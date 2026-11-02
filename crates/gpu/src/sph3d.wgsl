// 3D weakly compressible SPH on the GPU, several liquids and heat: the same
// equations as the CPU solver in `engine3d::fluid` (one compute pass per
// stage).
//
// Each substep runs, in order:
//   clear_heads       empty the neighbor hash table
//   insert            push every particle onto its cell's linked list
//   density           number density (fluid + boundary), density, pressure
//   forces            pressure, viscosity, gravity -> next velocity;
//                     conduction -> rate of temperature change
//   boundary_impulse  the reaction each boundary particle feels (coupling)
//   integrate         semi-implicit Euler with the speed cap
//
// Several liquids (Solenthaler & Pajarola 2008): each particle sums the
// *number density* δ = Σ W (boundary particles count ψ/m_ref each), its
// density is its own mass times that, ρ = m δ, and the pressure force is
// −(1/m_i) Σ (p_i/δ_i² + p_j/δ_j²) ∇W. Every liquid shares one stiffness.
//
// Neighbor grid: cells of side h, hashed into `heads`. Each particle stores
// its cell and the index of the next particle in the same bucket, so a
// bucket is a linked list built with atomicExchange in a single pass.
//
// Temperatures: each particle's temperature at the start of the step is
// `base` (constant during the step) and the change since then is `delta`.
// Conduction changes a temperature by far less per substep than an f32 can
// resolve at 300 K, so the small `delta` carries it, and the CPU adds it to
// its f64 temperature afterwards.
//
// Buffers are packed into vec4s to stay within WebGPU's 8 storage buffers.

struct Params {
    n: u32,             // fluid particles
    nb: u32,            // boundary particles
    table_mask: u32,    // fluid hash table size - 1
    btable_size: u32,   // boundary hash table size (power of two)
    h: f32,             // kernel support
    h2: f32,
    inv_cell: f32,      // 1 / h
    m_ref: f32,         // reference particle mass (water at rest density)
    v0: f32,            // rest volume of every particle
    stiffness: f32,     // shared Tait B
    c: f32,             // speed of sound
    alpha: f32,         // artificial viscosity
    alpha_wall: f32,    // wall viscosity
    h_visc: f32,        // smoothing length for viscosity (h / 2)
    eps: f32,           // 0.01 h_visc^2
    dt: f32,            // substep
    gravity: vec4<f32>, // xyz; 16-byte aligned
    max_speed: f32,
    sigma: f32,         // kernel normalization 8 / (pi h^3)
    use_viscosity: u32, // each liquid's physical viscosity (Morris)
    use_heat: u32,      // conduction
};

@group(0) @binding(0) var<uniform> P: Params;
// [3i] = x, y, z, temperature delta; [3i + 1] = velocity;
// [3i + 2] = next velocity xyz, rate of temperature change
@group(0) @binding(1) var<storage, read_write> particles: array<vec4<f32>>;
// density, pressure, number density, pressure / δ²
@group(0) @binding(2) var<storage, read_write> scratch: array<vec4<f32>>;
// [2i] = mass, viscosity, conductivity, specific heat;
// [2i + 1] = rest-volume ratio, base temperature, wall conductivity, unused
@group(0) @binding(3) var<storage, read_write> material: array<vec4<f32>>;
// cell x, y, z, next particle in bucket (-1 = end)
@group(0) @binding(4) var<storage, read_write> links: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> heads: array<atomic<i32>>;
// [2b] = boundary position, held temperature (< 0: none); [2b + 1] = velocity
@group(0) @binding(6) var<storage, read_write> boundary: array<vec4<f32>>;
// boundary psi, accumulated impulse x, y, z
@group(0) @binding(7) var<storage, read_write> boundary_extra: array<vec4<f32>>;
// Built on the CPU. Entries [0, btable_size): .x = first particle in bucket.
// Entries [btable_size + b]: cell x, y, z, next for particle b.
@group(0) @binding(8) var<storage, read_write> boundary_grid: array<vec4<i32>>;

const WG: u32 = 64u;

fn cell_of(x: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(clamp(floor(x * P.inv_cell), vec3<f32>(-1.0e9), vec3<f32>(1.0e9)));
}

// Same primes and wrapping arithmetic as the CPU grid.
fn hash(c: vec3<i32>, mask: u32) -> u32 {
    return ((bitcast<u32>(c.x) * 73856093u)
        ^ (bitcast<u32>(c.y) * 19349663u)
        ^ (bitcast<u32>(c.z) * 83492791u)) & mask;
}

fn pos(i: u32) -> vec3<f32> { return particles[3u * i].xyz; }
fn tdelta(i: u32) -> f32 { return particles[3u * i].w; }
fn vel(i: u32) -> vec3<f32> { return particles[3u * i + 1u].xyz; }
fn bpos(b: u32) -> vec3<f32> { return boundary[2u * b].xyz; }
fn bheld(b: u32) -> f32 { return boundary[2u * b].w; }
fn bvel(b: u32) -> vec3<f32> { return boundary[2u * b + 1u].xyz; }
fn same_cell(l: vec4<i32>, c: vec3<i32>) -> bool { return all(l.xyz == c); }

fn kernel_w(r: f32) -> f32 {
    let q = r / P.h;
    if (q <= 0.5) {
        return P.sigma * (6.0 * (q * q * q - q * q) + 1.0);
    } else if (q <= 1.0) {
        let t = 1.0 - q;
        return P.sigma * 2.0 * t * t * t;
    }
    return 0.0;
}

// Gradient with respect to x_i of W(x_i - x_j), for r_vec = x_i - x_j.
fn kernel_grad(r_vec: vec3<f32>, r: f32) -> vec3<f32> {
    if (r <= 1.0e-9 || r >= P.h) {
        return vec3<f32>(0.0);
    }
    let q = r / P.h;
    var dw_dq: f32;
    if (q <= 0.5) {
        dw_dq = 6.0 * (3.0 * q * q - 2.0 * q);
    } else {
        let t = 1.0 - q;
        dw_dq = -6.0 * t * t;
    }
    return r_vec * (P.sigma * dw_dq / (P.h * r));
}

@compute @workgroup_size(WG)
fn clear_heads(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x <= P.table_mask) {
        atomicStore(&heads[id.x], -1);
    }
}

@compute @workgroup_size(WG)
fn insert(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let c = cell_of(pos(i));
    let previous = atomicExchange(&heads[hash(c, P.table_mask)], i32(i));
    links[i] = vec4<i32>(c, previous);
}

@compute @workgroup_size(WG)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let xi = pos(i);
    let ci = cell_of(xi);
    var delta = kernel_w(0.0);

    for (var cell_index = 0; cell_index < 27; cell_index++) {
        {
            let c = ci + vec3<i32>(cell_index % 3 - 1, (cell_index / 3) % 3 - 1, cell_index / 9 - 1);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (same_cell(l, c) && u32(j) != i) {
                    let r = length(xi - pos(u32(j)));
                    if (r < P.h) {
                        delta += kernel_w(r);
                    }
                }
                j = l.w;
            }

            if (P.nb > 0u) {
                var b = boundary_grid[hash(c, P.btable_size - 1u)].x;
                loop {
                    if (b < 0) { break; }
                    let l = boundary_grid[P.btable_size + u32(b)];
                    if (same_cell(l, c)) {
                        let r = length(xi - bpos(u32(b)));
                        if (r < P.h) {
                            delta += boundary_extra[b].x / P.m_ref * kernel_w(r);
                        }
                    }
                    b = l.w;
                }
            }
        }
    }

    let m = material[2u * i];
    let volume_ratio = material[2u * i + 1u].x;
    let rho = m.x * delta;
    let pressure = max(P.stiffness * (pow(delta * P.v0 * volume_ratio, 7.0) - 1.0), 0.0);
    scratch[i] = vec4<f32>(rho, pressure, delta, pressure / (delta * delta));
}

@compute @workgroup_size(WG)
fn forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let xi = pos(i);
    let vi = vel(i);
    let si = scratch[i];
    let mi = material[2u * i];
    let mi2 = material[2u * i + 1u];
    let mass_i = mi.x;
    let inv_m = 1.0 / mass_i;
    let base_i = mi2.y;
    let td_i = tdelta(i);
    let ci = cell_of(xi);
    var a = P.gravity.xyz;
    var heat = 0.0;

    for (var cell_index = 0; cell_index < 27; cell_index++) {
        {
            let c = ci + vec3<i32>(cell_index % 3 - 1, (cell_index / 3) % 3 - 1, cell_index / 9 - 1);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                let uj = u32(j);
                if (same_cell(l, c) && uj != i) {
                    let r = xi - pos(uj);
                    let dist = length(r);
                    if (dist < P.h) {
                        let grad = kernel_grad(r, dist);
                        let sj = scratch[uj];
                        let mj = material[2u * uj];
                        let r2e = dist * dist + P.eps;
                        a -= inv_m * (si.w + sj.w) * grad;
                        let v_ij = vi - vel(uj);
                        let vr = dot(v_ij, r);
                        if (vr < 0.0) {
                            let nu = 2.0 * P.alpha * P.h_visc * P.c / (si.x + sj.x);
                            a -= mj.x * (-nu * vr / r2e) * grad;
                        }
                        let rg = dot(r, grad);
                        if (P.use_viscosity != 0u) {
                            a += inv_m * (mi.y + mj.y) * rg / (si.z * sj.z * r2e) * v_ij;
                        }
                        if (P.use_heat != 0u) {
                            let k_ij = 4.0 * mi.z * mj.z / (mi.z + mj.z);
                            // Base difference and delta difference apart,
                            // so small changes aren't lost to rounding.
                            let diff = (base_i - material[2u * uj + 1u].y) + (td_i - tdelta(uj));
                            heat += k_ij * diff * rg / (sj.z * r2e);
                        }
                    }
                }
                j = l.w;
            }

            if (P.nb > 0u) {
                var b = boundary_grid[hash(c, P.btable_size - 1u)].x;
                loop {
                    if (b < 0) { break; }
                    let l = boundary_grid[P.btable_size + u32(b)];
                    let ub = u32(b);
                    if (same_cell(l, c)) {
                        let r = xi - bpos(ub);
                        let dist = length(r);
                        if (dist < P.h) {
                            let grad = kernel_grad(r, dist);
                            let w = boundary_extra[ub].x / P.m_ref;
                            let r2e = dist * dist + P.eps;
                            a -= w * inv_m * si.w * grad;
                            let v_ib = vi - bvel(ub);
                            let vr = dot(v_ib, r);
                            if (vr < 0.0) {
                                let nu = P.alpha_wall * P.h_visc * P.c / si.x;
                                a -= w * mass_i * (-nu * vr / r2e) * grad;
                            }
                            let rg = dot(r, grad);
                            if (P.use_viscosity != 0u) {
                                a += w * inv_m * 2.0 * mi.y * rg / (si.z * si.z * r2e) * v_ib;
                            }
                            let held = bheld(ub);
                            if (P.use_heat != 0u && held >= 0.0) {
                                let diff = (base_i - held) + td_i;
                                heat += 2.0 * mi2.z * w * P.v0 * diff * rg / r2e;
                            }
                        }
                    }
                    b = l.w;
                }
            }
        }
    }

    var v = vi + a * P.dt;
    let speed = length(v);
    if (speed > P.max_speed) {
        v *= P.max_speed / speed;
    }
    particles[3u * i + 2u] = vec4<f32>(v, heat / (mass_i * si.z * mi.w));
}

// For each boundary particle, the exact opposite of the force it applied to
// every nearby fluid particle in `forces`, times dt.
@compute @workgroup_size(WG)
fn boundary_impulse(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if (b >= P.nb) {
        return;
    }
    let xb = bpos(b);
    let vb = bvel(b);
    let extra = boundary_extra[b];
    let w = extra.x / P.m_ref;
    let cb = cell_of(xb);
    var f = vec3<f32>(0.0);

    for (var cell_index = 0; cell_index < 27; cell_index++) {
        {
            let c = cb + vec3<i32>(cell_index % 3 - 1, (cell_index / 3) % 3 - 1, cell_index / 9 - 1);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                let uj = u32(j);
                if (same_cell(l, c)) {
                    let r = pos(uj) - xb;
                    let dist = length(r);
                    if (dist < P.h) {
                        let grad = kernel_grad(r, dist);
                        let sj = scratch[uj];
                        let mj = material[2u * uj];
                        let r2e = dist * dist + P.eps;
                        let v_ib = vel(uj) - vb;
                        let vr = dot(v_ib, r);
                        var force = w * sj.w * grad;
                        if (vr < 0.0) {
                            let nu = P.alpha_wall * P.h_visc * P.c / sj.x;
                            force += w * mj.x * mj.x * (-nu * vr / r2e) * grad;
                        }
                        if (P.use_viscosity != 0u) {
                            force -= w * 2.0 * mj.y * dot(r, grad) / (sj.z * sj.z * r2e) * v_ib;
                        }
                        f += force;
                    }
                }
                j = l.w;
            }
        }
    }
    boundary_extra[b] = vec4<f32>(extra.x, extra.yzw + f * P.dt);
}

@compute @workgroup_size(WG)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let next = particles[3u * i + 2u];
    let v = next.xyz;
    particles[3u * i] = vec4<f32>(pos(i) + v * P.dt, tdelta(i) + next.w * P.dt);
    particles[3u * i + 1u] = vec4<f32>(v, 0.0);
}
