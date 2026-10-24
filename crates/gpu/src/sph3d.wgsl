// 3D weakly compressible SPH on the GPU: the same equations as the CPU
// solver in `engine3d::fluid`, and the same structure as the 2D shader
// (`sph.wgsl`), one compute pass per stage.
//
// Each substep runs, in order:
//   clear_heads       empty the neighbor hash table
//   insert            push every particle onto its cell's linked list
//   density           density (fluid + boundary) and Tait pressure
//   forces            pressure + viscosity + gravity -> next velocity
//   boundary_impulse  the reaction each boundary particle feels (M3 coupling)
//   integrate         semi-implicit Euler with the speed cap
//
// Neighbor grid: cells of side h, hashed into `heads`. Each particle stores
// its cell and the index of the next particle in the same bucket, so a
// bucket is a linked list built with atomicExchange in a single pass (no
// sorting or prefix sums). Different cells can share a bucket, so walks
// compare the stored cell with the queried one.
//
// Buffers are packed into vec4s to stay within WebGPU's limit of 8 storage
// buffers per shader stage. A 3D particle needs six floats, so positions and
// velocities share one buffer: element 2i is particle i's position, 2i + 1
// its velocity (likewise for boundary particles).

struct Params {
    n: u32,             // fluid particles
    nb: u32,            // boundary particles
    table_mask: u32,    // fluid hash table size - 1
    btable_size: u32,   // boundary hash table size (power of two)
    h: f32,             // kernel support
    h2: f32,
    inv_cell: f32,      // 1 / h
    mass: f32,          // per fluid particle
    rho0: f32,
    stiffness: f32,     // Tait B
    c: f32,             // speed of sound
    alpha: f32,         // fluid viscosity
    alpha_wall: f32,    // wall viscosity
    h_visc: f32,        // smoothing length for viscosity (h / 2)
    eps: f32,           // 0.01 h_visc^2
    dt: f32,            // substep
    gravity: vec4<f32>, // xyz; 16-byte aligned
    max_speed: f32,
    sigma: f32,         // kernel normalization 8 / (pi h^3)
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
// [2i] = x, y, z, unused; [2i + 1] = vx, vy, vz, unused
@group(0) @binding(1) var<storage, read_write> particles: array<vec4<f32>>;
// density, pressure, unused, unused
@group(0) @binding(2) var<storage, read_write> scratch: array<vec4<f32>>;
// velocity after this substep's forces
@group(0) @binding(3) var<storage, read_write> next_velocity: array<vec4<f32>>;
// cell x, y, z, next particle in bucket (-1 = end)
@group(0) @binding(4) var<storage, read_write> links: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> heads: array<atomic<i32>>;
// [2b] = boundary position, [2b + 1] = its velocity
@group(0) @binding(6) var<storage, read_write> boundary: array<vec4<f32>>;
// boundary psi, accumulated impulse x, y, z
@group(0) @binding(7) var<storage, read_write> boundary_extra: array<vec4<f32>>;
// Built on the CPU. Entries [0, btable_size): .x = first particle in bucket.
// Entries [btable_size + b]: cell x, y, z, next for particle b.
@group(0) @binding(8) var<storage, read_write> boundary_grid: array<vec4<i32>>;

const WG: u32 = 64u;

fn cell_of(x: vec3<f32>) -> vec3<i32> {
    // Clamp so far-flung particles can't overflow the integer conversion.
    return vec3<i32>(clamp(floor(x * P.inv_cell), vec3<f32>(-1.0e9), vec3<f32>(1.0e9)));
}

// Same primes and wrapping arithmetic as the CPU grid.
fn hash(c: vec3<i32>, mask: u32) -> u32 {
    return ((bitcast<u32>(c.x) * 73856093u)
        ^ (bitcast<u32>(c.y) * 19349663u)
        ^ (bitcast<u32>(c.z) * 83492791u)) & mask;
}

fn pos(i: u32) -> vec3<f32> { return particles[2u * i].xyz; }
fn vel(i: u32) -> vec3<f32> { return particles[2u * i + 1u].xyz; }
fn bpos(b: u32) -> vec3<f32> { return boundary[2u * b].xyz; }
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
    var rho = P.mass * kernel_w(0.0);

    // The 27 cells around this one: offsets -1..1 on each axis.
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
                        rho += P.mass * kernel_w(r);
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
                            rho += boundary_extra[b].x * kernel_w(r);
                        }
                    }
                    b = l.w;
                }
            }
        }
    }

    let pressure = max(P.stiffness * (pow(rho / P.rho0, 7.0) - 1.0), 0.0);
    scratch[i] = vec4<f32>(rho, pressure, 0.0, 0.0);
}

@compute @workgroup_size(WG)
fn forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let xi = pos(i);
    let vi = vel(i);
    let rho_i = scratch[i].x;
    let pi_term = scratch[i].y / (rho_i * rho_i);
    let ci = cell_of(xi);
    var a = P.gravity.xyz;

    // The 27 cells around this one: offsets -1..1 on each axis.
    for (var cell_index = 0; cell_index < 27; cell_index++) {
        {
            let c = ci + vec3<i32>(cell_index % 3 - 1, (cell_index / 3) % 3 - 1, cell_index / 9 - 1);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (same_cell(l, c) && u32(j) != i) {
                    let r = xi - pos(u32(j));
                    let dist = length(r);
                    if (dist < P.h) {
                        let grad = kernel_grad(r, dist);
                        let sj = scratch[j];
                        a -= P.mass * (pi_term + sj.y / (sj.x * sj.x)) * grad;
                        let vr = dot(vi - vel(u32(j)), r);
                        if (vr < 0.0) {
                            let nu = 2.0 * P.alpha * P.h_visc * P.c / (rho_i + sj.x);
                            a -= P.mass * (-nu * vr / (dist * dist + P.eps)) * grad;
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
                    if (same_cell(l, c)) {
                        let r = xi - bpos(u32(b));
                        let dist = length(r);
                        if (dist < P.h) {
                            let grad = kernel_grad(r, dist);
                            let psi = boundary_extra[b].x;
                            a -= psi * pi_term * grad;
                            let vr = dot(vi - bvel(u32(b)), r);
                            if (vr < 0.0) {
                                let nu = P.alpha_wall * P.h_visc * P.c / rho_i;
                                a -= psi * (-nu * vr / (dist * dist + P.eps)) * grad;
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
    next_velocity[i] = vec4<f32>(v, 0.0);
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
    let psi = extra.x;
    let cb = cell_of(xb);
    var f = vec3<f32>(0.0);

    // The 27 cells around this one: offsets -1..1 on each axis.
    for (var cell_index = 0; cell_index < 27; cell_index++) {
        {
            let c = cb + vec3<i32>(cell_index % 3 - 1, (cell_index / 3) % 3 - 1, cell_index / 9 - 1);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (same_cell(l, c)) {
                    let r = pos(u32(j)) - xb;
                    let dist = length(r);
                    if (dist < P.h) {
                        let sj = scratch[j];
                        var coefficient = sj.y / (sj.x * sj.x);
                        let vr = dot(vel(u32(j)) - vb, r);
                        if (vr < 0.0) {
                            let nu = P.alpha_wall * P.h_visc * P.c / sj.x;
                            coefficient += -nu * vr / (dist * dist + P.eps);
                        }
                        f += P.mass * psi * coefficient * kernel_grad(r, dist);
                    }
                }
                j = l.w;
            }
        }
    }
    boundary_extra[b] = vec4<f32>(psi, extra.yzw + f * P.dt);
}

@compute @workgroup_size(WG)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let v = next_velocity[i].xyz;
    particles[2u * i] = vec4<f32>(pos(i) + v * P.dt, 0.0);
    particles[2u * i + 1u] = vec4<f32>(v, 0.0);
}
