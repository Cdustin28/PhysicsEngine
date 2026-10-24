// Weakly compressible SPH on the GPU: the same equations as the CPU solver
// in `engine::fluid` (see its docs), split into one compute pass per stage.
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
// buffers per shader stage.

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
    gravity: vec2<f32>,
    max_speed: f32,
    sigma: f32,         // kernel normalization 40 / (7 pi h^2)
};

@group(0) @binding(0) var<uniform> P: Params;
// x, y, vx, vy
@group(0) @binding(1) var<storage, read_write> particles: array<vec4<f32>>;
// density, pressure, unused, unused
@group(0) @binding(2) var<storage, read_write> scratch: array<vec4<f32>>;
// velocity after this substep's forces
@group(0) @binding(3) var<storage, read_write> next_velocity: array<vec2<f32>>;
// cell x, cell y, next particle in bucket (-1 = end), unused
@group(0) @binding(4) var<storage, read_write> links: array<vec4<i32>>;
@group(0) @binding(5) var<storage, read_write> heads: array<atomic<i32>>;
// boundary x, y, vx, vy
@group(0) @binding(6) var<storage, read_write> boundary: array<vec4<f32>>;
// boundary psi, accumulated impulse x, impulse y, unused
@group(0) @binding(7) var<storage, read_write> boundary_extra: array<vec4<f32>>;
// Built on the CPU. Entries [0, btable_size): .x = first particle in bucket.
// Entries [btable_size + b]: cell x, cell y, next, unused for particle b.
@group(0) @binding(8) var<storage, read_write> boundary_grid: array<vec4<i32>>;

const WG: u32 = 64u;

fn cell_of(x: vec2<f32>) -> vec2<i32> {
    // Clamp so far-flung particles can't overflow the integer conversion.
    return vec2<i32>(clamp(floor(x * P.inv_cell), vec2<f32>(-1.0e9), vec2<f32>(1.0e9)));
}

// Same primes and wrapping arithmetic as the CPU grid.
fn hash(c: vec2<i32>, mask: u32) -> u32 {
    return ((bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u)) & mask;
}

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
fn kernel_grad(r_vec: vec2<f32>, r: f32) -> vec2<f32> {
    if (r <= 1.0e-9 || r >= P.h) {
        return vec2<f32>(0.0);
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
    let c = cell_of(particles[i].xy);
    let previous = atomicExchange(&heads[hash(c, P.table_mask)], i32(i));
    links[i] = vec4<i32>(c.x, c.y, previous, 0);
}

@compute @workgroup_size(WG)
fn density(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let xi = particles[i].xy;
    let ci = cell_of(xi);
    var rho = P.mass * kernel_w(0.0);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = ci + vec2<i32>(dx, dy);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (l.x == c.x && l.y == c.y && u32(j) != i) {
                    let r = length(xi - particles[j].xy);
                    if (r < P.h) {
                        rho += P.mass * kernel_w(r);
                    }
                }
                j = l.z;
            }

            if (P.nb > 0u) {
                var b = boundary_grid[hash(c, P.btable_size - 1u)].x;
                loop {
                    if (b < 0) { break; }
                    let l = boundary_grid[P.btable_size + u32(b)];
                    if (l.x == c.x && l.y == c.y) {
                        let r = length(xi - boundary[b].xy);
                        if (r < P.h) {
                            rho += boundary_extra[b].x * kernel_w(r);
                        }
                    }
                    b = l.z;
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
    let pi = particles[i];
    let xi = pi.xy;
    let vi = pi.zw;
    let rho_i = scratch[i].x;
    let pi_term = scratch[i].y / (rho_i * rho_i);
    let ci = cell_of(xi);
    var a = P.gravity;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = ci + vec2<i32>(dx, dy);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (l.x == c.x && l.y == c.y && u32(j) != i) {
                    let pj = particles[j];
                    let r = xi - pj.xy;
                    let dist = length(r);
                    if (dist < P.h) {
                        let grad = kernel_grad(r, dist);
                        let sj = scratch[j];
                        a -= P.mass * (pi_term + sj.y / (sj.x * sj.x)) * grad;
                        let vr = dot(vi - pj.zw, r);
                        if (vr < 0.0) {
                            let nu = 2.0 * P.alpha * P.h_visc * P.c / (rho_i + sj.x);
                            a -= P.mass * (-nu * vr / (dist * dist + P.eps)) * grad;
                        }
                    }
                }
                j = l.z;
            }

            if (P.nb > 0u) {
                var b = boundary_grid[hash(c, P.btable_size - 1u)].x;
                loop {
                    if (b < 0) { break; }
                    let l = boundary_grid[P.btable_size + u32(b)];
                    if (l.x == c.x && l.y == c.y) {
                        let pb = boundary[b];
                        let r = xi - pb.xy;
                        let dist = length(r);
                        if (dist < P.h) {
                            let grad = kernel_grad(r, dist);
                            let psi = boundary_extra[b].x;
                            a -= psi * pi_term * grad;
                            let vr = dot(vi - pb.zw, r);
                            if (vr < 0.0) {
                                let nu = P.alpha_wall * P.h_visc * P.c / rho_i;
                                a -= psi * (-nu * vr / (dist * dist + P.eps)) * grad;
                            }
                        }
                    }
                    b = l.z;
                }
            }
        }
    }

    var v = vi + a * P.dt;
    let speed = length(v);
    if (speed > P.max_speed) {
        v *= P.max_speed / speed;
    }
    next_velocity[i] = v;
}

// For each boundary particle, the exact opposite of the force it applied to
// every nearby fluid particle in `forces`, times dt.
@compute @workgroup_size(WG)
fn boundary_impulse(@builtin(global_invocation_id) id: vec3<u32>) {
    let b = id.x;
    if (b >= P.nb) {
        return;
    }
    let pb = boundary[b];
    let extra = boundary_extra[b];
    let psi = extra.x;
    let cb = cell_of(pb.xy);
    var f = vec2<f32>(0.0);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let c = cb + vec2<i32>(dx, dy);
            var j = atomicLoad(&heads[hash(c, P.table_mask)]);
            loop {
                if (j < 0) { break; }
                let l = links[j];
                if (l.x == c.x && l.y == c.y) {
                    let pj = particles[j];
                    let r = pj.xy - pb.xy;
                    let dist = length(r);
                    if (dist < P.h) {
                        let sj = scratch[j];
                        var coefficient = sj.y / (sj.x * sj.x);
                        let vr = dot(pj.zw - pb.zw, r);
                        if (vr < 0.0) {
                            let nu = P.alpha_wall * P.h_visc * P.c / sj.x;
                            coefficient += -nu * vr / (dist * dist + P.eps);
                        }
                        f += P.mass * psi * coefficient * kernel_grad(r, dist);
                    }
                }
                j = l.z;
            }
        }
    }
    boundary_extra[b] = vec4<f32>(psi, extra.yz + f * P.dt, 0.0);
}

@compute @workgroup_size(WG)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= P.n) {
        return;
    }
    let v = next_velocity[i];
    particles[i] = vec4<f32>(particles[i].xy + v * P.dt, v);
}
