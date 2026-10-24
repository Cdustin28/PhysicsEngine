use engine3d::{CpuSolver, FluidSolver, presets};
use std::time::Instant;
fn main() {
    let mut w = presets::dam_break();
    for _ in 0..120 { w.step(1.0 / 120.0); }
    let p = w.fluid.params;
    let t = Instant::now();
    let mut nb = 0;
    for _ in 0..50 { nb = p.boundary(&w.bodies, w.fluid.boundary_region()).len(); }
    println!("boundary build: {:.2} ms ({nb} particles)", t.elapsed().as_secs_f64() * 1000.0 / 50.0);
    let b = p.boundary(&w.bodies, w.fluid.boundary_region());
    let t = Instant::now();
    for _ in 0..50 { let n = w.bodies.len(); CpuSolver.step(&mut w.fluid, &b, w.gravity, 1.0/120.0, n); }
    println!("fluid step: {:.2} ms ({} substeps)", t.elapsed().as_secs_f64() * 1000.0 / 50.0, w.fluid.stats.substeps);
}
