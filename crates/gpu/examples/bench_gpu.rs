//! CPU vs GPU fluid benchmark at several particle counts.
//!
//! ```sh
//! cargo run --release -p gpu --example bench_gpu
//! cargo run --release -p gpu --example bench_gpu --features parallel   # multithreaded CPU
//! ```
//!
//! Each case is a dam break in a wide tank, timed over 30 world steps
//! (1/120 s each) after a short warm-up. Real time needs ≤ 8.3 ms per step.

use engine::{CpuSolver, FluidSolver, Vec2, World, presets};
use gpu::GpuSolver;
use std::time::Instant;

fn scene(target: usize) -> World {
    // A square column holding about `target` particles at 0.05 m spacing.
    let side = ((target as f32).sqrt() * 0.05).max(0.5);
    let mut w = presets::tank(3.0 * side + 2.0, 2.0 * side);
    let bodies = w.bodies.clone();
    let left = -0.5 * (3.0 * side + 2.0);
    w.fluid
        .fill_rect(Vec2::new(left, 0.0), Vec2::new(left + side, side), &bodies);
    w
}

fn time(world: &World, solver: &mut dyn FluidSolver, steps: usize) -> f64 {
    let mut w = world.clone();
    let dt = 1.0 / 120.0;
    for _ in 0..3 {
        w.step_with(dt, solver);
    }
    let start = Instant::now();
    for _ in 0..steps {
        w.step_with(dt, solver);
    }
    start.elapsed().as_secs_f64() * 1000.0 / steps as f64
}

fn main() {
    let mut gpu = match GpuSolver::headless() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            return;
        }
    };
    let cpu_kind = if cfg!(feature = "parallel") {
        "CPU (all cores)"
    } else {
        "CPU (1 thread)"
    };
    println!("GPU: {}", gpu.adapter_name());
    println!("{:>10} {:>18} {:>14}", "particles", cpu_kind, "GPU");
    for target in [5_000, 20_000, 100_000] {
        let world = scene(target);
        let n = world.fluid.len();
        let steps = if n > 50_000 { 5 } else { 30 };
        let cpu = time(&world, &mut CpuSolver, steps);
        let g = time(&world, &mut gpu, steps);
        println!("{n:>10} {cpu:>15.1} ms {g:>11.1} ms");
    }
}
