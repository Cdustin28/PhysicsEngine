//! Fluid performance benchmark: a dam break with about 5,000 particles.
//!
//! ```sh
//! cargo run --release -p engine --example bench_fluid
//! cargo run --release -p engine --example bench_fluid --features parallel
//! ```
//!
//! `WARM` and `STEPS` (environment variables) change the warm-up and timed
//! step counts, e.g. for a short run under a profiler.

use engine::{Vec2, presets};
use std::time::Instant;

fn main() {
    // A 3.5 m × 3.6 m column in a 10 m tank: about 5,000 particles.
    let mut world = presets::tank(10.0, 5.0);
    let bodies = world.bodies.clone();
    world
        .fluid
        .fill_rect(Vec2::new(-5.0, 0.0), Vec2::new(-1.5, 3.6), &bodies);
    let n = world.fluid.len();
    let dt = 1.0 / 120.0;

    // Warm up through the initial collapse, then time 1 s of simulation.
    for _ in 0..std::env::var("WARM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30)
    {
        world.step(dt);
    }
    let steps = std::env::var("STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let mut substeps = 0;
    let start = Instant::now();
    for _ in 0..steps {
        world.step(dt);
        substeps += world.fluid.stats.substeps;
    }
    let per_step = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let threads = if cfg!(feature = "parallel") {
        format!(
            "parallel ({} threads)",
            std::thread::available_parallelism().map_or(1, |n| n.get())
        )
    } else {
        "single-threaded".to_owned()
    };
    println!(
        "{n} particles, {threads}: {per_step:.1} ms per 1/120 s step, {:.1} substeps/step, {:.2} µs per particle-substep",
        substeps as f64 / steps as f64,
        per_step * 1000.0 / (n as f64 * substeps as f64 / steps as f64),
    );
    println!(
        "real time needs ≤ {:.1} ms per step; this is {:.0}% of that",
        dt * 1000.0,
        per_step / (f64::from(dt) * 1000.0) * 100.0
    );
}
