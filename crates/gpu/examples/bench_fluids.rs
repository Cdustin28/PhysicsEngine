//! The Fluids tab's kind of scene, CPU vs GPU: a 1 m box a third full of
//! water at 20 °C, a mercury ball, and an 80 °C plate, at several particle
//! spacings.
//!
//! ```sh
//! cargo run --release -p gpu --example bench_fluids --features parallel
//! ```
//!
//! Real time needs ≤ 8.3 ms per 1/120 s step.

use engine3d::fluid::{Liquid, LiquidSpec};
use engine3d::{CpuSolver, FluidSolver, RigidBody, Shape, Vec3, World, presets};
use gpu::GpuSolver3d;
use std::time::Instant;

fn scene(spacing: f32) -> World {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    w.fluid.params.particle_spacing = spacing;
    w.fluid.params.speed_of_sound = 31.0;
    let mut plate = RigidBody::new_static(Shape::cuboid(0.3, 0.02, 0.3), Vec3::new(0.3, 0.01, 0.3));
    plate.temperature = Some(353.15);
    w.add_body(plate);
    let b = w.bodies.clone();
    w.fluid
        .fill_box(Vec3::new(-0.5, 0.02, -0.5), Vec3::new(0.5, 0.33, 0.5), &b);
    w.fluid.fill_sphere_as(
        Vec3::new(-0.2, 0.6, -0.2),
        0.12,
        &b,
        LiquidSpec::new(Liquid::Mercury),
    );
    w
}

fn time(world: &World, solver: &mut dyn FluidSolver, steps: usize) -> (f64, u32) {
    let mut w = world.clone();
    let dt = 1.0 / 120.0;
    for _ in 0..3 {
        w.step_with(dt, solver);
    }
    let start = Instant::now();
    for _ in 0..steps {
        w.step_with(dt, solver);
    }
    (
        start.elapsed().as_secs_f64() * 1000.0 / steps as f64,
        w.fluid.stats.substeps,
    )
}

fn main() {
    let mut gpu = match GpuSolver3d::headless() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            return;
        }
    };
    println!("GPU: {}", gpu.adapter_name());
    println!(
        "{:>8} {:>10} {:>10} {:>12} {:>12}   last GPU step: prepare / compute / finish",
        "spacing", "particles", "substeps", "CPU", "GPU"
    );
    for spacing in [0.04, 0.03, 0.025] {
        let world = scene(spacing);
        let n = world.fluid.len();
        let steps = if n > 20_000 { 3 } else { 8 };
        let (cpu, _) = time(&world, &mut CpuSolver, steps);
        let (g, subs) = time(&world, &mut gpu, steps);
        let t = gpu.timings;
        println!(
            "{:>6.1} cm {n:>10} {subs:>10} {cpu:>9.1} ms {g:>9.1} ms   {:.1} / {:.1} / {:.1} ms",
            spacing * 100.0,
            t.prepare,
            t.gpu,
            t.finish
        );
    }
}
