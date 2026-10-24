//! The GPU solver must agree with the CPU solver. Every test here runs the
//! same scene both ways (or checks the GPU result against the same physics
//! the CPU tests use).
//!
//! The tests need a GPU adapter. Software Vulkan (Mesa's lavapipe) is
//! enough, and that's what CI uses. Without any adapter they print a
//! message and pass, so a machine without one can still run the rest of
//! the suite.

use engine::{CpuSolver, RigidBody, Shape, Vec2, World, presets};
use gpu::GpuSolver;

const DT: f32 = 1.0 / 120.0;

fn gpu() -> Option<GpuSolver> {
    match GpuSolver::headless() {
        Ok(solver) => {
            eprintln!("GPU tests running on: {}", solver.adapter_name());
            Some(solver)
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn column(width: f32, height: f32) -> World {
    let mut w = presets::tank(width, 2.0 * height);
    let bodies = w.bodies.clone();
    w.fluid.fill_rect(
        Vec2::new(-0.5 * width, 0.0),
        Vec2::new(0.5 * width, height),
        &bodies,
    );
    w
}

/// After one world step from the same state, both solvers put every particle
/// in nearly the same place. (Not bit-identical: the GPU always takes the
/// worst-case substep count and sums in a different order.)
#[test]
fn one_step_matches_cpu() {
    let Some(mut gpu) = gpu() else { return };
    let mut on_cpu = presets::dam_break();
    // Let the column start moving so the step isn't trivial.
    for _ in 0..30 {
        on_cpu.step(DT);
    }
    let mut on_gpu = on_cpu.clone();
    on_cpu.step_with(DT, &mut CpuSolver);
    on_gpu.step_with(DT, &mut gpu);

    let d = on_cpu.fluid.params.particle_spacing;
    // The CPU solver sorts particles each step and the GPU one every 10, so
    // compare by matching nearest particles rather than by index.
    let mut cpu_sorted = on_cpu.fluid.positions.clone();
    let mut gpu_sorted = on_gpu.fluid.positions.clone();
    let key = |p: &Vec2| ((p.y / d).round() as i32, (p.x / d).round() as i32);
    cpu_sorted.sort_by_key(key);
    gpu_sorted.sort_by_key(key);
    let worst = cpu_sorted
        .iter()
        .zip(&gpu_sorted)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0_f32, f32::max);
    assert!(worst < 0.2 * d, "positions differ by up to {worst} m");

    let mean = |w: &World| w.fluid.densities().iter().sum::<f32>() / w.fluid.len() as f32;
    let (a, b) = (mean(&on_cpu), mean(&on_gpu));
    assert!(
        (a - b).abs() / a < 0.01,
        "mean density {a} (CPU) vs {b} (GPU)"
    );
}

/// Hydrostatic pressure gradient ρ₀g, the same check as the CPU suite.
#[test]
fn gpu_hydrostatic_pressure_gradient_is_rho_g() {
    let Some(mut gpu) = gpu() else { return };
    let mut w = column(1.0, 1.0);
    for _ in 0..480 {
        w.step_with(DT, &mut gpu);
    }
    let rho_g = w.fluid.params.rest_density * engine::STANDARD_GRAVITY;
    let band = |w: &World, lo: f32, hi: f32| {
        let (s, n) = w
            .fluid
            .positions
            .iter()
            .zip(w.fluid.pressures())
            .filter(|(p, _)| p.x.abs() < 0.35 && p.y > lo && p.y < hi)
            .fold((0.0, 0), |(s, n), (_, &p)| (s + p, n + 1));
        s / n as f32
    };
    let samples = 60;
    let mut ratio = 0.0;
    for _ in 0..samples {
        w.step_with(DT, &mut gpu);
        ratio += (band(&w, 0.2, 0.3) - band(&w, 0.6, 0.7)) / (0.4 * rho_g);
    }
    ratio /= samples as f32;
    assert!(
        (ratio - 1.0).abs() < 0.1,
        "dp/dy = {ratio:.3} × ρ₀g on the GPU"
    );
}

/// Archimedes with the GPU's coupling forces: a half-density box floats
/// half submerged.
#[test]
fn gpu_half_density_box_floats_half_submerged() {
    let Some(mut gpu) = gpu() else { return };
    let mut w = presets::tank(2.4, 2.0);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-1.2, 0.0), Vec2::new(1.2, 1.0), &bodies);
    let mut b = RigidBody::new(Shape::rect(0.6, 0.4), Vec2::new(0.0, 1.25));
    b.density = 5.0;
    let id = w.add_body(b);
    for _ in 0..720 {
        w.step_with(DT, &mut gpu);
    }
    let d = w.fluid.params.particle_spacing;
    let mut fraction = 0.0;
    let samples = 60;
    for _ in 0..samples {
        w.step_with(DT, &mut gpu);
        let body = w.body(id).unwrap();
        let mut ys: Vec<f32> = w
            .fluid
            .positions
            .iter()
            .filter(|p| (p.x - body.position.x).abs() > 0.6)
            .map(|p| p.y)
            .collect();
        ys.sort_by(|a, b| b.total_cmp(a));
        let level = ys[..12].iter().sum::<f32>() / 12.0 + 0.5 * d;
        fraction += (level - (body.position.y - 0.2)) / 0.4;
    }
    fraction /= samples as f32;
    assert!(
        (fraction - 0.5).abs() < 0.08,
        "{fraction:.3} submerged on the GPU"
    );
}

/// Newton's third law across the CPU/GPU boundary: momentum the box loses
/// is exactly what the water gains.
#[test]
fn gpu_momentum_is_exchanged_exactly() {
    let Some(mut gpu) = gpu() else { return };
    let mut w = World::new();
    w.gravity = Vec2::ZERO;
    w.fluid
        .fill_rect(Vec2::new(0.0, -0.6), Vec2::new(1.2, 0.6), &[]);
    let mut body = RigidBody::new(Shape::rect(0.3, 0.5), Vec2::new(-0.4, 0.0));
    body.velocity = Vec2::new(2.0, 0.0);
    body.density = 20.0;
    let id = w.add_body(body);
    let total = |w: &World| w.momentum() + w.fluid.momentum();
    let before = total(&w);
    for _ in 0..60 {
        w.step_with(DT, &mut gpu);
    }
    let v = w.body(id).unwrap().velocity.x;
    assert!(v < 1.8, "box didn't slow down ({v} m/s)");
    let after = total(&w);
    assert!(
        (after - before).length() < 0.02 * before.length(),
        "momentum {before} -> {after}"
    );
}
