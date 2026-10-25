//! The 3D GPU solver must agree with the 3D CPU solver. Like the 2D tests,
//! these need a GPU adapter (lavapipe is enough) and pass with a message
//! when there is none.

use engine3d::{CpuSolver, RigidBody, Shape, Vec3, World, presets};
use gpu::GpuSolver3d;

const DT: f32 = 1.0 / 120.0;

fn gpu() -> Option<GpuSolver3d> {
    match GpuSolver3d::headless() {
        Ok(solver) => {
            eprintln!("3D GPU tests running on: {}", solver.adapter_name());
            Some(solver)
        }
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

/// One world step from the same moving state lands every particle in nearly
/// the same place on both solvers.
#[test]
fn one_step_matches_cpu() {
    let Some(mut gpu) = gpu() else { return };
    let mut on_cpu = presets::dam_break();
    for _ in 0..30 {
        on_cpu.step(DT);
    }
    let mut on_gpu = on_cpu.clone();
    on_cpu.step_with(DT, &mut CpuSolver);
    on_gpu.step_with(DT, &mut gpu);

    let d = on_cpu.fluid.params.particle_spacing;
    // The solvers order particles differently, so match each CPU particle
    // with the nearest GPU one. (In one step particles move up to about
    // 3 m/s × 8 ms = 2.5 cm, so agreement to a few mm is meaningful.)
    let mean_error = on_cpu
        .fluid
        .positions
        .iter()
        .map(|p| {
            on_gpu
                .fluid
                .positions
                .iter()
                .map(|q| (*p - *q).length())
                .fold(f32::INFINITY, f32::min)
        })
        .sum::<f32>()
        / on_cpu.fluid.len() as f32;
    assert!(
        mean_error < 0.02 * d,
        "positions differ by {mean_error} m on average"
    );

    let mean = |w: &World| w.fluid.densities().iter().sum::<f32>() / w.fluid.len() as f32;
    let (x, y) = (mean(&on_cpu), mean(&on_gpu));
    assert!(
        (x - y).abs() / x < 0.01,
        "mean density {x} (CPU) vs {y} (GPU)"
    );

    // Coupling: the push on the walls agrees too.
    let force = |w: &World| w.fluid_forces.iter().map(|f| f.0).sum::<Vec3>();
    let (fc, fg) = (force(&on_cpu), force(&on_gpu));
    assert!(
        (fc - fg).length() < 0.02 * fc.length(),
        "net fluid force {fc} (CPU) vs {fg} (GPU)"
    );
}

/// Archimedes with the GPU's coupling forces: pine (500 kg/m³) floats half
/// submerged.
#[test]
fn gpu_pine_cube_floats_half_submerged() {
    let Some(mut gpu) = gpu() else { return };
    let mut w = presets::tank(1.2, 1.2, 1.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_box(
        Vec3::new(-0.6, 0.0, -0.6),
        Vec3::new(0.6, 0.5, 0.6),
        &bodies,
    );
    let mut b = RigidBody::new(Shape::cube(0.3), Vec3::new(0.0, 0.75, 0.0));
    b.density = 500.0;
    let id = w.add_body(b);
    for _ in 0..360 {
        w.step_with(DT, &mut gpu);
    }
    let samples = 60;
    let mut draft = 0.0;
    for _ in 0..samples {
        w.step_with(DT, &mut gpu);
        let level = w
            .fluid
            .positions
            .iter()
            .filter(|p| p.x.abs() > 0.4 || p.z.abs() > 0.4)
            .map(|p| p.y)
            .fold(0.0_f32, f32::max)
            + 0.05;
        draft += (level - (w.body(id).unwrap().position.y - 0.15)) / samples as f32;
    }
    assert!(
        (draft - 0.15).abs() < 0.03,
        "draft {draft:.3} m on the GPU, expected 0.15"
    );
}

/// Newton's third law across the CPU/GPU boundary.
#[test]
fn gpu_momentum_is_exchanged_exactly() {
    let Some(mut gpu) = gpu() else { return };
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    w.fluid.fill_box(Vec3::splat(-0.3), Vec3::splat(0.3), &[]);
    let mut ball = RigidBody::new(Shape::Sphere { radius: 0.15 }, Vec3::new(-0.8, 0.0, 0.0));
    ball.density = 1000.0;
    ball.velocity = Vec3::new(2.0, 0.0, 0.0);
    let id = w.add_body(ball);
    let total = |w: &World| {
        let b = w.body(id).unwrap();
        w.fluid.momentum() + b.mass() * b.velocity
    };
    let before = total(&w);
    for _ in 0..120 {
        w.step_with(DT, &mut gpu);
    }
    assert!(
        w.body(id).unwrap().velocity.x < 1.8,
        "the ball should slow down"
    );
    let after = total(&w);
    assert!(
        (after - before).length() < 0.02 * before.length(),
        "momentum {before} -> {after}"
    );
}
