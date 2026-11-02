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

/// Several liquids and heat: one step from the same moving state (mercury
/// falling into oil over warm water, beside an 80 °C plate) agrees on
/// positions, temperatures, and the energy conduction moved.
#[test]
fn liquids_and_heat_match_cpu() {
    use engine3d::fluid::{Liquid, LiquidSpec};
    let Some(mut gpu) = gpu() else { return };
    let mut w = presets::tank(1.0, 1.0, 1.2);
    let mut plate = RigidBody::new_static(Shape::cuboid(0.4, 0.04, 0.4), Vec3::new(0.0, 0.02, 0.0));
    plate.temperature = Some(353.15);
    w.add_body(plate);
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.04, -0.5),
        Vec3::new(0.5, 0.3, 0.5),
        &b,
        LiquidSpec::water().at(310.0),
    );
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.3, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
        &b,
        LiquidSpec::new(Liquid::Oil),
    );
    w.fluid.fill_sphere_as(
        Vec3::new(0.0, 0.75, 0.0),
        0.15,
        &b,
        LiquidSpec::new(Liquid::Mercury).at(330.0),
    );
    for _ in 0..20 {
        w.step(DT);
    }
    let mut on_cpu = w.clone();
    let mut on_gpu = w;
    let e0 = on_cpu.fluid.thermal_energy();
    on_cpu.step_with(DT, &mut CpuSolver);
    on_gpu.step_with(DT, &mut gpu);

    let d = on_cpu.fluid.params.particle_spacing;
    // Match particles by nearest position (the solvers order them
    // differently).
    let pos_err = on_cpu
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
        pos_err < 0.05 * d,
        "positions differ by {pos_err} m on average"
    );
    // Each liquid's mean temperature. (Particles of different liquids too
    // finely mixed to separate trade places, and which pair trades depends
    // on exact positions, so particle-by-particle temperatures aren't
    // comparable; per-liquid means are.)
    let mean_t = |w: &World, l: Liquid| {
        let (s, n) = (0..w.fluid.len())
            .filter(|&i| w.fluid.liquid(i) == l)
            .fold((0.0, 0), |(s, n), i| (s + w.fluid.temperatures[i], n + 1));
        s / f64::from(n)
    };
    for l in Liquid::ALL {
        let (tc, tg) = (mean_t(&on_cpu, l), mean_t(&on_gpu, l));
        assert!(
            (tc - tg).abs() < 1e-3,
            "{l:?}: mean {tc} K (CPU) vs {tg} K (GPU)"
        );
    }

    // Heat moved by conduction (mostly in from the plate) agrees.
    let (dc, dg) = (
        on_cpu.fluid.thermal_energy() - e0,
        on_gpu.fluid.thermal_energy() - e0,
    );
    assert!(dc.abs() > 1.0, "no heat moved ({dc} J)");
    assert!(
        (dg / dc - 1.0).abs() < 0.05,
        "heat in {dc} J (CPU) vs {dg} J (GPU)"
    );

    let force = |w: &World| w.fluid_forces.iter().map(|f| f.0).sum::<Vec3>();
    let (fc, fg) = (force(&on_cpu), force(&on_gpu));
    assert!(
        (fc - fg).length() < 0.03 * fc.length(),
        "net fluid force {fc} (CPU) vs {fg} (GPU)"
    );
}

/// Mercury sinks through water on the GPU as on the CPU.
#[test]
fn gpu_mercury_sinks_through_water() {
    use engine3d::fluid::{Liquid, LiquidSpec};
    let Some(mut gpu) = gpu() else { return };
    let mut w = presets::tank(0.8, 0.8, 1.0);
    let b = w.bodies.clone();
    w.fluid
        .fill_box(Vec3::new(-0.4, 0.0, -0.4), Vec3::new(0.4, 0.4, 0.4), &b);
    w.fluid.fill_box_as(
        Vec3::new(-0.4, 0.4, -0.4),
        Vec3::new(0.4, 0.5, 0.4),
        &b,
        LiquidSpec::new(Liquid::Mercury),
    );
    let mercury_height = |w: &World| {
        let (s, n) = (0..w.fluid.len())
            .filter(|&i| w.fluid.liquid(i) == Liquid::Mercury)
            .fold((0.0, 0), |(s, n), i| (s + w.fluid.positions[i].y, n + 1));
        s / n as f32
    };
    let start = mercury_height(&w);
    for _ in 0..(3.0 / DT) as usize {
        w.step_with(DT, &mut gpu);
    }
    let end = mercury_height(&w);
    assert!(end < 0.15, "mercury from {start} m to {end} m");
}
