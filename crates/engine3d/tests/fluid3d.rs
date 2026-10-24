//! 3D SPH validation: hydrostatics, leaks, buoyancy, momentum exchange.
//!
//! Resolution is the default 0.1 m particle spacing, so each test has about
//! 1,000 particles and runs in a few seconds.

use engine3d::{RigidBody, Shape, Vec3, World, presets};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine3d::STANDARD_GRAVITY;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

/// A 1 m × 1 m tank filled `depth` deep.
fn still_tank(depth: f32) -> World {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_box(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, depth, 0.5),
        &bodies,
    );
    w
}

#[test]
fn still_water_has_hydrostatic_pressure_and_does_not_leak() {
    let mut w = still_tank(0.6);
    let n = w.fluid.len();
    assert_eq!(n, 600);
    run(&mut w, 2.0);
    // Average over the next second: single snapshots are noisy.
    let (mut sums, mut counts) = ([0.0_f64; 3], [0_u32; 3]);
    for _ in 0..120 {
        w.step(DT);
        // Free surface: half a spacing above the highest particle center.
        let surface = w
            .fluid
            .positions
            .iter()
            .map(|p| p.y)
            .fold(0.0_f32, f32::max)
            + 0.05;
        for (p, &pressure) in w.fluid.positions.iter().zip(w.fluid.pressures()) {
            // Away from walls and floor, where boundary handling skews it.
            if p.x.abs() < 0.25 && p.z.abs() < 0.25 {
                for (k, depth) in [0.2_f32, 0.3, 0.4].into_iter().enumerate() {
                    if ((surface - p.y) - depth).abs() < 0.05 {
                        sums[k] += f64::from(pressure);
                        counts[k] += 1;
                    }
                }
            }
        }
    }
    for (k, depth) in [0.2_f32, 0.3, 0.4].into_iter().enumerate() {
        let measured = sums[k] / f64::from(counts[k]);
        let expected = f64::from(1000.0 * G * depth);
        assert!(
            (measured - expected).abs() < 0.15 * expected,
            "depth {depth} m: {measured:.0} Pa, expected ρgh = {expected:.0} Pa"
        );
    }
    assert_eq!(w.fluid.len(), n, "particles were lost");
    let outside = w
        .fluid
        .positions
        .iter()
        .filter(|p| p.x.abs() > 0.5 || p.z.abs() > 0.5 || p.y < 0.0)
        .count();
    assert_eq!(outside, 0, "{outside} particles leaked through the tank");
}

#[test]
fn dam_break_stays_in_the_tank() {
    let mut w = presets::dam_break();
    let n = w.fluid.len();
    run(&mut w, 2.0);
    // The front reaches the far wall at x = 1.5 within the first second.
    let front = w
        .fluid
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::MIN, f32::max);
    assert!(front > 1.3, "front at {front}");
    let outside = w
        .fluid
        .positions
        .iter()
        .filter(|p| p.x.abs() > 1.5 || p.z.abs() > 0.6 || p.y < 0.0)
        .count();
    assert_eq!((w.fluid.len(), outside), (n, 0));
    // Volume is conserved: 1,080 liters over 3.6 m² is 0.30 m deep, so once
    // it settles the particles' mean height is half that. (The mean, unlike
    // the highest particle, barely moves with the remaining slosh.)
    run(&mut w, 2.0);
    let mean_y = w.fluid.positions.iter().map(|p| p.y).sum::<f32>() / w.fluid.len() as f32;
    assert!((mean_y - 0.15).abs() < 0.02, "mean height {mean_y}");
}

#[test]
fn floating_cubes_sink_to_archimedes_depth() {
    // A 2 m × 1 m tank, 0.5 m of water, two 0.3 m cubes: pine (500 kg/m³)
    // should float with half its height under water and cork (250) with a
    // quarter.
    let mut w = presets::tank(2.0, 1.0, 1.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_box(
        Vec3::new(-1.0, 0.0, -0.5),
        Vec3::new(1.0, 0.5, 0.5),
        &bodies,
    );
    let mut ids = Vec::new();
    for (x, density) in [(-0.5, 500.0), (0.5, 250.0)] {
        let mut b = RigidBody::new(Shape::cube(0.3), Vec3::new(x, 0.75, 0.0));
        b.density = density;
        ids.push((w.add_body(b), density));
    }
    run(&mut w, 3.0);
    let mut drafts = [0.0_f32; 2];
    let samples = 120;
    for _ in 0..samples {
        w.step(DT);
        // Water level away from the cubes.
        let level = w
            .fluid
            .positions
            .iter()
            .filter(|p| p.x.abs() < 0.15 || p.x.abs() > 0.85)
            .map(|p| p.y)
            .fold(0.0_f32, f32::max)
            + 0.05;
        for (k, &(id, _)) in ids.iter().enumerate() {
            let bottom = w.body(id).unwrap().position.y - 0.15;
            drafts[k] += (level - bottom) / samples as f32;
        }
    }
    for (k, &(_, density)) in ids.iter().enumerate() {
        let expected = density / 1000.0 * 0.3;
        assert!(
            (drafts[k] - expected).abs() < 0.03,
            "density {density}: draft {:.3} m, expected {expected:.3} m",
            drafts[k]
        );
    }
}

#[test]
fn dense_body_sinks_to_the_floor() {
    let mut w = still_tank(0.5);
    let mut b = RigidBody::new(Shape::cube(0.3), Vec3::new(0.0, 0.75, 0.0));
    b.density = 2000.0;
    let id = w.add_body(b);
    run(&mut w, 3.0);
    let y = w.body(id).unwrap().position.y;
    // Resting on the floor (center 0.15 m up), not on a film of water.
    assert!((y - 0.15).abs() < 0.02, "center at {y}");
}

#[test]
fn fluid_and_body_exchange_momentum_exactly() {
    // No gravity, no walls: a ball flies into a free blob of water. Fluid
    // and ball push on each other with equal and opposite forces, so their
    // total momentum can't change.
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
    run(&mut w, 1.0);
    let after = total(&w);
    let ball_v = w.body(id).unwrap().velocity.x;
    assert!(
        ball_v < 1.8,
        "the ball should have been slowed, v = {ball_v}"
    );
    assert!(
        (after - before).length() < 0.01 * before.length(),
        "momentum {before} -> {after}"
    );
}
