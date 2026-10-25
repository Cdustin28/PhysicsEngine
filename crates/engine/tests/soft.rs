//! Soft-body (XPBD) validation.

use engine::soft::{Edge, SoftBody};
use engine::{RigidBody, Shape, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine::STANDARD_GRAVITY;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

/// A vertical chain of `n` particles of mass `m` hanging from a pinned
/// particle by springs of stiffness `k` and rest length `l`.
fn spring_chain(n: usize, m: f32, k: f32, l: f32) -> SoftBody {
    let positions: Vec<Vec2> = (0..=n).map(|i| Vec2::new(0.0, -(i as f32) * l)).collect();
    let mut inv_mass = vec![1.0 / m; n + 1];
    inv_mass[0] = 0.0;
    SoftBody {
        velocities: vec![Vec2::ZERO; n + 1],
        edges: (0..n)
            .map(|i| Edge {
                i: i as u32,
                j: i as u32 + 1,
                rest: l,
            })
            .collect(),
        positions,
        inv_mass,
        edge_stiffness: k,
        outline: Vec::new(),
        triangles: Vec::new(),
        rest_area: 0.0,
        area_stiffness: 0.0,
        radius: 0.01,
        friction: 0.0,
        damping: 0.0,
    }
}

#[test]
fn hanging_spring_chain_stretches_by_hookes_law_at_any_substep_count() {
    // Edge i (from the top) carries the weight of the n − i masses below it,
    // so stretches (n − i) m g / k; in total m g n (n + 1) / (2k).
    // 5 × 0.1 kg on 200 N/m: 0.1 × 9.80665 × 15 / 200 = 0.07355 m.
    let (n, m, k, l) = (5, 0.1, 200.0, 0.2);
    let expected = m * G * (n * (n + 1)) as f32 / (2.0 * k);
    for substeps in [10, 40] {
        let mut w = World::new();
        w.soft.substeps = substeps;
        w.soft_bodies.push(spring_chain(n, m, k, l));
        // Let it settle; add a little damping by averaging the tip height.
        run(&mut w, 3.0);
        let mut tip = 0.0;
        let samples = 240;
        for _ in 0..samples {
            w.step(DT);
            tip += w.soft_bodies[0].positions[n].y / samples as f32;
        }
        let stretch = -tip - n as f32 * l;
        assert!(
            (stretch - expected).abs() < 0.03 * expected,
            "{substeps} substeps: stretch {stretch:.5} m, expected {expected:.5} m"
        );
    }
}

#[test]
fn single_spring_oscillates_at_its_natural_frequency() {
    // One 0.5 kg particle on a 50 N/m spring: T = 2π √(m/k) = 0.6283 s.
    let (m, k) = (0.5, 50.0);
    let mut w = World::new();
    w.gravity = Vec2::ZERO;
    let mut s = spring_chain(1, m, k, 1.0);
    s.positions[1].y = -1.1; // stretched 0.1 m
    s.damping = 0.0;
    w.soft_bodies.push(s);
    let mut crossings = Vec::new();
    let mut prev = 0.1_f32; // starts stretched
    for step in 0..(5.0 / DT) as usize {
        w.step(DT);
        let d = -w.soft_bodies[0].positions[1].y - 1.0;
        if prev < 0.0 && d >= 0.0 {
            crossings.push((step as f32 + prev / (prev - d)) * DT);
        }
        prev = d;
    }
    let period = (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32;
    let expected = std::f32::consts::TAU * (m / k).sqrt();
    assert!(
        (period - expected).abs() < 0.01 * expected,
        "period {period:.4} s, expected {expected:.4} s"
    );
}

#[test]
fn blob_lands_without_tunneling_and_keeps_its_area() {
    let mut w = presets::soft_bodies();
    // Keep just the floor and one blob, dropped from 3 m.
    w.bodies.truncate(1);
    w.joints.clear();
    w.soft_bodies = vec![SoftBody::blob(Vec2::new(0.0, 3.0), 0.5, 32, 1.0)];
    let rest = w.soft_bodies[0].rest_area;
    let mut lowest = f32::MAX;
    let mut worst_area = 0.0_f32;
    // A pressurized blob bounces like a balloon for a few seconds.
    for _ in 0..(8.0 / DT) as usize {
        w.step(DT);
        let s = &w.soft_bodies[0];
        lowest = lowest.min(s.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min));
        worst_area = worst_area.max((s.area() / rest - 1.0).abs());
    }
    let r = w.soft_bodies[0].radius;
    // Particle centers stay at least (nearly) a radius above the floor.
    assert!(lowest > 0.5 * r, "a particle reached y = {lowest}");
    assert!(
        worst_area < 0.05,
        "area changed by {:.1}%",
        100.0 * worst_area
    );
    // It comes to rest: under 0.01 J of the ≈ 19 J it hit the floor with
    // (m g h = 0.78 kg × 9.8 × 2.5 m).
    let ke = w.soft_bodies[0].kinetic_energy();
    assert!(ke < 1e-2, "still moving: {ke} J");
}

#[test]
fn soft_bodies_never_gain_energy() {
    let mut w = World::new();
    w.add_body(RigidBody::new_static(
        Shape::rect(20.0, 0.5),
        Vec2::new(0.0, -0.25),
    ));
    let mut jelly = SoftBody::block(Vec2::new(0.0, 2.0), 1.0, 0.6, 8, 5, 2.0);
    jelly.velocities = vec![Vec2::new(1.0, 0.0); jelly.len()];
    w.soft_bodies.push(jelly);
    let energy = |w: &World| {
        let s = &w.soft_bodies[0];
        s.kinetic_energy() + s.mass() * G * s.center().y
    };
    let start = energy(&w);
    let mut highest = start;
    for _ in 0..(5.0 / DT) as usize {
        w.step(DT);
        highest = highest.max(energy(&w));
    }
    // Elastic energy isn't counted, so allow the 1% that can be stored in
    // the springs at the start of a bounce.
    assert!(
        highest < start + 0.01 * start.abs(),
        "energy rose from {start} to {highest}"
    );
}

#[test]
fn soft_and_rigid_bodies_exchange_momentum() {
    // No gravity: a jelly block flies into a free crate. Their total
    // momentum can't change.
    let mut w = World::new();
    w.gravity = Vec2::ZERO;
    let mut jelly = SoftBody::block(Vec2::new(-1.5, 0.0), 0.6, 0.6, 6, 6, 2.0);
    jelly.velocities = vec![Vec2::new(2.0, 0.0); jelly.len()];
    jelly.friction = 0.0;
    w.soft_bodies.push(jelly);
    let mut crate_ = RigidBody::new(Shape::rect(0.5, 0.5), Vec2::new(0.0, 0.0));
    crate_.density = 2.0;
    crate_.friction = 0.0;
    let id = w.add_body(crate_);
    let total = |w: &World| w.soft_bodies[0].momentum() + w.momentum();
    let before = total(&w);
    run(&mut w, 1.5);
    let crate_v = w.body(id).unwrap().velocity.x;
    assert!(
        crate_v > 0.5,
        "the crate should be knocked along, v = {crate_v}"
    );
    let after = total(&w);
    assert!(
        (after - before).length() < 0.02 * before.length(),
        "momentum {before} -> {after}"
    );
}

#[test]
fn soft_bodies_round_trip_through_scene_files() {
    let w = presets::soft_bodies();
    let loaded = engine::Scene::from_json(&engine::Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
}
