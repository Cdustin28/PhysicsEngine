//! 3D soft-body (XPBD) validation.

use engine3d::soft::{Edge, SoftBody};
use engine3d::{RigidBody, Shape, Vec3, World, presets};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine3d::STANDARD_GRAVITY;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn spring_chain(n: usize, m: f32, k: f32, l: f32) -> SoftBody {
    let positions: Vec<Vec3> = (0..=n)
        .map(|i| Vec3::new(0.0, -(i as f32) * l, 0.0))
        .collect();
    let mut inv_mass = vec![1.0 / m; n + 1];
    inv_mass[0] = 0.0;
    SoftBody {
        velocities: vec![Vec3::ZERO; n + 1],
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
        triangles: Vec::new(),
        rest_volume: 0.0,
        volume_stiffness: 0.0,
        radius: 0.01,
        friction: 0.0,
        damping: 2.0,
    }
}

#[test]
fn hanging_spring_chain_stretches_by_hookes_law() {
    // Total stretch m g n (n + 1) / (2k), the same as in 2D.
    let (n, m, k, l) = (5, 0.1, 200.0, 0.2);
    let expected = m * G * (n * (n + 1)) as f32 / (2.0 * k);
    for substeps in [10, 40] {
        let mut w = World::new();
        w.soft.substeps = substeps;
        w.soft_bodies.push(spring_chain(n, m, k, l));
        run(&mut w, 3.0);
        // Average over 2 s (the chain's slowest mode has a ~0.5 s period).
        let mut tip = 0.0;
        let samples = 240;
        for _ in 0..samples {
            w.step(DT);
            tip += w.soft_bodies[0].positions[n].y / samples as f32;
        }
        let stretch = -tip - n as f32 * l;
        assert!(
            (stretch - expected).abs() < 0.03 * expected,
            "{substeps} substeps: stretch {stretch:.5}, expected {expected:.5}"
        );
    }
}

#[test]
fn ball_lands_without_tunneling_keeps_its_volume_and_settles() {
    let mut w = presets::with_floor();
    w.soft_bodies
        .push(SoftBody::ball(Vec3::new(0.0, 2.5, 0.0), 0.5, 2, 200.0));
    let rest = w.soft_bodies[0].rest_volume;
    let (mut lowest, mut worst) = (f32::MAX, 0.0_f32);
    for _ in 0..(8.0 / DT) as usize {
        w.step(DT);
        let s = &w.soft_bodies[0];
        lowest = lowest.min(s.positions.iter().map(|p| p.y).fold(f32::MAX, f32::min));
        worst = worst.max((s.volume() / rest - 1.0).abs());
    }
    let r = w.soft_bodies[0].radius;
    assert!(lowest > 0.5 * r, "a particle reached y = {lowest}");
    assert!(worst < 0.05, "volume changed by {:.1}%", 100.0 * worst);
    // At rest: a ten-thousandth of the m g h it hit the floor with.
    let s = &w.soft_bodies[0];
    let impact = s.mass() * G * 2.0;
    let ke = s.kinetic_energy();
    assert!(ke < 1e-4 * impact, "still moving: {ke} J of {impact} J");
}

#[test]
fn cloth_drapes_over_a_ball_without_passing_through() {
    let mut w = presets::with_floor();
    let r = 0.6;
    let center = Vec3::new(0.0, r, 0.0);
    w.add_body(RigidBody::new_static(Shape::Sphere { radius: r }, center));
    w.soft_bodies.push(SoftBody::cloth(
        Vec3::new(0.0, 1.8, 0.0),
        2.0,
        2.0,
        16,
        16,
        0.4,
    ));
    run(&mut w, 4.0);
    let cloth = &w.soft_bodies[0];
    // No particle inside the ball, and the middle rests on top of it.
    let inside = cloth
        .positions
        .iter()
        .filter(|p| (**p - center).length() < r)
        .count();
    assert_eq!(inside, 0, "{inside} particles inside the ball");
    let middle = cloth.positions[cloth.len() / 2];
    assert!(
        (middle.y - (2.0 * r + cloth.radius)).abs() < 0.05,
        "middle at {middle}"
    );
}

#[test]
fn soft_and_rigid_bodies_exchange_momentum() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let mut jelly = SoftBody::jelly(Vec3::new(-1.2, 0.0, 0.0), Vec3::splat(0.4), 3, 1000.0);
    jelly.velocities = vec![Vec3::new(2.0, 0.0, 0.0); jelly.len()];
    jelly.friction = 0.0;
    w.soft_bodies.push(jelly);
    let mut crate_ = RigidBody::new(Shape::cube(0.4), Vec3::ZERO);
    crate_.density = 1000.0;
    crate_.friction = 0.0;
    let id = w.add_body(crate_);
    let total = |w: &World| w.soft_bodies[0].momentum() + w.momentum();
    let before = total(&w);
    run(&mut w, 1.5);
    assert!(
        w.body(id).unwrap().velocity.x > 0.5,
        "the crate should be knocked along"
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
    let loaded = engine3d::Scene::from_json(&engine3d::Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
}
