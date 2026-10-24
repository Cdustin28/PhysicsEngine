//! End-to-end checks of the 3D engine against closed-form physics.
//!
//! Each test builds its own small scene, runs it at the editor's fixed step
//! (120 Hz), and compares with the textbook answer. Tolerances are a few
//! times the error actually measured, so a regression shows up clearly.

use engine3d::STANDARD_GRAVITY as G;
use engine3d::body::{BodyId, RigidBody};
use engine3d::presets;
use engine3d::shape::Shape;
use engine3d::world::World;
use glam::{Quat, Vec3};

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn pos(w: &World, id: BodyId) -> Vec3 {
    w.body(id).unwrap().position
}

/// A static 6 m × 0.3 m × 2 m slab tilted `degrees` about z, centered at the
/// origin, plus a body of half-height `half` resting on it `along` meters
/// from the center (positive = uphill), both with friction `mu`.
fn on_ramp(degrees: f32, mu: f32, shape: Shape, half: f32, along: f32) -> (World, BodyId) {
    let mut w = World::new();
    let tilt = Quat::from_rotation_z(degrees.to_radians());
    let mut ramp =
        RigidBody::new_static(Shape::cuboid(6.0, 0.3, 2.0), Vec3::ZERO).with_orientation(tilt);
    ramp.friction = mu;
    w.add_body(ramp);
    let mut body =
        RigidBody::new(shape, tilt * Vec3::new(along, 0.15 + half, 0.0)).with_orientation(tilt);
    body.friction = mu;
    let id = w.add_body(body);
    (w, id)
}

#[test]
fn elastic_head_on_collision_swaps_velocities() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let mut a = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::new(-2.0, 0.0, 0.0));
    let mut b = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 0.0, 0.0));
    a.velocity = Vec3::new(3.0, 0.0, 0.0);
    a.restitution = 1.0;
    b.restitution = 1.0;
    let (ia, ib) = (w.add_body(a), w.add_body(b));
    run(&mut w, 1.0);
    let (va, vb) = (w.body(ia).unwrap().velocity, w.body(ib).unwrap().velocity);
    assert!(va.length() < 1e-3, "A should stop, has {va}");
    assert!(
        (vb - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-3,
        "B should carry 3 m/s, has {vb}"
    );
}

#[test]
fn momentum_is_conserved_without_gravity() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let shapes = [
        Shape::cube(1.0),
        Shape::Sphere { radius: 0.4 },
        Shape::cuboid(1.2, 0.5, 0.7),
        Shape::Sphere { radius: 0.6 },
    ];
    for (i, shape) in shapes.into_iter().enumerate() {
        let angle = i as f32 * std::f32::consts::FRAC_PI_2;
        let p = 3.0 * Vec3::new(angle.cos(), 0.3 * i as f32 - 0.4, angle.sin());
        let mut b =
            RigidBody::new(shape, p).with_orientation(Quat::from_rotation_x(0.3 * i as f32));
        b.velocity = -p * (0.8 + 0.2 * i as f32);
        b.angular_velocity = Vec3::new(0.5, -1.0, 0.2) * i as f32;
        w.add_body(b);
    }
    let before = w.momentum();
    run(&mut w, 3.0);
    let after = w.momentum();
    assert!(
        (after - before).length() < 1e-3 * before.length().max(1.0),
        "momentum {before} -> {after}"
    );
}

#[test]
fn box_comes_to_rest_on_floor() {
    let mut w = presets::with_floor();
    let id = w.add_body(RigidBody::new(Shape::cube(1.0), Vec3::new(0.0, 0.5, 0.0)));
    run(&mut w, 2.0);
    let b = w.body(id).unwrap();
    assert!((b.position.y - 0.5).abs() < 2e-3, "y = {}", b.position.y);
    assert!(b.velocity.length() < 1e-3 && b.angular_velocity.length() < 1e-3);
}

#[test]
fn tumbling_box_settles_flat() {
    let mut w = presets::with_floor();
    let id = w.add_body(
        RigidBody::new(Shape::cube(1.0), Vec3::new(0.0, 3.0, 0.0))
            .with_orientation(Quat::from_euler(glam::EulerRot::XYZ, 0.5, 0.3, 0.8)),
    );
    run(&mut w, 5.0);
    let b = w.body(id).unwrap();
    assert!((b.position.y - 0.5).abs() < 5e-3, "y = {}", b.position.y);
    assert!(w.kinetic_energy() < 1e-4, "KE = {}", w.kinetic_energy());
}

#[test]
fn six_box_tower_stands() {
    let mut w = presets::with_floor();
    let ids: Vec<BodyId> = (0..6)
        .map(|i| {
            w.add_body(RigidBody::new(
                Shape::cube(1.0),
                Vec3::new(0.0, 0.5 + i as f32, 0.0),
            ))
        })
        .collect();
    run(&mut w, 5.0);
    for (i, &id) in ids.iter().enumerate() {
        let p = pos(&w, id);
        let ideal = Vec3::new(0.0, 0.5 + i as f32, 0.0);
        assert!(
            (p - ideal).length() < 0.01,
            "box {i} at {p}, expected {ideal}"
        );
    }
    assert!(w.kinetic_energy() < 1e-3, "KE = {}", w.kinetic_energy());
}

#[test]
fn pyramid_holds_its_shape() {
    let mut w = presets::pyramid();
    let start: Vec<Vec3> = w.bodies.iter().map(|b| b.position).collect();
    run(&mut w, 5.0);
    let worst = w
        .bodies
        .iter()
        .zip(&start)
        .map(|(b, s)| (b.position - *s).length())
        .fold(0.0, f32::max);
    assert!(worst < 5e-3, "a cube moved {worst} m");
    assert!(w.kinetic_energy() < 1e-3);
}

#[test]
fn block_holds_below_friction_angle() {
    // tan 20° = 0.364 < μ = 0.5: static friction holds.
    let (mut w, id) = on_ramp(20.0, 0.5, Shape::cube(0.5), 0.25, 1.5);
    let start = pos(&w, id);
    run(&mut w, 1.0);
    let moved = (pos(&w, id) - start).length();
    assert!(moved < 5e-3, "slid {moved} m");
}

#[test]
fn block_slides_with_coulomb_acceleration() {
    // a = g (sin θ − μ cos θ) = 9.80665 × (0.5736 − 0.5 × 0.8192) = 1.608 m/s²
    // d = ½ a t² = 0.804 m after 1 s.
    let (theta, mu) = (35f32.to_radians(), 0.5);
    let expected = 0.5 * G * (theta.sin() - mu * theta.cos());
    let (mut w, id) = on_ramp(35.0, mu, Shape::cube(0.5), 0.25, 1.5);
    let start = pos(&w, id);
    run(&mut w, 1.0);
    let moved = (pos(&w, id) - start).length();
    assert!(
        (moved - expected).abs() < 0.03 * expected,
        "slid {moved} m, expected {expected}"
    );
    // It slides straight down without spinning about the ramp normal.
    let w_body = w.body(id).unwrap().angular_velocity;
    assert!(w_body.length() < 0.05, "spin {w_body}");
}

#[test]
fn ball_rolls_at_five_sevenths_g_sin_theta() {
    // Solid sphere rolling without slipping: a = 5/7 g sin θ.
    // 20°: 5/7 × 9.80665 × 0.3420 = 2.396 m/s², d = 1.198 m after 1 s.
    let theta = 20f32.to_radians();
    let expected = 0.5 * 5.0 / 7.0 * G * theta.sin();
    let r = 0.3;
    let (mut w, id) = on_ramp(20.0, 1.0, Shape::Sphere { radius: r }, r, 1.5);
    let start = pos(&w, id);
    run(&mut w, 1.0);
    let b = w.body(id).unwrap();
    let moved = (b.position - start).length();
    assert!(
        (moved - expected).abs() < 0.03 * expected,
        "rolled {moved} m, expected {expected}"
    );
    // No slip: spin matches v / r.
    let ratio = b.angular_velocity.length() * r / b.velocity.length();
    assert!((ratio - 1.0).abs() < 0.02, "ω r / v = {ratio}");
}

/// Highest point after the first bounce of a ball dropped from `h`.
fn bounce_peak(restitution: f32, h: f32) -> f32 {
    let mut w = presets::with_floor();
    let r = 0.25;
    let mut ball = RigidBody::new(Shape::Sphere { radius: r }, Vec3::new(0.0, h + r, 0.0));
    ball.restitution = restitution;
    let id = w.add_body(ball);
    let mut bounced = false;
    let mut peak = 0.0_f32;
    for _ in 0..(4.0 / DT) as usize {
        w.step(DT);
        let b = w.body(id).unwrap();
        if b.velocity.y > 0.0 {
            bounced = true;
            peak = peak.max(b.position.y - r);
        } else if bounced {
            break;
        }
    }
    peak
}

#[test]
fn bounce_height_scales_with_restitution_squared() {
    for (e, tol) in [(0.5, 0.02), (0.9, 0.04)] {
        let h = 3.0;
        let peak = bounce_peak(e, h);
        let expected = e * e * h;
        assert!(
            (peak - expected).abs() < tol * expected,
            "e = {e}: peak {peak}, expected {expected}"
        );
    }
}

#[test]
fn scene_round_trip_preserves_simulation() {
    let mut a = presets::demo();
    let json = engine3d::scene::Scene::from_world(&a).to_json();
    let mut b = engine3d::scene::Scene::from_json(&json).unwrap().world;
    run(&mut a, 1.0);
    run(&mut b, 1.0);
    for (x, y) in a.bodies.iter().zip(&b.bodies) {
        assert_eq!(x.position, y.position);
    }
}
