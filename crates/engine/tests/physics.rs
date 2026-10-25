//! Physics validation: every test compares the simulation against an answer
//! known from mechanics, not against "it looked right".

use engine::{RigidBody, Shape, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(world: &mut World, seconds: f32) {
    let steps = (seconds / DT).round() as usize;
    for _ in 0..steps {
        world.step(DT);
    }
}

fn zero_gravity() -> World {
    let mut w = World::new();
    w.gravity = Vec2::ZERO;
    w
}

/// Equal masses, head-on, perfectly elastic: the balls swap velocities.
/// (1D elastic collision: v1' = v2, v2' = v1 when m1 = m2.)
#[test]
fn elastic_head_on_collision_swaps_velocities() {
    let mut w = zero_gravity();
    let mut a = RigidBody::new(Shape::Circle { radius: 0.5 }, Vec2::new(-2.0, 0.0));
    a.velocity = Vec2::new(3.0, 0.0);
    a.restitution = 1.0;
    a.friction = 0.0;
    let mut b = RigidBody::new(Shape::Circle { radius: 0.5 }, Vec2::new(0.0, 0.0));
    b.restitution = 1.0;
    b.friction = 0.0;
    let (ia, ib) = (w.add_body(a), w.add_body(b));

    let p0 = w.momentum();
    let ke0 = w.kinetic_energy();
    run(&mut w, 1.0);

    let va = w.body(ia).unwrap().velocity;
    let vb = w.body(ib).unwrap().velocity;
    assert!(va.length() < 0.05, "A should stop, got {va}");
    assert!(
        (vb - Vec2::new(3.0, 0.0)).length() < 0.05,
        "B should take A's velocity, got {vb}"
    );
    assert!(
        (w.momentum() - p0).length() < 1e-3,
        "momentum {} -> {}",
        p0,
        w.momentum()
    );
    let ke = w.kinetic_energy();
    assert!(
        (ke - ke0).abs() / ke0 < 0.02,
        "kinetic energy {ke0} -> {ke}"
    );
}

/// Unequal masses, elastic: v1' = (m1 - m2)/(m1 + m2) v1,
/// v2' = 2 m1/(m1 + m2) v1.
#[test]
fn elastic_collision_with_unequal_masses() {
    let mut w = zero_gravity();
    let mut heavy = RigidBody::new(Shape::Circle { radius: 0.5 }, Vec2::new(-2.0, 0.0));
    heavy.density = 30.0; // 3x the mass of the light ball
    heavy.velocity = Vec2::new(2.0, 0.0);
    let mut light = RigidBody::new(Shape::Circle { radius: 0.5 }, Vec2::ZERO);
    for b in [&mut heavy, &mut light] {
        b.restitution = 1.0;
        b.friction = 0.0;
    }
    let (m1, m2) = (heavy.mass(), light.mass());
    let (ih, il) = (w.add_body(heavy), w.add_body(light));
    run(&mut w, 1.5);

    let v1 = (m1 - m2) / (m1 + m2) * 2.0; // = 1.0 m/s
    let v2 = 2.0 * m1 / (m1 + m2) * 2.0; // = 3.0 m/s
    let got1 = w.body(ih).unwrap().velocity.x;
    let got2 = w.body(il).unwrap().velocity.x;
    assert!((got1 - v1).abs() < 0.05, "heavy: expected {v1}, got {got1}");
    assert!((got2 - v2).abs() < 0.05, "light: expected {v2}, got {got2}");
}

/// Whatever the restitution, friction, or shapes, contact impulses come in
/// equal and opposite pairs, so total momentum never changes.
#[test]
fn momentum_is_conserved_in_glancing_box_circle_collisions() {
    let mut w = zero_gravity();
    let mut bx = RigidBody::new(Shape::rect(1.0, 0.6), Vec2::new(0.0, 0.0)).with_angle(0.3);
    bx.restitution = 0.2;
    bx.angular_velocity = 1.0;
    let mut ball = RigidBody::new(Shape::Circle { radius: 0.3 }, Vec2::new(-3.0, 0.25));
    ball.velocity = Vec2::new(4.0, 0.0);
    ball.density = 20.0;
    w.add_body(bx);
    w.add_body(ball);
    let p0 = w.momentum();
    run(&mut w, 2.0);
    assert!(
        !w.contacts().is_empty() || w.bodies[0].velocity.length() > 0.1,
        "the bodies should have collided"
    );
    assert!(
        (w.momentum() - p0).length() < 1e-3 * p0.length(),
        "momentum {} -> {}",
        p0,
        w.momentum()
    );
}

/// A box dropped onto the floor comes to rest on top of it, not inside it.
#[test]
fn box_comes_to_rest_on_floor() {
    let mut w = presets::bounce(); // just for the floor
    w.bodies.truncate(1);
    w.add_body(RigidBody::new(Shape::rect(1.0, 1.0), Vec2::new(0.0, 3.0)));
    run(&mut w, 3.0);
    let b = &w.bodies[1];
    let slop = w.solver.allowed_penetration;
    assert!(
        (b.position.y - 0.5).abs() < 2.0 * slop,
        "y = {}",
        b.position.y
    );
    assert!(b.angle.abs() < 1e-3, "box tipped over: angle {}", b.angle);
    assert!(b.velocity.length() < 0.01, "still moving: {}", b.velocity);
}

/// A tower of 10 boxes stands for 10 s without drifting or sinking.
#[test]
fn ten_box_stack_is_stable() {
    let mut w = presets::bounce();
    w.bodies.truncate(1);
    for i in 0..10 {
        w.add_body(RigidBody::new(
            Shape::rect(1.0, 1.0),
            Vec2::new(0.0, 0.5 + i as f32),
        ));
    }
    run(&mut w, 10.0);
    let top = w.bodies.last().unwrap();
    // Each of 10 contacts may settle into its slop allowance.
    let max_sink = 10.0 * 2.0 * w.solver.allowed_penetration;
    assert!(
        top.position.x.abs() < 0.02,
        "tower drifted sideways: x = {}",
        top.position.x
    );
    assert!(
        top.position.y > 9.5 - max_sink,
        "tower sank: top y = {}",
        top.position.y
    );
    assert!(
        w.kinetic_energy() < 1e-2,
        "tower still moving: KE = {}",
        w.kinetic_energy()
    );
}

/// The 55-box pyramid preset stands for 10 s.
#[test]
fn pyramid_is_stable() {
    let mut w = presets::pyramid();
    let start: Vec<Vec2> = w.bodies.iter().map(|b| b.position).collect();
    run(&mut w, 10.0);
    let worst = w
        .bodies
        .iter()
        .zip(&start)
        .map(|(b, s)| (b.position - *s).length())
        .fold(0.0_f32, f32::max);
    assert!(worst < 0.01, "a box moved {worst} m");
    assert!(
        w.kinetic_energy() < 0.05,
        "pyramid still moving: KE = {}",
        w.kinetic_energy()
    );
}

/// A block on a ramp stays put when tan(θ) < μ and slides when tan(θ) > μ.
/// With μ = 0.5 on both surfaces the combined μ is 0.5, so the threshold
/// angle is atan(0.5) ≈ 26.6°.
#[test]
fn friction_holds_below_critical_angle_and_slips_above() {
    let mut w = presets::ramp();
    // Bodies: floor, ramp 20°, block, ramp 35°, block.
    let (shallow, steep) = (2, 4);
    assert_eq!(w.bodies[shallow].friction, 0.5);
    let start_shallow = w.bodies[shallow].position;
    let start_steep = w.bodies[steep].position;
    run(&mut w, 1.0);
    let moved_shallow = (w.bodies[shallow].position - start_shallow).length();
    let moved_steep = (w.bodies[steep].position - start_steep).length();
    assert!(
        moved_shallow < 0.005,
        "block on 20° ramp slid {moved_shallow} m"
    );

    // On the steep ramp the block accelerates at g (sin θ - μ cos θ).
    let theta = 35f32.to_radians();
    let a = engine::STANDARD_GRAVITY * (theta.sin() - 0.5 * theta.cos());
    let expected = 0.5 * a; // distance after 1 s: a t² / 2
    assert!(
        (moved_steep - expected).abs() < 0.05 * expected,
        "block on 35° ramp slid {moved_steep} m, expected about {expected} m"
    );
}

/// A ball with restitution e dropped from height h bounces back to about
/// e² h. Restitution scales speed, and height goes as speed squared.
#[test]
fn bounce_height_follows_restitution() {
    for e in [0.5_f32, 0.9] {
        let mut w = presets::bounce();
        w.bodies.truncate(1);
        let r = 0.4;
        let drop = 3.0; // height of the ball's bottom above the floor
        let mut ball = RigidBody::new(Shape::Circle { radius: r }, Vec2::new(0.0, r + drop));
        ball.restitution = e;
        w.add_body(ball);

        // Wait until it's moving up, then record the peak.
        let mut bounced = false;
        let mut peak = 0.0_f32;
        for _ in 0..(3.0 / DT) as usize {
            w.step(DT);
            let b = &w.bodies[1];
            if b.velocity.y > 0.0 {
                bounced = true;
            }
            if bounced {
                peak = peak.max(b.position.y - r);
                if b.velocity.y < 0.0 {
                    break;
                }
            }
        }
        let expected = e * e * drop;
        assert!(bounced, "e = {e}: ball never bounced");
        assert!(
            (peak - expected).abs() < 0.1 * expected,
            "e = {e}: bounced to {peak} m, expected about {expected} m"
        );
    }
}

/// With restitution 0 a dropped ball does not bounce.
#[test]
fn zero_restitution_does_not_bounce() {
    let mut w = presets::bounce();
    w.bodies.truncate(2); // floor + the e = 0 ball
    assert_eq!(w.bodies[1].restitution, 0.0);
    let mut max_up = 0.0_f32;
    for _ in 0..(3.0 / DT) as usize {
        w.step(DT);
        max_up = max_up.max(w.bodies[1].velocity.y);
    }
    assert!(max_up < 0.2, "ball bounced up at {max_up} m/s");
}

/// Two dynamic bodies that start overlapping get pushed apart smoothly
/// rather than exploding.
#[test]
fn overlapping_bodies_separate_gently() {
    let mut w = zero_gravity();
    w.add_body(RigidBody::new(Shape::rect(1.0, 1.0), Vec2::ZERO));
    w.add_body(RigidBody::new(Shape::rect(1.0, 1.0), Vec2::new(0.6, 0.0)));
    run(&mut w, 2.0);
    let gap = w.bodies[1].position.x - w.bodies[0].position.x;
    assert!(
        gap > 1.0 - 2.0 * w.solver.allowed_penetration,
        "still overlapping: gap {gap}"
    );
    for b in &w.bodies {
        assert!(
            b.velocity.length() < 1.0,
            "pushed apart too violently: {}",
            b.velocity
        );
    }
}
