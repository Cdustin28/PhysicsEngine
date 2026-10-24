//! Joint validation against closed-form physics (120 Hz, like the editor).

use engine::joint::{Anchor, Joint, add_chain};
use engine::{BodyId, RigidBody, Scene, Shape, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine::STANDARD_GRAVITY;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

/// A disk of radius `r` on a massless rod of length `l` from a fixed pivot
/// at the origin, released `degrees` from vertical.
fn pendulum(l: f32, r: f32, degrees: f32) -> (World, BodyId) {
    let mut w = World::new();
    let a = degrees.to_radians();
    let p = Vec2::new(l * a.sin(), -l * a.cos());
    let id = w.add_body(RigidBody::new(Shape::Circle { radius: r }, p));
    w.joints
        .push(Joint::rod(&w.bodies, id.0, p, None, Vec2::ZERO));
    (w, id)
}

/// Mean period from upward zero crossings of x over `seconds`.
fn measure_period(w: &mut World, id: BodyId, seconds: f32) -> f32 {
    let mut crossings = Vec::new();
    let mut prev = w.body(id).unwrap().position.x;
    for step in 0..(seconds / DT) as usize {
        w.step(DT);
        let x = w.body(id).unwrap().position.x;
        if prev < 0.0 && x >= 0.0 {
            // Interpolate the crossing time within the step.
            crossings.push((step as f32 + prev / (prev - x)) * DT);
        }
        prev = x;
    }
    (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32
}

#[test]
fn small_swing_pendulum_has_the_textbook_period() {
    // A disk on a rod is a physical pendulum: I/(m g l) with
    // I = m l² + ½ m r², so T = 2π √((l² + r²/2) / (g l)).
    let (l, r) = (1.0, 0.05);
    let expected = std::f32::consts::TAU * ((l * l + 0.5 * r * r) / (G * l)).sqrt();
    let (mut w, id) = pendulum(l, r, 5.0);
    let period = measure_period(&mut w, id, 12.0);
    assert!(
        (period - expected).abs() < 0.005 * expected,
        "period {period:.4} s, expected {expected:.4} s"
    );
}

#[test]
fn large_swing_pendulum_conserves_energy() {
    let (mut w, id) = pendulum(1.0, 0.05, 60.0);
    let energy = |w: &World| {
        let b = w.body(id).unwrap();
        b.kinetic_energy() + b.mass() * G * b.position.y
    };
    let start = energy(&w);
    let mut worst = 0.0_f32;
    let mut history = Vec::new();
    for _ in 0..(30.0 / DT) as usize {
        w.step(DT);
        let e = energy(&w);
        worst = worst.max((e - start).abs());
        history.push(e);
    }
    // Swing height is l (1 − cos 60°) = 0.5 m. The solver's energy error is
    // bounded (it wobbles about ±1.2%) rather than steadily draining: the
    // average over the last 5 s matches the first 5 s.
    let scale = w.body(id).unwrap().mass() * G * 0.5;
    assert!(
        worst < 0.02 * scale,
        "energy error reached {:.2}%",
        100.0 * worst / scale
    );
    let window = (5.0 / DT) as usize;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let drift = mean(&history[history.len() - window..]) - mean(&history[..window]);
    assert!(
        drift.abs() < 0.005 * scale,
        "energy drifted {:.2}%",
        100.0 * drift / scale
    );
    // And the rod keeps its length.
    let len = w.body(id).unwrap().position.length();
    assert!((len - 1.0).abs() < 1e-3, "rod length {len}");
}

#[test]
fn swinging_chain_holds_together() {
    let mut w = World::new();
    // Chains whip: the tip spins at up to 80 rad/s. 4 substeps keep the
    // pins within about a millimeter (1 substep: up to 1 cm).
    w.solver.substeps = 4;
    let ids = add_chain(
        &mut w,
        Vec2::ZERO,
        Vec2::new(2.0, 0.0),
        10,
        0.05,
        Anchor::World,
        Anchor::Free,
    );
    // Released horizontal, it swings (nothing damps it).
    let energy = |w: &World| {
        w.bodies
            .iter()
            .map(|b| b.kinetic_energy() + b.mass() * G * b.position.y)
            .sum::<f32>()
    };
    let start = energy(&w);
    let mut highest = start;
    for _ in 0..(10.0 / DT) as usize {
        w.step(DT);
        highest = highest.max(energy(&w));
        // Every pin's two anchors stay together.
        for j in &w.joints {
            let (a, b) = j.anchors(&w.bodies);
            assert!(
                (a - b).length() < 2e-3,
                "pin opened by {}",
                (a - b).length()
            );
        }
    }
    // The tip never gets farther than the chain's 2 m length from the pivot,
    // and the chain never gains energy.
    let last = &w.bodies[*ids.last().unwrap()];
    let tip = last.local_to_world(Vec2::new(0.1, 0.0));
    assert!(tip.length() < 2.01, "tip {} m from the pivot", tip.length());
    let mass: f32 = w.bodies.iter().map(|b| b.mass()).sum();
    assert!(
        highest - start < 0.01 * mass * G * 1.0,
        "chain gained {} J",
        highest - start
    );
}

#[test]
fn spring_has_hookes_law_stretch_and_period() {
    // 1 kg on a 40 N/m spring: stretch mg/k = 0.245 m, T = 2π √(m/k) = 0.993 s.
    let (k, m) = (40.0, 1.0);
    let mut w = World::new();
    let mut block = RigidBody::new(Shape::Circle { radius: 0.1 }, Vec2::new(0.0, -1.0));
    block.density = m / (std::f32::consts::PI * 0.01);
    let id = w.add_body(block);
    w.joints.push(Joint::spring(
        &w.bodies,
        id.0,
        Vec2::new(0.0, -1.0),
        None,
        Vec2::ZERO,
        k,
        0.0,
    ));
    // Released at the rest length, it oscillates about y = −1 − mg/k.
    let center = -1.0 - m * G / k;
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    let mut crossings = Vec::new();
    let mut prev = -1.0 - center;
    for step in 0..(6.0 / DT) as usize {
        w.step(DT);
        let y = w.body(id).unwrap().position.y;
        lo = lo.min(y);
        hi = hi.max(y);
        let d = y - center;
        if prev < 0.0 && d >= 0.0 {
            crossings.push((step as f32 + prev / (prev - d)) * DT);
        }
        prev = d;
    }
    let mid = 0.5 * (lo + hi);
    assert!(
        (mid - center).abs() < 0.01,
        "oscillates about {mid}, expected {center}"
    );
    let period = (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32;
    let expected = std::f32::consts::TAU * (m / k).sqrt();
    assert!(
        (period - expected).abs() < 0.01 * expected,
        "period {period:.4} s, expected {expected:.4} s"
    );
}

#[test]
fn slack_rope_does_nothing_until_taut() {
    let mut w = World::new();
    // A 1 m rope from the origin to a ball released 0.5 m below: slack.
    let id = w.add_body(RigidBody::new(
        Shape::Circle { radius: 0.05 },
        Vec2::new(0.0, -0.5),
    ));
    w.joints.push(Joint::rope(
        &w.bodies,
        id.0,
        Vec2::new(0.0, -0.5),
        None,
        Vec2::ZERO,
    ));
    if let engine::JointKind::Distance { max_length, .. } = &mut w.joints[0].kind {
        *max_length = 1.0;
    }
    // Free fall for 0.2 s: ½ g t² = 0.196 m (under the 0.5 m of slack).
    run(&mut w, 0.2);
    let y = w.body(id).unwrap().position.y;
    let n = (0.2 / DT).round();
    let fall = G * DT * DT * n * (n + 1.0) / 2.0; // semi-implicit Euler
    assert!((y - (-0.5 - fall)).abs() < 1e-4, "y = {y}");
    // Later it hangs at the rope's length, not past it.
    w.bodies[id.0].velocity = Vec2::ZERO;
    run(&mut w, 3.0);
    let len = w.body(id).unwrap().position.length();
    assert!(len < 1.002, "stretched to {len}");
}

#[test]
fn removing_a_body_removes_its_joints() {
    let mut w = presets::pendulums();
    let before = w.joints.len();
    // Body 1 is the simple pendulum's bob (body 0 is the floor).
    w.remove_body(BodyId(1));
    assert_eq!(w.joints.len(), before - 1);
    for j in &w.joints {
        assert!(j.validate(w.bodies.len()).is_ok());
    }
}

#[test]
fn joints_round_trip_through_scene_files() {
    let w = presets::rope_bridge();
    let loaded = Scene::from_json(&Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
    assert!(!loaded.world.joints.is_empty());
}

#[test]
fn newtons_cradle_passes_the_momentum_along() {
    let mut w = presets::newtons_cradle();
    // Wait for the first ball to strike and the last one to swing out.
    let mut peak_last = 0.0_f32;
    let mut peak_middle = 0.0_f32;
    for _ in 0..(1.0 / DT) as usize {
        w.step(DT);
        peak_last = peak_last.max(w.bodies[5].velocity.length());
        for i in 2..=4 {
            peak_middle = peak_middle.max(w.bodies[i].position.x - w.bodies[i].position.x.round());
        }
    }
    // Ball 1 dropped from 2 m (1 − cos 40°) = 0.468 m hits at √(2 g h) =
    // 3.03 m/s; the last ball should leave with most of that.
    assert!(
        peak_last > 0.9 * 3.03,
        "last ball reached only {peak_last:.2} m/s"
    );
}
