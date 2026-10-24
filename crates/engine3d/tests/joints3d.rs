//! 3D joint validation against closed-form physics (120 Hz).

use engine3d::joint::{Anchor, Joint, add_chain};
use engine3d::{BodyId, RigidBody, Scene, Shape, Vec3, World, presets};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine3d::STANDARD_GRAVITY;

fn pendulum(l: f32, r: f32, degrees: f32) -> (World, BodyId) {
    let mut w = World::new();
    let a = degrees.to_radians();
    let p = Vec3::new(l * a.sin(), -l * a.cos(), 0.0);
    let id = w.add_body(RigidBody::new(Shape::Sphere { radius: r }, p));
    w.joints
        .push(Joint::rod(&w.bodies, id.0, p, None, Vec3::ZERO));
    (w, id)
}

#[test]
fn small_swing_pendulum_has_the_textbook_period() {
    // Ball on a rod: I = m l² + (2/5) m r², T = 2π √((l² + 0.4 r²) / (g l)).
    let (l, r) = (1.0, 0.05);
    let expected = std::f32::consts::TAU * ((l * l + 0.4 * r * r) / (G * l)).sqrt();
    let (mut w, id) = pendulum(l, r, 5.0);
    let mut crossings = Vec::new();
    let mut prev = w.body(id).unwrap().position.x;
    for step in 0..(12.0 / DT) as usize {
        w.step(DT);
        let x = w.body(id).unwrap().position.x;
        if prev < 0.0 && x >= 0.0 {
            crossings.push((step as f32 + prev / (prev - x)) * DT);
        }
        prev = x;
    }
    let period = (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32;
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
    let scale = w.body(id).unwrap().mass() * G * 0.5;
    let start = energy(&w);
    let mut history = Vec::new();
    for _ in 0..(30.0 / DT) as usize {
        w.step(DT);
        history.push(energy(&w));
    }
    let worst = history
        .iter()
        .map(|e| (e - start).abs())
        .fold(0.0, f32::max);
    assert!(
        worst < 0.02 * scale,
        "energy error {:.2}%",
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
}

#[test]
fn conical_pendulum_keeps_its_angular_momentum_and_cone() {
    // Gravity and the rod (through the pivot) exert no torque about the
    // vertical axis, so L_y = m (x v_z − z v_x) is conserved. Launched at the
    // steady-cone speed v = √(g l sin θ tan θ), it stays at θ.
    let (l, theta) = (1.5_f32, 40f32.to_radians());
    let p = Vec3::new(l * theta.sin(), -l * theta.cos(), 0.0);
    let mut w = World::new();
    let mut ball = RigidBody::new(Shape::Sphere { radius: 0.05 }, p);
    ball.velocity = Vec3::new(0.0, 0.0, (G * l * theta.sin() * theta.tan()).sqrt());
    let id = w.add_body(ball);
    w.joints
        .push(Joint::rod(&w.bodies, id.0, p, None, Vec3::ZERO));
    let ly = |w: &World| {
        let b = w.body(id).unwrap();
        b.position.x * b.velocity.z - b.position.z * b.velocity.x
    };
    let start = ly(&w);
    let mut worst_angle = 0.0_f32;
    for _ in 0..(10.0 / DT) as usize {
        w.step(DT);
        let q = w.body(id).unwrap().position;
        let angle = (q.x.hypot(q.z) / l).asin();
        worst_angle = worst_angle.max((angle - theta).abs());
    }
    assert!(
        (ly(&w) - start).abs() < 0.01 * start.abs(),
        "L_y {start} -> {}",
        ly(&w)
    );
    assert!(
        worst_angle < 1f32.to_radians(),
        "cone wandered {:.2}°",
        worst_angle.to_degrees()
    );
}

#[test]
fn swinging_chain_holds_together() {
    let mut w = World::new();
    w.solver.substeps = 4;
    add_chain(
        &mut w,
        Vec3::ZERO,
        Vec3::new(2.0, 0.0, 0.0),
        10,
        0.05,
        Anchor::World,
        Anchor::Free,
    );
    for _ in 0..(10.0 / DT) as usize {
        w.step(DT);
        for j in &w.joints {
            let (a, b) = j.anchors(&w.bodies);
            assert!(
                (a - b).length() < 3e-3,
                "pin opened by {}",
                (a - b).length()
            );
        }
    }
}

#[test]
fn spring_has_hookes_law_period() {
    // 1 kg on 40 N/m: T = 2π √(m/k) = 0.993 s.
    let (k, m) = (40.0, 1.0);
    let mut w = World::new();
    let mut ball = RigidBody::new(Shape::Sphere { radius: 0.1 }, Vec3::new(0.0, -1.0, 0.0));
    ball.density = m / (4.0 / 3.0 * std::f32::consts::PI * 0.001);
    let id = w.add_body(ball);
    w.joints.push(Joint::spring(
        &w.bodies,
        id.0,
        Vec3::new(0.0, -1.0, 0.0),
        None,
        Vec3::ZERO,
        k,
        0.0,
    ));
    let center = -1.0 - m * G / k;
    let mut crossings = Vec::new();
    let mut prev = -1.0 - center;
    for step in 0..(6.0 / DT) as usize {
        w.step(DT);
        let d = w.body(id).unwrap().position.y - center;
        if prev < 0.0 && d >= 0.0 {
            crossings.push((step as f32 + prev / (prev - d)) * DT);
        }
        prev = d;
    }
    let period = (crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32;
    let expected = std::f32::consts::TAU * (m / k).sqrt();
    assert!(
        (period - expected).abs() < 0.01 * expected,
        "period {period}, expected {expected}"
    );
}

#[test]
fn joints_survive_removal_and_scene_files() {
    let mut w = presets::pendulums();
    let loaded = Scene::from_json(&Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
    let before = w.joints.len();
    w.remove_body(BodyId(1)); // the conical pendulum's ball
    assert_eq!(w.joints.len(), before - 1);
    assert!(w.joints.iter().all(|j| j.validate(w.bodies.len()).is_ok()));
}
