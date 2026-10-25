//! Two-way fluid ↔ body coupling, checked against Archimedes' principle and
//! conservation of momentum.

use engine::{BodyId, RigidBody, Shape, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;
const WATER_DEPTH: f32 = 1.0;
const TANK_WIDTH: f32 = 2.4;

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

/// A 2.4 m tank with 1 m of still water, plus a `width` × `height` box of
/// the given density held just above the surface.
fn tank_with_box(width: f32, height: f32, density: f32, angle: f32) -> (World, BodyId) {
    let mut w = presets::tank(TANK_WIDTH, 2.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_rect(
        Vec2::new(-0.5 * TANK_WIDTH, 0.0),
        Vec2::new(0.5 * TANK_WIDTH, WATER_DEPTH),
        &bodies,
    );
    let mut body = RigidBody::new(
        Shape::rect(width, height),
        Vec2::new(0.0, WATER_DEPTH + 0.5 * height + 0.05),
    )
    .with_angle(angle);
    body.density = density;
    let id = w.add_body(body);
    (w, id)
}

/// Height of the free surface, measured away from the body: the mean of the
/// highest particle centers plus half a spacing (each particle stands for a
/// d × d cell).
fn water_level(w: &World, away_from_x: f32) -> f32 {
    let mut ys: Vec<f32> = w
        .fluid
        .positions
        .iter()
        .filter(|p| (p.x - away_from_x).abs() > 0.6)
        .map(|p| p.y)
        .collect();
    ys.sort_by(|a, b| b.total_cmp(a));
    let top = &ys[..12];
    top.iter().sum::<f32>() / top.len() as f32 + 0.5 * w.fluid.params.particle_spacing
}

/// Fraction of a floating box's height below the waterline, averaged over
/// `seconds` after settling.
fn submerged_fraction(density: f32) -> f32 {
    let (width, height) = (0.6, 0.4);
    let (mut w, id) = tank_with_box(width, height, density, 0.0);
    run(&mut w, 6.0);
    let samples = 120;
    let mut total = 0.0;
    for _ in 0..samples {
        w.step(DT);
        let b = w.body(id).unwrap();
        let level = water_level(&w, b.position.x);
        total += (level - (b.position.y - 0.5 * height)) / height;
    }
    total / samples as f32
}

fn check_floating(density: f32) {
    let water = presets::tank(1.0, 1.0).fluid.params.rest_density;
    let expected = density / water;
    let got = submerged_fraction(density);
    assert!(
        (got - expected).abs() < 0.08,
        "density {density}: {got:.3} of the box is under water, Archimedes says {expected:.3}"
    );
}

// Archimedes: a floating body sinks until it displaces its own weight of
// water, so the submerged fraction equals body density / water density.

#[test]
fn quarter_density_box_floats_a_quarter_submerged() {
    check_floating(2.5);
}

#[test]
fn half_density_box_floats_half_submerged() {
    check_floating(5.0);
}

#[test]
fn three_quarter_density_box_floats_three_quarters_submerged() {
    check_floating(7.5);
}

/// A body denser than water sinks to the bottom. (It ends up resting on a
/// film about one particle thick: a single layer of SPH particles squeezed
/// between two boundaries can't drain. See ROADMAP M3.)
#[test]
fn dense_box_sinks_to_the_bottom() {
    let (mut w, id) = tank_with_box(0.6, 0.4, 20.0, 0.0);
    let start = w.body(id).unwrap().position.y;
    run(&mut w, 4.0);
    let b = w.body(id).unwrap();
    let bottom = b.position.y - 0.2;
    let d = w.fluid.params.particle_spacing;
    assert!(
        start - b.position.y > 0.9,
        "only sank {} m",
        start - b.position.y
    );
    assert!(bottom < 1.5 * d, "bottom is {bottom:.3} m above the floor");
    assert!(b.velocity.length() < 0.05, "still moving at {}", b.velocity);
}

/// Archimedes again, from the water's side: sinking a body raises the water
/// level by (body area) / (tank width).
#[test]
fn sunk_box_raises_the_water_level_by_its_volume() {
    let (mut with_box, id) = tank_with_box(0.6, 0.4, 20.0, 0.0);
    let (mut without, _) = tank_with_box(0.6, 0.4, 20.0, 0.0);
    without.remove_body(id);
    run(&mut with_box, 5.0);
    run(&mut without, 5.0);
    // Average over 1 s: the splash from the drop takes a while to die down.
    let samples = 120;
    let mut rise = 0.0;
    for _ in 0..samples {
        with_box.step(DT);
        without.step(DT);
        let x = with_box.body(id).unwrap().position.x;
        rise += water_level(&with_box, x) - water_level(&without, x);
    }
    let rise = rise / samples as f32;
    let expected = 0.6 * 0.4 / TANK_WIDTH; // 0.1 m
    assert!(
        (rise - expected).abs() < 0.25 * expected,
        "water rose {rise:.3} m, expected {expected:.3} m"
    );
}

/// Buoyancy also produces torque: a wide, light box dropped in tilted
/// rights itself and floats flat.
#[test]
fn tilted_floating_box_rights_itself() {
    let (mut w, id) = tank_with_box(1.0, 0.3, 5.0, 0.5);
    run(&mut w, 6.0);
    let angle = w.body(id).unwrap().angle;
    // A 1.0 × 0.3 box is only stable lying flat, so it should end near 0.
    assert!(angle.abs() < 0.1, "box settled at {angle:.3} rad");
}

/// Fluid and body exchange momentum exactly: a box coasting into still
/// water (no gravity) slows down, and the water picks up what the box lost.
#[test]
fn momentum_is_exchanged_between_fluid_and_body() {
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
    let v0 = w.body(id).unwrap().velocity.x;
    run(&mut w, 0.5);
    let after = total(&w);
    let v1 = w.body(id).unwrap().velocity.x;
    assert!(v1 < 0.9 * v0, "box didn't slow down: {v0} -> {v1} m/s");
    assert!(
        (after - before).length() < 0.02 * before.length(),
        "momentum {before} -> {after}"
    );
}
