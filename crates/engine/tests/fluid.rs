//! Fluid validation: SPH results compared with hydrostatics and the dam-break
//! front-speed limit, plus no-leak checks.

use engine::{RigidBody, Shape, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(world: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        world.step(DT);
    }
}

fn column(width: f32, height: f32) -> World {
    let mut w = presets::tank(width, 2.0 * height);
    let bodies = w.bodies.clone();
    w.fluid.fill_rect(
        Vec2::new(-0.5 * width, 0.0),
        Vec2::new(0.5 * width, height),
        &bodies,
    );
    w
}

/// Mean pressure of particles in a horizontal band, away from the side walls.
fn band_pressure(w: &World, y_lo: f32, y_hi: f32, half_width: f32) -> f32 {
    let f = &w.fluid;
    let (sum, n) = f
        .positions
        .iter()
        .zip(f.pressures())
        .filter(|(p, _)| p.x.abs() < half_width && p.y > y_lo && p.y < y_hi)
        .fold((0.0, 0), |(s, n), (_, &p)| (s + p, n + 1));
    assert!(n > 5, "too few particles in band {y_lo}..{y_hi}");
    sum / n as f32
}

/// In still water, pressure grows with depth at dp/dy = ρ₀ g. Measured as
/// the difference between two depth bands, averaged over 1 s to smooth out
/// WCSPH pressure noise.
#[test]
fn hydrostatic_pressure_gradient_is_rho_g() {
    let mut w = column(1.0, 1.0);
    run(&mut w, 4.0);

    let rho_g = w.fluid.params.rest_density * engine::STANDARD_GRAVITY;
    let samples = 120;
    let mut ratio = 0.0;
    for _ in 0..samples {
        w.step(DT);
        let dp = band_pressure(&w, 0.2, 0.3, 0.35) - band_pressure(&w, 0.6, 0.7, 0.35);
        ratio += dp / (0.4 * rho_g);
    }
    ratio /= samples as f32;
    assert!((ratio - 1.0).abs() < 0.1, "dp/dy = {ratio:.3} × ρ₀g");
    assert!(
        w.fluid.kinetic_energy() < 0.1,
        "water still sloshing: KE = {}",
        w.fluid.kinetic_energy()
    );
    assert!(
        w.fluid.stats.mean_compression < 0.03,
        "mean compression {:.3}",
        w.fluid.stats.mean_compression
    );
}

/// Mass is conserved (nothing leaks or disappears), so the settled water
/// fills about the volume it started with.
#[test]
fn settled_column_keeps_its_volume() {
    let mut w = column(1.0, 1.0);
    let n = w.fluid.len();
    run(&mut w, 3.0);
    assert_eq!(w.fluid.len(), n);
    let d = w.fluid.params.particle_spacing;
    // Free-surface height estimate: top particle centers plus half a
    // spacing, averaged over the middle of the tank.
    let top_layer: Vec<f32> = {
        let mut ys: Vec<f32> = w
            .fluid
            .positions
            .iter()
            .filter(|p| p.x.abs() < 0.35)
            .map(|p| p.y)
            .collect();
        ys.sort_by(|a, b| b.total_cmp(a));
        ys.into_iter().take(10).collect()
    };
    let surface = top_layer.iter().sum::<f32>() / top_layer.len() as f32 + 0.5 * d;
    // Volume per particle is d² at rest; the tank is 1 m wide.
    let expected = n as f32 * d * d / 1.0;
    assert!(
        (surface - expected).abs() < 0.1 * expected,
        "surface at {surface:.3} m, expected about {expected:.3} m"
    );
}

/// A released column spreads along the floor. Its front can't outrun the
/// ideal (frictionless, shallow-water) dam-break front, 2√(gH) (Ritter
/// 1892), but it should cover a good fraction of that distance.
#[test]
fn dam_break_front_is_bounded_by_ritter_speed() {
    let height = 1.0;
    let mut w = presets::tank(4.0, 3.0);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-2.0, 0.0), Vec2::new(-1.0, height), &bodies);
    let start = w
        .fluid
        .positions
        .iter()
        .map(|p| p.x)
        .fold(f32::MIN, f32::max);
    let t = 0.5;
    run(&mut w, t);
    let front = w
        .fluid
        .positions
        .iter()
        .filter(|p| p.y < 0.3)
        .map(|p| p.x)
        .fold(f32::MIN, f32::max);
    let ritter = 2.0 * (engine::STANDARD_GRAVITY * height).sqrt() * t;
    let moved = front - start;
    assert!(
        moved < ritter,
        "front moved {moved:.2} m, faster than the ideal {ritter:.2} m"
    );
    assert!(
        moved > 0.3 * ritter,
        "front moved only {moved:.2} m (ideal {ritter:.2} m)"
    );
}

/// After a violent dam break hits the far wall, every particle is still in
/// the tank.
#[test]
fn dam_break_does_not_leak() {
    let mut w = presets::tank(4.0, 3.0);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-2.0, 0.0), Vec2::new(-1.0, 1.5), &bodies);
    let n = w.fluid.len();
    run(&mut w, 3.0);
    assert_eq!(w.fluid.len(), n, "particles were lost");
    let outside: Vec<Vec2> = w
        .fluid
        .positions
        .iter()
        .copied()
        .filter(|p| p.x < -2.0 || p.x > 2.0 || p.y < 0.0)
        .collect();
    assert!(
        outside.is_empty(),
        "{} particles escaped, e.g. {:?}",
        outside.len(),
        outside.first()
    );
}

/// A blob fired at the floor at 8 m/s splashes but doesn't tunnel into it.
#[test]
fn fast_fluid_does_not_tunnel_into_the_floor() {
    let mut w = presets::tank(4.0, 3.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_disc(Vec2::new(0.0, 0.8), 0.3, &bodies);
    for v in &mut w.fluid.velocities {
        *v = Vec2::new(0.0, -8.0);
    }
    let mut lowest = f32::MAX;
    for _ in 0..(1.0 / DT) as usize {
        w.step(DT);
        lowest = w.fluid.positions.iter().map(|p| p.y).fold(lowest, f32::min);
    }
    assert!(lowest > 0.0, "a particle went {lowest} m into the floor");
}

/// A body moving through still water pushes it: the fluid gains momentum in
/// the body's direction, and no particle ends up inside the body.
#[test]
fn moving_body_pushes_fluid() {
    let mut w = World::new();
    w.gravity = Vec2::ZERO;
    w.fluid
        .fill_rect(Vec2::new(0.0, -0.5), Vec2::new(1.0, 0.5), &[]);
    let mut paddle = RigidBody::new(Shape::rect(0.2, 0.6), Vec2::new(-0.3, 0.0));
    paddle.velocity = Vec2::new(2.0, 0.0);
    paddle.density = 1000.0; // heavy, so it keeps going after M3 coupling too
    let id = w.add_body(paddle);
    run(&mut w, 0.4);
    let p = w.fluid.momentum();
    assert!(p.x > 0.5, "fluid momentum {p}");
    let body = w.body(id).unwrap();
    let inside = w
        .fluid
        .positions
        .iter()
        .filter(|&&x| body.contains_point(x))
        .count();
    assert_eq!(inside, 0, "{inside} particles inside the paddle");
}
