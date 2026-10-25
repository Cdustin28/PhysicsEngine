//! The validation scenarios. Each runs one of the engines and compares it
//! with a closed-form answer, returning a chart and pass/fail checks.

use crate::plot::{BLUE, CARDINAL, Chart, GOLD, SLATE, Series};
use std::f64::consts::PI;

const DT: f32 = 1.0 / 120.0;
const G: f64 = 9.806_65;

pub struct Check {
    pub quantity: String,
    pub expected: String,
    pub measured: String,
    pub pass: bool,
}

pub struct Case {
    pub id: &'static str,
    pub title: &'static str,
    pub engine: &'static str,
    /// HTML paragraph(s) explaining the setup and the theory.
    pub explanation: String,
    pub chart: Option<String>,
    pub checks: Vec<Check>,
}

fn check(quantity: &str, expected: f64, measured: f64, tolerance: f64, unit: &str) -> Check {
    let error = (measured - expected).abs() / expected.abs().max(1e-12);
    Check {
        quantity: quantity.into(),
        expected: format!("{} {unit}", sig(expected)),
        measured: format!(
            "{} {unit} ({:+.2}%)",
            sig(measured),
            100.0 * (measured - expected) / expected.abs().max(1e-12)
        ),
        pass: error <= tolerance,
    }
}

/// Steps covering `seconds` (rounded: 1.5 / DT in f32 is 179.99998).
fn steps(seconds: f32) -> usize {
    (seconds / DT).round() as usize
}

/// A check with an absolute tolerance, for quantities that can be near zero.
fn check_abs(quantity: &str, expected: f64, measured: f64, tolerance: f64, unit: &str) -> Check {
    Check {
        quantity: quantity.into(),
        expected: format!("{} {unit}", sig(expected)),
        measured: format!("{} {unit} ({:+.3})", sig(measured), measured - expected),
        pass: (measured - expected).abs() <= tolerance,
    }
}

/// Four significant figures.
fn sig(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let digits = (3 - v.abs().log10().floor() as i32).max(0) as usize;
    format!("{v:.digits$}")
}

/// Complete elliptic integral of the first kind K(k), by the
/// arithmetic–geometric mean: K = π / (2 AGM(1, √(1 − k²))).
fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0, (1.0 - k * k).sqrt());
    for _ in 0..30 {
        let (an, bn) = (0.5 * (a + b), (a * b).sqrt());
        a = an;
        b = bn;
    }
    PI / (2.0 * a)
}

// ---------------------------------------------------------------------------
// Rigid bodies and joints

pub fn pendulum_period() -> Case {
    use engine::{Joint, RigidBody, Shape, Vec2, World};
    // Disk (radius r) on a massless rod (length l) from a fixed pivot. Its
    // exact period at amplitude θ₀ is T = 4 √(L/g) K(sin(θ₀/2)), with the
    // equivalent length L = (l² + r²/2) / l for the disk's own inertia.
    let (l, r) = (1.0_f32, 0.02_f32);
    let length = f64::from((l * l + 0.5 * r * r) / l);
    let t0 = 2.0 * PI * (length / G).sqrt();
    let theory =
        |a_deg: f64| 4.0 * (length / G).sqrt() * elliptic_k((a_deg.to_radians() / 2.0).sin());
    let mut measured = Vec::new();
    let mut checks = Vec::new();
    for amp in [5.0_f32, 20.0, 40.0, 60.0, 80.0] {
        let mut w = World::new();
        let a = amp.to_radians();
        let p = Vec2::new(l * a.sin(), -l * a.cos());
        let id = w.add_body(RigidBody::new(Shape::Circle { radius: r }, p));
        w.joints
            .push(Joint::rod(&w.bodies, id.0, p, None, Vec2::ZERO));
        let mut crossings = Vec::new();
        let mut prev = p.x;
        for step in 0..steps(16.0) {
            w.step(DT);
            let x = w.bodies[id.0].position.x;
            if prev < 0.0 && x >= 0.0 {
                crossings.push((step as f32 + prev / (prev - x)) * DT);
            }
            prev = x;
        }
        let period =
            f64::from((crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32);
        measured.push((f64::from(amp), period / t0));
        checks.push(check(
            &format!("Period at {amp}°"),
            theory(f64::from(amp)),
            period,
            0.005,
            "s",
        ));
    }
    let curve: Vec<(f64, f64)> = (0..=85)
        .map(|a| (f64::from(a), theory(f64::from(a)) / t0))
        .collect();
    Case {
        id: "pendulum-period",
        title: "Pendulum period vs. amplitude",
        engine: "2D rigid bodies + joints",
        explanation: format!(
            "<p>A disk on a 1 m rod from a fixed pin, released from rest at 5° to 80°. Small swings take \
             T₀ = 2π√(L/g) = {t0:.4} s; larger ones take longer, exactly T = 4√(L/g)·K(sin(θ₀/2)) \
             where K is the complete elliptic integral of the first kind. That curve is far from the \
             small-angle formula at 80° (+{:.1}%), so matching it tests the full nonlinear motion and the \
             joint solver's energy behavior.</p>",
            100.0 * (theory(80.0) / t0 - 1.0)
        ),
        chart: Some(
            Chart::new("Amplitude θ₀ (degrees)", "Period / small-swing period")
                .with(Series::line("Exact: 4√(L/g) K(sin θ₀/2)", curve, CARDINAL))
                .with(Series::points("Simulated", measured, GOLD))
                .svg(),
        ),
        checks,
    }
}

pub fn pendulum_energy() -> Case {
    use engine::{Joint, RigidBody, Shape, Vec2, World};
    let mut w = World::new();
    let a = 60f32.to_radians();
    let p = Vec2::new(a.sin(), -a.cos());
    let id = w.add_body(RigidBody::new(Shape::Circle { radius: 0.05 }, p));
    w.joints
        .push(Joint::rod(&w.bodies, id.0, p, None, Vec2::ZERO));
    let energy = |w: &World| {
        let b = &w.bodies[id.0];
        f64::from(b.kinetic_energy() + b.mass() * G as f32 * b.position.y)
    };
    let scale = f64::from(w.bodies[id.0].mass()) * G * 0.5;
    let e0 = energy(&w);
    let mut series = Vec::new();
    let mut worst = 0.0_f64;
    let mut first = Vec::new();
    let mut last = Vec::new();
    let steps = steps(30.0);
    for step in 0..steps {
        w.step(DT);
        let rel = (energy(&w) - e0) / scale;
        worst = worst.max(rel.abs());
        if step % 4 == 0 {
            series.push((f64::from(step as f32 * DT), 100.0 * rel));
        }
        if step < 600 {
            first.push(rel);
        }
        if step >= steps - 600 {
            last.push(rel);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let drift = mean(&last) - mean(&first);
    Case {
        id: "pendulum-energy",
        title: "Pendulum energy over 30 s",
        engine: "2D rigid bodies + joints",
        explanation: "<p>The same pendulum swinging 60° for 30 s, with no friction: its energy should stay \
             constant. The joint solver aims each step at the joint's predicted end-of-step position \
             (a SHAKE-style scheme), so the error oscillates within a bounded band instead of steadily \
             draining. (The textbook velocity-only constraint loses 23% of this pendulum's energy in \
             10 s.)</p>"
            .into(),
        chart: Some(
            Chart::new("Time (s)", "Energy error (% of swing energy)")
                .with(Series::line("Simulated", series, CARDINAL))
                .svg(),
        ),
        checks: vec![
            Check {
                quantity: "Largest energy error".into(),
                expected: "< 2%".into(),
                measured: format!("{:.2}%", 100.0 * worst),
                pass: worst < 0.02,
            },
            Check {
                quantity: "Drift (last 5 s vs. first 5 s)".into(),
                expected: "< 0.5%".into(),
                measured: format!("{:+.3}%", 100.0 * drift),
                pass: drift.abs() < 0.005,
            },
        ],
    }
}

pub fn rolling_and_sliding() -> Case {
    use engine3d::{Quat, RigidBody, Shape, Vec3, World};
    // A 20° slab; a ball (μ = 1, so it rolls) and a box (μ = 0.2, slides).
    let theta = 20f32.to_radians();
    let make = |shape: Shape, half: f32, mu: f32| {
        let mut w = World::new();
        let tilt = Quat::from_rotation_z(theta);
        let mut ramp =
            RigidBody::new_static(Shape::cuboid(8.0, 0.3, 2.0), Vec3::ZERO).with_orientation(tilt);
        ramp.friction = mu;
        w.add_body(ramp);
        let mut b =
            RigidBody::new(shape, tilt * Vec3::new(3.0, 0.15 + half, 0.0)).with_orientation(tilt);
        b.friction = mu;
        let id = w.add_body(b);
        (w, id)
    };
    let a_roll = 5.0 / 7.0 * G * f64::from(theta.sin());
    // Box friction combines √(μ·μ) = μ.
    let a_slide = G * (f64::from(theta.sin()) - 0.2 * f64::from(theta.cos()));
    let mut series = Vec::new();
    let mut checks = Vec::new();
    for (name, (mut w, id), a, color) in [
        (
            "Ball, rolling",
            make(Shape::Sphere { radius: 0.3 }, 0.3, 1.0),
            a_roll,
            GOLD,
        ),
        (
            "Box, sliding (μ = 0.2)",
            make(Shape::cube(0.5), 0.25, 0.2),
            a_slide,
            BLUE,
        ),
    ] {
        let start = w.bodies[id.0].position;
        let mut points = Vec::new();
        for step in 1..=steps(1.5) {
            w.step(DT);
            if step % 6 == 0 {
                let d = f64::from((w.bodies[id.0].position - start).length());
                points.push((f64::from(step as f32 * DT), d));
            }
        }
        let t = 1.5;
        let d = points.last().unwrap().1;
        checks.push(check(
            &format!("{name}: distance after {t} s"),
            0.5 * a * t * t,
            d,
            0.03,
            "m",
        ));
        series.push(Series::points(name, points, color));
    }
    let curve = |a: f64| {
        (0..=30)
            .map(|k| (k as f64 * 0.05, 0.5 * a * (k as f64 * 0.05).powi(2)))
            .collect()
    };
    let mut chart = Chart::new("Time (s)", "Distance down the ramp (m)")
        .with(Series::line(
            "Rolling: ½ (5/7) g sin θ t²",
            curve(a_roll),
            CARDINAL,
        ))
        .with(Series::line(
            "Sliding: ½ g (sin θ − μ cos θ) t²",
            curve(a_slide),
            SLATE,
        ));
    for s in series {
        chart = chart.with(s);
    }
    Case {
        id: "ramp",
        title: "Rolling vs. sliding down a ramp",
        engine: "3D rigid bodies",
        explanation: "<p>A ball and a box released on a 20° slope. The ball rolls without slipping, so part of \
             its energy goes into spin: it accelerates at (5/7) g sin θ. The box slides against Coulomb \
             friction μ = 0.2: g (sin θ − μ cos θ). Both are distance-vs-time parabolas.</p>"
            .into(),
        chart: Some(chart.svg()),
        checks,
    }
}

pub fn bounce_heights() -> Case {
    use engine3d::{RigidBody, Shape, Vec3, presets};
    let h = 3.0_f32;
    let mut points = Vec::new();
    let mut checks = Vec::new();
    for e in [0.2_f32, 0.4, 0.6, 0.8, 0.9] {
        let mut w = presets::with_floor();
        w.bodies[0].restitution = 0.0;
        let r = 0.25;
        let mut ball = RigidBody::new(Shape::Sphere { radius: r }, Vec3::new(0.0, h + r, 0.0));
        ball.restitution = e;
        let id = w.add_body(ball);
        let (mut bounced, mut peak) = (false, 0.0_f32);
        for _ in 0..steps(4.0) {
            w.step(DT);
            let b = &w.bodies[id.0];
            if b.velocity.y > 0.0 {
                bounced = true;
                peak = peak.max(b.position.y - r);
            } else if bounced {
                break;
            }
        }
        let ratio = f64::from(peak / h);
        points.push((f64::from(e), ratio));
        checks.push(check_abs(
            &format!("Rebound height, e = {e}"),
            f64::from(e * e),
            ratio,
            0.03,
            "× h",
        ));
    }
    let curve = (0..=50)
        .map(|k| (k as f64 * 0.02, (k as f64 * 0.02).powi(2)))
        .collect();
    Case {
        id: "bounce",
        title: "Bounce height vs. restitution",
        engine: "3D rigid bodies",
        explanation: "<p>A ball dropped 3 m onto the floor with coefficient of restitution e leaves at e times \
             its impact speed, so it rises to e² of the drop height. The engine applies restitution to the \
             velocity at the start of the contact step, which includes that step's gravity, so lively \
             balls bounce about 2% higher than e²; the check allows ±0.03 h.</p>"
            .into(),
        chart: Some(
            Chart::new("Coefficient of restitution e", "Rebound height / drop height")
                .with(Series::line("e²", curve, CARDINAL))
                .with(Series::points("Simulated", points, GOLD))
                .svg(),
        ),
        checks,
    }
}

pub fn spring_period() -> Case {
    use engine::{Joint, RigidBody, Shape, Vec2, World};
    let k = 40.0_f32;
    let mut points = Vec::new();
    let mut checks = Vec::new();
    for m in [0.25_f32, 0.5, 1.0, 2.0, 4.0] {
        let mut w = World::new();
        let mut b = RigidBody::new(Shape::Circle { radius: 0.1 }, Vec2::new(0.0, -1.0));
        b.density = m / (std::f32::consts::PI * 0.01);
        let id = w.add_body(b);
        w.joints.push(Joint::spring(
            &w.bodies,
            id.0,
            Vec2::new(0.0, -1.0),
            None,
            Vec2::ZERO,
            k,
            0.0,
        ));
        let center = -1.0 - m * G as f32 / k;
        let mut crossings = Vec::new();
        let mut prev = -1.0 - center;
        for step in 0..steps(10.0) {
            w.step(DT);
            let d = w.bodies[id.0].position.y - center;
            if prev < 0.0 && d >= 0.0 {
                crossings.push((step as f32 + prev / (prev - d)) * DT);
            }
            prev = d;
        }
        let period =
            f64::from((crossings.last().unwrap() - crossings[0]) / (crossings.len() - 1) as f32);
        points.push((f64::from(m), period));
        let expected = 2.0 * PI * (f64::from(m / k)).sqrt();
        checks.push(check(
            &format!("Period, m = {m} kg"),
            expected,
            period,
            0.01,
            "s",
        ));
    }
    let curve = (1..=100).map(|i| {
        let m = 0.2 + i as f64 * 0.04;
        (m, 2.0 * PI * (m / f64::from(k)).sqrt())
    });
    Case {
        id: "spring",
        title: "Mass on a spring",
        engine: "2D joints (spring)",
        explanation: "<p>Masses from 0.25 to 4 kg hanging on a 40 N/m spring joint, released at the spring's \
             rest length. Each oscillates about the stretched equilibrium with period T = 2π√(m/k). The \
             spring is integrated implicitly (a soft constraint), so it's stable at any stiffness.</p>"
            .into(),
        chart: Some(
            Chart::new("Mass (kg)", "Period (s)")
                .with(Series::line("2π √(m/k)", curve.collect(), CARDINAL))
                .with(Series::points("Simulated", points, GOLD))
                .svg(),
        ),
        checks,
    }
}

// ---------------------------------------------------------------------------
// Soft bodies

pub fn xpbd_stiffness() -> Case {
    use engine::soft::{Edge, SoftBody};
    use engine::{Vec2, World};
    let (n, m, l) = (5usize, 0.1_f32, 0.2_f32);
    let chain = |k: f32| {
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
            damping: 3.0,
        }
    };
    let expected = |k: f64| f64::from(m) * G * (n * (n + 1)) as f64 / (2.0 * k);
    let mut series = Vec::new();
    let mut checks = Vec::new();
    for (substeps, color) in [(10u32, CARDINAL), (40, GOLD)] {
        let mut points = Vec::new();
        for k in [50.0_f32, 100.0, 200.0, 500.0, 1000.0, 2000.0] {
            let mut w = World::new();
            w.soft.substeps = substeps;
            w.soft_bodies.push(chain(k));
            for _ in 0..steps(6.0) {
                w.step(DT);
            }
            let mut tip = 0.0;
            for _ in 0..120 {
                w.step(DT);
                tip += f64::from(w.soft_bodies[0].positions[n].y) / 120.0;
            }
            let stretch = -tip - f64::from(n as f32 * l);
            points.push((f64::from(k), stretch));
            if k == 200.0 {
                checks.push(check(
                    &format!("Stretch at 200 N/m, {substeps} substeps"),
                    expected(200.0),
                    stretch,
                    0.03,
                    "m",
                ));
            }
        }
        let name = format!("Simulated, {substeps} substeps");
        series.push(if substeps == 10 {
            Series::rings(&name, points, color)
        } else {
            Series::points(&name, points, color)
        });
    }
    let curve = (0..=40).map(|i| {
        let k = 40.0 * 10f64.powf(i as f64 * 0.045);
        (k, expected(k))
    });
    let mut chart = Chart::new("Edge stiffness k (N/m)", "Total stretch (m)")
        .log_log()
        .with(Series::line(
            "Hooke: m g n(n+1)/(2k)",
            curve.collect(),
            SLATE,
        ));
    for s in series {
        chart = chart.with(s);
    }
    Case {
        id: "xpbd",
        title: "Soft-body stiffness is physical",
        engine: "2D soft bodies (XPBD)",
        explanation: "<p>Five 0.1 kg particles hanging from a pin by springs of stiffness k. Edge i carries the \
             weight of everything below it, so the chain stretches m g n(n+1)/(2k) in total. Plain \
             position-based dynamics gets stiffer the more iterations it runs; XPBD's compliance term \
             makes stiffness a real N/m, so 10 and 40 substeps give the same, correct answer.</p>"
            .into(),
        chart: Some(chart.svg()),
        checks,
    }
}

// ---------------------------------------------------------------------------
// Fluids

pub fn hydrostatic_pressure() -> Case {
    use engine::{Vec2, presets};
    let mut w = presets::tank(1.0, 2.0);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 1.0), &bodies);
    for _ in 0..steps(4.0) {
        w.step(DT);
    }
    let rho_g = f64::from(w.fluid.params.rest_density) * G;
    let d = w.fluid.params.particle_spacing;
    let bins = 9;
    let (mut sums, mut counts) = (vec![0.0_f64; bins], vec![0_u32; bins]);
    let samples = 120;
    for _ in 0..samples {
        w.step(DT);
        let surface = w
            .fluid
            .positions
            .iter()
            .map(|p| p.y)
            .fold(0.0_f32, f32::max)
            + 0.5 * d;
        for (p, &pr) in w.fluid.positions.iter().zip(w.fluid.pressures()) {
            if p.x.abs() < 0.35 {
                let depth = surface - p.y;
                let bin = (depth / 0.1) as usize;
                if (1..bins).contains(&bin) {
                    sums[bin] += f64::from(pr);
                    counts[bin] += 1;
                }
            }
        }
    }
    let points: Vec<(f64, f64)> = (1..bins)
        .filter(|&b| counts[b] > 0)
        .map(|b| ((b as f64 + 0.5) * 0.1, sums[b] / f64::from(counts[b])))
        .collect();
    // Gradient from a least-squares fit over the interior bins (2..7),
    // away from the surface and floor layers.
    let fit: Vec<&(f64, f64)> = points
        .iter()
        .filter(|(x, _)| *x > 0.2 && *x < 0.75)
        .collect();
    let n = fit.len() as f64;
    let (sx, sy) = (
        fit.iter().map(|p| p.0).sum::<f64>(),
        fit.iter().map(|p| p.1).sum::<f64>(),
    );
    let sxx = fit.iter().map(|p| p.0 * p.0).sum::<f64>();
    let sxy = fit.iter().map(|p| p.0 * p.1).sum::<f64>();
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let line = vec![(0.0, 0.0), (1.0, rho_g)];
    Case {
        id: "hydrostatic",
        title: "Hydrostatic pressure in still water",
        engine: "2D SPH fluid",
        explanation: format!(
            "<p>A 1 m column of water left to settle, then particle pressures averaged over 1 s in 0.1 m \
             depth bands away from the walls. Still water's pressure grows linearly with depth, \
             p = ρ₀ g h (ρ₀ = {:.0} kg/m², a 1 cm slab of water in 2D). The bands nearest the free surface \
             and the floor read low and high: SPH particles there have one-sided neighborhoods.</p>",
            w.fluid.params.rest_density
        ),
        chart: Some(
            Chart::new("Depth below the surface (m)", "Pressure (N/m)")
                .with(Series::line("ρ₀ g h", line, CARDINAL))
                .with(Series::points("Simulated (band means)", points, GOLD))
                .svg(),
        ),
        checks: vec![check(
            "Pressure gradient dp/dh (interior fit)",
            rho_g,
            slope,
            0.10,
            "N/m²",
        )],
    }
}

pub fn archimedes() -> Case {
    use engine3d::{RigidBody, Shape, Vec3, presets};
    let mut points = Vec::new();
    let mut checks = Vec::new();
    for density in [250.0_f32, 500.0, 750.0] {
        let mut w = presets::tank(1.2, 1.2, 1.0);
        let bodies = w.bodies.clone();
        w.fluid.fill_box(
            Vec3::new(-0.6, 0.0, -0.6),
            Vec3::new(0.6, 0.5, 0.6),
            &bodies,
        );
        let mut b = RigidBody::new(Shape::cube(0.3), Vec3::new(0.0, 0.75, 0.0));
        b.density = density;
        let id = w.add_body(b);
        for _ in 0..steps(3.0) {
            w.step(DT);
        }
        let mut draft = 0.0;
        let samples = 120;
        for _ in 0..samples {
            w.step(DT);
            let level = w
                .fluid
                .positions
                .iter()
                .filter(|p| p.x.abs() > 0.4 || p.z.abs() > 0.4)
                .map(|p| p.y)
                .fold(0.0_f32, f32::max)
                + 0.05;
            draft += f64::from(level - (w.bodies[id.0].position.y - 0.15)) / f64::from(samples);
        }
        let fraction = draft / 0.3;
        points.push((f64::from(density / 1000.0), fraction));
        checks.push(check(
            &format!("Submerged fraction, {density} kg/m³"),
            f64::from(density / 1000.0),
            fraction,
            0.10,
            "",
        ));
    }
    Case {
        id: "archimedes",
        title: "Archimedes' principle",
        engine: "3D SPH fluid + rigid bodies",
        explanation: "<p>0.3 m cubes of density 250, 500, and 750 kg/m³ floating in water (1000 kg/m³). A \
             floating body displaces its own weight of water, so the fraction under water equals the \
             density ratio. The water and the cubes push on each other with exactly opposite impulses.</p>"
            .into(),
        chart: Some(
            Chart::new("Density / water density", "Fraction submerged")
                .with(Series::line("ρ / ρ_water", vec![(0.0, 0.0), (1.0, 1.0)], CARDINAL))
                .with(Series::points("Simulated", points, GOLD))
                .svg(),
        ),
        checks,
    }
}

// ---------------------------------------------------------------------------
// Heat

pub fn heat_diffusion() -> Case {
    use engine::{Gas, Vec2};
    let (n, dx) = (64, 0.05_f32);
    let kappa = 0.01_f64;
    let s0 = 0.2_f64;
    let mut gas = Gas::new(Vec2::ZERO, dx, n, n);
    gas.params.thermal_diffusivity = kappa as f32;
    gas.params.open_top = false;
    let center = gas.cell_center(n / 2, n / 2);
    for j in 0..n {
        for i in 0..n {
            let r2 = f64::from((gas.cell_center(i, j) - center).length_squared());
            gas.heat[j * n + i] = (10.0 * (-r2 / (2.0 * s0 * s0)).exp()) as f32;
        }
    }
    let analytic = |x: f64, t: f64| {
        let s2 = s0 * s0 + 2.0 * kappa * t;
        10.0 * s0 * s0 / s2 * (-x * x / (2.0 * s2)).exp()
    };
    let profile = |gas: &Gas| -> Vec<(f64, f64)> {
        (0..n)
            .map(|i| {
                let x = f64::from(gas.cell_center(i, n / 2).x - center.x);
                (x, f64::from(gas.heat[(n / 2) * n + i]))
            })
            .collect()
    };
    let mut chart = Chart::new(
        "Distance from the center (m)",
        "Temperature above ambient (K)",
    )
    .include_zero();
    let colors = [GOLD, BLUE, SLATE];
    let mut checks = Vec::new();
    let mut t = 0.0;
    for (k, target) in [0.5_f64, 2.0, 5.0].into_iter().enumerate() {
        while t < target - 1e-9 {
            gas.step(&[], Vec2::ZERO, DT);
            t += f64::from(DT);
        }
        let curve = (-80..=80).map(|i| {
            let x = i as f64 * 0.02;
            (x, analytic(x, t))
        });
        chart = chart
            .with(Series::line(
                if k == 0 {
                    "Analytic: σ² = σ₀² + 2κt"
                } else {
                    ""
                },
                curve.collect(),
                CARDINAL,
            ))
            .with(Series::points(
                &format!("Simulated, t = {target} s"),
                profile(&gas),
                colors[k],
            ));
        let peak = f64::from(gas.heat[(n / 2) * n + n / 2]);
        checks.push(check(
            &format!("Peak at t = {target} s"),
            analytic(0.0, t),
            peak,
            0.02,
            "K",
        ));
    }
    Case {
        id: "heat",
        title: "Heat diffusion",
        engine: "2D smoke & heat grid",
        explanation: format!(
            "<p>A Gaussian hot spot (σ₀ = 0.2 m, 10 K) in still air with thermal diffusivity κ = {kappa} m²/s. \
             It stays Gaussian with variance σ² = σ₀² + 2κt, and in 2D its peak falls as σ₀²/σ² so \
             the total heat is unchanged. The solver diffuses heat implicitly, so it's stable at any κ \
             and step size, and its insulated walls conserve heat exactly.</p>"
        ),
        chart: Some(chart.svg()),
        checks,
    }
}

pub fn buoyant_start() -> Case {
    use engine::Vec2;
    use engine3d::Vec3;
    let g_prime = G * 10.0 / 293.15;
    // 2D: circular warm patch.
    let ratio_2d = {
        let (n, dx) = (96, 0.05);
        let mut gas = engine::Gas::new(Vec2::ZERO, dx, n, n);
        gas.params.open_top = false;
        gas.params.thermal_diffusivity = 0.0;
        let c = gas.cell_center(n / 2, n / 2);
        for j in 0..n {
            for i in 0..n {
                if (gas.cell_center(i, j) - c).length() < 0.4 {
                    gas.heat[j * n + i] = 10.0;
                }
            }
        }
        gas.step(&[], Vec2::new(0.0, -G as f32), DT);
        let (mut s, mut k) = (0.0, 0);
        for j in 0..n {
            for i in 0..n {
                if (gas.cell_center(i, j) - c).length() < 0.25 {
                    s += f64::from(gas.velocity_at_cell(i, j).y);
                    k += 1;
                }
            }
        }
        s / f64::from(k) / (g_prime * f64::from(DT))
    };
    // 3D: spherical warm patch.
    let ratio_3d = {
        let (n, dx) = (48, 0.06);
        let mut gas = engine3d::Gas::new(Vec3::ZERO, dx, n, n, n);
        gas.params.open_top = false;
        gas.params.thermal_diffusivity = 0.0;
        let c = gas.cell_center(n / 2, n / 2, n / 2);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    if (gas.cell_center(i, j, k) - c).length() < 0.4 {
                        let m = gas.idx(i, j, k);
                        gas.heat[m] = 10.0;
                    }
                }
            }
        }
        gas.step(&[], Vec3::new(0.0, -G as f32, 0.0), DT);
        let (mut s, mut q) = (0.0, 0);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    if (gas.cell_center(i, j, k) - c).length() < 0.25 {
                        s += f64::from(gas.velocity_at_cell(i, j, k).y);
                        q += 1;
                    }
                }
            }
        }
        s / f64::from(q) / (g_prime * f64::from(DT))
    };
    Case {
        id: "buoyancy",
        title: "Warm air starts to rise",
        engine: "2D and 3D smoke & heat grids",
        explanation: format!(
            "<p>A patch of air 10 K warmer than its surroundings is lighter (Boussinesq: g′ = g β ΔT with \
             β = 1/T for an ideal gas, here {g_prime:.4} m/s²), but it can't rise without pushing the \
             surrounding air out of the way. That air's inertia acts as added mass: equal to the patch's \
             own for a 2D cylinder, half of it for a sphere. So the patch starts out accelerating at \
             g′/2 in 2D and 2g′/3 in 3D, a sharp test of the pressure solve.</p>"
        ),
        chart: None,
        checks: vec![
            check(
                "Initial acceleration, 2D cylinder",
                0.5,
                ratio_2d,
                0.05,
                "× g′",
            ),
            check(
                "Initial acceleration, 3D sphere",
                2.0 / 3.0,
                ratio_3d,
                0.05,
                "× g′",
            ),
        ],
    }
}

pub type Scenario = fn() -> Case;

// ---------------------------------------------------------------------------
// Orbits

pub fn kepler() -> Case {
    use engine3d::orbits::{AU, DAY, G, M_SUN, YEAR, solar_system, sun_only};
    use std::f64::consts::TAU;
    // Light test planets, one at a time, from Mercury's distance to
    // Neptune's. Period measured from the angle swept over two orbits.
    let mut points = Vec::new();
    let mut checks = Vec::new();
    for a_au in [0.387, 0.723, 1.0, 1.524, 5.2, 9.54, 19.2, 30.1] {
        let mut s = sun_only();
        let p = s.add_planet("Test", 1e20, 1e6, [255, 255, 255], a_au * AU, 0.0);
        let angle = |s: &engine3d::orbits::OrbitSystem| {
            let d = s.bodies[p].position - s.bodies[0].position;
            d.z.atan2(d.x)
        };
        let (mut prev, mut swept) = (angle(&s), 0.0_f64);
        let target = 2.0 * TAU;
        let period = loop {
            let dt = s.stable_dt();
            s.step(dt);
            let a = angle(&s);
            let mut d = a - prev;
            d -= TAU * (d / TAU).round();
            if swept.abs() + d.abs() >= target {
                let frac = (target - swept.abs()) / d.abs();
                break (s.time - dt + frac * dt) / 2.0;
            }
            swept += d;
            prev = a;
        };
        let expected = TAU * ((a_au * AU).powi(3) / (G * (M_SUN + 1e20))).sqrt();
        points.push((a_au, period / YEAR));
        if [0.387, 1.0, 30.1].contains(&a_au) {
            checks.push(check(
                &format!("Period at {a_au} AU"),
                expected / YEAR,
                period / YEAR,
                5e-4,
                "years",
            ));
        }
    }
    // The real Solar System, all bodies pulling on each other, for 20 years.
    let mut s = solar_system();
    let (e0, l0) = (s.energy(), s.angular_momentum());
    let end = 20.0 * YEAR;
    while s.time < end {
        s.advance((end - s.time).min(DAY), 100_000);
    }
    let de = ((s.energy() - e0) / e0).abs();
    let dl = (s.angular_momentum() - l0).length() / l0.length();
    checks.push(Check {
        quantity: "Solar System energy change, 20 years".into(),
        expected: "< 10⁻⁶".into(),
        measured: format!("{de:.1e}"),
        pass: de < 1e-6,
    });
    checks.push(Check {
        quantity: "Solar System angular momentum change, 20 years".into(),
        expected: "< 10⁻⁹".into(),
        measured: format!("{dl:.1e}"),
        pass: dl < 1e-9,
    });
    let curve = (0..=40)
        .map(|k| {
            let a = 0.3 * 10f64.powf(k as f64 * 0.05);
            (a, a.powf(1.5))
        })
        .collect();
    Case {
        id: "kepler",
        title: "Kepler's third law",
        engine: "N-body orbits",
        explanation: "<p>Planets placed on circular orbits from Mercury's distance to Neptune's, each              timed over two orbits. Kepler's third law says the period grows as the distance to the              1.5 power: T = 2π√(a³/GM), one year at 1 AU. The integrator (leapfrog, at least 256              steps per orbit) runs orbits long by a fixed (2π/256)²/3 = 0.02%, and being              symplectic, its energy error stays bounded instead of drifting: the full Solar              System, with every planet and moon pulling on every other, is checked over 20              years.</p>"
            .into(),
        chart: Some(
            Chart::new("Distance from the Sun (AU)", "Orbital period (years)")
                .log_log()
                .with(Series::line("T = a^1.5 (years, AU)", curve, CARDINAL))
                .with(Series::points("Simulated", points, GOLD))
                .svg(),
        ),
        checks,
    }
}

/// Every case, in page order.
pub fn all() -> Vec<(&'static str, Scenario)> {
    vec![
        ("pendulum period", pendulum_period),
        ("pendulum energy", pendulum_energy),
        ("ramp", rolling_and_sliding),
        ("bounce", bounce_heights),
        ("spring", spring_period),
        ("xpbd", xpbd_stiffness),
        ("hydrostatic", hydrostatic_pressure),
        ("archimedes", archimedes),
        ("heat", heat_diffusion),
        ("buoyancy", buoyant_start),
        ("kepler", kepler),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elliptic_k_matches_known_values() {
        // K(0) = π/2; K(sin 45°) = 1.854075 (tabulated).
        assert!((elliptic_k(0.0) - PI / 2.0).abs() < 1e-12);
        assert!((elliptic_k(std::f64::consts::FRAC_1_SQRT_2) - 1.854_075).abs() < 1e-6);
    }
}
