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

// ---------------------------------------------------------------------------
// Liquids, heat, and boiling (M12)

pub fn layered_hydrostatics() -> Case {
    use engine::fluid::Liquid;
    use engine::{Vec2, World, presets};
    // 0.3 m of mercury under 0.5 m of water.
    let mut w = presets::tank(1.0, 2.0);
    let b = w.bodies.clone();
    w.fluid.fill_rect_with(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.3),
        &b,
        Liquid::Mercury,
        0.0,
    );
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.3), Vec2::new(0.5, 0.8), &b);
    for _ in 0..steps(3.0) {
        w.step(DT);
    }
    let g = f64::from(w.gravity.length());
    let rho_w = f64::from(w.fluid.params.rest_density);
    let rho_hg = rho_w * 13.546;
    let band = |w: &World, lo: f32, hi: f32| {
        let (s, n) = (0..w.fluid.len())
            .filter(|&i| {
                let p = w.fluid.positions[i];
                p.x.abs() < 0.3 && p.y > lo && p.y < hi
            })
            .fold((0.0_f64, 0), |(s, n), i| {
                (s + f64::from(w.fluid.pressures()[i]), n + 1)
            });
        s / f64::from(n.max(1))
    };
    let edges: Vec<f32> = (0..16).map(|k| 0.05 * k as f32).collect();
    let mut sums = vec![0.0_f64; edges.len() - 1];
    let (mut grad_w, mut grad_hg) = (0.0, 0.0);
    let samples = 240;
    for _ in 0..samples {
        w.step(DT);
        for (k, s) in sums.iter_mut().enumerate() {
            *s += band(&w, edges[k], edges[k + 1]) / f64::from(samples);
        }
        grad_w += (band(&w, 0.45, 0.5) - band(&w, 0.65, 0.7)) / 0.2 / f64::from(samples);
        grad_hg += (band(&w, 0.12, 0.17) - band(&w, 0.22, 0.27)) / 0.1 / f64::from(samples);
    }
    let surface = 0.8;
    let points: Vec<(f64, f64)> = sums
        .iter()
        .enumerate()
        .map(|(k, &p)| (surface - f64::from(edges[k] + 0.025), p))
        .collect();
    let exact = |depth: f64| {
        if depth <= 0.5 {
            rho_w * g * depth
        } else {
            rho_w * g * 0.5 + rho_hg * g * (depth - 0.5)
        }
    };
    let curve: Vec<(f64, f64)> = (0..=80)
        .map(|k| (k as f64 * 0.01, exact(k as f64 * 0.01)))
        .collect();
    Case {
        id: "layered",
        title: "Pressure through two liquids",
        engine: "2D SPH fluid, several liquids",
        explanation: "<p>0.3 m of mercury under 0.5 m of water, settled, then pressures averaged over 2 s in \
             5 cm bands. Pressure grows at ρg in each liquid: 13.5× faster in the mercury. This needs the \
             multiphase formulation (each particle sums neighbors by number, not mass, so a water particle \
             beside mercury doesn't count mercury's mass as its own density). Bands at the floor and at the \
             interface read off: one-sided SPH neighborhoods there.</p>"
            .into(),
        chart: Some(
            Chart::new("Depth below the surface (m)", "Pressure (N/m)")
                .with(Series::line("ρ_water g h, then ρ_mercury g h", curve, CARDINAL))
                .with(Series::points("Simulated (band means)", points, GOLD))
                .svg(),
        ),
        checks: vec![
            check("dp/dh in the water", rho_w * g, grad_w, 0.15, "N/m²"),
            check("dp/dh in the mercury", rho_hg * g, grad_hg, 0.20, "N/m²"),
        ],
    }
}

pub fn conduction_in_liquid() -> Case {
    use engine::fluid::{Liquid, LiquidSpec, ROOM_TEMPERATURE};
    use engine::{RigidBody, Shape, Vec2, World, presets};
    // Mercury at rest, T = T₀ + A cos(2πx/L): the amplitude decays as
    // e^(−α k² t), α = k_th / (ρ c).
    // A closed box: without gravity, surface tension would pull a free
    // surface into a round drop and stir the liquid.
    let mut w = presets::tank(0.05, 0.06);
    w.add_body(RigidBody::new_static(
        Shape::rect(0.45, 0.2),
        Vec2::new(0.0, 0.05 + 0.1),
    ));
    w.gravity = Vec2::ZERO;
    w.fluid.params.particle_spacing = 0.0025;
    w.fluid.params.speed_of_sound = 5.0;
    let b = w.bodies.clone();
    w.fluid.fill_rect_as(
        Vec2::new(-0.025, 0.0),
        Vec2::new(0.025, 0.05),
        &b,
        LiquidSpec::new(Liquid::Mercury),
    );
    for _ in 0..steps(1.0) {
        w.step(DT);
    }
    let l = 0.05_f32;
    let k = std::f32::consts::TAU / l;
    for i in 0..w.fluid.len() {
        let x = w.fluid.positions[i].x;
        w.fluid.temperatures[i] = f64::from(ROOM_TEMPERATURE + 5.0 * (k * x).cos());
    }
    let amplitude = |w: &World| -> f64 {
        let n = w.fluid.len() as f64;
        let mean = w.fluid.temperatures.iter().sum::<f64>() / n;
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..w.fluid.len() {
            let c = f64::from((k * w.fluid.positions[i].x).cos());
            num += (w.fluid.temperatures[i] - mean) * c;
            den += c * c;
        }
        num / den
    };
    let alpha = f64::from(
        Liquid::Mercury.conductivity(0.0)
            / (Liquid::Mercury.density(0.0) * Liquid::Mercury.specific_heat(0.0)),
    );
    let kk = f64::from(k * k);
    let a0 = amplitude(&w);
    let mut points = vec![(0.0, 1.0)];
    for step in 1..=8 {
        for _ in 0..steps(0.25) {
            w.step(DT);
        }
        points.push((0.25 * f64::from(step), amplitude(&w) / a0));
    }
    // Least-squares slope of ln A against t, through A(0) = 1.
    let (sxy, sxx) = points
        .iter()
        .skip(1)
        .fold((0.0, 0.0), |(sxy, sxx), &(t, a)| {
            (sxy + t * a.ln(), sxx + t * t)
        });
    let rate = -sxy / sxx;
    let curve: Vec<(f64, f64)> = (0..=40)
        .map(|i| {
            let t = 0.05 * i as f64;
            (t, (-alpha * kk * t).exp())
        })
        .collect();
    Case {
        id: "conduction",
        title: "Heat conduction in a liquid",
        engine: "2D SPH fluid, heat",
        explanation: format!(
            "<p>Still mercury with a sinusoidal temperature pattern, 5 K amplitude over a 5 cm wavelength (20 particles per wavelength). \
             The heat equation says each Fourier mode decays independently, A(t) = A₀ e^(−αk²t), with \
             mercury's thermal diffusivity α = k/(ρc) = {:.2e} m²/s. The SPH conduction term \
             (Cleary &amp; Monaghan) uses each liquid's real conductivity and specific heat and conserves \
             energy pair by pair. The box is closed and there's no gravity (so nothing convects): with an open \
             top, surface tension, which rules in zero gravity, would pull the mercury into a drop and stir it.</p>",
            alpha
        ),
        chart: Some(
            Chart::new("Time (s)", "Amplitude / initial amplitude")
                .with(Series::line("e^(−αk²t)", curve, CARDINAL))
                .with(Series::points("Simulated", points, GOLD))
                .svg(),
        ),
        checks: vec![check("Decay rate αk²", alpha * kk, rate, 0.10, "1/s")],
    }
}

pub fn vacuum_flash() -> Case {
    use engine3d::fluid::{LiquidSpec, water};
    use engine3d::{Vec3, presets};
    // Water at 20 °C suddenly in vacuum. Wherever its own hydrostatic
    // pressure is below its vapor pressure it boils, cooling until it
    // reaches the boiling point for that depth: T = T_sat(ρ g h). Below
    // the depth where that equals 20 °C, it's held liquid.
    let d = 0.04;
    let mut w = presets::tank(0.4, 0.4, 0.6);
    w.fluid.params.particle_spacing = d;
    w.ambient_pressure = 0.0;
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.2, 0.0, -0.2),
        Vec3::new(0.2, 0.4, 0.2),
        &b,
        LiquidSpec::water().at(293.15),
    );
    let m0 = f64::from(w.fluid.water_mass());
    let e0 = w.fluid.thermal_energy();
    for _ in 0..steps(0.25) {
        w.step(DT);
    }
    let surface = w
        .fluid
        .positions
        .iter()
        .map(|p| p.y)
        .fold(0.0_f32, f32::max)
        + 0.5 * d;
    let rho = 998.2_f64;
    let exact = |depth: f64| {
        let p = (rho * G * depth.max(0.0)) as f32;
        f64::from(water::saturation_temperature(p).min(293.15)) - 273.15
    };
    // Band means by depth, away from the side walls.
    let bands = 9;
    let (mut sum, mut n) = (vec![0.0_f64; bands], vec![0_u32; bands]);
    for (i, p) in w.fluid.positions.iter().enumerate() {
        if p.x.abs() > 0.12 || p.z.abs() > 0.12 {
            continue;
        }
        let depth = surface - p.y;
        let k = (depth / 0.04) as usize;
        if k < bands {
            sum[k] += w.fluid.temperatures[i] - 273.15;
            n[k] += 1;
        }
    }
    let points: Vec<(f64, f64)> = (0..bands)
        .filter(|&k| n[k] > 0)
        .map(|k| ((k as f64 + 0.5) * 0.04, sum[k] / f64::from(n[k])))
        .collect();
    let worst = points
        .iter()
        .filter(|(depth, _)| *depth > 0.04)
        .map(|(depth, t)| (t - exact(*depth)).abs())
        .fold(0.0, f64::max);
    let curve: Vec<(f64, f64)> = (0..=40)
        .map(|i| (0.01 * i as f64, exact(0.01 * i as f64)))
        .collect();
    let held_depth = f64::from(water::saturation_pressure(293.15)) / (rho * G);
    let m1 = f64::from(w.fluid.water_mass() + w.water.vented);
    let lost = e0 - w.fluid.thermal_energy();
    Case {
        id: "vacuum-flash",
        title: "Water flash-boiling in vacuum",
        engine: "3D SPH fluid, boiling",
        explanation: format!(
            "<p>A 0.4 m pool of 20 °C water with nothing above it. Water boils where the pressure on it is \
             below its vapor pressure (2.34 kPa at 20 °C), and the only pressure here is the water's own \
             weight, ρgh. So the top {:.0} cm boils, each layer cooling until it reaches its own boiling \
             point, T_sat(ρgh) from the IAPWS-IF97 steam-table equation, while the water below stays at \
             20 °C. The heat that leaves is exactly the latent heat of the vapor that escapes. The profile is \
             read after 0.25 s: the top band lags because a free surface can only lose molecules at the \
             Hertz–Knudsen rate, and later the chilled water (denser than 20 °C water) sinks and mixes, as it \
             would in reality. (Without freezing in the model, the surface stops at 0.01 °C instead of \
             icing over.)</p>",
            100.0 * held_depth
        ),
        chart: Some(
            Chart::new("Depth below the surface (m)", "Temperature (°C)")
                .with(Series::line(
                    "Boiling point at depth, T_sat(ρgh), up to 20 °C",
                    curve,
                    CARDINAL,
                ))
                .with(Series::points(
                    "Simulated (band means, 0.25 s)",
                    points,
                    GOLD,
                ))
                .svg(),
        ),
        checks: vec![
            check_abs(
                "Largest band deviation from T_sat(ρgh)",
                0.0,
                worst,
                1.5,
                "K",
            ),
            check("Water (liquid + vapor) conserved", m0, m1, 1e-4, "kg"),
            check(
                "Heat lost = latent heat of the vapor",
                w.water.latent,
                lost,
                0.01,
                "J",
            ),
        ],
    }
}

pub fn plate_convection() -> Case {
    use engine3d::fluid::Liquid;
    use engine3d::{RigidBody, Shape, Vec3, presets};
    // An 80 °C plate on the floor under 20 °C water. Over the first
    // moments the heat in should be h A ΔT t with h from turbulent natural
    // convection.
    let mut points = Vec::new();
    let mut checks = Vec::new();
    for celsius in [40.0_f32, 60.0, 80.0] {
        let mut w = presets::tank(1.0, 1.0, 1.0);
        let mut plate =
            RigidBody::new_static(Shape::cuboid(0.6, 0.04, 0.6), Vec3::new(0.0, 0.02, 0.0));
        plate.temperature = Some(celsius + 273.15);
        w.add_body(plate);
        let b = w.bodies.clone();
        w.fluid
            .fill_box(Vec3::new(-0.5, 0.04, -0.5), Vec3::new(0.5, 0.4, 0.5), &b);
        let e0 = w.fluid.thermal_energy();
        for _ in 0..steps(0.25) {
            w.step(DT);
        }
        let got = w.fluid.thermal_energy() - e0;
        let h = Liquid::Water.natural_convection(0.0, 293.15, celsius + 273.15);
        let expected = f64::from(h * 0.36 * (celsius - 20.0)) * 0.25;
        points.push((f64::from(celsius - 20.0), got / 0.25 / 0.36));
        checks.push(check(
            &format!("Heat in over 0.25 s, plate at {celsius} °C"),
            expected,
            got,
            0.25,
            "J",
        ));
    }
    let curve: Vec<(f64, f64)> = (1..=70)
        .map(|dt| {
            let dt = f64::from(dt);
            let h = Liquid::Water.natural_convection(0.0, 293.15, 293.15 + dt as f32);
            (dt, f64::from(h) * dt)
        })
        .collect();
    Case {
        id: "plate",
        title: "Heat from a hot plate into water",
        engine: "3D SPH fluid, heat",
        explanation: "<p>A 0.6 m square plate held at 40, 60, or 80 °C under 20 °C water. The thermal boundary \
             layer on a heated wall is far thinner than a particle, so the engine carries heat across the \
             first layer at the natural-convection rate q = hΔT, with h = 0.13 k (gβΔT/να)^(1/3) (turbulent, \
             McAdams) from water's properties at the film temperature: the flux grows as ΔT^(4/3). This checks \
             that the SPH wall term delivers that flux (plate area 0.36 m²; the plate's edges add a little).</p>"
            .into(),
        chart: Some(
            Chart::new("Plate above the water (K)", "Heat flux (W/m²)")
                .with(Series::line("q = h(ΔT) ΔT", curve, CARDINAL))
                .with(Series::points("Simulated, first 0.25 s", points, GOLD))
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
        ("layered", layered_hydrostatics),
        ("conduction", conduction_in_liquid),
        ("plate", plate_convection),
        ("vacuum flash", vacuum_flash),
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
