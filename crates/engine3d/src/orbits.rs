//! Planetary systems: a star, planets, and moons pulling on each other by
//! Newtonian gravity (full N-body), in real units.
//!
//! Everything is SI and `f64`: positions in meters (a planet 30 AU out is
//! 4.5 × 10¹² m, which `f64` holds to a fraction of a millimeter), masses in
//! kilograms, time in seconds. [`AU`], [`DAY`], [`M_SUN`] and friends convert.
//!
//! The integrator is kick-drift-kick leapfrog: symplectic, so energy errors
//! stay bounded over millions of orbits instead of drifting. The step size
//! comes from the tightest pair of bodies ([`OrbitSystem::stable_dt`]), so a
//! close moon or a near miss automatically gets small steps.
//!
//! Bodies that touch merge, conserving mass and momentum.
//!
//! Frame: +y is ecliptic north (the same y-up frame as the rest of the
//! engine); x points to the vernal equinox. Orbits that look counterclockwise
//! from above (+y) are prograde.

use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

/// Newtonian constant of gravitation, m³/(kg·s²) (CODATA 2018).
pub const G: f64 = 6.674_30e-11;
/// Astronomical unit, m (IAU 2012, exact).
pub const AU: f64 = 1.495_978_707e11;
pub const DAY: f64 = 86_400.0;
/// Julian year, s.
pub const YEAR: f64 = 365.25 * DAY;
/// Solar mass, kg: the IAU nominal GM☉ divided by G.
pub const M_SUN: f64 = 1.327_124_400_18e20 / G;
/// Earth mass, kg.
pub const M_EARTH: f64 = 5.9722e24;
/// Solar radius, m (IAU nominal).
pub const R_SUN: f64 = 6.957e8;
/// Earth's mean radius, m.
pub const R_EARTH: f64 = 6.371e6;

/// Steps per orbit for the tightest orbit in the system. Leapfrog's period
/// error goes as the square of the step: orbits run long by (2π/256)²/3 =
/// 0.02% (Earth's year by 1.8 hours), with no drift in their size or shape.
pub const STEPS_PER_ORBIT: f64 = 256.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Star,
    Planet,
    Moon,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    pub kind: Kind,
    /// What it orbits: the star for planets, a planet for moons, `None` for
    /// the star.
    pub parent: Option<usize>,
    /// kg.
    pub mass: f64,
    /// m.
    pub radius: f64,
    /// sRGB, for drawing.
    pub color: [u8; 3],
    /// m, relative to the system's center of mass.
    pub position: DVec3,
    /// m/s.
    pub velocity: DVec3,
}

/// The shape and timing of an orbit around its parent, from the two-body
/// approximation of the current position and velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elements {
    /// Current distance from the parent, m.
    pub distance: f64,
    /// Current speed relative to the parent, m/s.
    pub speed: f64,
    /// Semi-major axis, m; negative when escaping (hyperbolic).
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    /// Degrees from the ecliptic (x–z plane); above 90° is retrograde.
    pub inclination_deg: f64,
    /// s; `None` if the body isn't bound to its parent.
    pub period: Option<f64>,
}

/// What a call to [`OrbitSystem::advance`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Advance {
    pub steps: usize,
    /// Simulated time, s. Less than asked for if the step budget ran out.
    pub simulated: f64,
    /// Bodies that collided: (absorbed body's name, survivor's name).
    pub mergers: Vec<(String, String)>,
    /// The same collisions by index, in the order they happened: (index
    /// removed, survivor's index at that moment). Later indices shift down
    /// by one after each removal, as with `Vec::remove`.
    pub removed: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrbitSystem {
    pub bodies: Vec<Body>,
    /// Simulated time since the system was built, s.
    pub time: f64,
}

impl OrbitSystem {
    /// A lone star.
    pub fn with_star(name: &str, mass: f64, radius: f64) -> Self {
        Self {
            bodies: vec![Body {
                name: name.into(),
                kind: Kind::Star,
                parent: None,
                mass,
                radius,
                color: [255, 214, 120],
                position: DVec3::ZERO,
                velocity: DVec3::ZERO,
            }],
            time: 0.0,
        }
    }

    /// The star (body 0), if there is one.
    pub fn star(&self) -> Option<usize> {
        self.bodies.iter().position(|b| b.kind == Kind::Star)
    }

    /// Moons of body `i`.
    pub fn moons_of(&self, i: usize) -> Vec<usize> {
        (0..self.bodies.len())
            .filter(|&k| self.bodies[k].parent == Some(i) && self.bodies[k].kind == Kind::Moon)
            .collect()
    }

    /// Body `i` and its moons.
    fn group(&self, i: usize) -> Vec<usize> {
        let mut g = vec![i];
        if self.bodies[i].kind != Kind::Star {
            g.extend(self.moons_of(i));
        }
        g
    }

    /// Total mass, center of mass, and its velocity of some bodies.
    fn barycenter(&self, group: &[usize]) -> (f64, DVec3, DVec3) {
        let m: f64 = group.iter().map(|&k| self.bodies[k].mass).sum();
        let (mut p, mut v) = (DVec3::ZERO, DVec3::ZERO);
        for &k in group {
            p += self.bodies[k].mass * self.bodies[k].position;
            v += self.bodies[k].mass * self.bodies[k].velocity;
        }
        (m, p / m, v / m)
    }

    /// Add a planet around the star with the given heliocentric orbital
    /// elements (see [`state_from_elements`]). Returns its index.
    pub fn add_planet_with_elements(
        &mut self,
        name: &str,
        mass: f64,
        radius: f64,
        color: [u8; 3],
        orbit: OrbitSpec,
    ) -> usize {
        let star = self.star().expect("a system needs a star before planets");
        self.add_orbiting(name, Kind::Planet, star, mass, radius, color, orbit)
    }

    /// Add a planet on a circular orbit in the ecliptic, `distance` meters
    /// from the star, at `angle_deg` degrees around from +x.
    pub fn add_planet(
        &mut self,
        name: &str,
        mass: f64,
        radius: f64,
        color: [u8; 3],
        distance: f64,
        angle_deg: f64,
    ) -> usize {
        self.add_planet_with_elements(
            name,
            mass,
            radius,
            color,
            OrbitSpec::circular(distance, angle_deg),
        )
    }

    /// Add a moon to planet `planet`. Returns its index.
    pub fn add_moon(
        &mut self,
        planet: usize,
        name: &str,
        mass: f64,
        radius: f64,
        color: [u8; 3],
        orbit: OrbitSpec,
    ) -> usize {
        self.add_orbiting(name, Kind::Moon, planet, mass, radius, color, orbit)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_orbiting(
        &mut self,
        name: &str,
        kind: Kind,
        parent: usize,
        mass: f64,
        radius: f64,
        color: [u8; 3],
        orbit: OrbitSpec,
    ) -> usize {
        // The parent's group (the parent and its moons) keeps its center of
        // mass and its motion: adding a moon doesn't knock a planet off its
        // orbit.
        let group = self.group(parent);
        let (_, before_p, before_v) = self.barycenter(&group);
        let mu = G * (self.bodies[parent].mass + mass);
        let (r, v) = state_from_elements(mu, &orbit);
        let p = &self.bodies[parent];
        let (position, velocity) = (p.position + r, p.velocity + v);
        self.bodies.push(Body {
            name: name.into(),
            kind,
            parent: Some(parent),
            mass,
            radius,
            color,
            position,
            velocity,
        });
        let i = self.bodies.len() - 1;
        if kind == Kind::Moon {
            let mut group = group;
            group.push(i);
            self.shift_group_to(&group, before_p, before_v);
        }
        self.zero_momentum();
        i
    }

    /// Translate and boost a group so its center of mass is at `p`, moving
    /// with `v`.
    fn shift_group_to(&mut self, group: &[usize], p: DVec3, v: DVec3) {
        let (_, now_p, now_v) = self.barycenter(group);
        for &k in group {
            self.bodies[k].position += p - now_p;
            self.bodies[k].velocity += v - now_v;
        }
    }

    /// Remove the system's net momentum, so the star stays put on average.
    pub fn zero_momentum(&mut self) {
        if self.bodies.is_empty() {
            return;
        }
        let all: Vec<usize> = (0..self.bodies.len()).collect();
        let (_, _, v) = self.barycenter(&all);
        for b in &mut self.bodies {
            b.velocity -= v;
        }
    }

    /// Where body `i` orbits from and what it orbits: for a planet, its
    /// group's center of mass relative to the star (so a big moon's tug
    /// doesn't read as eccentricity); for a moon, itself relative to its
    /// planet. Returns (μ = G(M + m), relative position, relative velocity).
    fn relative_state(&self, i: usize) -> Option<(f64, DVec3, DVec3)> {
        let parent = self.bodies[i].parent?;
        let group = if self.bodies[i].kind == Kind::Planet {
            self.group(i)
        } else {
            vec![i]
        };
        let (m, p, v) = self.barycenter(&group);
        let host = &self.bodies[parent];
        Some((G * (host.mass + m), p - host.position, v - host.velocity))
    }

    /// Orbital elements of body `i` around its parent.
    pub fn elements(&self, i: usize) -> Option<Elements> {
        let (mu, r, v) = self.relative_state(i)?;
        Some(elements_from_state(mu, r, v))
    }

    /// Put body `i` on a circular orbit `distance` meters from its parent,
    /// keeping its direction from the parent and its orbital plane. A
    /// planet's moons move with it.
    pub fn set_distance(&mut self, i: usize, distance: f64) {
        let Some((mu, r, v)) = self.relative_state(i) else {
            return;
        };
        let parent = self.bodies[i].parent.unwrap_or(0);
        let normal = r.cross(v).try_normalize().unwrap_or(DVec3::Y);
        // Direction from the parent, flattened into the orbital plane.
        let along = (r - normal * r.dot(normal))
            .try_normalize()
            .unwrap_or_else(|| normal.any_orthonormal_vector());
        let speed = (mu / distance).sqrt();
        // Moving a moon shifts its planet's group's center of mass; it's put
        // back afterwards so the planet's own orbit is untouched.
        let host_group = (self.bodies[i].kind == Kind::Moon).then(|| self.group(parent));
        let host_before = host_group.as_ref().map(|g| self.barycenter(g));
        let host = &self.bodies[parent];
        let target_p = host.position + along * distance;
        let target_v = host.velocity + normal.cross(along) * speed;
        let group = if self.bodies[i].kind == Kind::Planet {
            self.group(i)
        } else {
            vec![i]
        };
        self.shift_group_to(&group, target_p, target_v);
        if let (Some(g), Some((_, p, v))) = (host_group, host_before) {
            self.shift_group_to(&g, p, v);
        }
        self.zero_momentum();
    }

    /// Change body `i`'s mass. Bodies orbiting it keep their positions and
    /// velocities, so their orbits change shape (as they physically would).
    pub fn set_mass(&mut self, i: usize, mass: f64) {
        let group = self.group(i);
        let (_, p, v) = self.barycenter(&group);
        self.bodies[i].mass = mass.max(1.0);
        if self.bodies[i].kind == Kind::Planet {
            self.shift_group_to(&group, p, v);
        }
        self.zero_momentum();
    }

    /// Remove body `i` and, if it's a planet, its moons. Returns the removed
    /// indices in the original numbering, so callers can fix up references.
    pub fn remove(&mut self, i: usize) -> Vec<usize> {
        let mut gone = self.group(i);
        gone.sort_unstable();
        for &k in gone.iter().rev() {
            self.remove_one(k);
        }
        self.zero_momentum();
        gone
    }

    /// Remove one body and renumber `parent` links.
    fn remove_one(&mut self, k: usize) {
        self.bodies.remove(k);
        for b in &mut self.bodies {
            b.parent = match b.parent {
                Some(p) if p == k => None,
                Some(p) if p > k => Some(p - 1),
                p => p,
            };
        }
    }

    /// Gravitational acceleration on every body.
    fn accelerations(&self, out: &mut Vec<DVec3>) {
        out.clear();
        out.resize(self.bodies.len(), DVec3::ZERO);
        for i in 0..self.bodies.len() {
            for j in i + 1..self.bodies.len() {
                let d = self.bodies[j].position - self.bodies[i].position;
                let r2 = d.length_squared();
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                out[i] += d * (G * self.bodies[j].mass * inv_r3);
                out[j] -= d * (G * self.bodies[i].mass * inv_r3);
            }
        }
    }

    /// The largest step that resolves the tightest pair: 1/256 of the
    /// circular-orbit period at their current separation, 2π √(r³/G(m₁+m₂)).
    pub fn stable_dt(&self) -> f64 {
        let mut t: f64 = f64::INFINITY;
        for i in 0..self.bodies.len() {
            for j in i + 1..self.bodies.len() {
                let (a, b) = (&self.bodies[i], &self.bodies[j]);
                let r = (a.position - b.position).length();
                t = t.min((r * r * r / (G * (a.mass + b.mass))).sqrt());
            }
        }
        if t.is_finite() {
            TAU * t / STEPS_PER_ORBIT
        } else {
            DAY
        }
    }

    /// The pair that limits the step size, by name.
    pub fn tightest_pair(&self) -> Option<(usize, usize)> {
        let mut best = None;
        let mut t = f64::INFINITY;
        for i in 0..self.bodies.len() {
            for j in i + 1..self.bodies.len() {
                let (a, b) = (&self.bodies[i], &self.bodies[j]);
                let r = (a.position - b.position).length();
                let tij = r * r * r / (G * (a.mass + b.mass));
                if tij < t {
                    t = tij;
                    best = Some((i, j));
                }
            }
        }
        best
    }

    /// One kick-drift-kick leapfrog step of `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        let mut acc = Vec::new();
        self.accelerations(&mut acc);
        self.kick_drift_kick(dt, &mut acc);
    }

    fn kick_drift_kick(&mut self, dt: f64, acc: &mut Vec<DVec3>) {
        for (b, a) in self.bodies.iter_mut().zip(acc.iter()) {
            b.velocity += 0.5 * dt * *a;
            b.position += dt * b.velocity;
        }
        self.accelerations(acc);
        for (b, a) in self.bodies.iter_mut().zip(acc.iter()) {
            b.velocity += 0.5 * dt * *a;
        }
        self.time += dt;
    }

    /// Simulate `seconds` of time in equal steps no longer than
    /// [`Self::stable_dt`], taking at most `max_steps`; if that isn't enough,
    /// simulate less. Merges any bodies that touch.
    pub fn advance(&mut self, seconds: f64, max_steps: usize) -> Advance {
        self.advance_observed(seconds, max_steps, |_| {})
    }

    /// [`Self::advance`], calling `each_step` after every step (to record
    /// trails without missing fast orbits).
    pub fn advance_observed(
        &mut self,
        seconds: f64,
        max_steps: usize,
        mut each_step: impl FnMut(&OrbitSystem),
    ) -> Advance {
        let mut out = Advance::default();
        if seconds <= 0.0 || self.bodies.len() < 2 {
            return out;
        }
        let dt_max = self.stable_dt();
        let wanted = (seconds / dt_max).ceil().max(1.0) as usize;
        let steps = wanted.min(max_steps.max(1));
        let dt = if steps < wanted {
            dt_max
        } else {
            seconds / steps as f64
        };
        let mut acc = Vec::new();
        self.accelerations(&mut acc);
        for _ in 0..steps {
            self.kick_drift_kick(dt, &mut acc);
            let merged = self.merge_touching();
            if !merged.is_empty() {
                for (names, indices) in merged {
                    out.mergers.push(names);
                    out.removed.push(indices);
                }
                self.accelerations(&mut acc);
            }
            each_step(self);
        }
        out.steps = steps;
        out.simulated = steps as f64 * dt;
        out
    }

    /// Merge bodies whose surfaces touch: the lighter one is absorbed,
    /// conserving mass, momentum, and volume; its moons now orbit the
    /// survivor.
    #[allow(clippy::type_complexity)]
    fn merge_touching(&mut self) -> Vec<((String, String), (usize, usize))> {
        let mut out = Vec::new();
        'again: loop {
            for i in 0..self.bodies.len() {
                for j in i + 1..self.bodies.len() {
                    let (a, b) = (&self.bodies[i], &self.bodies[j]);
                    if (a.position - b.position).length() > a.radius + b.radius {
                        continue;
                    }
                    let (keep, lose) = if a.mass >= b.mass { (i, j) } else { (j, i) };
                    let (k, l) = (&self.bodies[keep], &self.bodies[lose]);
                    let m = k.mass + l.mass;
                    let position = (k.mass * k.position + l.mass * l.position) / m;
                    let velocity = (k.mass * k.velocity + l.mass * l.velocity) / m;
                    let radius = (k.radius.powi(3) + l.radius.powi(3)).cbrt();
                    out.push(((l.name.clone(), k.name.clone()), (lose, keep)));
                    let kb = &mut self.bodies[keep];
                    kb.mass = m;
                    kb.position = position;
                    kb.velocity = velocity;
                    kb.radius = radius;
                    for b in &mut self.bodies {
                        if b.parent == Some(lose) {
                            b.parent = Some(keep);
                        }
                    }
                    self.remove_one(lose);
                    continue 'again;
                }
            }
            break;
        }
        out
    }

    /// Kinetic plus gravitational potential energy, J.
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for (i, a) in self.bodies.iter().enumerate() {
            e += 0.5 * a.mass * a.velocity.length_squared();
            for b in &self.bodies[i + 1..] {
                e -= G * a.mass * b.mass / (a.position - b.position).length();
            }
        }
        e
    }

    /// Total angular momentum about the origin, kg·m²/s.
    pub fn angular_momentum(&self) -> DVec3 {
        self.bodies
            .iter()
            .map(|b| b.mass * b.position.cross(b.velocity))
            .sum()
    }

    /// A unique name like "Planet 3" or "Earth moon 2".
    pub fn unique_name(&self, base: &str) -> String {
        (1..)
            .map(|n| format!("{base} {n}"))
            .find(|name| !self.bodies.iter().any(|b| &b.name == name))
            .expect("unbounded")
    }
}

/// Classical orbital elements, ecliptic frame, angles in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitSpec {
    /// m.
    pub semi_major_axis: f64,
    pub eccentricity: f64,
    pub inclination: f64,
    /// Longitude of the ascending node, Ω.
    pub ascending_node: f64,
    /// Longitude of periapsis, ϖ = Ω + ω.
    pub periapsis_longitude: f64,
    /// Mean longitude, L = ϖ + M.
    pub mean_longitude: f64,
}

impl OrbitSpec {
    /// A circular orbit in the ecliptic, `angle_deg` around from +x.
    pub fn circular(distance: f64, angle_deg: f64) -> Self {
        Self {
            semi_major_axis: distance,
            eccentricity: 0.0,
            inclination: 0.0,
            ascending_node: 0.0,
            periapsis_longitude: 0.0,
            mean_longitude: angle_deg,
        }
    }
}

/// Position and velocity relative to the central body, from orbital
/// elements, for gravitational parameter `mu` = G(M + m).
pub fn state_from_elements(mu: f64, o: &OrbitSpec) -> (DVec3, DVec3) {
    let (a, e) = (o.semi_major_axis, o.eccentricity.clamp(0.0, 0.99));
    let (i, node) = (o.inclination.to_radians(), o.ascending_node.to_radians());
    let omega = (o.periapsis_longitude - o.ascending_node).to_radians();
    let mean = (o.mean_longitude - o.periapsis_longitude)
        .to_radians()
        .rem_euclid(TAU);
    // Kepler's equation, M = E − e sin E, by Newton's method.
    let mut ecc = if e > 0.8 { PI } else { mean };
    for _ in 0..30 {
        let f = ecc - e * ecc.sin() - mean;
        ecc -= f / (1.0 - e * ecc.cos());
        if f.abs() < 1e-15 {
            break;
        }
    }
    // In the orbital plane, periapsis along +x'.
    let (se, ce) = ecc.sin_cos();
    let b = a * (1.0 - e * e).sqrt();
    let (x, y) = (a * (ce - e), b * se);
    let n = (mu / (a * a * a)).sqrt();
    let edot = n / (1.0 - e * ce);
    let (vx, vy) = (-a * se * edot, b * ce * edot);
    // Rotate by ω, i, Ω into the ecliptic (x, y, z with z north).
    let rotate = |px: f64, py: f64| {
        let (so, co) = omega.sin_cos();
        let (si, ci) = i.sin_cos();
        let (sn, cn) = node.sin_cos();
        let (x1, y1) = (co * px - so * py, so * px + co * py);
        let (x2, y2, z2) = (x1, ci * y1, si * y1);
        DVec3::new(cn * x2 - sn * y2, sn * x2 + cn * y2, z2)
    };
    (to_y_up(rotate(x, y)), to_y_up(rotate(vx, vy)))
}

/// Ecliptic (z north) to the engine's y-up frame: (x, y, z) → (x, z, −y).
/// A proper rotation, so prograde stays counterclockwise seen from above.
fn to_y_up(v: DVec3) -> DVec3 {
    DVec3::new(v.x, v.z, -v.y)
}

/// Two-body orbital elements from a relative position and velocity.
pub fn elements_from_state(mu: f64, r: DVec3, v: DVec3) -> Elements {
    let (dist, speed) = (r.length(), v.length());
    let energy = 0.5 * speed * speed - mu / dist;
    let semi_major_axis = -mu / (2.0 * energy);
    let e_vec = ((speed * speed - mu / dist) * r - r.dot(v) * v) / mu;
    let h = r.cross(v);
    let inclination_deg = (h.y / h.length().max(1e-300))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    let period = (energy < 0.0).then(|| TAU * (semi_major_axis.powi(3) / mu).sqrt());
    Elements {
        distance: dist,
        speed,
        semi_major_axis,
        eccentricity: e_vec.length(),
        inclination_deg,
        period,
    }
}

/// name, a (AU), e, I, L, ϖ, Ω (deg), mass (Earths), radius (km), color.
type PlanetRow = (
    &'static str,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    [u8; 3],
);
/// planet, name, a (km), e, inclination (deg), mass (kg), radius (km), color.
type MoonRow = (&'static str, &'static str, f64, f64, f64, f64, f64, [u8; 3]);

/// An empty system: the Sun alone.
pub fn sun_only() -> OrbitSystem {
    OrbitSystem::with_star("Sun", M_SUN, R_SUN)
}

/// The Sun, the eight planets at their J2000 positions (JPL's mean orbital
/// elements), and the major moons: Earth's Moon, Jupiter's four Galilean
/// moons, Titan, Uranus's Titania and Oberon, and Neptune's retrograde
/// Triton. Mars's tiny, fast moons are left out: Phobos's 7.7-hour orbit
/// would need steps 14 times shorter than Io's.
pub fn solar_system() -> OrbitSystem {
    let mut s = sun_only();
    // name, a (AU), e, I, L, ϖ, Ω (deg), mass (Earth = 1), radius (km), color
    #[rustfmt::skip]
    let planets: [PlanetRow; 8] = [
        ("Mercury", 0.387_099_27, 0.205_635_93, 7.004_979_02, 252.250_323_50, 77.457_796_28, 48.330_765_93, 0.0553, 2439.7, [158, 150, 142]),
        ("Venus",   0.723_335_66, 0.006_776_72, 3.394_676_05, 181.979_099_50, 131.602_467_18, 76.679_842_55, 0.815, 6051.8, [226, 196, 140]),
        ("Earth",   1.000_002_61, 0.016_711_23, 0.0, 100.464_571_66, 102.937_681_93, 0.0, 1.0, 6371.0, [70, 120, 200]),
        ("Mars",    1.523_710_34, 0.093_394_10, 1.849_691_42, -4.553_432_05, -23.943_629_59, 49.559_538_91, 0.107, 3389.5, [196, 92, 60]),
        ("Jupiter", 5.202_887_00, 0.048_386_24, 1.304_396_95, 34.396_440_51, 14.728_479_83, 100.473_909_09, 317.8, 69_911.0, [210, 170, 130]),
        ("Saturn",  9.536_675_94, 0.053_861_79, 2.485_991_87, 49.954_244_23, 92.598_878_31, 113.662_424_48, 95.2, 58_232.0, [222, 200, 150]),
        ("Uranus", 19.189_164_64, 0.047_257_44, 0.772_637_83, 313.238_104_51, 170.954_276_30, 74.016_925_03, 14.54, 25_362.0, [150, 210, 220]),
        ("Neptune",30.069_922_76, 0.008_590_48, 1.770_043_47, -55.120_029_69, 44.964_762_27, 131.784_225_74, 17.15, 24_622.0, [80, 110, 210]),
    ];
    let mut index = std::collections::HashMap::new();
    for (name, a, e, inc, l, w, node, mass, radius, color) in planets {
        let orbit = OrbitSpec {
            semi_major_axis: a * AU,
            eccentricity: e,
            inclination: inc,
            ascending_node: node,
            periapsis_longitude: w,
            mean_longitude: l,
        };
        let i = s.add_planet_with_elements(name, mass * M_EARTH, radius * 1e3, color, orbit);
        index.insert(name, i);
    }
    // planet, name, a (km), e, inclination to the ecliptic (deg, approximate:
    // moons orbit near their planet's equator), mass (kg), radius (km).
    #[rustfmt::skip]
    let moons: [MoonRow; 9] = [
        ("Earth",   "Moon",     384_399.0,   0.0549, 5.145,  7.342e22,  1737.4, [190, 190, 186]),
        ("Jupiter", "Io",       421_700.0,   0.0041, 2.2,    8.932e22,  1821.6, [230, 210, 110]),
        ("Jupiter", "Europa",   671_034.0,   0.009,  2.7,    4.800e22,  1560.8, [200, 185, 160]),
        ("Jupiter", "Ganymede", 1_070_412.0, 0.0013, 2.4,    1.4819e23, 2634.1, [160, 150, 140]),
        ("Jupiter", "Callisto", 1_882_709.0, 0.0074, 2.0,    1.0759e23, 2410.3, [110, 100, 90]),
        ("Saturn",  "Titan",    1_221_870.0, 0.0288, 27.0,   1.3452e23, 2574.7, [220, 170, 90]),
        ("Uranus",  "Titania",  435_910.0,   0.0011, 97.8,   3.455e21,  788.4,  [180, 175, 170]),
        ("Uranus",  "Oberon",   583_520.0,   0.0014, 97.8,   3.076e21,  761.4,  [160, 150, 145]),
        ("Neptune", "Triton",   354_759.0,   0.0,    130.0,  2.139e22,  1353.4, [210, 195, 190]),
    ];
    for (k, (planet, name, a, e, inc, mass, radius, color)) in moons.into_iter().enumerate() {
        let orbit = OrbitSpec {
            semi_major_axis: a * 1e3,
            eccentricity: e,
            inclination: inc,
            ascending_node: 0.0,
            periapsis_longitude: 0.0,
            // Spread the moons around their orbits.
            mean_longitude: 77.0 * k as f64,
        };
        s.add_moon(index[planet], name, mass, radius * 1e3, color, orbit);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements_round_trip() {
        let mu = G * M_SUN;
        let spec = OrbitSpec {
            semi_major_axis: 1.7 * AU,
            eccentricity: 0.3,
            inclination: 20.0,
            ascending_node: 40.0,
            periapsis_longitude: 100.0,
            mean_longitude: 210.0,
        };
        let (r, v) = state_from_elements(mu, &spec);
        let el = elements_from_state(mu, r, v);
        assert!((el.semi_major_axis / (1.7 * AU) - 1.0).abs() < 1e-12);
        assert!((el.eccentricity - 0.3).abs() < 1e-12);
        assert!((el.inclination_deg - 20.0).abs() < 1e-9);
    }

    #[test]
    fn prograde_orbits_turn_counterclockwise_seen_from_above() {
        let (r, v) = state_from_elements(G * M_SUN, &OrbitSpec::circular(AU, 0.0));
        // Angular momentum points up (+y).
        assert!(r.cross(v).y > 0.0);
    }
}
