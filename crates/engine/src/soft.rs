//! Soft bodies: deformable blobs and jelly, simulated with XPBD (extended
//! position-based dynamics; Macklin, Müller & Chentanez 2016, and the
//! "small steps" variant, Macklin et al. 2019).
//!
//! A soft body is a set of particles joined by **edge constraints** (each
//! edge wants its rest length) and, for a closed shape, an **area
//! constraint** on its outline (the shape wants its rest area, like gas
//! pressure in a balloon).
//!
//! # How XPBD works
//!
//! Each substep, particles move freely (velocity and gravity), then every
//! constraint C(x) = 0 is enforced by moving the particles along ∇C:
//!
//! ```text
//! Δλ = (−C − α̃ λ) / (Σᵢ wᵢ |∇ᵢC|² + α̃),   Δxᵢ = wᵢ ∇ᵢC Δλ,   α̃ = α / h²
//! ```
//!
//! with wᵢ = 1/mᵢ and α the constraint's **compliance**, the inverse of its
//! stiffness (an edge of stiffness k N/m has α = 1/k). Velocities are then
//! recovered from how far particles actually moved: v = (x − x_prev)/h.
//!
//! What makes XPBD worth using: the compliance term makes stiffness a real
//! physical quantity. Plain PBD gets stiffer the more iterations you run;
//! XPBD converges to the true spring force (a hanging chain of springs
//! stretches exactly m·g/k per edge whatever the substep count, which
//! `tests/soft.rs` checks). With many small substeps and one constraint pass
//! each, it is also very stable and cheap.
//!
//! # Collisions
//!
//! Particles have a radius. They are pushed out of rigid bodies with the
//! correction split by inverse mass, so a soft body can shove a crate: the
//! body's share is applied to its position and velocity directly (as XPBD
//! does for rigid bodies). Friction is positional: tangential sliding during
//! a substep is cancelled up to μ times the penetration. Particles of
//! different soft bodies push each other apart the same way.

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use crate::fluid::grid::NeighborGrid;
use glam::{DVec2, Vec2};
use serde::{Deserialize, Serialize};

/// Two particles that want to stay `rest` apart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub i: u32,
    pub j: u32,
    /// m
    pub rest: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftBody {
    pub positions: Vec<Vec2>,
    #[serde(default)]
    pub velocities: Vec<Vec2>,
    /// 1 / mass per particle, 1/kg. Zero pins a particle in place.
    pub inv_mass: Vec<f32>,
    pub edges: Vec<Edge>,
    /// Stiffness of each edge, N/m.
    pub edge_stiffness: f32,
    /// The outline, counter-clockwise, as particle indices: drawn, and
    /// (when `area_stiffness` > 0) held at `rest_area`.
    pub outline: Vec<u32>,
    /// m² (2D "volume")
    pub rest_area: f32,
    /// How strongly the outline keeps its area: the inverse of the area
    /// constraint's compliance (N/m³ in 2D: pressure per unit of area
    /// change). 0 disables it.
    pub area_stiffness: f32,
    /// Triangles covering the body, for drawing. Empty for a blob, which is
    /// drawn as a fan from its center.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub triangles: Vec<[u32; 3]>,
    /// Collision radius of each particle, m.
    pub radius: f32,
    pub friction: f32,
    /// Internal damping, 1/s: how fast jiggling dies out (the body's overall
    /// motion and spin are untouched). 0 makes a perfectly elastic body that
    /// wobbles forever.
    #[serde(default = "default_damping")]
    pub damping: f32,
}

fn default_damping() -> f32 {
    2.0
}

impl PartialEq for SoftBody {
    fn eq(&self, other: &Self) -> bool {
        self.positions == other.positions
            && self.velocities == other.velocities
            && self.inv_mass == other.inv_mass
            && self.edges == other.edges
            && self.edge_stiffness == other.edge_stiffness
            && self.outline == other.outline
            && self.rest_area == other.rest_area
            && self.area_stiffness == other.area_stiffness
            && self.radius == other.radius
            && self.friction == other.friction
            && self.damping == other.damping
            && self.triangles == other.triangles
    }
}

/// Signed area of a closed polygon (positive when counter-clockwise).
fn polygon_area(points: &[Vec2], outline: &[u32]) -> f32 {
    let n = outline.len();
    0.5 * (0..n)
        .map(|k| {
            let a = points[outline[k] as usize];
            let b = points[outline[(k + 1) % n] as usize];
            a.perp_dot(b)
        })
        .sum::<f32>()
}

impl SoftBody {
    /// A round blob: `segments` particles around a circle, each joined to
    /// its neighbors and next-but-one neighbors, holding its area. `density`
    /// is kg/m² (as for rigid bodies) and sets the total mass.
    pub fn blob(center: Vec2, radius: f32, segments: usize, density: f32) -> Self {
        let n = segments.max(6);
        let positions: Vec<Vec2> = (0..n)
            .map(|k| {
                center + radius * Vec2::from_angle(std::f32::consts::TAU * k as f32 / n as f32)
            })
            .collect();
        let mut edges = Vec::new();
        for k in 0..n {
            for step in [1, 2] {
                let j = (k + step) % n;
                edges.push(Edge {
                    i: k as u32,
                    j: j as u32,
                    rest: (positions[j] - positions[k]).length(),
                });
            }
        }
        let outline: Vec<u32> = (0..n as u32).collect();
        let rest_area = polygon_area(&positions, &outline);
        let mass = density * rest_area;
        let spacing = (positions[1] - positions[0]).length();
        Self {
            inv_mass: vec![n as f32 / mass; n],
            velocities: vec![Vec2::ZERO; n],
            positions,
            edges,
            edge_stiffness: 2000.0,
            outline,
            rest_area,
            triangles: Vec::new(),
            area_stiffness: 50_000.0,
            radius: 0.5 * spacing,
            friction: 0.5,
            damping: default_damping(),
        }
    }

    /// A jelly block: a lattice of (nx + 1) × (ny + 1) particles with edges
    /// along rows, columns, and both diagonals.
    pub fn block(
        center: Vec2,
        width: f32,
        height: f32,
        nx: usize,
        ny: usize,
        density: f32,
    ) -> Self {
        let (nx, ny) = (nx.max(1), ny.max(1));
        let idx = |i: usize, j: usize| (j * (nx + 1) + i) as u32;
        let origin = center - Vec2::new(width, height) * 0.5;
        let (dx, dy) = (width / nx as f32, height / ny as f32);
        let mut positions = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                positions.push(origin + Vec2::new(i as f32 * dx, j as f32 * dy));
            }
        }
        let mut edges = Vec::new();
        let mut link = |a: u32, b: u32| {
            edges.push(Edge {
                i: a,
                j: b,
                rest: (positions[b as usize] - positions[a as usize]).length(),
            });
        };
        for j in 0..=ny {
            for i in 0..=nx {
                if i < nx {
                    link(idx(i, j), idx(i + 1, j));
                }
                if j < ny {
                    link(idx(i, j), idx(i, j + 1));
                }
                if i < nx && j < ny {
                    link(idx(i, j), idx(i + 1, j + 1));
                    link(idx(i + 1, j), idx(i, j + 1));
                }
            }
        }
        // Outline, counter-clockwise from the bottom-left corner.
        let mut outline = Vec::new();
        outline.extend((0..nx).map(|i| idx(i, 0)));
        outline.extend((0..ny).map(|j| idx(nx, j)));
        outline.extend((1..=nx).rev().map(|i| idx(i, ny)));
        outline.extend((1..=ny).rev().map(|j| idx(0, j)));
        let mut triangles = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        let n = positions.len();
        let mass = density * width * height;
        Self {
            inv_mass: vec![n as f32 / mass; n],
            velocities: vec![Vec2::ZERO; n],
            rest_area: width * height,
            radius: 0.5 * dx.min(dy),
            positions,
            edges,
            edge_stiffness: 2000.0,
            outline,
            triangles,
            area_stiffness: 0.0,
            friction: 0.5,
            damping: default_damping(),
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn mass(&self) -> f32 {
        self.inv_mass
            .iter()
            .filter(|&&w| w > 0.0)
            .map(|w| 1.0 / w)
            .sum()
    }

    pub fn area(&self) -> f32 {
        polygon_area(&self.positions, &self.outline)
    }

    pub fn center(&self) -> Vec2 {
        self.positions.iter().sum::<Vec2>() / self.len().max(1) as f32
    }

    pub fn kinetic_energy(&self) -> f32 {
        self.velocities
            .iter()
            .zip(&self.inv_mass)
            .filter(|(_, w)| **w > 0.0)
            .map(|(v, w)| 0.5 * v.length_squared() / w)
            .sum()
    }

    pub fn momentum(&self) -> Vec2 {
        self.velocities
            .iter()
            .zip(&self.inv_mass)
            .filter(|(_, w)| **w > 0.0)
            .map(|(v, w)| *v / *w)
            .sum()
    }

    /// Whether `p` is inside the outline.
    pub fn contains_point(&self, p: Vec2) -> bool {
        let n = self.outline.len();
        let mut inside = false;
        for k in 0..n {
            let a = self.positions[self.outline[k] as usize];
            let b = self.positions[self.outline[(k + 1) % n] as usize];
            if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                inside = !inside;
            }
        }
        inside
    }

    /// Move every particle by `delta` (dragging in the editor).
    pub fn translate(&mut self, delta: Vec2) {
        for p in &mut self.positions {
            *p += delta;
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.positions.len();
        if self.inv_mass.len() != n || (!self.velocities.is_empty() && self.velocities.len() != n) {
            return Err("soft body arrays have different lengths".into());
        }
        if self
            .edges
            .iter()
            .any(|e| e.i as usize >= n || e.j as usize >= n)
            || self.outline.iter().any(|&i| i as usize >= n)
            || self.triangles.iter().flatten().any(|&i| i as usize >= n)
        {
            return Err("soft body refers to a particle that doesn't exist".into());
        }
        let ok = |v: f32| v.is_finite() && v >= 0.0;
        if !(ok(self.edge_stiffness)
            && ok(self.area_stiffness)
            && self.radius > 0.0
            && ok(self.friction)
            && ok(self.damping))
            || !self.inv_mass.iter().all(|&w| ok(w))
            || !self.positions.iter().all(|p| p.is_finite())
        {
            return Err("soft body values must be finite and non-negative".into());
        }
        Ok(())
    }
}

/// Area-constraint damping time per unit of `SoftBody::damping`, s. With
/// the default damping of 2/s, the area mode's damping coefficient is
/// 0.01 s × its stiffness.
const AREA_DAMPING_TIME: f32 = 0.005;

/// Settings for the soft-body solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftSettings {
    /// XPBD substeps per world step. More is stiffer-capable and more
    /// accurate (XPBD's stiffness doesn't depend on it).
    pub substeps: u32,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self { substeps: 20 }
    }
}

/// Working copy of one soft body's state in double precision.
///
/// XPBD takes many tiny substeps and rebuilds velocity from how far each
/// particle moved, v = (x − x_prev)/h. In single precision that fails: at
/// 80 substeps per 1/120 s step a particle at rest 3 m up first moves
/// v·h ≈ 1e-7 m, below an f32's resolution there (2.4e-7 m), so it never
/// moves and its velocity is rebuilt as zero; a blob hung in the air. The
/// substep loop therefore runs in f64 and converts back once per step.
struct Work {
    x: Vec<DVec2>,
    v: Vec<DVec2>,
    prev: Vec<DVec2>,
    w: Vec<f64>,
}

impl Work {
    fn of(s: &SoftBody) -> Self {
        let x: Vec<DVec2> = s.positions.iter().map(|p| p.as_dvec2()).collect();
        Self {
            v: s.velocities.iter().map(|v| v.as_dvec2()).collect(),
            prev: x.clone(),
            x,
            w: s.inv_mass.iter().map(|&w| f64::from(w)).collect(),
        }
    }
}

/// Advance every soft body by `dt`, colliding with `bodies` (which they can
/// push) and with each other.
pub fn step(
    soft: &mut [SoftBody],
    bodies: &mut [RigidBody],
    gravity: Vec2,
    dt: f32,
    settings: &SoftSettings,
) {
    if soft.is_empty() {
        return;
    }
    for s in soft.iter_mut() {
        s.velocities.resize(s.positions.len(), Vec2::ZERO);
    }
    let mut work: Vec<Work> = soft.iter().map(Work::of).collect();
    let substeps = settings.substeps.clamp(1, 200);
    let h = f64::from(dt) / f64::from(substeps);
    let g = gravity.as_dvec2();
    for _ in 0..substeps {
        // Predict.
        for wk in &mut work {
            wk.prev.copy_from_slice(&wk.x);
            for k in 0..wk.x.len() {
                if wk.w[k] > 0.0 {
                    wk.v[k] += g * h;
                    wk.x[k] += wk.v[k] * h;
                }
            }
        }
        // Contacts first too, so the shape constraints below see particles
        // that are already out of the floor and don't fight it as hard.
        collide_rigid(soft, &mut work, bodies, h);
        // Constraints (small steps: one pass, λ starts at zero).
        for (s, wk) in soft.iter().zip(&mut work) {
            solve_edges(s, wk, h);
            solve_area(s, wk, h);
        }
        collide_rigid(soft, &mut work, bodies, h);
        collide_soft(soft, &mut work);
        // Velocities from motion, then damping.
        for (s, wk) in soft.iter().zip(&mut work) {
            for k in 0..wk.x.len() {
                wk.v[k] = if wk.w[k] > 0.0 {
                    (wk.x[k] - wk.prev[k]) / h
                } else {
                    DVec2::ZERO
                };
            }
            damp_deformation(s, wk, h);
        }
    }
    for (s, wk) in soft.iter_mut().zip(&work) {
        for k in 0..s.positions.len() {
            s.positions[k] = wk.x[k].as_vec2();
            s.velocities[k] = wk.v[k].as_vec2();
        }
    }
}

/// Damp every particle's velocity toward the body's rigid motion (its
/// center-of-mass velocity plus its overall spin), by a fraction
/// `damping · h` (Müller et al. 2007, "Position Based Dynamics", §3.5).
/// This damps every way the body can deform, including wobbles that barely
/// change any edge's length (damping along edges alone left a blob jiggling
/// for many seconds), while leaving linear and angular momentum unchanged.
fn damp_deformation(s: &SoftBody, wk: &mut Work, h: f64) {
    let factor = (f64::from(s.damping) * h).min(1.0);
    if factor <= 0.0 {
        return;
    }
    let (mut mass, mut x_cm, mut p) = (0.0, DVec2::ZERO, DVec2::ZERO);
    for k in 0..wk.x.len() {
        if wk.w[k] > 0.0 {
            let m = 1.0 / wk.w[k];
            mass += m;
            x_cm += m * wk.x[k];
            p += m * wk.v[k];
        }
    }
    if mass == 0.0 {
        return;
    }
    x_cm /= mass;
    let v_cm = p / mass;
    // Angular velocity of the best-fit rigid rotation: ω = L / I.
    let (mut l, mut inertia) = (0.0, 0.0);
    for k in 0..wk.x.len() {
        if wk.w[k] > 0.0 {
            let r = wk.x[k] - x_cm;
            l += r.perp_dot(wk.v[k] - v_cm) / wk.w[k];
            inertia += r.length_squared() / wk.w[k];
        }
    }
    let omega = if inertia > 0.0 { l / inertia } else { 0.0 };
    for k in 0..wk.x.len() {
        if wk.w[k] > 0.0 {
            let rigid = v_cm + omega * (wk.x[k] - x_cm).perp();
            let v = wk.v[k];
            wk.v[k] = v + factor * (rigid - v);
        }
    }
}

fn solve_edges(s: &SoftBody, wk: &mut Work, h: f64) {
    if s.edge_stiffness <= 0.0 {
        return;
    }
    let alpha = 1.0 / (f64::from(s.edge_stiffness) * h * h);
    for e in &s.edges {
        let (i, j) = (e.i as usize, e.j as usize);
        let (wi, wj) = (wk.w[i], wk.w[j]);
        let w = wi + wj;
        if w == 0.0 {
            continue;
        }
        let d = wk.x[j] - wk.x[i];
        let len = d.length();
        if len < 1e-12 {
            continue;
        }
        let n = d / len;
        let lambda = -(len - f64::from(e.rest)) / (w + alpha);
        wk.x[i] -= wi * lambda * n;
        wk.x[j] += wj * lambda * n;
    }
}

/// The area constraint, with XPBD constraint damping (Macklin et al. 2016,
/// eq. 26): Δλ = (−C − α̃λ − γ ∇C·(x − x_prev)) / ((1 + γ) Σ w|∇C|² + α̃),
/// with γ = α̃ β̃ / h and the damping coefficient β proportional to the
/// stiffness, so γ = (AREA_DAMPING_TIME × damping) / h.
fn solve_area(s: &SoftBody, wk: &mut Work, h: f64) {
    let n = s.outline.len();
    if s.area_stiffness <= 0.0 || n < 3 {
        return;
    }
    let alpha = 1.0 / (f64::from(s.area_stiffness) * h * h);
    let gamma = f64::from(AREA_DAMPING_TIME * s.damping) / h;
    let at = |k: usize| wk.x[s.outline[k % n] as usize];
    let area = 0.5 * (0..n).map(|k| at(k).perp_dot(at(k + 1))).sum::<f64>();
    let c = area - f64::from(s.rest_area);
    // ∇ᵢA = ½ (y_{i+1} − y_{i−1}, x_{i−1} − x_{i+1})
    let grads: Vec<DVec2> = (0..n)
        .map(|k| {
            let (next, prev) = (at(k + 1), at(k + n - 1));
            0.5 * DVec2::new(next.y - prev.y, prev.x - next.x)
        })
        .collect();
    let mut denom = 0.0;
    let mut rate = 0.0;
    for (k, g) in grads.iter().enumerate() {
        let i = s.outline[k] as usize;
        denom += wk.w[i] * g.length_squared();
        rate += g.dot(wk.x[i] - wk.prev[i]);
    }
    if denom + alpha <= 0.0 {
        return;
    }
    let lambda = -(c + gamma * rate) / ((1.0 + gamma) * denom + alpha);
    for (k, g) in grads.into_iter().enumerate() {
        let i = s.outline[k] as usize;
        wk.x[i] += wk.w[i] * lambda * g;
    }
}

fn collide_rigid(soft: &[SoftBody], work: &mut [Work], bodies: &mut [RigidBody], h: f64) {
    let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
    for (s, wk) in soft.iter().zip(work.iter_mut()) {
        let r = f64::from(s.radius);
        for k in 0..wk.x.len() {
            let wp = wk.w[k];
            if wp == 0.0 {
                continue;
            }
            for (bi, bb) in boxes.iter().enumerate() {
                let x = wk.x[k];
                let (lo, hi) = (bb.min.as_dvec2(), bb.max.as_dvec2());
                if x.x < lo.x - r || x.x > hi.x + r || x.y < lo.y - r || x.y > hi.y + r {
                    continue;
                }
                let body = &bodies[bi];
                let (surface, normal, inside) = body.closest_surface_point(x.as_vec2());
                let (surface, normal) = (surface.as_dvec2(), normal.as_dvec2());
                let dist = (x - surface).length();
                let depth = if inside { r + dist } else { r - dist };
                if depth <= 0.0 {
                    continue;
                }
                // Generalized inverse mass of the body at the contact.
                let arm = surface - body.position.as_dvec2();
                let wb = f64::from(body.inverse_mass())
                    + f64::from(body.inverse_inertia()) * arm.perp_dot(normal).powi(2);
                let share = depth / (wp + wb);
                wk.x[k] += wp * share * normal;
                // Positional friction against the surface's motion.
                let surface_motion = body.velocity_at(surface.as_vec2()).as_dvec2() * h;
                let slide = (wk.x[k] - wk.prev[k]) - surface_motion;
                let tangential = slide - slide.dot(normal) * normal;
                let t_len = tangential.length();
                if t_len > 1e-15 {
                    let limit = f64::from(s.friction.max(body.friction)) * wp * share;
                    wk.x[k] -= tangential * (limit / t_len).min(1.0);
                }
                // The body's share: move it and give it the matching velocity.
                if wb > 0.0 {
                    let push = (-share * normal).as_vec2();
                    let arm = arm.as_vec2();
                    let hf = h as f32;
                    let body = &mut bodies[bi];
                    let (im, ii) = (body.inverse_mass(), body.inverse_inertia());
                    body.position += im * push;
                    body.angle += ii * arm.perp_dot(push);
                    body.velocity += im * push / hf;
                    body.angular_velocity += ii * arm.perp_dot(push) / hf;
                }
            }
        }
    }
}

/// Particles of different soft bodies push apart.
fn collide_soft(soft: &[SoftBody], work: &mut [Work]) {
    if soft.len() < 2 {
        return;
    }
    let mut owner = Vec::new();
    let mut points = Vec::new();
    let mut max_r = 0.0_f32;
    for (b, (s, wk)) in soft.iter().zip(work.iter()).enumerate() {
        max_r = max_r.max(s.radius);
        for (k, p) in wk.x.iter().enumerate() {
            owner.push((b, k));
            points.push(p.as_vec2());
        }
    }
    let grid = NeighborGrid::build(&points, 2.0 * max_r);
    for a in 0..points.len() {
        let (ba, ka) = owner[a];
        grid.for_each_candidate(points[a], |b| {
            let (bb, kb) = owner[b];
            if bb <= ba {
                return;
            }
            let d = work[bb].x[kb] - work[ba].x[ka];
            let len = d.length();
            let min = f64::from(soft[ba].radius + soft[bb].radius);
            if len >= min || len < 1e-12 {
                return;
            }
            let (wa, wb) = (work[ba].w[ka], work[bb].w[kb]);
            if wa + wb == 0.0 {
                return;
            }
            let n = d / len;
            let corr = (min - len) / (wa + wb);
            work[ba].x[ka] -= wa * corr * n;
            work[bb].x[kb] += wb * corr * n;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_have_the_requested_mass_and_area() {
        let b = SoftBody::blob(Vec2::ZERO, 1.0, 32, 2.0);
        // A 32-gon of radius 1 has area 32/2 · sin(2π/32) = 3.1214.
        assert!((b.rest_area - 3.1214).abs() < 1e-3);
        assert!((b.mass() - 2.0 * b.rest_area).abs() < 1e-3);
        let j = SoftBody::block(Vec2::ZERO, 2.0, 1.0, 8, 4, 3.0);
        assert!((j.area() - 2.0).abs() < 1e-5, "outline area {}", j.area());
        assert!((j.mass() - 6.0).abs() < 1e-3);
        assert_eq!(j.outline.len(), 2 * (8 + 4));
        assert!(j.validate().is_ok());
        assert!(j.contains_point(Vec2::new(0.3, 0.2)) && !j.contains_point(Vec2::new(1.3, 0.0)));
    }
}
