//! Soft bodies in 3D: the same XPBD solver as the 2D engine (see the
//! `engine` crate's `soft` module for the method), with 3D shapes:
//!
//! - **Ball**: a subdivided icosahedron surface whose edges hold their
//!   lengths and whose enclosed **volume** is held like gas pressure.
//! - **Cloth**: a sheet of particles with stretch, shear, and bend springs.
//!   Pin particles (inverse mass 0) to hang it.
//! - **Jelly**: a solid lattice with edges along the axes and across every
//!   face and cell.
//!
//! Particles collide with rigid bodies (pushing them, with the correction
//! split by generalized inverse mass including the inertia tensor) and with
//! other soft bodies. As in 2D, the substep loop runs in f64 so tiny
//! substeps don't lose motion to rounding.

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use crate::fluid::grid::NeighborGrid;
use glam::{DMat3, DQuat, DVec3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub i: u32,
    pub j: u32,
    /// m
    pub rest: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftBody {
    pub positions: Vec<Vec3>,
    #[serde(default)]
    pub velocities: Vec<Vec3>,
    /// 1 / mass per particle, 1/kg. Zero pins a particle in place.
    pub inv_mass: Vec<f32>,
    pub edges: Vec<Edge>,
    /// N/m
    pub edge_stiffness: f32,
    /// Surface triangles, counter-clockwise seen from outside: drawn, and
    /// (when `volume_stiffness` > 0) used to hold `rest_volume`.
    pub triangles: Vec<[u32; 3]>,
    /// m³
    pub rest_volume: f32,
    /// Inverse compliance of the volume constraint (pressure per unit of
    /// volume change, Pa/m³). 0 disables it.
    pub volume_stiffness: f32,
    /// Collision radius of each particle, m.
    pub radius: f32,
    pub friction: f32,
    /// How quickly jiggling dies out, 1/s.
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
            && self.triangles == other.triangles
            && self.rest_volume == other.rest_volume
            && self.volume_stiffness == other.volume_stiffness
            && self.radius == other.radius
            && self.friction == other.friction
            && self.damping == other.damping
    }
}

fn mesh_volume(points: &[Vec3], triangles: &[[u32; 3]]) -> f32 {
    triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (
                points[t[0] as usize],
                points[t[1] as usize],
                points[t[2] as usize],
            );
            a.dot(b.cross(c))
        })
        .sum::<f32>()
        / 6.0
}

/// Edges along every triangle side, each once.
fn triangle_edges(positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Edge> {
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for t in triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if seen.insert((a.min(b), a.max(b))) {
                edges.push(Edge {
                    i: a,
                    j: b,
                    rest: (positions[b as usize] - positions[a as usize]).length(),
                });
            }
        }
    }
    edges
}

/// A unit icosphere: an icosahedron with each triangle split into four,
/// `subdivisions` times, projected onto the sphere.
fn icosphere(subdivisions: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut v: Vec<Vec3> = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect();
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut mid = |a: u32, b: u32, v: &mut Vec<Vec3>| {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                v.push(((v[a as usize] + v[b as usize]) * 0.5).normalize());
                v.len() as u32 - 1
            })
        };
        let mut next = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let (ab, bc, ca) = (mid(a, b, &mut v), mid(b, c, &mut v), mid(c, a, &mut v));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = next;
    }
    (v, f)
}

impl SoftBody {
    /// A pressurized ball of the given radius, `density` kg/m³ (setting the
    /// total mass from its volume).
    pub fn ball(center: Vec3, radius: f32, subdivisions: u32, density: f32) -> Self {
        let (unit, triangles) = icosphere(subdivisions.min(4));
        let positions: Vec<Vec3> = unit.iter().map(|&p| center + radius * p).collect();
        let edges = triangle_edges(&positions, &triangles);
        let rest_volume = mesh_volume(&positions, &triangles);
        let n = positions.len();
        let spacing = edges.iter().map(|e| e.rest).sum::<f32>() / edges.len() as f32;
        Self {
            inv_mass: vec![n as f32 / (density * rest_volume); n],
            velocities: vec![Vec3::ZERO; n],
            positions,
            edges,
            edge_stiffness: 2000.0,
            triangles,
            rest_volume,
            volume_stiffness: 200_000.0,
            radius: 0.5 * spacing,
            friction: 0.5,
            damping: default_damping(),
        }
    }

    /// A horizontal sheet of cloth, `width` (x) by `depth` (z), with
    /// (nx + 1) × (nz + 1) particles, centered at `center`, total mass
    /// `mass` kg.
    pub fn cloth(center: Vec3, width: f32, depth: f32, nx: usize, nz: usize, mass: f32) -> Self {
        let (nx, nz) = (nx.max(1), nz.max(1));
        let idx = |i: usize, k: usize| (k * (nx + 1) + i) as u32;
        let origin = center - Vec3::new(width, 0.0, depth) * 0.5;
        let (dx, dz) = (width / nx as f32, depth / nz as f32);
        let mut positions = Vec::new();
        for k in 0..=nz {
            for i in 0..=nx {
                positions.push(origin + Vec3::new(i as f32 * dx, 0.0, k as f32 * dz));
            }
        }
        let mut triangles = Vec::new();
        for k in 0..nz {
            for i in 0..nx {
                let (a, b, c, d) = (idx(i, k), idx(i + 1, k), idx(i + 1, k + 1), idx(i, k + 1));
                // Counter-clockwise seen from above (+y).
                triangles.push([a, c, b]);
                triangles.push([a, d, c]);
            }
        }
        let mut edges = triangle_edges(&positions, &triangles);
        // Shear (the other diagonal) and bend (skip one) springs.
        let mut link = |a: u32, b: u32| {
            edges.push(Edge {
                i: a,
                j: b,
                rest: (positions[b as usize] - positions[a as usize]).length(),
            });
        };
        for k in 0..=nz {
            for i in 0..=nx {
                if i < nx && k < nz {
                    link(idx(i + 1, k), idx(i, k + 1));
                }
                if i + 2 <= nx {
                    link(idx(i, k), idx(i + 2, k));
                }
                if k + 2 <= nz {
                    link(idx(i, k), idx(i, k + 2));
                }
            }
        }
        let n = positions.len();
        Self {
            inv_mass: vec![n as f32 / mass; n],
            velocities: vec![Vec3::ZERO; n],
            positions,
            edges,
            edge_stiffness: 1000.0,
            triangles,
            rest_volume: 0.0,
            volume_stiffness: 0.0,
            radius: 0.5 * dx.min(dz),
            friction: 0.5,
            damping: default_damping(),
        }
    }

    /// A solid jelly box with n × n × n cells.
    pub fn jelly(center: Vec3, size: Vec3, n: usize, density: f32) -> Self {
        let n = n.max(1);
        let idx = |i: usize, j: usize, k: usize| ((k * (n + 1) + j) * (n + 1) + i) as u32;
        let step = size / n as f32;
        let origin = center - size * 0.5;
        let mut positions = Vec::new();
        for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    positions.push(origin + Vec3::new(i as f32, j as f32, k as f32) * step);
                }
            }
        }
        let mut edges = Vec::new();
        // Every pair within one cell: axes, face diagonals, body diagonals.
        for k in 0..=n {
            for j in 0..=n {
                for i in 0..=n {
                    for (di, dj, dk) in [
                        (1, 0, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                        (1, 1, 0),
                        (1, -1, 0),
                        (1, 0, 1),
                        (1, 0, -1),
                        (0, 1, 1),
                        (0, 1, -1),
                        (1, 1, 1),
                        (1, 1, -1),
                        (1, -1, 1),
                        (1, -1, -1),
                    ] {
                        let (a, b, c) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                        let inside = |x: i32| (0..=n as i32).contains(&x);
                        if inside(a) && inside(b) && inside(c) {
                            let (p, q) = (idx(i, j, k), idx(a as usize, b as usize, c as usize));
                            edges.push(Edge {
                                i: p,
                                j: q,
                                rest: (positions[q as usize] - positions[p as usize]).length(),
                            });
                        }
                    }
                }
            }
        }
        // Surface triangles for drawing, wound outward.
        let mut triangles = Vec::new();
        let mut quad = |a: u32, b: u32, c: u32, d: u32| {
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        };
        for u in 0..n {
            for v in 0..n {
                // −z and +z faces
                quad(
                    idx(u, v, 0),
                    idx(u, v + 1, 0),
                    idx(u + 1, v + 1, 0),
                    idx(u + 1, v, 0),
                );
                quad(
                    idx(u, v, n),
                    idx(u + 1, v, n),
                    idx(u + 1, v + 1, n),
                    idx(u, v + 1, n),
                );
                // −y and +y
                quad(
                    idx(u, 0, v),
                    idx(u + 1, 0, v),
                    idx(u + 1, 0, v + 1),
                    idx(u, 0, v + 1),
                );
                quad(
                    idx(u, n, v),
                    idx(u, n, v + 1),
                    idx(u + 1, n, v + 1),
                    idx(u + 1, n, v),
                );
                // −x and +x
                quad(
                    idx(0, u, v),
                    idx(0, u, v + 1),
                    idx(0, u + 1, v + 1),
                    idx(0, u + 1, v),
                );
                quad(
                    idx(n, u, v),
                    idx(n, u + 1, v),
                    idx(n, u + 1, v + 1),
                    idx(n, u, v + 1),
                );
            }
        }
        let count = positions.len();
        let volume = size.x * size.y * size.z;
        Self {
            inv_mass: vec![count as f32 / (density * volume); count],
            velocities: vec![Vec3::ZERO; count],
            positions,
            edges,
            edge_stiffness: 2000.0,
            triangles,
            rest_volume: volume,
            volume_stiffness: 0.0,
            radius: 0.5 * step.min_element(),
            friction: 0.5,
            damping: default_damping(),
        }
    }

    /// Pin the particles for which `pick` is true (inverse mass 0).
    pub fn pin_where(&mut self, pick: impl Fn(Vec3) -> bool) -> usize {
        let mut count = 0;
        for (p, w) in self.positions.iter().zip(&mut self.inv_mass) {
            if pick(*p) {
                *w = 0.0;
                count += 1;
            }
        }
        count
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

    pub fn volume(&self) -> f32 {
        mesh_volume(&self.positions, &self.triangles)
    }

    pub fn center(&self) -> Vec3 {
        self.positions.iter().sum::<Vec3>() / self.len().max(1) as f32
    }

    pub fn kinetic_energy(&self) -> f32 {
        self.velocities
            .iter()
            .zip(&self.inv_mass)
            .filter(|(_, w)| **w > 0.0)
            .map(|(v, w)| 0.5 * v.length_squared() / w)
            .sum()
    }

    pub fn momentum(&self) -> Vec3 {
        self.velocities
            .iter()
            .zip(&self.inv_mass)
            .filter(|(_, w)| **w > 0.0)
            .map(|(v, w)| *v / *w)
            .sum()
    }

    pub fn translate(&mut self, delta: Vec3) {
        for p in &mut self.positions {
            *p += delta;
        }
    }

    /// Distance along the ray `origin + t·dir` to the nearest particle
    /// sphere it hits (for picking).
    pub fn raycast(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let r2 = (self.radius * 1.5).powi(2);
        self.positions
            .iter()
            .filter_map(|&p| {
                let t = (p - origin).dot(dir);
                (t > 0.0 && (origin + t * dir - p).length_squared() < r2).then_some(t)
            })
            .min_by(f32::total_cmp)
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
            || self.triangles.iter().flatten().any(|&i| i as usize >= n)
        {
            return Err("soft body refers to a particle that doesn't exist".into());
        }
        let ok = |v: f32| v.is_finite() && v >= 0.0;
        if !(ok(self.edge_stiffness)
            && ok(self.volume_stiffness)
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoftSettings {
    pub substeps: u32,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self { substeps: 20 }
    }
}

/// Volume damping time per unit of `damping`, s (as the 2D area damping).
const VOLUME_DAMPING_TIME: f64 = 0.005;

struct Work {
    x: Vec<DVec3>,
    v: Vec<DVec3>,
    prev: Vec<DVec3>,
    w: Vec<f64>,
}

impl Work {
    fn of(s: &SoftBody) -> Self {
        let x: Vec<DVec3> = s.positions.iter().map(|p| p.as_dvec3()).collect();
        Self {
            v: s.velocities.iter().map(|v| v.as_dvec3()).collect(),
            prev: x.clone(),
            x,
            w: s.inv_mass.iter().map(|&w| f64::from(w)).collect(),
        }
    }
}

/// Advance every soft body by `dt`, colliding with `bodies` and each other.
pub fn step(
    soft: &mut [SoftBody],
    bodies: &mut [RigidBody],
    gravity: Vec3,
    dt: f32,
    settings: &SoftSettings,
) {
    if soft.is_empty() {
        return;
    }
    for s in soft.iter_mut() {
        s.velocities.resize(s.positions.len(), Vec3::ZERO);
    }
    let mut work: Vec<Work> = soft.iter().map(Work::of).collect();
    let substeps = settings.substeps.clamp(1, 200);
    let h = f64::from(dt) / f64::from(substeps);
    let g = gravity.as_dvec3();
    for _ in 0..substeps {
        for wk in &mut work {
            wk.prev.copy_from_slice(&wk.x);
            for k in 0..wk.x.len() {
                if wk.w[k] > 0.0 {
                    wk.v[k] += g * h;
                    wk.x[k] += wk.v[k] * h;
                }
            }
        }
        collide_rigid(soft, &mut work, bodies, h);
        for (s, wk) in soft.iter().zip(&mut work) {
            solve_edges(s, wk, h);
            solve_volume(s, wk, h);
        }
        collide_rigid(soft, &mut work, bodies, h);
        collide_soft(soft, &mut work);
        for (s, wk) in soft.iter().zip(&mut work) {
            for k in 0..wk.x.len() {
                wk.v[k] = if wk.w[k] > 0.0 {
                    (wk.x[k] - wk.prev[k]) / h
                } else {
                    DVec3::ZERO
                };
            }
            damp_deformation(s, wk, h);
        }
    }
    for (s, wk) in soft.iter_mut().zip(&work) {
        for k in 0..s.positions.len() {
            s.positions[k] = wk.x[k].as_vec3();
            s.velocities[k] = wk.v[k].as_vec3();
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
        if wi + wj == 0.0 {
            continue;
        }
        let d = wk.x[j] - wk.x[i];
        let len = d.length();
        if len < 1e-12 {
            continue;
        }
        let n = d / len;
        let lambda = -(len - f64::from(e.rest)) / (wi + wj + alpha);
        wk.x[i] -= wi * lambda * n;
        wk.x[j] += wj * lambda * n;
    }
}

/// Volume constraint with XPBD damping (see the 2D area constraint):
/// V = ⅙ Σ a·(b × c), ∇_a V = ⅙ Σ (b × c) over the triangles at a.
fn solve_volume(s: &SoftBody, wk: &mut Work, h: f64) {
    if s.volume_stiffness <= 0.0 || s.triangles.is_empty() {
        return;
    }
    let alpha = 1.0 / (f64::from(s.volume_stiffness) * h * h);
    let gamma = VOLUME_DAMPING_TIME * f64::from(s.damping) / h;
    let mut grads = vec![DVec3::ZERO; wk.x.len()];
    let mut volume = 0.0;
    for t in &s.triangles {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (xa, xb, xc) = (wk.x[a], wk.x[b], wk.x[c]);
        volume += xa.dot(xb.cross(xc)) / 6.0;
        grads[a] += xb.cross(xc) / 6.0;
        grads[b] += xc.cross(xa) / 6.0;
        grads[c] += xa.cross(xb) / 6.0;
    }
    let c = volume - f64::from(s.rest_volume);
    let mut denom = 0.0;
    let mut rate = 0.0;
    for (k, g) in grads.iter().enumerate() {
        denom += wk.w[k] * g.length_squared();
        rate += g.dot(wk.x[k] - wk.prev[k]);
    }
    if denom + alpha <= 0.0 {
        return;
    }
    let lambda = -(c + gamma * rate) / ((1.0 + gamma) * denom + alpha);
    for (k, g) in grads.iter().enumerate() {
        wk.x[k] += wk.w[k] * lambda * *g;
    }
}

fn damp_deformation(s: &SoftBody, wk: &mut Work, h: f64) {
    let factor = (f64::from(s.damping) * h).min(1.0);
    if factor <= 0.0 {
        return;
    }
    let (mut mass, mut x_cm, mut p) = (0.0, DVec3::ZERO, DVec3::ZERO);
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
    // A pinned body can't move rigidly, so all its motion is deformation:
    // damp toward rest (like air drag on hanging cloth or a chain).
    if wk.w.contains(&0.0) {
        for k in 0..wk.x.len() {
            let v = wk.v[k];
            wk.v[k] = v - factor * v;
        }
        return;
    }
    x_cm /= mass;
    let v_cm = p / mass;
    let (mut l, mut inertia) = (DVec3::ZERO, DMat3::ZERO);
    for k in 0..wk.x.len() {
        if wk.w[k] > 0.0 {
            let m = 1.0 / wk.w[k];
            let r = wk.x[k] - x_cm;
            l += m * r.cross(wk.v[k] - v_cm);
            // I += m (|r|² 1 − r rᵀ)
            inertia += DMat3::from_diagonal(DVec3::splat(m * r.length_squared()))
                - DMat3::from_cols(r * (m * r.x), r * (m * r.y), r * (m * r.z));
        }
    }
    let omega = if inertia.determinant().abs() > 1e-18 {
        inertia.inverse() * l
    } else {
        DVec3::ZERO
    };
    for k in 0..wk.x.len() {
        if wk.w[k] > 0.0 {
            let rigid = v_cm + omega.cross(wk.x[k] - x_cm);
            let v = wk.v[k];
            wk.v[k] = v + factor * (rigid - v);
        }
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
                let (lo, hi) = (bb.min.as_dvec3() - r, bb.max.as_dvec3() + r);
                if x.cmplt(lo).any() || x.cmpgt(hi).any() {
                    continue;
                }
                let body = &bodies[bi];
                let (surface, normal, inside) = body.closest_surface_point(x.as_vec3());
                let (surface, normal) = (surface.as_dvec3(), normal.as_dvec3());
                let dist = (x - surface).length();
                let depth = if inside { r + dist } else { r - dist };
                if depth <= 0.0 {
                    continue;
                }
                let arm = surface - body.position.as_dvec3();
                let inv_i = body.inverse_inertia_world().as_dmat3();
                let rn = arm.cross(normal);
                let wb = f64::from(body.inverse_mass()) + rn.dot(inv_i * rn);
                let share = depth / (wp + wb);
                wk.x[k] += wp * share * normal;
                let surface_motion = body.velocity_at(surface.as_vec3()).as_dvec3() * h;
                let slide = (wk.x[k] - wk.prev[k]) - surface_motion;
                let tangential = slide - slide.dot(normal) * normal;
                let t_len = tangential.length();
                if t_len > 1e-15 {
                    let limit = f64::from(s.friction.max(body.friction)) * wp * share;
                    wk.x[k] -= tangential * (limit / t_len).min(1.0);
                }
                if wb > 0.0 {
                    let push = -share * normal;
                    let im = f64::from(body.inverse_mass());
                    let spin = inv_i * arm.cross(push);
                    let body = &mut bodies[bi];
                    body.position += (im * push).as_vec3();
                    body.velocity += (im * push / h).as_vec3();
                    body.angular_velocity += (spin / h).as_vec3();
                    // Small rotation by `spin` radians.
                    let q = body.orientation.as_dquat();
                    let dq = DQuat::from_xyzw(spin.x, spin.y, spin.z, 0.0) * q * 0.5;
                    body.orientation = (q + dq).normalize().as_quat();
                }
            }
        }
    }
}

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
            points.push(p.as_vec3());
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
    fn shapes_are_closed_with_the_right_volume() {
        let b = SoftBody::ball(Vec3::ZERO, 1.0, 2, 500.0);
        assert_eq!(b.len(), 162);
        // A subdivided icosahedron is a little smaller than the sphere.
        let sphere = 4.0 / 3.0 * std::f32::consts::PI;
        assert!(
            b.rest_volume > 0.95 * sphere && b.rest_volume < sphere,
            "{}",
            b.rest_volume
        );
        assert!((b.mass() - 500.0 * b.rest_volume).abs() < 0.01 * b.mass());
        let j = SoftBody::jelly(Vec3::ZERO, Vec3::new(1.0, 0.5, 0.8), 4, 1000.0);
        assert!(
            (j.volume() - 0.4).abs() < 1e-4,
            "jelly volume {}",
            j.volume()
        );
        let c = SoftBody::cloth(Vec3::ZERO, 2.0, 1.0, 10, 5, 0.3);
        assert_eq!(c.len(), 66);
        assert!(b.validate().is_ok() && j.validate().is_ok() && c.validate().is_ok());
    }
}
