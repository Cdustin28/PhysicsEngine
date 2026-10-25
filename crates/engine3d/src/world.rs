//! The 3D world and its time step.

use crate::body::{BodyHandle, BodyId, RigidBody};
use crate::broad_phase;
use crate::collision::{self, Manifold};
use crate::fluid::{CpuSolver, Fluid, FluidSolver};
use crate::solver::{self, SolverSettings};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Contact points closer than this to one from the last step (same body
/// pair) are treated as the same contact for warm starting, m. 3D manifolds
/// don't carry stable feature ids like the 2D ones, so contacts are matched
/// by position instead, as Bullet does.
const WARM_START_MATCH_DISTANCE: f32 = 0.02;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World {
    /// Gravitational acceleration, m/s².
    pub gravity: Vec3,
    pub bodies: Vec<RigidBody>,
    #[serde(default)]
    pub solver: SolverSettings,
    /// SPH water. Left out of scene files when unused.
    #[serde(default, skip_serializing_if = "Fluid::is_unused")]
    pub fluid: Fluid,
    /// Force and torque the fluid exerted on each body in the last step
    /// (N, N·m), for display.
    #[serde(skip)]
    pub fluid_forces: Vec<(Vec3, Vec3)>,
    #[serde(skip)]
    pub time: f64,
    #[serde(skip)]
    contacts: Vec<Manifold>,
    #[serde(skip)]
    next_handle: u64,
}

impl PartialEq for World {
    /// Same scene: gravity, bodies, and solver settings.
    fn eq(&self, other: &Self) -> bool {
        self.gravity == other.gravity
            && self.bodies == other.bodies
            && self.solver == other.solver
            && self.fluid == other.fluid
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world with standard gravity along −y.
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -crate::STANDARD_GRAVITY, 0.0),
            bodies: Vec::new(),
            solver: SolverSettings::default(),
            fluid: Fluid::default(),
            fluid_forces: Vec::new(),
            time: 0.0,
            contacts: Vec::new(),
            next_handle: 1,
        }
    }

    pub fn add_body(&mut self, mut body: RigidBody) -> BodyId {
        body.handle = 0;
        self.bodies.push(body);
        self.assign_handles();
        BodyId(self.bodies.len() - 1)
    }

    /// Give every body without a handle one (bodies loaded from a file or
    /// pushed onto `bodies` directly).
    pub fn assign_handles(&mut self) {
        let highest = self.bodies.iter().map(|b| b.handle).max().unwrap_or(0);
        self.next_handle = self.next_handle.max(highest + 1).max(1);
        for body in &mut self.bodies {
            if body.handle == 0 {
                body.handle = self.next_handle;
                self.next_handle += 1;
            }
        }
    }

    pub fn handle(&mut self, id: BodyId) -> Option<BodyHandle> {
        self.assign_handles();
        self.bodies.get(id.0).and_then(RigidBody::handle)
    }

    pub fn find(&self, handle: BodyHandle) -> Option<BodyId> {
        self.bodies
            .iter()
            .position(|b| b.handle == handle.0)
            .map(BodyId)
    }

    pub fn remove_body(&mut self, id: BodyId) -> Option<RigidBody> {
        if id.0 >= self.bodies.len() {
            return None;
        }
        self.contacts.clear();
        Some(self.bodies.remove(id.0))
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.get(id.0)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.get_mut(id.0)
    }

    pub fn contacts(&self) -> &[Manifold] {
        &self.contacts
    }

    /// Kinetic energy of the bodies and the fluid, J.
    pub fn kinetic_energy(&self) -> f32 {
        self.bodies
            .iter()
            .map(RigidBody::kinetic_energy)
            .sum::<f32>()
            + self.fluid.kinetic_energy()
    }

    pub fn momentum(&self) -> Vec3 {
        self.bodies
            .iter()
            .filter(|b| !b.is_static)
            .map(|b| b.mass() * b.velocity)
            .sum()
    }

    /// The first body hit by the ray `origin + t·direction` (t ≥ 0), and the
    /// distance t to it. `direction` must be unit length. Used for picking
    /// with the mouse.
    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<(BodyId, f32)> {
        self.bodies
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                crate::ray::intersect(b, origin, direction).map(|t| (BodyId(i), t))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Advance by exactly `dt` seconds with the CPU fluid solver.
    pub fn step(&mut self, dt: f32) {
        self.step_with(dt, &mut CpuSolver);
    }

    /// Advance by exactly `dt` seconds (fixed steps, as in 2D): fluid (whose
    /// push on the bodies becomes forces on them), broad phase, narrow
    /// phase, warm-start bookkeeping, velocities, contact solve, positions.
    pub fn step_with(&mut self, dt: f32, fluid_solver: &mut dyn FluidSolver) {
        assert!(
            dt > 0.0 && dt.is_finite(),
            "time step must be positive, got {dt}"
        );
        self.fluid_forces.clear();
        if !self.fluid.is_empty() {
            let boundary = self
                .fluid
                .params
                .boundary(&self.bodies, self.fluid.boundary_region());
            let impulses = fluid_solver.step(
                &mut self.fluid,
                &boundary,
                self.gravity,
                dt,
                self.bodies.len(),
            );
            self.fluid.project_out_of_bodies(&self.bodies);
            self.fluid.remove_escaped();
            for (body, impulse) in self.bodies.iter_mut().zip(&impulses) {
                let (force, torque) = (impulse.linear / dt, impulse.angular / dt);
                if !body.is_static {
                    body.apply_force(force);
                    body.apply_torque(torque);
                }
                self.fluid_forces.push((force, torque));
            }
        }
        let mut contacts: Vec<Manifold> = broad_phase::sweep_and_prune_pairs(&self.bodies)
            .into_iter()
            .filter_map(|(i, j)| collision::collide(i, &self.bodies[i], j, &self.bodies[j]))
            .collect();
        if self.solver.warm_starting {
            carry_over_impulses(&self.contacts, &mut contacts);
        }
        let gravity = self.gravity;
        for body in &mut self.bodies {
            body.integrate_velocity(gravity, dt);
        }
        let correction = solver::solve(&mut self.bodies, &mut contacts, &self.solver, dt);
        for (i, body) in self.bodies.iter_mut().enumerate() {
            body.integrate_position(dt, correction.v[i], correction.w[i]);
        }
        self.contacts = contacts;
        self.time += f64::from(dt);
    }
}

fn carry_over_impulses(old: &[Manifold], new: &mut [Manifold]) {
    let previous: HashMap<(usize, usize), &Manifold> =
        old.iter().map(|m| ((m.a, m.b), m)).collect();
    for m in new {
        let Some(prev) = previous.get(&(m.a, m.b)) else {
            continue;
        };
        // Tangent directions are derived from the normal, so impulses only
        // carry over if the normal hasn't turned much.
        if prev.normal.dot(m.normal) < 0.99 {
            continue;
        }
        m.friction_impulse = prev.friction_impulse;
        m.twist_impulse = prev.twist_impulse;
        for point in &mut m.points {
            let nearest = prev
                .points
                .iter()
                .map(|p| (p, (p.position - point.position).length_squared()))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((p, d2)) = nearest
                && d2 < WARM_START_MATCH_DISTANCE * WARM_START_MATCH_DISTANCE
            {
                point.normal_impulse = p.normal_impulse;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn free_fall_matches_semi_implicit_euler() {
        let mut w = World::new();
        let id = w.add_body(RigidBody::new(
            Shape::Sphere { radius: 0.1 },
            Vec3::new(0.0, 100.0, 0.0),
        ));
        let (dt, n) = (1.0 / 120.0_f32, 120);
        for _ in 0..n {
            w.step(dt);
        }
        let g = crate::STANDARD_GRAVITY;
        let nf = n as f32;
        let expected = 100.0 - g * dt * dt * nf * (nf + 1.0) / 2.0;
        assert!((w.body(id).unwrap().position.y - expected).abs() < 1e-3);
    }

    #[test]
    fn handles_survive_removal() {
        let mut w = World::new();
        let a = w.add_body(RigidBody::new(Shape::cube(1.0), Vec3::ZERO));
        let b = w.add_body(RigidBody::new(Shape::cube(1.0), Vec3::X));
        let hb = w.handle(b).unwrap();
        w.remove_body(a);
        assert_eq!(w.find(hb), Some(BodyId(0)));
    }

    #[test]
    fn raycast_hits_nearest() {
        let mut w = World::new();
        w.add_body(RigidBody::new(Shape::cube(1.0), Vec3::new(0.0, 0.0, -5.0)));
        let near = w.add_body(RigidBody::new(
            Shape::Sphere { radius: 0.5 },
            Vec3::new(0.0, 0.0, -2.0),
        ));
        let (id, t) = w.raycast(Vec3::ZERO, -Vec3::Z).unwrap();
        assert_eq!(id, near);
        assert!((t - 1.5).abs() < 1e-5);
        assert!(w.raycast(Vec3::ZERO, Vec3::Z).is_none());
    }
}
