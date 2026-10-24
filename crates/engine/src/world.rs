//! The simulation world and its time stepping.

use crate::body::{BodyId, RigidBody};
use crate::broad_phase;
use crate::collision::{self, Manifold};
use crate::solver::{self, SolverSettings};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Everything that is simulated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct World {
    /// Gravitational acceleration, m/s².
    pub gravity: Vec2,
    pub bodies: Vec<RigidBody>,
    /// Contact solver tuning. Optional in scene files; older scenes without
    /// it get the defaults.
    #[serde(default)]
    pub solver: SolverSettings,
    /// Simulated time elapsed, s. Not saved in scene files.
    #[serde(skip)]
    pub time: f64,
    /// Contacts found during the last step, including their accumulated
    /// impulses for warm starting. Not saved in scene files.
    #[serde(skip)]
    contacts: Vec<Manifold>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world with standard Earth gravity.
    pub fn new() -> Self {
        Self {
            gravity: Vec2::new(0.0, -crate::STANDARD_GRAVITY),
            bodies: Vec::new(),
            solver: SolverSettings::default(),
            time: 0.0,
            contacts: Vec::new(),
        }
    }

    /// The scene the editor opens with. See [`crate::presets`].
    pub fn demo() -> Self {
        crate::presets::demo()
    }

    pub fn add_body(&mut self, body: RigidBody) -> BodyId {
        self.bodies.push(body);
        BodyId(self.bodies.len() - 1)
    }

    /// Remove a body. Note that this shifts the indices of every body after
    /// it, so any `BodyId`s you were holding past `id` are now stale.
    pub fn remove_body(&mut self, id: BodyId) -> Option<RigidBody> {
        if id.0 >= self.bodies.len() {
            return None;
        }
        // Cached contacts refer to bodies by index, so they're now wrong.
        self.contacts.clear();
        Some(self.bodies.remove(id.0))
    }

    pub fn body(&self, id: BodyId) -> Option<&RigidBody> {
        self.bodies.get(id.0)
    }

    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut RigidBody> {
        self.bodies.get_mut(id.0)
    }

    /// Topmost body containing `point`. Later bodies are drawn on top, so we
    /// search back to front.
    pub fn body_at_point(&self, point: Vec2) -> Option<BodyId> {
        self.bodies
            .iter()
            .rposition(|b| b.contains_point(point))
            .map(BodyId)
    }

    /// Contacts from the most recent step, for debug drawing.
    pub fn contacts(&self) -> &[Manifold] {
        &self.contacts
    }

    /// Total kinetic energy, J.
    pub fn kinetic_energy(&self) -> f32 {
        self.bodies.iter().map(RigidBody::kinetic_energy).sum()
    }

    /// Total linear momentum of the dynamic bodies, kg·m/s.
    pub fn momentum(&self) -> Vec2 {
        self.bodies
            .iter()
            .filter(|b| !b.is_static)
            .map(|b| b.mass() * b.velocity)
            .sum()
    }

    /// Advance the simulation by exactly `dt` seconds.
    ///
    /// Call this with a *fixed* `dt` (see [`FixedTimestep`]). Variable time
    /// steps make the simulation non-deterministic and less stable.
    ///
    /// One step is:
    /// 1. **Broad phase:** find pairs whose bounding boxes overlap.
    /// 2. **Narrow phase:** compute exact contact points for those pairs.
    /// 3. **Warm start bookkeeping:** carry impulses over from last step's
    ///    matching contacts.
    /// 4. **Integrate velocities:** apply gravity and forces.
    /// 5. **Solve contacts:** adjust velocities so bodies stop moving into each
    ///    other, and compute a separate position correction for any overlap.
    /// 6. **Integrate positions** with the corrected velocities, plus the
    ///    position correction.
    pub fn step(&mut self, dt: f32) {
        assert!(
            dt > 0.0 && dt.is_finite(),
            "time step must be positive, got {dt}"
        );

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
            body.integrate_position(dt);
            // Push overlapping bodies apart without changing their velocity.
            if !body.is_static {
                body.position += correction.v[i] * dt;
                body.angle += correction.w[i] * dt;
            }
        }

        self.contacts = contacts;
        self.time += f64::from(dt);
    }
}

/// Copy accumulated impulses from last step's contacts onto this step's,
/// matching by body pair and feature id.
fn carry_over_impulses(old: &[Manifold], new: &mut [Manifold]) {
    let previous: HashMap<(usize, usize), &Manifold> =
        old.iter().map(|m| ((m.a, m.b), m)).collect();
    for m in new {
        let Some(prev) = previous.get(&(m.a, m.b)) else {
            continue;
        };
        for point in &mut m.points {
            if let Some(p) = prev.points.iter().find(|p| p.feature == point.feature) {
                point.normal_impulse = p.normal_impulse;
                point.tangent_impulse = p.tangent_impulse;
            }
        }
    }
}

/// Turns variable frame times into a whole number of fixed physics steps.
///
/// The leftover fraction of a step carries over to the next frame, so the
/// simulation runs at real-time speed regardless of frame rate. If a frame
/// takes too long, at most `max_steps_per_frame` steps run and the rest of
/// the backlog is dropped; otherwise a slow frame causes more steps, which
/// causes a slower frame (the "spiral of death").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedTimestep {
    pub dt: f32,
    pub max_steps_per_frame: u32,
    accumulator: f32,
}

impl FixedTimestep {
    pub fn new(dt: f32) -> Self {
        Self {
            dt,
            max_steps_per_frame: 8,
            accumulator: 0.0,
        }
    }

    /// Add `frame_time` seconds of real time and return how many fixed steps
    /// to run now.
    pub fn advance(&mut self, frame_time: f32) -> u32 {
        self.accumulator += frame_time.max(0.0);
        let mut steps = 0;
        while self.accumulator >= self.dt && steps < self.max_steps_per_frame {
            self.accumulator -= self.dt;
            steps += 1;
        }
        if steps == self.max_steps_per_frame {
            // Drop the backlog rather than trying to catch up forever.
            self.accumulator = self.accumulator.min(self.dt);
        }
        steps
    }

    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }
}

impl Default for FixedTimestep {
    /// 120 Hz: a common choice for rigid bodies, and SPH usually wants
    /// something at least this small.
    fn default() -> Self {
        Self::new(1.0 / 120.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    /// Semi-implicit Euler has a closed form for constant acceleration:
    /// after n steps, y_n = y_0 + v_0 n dt - g dt² n(n+1)/2.
    #[test]
    fn free_fall_matches_semi_implicit_euler_closed_form() {
        let mut w = World::new();
        let g = w.gravity.y.abs();
        let id = w.add_body(RigidBody::new(
            Shape::Circle { radius: 0.1 },
            Vec2::new(0.0, 100.0),
        ));

        let dt = 1.0 / 120.0;
        let n = 120; // one second
        for _ in 0..n {
            w.step(dt);
        }

        let nf = n as f32;
        let expected_y = 100.0 - g * dt * dt * nf * (nf + 1.0) / 2.0;
        let y = w.body(id).unwrap().position.y;
        assert!(
            (y - expected_y).abs() < 1e-3,
            "y = {y}, expected {expected_y}"
        );

        // Sanity check against the exact answer, 100 - g t²/2 = 95.097 m.
        // Semi-implicit Euler overshoots by g dt t / 2 ≈ 0.041 m.
        let exact = 100.0 - 0.5 * g;
        assert!((y - exact).abs() < 0.05, "y = {y}, exact {exact}");

        // v = -g t exactly, because velocity integration has no error here.
        let vy = w.body(id).unwrap().velocity.y;
        assert!((vy + g).abs() < 1e-3);
    }

    #[test]
    fn static_bodies_do_not_move() {
        let mut w = World::demo();
        let floor_before = w.bodies[0].position;
        for _ in 0..100 {
            w.step(1.0 / 120.0);
        }
        assert_eq!(w.bodies[0].position, floor_before);
    }

    #[test]
    fn picking_prefers_topmost_body() {
        let mut w = World::new();
        w.add_body(RigidBody::new(Shape::rect(2.0, 2.0), Vec2::ZERO));
        let top = w.add_body(RigidBody::new(Shape::Circle { radius: 0.5 }, Vec2::ZERO));
        assert_eq!(w.body_at_point(Vec2::ZERO), Some(top));
        assert_eq!(w.body_at_point(Vec2::new(5.0, 5.0)), None);
    }

    #[test]
    fn removing_a_body_clears_cached_contacts() {
        let mut w = World::demo();
        for _ in 0..240 {
            w.step(1.0 / 120.0);
        }
        assert!(!w.contacts().is_empty());
        w.remove_body(BodyId(1));
        assert!(w.contacts().is_empty());
        w.step(1.0 / 120.0); // must not panic on stale indices
    }

    #[test]
    fn fixed_timestep_carries_remainder_and_caps_steps() {
        let mut ts = FixedTimestep::new(0.01);
        assert_eq!(ts.advance(0.025), 2); // 0.005 left over
        assert_eq!(ts.advance(0.006), 1); // 0.011 -> 1 step
        assert_eq!(ts.advance(10.0), ts.max_steps_per_frame); // backlog capped
        assert!(ts.advance(0.0) <= 1);
    }
}
