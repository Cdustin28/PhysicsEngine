//! The simulation world and its time stepping.

use crate::body::{BodyId, RigidBody};
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Everything that is simulated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct World {
    /// Gravitational acceleration, m/s².
    pub gravity: Vec2,
    pub bodies: Vec<RigidBody>,
    /// Simulated time elapsed, s. Not saved in scene files.
    #[serde(skip)]
    pub time: f64,
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
            time: 0.0,
        }
    }

    /// The scene the editor opens with: a static floor and two objects.
    pub fn demo() -> Self {
        use crate::shape::Shape;
        let mut w = Self::new();
        w.add_body(RigidBody::new_static(
            Shape::rect(20.0, 0.5),
            Vec2::new(0.0, -0.25),
        ));
        w.add_body(RigidBody::new(
            Shape::Circle { radius: 0.5 },
            Vec2::new(-1.5, 4.0),
        ));
        w.add_body(RigidBody::new(Shape::rect(1.0, 1.0), Vec2::new(1.5, 5.0)).with_angle(0.4));
        w
    }

    pub fn add_body(&mut self, body: RigidBody) -> BodyId {
        self.bodies.push(body);
        BodyId(self.bodies.len() - 1)
    }

    /// Remove a body. Note that this shifts the indices of every body after
    /// it, so any `BodyId`s you were holding past `id` are now stale.
    pub fn remove_body(&mut self, id: BodyId) -> Option<RigidBody> {
        (id.0 < self.bodies.len()).then(|| self.bodies.remove(id.0))
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

    /// Total kinetic energy, J. Handy for spotting instability: with no
    /// collisions or forces other than gravity, KE + PE should stay constant.
    pub fn kinetic_energy(&self) -> f32 {
        self.bodies.iter().map(RigidBody::kinetic_energy).sum()
    }

    /// Advance the simulation by exactly `dt` seconds.
    ///
    /// Call this with a *fixed* `dt` (see [`FixedTimestep`]); variable time
    /// steps make the simulation non-deterministic and, once collisions and
    /// fluids land, unstable.
    pub fn step(&mut self, dt: f32) {
        assert!(
            dt > 0.0 && dt.is_finite(),
            "time step must be positive, got {dt}"
        );

        // Milestone 1 inserts collision detection and response here:
        //   broad phase -> narrow phase -> contact solver.
        // Milestone 2 adds the SPH fluid step and fluid <-> body coupling.

        let gravity = self.gravity;
        for body in &mut self.bodies {
            body.integrate(gravity, dt);
        }
        self.time += f64::from(dt);
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
    fn fixed_timestep_carries_remainder_and_caps_steps() {
        let mut ts = FixedTimestep::new(0.01);
        assert_eq!(ts.advance(0.025), 2); // 0.005 left over
        assert_eq!(ts.advance(0.006), 1); // 0.011 -> 1 step
        assert_eq!(ts.advance(10.0), ts.max_steps_per_frame); // backlog capped
        assert!(ts.advance(0.0) <= 1);
    }
}
