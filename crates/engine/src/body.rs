//! Rigid bodies.

use crate::shape::Shape;
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Index of a body inside a [`World`](crate::World).
///
/// Handles are plain indices into `World::bodies`. Removing a body
/// invalidates any `BodyId` greater than or equal to the removed one, so the
/// editor clears its selection whenever it removes bodies. Generational
/// handles are on the roadmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BodyId(pub usize);

/// A 2D rigid body.
///
/// Mass and moment of inertia are *derived* from `shape` and `density` every
/// time they are needed rather than cached. That costs a few multiplies but
/// means they can never go stale when the editor changes a shape or density.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub shape: Shape,
    /// Center of mass in world space, m.
    pub position: Vec2,
    /// Orientation, rad (counter-clockwise).
    pub angle: f32,
    /// Linear velocity of the center of mass, m/s.
    pub velocity: Vec2,
    /// Angular velocity, rad/s (counter-clockwise).
    pub angular_velocity: f32,
    /// Areal density, kg/m². Ignored for static bodies.
    pub density: f32,
    /// Coefficient of restitution in [0, 1]. Used once collisions land (M1).
    pub restitution: f32,
    /// Coulomb friction coefficient. Used once collisions land (M1).
    pub friction: f32,
    /// Static bodies have infinite mass and never move (floors, walls).
    pub is_static: bool,

    /// Force accumulated this step, N. Cleared after every step.
    #[serde(skip)]
    force: Vec2,
    /// Torque accumulated this step, N·m. Cleared after every step.
    #[serde(skip)]
    torque: f32,
}

impl RigidBody {
    /// Default density: roughly a dense plastic slab 1 cm thick
    /// (1000 kg/m³ × 0.01 m = 10 kg/m²). Picked so default objects have
    /// sensible masses at human scale.
    pub const DEFAULT_DENSITY: f32 = 10.0;

    /// A dynamic body at rest at `position`.
    pub fn new(shape: Shape, position: Vec2) -> Self {
        Self {
            shape,
            position,
            angle: 0.0,
            velocity: Vec2::ZERO,
            angular_velocity: 0.0,
            density: Self::DEFAULT_DENSITY,
            restitution: 0.3,
            friction: 0.5,
            is_static: false,
            force: Vec2::ZERO,
            torque: 0.0,
        }
    }

    /// A static (immovable) body at `position`.
    pub fn new_static(shape: Shape, position: Vec2) -> Self {
        Self {
            is_static: true,
            ..Self::new(shape, position)
        }
    }

    /// Builder-style helper to set the orientation.
    pub fn with_angle(mut self, angle: f32) -> Self {
        self.angle = angle;
        self
    }

    /// Mass in kg. Static bodies report `f32::INFINITY`.
    pub fn mass(&self) -> f32 {
        if self.is_static {
            f32::INFINITY
        } else {
            self.density * self.shape.area()
        }
    }

    /// 1 / mass, or 0 for static bodies. Solvers work with inverse mass so
    /// that infinite mass is just a zero, with no special cases.
    pub fn inverse_mass(&self) -> f32 {
        if self.is_static {
            0.0
        } else {
            1.0 / self.mass()
        }
    }

    /// Moment of inertia about the center of mass, kg·m².
    pub fn inertia(&self) -> f32 {
        if self.is_static {
            f32::INFINITY
        } else {
            self.shape.moment_of_inertia(self.mass())
        }
    }

    /// 1 / inertia, or 0 for static bodies.
    pub fn inverse_inertia(&self) -> f32 {
        if self.is_static {
            0.0
        } else {
            1.0 / self.inertia()
        }
    }

    /// Unit vector `(cos θ, sin θ)` used to rotate local vectors into world
    /// space with [`Vec2::rotate`].
    pub fn rotation(&self) -> Vec2 {
        Vec2::from_angle(self.angle)
    }

    /// Transform a point from the body's local frame to world space.
    pub fn local_to_world(&self, local: Vec2) -> Vec2 {
        self.position + self.rotation().rotate(local)
    }

    /// Transform a point from world space to the body's local frame.
    pub fn world_to_local(&self, world: Vec2) -> Vec2 {
        Vec2::from_angle(-self.angle).rotate(world - self.position)
    }

    /// Whether a world-space point lies inside this body.
    pub fn contains_point(&self, world: Vec2) -> bool {
        self.shape.contains_local_point(self.world_to_local(world))
    }

    /// World-space corners of a box, counter-clockwise starting bottom-left.
    /// Returns `None` for circles.
    pub fn box_corners(&self) -> Option<[Vec2; 4]> {
        match self.shape {
            Shape::Box { half_extents: h } => Some(
                [
                    Vec2::new(-h.x, -h.y),
                    Vec2::new(h.x, -h.y),
                    Vec2::new(h.x, h.y),
                    Vec2::new(-h.x, h.y),
                ]
                .map(|c| self.local_to_world(c)),
            ),
            Shape::Circle { .. } => None,
        }
    }

    /// Kinetic energy (translational + rotational), J.
    pub fn kinetic_energy(&self) -> f32 {
        if self.is_static {
            return 0.0;
        }
        0.5 * self.mass() * self.velocity.length_squared()
            + 0.5 * self.inertia() * self.angular_velocity * self.angular_velocity
    }

    /// Apply a force through the center of mass for the next step.
    pub fn apply_force(&mut self, force: Vec2) {
        self.force += force;
    }

    /// Apply a force at a world-space point. Off-center forces also produce
    /// torque: τ = r × F (the 2D cross product is a scalar).
    pub fn apply_force_at_point(&mut self, force: Vec2, world_point: Vec2) {
        self.force += force;
        self.torque += (world_point - self.position).perp_dot(force);
    }

    /// Apply a pure torque for the next step.
    pub fn apply_torque(&mut self, torque: f32) {
        self.torque += torque;
    }

    /// Advance velocity then position by `dt` seconds using semi-implicit
    /// (symplectic) Euler, then clear accumulated forces.
    ///
    /// Updating velocity *before* position is what makes this "semi-implicit".
    /// It is just as cheap as explicit Euler but does not gain energy over
    /// time, which keeps orbits and springs stable.
    ///
    /// `World::step` calls the two halves separately so the contact solver
    /// can adjust velocities in between.
    #[cfg(test)]
    pub(crate) fn integrate(&mut self, gravity: Vec2, dt: f32) {
        self.integrate_velocity(gravity, dt);
        self.integrate_position(dt);
    }

    /// First half of a step: apply gravity and accumulated forces to the
    /// velocities, then clear the accumulators.
    pub(crate) fn integrate_velocity(&mut self, gravity: Vec2, dt: f32) {
        if !self.is_static {
            let linear_accel = gravity + self.force * self.inverse_mass();
            let angular_accel = self.torque * self.inverse_inertia();
            self.velocity += linear_accel * dt;
            self.angular_velocity += angular_accel * dt;
        }
        self.force = Vec2::ZERO;
        self.torque = 0.0;
    }

    /// Second half of a step: move by the (now contact-corrected) velocity.
    pub(crate) fn integrate_position(&mut self, dt: f32) {
        if !self.is_static {
            self.position += self.velocity * dt;
            self.angle += self.angular_velocity * dt;
        }
    }

    /// Check that the body's parameters are physically meaningful.
    pub fn validate(&self) -> Result<(), String> {
        self.shape.validate()?;
        if !self.is_static && !(self.density.is_finite() && self.density > 0.0) {
            return Err(format!("density must be positive, got {}", self.density));
        }
        if !(self.position.is_finite() && self.velocity.is_finite()) {
            return Err("position and velocity must be finite".into());
        }
        if !(self.angle.is_finite() && self.angular_velocity.is_finite()) {
            return Err("angle and angular velocity must be finite".into());
        }
        if !(0.0..=1.0).contains(&self.restitution) {
            return Err(format!(
                "restitution must be in [0, 1], got {}",
                self.restitution
            ));
        }
        if !(self.friction.is_finite() && self.friction >= 0.0) {
            return Err(format!(
                "friction must be non-negative, got {}",
                self.friction
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn mass_follows_density_and_area() {
        let b = RigidBody::new(Shape::rect(2.0, 0.5), Vec2::ZERO); // 1 m²
        assert!((b.mass() - RigidBody::DEFAULT_DENSITY).abs() < 1e-5);
        assert!((b.inverse_mass() * b.mass() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn static_bodies_have_zero_inverse_mass() {
        let b = RigidBody::new_static(Shape::rect(10.0, 1.0), Vec2::ZERO);
        assert_eq!(b.inverse_mass(), 0.0);
        assert_eq!(b.inverse_inertia(), 0.0);
        assert_eq!(b.kinetic_energy(), 0.0);
    }

    #[test]
    fn rotated_box_contains_point() {
        // A 4 x 1 box rotated 90° is 1 wide and 4 tall.
        let b = RigidBody::new(Shape::rect(4.0, 1.0), Vec2::new(10.0, 0.0)).with_angle(FRAC_PI_2);
        assert!(b.contains_point(Vec2::new(10.0, 1.9)));
        assert!(!b.contains_point(Vec2::new(11.9, 0.0)));
    }

    #[test]
    fn local_world_round_trip() {
        let b = RigidBody::new(Shape::Circle { radius: 1.0 }, Vec2::new(3.0, -2.0)).with_angle(0.7);
        let p = Vec2::new(0.25, -0.4);
        let back = b.world_to_local(b.local_to_world(p));
        assert!((back - p).length() < 1e-5);
    }

    #[test]
    fn off_center_force_produces_torque() {
        let mut b = RigidBody::new(Shape::rect(2.0, 2.0), Vec2::ZERO);
        // Push up on the right edge: should spin counter-clockwise.
        b.apply_force_at_point(Vec2::new(0.0, 10.0), Vec2::new(1.0, 0.0));
        b.integrate(Vec2::ZERO, 0.1);
        assert!(b.angular_velocity > 0.0);
        assert!(b.velocity.y > 0.0);
    }
}
