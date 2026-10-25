//! 3D rigid bodies.

use crate::shape::Shape;
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Index of a body in [`World::bodies`](crate::World::bodies). Shifts when
/// an earlier body is removed; use a [`BodyHandle`] to refer to a body
/// across removals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyId(pub usize);

/// A name for a body that stays valid while others are added and removed.
/// Never reused. Look it up with [`World::find`](crate::World::find).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle(pub(crate) u64);

/// A 3D rigid body. Mass and inertia derive from `shape` and `density`, so
/// they can't go stale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidBody {
    pub shape: Shape,
    /// Center of mass, m.
    pub position: Vec3,
    /// Rotation from the body's frame to the world. Kept normalized.
    pub orientation: Quat,
    /// Linear velocity of the center of mass, m/s.
    pub velocity: Vec3,
    /// Angular velocity in world coordinates, rad/s (axis × rate).
    pub angular_velocity: Vec3,
    /// Density, kg/m³. Water is 1000. Ignored for static bodies.
    pub density: f32,
    /// Coefficient of restitution in [0, 1].
    pub restitution: f32,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Static bodies have infinite mass and never move.
    pub is_static: bool,

    #[serde(skip)]
    force: Vec3,
    #[serde(skip)]
    torque: Vec3,
    #[serde(skip)]
    pub(crate) handle: u64,
}

impl PartialEq for RigidBody {
    /// Physically identical; the world-assigned handle doesn't count.
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.position == other.position
            && self.orientation == other.orientation
            && self.velocity == other.velocity
            && self.angular_velocity == other.angular_velocity
            && self.density == other.density
            && self.restitution == other.restitution
            && self.friction == other.friction
            && self.is_static == other.is_static
    }
}

impl RigidBody {
    /// Default density, kg/m³: about that of pine wood, so a default body
    /// floats half-submerged in water.
    pub const DEFAULT_DENSITY: f32 = 500.0;

    /// A dynamic body at rest at `position`, unrotated.
    pub fn new(shape: Shape, position: Vec3) -> Self {
        Self {
            shape,
            position,
            orientation: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            density: Self::DEFAULT_DENSITY,
            restitution: 0.3,
            friction: 0.5,
            is_static: false,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            handle: 0,
        }
    }

    /// A static body at `position`.
    pub fn new_static(shape: Shape, position: Vec3) -> Self {
        Self {
            is_static: true,
            ..Self::new(shape, position)
        }
    }

    pub fn with_orientation(mut self, orientation: Quat) -> Self {
        self.orientation = orientation.normalize();
        self
    }

    pub fn handle(&self) -> Option<BodyHandle> {
        (self.handle != 0).then_some(BodyHandle(self.handle))
    }

    /// Mass in kg (`f32::INFINITY` for static bodies).
    pub fn mass(&self) -> f32 {
        if self.is_static {
            f32::INFINITY
        } else {
            self.density * self.shape.volume()
        }
    }

    pub fn inverse_mass(&self) -> f32 {
        if self.is_static {
            0.0
        } else {
            1.0 / self.mass()
        }
    }

    /// Inverse inertia tensor in world coordinates: R I⁻¹ Rᵀ. Zero for
    /// static bodies, so the solver needs no special cases.
    pub fn inverse_inertia_world(&self) -> Mat3 {
        if self.is_static {
            return Mat3::ZERO;
        }
        let r = Mat3::from_quat(self.orientation);
        let inv_local = Mat3::from_diagonal(self.shape.principal_inertia(self.mass()).recip());
        r * inv_local * r.transpose()
    }

    /// Inertia tensor in world coordinates, kg·m².
    pub fn inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.orientation);
        r * Mat3::from_diagonal(self.shape.principal_inertia(self.mass())) * r.transpose()
    }

    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        self.position + self.orientation * local
    }

    pub fn world_to_local(&self, world: Vec3) -> Vec3 {
        self.orientation.inverse() * (world - self.position)
    }

    /// Velocity of the material at a world point: v + ω × r.
    pub fn velocity_at(&self, world: Vec3) -> Vec3 {
        if self.is_static {
            Vec3::ZERO
        } else {
            self.velocity + self.angular_velocity.cross(world - self.position)
        }
    }

    pub fn contains_point(&self, world: Vec3) -> bool {
        self.shape.contains_local_point(self.world_to_local(world))
    }

    /// The body's local axes in world space (columns of its rotation).
    pub fn axes(&self) -> [Vec3; 3] {
        let m = Mat3::from_quat(self.orientation);
        [m.x_axis, m.y_axis, m.z_axis]
    }

    /// Kinetic energy, J: ½ m v² + ½ ωᵀ I ω.
    pub fn kinetic_energy(&self) -> f32 {
        if self.is_static {
            return 0.0;
        }
        let w = self.angular_velocity;
        0.5 * self.mass() * self.velocity.length_squared() + 0.5 * w.dot(self.inertia_world() * w)
    }

    pub fn apply_force(&mut self, force: Vec3) {
        self.force += force;
    }

    /// A force at a world point also produces torque τ = r × F.
    pub fn apply_force_at_point(&mut self, force: Vec3, world_point: Vec3) {
        self.force += force;
        self.torque += (world_point - self.position).cross(force);
    }

    pub fn apply_torque(&mut self, torque: Vec3) {
        self.torque += torque;
    }

    /// First half of a step: gravity and accumulated forces into velocity.
    ///
    /// The gyroscopic term ω × (Iω) is left out, as in many game engines:
    /// explicit integration of it adds energy and destabilizes fast
    /// spinners. It matters for tumbling non-symmetric bodies (the "tennis
    /// racket" effect) and is a candidate for later.
    pub(crate) fn integrate_velocity(&mut self, gravity: Vec3, dt: f32) {
        if !self.is_static {
            self.velocity += (gravity + self.force * self.inverse_mass()) * dt;
            self.angular_velocity += self.inverse_inertia_world() * self.torque * dt;
        }
        self.force = Vec3::ZERO;
        self.torque = Vec3::ZERO;
    }

    /// Second half: move and turn. `extra_v`/`extra_w` are position-only
    /// corrections from the solver (split impulse); they move the body
    /// without changing its velocity.
    pub(crate) fn integrate_position(&mut self, dt: f32, extra_v: Vec3, extra_w: Vec3) {
        if self.is_static {
            return;
        }
        self.position += (self.velocity + extra_v) * dt;
        let w = self.angular_velocity + extra_w;
        // dq/dt = ½ (0, ω) q
        let spin = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * self.orientation;
        self.orientation = (self.orientation + spin * (0.5 * dt)).normalize();
    }

    pub fn validate(&self) -> Result<(), String> {
        self.shape.validate()?;
        if !self.is_static && !(self.density.is_finite() && self.density > 0.0) {
            return Err(format!("density must be positive, got {}", self.density));
        }
        let finite = self.position.is_finite()
            && self.velocity.is_finite()
            && self.angular_velocity.is_finite()
            && self.orientation.is_finite();
        if !finite {
            return Err("position, orientation, and velocities must be finite".into());
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
    fn world_inertia_rotates_with_the_body() {
        // A long box along x; rotated 90° about z it's long along y.
        let b = RigidBody::new(Shape::cuboid(4.0, 1.0, 1.0), Vec3::ZERO);
        let i = b.inertia_world();
        let r = b.with_orientation(Quat::from_rotation_z(FRAC_PI_2));
        let ir = r.inertia_world();
        assert!((ir.x_axis.x - i.y_axis.y).abs() < 1e-3);
        assert!((ir.y_axis.y - i.x_axis.x).abs() < 1e-3);
        // Inverse really is the inverse.
        let prod = ir * r.inverse_inertia_world();
        assert!((prod - Mat3::IDENTITY).abs_diff_eq(Mat3::ZERO, 1e-4));
    }

    #[test]
    fn spin_integration_turns_the_body() {
        let mut b = RigidBody::new(Shape::cube(1.0), Vec3::ZERO);
        b.angular_velocity = Vec3::new(0.0, 0.0, FRAC_PI_2); // 90°/s about z
        for _ in 0..1200 {
            b.integrate_position(1.0 / 1200.0, Vec3::ZERO, Vec3::ZERO);
        }
        let x = b.orientation * Vec3::X;
        assert!((x - Vec3::Y).length() < 1e-3, "x axis ended at {x}");
    }

    #[test]
    fn off_center_force_spins() {
        let mut b = RigidBody::new(Shape::cube(1.0), Vec3::ZERO);
        b.apply_force_at_point(Vec3::Y, Vec3::new(0.5, 0.0, 0.0));
        b.integrate_velocity(Vec3::ZERO, 0.1);
        assert!(b.angular_velocity.z > 0.0);
    }
}
