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
    /// A temperature this body holds its surface at, K (a heater or cooler
    /// plate), or `None` to use the fluid's wall setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

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
            && self.temperature == other.temperature
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
            temperature: None,
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

    /// Nearest point on the surface to a world point, the outward normal
    /// there, and whether the point is inside the body.
    pub fn closest_surface_point(&self, world: Vec3) -> (Vec3, Vec3, bool) {
        let local = self.world_to_local(world);
        let (surface, normal, inside) = match self.shape {
            Shape::Sphere { radius } => {
                let len = local.length();
                let n = if len > 1e-9 { local / len } else { Vec3::Y };
                (n * radius, n, len < radius)
            }
            Shape::Cuboid { half_extents: h } => {
                let clamped = local.clamp(-h, h);
                if clamped != local {
                    (clamped, (local - clamped).normalize_or(Vec3::Y), false)
                } else {
                    // Inside: leave through the nearest face.
                    let gap = h - local.abs();
                    let k = if gap.x <= gap.y && gap.x <= gap.z {
                        0
                    } else if gap.y <= gap.z {
                        1
                    } else {
                        2
                    };
                    let sign = if local[k] < 0.0 { -1.0 } else { 1.0 };
                    let mut surface = local;
                    surface[k] = sign * h[k];
                    let mut n = Vec3::ZERO;
                    n[k] = sign;
                    (surface, n, true)
                }
            }
        };
        (
            self.local_to_world(surface),
            self.orientation * normal,
            inside,
        )
    }

    /// Points about `spacing` apart over the body's surface, moved `inset`
    /// inward (never more than halfway to the center), in world space. Only
    /// the part of the surface inside `region` (world min, max corners) is
    /// sampled: a 20 m floor at fluid resolution would otherwise be hundreds
    /// of thousands of points, nearly all far from any water. Used as the
    /// fluid's boundary particles.
    pub fn sample_surface(
        &self,
        spacing: f32,
        inset: f32,
        region: (Vec3, Vec3),
        out: &mut Vec<Vec3>,
    ) {
        let inside_region = |p: Vec3| p.cmpge(region.0).all() && p.cmple(region.1).all();
        match self.shape {
            Shape::Sphere { radius } => {
                let r = (radius - inset).max(0.5 * radius);
                // Fibonacci sphere: near-uniform points, no clustering at
                // the poles.
                let n = ((4.0 * std::f32::consts::PI * r * r / (spacing * spacing)).ceil()
                    as usize)
                    .max(12);
                let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
                for k in 0..n {
                    let y = 1.0 - 2.0 * (k as f32 + 0.5) / n as f32;
                    let ring = (1.0 - y * y).max(0.0).sqrt();
                    let (s, c) = (golden * k as f32).sin_cos();
                    let p = self.local_to_world(Vec3::new(ring * c, y, ring * s) * r);
                    if inside_region(p) {
                        out.push(p);
                    }
                }
            }
            Shape::Cuboid { half_extents } => {
                let h = (half_extents - Vec3::splat(inset)).max(0.5 * half_extents);
                // Lattice over the box with n[k] intervals per axis; only
                // points with an index on a face are kept.
                let n = (2.0 * h / spacing).ceil().max(Vec3::ONE);
                let step = 2.0 * h / n;
                let n = [n.x as i32, n.y as i32, n.z as i32];
                // The region in the box's frame (a conservative box around
                // its 8 transformed corners), as index ranges.
                let mut lo = Vec3::splat(f32::INFINITY);
                let mut hi = Vec3::splat(f32::NEG_INFINITY);
                for c in 0..8 {
                    let corner = Vec3::new(
                        if c & 1 == 0 { region.0.x } else { region.1.x },
                        if c & 2 == 0 { region.0.y } else { region.1.y },
                        if c & 4 == 0 { region.0.z } else { region.1.z },
                    );
                    let l = self.world_to_local(corner);
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
                let range = |k: usize| {
                    let a = ((lo[k] + h[k]) / step[k]).floor().max(0.0) as i32;
                    let b = ((hi[k] + h[k]) / step[k]).ceil().min(n[k] as f32) as i32;
                    a..=b
                };
                let point = |i: i32, j: i32, k: i32| {
                    self.local_to_world(-h + Vec3::new(i as f32, j as f32, k as f32) * step)
                };
                let (rx, ry, rz) = (range(0), range(1), range(2));
                for i in rx {
                    for j in ry.clone() {
                        let on_face = i == 0 || i == n[0] || j == 0 || j == n[1];
                        if on_face {
                            for k in rz.clone() {
                                out.push(point(i, j, k));
                            }
                        } else {
                            for k in [0, n[2]] {
                                if rz.contains(&k) {
                                    out.push(point(i, j, k));
                                }
                            }
                        }
                    }
                }
            }
        }
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

#[cfg(test)]
mod surface_tests {
    use super::*;

    const EVERYWHERE: (Vec3, Vec3) = (Vec3::splat(-1e6), Vec3::splat(1e6));

    #[test]
    fn box_surface_points_are_on_the_surface_and_unique() {
        let b = RigidBody::new(Shape::cuboid(1.0, 0.5, 0.3), Vec3::new(1.0, 2.0, 3.0))
            .with_orientation(Quat::from_rotation_y(0.7));
        let mut pts = Vec::new();
        b.sample_surface(0.05, 0.0, EVERYWHERE, &mut pts);
        // Area 2(1·0.5 + 1·0.3 + 0.5·0.3) = 1.9 m² at 0.05 m spacing ≈ 760.
        assert!((700..900).contains(&pts.len()), "{} points", pts.len());
        for p in &pts {
            let (s, _, _) = b.closest_surface_point(*p);
            assert!((s - *p).length() < 1e-4);
        }
        for (i, a) in pts.iter().enumerate() {
            assert!(pts[i + 1..].iter().all(|b| (*a - *b).length() > 1e-4));
        }
    }

    #[test]
    fn sampling_is_clipped_to_the_region() {
        let floor =
            RigidBody::new_static(Shape::cuboid(20.0, 0.5, 20.0), Vec3::new(0.0, -0.25, 0.0));
        let mut pts = Vec::new();
        floor.sample_surface(
            0.05,
            0.0,
            (Vec3::new(-0.5, -0.1, -0.5), Vec3::new(0.5, 1.0, 0.5)),
            &mut pts,
        );
        // Just the 1 m × 1 m patch of the top face: about 21 × 21 points.
        assert!((400..600).contains(&pts.len()), "{} points", pts.len());
    }

    #[test]
    fn sphere_samples_cover_it_evenly() {
        let s = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::ZERO);
        let mut pts = Vec::new();
        s.sample_surface(0.05, 0.0, EVERYWHERE, &mut pts);
        // 4π·0.25 / 0.0025 = 1257
        assert_eq!(pts.len(), 1257);
        assert!(pts.iter().all(|p| (p.length() - 0.5).abs() < 1e-5));
        assert!(pts.iter().map(|p| p.y).sum::<f32>().abs() < 0.5);
    }

    #[test]
    fn closest_point_inside_box_exits_nearest_face() {
        let b = RigidBody::new(Shape::cube(1.0), Vec3::ZERO);
        let (s, n, inside) = b.closest_surface_point(Vec3::new(0.1, 0.45, 0.0));
        assert!(inside);
        assert_eq!(n, Vec3::Y);
        assert!((s - Vec3::new(0.1, 0.5, 0.0)).length() < 1e-6);
    }
}
