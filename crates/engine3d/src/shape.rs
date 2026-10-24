//! 3D collision shapes and their mass properties.

use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// The geometry of a body, in its local frame (centered on the center of
/// mass, unrotated).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape {
    /// A ball of the given radius, m.
    Sphere { radius: f32 },
    /// A box described by half its size along each local axis, m.
    Cuboid { half_extents: Vec3 },
}

impl Shape {
    /// A box from its full width (x), height (y), and depth (z).
    pub fn cuboid(width: f32, height: f32, depth: f32) -> Self {
        Shape::Cuboid {
            half_extents: Vec3::new(width, height, depth) * 0.5,
        }
    }

    /// A cube with sides of length `side`.
    pub fn cube(side: f32) -> Self {
        Self::cuboid(side, side, side)
    }

    /// Volume, m³.
    pub fn volume(&self) -> f32 {
        match *self {
            Shape::Sphere { radius } => 4.0 / 3.0 * PI * radius.powi(3),
            Shape::Cuboid { half_extents: h } => 8.0 * h.x * h.y * h.z,
        }
    }

    /// Principal moments of inertia about the center of mass, in the local
    /// frame, for a body of `mass` kg, kg·m². Both shapes are symmetric
    /// enough that the inertia tensor is diagonal in their own frame.
    ///
    /// - Solid sphere: 2/5 m r² about every axis.
    /// - Solid box of full size (w, h, d): Iₓ = m (h² + d²) / 12, and so on;
    ///   with half extents (a, b, c) that's m (b² + c²) / 3.
    pub fn principal_inertia(&self, mass: f32) -> Vec3 {
        match *self {
            Shape::Sphere { radius } => Vec3::splat(0.4 * mass * radius * radius),
            Shape::Cuboid { half_extents: h } => {
                let sq = h * h;
                Vec3::new(sq.y + sq.z, sq.x + sq.z, sq.x + sq.y) * (mass / 3.0)
            }
        }
    }

    /// Radius of the smallest sphere around the center that contains the
    /// shape.
    pub fn bounding_radius(&self) -> f32 {
        match *self {
            Shape::Sphere { radius } => radius,
            Shape::Cuboid { half_extents } => half_extents.length(),
        }
    }

    /// Whether a point in the shape's local frame lies inside it.
    pub fn contains_local_point(&self, p: Vec3) -> bool {
        match *self {
            Shape::Sphere { radius } => p.length_squared() <= radius * radius,
            Shape::Cuboid { half_extents } => p.abs().cmple(half_extents).all(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Shape::Sphere { radius } if !(radius.is_finite() && radius > 0.0) => {
                Err(format!("sphere radius must be positive, got {radius}"))
            }
            Shape::Cuboid { half_extents: h } if !(h.is_finite() && h.cmpgt(Vec3::ZERO).all()) => {
                Err(format!("box half extents must be positive, got {h}"))
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_mass_properties() {
        let s = Shape::Sphere { radius: 0.5 };
        assert!((s.volume() - 4.0 / 3.0 * PI * 0.125).abs() < 1e-6);
        assert!((s.principal_inertia(2.0) - Vec3::splat(0.2)).length() < 1e-6);
    }

    #[test]
    fn box_inertia_matches_full_size_formula() {
        let (w, h, d, m) = (1.0_f32, 2.0_f32, 3.0_f32, 6.0_f32);
        let i = Shape::cuboid(w, h, d).principal_inertia(m);
        let expected = Vec3::new(h * h + d * d, w * w + d * d, w * w + h * h) * (m / 12.0);
        assert!((i - expected).length() < 1e-5, "{i} vs {expected}");
        assert!((Shape::cuboid(w, h, d).volume() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn validation() {
        assert!(Shape::Sphere { radius: -1.0 }.validate().is_err());
        assert!(Shape::cuboid(1.0, 0.0, 1.0).validate().is_err());
        assert!(Shape::cube(1.0).validate().is_ok());
    }
}
