//! Collision shapes and their geometric properties.

use glam::Vec2;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// The geometry of a rigid body, described in the body's local frame
/// (centered on the body's center of mass, unrotated).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape {
    /// A circle of the given radius, in meters.
    Circle { radius: f32 },
    /// A rectangle described by half its width and half its height, in meters.
    Box { half_extents: Vec2 },
}

impl Shape {
    /// Convenience constructor for a box from its full width and height.
    pub fn rect(width: f32, height: f32) -> Self {
        Shape::Box {
            half_extents: Vec2::new(width, height) * 0.5,
        }
    }

    /// Area in m².
    pub fn area(&self) -> f32 {
        match *self {
            Shape::Circle { radius } => PI * radius * radius,
            Shape::Box { half_extents } => 4.0 * half_extents.x * half_extents.y,
        }
    }

    /// Moment of inertia about the center of mass for a body of `mass` kg,
    /// in kg·m².
    ///
    /// - Solid disk: `m r² / 2`
    /// - Solid rectangle of width `w`, height `h`: `m (w² + h²) / 12`,
    ///   which with half extents `(a, b)` is `m (a² + b²) / 3`.
    pub fn moment_of_inertia(&self, mass: f32) -> f32 {
        match *self {
            Shape::Circle { radius } => 0.5 * mass * radius * radius,
            Shape::Box { half_extents } => mass * half_extents.length_squared() / 3.0,
        }
    }

    /// Radius of the smallest circle centered on the body that contains the
    /// shape. Useful for cheap broad-phase rejection.
    pub fn bounding_radius(&self) -> f32 {
        match *self {
            Shape::Circle { radius } => radius,
            Shape::Box { half_extents } => half_extents.length(),
        }
    }

    /// Whether a point given in the shape's *local* frame lies inside it.
    pub fn contains_local_point(&self, p: Vec2) -> bool {
        match *self {
            Shape::Circle { radius } => p.length_squared() <= radius * radius,
            Shape::Box { half_extents } => {
                p.x.abs() <= half_extents.x && p.y.abs() <= half_extents.y
            }
        }
    }

    /// Returns `Err` with a description if the shape has non-positive or
    /// non-finite dimensions.
    pub fn validate(&self) -> Result<(), String> {
        match *self {
            Shape::Circle { radius } if !(radius.is_finite() && radius > 0.0) => {
                Err(format!("circle radius must be positive, got {radius}"))
            }
            Shape::Box { half_extents }
                if !(half_extents.is_finite() && half_extents.x > 0.0 && half_extents.y > 0.0) =>
            {
                Err(format!(
                    "box half extents must be positive, got {half_extents}"
                ))
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_area_and_inertia() {
        let s = Shape::Circle { radius: 2.0 };
        assert!((s.area() - 4.0 * PI).abs() < 1e-5);
        // m r^2 / 2 with m = 3, r = 2 -> 6
        assert!((s.moment_of_inertia(3.0) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn box_area_and_inertia_match_full_size_formula() {
        let (w, h, m) = (2.0_f32, 4.0_f32, 6.0_f32);
        let s = Shape::rect(w, h);
        assert!((s.area() - w * h).abs() < 1e-6);
        let expected = m * (w * w + h * h) / 12.0; // = 10
        assert!((s.moment_of_inertia(m) - expected).abs() < 1e-5);
    }

    #[test]
    fn validation_rejects_bad_dimensions() {
        assert!(Shape::Circle { radius: 0.0 }.validate().is_err());
        assert!(Shape::Circle { radius: f32::NAN }.validate().is_err());
        assert!(Shape::rect(1.0, -1.0).validate().is_err());
        assert!(Shape::rect(1.0, 1.0).validate().is_ok());
    }
}
