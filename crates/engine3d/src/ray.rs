//! Ray casts against bodies, for picking with the mouse.

use crate::body::RigidBody;
use crate::shape::Shape;
use glam::Vec3;

/// Distance along the ray `origin + t·dir` (unit `dir`, t ≥ 0) to the first
/// hit on `body`, or `None` if it misses. A ray starting inside reports 0.
pub fn intersect(body: &RigidBody, origin: Vec3, dir: Vec3) -> Option<f32> {
    // Work in the body's frame: the box becomes axis-aligned.
    let o = body.world_to_local(origin);
    let d = body.orientation.inverse() * dir;
    match body.shape {
        Shape::Sphere { radius } => {
            // |o + t d|² = r²  →  t² + 2(o·d)t + |o|² − r² = 0
            let b = o.dot(d);
            let c = o.length_squared() - radius * radius;
            if c <= 0.0 {
                return Some(0.0);
            }
            let disc = b * b - c;
            if disc < 0.0 {
                return None;
            }
            let t = -b - disc.sqrt();
            (t >= 0.0).then_some(t)
        }
        Shape::Cuboid { half_extents: h } => {
            // Slab method: intersect the three pairs of parallel planes.
            let (mut t_min, mut t_max) = (0.0_f32, f32::INFINITY);
            for k in 0..3 {
                if d[k].abs() < 1e-12 {
                    if o[k].abs() > h[k] {
                        return None;
                    }
                } else {
                    let (t1, t2) = ((-h[k] - o[k]) / d[k], (h[k] - o[k]) / d[k]);
                    t_min = t_min.max(t1.min(t2));
                    t_max = t_max.min(t1.max(t2));
                    if t_min > t_max {
                        return None;
                    }
                }
            }
            Some(t_min)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn rotated_box() {
        // A 2 × 2 × 2 cube rotated 45° about y: its corner points along +x
        // at distance √2.
        let b = RigidBody::new(Shape::cube(2.0), Vec3::ZERO)
            .with_orientation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4));
        let t = intersect(&b, Vec3::new(5.0, 0.0, 0.0), -Vec3::X).unwrap();
        assert!((t - (5.0 - 2f32.sqrt())).abs() < 1e-4, "t = {t}");
        assert!(intersect(&b, Vec3::new(5.0, 2.0, 0.0), -Vec3::X).is_none());
    }

    #[test]
    fn sphere_from_outside_and_inside() {
        let s = RigidBody::new(Shape::Sphere { radius: 1.0 }, Vec3::new(0.0, 0.0, -3.0));
        assert!((intersect(&s, Vec3::ZERO, -Vec3::Z).unwrap() - 2.0).abs() < 1e-5);
        assert_eq!(intersect(&s, Vec3::new(0.0, 0.0, -3.0), Vec3::X), Some(0.0));
        assert!(intersect(&s, Vec3::ZERO, Vec3::Z).is_none());
    }
}
