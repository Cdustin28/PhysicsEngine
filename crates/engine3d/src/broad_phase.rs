//! Broad phase: sort-and-sweep on axis-aligned bounding boxes, as in 2D.

use crate::body::RigidBody;
use crate::shape::Shape;
use glam::{Mat3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn of(body: &RigidBody) -> Self {
        let extent = match body.shape {
            Shape::Sphere { radius } => Vec3::splat(radius),
            // A rotated box's half-size along each world axis is
            // |R| · h (absolute values of the rotation matrix entries).
            Shape::Cuboid { half_extents } => {
                let r = Mat3::from_quat(body.orientation);
                let abs = Mat3::from_cols(r.x_axis.abs(), r.y_axis.abs(), r.z_axis.abs());
                abs * half_extents
            }
        };
        Self {
            min: body.position - extent,
            max: body.position + extent,
        }
    }

    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

/// Every pair `(i, j)`, `i < j`, sorted, whose boxes overlap and that aren't
/// both static.
pub fn sweep_and_prune_pairs(bodies: &[RigidBody]) -> Vec<(usize, usize)> {
    let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
    let mut order: Vec<usize> = (0..bodies.len()).collect();
    order.sort_by(|&i, &j| boxes[i].min.x.total_cmp(&boxes[j].min.x));
    let mut active: Vec<usize> = Vec::new();
    let mut pairs = Vec::new();
    for &i in &order {
        active.retain(|&j| boxes[j].max.x >= boxes[i].min.x);
        for &j in &active {
            if !(bodies[i].is_static && bodies[j].is_static) && boxes[i].overlaps(&boxes[j]) {
                pairs.push((i.min(j), i.max(j)));
            }
        }
        active.push(i);
    }
    pairs.sort_unstable();
    pairs
}

/// Reference implementation for testing.
pub fn brute_force_pairs(bodies: &[RigidBody]) -> Vec<(usize, usize)> {
    let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
    let mut pairs = Vec::new();
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            if !(bodies[i].is_static && bodies[j].is_static) && boxes[i].overlaps(&boxes[j]) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0 as f32 / u32::MAX as f32
        }
    }

    #[test]
    fn sweep_matches_brute_force() {
        let mut rng = Rng(0xC0FF_EE11);
        for _ in 0..30 {
            let bodies: Vec<RigidBody> = (0..60)
                .map(|_| {
                    let p = Vec3::new(rng.next(), rng.next(), rng.next()) * 10.0 - 5.0;
                    let mut b = if rng.next() < 0.5 {
                        RigidBody::new(
                            Shape::Sphere {
                                radius: 0.2 + rng.next(),
                            },
                            p,
                        )
                    } else {
                        RigidBody::new(
                            Shape::cuboid(
                                0.2 + rng.next() * 2.0,
                                0.2 + rng.next() * 2.0,
                                0.2 + rng.next() * 2.0,
                            ),
                            p,
                        )
                        .with_orientation(Quat::from_euler(
                            glam::EulerRot::XYZ,
                            rng.next() * 6.0,
                            rng.next() * 6.0,
                            rng.next() * 6.0,
                        ))
                    };
                    b.is_static = rng.next() < 0.2;
                    b
                })
                .collect();
            assert_eq!(sweep_and_prune_pairs(&bodies), brute_force_pairs(&bodies));
        }
    }
}
