//! Broad phase: quickly find the pairs of bodies that *might* be touching,
//! so the expensive narrow phase only runs on those.
//!
//! Testing every pair is O(n²). Sort-and-sweep sorts bodies by the left edge
//! of their bounding box, then sweeps left to right keeping a list of boxes
//! that are still "open". Only boxes that overlap along x are ever compared.
//! That is close to O(n log n) for typical scenes.

use crate::body::RigidBody;
use crate::shape::Shape;
use glam::Vec2;

/// Axis-aligned bounding box in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb {
    pub fn of(body: &RigidBody) -> Self {
        match body.shape {
            Shape::Circle { radius } => Self {
                min: body.position - Vec2::splat(radius),
                max: body.position + Vec2::splat(radius),
            },
            Shape::Box { .. } => {
                let corners = body.box_corners().expect("box has corners");
                let (min, max) = corners
                    .iter()
                    .fold((corners[0], corners[0]), |(lo, hi), &c| {
                        (lo.min(c), hi.max(c))
                    });
                Self { min, max }
            }
        }
    }

    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
    }
}

/// Two static bodies can never need a collision response.
fn can_collide(a: &RigidBody, b: &RigidBody) -> bool {
    !(a.is_static && b.is_static)
}

/// Reference implementation: test every pair. Pairs are `(i, j)` with
/// `i < j`, sorted.
pub fn brute_force_pairs(bodies: &[RigidBody]) -> Vec<(usize, usize)> {
    let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
    let mut pairs = Vec::new();
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            if can_collide(&bodies[i], &bodies[j]) && boxes[i].overlaps(&boxes[j]) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Sort-and-sweep along x. Returns the same set as [`brute_force_pairs`],
/// as `(i, j)` with `i < j`, sorted.
pub fn sweep_and_prune_pairs(bodies: &[RigidBody]) -> Vec<(usize, usize)> {
    let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
    let mut order: Vec<usize> = (0..bodies.len()).collect();
    order.sort_by(|&i, &j| boxes[i].min.x.total_cmp(&boxes[j].min.x));

    let mut active: Vec<usize> = Vec::new();
    let mut pairs = Vec::new();
    for &i in &order {
        let left = boxes[i].min.x;
        // Anything whose right edge is left of this box's left edge can't
        // overlap this box or any box after it in the sweep.
        active.retain(|&j| boxes[j].max.x >= left);
        for &j in &active {
            if can_collide(&bodies[i], &bodies[j]) && boxes[i].overlaps(&boxes[j]) {
                pairs.push((i.min(j), i.max(j)));
            }
        }
        active.push(i);
    }
    // Deterministic order makes the solver deterministic too.
    pairs.sort_unstable();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Tiny deterministic random number generator so tests don't need a
    /// `rand` dependency. (xorshift32)
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0 as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next()
        }
    }

    #[test]
    fn rotated_box_aabb() {
        let b = RigidBody::new(Shape::rect(2.0, 2.0), Vec2::ZERO).with_angle(TAU / 8.0);
        let bb = Aabb::of(&b);
        let r = 2f32.sqrt();
        assert!((bb.max - Vec2::splat(r)).length() < 1e-5);
        assert!((bb.min + Vec2::splat(r)).length() < 1e-5);
    }

    #[test]
    fn sweep_matches_brute_force_on_random_scenes() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..50 {
            let n = 1 + (rng.next() * 80.0) as usize;
            let bodies: Vec<RigidBody> = (0..n)
                .map(|_| {
                    let pos = Vec2::new(rng.range(-10.0, 10.0), rng.range(-10.0, 10.0));
                    let mut b = if rng.next() < 0.5 {
                        RigidBody::new(
                            Shape::Circle {
                                radius: rng.range(0.1, 1.5),
                            },
                            pos,
                        )
                    } else {
                        RigidBody::new(Shape::rect(rng.range(0.1, 3.0), rng.range(0.1, 3.0)), pos)
                            .with_angle(rng.range(0.0, TAU))
                    };
                    b.is_static = rng.next() < 0.2;
                    b
                })
                .collect();
            assert_eq!(sweep_and_prune_pairs(&bodies), brute_force_pairs(&bodies));
        }
    }

    #[test]
    fn static_pairs_are_skipped() {
        let bodies = vec![
            RigidBody::new_static(Shape::rect(1.0, 1.0), Vec2::ZERO),
            RigidBody::new_static(Shape::rect(1.0, 1.0), Vec2::new(0.5, 0.0)),
        ];
        assert!(sweep_and_prune_pairs(&bodies).is_empty());
    }
}
