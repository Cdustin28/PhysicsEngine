//! Joints in 3D: the same three kinds and the same solver as the 2D engine
//! (see the `engine` crate's `joint` module for the full explanation).
//!
//! - **Pin**: a ball-and-socket joint. The anchors coincide; the bodies turn
//!   freely about it in every direction.
//! - **Distance**: anchors held between `min_length` and `max_length` apart
//!   (a rod when equal, a rope segment when `min_length = 0`).
//! - **Spring**: k·stretch + c·stretch rate along the anchor line (N/m,
//!   N·s/m), integrated implicitly as a soft constraint.
//!
//! Each sweep aims the joint at its predicted end-of-step state (the change
//! over the step only; existing error goes to the split-impulse position
//! pass). The prediction is linear in the velocities plus a fixed curvature
//! term, the part of each lever arm's swing that a straight line misses,
//! here rotated with a quaternion. That keeps pendulums from losing energy
//! without destabilizing fast spinners.

use crate::body::RigidBody;
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JointKind {
    Pin,
    Distance {
        /// m
        min_length: f32,
        /// m
        max_length: f32,
    },
    Spring {
        /// m
        rest_length: f32,
        /// N/m
        stiffness: f32,
        /// N·s/m
        damping: f32,
    },
}

/// A joint between body `a` and body `b` (or a fixed world point when `b`
/// is `None`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    #[serde(flatten)]
    pub kind: JointKind,
    pub a: usize,
    pub b: Option<usize>,
    /// Anchor in body A's frame, m.
    pub local_anchor_a: Vec3,
    /// Anchor in body B's frame, or a world point if `b` is `None`.
    pub anchor_b: Vec3,
    /// Warm-start cache: pin, the 3D impulse; distance, (lower, upper, 0);
    /// spring, (λ, 0, 0).
    #[serde(skip)]
    pub(crate) impulse: Vec3,
}

impl PartialEq for Joint {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.a == other.a
            && self.b == other.b
            && self.local_anchor_a == other.local_anchor_a
            && self.anchor_b == other.anchor_b
    }
}

impl Joint {
    /// A joint with anchors at the world points `pa` (on `a`) and `pb`.
    pub fn new(
        kind: JointKind,
        bodies: &[RigidBody],
        a: usize,
        pa: Vec3,
        b: Option<usize>,
        pb: Vec3,
    ) -> Self {
        Self {
            kind,
            a,
            b,
            local_anchor_a: bodies[a].world_to_local(pa),
            anchor_b: match b {
                Some(b) => bodies[b].world_to_local(pb),
                None => pb,
            },
            impulse: Vec3::ZERO,
        }
    }

    pub fn pin(bodies: &[RigidBody], a: usize, b: Option<usize>, p: Vec3) -> Self {
        Self::new(JointKind::Pin, bodies, a, p, b, p)
    }

    pub fn rod(bodies: &[RigidBody], a: usize, pa: Vec3, b: Option<usize>, pb: Vec3) -> Self {
        let l = (pb - pa).length();
        let kind = JointKind::Distance {
            min_length: l,
            max_length: l,
        };
        Self::new(kind, bodies, a, pa, b, pb)
    }

    pub fn rope(bodies: &[RigidBody], a: usize, pa: Vec3, b: Option<usize>, pb: Vec3) -> Self {
        let kind = JointKind::Distance {
            min_length: 0.0,
            max_length: (pb - pa).length(),
        };
        Self::new(kind, bodies, a, pa, b, pb)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spring(
        bodies: &[RigidBody],
        a: usize,
        pa: Vec3,
        b: Option<usize>,
        pb: Vec3,
        stiffness: f32,
        damping: f32,
    ) -> Self {
        let kind = JointKind::Spring {
            rest_length: (pb - pa).length(),
            stiffness,
            damping,
        };
        Self::new(kind, bodies, a, pa, b, pb)
    }

    pub fn anchors(&self, bodies: &[RigidBody]) -> (Vec3, Vec3) {
        let pa = bodies[self.a].local_to_world(self.local_anchor_a);
        let pb = match self.b {
            Some(b) => bodies[b].local_to_world(self.anchor_b),
            None => self.anchor_b,
        };
        (pa, pb)
    }

    pub fn involves(&self, i: usize) -> bool {
        self.a == i || self.b == Some(i)
    }

    pub(crate) fn reindex_after_removal(&mut self, removed: usize) -> bool {
        if self.involves(removed) {
            return false;
        }
        if self.a > removed {
            self.a -= 1;
        }
        if let Some(b) = self.b.as_mut()
            && *b > removed
        {
            *b -= 1;
        }
        true
    }

    pub fn validate(&self, body_count: usize) -> Result<(), String> {
        if self.a >= body_count || self.b.is_some_and(|b| b >= body_count) {
            return Err("joint refers to a body that doesn't exist".into());
        }
        if self.b == Some(self.a) {
            return Err("joint connects a body to itself".into());
        }
        if !(self.local_anchor_a.is_finite() && self.anchor_b.is_finite()) {
            return Err("joint anchors must be finite".into());
        }
        match self.kind {
            JointKind::Pin => Ok(()),
            JointKind::Distance {
                min_length,
                max_length,
            } => {
                if 0.0 <= min_length && min_length <= max_length && max_length.is_finite() {
                    Ok(())
                } else {
                    Err(format!(
                        "distance joint needs 0 ≤ min ≤ max, got {min_length}..{max_length}"
                    ))
                }
            }
            JointKind::Spring {
                rest_length,
                stiffness,
                damping,
            } => {
                if rest_length >= 0.0
                    && stiffness >= 0.0
                    && damping >= 0.0
                    && (rest_length + stiffness + damping).is_finite()
                {
                    Ok(())
                } else {
                    Err("spring needs non-negative length, stiffness, damping".into())
                }
            }
        }
    }
}

/// What one end of a chain built by [`add_chain`] is attached to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anchor {
    Free,
    World,
    Body(usize),
}

/// Add a chain of `links` boxes (square cross-section `thickness`) from
/// `start` to `end`, each pinned to the next. Returns the new bodies'
/// indices.
pub fn add_chain(
    world: &mut crate::World,
    start: Vec3,
    end: Vec3,
    links: usize,
    thickness: f32,
    start_anchor: Anchor,
    end_anchor: Anchor,
) -> Vec<usize> {
    let links = links.max(1);
    let step = (end - start) / links as f32;
    let orientation = Quat::from_rotation_arc(Vec3::X, step.normalize_or(Vec3::X));
    let first = world.bodies.len();
    for k in 0..links {
        let center = start + step * (k as f32 + 0.5);
        let link = RigidBody::new(
            crate::Shape::cuboid(step.length(), thickness, thickness),
            center,
        )
        .with_orientation(orientation);
        world.add_body(link);
    }
    let ids: Vec<usize> = (first..first + links).collect();
    for k in 1..links {
        let j = Joint::pin(
            &world.bodies,
            ids[k - 1],
            Some(ids[k]),
            start + step * k as f32,
        );
        world.joints.push(j);
    }
    for (anchor, link, p) in [
        (start_anchor, ids[0], start),
        (end_anchor, ids[links - 1], end),
    ] {
        let other = match anchor {
            Anchor::Free => continue,
            Anchor::World => None,
            Anchor::Body(b) => Some(b),
        };
        let j = Joint::pin(&world.bodies, link, other, p);
        world.joints.push(j);
    }
    ids
}

/// Velocity state shared with the contact solver; index `world` is the
/// immovable world.
pub(crate) struct JointVelocities<'a> {
    pub v: &'a mut [Vec3],
    pub w: &'a mut [Vec3],
    pub inv_mass: &'a [f32],
    pub inv_inertia: &'a [Mat3],
}

impl JointVelocities<'_> {
    fn at(&self, i: usize, r: Vec3) -> Vec3 {
        self.v[i] + self.w[i].cross(r)
    }

    fn apply(&mut self, a: usize, ra: Vec3, b: usize, rb: Vec3, j: Vec3) {
        self.v[a] -= self.inv_mass[a] * j;
        self.w[a] -= self.inv_inertia[a] * ra.cross(j);
        self.v[b] += self.inv_mass[b] * j;
        self.w[b] += self.inv_inertia[b] * rb.cross(j);
    }

    fn k_along(&self, a: usize, ra: Vec3, b: usize, rb: Vec3, d: Vec3) -> f32 {
        self.inv_mass[a]
            + self.inv_mass[b]
            + (self.inv_inertia[a] * ra.cross(d)).cross(ra).dot(d)
            + (self.inv_inertia[b] * rb.cross(d)).cross(rb).dot(d)
    }
}

/// The cross-product matrix: skew(r) · x = r × x.
fn skew(r: Vec3) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(0.0, r.z, -r.y),
        Vec3::new(-r.z, 0.0, r.x),
        Vec3::new(r.y, -r.x, 0.0),
    )
}

pub(crate) struct PreparedJoint {
    a: usize,
    b: usize,
    ra: Vec3,
    rb: Vec3,
    gap: Vec3,
    curvature: Vec3,
    rows: Rows,
}

enum Rows {
    Pin {
        inv_k: Mat3,
        error: Vec3,
    },
    Distance {
        n: Vec3,
        mass: f32,
        length: f32,
        min: f32,
        max: f32,
    },
    Spring {
        n: Vec3,
        mass: f32,
        bias: f32,
        gamma: f32,
    },
    None,
}

const MIN_LENGTH: f32 = 1e-6;

impl PreparedJoint {
    fn predicted_motion(&self, vel: &JointVelocities, dt: f32) -> Vec3 {
        (vel.at(self.b, self.rb) - vel.at(self.a, self.ra)) * dt + self.curvature
    }
}

pub(crate) fn prepare(
    joint: &Joint,
    bodies: &[RigidBody],
    vel: &JointVelocities,
    world: usize,
    dt: f32,
) -> PreparedJoint {
    let (pa, pb) = joint.anchors(bodies);
    let a = joint.a;
    let b = joint.b.unwrap_or(world);
    let ra = pa - bodies[a].position;
    let rb = match joint.b {
        Some(i) => pb - bodies[i].position,
        None => Vec3::ZERO,
    };
    let rows = match joint.kind {
        JointKind::Pin => {
            // K = (ma + mb) I − [ra]× Ia⁻¹ [ra]× − [rb]× Ib⁻¹ [rb]×
            let (sa, sb) = (skew(ra), skew(rb));
            let k = Mat3::from_diagonal(Vec3::splat(vel.inv_mass[a] + vel.inv_mass[b]))
                - sa * vel.inv_inertia[a] * sa
                - sb * vel.inv_inertia[b] * sb;
            if k.determinant().abs() < 1e-12 {
                Rows::None
            } else {
                Rows::Pin {
                    inv_k: k.inverse(),
                    error: pb - pa,
                }
            }
        }
        JointKind::Distance {
            min_length,
            max_length,
        } => {
            let d = pb - pa;
            let length = d.length();
            let k = if length > MIN_LENGTH {
                vel.k_along(a, ra, b, rb, d / length)
            } else {
                0.0
            };
            if k <= 0.0 {
                Rows::None
            } else {
                Rows::Distance {
                    n: d / length,
                    mass: 1.0 / k,
                    length,
                    min: min_length,
                    max: max_length,
                }
            }
        }
        JointKind::Spring {
            rest_length,
            stiffness,
            damping,
        } => {
            let d = pb - pa;
            let length = d.length();
            let k = if length > MIN_LENGTH {
                vel.k_along(a, ra, b, rb, d / length)
            } else {
                0.0
            };
            let soft = dt * (damping + dt * stiffness);
            if k <= 0.0 || soft <= 0.0 {
                Rows::None
            } else {
                let gamma = 1.0 / soft;
                Rows::Spring {
                    n: d / length,
                    mass: 1.0 / (k + gamma),
                    bias: (length - rest_length) * dt * stiffness * gamma,
                    gamma,
                }
            }
        }
    };
    let curve = |r: Vec3, w: Vec3| Quat::from_scaled_axis(w * dt) * r - r - w.cross(r) * dt;
    PreparedJoint {
        a,
        b,
        ra,
        rb,
        gap: pb - pa,
        curvature: curve(rb, vel.w[b]) - curve(ra, vel.w[a]),
        rows,
    }
}

pub(crate) fn warm_start(
    p: &PreparedJoint,
    joint: &mut Joint,
    vel: &mut JointVelocities,
    on: bool,
) {
    if !on {
        joint.impulse = Vec3::ZERO;
        return;
    }
    let impulse = match p.rows {
        Rows::Pin { .. } => joint.impulse,
        Rows::Distance { n, .. } => (joint.impulse.x + joint.impulse.y) * n,
        Rows::Spring { n, .. } => joint.impulse.x * n,
        Rows::None => {
            joint.impulse = Vec3::ZERO;
            Vec3::ZERO
        }
    };
    vel.apply(p.a, p.ra, p.b, p.rb, impulse);
}

pub(crate) fn solve_velocity(
    p: &PreparedJoint,
    joint: &mut Joint,
    vel: &mut JointVelocities,
    dt: f32,
) {
    let (a, b, ra, rb) = (p.a, p.b, p.ra, p.rb);
    match p.rows {
        Rows::Pin { inv_k, .. } => {
            let lambda = -(inv_k * (p.predicted_motion(vel, dt) / dt));
            joint.impulse += lambda;
            vel.apply(a, ra, b, rb, lambda);
        }
        Rows::Distance {
            n, mass, min, max, ..
        } => {
            let start = p.gap.normalize_or_zero() * p.gap.length().clamp(min, max);
            let predicted = |vel: &JointVelocities| (start + p.predicted_motion(vel, dt)).length();
            if min == max {
                let lambda = -mass * (predicted(vel) - max) / dt;
                joint.impulse.x += lambda;
                vel.apply(a, ra, b, rb, lambda * n);
                return;
            }
            let old = joint.impulse.x;
            joint.impulse.x = (old - mass * (predicted(vel) - min) / dt).max(0.0);
            vel.apply(a, ra, b, rb, (joint.impulse.x - old) * n);
            let old = joint.impulse.y;
            joint.impulse.y = (old - mass * (predicted(vel) - max) / dt).min(0.0);
            vel.apply(a, ra, b, rb, (joint.impulse.y - old) * n);
        }
        Rows::Spring {
            n,
            mass,
            bias,
            gamma,
        } => {
            let cdot = (vel.at(b, rb) - vel.at(a, ra)).dot(n);
            let lambda = -mass * (cdot + bias + gamma * joint.impulse.x);
            joint.impulse.x += lambda;
            vel.apply(a, ra, b, rb, lambda * n);
        }
        Rows::None => {}
    }
}

pub(crate) fn solve_position(
    p: &PreparedJoint,
    total: &mut Vec3,
    pseudo: &mut JointVelocities,
    fraction: f32,
    dt: f32,
) {
    let (a, b, ra, rb) = (p.a, p.b, p.ra, p.rb);
    match p.rows {
        Rows::Pin { inv_k, error } => {
            let target = -error * (fraction / dt);
            let lambda = -(inv_k * (pseudo.at(b, rb) - pseudo.at(a, ra) - target));
            *total += lambda;
            pseudo.apply(a, ra, b, rb, lambda);
        }
        Rows::Distance {
            n,
            mass,
            length,
            min,
            max,
        } => {
            let error = if length > max {
                length - max
            } else if length < min {
                length - min
            } else {
                return;
            };
            let target = -error * (fraction / dt);
            let rate = (pseudo.at(b, rb) - pseudo.at(a, ra)).dot(n);
            let lambda = -mass * (rate - target);
            total.x += lambda;
            pseudo.apply(a, ra, b, rb, lambda * n);
        }
        Rows::Spring { .. } | Rows::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skew_is_the_cross_product() {
        let (r, x) = (Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.3, 0.7, -1.1));
        assert!((skew(r) * x - r.cross(x)).length() < 1e-6);
    }
}
