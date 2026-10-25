//! Joints: constraints that tie two bodies together (or a body to a fixed
//! point in the world).
//!
//! - **Pin**: an anchor point on each body is held at the same place. The
//!   bodies can turn freely about it: a hinge, a chain link, a pendulum
//!   pivot.
//! - **Distance**: the anchors are held between `min_length` and
//!   `max_length` apart. Equal lengths make a rigid rod; `min_length = 0`
//!   makes a rope segment that can go slack but not stretch.
//! - **Spring**: a force k·(stretch) + c·(stretch rate) along the line
//!   between the anchors, in physical units (N/m and N·s/m).
//!
//! # How they're solved
//!
//! Like contacts (see [`solver`](crate::solver)), joints are solved by
//! sequential impulses, in the same sweeps as the contacts, with warm
//! starting. Each joint has a constraint function C (pin: the gap between
//! the anchors; distance: length minus target) that should stay zero.
//!
//! The textbook approach makes the *rate* Ċ = J·v zero. That quietly drains
//! energy: the velocity is made tangent to the circle the body is on at the
//! start of the step, the body then moves along that tangent, and at its new
//! position part of its velocity points outward, which the next step
//! removes. Each step loses a fraction (v·dt/L)² of the kinetic energy; a
//! pendulum swinging 60° lost 23% of its energy in 10 s at 120 Hz.
//!
//! Instead, each sweep aims at the constraint's value at the *end* of the
//! step, C(x + v·dt), predicted from the current velocities including the
//! rotation of each anchor's lever arm, and applies the impulse that brings
//! it to zero: λ = −C_pred / (k·dt), with k = J M⁻¹ Jᵀ the effective mass.
//! Sweeping that to convergence is the SHAKE scheme from molecular dynamics
//! (Ryckaert et al. 1977), which has no long-term energy drift: the same
//! pendulum stays within 1% of its starting energy indefinitely. A
//! split-impulse pass (as for contact overlap) cleans up whatever error is
//! left, without touching velocities.
//!
//! Springs are *soft* constraints (Catto, "Soft Constraints", GDC 2011):
//! the same machinery with a compliance term, which integrates the spring
//! implicitly. That stays stable however stiff the spring is, where an
//! explicit spring force would blow up.

use crate::body::RigidBody;
use glam::Vec2;
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
        /// Length at which the spring pushes and pulls nothing, m.
        rest_length: f32,
        /// N/m
        stiffness: f32,
        /// N·s/m
        damping: f32,
    },
}

/// A joint between body `a` and body `b` (or, with `b = None`, a fixed point
/// in the world).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    #[serde(flatten)]
    pub kind: JointKind,
    /// Index of the first body in `World::bodies`.
    pub a: usize,
    /// Index of the second body, or `None` to anchor to the world.
    pub b: Option<usize>,
    /// Anchor in body A's local frame, m.
    pub local_anchor_a: Vec2,
    /// Anchor in body B's local frame, or a world point if `b` is `None`.
    pub anchor_b: Vec2,
    /// Accumulated impulse for warm starting (pin: x and y; distance:
    /// lower and upper limit; spring: x).
    #[serde(skip)]
    pub(crate) impulse: Vec2,
}

impl PartialEq for Joint {
    /// Same joint; the solver's warm-start cache doesn't count.
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.a == other.a
            && self.b == other.b
            && self.local_anchor_a == other.local_anchor_a
            && self.anchor_b == other.anchor_b
    }
}

impl Joint {
    /// A joint whose anchors are the world points `pa` (on body `a`) and `pb`
    /// (on body `b`, or fixed in the world). Distance and spring lengths
    /// default to the current distance between the anchors, so a new joint
    /// starts relaxed.
    pub fn new(
        kind: JointKind,
        bodies: &[RigidBody],
        a: usize,
        pa: Vec2,
        b: Option<usize>,
        pb: Vec2,
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
            impulse: Vec2::ZERO,
        }
    }

    /// A pin joint at world point `p`.
    pub fn pin(bodies: &[RigidBody], a: usize, b: Option<usize>, p: Vec2) -> Self {
        Self::new(JointKind::Pin, bodies, a, p, b, p)
    }

    /// A rigid rod between the world points `pa` and `pb`, at its current
    /// length.
    pub fn rod(bodies: &[RigidBody], a: usize, pa: Vec2, b: Option<usize>, pb: Vec2) -> Self {
        let l = (pb - pa).length();
        Self::new(
            JointKind::Distance {
                min_length: l,
                max_length: l,
            },
            bodies,
            a,
            pa,
            b,
            pb,
        )
    }

    /// A rope segment: can go slack, can't stretch past its current length.
    pub fn rope(bodies: &[RigidBody], a: usize, pa: Vec2, b: Option<usize>, pb: Vec2) -> Self {
        Self::new(
            JointKind::Distance {
                min_length: 0.0,
                max_length: (pb - pa).length(),
            },
            bodies,
            a,
            pa,
            b,
            pb,
        )
    }

    /// A spring of the given stiffness (N/m) and damping (N·s/m), relaxed at
    /// its current length.
    #[allow(clippy::too_many_arguments)]
    pub fn spring(
        bodies: &[RigidBody],
        a: usize,
        pa: Vec2,
        b: Option<usize>,
        pb: Vec2,
        stiffness: f32,
        damping: f32,
    ) -> Self {
        Self::new(
            JointKind::Spring {
                rest_length: (pb - pa).length(),
                stiffness,
                damping,
            },
            bodies,
            a,
            pa,
            b,
            pb,
        )
    }

    /// World positions of the two anchors.
    pub fn anchors(&self, bodies: &[RigidBody]) -> (Vec2, Vec2) {
        let pa = bodies[self.a].local_to_world(self.local_anchor_a);
        let pb = match self.b {
            Some(b) => bodies[b].local_to_world(self.anchor_b),
            None => self.anchor_b,
        };
        (pa, pb)
    }

    /// Whether this joint connects body `i`.
    pub fn involves(&self, i: usize) -> bool {
        self.a == i || self.b == Some(i)
    }

    /// Fix body indices after body `removed` was taken out of the list.
    /// Returns false if the joint referenced it (and should be dropped).
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
            JointKind::Pin => {}
            JointKind::Distance {
                min_length,
                max_length,
            } => {
                if !(0.0 <= min_length && min_length <= max_length && max_length.is_finite()) {
                    return Err(format!(
                        "distance joint needs 0 ≤ min ≤ max, got {min_length}..{max_length}"
                    ));
                }
            }
            JointKind::Spring {
                rest_length,
                stiffness,
                damping,
            } => {
                if !(rest_length >= 0.0 && stiffness >= 0.0 && damping >= 0.0)
                    || !(rest_length + stiffness + damping).is_finite()
                {
                    return Err("spring needs non-negative length, stiffness, damping".into());
                }
            }
        }
        Ok(())
    }
}

/// What one end of a chain built by [`add_chain`] is attached to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Anchor {
    Free,
    /// Pinned to its starting point in the world.
    World,
    /// Pinned to this body (index into `World::bodies`).
    Body(usize),
}

/// Add a chain of `links` boxes from `start` to `end`, each pinned to the
/// next where they meet. Thin links make a rope; with both ends anchored and
/// some slack it's a rope bridge. Returns the indices of the new bodies.
pub fn add_chain(
    world: &mut crate::World,
    start: Vec2,
    end: Vec2,
    links: usize,
    thickness: f32,
    start_anchor: Anchor,
    end_anchor: Anchor,
) -> Vec<usize> {
    let links = links.max(1);
    let span = end - start;
    let step = span / links as f32;
    let angle = span.y.atan2(span.x);
    let length = step.length();
    let first = world.bodies.len();
    for k in 0..links {
        let center = start + step * (k as f32 + 0.5);
        let mut link = RigidBody::new(crate::Shape::rect(length, thickness), center);
        link.angle = angle;
        world.add_body(link);
    }
    let ids: Vec<usize> = (first..first + links).collect();
    for k in 1..links {
        let p = start + step * k as f32;
        let j = Joint::pin(&world.bodies, ids[k - 1], Some(ids[k]), p);
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

/// Velocity state shared with the contact solver. Index `world` (one past
/// the last body) is the world: immovable, zero velocity.
pub(crate) struct JointVelocities<'a> {
    pub v: &'a mut [Vec2],
    pub w: &'a mut [f32],
    pub inv_mass: &'a [f32],
    pub inv_inertia: &'a [f32],
}

impl JointVelocities<'_> {
    fn at(&self, i: usize, r: Vec2) -> Vec2 {
        self.v[i] + self.w[i] * r.perp()
    }

    fn apply(&mut self, a: usize, ra: Vec2, b: usize, rb: Vec2, impulse: Vec2) {
        self.v[a] -= self.inv_mass[a] * impulse;
        self.w[a] -= self.inv_inertia[a] * ra.perp_dot(impulse);
        self.v[b] += self.inv_mass[b] * impulse;
        self.w[b] += self.inv_inertia[b] * rb.perp_dot(impulse);
    }

    /// Effective mass k = J M⁻¹ Jᵀ along direction `d`.
    fn k_along(&self, a: usize, ra: Vec2, b: usize, rb: Vec2, d: Vec2) -> f32 {
        self.inv_mass[a]
            + self.inv_mass[b]
            + self.inv_inertia[a] * ra.perp_dot(d).powi(2)
            + self.inv_inertia[b] * rb.perp_dot(d).powi(2)
    }
}

/// Everything about a joint that stays fixed during one step's iterations.
pub(crate) struct PreparedJoint {
    a: usize,
    b: usize,
    ra: Vec2,
    rb: Vec2,
    /// pb − pa at the start of the step.
    gap: Vec2,
    /// The curved part of the anchors' motion over the step: how far each
    /// lever arm's true rotation by ω·dt departs from the straight-line
    /// estimate ω × r·dt (B's minus A's), using the spins at the start of the
    /// step.
    curvature: Vec2,
    rows: Rows,
}

impl PreparedJoint {
    /// How pb − pa changes over the step if the bodies keep their current
    /// velocities. Linear in the velocities, so sweeping it is an ordinary
    /// convergent Gauss–Seidel solve, plus the fixed `curvature` term that
    /// carries the centripetal effect. (Predicting the rotation exactly
    /// inside the sweeps instead diverges for fast spinners: with ω·dt near
    /// 1 rad the impulse direction no longer matches the prediction. A
    /// whipping chain blew up that way.)
    ///
    /// Only the *change* is predicted: the velocity sweeps keep a joint
    /// that's satisfied now satisfied at the end of the step, and any error
    /// already there is left to the position pass. Folding the existing
    /// error into the velocity target instead is Baumgarte stabilization at
    /// full strength, which pumps energy in (a chain at 8 substeps gained
    /// 42%).
    fn predicted_motion(&self, vel: &JointVelocities, dt: f32) -> Vec2 {
        (vel.at(self.b, self.rb) - vel.at(self.a, self.ra)) * dt + self.curvature
    }
}

enum Rows {
    /// Inverse of the 2×2 effective-mass matrix; `error` = pb − pa.
    Pin { inv_k: [[f32; 2]; 2], error: Vec2 },
    /// Unit direction a → b, effective mass, current length.
    Distance {
        n: Vec2,
        mass: f32,
        length: f32,
        min: f32,
        max: f32,
    },
    /// Soft constraint: `mass` includes compliance, `bias` and `gamma` per
    /// Catto's soft-constraint formulation.
    Spring {
        n: Vec2,
        mass: f32,
        bias: f32,
        gamma: f32,
    },
    /// Degenerate (zero length or both bodies immovable): skip.
    None,
}

/// Anchors closer than this have no defined direction; distance joints and
/// springs skip the step.
const MIN_LENGTH: f32 = 1e-6;

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
        None => Vec2::ZERO,
    };
    let rows = match joint.kind {
        JointKind::Pin => {
            let (ma, mb) = (vel.inv_mass[a], vel.inv_mass[b]);
            let (ia, ib) = (vel.inv_inertia[a], vel.inv_inertia[b]);
            // K = (ma + mb) I + ia [ra⊥][ra⊥]ᵀ + ib [rb⊥][rb⊥]ᵀ, ra⊥ = (−ra.y, ra.x)
            let k11 = ma + mb + ia * ra.y * ra.y + ib * rb.y * rb.y;
            let k12 = -ia * ra.x * ra.y - ib * rb.x * rb.y;
            let k22 = ma + mb + ia * ra.x * ra.x + ib * rb.x * rb.x;
            let det = k11 * k22 - k12 * k12;
            if det.abs() < 1e-12 {
                Rows::None
            } else {
                Rows::Pin {
                    inv_k: [[k22 / det, -k12 / det], [-k12 / det, k11 / det]],
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
                // Implicit spring: λ = −m_soft (Ċ + β + γ λ_acc), with
                // γ = 1 / (dt (c + dt k_s)), β = C dt k_s γ.
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
    // rot(ω dt) r − r − ω dt r⊥: the second-order and higher part of the
    // arm's swing.
    let curve = |r: Vec2, w: f32| Vec2::from_angle(w * dt).rotate(r) - r - w * dt * r.perp();
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
        joint.impulse = Vec2::ZERO;
        return;
    }
    let impulse = match p.rows {
        Rows::Pin { .. } => joint.impulse,
        Rows::Distance { n, .. } => (joint.impulse.x + joint.impulse.y) * n,
        Rows::Spring { n, .. } => joint.impulse.x * n,
        Rows::None => {
            joint.impulse = Vec2::ZERO;
            Vec2::ZERO
        }
    };
    vel.apply(p.a, p.ra, p.b, p.rb, impulse);
}

/// One velocity sweep over a joint: the impulse that zeroes the predicted
/// end-of-step error (see the module docs).
pub(crate) fn solve_velocity(
    p: &PreparedJoint,
    joint: &mut Joint,
    vel: &mut JointVelocities,
    dt: f32,
) {
    let (a, b, ra, rb) = (p.a, p.b, p.ra, p.rb);
    match p.rows {
        Rows::Pin { inv_k, .. } => {
            // λ = −K⁻¹ C_pred / dt, starting from a closed pin.
            let e = p.predicted_motion(vel, dt) / dt;
            let lambda = -Vec2::new(
                inv_k[0][0] * e.x + inv_k[0][1] * e.y,
                inv_k[1][0] * e.x + inv_k[1][1] * e.y,
            );
            joint.impulse += lambda;
            vel.apply(a, ra, b, rb, lambda);
        }
        Rows::Distance {
            n, mass, min, max, ..
        } => {
            // Impulses act along the joint's direction at the start of the
            // step; the predicted length decides how much. The prediction
            // starts from the current length clamped to the allowed range
            // (a violation is the position pass's job).
            let start = p.gap.normalize_or_zero() * p.gap.length().clamp(min, max);
            let predicted = |vel: &JointVelocities| (start + p.predicted_motion(vel, dt)).length();
            if min == max {
                let lambda = -mass * (predicted(vel) - max) / dt;
                joint.impulse.x += lambda;
                vel.apply(a, ra, b, rb, lambda * n);
                return;
            }
            // Lower limit: push only (λ ≥ 0).
            let old = joint.impulse.x;
            joint.impulse.x = (old - mass * (predicted(vel) - min) / dt).max(0.0);
            vel.apply(a, ra, b, rb, (joint.impulse.x - old) * n);
            // Upper limit: pull only (λ ≤ 0).
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

/// One sweep of position correction on the pseudo-velocities: aim to remove
/// `fraction` of the joint's position error this step. Springs aren't
/// corrected (their stretch is the point).
pub(crate) fn solve_position(
    p: &PreparedJoint,
    total: &mut Vec2,
    pseudo: &mut JointVelocities,
    fraction: f32,
    dt: f32,
) {
    let (a, b, ra, rb) = (p.a, p.b, p.ra, p.rb);
    match p.rows {
        Rows::Pin { inv_k, error } => {
            let target = -error * (fraction / dt);
            let e = pseudo.at(b, rb) - pseudo.at(a, ra) - target;
            let lambda = -Vec2::new(
                inv_k[0][0] * e.x + inv_k[0][1] * e.y,
                inv_k[1][0] * e.x + inv_k[1][1] * e.y,
            );
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
            // Only correct a violated limit.
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
    use crate::shape::Shape;

    #[test]
    fn reindexing_drops_joints_on_removed_bodies() {
        let bodies: Vec<RigidBody> = (0..4)
            .map(|i| RigidBody::new(Shape::Circle { radius: 0.1 }, Vec2::new(i as f32, 0.0)))
            .collect();
        let mut on_removed = Joint::pin(&bodies, 1, Some(2), Vec2::new(1.5, 0.0));
        let mut after = Joint::pin(&bodies, 0, Some(3), Vec2::new(1.5, 0.0));
        assert!(!on_removed.reindex_after_removal(2));
        assert!(after.reindex_after_removal(2));
        assert_eq!((after.a, after.b), (0, Some(2)));
    }

    #[test]
    fn new_distance_joints_start_relaxed() {
        let bodies = vec![
            RigidBody::new(Shape::Circle { radius: 0.1 }, Vec2::ZERO),
            RigidBody::new(Shape::Circle { radius: 0.1 }, Vec2::new(3.0, 4.0)),
        ];
        let rod = Joint::rod(&bodies, 0, Vec2::ZERO, Some(1), Vec2::new(3.0, 4.0));
        assert_eq!(
            rod.kind,
            JointKind::Distance {
                min_length: 5.0,
                max_length: 5.0
            }
        );
        assert!(rod.validate(2).is_ok());
        assert!(
            Joint::pin(&bodies, 0, Some(0), Vec2::ZERO)
                .validate(2)
                .is_err()
        );
    }
}
