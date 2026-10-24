//! 3D contact solver: sequential impulses, as in the 2D engine (see its
//! `solver` module for the background), with these changes for 3D:
//!
//! - **Inertia is a tensor.** Angular effects use each body's world-space
//!   inverse inertia matrix, computed once per step.
//! - **Patch friction.** Friction is solved once per *manifold*, at its
//!   center of pressure (the contact points weighted by their current
//!   normal impulses), rather than once per point. The center
//!   gets two tangential friction impulses (clamped together to a disc of
//!   radius μ × the manifold's total normal impulse, which is Coulomb's
//!   cone) and a twisting-friction impulse about the normal (limited by μ ×
//!   normal × the patch's mean radius).
//!
//!   Why: with per-point friction, solving a box's four corners one after
//!   another gives a slightly lopsided push that sets a sliding box spinning.
//!   Once it spins, part of the friction goes into fighting the spin instead
//!   of slowing the slide. A block on a 35° ramp slid 11% too far because of
//!   it. Patch friction has no ordering bias. PhysX offers the same idea as
//!   its "patch friction" mode.
//!
//!   Why the center of *pressure*: when a box rocks, its load shifts to one
//!   edge. Friction applied at the plain geometric center then acts in the
//!   wrong place and pumps energy into the rocking; a 6-box tower swayed
//!   indefinitely. Weighted by load, it settles to rest.
//!
//! Warm starting, accumulated impulse clamping, restitution with a speed
//! threshold, and split-impulse position correction are as in 2D.

use crate::body::RigidBody;
use crate::collision::Manifold;
use crate::joint::{self, Joint, JointVelocities};
use glam::{Mat3, Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverSettings {
    pub velocity_iterations: u32,
    /// Fraction of the remaining overlap removed per step.
    pub baumgarte: f32,
    /// Overlap tolerated without correction, m.
    pub allowed_penetration: f32,
    /// Impacts slower than this don't bounce, m/s.
    pub restitution_threshold: f32,
    pub warm_starting: bool,
    /// Rigid-body substeps per world step (see the 2D engine's
    /// `SolverSettings::substeps`). Chains and ropes stay tighter with 4–8.
    pub substeps: u32,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            velocity_iterations: 12,
            baumgarte: 0.2,
            allowed_penetration: 0.005,
            restitution_threshold: 1.0,
            warm_starting: true,
            substeps: 1,
        }
    }
}

pub struct PositionCorrection {
    pub v: Vec<Vec3>,
    pub w: Vec<Vec3>,
}

struct Velocities {
    v: Vec<Vec3>,
    w: Vec<Vec3>,
    inv_mass: Vec<f32>,
    inv_inertia: Vec<Mat3>,
}

impl Velocities {
    fn relative(&self, a: usize, ra: Vec3, b: usize, rb: Vec3) -> Vec3 {
        (self.v[b] + self.w[b].cross(rb)) - (self.v[a] + self.w[a].cross(ra))
    }

    /// Impulse `j` at offsets `ra`/`rb`: −j on A, +j on B.
    fn apply(&mut self, a: usize, ra: Vec3, b: usize, rb: Vec3, j: Vec3) {
        self.v[a] -= self.inv_mass[a] * j;
        self.w[a] -= self.inv_inertia[a] * ra.cross(j);
        self.v[b] += self.inv_mass[b] * j;
        self.w[b] += self.inv_inertia[b] * rb.cross(j);
    }

    /// Pure angular impulse `l` about an axis: −l on A, +l on B.
    fn apply_angular(&mut self, a: usize, b: usize, l: Vec3) {
        self.w[a] -= self.inv_inertia[a] * l;
        self.w[b] += self.inv_inertia[b] * l;
    }

    fn joint_view(&mut self) -> JointVelocities<'_> {
        JointVelocities {
            v: &mut self.v,
            w: &mut self.w,
            inv_mass: &self.inv_mass,
            inv_inertia: &self.inv_inertia,
        }
    }

    /// 1 / effective mass along direction `d` at offsets `ra`/`rb`.
    fn effective_mass(&self, a: usize, ra: Vec3, b: usize, rb: Vec3, d: Vec3) -> f32 {
        let k = self.inv_mass[a]
            + self.inv_mass[b]
            + (self.inv_inertia[a] * ra.cross(d)).cross(ra).dot(d)
            + (self.inv_inertia[b] * rb.cross(d)).cross(rb).dot(d);
        if k > 0.0 { 1.0 / k } else { 0.0 }
    }
}

/// Per contact point: the normal constraint.
struct PointConstraint {
    point: usize,
    ra: Vec3,
    rb: Vec3,
    normal_mass: f32,
    restitution_bias: f32,
    position_bias: f32,
}

/// Per manifold: the points plus patch friction at their center.
struct ManifoldConstraint {
    a: usize,
    b: usize,
    normal: Vec3,
    tangents: [Vec3; 2],
    friction: f32,
    points: Vec<PointConstraint>,
    /// World-space contact positions (to locate the center of pressure).
    positions: Vec<Vec3>,
    /// Body centers of mass.
    pa: Vec3,
    pb: Vec3,
    twist_mass: f32,
}

/// Where patch friction acts, and how strongly it resists twisting.
struct Patch {
    ca: Vec3,
    cb: Vec3,
    tangent_mass: [f32; 2],
    /// Load-weighted mean distance of the points from the center, m.
    radius: f32,
}

impl ManifoldConstraint {
    /// Patch friction acts at the *center of pressure*: the contact points
    /// weighted by how hard each one currently pushes.
    fn patch(&self, impulses: &[f32], vel: &Velocities) -> Patch {
        let total: f32 = impulses.iter().sum();
        let (center, radius) = if total > 1e-9 {
            let c = self
                .positions
                .iter()
                .zip(impulses)
                .map(|(&p, &j)| p * j)
                .sum::<Vec3>()
                / total;
            let r = self
                .positions
                .iter()
                .zip(impulses)
                .map(|(&p, &j)| (p - c).length() * j)
                .sum::<f32>()
                / total;
            (c, r)
        } else {
            (
                self.positions.iter().sum::<Vec3>() / self.positions.len() as f32,
                0.0,
            )
        };
        let (ca, cb) = (center - self.pa, center - self.pb);
        Patch {
            ca,
            cb,
            tangent_mass: [
                vel.effective_mass(self.a, ca, self.b, cb, self.tangents[0]),
                vel.effective_mass(self.a, ca, self.b, cb, self.tangents[1]),
            ],
            radius,
        }
    }
}

/// Resolve all contacts by changing body velocities. Returns the
/// position-only corrections (split impulse) to apply this step.
pub fn solve(
    bodies: &mut [RigidBody],
    manifolds: &mut [Manifold],
    joints: &mut [Joint],
    settings: &SolverSettings,
    dt: f32,
) -> PositionCorrection {
    let n = bodies.len();
    let mut vel = Velocities {
        v: bodies
            .iter()
            .map(|b| if b.is_static { Vec3::ZERO } else { b.velocity })
            .collect(),
        w: bodies
            .iter()
            .map(|b| {
                if b.is_static {
                    Vec3::ZERO
                } else {
                    b.angular_velocity
                }
            })
            .collect(),
        inv_mass: bodies.iter().map(RigidBody::inverse_mass).collect(),
        inv_inertia: bodies
            .iter()
            .map(RigidBody::inverse_inertia_world)
            .collect(),
    };
    // Index n: the immovable world, for joints anchored to a fixed point.
    vel.v.push(Vec3::ZERO);
    vel.w.push(Vec3::ZERO);
    vel.inv_mass.push(0.0);
    vel.inv_inertia.push(Mat3::ZERO);
    let world = n;
    let prepared_joints: Vec<joint::PreparedJoint> = joints
        .iter()
        .map(|j| joint::prepare(j, bodies, &vel.joint_view(), world, dt))
        .collect();
    for (pj, j) in prepared_joints.iter().zip(joints.iter_mut()) {
        joint::warm_start(pj, j, &mut vel.joint_view(), settings.warm_starting);
    }

    // --- Prepare.
    let constraints: Vec<ManifoldConstraint> = manifolds
        .iter()
        .map(|m| {
            let (ba, bb) = (&bodies[m.a], &bodies[m.b]);
            let (a, b) = (m.a, m.b);
            let restitution = ba.restitution.max(bb.restitution);
            let points = m
                .points
                .iter()
                .enumerate()
                .map(|(pi, point)| {
                    let ra = point.position - ba.position;
                    let rb = point.position - bb.position;
                    let vn = vel.relative(a, ra, b, rb).dot(m.normal);
                    PointConstraint {
                        point: pi,
                        ra,
                        rb,
                        normal_mass: vel.effective_mass(a, ra, b, rb, m.normal),
                        restitution_bias: if vn < -settings.restitution_threshold {
                            -restitution * vn
                        } else {
                            0.0
                        },
                        position_bias: settings.baumgarte / dt
                            * (point.penetration - settings.allowed_penetration).max(0.0),
                    }
                })
                .collect();
            let (t1, t2) = m.normal.any_orthonormal_pair();
            let twist_k = m.normal.dot(vel.inv_inertia[a] * m.normal)
                + m.normal.dot(vel.inv_inertia[b] * m.normal);
            ManifoldConstraint {
                a,
                b,
                normal: m.normal,
                tangents: [t1, t2],
                friction: (ba.friction * bb.friction).sqrt(),
                points,
                positions: m.points.iter().map(|p| p.position).collect(),
                pa: ba.position,
                pb: bb.position,
                twist_mass: if twist_k > 0.0 { 1.0 / twist_k } else { 0.0 },
            }
        })
        .collect();

    // --- Warm start.
    for (c, m) in constraints.iter().zip(manifolds.iter_mut()) {
        if !settings.warm_starting {
            m.friction_impulse = Vec2::ZERO;
            m.twist_impulse = 0.0;
            for p in &mut m.points {
                p.normal_impulse = 0.0;
            }
            continue;
        }
        for pc in &c.points {
            let j = m.points[pc.point].normal_impulse * c.normal;
            vel.apply(c.a, pc.ra, c.b, pc.rb, j);
        }
        let impulses: Vec<f32> = m.points.iter().map(|p| p.normal_impulse).collect();
        let patch = c.patch(&impulses, &vel);
        let f = m.friction_impulse.x * c.tangents[0] + m.friction_impulse.y * c.tangents[1];
        vel.apply(c.a, patch.ca, c.b, patch.cb, f);
        vel.apply_angular(c.a, c.b, m.twist_impulse * c.normal);
    }

    // --- Iterate.
    for _ in 0..settings.velocity_iterations {
        // Joints first; contacts get the last word.
        for (pj, j) in prepared_joints.iter().zip(joints.iter_mut()) {
            joint::solve_velocity(pj, j, &mut vel.joint_view(), dt);
        }
        for (c, m) in constraints.iter().zip(manifolds.iter_mut()) {
            let impulses: Vec<f32> = m.points.iter().map(|p| p.normal_impulse).collect();
            let limit = c.friction * impulses.iter().sum::<f32>();
            let patch = c.patch(&impulses, &vel);

            // Tangential patch friction at the center, clamped to the disc.
            let dv = vel.relative(c.a, patch.ca, c.b, patch.cb);
            let old = m.friction_impulse;
            let step = Vec2::new(
                -patch.tangent_mass[0] * dv.dot(c.tangents[0]),
                -patch.tangent_mass[1] * dv.dot(c.tangents[1]),
            );
            m.friction_impulse = (old + step).clamp_length_max(limit);
            let d = m.friction_impulse - old;
            vel.apply(
                c.a,
                patch.ca,
                c.b,
                patch.cb,
                d.x * c.tangents[0] + d.y * c.tangents[1],
            );

            // Twisting friction about the normal.
            if patch.radius > 0.0 {
                let spin = (vel.w[c.b] - vel.w[c.a]).dot(c.normal);
                let old = m.twist_impulse;
                let max = limit * patch.radius;
                m.twist_impulse = (old - c.twist_mass * spin).clamp(-max, max);
                vel.apply_angular(c.a, c.b, (m.twist_impulse - old) * c.normal);
            }

            // Normal at each point: push, never pull.
            for pc in &c.points {
                let point = &mut m.points[pc.point];
                let vn = vel.relative(c.a, pc.ra, c.b, pc.rb).dot(c.normal);
                let old = point.normal_impulse;
                point.normal_impulse = (old + pc.normal_mass * (pc.restitution_bias - vn)).max(0.0);
                vel.apply(
                    c.a,
                    pc.ra,
                    c.b,
                    pc.rb,
                    (point.normal_impulse - old) * c.normal,
                );
            }
        }
    }

    for (i, body) in bodies.iter_mut().enumerate() {
        if !body.is_static {
            body.velocity = vel.v[i];
            body.angular_velocity = vel.w[i];
        }
    }

    // --- Split-impulse position correction on separate pseudo-velocities.
    let mut pseudo = Velocities {
        v: vec![Vec3::ZERO; n + 1],
        w: vec![Vec3::ZERO; n + 1],
        inv_mass: vel.inv_mass,
        inv_inertia: vel.inv_inertia,
    };
    let mut joint_totals = vec![Vec3::ZERO; joints.len()];
    let mut totals: Vec<Vec<f32>> = constraints
        .iter()
        .map(|c| vec![0.0; c.points.len()])
        .collect();
    for _ in 0..settings.velocity_iterations {
        for (pj, total) in prepared_joints.iter().zip(&mut joint_totals) {
            joint::solve_position(pj, total, &mut pseudo.joint_view(), settings.baumgarte, dt);
        }
        for (c, t) in constraints.iter().zip(&mut totals) {
            for (pc, total) in c.points.iter().zip(t.iter_mut()) {
                if pc.position_bias == 0.0 && *total == 0.0 {
                    continue;
                }
                let vn = pseudo.relative(c.a, pc.ra, c.b, pc.rb).dot(c.normal);
                let old = *total;
                *total = (old + pc.normal_mass * (pc.position_bias - vn)).max(0.0);
                pseudo.apply(c.a, pc.ra, c.b, pc.rb, (*total - old) * c.normal);
            }
        }
    }
    pseudo.v.truncate(n);
    pseudo.w.truncate(n);
    PositionCorrection {
        v: pseudo.v,
        w: pseudo.w,
    }
}
