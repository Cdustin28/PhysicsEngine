//! Contact solver: turns contact manifolds into velocity changes.
//!
//! This is the *sequential impulse* method from Erin Catto's Box2D. Each
//! contact point gets a normal impulse that stops the bodies moving into each
//! other (it can push but never pull), and a friction impulse that resists
//! sliding, limited by Coulomb's law `|friction| <= μ × normal`. Contacts are
//! solved one at a time and the whole set is swept several times. Each sweep
//! brings the system closer to satisfying every contact at once.
//!
//! Four refinements make stacks stable:
//!
//! - **Accumulated impulse clamping.** The *running total* impulse per
//!   point is clamped, not each increment. That lets a later sweep take back
//!   some of an earlier sweep's push.
//! - **Warm starting.** Each contact starts from last step's total impulse,
//!   so a resting stack begins every step already almost solved.
//! - **Block solving.** The two points of a flat contact (a box lying on a
//!   face) are solved *together* as a 2×2 linear complementarity problem.
//!   Solved one at a time they fight each other and leave a small residual
//!   twist that makes boxes creep and towers sway.
//! - **Split-impulse position correction.** Overlap is removed by a
//!   separate *pseudo-velocity* that moves bodies apart this step and is
//!   then thrown away. Classic Baumgarte stabilization feeds the correction
//!   into the real velocity instead, which makes overlapping bodies fly apart
//!   and makes "dead" (restitution 0) objects bounce. A slop allowance stops
//!   the correction fighting tiny overlaps, which would cause jitter.

use crate::body::RigidBody;
use crate::collision::Manifold;
use crate::joint::{self, Joint, JointVelocities};
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Tuning knobs for the contact solver. Defaults suit human-scale scenes
/// (objects 0.1–10 m) at 60–240 Hz.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverSettings {
    /// Sweeps over all contacts per step. More iterations give stiffer stacks
    /// but cost more time.
    pub velocity_iterations: u32,
    /// Fraction of the remaining overlap removed per step, in [0, 1].
    /// Higher removes overlap faster but can make stacks jittery.
    /// 0.1–0.3 is typical.
    pub baumgarte: f32,
    /// Overlap tolerated without correction, m. Prevents jitter.
    pub allowed_penetration: f32,
    /// Impacts slower than this don't bounce, m/s. Stops resting bodies with
    /// restitution > 0 from vibrating forever.
    pub restitution_threshold: f32,
    /// Start each step from last step's impulses.
    pub warm_starting: bool,
    /// Split each world step into this many rigid-body substeps (collision,
    /// solve, move). 1 is fine for stacks and piles; chains and ropes stay
    /// much tighter with 4–8, at proportional cost. The fluid still steps
    /// once per world step.
    pub substeps: u32,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            velocity_iterations: 10,
            baumgarte: 0.2,
            allowed_penetration: 0.005,
            restitution_threshold: 1.0,
            warm_starting: true,
            substeps: 1,
        }
    }
}

/// Per-body pseudo-velocities from position correction. Add these to the
/// real velocities when integrating positions this step, then discard them.
pub struct PositionCorrection {
    pub v: Vec<Vec2>,
    pub w: Vec<f32>,
}

/// Per-point quantities that don't change during the solver iterations.
struct PreparedPoint {
    /// Contact point relative to each body's center of mass, m.
    ra: Vec2,
    rb: Vec2,
    /// 1 / (effective mass along the normal and tangent).
    normal_mass: f32,
    tangent_mass: f32,
    /// Target separating velocity from restitution, m/s.
    restitution_bias: f32,
    /// Target pseudo-velocity for position correction, m/s.
    position_bias: f32,
}

/// The 2×2 effective-mass matrix K for a two-point manifold, and its inverse.
/// Entry K[i][j] is how much an impulse at point j changes the normal
/// velocity at point i.
struct Block {
    k: [[f32; 2]; 2],
    inv_k: [[f32; 2]; 2],
}

struct PreparedManifold {
    a: usize,
    b: usize,
    normal: Vec2,
    tangent: Vec2,
    friction: f32,
    points: Vec<PreparedPoint>,
    /// Present only for well-conditioned two-point manifolds.
    block: Option<Block>,
}

/// Velocity state copied out of the bodies, so the solver can update two
/// bodies at once without fighting the borrow checker over `&mut bodies[a]`
/// and `&mut bodies[b]`.
struct Velocities {
    v: Vec<Vec2>,
    w: Vec<f32>,
    inv_mass: Vec<f32>,
    inv_inertia: Vec<f32>,
}

impl Velocities {
    /// Velocity of body `i`'s material at offset `r` from its center:
    /// v + ω × r.
    fn at(&self, i: usize, r: Vec2) -> Vec2 {
        self.v[i] + self.w[i] * r.perp()
    }

    /// The same state as seen by the joint solver.
    fn joint_view(&mut self) -> JointVelocities<'_> {
        JointVelocities {
            v: &mut self.v,
            w: &mut self.w,
            inv_mass: &self.inv_mass,
            inv_inertia: &self.inv_inertia,
        }
    }

    /// Relative velocity of B with respect to A at a contact point.
    fn relative(&self, m: &PreparedManifold, p: &PreparedPoint) -> Vec2 {
        self.at(m.b, p.rb) - self.at(m.a, p.ra)
    }

    /// Apply impulse `impulse` at the contact point to body B and the
    /// opposite impulse to body A (Newton's third law).
    fn apply(&mut self, m: &PreparedManifold, p: &PreparedPoint, impulse: Vec2) {
        let (a, b) = (m.a, m.b);
        self.v[a] -= self.inv_mass[a] * impulse;
        self.w[a] -= self.inv_inertia[a] * p.ra.perp_dot(impulse);
        self.v[b] += self.inv_mass[b] * impulse;
        self.w[b] += self.inv_inertia[b] * p.rb.perp_dot(impulse);
    }
}

/// Solve the normal impulses of one manifold, aiming for a separating normal
/// velocity of `target[i]` at each point. `totals` holds the accumulated
/// impulses and is updated in place. `normal_velocity(vel, i)` reads the
/// current normal velocity at point `i`.
fn solve_normal(
    vel: &mut Velocities,
    m: &PreparedManifold,
    totals: &mut [f32],
    target: impl Fn(&PreparedPoint) -> f32,
) {
    if let (Some(block), [p1, p2]) = (&m.block, &m.points[..]) {
        let vn1 = vel.relative(m, p1).dot(m.normal);
        let vn2 = vel.relative(m, p2).dot(m.normal);
        let old = [totals[0], totals[1]];
        let new = solve_block(
            block,
            [vn1, vn2],
            [target(p1), target(p2)],
            old,
            [p1.normal_mass, p2.normal_mass],
        );
        vel.apply(m, p1, (new[0] - old[0]) * m.normal);
        vel.apply(m, p2, (new[1] - old[1]) * m.normal);
        totals[0] = new[0];
        totals[1] = new[1];
        return;
    }
    for (p, total) in m.points.iter().zip(totals.iter_mut()) {
        let vn = vel.relative(m, p).dot(m.normal);
        let old = *total;
        *total = (old + p.normal_mass * (target(p) - vn)).max(0.0);
        vel.apply(m, p, (*total - old) * m.normal);
    }
}

/// Exact solution of the two-point contact problem, by trying each of the
/// four possible active sets (Box2D's block solver).
///
/// We want total impulses `x >= 0` such that the resulting normal velocities
/// `vn' = vn + K (x - old)` satisfy `vn' >= target`, with
/// `x_i * (vn'_i - target_i) = 0` (each point either pushes, or is
/// separating, never both).
fn solve_block(
    block: &Block,
    vn: [f32; 2],
    target: [f32; 2],
    old: [f32; 2],
    normal_mass: [f32; 2],
) -> [f32; 2] {
    let k = &block.k;
    // Velocity error if all impulses were removed: b = vn - target - K old.
    let b = [
        vn[0] - target[0] - (k[0][0] * old[0] + k[0][1] * old[1]),
        vn[1] - target[1] - (k[1][0] * old[0] + k[1][1] * old[1]),
    ];

    // Case 1: both points push. Solve K x = -b.
    let inv = &block.inv_k;
    let x = [
        -(inv[0][0] * b[0] + inv[0][1] * b[1]),
        -(inv[1][0] * b[0] + inv[1][1] * b[1]),
    ];
    if x[0] >= 0.0 && x[1] >= 0.0 {
        return x;
    }
    // Case 2: only point 1 pushes; point 2 must then be separating.
    let x1 = -normal_mass[0] * b[0];
    if x1 >= 0.0 && k[1][0] * x1 + b[1] >= 0.0 {
        return [x1, 0.0];
    }
    // Case 3: only point 2 pushes.
    let x2 = -normal_mass[1] * b[1];
    if x2 >= 0.0 && k[0][1] * x2 + b[0] >= 0.0 {
        return [0.0, x2];
    }
    // Case 4: neither pushes; both must be separating.
    if b[0] >= 0.0 && b[1] >= 0.0 {
        return [0.0, 0.0];
    }
    // No consistent solution (can happen through round-off): keep the old
    // impulses rather than inject garbage.
    old
}

/// Resolve all contacts by changing body velocities. Accumulated impulses are
/// written back into `manifolds` for warm starting next step.
///
/// Returns the pseudo-velocities that remove overlap. See
/// [`PositionCorrection`].
pub fn solve(
    bodies: &mut [RigidBody],
    manifolds: &mut [Manifold],
    joints: &mut [Joint],
    settings: &SolverSettings,
    dt: f32,
) -> PositionCorrection {
    let n = bodies.len();
    let mut vel = Velocities {
        // Static bodies never move, whatever velocity is stored on them.
        v: bodies
            .iter()
            .map(|b| if b.is_static { Vec2::ZERO } else { b.velocity })
            .collect(),
        w: bodies
            .iter()
            .map(|b| if b.is_static { 0.0 } else { b.angular_velocity })
            .collect(),
        inv_mass: bodies.iter().map(RigidBody::inverse_mass).collect(),
        inv_inertia: bodies.iter().map(RigidBody::inverse_inertia).collect(),
    };
    // One extra, immovable slot at index n: the world, for joints anchored
    // to a fixed point.
    vel.v.push(Vec2::ZERO);
    vel.w.push(0.0);
    vel.inv_mass.push(0.0);
    vel.inv_inertia.push(0.0);
    let world = n;

    // --- Prepare everything that stays fixed during the iterations.
    let prepared: Vec<PreparedManifold> = manifolds
        .iter()
        .map(|m| prepare(m, bodies, &vel, settings, dt))
        .collect();

    let prepared_joints: Vec<joint::PreparedJoint> = joints
        .iter()
        .map(|j| joint::prepare(j, bodies, &vel.joint_view(), world, dt))
        .collect();
    for (pj, j) in prepared_joints.iter().zip(joints.iter_mut()) {
        joint::warm_start(pj, j, &mut vel.joint_view(), settings.warm_starting);
    }

    // --- Warm start: re-apply last step's impulses.
    for (pm, m) in prepared.iter().zip(manifolds.iter_mut()) {
        for (p, point) in pm.points.iter().zip(&mut m.points) {
            if settings.warm_starting {
                let impulse = point.normal_impulse * pm.normal + point.tangent_impulse * pm.tangent;
                vel.apply(pm, p, impulse);
            } else {
                point.normal_impulse = 0.0;
                point.tangent_impulse = 0.0;
            }
        }
    }

    // --- Velocity iterations.
    let mut normal_totals: Vec<f32> = Vec::with_capacity(2);
    for _ in 0..settings.velocity_iterations {
        // Joints first, then contacts, so contacts (which can't be
        // violated by pushing through a wall) get the last word.
        for (pj, j) in prepared_joints.iter().zip(joints.iter_mut()) {
            joint::solve_velocity(pj, j, &mut vel.joint_view(), dt);
        }
        for (pm, m) in prepared.iter().zip(manifolds.iter_mut()) {
            // Friction first: it depends on the normal impulse, and solving
            // the normal last means non-penetration gets the final word.
            for (p, point) in pm.points.iter().zip(&mut m.points) {
                let vt = vel.relative(pm, p).dot(pm.tangent);
                let max_friction = pm.friction * point.normal_impulse;
                let old = point.tangent_impulse;
                point.tangent_impulse =
                    (old - p.tangent_mass * vt).clamp(-max_friction, max_friction);
                vel.apply(pm, p, (point.tangent_impulse - old) * pm.tangent);
            }

            // Normal: aim for a separating speed of `restitution_bias`
            // (zero for resting contact). Contacts push but never pull.
            normal_totals.clear();
            normal_totals.extend(m.points.iter().map(|p| p.normal_impulse));
            solve_normal(&mut vel, pm, &mut normal_totals, |p| p.restitution_bias);
            for (point, &total) in m.points.iter_mut().zip(&normal_totals) {
                point.normal_impulse = total;
            }
        }
    }

    // --- Write velocities back.
    for (i, body) in bodies.iter_mut().enumerate() {
        if !body.is_static {
            body.velocity = vel.v[i];
            body.angular_velocity = vel.w[i];
        }
    }

    // --- Position correction (split impulse): the same normal solve, run on
    // separate pseudo-velocities that start at zero, aiming for a separating
    // speed of `position_bias`. No friction and no warm starting.
    let mut pseudo = Velocities {
        v: vec![Vec2::ZERO; n + 1],
        w: vec![0.0; n + 1],
        inv_mass: vel.inv_mass,
        inv_inertia: vel.inv_inertia,
    };
    let mut joint_totals = vec![Vec2::ZERO; joints.len()];
    let mut totals: Vec<Vec<f32>> = prepared
        .iter()
        .map(|pm| vec![0.0; pm.points.len()])
        .collect();
    let needs_correction: Vec<bool> = prepared
        .iter()
        .map(|pm| pm.points.iter().any(|p| p.position_bias > 0.0))
        .collect();
    for _ in 0..settings.velocity_iterations {
        for (pj, total) in prepared_joints.iter().zip(&mut joint_totals) {
            joint::solve_position(pj, total, &mut pseudo.joint_view(), settings.baumgarte, dt);
        }
        for ((pm, t), &needed) in prepared.iter().zip(&mut totals).zip(&needs_correction) {
            if needed {
                solve_normal(&mut pseudo, pm, t, |p| p.position_bias);
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

fn prepare(
    m: &Manifold,
    bodies: &[RigidBody],
    vel: &Velocities,
    settings: &SolverSettings,
    dt: f32,
) -> PreparedManifold {
    let (body_a, body_b) = (&bodies[m.a], &bodies[m.b]);
    let (a, b) = (m.a, m.b);
    let normal = m.normal;
    let tangent = normal.perp();
    // Common mixing rules: the bouncier material wins; friction is the
    // geometric mean, so anything on ice is slippery.
    let restitution = body_a.restitution.max(body_b.restitution);
    let friction = (body_a.friction * body_b.friction).sqrt();
    let (ima, imb) = (vel.inv_mass[a], vel.inv_mass[b]);
    let (iia, iib) = (vel.inv_inertia[a], vel.inv_inertia[b]);

    let points: Vec<PreparedPoint> = m
        .points
        .iter()
        .map(|point| {
            let ra = point.position - body_a.position;
            let rb = point.position - body_b.position;
            // Effective mass along a direction d:
            //   k = 1/mA + 1/mB + (rA × d)²/IA + (rB × d)²/IB
            let k =
                |d: Vec2| ima + imb + iia * ra.perp_dot(d).powi(2) + iib * rb.perp_dot(d).powi(2);
            let vn = (vel.at(b, rb) - vel.at(a, ra)).dot(normal);
            PreparedPoint {
                ra,
                rb,
                normal_mass: 1.0 / k(normal),
                tangent_mass: 1.0 / k(tangent),
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

    let block = match &points[..] {
        [p1, p2] => {
            let (rn1a, rn1b) = (p1.ra.perp_dot(normal), p1.rb.perp_dot(normal));
            let (rn2a, rn2b) = (p2.ra.perp_dot(normal), p2.rb.perp_dot(normal));
            let k11 = ima + imb + iia * rn1a * rn1a + iib * rn1b * rn1b;
            let k22 = ima + imb + iia * rn2a * rn2a + iib * rn2b * rn2b;
            let k12 = ima + imb + iia * rn1a * rn2a + iib * rn1b * rn2b;
            let det = k11 * k22 - k12 * k12;
            // If the two points are nearly on top of each other the matrix is
            // nearly singular; fall back to solving them one at a time.
            const MAX_CONDITION: f32 = 1000.0;
            (k11 * k11 < MAX_CONDITION * det).then(|| Block {
                k: [[k11, k12], [k12, k22]],
                inv_k: [[k22 / det, -k12 / det], [-k12 / det, k11 / det]],
            })
        }
        _ => None,
    };

    PreparedManifold {
        a,
        b,
        normal,
        tangent,
        friction,
        points,
        block,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block solver must satisfy the complementarity conditions it
    /// claims to.
    #[test]
    fn block_solver_finds_complementary_solution() {
        let k = [[2.0, 0.5], [0.5, 1.5]];
        let det = k[0][0] * k[1][1] - k[0][1] * k[1][0];
        let block = Block {
            k,
            inv_k: [
                [k[1][1] / det, -k[0][1] / det],
                [-k[1][0] / det, k[0][0] / det],
            ],
        };
        let nm = [1.0 / k[0][0], 1.0 / k[1][1]];
        // Various approaching / separating situations.
        for vn in [
            [-1.0, -1.0],
            [-1.0, 0.5],
            [0.5, -2.0],
            [1.0, 1.0],
            [-0.1, -3.0],
        ] {
            let x = solve_block(&block, vn, [0.0, 0.0], [0.0, 0.0], nm);
            let after = [
                vn[0] + k[0][0] * x[0] + k[0][1] * x[1],
                vn[1] + k[1][0] * x[0] + k[1][1] * x[1],
            ];
            for i in 0..2 {
                assert!(x[i] >= 0.0, "vn {vn:?}: negative impulse {x:?}");
                assert!(after[i] >= -1e-5, "vn {vn:?}: still approaching {after:?}");
                assert!(
                    (x[i] * after[i]).abs() < 1e-5,
                    "vn {vn:?}: not complementary"
                );
            }
        }
    }
}
