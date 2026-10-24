//! 3D narrow phase: contact manifolds for sphere and box pairs.
//!
//! Conventions match the 2D engine: the normal points **from body A to
//! body B**, `penetration` > 0 is the overlap along it, and each contact
//! point sits halfway between the two surfaces.
//!
//! Box vs box uses the Separating Axis Theorem over 15 candidate axes (3
//! face normals of each box and the 9 cross products of their edge
//! directions). If the best axis is a face normal, the contact is a face:
//! the other box's most opposed face is clipped against the side planes of
//! the reference face (Sutherland–Hodgman), giving up to 4 points, which is
//! what keeps a box resting flat instead of rocking. If the best axis is an
//! edge cross product, the contact is a single point between two edges.

use crate::body::RigidBody;
use crate::shape::Shape;
use glam::{Vec2, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct Manifold {
    /// Index of body A (always `a < b`).
    pub a: usize,
    pub b: usize,
    /// Unit normal from A toward B.
    pub normal: Vec3,
    pub points: Vec<ContactPoint>,
    /// Accumulated friction impulse at the manifold's center, along its two
    /// tangent directions (see `solver`: friction is per manifold).
    pub friction_impulse: Vec2,
    /// Accumulated twisting-friction impulse about the normal.
    pub twist_impulse: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    pub position: Vec3,
    pub penetration: f32,
    /// Accumulated normal impulse, kept for warm starting.
    pub normal_impulse: f32,
}

impl ContactPoint {
    fn new(position: Vec3, penetration: f32) -> Self {
        Self {
            position,
            penetration,
            normal_impulse: 0.0,
        }
    }
}

/// Test bodies `a` and `b` (indices `ia < ib`) for contact.
pub fn collide(ia: usize, a: &RigidBody, ib: usize, b: &RigidBody) -> Option<Manifold> {
    let (normal, points) = match (a.shape, b.shape) {
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            sphere_sphere(a.position, ra, b.position, rb)?
        }
        (Shape::Cuboid { .. }, Shape::Sphere { radius }) => box_sphere(a, b.position, radius)?,
        (Shape::Sphere { radius }, Shape::Cuboid { .. }) => {
            let (n, pts) = box_sphere(b, a.position, radius)?;
            (-n, pts)
        }
        (Shape::Cuboid { .. }, Shape::Cuboid { .. }) => box_box(a, b)?,
    };
    Some(Manifold {
        a: ia,
        b: ib,
        normal,
        points,
        friction_impulse: Vec2::ZERO,
        twist_impulse: 0.0,
    })
}

fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32) -> Option<(Vec3, Vec<ContactPoint>)> {
    let d = cb - ca;
    let r = ra + rb;
    let dist_sq = d.length_squared();
    if dist_sq > r * r {
        return None;
    }
    let dist = dist_sq.sqrt();
    let normal = if dist > 1e-6 { d / dist } else { Vec3::Y };
    let penetration = r - dist;
    Some((
        normal,
        vec![ContactPoint::new(
            ca + normal * (ra - 0.5 * penetration),
            penetration,
        )],
    ))
}

/// Box (A) vs sphere with center `c` and radius `r` (B).
fn box_sphere(bx: &RigidBody, c: Vec3, r: f32) -> Option<(Vec3, Vec<ContactPoint>)> {
    let Shape::Cuboid { half_extents: h } = bx.shape else {
        unreachable!("box_sphere called with a non-box");
    };
    let local = bx.world_to_local(c);
    let clamped = local.clamp(-h, h);
    let (surface, local_normal) = if clamped != local {
        let delta = local - clamped;
        if delta.length_squared() > r * r {
            return None;
        }
        (clamped, delta.normalize())
    } else {
        // Center inside: leave through the nearest face.
        let gap = h - local.abs();
        let axis = if gap.x <= gap.y && gap.x <= gap.z {
            0
        } else if gap.y <= gap.z {
            1
        } else {
            2
        };
        let mut n = Vec3::ZERO;
        n[axis] = if local[axis] < 0.0 { -1.0 } else { 1.0 };
        let mut s = local;
        s[axis] = n[axis] * h[axis];
        (s, n)
    };
    let normal = bx.orientation * local_normal;
    let box_point = bx.local_to_world(surface);
    let sphere_point = c - normal * r;
    let penetration = (box_point - sphere_point).dot(normal);
    if penetration < 0.0 {
        return None;
    }
    Some((
        normal,
        vec![ContactPoint::new(
            0.5 * (box_point + sphere_point),
            penetration,
        )],
    ))
}

/// Edge axes must beat the best face axis by this relative margin to be
/// used. Faces give multi-point manifolds and are more stable; without the
/// bias the choice flickers between nearly equal axes and boxes jitter.
const EDGE_BIAS: f32 = 0.95;
const EDGE_ABSOLUTE_BIAS: f32 = 1e-3;

#[derive(Clone, Copy)]
enum Axis {
    /// Face `k` of box A.
    FaceA(usize),
    /// Face `k` of box B.
    FaceB(usize),
    /// Edge `i` of A crossed with edge `j` of B.
    Edge(usize, usize),
}

fn box_box(a: &RigidBody, b: &RigidBody) -> Option<(Vec3, Vec<ContactPoint>)> {
    let (Shape::Cuboid { half_extents: ha }, Shape::Cuboid { half_extents: hb }) =
        (a.shape, b.shape)
    else {
        unreachable!("box_box called with a non-box");
    };
    let (ua, ub) = (a.axes(), b.axes());
    let d = b.position - a.position;

    // Overlap of the two boxes' projections onto axis `l` (unit length).
    let overlap = |l: Vec3| {
        let ra: f32 = (0..3).map(|k| ha[k] * ua[k].dot(l).abs()).sum();
        let rb: f32 = (0..3).map(|k| hb[k] * ub[k].dot(l).abs()).sum();
        ra + rb - d.dot(l).abs()
    };

    let mut best_face = (f32::INFINITY, Axis::FaceA(0), Vec3::X);
    for k in 0..3 {
        for (axis, l) in [(Axis::FaceA(k), ua[k]), (Axis::FaceB(k), ub[k])] {
            let o = overlap(l);
            if o < 0.0 {
                return None; // separating axis found
            }
            if o < best_face.0 {
                best_face = (o, axis, l);
            }
        }
    }
    let mut best_edge = (f32::INFINITY, Axis::Edge(0, 0), Vec3::X);
    for (i, ea) in ua.iter().enumerate() {
        for (j, eb) in ub.iter().enumerate() {
            let c = ea.cross(*eb);
            let len = c.length();
            if len < 1e-5 {
                continue; // parallel edges: covered by the face axes
            }
            let l = c / len;
            let o = overlap(l);
            if o < 0.0 {
                return None;
            }
            if o < best_edge.0 {
                best_edge = (o, Axis::Edge(i, j), l);
            }
        }
    }

    let (penetration, axis, l) = if best_edge.0 < EDGE_BIAS * best_face.0 - EDGE_ABSOLUTE_BIAS {
        best_edge
    } else {
        best_face
    };
    // Orient the normal from A toward B.
    let normal = if d.dot(l) < 0.0 { -l } else { l };

    let points = match axis {
        Axis::FaceA(k) => face_contact(a.position, ua, ha, k, normal, b.position, ub, hb),
        Axis::FaceB(k) => face_contact(b.position, ub, hb, k, -normal, a.position, ua, ha),
        Axis::Edge(i, j) => {
            vec![edge_contact(
                a.position,
                ua,
                ha,
                i,
                b.position,
                ub,
                hb,
                j,
                normal,
                penetration,
            )]
        }
    };
    (!points.is_empty()).then_some((normal, points))
}

/// Contact where box R's face `k` (outward normal `n`, pointing at box I)
/// is the reference face. Returns points with positions and penetrations.
#[allow(clippy::too_many_arguments)]
fn face_contact(
    cr: Vec3,
    ur: [Vec3; 3],
    hr: Vec3,
    k: usize,
    n: Vec3,
    ci: Vec3,
    ui: [Vec3; 3],
    hi: Vec3,
) -> Vec<ContactPoint> {
    let face_center = cr + n * hr[k];
    let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);

    // Incident face: the face of box I most opposed to n.
    let m = (0..3)
        .max_by(|&p, &q| ui[p].dot(n).abs().total_cmp(&ui[q].dot(n).abs()))
        .unwrap_or(0);
    let face_normal = if ui[m].dot(n) > 0.0 { -ui[m] } else { ui[m] };
    let ic = ci + face_normal * hi[m];
    let (m1, m2) = ((m + 1) % 3, (m + 2) % 3);
    let (e1, e2) = (ui[m1] * hi[m1], ui[m2] * hi[m2]);
    let mut polygon = vec![ic + e1 + e2, ic - e1 + e2, ic - e1 - e2, ic + e1 - e2];

    // Clip against the four side planes of the reference face.
    for (dir, half) in [
        (ur[k1], hr[k1]),
        (-ur[k1], hr[k1]),
        (ur[k2], hr[k2]),
        (-ur[k2], hr[k2]),
    ] {
        polygon = clip_polygon(&polygon, dir, dir.dot(face_center) + half);
        if polygon.is_empty() {
            return Vec::new();
        }
    }

    let mut points: Vec<ContactPoint> = polygon
        .into_iter()
        .filter_map(|p| {
            let separation = n.dot(p - face_center);
            (separation <= 0.0).then(|| ContactPoint::new(p - n * (0.5 * separation), -separation))
        })
        .collect();
    reduce_to_four(&mut points, n);
    points
}

/// Keep the part of a convex polygon with `dot(n, p) <= offset`.
fn clip_polygon(polygon: &[Vec3], n: Vec3, offset: f32) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(polygon.len() + 1);
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % polygon.len()];
        let (dp, dq) = (n.dot(p) - offset, n.dot(q) - offset);
        if dp <= 0.0 {
            out.push(p);
        }
        if (dp < 0.0) != (dq < 0.0) && dp != dq {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}

/// Clipping a quad can yield up to 8 points; 4 well-spread ones hold a box
/// just as well and keep the solver cheaper. Keep the deepest point, the one
/// farthest from it, and then the two that add the most area.
fn reduce_to_four(points: &mut Vec<ContactPoint>, n: Vec3) {
    if points.len() <= 4 {
        return;
    }
    let mut chosen: Vec<ContactPoint> = Vec::with_capacity(4);
    let deepest = points
        .iter()
        .copied()
        .max_by(|p, q| p.penetration.total_cmp(&q.penetration))
        .expect("non-empty");
    chosen.push(deepest);
    let far = points
        .iter()
        .copied()
        .max_by(|p, q| {
            (p.position - deepest.position)
                .length_squared()
                .total_cmp(&(q.position - deepest.position).length_squared())
        })
        .expect("non-empty");
    chosen.push(far);
    // Signed area of the triangle (a, b, p) about the normal.
    let area = |a: Vec3, b: Vec3, p: Vec3| (b - a).cross(p - a).dot(n);
    let (a, b) = (deepest.position, far.position);
    let most = |sign: f32| {
        points
            .iter()
            .copied()
            .max_by(|p, q| {
                (sign * area(a, b, p.position)).total_cmp(&(sign * area(a, b, q.position)))
            })
            .expect("non-empty")
    };
    for p in [most(1.0), most(-1.0)] {
        if !chosen
            .iter()
            .any(|c| (c.position - p.position).length_squared() < 1e-10)
        {
            chosen.push(p);
        }
    }
    *points = chosen;
}

/// Single contact point between edge `i` of box A and edge `j` of box B.
#[allow(clippy::too_many_arguments)]
fn edge_contact(
    ca: Vec3,
    ua: [Vec3; 3],
    ha: Vec3,
    i: usize,
    cb: Vec3,
    ub: [Vec3; 3],
    hb: Vec3,
    j: usize,
    normal: Vec3,
    penetration: f32,
) -> ContactPoint {
    // The edge of A parallel to ua[i] that lies furthest along +normal, and
    // the edge of B parallel to ub[j] furthest along -normal.
    let edge_point = |c: Vec3, u: [Vec3; 3], h: Vec3, along: usize, dir: Vec3| {
        let mut p = c;
        for k in 0..3 {
            if k != along {
                p += u[k] * h[k] * u[k].dot(dir).signum();
            }
        }
        p
    };
    let pa = edge_point(ca, ua, ha, i, normal);
    let pb = edge_point(cb, ub, hb, j, -normal);
    let (da, db) = (ua[i], ub[j]);

    // Closest points between the lines pa + s·da and pb + t·db, clamped to
    // the edges' lengths.
    let r = pa - pb;
    let (a, e, b) = (da.dot(da), db.dot(db), da.dot(db));
    let (c, f) = (da.dot(r), db.dot(r));
    let denom = a * e - b * b;
    let s = if denom.abs() > 1e-9 {
        ((b * f - c * e) / denom).clamp(-ha[i], ha[i])
    } else {
        0.0
    };
    let t = ((b * s + f) / e).clamp(-hb[j], hb[j]);
    let qa = pa + da * s;
    let qb = pb + db * t;
    ContactPoint::new(0.5 * (qa + qb), penetration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::f32::consts::FRAC_PI_4;

    fn cube(side: f32, p: Vec3) -> RigidBody {
        RigidBody::new(Shape::cube(side), p)
    }

    #[test]
    fn spheres() {
        let a = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::ZERO);
        let b = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 0.9, 0.0));
        let m = collide(0, &a, 1, &b).unwrap();
        assert!((m.normal - Vec3::Y).length() < 1e-5);
        assert!((m.points[0].penetration - 0.1).abs() < 1e-5);
        let far = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::new(0.0, 1.1, 0.0));
        assert!(collide(0, &a, 1, &far).is_none());
    }

    #[test]
    fn sphere_on_box_face_and_corner() {
        let bx = cube(2.0, Vec3::ZERO);
        let s = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::new(0.2, 1.45, -0.3));
        let m = collide(0, &bx, 1, &s).unwrap();
        assert!((m.normal - Vec3::Y).length() < 1e-5);
        assert!((m.points[0].penetration - 0.05).abs() < 1e-5);
        // Diagonally off a corner at (1, 1, 1).
        let diag = Vec3::ONE.normalize();
        let s = RigidBody::new(Shape::Sphere { radius: 0.5 }, Vec3::ONE + diag * 0.45);
        let m = collide(0, &bx, 1, &s).unwrap();
        assert!((m.normal - diag).length() < 1e-4);
        assert!((m.points[0].penetration - 0.05).abs() < 1e-4);
        // Sphere first: normal flips to point from sphere to box.
        let m = collide(0, &s, 1, &bx).unwrap();
        assert!((m.normal + diag).length() < 1e-4);
    }

    #[test]
    fn box_resting_on_floor_gets_four_points() {
        let floor =
            RigidBody::new_static(Shape::cuboid(10.0, 1.0, 10.0), Vec3::new(0.0, -0.5, 0.0));
        let b = cube(1.0, Vec3::new(0.3, 0.48, -0.2)); // 2 cm into the floor
        let m = collide(0, &floor, 1, &b).unwrap();
        assert!((m.normal - Vec3::Y).length() < 1e-5, "normal {}", m.normal);
        assert_eq!(m.points.len(), 4);
        for p in &m.points {
            assert!((p.penetration - 0.02).abs() < 1e-4, "pen {}", p.penetration);
            assert!((p.position.y + 0.01).abs() < 1e-4);
        }
    }

    #[test]
    fn small_box_on_rotated_big_box() {
        // Reference face belongs to whichever box gives the best axis; the
        // normal must still point from A to B.
        let top = cube(1.0, Vec3::new(0.0, 1.0, 0.0));
        let base = cube(1.2, Vec3::ZERO).with_orientation(Quat::from_rotation_y(0.3));
        let m = collide(0, &top, 1, &base).unwrap();
        assert!((m.normal + Vec3::Y).length() < 1e-4, "normal {}", m.normal);
        assert!(m.points.len() >= 3, "{} points", m.points.len());
    }

    #[test]
    fn separated_boxes_dont_collide() {
        let a = cube(1.0, Vec3::ZERO);
        assert!(collide(0, &a, 1, &cube(1.0, Vec3::new(1.01, 0.0, 0.0))).is_none());
        // Rotated so bounding spheres overlap but the boxes don't.
        let b =
            cube(1.0, Vec3::new(1.2, 1.2, 0.0)).with_orientation(Quat::from_rotation_z(FRAC_PI_4));
        assert!(collide(0, &a, 1, &b).is_none());
    }

    #[test]
    fn crossed_edges_give_one_point() {
        // Box B rotated 45° about z and then about x, so one of its edges
        // pokes across the top edge of A: an edge-edge contact.
        let a = cube(1.0, Vec3::ZERO).with_orientation(Quat::from_rotation_x(FRAC_PI_4));
        let top_of_a = 0.5 * 2f32.sqrt();
        let b = cube(
            1.0,
            Vec3::new(0.0, top_of_a + 0.5 * 2f32.sqrt() - 0.02, 0.0),
        )
        .with_orientation(Quat::from_rotation_z(FRAC_PI_4));
        let m = collide(0, &a, 1, &b).unwrap();
        assert_eq!(m.points.len(), 1, "{:?}", m.points);
        assert!((m.normal - Vec3::Y).length() < 1e-3, "normal {}", m.normal);
        assert!((m.points[0].penetration - 0.02).abs() < 2e-3);
        assert!(m.points[0].position.length() - top_of_a < 0.05);
    }

    #[test]
    fn clipping_keeps_inside_part() {
        let square = vec![
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];
        let clipped = clip_polygon(&square, Vec3::X, 0.5);
        assert_eq!(clipped.len(), 4);
        assert!(clipped.iter().all(|p| p.x <= 0.5 + 1e-6));
    }
}
