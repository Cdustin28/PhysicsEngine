//! Narrow-phase collision detection: given two bodies whose bounding boxes
//! overlap, find out whether they actually touch and where.
//!
//! The result is a [`Manifold`]: a contact normal plus one or two contact
//! points. Boxes resting flat on each other need *two* points. With only one,
//! the solver can't stop them rocking back and forth.
//!
//! # Conventions
//!
//! - The normal is a unit vector pointing **from body A toward body B**.
//! - `penetration` is how far the shapes overlap along the normal, m (> 0).
//! - Each contact point sits halfway between the two surfaces.
//! - Each point carries a `feature` id describing which edge/vertex pair
//!   produced it. If the same pair touches again next step with the same
//!   feature, it is "the same" contact, and the solver reuses last step's
//!   impulse as a starting guess (warm starting).

use crate::body::RigidBody;
use crate::shape::Shape;
use glam::Vec2;

/// Up to two points where body A and body B touch.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifold {
    /// Index of body A in `World::bodies`. Always `a < b`.
    pub a: usize,
    /// Index of body B in `World::bodies`.
    pub b: usize,
    /// Unit normal pointing from A toward B.
    pub normal: Vec2,
    pub points: Vec<ContactPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactPoint {
    /// World-space contact position, m.
    pub position: Vec2,
    /// Overlap depth along the normal, m.
    pub penetration: f32,
    /// Identifies the geometric feature pair that produced this point.
    pub feature: u32,
    /// Accumulated normal impulse from the solver, N·s. Persisted across
    /// steps for warm starting.
    pub normal_impulse: f32,
    /// Accumulated friction impulse from the solver, N·s.
    pub tangent_impulse: f32,
}

impl ContactPoint {
    fn new(position: Vec2, penetration: f32, feature: u32) -> Self {
        Self {
            position,
            penetration,
            feature,
            normal_impulse: 0.0,
            tangent_impulse: 0.0,
        }
    }
}

/// Test bodies `a` and `b` (indices `ia < ib`) for contact.
pub fn collide(ia: usize, a: &RigidBody, ib: usize, b: &RigidBody) -> Option<Manifold> {
    debug_assert!(ia < ib);
    let (normal, points) = match (a.shape, b.shape) {
        (Shape::Circle { radius: ra }, Shape::Circle { radius: rb }) => {
            circle_circle(a.position, ra, b.position, rb)?
        }
        (Shape::Box { .. }, Shape::Circle { radius }) => box_circle(a, b.position, radius)?,
        (Shape::Circle { radius }, Shape::Box { .. }) => {
            // Same test with the roles swapped, then flip the normal so it
            // still points from A (the circle) to B (the box).
            let (n, pts) = box_circle(b, a.position, radius)?;
            (-n, pts)
        }
        (Shape::Box { .. }, Shape::Box { .. }) => box_box(a, b)?,
    };
    Some(Manifold {
        a: ia,
        b: ib,
        normal,
        points,
    })
}

fn circle_circle(ca: Vec2, ra: f32, cb: Vec2, rb: f32) -> Option<(Vec2, Vec<ContactPoint>)> {
    let d = cb - ca;
    let dist_sq = d.length_squared();
    let r = ra + rb;
    if dist_sq > r * r {
        return None;
    }
    let dist = dist_sq.sqrt();
    // Perfectly concentric circles have no meaningful direction; push up.
    let normal = if dist > 1e-6 { d / dist } else { Vec2::Y };
    let penetration = r - dist;
    let position = ca + normal * (ra - 0.5 * penetration);
    Some((normal, vec![ContactPoint::new(position, penetration, 0)]))
}

/// Box (as A) against a circle with center `c` and radius `r` (as B).
fn box_circle(bx: &RigidBody, c: Vec2, r: f32) -> Option<(Vec2, Vec<ContactPoint>)> {
    let Shape::Box { half_extents: h } = bx.shape else {
        unreachable!("box_circle called with a non-box");
    };
    let local = bx.world_to_local(c);
    let clamped = local.clamp(-h, h);

    // Closest point on the box *surface* and the outward normal there,
    // both in the box's local frame.
    let (surface, local_normal) = if clamped != local {
        // Center is outside the box: the closest point is the clamped one.
        let delta = local - clamped;
        let dist_sq = delta.length_squared();
        if dist_sq > r * r {
            return None;
        }
        (clamped, delta / dist_sq.sqrt())
    } else {
        // Center is inside the box: push out through the nearest face.
        let to_x = h.x - local.x.abs();
        let to_y = h.y - local.y.abs();
        if to_x < to_y {
            let s = local.x.signum();
            (Vec2::new(s * h.x, local.y), Vec2::new(s, 0.0))
        } else {
            let s = local.y.signum();
            (Vec2::new(local.x, s * h.y), Vec2::new(0.0, s))
        }
    };

    let normal = bx.rotation().rotate(local_normal);
    let box_point = bx.local_to_world(surface);
    let circle_point = c - normal * r;
    // Positive when the circle's surface has crossed the box's surface.
    let penetration = (box_point - circle_point).dot(normal);
    if penetration < 0.0 {
        return None;
    }
    let position = 0.5 * (box_point + circle_point);
    Some((normal, vec![ContactPoint::new(position, penetration, 0)]))
}

/// A convex polygon in world space with counter-clockwise vertices.
struct Polygon {
    vertices: [Vec2; 4],
    /// `normals[i]` is the outward normal of the edge from vertex `i` to
    /// vertex `i + 1`.
    normals: [Vec2; 4],
}

impl Polygon {
    fn from_box(body: &RigidBody) -> Self {
        let vertices = body.box_corners().expect("from_box called with a non-box");
        // Local outward normals of the edges bottom, right, top, left.
        let rot = body.rotation();
        let normals = [
            Vec2::new(0.0, -1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
        ]
        .map(|n| rot.rotate(n));
        Self { vertices, normals }
    }

    /// Of all of `self`'s edge normals, the one along which `other` is
    /// furthest outside `self`. Returns (edge index, separation). A positive
    /// separation means a gap exists, so by the Separating Axis Theorem the
    /// polygons don't touch.
    fn max_separation(&self, other: &Polygon) -> (usize, f32) {
        let mut best = (0, f32::NEG_INFINITY);
        for (i, (&n, &v)) in self.normals.iter().zip(&self.vertices).enumerate() {
            // How deep is the deepest vertex of `other` behind this edge?
            let sep = other
                .vertices
                .iter()
                .map(|&w| n.dot(w - v))
                .fold(f32::INFINITY, f32::min);
            if sep > best.1 {
                best = (i, sep);
            }
        }
        best
    }
}

/// A point produced while clipping, tagged with the feature it came from.
#[derive(Clone, Copy)]
struct ClipVertex {
    v: Vec2,
    id: u32,
}

/// Keep the parts of segment `points` with `dot(n, p) <= offset`. If the
/// segment crosses the plane, the crossing point replaces the clipped end and
/// gets `clip_id` as its feature id.
fn clip_segment(points: [ClipVertex; 2], n: Vec2, offset: f32, clip_id: u32) -> Vec<ClipVertex> {
    let mut out = Vec::with_capacity(2);
    let d0 = n.dot(points[0].v) - offset;
    let d1 = n.dot(points[1].v) - offset;
    if d0 <= 0.0 {
        out.push(points[0]);
    }
    if d1 <= 0.0 {
        out.push(points[1]);
    }
    if d0 * d1 < 0.0 {
        let t = d0 / (d0 - d1);
        out.push(ClipVertex {
            v: points[0].v + t * (points[1].v - points[0].v),
            id: clip_id,
        });
    }
    out
}

/// Box vs box using the Separating Axis Theorem, then reference-face
/// clipping to build a manifold with up to two points.
///
/// This follows the approach in Box2D (`b2CollidePolygons`):
/// 1. Find the axis of least penetration among both boxes' face normals.
///    The box owning it is the *reference*, and that face is the reference face.
/// 2. On the other (*incident*) box, find the edge facing most directly
///    against the reference face.
/// 3. Clip that incident edge to the side planes of the reference face.
/// 4. Keep the clipped points that are behind the reference face.
fn box_box(a: &RigidBody, b: &RigidBody) -> Option<(Vec2, Vec<ContactPoint>)> {
    let pa = Polygon::from_box(a);
    let pb = Polygon::from_box(b);

    let (edge_a, sep_a) = pa.max_separation(&pb);
    if sep_a > 0.0 {
        return None;
    }
    let (edge_b, sep_b) = pb.max_separation(&pa);
    if sep_b > 0.0 {
        return None;
    }

    // Prefer A as the reference unless B is clearly better. The tolerance
    // stops the choice from flip-flopping between steps when the two are
    // nearly equal, which would break warm starting and cause jitter.
    const RELATIVE_TOL: f32 = 0.98;
    const ABSOLUTE_TOL: f32 = 0.001;
    let flip = sep_b > RELATIVE_TOL * sep_a + ABSOLUTE_TOL;
    let (reference, incident, ref_edge) = if flip {
        (&pb, &pa, edge_b)
    } else {
        (&pa, &pb, edge_a)
    };

    // Incident edge: the one whose normal is most anti-parallel to the
    // reference normal.
    let ref_normal = reference.normals[ref_edge];
    let inc_edge = (0..4)
        .min_by(|&i, &j| {
            let di = incident.normals[i].dot(ref_normal);
            let dj = incident.normals[j].dot(ref_normal);
            di.total_cmp(&dj)
        })
        .unwrap_or(0);
    let inc_next = (inc_edge + 1) % 4;
    let incident_segment = [
        ClipVertex {
            v: incident.vertices[inc_edge],
            id: inc_edge as u32,
        },
        ClipVertex {
            v: incident.vertices[inc_next],
            id: inc_next as u32,
        },
    ];

    // Reference face endpoints and tangent.
    let v1 = reference.vertices[ref_edge];
    let v2 = reference.vertices[(ref_edge + 1) % 4];
    let tangent = (v2 - v1).normalize();

    // Clip against the side plane at v1 (keep dot(-t, p) <= dot(-t, v1)),
    // then against the side plane at v2 (keep dot(t, p) <= dot(t, v2)).
    let clipped = clip_segment(incident_segment, -tangent, -tangent.dot(v1), 4);
    let [c0, c1] = clipped[..] else {
        return None; // numerically degenerate (edge-on contact)
    };
    let clipped = clip_segment([c0, c1], tangent, tangent.dot(v2), 5);
    if clipped.len() < 2 {
        return None;
    }

    let face_offset = ref_normal.dot(v1);
    let points: Vec<ContactPoint> = clipped
        .iter()
        .filter_map(|cv| {
            let separation = ref_normal.dot(cv.v) - face_offset;
            (separation <= 0.0).then(|| {
                // cv.v is on the incident box. Move halfway back to the
                // reference face so the point sits between the surfaces.
                let position = cv.v - ref_normal * (0.5 * separation);
                let feature = (u32::from(flip) << 16) | ((ref_edge as u32) << 8) | cv.id;
                ContactPoint::new(position, -separation, feature)
            })
        })
        .collect();

    if points.is_empty() {
        return None;
    }
    let normal = if flip { -ref_normal } else { ref_normal };
    Some((normal, points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    fn circle(r: f32, x: f32, y: f32) -> RigidBody {
        RigidBody::new(Shape::Circle { radius: r }, Vec2::new(x, y))
    }

    fn boxed(w: f32, h: f32, x: f32, y: f32) -> RigidBody {
        RigidBody::new(Shape::rect(w, h), Vec2::new(x, y))
    }

    fn approx(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn circles_apart_do_not_collide() {
        assert!(collide(0, &circle(0.5, 0.0, 0.0), 1, &circle(0.5, 1.01, 0.0)).is_none());
    }

    #[test]
    fn circles_overlapping() {
        let m = collide(0, &circle(0.5, 0.0, 0.0), 1, &circle(0.5, 0.9, 0.0)).unwrap();
        assert!(approx(m.normal, Vec2::X));
        assert_eq!(m.points.len(), 1);
        assert!((m.points[0].penetration - 0.1).abs() < 1e-5);
        assert!(approx(m.points[0].position, Vec2::new(0.45, 0.0)));
    }

    #[test]
    fn circle_resting_on_box_face() {
        // 2 x 1 box centered at the origin; circle of radius 0.5 sinking 0.05
        // into its top face.
        let bx = boxed(2.0, 1.0, 0.0, 0.0);
        let c = circle(0.5, 0.3, 0.95);
        let m = collide(0, &bx, 1, &c).unwrap();
        assert!(approx(m.normal, Vec2::Y), "normal {}", m.normal);
        assert!((m.points[0].penetration - 0.05).abs() < 1e-5);
    }

    #[test]
    fn circle_first_flips_normal() {
        let c = circle(0.5, 0.3, 0.95);
        let bx = boxed(2.0, 1.0, 0.0, 0.0);
        let m = collide(0, &c, 1, &bx).unwrap();
        // Normal points from A (circle) to B (box), i.e. downward.
        assert!(approx(m.normal, -Vec2::Y));
    }

    #[test]
    fn circle_near_box_corner() {
        let bx = boxed(2.0, 2.0, 0.0, 0.0);
        // Corner is at (1, 1). The circle center is diagonally out from it.
        let d = 0.45 / 2f32.sqrt();
        let m = collide(0, &bx, 1, &circle(0.5, 1.0 + d, 1.0 + d)).unwrap();
        let diag = Vec2::ONE.normalize();
        assert!(approx(m.normal, diag));
        assert!((m.points[0].penetration - 0.05).abs() < 1e-4);
        // A little further away: no contact.
        let d = 0.55 / 2f32.sqrt();
        assert!(collide(0, &bx, 1, &circle(0.5, 1.0 + d, 1.0 + d)).is_none());
    }

    #[test]
    fn circle_center_inside_box() {
        let bx = boxed(4.0, 2.0, 0.0, 0.0);
        // Center inside, nearest face is the right one (x = 2).
        let m = collide(0, &bx, 1, &circle(0.25, 1.9, 0.0)).unwrap();
        assert!(approx(m.normal, Vec2::X));
        assert!((m.points[0].penetration - 0.35).abs() < 1e-5);
    }

    #[test]
    fn circle_against_rotated_box() {
        // A 2 x 2 box rotated 45°: its top corner is at (0, √2).
        let bx = boxed(2.0, 2.0, 0.0, 0.0).with_angle(FRAC_PI_4);
        let top = 2f32.sqrt();
        let m = collide(0, &bx, 1, &circle(0.5, 0.0, top + 0.45)).unwrap();
        assert!(approx(m.normal, Vec2::Y), "normal {}", m.normal);
        assert!((m.points[0].penetration - 0.05).abs() < 1e-4);
    }

    #[test]
    fn stacked_boxes_get_two_contact_points() {
        let floor = boxed(10.0, 1.0, 0.0, 0.0); // top face at y = 0.5
        let top = boxed(1.0, 1.0, 0.2, 0.98); // bottom face at y = 0.48
        let m = collide(0, &floor, 1, &top).unwrap();
        assert!(approx(m.normal, Vec2::Y), "normal {}", m.normal);
        assert_eq!(m.points.len(), 2);
        for p in &m.points {
            assert!((p.penetration - 0.02).abs() < 1e-4, "pen {}", p.penetration);
            assert!((p.position.y - 0.49).abs() < 1e-4);
        }
        let mut xs: Vec<f32> = m.points.iter().map(|p| p.position.x).collect();
        xs.sort_by(f32::total_cmp);
        assert!(
            (xs[0] + 0.3).abs() < 1e-4 && (xs[1] - 0.7).abs() < 1e-4,
            "{xs:?}"
        );
        // Distinct feature ids so warm starting can tell them apart.
        assert_ne!(m.points[0].feature, m.points[1].feature);
    }

    #[test]
    fn boxes_apart_do_not_collide() {
        let a = boxed(1.0, 1.0, 0.0, 0.0);
        assert!(collide(0, &a, 1, &boxed(1.0, 1.0, 1.01, 0.0)).is_none());
        // Rotated box whose bounding circle overlaps but which doesn't touch.
        let b = boxed(1.0, 1.0, 1.15, 1.15).with_angle(FRAC_PI_4);
        assert!(collide(0, &a, 1, &b).is_none());
    }

    #[test]
    fn corner_into_face_gives_one_point() {
        // A diamond (box rotated 45°) poking its bottom corner 0.05 into a
        // floor whose top is at y = 0.
        let floor = boxed(10.0, 1.0, 0.0, -0.5);
        let half_diag = 0.5 * 2f32.sqrt();
        let diamond = boxed(1.0, 1.0, 0.0, half_diag - 0.05).with_angle(FRAC_PI_4);
        let m = collide(0, &floor, 1, &diamond).unwrap();
        assert!(approx(m.normal, Vec2::Y), "normal {}", m.normal);
        assert_eq!(m.points.len(), 1);
        assert!((m.points[0].penetration - 0.05).abs() < 1e-4);
        assert!(m.points[0].position.x.abs() < 1e-4);
    }

    #[test]
    fn box_box_normal_points_from_a_to_b() {
        // B sits below A, so the normal must point down regardless of which
        // box ends up as the reference.
        let upper = boxed(1.0, 1.0, 0.0, 0.0);
        let lower = boxed(10.0, 1.0, 0.0, -0.98);
        let m = collide(0, &upper, 1, &lower).unwrap();
        assert!(approx(m.normal, -Vec2::Y), "normal {}", m.normal);
        assert_eq!(m.points.len(), 2);
    }
}
