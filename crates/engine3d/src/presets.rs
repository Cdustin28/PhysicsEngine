//! Built-in 3D example scenes.

use crate::body::RigidBody;
use crate::shape::Shape;
use crate::world::World;
use glam::{Quat, Vec3};

pub type Preset = (&'static str, fn() -> World);

/// Every preset, in menu order.
pub const ALL: &[Preset] = &[
    ("demo", demo),
    ("pyramid", pyramid),
    ("ramp", ramp),
    ("rolling", rolling),
    ("bounce", bounce),
    ("dam_break", dam_break),
    ("buoyancy", buoyancy),
    ("pendulums", pendulums),
    ("newtons_cradle", newtons_cradle),
    ("chain_drape", chain_drape),
    ("soft_bodies", soft_bodies),
];

/// A 20 m × 20 m static floor whose top is at y = 0.
pub fn with_floor() -> World {
    let mut w = World::new();
    w.add_body(RigidBody::new_static(
        Shape::cuboid(20.0, 0.5, 20.0),
        Vec3::new(0.0, -0.25, 0.0),
    ));
    w
}

/// An open-topped glass tank on the floor: inner size `width` (x) ×
/// `depth` (z), walls `height` tall and 0.2 m thick, centered on the origin.
pub fn tank(width: f32, depth: f32, height: f32) -> World {
    let mut w = with_floor();
    w.bodies[0].friction = 0.5;
    let t = 0.2;
    let (hx, hz) = (0.5 * width, 0.5 * depth);
    for sign in [-1.0, 1.0] {
        w.add_body(RigidBody::new_static(
            Shape::cuboid(t, height, depth + 2.0 * t),
            Vec3::new(sign * (hx + 0.5 * t), 0.5 * height, 0.0),
        ));
        w.add_body(RigidBody::new_static(
            Shape::cuboid(width, height, t),
            Vec3::new(0.0, 0.5 * height, sign * (hz + 0.5 * t)),
        ));
    }
    w
}

/// A column of water collapsing across a tank.
pub fn dam_break() -> World {
    let mut w = tank(3.0, 1.2, 1.2);
    let bodies = w.bodies.clone();
    w.fluid.fill_box(
        Vec3::new(-1.5, 0.0, -0.6),
        Vec3::new(-0.6, 1.0, 0.6),
        &bodies,
    );
    w
}

/// Water with bodies of different densities: pine (500 kg/m³) floats half
/// in, cork (250) floats high, a ball of 600 floats low, and steel-ish
/// (2000) sinks.
pub fn buoyancy() -> World {
    let mut w = tank(2.0, 2.0, 1.0);
    let bodies = w.bodies.clone();
    w.fluid.fill_box(
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.6, 1.0),
        &bodies,
    );
    for (x, z, density) in [(-0.5, -0.5, 500.0), (0.5, -0.5, 250.0), (-0.5, 0.5, 2000.0)] {
        let mut b = RigidBody::new(Shape::cube(0.4), Vec3::new(x, 1.2, z));
        b.density = density;
        w.add_body(b);
    }
    let mut ball = RigidBody::new(Shape::Sphere { radius: 0.25 }, Vec3::new(0.5, 1.2, 0.5));
    ball.density = 600.0;
    w.add_body(ball);
    w
}

/// A short tower of cubes, two balls, and a tumbling box.
pub fn demo() -> World {
    let mut w = with_floor();
    for i in 0..4 {
        w.add_body(RigidBody::new(
            Shape::cube(1.0),
            Vec3::new(-1.5, 0.5 + i as f32, 0.0),
        ));
    }
    w.add_body(RigidBody::new(
        Shape::Sphere { radius: 0.5 },
        Vec3::new(1.0, 3.0, 0.5),
    ));
    w.add_body(RigidBody::new(
        Shape::Sphere { radius: 0.3 },
        Vec3::new(1.3, 5.0, -0.8),
    ));
    w.add_body(
        RigidBody::new(Shape::cuboid(1.4, 0.4, 0.8), Vec3::new(0.5, 6.0, 1.5))
            .with_orientation(Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.7, 0.2)),
    );
    w
}

/// A square pyramid of 30 cubes (4 × 4, 3 × 3, 2 × 2, 1).
pub fn pyramid() -> World {
    let mut w = with_floor();
    let size = 0.5;
    for level in 0..4 {
        let n = 4 - level;
        let offset = -0.5 * size * (n as f32 - 1.0);
        for i in 0..n {
            for j in 0..n {
                w.add_body(RigidBody::new(
                    Shape::cube(size),
                    Vec3::new(
                        offset + size * i as f32,
                        size * (0.5 + level as f32),
                        offset + size * j as f32,
                    ),
                ));
            }
        }
    }
    w
}

/// A static ramp tilted `degrees` about z, `x` along the floor, with its top
/// surface's center at height `y`.
fn ramp_at(w: &mut World, x: f32, y: f32, degrees: f32) -> RigidBody {
    let r = RigidBody::new_static(Shape::cuboid(6.0, 0.3, 2.0), Vec3::new(x, y, 0.0))
        .with_orientation(Quat::from_rotation_z(degrees.to_radians()));
    w.add_body(r.clone());
    r
}

/// Blocks on a 20° ramp (friction holds) and a 35° ramp (it slides).
pub fn ramp() -> World {
    let mut w = with_floor();
    for (x, degrees) in [(-4.0_f32, 20.0_f32), (4.0, 35.0)] {
        let r = ramp_at(&mut w, x, 3.0, degrees);
        let block = RigidBody::new(
            Shape::cube(0.5),
            r.local_to_world(Vec3::new(1.5, 0.15 + 0.25, 0.0)),
        )
        .with_orientation(r.orientation);
        w.add_body(block);
    }
    w
}

/// A ball and a box released side by side on the same 20° ramp: the ball
/// rolls (acceleration 5/7 g sin θ) while the box slides.
pub fn rolling() -> World {
    let mut w = with_floor();
    let r = ramp_at(&mut w, 0.0, 3.0, 20.0);
    let mut ball = RigidBody::new(
        Shape::Sphere { radius: 0.3 },
        r.local_to_world(Vec3::new(2.0, 0.15 + 0.3, -0.5)),
    );
    ball.friction = 1.0;
    w.add_body(ball);
    let mut block = RigidBody::new(
        Shape::cube(0.5),
        r.local_to_world(Vec3::new(2.0, 0.15 + 0.25, 0.5)),
    )
    .with_orientation(r.orientation);
    block.friction = 0.2;
    w.add_body(block);
    w
}

/// Balls with restitution 0, 0.5, and 0.9 dropped side by side.
pub fn bounce() -> World {
    let mut w = with_floor();
    w.bodies[0].restitution = 0.0;
    for (i, e) in [0.0_f32, 0.5, 0.9].into_iter().enumerate() {
        let mut ball = RigidBody::new(
            Shape::Sphere { radius: 0.4 },
            Vec3::new(-2.0 + 2.0 * i as f32, 5.0, 0.0),
        );
        ball.restitution = e;
        w.add_body(ball);
    }
    w
}

/// A ball on a rod swinging in a cone, a double pendulum, and a hanging
/// chain.
pub fn pendulums() -> World {
    use crate::joint::{Anchor, Joint, add_chain};
    let mut w = with_floor();
    w.solver.substeps = 4;
    // Conical pendulum: 1.5 m rod, 40° out, pushed sideways.
    let pivot = Vec3::new(-3.0, 4.0, 0.0);
    let a = 40f32.to_radians();
    let bob = pivot + 1.5 * Vec3::new(a.sin(), -a.cos(), 0.0);
    let mut ball = RigidBody::new(Shape::Sphere { radius: 0.2 }, bob);
    // Speed for a steady cone: v = √(g L sin θ tan θ).
    ball.velocity = Vec3::new(
        0.0,
        0.0,
        (crate::STANDARD_GRAVITY * 1.5 * a.sin() * a.tan()).sqrt(),
    );
    let id = w.add_body(ball).0;
    w.joints.push(Joint::rod(&w.bodies, id, bob, None, pivot));
    // Double pendulum, released horizontally.
    let pivot = Vec3::new(0.0, 4.0, 0.0);
    let (p1, p2) = (pivot + Vec3::X, pivot + 2.0 * Vec3::X);
    let b1 = w
        .add_body(RigidBody::new(Shape::Sphere { radius: 0.15 }, p1))
        .0;
    let b2 = w
        .add_body(RigidBody::new(Shape::Sphere { radius: 0.15 }, p2))
        .0;
    w.joints.push(Joint::rod(&w.bodies, b1, p1, None, pivot));
    w.joints.push(Joint::rod(&w.bodies, b2, p2, Some(b1), p1));
    // A chain released horizontally.
    add_chain(
        &mut w,
        Vec3::new(3.0, 4.0, 0.0),
        Vec3::new(3.0, 4.0, 3.0),
        12,
        0.08,
        Anchor::World,
        Anchor::Free,
    );
    w
}

/// Five steel balls on rods, the first pulled back 40°.
pub fn newtons_cradle() -> World {
    use crate::joint::Joint;
    let mut w = with_floor();
    let (r, length) = (0.25, 2.0);
    for i in 0..5 {
        let x = (i as f32 - 2.0) * (2.0 * r + 0.001);
        let pivot = Vec3::new(x, 4.0, 0.0);
        let pos = if i == 0 {
            pivot + length * Vec3::new(-40f32.to_radians().sin(), -40f32.to_radians().cos(), 0.0)
        } else {
            pivot - Vec3::new(0.0, length, 0.0)
        };
        let mut ball = RigidBody::new(Shape::Sphere { radius: r }, pos);
        ball.restitution = 1.0;
        ball.friction = 0.0;
        ball.density = 7800.0;
        let id = w.add_body(ball).0;
        w.joints.push(Joint::rod(&w.bodies, id, pos, None, pivot));
    }
    w.solver.restitution_threshold = 0.05;
    w
}

/// A chain falling across a ball and draping over it.
pub fn chain_drape() -> World {
    use crate::joint::{Anchor, add_chain};
    let mut w = with_floor();
    w.solver.substeps = 4;
    w.add_body(RigidBody::new_static(
        Shape::Sphere { radius: 1.0 },
        Vec3::new(0.0, 1.0, 0.0),
    ));
    add_chain(
        &mut w,
        Vec3::new(-2.5, 2.6, 0.2),
        Vec3::new(2.5, 2.6, -0.2),
        24,
        0.08,
        Anchor::Free,
        Anchor::Free,
    );
    w
}

/// A cloth falling onto a ball, a bouncy ball, and a jelly cube.
pub fn soft_bodies() -> World {
    use crate::soft::SoftBody;
    let mut w = with_floor();
    w.add_body(RigidBody::new_static(
        Shape::Sphere { radius: 0.7 },
        Vec3::new(-1.5, 0.7, 0.0),
    ));
    w.soft_bodies.push(SoftBody::cloth(
        Vec3::new(-1.5, 2.2, 0.0),
        2.4,
        2.4,
        20,
        20,
        0.5,
    ));
    let mut ball = SoftBody::ball(Vec3::new(1.2, 2.0, -0.8), 0.45, 2, 200.0);
    ball.edge_stiffness = 500.0;
    ball.volume_stiffness = 50_000.0;
    w.soft_bodies.push(ball);
    w.soft_bodies.push(SoftBody::jelly(
        Vec3::new(1.2, 1.0, 1.0),
        Vec3::splat(0.6),
        5,
        1000.0,
    ));
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        for (name, make) in ALL {
            for (i, b) in make().bodies.iter().enumerate() {
                b.validate()
                    .unwrap_or_else(|e| panic!("{name} body {i}: {e}"));
            }
        }
    }
}
