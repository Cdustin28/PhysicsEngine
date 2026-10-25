//! Built-in example scenes. The editor lists them under "Examples", and
//! `cargo run -p engine --example make_scenes` writes them to `scenes/`.

use crate::body::RigidBody;
use crate::shape::Shape;
use crate::world::World;
use glam::Vec2;

/// A preset: its name and a function that builds it.
pub type Preset = (&'static str, fn() -> World);

/// Every preset, in menu order.
pub const ALL: &[Preset] = &[
    ("demo", demo),
    ("pyramid", pyramid),
    ("ramp", ramp),
    ("bounce", bounce),
    ("dam_break", dam_break),
    ("buoyancy", buoyancy),
    ("pendulums", pendulums),
    ("rope_bridge", rope_bridge),
    ("newtons_cradle", newtons_cradle),
    ("springs", springs),
];

/// A 20 m static floor whose top surface is at y = 0.
fn with_floor() -> World {
    let mut w = World::new();
    w.add_body(RigidBody::new_static(
        Shape::rect(20.0, 0.5),
        Vec2::new(0.0, -0.25),
    ));
    w
}

/// A short stack of boxes, a ball, and a tumbling box.
pub fn demo() -> World {
    let mut w = with_floor();
    for i in 0..3 {
        w.add_body(RigidBody::new(
            Shape::rect(1.0, 1.0),
            Vec2::new(-2.0, 0.5 + i as f32),
        ));
    }
    w.add_body(RigidBody::new(
        Shape::Circle { radius: 0.5 },
        Vec2::new(0.5, 4.0),
    ));
    w.add_body(RigidBody::new(Shape::rect(1.2, 0.6), Vec2::new(2.5, 5.0)).with_angle(0.4));
    w
}

/// A 10-row pyramid of 55 boxes: the classic solver stress test.
pub fn pyramid() -> World {
    let mut w = with_floor();
    let size = 0.5;
    let rows = 10;
    for row in 0..rows {
        let count = rows - row;
        let y = size * (0.5 + row as f32);
        let x0 = -0.5 * size * (count as f32 - 1.0);
        for i in 0..count {
            w.add_body(RigidBody::new(
                Shape::rect(size, size),
                Vec2::new(x0 + size * i as f32, y),
            ));
        }
    }
    w
}

/// Blocks on two ramps: one shallow enough that friction holds the block,
/// one steep enough that it slides.
pub fn ramp() -> World {
    let mut w = with_floor();
    for (x, degrees) in [(-4.0_f32, 20.0_f32), (4.0, 35.0)] {
        let angle = degrees.to_radians();
        let ramp =
            RigidBody::new_static(Shape::rect(6.0, 0.3), Vec2::new(x, 3.0)).with_angle(angle);
        // Sit a 0.5 m block on the ramp's top surface, near its upper end.
        let block_local = Vec2::new(1.5, 0.15 + 0.25);
        let block = RigidBody::new(Shape::rect(0.5, 0.5), ramp.local_to_world(block_local))
            .with_angle(angle);
        w.add_body(ramp);
        w.add_body(block);
    }
    w
}

/// Balls with restitution 0, 0.5, and 0.9 dropped side by side.
pub fn bounce() -> World {
    let mut w = with_floor();
    w.bodies[0].restitution = 0.0;
    for (i, e) in [0.0_f32, 0.5, 0.9].into_iter().enumerate() {
        let mut ball = RigidBody::new(
            Shape::Circle { radius: 0.4 },
            Vec2::new(-2.0 + 2.0 * i as f32, 5.0),
        );
        ball.restitution = e;
        w.add_body(ball);
    }
    w
}

/// A water tank: floor and two walls, `width` m wide inside, walls
/// `wall_height` m tall. The inside floor is at y = 0, centered on x = 0.
pub fn tank(width: f32, wall_height: f32) -> World {
    let mut w = World::new();
    let t = 0.2; // wall thickness
    w.add_body(RigidBody::new_static(
        Shape::rect(width + 2.0 * t, t),
        Vec2::new(0.0, -0.5 * t),
    ));
    for side in [-1.0, 1.0] {
        w.add_body(RigidBody::new_static(
            Shape::rect(t, wall_height),
            Vec2::new(side * 0.5 * (width + t), 0.5 * wall_height),
        ));
    }
    w
}

/// The classic dam break: a 2 m × 2 m column of water released at one end
/// of a 6 m tank (about 1,500 particles at the default spacing).
pub fn dam_break() -> World {
    let mut w = tank(6.0, 3.0);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-3.0, 0.0), Vec2::new(-1.0, 2.0), &bodies);
    w
}

/// Archimedes in action: boxes of density 2.5, 5, and 7.5 (water is 10)
/// float a quarter, half, and three quarters submerged; a density-20 box
/// sinks; a light ball bobs up.
pub fn buoyancy() -> World {
    let mut w = tank(5.0, 2.5);
    let bodies = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-2.5, 0.0), Vec2::new(2.5, 1.2), &bodies);
    for (x, density) in [(-1.8_f32, 2.5_f32), (-0.6, 5.0), (0.6, 7.5), (1.8, 20.0)] {
        let mut b = RigidBody::new(Shape::rect(0.6, 0.4), Vec2::new(x, 1.6));
        b.density = density;
        w.add_body(b);
    }
    // Starts under water and is released.
    let mut ball = RigidBody::new(Shape::Circle { radius: 0.2 }, Vec2::new(0.0, 0.5));
    ball.density = 3.0;
    w.fluid.remove_disc(ball.position, 0.25);
    w.add_body(ball);
    w
}

/// Three pendulums: a ball on a rod, a double pendulum (chaotic), and a
/// hanging chain.
pub fn pendulums() -> World {
    use crate::joint::{Anchor, Joint, add_chain};
    let mut w = with_floor();
    // Simple pendulum: 1.5 m rod, released 45° out.
    let pivot = Vec2::new(-4.0, 5.0);
    let bob = pivot
        + 1.5
            * Vec2::new(
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
            );
    let a = w
        .add_body(RigidBody::new(Shape::Circle { radius: 0.2 }, bob))
        .0;
    w.joints.push(Joint::rod(&w.bodies, a, bob, None, pivot));
    // Double pendulum, released horizontally.
    let pivot = Vec2::new(0.0, 5.0);
    let p1 = pivot + Vec2::new(1.0, 0.0);
    let p2 = p1 + Vec2::new(1.0, 0.0);
    let b1 = w
        .add_body(RigidBody::new(Shape::Circle { radius: 0.15 }, p1))
        .0;
    let b2 = w
        .add_body(RigidBody::new(Shape::Circle { radius: 0.15 }, p2))
        .0;
    w.joints.push(Joint::rod(&w.bodies, b1, p1, None, pivot));
    w.joints.push(Joint::rod(&w.bodies, b2, p2, Some(b1), p1));
    // Chains whip; substeps keep their pins tight (see SolverSettings).
    w.solver.substeps = 4;
    // A 12-link chain hanging from one end, starting horizontal.
    add_chain(
        &mut w,
        Vec2::new(3.0, 5.0),
        Vec2::new(6.0, 5.0),
        12,
        0.08,
        Anchor::World,
        Anchor::Free,
    );
    w
}

/// A sagging bridge of planks between two towers, with crates dropped on it.
pub fn rope_bridge() -> World {
    use crate::joint::{Anchor, add_chain};
    let mut w = with_floor();
    w.solver.substeps = 4;
    for x in [-4.5, 4.5] {
        w.add_body(RigidBody::new_static(
            Shape::rect(1.0, 3.0),
            Vec2::new(x, 1.5),
        ));
    }
    // Built as a shallow V so it has slack: 2 × √(4² + 0.8²) = 8.16 m of
    // planks across the 8 m gap. It settles into a sag.
    let middle = Vec2::new(0.0, 2.2);
    let left = add_chain(
        &mut w,
        Vec2::new(-4.0, 3.0),
        middle,
        8,
        0.12,
        Anchor::World,
        Anchor::Free,
    );
    add_chain(
        &mut w,
        middle,
        Vec2::new(4.0, 3.0),
        8,
        0.12,
        Anchor::Body(left[7]),
        Anchor::World,
    );
    for (i, x) in [-1.5, 0.0, 1.5].into_iter().enumerate() {
        let mut crate_ = RigidBody::new(Shape::rect(0.6, 0.6), Vec2::new(x, 5.0 + i as f32));
        crate_.density = 2.0;
        w.add_body(crate_);
    }
    w
}

/// Five steel balls on strings, touching: pull one back and let go.
pub fn newtons_cradle() -> World {
    use crate::joint::Joint;
    let mut w = with_floor();
    let r = 0.25;
    let length = 2.0;
    for i in 0..5 {
        // A hair apart (1 mm) so each collision is a separate event.
        let x = (i as f32 - 2.0) * (2.0 * r + 0.001);
        let pivot = Vec2::new(x, 4.0);
        let rest = pivot - Vec2::new(0.0, length);
        // The first ball starts pulled out 40°.
        let pos = if i == 0 {
            pivot + Vec2::new(-length * 0.6428, -length * 0.7660)
        } else {
            rest
        };
        let mut ball = RigidBody::new(Shape::Circle { radius: r }, pos);
        ball.restitution = 1.0;
        ball.friction = 0.0;
        ball.density = 78.0; // steel in a 1 cm slab
        let id = w.add_body(ball).0;
        w.joints.push(Joint::rod(&w.bodies, id, pos, None, pivot));
    }
    w.solver.restitution_threshold = 0.05;
    w
}

/// Masses on springs: soft and stiff, damped and undamped.
pub fn springs() -> World {
    use crate::joint::Joint;
    let mut w = with_floor();
    for (i, (stiffness, damping)) in [(20.0, 0.0), (60.0, 0.0), (60.0, 3.0)]
        .into_iter()
        .enumerate()
    {
        let x = -3.0 + 3.0 * i as f32;
        let top = Vec2::new(x, 6.0);
        let start = Vec2::new(x, 4.0);
        let mut block = RigidBody::new(Shape::rect(0.5, 0.5), start);
        block.density = 4.0; // 1 kg
        let id = w.add_body(block).0;
        let anchor = start + Vec2::new(0.0, 0.25);
        w.joints.push(Joint::spring(
            &w.bodies, id, anchor, None, top, stiffness, damping,
        ));
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid() {
        for (name, make) in ALL {
            let w = make();
            for (i, b) in w.bodies.iter().enumerate() {
                b.validate()
                    .unwrap_or_else(|e| panic!("preset {name}, body {i}: {e}"));
            }
        }
    }
}
