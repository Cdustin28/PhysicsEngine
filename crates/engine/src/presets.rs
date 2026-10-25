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
