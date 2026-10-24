//! Core physics for physicsengine.dev.
//!
//! This crate has no graphics or GUI dependencies so it can be unit-tested,
//! benchmarked, and later reused by a 3D or GPU backend.
//!
//! # Units
//!
//! Everything is SI: meters, kilograms, seconds, radians. Because the world is
//! 2D, `density` is an *areal* density in kg/m² (mass per unit area). The
//! editor converts meters to pixels only when drawing.
//!
//! # Coordinate frame
//!
//! +x points right, +y points up, positive angles rotate counter-clockwise.

pub mod body;
pub mod scene;
pub mod shape;
pub mod world;

pub use body::{BodyId, RigidBody};
pub use glam::Vec2;
pub use scene::{Scene, SceneError};
pub use shape::Shape;
pub use world::{FixedTimestep, World};

/// Standard gravity at Earth's surface, m/s².
pub const STANDARD_GRAVITY: f32 = 9.806_65;
