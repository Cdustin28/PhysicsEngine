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
//!
//! # Module map
//!
//! - [`shape`], [`body`]: geometry and rigid-body state
//! - [`broad_phase`]: which pairs *might* touch (sort-and-sweep)
//! - [`collision`]: exactly where they touch (contact manifolds)
//! - [`solver`]: sequential impulses for contacts, friction, and restitution
//! - [`fluid`]: SPH particle fluid and its boundary handling
//! - [`world`]: owns the bodies and fluid and runs a step
//! - [`scene`], [`presets`]: JSON scene files and built-in examples

pub mod body;
pub mod broad_phase;
pub mod collision;
pub mod fluid;
pub mod gas;
pub mod joint;
pub mod presets;
pub mod scene;
pub mod shape;
pub mod soft;
pub mod solver;
pub mod world;

pub use body::{BodyHandle, BodyId, RigidBody};
pub use collision::{ContactPoint, Manifold};
pub use fluid::{BodyImpulse, CpuSolver, Fluid, FluidParams, FluidSolver};
pub use gas::{Emitter, Gas, GasParams};
pub use glam::Vec2;
pub use joint::{Joint, JointKind};
pub use scene::{Scene, SceneError};
pub use shape::Shape;
pub use soft::{SoftBody, SoftSettings};
pub use solver::SolverSettings;
pub use world::{FixedTimestep, World};

/// Standard gravity at Earth's surface, m/s².
pub const STANDARD_GRAVITY: f32 = 9.806_65;
