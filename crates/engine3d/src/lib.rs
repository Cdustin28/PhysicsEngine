//! 3D rigid-body physics for physicsengine.dev.
//!
//! The same design as the 2D `engine` crate, a dimension up: spheres and
//! boxes, sort-and-sweep broad phase, SAT-based contact manifolds, and a
//! sequential-impulse solver with friction, restitution, warm starting, and
//! split-impulse position correction; plus SPH water that pushes bodies and
//! is pushed by them. No GUI or graphics dependencies.
//!
//! # Units and frame
//!
//! SI throughout, and in 3D that means real densities: kg/m³ (water is
//! 1000). +y is up; the world is right-handed.
//!
//! # Module map
//!
//! - [`shape`], [`body`]: geometry, mass properties, rigid-body state
//! - [`broad_phase`], [`collision`]: which pairs touch, and where
//! - [`solver`]: contact impulses
//! - [`world`]: owns the bodies and runs a step
//! - [`fluid`]: SPH water and its coupling with bodies
//! - [`ray`]: ray casts for picking
//! - [`scene`], [`presets`]: JSON files and built-in examples

pub mod body;
pub mod broad_phase;
pub mod collision;
pub mod fluid;
pub mod presets;
pub mod ray;
pub mod scene;
pub mod shape;
pub mod solver;
pub mod world;

pub use body::{BodyHandle, BodyId, RigidBody};
pub use fluid::{CpuSolver, Fluid, FluidParams, FluidSolver};
pub use glam::{Quat, Vec3};
pub use scene::{Scene, SceneError};
pub use shape::Shape;
pub use solver::SolverSettings;
pub use world::World;

/// Standard gravity, m/s².
pub const STANDARD_GRAVITY: f32 = 9.806_65;
