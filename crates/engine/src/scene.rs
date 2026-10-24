//! Saving and loading worlds as JSON scene files.
//!
//! A scene file looks like:
//!
//! ```json
//! {
//!   "format": "physicsengine-scene",
//!   "version": 2,
//!   "world": {
//!     "gravity": [0.0, -9.80665],
//!     "bodies": [
//!       {
//!         "shape": { "type": "circle", "radius": 0.5 },
//!         "position": [0.0, 2.0],
//!         "angle": 0.0,
//!         "velocity": [0.0, 0.0],
//!         "angular_velocity": 0.0,
//!         "density": 10.0,
//!         "restitution": 0.3,
//!         "friction": 0.5,
//!         "is_static": false
//!       }
//!     ],
//!     "fluid": {
//!       "params": { "particle_spacing": 0.05, "rest_density": 10.0, ... },
//!       "positions": [
//!         [0.025, 0.025],
//!         [0.075, 0.025]
//!       ]
//!     },
//!     "solver": { "velocity_iterations": 10, ... }
//!   }
//! }
//! ```
//!
//! `fluid` and `solver` are optional; fluid velocities are left out when
//! every particle is at rest.
//!
//! All values are SI units (see the crate docs). Bump [`Scene::VERSION`]
//! whenever the format changes incompatibly, and add a migration in
//! [`Scene::from_json`] so old files keep loading.
//!
//! # Version history
//!
//! - **1:** bodies, gravity, and (optional) solver settings.
//! - **2:** adds the optional `fluid`. Version 1 files load unchanged; a
//!   version 1 build would silently drop the fluid from a version 2 file,
//!   which is why the version was bumped.

use crate::world::World;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub format: String,
    pub version: u32,
    pub world: World,
}

impl Scene {
    pub const FORMAT: &'static str = "physicsengine-scene";
    /// 2 added `fluid`, 3 added `joints`. Older files load unchanged.
    pub const VERSION: u32 = 3;
    /// Oldest version this build can read.
    pub const OLDEST_SUPPORTED: u32 = 1;

    pub fn from_world(world: &World) -> Self {
        Self {
            format: Self::FORMAT.to_owned(),
            version: Self::VERSION,
            world: world.clone(),
        }
    }

    /// Pretty-printed JSON, suitable for committing to git or editing by
    /// hand. Short arrays of numbers (vectors) stay on one line, so a fluid
    /// with thousands of particles is one line per particle, not four.
    pub fn to_json(&self) -> String {
        // Serializing plain data with no maps keyed by non-strings can't fail.
        let pretty = serde_json::to_string_pretty(self).expect("scene serialization is infallible");
        collapse_number_arrays(&pretty)
    }

    /// Parse and validate a scene file.
    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        let scene: Scene = serde_json::from_str(json).map_err(SceneError::Json)?;
        if scene.format != Self::FORMAT {
            return Err(SceneError::WrongFormat(scene.format));
        }
        if !(Self::OLDEST_SUPPORTED..=Self::VERSION).contains(&scene.version) {
            return Err(SceneError::UnsupportedVersion(scene.version));
        }
        // Migrations from older versions go here. 1 -> 2 needs none: a
        // version 1 file simply has no fluid.
        let mut scene = scene;
        scene.version = Self::VERSION;
        if !scene.world.gravity.is_finite() {
            return Err(SceneError::Invalid("gravity must be finite".into()));
        }
        for (i, body) in scene.world.bodies.iter().enumerate() {
            body.validate()
                .map_err(|e| SceneError::Invalid(format!("body {i}: {e}")))?;
        }
        scene.world.fluid.validate().map_err(SceneError::Invalid)?;
        for (i, joint) in scene.world.joints.iter().enumerate() {
            joint
                .validate(scene.world.bodies.len())
                .map_err(|e| SceneError::Invalid(format!("joint {i}: {e}")))?;
        }
        scene.world.fluid.fill_missing_velocities();
        Ok(scene)
    }
}

/// Rewrite serde_json's pretty output so arrays whose elements are all
/// numbers sit on one line:
///
/// ```text
/// "position": [        ->   "position": [0.0, -0.25],
///   0.0,
///   -0.25
/// ],
/// ```
fn collapse_number_arrays(pretty: &str) -> String {
    let lines: Vec<&str> = pretty.lines().collect();
    let mut out = String::with_capacity(pretty.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.trim_end().ends_with('[') {
            // Look ahead for number-only elements up to the closing bracket.
            let mut numbers = Vec::new();
            let mut j = i + 1;
            while j < lines.len() {
                let t = lines[j].trim();
                if t.starts_with(']') {
                    break;
                }
                let value = t.trim_end_matches(',');
                if value.parse::<f64>().is_err() {
                    break;
                }
                numbers.push(value);
                j += 1;
            }
            if j < lines.len() && !numbers.is_empty() && lines[j].trim().starts_with(']') {
                out.push_str(line.trim_end());
                out.push_str(&numbers.join(", "));
                out.push_str(lines[j].trim());
                out.push('\n');
                i = j + 1;
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    out.pop(); // match serde_json: no trailing newline
    out
}

#[derive(Debug)]
pub enum SceneError {
    Json(serde_json::Error),
    WrongFormat(String),
    UnsupportedVersion(u32),
    Invalid(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::Json(e) => write!(f, "not a valid scene file: {e}"),
            SceneError::WrongFormat(got) => {
                write!(f, "expected format \"{}\", found \"{got}\"", Scene::FORMAT)
            }
            SceneError::UnsupportedVersion(v) => write!(
                f,
                "scene version {v} is not supported (this build reads versions {} to {})",
                Scene::OLDEST_SUPPORTED,
                Scene::VERSION
            ),
            SceneError::Invalid(msg) => write!(f, "invalid scene: {msg}"),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SceneError::Json(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_world() {
        let world = World::demo();
        let json = Scene::from_world(&world).to_json();
        let loaded = Scene::from_json(&json).unwrap();
        assert_eq!(loaded.world, world);
    }

    #[test]
    fn rejects_future_versions() {
        let mut scene = Scene::from_world(&World::demo());
        scene.version = 99;
        let err = Scene::from_json(&scene.to_json()).unwrap_err();
        assert!(matches!(err, SceneError::UnsupportedVersion(99)));
    }

    #[test]
    fn rejects_invalid_bodies() {
        let mut scene = Scene::from_world(&World::demo());
        scene.world.bodies[1].density = -1.0;
        let err = Scene::from_json(&scene.to_json()).unwrap_err();
        assert!(matches!(err, SceneError::Invalid(_)), "{err}");
    }

    /// The files in `scenes/` must match the presets they were generated
    /// from. If this fails after changing a preset, regenerate them with
    /// `cargo run -p engine --example make_scenes`.
    #[test]
    fn example_scene_files_match_presets() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scenes");
        for (name, make) in crate::presets::ALL {
            let path = format!("{dir}/{name}.json");
            let json = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{path}: {e}. Run the make_scenes example."));
            let scene = Scene::from_json(&json).unwrap_or_else(|e| panic!("{path}: {e}"));
            assert!(
                scene.world == make(),
                "{path} is out of date. Run `cargo run -p engine --example make_scenes`."
            );
        }
    }

    /// Version 1 scenes (no fluid, maybe no solver settings) must still load.
    #[test]
    fn version_1_scene_loads() {
        let json = r#"{
            "format": "physicsengine-scene",
            "version": 1,
            "world": { "gravity": [0.0, -9.8], "bodies": [] }
        }"#;
        let scene = Scene::from_json(json).unwrap();
        assert_eq!(scene.version, Scene::VERSION);
        assert_eq!(scene.world.solver, crate::SolverSettings::default());
        assert!(scene.world.fluid.is_empty());
    }

    #[test]
    fn fluid_round_trips_and_resting_velocities_are_omitted() {
        let mut world = World::new();
        world
            .fluid
            .fill_rect(crate::Vec2::ZERO, crate::Vec2::new(0.2, 0.1), &[]);
        let json = Scene::from_world(&world).to_json();
        assert!(
            !json.contains("velocities"),
            "resting fluid should omit velocities"
        );
        assert!(
            json.contains("[0.025, 0.025]"),
            "vectors should be on one line:\n{json}"
        );
        let loaded = Scene::from_json(&json).unwrap();
        assert_eq!(loaded.world, world);
        assert_eq!(
            loaded.world.fluid.velocities.len(),
            loaded.world.fluid.len()
        );

        world.fluid.velocities[3] = crate::Vec2::new(1.0, 2.0);
        let json = Scene::from_world(&world).to_json();
        assert!(json.contains("velocities"));
        assert_eq!(Scene::from_json(&json).unwrap().world, world);
    }

    #[test]
    fn mismatched_fluid_arrays_are_rejected() {
        let mut world = World::new();
        world
            .fluid
            .fill_rect(crate::Vec2::ZERO, crate::Vec2::new(0.2, 0.1), &[]);
        world.fluid.velocities = vec![crate::Vec2::ONE; 2];
        let err = Scene::from_json(&Scene::from_world(&world).to_json()).unwrap_err();
        assert!(matches!(err, SceneError::Invalid(_)), "{err}");
    }

    #[test]
    fn number_arrays_collapse_but_nested_arrays_do_not() {
        let input = "{\n  \"a\": [\n    1.0,\n    -2.5\n  ],\n  \"b\": [\n    [\n      1.0,\n      2.0\n    ]\n  ]\n}";
        let expected = "{\n  \"a\": [1.0, -2.5],\n  \"b\": [\n    [1.0, 2.0]\n  ]\n}";
        assert_eq!(collapse_number_arrays(input), expected);
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(Scene::from_json("{"), Err(SceneError::Json(_))));
    }
}
