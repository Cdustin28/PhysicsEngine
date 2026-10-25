//! Saving and loading worlds as JSON scene files.
//!
//! A scene file looks like:
//!
//! ```json
//! {
//!   "format": "physicsengine-scene",
//!   "version": 1,
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
//!     ]
//!   }
//! }
//! ```
//!
//! All values are SI units (see the crate docs). Bump [`Scene::VERSION`]
//! whenever the format changes incompatibly, and add a migration in
//! [`Scene::from_json`] so old files keep loading.

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
    pub const VERSION: u32 = 1;

    pub fn from_world(world: &World) -> Self {
        Self {
            format: Self::FORMAT.to_owned(),
            version: Self::VERSION,
            world: world.clone(),
        }
    }

    /// Pretty-printed JSON, suitable for committing to git or editing by hand.
    pub fn to_json(&self) -> String {
        // Serializing plain data with no maps keyed by non-strings can't fail.
        serde_json::to_string_pretty(self).expect("scene serialization is infallible")
    }

    /// Parse and validate a scene file.
    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        let scene: Scene = serde_json::from_str(json).map_err(SceneError::Json)?;
        if scene.format != Self::FORMAT {
            return Err(SceneError::WrongFormat(scene.format));
        }
        if scene.version != Self::VERSION {
            return Err(SceneError::UnsupportedVersion(scene.version));
        }
        if !scene.world.gravity.is_finite() {
            return Err(SceneError::Invalid("gravity must be finite".into()));
        }
        for (i, body) in scene.world.bodies.iter().enumerate() {
            body.validate()
                .map_err(|e| SceneError::Invalid(format!("body {i}: {e}")))?;
        }
        Ok(scene)
    }
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
                "scene version {v} is not supported (this build reads version {})",
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

    /// Scenes saved before the solver settings existed must still load.
    #[test]
    fn scene_without_solver_settings_uses_defaults() {
        let json = r#"{
            "format": "physicsengine-scene",
            "version": 1,
            "world": { "gravity": [0.0, -9.8], "bodies": [] }
        }"#;
        let scene = Scene::from_json(json).unwrap();
        assert_eq!(scene.world.solver, crate::SolverSettings::default());
    }

    #[test]
    fn rejects_garbage() {
        assert!(matches!(Scene::from_json("{"), Err(SceneError::Json(_))));
    }
}
