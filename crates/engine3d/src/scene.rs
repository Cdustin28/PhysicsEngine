//! 3D scene files: JSON, like the 2D ones, with their own format name.
//!
//! ```json
//! {
//!   "format": "physicsengine-scene-3d",
//!   "version": 1,
//!   "world": {
//!     "gravity": [0.0, -9.80665, 0.0],
//!     "bodies": [
//!       {
//!         "shape": { "type": "sphere", "radius": 0.5 },
//!         "position": [0.0, 2.0, 0.0],
//!         "orientation": [0.0, 0.0, 0.0, 1.0],
//!         ...
//!       }
//!     ]
//!   }
//! }
//! ```
//!
//! `orientation` is a quaternion [x, y, z, w]. Densities are kg/m³.

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
    pub const FORMAT: &'static str = "physicsengine-scene-3d";
    pub const VERSION: u32 = 1;

    pub fn from_world(world: &World) -> Self {
        Self {
            format: Self::FORMAT.to_owned(),
            version: Self::VERSION,
            world: world.clone(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("scene serialization is infallible")
    }

    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        let mut scene: Scene = serde_json::from_str(json).map_err(SceneError::Json)?;
        if scene.format != Self::FORMAT {
            return Err(SceneError::WrongFormat(scene.format));
        }
        if scene.version != Self::VERSION {
            return Err(SceneError::UnsupportedVersion(scene.version));
        }
        if !scene.world.gravity.is_finite() {
            return Err(SceneError::Invalid("gravity must be finite".into()));
        }
        for (i, body) in scene.world.bodies.iter_mut().enumerate() {
            body.validate()
                .map_err(|e| SceneError::Invalid(format!("body {i}: {e}")))?;
            body.orientation = body.orientation.normalize();
        }
        scene.world.assign_handles();
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
            SceneError::Json(e) => write!(f, "not a valid 3D scene file: {e}"),
            SceneError::WrongFormat(got) => write!(
                f,
                "expected format \"{}\", found \"{got}\" (is this a 2D scene?)",
                Scene::FORMAT
            ),
            SceneError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "3D scene version {v} is not supported (this build reads version {})",
                    Scene::VERSION
                )
            }
            SceneError::Invalid(msg) => write!(f, "invalid scene: {msg}"),
        }
    }
}

impl std::error::Error for SceneError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let w = crate::presets::demo();
        let loaded = Scene::from_json(&Scene::from_world(&w).to_json()).unwrap();
        assert_eq!(loaded.world, w);
    }

    #[test]
    fn rejects_2d_scenes() {
        let json = r#"{"format": "physicsengine-scene", "version": 2, "world": {"gravity": [0, -9.8, 0], "bodies": []}}"#;
        assert!(matches!(
            Scene::from_json(json),
            Err(SceneError::WrongFormat(_))
        ));
    }
}
