//! 3D scene files: JSON, like the 2D ones, with their own format name.
//!
//! ```json
//! {
//!   "format": "physicsengine-scene-3d",
//!   "version": 2,
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
//! `orientation` is a quaternion [x, y, z, w]. Densities are kg/m³. A
//! `fluid` object (particle positions, velocities, and settings) is present
//! only when the scene has water.

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
    /// 2 added `fluid`, 3 `joints`, 4 `soft_bodies`, 5 `gas`. Older files
    /// load unchanged.
    pub const VERSION: u32 = 5;
    pub const OLDEST_SUPPORTED: u32 = 1;

    pub fn from_world(world: &World) -> Self {
        Self {
            format: Self::FORMAT.to_owned(),
            version: Self::VERSION,
            world: world.clone(),
        }
    }

    pub fn to_json(&self) -> String {
        let pretty = serde_json::to_string_pretty(self).expect("scene serialization is infallible");
        collapse_number_arrays(&pretty)
    }

    pub fn from_json(json: &str) -> Result<Self, SceneError> {
        let mut scene: Scene = serde_json::from_str(json).map_err(SceneError::Json)?;
        if scene.format != Self::FORMAT {
            return Err(SceneError::WrongFormat(scene.format));
        }
        if !(Self::OLDEST_SUPPORTED..=Self::VERSION).contains(&scene.version) {
            return Err(SceneError::UnsupportedVersion(scene.version));
        }
        scene.version = Self::VERSION;
        if !scene.world.gravity.is_finite() {
            return Err(SceneError::Invalid("gravity must be finite".into()));
        }
        for (i, body) in scene.world.bodies.iter_mut().enumerate() {
            body.validate()
                .map_err(|e| SceneError::Invalid(format!("body {i}: {e}")))?;
            body.orientation = body.orientation.normalize();
        }
        scene.world.fluid.validate().map_err(SceneError::Invalid)?;
        if let Some(gas) = scene.world.gas.as_mut() {
            gas.validate().map_err(SceneError::Invalid)?;
            gas.fill_missing();
        }
        for (i, s) in scene.world.soft_bodies.iter_mut().enumerate() {
            s.validate()
                .map_err(|e| SceneError::Invalid(format!("soft body {i}: {e}")))?;
            s.velocities.resize(s.positions.len(), glam::Vec3::ZERO);
        }
        for (i, joint) in scene.world.joints.iter().enumerate() {
            joint
                .validate(scene.world.bodies.len())
                .map_err(|e| SceneError::Invalid(format!("joint {i}: {e}")))?;
        }
        scene.world.fluid.fill_missing_velocities();
        scene.world.assign_handles();
        Ok(scene)
    }
}

/// Same layout as 2D scene files (see `engine::scene`).
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
