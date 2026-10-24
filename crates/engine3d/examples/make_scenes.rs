//! Regenerate the 3D example scene files in `scenes/3d/` from the built-in
//! presets.
//!
//! ```sh
//! cargo run -p engine3d --example make_scenes
//! ```

use engine3d::{Scene, presets};
use std::path::Path;

fn main() -> std::io::Result<()> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenes/3d");
    std::fs::create_dir_all(&dir)?;
    for (name, make) in presets::ALL {
        let path = dir.join(format!("{name}.json"));
        std::fs::write(&path, Scene::from_world(&make()).to_json() + "\n")?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
