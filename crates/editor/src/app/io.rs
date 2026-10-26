//! Opening and saving scene files.
//!
//! On desktop these are ordinary OS file dialogs. In the browser, **Open…**
//! uses the browser's file picker (the result arrives asynchronously, so it
//! is parked in a [`PendingFile`] and picked up on the next frame) and
//! **Save…** downloads the scene as `scene.json`.

use super::{EditorApp, Mode};
use eframe::egui;
use engine::Scene;
use std::sync::{Arc, Mutex};

/// A picked file's name and contents, waiting to be loaded.
pub(super) type PendingFile = Arc<Mutex<Option<(String, String)>>>;

impl EditorApp {
    /// The scene in the current editor, as JSON.
    pub(super) fn scene_json(&self) -> String {
        match self.mode {
            Mode::TwoD => Scene::from_world(&self.world).to_json(),
            Mode::ThreeD => self.view3d.scene_json(),
            Mode::Orbits => self.orbits.system_json(),
            Mode::Fluids => self.fluids.scene_json(),
        }
    }

    /// Load a 2D or 3D scene, switching to the matching editor.
    pub(super) fn load_scene_json(&mut self, json: &str, source: &str) -> bool {
        if json.contains(&format!("\"{}\"", engine3d::Scene::FORMAT)) {
            let ok = self.view3d.load_scene_json(json, source);
            if ok {
                self.set_mode(Mode::ThreeD);
            }
            self.take_3d_status();
            return ok;
        }
        match Scene::from_json(json) {
            Ok(scene) => {
                self.set_mode(Mode::TwoD);
                self.replace_world(scene.world);
                self.status = format!("Loaded {source}.");
                true
            }
            Err(e) => {
                self.status = format!("Could not load {source}: {e}");
                false
            }
        }
    }

    /// Show the 3D editor's latest message in the shared status bar.
    pub(super) fn take_3d_status(&mut self) {
        if !self.view3d.status.is_empty() {
            self.status = std::mem::take(&mut self.view3d.status);
        }
    }

    /// Load a file the browser's picker delivered since last frame.
    pub(super) fn poll_pending_open(&mut self) {
        let picked = self
            .pending_open
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        if let Some((name, json)) = picked {
            self.load_scene_json(&json, &name);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn open_scene_dialog(&mut self, _ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Scene (JSON)", &["json"])
            .pick_file()
        else {
            return;
        };
        match std::fs::read_to_string(&path) {
            Ok(json) => {
                self.load_scene_json(&json, &path.display().to_string());
            }
            Err(e) => self.status = format!("Could not read {}: {e}", path.display()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn open_scene_dialog(&mut self, ctx: &egui::Context) {
        let slot = self.pending_open.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("Scene (JSON)", &["json"])
                .pick_file()
                .await
            else {
                return;
            };
            let bytes = file.read().await;
            if let Ok(mut slot) = slot.lock() {
                *slot = Some((
                    file.file_name(),
                    String::from_utf8_lossy(&bytes).into_owned(),
                ));
            }
            ctx.request_repaint();
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn save_scene_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Scene (JSON)", &["json"])
            .set_file_name("scene.json")
            .save_file()
        else {
            return;
        };
        self.status = match std::fs::write(&path, self.scene_json()) {
            Ok(()) => format!("Saved {}.", path.display()),
            Err(e) => format!("Could not save {}: {e}", path.display()),
        };
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) fn save_scene_dialog(&mut self) {
        self.status = match download("scene.json", &self.scene_json()) {
            Ok(()) => "Downloaded scene.json.".to_owned(),
            Err(e) => format!("Could not download the scene: {e}"),
        };
    }
}

/// Have the browser download `contents` as a file named `name`: wrap it in
/// a Blob, point a temporary link at it, and click the link.
#[cfg(target_arch = "wasm32")]
fn download(name: &str, contents: &str) -> Result<(), String> {
    use wasm_bindgen::{JsCast as _, JsValue};
    let js = |e: JsValue| format!("{e:?}");

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let parts = js_sys::Array::of1(&JsValue::from_str(contents));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("application/json");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&parts, &options).map_err(js)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js)?;
    let link: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(js)?
        .dyn_into()
        .map_err(|_| "could not create a link".to_owned())?;
    link.set_href(&url);
    link.set_download(name);
    link.click();
    web_sys::Url::revoke_object_url(&url).map_err(js)?;
    Ok(())
}
