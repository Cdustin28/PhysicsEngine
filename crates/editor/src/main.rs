//! physicsengine.dev editor. Builds for desktop (Windows, macOS, Linux) and,
//! through Trunk, for the browser.

mod app;
mod camera;
mod fluids;
mod history;
mod liquid_ui;
mod orbits;
mod particles;
mod theme;
mod view3d;

use app::EditorApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("PhysicsEngine")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 400.0]),
        // The 3D view needs a depth buffer (see `view3d/render.rs`).
        depth_buffer: 24,
        ..Default::default()
    };
    eframe::run_native(
        "PhysicsEngine",
        options,
        Box::new(|cc| Ok(Box::new(EditorApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    wasm_bindgen_futures::spawn_local(async {
        let canvas = eframe::web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("the_canvas_id"))
            .and_then(|e| e.dyn_into::<eframe::web_sys::HtmlCanvasElement>().ok())
            .expect("index.html must contain <canvas id=\"the_canvas_id\">");

        let result = eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions {
                    depth_buffer: 24,
                    ..Default::default()
                },
                Box::new(|cc| Ok(Box::new(EditorApp::new(cc)))),
            )
            .await;

        // Replace the "Loading…" text with the outcome.
        if let Some(status) = eframe::web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("loading_text"))
        {
            match result {
                Ok(()) => status.remove(),
                Err(e) => status.set_inner_html(&format!(
                    "<p>The simulator failed to start. Your browser may not support WebGPU or WebGL2.</p><pre>{e:?}</pre>"
                )),
            }
        }
    });
}
