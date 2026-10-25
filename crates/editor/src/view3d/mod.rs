//! The 3D editor: the same workflow as 2D (place, drag, play, undo, save)
//! for spheres and boxes in 3D.
//!
//! - `mod.rs` (this file): state, simulation control, undo, shortcuts
//! - `ui.rs`: toolbar, sidebar, status bar, and the viewport
//! - `orbit.rs`: the orbit camera, picking rays
//! - `render.rs`: GPU drawing of bodies

mod orbit;
pub mod render;
mod ui;

use crate::history::History;
use eframe::egui::{self, Key, KeyboardShortcut, Modifiers};
use engine::FixedTimestep;
use engine3d::{BodyHandle, BodyId, RigidBody, Scene, Shape, Vec3, World};
use orbit::OrbitCamera;
use serde::{Deserialize, Serialize};

const UNDO_LIMIT: usize = 100;
const STEP_HZ: f32 = 120.0;
/// Fastest a thrown body leaves the mouse, m/s.
const MAX_THROW_SPEED: f32 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Sphere,
    Box,
    /// Click to drop a block of water.
    Water,
}

/// Parameters for the next body placed with the Sphere or Box tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NewBody {
    radius: f32,
    /// Full width, height, depth, m.
    size: Vec3,
    density: f32,
    is_static: bool,
    /// Size of the block of water the Water tool drops, m.
    water_size: Vec3,
}

impl Default for NewBody {
    fn default() -> Self {
        Self {
            radius: 0.5,
            size: Vec3::ONE,
            density: RigidBody::DEFAULT_DENSITY,
            is_static: false,
            water_size: Vec3::new(1.0, 0.6, 1.0),
        }
    }
}

impl NewBody {
    fn shape(&self, tool: Tool) -> Option<Shape> {
        match tool {
            Tool::Select | Tool::Water => None,
            Tool::Sphere => Some(Shape::Sphere {
                radius: self.radius,
            }),
            Tool::Box => Some(Shape::cuboid(self.size.x, self.size.y, self.size.z)),
        }
    }
}

/// Dragging a body: it follows the mouse across a plane through the point
/// where it was grabbed.
struct Grab {
    handle: BodyHandle,
    plane_point: Vec3,
    plane_normal: Vec3,
    /// Body center minus the grab point.
    offset: Vec3,
    prev_position: Vec3,
}

/// Settings remembered between sessions.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Persisted3d {
    camera: OrbitCamera,
    new_body: NewBody,
    show_contacts: bool,
    time_scale: f32,
    use_gpu: bool,
    /// The 3D scene as it was when the app closed, as scene JSON.
    scene: Option<String>,
}

impl Default for Persisted3d {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            new_body: NewBody::default(),
            show_contacts: false,
            time_scale: 1.0,
            use_gpu: false,
            scene: None,
        }
    }
}

pub struct Editor3d {
    world: World,
    snapshot: Option<World>,
    running: bool,
    timestep: FixedTimestep,
    time_scale: f32,
    steps_last_frame: u32,
    history: History<World>,
    camera: OrbitCamera,
    tool: Tool,
    new_body: NewBody,
    selection: Option<BodyHandle>,
    grab: Option<Grab>,
    show_contacts: bool,
    pub status: String,
    /// Whether the GPU body renderer is available.
    gpu_render: bool,
    /// The GPU fluid solver, if this machine can run it (desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    gpu_solver: Option<gpu::GpuSolver3d>,
    use_gpu: bool,
}

impl Editor3d {
    pub fn new(p: Persisted3d, gpu_render: bool) -> Self {
        let (world, status) = match p.scene.as_deref().map(Scene::from_json) {
            Some(Ok(scene)) => (scene.world, "Restored your last 3D scene.".to_owned()),
            Some(Err(e)) => (
                engine3d::presets::demo(),
                format!("Couldn't restore your last 3D scene: {e}"),
            ),
            None => (
                engine3d::presets::demo(),
                "Ready. Press Space to play.".to_owned(),
            ),
        };
        let mut editor = Self {
            world,
            snapshot: None,
            running: false,
            timestep: FixedTimestep::new(1.0 / STEP_HZ),
            time_scale: p.time_scale.clamp(0.05, 2.0),
            steps_last_frame: 0,
            history: History::new(UNDO_LIMIT),
            camera: p.camera,
            tool: Tool::Select,
            new_body: p.new_body,
            selection: None,
            grab: None,
            show_contacts: p.show_contacts,
            status,
            gpu_render,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_solver: None,
            use_gpu: p.use_gpu,
        };
        editor.history.rebase(&editor.world);
        editor
    }

    pub fn persisted(&self) -> Persisted3d {
        Persisted3d {
            camera: self.camera,
            new_body: self.new_body.clone(),
            show_contacts: self.show_contacts,
            time_scale: self.time_scale,
            use_gpu: self.use_gpu,
            scene: Some(self.scene_json()),
        }
    }

    pub fn scene_json(&self) -> String {
        Scene::from_world(&self.world).to_json()
    }

    /// Load a 3D scene file. Returns false (with the reason in `status`) if
    /// it isn't one.
    pub fn load_scene_json(&mut self, json: &str, source: &str) -> bool {
        match Scene::from_json(json) {
            Ok(scene) => {
                self.replace_world(scene.world);
                self.frame_scene();
                self.status = format!("Loaded {source}.");
                true
            }
            Err(e) => {
                self.status = format!("Could not load {source}: {e}");
                false
            }
        }
    }

    /// Run the simulation for this frame. Called before any UI is drawn.
    pub fn logic(&mut self, ctx: &egui::Context) {
        self.steps_last_frame = 0;
        if self.running {
            let frame_time = ctx.input(|i| i.stable_dt).min(0.1) * self.time_scale;
            let steps = self.timestep.advance(frame_time);
            for _ in 0..steps {
                self.step_world();
            }
            self.steps_last_frame = steps;
        }
    }

    // ---------------------------------------------------------------------
    // Simulation control (as in 2D; see `app/mod.rs`)

    fn play(&mut self) {
        if !self.running {
            self.history.record(&self.world);
            self.snapshot = Some(self.world.clone());
            self.timestep.reset();
            self.running = true;
        }
    }

    pub fn set_gpu_render(&mut self, on: bool) {
        self.gpu_render = on;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_gpu_solver(&mut self, solver: Option<gpu::GpuSolver3d>) {
        self.gpu_solver = solver;
    }

    /// The GPU's name, if the GPU fluid solver can be used here.
    fn gpu_name(&self) -> Option<&str> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(solver) = &self.gpu_solver {
            return Some(solver.adapter_name());
        }
        None
    }

    /// One fixed step with the chosen fluid solver.
    fn step_world(&mut self) {
        let dt = self.timestep.dt;
        #[cfg(not(target_arch = "wasm32"))]
        if self.use_gpu
            && let Some(solver) = self.gpu_solver.as_mut()
        {
            self.world.step_with(dt, solver);
            return;
        }
        self.world.step(dt);
    }

    /// Pause (when switching to the 2D editor).
    pub fn pause_public(&mut self) {
        self.pause();
    }

    fn pause(&mut self) {
        if self.running {
            self.running = false;
            self.history.rebase(&self.world);
        }
    }

    fn step_once(&mut self) {
        self.pause();
        self.history.record(&self.world);
        self.step_world();
        self.history.rebase(&self.world);
    }

    fn reset(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.running = false;
            self.world = snapshot;
            self.grab = None;
            self.timestep.reset();
            self.history.rebase(&self.world);
            self.status = "Reset to the state before Play.".into();
        }
    }

    fn replace_world(&mut self, world: World) {
        self.pause();
        self.history.record(&self.world);
        self.world = world;
        self.snapshot = None;
        self.selection = None;
        self.grab = None;
        self.timestep.reset();
    }

    fn undo(&mut self) {
        self.pause();
        self.history.record(&self.world);
        if let Some(world) = self.history.undo() {
            self.restore(world);
            self.status = "Undone.".into();
        }
    }

    fn redo(&mut self) {
        self.pause();
        if let Some(world) = self.history.redo() {
            self.restore(world);
            self.status = "Redone.".into();
        }
    }

    fn restore(&mut self, world: World) {
        self.world = world;
        self.grab = None;
        self.snapshot = None;
        if let Some(h) = self.selection
            && self.world.find(h).is_none()
        {
            self.selection = None;
        }
    }

    /// Record an edit once the user has finished one (end of every frame).
    pub fn record_finished_edit(&mut self, ctx: &egui::Context) {
        let busy = self.running || self.grab.is_some() || ctx.input(|i| i.pointer.any_down());
        if !busy {
            self.history.record(&self.world);
        }
    }

    fn selected(&self) -> Option<BodyId> {
        self.selection.and_then(|h| self.world.find(h))
    }

    fn select(&mut self, id: Option<BodyId>) {
        self.selection = id.and_then(|id| self.world.handle(id));
    }

    fn delete_selected(&mut self) {
        if let Some(id) = self.selected() {
            self.world.remove_body(id);
            self.selection = None;
            self.grab = None;
            self.status = "Deleted 1 body.".into();
        }
    }

    /// Point the camera at the scene: the bodies (except huge ones like the
    /// floor) and the water, from above and at a distance that fits them.
    fn frame_scene(&mut self) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for b in &self.world.bodies {
            let extent = match b.shape {
                Shape::Sphere { radius } => Vec3::splat(radius),
                Shape::Cuboid { half_extents } => Vec3::splat(half_extents.length()),
            };
            if extent.x < 5.0 {
                lo = lo.min(b.position - extent);
                hi = hi.max(b.position + extent);
            }
        }
        for &p in &self.world.fluid.positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        if !lo.is_finite() {
            self.camera = OrbitCamera::default();
            return;
        }
        let radius = 0.5 * (hi - lo).length();
        self.camera.target = 0.5 * (lo + hi);
        self.camera.distance = (2.5 * radius).clamp(4.0, 60.0);
        // Steep enough (about 50° down) to see into a tank over its walls.
        self.camera.pitch = self.camera.pitch.max(0.85);
    }

    /// Aim the camera at the selected body.
    fn focus_selected(&mut self) {
        if let Some(b) = self.selected().and_then(|id| self.world.body(id)) {
            self.camera.target = b.position;
        }
    }

    pub fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let shortcut =
            |mods, key| ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(mods, key)));
        if shortcut(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || shortcut(Modifiers::COMMAND, Key::Y)
        {
            self.redo();
        } else if shortcut(Modifiers::COMMAND, Key::Z) {
            self.undo();
        }
        let mods = ctx.input(|i| i.modifiers);
        if mods.command || mods.alt {
            return;
        }
        let pressed = |k| ctx.input(|i| i.key_pressed(k));
        if pressed(Key::Space) {
            if self.running {
                self.pause();
            } else {
                self.play();
            }
        }
        if pressed(Key::Period) {
            self.step_once();
        }
        if pressed(Key::R) {
            self.reset();
        }
        if pressed(Key::Escape) {
            self.tool = Tool::Select;
            self.selection = None;
        }
        if pressed(Key::V) {
            self.tool = Tool::Select;
        }
        if pressed(Key::C) {
            self.tool = Tool::Sphere;
        }
        if pressed(Key::B) {
            self.tool = Tool::Box;
        }
        if pressed(Key::W) {
            self.tool = Tool::Water;
        }
        if pressed(Key::F) {
            self.focus_selected();
        }
        if pressed(Key::D) {
            self.show_contacts = !self.show_contacts;
        }
        if pressed(Key::Delete) || pressed(Key::Backspace) {
            self.delete_selected();
        }
    }
}

/// Where the ray `origin + t·dir` crosses the plane, if in front.
fn ray_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    (t >= 0.0).then(|| origin + t * dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_plane_hits_floor() {
        let p = ray_plane(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y).unwrap();
        assert!(p.length() < 1e-6);
        assert!(ray_plane(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::Y).is_none());
    }

    #[test]
    fn new_box_has_requested_size() {
        let nb = NewBody {
            size: Vec3::new(1.0, 2.0, 3.0),
            ..NewBody::default()
        };
        assert_eq!(
            nb.shape(Tool::Box),
            Some(Shape::Cuboid {
                half_extents: Vec3::new(0.5, 1.0, 1.5)
            })
        );
    }
}
