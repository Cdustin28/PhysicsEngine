//! The editor application.
//!
//! - `mod.rs` (this file): state, simulation control, selection, undo/redo,
//!   keyboard shortcuts, and saving settings between sessions
//! - `panels.rs`: toolbar, sidebar, inspector, status bar
//! - `canvas.rs`: the canvas: mouse interaction and drawing
//! - `io.rs`: opening and saving scene files, on desktop and in the browser
//!
//! The 3D editor lives in `crate::view3d`; this app switches between them.

mod canvas;
mod gas;
mod io;
mod joints;
mod panels;
mod soft;

pub(crate) use panels::title_case;

use crate::camera::Camera;
use crate::history::History;
use crate::view3d::{Editor3d, Persisted3d};
use eframe::egui::{self, Color32, Key, KeyboardShortcut, Modifiers, Ui};
use engine::{BodyHandle, BodyId, FixedTimestep, RigidBody, Scene, Shape, Vec2, World};
use serde::{Deserialize, Serialize};

/// Iowa State cardinal, used for dynamic bodies.
const CARDINAL: Color32 = Color32::from_rgb(0xC8, 0x10, 0x2E);
/// Iowa State gold, used for selection, handles, and previews.
const GOLD: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);
const STATIC_FILL: Color32 = Color32::from_rgb(0x8A, 0x8D, 0x91);
/// Contact points and normals in the debug overlay.
const CONTACT_COLOR: Color32 = Color32::from_rgb(0x1F, 0x4E, 0x9A);
/// Fluid particle colors: deep blue at rest shading to pale blue when fast
/// (or highly pressurized).
const WATER_DEEP: Color32 = Color32::from_rgb(0x1F, 0x4E, 0x9A);
const WATER_LIGHT: Color32 = Color32::from_rgb(0xA8, 0xDD, 0xF7);
/// Speed shown at full brightness, m/s.
const SPEED_COLOR_SCALE: f32 = 3.0;
/// Depth of water whose hydrostatic pressure is shown at full brightness, m.
const PRESSURE_COLOR_DEPTH: f32 = 2.0;

/// Distance of the rotate handle beyond the body's bounding circle, px.
const ROTATE_HANDLE_OFFSET_PX: f32 = 28.0;
/// Radius of the rotate and resize handles, px.
const HANDLE_RADIUS_PX: f32 = 6.0;
/// Smallest size the resize handle allows (half extent or radius), m.
const MIN_HALF_SIZE: f32 = 0.025;
/// Rotation snaps to multiples of this while snapping is on.
const ANGLE_SNAP_DEGREES: f32 = 15.0;
/// How many undo steps to keep.
const UNDO_LIMIT: usize = 100;

/// Fastest speed a "thrown" body can leave the mouse with, m/s. Keeps a
/// flick of the mouse from launching a body into orbit.
const MAX_THROW_SPEED: f32 = 30.0;

/// Which editor is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
enum Mode {
    #[default]
    TwoD,
    ThreeD,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Circle,
    Box,
    /// Add fluid: drag a rectangle, or paint with a brush.
    Fluid,
    /// Remove fluid particles under the brush.
    Erase,
    /// Pin, rod, rope, or spring between bodies.
    Joint,
    /// Drag out a chain of pinned links.
    Chain,
    /// Place a soft blob or jelly block.
    Soft,
    /// Paint smoke and heat, or place emitters.
    Smoke,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FluidShape {
    Rectangle,
    Brush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ColorBy {
    Speed,
    Pressure,
}

/// Parameters for the next object placed with the Circle or Box tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NewBodySettings {
    radius: f32,
    width: f32,
    height: f32,
    density: f32,
    is_static: bool,
}

impl Default for NewBodySettings {
    fn default() -> Self {
        Self {
            radius: 0.5,
            width: 1.0,
            height: 1.0,
            density: RigidBody::DEFAULT_DENSITY,
            is_static: false,
        }
    }
}

impl NewBodySettings {
    fn shape(&self, tool: Tool) -> Option<Shape> {
        match tool {
            Tool::Circle => Some(Shape::Circle {
                radius: self.radius,
            }),
            Tool::Box => Some(Shape::rect(self.width, self.height)),
            _ => None,
        }
    }

    fn build(&self, tool: Tool, position: Vec2) -> Option<RigidBody> {
        let shape = self.shape(tool)?;
        let mut body = if self.is_static {
            RigidBody::new_static(shape, position)
        } else {
            RigidBody::new(shape, position)
        };
        body.density = self.density;
        Some(body)
    }
}

/// One selected body being moved by a drag.
struct Grab {
    handle: BodyHandle,
    /// Body position minus the cursor position at the start of the drag, so
    /// bodies keep their place relative to the cursor.
    offset: Vec2,
    /// Position last frame, for the throw velocity.
    prev_position: Vec2,
}

/// What a left-drag with the Select tool is doing.
enum Drag {
    /// Moving every selected body.
    Move { grabs: Vec<Grab> },
    /// Turning one body with its rotate handle.
    Rotate {
        handle: BodyHandle,
        /// Body angle minus the cursor's angle around the body's center at
        /// the start, so the body doesn't snap to the cursor.
        offset: f32,
    },
    /// Resizing one body with its resize handle.
    Resize { handle: BodyHandle },
    /// Dragging a selection rectangle over empty space.
    Marquee {
        start: Vec2,
        end: Vec2,
        /// Shift held: add to the selection instead of replacing it.
        additive: bool,
    },
}

/// Grid snapping: positions and lengths to multiples of `step`, angles to
/// multiples of 15°. Does nothing while `on` is false.
#[derive(Debug, Clone, Copy)]
struct Snapper {
    on: bool,
    step: f32,
}

impl Snapper {
    fn point(self, p: Vec2) -> Vec2 {
        if self.on {
            (p / self.step).round() * self.step
        } else {
            p
        }
    }

    /// Snap a full length (width, height, diameter) to whole steps.
    fn length(self, length: f32) -> f32 {
        if self.on {
            ((length / self.step).round() * self.step).max(self.step)
        } else {
            length
        }
    }

    fn angle(self, angle: f32) -> f32 {
        if self.on {
            let step = ANGLE_SNAP_DEGREES.to_radians();
            (angle / step).round() * step
        } else {
            angle
        }
    }
}

/// Settings remembered between sessions (desktop: a file in the OS config
/// folder; web: the browser's local storage).
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Persisted {
    mode: Mode,
    joint_settings: joints::JointSettings,
    soft_settings: soft::SoftSettings,
    gas_settings: gas::GasSettings,
    view3d: Persisted3d,
    camera: Camera,
    new_body: NewBodySettings,
    fluid_shape: FluidShape,
    brush_radius: f32,
    color_by: ColorBy,
    show_contacts: bool,
    show_fluid_forces: bool,
    snap: bool,
    time_scale: f32,
    step_hz: u32,
    use_gpu: bool,
    /// The scene as it was when the app closed, as scene JSON.
    scene: Option<String>,
}

impl Default for Persisted {
    fn default() -> Self {
        Self {
            mode: Mode::TwoD,
            joint_settings: joints::JointSettings::default(),
            soft_settings: soft::SoftSettings::default(),
            gas_settings: gas::GasSettings::default(),
            view3d: Persisted3d::default(),
            camera: Camera::default(),
            new_body: NewBodySettings::default(),
            fluid_shape: FluidShape::Rectangle,
            brush_radius: 0.3,
            color_by: ColorBy::Speed,
            show_contacts: false,
            show_fluid_forces: true,
            snap: false,
            time_scale: 1.0,
            step_hz: 120,
            use_gpu: false,
            scene: None,
        }
    }
}

pub struct EditorApp {
    mode: Mode,
    /// The 3D editor (its own world, camera, history).
    view3d: Editor3d,
    world: World,
    /// World state captured when Play was pressed, restored by Reset.
    snapshot: Option<World>,
    running: bool,
    timestep: FixedTimestep,
    /// 1.0 = real time.
    time_scale: f32,
    steps_last_frame: u32,
    history: History<World>,

    camera: Camera,
    tool: Tool,
    new_body: NewBodySettings,
    selection: Vec<BodyHandle>,
    drag: Option<Drag>,
    snap: bool,
    show_contacts: bool,
    fluid_shape: FluidShape,
    /// Radius of the fluid brush and the eraser, m.
    brush_radius: f32,
    /// Where a rectangle fill drag started.
    fluid_rect_start: Option<Vec2>,
    color_by: ColorBy,
    show_fluid_forces: bool,
    joint_settings: joints::JointSettings,
    /// Body and point where a rod/rope/spring drag started.
    joint_drag: Option<(usize, Vec2)>,
    /// Where a chain drag started.
    chain_start: Option<Vec2>,
    soft_settings: soft::SoftSettings,
    /// The soft body selected (separately from rigid bodies).
    selected_soft: Option<usize>,
    /// Soft body being dragged, and the cursor's last position.
    soft_drag: Option<(usize, Vec2)>,
    gas_settings: gas::GasSettings,
    /// The smoke/heat grid's image, updated every frame.
    gas_texture: Option<egui::TextureHandle>,
    /// The world rectangle on screen last frame (for sizing new grids).
    viewport_world: (Vec2, Vec2),

    /// Text of the "Paste scene JSON" window while it is open.
    paste_buffer: Option<String>,
    /// A scene file picked in the browser, delivered asynchronously.
    pending_open: io::PendingFile,
    status: String,
    /// Cursor position over the canvas, m. Updated while drawing the canvas
    /// and shown in the status bar on the next frame.
    cursor_world: Option<Vec2>,

    /// Fluid particles are drawn by the GPU renderer (`particles.rs`) when
    /// eframe runs on wgpu; otherwise with egui circles.
    gpu_render: bool,
    /// The GPU fluid solver, if this machine can run it (desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    gpu_solver: Option<gpu::GpuSolver>,
    /// Use the GPU solver when it's available.
    use_gpu: bool,
}

impl EditorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let persisted = cc
            .storage
            .and_then(|s| eframe::get_value::<Persisted>(s, eframe::APP_KEY))
            .unwrap_or_default();
        let mut app = Self::from_persisted(persisted);
        app.history.rebase(&app.world);
        app.gpu_render = crate::particles::ParticleRenderer::install(cc);
        let render3d = crate::view3d::render::BodyRenderer::install(cc);
        app.view3d.set_gpu_render(render3d);
        #[cfg(not(target_arch = "wasm32"))]
        {
            app.view3d
                .set_gpu_solver(cc.wgpu_render_state.as_ref().and_then(|rs| {
                    let compute = rs
                        .adapter
                        .get_downlevel_capabilities()
                        .flags
                        .contains(eframe::egui_wgpu::wgpu::DownlevelFlags::COMPUTE_SHADERS);
                    compute.then(|| {
                        gpu::GpuSolver3d::new(
                            rs.device.clone(),
                            rs.queue.clone(),
                            rs.adapter.get_info().name,
                        )
                    })
                }));
            app.gpu_solver = cc.wgpu_render_state.as_ref().and_then(|rs| {
                let compute = rs
                    .adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(eframe::egui_wgpu::wgpu::DownlevelFlags::COMPUTE_SHADERS);
                compute.then(|| {
                    gpu::GpuSolver::new(
                        rs.device.clone(),
                        rs.queue.clone(),
                        rs.adapter.get_info().name,
                    )
                })
            });
        }
        app
    }

    /// Advance the world one fixed step with the chosen fluid solver.
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

    /// The GPU's name, if the GPU fluid solver can be used here.
    fn gpu_name(&self) -> Option<&str> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(solver) = &self.gpu_solver {
            return Some(solver.adapter_name());
        }
        None
    }

    /// Which fluid solver the next step will use.
    fn solver_label(&self) -> &'static str {
        if self.use_gpu && self.gpu_name().is_some() {
            "GPU"
        } else {
            "CPU"
        }
    }

    fn from_persisted(mut p: Persisted) -> Self {
        let (world, status) = match p.scene.as_deref().map(Scene::from_json) {
            Some(Ok(scene)) => (scene.world, "Restored your last scene.".to_owned()),
            Some(Err(e)) => (
                World::demo(),
                format!("Couldn't restore your last scene: {e}"),
            ),
            None => (World::demo(), "Ready. Press Space to play.".to_owned()),
        };
        Self {
            mode: p.mode,
            view3d: Editor3d::new(std::mem::take(&mut p.view3d), false),
            world,
            snapshot: None,
            running: false,
            timestep: FixedTimestep::new(1.0 / p.step_hz.clamp(30, 960) as f32),
            time_scale: p.time_scale.clamp(0.05, 2.0),
            steps_last_frame: 0,
            history: History::new(UNDO_LIMIT),
            camera: p.camera,
            tool: Tool::Select,
            new_body: p.new_body,
            selection: Vec::new(),
            drag: None,
            snap: p.snap,
            show_contacts: p.show_contacts,
            fluid_shape: p.fluid_shape,
            brush_radius: p.brush_radius,
            fluid_rect_start: None,
            color_by: p.color_by,
            show_fluid_forces: p.show_fluid_forces,
            joint_settings: p.joint_settings.clone(),
            joint_drag: None,
            chain_start: None,
            soft_settings: p.soft_settings.clone(),
            selected_soft: None,
            soft_drag: None,
            gas_settings: p.gas_settings.clone(),
            gas_texture: None,
            viewport_world: (Vec2::new(-5.0, 0.0), Vec2::new(5.0, 7.0)),
            paste_buffer: None,
            pending_open: io::PendingFile::default(),
            status,
            cursor_world: None,
            gpu_render: false,
            #[cfg(not(target_arch = "wasm32"))]
            gpu_solver: None,
            use_gpu: p.use_gpu,
        }
    }

    fn persisted(&self) -> Persisted {
        Persisted {
            mode: self.mode,
            joint_settings: self.joint_settings.clone(),
            soft_settings: self.soft_settings.clone(),
            gas_settings: self.gas_settings.clone(),
            view3d: self.view3d.persisted(),
            camera: self.camera,
            new_body: self.new_body.clone(),
            fluid_shape: self.fluid_shape,
            brush_radius: self.brush_radius,
            color_by: self.color_by,
            show_contacts: self.show_contacts,
            show_fluid_forces: self.show_fluid_forces,
            snap: self.snap,
            time_scale: self.time_scale,
            step_hz: (1.0 / self.timestep.dt).round() as u32,
            use_gpu: self.use_gpu,
            scene: Some(Scene::from_world(&self.world).to_json()),
        }
    }

    // ---------------------------------------------------------------------
    // Simulation control

    fn play(&mut self) {
        if !self.running {
            // Any edit made just before pressing Play is still undoable.
            self.history.record(&self.world);
            self.snapshot = Some(self.world.clone());
            self.timestep.reset();
            self.running = true;
        }
    }

    fn pause(&mut self) {
        if self.running {
            self.running = false;
            // Where the simulation got to isn't an edit to undo.
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
            self.drag = None;
            self.timestep.reset();
            self.history.rebase(&self.world);
            self.status = "Reset to the state before Play.".into();
        }
    }

    /// Swap in a whole new world (New, an example, a loaded file). The swap
    /// itself is recorded as an undoable edit on the next idle frame.
    fn replace_world(&mut self, world: World) {
        self.pause();
        self.history.record(&self.world);
        self.world = world;
        self.snapshot = None;
        self.selection.clear();
        self.selected_soft = None;
        self.soft_drag = None;
        self.drag = None;
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

    /// Show a state from the undo history. Selected bodies that still exist
    /// stay selected, since handles survive in the saved copies.
    fn restore(&mut self, world: World) {
        self.world = world;
        self.drag = None;
        self.snapshot = None;
        let world = &self.world;
        self.selection.retain(|&h| world.find(h).is_some());
        self.soft_drag = None;
        if self
            .selected_soft
            .is_some_and(|i| i >= self.world.soft_bodies.len())
        {
            self.selected_soft = None;
        }
    }

    /// Record an edit if the user has finished one: called at the end of
    /// every frame, it records only when nothing is in progress.
    fn record_finished_edit(&mut self, ctx: &egui::Context) {
        let busy = self.running
            || self.drag.is_some()
            || self.fluid_rect_start.is_some()
            || self.joint_drag.is_some()
            || self.chain_start.is_some()
            || self.soft_drag.is_some()
            || ctx.input(|i| i.pointer.any_down());
        if !busy {
            self.history.record(&self.world);
        }
    }

    // ---------------------------------------------------------------------
    // Selection

    /// Indices of the selected bodies that still exist.
    fn selected_ids(&self) -> Vec<BodyId> {
        self.selection
            .iter()
            .filter_map(|&h| self.world.find(h))
            .collect()
    }

    /// The selected body, if exactly one is selected.
    fn single_selected(&self) -> Option<BodyId> {
        match self.selected_ids()[..] {
            [id] => Some(id),
            _ => None,
        }
    }

    fn is_selected(&self, id: BodyId) -> bool {
        self.world
            .body(id)
            .and_then(RigidBody::handle)
            .is_some_and(|h| self.selection.contains(&h))
    }

    fn select_only(&mut self, id: BodyId) {
        self.selection.clear();
        self.add_to_selection(id);
    }

    fn add_to_selection(&mut self, id: BodyId) {
        if let Some(h) = self.world.handle(id)
            && !self.selection.contains(&h)
        {
            self.selection.push(h);
        }
    }

    fn toggle_selected(&mut self, id: BodyId) {
        if let Some(h) = self.world.handle(id) {
            if let Some(i) = self.selection.iter().position(|&s| s == h) {
                self.selection.remove(i);
            } else {
                self.selection.push(h);
            }
        }
    }

    fn delete_selected(&mut self) {
        if self.delete_selected_soft() {
            return;
        }
        let mut ids = self.selected_ids();
        // Remove from the back so earlier indices stay valid.
        ids.sort_unstable_by_key(|id| std::cmp::Reverse(id.0));
        for id in &ids {
            self.world.remove_body(*id);
        }
        self.selection.clear();
        self.drag = None;
        if !ids.is_empty() {
            self.status = format!("Deleted {} bodies.", ids.len());
        }
    }

    // ---------------------------------------------------------------------
    // Grid snapping

    /// Snap increment: a quarter of the grid currently shown, m.
    fn snap_step(&self) -> f32 {
        self.camera.grid_spacing(80.0) / 4.0
    }

    /// The current snap settings, copied out so they can be used while a
    /// body is borrowed mutably.
    fn snapper(&self) -> Snapper {
        Snapper {
            on: self.snap,
            step: self.snap_step(),
        }
    }

    fn snap_point(&self, p: Vec2) -> Vec2 {
        self.snapper().point(p)
    }

    // ---------------------------------------------------------------------
    // Keyboard

    /// Switch between the 2D and 3D editors. Whichever is left is paused.
    fn set_mode(&mut self, mode: Mode) {
        if mode != self.mode {
            self.pause();
            self.view3d.pause_public();
            self.mode = mode;
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.mode == Mode::ThreeD {
            self.view3d.handle_shortcuts(ctx);
            return;
        }
        if ctx.egui_wants_keyboard_input() {
            return; // typing in a text field
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
        if shortcut(Modifiers::COMMAND, Key::A) {
            self.selection = (0..self.world.bodies.len())
                .filter_map(|i| self.world.handle(BodyId(i)))
                .collect();
        }

        // Single-key shortcuts only without Ctrl/Cmd/Alt, so Ctrl+C doesn't
        // also pick the Circle tool.
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
            self.selection.clear();
            self.selected_soft = None;
        }
        if pressed(Key::V) {
            self.tool = Tool::Select;
        }
        if pressed(Key::C) {
            self.tool = Tool::Circle;
        }
        if pressed(Key::B) {
            self.tool = Tool::Box;
        }
        if pressed(Key::F) {
            self.tool = Tool::Fluid;
        }
        if pressed(Key::E) {
            self.tool = Tool::Erase;
        }
        if pressed(Key::J) {
            self.tool = Tool::Joint;
        }
        if pressed(Key::L) {
            self.tool = Tool::Chain;
        }
        if pressed(Key::S) {
            self.tool = Tool::Soft;
        }
        if pressed(Key::K) {
            self.tool = Tool::Smoke;
        }
        if pressed(Key::G) {
            self.snap = !self.snap;
        }
        if pressed(Key::Delete) || pressed(Key::Backspace) {
            self.delete_selected();
        }
        if pressed(Key::D) {
            self.show_contacts = !self.show_contacts;
        }
    }
}

impl eframe::App for EditorApp {
    /// Physics runs here, once per frame, before any UI is drawn.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_pending_open();
        self.steps_last_frame = 0;
        if self.mode == Mode::ThreeD {
            self.view3d.logic(ctx);
            return;
        }
        if self.running {
            // Cap the frame time so a stall (window drag, breakpoint) doesn't
            // dump seconds of backlog into the simulation.
            let frame_time = ctx.input(|i| i.stable_dt).min(0.1) * self.time_scale;
            let steps = self.timestep.advance(frame_time);
            for _ in 0..steps {
                self.step_world();
            }
            self.steps_last_frame = steps;
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.handle_shortcuts(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));

        // Side panels must be laid out before the central canvas, so the
        // status bar shows last frame's cursor position. Nobody can tell.
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));

        if self.mode == Mode::ThreeD {
            egui::Panel::left("sidebar3d")
                .resizable(true)
                .default_size(290.0)
                .show(ui, |ui| self.view3d.sidebar(ui));
            egui::CentralPanel::no_frame().show(ui, |ui| self.view3d.canvas(ui));
            self.paste_window(&ctx);
            self.view3d.record_finished_edit(&ctx);
            self.take_3d_status();
            return;
        }

        egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(290.0)
            .show(ui, |ui| self.sidebar(ui));

        self.cursor_world = egui::CentralPanel::no_frame()
            .show(ui, |ui| self.canvas(ui))
            .inner;

        self.paste_window(&ctx);
        self.record_finished_edit(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.persisted());
    }
}
