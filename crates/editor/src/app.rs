//! The editor application: toolbar, sidebar, status bar, and the canvas.

use crate::camera::Camera;
use eframe::egui::{self, Color32, Key, PointerButton, Pos2, Rect, Sense, Stroke, Ui};
use engine::{BodyId, FixedTimestep, RigidBody, Scene, Shape, Vec2, World, presets};

/// Iowa State cardinal, used for dynamic bodies.
const CARDINAL: Color32 = Color32::from_rgb(0xC8, 0x10, 0x2E);
/// Iowa State gold, used for selection and previews.
const GOLD: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);
const STATIC_FILL: Color32 = Color32::from_rgb(0x8A, 0x8D, 0x91);
/// Contact points and normals in the debug overlay.
const CONTACT_COLOR: Color32 = Color32::from_rgb(0x1F, 0x4E, 0x9A);

/// Distance of the rotate handle beyond the body's bounding circle, px.
const ROTATE_HANDLE_OFFSET_PX: f32 = 28.0;
/// Radius of the rotate handle, px.
const ROTATE_HANDLE_RADIUS_PX: f32 = 6.0;

/// Fastest speed a "thrown" body can leave the mouse with, m/s. Keeps a
/// flick of the mouse from launching a body into orbit.
const MAX_THROW_SPEED: f32 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Circle,
    Box,
}

/// Parameters for the next object placed with the Circle or Box tool.
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
            Tool::Select => None,
            Tool::Circle => Some(Shape::Circle {
                radius: self.radius,
            }),
            Tool::Box => Some(Shape::rect(self.width, self.height)),
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

/// A body being dragged with the mouse.
enum Drag {
    Move {
        id: BodyId,
        /// Body position minus the grab point, so the body doesn't jump to
        /// put its center under the cursor.
        offset: Vec2,
        prev_position: Vec2,
    },
    Rotate {
        id: BodyId,
        /// Body angle minus the cursor's angle around the body's center at
        /// the moment of grabbing, so the body doesn't snap.
        offset: f32,
    },
}

impl Drag {
    fn id(&self) -> BodyId {
        match *self {
            Drag::Move { id, .. } | Drag::Rotate { id, .. } => id,
        }
    }
}

pub struct EditorApp {
    world: World,
    /// World state captured when Play was pressed, restored by Reset.
    snapshot: Option<World>,
    running: bool,
    timestep: FixedTimestep,
    /// 1.0 = real time.
    time_scale: f32,
    steps_last_frame: u32,

    camera: Camera,
    tool: Tool,
    new_body: NewBodySettings,
    selected: Option<BodyId>,
    drag: Option<Drag>,
    show_contacts: bool,

    /// Text of the "Paste scene JSON" window while it is open.
    paste_buffer: Option<String>,
    status: String,
    /// Cursor position over the canvas, m. Updated while drawing the canvas
    /// and shown in the status bar on the next frame.
    cursor_world: Option<Vec2>,
}

impl EditorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        Self {
            world: World::demo(),
            snapshot: None,
            running: false,
            timestep: FixedTimestep::default(),
            time_scale: 1.0,
            steps_last_frame: 0,
            camera: Camera::default(),
            tool: Tool::Select,
            new_body: NewBodySettings::default(),
            selected: None,
            drag: None,
            show_contacts: false,
            paste_buffer: None,
            status: "Ready. Press Space to play.".into(),
            cursor_world: None,
        }
    }

    // ---------------------------------------------------------------------
    // Simulation control

    fn play(&mut self) {
        if !self.running {
            self.snapshot = Some(self.world.clone());
            self.timestep.reset();
            self.running = true;
        }
    }

    fn pause(&mut self) {
        self.running = false;
    }

    fn step_once(&mut self) {
        self.running = false;
        self.world.step(self.timestep.dt);
    }

    fn reset(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.replace_world(snapshot);
            self.status = "Reset to the state before Play.".into();
        }
    }

    /// Swap in a whole new world, dropping any handles into the old one.
    fn replace_world(&mut self, world: World) {
        self.world = world;
        self.running = false;
        self.selected = None;
        self.drag = None;
        self.timestep.reset();
    }

    fn delete_selected(&mut self) {
        if let Some(id) = self.selected.take() {
            self.world.remove_body(id);
            // Removal shifts indices, so any in-progress drag is now stale.
            self.drag = None;
        }
    }

    // ---------------------------------------------------------------------
    // Scene files

    fn load_scene_json(&mut self, json: &str, source: &str) -> bool {
        match Scene::from_json(json) {
            Ok(scene) => {
                self.replace_world(scene.world);
                self.snapshot = None;
                self.status = format!("Loaded {source}.");
                true
            }
            Err(e) => {
                self.status = format!("Could not load {source}: {e}");
                false
            }
        }
    }

    fn scene_json(&self) -> String {
        Scene::from_world(&self.world).to_json()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_dialog(&mut self) {
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

    #[cfg(not(target_arch = "wasm32"))]
    fn open_dialog(&mut self) {
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

    // ---------------------------------------------------------------------
    // UI

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return; // typing in a text field
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
        if pressed(Key::V) || pressed(Key::Escape) {
            self.tool = Tool::Select;
        }
        if pressed(Key::C) {
            self.tool = Tool::Circle;
        }
        if pressed(Key::B) {
            self.tool = Tool::Box;
        }
        if pressed(Key::Delete) || pressed(Key::Backspace) {
            self.delete_selected();
        }
        if pressed(Key::D) {
            self.show_contacts = !self.show_contacts;
        }
    }

    fn toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            if self.running {
                if ui.button("Pause").on_hover_text("Space").clicked() {
                    self.pause();
                }
            } else if ui.button("Play").on_hover_text("Space").clicked() {
                self.play();
            }
            if ui
                .button("Step")
                .on_hover_text("Advance one physics step (.)")
                .clicked()
            {
                self.step_once();
            }
            if ui
                .add_enabled(self.snapshot.is_some(), egui::Button::new("Reset"))
                .on_hover_text("Restore the scene from before Play (R)")
                .clicked()
            {
                self.reset();
            }
            ui.add(
                egui::Slider::new(&mut self.time_scale, 0.05..=2.0)
                    .logarithmic(true)
                    .text("speed"),
            );

            ui.separator();

            if ui
                .button("New")
                .on_hover_text("Empty world with a floor")
                .clicked()
            {
                let mut world = World::new();
                world.add_body(RigidBody::new_static(
                    Shape::rect(20.0, 0.5),
                    Vec2::new(0.0, -0.25),
                ));
                self.replace_world(world);
                self.snapshot = None;
            }
            let mut chosen = None;
            egui::ComboBox::from_id_salt("examples")
                .selected_text("Examples")
                .show_ui(ui, |ui| {
                    for &(name, make) in presets::ALL {
                        if ui.selectable_label(false, title_case(name)).clicked() {
                            chosen = Some((name, make));
                        }
                    }
                });
            if let Some((name, make)) = chosen {
                self.replace_world(make());
                self.snapshot = None;
                self.status = format!("Loaded the {name} example. Press Space to play.");
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                if ui.button("Open…").clicked() {
                    self.open_dialog();
                }
                if ui.button("Save…").clicked() {
                    self.save_dialog();
                }
            }
            if ui
                .button("Copy JSON")
                .on_hover_text("Copy the scene to the clipboard")
                .clicked()
            {
                ui.ctx().copy_text(self.scene_json());
                self.status = "Scene JSON copied to clipboard.".into();
            }
            if ui.button("Paste JSON…").clicked() {
                self.paste_buffer = Some(String::new());
            }
        });
    }

    fn sidebar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Tools");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Select, "Select (V)");
                ui.selectable_value(&mut self.tool, Tool::Circle, "Circle (C)");
                ui.selectable_value(&mut self.tool, Tool::Box, "Box (B)");
            });

            if self.tool != Tool::Select {
                ui.add_space(6.0);
                ui.label(egui::RichText::new("New object").strong());
                let nb = &mut self.new_body;
                egui::Grid::new("new_body").num_columns(2).show(ui, |ui| {
                    if self.tool == Tool::Circle {
                        length_row(ui, "Radius", &mut nb.radius);
                    } else {
                        length_row(ui, "Width", &mut nb.width);
                        length_row(ui, "Height", &mut nb.height);
                    }
                    ui.label("Density");
                    ui.add(
                        egui::DragValue::new(&mut nb.density)
                            .speed(0.1)
                            .range(0.01..=10_000.0)
                            .suffix(" kg/m²"),
                    );
                    ui.end_row();
                    ui.label("Static");
                    ui.checkbox(&mut nb.is_static, "");
                    ui.end_row();
                });
                if let Some(body) = nb.build(self.tool, Vec2::ZERO)
                    && !body.is_static
                {
                    ui.weak(format!("Mass: {:.3} kg", body.mass()));
                }
                ui.weak("Click on the canvas to place.");
            }

            ui.separator();
            ui.heading("World");
            egui::Grid::new("world").num_columns(2).show(ui, |ui| {
                ui.label("Gravity x");
                ui.add(
                    egui::DragValue::new(&mut self.world.gravity.x)
                        .speed(0.1)
                        .suffix(" m/s²"),
                );
                ui.end_row();
                ui.label("Gravity y");
                ui.add(
                    egui::DragValue::new(&mut self.world.gravity.y)
                        .speed(0.1)
                        .suffix(" m/s²"),
                );
                ui.end_row();
                ui.label("Step rate");
                let mut hz = (1.0 / self.timestep.dt).round() as u32;
                egui::ComboBox::from_id_salt("hz")
                    .selected_text(format!("{hz} Hz"))
                    .show_ui(ui, |ui| {
                        for option in [60, 120, 240, 480] {
                            ui.selectable_value(&mut hz, option, format!("{option} Hz"));
                        }
                    });
                self.timestep.dt = 1.0 / hz as f32;
                ui.end_row();
            });
            ui.checkbox(&mut self.show_contacts, "Show contacts (D)");

            ui.collapsing("Solver", |ui| {
                let s = &mut self.world.solver;
                egui::Grid::new("solver").num_columns(2).show(ui, |ui| {
                    ui.label("Iterations");
                    ui.add(egui::Slider::new(&mut s.velocity_iterations, 1..=50));
                    ui.end_row();
                    ui.label("Position fix")
                        .on_hover_text("Fraction of overlap removed per step");
                    ui.add(egui::Slider::new(&mut s.baumgarte, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Slop")
                        .on_hover_text("Overlap tolerated without correction");
                    ui.add(
                        egui::DragValue::new(&mut s.allowed_penetration)
                            .speed(0.0005)
                            .range(0.0..=0.1)
                            .suffix(" m"),
                    );
                    ui.end_row();
                    ui.label("Bounce threshold")
                        .on_hover_text("Impacts slower than this don't bounce");
                    ui.add(
                        egui::DragValue::new(&mut s.restitution_threshold)
                            .speed(0.05)
                            .range(0.0..=10.0)
                            .suffix(" m/s"),
                    );
                    ui.end_row();
                    ui.label("Warm starting");
                    ui.checkbox(&mut s.warm_starting, "");
                    ui.end_row();
                });
                if ui.button("Reset to defaults").clicked() {
                    *s = engine::SolverSettings::default();
                }
            });

            ui.separator();
            ui.heading("Selection");
            match self.selected {
                Some(id) if self.world.body(id).is_some() => {
                    if self.inspector(ui, id) {
                        self.delete_selected();
                    }
                }
                _ => {
                    ui.weak("Nothing selected. Use the Select tool and click a body.");
                }
            }

            ui.separator();
            ui.collapsing("Controls", |ui| {
                ui.label("Left-click: select / place");
                ui.label("Left-drag: move a body (release to throw while playing)");
                ui.label("Drag the gold handle: rotate the selected body");
                ui.label("Right- or middle-drag: pan");
                ui.label("Scroll or pinch: zoom");
                ui.label("Space: play/pause · . step · R reset");
                ui.label("Delete: remove selected · D contact overlay");
            });
        });
    }

    /// Property editor for one body. Returns true if Delete was clicked.
    fn inspector(&mut self, ui: &mut Ui, id: BodyId) -> bool {
        let Some(body) = self.world.body_mut(id) else {
            return false;
        };
        let mut delete = false;

        egui::Grid::new("inspector").num_columns(2).show(ui, |ui| {
            match &mut body.shape {
                Shape::Circle { radius } => {
                    length_row(ui, "Radius", radius);
                }
                Shape::Box { half_extents } => {
                    let mut w = half_extents.x * 2.0;
                    let mut h = half_extents.y * 2.0;
                    length_row(ui, "Width", &mut w);
                    length_row(ui, "Height", &mut h);
                    *half_extents = Vec2::new(w, h) * 0.5;
                }
            }

            ui.label("Position");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut body.position.x)
                        .speed(0.01)
                        .prefix("x ")
                        .suffix(" m"),
                );
                ui.add(
                    egui::DragValue::new(&mut body.position.y)
                        .speed(0.01)
                        .prefix("y ")
                        .suffix(" m"),
                );
            });
            ui.end_row();

            ui.label("Angle");
            let mut degrees = body.angle.to_degrees();
            if ui
                .add(egui::DragValue::new(&mut degrees).speed(1.0).suffix("°"))
                .changed()
            {
                body.angle = degrees.to_radians();
            }
            ui.end_row();

            ui.label("Velocity");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut body.velocity.x)
                        .speed(0.05)
                        .prefix("x ")
                        .suffix(" m/s"),
                );
                ui.add(
                    egui::DragValue::new(&mut body.velocity.y)
                        .speed(0.05)
                        .prefix("y ")
                        .suffix(" m/s"),
                );
            });
            ui.end_row();

            ui.label("Spin");
            ui.add(
                egui::DragValue::new(&mut body.angular_velocity)
                    .speed(0.05)
                    .suffix(" rad/s"),
            );
            ui.end_row();

            ui.label("Static");
            ui.checkbox(&mut body.is_static, "");
            ui.end_row();

            ui.label("Density");
            ui.add_enabled(
                !body.is_static,
                egui::DragValue::new(&mut body.density)
                    .speed(0.1)
                    .range(0.01..=10_000.0)
                    .suffix(" kg/m²"),
            );
            ui.end_row();

            ui.label("Restitution");
            ui.add(egui::Slider::new(&mut body.restitution, 0.0..=1.0));
            ui.end_row();

            ui.label("Friction");
            ui.add(egui::Slider::new(&mut body.friction, 0.0..=2.0));
            ui.end_row();
        });

        if !body.is_static {
            ui.weak(format!(
                "Mass {:.3} kg · inertia {:.4} kg·m² · KE {:.2} J",
                body.mass(),
                body.inertia(),
                body.kinetic_energy()
            ));
        }
        if ui.button("Delete").clicked() {
            delete = true;
        }
        delete
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(if self.running { "Running" } else { "Paused" });
            ui.separator();
            ui.label(format!("t = {:.2} s", self.world.time));
            ui.separator();
            ui.label(format!("{} bodies", self.world.bodies.len()));
            ui.separator();
            ui.label(format!("KE {:.2} J", self.world.kinetic_energy()));
            ui.separator();
            ui.label(format!(
                "{:.0} Hz × {} steps/frame",
                1.0 / self.timestep.dt,
                self.steps_last_frame
            ));
            if let Some(p) = self.cursor_world {
                ui.separator();
                ui.label(format!("cursor ({:.2}, {:.2}) m", p.x, p.y));
            }
            ui.separator();
            ui.weak(&self.status);
        });
    }

    fn paste_window(&mut self, ctx: &egui::Context) {
        let Some(buffer) = self.paste_buffer.as_mut() else {
            return;
        };
        let mut open = true;
        let mut load = false;
        egui::Window::new("Paste scene JSON")
            .open(&mut open)
            .default_width(420.0)
            .show(ctx, |ui| {
                ui.add(
                    egui::TextEdit::multiline(buffer)
                        .code_editor()
                        .desired_rows(14)
                        .desired_width(f32::INFINITY),
                );
                load = ui.button("Load").clicked();
            });

        if load {
            let json = buffer.clone();
            if self.load_scene_json(&json, "pasted scene") {
                self.paste_buffer = None;
            }
        } else if !open {
            self.paste_buffer = None;
        }
    }

    // ---------------------------------------------------------------------
    // Canvas

    /// Draws the world and handles mouse interaction. Returns the cursor's
    /// world position, if it's over the canvas.
    fn canvas(&mut self, ui: &mut Ui) -> Option<Vec2> {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let viewport = response.rect;
        let cursor_screen = response.hover_pos();
        let cursor_world = cursor_screen.map(|p| self.camera.screen_to_world(viewport, p));

        // --- Camera: zoom with scroll/pinch, pan with right or middle drag.
        if let Some(anchor) = cursor_screen {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll * 0.0015).exp();
            if factor != 1.0 {
                self.camera.zoom_about(viewport, anchor, factor);
            }
        }
        if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
        {
            self.camera.pan_pixels(response.drag_delta());
        }

        // --- Select tool: click to select, drag to move/throw, drag the
        // handle to rotate.
        if self.tool == Tool::Select {
            if response.drag_started_by(PointerButton::Primary) {
                // Use where the button went down, not where the drag was
                // recognized a few pixels later.
                let origin = ui.input(|i| i.pointer.press_origin()).or(cursor_screen);
                self.drag = origin.and_then(|press| self.start_drag(viewport, press));
                if let Some(drag) = &self.drag {
                    self.selected = Some(drag.id());
                } else {
                    self.selected = None;
                }
            }
            if response.dragged_by(PointerButton::Primary)
                && let Some(pointer) = response.interact_pointer_pos()
            {
                let pointer = self.camera.screen_to_world(viewport, pointer);
                let dt = ui.input(|i| i.stable_dt).max(1e-4);
                self.continue_drag(pointer, dt);
            }
            if response.drag_stopped_by(PointerButton::Primary) {
                self.drag = None;
            }
            if response.clicked_by(PointerButton::Primary) {
                self.selected = cursor_world.and_then(|p| self.world.body_at_point(p));
            }
        } else if response.clicked_by(PointerButton::Primary)
            && let Some(p) = cursor_world
            && let Some(body) = self.new_body.build(self.tool, p)
        {
            self.selected = Some(self.world.add_body(body));
            self.status = "Placed. Esc or V returns to Select.".into();
        }

        // --- Drawing.
        let visuals = ui.visuals();
        painter.rect_filled(viewport, 0.0, visuals.extreme_bg_color);
        self.draw_grid(&painter, viewport, visuals.weak_text_color());

        for (i, body) in self.world.bodies.iter().enumerate() {
            if self.selected != Some(BodyId(i)) {
                self.draw_body(&painter, viewport, body, false, None);
            }
        }
        // Draw the selected body last so its outline isn't hidden under its
        // neighbors.
        if let Some(body) = self.selected.and_then(|id| self.world.body(id)) {
            self.draw_body(&painter, viewport, body, true, None);
        }
        if self.show_contacts {
            self.draw_contacts(&painter, viewport);
        }
        if self.tool == Tool::Select
            && let Some(body) = self.selected.and_then(|id| self.world.body(id))
        {
            let center = self.camera.world_to_screen(viewport, body.position);
            let handle = self
                .camera
                .world_to_screen(viewport, self.rotate_handle(body));
            painter.line_segment([center, handle], Stroke::new(1.5, GOLD));
            painter.circle_filled(handle, ROTATE_HANDLE_RADIUS_PX, GOLD);
            painter.circle_stroke(
                handle,
                ROTATE_HANDLE_RADIUS_PX,
                Stroke::new(1.0, Color32::from_black_alpha(160)),
            );
        }

        // Ghost preview of the object about to be placed.
        if let Some(p) = cursor_world
            && let Some(ghost) = self.new_body.build(self.tool, p)
        {
            self.draw_body(
                &painter,
                viewport,
                &ghost,
                false,
                Some(GOLD.gamma_multiply(0.45)),
            );
        }

        if self.running || response.dragged() {
            ui.ctx().request_repaint();
        }
        cursor_world
    }

    /// World position of the rotate handle for `body`. It rotates with the
    /// body so you can see which way is "up" for it.
    fn rotate_handle(&self, body: &RigidBody) -> Vec2 {
        let reach =
            body.shape.bounding_radius() + ROTATE_HANDLE_OFFSET_PX / self.camera.pixels_per_meter;
        body.local_to_world(Vec2::new(0.0, reach))
    }

    /// Decide what a left-drag starting at screen point `press` grabs: the
    /// selected body's rotate handle, a body to move, or nothing.
    fn start_drag(&self, viewport: Rect, press: Pos2) -> Option<Drag> {
        if let Some(id) = self.selected
            && let Some(body) = self.world.body(id)
        {
            let handle = self
                .camera
                .world_to_screen(viewport, self.rotate_handle(body));
            if handle.distance(press) <= ROTATE_HANDLE_RADIUS_PX + 4.0 {
                let grab = self.camera.screen_to_world(viewport, press) - body.position;
                return Some(Drag::Rotate {
                    id,
                    offset: body.angle - grab.to_angle(),
                });
            }
        }
        let grab = self.camera.screen_to_world(viewport, press);
        let id = self.world.body_at_point(grab)?;
        let position = self.world.body(id)?.position;
        Some(Drag::Move {
            id,
            offset: position - grab,
            prev_position: position,
        })
    }

    /// Apply an in-progress drag with the cursor at world point `pointer`.
    fn continue_drag(&mut self, pointer: Vec2, dt: f32) {
        let running = self.running;
        match self.drag.as_mut() {
            Some(Drag::Move {
                id,
                offset,
                prev_position,
            }) => {
                let target = pointer + *offset;
                if let Some(body) = self.world.body_mut(*id) {
                    body.position = target;
                    body.angular_velocity = 0.0;
                    // While playing, the body leaves the mouse with the speed
                    // it was dragged at, so you can throw things.
                    body.velocity = if running {
                        ((target - *prev_position) / dt).clamp_length_max(MAX_THROW_SPEED)
                    } else {
                        Vec2::ZERO
                    };
                }
                *prev_position = target;
            }
            Some(Drag::Rotate { id, offset }) => {
                if let Some(body) = self.world.body_mut(*id) {
                    let to_cursor = pointer - body.position;
                    if to_cursor.length_squared() > 1e-8 {
                        body.angle = to_cursor.to_angle() + *offset;
                    }
                    // Hold it still while rotating so it doesn't fall away.
                    body.velocity = Vec2::ZERO;
                    body.angular_velocity = 0.0;
                }
            }
            None => {}
        }
    }

    /// Debug overlay: a dot at each contact point and a short line along the
    /// contact normal (pointing from body A toward body B).
    fn draw_contacts(&self, painter: &egui::Painter, viewport: Rect) {
        let normal_px = 18.0;
        for m in self.world.contacts() {
            for p in &m.points {
                let at = self.camera.world_to_screen(viewport, p.position);
                let tip = self.camera.world_to_screen(
                    viewport,
                    p.position + m.normal * (normal_px / self.camera.pixels_per_meter),
                );
                painter.line_segment([at, tip], Stroke::new(2.0, CONTACT_COLOR));
                painter.circle_filled(at, 3.5, CONTACT_COLOR);
            }
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, viewport: Rect, color: Color32) {
        let spacing = self.camera.grid_spacing(80.0);
        let (min, max) = self.camera.visible_world_bounds(viewport);
        let minor = Stroke::new(1.0, color.gamma_multiply(0.18));
        let axis = Stroke::new(1.5, color.gamma_multiply(0.6));

        let to_screen = |p: Vec2| self.camera.world_to_screen(viewport, p);

        let mut x = (min.x / spacing).floor() * spacing;
        while x <= max.x {
            let stroke = if x.abs() < spacing * 0.5 { axis } else { minor };
            painter.line_segment(
                [
                    to_screen(Vec2::new(x, min.y)),
                    to_screen(Vec2::new(x, max.y)),
                ],
                stroke,
            );
            x += spacing;
        }
        let mut y = (min.y / spacing).floor() * spacing;
        while y <= max.y {
            let stroke = if y.abs() < spacing * 0.5 { axis } else { minor };
            painter.line_segment(
                [
                    to_screen(Vec2::new(min.x, y)),
                    to_screen(Vec2::new(max.x, y)),
                ],
                stroke,
            );
            y += spacing;
        }

        painter.text(
            viewport.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            format!("grid {spacing} m"),
            egui::FontId::proportional(12.0),
            color,
        );
    }

    fn draw_body(
        &self,
        painter: &egui::Painter,
        viewport: Rect,
        body: &RigidBody,
        selected: bool,
        override_fill: Option<Color32>,
    ) {
        let fill = override_fill.unwrap_or(if body.is_static {
            STATIC_FILL
        } else {
            CARDINAL
        });
        let outline = if selected {
            Stroke::new(3.0, GOLD)
        } else {
            Stroke::new(1.0, Color32::from_black_alpha(140))
        };
        let to_screen = |p: Vec2| -> Pos2 { self.camera.world_to_screen(viewport, p) };

        match body.shape {
            Shape::Circle { radius } => {
                let center = to_screen(body.position);
                let r = self.camera.scale(radius);
                painter.circle_filled(center, r, fill);
                painter.circle_stroke(center, r, outline);
                // A spoke so you can see it rotate.
                let rim = to_screen(body.local_to_world(Vec2::new(radius, 0.0)));
                painter.line_segment(
                    [center, rim],
                    Stroke::new(1.5, Color32::from_white_alpha(200)),
                );
            }
            Shape::Box { .. } => {
                if let Some(corners) = body.box_corners() {
                    let points = corners.iter().map(|&c| to_screen(c)).collect();
                    painter.add(egui::Shape::convex_polygon(points, fill, outline));
                }
            }
        }
    }
}

/// "pyramid" -> "Pyramid".
fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// A labeled DragValue for a positive length in meters.
fn length_row(ui: &mut Ui, label: &str, value: &mut f32) {
    ui.label(label);
    ui.add(
        egui::DragValue::new(value)
            .speed(0.01)
            .range(0.01..=100.0)
            .suffix(" m"),
    );
    ui.end_row();
}

impl eframe::App for EditorApp {
    /// Physics runs here, once per frame, before any UI is drawn.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.steps_last_frame = 0;
        if self.running {
            // Cap the frame time so a stall (window drag, breakpoint) doesn't
            // dump seconds of backlog into the simulation.
            let frame_time = ctx.input(|i| i.stable_dt).min(0.1) * self.time_scale;
            let steps = self.timestep.advance(frame_time);
            for _ in 0..steps {
                self.world.step(self.timestep.dt);
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

        egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(290.0)
            .show(ui, |ui| self.sidebar(ui));

        self.cursor_world = egui::CentralPanel::no_frame()
            .show(ui, |ui| self.canvas(ui))
            .inner;

        self.paste_window(&ctx);
    }
}
