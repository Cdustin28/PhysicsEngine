//! Toolbar, sidebar, inspector, and status bar.

use super::{ColorBy, EditorApp, FluidShape, Mode, Tool};
use crate::theme::Theme;
use eframe::egui::{self, Ui};
use engine::{BodyId, RigidBody, Shape, Vec2, World, presets};

impl EditorApp {
    pub(super) fn toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            let mut mode = self.mode;
            ui.selectable_value(&mut mode, Mode::TwoD, "2D");
            ui.selectable_value(&mut mode, Mode::ThreeD, "3D");
            ui.selectable_value(&mut mode, Mode::Orbits, "Orbits");
            self.set_mode(mode);
            ui.separator();
            match self.mode {
                Mode::ThreeD => self.view3d.toolbar(ui),
                Mode::TwoD => self.toolbar_2d(ui),
                Mode::Orbits => self.orbits.toolbar(ui),
            }
            self.file_buttons(ui);
        });
    }

    fn toolbar_2d(&mut self, ui: &mut Ui) {
        {
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
                .add_enabled(self.history.can_undo(), egui::Button::new("Undo"))
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(self.history.can_redo(), egui::Button::new("Redo"))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
                .clicked()
            {
                self.redo();
            }
            ui.checkbox(&mut self.snap, "Snap").on_hover_text(format!(
                "Snap positions and sizes to {} m and angles to 15° (G)",
                self.snap_step()
            ));

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
            }
            let mut chosen = None;
            egui::ComboBox::from_id_salt("examples")
                .selected_text("Examples")
                // Tall enough to list every example without scrolling.
                .height(600.0)
                .show_ui(ui, |ui| {
                    for &(name, make) in presets::ALL {
                        if ui.selectable_label(false, title_case(name)).clicked() {
                            chosen = Some((name, make));
                        }
                    }
                });
            if let Some((name, make)) = chosen {
                self.replace_world(make());
                self.status = format!(
                    "Loaded the {} example. Press Space to play.",
                    title_case(name)
                );
            }
        }
    }

    /// Open, Save, Copy, Paste: shared by both editors.
    fn file_buttons(&mut self, ui: &mut Ui) {
        ui.separator();
        // Scene files are 2D/3D scenes; the Orbits tab saves its system
        // between sessions instead.
        if self.mode != Mode::Orbits {
            if ui
                .button("Open…")
                .on_hover_text("Open a scene file")
                .clicked()
            {
                self.open_scene_dialog(ui.ctx());
            }
            if ui
                .button("Save…")
                .on_hover_text("Save the scene as a file")
                .clicked()
            {
                self.save_scene_dialog();
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
        }
        ui.separator();
        let mut dark = self.theme == Theme::Dark;
        if ui
            .toggle_value(&mut dark, "Dark mode")
            .on_hover_text("Switch between the light and dark grey themes")
            .changed()
        {
            self.theme = if dark { Theme::Dark } else { Theme::Light };
            self.theme.apply(ui.ctx());
        }
    }

    pub(super) fn sidebar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Tools");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Select, "Select (V)");
                ui.selectable_value(&mut self.tool, Tool::Circle, "Circle (C)");
                ui.selectable_value(&mut self.tool, Tool::Box, "Box (B)");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Fluid, "Fluid (F)");
                ui.selectable_value(&mut self.tool, Tool::Erase, "Erase (E)");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Joint, "Joint (J)");
                ui.selectable_value(&mut self.tool, Tool::Chain, "Chain (L)");
                ui.selectable_value(&mut self.tool, Tool::Soft, "Soft (S)");
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Smoke, "Smoke (K)");
            });
            self.tool_options(ui);

            ui.separator();
            self.world_panel(ui);

            ui.separator();
            self.fluid_panel(ui);

            ui.separator();
            let view = self.viewport_world;
            self.gas_panel(ui, view);

            ui.separator();
            ui.heading("Selection");
            let ids = self.selected_ids();
            if let Some(i) = self.selected_soft {
                self.soft_inspector(ui, i);
            }
            match ids[..] {
                [] if self.selected_soft.is_some() => {}
                [] => {
                    ui.weak("Nothing selected. Click a body, Shift+click to add more, or drag a box around several.");
                }
                [id] => {
                    if self.inspector(ui, id) {
                        self.delete_selected();
                    }
                    let attached = self.world.joints.iter().filter(|j| j.involves(id.0)).count();
                    if attached > 0
                        && ui
                            .button(format!("Remove {attached} joint(s)"))
                            .clicked()
                    {
                        self.world.remove_joints_of(id);
                        self.status = format!("Removed {attached} joint(s).");
                    }
                }
                _ => {
                    ui.label(format!("{} bodies selected", ids.len()));
                    ui.weak("Drag any of them to move them together.");
                    ui.horizontal(|ui| {
                        if ui.button("Make static").clicked() {
                            for &id in &ids {
                                if let Some(b) = self.world.body_mut(id) {
                                    b.is_static = true;
                                }
                            }
                        }
                        if ui.button("Make dynamic").clicked() {
                            for &id in &ids {
                                if let Some(b) = self.world.body_mut(id) {
                                    b.is_static = false;
                                }
                            }
                        }
                        if ui.button("Delete").clicked() {
                            self.delete_selected();
                        }
                    });
                }
            }

            ui.separator();
            ui.collapsing("Controls", |ui| {
                ui.label("Click: select · Shift+click: add/remove · drag empty space: box select");
                ui.label("Drag a body: move the selection (release to throw while playing)");
                ui.label("Gold handles: round = rotate, square = resize");
                ui.label("Right- or middle-drag: pan · scroll or pinch: zoom");
                ui.label("Space: play/pause · . step · R reset");
                ui.label("Ctrl+Z undo · Ctrl+Shift+Z / Ctrl+Y redo · Ctrl+A select all");
                ui.label("V select · C circle · B box · F fluid · E erase · Esc deselect");
                ui.label("G snap · D contact overlay · Delete: remove selected");
                ui.label("Blue arrows: fluid force on a body (weight = 1 m)");
            });
        });
    }

    fn tool_options(&mut self, ui: &mut Ui) {
        match self.tool {
            Tool::Select => {}
            Tool::Joint => self.joint_tool_options(ui),
            Tool::Chain => self.chain_tool_options(ui),
            Tool::Soft => self.soft_tool_options(ui),
            Tool::Smoke => self.smoke_tool_options(ui),
            Tool::Fluid => {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.fluid_shape, FluidShape::Rectangle, "Rectangle");
                    ui.selectable_value(&mut self.fluid_shape, FluidShape::Brush, "Brush");
                });
                self.liquid.picker(ui, &mut self.liquid_celsius);
                if self.fluid_shape == FluidShape::Brush {
                    egui::Grid::new("brush").num_columns(2).show(ui, |ui| {
                        length_row(ui, "Brush radius", &mut self.brush_radius);
                    });
                    ui.weak(format!(
                        "Drag on the canvas to paint {}.",
                        self.liquid.noun()
                    ));
                } else {
                    ui.weak(format!(
                        "Drag a rectangle on the canvas to fill it with {}.",
                        self.liquid.noun()
                    ));
                }
                crate::liquid_ui::hint(ui);
            }
            Tool::Erase => {
                ui.add_space(6.0);
                egui::Grid::new("eraser").num_columns(2).show(ui, |ui| {
                    length_row(ui, "Eraser radius", &mut self.brush_radius);
                });
                ui.weak("Drag over water to remove it.");
            }
            Tool::Circle | Tool::Box => {
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
        }
    }

    fn world_panel(&mut self, ui: &mut Ui) {
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
                ui.label("Substeps").on_hover_text(
                    "Rigid-body substeps per step. Chains and ropes stay tighter with 4–8.",
                );
                ui.add(egui::Slider::new(&mut s.substeps, 1..=16));
                ui.end_row();
            });
            if ui.button("Reset to defaults").clicked() {
                *s = engine::SolverSettings::default();
            }
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
            let weight = body.mass() * engine::STANDARD_GRAVITY;
            let (force, torque) = self.world.fluid_force(id);
            if force != Vec2::ZERO || torque != 0.0 {
                ui.weak(format!(
                    "Fluid force ({:.1}, {:.1}) N = {:.2}× weight · torque {:.2} N·m",
                    force.x,
                    force.y,
                    force.length() / weight,
                    torque
                ));
            }
        }
        if ui.button("Delete").clicked() {
            delete = true;
        }
        delete
    }

    pub(super) fn fluid_panel(&mut self, ui: &mut Ui) {
        let gpu_name = self.gpu_name().map(str::to_owned);
        let fluid = &mut self.world.fluid;
        let header = if fluid.is_empty() {
            "Fluid".to_owned()
        } else {
            format!("Fluid ({} particles)", fluid.len())
        };
        egui::CollapsingHeader::new(header)
            .id_salt("fluid_panel")
            .default_open(true)
            .show(ui, |ui| {
                let empty = fluid.is_empty();
                let p = &mut fluid.params;
                egui::Grid::new("fluid").num_columns(2).show(ui, |ui| {
                    ui.label("Spacing").on_hover_text(
                        "Distance between particles. Can only change while there is no fluid, \
                         since every particle's mass depends on it.",
                    );
                    ui.add_enabled(
                        empty,
                        egui::DragValue::new(&mut p.particle_spacing)
                            .speed(0.001)
                            .range(0.01..=0.5)
                            .suffix(" m"),
                    );
                    ui.end_row();
                    ui.label("Density")
                        .on_hover_text("Rest density. Water is 10 kg/m² (1000 kg/m³ × 1 cm).");
                    ui.add(
                        egui::DragValue::new(&mut p.rest_density)
                            .speed(0.1)
                            .range(0.1..=1000.0)
                            .suffix(" kg/m²"),
                    );
                    ui.end_row();
                    ui.label("Stiffness").on_hover_text(
                        "Numerical speed of sound. Higher compresses less but needs more substeps.",
                    );
                    ui.add(
                        egui::DragValue::new(&mut p.speed_of_sound)
                            .speed(0.5)
                            .range(5.0..=100.0)
                            .suffix(" m/s"),
                    );
                    ui.end_row();
                    ui.label("Viscosity");
                    ui.add(egui::Slider::new(&mut p.viscosity, 0.0..=1.0).logarithmic(true));
                    ui.end_row();
                    ui.label("Wall friction");
                    ui.add(egui::Slider::new(&mut p.wall_viscosity, 0.0..=2.0).logarithmic(true));
                    ui.end_row();
                    ui.label("Color by");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.color_by, ColorBy::Speed, "Speed");
                        ui.selectable_value(&mut self.color_by, ColorBy::Pressure, "Pressure");
                        ui.selectable_value(&mut self.color_by, ColorBy::Temperature, "Temperature");
                    });
                    ui.end_row();
                });
                match &gpu_name {
                    Some(name) => {
                        ui.checkbox(&mut self.use_gpu, "Simulate on the GPU")
                            .on_hover_text(format!("Run the fluid solver as compute shaders on {name}."));
                    }
                    None => {
                        ui.add_enabled(false, egui::Checkbox::new(&mut false, "Simulate on the GPU"))
                            .on_disabled_hover_text(if cfg!(target_arch = "wasm32") {
                                "The GPU fluid solver needs the desktop app for now."
                            } else {
                                "This GPU (or driver) doesn't support compute shaders."
                            });
                    }
                }
                ui.checkbox(&mut self.show_fluid_forces, "Show forces on bodies")
                    .on_hover_text(
                        "An arrow from each body's center showing the fluid's push (buoyancy and drag), \
                         drawn so the body's own weight would be 1 m long.",
                    );
                if !fluid.is_empty() {
                    ui.weak(format!(
                        "{} substeps per step · max compression {:.1}%",
                        fluid.stats.substeps,
                        fluid.stats.max_compression * 100.0
                    ));
                    if ui.button("Clear fluid").clicked() {
                        fluid.clear();
                    }
                }
            });
    }

    pub(super) fn status_bar(&mut self, ui: &mut Ui) {
        if self.mode == Mode::Orbits {
            ui.horizontal(|ui| {
                self.orbits.status_bar(ui);
                ui.separator();
                ui.weak(&self.status);
            });
            return;
        }
        if self.mode == Mode::ThreeD {
            ui.horizontal(|ui| {
                self.view3d.status_bar(ui);
                ui.separator();
                ui.weak(&self.status);
            });
            return;
        }
        ui.horizontal(|ui| {
            ui.label(if self.running { "Running" } else { "Paused" });
            ui.separator();
            ui.label(format!("t = {:.2} s", self.world.time));
            ui.separator();
            ui.label(format!("{} bodies", self.world.bodies.len()));
            if !self.world.fluid.is_empty() {
                ui.separator();
                ui.label(format!(
                    "{} particles · {} substeps · {}",
                    self.world.fluid.len(),
                    self.world.fluid.stats.substeps,
                    self.solver_label()
                ));
            }
            ui.separator();
            // abs(): an empty sum is -0.0, which would print as "-0.00".
            ui.label(format!("KE {:.2} J", self.world.kinetic_energy().abs()));
            ui.separator();
            ui.label(format!(
                "{:.0} Hz × {} steps/frame",
                1.0 / self.timestep.dt,
                self.steps_last_frame
            ));
            if self.snap {
                ui.separator();
                ui.label(format!("snap {} m", self.snap_step()));
            }
            if let Some(p) = self.cursor_world {
                ui.separator();
                ui.label(format!("cursor ({:.2}, {:.2}) m", p.x, p.y));
            }
            ui.separator();
            ui.weak(&self.status);
        });
    }

    pub(super) fn paste_window(&mut self, ctx: &egui::Context) {
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
}

/// "pyramid" -> "Pyramid", "dam_break" -> "Dam break".
pub(crate) fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect::<String>())
        .unwrap_or_default()
        .replace('_', " ")
        .replace("Newtons", "Newton's")
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
