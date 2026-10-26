//! The Fluids tab's toolbar, sidebar, status bar, and 3D view.

use super::{
    BOX_BODIES, ColorBy, Environment, FluidsEditor, Grab, SPACINGS, SolidMaterial, SolidShape,
    Tool, celsius, graph::format_energy, pick_particle, ray_plane,
};
use crate::view3d::render::{BodyCallback, CameraView, Instance, SoftMeshes, SurfaceFrame};
use crate::view3d::surface::{SurfaceView, particles_colored};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui};
use engine3d::fluid::Liquid;
use engine3d::{BodyId, Shape, Vec3};

const GLASS: Color32 = Color32::from_rgb(0x8C, 0x96, 0xA0);
const SELECTED: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);
const PROBE: Color32 = Color32::from_rgb(0xC8, 0x10, 0x2E);
/// Speed that gets the palest tint, m/s.
const SPEED_SCALE: f32 = 2.0;
/// Fastest a dragged solid is thrown, m/s.
const MAX_THROW: f32 = 10.0;

impl FluidsEditor {
    pub fn toolbar(&mut self, ui: &mut Ui) {
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
            .on_hover_text("Back to when Play was pressed (R)")
            .clicked()
        {
            self.reset();
        }
        ui.add(
            egui::Slider::new(&mut self.settings.time_scale, 0.05..=1.0)
                .logarithmic(true)
                .max_decimals(2),
        )
        .on_hover_text("Slow motion");
        ui.label("speed");
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
            .on_hover_text("Ctrl+Shift+Z")
            .clicked()
        {
            self.redo();
        }
        ui.separator();
        ui.label("View");
        for (label, yaw, pitch) in [("3D", 0.55, 0.5), ("Top", 0.0, 1.56), ("Side", 0.0, 0.0)] {
            if ui.button(label).clicked() {
                self.camera.yaw = yaw;
                self.camera.pitch = pitch;
                let s = self.box_size();
                self.camera.target = Vec3::new(0.0, 0.4 * s, 0.0);
            }
        }
    }

    pub fn status_bar(&mut self, ui: &mut Ui) {
        ui.label(if self.running { "Running" } else { "Paused" });
        ui.separator();
        ui.label(format!("t = {:.2} s", self.world.time));
        ui.separator();
        let fluid = &self.world.fluid;
        ui.label(format!(
            "{} particles · {} substeps",
            fluid.len(),
            fluid.stats.substeps
        ));
        if self.running {
            ui.separator();
            ui.label(format!("{:.2}× real time", self.speed))
                .on_hover_text(
                    "Simulated seconds per real second. Below the speed slider when the \
                 solver can't keep up (fewer or coarser particles run faster).",
                );
        }
        ui.separator();
        ui.label(match self.settings.environment {
            Environment::Vacuum => "Vacuum",
            Environment::Air => "Air, 1 atm",
        });
    }

    pub fn sidebar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Tools");
            self.tool_buttons(ui);
            self.tool_options(ui);
            ui.separator();
            self.box_panel(ui);
            ui.separator();
            self.view_panel(ui);
            ui.separator();
            self.readout_panel(ui);
            if let Some(id) = self.selected() {
                ui.separator();
                self.inspector(ui, id);
            }
        });
    }

    fn tool_buttons(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for (tool, label, hint) in [
                (
                    Tool::Select,
                    "Select (V)",
                    "Select and drag solids and plates",
                ),
                (
                    Tool::Block,
                    "Block (B)",
                    "Drag a rectangle on the floor: a block of liquid",
                ),
                (
                    Tool::Ball,
                    "Ball (C)",
                    "Click: a ball of liquid dropped from above",
                ),
                (
                    Tool::Solid,
                    "Solid (O)",
                    "Click: a floating or sinking solid",
                ),
                (
                    Tool::Plate,
                    "Plate (P)",
                    "Click the floor: a heater or cooler plate",
                ),
                (
                    Tool::Heat,
                    "Heat (H)",
                    "Hold over the liquid to heat or cool it",
                ),
                (
                    Tool::Probe,
                    "Probe (I)",
                    "Hover to read the liquid; click to pin",
                ),
            ] {
                ui.selectable_value(&mut self.tool, tool, label)
                    .on_hover_text(hint);
            }
        });
    }

    fn tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.settings;
        match self.tool {
            Tool::Select => {
                ui.weak("Click a solid or plate to select it; drag to move it. Right-drag orbits the view, the wheel zooms.");
            }
            Tool::Block | Tool::Ball => {
                s.liquid.picker(ui, &mut s.liquid_celsius);
                if self.tool == Tool::Block {
                    length_row(ui, "Height", &mut s.block_height, 0.02..=2.0);
                    ui.weak("Drag on the floor; the block stands on the liquid already there. A click pours one the ball's width.");
                } else {
                    length_row(ui, "Radius", &mut s.ball_radius, 0.03..=0.5);
                    length_row(ui, "Drop from", &mut s.drop_height, 0.0..=1.0);
                }
                let mut fill = false;
                ui.horizontal(|ui| {
                    ui.label("Fill to");
                    ui.add(
                        egui::Slider::new(&mut s.fill_level, 0.02..=1.0)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                    );
                    fill = ui
                        .button("Fill")
                        .on_hover_text(
                            "Fill the box with this liquid up to the level, wherever there's room",
                        )
                        .clicked();
                });
                if fill {
                    self.fill_to_level();
                }
                crate::liquid_ui::hint(ui);
            }
            Tool::Solid => {
                ui.horizontal_wrapped(|ui| {
                    for m in SolidMaterial::ALL {
                        ui.selectable_value(&mut s.solid, m, m.name())
                            .on_hover_text(format!("{:.0} kg/m³", m.density()));
                    }
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut s.solid_shape, SolidShape::Ball, "Ball");
                    ui.selectable_value(&mut s.solid_shape, SolidShape::Cube, "Cube");
                });
                length_row(ui, "Size", &mut s.solid_size, 0.03..=0.5);
                ui.weak(
                    "Floats if it's lighter than the liquid: steel floats on mercury, \
                     gold sinks through it, plastic floats on oil only just.",
                );
            }
            Tool::Plate => {
                length_row(ui, "Size", &mut s.plate_size, 0.05..=2.0);
                ui.horizontal(|ui| {
                    ui.label("Temperature");
                    ui.add(
                        egui::DragValue::new(&mut s.plate_celsius)
                            .range(-20.0..=99.0)
                            .speed(0.5)
                            .suffix(" °C"),
                    );
                });
                ui.weak(
                    "Heats or cools the liquid touching it by natural convection \
                     (about 1 kW/m²·K for water). Up to 99 °C until boiling is in.",
                );
            }
            Tool::Heat => {
                ui.horizontal(|ui| {
                    ui.label("Power");
                    ui.add(
                        egui::DragValue::new(&mut s.heat_power)
                            .range(-20_000.0..=20_000.0)
                            .speed(10.0)
                            .suffix(" W"),
                    )
                    .on_hover_text("Negative cools. A kettle element is about 2 kW.");
                });
                length_row(ui, "Radius", &mut s.heat_radius, 0.02..=0.5);
                ui.weak(
                    "Hold the mouse on the liquid while it runs. The energy goes \
                     exactly into the liquid inside the radius (shown under Totals).",
                );
            }
            Tool::Probe => {
                ui.weak("Hover over the liquid to read it; click to pin a probe there.");
                if self.probe_pin.is_some() && ui.button("Remove probe").clicked() {
                    self.probe_pin = None;
                }
            }
        }
    }

    fn box_panel(&mut self, ui: &mut Ui) {
        ui.heading("Box");
        let mut env = self.settings.environment;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut env, Environment::Air, "Air, 1 atm, 20 °C")
                .on_hover_text("Room air above the liquid: it takes heat off the surface");
            ui.selectable_value(&mut env, Environment::Vacuum, "Vacuum")
                .on_hover_text("Nothing above the liquid: no heat leaves the surface");
        });
        if env != self.settings.environment {
            self.set_environment(env);
        }
        let mut hold = self.settings.hold_walls;
        let mut wall = self.settings.wall_celsius;
        ui.horizontal(|ui| {
            ui.checkbox(&mut hold, "Walls held at")
                .on_hover_text("Otherwise the walls are insulated");
            ui.add_enabled(
                hold,
                egui::DragValue::new(&mut wall)
                    .range(-20.0..=99.0)
                    .speed(0.5)
                    .suffix(" °C"),
            );
        });
        if hold != self.settings.hold_walls || wall != self.settings.wall_celsius {
            self.settings.hold_walls = hold;
            self.settings.wall_celsius = wall;
            self.apply_wall_temperature();
        }
        egui::Grid::new("fluids box").num_columns(2).show(ui, |ui| {
            ui.label("Size");
            ui.add(
                egui::DragValue::new(&mut self.settings.box_size)
                    .range(0.25..=2.0)
                    .speed(0.01)
                    .suffix(" m"),
            )
            .on_hover_text("Inside width, depth, and height (takes effect with New box)");
            ui.end_row();
            ui.label("Spacing");
            egui::ComboBox::from_id_salt("fluids spacing")
                .selected_text(format!("{:.1} cm", self.settings.spacing * 100.0))
                .show_ui(ui, |ui| {
                    for d in SPACINGS {
                        let full = (self.settings.box_size / d).powi(3);
                        ui.selectable_value(
                            &mut self.settings.spacing,
                            d,
                            format!("{:.1} cm ({:.0}k full)", d * 100.0, full / 1000.0),
                        );
                    }
                })
                .response
                .on_hover_text(
                    "Distance between particles (takes effect with New box). Halving it \
                     means 8× the particles: finer but much slower.",
                );
            ui.end_row();
        });
        ui.horizontal(|ui| {
            if ui
                .button("New box")
                .on_hover_text("Empty box with these settings")
                .clicked()
            {
                self.new_box();
            }
            if ui.button("Drain liquid").clicked() {
                self.world.fluid.clear();
                self.say("Drained.");
            }
        });
    }

    fn view_panel(&mut self, ui: &mut Ui) {
        ui.heading("View");
        ui.checkbox(&mut self.settings.smooth, "Smooth surfaces")
            .on_hover_text("Draw the liquid as smooth, see-through surfaces instead of balls");
        ui.horizontal_wrapped(|ui| {
            ui.label("Color by");
            for (c, label) in [
                (ColorBy::Liquid, "Liquid"),
                (ColorBy::Temperature, "Temperature"),
                (ColorBy::Speed, "Speed"),
                (ColorBy::Pressure, "Pressure"),
                (ColorBy::Density, "Density"),
            ] {
                ui.selectable_value(&mut self.settings.color_by, c, label);
            }
        });
        if let Some(legend) = self.legend() {
            ui.weak(legend);
        }
    }

    fn legend(&self) -> Option<String> {
        Some(match self.settings.color_by {
            ColorBy::Liquid => return None,
            ColorBy::Temperature => {
                "Blue 0 °C · each liquid's color at 20 °C · red at 100 °C".into()
            }
            ColorBy::Speed => format!("Dark at rest · pale at {SPEED_SCALE} m/s and faster"),
            ColorBy::Pressure => format!(
                "Gauge pressure: dark 0 · bright at {:.1} kPa (the box's depth of water)",
                self.full_pressure() / 1000.0
            ),
            ColorBy::Density => {
                "Blue 900 kg/m³ (oil) · white 1000 (water) · red 1100 and up".into()
            }
        })
    }

    fn full_pressure(&self) -> f32 {
        1000.0 * 9.81 * self.box_size()
    }

    fn readout_panel(&mut self, ui: &mut Ui) {
        ui.heading("Probe");
        let point = self.probe_hover.or(self.probe_pin);
        match point.and_then(|p| self.world.fluid.probe(p)) {
            Some(p) => {
                egui::Grid::new("probe").num_columns(2).show(ui, |ui| {
                    let s = if self.probe_hover.is_some() {
                        "here"
                    } else {
                        "pinned"
                    };
                    ui.label("Liquid");
                    ui.label(format!("{} ({s})", p.liquid.name()));
                    ui.end_row();
                    ui.label("Temperature");
                    ui.label(format!("{:.2} °C", celsius(p.temperature)));
                    ui.end_row();
                    ui.label("Pressure");
                    let absolute = p.pressure + self.settings.environment.pressure();
                    ui.label(format!("{:.2} kPa", absolute / 1000.0))
                        .on_hover_text(format!(
                            "Absolute; {:.0} Pa above the surroundings",
                            p.pressure
                        ));
                    ui.end_row();
                    ui.label("Density");
                    ui.label(format!("{:.0} kg/m³", p.density));
                    ui.end_row();
                    ui.label("Speed");
                    ui.label(format!("{:.3} m/s", p.velocity.length()));
                    ui.end_row();
                    ui.label("Kinetic energy");
                    ui.label(format!(
                        "{:.2} J/L",
                        0.5 * p.density * p.velocity.length_squared() / 1000.0
                    ));
                    ui.end_row();
                });
            }
            None => {
                ui.weak("Use the Probe tool (I) and point at the liquid.");
            }
        }

        ui.heading("Totals");
        let fluid = &self.world.fluid;
        let masses = fluid.mass_by_liquid();
        egui::Grid::new("totals").num_columns(2).show(ui, |ui| {
            for liquid in Liquid::ALL {
                let m = masses[liquid.index()];
                if m > 0.0 {
                    ui.label(liquid.name());
                    ui.label(format!(
                        "{m:.2} kg · {:.2} L",
                        m / liquid.density(0.0) * 1000.0
                    ));
                    ui.end_row();
                }
            }
            if let Some(s) = self.graph.latest() {
                for (label, v, hint) in [
                    ("Kinetic", s.kinetic, "½ m v², summed"),
                    ("Potential", s.potential, "m g y above the floor"),
                    ("Thermal", s.thermal, "m c (T − 20 °C), summed"),
                    ("Heat brush", s.brush, "Energy the heat brush has put in"),
                ] {
                    ui.label(label).on_hover_text(hint);
                    ui.label(format_energy(v));
                    ui.end_row();
                }
                ui.label("Temperature");
                ui.label(format!(
                    "{:.2} °C mean · {:.2} °C max",
                    s.mean_celsius, s.max_celsius
                ));
                ui.end_row();
            }
        });
        self.graph.show(ui);
    }

    fn inspector(&mut self, ui: &mut Ui, id: BodyId) {
        let Some(b) = self.world.body_mut(id) else {
            return;
        };
        if let Some(t) = b.temperature.as_mut() {
            ui.heading("Plate");
            let mut c = celsius(*t);
            ui.horizontal(|ui| {
                ui.label("Temperature");
                ui.add(
                    egui::DragValue::new(&mut c)
                        .range(-20.0..=99.0)
                        .speed(0.5)
                        .suffix(" °C"),
                );
            });
            *t = c + super::KELVIN;
        } else {
            let material = SolidMaterial::of_density(b.density);
            ui.heading(material.name());
            let mut m = material;
            ui.horizontal_wrapped(|ui| {
                for option in SolidMaterial::ALL {
                    ui.selectable_value(&mut m, option, option.name());
                }
            });
            if m != material {
                b.density = m.density();
            }
            ui.label(format!("{:.0} kg/m³ · {:.3} kg", b.density, b.mass()));
            if let Some((force, _)) = self.world.fluid_forces.get(id.0) {
                ui.label(format!("Force from the liquid: {:.2} N up", force.y))
                    .on_hover_text("Buoyancy plus drag, last step");
            }
        }
        if ui.button("Delete").on_hover_text("Delete key").clicked() {
            self.delete_selected();
        }
    }

    // ---------------------------------------------------------------------
    // The view

    pub fn canvas(&mut self, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;

        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (-scroll * 0.0015).exp() / pinch;
            if factor != 1.0 {
                self.camera.zoom(factor);
            }
        }
        let shift = ui.input(|i| i.modifiers.shift);
        if response.dragged_by(PointerButton::Middle)
            || (shift && response.dragged_by(PointerButton::Secondary))
        {
            self.camera.pan(rect, response.drag_delta());
        } else if response.dragged_by(PointerButton::Secondary) {
            self.camera.orbit(response.drag_delta());
        }

        self.heating = None;
        self.probe_hover = None;
        let hover = response.hover_pos();
        match self.tool {
            Tool::Select => self.select_input(ui, &response, rect),
            Tool::Block => self.block_input(&response, rect),
            Tool::Ball | Tool::Solid | Tool::Plate => {
                if response.clicked_by(PointerButton::Primary)
                    && let Some(p) = response.interact_pointer_pos()
                    && let Some(at) = self.floor_point(rect, p)
                {
                    match self.tool {
                        Tool::Ball => self.drop_ball(at),
                        Tool::Solid => self.drop_solid(at),
                        _ => self.place_plate(at),
                    }
                }
            }
            Tool::Heat => {
                if response.is_pointer_button_down_on()
                    && let Some(p) = response.interact_pointer_pos()
                {
                    self.heating = self.liquid_point(rect, p);
                    if self.heating.is_some() && !self.running {
                        self.say("Press Space to play; the brush heats while time runs.");
                    }
                }
            }
            Tool::Probe => {
                if let Some(p) = hover {
                    self.probe_hover = self.liquid_point(rect, p);
                }
                if response.clicked_by(PointerButton::Primary) {
                    self.probe_pin = self.probe_hover;
                }
            }
        }

        let dark = crate::theme::is_dark(ui.ctx());
        painter.rect_filled(rect, 0.0, crate::theme::sky(dark));
        if self.gpu_render {
            painter.add(self.body_callback(rect, dark).into_shape(rect));
        } else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "The Fluids tab needs GPU rendering (wgpu), which isn't available here.",
                egui::FontId::proportional(16.0),
                ui.visuals().weak_text_color(),
            );
        }
        self.draw_glass(&painter, rect, dark);
        self.draw_overlays(&painter, rect, hover);

        if self.running || response.dragged() || response.hovered() {
            ui.ctx().request_repaint();
        }
    }

    /// The point on the floor under the mouse, inside the box.
    fn floor_point(&mut self, rect: Rect, p: Pos2) -> Option<Vec3> {
        let (origin, dir) = self.camera.ray(rect, p);
        let (lo, hi) = self.inner();
        let hit = ray_plane(origin, dir, Vec3::ZERO, Vec3::Y)
            .filter(|h| h.x >= lo.x && h.x <= hi.x && h.z >= lo.z && h.z <= hi.z);
        if hit.is_none() {
            self.say("Point at the box's floor.");
        }
        hit
    }

    /// The liquid under the mouse: the nearest particle along the ray.
    fn liquid_point(&self, rect: Rect, p: Pos2) -> Option<Vec3> {
        let (origin, dir) = self.camera.ray(rect, p);
        let d = self.world.fluid.params.particle_spacing;
        pick_particle(&self.world.fluid.positions, origin, dir, 0.6 * d)
    }

    fn block_input(&mut self, response: &egui::Response, rect: Rect) {
        if response.drag_started_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            self.block_start = self.floor_point(rect, p);
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            if let (Some(a), Some(p)) = (self.block_start.take(), response.interact_pointer_pos()) {
                let (origin, dir) = self.camera.ray(rect, p);
                if let Some(b) = ray_plane(origin, dir, Vec3::ZERO, Vec3::Y) {
                    self.pour_block(a, b);
                }
            }
        } else if response.clicked_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
            && let Some(a) = self.floor_point(rect, p)
        {
            self.pour_block(a, a);
        }
    }

    fn select_input(&mut self, ui: &Ui, response: &egui::Response, rect: Rect) {
        if response.drag_started_by(PointerButton::Primary)
            && let Some(p) = ui
                .input(|i| i.pointer.press_origin())
                .or(response.hover_pos())
        {
            let (origin, dir) = self.camera.ray(rect, p);
            self.grab = self
                .world
                .raycast(origin, dir)
                .filter(|(id, _)| id.0 >= BOX_BODIES)
                .and_then(|(id, t)| {
                    let point = origin + t * dir;
                    let (position, is_static) = {
                        let body = self.world.body(id)?;
                        (body.position, body.is_static)
                    };
                    if is_static && self.running {
                        return None;
                    }
                    let handle = self.world.handle(id)?;
                    self.selection = Some(handle);
                    Some(Grab {
                        handle,
                        height: point.y,
                        offset: position - point,
                        prev: position,
                    })
                });
        }
        if response.dragged_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
            && let Some(grab) = self.grab.as_mut()
        {
            let dt = ui.input(|i| i.stable_dt).max(1e-4);
            let (origin, dir) = self.camera.ray(rect, p);
            if let Some(hit) = ray_plane(origin, dir, Vec3::new(0.0, grab.height, 0.0), Vec3::Y)
                && let Some(body) = self
                    .world
                    .find(grab.handle)
                    .and_then(|id| self.world.body_mut(id))
            {
                let target = hit + grab.offset;
                body.position.x = target.x;
                body.position.z = target.z;
                body.angular_velocity = Vec3::ZERO;
                body.velocity = if self.running && !body.is_static {
                    ((body.position - grab.prev) / dt).clamp_length_max(MAX_THROW)
                } else {
                    Vec3::ZERO
                };
                grab.prev = body.position;
            }
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            let dropped = self.grab.take().is_some();
            if dropped && !self.running {
                let bodies = self.world.bodies.clone();
                self.world.fluid.project_out_of_bodies(&bodies);
            }
        }
        if response.clicked_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            let (origin, dir) = self.camera.ray(rect, p);
            let hit = self
                .world
                .raycast(origin, dir)
                .map(|(id, _)| id)
                .filter(|id| id.0 >= BOX_BODIES);
            self.selection = hit.and_then(|id| self.world.handle(id));
        }
    }

    /// Color of particle `i` for the current color-by mode.
    fn particle_color(&self, i: usize) -> Color32 {
        let fluid = &self.world.fluid;
        let liquid = fluid.liquid(i);
        let salinity = fluid.salinity.get(i).copied().unwrap_or(0.0);
        let own = || {
            let [r, g, b] = liquid.color(salinity);
            Color32::from_rgb(r, g, b)
        };
        match self.settings.color_by {
            ColorBy::Liquid => own(),
            ColorBy::Temperature => {
                crate::liquid_ui::temperature_color(liquid, salinity, fluid.temperature(i))
            }
            ColorBy::Speed => {
                let v = fluid.velocities.get(i).map_or(0.0, |v| v.length());
                ramp(own(), Color32::WHITE, v / SPEED_SCALE)
            }
            ColorBy::Pressure => {
                let p = fluid.pressures().get(i).copied().unwrap_or(0.0);
                ramp(
                    Color32::from_rgb(0x1A, 0x23, 0x40),
                    Color32::from_rgb(0xF1, 0xBE, 0x48),
                    p / self.full_pressure(),
                )
            }
            ColorBy::Density => {
                let rho = fluid
                    .densities()
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| liquid.density(salinity));
                if rho < 1000.0 {
                    ramp(
                        Color32::from_rgb(0x3B, 0x6E, 0xC8),
                        Color32::WHITE,
                        (rho - 900.0) / 100.0,
                    )
                } else {
                    ramp(
                        Color32::WHITE,
                        Color32::from_rgb(0xC8, 0x10, 0x2E),
                        (rho - 1000.0) / 100.0,
                    )
                }
            }
        }
    }

    fn body_callback(&self, rect: Rect, dark: bool) -> BodyCallback {
        let forward = (self.camera.target - self.camera.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        let selected = self.selected();
        let (mut cubes, mut spheres) = (Vec::new(), Vec::new());
        for (i, b) in self.world.bodies.iter().enumerate() {
            // The walls are glass: drawn as edges, not solid.
            if (1..BOX_BODIES).contains(&i) {
                continue;
            }
            let color = if i == 0 {
                crate::theme::static_fill(dark, true).to_array()
            } else if let Some(t) = b.temperature {
                plate_color(t)
            } else {
                SolidMaterial::of_density(b.density).color()
            };
            let (scale, list) = match b.shape {
                Shape::Sphere { radius } => (Vec3::splat(radius), &mut spheres),
                Shape::Cuboid { half_extents } => (half_extents, &mut cubes),
            };
            list.push(Instance {
                position: b.position.to_array(),
                checker: if i == 0 { 1.0 } else { 0.0 },
                scale: scale.to_array(),
                highlight: if selected == Some(BodyId(i)) {
                    1.0
                } else {
                    0.0
                },
                rotation: b.orientation.to_array(),
                color,
            });
        }
        let fluid = &self.world.fluid;
        let colors: Vec<Color32> = (0..fluid.len()).map(|i| self.particle_color(i)).collect();
        let smooth = self.settings.smooth && !fluid.is_empty();
        let particles = if smooth {
            Vec::new()
        } else {
            let radius = Vec3::splat(0.55 * fluid.params.particle_spacing).to_array();
            fluid
                .positions
                .iter()
                .zip(&colors)
                .map(|(p, c)| Instance {
                    position: p.to_array(),
                    checker: 0.0,
                    scale: radius,
                    highlight: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    color: c.to_array(),
                })
                .collect()
        };
        let mut callback = BodyCallback::new(
            cubes,
            spheres,
            particles,
            SoftMeshes::default(),
            Vec::new(),
            CameraView {
                view_proj: self.camera.view_proj(rect),
                eye: self.camera.eye(),
                right,
                up,
            },
        );
        if smooth {
            let radius = 0.9 * fluid.params.particle_spacing;
            let sky = crate::theme::sky(dark);
            let metal = self.settings.color_by == ColorBy::Liquid;
            callback = callback.with_surface(SurfaceFrame {
                particles: particles_colored(fluid, radius, metal, |i| {
                    let c = colors[i];
                    [c.r(), c.g(), c.b()]
                }),
                view: SurfaceView {
                    view: self.camera.view(),
                    proj: self.camera.proj(rect),
                    sky: [
                        f32::from(sky.r()) / 255.0,
                        f32::from(sky.g()) / 255.0,
                        f32::from(sky.b()) / 255.0,
                    ],
                    light: Vec3::new(0.45, 0.8, 0.35),
                    radius,
                },
                rect,
            });
        }
        callback
    }

    /// The glass walls: the box's twelve edges, the back ones fainter.
    fn draw_glass(&self, painter: &egui::Painter, rect: Rect, dark: bool) {
        let (lo, hi) = self.inner();
        let corner = |i: usize| {
            Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            )
        };
        let eye = self.camera.eye();
        let center = 0.5 * (lo + hi);
        let color = if dark {
            Color32::from_rgb(0xA0, 0xA8, 0xB4)
        } else {
            GLASS
        };
        for a in 0..8usize {
            for bit in [1usize, 2, 4] {
                let b = a | bit;
                if b == a {
                    continue;
                }
                let (pa, pb) = (corner(a), corner(b));
                // Edges on the far side of the box from the camera are fainter.
                let mid = 0.5 * (pa + pb);
                let near = (mid - center).dot(eye - center) > 0.0;
                let stroke = if near {
                    Stroke::new(1.5, color)
                } else {
                    Stroke::new(1.0, color.gamma_multiply(0.45))
                };
                if let (Some(sa), Some(sb)) =
                    (self.camera.project(rect, pa), self.camera.project(rect, pb))
                {
                    painter.line_segment([sa, sb], stroke);
                }
            }
        }
    }

    fn draw_overlays(&self, painter: &egui::Painter, rect: Rect, hover: Option<Pos2>) {
        // The fill level and the block being dragged.
        if matches!(self.tool, Tool::Block | Tool::Ball) {
            let (lo, hi) = self.inner();
            let y = self.settings.fill_level * self.box_size();
            let ring = [
                Vec3::new(lo.x, y, lo.z),
                Vec3::new(hi.x, y, lo.z),
                Vec3::new(hi.x, y, hi.z),
                Vec3::new(lo.x, y, hi.z),
            ];
            self.polyline(
                painter,
                rect,
                &ring,
                Stroke::new(1.0, GLASS.gamma_multiply(0.6)),
            );
        }
        if let (Some(a), Some(p)) = (self.block_start, hover) {
            let (origin, dir) = self.camera.ray(rect, p);
            if let Some(b) = ray_plane(origin, dir, Vec3::ZERO, Vec3::Y) {
                let ring = [
                    Vec3::new(a.x, 0.0, a.z),
                    Vec3::new(b.x, 0.0, a.z),
                    Vec3::new(b.x, 0.0, b.z),
                    Vec3::new(a.x, 0.0, b.z),
                ];
                self.polyline(painter, rect, &ring, Stroke::new(2.0, SELECTED));
            }
        }
        // Heat brush and probes.
        if let Some(c) = self.heating
            && let Some(s) = self.camera.project(rect, c)
        {
            let r = self.settings.heat_radius
                / self
                    .camera
                    .units_per_pixel(rect, (c - self.camera.eye()).length());
            let color = if self.settings.heat_power >= 0.0 {
                Color32::from_rgb(0xE8, 0x3A, 0x14)
            } else {
                Color32::from_rgb(0x4F, 0xA8, 0xF0)
            };
            painter.circle_stroke(s, r, Stroke::new(2.0, color));
        }
        for (p, filled) in [(self.probe_pin, true), (self.probe_hover, false)] {
            if let Some(p) = p
                && let Some(s) = self.camera.project(rect, p)
            {
                if filled {
                    painter.circle_filled(s, 4.0, PROBE);
                } else {
                    painter.circle_stroke(s, 6.0, Stroke::new(1.5, PROBE));
                }
            }
        }
    }

    fn polyline(&self, painter: &egui::Painter, rect: Rect, ring: &[Vec3], stroke: Stroke) {
        for k in 0..ring.len() {
            let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
            if let (Some(sa), Some(sb)) =
                (self.camera.project(rect, a), self.camera.project(rect, b))
            {
                painter.line_segment([sa, sb], stroke);
            }
        }
    }
}

/// Plates: red-orange when hot, blue when cold, grey at room temperature.
fn plate_color(kelvin: f32) -> [u8; 4] {
    let dt = kelvin - super::ROOM;
    let c = if dt >= 0.0 {
        ramp(
            Color32::from_rgb(0x8A, 0x8E, 0x94),
            Color32::from_rgb(0xE8, 0x3A, 0x14),
            dt / 60.0,
        )
    } else {
        ramp(
            Color32::from_rgb(0x8A, 0x8E, 0x94),
            Color32::from_rgb(0x4F, 0xA8, 0xF0),
            -dt / 30.0,
        )
    };
    c.to_array()
}

fn ramp(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

fn length_row(ui: &mut Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(
            egui::DragValue::new(value)
                .range(range)
                .speed(0.005)
                .suffix(" m"),
        );
    });
}
