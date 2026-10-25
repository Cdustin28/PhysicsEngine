//! 3D toolbar, sidebar, status bar, and viewport.

use super::render::{BodyCallback, Instance};
use super::{Editor3d, Grab, MAX_THROW_SPEED, Tool, ray_plane};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Sense, Ui};
use engine3d::{BodyId, RigidBody, Shape, Vec3, presets};

const CARDINAL: Color32 = Color32::from_rgb(0xC8, 0x10, 0x2E);
const STATIC_FILL: Color32 = Color32::from_rgb(0xB4, 0xB6, 0xB9);
const CONTACT_COLOR: Color32 = Color32::from_rgb(0x1F, 0x4E, 0x9A);
const SKY: Color32 = Color32::from_rgb(0xEE, 0xF1, 0xF5);

impl Editor3d {
    /// Play/Pause, Step, Reset, speed, Undo/Redo, New, Examples.
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
        ui.separator();
        if ui
            .button("New")
            .on_hover_text("Empty world with a floor")
            .clicked()
        {
            self.replace_world(presets::with_floor());
        }
        let mut chosen = None;
        egui::ComboBox::from_id_salt("examples3d")
            .selected_text("Examples")
            .show_ui(ui, |ui| {
                for &(name, make) in presets::ALL {
                    if ui
                        .selectable_label(false, crate::app::title_case(name))
                        .clicked()
                    {
                        chosen = Some((name, make));
                    }
                }
            });
        if let Some((name, make)) = chosen {
            self.replace_world(make());
            self.status = format!(
                "Loaded the {} example. Press Space to play.",
                crate::app::title_case(name)
            );
        }
    }

    pub fn sidebar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Tools");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Select, "Select (V)");
                ui.selectable_value(&mut self.tool, Tool::Sphere, "Sphere (C)");
                ui.selectable_value(&mut self.tool, Tool::Box, "Box (B)");
            });
            self.tool_options(ui);

            ui.separator();
            self.world_panel(ui);

            ui.separator();
            ui.heading("Selection");
            match self.selected() {
                Some(id) => {
                    if self.inspector(ui, id) {
                        self.delete_selected();
                    }
                }
                None => {
                    ui.weak("Click a body to select it.");
                }
            }

            ui.separator();
            ui.collapsing("Controls", |ui| {
                ui.label("Left-click: select, or place with Sphere/Box");
                ui.label("Left-drag: move on the ground plane");
                ui.label("Shift + left-drag: move up and down");
                ui.label("Right-drag: orbit");
                ui.label("Shift + right-drag or middle-drag: pan");
                ui.label("Scroll: zoom");
                ui.label("F: focus on the selection");
            });
        });
    }

    fn tool_options(&mut self, ui: &mut Ui) {
        let nb = &mut self.new_body;
        match self.tool {
            Tool::Select => {
                ui.weak("Drag bodies to move them. While running, let go to throw.");
            }
            Tool::Sphere | Tool::Box => {
                egui::Grid::new("new_body3d").num_columns(2).show(ui, |ui| {
                    if self.tool == Tool::Sphere {
                        length_row(ui, "Radius", &mut nb.radius);
                    } else {
                        length_row(ui, "Width", &mut nb.size.x);
                        length_row(ui, "Height", &mut nb.size.y);
                        length_row(ui, "Depth", &mut nb.size.z);
                    }
                    density_row(ui, &mut nb.density);
                    ui.label("Static");
                    ui.checkbox(&mut nb.is_static, "");
                    ui.end_row();
                });
                ui.weak("Click in the view to place.");
            }
        }
    }

    fn world_panel(&mut self, ui: &mut Ui) {
        ui.heading("World");
        egui::Grid::new("world3d").num_columns(2).show(ui, |ui| {
            ui.label("Gravity y");
            ui.add(
                egui::DragValue::new(&mut self.world.gravity.y)
                    .speed(0.1)
                    .suffix(" m/s²"),
            );
            ui.end_row();
            ui.label("Iterations");
            ui.add(egui::Slider::new(
                &mut self.world.solver.velocity_iterations,
                1..=50,
            ));
            ui.end_row();
        });
        ui.checkbox(&mut self.show_contacts, "Show contacts (D)");
    }

    /// Property editor for one body. Returns true if Delete was clicked.
    fn inspector(&mut self, ui: &mut Ui, id: BodyId) -> bool {
        let mut focus = false;
        let mut delete = false;
        let Some(body) = self.world.body_mut(id) else {
            return false;
        };
        egui::Grid::new("inspector3d")
            .num_columns(2)
            .show(ui, |ui| {
                match &mut body.shape {
                    Shape::Sphere { radius } => length_row(ui, "Radius", radius),
                    Shape::Cuboid { half_extents } => {
                        let mut size = *half_extents * 2.0;
                        length_row(ui, "Width", &mut size.x);
                        length_row(ui, "Height", &mut size.y);
                        length_row(ui, "Depth", &mut size.z);
                        *half_extents = size * 0.5;
                    }
                }
                vec3_row(ui, "Position", &mut body.position, " m");
                vec3_row(ui, "Velocity", &mut body.velocity, " m/s");
                if !body.is_static {
                    density_row(ui, &mut body.density);
                    ui.label("Mass");
                    ui.label(format!("{:.3} kg", body.mass()));
                    ui.end_row();
                }
                ui.label("Friction");
                ui.add(egui::Slider::new(&mut body.friction, 0.0..=1.5));
                ui.end_row();
                ui.label("Restitution");
                ui.add(egui::Slider::new(&mut body.restitution, 0.0..=1.0));
                ui.end_row();
                ui.label("Static");
                if ui.checkbox(&mut body.is_static, "").changed() {
                    body.velocity = Vec3::ZERO;
                    body.angular_velocity = Vec3::ZERO;
                }
                ui.end_row();
            });
        ui.horizontal(|ui| {
            focus = ui.button("Focus (F)").clicked();
            delete = ui.button("Delete").clicked();
        });
        if focus {
            self.focus_selected();
        }
        delete
    }

    pub fn status_bar(&mut self, ui: &mut Ui) {
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
    }

    // ---------------------------------------------------------------------
    // Viewport

    pub fn canvas(&mut self, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let shift = ui.input(|i| i.modifiers.shift);

        // --- Camera.
        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (-scroll * 0.0015).exp() / pinch;
            if factor != 1.0 {
                self.camera.zoom(factor);
            }
        }
        if response.dragged_by(PointerButton::Middle)
            || (shift && response.dragged_by(PointerButton::Secondary))
        {
            self.camera.pan(rect, response.drag_delta());
        } else if response.dragged_by(PointerButton::Secondary) {
            self.camera.orbit(response.drag_delta());
        }

        // --- Tools.
        match self.tool {
            Tool::Select => self.select_input(ui, &response, rect, shift),
            Tool::Sphere | Tool::Box => {
                if response.clicked_by(PointerButton::Primary)
                    && let Some(p) = response.interact_pointer_pos()
                {
                    self.place(rect, p);
                }
            }
        }

        // --- Drawing.
        painter.rect_filled(rect, 0.0, SKY);
        if self.gpu_render {
            painter.add(self.body_callback(rect).into_shape(rect));
        } else {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "The 3D view needs GPU rendering (wgpu), which isn't available here.",
                egui::FontId::proportional(16.0),
                Color32::DARK_GRAY,
            );
        }
        if self.show_contacts {
            self.draw_contacts(&painter, rect);
        }
        self.draw_axes(&painter, rect);

        if self.running || response.dragged() {
            ui.ctx().request_repaint();
        }
    }

    fn select_input(&mut self, ui: &Ui, response: &egui::Response, rect: Rect, shift: bool) {
        if response.drag_started_by(PointerButton::Primary) {
            let press = ui
                .input(|i| i.pointer.press_origin())
                .or(response.hover_pos());
            self.grab = press.and_then(|p| self.start_grab(rect, p, shift));
        }
        if response.dragged_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            let dt = ui.input(|i| i.stable_dt).max(1e-4);
            self.continue_grab(rect, p, dt);
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            self.grab = None;
        }
        if response.clicked_by(PointerButton::Primary)
            && let Some(p) = response.interact_pointer_pos()
        {
            let (origin, dir) = self.camera.ray(rect, p);
            let hit = self.world.raycast(origin, dir).map(|(id, _)| id);
            self.select(hit);
        }
    }

    /// Pick the body under `p` and start dragging it. Shift drags in the
    /// vertical plane facing the camera; otherwise in the horizontal plane.
    fn start_grab(&mut self, rect: Rect, p: Pos2, shift: bool) -> Option<Grab> {
        let (origin, dir) = self.camera.ray(rect, p);
        let (id, t) = self.world.raycast(origin, dir)?;
        self.select(Some(id));
        let handle = self.world.handle(id)?;
        let body = self.world.body(id)?;
        if body.is_static && self.running {
            return None;
        }
        let point = origin + t * dir;
        let plane_normal = if shift {
            let forward = self.camera.target - self.camera.eye();
            Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::Z)
        } else {
            Vec3::Y
        };
        Some(Grab {
            handle,
            plane_point: point,
            plane_normal,
            offset: body.position - point,
            prev_position: body.position,
        })
    }

    fn continue_grab(&mut self, rect: Rect, p: Pos2, dt: f32) {
        let Some(grab) = self.grab.as_mut() else {
            return;
        };
        let (origin, dir) = self.camera.ray(rect, p);
        let Some(hit) = ray_plane(origin, dir, grab.plane_point, grab.plane_normal) else {
            return;
        };
        let Some(body) = self
            .world
            .find(grab.handle)
            .and_then(|id| self.world.body_mut(id))
        else {
            self.grab = None;
            return;
        };
        let target = hit + grab.offset;
        body.position = target;
        body.angular_velocity = Vec3::ZERO;
        body.velocity = if self.running && !body.is_static {
            ((target - grab.prev_position) / dt).clamp_length_max(MAX_THROW_SPEED)
        } else {
            Vec3::ZERO
        };
        grab.prev_position = target;
    }

    /// Place a new body where the mouse points: on top of whatever is there,
    /// or on the ground plane y = 0.
    fn place(&mut self, rect: Rect, p: Pos2) {
        let Some(shape) = self.new_body.shape(self.tool) else {
            return;
        };
        let (origin, dir) = self.camera.ray(rect, p);
        let ground = self
            .world
            .raycast(origin, dir)
            .map(|(_, t)| origin + t * dir)
            .or_else(|| ray_plane(origin, dir, Vec3::ZERO, Vec3::Y));
        let Some(ground) = ground else {
            self.status = "Point at the ground or a body to place.".into();
            return;
        };
        let half_height = match shape {
            Shape::Sphere { radius } => radius,
            Shape::Cuboid { half_extents } => half_extents.y,
        };
        let position = ground + Vec3::Y * (half_height + 0.001);
        let mut body = if self.new_body.is_static {
            RigidBody::new_static(shape, position)
        } else {
            RigidBody::new(shape, position)
        };
        body.density = self.new_body.density;
        let id = self.world.add_body(body);
        self.select(Some(id));
        self.status = "Placed. Esc or V returns to Select.".into();
    }

    fn body_callback(&self, rect: Rect) -> BodyCallback {
        let selected = self.selected();
        let (mut cubes, mut spheres) = (Vec::new(), Vec::new());
        for (i, b) in self.world.bodies.iter().enumerate() {
            let color = if b.is_static { STATIC_FILL } else { CARDINAL };
            let (scale, list) = match b.shape {
                Shape::Sphere { radius } => (Vec3::splat(radius), &mut spheres),
                Shape::Cuboid { half_extents } => (half_extents, &mut cubes),
            };
            list.push(Instance {
                position: b.position.to_array(),
                checker: if b.is_static { 1.0 } else { 0.0 },
                scale: scale.to_array(),
                highlight: if selected == Some(BodyId(i)) {
                    1.0
                } else {
                    0.0
                },
                rotation: b.orientation.to_array(),
                color: color.to_array(),
            });
        }
        BodyCallback::new(
            cubes,
            spheres,
            self.camera.view_proj(rect),
            self.camera.eye(),
        )
    }

    fn draw_contacts(&self, painter: &egui::Painter, rect: Rect) {
        for m in self.world.contacts() {
            for c in &m.points {
                if let (Some(a), Some(b)) = (
                    self.camera.project(rect, c.position),
                    self.camera.project(rect, c.position + m.normal * 0.3),
                ) {
                    painter.circle_filled(a, 3.0, CONTACT_COLOR);
                    painter.line_segment([a, b], egui::Stroke::new(1.5, CONTACT_COLOR));
                }
            }
        }
    }

    /// A small x/y/z gizmo in the corner showing which way the axes point.
    fn draw_axes(&self, painter: &egui::Painter, rect: Rect) {
        let origin = Pos2::new(rect.left() + 40.0, rect.bottom() - 40.0);
        let eye_dir = (self.camera.eye() - self.camera.target).normalize();
        let right = Vec3::Y.cross(eye_dir).normalize_or(Vec3::X);
        let up = eye_dir.cross(right);
        for (axis, label, color) in [
            (Vec3::X, "x", Color32::from_rgb(0xC8, 0x10, 0x2E)),
            (Vec3::Y, "y", Color32::from_rgb(0x2E, 0x8B, 0x57)),
            (Vec3::Z, "z", CONTACT_COLOR),
        ] {
            let tip = origin + egui::vec2(axis.dot(right), -axis.dot(up)) * 26.0;
            painter.line_segment([origin, tip], egui::Stroke::new(2.0, color));
            painter.text(
                tip + (tip - origin).normalized() * 8.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }
}

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

fn density_row(ui: &mut Ui, value: &mut f32) {
    ui.label("Density");
    ui.add(
        egui::DragValue::new(value)
            .speed(10.0)
            .range(1.0..=20_000.0)
            .suffix(" kg/m³"),
    );
    ui.end_row();
}

fn vec3_row(ui: &mut Ui, label: &str, v: &mut Vec3, suffix: &str) {
    ui.label(label);
    ui.horizontal(|ui| {
        for c in [&mut v.x, &mut v.y, &mut v.z] {
            ui.add(egui::DragValue::new(c).speed(0.01).suffix(suffix));
        }
    });
    ui.end_row();
}
