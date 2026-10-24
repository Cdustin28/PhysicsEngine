//! Soft-body tool and drawing.
//!
//! **Soft tool (S)**: click to place a blob or a jelly block. With the
//! Select tool, soft bodies can be clicked, dragged (and thrown while
//! playing), and deleted like rigid bodies.

use super::{EditorApp, GOLD};
use eframe::egui::{self, Color32, PointerButton, Rect, Stroke, Ui};
use engine::Vec2;
use engine::soft::SoftBody;
use serde::{Deserialize, Serialize};

/// Soft bodies: a lighter shade of cardinal, so they read as squishy.
const SOFT_FILL: Color32 = Color32::from_rgb(0xE0, 0x5A, 0x6E);
const SOFT_EDGE: Color32 = Color32::from_rgb(0x8F, 0x0C, 0x21);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum SoftShape {
    Blob,
    Jelly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct SoftSettings {
    pub shape: SoftShape,
    /// Blob radius or jelly width, m.
    pub size: f32,
    /// N/m per edge.
    pub stiffness: f32,
    /// kg/m²
    pub density: f32,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self {
            shape: SoftShape::Blob,
            size: 0.5,
            stiffness: 1000.0,
            density: 1.0,
        }
    }
}

impl SoftSettings {
    pub fn build(&self, at: Vec2) -> SoftBody {
        let mut s = match self.shape {
            SoftShape::Blob => {
                let segments =
                    ((std::f32::consts::TAU * self.size / 0.1).round() as usize).clamp(12, 64);
                SoftBody::blob(at, self.size, segments, self.density)
            }
            SoftShape::Jelly => {
                let (w, h) = (self.size * 2.0, self.size * 1.4);
                let nx = ((w / 0.12).round() as usize).clamp(3, 24);
                let ny = ((h / 0.12).round() as usize).clamp(2, 24);
                SoftBody::block(at, w, h, nx, ny, self.density)
            }
        };
        s.edge_stiffness = self.stiffness;
        s
    }
}

impl EditorApp {
    pub(super) fn soft_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.soft_settings;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.shape, SoftShape::Blob, "Blob");
            ui.selectable_value(&mut s.shape, SoftShape::Jelly, "Jelly");
        });
        egui::Grid::new("soft").num_columns(2).show(ui, |ui| {
            ui.label(if s.shape == SoftShape::Blob {
                "Radius"
            } else {
                "Half width"
            });
            ui.add(
                egui::DragValue::new(&mut s.size)
                    .speed(0.01)
                    .range(0.15..=3.0)
                    .suffix(" m"),
            );
            ui.end_row();
            ui.label("Stiffness")
                .on_hover_text("Per edge between particles. Higher is firmer.");
            ui.add(
                egui::DragValue::new(&mut s.stiffness)
                    .speed(10.0)
                    .range(10.0..=1.0e6)
                    .suffix(" N/m"),
            );
            ui.end_row();
            ui.label("Density");
            ui.add(
                egui::DragValue::new(&mut s.density)
                    .speed(0.05)
                    .range(0.01..=1000.0)
                    .suffix(" kg/m²"),
            );
            ui.end_row();
        });
        ui.weak("Click on the canvas to place.");
    }

    pub(super) fn soft_tool_input(&mut self, response: &egui::Response, cursor: Option<Vec2>) {
        if response.clicked_by(PointerButton::Primary)
            && let Some(p) = cursor
        {
            let body = self.soft_settings.build(self.snap_point(p));
            self.world.soft_bodies.push(body);
            self.selected_soft = Some(self.world.soft_bodies.len() - 1);
            self.status = "Placed a soft body. Esc or V returns to Select.".into();
        }
    }

    /// The topmost soft body containing `p`.
    pub(super) fn soft_at(&self, p: Vec2) -> Option<usize> {
        (0..self.world.soft_bodies.len())
            .rev()
            .find(|&i| self.world.soft_bodies[i].contains_point(p))
    }

    /// Start dragging a soft body if the press landed on one (and on no
    /// rigid body). Returns whether it did.
    pub(super) fn start_soft_drag(&mut self, p: Vec2) -> bool {
        if self.world.body_at_point(p).is_some() {
            return false;
        }
        let Some(i) = self.soft_at(p) else {
            return false;
        };
        self.selected_soft = Some(i);
        self.selection.clear();
        self.soft_drag = Some((i, p));
        true
    }

    pub(super) fn continue_soft_drag(&mut self, p: Vec2, dt: f32) {
        let Some((i, last)) = self.soft_drag else {
            return;
        };
        let Some(s) = self.world.soft_bodies.get_mut(i) else {
            self.soft_drag = None;
            return;
        };
        let delta = p - last;
        s.translate(delta);
        let v = if self.running {
            (delta / dt).clamp_length_max(30.0)
        } else {
            Vec2::ZERO
        };
        for (vel, &w) in s.velocities.iter_mut().zip(&s.inv_mass) {
            if w > 0.0 {
                *vel = v;
            }
        }
        self.soft_drag = Some((i, p));
    }

    pub(super) fn delete_selected_soft(&mut self) -> bool {
        match self.selected_soft.take() {
            Some(i) if i < self.world.soft_bodies.len() => {
                self.world.soft_bodies.remove(i);
                self.soft_drag = None;
                self.status = "Deleted a soft body.".into();
                true
            }
            _ => false,
        }
    }

    pub(super) fn soft_inspector(&mut self, ui: &mut Ui, i: usize) {
        let Some(s) = self.world.soft_bodies.get_mut(i) else {
            return;
        };
        ui.label(format!(
            "Soft body: {} particles, {:.3} kg, area {:.0}% of rest",
            s.len(),
            s.mass(),
            100.0 * s.area() / s.rest_area.max(1e-9)
        ));
        egui::Grid::new("soft_inspector")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Stiffness");
                ui.add(
                    egui::DragValue::new(&mut s.edge_stiffness)
                        .speed(10.0)
                        .range(0.0..=1.0e7)
                        .suffix(" N/m"),
                );
                ui.end_row();
                ui.label("Pressure")
                    .on_hover_text("How strongly it keeps its area (0 = none)");
                ui.add(
                    egui::DragValue::new(&mut s.area_stiffness)
                        .speed(100.0)
                        .range(0.0..=1.0e7),
                );
                ui.end_row();
                ui.label("Damping")
                    .on_hover_text("How quickly wobbling dies out, 1/s");
                ui.add(egui::Slider::new(&mut s.damping, 0.0..=20.0));
                ui.end_row();
                ui.label("Friction");
                ui.add(egui::Slider::new(&mut s.friction, 0.0..=1.5));
                ui.end_row();
            });
        if ui.button("Delete").clicked() {
            self.delete_selected_soft();
        }
    }

    pub(super) fn draw_soft_bodies(&self, painter: &egui::Painter, viewport: Rect) {
        for (i, s) in self.world.soft_bodies.iter().enumerate() {
            let screen: Vec<egui::Pos2> = s
                .positions
                .iter()
                .map(|&p| self.camera.world_to_screen(viewport, p))
                .collect();
            let mut mesh = egui::Mesh::default();
            if s.triangles.is_empty() {
                // Blob: a fan from its center.
                let center = self.camera.world_to_screen(viewport, s.center());
                mesh.colored_vertex(center, SOFT_FILL);
                for &k in &s.outline {
                    mesh.colored_vertex(screen[k as usize], SOFT_FILL);
                }
                let n = s.outline.len() as u32;
                for k in 0..n {
                    mesh.add_triangle(0, 1 + k, 1 + (k + 1) % n);
                }
            } else {
                for &p in &screen {
                    mesh.colored_vertex(p, SOFT_FILL);
                }
                for t in &s.triangles {
                    mesh.add_triangle(t[0], t[1], t[2]);
                }
            }
            painter.add(egui::Shape::mesh(mesh));
            let outline: Vec<egui::Pos2> = s.outline.iter().map(|&k| screen[k as usize]).collect();
            let selected = self.selected_soft == Some(i);
            let stroke = if selected {
                Stroke::new(2.5, GOLD)
            } else {
                Stroke::new(1.5, SOFT_EDGE)
            };
            painter.add(egui::Shape::closed_line(outline, stroke));
        }
    }
}
