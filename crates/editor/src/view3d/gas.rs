//! Smoke and heat in the 3D editor: the gas panel, the Smoke tool (K), and
//! turning the grid into splats for the renderer.

use super::render::SplatInstance;
use super::{Editor3d, ray_plane};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Stroke, Ui};
use engine3d::{Emitter, Gas, Vec3};
use serde::{Deserialize, Serialize};

const KELVIN: f32 = 273.15;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum GasView {
    Smoke,
    Temperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum SmokeMode {
    Brush,
    Emitter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct GasSettings {
    pub view: GasView,
    pub mode: SmokeMode,
    pub temperature_c: f32,
    pub smoke: f32,
    pub radius: f32,
    pub cell: f32,
}

impl Default for GasSettings {
    fn default() -> Self {
        Self {
            view: GasView::Smoke,
            mode: SmokeMode::Emitter,
            temperature_c: 80.0,
            smoke: 1.5,
            radius: 0.25,
            cell: 0.1,
        }
    }
}

impl Editor3d {
    pub(super) fn gas_panel(&mut self, ui: &mut Ui) {
        let header = match &self.world.gas {
            Some(g) => format!("Smoke and heat ({}×{}×{})", g.nx, g.ny, g.nz),
            None => "Smoke and heat".to_owned(),
        };
        egui::CollapsingHeader::new(header)
            .default_open(self.world.gas.is_some())
            .show(ui, |ui| {
                let s = &mut self.gas_settings;
                if self.world.gas.is_none() {
                    egui::Grid::new("gas3d_new").num_columns(2).show(ui, |ui| {
                        ui.label("Cell size");
                        ui.add(
                            egui::DragValue::new(&mut s.cell)
                                .speed(0.005)
                                .range(0.04..=1.0)
                                .suffix(" m"),
                        );
                        ui.end_row();
                    });
                    if ui
                        .button("Add a smoke grid here")
                        .on_hover_text("4 m × 3 m × 4 m around where the camera is looking")
                        .clicked()
                    {
                        let size = Vec3::new(4.0, 3.0, 4.0);
                        let t = self.camera.target;
                        let origin = Vec3::new(t.x - 0.5 * size.x, 0.0, t.z - 0.5 * size.z);
                        let n = (size / s.cell).as_uvec3();
                        let (nx, ny, nz) = (n.x as usize, n.y as usize, n.z as usize);
                        self.world.gas = Some(Gas::new(origin, s.cell, nx, ny, nz));
                        self.status =
                            format!("Added a {nx}×{ny}×{nz} smoke grid. Use the Smoke tool (K).");
                    }
                    return;
                }
                ui.horizontal(|ui| {
                    ui.label("Show");
                    ui.selectable_value(&mut s.view, GasView::Smoke, "Smoke");
                    ui.selectable_value(&mut s.view, GasView::Temperature, "Temperature");
                });
                let gas = self.world.gas.as_mut().expect("checked above");
                let p = &mut gas.params;
                egui::Grid::new("gas3d_params")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Ambient");
                        let mut c = p.ambient_temperature - KELVIN;
                        if ui
                            .add(
                                egui::DragValue::new(&mut c)
                                    .speed(0.5)
                                    .range(-50.0..=60.0)
                                    .suffix(" °C"),
                            )
                            .changed()
                        {
                            p.ambient_temperature = c + KELVIN;
                        }
                        ui.end_row();
                        ui.label("Heat diffusivity");
                        ui.add(
                            egui::DragValue::new(&mut p.thermal_diffusivity)
                                .speed(0.0001)
                                .range(0.0..=0.1)
                                .suffix(" m²/s"),
                        );
                        ui.end_row();
                        ui.label("Smoke weight");
                        ui.add(egui::Slider::new(&mut p.smoke_weight, 0.0..=0.5));
                        ui.end_row();
                        ui.label("Smoke fades");
                        ui.add(egui::Slider::new(&mut p.smoke_decay, 0.0..=1.0));
                        ui.end_row();
                        ui.label("Cooling");
                        ui.add(egui::Slider::new(&mut p.cooling, 0.0..=1.0));
                        ui.end_row();
                        ui.label("Open top");
                        ui.checkbox(&mut p.open_top, "");
                        ui.end_row();
                    });
                ui.weak(format!(
                    "{} emitters · {} pressure iterations",
                    gas.emitters.len(),
                    gas.stats.pressure_iterations
                ));
                let mut remove = false;
                ui.horizontal(|ui| {
                    if ui.button("Clear air").clicked() {
                        gas.clear();
                    }
                    if ui.button("Remove emitters").clicked() {
                        gas.emitters.clear();
                    }
                    remove = ui.button("Remove grid").clicked();
                });
                if remove {
                    self.world.gas = None;
                }
            });
    }

    pub(super) fn smoke_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.gas_settings;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.mode, SmokeMode::Emitter, "Emitter");
            ui.selectable_value(&mut s.mode, SmokeMode::Brush, "Brush");
        });
        egui::Grid::new("smoke3d_tool")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Temperature");
                ui.add(
                    egui::DragValue::new(&mut s.temperature_c)
                        .speed(1.0)
                        .range(-50.0..=1000.0)
                        .suffix(" °C"),
                );
                ui.end_row();
                ui.label(if s.mode == SmokeMode::Brush {
                    "Smoke"
                } else {
                    "Smoke per second"
                });
                ui.add(
                    egui::DragValue::new(&mut s.smoke)
                        .speed(0.05)
                        .range(0.0..=10.0),
                );
                ui.end_row();
                ui.label("Radius");
                ui.add(
                    egui::DragValue::new(&mut s.radius)
                        .speed(0.01)
                        .range(0.05..=2.0)
                        .suffix(" m"),
                );
                ui.end_row();
            });
        ui.weak(if self.world.gas.is_none() {
            "Add a smoke grid first (Smoke and heat panel)."
        } else if s.mode == SmokeMode::Brush {
            "Drag over surfaces to paint smoke and heat."
        } else {
            "Click a surface to place a heater/smoke source."
        });
    }

    /// The surface point under the mouse, raised by the brush radius.
    fn smoke_point(&self, rect: Rect, p: Pos2) -> Option<Vec3> {
        let (origin, dir) = self.camera.ray(rect, p);
        let hit = self
            .world
            .raycast(origin, dir)
            .map(|(_, t)| origin + t * dir)
            .or_else(|| ray_plane(origin, dir, Vec3::ZERO, Vec3::Y))?;
        Some(hit + Vec3::Y * self.gas_settings.radius)
    }

    pub(super) fn smoke_tool_input(&mut self, response: &egui::Response, rect: Rect) {
        let Some(p) = response.interact_pointer_pos() else {
            return;
        };
        let point = self.smoke_point(rect, p);
        let Some(gas) = self.world.gas.as_mut() else {
            if response.clicked_by(PointerButton::Primary) {
                self.status = "Add a smoke grid first (Smoke and heat panel).".into();
            }
            return;
        };
        let (Some(point), s) = (point, &self.gas_settings) else {
            return;
        };
        match s.mode {
            SmokeMode::Brush => {
                if response.dragged_by(PointerButton::Primary)
                    || response.clicked_by(PointerButton::Primary)
                {
                    gas.paint(point, s.radius, s.smoke, s.temperature_c + KELVIN);
                }
            }
            SmokeMode::Emitter => {
                if response.clicked_by(PointerButton::Primary) {
                    gas.emitters.push(Emitter {
                        center: point,
                        radius: s.radius,
                        smoke_rate: s.smoke,
                        temperature: s.temperature_c + KELVIN,
                        velocity: Vec3::ZERO,
                    });
                    self.status = format!("{} emitters. Press Space to play.", gas.emitters.len());
                }
            }
        }
    }

    /// Cells with visible smoke (or heat) as splats.
    pub(super) fn gas_splats(&self, dark: bool) -> Vec<SplatInstance> {
        let Some(gas) = &self.world.gas else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let size = 0.9 * gas.cell;
        for k in 0..gas.nz {
            for j in 0..gas.ny {
                for i in 0..gas.nx {
                    let c = gas.idx(i, j, k);
                    let (tint, strength) = match self.gas_settings.view {
                        GasView::Smoke => {
                            (crate::theme::smoke_rgb(dark), gas.smoke[c].min(1.0) * 0.5)
                        }
                        GasView::Temperature => {
                            let t = (gas.heat[c] / 100.0).clamp(-0.3, 1.0);
                            if t >= 0.0 {
                                ([255, (200.0 - 150.0 * t) as u8, 40], t.sqrt() * 0.6)
                            } else {
                                ([60, 120, 255], (-t).sqrt() * 0.6)
                            }
                        }
                    };
                    if strength < 0.01 {
                        continue;
                    }
                    out.push(SplatInstance {
                        position: gas.cell_center(i, j, k).to_array(),
                        size,
                        color: [tint[0], tint[1], tint[2], (strength * 255.0) as u8],
                    });
                }
            }
        }
        out
    }

    /// The grid's outline and the emitters, drawn over the view.
    pub(super) fn draw_gas_overlay(&self, painter: &egui::Painter, rect: Rect) {
        let Some(gas) = &self.world.gas else {
            return;
        };
        let (lo, hi) = (gas.origin, gas.origin + gas.size());
        let corner = |m: usize| {
            Vec3::new(
                if m & 1 == 0 { lo.x } else { hi.x },
                if m & 2 == 0 { lo.y } else { hi.y },
                if m & 4 == 0 { lo.z } else { hi.z },
            )
        };
        let stroke = Stroke::new(1.0, Color32::from_gray(150));
        for (a, b) in [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ] {
            if let (Some(p), Some(q)) = (
                self.camera.project(rect, corner(a)),
                self.camera.project(rect, corner(b)),
            ) {
                painter.line_segment([p, q], stroke);
            }
        }
        for e in &gas.emitters {
            if let Some(p) = self.camera.project(rect, e.center) {
                painter.circle(
                    p,
                    6.0,
                    Color32::from_rgba_unmultiplied(255, 140, 0, 120),
                    Stroke::new(1.5, Color32::from_rgb(0xE0, 0x6C, 0x00)),
                );
            }
        }
    }
}
