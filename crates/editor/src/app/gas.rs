//! Smoke and heat in the 2D editor: the gas panel, the Smoke tool (K), and
//! drawing the grid as a texture.

use super::EditorApp;
use eframe::egui::{self, Color32, PointerButton, Rect, Stroke, Ui};
use engine::{Emitter, Gas, Vec2};
use serde::{Deserialize, Serialize};

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
    /// Brush and emitter temperature, °C.
    pub temperature_c: f32,
    /// Brush concentration / emitter rate per second.
    pub smoke: f32,
    pub radius: f32,
    /// Cell size for new grids, m.
    pub cell: f32,
}

impl Default for GasSettings {
    fn default() -> Self {
        Self {
            view: GasView::Smoke,
            mode: SmokeMode::Brush,
            temperature_c: 80.0,
            smoke: 1.0,
            radius: 0.3,
            cell: 0.1,
        }
    }
}

const KELVIN: f32 = 273.15;

impl EditorApp {
    pub(super) fn gas_panel(&mut self, ui: &mut Ui, viewport_world: (Vec2, Vec2)) {
        let header = match &self.world.gas {
            Some(g) => format!("Smoke and heat ({}×{})", g.nx, g.ny),
            None => "Smoke and heat".to_owned(),
        };
        egui::CollapsingHeader::new(header)
            .default_open(self.world.gas.is_some())
            .show(ui, |ui| {
                let s = &mut self.gas_settings;
                if self.world.gas.is_none() {
                    egui::Grid::new("gas_new").num_columns(2).show(ui, |ui| {
                        ui.label("Cell size");
                        ui.add(
                            egui::DragValue::new(&mut s.cell)
                                .speed(0.005)
                                .range(0.02..=1.0)
                                .suffix(" m"),
                        );
                        ui.end_row();
                    });
                    if ui
                        .button("Add a smoke grid over the view")
                        .on_hover_text("Covers what's on screen now")
                        .clicked()
                    {
                        let (lo, hi) = viewport_world;
                        let size = hi - lo;
                        let (nx, ny) = (
                            ((size.x / s.cell) as usize).clamp(8, 400),
                            ((size.y / s.cell) as usize).clamp(8, 400),
                        );
                        self.world.gas = Some(Gas::new(lo, s.cell, nx, ny));
                        self.status =
                            format!("Added a {nx}×{ny} smoke grid. Use the Smoke tool (K).");
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
                egui::Grid::new("gas_params").num_columns(2).show(ui, |ui| {
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
                    ui.label("Heat diffusivity")
                        .on_hover_text("κ: air is 2.2e-5 m²/s");
                    ui.add(
                        egui::DragValue::new(&mut p.thermal_diffusivity)
                            .speed(0.0001)
                            .range(0.0..=0.1)
                            .suffix(" m²/s"),
                    );
                    ui.end_row();
                    ui.label("Smoke weight")
                        .on_hover_text("0 = a weightless dye");
                    ui.add(egui::Slider::new(&mut p.smoke_weight, 0.0..=0.5));
                    ui.end_row();
                    ui.label("Smoke fades")
                        .on_hover_text("Fraction lost per second");
                    ui.add(egui::Slider::new(&mut p.smoke_decay, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Cooling")
                        .on_hover_text("Newton cooling toward ambient, 1/s");
                    ui.add(egui::Slider::new(&mut p.cooling, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Open top");
                    ui.checkbox(&mut p.open_top, "");
                    ui.end_row();
                });
                ui.weak(format!(
                    "{} emitters · {} substeps · {} pressure iterations",
                    gas.emitters.len(),
                    gas.stats.substeps,
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
                    self.gas_texture = None;
                }
            });
    }

    pub(super) fn smoke_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.gas_settings;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.mode, SmokeMode::Brush, "Brush");
            ui.selectable_value(&mut s.mode, SmokeMode::Emitter, "Emitter");
        });
        egui::Grid::new("smoke_tool").num_columns(2).show(ui, |ui| {
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
                    .range(0.05..=3.0)
                    .suffix(" m"),
            );
            ui.end_row();
        });
        ui.weak(if self.world.gas.is_none() {
            "Add a smoke grid first (Smoke and heat panel)."
        } else if s.mode == SmokeMode::Brush {
            "Drag to paint smoke and heat."
        } else {
            "Click to place a heater/smoke source."
        });
    }

    pub(super) fn smoke_tool_input(&mut self, response: &egui::Response, cursor: Option<Vec2>) {
        let Some(gas) = self.world.gas.as_mut() else {
            if response.clicked_by(PointerButton::Primary) {
                self.status = "Add a smoke grid first (Smoke and heat panel).".into();
            }
            return;
        };
        let s = &self.gas_settings;
        let Some(p) = cursor else { return };
        match s.mode {
            SmokeMode::Brush => {
                if response.dragged_by(PointerButton::Primary)
                    || response.clicked_by(PointerButton::Primary)
                {
                    gas.paint(p, s.radius, s.smoke, s.temperature_c + KELVIN);
                }
            }
            SmokeMode::Emitter => {
                if response.clicked_by(PointerButton::Primary) {
                    gas.emitters.push(Emitter {
                        center: p,
                        radius: s.radius,
                        smoke_rate: s.smoke,
                        temperature: s.temperature_c + KELVIN,
                        velocity: Vec2::ZERO,
                    });
                    self.status = format!("{} emitters.", gas.emitters.len());
                }
            }
        }
    }

    /// Draw the grid as a texture stretched over its rectangle, plus the
    /// emitters and the grid's outline.
    pub(super) fn draw_gas(&mut self, ui: &Ui, painter: &egui::Painter, viewport: Rect) {
        let Some(gas) = &self.world.gas else {
            return;
        };
        let (nx, ny) = (gas.nx, gas.ny);
        let mut pixels = vec![Color32::TRANSPARENT; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                let color = if gas.is_solid(i, j) {
                    Color32::TRANSPARENT
                } else {
                    match self.gas_settings.view {
                        GasView::Smoke => {
                            let a = (gas.smoke[k].min(1.0) * 220.0) as u8;
                            Color32::from_rgba_unmultiplied(60, 60, 64, a)
                        }
                        GasView::Temperature => heat_color(gas.heat[k]),
                    }
                };
                // Image rows go top-down; the grid's go bottom-up.
                pixels[(ny - 1 - j) * nx + i] = color;
            }
        }
        let image = egui::ColorImage::new([nx, ny], pixels);
        let options = egui::TextureOptions::LINEAR;
        let texture = match &mut self.gas_texture {
            Some(t) if t.size() == [nx, ny] => {
                t.set(image, options);
                t.clone()
            }
            slot => {
                let t = ui.ctx().load_texture("gas", image, options);
                *slot = Some(t.clone());
                t
            }
        };
        let a = self.camera.world_to_screen(viewport, gas.origin);
        let b = self
            .camera
            .world_to_screen(viewport, gas.origin + gas.size());
        let rect = Rect::from_two_pos(a, b);
        painter.image(
            texture.id(),
            rect,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.0, Color32::from_gray(170)),
            egui::StrokeKind::Middle,
        );
        for e in &gas.emitters {
            let c = self.camera.world_to_screen(viewport, e.center);
            let r = self.camera.scale(e.radius);
            painter.circle(
                c,
                r,
                Color32::from_rgba_unmultiplied(255, 140, 0, 40),
                Stroke::new(1.5, Color32::from_rgb(0xE0, 0x6C, 0x00)),
            );
        }
    }
}

/// Transparent at ambient; warm shades through orange to deep red by 100 K
/// above; blue below ambient.
fn heat_color(dt: f32) -> Color32 {
    if dt >= 0.0 {
        let t = (dt / 100.0).min(1.0);
        let (r, g, b) = if t < 0.5 {
            let u = t / 0.5;
            (255.0, 220.0 - 80.0 * u, 120.0 - 120.0 * u)
        } else {
            let u = (t - 0.5) / 0.5;
            (255.0 - 55.0 * u, 140.0 - 120.0 * u, 0.0)
        };
        Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, (40.0 + 200.0 * t.sqrt()) as u8)
    } else {
        let t = (-dt / 30.0).min(1.0);
        Color32::from_rgba_unmultiplied(40, 110, 230, (200.0 * t.sqrt()) as u8)
    }
}
