//! 3D soft bodies in the editor: the Soft tool (S), drawing them as meshes,
//! and picking/dragging them with Select.

use super::Editor3d;
use super::render::{Instance, SoftMeshes, Vertex};
use eframe::egui::{self, Color32, Pos2, Rect, Ui};
use engine3d::Vec3;
use engine3d::soft::SoftBody;
use serde::{Deserialize, Serialize};

const SOFT_COLOR: Color32 = Color32::from_rgb(0xE0, 0x5A, 0x6E);
const CLOTH_COLOR: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum SoftShape {
    Ball,
    Cloth,
    Jelly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct SoftSettings {
    pub shape: SoftShape,
    /// Ball radius, cloth or jelly side, m.
    pub size: f32,
    /// N/m per edge.
    pub stiffness: f32,
    /// kg/m³ (cloth: total mass 0.5 kg per m²).
    pub density: f32,
    /// Pin two corners of new cloth to the world.
    pub pin_cloth: bool,
}

impl Default for SoftSettings {
    fn default() -> Self {
        Self {
            shape: SoftShape::Ball,
            size: 0.5,
            stiffness: 1000.0,
            density: 300.0,
            pin_cloth: false,
        }
    }
}

impl SoftSettings {
    /// A new soft body resting on the point `ground`.
    fn build(&self, ground: Vec3) -> SoftBody {
        let mut s = match self.shape {
            SoftShape::Ball => SoftBody::ball(
                ground + Vec3::Y * (self.size + 0.05),
                self.size,
                2,
                self.density,
            ),
            SoftShape::Cloth => {
                let side = self.size * 4.0;
                let n = ((side / 0.12).round() as usize).clamp(6, 30);
                let mut c =
                    SoftBody::cloth(ground + Vec3::Y * 1.5, side, side, n, n, 0.5 * side * side);
                if self.pin_cloth {
                    let back = c.positions.iter().map(|p| p.z).fold(f32::MAX, f32::min);
                    let (lo, hi) = (
                        c.positions.iter().map(|p| p.x).fold(f32::MAX, f32::min),
                        c.positions.iter().map(|p| p.x).fold(f32::MIN, f32::max),
                    );
                    c.pin_where(|p| {
                        (p.z - back).abs() < 1e-4
                            && ((p.x - lo).abs() < 1e-4 || (p.x - hi).abs() < 1e-4)
                    });
                }
                c
            }
            SoftShape::Jelly => {
                let side = self.size * 1.6;
                SoftBody::jelly(
                    ground + Vec3::Y * (0.5 * side + 0.05),
                    Vec3::splat(side),
                    5,
                    self.density,
                )
            }
        };
        s.edge_stiffness = self.stiffness;
        s
    }
}

impl Editor3d {
    pub(super) fn soft_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.soft_settings;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.shape, SoftShape::Ball, "Ball");
            ui.selectable_value(&mut s.shape, SoftShape::Cloth, "Cloth");
            ui.selectable_value(&mut s.shape, SoftShape::Jelly, "Jelly");
        });
        egui::Grid::new("soft3d").num_columns(2).show(ui, |ui| {
            ui.label("Size");
            ui.add(
                egui::DragValue::new(&mut s.size)
                    .speed(0.01)
                    .range(0.15..=3.0)
                    .suffix(" m"),
            );
            ui.end_row();
            ui.label("Stiffness");
            ui.add(
                egui::DragValue::new(&mut s.stiffness)
                    .speed(10.0)
                    .range(10.0..=1.0e6)
                    .suffix(" N/m"),
            );
            ui.end_row();
            if s.shape != SoftShape::Cloth {
                ui.label("Density");
                ui.add(
                    egui::DragValue::new(&mut s.density)
                        .speed(10.0)
                        .range(10.0..=20_000.0)
                        .suffix(" kg/m³"),
                );
                ui.end_row();
            }
        });
        if s.shape == SoftShape::Cloth {
            ui.checkbox(&mut s.pin_cloth, "Pin two corners (hang it)");
        }
        ui.weak("Click to place.");
    }

    pub(super) fn place_soft(&mut self, rect: Rect, p: Pos2) {
        let (origin, dir) = self.camera.ray(rect, p);
        let ground = self
            .world
            .raycast(origin, dir)
            .map(|(_, t)| origin + t * dir)
            .or_else(|| super::ray_plane(origin, dir, Vec3::ZERO, Vec3::Y));
        let Some(ground) = ground else {
            self.status = "Point at the ground or a body to place.".into();
            return;
        };
        self.world
            .soft_bodies
            .push(self.soft_settings.build(ground));
        self.selected_soft = Some(self.world.soft_bodies.len() - 1);
        self.selection = None;
        self.status = "Placed a soft body. Press Space to play.".into();
    }

    /// The soft body under the mouse and its distance, if nearer than any
    /// rigid body.
    pub(super) fn pick_soft(&self, rect: Rect, p: Pos2) -> Option<(usize, f32)> {
        let (origin, dir) = self.camera.ray(rect, p);
        let rigid = self
            .world
            .raycast(origin, dir)
            .map_or(f32::INFINITY, |(_, t)| t);
        self.world
            .soft_bodies
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.raycast(origin, dir).map(|t| (i, t)))
            .filter(|&(_, t)| t < rigid)
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    pub(super) fn delete_selected_soft(&mut self) -> bool {
        match self.selected_soft.take() {
            Some(i) if i < self.world.soft_bodies.len() => {
                self.world.soft_bodies.remove(i);
                self.soft_grab = None;
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
            "Soft body: {} particles, {:.2} kg",
            s.len(),
            s.mass()
        ));
        egui::Grid::new("soft_inspector3d")
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
                if s.rest_volume > 0.0 {
                    ui.label("Pressure");
                    ui.add(
                        egui::DragValue::new(&mut s.volume_stiffness)
                            .speed(100.0)
                            .range(0.0..=1.0e8),
                    );
                    ui.end_row();
                }
                ui.label("Damping");
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

    /// Every soft body's surface as world-space triangles with smooth
    /// normals.
    pub(super) fn soft_meshes(&self) -> SoftMeshes {
        let mut out = SoftMeshes::default();
        for (i, s) in self.world.soft_bodies.iter().enumerate() {
            let base = out.vertices.len() as u32;
            let mut normals = vec![Vec3::ZERO; s.len()];
            for t in &s.triangles {
                let (a, b, c) = (
                    s.positions[t[0] as usize],
                    s.positions[t[1] as usize],
                    s.positions[t[2] as usize],
                );
                let n = (b - a).cross(c - a); // area-weighted
                for &k in t {
                    normals[k as usize] += n;
                }
            }
            out.vertices
                .extend(s.positions.iter().zip(&normals).map(|(p, n)| Vertex {
                    position: p.to_array(),
                    normal: n.normalize_or(Vec3::Y).to_array(),
                }));
            let start = out.indices.len() as u32;
            out.indices
                .extend(s.triangles.iter().flatten().map(|&k| base + k));
            let end = out.indices.len() as u32;
            let color = if s.rest_volume == 0.0 {
                CLOTH_COLOR
            } else {
                SOFT_COLOR
            };
            out.bodies.push((
                start..end,
                Instance {
                    position: [0.0; 3],
                    checker: 0.0,
                    scale: [1.0; 3],
                    highlight: if self.selected_soft == Some(i) {
                        1.0
                    } else {
                        0.0
                    },
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    color: color.to_array(),
                },
            ));
        }
        out
    }
}
