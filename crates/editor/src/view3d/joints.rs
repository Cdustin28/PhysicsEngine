//! 3D joint and chain tools: two clicks each.
//!
//! - **Joint (J)**: click a point on a body, then a point on another body
//!   (or the ground or empty space, for the world). Pin joins them at the
//!   first point; Rod, Rope, and Spring connect the two points.
//! - **Chain (L)**: click the two ends. Each end is pinned to the body under
//!   it, or to the world.

use super::{Editor3d, ray_plane};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Ui};
use engine3d::Vec3;
use engine3d::joint::{Anchor, Joint, JointKind, add_chain};
use serde::{Deserialize, Serialize};

const GOLD: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum JointType {
    Pin,
    Rod,
    Rope,
    Spring,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct JointSettings {
    pub kind: JointType,
    pub stiffness: f32,
    pub damping: f32,
    pub link_length: f32,
    pub link_thickness: f32,
}

impl Default for JointSettings {
    fn default() -> Self {
        Self {
            kind: JointType::Pin,
            stiffness: 200.0,
            damping: 5.0,
            link_length: 0.25,
            link_thickness: 0.08,
        }
    }
}

impl Editor3d {
    /// The body and point under the mouse: (Some(body), hit point) on a
    /// body, (None, point) on the ground plane.
    fn pick_point(&self, rect: Rect, p: Pos2) -> Option<(Option<usize>, Vec3)> {
        let (origin, dir) = self.camera.ray(rect, p);
        if let Some((id, t)) = self.world.raycast(origin, dir) {
            return Some((Some(id.0), origin + t * dir));
        }
        ray_plane(origin, dir, Vec3::ZERO, Vec3::Y).map(|p| (None, p))
    }

    pub(super) fn joint_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.joint_settings;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.kind, JointType::Pin, "Pin");
            ui.selectable_value(&mut s.kind, JointType::Rod, "Rod");
            ui.selectable_value(&mut s.kind, JointType::Rope, "Rope");
            ui.selectable_value(&mut s.kind, JointType::Spring, "Spring");
        });
        if s.kind == JointType::Spring {
            egui::Grid::new("spring3d").num_columns(2).show(ui, |ui| {
                ui.label("Stiffness");
                ui.add(
                    egui::DragValue::new(&mut s.stiffness)
                        .speed(1.0)
                        .range(0.0..=1.0e7)
                        .suffix(" N/m"),
                );
                ui.end_row();
                ui.label("Damping");
                ui.add(
                    egui::DragValue::new(&mut s.damping)
                        .speed(0.1)
                        .range(0.0..=1.0e6)
                        .suffix(" N·s/m"),
                );
                ui.end_row();
            });
        }
        ui.weak(if self.joint_click.is_some() {
            "Now click the second point: another body, or the ground for the world."
        } else {
            "Click a point on a body to start."
        });
    }

    pub(super) fn chain_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.joint_settings;
        egui::Grid::new("chain3d").num_columns(2).show(ui, |ui| {
            ui.label("Link length");
            ui.add(
                egui::DragValue::new(&mut s.link_length)
                    .speed(0.01)
                    .range(0.05..=5.0)
                    .suffix(" m"),
            );
            ui.end_row();
            ui.label("Thickness");
            ui.add(
                egui::DragValue::new(&mut s.link_thickness)
                    .speed(0.005)
                    .range(0.01..=1.0)
                    .suffix(" m"),
            );
            ui.end_row();
        });
        ui.weak(if self.joint_click.is_some() {
            "Now click where the chain should end."
        } else {
            "Click where the chain should start. Ends are pinned to the body under them, or to the world."
        });
    }

    pub(super) fn joint_click(&mut self, rect: Rect, p: Pos2) {
        let Some((body, point)) = self.pick_point(rect, p) else {
            return;
        };
        let Some((a, pa)) = self.joint_click.take() else {
            if body.is_none() {
                self.status = "Start on a body.".into();
            } else {
                self.joint_click = Some((body, point));
            }
            return;
        };
        let a = a.expect("first click is always on a body");
        let b = body.filter(|&b| b != a);
        let bodies = &self.world.bodies;
        if b.is_none() && bodies[a].is_static {
            self.status = "Both ends would be fixed; start on a dynamic body.".into();
            return;
        }
        let s = &self.joint_settings;
        let joint = match s.kind {
            JointType::Pin => Joint::pin(bodies, a, b, pa),
            JointType::Rod => Joint::rod(bodies, a, pa, b, point),
            JointType::Rope => Joint::rope(bodies, a, pa, b, point),
            JointType::Spring => Joint::spring(bodies, a, pa, b, point, s.stiffness, s.damping),
        };
        self.status = match self.world.add_joint(joint) {
            Ok(_) => format!("Joined. {} joints in the scene.", self.world.joints.len()),
            Err(e) => format!("Couldn't add that joint: {e}"),
        };
    }

    pub(super) fn chain_click(&mut self, rect: Rect, p: Pos2) {
        let Some((body, point)) = self.pick_point(rect, p) else {
            return;
        };
        let Some((start_body, start)) = self.joint_click.take() else {
            self.joint_click = Some((body, point));
            return;
        };
        let s = self.joint_settings.clone();
        let length = (point - start).length();
        if length < s.link_length {
            self.status = "Click farther away to make a chain.".into();
            return;
        }
        let anchor = |b: Option<usize>| b.map_or(Anchor::World, Anchor::Body);
        let links = ((length / s.link_length).round() as usize).max(2);
        add_chain(
            &mut self.world,
            start,
            point,
            links,
            s.link_thickness,
            anchor(start_body),
            anchor(body),
        );
        self.world.solver.substeps = self.world.solver.substeps.max(4);
        self.status = format!("Added a chain of {links} links.");
    }

    /// Joints drawn over the scene: rods dark, ropes brown (pale when
    /// slack), springs as zigzags, gold dots at the anchors.
    pub(super) fn draw_joints(&self, painter: &egui::Painter, rect: Rect) {
        let dark = crate::theme::is_dark(painter.ctx());
        let (ink, rope) = (crate::theme::ink(dark), crate::theme::rope(dark));
        for joint in &self.world.joints {
            let (pa, pb) = joint.anchors(&self.world.bodies);
            let (Some(a), Some(b)) = (self.camera.project(rect, pa), self.camera.project(rect, pb))
            else {
                continue;
            };
            match joint.kind {
                JointKind::Pin => {}
                JointKind::Distance {
                    min_length,
                    max_length,
                } => {
                    let color = if min_length == max_length {
                        ink
                    } else if (pb - pa).length() < max_length - 1e-3 {
                        rope.gamma_multiply(0.5)
                    } else {
                        rope
                    };
                    painter.line_segment([a, b], Stroke::new(2.0, color));
                }
                JointKind::Spring { .. } => {
                    let d = b - a;
                    let across = egui::vec2(-d.y, d.x).normalized() * 5.0;
                    let mut pts = vec![a];
                    for k in 0..10 {
                        let t = 0.1 + 0.8 * (k as f32 + 0.5) / 10.0;
                        pts.push(a + d * t + across * if k % 2 == 0 { 1.0 } else { -1.0 });
                    }
                    pts.push(b);
                    painter.add(egui::Shape::line(pts, Stroke::new(1.5, ink)));
                }
            }
            painter.circle(b, 3.5, GOLD, Stroke::new(1.0, ink));
        }
        if let Some((_, p)) = self.joint_click
            && let Some(s) = self.camera.project(rect, p)
        {
            painter.circle(s, 5.0, GOLD, Stroke::new(1.5, ink));
        }
    }
}
