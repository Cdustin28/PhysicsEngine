//! Joint and chain tools, and drawing joints.
//!
//! - **Joint tool (J)**: Pin: click where two bodies overlap to pin them
//!   together there, or on one body to pin it to the world. Rod, Rope,
//!   Spring: drag from a point on one body to a point on another (or to
//!   empty space, for the world).
//! - **Chain tool (L)**: drag to lay a chain; each end is pinned to the body
//!   under it, or to the world.

use super::{EditorApp, GOLD};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Stroke, Ui};
use engine::Vec2;
use engine::joint::{Anchor, Joint, JointKind, add_chain};
use serde::{Deserialize, Serialize};

const JOINT_COLOR: Color32 = Color32::from_rgb(0x3C, 0x3C, 0x3C);
const ROPE_COLOR: Color32 = Color32::from_rgb(0x8B, 0x5A, 0x2B);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum JointType {
    Pin,
    Rod,
    Rope,
    Spring,
}

/// Settings for new joints and chains, remembered between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct JointSettings {
    pub kind: JointType,
    /// N/m
    pub stiffness: f32,
    /// N·s/m
    pub damping: f32,
    /// Approximate length of each chain link, m.
    pub link_length: f32,
    pub link_thickness: f32,
    /// Pin chain ends that aren't on a body to the world.
    pub pin_free_ends: bool,
}

impl Default for JointSettings {
    fn default() -> Self {
        Self {
            kind: JointType::Pin,
            stiffness: 50.0,
            damping: 1.0,
            link_length: 0.25,
            link_thickness: 0.08,
            pin_free_ends: true,
        }
    }
}

impl EditorApp {
    /// Indices of the bodies containing `p`, topmost (drawn last) first.
    fn bodies_at(&self, p: Vec2) -> Vec<usize> {
        (0..self.world.bodies.len())
            .rev()
            .filter(|&i| self.world.bodies[i].contains_point(p))
            .collect()
    }

    pub(super) fn joint_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.joint_settings;
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut s.kind, JointType::Pin, "Pin");
            ui.selectable_value(&mut s.kind, JointType::Rod, "Rod");
            ui.selectable_value(&mut s.kind, JointType::Rope, "Rope");
            ui.selectable_value(&mut s.kind, JointType::Spring, "Spring");
        });
        if s.kind == JointType::Spring {
            egui::Grid::new("spring").num_columns(2).show(ui, |ui| {
                ui.label("Stiffness");
                ui.add(
                    egui::DragValue::new(&mut s.stiffness)
                        .speed(1.0)
                        .range(0.0..=1.0e6)
                        .suffix(" N/m"),
                );
                ui.end_row();
                ui.label("Damping");
                ui.add(
                    egui::DragValue::new(&mut s.damping)
                        .speed(0.1)
                        .range(0.0..=1.0e5)
                        .suffix(" N·s/m"),
                );
                ui.end_row();
            });
        }
        ui.weak(match s.kind {
            JointType::Pin => {
                "Click where two bodies overlap to pin them together, or on one body to pin it to the world."
            }
            JointType::Rod => "Drag from one body to another (or to empty space) for a rigid rod.",
            JointType::Rope => "Drag from one body to another (or to empty space) for a rope: it can go slack but not stretch.",
            JointType::Spring => "Drag from one body to another (or to empty space) for a spring, relaxed at the length you drag.",
        });
    }

    pub(super) fn chain_tool_options(&mut self, ui: &mut Ui) {
        let s = &mut self.joint_settings;
        ui.add_space(6.0);
        egui::Grid::new("chain").num_columns(2).show(ui, |ui| {
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
        ui.checkbox(
            &mut s.pin_free_ends,
            "Pin ends to the world when not on a body",
        );
        ui.weak("Drag to lay a chain. Each end is pinned to the body under it.");
    }

    pub(super) fn joint_tool_input(
        &mut self,
        ui: &Ui,
        response: &egui::Response,
        cursor: Option<Vec2>,
    ) {
        if self.joint_settings.kind == JointType::Pin {
            if response.clicked_by(PointerButton::Primary)
                && let Some(p) = cursor
            {
                let p = self.snap_point(p);
                let under = self.bodies_at(p);
                let (a, b) = match under[..] {
                    [] => {
                        self.status = "Click on a body to pin it.".into();
                        return;
                    }
                    [a] => (a, None),
                    [a, b, ..] => (a, Some(b)),
                };
                if b.is_none() && self.world.bodies[a].is_static {
                    self.status = "That body is static already; pin a dynamic body to it.".into();
                    return;
                }
                let joint = Joint::pin(&self.world.bodies, a, b, p);
                self.add_joint(joint, "Pinned");
            }
            return;
        }
        if response.drag_started_by(PointerButton::Primary) {
            let press = ui.input(|i| i.pointer.press_origin()).and(cursor);
            self.joint_drag = press.and_then(|p| {
                let p = self.snap_point(p);
                let a = self.bodies_at(p).first().copied();
                if a.is_none() {
                    self.status = "Start the drag on a body.".into();
                }
                a.map(|a| (a, p))
            });
        }
        if response.drag_stopped_by(PointerButton::Primary)
            && let (Some((a, pa)), Some(pb)) = (self.joint_drag.take(), cursor)
        {
            let pb = self.snap_point(pb);
            let b = self.bodies_at(pb).into_iter().find(|&i| i != a);
            if (pb - pa).length() < 1e-3 {
                self.status = "Drag farther to make a joint.".into();
                return;
            }
            if b.is_none() && self.world.bodies[a].is_static {
                self.status = "Both ends would be fixed; start on a dynamic body.".into();
                return;
            }
            let bodies = &self.world.bodies;
            let s = &self.joint_settings;
            let joint = match s.kind {
                JointType::Rod => Joint::rod(bodies, a, pa, b, pb),
                JointType::Rope => Joint::rope(bodies, a, pa, b, pb),
                JointType::Spring => Joint::spring(bodies, a, pa, b, pb, s.stiffness, s.damping),
                JointType::Pin => unreachable!("pins are placed by clicking"),
            };
            self.add_joint(joint, "Joined");
        }
    }

    fn add_joint(&mut self, joint: Joint, verb: &str) {
        match self.world.add_joint(joint) {
            Ok(_) => {
                self.status = format!("{verb}. {} joints in the scene.", self.world.joints.len())
            }
            Err(e) => self.status = format!("Couldn't add that joint: {e}"),
        }
    }

    pub(super) fn chain_tool_input(
        &mut self,
        ui: &Ui,
        response: &egui::Response,
        cursor: Option<Vec2>,
    ) {
        if response.drag_started_by(PointerButton::Primary) {
            let press = ui.input(|i| i.pointer.press_origin()).and(cursor);
            self.chain_start = press.map(|p| self.snap_point(p));
        }
        if response.drag_stopped_by(PointerButton::Primary)
            && let (Some(start), Some(end)) = (self.chain_start.take(), cursor)
        {
            let end = self.snap_point(end);
            let s = self.joint_settings.clone();
            let length = (end - start).length();
            if length < s.link_length {
                self.status = "Drag farther to make a chain.".into();
                return;
            }
            let anchor = |editor: &EditorApp, p: Vec2| match editor.bodies_at(p).first() {
                Some(&i) => Anchor::Body(i),
                None if s.pin_free_ends => Anchor::World,
                None => Anchor::Free,
            };
            let (a0, a1) = (anchor(self, start), anchor(self, end));
            let links = ((length / s.link_length).round() as usize).max(2);
            add_chain(&mut self.world, start, end, links, s.link_thickness, a0, a1);
            // Chains whip; substeps keep their pins tight.
            self.world.solver.substeps = self.world.solver.substeps.max(4);
            self.status = format!("Added a chain of {links} links.");
        }
    }

    /// Draw every joint: pins as gold dots, rods as lines, ropes in brown
    /// (lighter when slack), springs as zigzags.
    pub(super) fn draw_joints(&self, painter: &egui::Painter, viewport: Rect) {
        for joint in &self.world.joints {
            let (pa, pb) = joint.anchors(&self.world.bodies);
            let (a, b) = (
                self.camera.world_to_screen(viewport, pa),
                self.camera.world_to_screen(viewport, pb),
            );
            match joint.kind {
                JointKind::Pin => {}
                JointKind::Distance {
                    min_length,
                    max_length,
                } => {
                    if min_length == max_length {
                        painter.line_segment([a, b], Stroke::new(2.5, JOINT_COLOR));
                    } else {
                        let slack = (pb - pa).length() < max_length - 1e-3;
                        let color = if slack {
                            ROPE_COLOR.gamma_multiply(0.5)
                        } else {
                            ROPE_COLOR
                        };
                        painter.line_segment([a, b], Stroke::new(2.0, color));
                    }
                }
                JointKind::Spring { .. } => draw_zigzag(painter, a, b),
            }
            painter.circle_filled(a, 3.0, JOINT_COLOR);
            painter.circle(b, 4.0, GOLD, Stroke::new(1.5, JOINT_COLOR));
        }
    }

    /// The rope/rod/spring or chain being dragged out.
    pub(super) fn draw_joint_preview(
        &self,
        painter: &egui::Painter,
        viewport: Rect,
        cursor: Option<Vec2>,
    ) {
        let start = self.joint_drag.map(|(_, p)| p).or(self.chain_start);
        if let (Some(start), Some(end)) = (start, cursor) {
            let a = self.camera.world_to_screen(viewport, start);
            let b = self.camera.world_to_screen(viewport, self.snap_point(end));
            painter.line_segment([a, b], Stroke::new(2.0, GOLD.gamma_multiply(0.8)));
            painter.circle_filled(a, 4.0, GOLD);
        }
    }
}

fn draw_zigzag(painter: &egui::Painter, a: Pos2, b: Pos2) {
    let d = b - a;
    let len = d.length();
    if len < 1.0 {
        return;
    }
    let along = d / len;
    let across = egui::vec2(-along.y, along.x) * 6.0;
    let coils = 10;
    // Straight leads at each end, zigzag in the middle.
    let lead = 0.1;
    let mut points = vec![a, a + d * lead];
    for k in 0..coils {
        let t = lead + (1.0 - 2.0 * lead) * (k as f32 + 0.5) / coils as f32;
        let side = if k % 2 == 0 { 1.0 } else { -1.0 };
        points.push(a + d * t + across * side);
    }
    points.push(a + d * (1.0 - lead));
    points.push(b);
    painter.add(egui::Shape::line(points, Stroke::new(1.5, JOINT_COLOR)));
}
