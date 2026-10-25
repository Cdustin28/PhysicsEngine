//! The Orbits tab's toolbar, sidebar, status bar, and 3D view.

use super::{M_MOON, OrbitEditor, SPEEDS, format_duration};
use crate::view3d::render::{BodyCallback, CameraView, Instance, SoftMeshes, SplatInstance};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui};
use engine3d::orbits::{AU, Kind, M_EARTH, M_SUN, R_EARTH, R_SUN};
use glam::Vec3;

/// Space is dark in both themes.
const SPACE: Color32 = Color32::from_rgb(0x0E, 0x10, 0x16);
const GOLD: Color32 = Color32::from_rgb(0xF1, 0xBE, 0x48);
const LABEL: Color32 = Color32::from_rgb(0xD6, 0xD8, 0xDE);

impl OrbitEditor {
    pub fn toolbar(&mut self, ui: &mut Ui) {
        if self.running {
            if ui.button("Pause").on_hover_text("Space").clicked() {
                self.pause();
            }
        } else if ui.button("Play").on_hover_text("Space").clicked() {
            self.play();
        }
        if ui
            .add_enabled(self.snapshot.is_some(), egui::Button::new("Reset"))
            .on_hover_text("Back to when Play was pressed (R)")
            .clicked()
        {
            self.reset();
        }
        egui::ComboBox::from_id_salt("orbit speed")
            .selected_text(SPEEDS[self.speed].0)
            .show_ui(ui, |ui| {
                for (k, (label, _)) in SPEEDS.iter().enumerate() {
                    ui.selectable_value(&mut self.speed, k, *label);
                }
            })
            .response
            .on_hover_text("Simulated time per second (+ and − keys)");
        ui.separator();
        ui.menu_button("Examples", |ui| {
            if ui.button("Solar System").clicked() {
                self.load_example(true);
                ui.close();
            }
            if ui.button("Empty star").clicked() {
                self.load_example(false);
                ui.close();
            }
        });
    }

    pub fn status_bar(&mut self, ui: &mut Ui) {
        ui.label(if self.running { "Running" } else { "Paused" });
        ui.separator();
        ui.label(format!("t = {}", format_duration(self.system.time)));
        ui.separator();
        let n = self.system.bodies.len();
        ui.label(format!("{n} {}", if n == 1 { "body" } else { "bodies" }));
        if self.running {
            ui.separator();
            if self.rate_limited {
                let why = self
                    .system
                    .tightest_pair()
                    .map(|(a, b)| {
                        format!(
                            " (limited by {} and {})",
                            self.system.bodies[a].name, self.system.bodies[b].name
                        )
                    })
                    .unwrap_or_default();
                ui.label(format!("{}/s{why}", format_duration(self.achieved_rate)))
                    .on_hover_text(
                        "The tightest orbit needs small steps, and there's a budget per frame, \
                         so time runs slower than the chosen speed.",
                    );
            } else {
                ui.label(format!("{} steps/frame", self.steps_last_frame));
            }
            if let Some(e0) = self.energy_at_play {
                ui.separator();
                let drift = (self.system.energy() - e0) / e0.abs();
                ui.label(format!("ΔE/E {drift:+.1e}"))
                    .on_hover_text("Change in total energy since Play: the integrator's error");
            }
        }
    }

    pub fn sidebar(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("System");
            self.body_tree(ui);
            ui.separator();
            self.inspector(ui);
            ui.separator();
            ui.heading("Add a planet");
            self.new_planet_ui(ui);
            ui.separator();
            ui.heading("View");
            ui.checkbox(&mut self.show_trails, "Orbit trails");
            ui.checkbox(&mut self.show_labels, "Names");
            ui.checkbox(&mut self.top_down, "Top-down (T)")
                .on_hover_text("Look straight down on the ecliptic");
            ui.horizontal(|ui| {
                ui.label("Smallest size");
                ui.add(egui::Slider::new(&mut self.min_size_px, 1.0..=20.0).suffix(" px"))
                    .on_hover_text(
                        "Bodies are drawn at their true size or this many pixels, whichever \
                         is bigger, so they stay visible zoomed out.",
                    );
            });
            ui.separator();
            egui::CollapsingHeader::new("Controls").show(ui, |ui| {
                ui.label(
                    "Click a body to select it; double-click (or F) to follow it.\n\
                     Drag to orbit the view, Shift+drag or middle-drag to pan, scroll to zoom.\n\
                     Space plays and pauses, R resets, + and − change speed, Delete removes.",
                );
            });
        });
    }

    /// The star, its planets, and their moons, as a clickable list.
    fn body_tree(&mut self, ui: &mut Ui) {
        let roots: Vec<usize> = (0..self.system.bodies.len())
            .filter(|&i| self.system.bodies[i].parent.is_none())
            .collect();
        for root in roots {
            self.tree_row(ui, root, 0);
        }
    }

    fn tree_row(&mut self, ui: &mut Ui, i: usize, depth: usize) {
        let b = &self.system.bodies[i];
        let color = Color32::from_rgb(b.color[0], b.color[1], b.color[2]);
        let name = b.name.clone();
        ui.horizontal(|ui| {
            ui.add_space(16.0 * depth as f32);
            let (dot, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
            ui.painter().circle_filled(dot.center(), 4.5, color);
            let response = ui.selectable_label(self.selected == Some(i), name);
            if response.clicked() {
                self.selected = Some(i);
            }
            if response.double_clicked() {
                self.follow(i);
            }
            if self.focus == i {
                ui.weak("(following)");
            }
        });
        let children: Vec<usize> = (0..self.system.bodies.len())
            .filter(|&k| self.system.bodies[k].parent == Some(i))
            .collect();
        for k in children {
            self.tree_row(ui, k, depth + 1);
        }
    }

    fn inspector(&mut self, ui: &mut Ui) {
        let Some(i) = self.selected.filter(|&i| i < self.system.bodies.len()) else {
            ui.heading("Selection");
            ui.weak("Click a body in the view or the list above.");
            return;
        };
        let kind = self.system.bodies[i].kind;
        ui.horizontal(|ui| {
            let mut name = self.system.bodies[i].name.clone();
            if ui.text_edit_singleline(&mut name).changed() && !name.trim().is_empty() {
                self.system.bodies[i].name = name;
            }
        });
        if let Some(p) = self.system.bodies[i].parent {
            ui.weak(format!(
                "{} of {}",
                if kind == Kind::Moon { "Moon" } else { "Planet" },
                self.system.bodies[p].name
            ));
        } else {
            ui.weak("Star");
        }

        // Distance: the edit that matters most, so first.
        if let Some(el) = self.system.elements(i) {
            let mut changed = None;
            match kind {
                Kind::Planet => {
                    let mut au = el.distance / AU;
                    if ui
                        .add(
                            egui::Slider::new(&mut au, 0.05..=100.0)
                                .logarithmic(true)
                                .text("AU from the star")
                                .clamping(egui::SliderClamping::Edits),
                        )
                        .on_hover_text(
                            "Puts it on a circular orbit at this distance; its moons come along",
                        )
                        .changed()
                    {
                        changed = Some(au * AU);
                    }
                }
                Kind::Moon => {
                    let parent = self.system.bodies[i].parent.unwrap_or(0);
                    let floor = self.system.bodies[parent].radius + self.system.bodies[i].radius;
                    let hill = self.hill_radius(parent);
                    let mut km = el.distance / 1e3;
                    let (lo, hi) = (
                        1.05 * floor / 1e3,
                        (1.2 * hill / 1e3).max(2.0 * floor / 1e3),
                    );
                    if ui
                        .add(
                            egui::Slider::new(&mut km, lo..=hi)
                                .logarithmic(true)
                                .text("km from its planet")
                                .clamping(egui::SliderClamping::Edits),
                        )
                        .on_hover_text(format!(
                            "Puts it on a circular orbit at this distance. Beyond about {:.0} km \
                             (half the planet's Hill radius) the star's pull takes over.",
                            0.5 * hill / 1e3
                        ))
                        .changed()
                    {
                        changed = Some(km * 1e3);
                    }
                }
                Kind::Star => {}
            }
            if let Some(m) = changed {
                self.set_distance(i, m);
            }
        }

        // Mass and size, in units that suit the body.
        let (mass_unit, mass_scale, radius_unit, radius_scale) = match kind {
            Kind::Star => ("Suns", M_SUN, "Sun radii", R_SUN),
            Kind::Planet => ("Earths", M_EARTH, "Earth radii", R_EARTH),
            Kind::Moon => ("Moons", M_MOON, "km", 1e3),
        };
        // Sliders here use `SliderClamping::Edits` and no rounding: with the
        // defaults, egui clamps and rounds the value every frame and reports
        // that as a change, which would re-place the body every frame.
        let mut mass = self.system.bodies[i].mass / mass_scale;
        let mass_range = match kind {
            Kind::Star => 0.08..=20.0,
            Kind::Planet => 0.01..=5000.0,
            Kind::Moon => 0.001..=100.0,
        };
        if ui
            .add(
                egui::Slider::new(&mut mass, mass_range)
                    .logarithmic(true)
                    .text(format!("mass ({mass_unit})"))
                    .clamping(egui::SliderClamping::Edits),
            )
            .changed()
        {
            self.set_mass(i, mass * mass_scale);
        }
        let mut radius = self.system.bodies[i].radius / radius_scale;
        let radius_range = match kind {
            Kind::Star => 0.1..=10.0,
            Kind::Planet => 0.05..=25.0,
            Kind::Moon => 5.0..=5000.0,
        };
        if ui
            .add(
                egui::Slider::new(&mut radius, radius_range)
                    .logarithmic(true)
                    .text(format!("radius ({radius_unit})"))
                    .clamping(egui::SliderClamping::Edits),
            )
            .changed()
        {
            self.system.bodies[i].radius = radius * radius_scale;
        }
        ui.horizontal(|ui| {
            ui.label("Color");
            ui.color_edit_button_srgb(&mut self.system.bodies[i].color);
        });

        // Live orbit readout.
        if let Some(el) = self.system.elements(i) {
            egui::Grid::new("orbit readout")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Distance");
                    ui.label(if kind == Kind::Moon {
                        format!("{} km", thousands(el.distance / 1e3))
                    } else {
                        format!("{:.4} AU", el.distance / AU)
                    });
                    ui.end_row();
                    ui.label("Speed");
                    ui.label(format!("{:.2} km/s", el.speed / 1e3));
                    ui.end_row();
                    ui.label("Period");
                    ui.label(match el.period {
                        Some(t) => format_duration(t),
                        None => "escaping".into(),
                    });
                    ui.end_row();
                    ui.label("Eccentricity");
                    ui.label(format!("{:.4}", el.eccentricity));
                    ui.end_row();
                    ui.label("Inclination");
                    ui.label(format!("{:.1}°", el.inclination_deg))
                        .on_hover_text("To the ecliptic; over 90° is retrograde");
                    ui.end_row();
                });
        }

        ui.horizontal(|ui| {
            if ui.button("Follow (F)").clicked() {
                self.follow(i);
            }
            if kind == Kind::Planet && ui.button("Add moon").clicked() {
                self.add_moon(i);
            }
            if kind != Kind::Star && ui.button("Remove").clicked() {
                self.remove(i);
            }
        });
    }

    fn new_planet_ui(&mut self, ui: &mut Ui) {
        let p = &mut self.new_planet;
        ui.add(
            egui::Slider::new(&mut p.distance_au, 0.05..=100.0)
                .logarithmic(true)
                .text("AU from the star")
                .max_decimals(3),
        );
        ui.add(
            egui::Slider::new(&mut p.mass_earths, 0.01..=5000.0)
                .logarithmic(true)
                .text("mass (Earths)")
                .max_decimals(3),
        );
        ui.add(
            egui::Slider::new(&mut p.radius_earths, 0.05..=25.0)
                .logarithmic(true)
                .text("radius (Earth radii)")
                .max_decimals(2),
        );
        ui.horizontal(|ui| {
            ui.label("Color");
            ui.color_edit_button_srgb(&mut p.color);
        });
        if ui
            .button("Add planet")
            .on_hover_text("On a circular orbit")
            .clicked()
        {
            self.add_planet();
        }
    }

    // ---------------------------------------------------------------------
    // The view

    pub fn canvas(&mut self, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let shift = ui.input(|i| i.modifiers.shift);

        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (-scroll * 0.0015).exp() / pinch;
            if factor != 1.0 {
                self.camera.zoom(factor);
            }
        }
        if response.dragged_by(PointerButton::Middle)
            || (shift
                && (response.dragged_by(PointerButton::Primary)
                    || response.dragged_by(PointerButton::Secondary)))
        {
            self.camera.pan(rect, response.drag_delta());
        } else if response.dragged_by(PointerButton::Primary)
            || response.dragged_by(PointerButton::Secondary)
        {
            let mut d = response.drag_delta();
            if self.top_down {
                d.y = 0.0;
            }
            self.camera.orbit(d);
        }
        self.apply_top_down();

        if let Some(p) = response.interact_pointer_pos() {
            if response.double_clicked_by(PointerButton::Primary) {
                if let Some(i) = self.pick(rect, p) {
                    self.selected = Some(i);
                    self.follow(i);
                }
            } else if response.clicked_by(PointerButton::Primary) {
                self.selected = self.pick(rect, p);
            }
        }

        painter.rect_filled(rect, 0.0, SPACE);
        if self.show_trails {
            self.draw_trails(&painter, rect);
        }
        if self.gpu_render {
            painter.add(self.body_callback(rect).into_shape(rect));
        } else {
            self.draw_flat(&painter, rect);
        }
        if self.show_labels {
            self.draw_labels(&painter, rect);
        }
        if let Some(i) = self.selected
            && let Some(s) = self.camera.project(rect, self.view_position(i))
        {
            let r = self.radius_px(rect, i) + 5.0;
            painter.circle_stroke(s, r, Stroke::new(1.5, GOLD));
        }
        self.draw_scale_bar(&painter, rect);
    }

    /// Distance from the eye to body `i`, view units.
    fn depth(&self, i: usize) -> f32 {
        (self.view_position(i) - self.camera.eye())
            .length()
            .max(1e-9)
    }

    /// How big body `i` is drawn, in view units (AU): its true radius or
    /// the minimum on-screen size, whichever is bigger.
    fn draw_radius(&self, rect: Rect, i: usize) -> f32 {
        let b = &self.system.bodies[i];
        let true_radius = (b.radius / AU) as f32;
        // Bigger planets get a bigger minimum, gently (Jupiter about 2.4×
        // Earth's, Mercury about 0.7×).
        let relative = ((b.radius / R_EARTH) as f32).powf(0.35).clamp(0.5, 3.0);
        let factor = match b.kind {
            Kind::Star => 2.4,
            Kind::Planet => relative,
            Kind::Moon => 0.6 * relative.max(0.8),
        };
        let floor = self.min_size_px * factor * self.camera.units_per_pixel(rect, self.depth(i));
        true_radius.max(floor)
    }

    fn radius_px(&self, rect: Rect, i: usize) -> f32 {
        self.draw_radius(rect, i) / self.camera.units_per_pixel(rect, self.depth(i))
    }

    /// The body under screen point `p`: the nearest one within a few pixels
    /// of its drawn edge.
    fn pick(&self, rect: Rect, p: Pos2) -> Option<usize> {
        let mut best = None;
        let mut best_d = f32::INFINITY;
        for i in 0..self.system.bodies.len() {
            let Some(s) = self.camera.project(rect, self.view_position(i)) else {
                continue;
            };
            let d = s.distance(p);
            if d <= self.radius_px(rect, i).max(6.0) + 4.0 && d < best_d {
                best = Some(i);
                best_d = d;
            }
        }
        best
    }

    fn body_callback(&self, rect: Rect) -> BodyCallback {
        let forward = (self.camera.target - self.camera.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        let mut spheres = Vec::new();
        let mut splats = Vec::new();
        let mut light = Vec3::ZERO;
        for (i, b) in self.system.bodies.iter().enumerate() {
            let position = self.view_position(i);
            let radius = self.draw_radius(rect, i);
            let star = b.kind == Kind::Star;
            if star {
                light = position;
                // A soft glow around the star.
                splats.push(SplatInstance {
                    position: position.to_array(),
                    size: radius * 4.0,
                    color: [255, 190, 110, 150],
                });
            }
            spheres.push(Instance {
                position: position.to_array(),
                checker: if star { 2.0 } else { 0.0 },
                scale: [radius; 3],
                highlight: f32::from(u8::from(self.selected == Some(i))),
                rotation: [0.0, 0.0, 0.0, 1.0],
                color: [b.color[0], b.color[1], b.color[2], 255],
            });
        }
        BodyCallback::new(
            Vec::new(),
            spheres,
            Vec::new(),
            SoftMeshes::default(),
            splats,
            CameraView {
                view_proj: self.camera.view_proj(rect),
                eye: self.camera.eye(),
                right,
                up,
            },
        )
        .with_point_light(light)
        .with_glowing_splats(true)
    }

    /// Without GPU rendering: flat discs.
    fn draw_flat(&self, painter: &egui::Painter, rect: Rect) {
        let mut order: Vec<usize> = (0..self.system.bodies.len()).collect();
        order.sort_by(|&a, &b| self.depth(b).total_cmp(&self.depth(a)));
        for i in order {
            if let Some(s) = self.camera.project(rect, self.view_position(i)) {
                let c = self.system.bodies[i].color;
                painter.circle_filled(
                    s,
                    self.radius_px(rect, i),
                    Color32::from_rgb(c[0], c[1], c[2]),
                );
            }
        }
    }

    /// Each body's path relative to its parent, fading toward the tail.
    fn draw_trails(&self, painter: &egui::Painter, rect: Rect) {
        let origin = self
            .system
            .bodies
            .get(self.focus)
            .map_or(glam::DVec3::ZERO, |b| b.position);
        for (i, trail) in self.trails.iter().enumerate() {
            let Some(b) = self.system.bodies.get(i) else {
                break;
            };
            let Some(parent) = b.parent else {
                continue;
            };
            let host = self.system.bodies[parent].position;
            let n = trail.points.len();
            if n < 2 {
                continue;
            }
            let screen: Vec<Option<Pos2>> = trail
                .points
                .iter()
                .chain(std::iter::once(&(b.position - host)))
                .map(|&rel| {
                    let world = ((host + rel - origin) / AU).as_vec3();
                    self.camera.project(rect, world)
                })
                .collect();
            let c = b.color;
            // Stop at the body's edge rather than drawing across it.
            let head = self.camera.project(rect, self.view_position(i));
            let clear = self.radius_px(rect, i) + 1.0;
            let outside = |p: Pos2| head.is_none_or(|h| h.distance(p) > clear);
            for k in 1..screen.len() {
                if let (Some(a), Some(z)) = (screen[k - 1], screen[k])
                    && outside(a)
                    && outside(z)
                {
                    let alpha = (k as f32 / screen.len() as f32).powf(1.5) * 0.85;
                    let color =
                        Color32::from_rgba_unmultiplied(c[0], c[1], c[2], (alpha * 255.0) as u8);
                    painter.line_segment([a, z], Stroke::new(1.3, color));
                }
            }
        }
    }

    /// Names beside bodies, most important first (the star, then the
    /// selected and followed bodies, then by mass); a name that would
    /// overlap one already drawn is skipped, as is a moon's while it's too
    /// close to its planet on screen to tell apart.
    fn draw_labels(&self, painter: &egui::Painter, rect: Rect) {
        let font = egui::FontId::proportional(12.0);
        let mut order: Vec<usize> = (0..self.system.bodies.len()).collect();
        let rank = |i: usize| {
            let b = &self.system.bodies[i];
            let first = b.kind == Kind::Star || Some(i) == self.selected || i == self.focus;
            (!first, -b.mass)
        };
        order.sort_by(|&a, &b| {
            rank(a)
                .partial_cmp(&rank(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut taken: Vec<Rect> = Vec::new();
        for i in order {
            let b = &self.system.bodies[i];
            let Some(s) = self.camera.project(rect, self.view_position(i)) else {
                continue;
            };
            if !rect.contains(s) {
                continue;
            }
            if let Some(p) = b.parent
                && b.kind == Kind::Moon
                && let Some(ps) = self.camera.project(rect, self.view_position(p))
                && ps.distance(s) < 16.0
            {
                continue;
            }
            let r = self.radius_px(rect, i);
            let galley = painter.layout_no_wrap(b.name.clone(), font.clone(), LABEL);
            let at = s + egui::vec2(r + 4.0, -r - 2.0 - galley.size().y);
            let label = Rect::from_min_size(at, galley.size()).expand(2.0);
            if taken.iter().any(|t| t.intersects(label)) {
                continue;
            }
            taken.push(label);
            painter.galley(at, galley, LABEL);
        }
    }

    /// A bar of round length in the corner, at the followed body's depth.
    fn draw_scale_bar(&self, painter: &egui::Painter, rect: Rect) {
        let per_px = self.camera.units_per_pixel(rect, self.camera.distance) as f64 * AU;
        let target_m = per_px * 120.0;
        let (value, unit, meters) = nice_length(target_m);
        let px = (meters / per_px) as f32;
        let y = rect.bottom() - 18.0;
        let x0 = rect.left() + 16.0;
        let stroke = Stroke::new(1.5, LABEL);
        painter.line_segment([Pos2::new(x0, y), Pos2::new(x0 + px, y)], stroke);
        for x in [x0, x0 + px] {
            painter.line_segment([Pos2::new(x, y - 4.0), Pos2::new(x, y + 4.0)], stroke);
        }
        painter.text(
            Pos2::new(x0 + px + 6.0, y),
            egui::Align2::LEFT_CENTER,
            format!("{value} {unit}"),
            egui::FontId::proportional(12.0),
            LABEL,
        );
    }
}

/// A round length near `meters`: (number shown, unit, its length in meters).
fn nice_length(meters: f64) -> (String, &'static str, f64) {
    let (unit, scale) = if meters >= 0.05 * AU {
        ("AU", AU)
    } else {
        ("km", 1e3)
    };
    let v = meters / scale;
    let p = 10f64.powf(v.log10().floor());
    let m = v / p;
    let nice = if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    } * p;
    let text = if nice >= 1.0 {
        thousands(nice)
    } else {
        let decimals = (-p.log10()).round().max(0.0) as usize;
        format!("{nice:.decimals$}")
    };
    (text, unit, nice * scale)
}

/// 384400 → "384,400".
pub(super) fn thousands(v: f64) -> String {
    let digits = format!("{}", v.round() as u64);
    let mut out = String::new();
    for (k, ch) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_bar_lengths_are_round() {
        assert_eq!(nice_length(3.1 * AU).0, "2");
        assert_eq!(nice_length(4.2e8).0, "500,000");
        assert_eq!(nice_length(0.3 * AU).1, "AU");
        assert_eq!(nice_length(0.3 * AU).0, "0.2");
    }
}
