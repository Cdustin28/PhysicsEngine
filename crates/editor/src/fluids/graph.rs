//! The Fluids tab's history of totals (energies, temperatures) and a small
//! line plot of one of them.

use eframe::egui::{self, Color32, Pos2, Stroke, Ui};
use engine3d::World;
use serde::{Deserialize, Serialize};

/// Simulated seconds between samples.
const SAMPLE_EVERY: f32 = 0.05;
/// Samples kept; when full, every other one is dropped and the interval
/// doubles, so the plot always spans the whole run.
const CAPACITY: usize = 600;

#[derive(Debug, Clone, Copy, Default)]
pub struct Sample {
    pub time: f32,
    /// J.
    pub kinetic: f64,
    /// J, above the floor.
    pub potential: f64,
    /// J, above 20 °C.
    pub thermal: f64,
    /// Heat the brush has added, J.
    pub brush: f64,
    /// °C.
    pub mean_celsius: f32,
    pub max_celsius: f32,
}

/// A plotted line: name, color, and how to read it from a sample.
type Series = (&'static str, Color32, fn(&Sample) -> f64);

/// Which quantity the plot shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Quantity {
    #[default]
    Energy,
    Temperature,
}

#[derive(Default)]
pub struct Graph {
    samples: Vec<Sample>,
    interval: f32,
    next: f32,
    pub quantity: Quantity,
}

impl Graph {
    pub fn clear(&mut self) {
        self.samples.clear();
        self.interval = 0.0;
        self.next = 0.0;
    }

    pub fn latest(&self) -> Option<&Sample> {
        self.samples.last()
    }

    /// Sample the world if it's time.
    pub fn record(&mut self, world: &World, brush: f64) {
        if self.interval == 0.0 {
            self.interval = SAMPLE_EVERY;
        }
        let t = world.time as f32;
        if t < self.next && !self.samples.is_empty() {
            return;
        }
        self.next = t + self.interval;
        self.samples.push(sample(world, brush));
        if self.samples.len() >= CAPACITY {
            let mut k = 0;
            self.samples.retain(|_| {
                k += 1;
                k % 2 == 1
            });
            self.interval *= 2.0;
        }
    }

    /// The plot: energies (kinetic, potential, thermal above 20 °C, and the
    /// brush's input) or temperatures (mean and hottest).
    pub fn show(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.quantity, Quantity::Energy, "Energy");
            ui.selectable_value(&mut self.quantity, Quantity::Temperature, "Temperature");
        });
        let width = ui.available_width().max(120.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 130.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);
        if self.samples.len() < 2 {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Press Play to record",
                egui::FontId::proportional(12.0),
                visuals.weak_text_color(),
            );
            return;
        }
        let series: Vec<Series> = match self.quantity {
            Quantity::Energy => vec![
                (
                    "kinetic",
                    Color32::from_rgb(0x2E, 0x8B, 0x57),
                    |s: &Sample| s.kinetic,
                ),
                (
                    "potential",
                    Color32::from_rgb(0x3B, 0x6E, 0xC8),
                    |s: &Sample| s.potential,
                ),
                (
                    "thermal",
                    Color32::from_rgb(0xC8, 0x10, 0x2E),
                    |s: &Sample| s.thermal,
                ),
                (
                    "brush",
                    Color32::from_rgb(0xF1, 0xBE, 0x48),
                    |s: &Sample| s.brush,
                ),
            ],
            Quantity::Temperature => vec![
                ("mean", Color32::from_rgb(0xC8, 0x10, 0x2E), |s: &Sample| {
                    f64::from(s.mean_celsius)
                }),
                (
                    "hottest",
                    Color32::from_rgb(0xF1, 0xBE, 0x48),
                    |s: &Sample| f64::from(s.max_celsius),
                ),
            ],
        };
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for (_, _, f) in &series {
            for s in &self.samples {
                let v = f(s);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        if hi - lo < 1e-9 {
            hi = lo + 1.0;
        }
        let pad = 0.05 * (hi - lo);
        let (lo, hi) = (lo - pad, hi + pad);
        let t0 = self.samples[0].time;
        let t1 = self.samples[self.samples.len() - 1].time.max(t0 + 1e-6);
        let plot = rect.shrink2(egui::vec2(6.0, 16.0));
        let to_screen = |t: f32, v: f64| {
            Pos2::new(
                plot.left() + (t - t0) / (t1 - t0) * plot.width(),
                plot.bottom() - ((v - lo) / (hi - lo)) as f32 * plot.height(),
            )
        };
        for (_, color, f) in &series {
            let points: Vec<Pos2> = self
                .samples
                .iter()
                .map(|s| to_screen(s.time, f(s)))
                .collect();
            painter.add(egui::Shape::line(points, Stroke::new(1.5, *color)));
        }
        let unit = |v: f64| match self.quantity {
            Quantity::Energy => format_energy(v),
            Quantity::Temperature => format!("{v:.1} °C"),
        };
        let text = visuals.weak_text_color();
        let font = egui::FontId::proportional(10.0);
        painter.text(
            rect.left_top() + egui::vec2(4.0, 2.0),
            egui::Align2::LEFT_TOP,
            unit(hi),
            font.clone(),
            text,
        );
        painter.text(
            rect.left_bottom() + egui::vec2(4.0, -2.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{} · {:.1} s", unit(lo), t1 - t0),
            font.clone(),
            text,
        );
        // Legend, right-aligned along the top.
        let mut x = rect.right() - 4.0;
        for (name, color, _) in series.iter().rev() {
            let galley = painter.layout_no_wrap((*name).to_owned(), font.clone(), *color);
            x -= galley.size().x;
            painter.galley(Pos2::new(x, rect.top() + 2.0), galley, *color);
            x -= 8.0;
        }
    }
}

fn sample(world: &World, brush: f64) -> Sample {
    let fluid = &world.fluid;
    let n = fluid.len().max(1) as f32;
    let (sum, max) = fluid
        .temperatures
        .iter()
        .fold((0.0f64, f64::MIN), |(s, m), &t| (s + t, m.max(t)));
    Sample {
        time: world.time as f32,
        kinetic: f64::from(fluid.kinetic_energy()),
        potential: super::potential_energy(world),
        thermal: super::thermal_energy_above_room(world),
        brush,
        mean_celsius: super::celsius((sum / f64::from(n)) as f32),
        max_celsius: if fluid.is_empty() {
            20.0
        } else {
            super::celsius(max as f32)
        },
    }
}

/// Joules with a sensible prefix.
pub fn format_energy(j: f64) -> String {
    let a = j.abs();
    if a >= 1e6 {
        format!("{:.2} MJ", j / 1e6)
    } else if a >= 1e3 {
        format!("{:.2} kJ", j / 1e3)
    } else {
        format!("{j:.1} J")
    }
}
