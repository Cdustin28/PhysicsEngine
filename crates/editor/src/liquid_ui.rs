//! Choosing a liquid to place, and coloring particles by liquid. Shared by
//! the 2D and 3D editors.

use eframe::egui::{Color32, Ui};
use engine::fluid::{Liquid, SEAWATER_SALINITY};
use serde::{Deserialize, Serialize};

/// The liquids the placing tools offer. Salt water is water with
/// seawater's salinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LiquidChoice {
    #[default]
    Water,
    SaltWater,
    Oil,
    Mercury,
}

impl LiquidChoice {
    const ALL: [LiquidChoice; 4] = [
        LiquidChoice::Water,
        LiquidChoice::SaltWater,
        LiquidChoice::Oil,
        LiquidChoice::Mercury,
    ];

    pub fn liquid(self) -> Liquid {
        match self {
            LiquidChoice::Water | LiquidChoice::SaltWater => Liquid::Water,
            LiquidChoice::Oil => Liquid::Oil,
            LiquidChoice::Mercury => Liquid::Mercury,
        }
    }

    pub fn salinity(self) -> f32 {
        if self == LiquidChoice::SaltWater {
            SEAWATER_SALINITY
        } else {
            0.0
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LiquidChoice::Water => "Water",
            LiquidChoice::SaltWater => "Salt water",
            LiquidChoice::Oil => "Oil",
            LiquidChoice::Mercury => "Mercury",
        }
    }

    /// Lowercase, for status messages.
    pub fn noun(self) -> &'static str {
        match self {
            LiquidChoice::Water => "water",
            LiquidChoice::SaltWater => "salt water",
            LiquidChoice::Oil => "oil",
            LiquidChoice::Mercury => "mercury",
        }
    }

    /// A row of buttons, one per liquid, with each one's density on hover.
    pub fn picker(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for c in Self::ALL {
                let rho = c.liquid().density(c.salinity());
                ui.selectable_value(self, c, c.name())
                    .on_hover_text(format!("{rho:.0} kg/m³"));
            }
        });
    }
}

/// Particle color: a liquid's own color, shaded toward a lighter tint as
/// `t` (0–1: speed or pressure) rises. Fresh water keeps the editors'
/// original deep-to-pale blue.
pub fn particle_color(
    liquid: Liquid,
    salinity: f32,
    t: f32,
    deep: Color32,
    light: Color32,
) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    if liquid == Liquid::Water && salinity <= 0.005 {
        return lerp(deep, light, t);
    }
    let [r, g, b] = liquid.color(salinity);
    let base = Color32::from_rgb(r, g, b);
    let pale = lerp(base, Color32::WHITE, 0.55);
    lerp(base, pale, t)
}

fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

/// Shown under the placing tools.
pub fn hint(ui: &mut Ui) {
    ui.weak(
        "Liquids sort by density: mercury sinks, oil floats, salt water settles under fresh. \
         Salt diffuses at its real (very slow) rate.",
    );
}
