//! Choosing a liquid to place, and coloring particles by liquid. Shared by
//! the 2D and 3D editors.

use eframe::egui::{self, Color32, Ui};
use engine::fluid::{Liquid, LiquidSpec, SEAWATER_SALINITY};
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

    /// Highest temperature this liquid can be placed at, °C: below boiling
    /// (water, at 1 atm) or the smoke point (oil). Boiling isn't simulated
    /// in these tabs.
    pub fn max_celsius(self) -> f32 {
        match self {
            LiquidChoice::Water | LiquidChoice::SaltWater => 99.0,
            LiquidChoice::Oil => 230.0,
            LiquidChoice::Mercury => 350.0,
        }
    }

    /// What to fill with at `celsius` (clamped to what this liquid allows).
    pub fn spec(self, celsius: f32) -> LiquidSpec {
        let c = celsius.clamp(0.0, self.max_celsius());
        LiquidSpec::new(self.liquid())
            .with_salinity(self.salinity())
            .at(c + 273.15)
    }

    /// A row of buttons, one per liquid, with each one's density on hover,
    /// and the temperature to place it at.
    pub fn picker(&mut self, ui: &mut Ui, celsius: &mut f32) {
        ui.horizontal_wrapped(|ui| {
            for c in Self::ALL {
                let rho = c.liquid().density(c.salinity());
                ui.selectable_value(self, c, c.name())
                    .on_hover_text(format!("{rho:.0} kg/m³ at 20 °C"));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Temperature");
            ui.add(
                egui::DragValue::new(celsius)
                    .range(0.0..=self.max_celsius())
                    .speed(0.5)
                    .suffix(" °C"),
            )
            .on_hover_text(
                "Warm liquid is lighter and rises; heat conducts at each liquid's real rate.",
            );
        });
        *celsius = celsius.clamp(0.0, self.max_celsius());
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

/// Temperature coloring: blue below 20 °C, the liquid's own color at 20 °C,
/// through orange to red by 100 °C above.
pub fn temperature_color(liquid: Liquid, salinity: f32, kelvin: f32) -> Color32 {
    let [r, g, b] = liquid.color(salinity);
    let base = Color32::from_rgb(r, g, b);
    let dt = kelvin - 293.15;
    if dt >= 0.0 {
        let hot = Color32::from_rgb(0xE8, 0x3A, 0x14);
        let warm = Color32::from_rgb(0xF5, 0x9E, 0x2A);
        let u = (dt / 100.0).min(1.0);
        if u < 0.5 {
            lerp(base, warm, u * 2.0)
        } else {
            lerp(warm, hot, (u - 0.5) * 2.0)
        }
    } else {
        lerp(
            base,
            Color32::from_rgb(0x4F, 0xA8, 0xF0),
            (-dt / 20.0).min(1.0),
        )
    }
}

fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

/// Shown under the placing tools.
pub fn hint(ui: &mut Ui) {
    ui.weak(
        "Liquids sort by density: mercury sinks, oil floats, salt water settles under fresh, \
         warm rises over cold. Salt and heat diffuse at their real (slow) rates.",
    );
}
