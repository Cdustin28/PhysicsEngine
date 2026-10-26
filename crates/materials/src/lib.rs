//! Physical properties of the liquids the 2D and 3D engines simulate, at
//! 20 °C and 1 atm, in SI units. Both engines read them from here, so the
//! numbers live in one place.
//!
//! Salt water isn't a separate liquid: it's [`Liquid::Water`] with a
//! salinity (mass fraction of salt). Salinity changes the density and
//! viscosity and diffuses between neighboring water particles.
//!
//! Sources: CRC Handbook of Chemistry and Physics (water, mercury);
//! UNESCO/TEOS-10 seawater tables (salt water at 35 g/kg); typical values
//! for vegetable (canola) oil. Interfacial tensions with oil vary by oil and
//! purity; the ones here are representative, not exact.

use serde::{Deserialize, Serialize};

/// Density of fresh water used as the reference, kg/m³. The engines'
/// `rest_density` setting is this, scaled for 2D (a 1 cm slab).
pub const WATER_DENSITY: f32 = 1000.0;

/// Salinity of ordinary seawater: 35 g of salt per kg.
pub const SEAWATER_SALINITY: f32 = 0.035;

/// Reference temperature for the properties below, K (20 °C).
pub const ROOM_TEMPERATURE: f32 = 293.15;

/// Diffusivity of dissolved salt (NaCl) in water, m²/s. Tiny: salt takes
/// hours to diffuse a few millimeters, so a brine layer under fresh water
/// stays sharp unless the water is stirred.
pub const SALT_DIFFUSIVITY: f32 = 1.5e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Liquid {
    #[default]
    Water,
    /// Vegetable (canola) oil.
    Oil,
    Mercury,
}

impl Liquid {
    pub const ALL: [Liquid; 3] = [Liquid::Water, Liquid::Oil, Liquid::Mercury];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Liquid::Water => "Water",
            Liquid::Oil => "Oil",
            Liquid::Mercury => "Mercury",
        }
    }

    /// Density, kg/m³, at `salinity` (water only; ignored otherwise).
    pub fn density(self, salinity: f32) -> f32 {
        match self {
            // Seawater at 20 °C: 1024.8 kg/m³ at 35 g/kg, close to linear
            // in salinity over that range.
            Liquid::Water => WATER_DENSITY * (1.0 + 0.76 * salinity.max(0.0)),
            Liquid::Oil => 920.0,
            Liquid::Mercury => 13_546.0,
        }
    }

    /// Dynamic viscosity, Pa·s.
    pub fn viscosity(self, salinity: f32) -> f32 {
        match self {
            // 1.002 mPa·s fresh, 1.08 at 35 g/kg.
            Liquid::Water => 1.002e-3 * (1.0 + 2.2 * salinity.max(0.0)),
            Liquid::Oil => 0.057,
            Liquid::Mercury => 1.554e-3,
        }
    }

    /// Surface tension against air (or vacuum), N/m.
    pub fn surface_tension(self) -> f32 {
        match self {
            Liquid::Water => 0.0728,
            Liquid::Oil => 0.032,
            Liquid::Mercury => 0.486,
        }
    }

    /// Density at temperature `t` (K) relative to its density at 20 °C:
    /// liquids expand when warmed. Water follows Tanaka et al. (2001)
    /// (densest at 4 °C: cold water sinks only down to 4 °C); oil and
    /// mercury expand linearly (β = 7.0 × 10⁻⁴ and 1.82 × 10⁻⁴ per K).
    pub fn expansion_factor(self, t: f32) -> f32 {
        let c = t - 273.15;
        match self {
            Liquid::Water => water_density_tanaka(c) / water_density_tanaka(20.0),
            Liquid::Oil => 1.0 - 7.0e-4 * (c - 20.0),
            Liquid::Mercury => 1.0 / (1.0 + 1.82e-4 * (c - 20.0)),
        }
    }

    /// Volumetric thermal expansion coefficient β = −(1/ρ) dρ/dT at `t`
    /// (K), 1/K. Water's is zero at 4 °C and negative below.
    pub fn expansion_coefficient(self, t: f32) -> f32 {
        let (a, b) = (
            self.expansion_factor(t - 0.5),
            self.expansion_factor(t + 0.5),
        );
        (a - b) / (0.5 * (a + b))
    }

    /// Heat transfer coefficient between this liquid at `t_liquid` and a
    /// surface held at `t_wall` (K) by natural convection, W/(m²·K).
    /// Turbulent natural convection, Nu = 0.13 Ra^(1/3) (between vertical
    /// walls' 0.10 and an upward-facing heated plate's 0.15; McAdams), with
    /// properties at the film temperature. With the 1/3 power the plate's
    /// size cancels, so this needs only the temperatures. For mercury
    /// (Prandtl number 0.025) it's rough; conduction dominates there anyway.
    pub fn natural_convection(self, salinity: f32, t_liquid: f32, t_wall: f32) -> f32 {
        let film = 0.5 * (t_liquid + t_wall);
        let rho = self.density_at(salinity, film);
        let k = self.conductivity(salinity);
        let nu = self.viscosity_at(salinity, film) / rho;
        let alpha = k / (rho * self.specific_heat(salinity));
        let beta = self.expansion_coefficient(film).abs();
        let ra_per_cubic_metre = 9.81 * beta * (t_wall - t_liquid).abs() / (nu * alpha);
        0.13 * k * ra_per_cubic_metre.cbrt()
    }

    /// Density at `salinity` and temperature `t` (K), kg/m³.
    pub fn density_at(self, salinity: f32, t: f32) -> f32 {
        self.density(salinity) * self.expansion_factor(t)
    }

    /// Dynamic viscosity at temperature `t` (K), Pa·s. Water: Vogel's
    /// equation, μ = 2.414 × 10⁻⁵ · 10^(247.8 / (T − 140)); oil and mercury:
    /// Andrade's, μ = A e^(B/T), through their 20 °C and 100 °C values
    /// (oil 0.057 → 0.0085, mercury 1.554 → 1.24 mPa·s).
    pub fn viscosity_at(self, salinity: f32, t: f32) -> f32 {
        let t = t.clamp(250.0, 650.0);
        let relative = match self {
            Liquid::Water => {
                let vogel = |t: f32| 2.414e-5 * 10f32.powf(247.8 / (t - 140.0));
                vogel(t) / vogel(ROOM_TEMPERATURE)
            }
            Liquid::Oil => (2603.0 * (1.0 / t - 1.0 / ROOM_TEMPERATURE)).exp(),
            Liquid::Mercury => (309.0 * (1.0 / t - 1.0 / ROOM_TEMPERATURE)).exp(),
        };
        self.viscosity(salinity) * relative
    }

    /// Specific heat capacity, J/(kg·K).
    pub fn specific_heat(self, salinity: f32) -> f32 {
        match self {
            // 4182 fresh, 3993 at 35 g/kg.
            Liquid::Water => 4182.0 * (1.0 - 1.29 * salinity.max(0.0)),
            Liquid::Oil => 1970.0,
            Liquid::Mercury => 139.5,
        }
    }

    /// The hottest the liquid can get as a liquid at 1 atm, K: water's and
    /// mercury's boiling points (salt raises water's by about 0.5 K at
    /// seawater salinity; ebullioscopic constant 0.512 K·kg/mol, two ions
    /// per NaCl), and for oil its smoke point, where it starts to break
    /// down.
    pub fn max_temperature(self, salinity: f32) -> f32 {
        match self {
            Liquid::Water => {
                let molality = salinity.max(0.0) / (0.058_44 * (1.0 - salinity.clamp(0.0, 0.5)));
                373.15 + 0.512 * 2.0 * molality
            }
            Liquid::Oil => 503.15,
            Liquid::Mercury => 629.88,
        }
    }

    /// The coldest the liquid gets before it freezes, K: water's freezing
    /// point (salt lowers it, cryoscopic constant 1.86 K·kg/mol: seawater
    /// freezes near −1.9 °C), canola oil's pour point, mercury's melting
    /// point.
    pub fn min_temperature(self, salinity: f32) -> f32 {
        match self {
            Liquid::Water => {
                let molality = salinity.max(0.0) / (0.058_44 * (1.0 - salinity.clamp(0.0, 0.5)));
                // 0.83: the osmotic coefficient of seawater-strength brine.
                273.15 - 1.86 * 2.0 * molality * 0.83
            }
            Liquid::Oil => 263.15,
            Liquid::Mercury => 234.32,
        }
    }

    /// Thermal conductivity, W/(m·K).
    pub fn conductivity(self, salinity: f32) -> f32 {
        match self {
            Liquid::Water => 0.598 * (1.0 - 0.1 * salinity.max(0.0)),
            Liquid::Oil => 0.17,
            Liquid::Mercury => 8.3,
        }
    }

    /// Display color, sRGB.
    pub fn color(self, salinity: f32) -> [u8; 3] {
        match self {
            Liquid::Water if salinity > 0.005 => [40, 120, 150],
            Liquid::Water => [31, 78, 154],
            Liquid::Oil => [214, 168, 52],
            Liquid::Mercury => [176, 180, 188],
        }
    }
}

/// What to fill with: a liquid, its salinity (water only), and its
/// temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiquidSpec {
    pub liquid: Liquid,
    /// Mass fraction of salt; ignored except for water.
    pub salinity: f32,
    /// K.
    pub temperature: f32,
}

impl LiquidSpec {
    /// `liquid`, fresh, at room temperature.
    pub fn new(liquid: Liquid) -> Self {
        Self {
            liquid,
            salinity: 0.0,
            temperature: ROOM_TEMPERATURE,
        }
    }

    /// Fresh water at room temperature.
    pub fn water() -> Self {
        Self::new(Liquid::Water)
    }

    /// Seawater (35 g/kg) at room temperature.
    pub fn seawater() -> Self {
        Self::water().with_salinity(SEAWATER_SALINITY)
    }

    pub fn with_salinity(mut self, salinity: f32) -> Self {
        self.salinity = salinity;
        self
    }

    /// At `temperature` K.
    pub fn at(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Salinity as stored: clamped, and zero for anything but water.
    pub fn stored_salinity(&self) -> f32 {
        if self.liquid == Liquid::Water {
            self.salinity.clamp(0.0, 0.4)
        } else {
            0.0
        }
    }
}

impl From<Liquid> for LiquidSpec {
    fn from(liquid: Liquid) -> Self {
        Self::new(liquid)
    }
}

/// Density of pure water at `c` °C, kg/m³ (Tanaka et al. 2001, valid
/// 0–40 °C to a few ppm and within 0.1% to 100 °C).
fn water_density_tanaka(c: f32) -> f32 {
    let c = f64::from(c.clamp(0.0, 100.0));
    let (a1, a2, a3, a4, a5) = (-3.983_035, 301.797, 522_528.9, 69.348_81, 999.974_95);
    (a5 * (1.0 - (c + a1) * (c + a1) * (c + a2) / (a3 * (c + a4)))) as f32
}

/// Interfacial tension between two different liquids, N/m (between a liquid
/// and itself: 0, there's no interface).
pub fn interfacial_tension(a: Liquid, b: Liquid) -> f32 {
    use Liquid::*;
    match (a, b) {
        (x, y) if x == y => 0.0,
        (Water, Oil) | (Oil, Water) => 0.025,
        (Water, Mercury) | (Mercury, Water) => 0.375,
        (Oil, Mercury) | (Mercury, Oil) => 0.35,
        _ => unreachable!(),
    }
}

/// The work of adhesion between `a` and `b` (Dupré): the energy per area to
/// pull them apart, W = σ_a + σ_b − σ_ab, N/m. For a liquid and itself it's
/// the work of cohesion, 2σ. Cohesive forces between particles are scaled
/// to this, which makes each liquid's surface tension, and the tension of
/// every interface between two liquids, come out right.
pub fn work_of_adhesion(a: Liquid, b: Liquid) -> f32 {
    a.surface_tension() + b.surface_tension() - interfacial_tension(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn densities_and_layering() {
        assert!((Liquid::Water.density(SEAWATER_SALINITY) - 1026.6).abs() < 2.0);
        // Oil floats on water, which floats on mercury.
        assert!(Liquid::Oil.density(0.0) < Liquid::Water.density(0.0));
        assert!(Liquid::Water.density(0.04) < Liquid::Mercury.density(0.0));
    }

    #[test]
    fn dupre_is_symmetric_and_cohesion_is_twice_sigma() {
        for a in Liquid::ALL {
            assert_eq!(work_of_adhesion(a, a), 2.0 * a.surface_tension());
            for b in Liquid::ALL {
                assert_eq!(work_of_adhesion(a, b), work_of_adhesion(b, a));
                // Immiscible: sticking to your own kind beats the other.
                if a != b {
                    let mixed = work_of_adhesion(a, b);
                    assert!(mixed < 0.5 * (work_of_adhesion(a, a) + work_of_adhesion(b, b)));
                }
            }
        }
    }

    #[test]
    fn thermal_properties_match_tables() {
        let k = |c: f32| c + 273.15;
        // Water: densest near 4 °C; 958.4 kg/m³ at 100 °C (CRC).
        let w = Liquid::Water;
        assert!(w.expansion_factor(k(4.0)) > w.expansion_factor(k(0.0)));
        assert!(w.expansion_factor(k(4.0)) > w.expansion_factor(k(10.0)));
        assert!((water_density_tanaka(100.0) - 958.4).abs() < 1.0);
        assert!((water_density_tanaka(20.0) - 998.2).abs() < 0.1);
        // Water viscosity: 1.002 mPa·s at 20 °C, 0.282 at 100 °C.
        assert!((w.viscosity_at(0.0, k(100.0)) - 0.282e-3).abs() < 0.01e-3);
        // Oil at 100 °C, mercury at 100 °C.
        assert!((Liquid::Oil.viscosity_at(0.0, k(100.0)) - 0.0085).abs() < 1e-4);
        assert!((Liquid::Mercury.viscosity_at(0.0, k(100.0)) - 1.24e-3).abs() < 1e-5);
        // Seawater's specific heat ≈ 3993 J/(kg·K).
        assert!((w.specific_heat(SEAWATER_SALINITY) - 3993.0).abs() < 5.0);
    }

    #[test]
    fn serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_string(&Liquid::Mercury).unwrap(),
            "\"mercury\""
        );
    }
}
