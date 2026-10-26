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
    fn serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_string(&Liquid::Mercury).unwrap(),
            "\"mercury\""
        );
    }
}
