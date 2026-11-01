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

    /// Heat transfer coefficient between this liquid at `t_liquid` and a
    /// surface at `t_wall` (K), W/(m²·K): natural convection, or for water
    /// against a surface past its boiling point, nucleate boiling (Jakob's
    /// correlation for water at 1 atm, h = 5.56 ΔT³ with ΔT the wall's
    /// excess over boiling), capped at the critical heat flux (1.26 MW/m²),
    /// where the surface would blanket itself in vapor.
    pub fn wall_heat_transfer(self, salinity: f32, t_liquid: f32, t_wall: f32) -> f32 {
        let natural = self.natural_convection(salinity, t_liquid, t_wall);
        let boil = self.max_temperature(salinity);
        if self != Liquid::Water || t_wall <= boil {
            return natural;
        }
        let excess = t_wall - boil;
        let nucleate = 5.56 * excess.powi(3);
        let critical = 1.26e6 / (t_wall - t_liquid).abs().max(1e-3);
        natural.max(nucleate.min(critical))
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
    let (a1, a2, a3, a4, a5) = (-3.983_035, 301.797, 522_528.9, 69.348_81, 999.974_95);
    let rho = |c: f64| a5 * (1.0 - (c + a1) * (c + a1) * (c + a2) / (a3 * (c + a4)));
    let c = f64::from(c.max(0.0));
    if c <= 100.0 {
        rho(c) as f32
    } else {
        // Past the fit's range (water just above boiling, before it
        // flashes), continue at its 100 °C expansion rate (7.5 × 10⁻⁴/K).
        let slope = rho(100.0) - rho(99.0);
        (rho(100.0) + slope * (c - 100.0)) as f32
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

/// Water's phase change: saturation (boiling point vs pressure), latent
/// heat, and vapor. Used for boiling, evaporation, and condensation.
pub mod water {
    /// Molar mass of water, kg/mol.
    pub const MOLAR_MASS: f32 = 0.018_015;
    /// Gas constant, J/(mol·K).
    pub const GAS_CONSTANT: f32 = 8.314_462;
    /// Triple point: below this pressure liquid water can't exist, Pa.
    pub const TRIPLE_PRESSURE: f32 = 611.657;
    /// One standard atmosphere, Pa.
    pub const ATMOSPHERE: f32 = 101_325.0;
    /// Density of dry air at 20 °C and 1 atm, kg/m³.
    pub const AIR_DENSITY: f32 = 1.204;
    /// Specific heat of air, J/(kg·K).
    pub const AIR_SPECIFIC_HEAT: f32 = 1005.0;

    /// IAPWS-IF97 region 4 (the saturation line) coefficients.
    const N: [f64; 10] = [
        0.116_705_214_527_67e4,
        -0.724_213_167_032_06e6,
        -0.170_738_469_400_92e2,
        0.120_208_247_024_70e5,
        -0.323_255_503_223_33e7,
        0.149_151_086_135_30e2,
        -0.482_326_573_615_91e4,
        0.405_113_405_420_57e6,
        -0.238_555_575_678_49,
        0.650_175_348_447_98e3,
    ];

    /// Saturation vapor pressure of pure water at `t` (K), Pa: where it
    /// boils at that temperature. IAPWS-IF97's saturation equation, the
    /// industrial steam-table standard (611 Pa at 0 °C, 2.34 kPa at 20 °C,
    /// 101.42 kPa at 100 °C, valid to the critical point).
    pub fn saturation_pressure(t: f32) -> f32 {
        let t = f64::from(t).clamp(273.15, 647.096);
        let theta = t + N[8] / (t - N[9]);
        let a = theta * theta + N[0] * theta + N[1];
        let b = N[2] * theta * theta + N[3] * theta + N[4];
        let c = N[5] * theta * theta + N[6] * theta + N[7];
        let p_mpa = (2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt())).powi(4);
        (p_mpa * 1e6) as f32
    }

    /// Boiling point of pure water at pressure `p` (Pa), K: the inverse of
    /// [`saturation_pressure`] (IAPWS-IF97's backward equation). Never
    /// below the triple point (0.01 °C).
    pub fn saturation_temperature(p: f32) -> f32 {
        let p_mpa = f64::from(p.max(TRIPLE_PRESSURE)).min(22.064e6) / 1e6;
        let beta = p_mpa.powf(0.25);
        let e = beta * beta + N[2] * beta + N[5];
        let f = N[0] * beta * beta + N[3] * beta + N[6];
        let g = N[1] * beta * beta + N[4] * beta + N[7];
        let d = 2.0 * g / (-f - (f * f - 4.0 * e * g).sqrt());
        let t = 0.5 * (N[9] + d - ((N[9] + d) * (N[9] + d) - 4.0 * (N[8] + N[9] * d)).sqrt());
        t.max(273.16) as f32
    }

    /// Water activity of brine at `salinity` (kg salt per kg): the factor
    /// salt lowers the vapor pressure by (0.98 for seawater). Two ions per
    /// NaCl, osmotic coefficient 0.9.
    pub fn activity(salinity: f32) -> f32 {
        let s = salinity.clamp(0.0, 0.5);
        let molality = s / (0.058_44 * (1.0 - s));
        (-0.9 * 2.0 * molality * MOLAR_MASS).exp()
    }

    /// Latent heat of vaporization at `t` (K), J/kg: 2.257 MJ at 100 °C,
    /// 2.45 MJ at 20 °C (Watson's correlation, zero at the critical point).
    pub fn latent_heat(t: f32) -> f32 {
        let t = t.clamp(273.15, 647.0);
        2.257e6 * ((647.1 - t) / (647.1 - 373.15)).max(0.0).powf(0.38)
    }

    /// Density of water vapor at partial pressure `p` (Pa) and `t` (K),
    /// kg/m³ (ideal gas).
    pub fn vapor_density(p: f32, t: f32) -> f32 {
        p * MOLAR_MASS / (GAS_CONSTANT * t.max(1.0))
    }

    /// Partial pressure of vapor at density `rho` (kg/m³) and `t` (K), Pa.
    pub fn vapor_pressure(rho: f32, t: f32) -> f32 {
        rho * GAS_CONSTANT * t.max(1.0) / MOLAR_MASS
    }

    /// Vapor density in air saturated at `t` (K), kg/m³: 17.3 g/m³ at
    /// 20 °C, 0.598 kg/m³ (pure steam) at 100 °C.
    pub fn saturation_vapor_density(t: f32) -> f32 {
        vapor_density(saturation_pressure(t), t)
    }

    /// Hertz–Knudsen: the most a surface at `t` (K) can evaporate into
    /// vacuum, kg/(m²·s), with evaporation coefficient 0.1 (measured values
    /// for water run from about 0.04 to 1).
    pub fn vacuum_evaporation_flux(t: f32, activity: f32) -> f32 {
        let p = saturation_pressure(t) * activity;
        let m = MOLAR_MASS;
        0.1 * p * (m / (2.0 * std::f32::consts::PI * GAS_CONSTANT * t.max(1.0))).sqrt()
    }
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

    #[test]
    fn water_saturation_is_real() {
        use water::*;
        let at = |c: f32| saturation_pressure(c + 273.15);
        assert!((at(0.01) / 611.66 - 1.0).abs() < 0.002, "{}", at(0.01));
        assert!((at(20.0) / 2339.2 - 1.0).abs() < 0.002, "{}", at(20.0));
        assert!((at(100.0) / 101_418.0 - 1.0).abs() < 0.001, "{}", at(100.0));
        assert!((at(200.0) / 1.5549e6 - 1.0).abs() < 0.001, "{}", at(200.0));
        // Boiling point vs pressure: Denver (83.4 kPa) boils at 94.4 °C.
        let boil = |p: f32| saturation_temperature(p) - 273.15;
        assert!((boil(101_325.0) - 99.97).abs() < 0.02);
        assert!((boil(83_400.0) - 94.4).abs() < 0.3, "{}", boil(83_400.0));
        // IF97 verification values: 0.1 MPa -> 372.756 K, 1 MPa -> 453.036 K.
        assert!(
            (boil(100_000.0) - 99.606).abs() < 0.005,
            "{}",
            boil(100_000.0)
        );
        assert!(
            (boil(1.0e6) + 273.15 - 453.036).abs() < 0.01,
            "{}",
            boil(1.0e6)
        );
        assert!((boil(0.0) - 0.01).abs() < 0.01);
        for c in [5.0, 30.0, 60.0, 99.0, 150.0] {
            let back = boil(at(c));
            assert!((back - c).abs() < 0.05, "{c} -> {back}");
        }
        assert!((latent_heat(373.15) - 2.257e6).abs() < 1.0);
        assert!((latent_heat(293.15) / 2.454e6 - 1.0).abs() < 0.02);
        assert!((saturation_vapor_density(293.15) * 1000.0 - 17.3).abs() < 0.3);
        assert!((activity(0.035) - 0.98).abs() < 0.003);
    }
}
