//! Heat in the liquids: conduction conserves energy and matches the heat
//! equation, warm liquid expands and rises, and walls can hold a
//! temperature.

use engine::fluid::{Liquid, LiquidSpec, ROOM_TEMPERATURE};
use engine::{Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

#[test]
fn conduction_conserves_thermal_energy() {
    // Hot and cold water side by side in an insulated tank, no gravity.
    let mut w = presets::tank(1.0, 1.0);
    w.gravity = Vec2::ZERO;
    let b = w.bodies.clone();
    w.fluid.fill_rect_as(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.0, 0.5),
        &b,
        LiquidSpec::water().at(353.15),
    );
    w.fluid.fill_rect_as(
        Vec2::new(0.0, 0.0),
        Vec2::new(0.5, 0.5),
        &b,
        LiquidSpec::water().at(283.15),
    );
    let e0 = w.fluid.thermal_energy();
    run(&mut w, 1.0);
    let e1 = w.fluid.thermal_energy();
    assert!(((e1 - e0) / e0).abs() < 1e-6, "{e0} → {e1}");
    // And heat did flow across the middle (slowly: water's thermal
    // diffusivity is 1.4 × 10⁻⁷ m²/s).
    let warmed = w
        .fluid
        .temperatures
        .iter()
        .filter(|&&t| t > 283.15 && t < 300.0)
        .count();
    assert!(warmed > 0, "no cold particle warmed");
}

#[test]
fn a_sine_wave_of_temperature_decays_at_the_heat_equation_rate() {
    // Mercury (thermal diffusivity α = k/ρc = 4.4 × 10⁻⁶ m²/s) at rest in a
    // closed box, with T = T₀ + A cos(2πx/L). The heat equation says the
    // amplitude decays as e^(−α (2π/L)² t).
    let mut w = presets::tank(0.1, 0.12);
    w.gravity = Vec2::ZERO;
    w.fluid.params.particle_spacing = 0.0025;
    // Nothing flows, so a soft equation of state is fine (fewer substeps).
    w.fluid.params.speed_of_sound = 5.0;
    let b = w.bodies.clone();
    w.fluid.fill_rect_as(
        Vec2::new(-0.05, 0.0),
        Vec2::new(0.05, 0.1),
        &b,
        LiquidSpec::new(Liquid::Mercury),
    );
    // Let the particles settle against the walls first: the startup
    // shuffle would otherwise stir the temperature pattern.
    run(&mut w, 1.0);
    let l = 0.1_f32;
    let k = std::f32::consts::TAU / l;
    for i in 0..w.fluid.len() {
        let x = w.fluid.positions[i].x;
        w.fluid.temperatures[i] = f64::from(ROOM_TEMPERATURE + 10.0 * (k * x).cos());
    }
    let amplitude = |w: &World| -> f32 {
        // Projection onto cos(kx), away from the side walls' influence on
        // the sum's normalization: 2/N Σ (T − T̄) cos kx.
        let n = w.fluid.len() as f32;
        let mean = (w.fluid.temperatures.iter().sum::<f64>() / f64::from(n)) as f32;
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..w.fluid.len() {
            let c = (k * w.fluid.positions[i].x).cos();
            num += (w.fluid.temperatures[i] as f32 - mean) * c;
            den += c * c;
        }
        num / den
    };
    let a0 = amplitude(&w);
    let t = 2.0;
    run(&mut w, t);
    let a1 = amplitude(&w);
    let alpha = 8.3 / (13_546.0 * 139.5);
    let expected = (-alpha * k * k * t).exp();
    let measured = a1 / a0;
    // 1 − ratio is the decay; compare decays.
    let (de, dm) = (1.0 - expected, 1.0 - measured);
    assert!(
        (dm / de - 1.0).abs() < 0.15,
        "decay {dm:.5}, expected {de:.5}"
    );
}

#[test]
fn warm_water_rises() {
    // A disc of 80 °C water (2.6% lighter) at the bottom of 20 °C water.
    let mut w = presets::tank(1.0, 2.0);
    let b = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 1.2), &b);
    let center = Vec2::new(0.0, 0.35);
    for i in 0..w.fluid.len() {
        if (w.fluid.positions[i] - center).length() < 0.3 {
            w.fluid.temperatures[i] = 353.15;
        }
    }
    let warm_height = |w: &World| {
        let (s, n) = (0..w.fluid.len())
            .filter(|&i| w.fluid.temperatures[i] > 330.0)
            .fold((0.0, 0), |(s, n), i| (s + w.fluid.positions[i].y, n + 1));
        s / n as f32
    };
    let start = warm_height(&w);
    run(&mut w, 3.0);
    let end = warm_height(&w);
    assert!(end > start + 0.05, "warm water from {start} to {end} m");
}

#[test]
fn hot_walls_heat_the_water() {
    let mut w = presets::tank(1.0, 1.0);
    w.fluid.params.wall_temperature = Some(373.15);
    let b = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.5), &b);
    run(&mut w, 1.0);
    let hottest = w.fluid.temperatures.iter().copied().fold(0.0, f64::max);
    let coldest = w
        .fluid
        .temperatures
        .iter()
        .copied()
        .fold(f64::MAX, f64::min);
    let room = f64::from(ROOM_TEMPERATURE);
    assert!(hottest > room + 1e-4, "hottest {hottest}");
    assert!(hottest <= 373.15 && coldest >= room - 1e-9);
}

#[test]
fn temperatures_round_trip_through_scene_files() {
    let mut w = presets::tank(1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_rect_as(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.2),
        &b,
        LiquidSpec::new(Liquid::Oil).at(400.0),
    );
    let json = engine::Scene::from_world(&w).to_json();
    let back = engine::Scene::from_json(&json).unwrap().world;
    assert_eq!(back.fluid, w.fluid);
    assert!(back.fluid.temperatures.iter().all(|&t| t == 400.0));
}

#[test]
fn hot_water_warms_the_air_above_it() {
    // 80 °C water under a smoke/heat grid: its surface gives heat to the
    // air, and what the water loses the air gains.
    let mut w = presets::tank(1.0, 1.5);
    let b = w.bodies.clone();
    w.fluid.fill_rect_as(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.4),
        &b,
        LiquidSpec::water().at(353.15),
    );
    let mut gas = engine::Gas::new(Vec2::new(-0.6, -0.1), 0.05, 24, 32);
    gas.params.open_top = false;
    w.gas = Some(gas);
    let thickness = w.fluid.params.rest_density / 1000.0;
    let air_capacity = 1.204 * 1005.0 * 0.05 * 0.05 * thickness;
    let air_heat = |w: &World| -> f64 {
        w.gas
            .as_ref()
            .unwrap()
            .heat
            .iter()
            .map(|&h| f64::from(h * air_capacity))
            .sum()
    };
    let e0 = w.fluid.thermal_energy() + air_heat(&w);
    run(&mut w, 2.0);
    let gained = air_heat(&w);
    assert!(gained > 0.0, "air gained {gained} J");
    let e1 = w.fluid.thermal_energy() + air_heat(&w);
    assert!(
        ((e1 - e0) / gained).abs() < 0.05,
        "energy {e0} → {e1}, air gained {gained}"
    );
}
