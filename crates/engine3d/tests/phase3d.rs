//! Water changing phase: boiling at the right temperature for the
//! pressure, flash evaporation in vacuum, evaporation into dry air, dew on
//! a cold plate, steam condensing to mist, and every kilogram and joule
//! accounted for.

use engine3d::fluid::LiquidSpec;
use engine3d::{Gas, RigidBody, Shape, Vec3, World, presets};
use materials::water;

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

/// A 0.6 m tank holding a 0.2 m layer of water at `celsius`.
fn pool(celsius: f32) -> World {
    let mut w = presets::tank(0.6, 0.6, 0.6);
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.3, 0.0, -0.3),
        Vec3::new(0.3, 0.2, 0.3),
        &b,
        LiquidSpec::water().at(celsius + 273.15),
    );
    w
}

/// Air over the tank at 20 °C with `humidity`; `still` stops the air
/// moving for the first second (the world steps air at least once a
/// second), so the water accounting can be checked exactly, without the
/// grid's advection.
fn air(w: &mut World, humidity: f32, still: bool) {
    let mut g = Gas::new(Vec3::new(-0.3, 0.0, -0.3), 0.05, 12, 16, 12);
    g.set_humidity(humidity);
    if still {
        g.params.step_hz = 1e-6;
    }
    w.gas = Some(g);
}

/// All the water, kg: liquid, vapor and mist in the air, dew, and vented.
fn all_water(w: &World) -> f64 {
    let gas = w
        .gas
        .as_ref()
        .map_or(0.0, |g| f64::from(g.total_vapor() + g.total_fog()));
    f64::from(w.fluid.water_mass()) + gas + f64::from(w.water.total_dew() + w.water.vented)
}

#[test]
fn water_boils_at_its_boiling_point_and_no_hotter() {
    let mut w = pool(99.0);
    air(&mut w, 0.0, false);
    let mut plate = RigidBody::new_static(Shape::cuboid(0.4, 0.02, 0.4), Vec3::new(0.0, 0.01, 0.0));
    plate.temperature = Some(150.0 + 273.15);
    w.add_body(plate);
    let b = w.bodies.clone();
    w.fluid.project_out_of_bodies(&b);
    run(&mut w, 2.0);
    let hottest = w.fluid.temperatures.iter().copied().fold(0.0, f64::max) - 273.15;
    // 100 °C at the surface; a little more under the water's own weight
    // (0.2 m of water adds 2 kPa: +0.55 K).
    assert!(hottest > 99.9 && hottest < 100.8, "hottest {hottest} °C");
    assert!(w.water.latent > 0.0, "nothing boiled");
    let vapor = w.gas.as_ref().unwrap();
    assert!(vapor.total_vapor() + vapor.total_fog() > 0.0);
}

#[test]
fn salt_water_boils_hotter() {
    let fresh = water::saturation_temperature(water::ATMOSPHERE);
    let brine = water::saturation_temperature(water::ATMOSPHERE / water::activity(0.035));
    let rise = brine - fresh;
    assert!(rise > 0.45 && rise < 0.65, "seawater boils {rise} K higher");
}

#[test]
fn water_in_vacuum_flash_boils_and_cools() {
    let mut w = pool(20.0);
    w.ambient_pressure = 0.0;
    let m0 = all_water(&w);
    run(&mut w, 1.0);
    // Surface water boils off until it's near freezing; deeper water is
    // held liquid by the pressure of the water above it.
    let coldest = w.fluid.temperatures.iter().copied().fold(1e9, f64::min) - 273.15;
    assert!(coldest < 5.0, "coldest {coldest} °C");
    assert!(w.water.vented > 0.0);
    let m1 = all_water(&w);
    assert!(((m1 - m0) / m0).abs() < 1e-5, "water {m0} -> {m1} kg");
    // The heat that left went into vaporizing what vented.
    let l = f64::from(water::latent_heat(283.15));
    let expected = f64::from(w.water.vented) * l;
    assert!(
        (w.water.latent / expected - 1.0).abs() < 0.15,
        "latent {} J vs {expected} J",
        w.water.latent
    );
}

#[test]
fn warm_water_evaporates_into_dry_air_and_every_kilogram_is_kept() {
    let mut w = pool(60.0);
    air(&mut w, 0.0, true);
    let m0 = all_water(&w);
    let e0 = w.fluid.thermal_energy();
    run(&mut w, 0.9);
    let g = w.gas.as_ref().unwrap();
    let vapor = g.total_vapor();
    assert!(vapor > 0.0, "nothing evaporated");
    let m1 = all_water(&w);
    assert!(((m1 - m0) / m0).abs() < 1e-5, "water {m0} -> {m1} kg");
    // Energy: what the liquid lost went into latent heat and warming air.
    let air_heat = f64::from(
        g.heat.iter().sum::<f32>() * g.cell.powi(3) * water::AIR_DENSITY * water::AIR_SPECIFIC_HEAT,
    );
    let lost = e0 - w.fluid.thermal_energy();
    let accounted = w.water.latent + air_heat;
    assert!(
        (lost / accounted - 1.0).abs() < 0.02,
        "lost {lost} J, accounted {accounted} J"
    );
}

#[test]
fn humid_air_leaves_dew_on_a_cold_plate() {
    let mut w = presets::tank(0.6, 0.6, 0.6);
    air(&mut w, 0.9, true);
    let mut plate = RigidBody::new_static(Shape::cuboid(0.3, 0.02, 0.3), Vec3::new(0.0, 0.3, 0.0));
    plate.temperature = Some(5.0 + 273.15);
    w.add_body(plate);
    let m0 = all_water(&w);
    run(&mut w, 0.9);
    assert!(w.water.total_dew() > 0.0, "no dew");
    let m1 = all_water(&w);
    assert!(((m1 - m0) / m0).abs() < 1e-5, "water {m0} -> {m1} kg");
}

#[test]
fn steam_in_cool_air_condenses_to_mist_and_warms_it() {
    let mut g = Gas::new(Vec3::new(-0.5, 0.0, -0.5), 0.05, 20, 20, 20);
    g.set_humidity(0.5);
    let added = g.add_steam(Vec3::new(0.0, 0.3, 0.0), 0.12, 373.15);
    let total0 = g.total_vapor() + g.total_fog();
    assert!(added > 0.0);
    let mut w = World::new();
    w.gas = Some(g);
    run(&mut w, 1.0);
    let g = w.gas.as_ref().unwrap();
    assert!(g.total_fog() > 0.0, "no mist");
    // Semi-Lagrangian advection isn't exactly conservative (a known limit
    // of the grid); the water stays within 20%.
    let total1 = g.total_vapor() + g.total_fog();
    assert!(
        (total1 / total0 - 1.0).abs() < 0.2,
        "{total0} -> {total1} kg"
    );
    // Nowhere is supersaturated after the adjustment.
    for c in 0..g.vapor.len() {
        let rho_s = water::saturation_vapor_density(g.temperature(c));
        assert!(g.vapor[c] <= rho_s * 1.001 + 1e-6);
    }
}
