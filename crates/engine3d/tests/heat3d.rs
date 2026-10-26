//! Heat in 3D liquids: conduction conserves energy, warm water rises, and
//! temperatures survive scene files.

use engine3d::fluid::{Liquid, LiquidSpec};
use engine3d::{Vec3, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

#[test]
fn conduction_conserves_thermal_energy() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    w.gravity = Vec3::ZERO;
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.0, 0.5, 0.5),
        &b,
        LiquidSpec::new(Liquid::Mercury).at(353.15),
    );
    w.fluid.fill_box_as(
        Vec3::new(0.0, 0.0, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
        &b,
        LiquidSpec::water().at(283.15),
    );
    let e0 = w.fluid.thermal_energy();
    run(&mut w, 1.0);
    let e1 = w.fluid.thermal_energy();
    assert!(((e1 - e0) / e0).abs() < 1e-6, "{e0} → {e1}");
    let warmed = w
        .fluid
        .temperatures
        .iter()
        .filter(|&&t| t > 283.15 && t < 300.0)
        .count();
    assert!(warmed > 0);
}

#[test]
fn warm_water_rises() {
    let mut w = presets::tank(1.0, 1.0, 1.5);
    let b = w.bodies.clone();
    w.fluid
        .fill_box(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 1.0, 0.5), &b);
    for i in 0..w.fluid.len() {
        if w.fluid.positions[i].y < 0.3 {
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
    run(&mut w, 4.0);
    let end = warm_height(&w);
    assert!(end > start + 0.3, "warm water from {start} to {end} m");
}

#[test]
fn temperatures_round_trip_through_scene_files() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.2, 0.5),
        &b,
        LiquidSpec::new(Liquid::Oil).at(400.0),
    );
    let json = engine3d::Scene::from_world(&w).to_json();
    let back = engine3d::Scene::from_json(&json).unwrap().world;
    assert_eq!(back.fluid, w.fluid);
}

#[test]
fn a_heater_plate_warms_the_water_toward_its_temperature() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    // A plate on the floor held at 80 °C.
    let mut plate = engine3d::RigidBody::new_static(
        engine3d::Shape::cuboid(0.6, 0.04, 0.6),
        Vec3::new(0.0, 0.02, 0.0),
    );
    plate.temperature = Some(353.15);
    w.add_body(plate);
    let b = w.bodies.clone();
    w.fluid
        .fill_box(Vec3::new(-0.5, 0.04, -0.5), Vec3::new(0.5, 0.4, 0.5), &b);
    let mean = |w: &World| w.fluid.temperatures.iter().sum::<f64>() / w.fluid.len() as f64;
    let e0 = w.fluid.thermal_energy();
    let t0 = mean(&w);
    run(&mut w, 0.25);
    // Natural convection off a hot plate: q = h A ΔT, h from the
    // turbulent correlation (about 1.2 kW/m²·K here).
    let h = Liquid::Water.natural_convection(0.0, 293.15, 353.15);
    let expected = f64::from(h * 0.36 * 60.0) * 0.25;
    let got = w.fluid.thermal_energy() - e0;
    eprintln!("h {h} W/m²K: plate gave {got:.0} J, h·A·ΔT·t = {expected:.0} J");
    assert!(
        (got / expected - 1.0).abs() < 0.2,
        "{got} J vs {expected} J"
    );
    run(&mut w, 1.75);
    let t1 = mean(&w);
    assert!(t1 > t0 + 0.01, "mean {t0} → {t1} K");
    let hottest = w.fluid.temperatures.iter().copied().fold(0.0, f64::max);
    assert!(hottest <= 353.15 + 1e-6, "hottest {hottest} K");
}

#[test]
fn the_heat_brush_adds_exactly_its_energy() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.3, 0.5),
        &b,
        LiquidSpec::new(Liquid::Oil),
    );
    let e0 = w.fluid.thermal_energy();
    let added = w
        .fluid
        .add_heat(Vec3::new(0.0, 0.15, 0.0), 0.2, 1000.0, 0.5);
    let e1 = w.fluid.thermal_energy();
    assert!((added - 500.0).abs() < 1e-3, "added {added} J");
    assert!(((e1 - e0) - 500.0).abs() < 0.5, "energy rose {} J", e1 - e0);
    // Cooling works too, and nothing passes the limits.
    w.fluid.add_heat(Vec3::ZERO, 10.0, -1e12, 1.0);
    let coldest = w.fluid.temperatures.iter().copied().fold(1e9, f64::min);
    assert!(coldest > 263.0 && coldest < 263.3, "coldest {coldest} K");
}

#[test]
fn the_probe_reads_the_local_liquid() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_box_as(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.4, 0.5),
        &b,
        LiquidSpec::seawater().at(300.0),
    );
    run(&mut w, 0.5);
    let p = w.fluid.probe(Vec3::new(0.0, 0.1, 0.0)).unwrap();
    assert_eq!(p.liquid, Liquid::Water);
    assert!((p.temperature - 300.0).abs() < 0.01);
    // Hydrostatic: about ρ g × 0.3 m of depth (WCSPH noise allowed).
    let expected = 1025.0 * 9.81 * 0.3;
    assert!(
        (p.pressure / expected - 1.0).abs() < 0.3,
        "{} Pa",
        p.pressure
    );
    assert!(w.fluid.probe(Vec3::new(0.0, 5.0, 0.0)).is_none());
}

#[test]
fn liquid_limits_are_real() {
    // Seawater boils about 0.6 K above fresh water and freezes near −1.9 °C.
    let fresh = Liquid::Water.max_temperature(0.0);
    let sea = Liquid::Water.max_temperature(0.035);
    assert!((fresh - 373.15).abs() < 1e-3);
    assert!(sea - fresh > 0.4 && sea - fresh < 0.8, "{}", sea - fresh);
    let freeze = Liquid::Water.min_temperature(0.035) - 273.15;
    assert!((freeze + 1.9).abs() < 0.2, "{freeze} °C");
    assert!((Liquid::Mercury.max_temperature(0.0) - 629.88).abs() < 1e-3);
}
