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
