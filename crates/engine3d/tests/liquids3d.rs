//! Several liquids in 3D: they sort by density, conserve momentum, and
//! salt diffuses between waters.

use engine3d::fluid::{Liquid, SEAWATER_SALINITY};
use engine3d::{Scene, Vec3, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

fn mean_height(w: &World, liquid: Liquid) -> f32 {
    let (sum, n) = (0..w.fluid.len())
        .filter(|&i| w.fluid.liquid(i) == liquid)
        .fold((0.0, 0), |(s, n), i| (s + w.fluid.positions[i].y, n + 1));
    assert!(n > 0, "no {liquid:?} left");
    sum / n as f32
}

#[test]
fn mercury_sinks_and_oil_floats() {
    // Inverted start: oil at the bottom, water, then mercury on top.
    let mut w = presets::tank(1.0, 1.0, 1.5);
    let b = w.bodies.clone();
    w.fluid.fill_box_with(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.3, 0.5),
        &b,
        Liquid::Oil,
        0.0,
    );
    w.fluid
        .fill_box(Vec3::new(-0.5, 0.3, -0.5), Vec3::new(0.5, 0.6, 0.5), &b);
    w.fluid.fill_box_with(
        Vec3::new(-0.5, 0.6, -0.5),
        Vec3::new(0.5, 0.7, 0.5),
        &b,
        Liquid::Mercury,
        0.0,
    );
    run(&mut w, 5.0);
    let (hg, water, oil) = (
        mean_height(&w, Liquid::Mercury),
        mean_height(&w, Liquid::Water),
        mean_height(&w, Liquid::Oil),
    );
    assert!(
        hg < water && water < oil,
        "mercury {hg}, water {water}, oil {oil}"
    );
    assert!(hg < 0.15, "mercury at {hg} m");
}

#[test]
fn liquids_conserve_momentum() {
    let mut w = World::new();
    w.bodies.clear();
    w.gravity = Vec3::ZERO;
    w.fluid
        .fill_sphere_with(Vec3::new(-0.5, 0.0, 0.0), 0.35, &[], Liquid::Oil, 0.0);
    let n_oil = w.fluid.len();
    w.fluid
        .fill_sphere_with(Vec3::new(0.5, 0.0, 0.0), 0.35, &[], Liquid::Mercury, 0.0);
    for i in 0..w.fluid.len() {
        w.fluid.velocities[i] = if i < n_oil { Vec3::X } else { -0.2 * Vec3::X };
    }
    let momentum = |w: &World| -> Vec3 {
        (0..w.fluid.len())
            .map(|i| w.fluid.mass(i) * w.fluid.velocities[i])
            .sum()
    };
    let p0 = momentum(&w);
    run(&mut w, 1.0);
    let p1 = momentum(&w);
    assert!((p1 - p0).length() < 1e-4 * p0.length(), "{p0} → {p1}");
}

#[test]
fn salt_diffuses_and_is_conserved() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    w.gravity = Vec3::ZERO;
    w.fluid.params.salt_diffusivity = 5e-3;
    let b = w.bodies.clone();
    w.fluid
        .fill_box(Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.0, 0.5, 0.5), &b);
    w.fluid.fill_box_with(
        Vec3::new(0.0, 0.0, -0.5),
        Vec3::new(0.5, 0.5, 0.5),
        &b,
        Liquid::Water,
        SEAWATER_SALINITY,
    );
    let total = |w: &World| w.fluid.salinity.iter().sum::<f32>();
    let s0 = total(&w);
    run(&mut w, 1.0);
    assert!((total(&w) / s0 - 1.0).abs() < 1e-3);
    let picked = (0..w.fluid.len())
        .filter(|&i| (-0.2..-0.05).contains(&w.fluid.positions[i].x))
        .map(|i| w.fluid.salinity[i])
        .fold(0.0, f32::max);
    assert!(
        picked > 0.1 * SEAWATER_SALINITY,
        "max salinity near interface {picked}"
    );
}

#[test]
fn multiple_liquids_round_trip_through_scene_files() {
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_box_with(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.2, 0.5),
        &b,
        Liquid::Oil,
        0.0,
    );
    w.fluid.fill_box_with(
        Vec3::new(-0.5, 0.2, -0.5),
        Vec3::new(0.5, 0.4, 0.5),
        &b,
        Liquid::Water,
        SEAWATER_SALINITY,
    );
    let json = Scene::from_world(&w).to_json();
    let back = Scene::from_json(&json).unwrap().world;
    assert_eq!(back.fluid, w.fluid);
}
