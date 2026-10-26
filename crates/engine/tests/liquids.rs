//! Several liquids at once: water, salt water, oil, and mercury sort by
//! density, carry the right hydrostatic pressure, conserve momentum, and
//! salt diffuses between waters.

use engine::fluid::{Liquid, SEAWATER_SALINITY};
use engine::{CpuSolver, FluidSolver, Scene, Vec2, World, presets};

const DT: f32 = 1.0 / 120.0;

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT);
    }
}

/// Mean height of the particles of one liquid.
fn mean_height(w: &World, liquid: Liquid) -> f32 {
    let (sum, n) = (0..w.fluid.len())
        .filter(|&i| w.fluid.liquid(i) == liquid)
        .fold((0.0, 0), |(s, n), i| (s + w.fluid.positions[i].y, n + 1));
    assert!(n > 0, "no {liquid:?} left");
    sum / n as f32
}

#[test]
fn mercury_sinks_to_the_bottom() {
    // A slab of mercury starts on top of a water column.
    let mut w = presets::tank(1.0, 2.0);
    let b = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.6), &b);
    w.fluid.fill_rect_with(
        Vec2::new(-0.5, 0.6),
        Vec2::new(0.5, 0.75),
        &b,
        Liquid::Mercury,
        0.0,
    );
    run(&mut w, 4.0);
    let (hg, water) = (
        mean_height(&w, Liquid::Mercury),
        mean_height(&w, Liquid::Water),
    );
    // Settled: mercury fills the bottom ~0.15 m (mean ~0.08), water above.
    assert!(hg < 0.15, "mercury at {hg} m");
    assert!(water > 0.3, "water at {water} m");
}

#[test]
fn a_blob_of_oil_rises_through_water() {
    let mut w = presets::tank(1.0, 2.0);
    let b = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 1.2), &b);
    // Replace a disc of water near the bottom with oil. (It needs to be
    // several particles across: a blob only one or two particles wide is
    // held in place by the particle lattice.)
    let center = Vec2::new(0.0, 0.35);
    for i in 0..w.fluid.len() {
        if (w.fluid.positions[i] - center).length() < 0.3 {
            w.fluid.liquids[i] = Liquid::Oil;
        }
    }
    let start = mean_height(&w, Liquid::Oil);
    run(&mut w, 3.0);
    let end = mean_height(&w, Liquid::Oil);
    assert!(end > start + 0.1, "oil rose from {start} to {end} m");
}

#[test]
fn layered_hydrostatic_pressure() {
    // 0.3 m of mercury under 0.5 m of water: pressure grows at ρ_w g in the
    // water and 13.5× faster in the mercury. Bands stay clear of the floor
    // and the interface, where one-sided neighborhoods skew SPH pressure.
    let mut w = presets::tank(1.0, 2.0);
    let b = w.bodies.clone();
    w.fluid.fill_rect_with(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.3),
        &b,
        Liquid::Mercury,
        0.0,
    );
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.3), Vec2::new(0.5, 0.8), &b);
    run(&mut w, 3.0);
    let g = w.gravity.length();
    let rho_w = w.fluid.params.rest_density;
    let band = |w: &World, lo: f32, hi: f32| {
        let (s, n) = (0..w.fluid.len())
            .filter(|&i| {
                let p = w.fluid.positions[i];
                p.x.abs() < 0.3 && p.y > lo && p.y < hi
            })
            .fold((0.0, 0), |(s, n), i| (s + w.fluid.pressures()[i], n + 1));
        s / n as f32
    };
    let mut gradient_w = 0.0;
    let mut gradient_hg = 0.0;
    let samples = 240;
    for _ in 0..samples {
        w.step(DT);
        gradient_w += (band(&w, 0.45, 0.5) - band(&w, 0.65, 0.7)) / 0.2 / samples as f32;
        gradient_hg += (band(&w, 0.12, 0.17) - band(&w, 0.22, 0.27)) / 0.1 / samples as f32;
    }
    let ratio_w = gradient_w / (rho_w * g);
    let ratio_hg = gradient_hg / (13.546 * rho_w * g);
    assert!(
        (ratio_w - 1.0).abs() < 0.15,
        "water dp/dy = {ratio_w:.3} ρg"
    );
    assert!(
        (ratio_hg - 1.0).abs() < 0.2,
        "mercury dp/dy = {ratio_hg:.3} ρg"
    );
}

#[test]
fn liquids_conserve_momentum() {
    // Oil and mercury blobs thrown at each other in empty space: every pair
    // force is equal and opposite, so total momentum can't change.
    let mut w = World::new();
    w.bodies.clear();
    w.gravity = Vec2::ZERO;
    w.fluid
        .fill_disc_with(Vec2::new(-0.4, 0.0), 0.25, &[], Liquid::Oil, 0.0);
    let n_oil = w.fluid.len();
    w.fluid
        .fill_disc_with(Vec2::new(0.4, 0.0), 0.25, &[], Liquid::Mercury, 0.0);
    for i in 0..w.fluid.len() {
        w.fluid.velocities[i] = if i < n_oil { Vec2::X } else { -0.3 * Vec2::X };
    }
    let momentum = |w: &World| -> Vec2 {
        (0..w.fluid.len())
            .map(|i| w.fluid.mass(i) * w.fluid.velocities[i])
            .sum()
    };
    let p0 = momentum(&w);
    run(&mut w, 1.0);
    let p1 = momentum(&w);
    assert!(
        (p1 - p0).length() < 1e-4 * p0.length().max(1e-3),
        "{p0} → {p1}"
    );
}

#[test]
fn salt_diffuses_and_is_conserved() {
    // Fresh and salty water side by side, no gravity, faster diffusion than
    // real so it shows in a second. Salt spreads across the interface and
    // the total amount stays the same.
    let mut w = presets::tank(1.0, 1.0);
    w.gravity = Vec2::ZERO;
    w.fluid.params.salt_diffusivity = 2e-3;
    let b = w.bodies.clone();
    w.fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.0, 0.5), &b);
    w.fluid.fill_rect_with(
        Vec2::new(0.0, 0.0),
        Vec2::new(0.5, 0.5),
        &b,
        Liquid::Water,
        SEAWATER_SALINITY,
    );
    let total = |w: &World| w.fluid.salinity.iter().sum::<f32>();
    let s0 = total(&w);
    run(&mut w, 1.0);
    assert!(
        (total(&w) / s0 - 1.0).abs() < 1e-3,
        "salt {s0} → {}",
        total(&w)
    );
    // Fresh side near the interface picked up salt.
    let picked = (0..w.fluid.len())
        .filter(|&i| {
            let x = w.fluid.positions[i].x;
            (-0.1..-0.02).contains(&x)
        })
        .map(|i| w.fluid.salinity[i])
        .fold(0.0, f32::max);
    assert!(
        picked > 0.1 * SEAWATER_SALINITY,
        "max salinity near interface {picked}"
    );
}

#[test]
fn multiple_liquids_round_trip_through_scene_files() {
    let mut w = presets::tank(1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_rect_with(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.0, 0.2),
        &b,
        Liquid::Oil,
        0.0,
    );
    w.fluid.fill_rect_with(
        Vec2::new(0.0, 0.0),
        Vec2::new(0.5, 0.2),
        &b,
        Liquid::Water,
        SEAWATER_SALINITY,
    );
    let json = Scene::from_world(&w).to_json();
    let back = Scene::from_json(&json).unwrap().world;
    assert_eq!(back.fluid, w.fluid);
    // Fresh water alone writes neither field, so old files are unchanged.
    let mut plain = presets::tank(1.0, 1.0);
    let b = plain.bodies.clone();
    plain
        .fluid
        .fill_rect(Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.2), &b);
    let json = Scene::from_world(&plain).to_json();
    assert!(!json.contains("liquids") && !json.contains("salinity"));
}

#[test]
fn cpu_solver_steps_mixed_liquids_through_the_trait() {
    let mut w = presets::tank(1.0, 1.0);
    let b = w.bodies.clone();
    w.fluid.fill_rect_with(
        Vec2::new(-0.5, 0.0),
        Vec2::new(0.5, 0.2),
        &b,
        Liquid::Oil,
        0.0,
    );
    let boundary = w.fluid.params.boundary(&w.bodies);
    let n = w.bodies.len();
    let impulses = CpuSolver.step(&mut w.fluid, &boundary, w.gravity, DT, n);
    assert_eq!(impulses.len(), n);
}
