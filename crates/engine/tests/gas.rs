//! Smoke and heat (grid gas) validation against analytic results.

use engine::{Emitter, Gas, RigidBody, Shape, Vec2, World};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine::STANDARD_GRAVITY;

#[test]
fn heat_spreads_like_the_gaussian_solution() {
    // In still air a Gaussian hot spot stays Gaussian. In 2D its variance
    // grows as σ² = σ₀² + 2κt and, since total heat is fixed, its peak
    // falls as ΔT₀ σ₀²/σ². With σ₀ = 0.2 m, κ = 0.01 m²/s, t = 1 s:
    // σ² = 0.04 + 0.02 = 0.06, peak = 10 K × 0.04/0.06 = 6.667 K.
    let (n, dx) = (64, 0.05);
    let mut gas = Gas::new(Vec2::ZERO, dx, n, n);
    gas.params.thermal_diffusivity = 0.01;
    gas.params.open_top = false;
    let center = gas.cell_center(n / 2, n / 2);
    let s0 = 0.2_f32;
    for j in 0..n {
        for i in 0..n {
            let r2 = (gas.cell_center(i, j) - center).length_squared();
            gas.heat[j * n + i] = 10.0 * (-r2 / (2.0 * s0 * s0)).exp();
        }
    }
    let heat_before = gas.total_heat();
    for _ in 0..120 {
        gas.step(&[], Vec2::ZERO, DT);
    }
    let peak = gas.heat[(n / 2) * n + n / 2];
    let expected = 10.0 * s0 * s0 / (s0 * s0 + 2.0 * 0.01 * 1.0);
    assert!(
        (peak - expected).abs() < 0.02 * expected,
        "peak {peak:.4} K, expected {expected:.4} K"
    );
    // Insulated walls: heat is conserved.
    let heat_after = gas.total_heat();
    assert!(
        (heat_after - heat_before).abs() < 1e-4 * heat_before,
        "heat {heat_before} -> {heat_after}"
    );
}

#[test]
fn warm_cylinder_starts_rising_at_half_g_beta_delta_t() {
    // A circular warm patch in still air feels buoyancy g' = g β ΔT, but it
    // has to push the surrounding air aside: in 2D that added mass equals its
    // own, so it starts out accelerating at exactly g'/2 (the pressure field
    // inside a uniformly forced circle is the 2D analogue of a uniformly
    // polarized cylinder). ΔT = 10 K: g' = 9.80665 × 10 / 293.15 = 0.3345.
    let (n, dx) = (96, 0.05);
    let mut gas = Gas::new(Vec2::ZERO, dx, n, n);
    gas.params.open_top = false;
    gas.params.thermal_diffusivity = 0.0;
    let center = gas.cell_center(n / 2, n / 2);
    let radius = 0.4;
    for j in 0..n {
        for i in 0..n {
            if (gas.cell_center(i, j) - center).length() < radius {
                gas.heat[j * n + i] = 10.0;
            }
        }
    }
    gas.step(&[], Vec2::new(0.0, -G), DT);
    // Mean vertical velocity well inside the patch.
    let (mut sum, mut count) = (0.0, 0);
    for j in 0..n {
        for i in 0..n {
            if (gas.cell_center(i, j) - center).length() < 0.25 {
                sum += gas.velocity_at_cell(i, j).y;
                count += 1;
            }
        }
    }
    let g_prime = G * 10.0 / gas.params.ambient_temperature;
    let ratio = (sum / count as f32) / (g_prime * DT);
    assert!(
        (ratio - 0.5).abs() < 0.025,
        "initial acceleration {ratio:.4} × g′"
    );
}

#[test]
fn uniformly_warm_sealed_box_stays_still() {
    // Uniform buoyancy in a closed box is balanced by pressure (hydrostatic
    // equilibrium): nothing should move.
    let mut gas = Gas::new(Vec2::ZERO, 0.1, 40, 30);
    gas.params.open_top = false;
    gas.heat.iter_mut().for_each(|h| *h = 20.0);
    for _ in 0..120 {
        gas.step(&[], Vec2::new(0.0, -G), DT);
    }
    let vmax = gas
        .u
        .iter()
        .chain(&gas.v)
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(vmax < 1e-4, "air moved at {vmax} m/s");
}

#[test]
fn hot_plume_rises_around_an_obstacle() {
    let mut w = World::new();
    let mut gas = Gas::new(Vec2::new(-2.0, 0.0), 0.05, 80, 120);
    gas.emitters.push(Emitter {
        center: Vec2::new(0.0, 0.3),
        radius: 0.2,
        smoke_rate: 2.0,
        temperature: 293.15 + 60.0,
        velocity: Vec2::ZERO,
    });
    w.gas = Some(gas);
    // A plate above the source the plume must flow around.
    w.add_body(RigidBody::new_static(
        Shape::rect(0.8, 0.1),
        Vec2::new(0.0, 2.0),
    ));
    for _ in 0..(4.0 / DT) as usize {
        w.step(DT);
    }
    let gas = w.gas.as_ref().unwrap();
    // Smoke has risen well above the source (its centroid > 1 m up)...
    let (mut sy, mut s) = (0.0, 0.0);
    for j in 0..gas.ny {
        for i in 0..gas.nx {
            let c = gas.smoke[j * gas.nx + i];
            sy += c * gas.cell_center(i, j).y;
            s += c;
        }
    }
    assert!(sy / s > 1.0, "smoke centroid at {} m", sy / s);
    // ...some of it past the plate, none inside it.
    let (i, j) = gas.cell_at(Vec2::new(0.0, 2.0)).unwrap();
    assert!(gas.is_solid(i, j) && gas.smoke[j * gas.nx + i] == 0.0);
    let above: f32 = (gas.cell_at(Vec2::new(0.0, 2.6)).unwrap().1..gas.ny)
        .flat_map(|j| (0..gas.nx).map(move |i| (i, j)))
        .map(|(i, j)| gas.smoke[j * gas.nx + i])
        .sum();
    assert!(above > 0.0, "no smoke got past the plate");
    // The flow stays divergence-free.
    assert!(
        gas.stats.max_divergence < 1e-2,
        "max ∇·u {}",
        gas.stats.max_divergence
    );
}

#[test]
fn gas_round_trips_through_scene_files() {
    let mut w = engine::presets::smoke_plume();
    w.step(DT);
    let loaded = engine::Scene::from_json(&engine::Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
}
