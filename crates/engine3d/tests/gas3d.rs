//! 3D smoke and heat validation against analytic results.

use engine3d::{Emitter, Gas, RigidBody, Shape, Vec3, World};

const DT: f32 = 1.0 / 120.0;
const G: f32 = engine3d::STANDARD_GRAVITY;

#[test]
fn heat_spreads_like_the_gaussian_solution() {
    // 3D: σ² = σ₀² + 2κt and the peak falls as (σ₀²/σ²)^{3/2}.
    // σ₀ = 0.2 m, κ = 0.01 m²/s, t = 0.5 s: σ² = 0.04 + 0.01 = 0.05,
    // peak = 10 K × 0.8^1.5 = 7.155 K.
    let (n, dx) = (40, 0.05);
    let mut gas = Gas::new(Vec3::ZERO, dx, n, n, n);
    gas.params.thermal_diffusivity = 0.01;
    gas.params.open_top = false;
    let center = gas.cell_center(n / 2, n / 2, n / 2);
    let s0 = 0.2_f32;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let r2 = (gas.cell_center(i, j, k) - center).length_squared();
                let c = gas.idx(i, j, k);
                gas.heat[c] = 10.0 * (-r2 / (2.0 * s0 * s0)).exp();
            }
        }
    }
    let before = gas.total_heat();
    for _ in 0..60 {
        gas.step(&[], Vec3::ZERO, DT);
    }
    let peak = gas.heat[gas.idx(n / 2, n / 2, n / 2)];
    let expected = 10.0 * (s0 * s0 / (s0 * s0 + 2.0 * 0.01 * 0.5)).powf(1.5);
    assert!(
        (peak - expected).abs() < 0.02 * expected,
        "peak {peak:.4} K, expected {expected:.4} K"
    );
    let after = gas.total_heat();
    assert!(
        (after - before).abs() < 1e-4 * before,
        "heat {before} -> {after}"
    );
}

#[test]
fn warm_sphere_starts_rising_at_two_thirds_g_beta_delta_t() {
    // A spherical warm patch must also accelerate the air around it; in 3D
    // that added mass is half its own, so it starts at g′/(1 + ½) = ⅔ g′.
    let (n, dx) = (48, 0.06);
    let mut gas = Gas::new(Vec3::ZERO, dx, n, n, n);
    gas.params.open_top = false;
    gas.params.thermal_diffusivity = 0.0;
    let center = gas.cell_center(n / 2, n / 2, n / 2);
    let radius = 0.4;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if (gas.cell_center(i, j, k) - center).length() < radius {
                    let c = gas.idx(i, j, k);
                    gas.heat[c] = 10.0;
                }
            }
        }
    }
    gas.step(&[], Vec3::new(0.0, -G, 0.0), DT);
    let (mut sum, mut count) = (0.0, 0);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if (gas.cell_center(i, j, k) - center).length() < 0.25 {
                    sum += gas.velocity_at_cell(i, j, k).y;
                    count += 1;
                }
            }
        }
    }
    let g_prime = G * 10.0 / gas.params.ambient_temperature;
    let ratio = (sum / count as f32) / (g_prime * DT);
    assert!(
        (ratio - 2.0 / 3.0).abs() < 0.035,
        "initial acceleration {ratio:.4} × g′"
    );
}

#[test]
fn uniformly_warm_sealed_box_stays_still() {
    let mut gas = Gas::new(Vec3::ZERO, 0.1, 16, 12, 10);
    gas.params.open_top = false;
    gas.heat.iter_mut().for_each(|h| *h = 20.0);
    for _ in 0..60 {
        gas.step(&[], Vec3::new(0.0, -G, 0.0), DT);
    }
    let vmax = gas
        .u
        .iter()
        .chain(&gas.v)
        .chain(&gas.w)
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(vmax < 1e-4, "air moved at {vmax} m/s");
}

#[test]
fn hot_plume_rises_past_an_obstacle() {
    let mut w = World::new();
    let mut gas = Gas::new(Vec3::new(-0.8, 0.0, -0.8), 0.05, 32, 48, 32);
    gas.emitters.push(Emitter {
        center: Vec3::new(0.0, 0.25, 0.0),
        radius: 0.15,
        smoke_rate: 2.0,
        temperature: 293.15 + 80.0,
        velocity: Vec3::ZERO,
    });
    w.gas = Some(gas);
    w.add_body(RigidBody::new_static(
        Shape::cuboid(0.4, 0.06, 0.4),
        Vec3::new(0.0, 1.0, 0.0),
    ));
    for _ in 0..(2.0 / DT) as usize {
        w.step(DT);
    }
    let gas = w.gas.as_ref().unwrap();
    let (mut sy, mut s) = (0.0, 0.0);
    for k in 0..gas.nz {
        for j in 0..gas.ny {
            for i in 0..gas.nx {
                let c = gas.smoke[gas.idx(i, j, k)];
                sy += c * gas.cell_center(i, j, k).y;
                s += c;
            }
        }
    }
    assert!(sy / s > 0.6, "smoke centroid at {} m", sy / s);
    assert!(
        gas.stats.max_divergence < 1e-2,
        "max ∇·u {}",
        gas.stats.max_divergence
    );
}

#[test]
fn gas_round_trips_through_scene_files() {
    let mut w = engine3d::presets::smoke_plume();
    w.step(DT);
    let loaded = engine3d::Scene::from_json(&engine3d::Scene::from_world(&w).to_json()).unwrap();
    assert_eq!(loaded.world, w);
}
