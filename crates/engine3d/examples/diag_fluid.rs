// Temporary diagnostics for 3D SPH.
use engine3d::{RigidBody, Shape, Vec3, presets};
use std::time::Instant;

fn main() {
    let dt = 1.0 / 120.0;
    // Still water in a tank: pressure vs depth, leaks, settling.
    let mut w = presets::tank(1.0, 1.0, 1.0);
    let bodies = w.bodies.clone();
    let n = w.fluid.fill_box(
        Vec3::new(-0.5, 0.0, -0.5),
        Vec3::new(0.5, 0.6, 0.5),
        &bodies,
    );
    println!("tank: {n} particles");
    let t0 = Instant::now();
    let mut acc = vec![(0.0f64, 0u32); 6];
    for step in 0..360 {
        w.step(dt);
        if step >= 240 {
            let top = w.fluid.positions.iter().map(|p| p.y).fold(0.0f32, f32::max);
            for (p, pr) in w.fluid.positions.iter().zip(w.fluid.pressures()) {
                if p.x.abs() < 0.25 && p.z.abs() < 0.25 {
                    let bin = ((top - p.y) / 0.1) as usize;
                    if bin < 6 {
                        acc[bin].0 += *pr as f64;
                        acc[bin].1 += 1;
                    }
                }
            }
        }
        if step % 60 == 59 {
            let top = w.fluid.positions.iter().map(|p| p.y).fold(0.0f32, f32::max);
            let min = w.fluid.positions.iter().map(|p| p.y).fold(9.0f32, f32::min);
            let out = w
                .fluid
                .positions
                .iter()
                .filter(|p| p.x.abs() > 0.5 || p.z.abs() > 0.5 || p.y < 0.0)
                .count();
            println!(
                "t={:.1} n={} KE={:.3} top={:.3} min={:.3} out={} substeps={} maxc={:.3}",
                w.time,
                w.fluid.len(),
                w.fluid.kinetic_energy(),
                top,
                min,
                out,
                w.fluid.stats.substeps,
                w.fluid.stats.max_compression
            );
        }
    }
    println!("{:.2} ms/step", t0.elapsed().as_secs_f64() * 1000.0 / 360.0);
    for (i, (s, c)) in acc.iter().enumerate() {
        let depth = (i as f32 + 0.5) * 0.1;
        println!(
            "depth {:.2}: p {:.0} Pa, rho g h {:.0}",
            depth,
            s / *c as f64,
            1000.0 * 9.80665 * depth
        );
    }

    // Buoyancy: equilibrium draft of cubes.
    for density in [250.0, 500.0, 750.0, 2000.0] {
        let mut w = presets::tank(1.6, 1.6, 1.2);
        let bodies = w.bodies.clone();
        w.fluid.fill_box(
            Vec3::new(-0.8, 0.0, -0.8),
            Vec3::new(0.8, 0.6, 0.8),
            &bodies,
        );
        let mut b = RigidBody::new(Shape::cube(0.4), Vec3::new(0.0, 0.85, 0.0));
        b.density = density;
        let id = w.add_body(b);
        let t0 = Instant::now();
        let mut ys = Vec::new();
        let mut levels = Vec::new();
        for step in 0..720 {
            w.step(dt);
            if step >= 480 {
                ys.push(w.body(id).unwrap().position.y);
                let far: Vec<f32> = w
                    .fluid
                    .positions
                    .iter()
                    .filter(|p| p.x.abs() > 0.5 || p.z.abs() > 0.5)
                    .map(|p| p.y)
                    .collect();
                let top = far.iter().cloned().fold(0.0f32, f32::max);
                levels.push(top);
            }
        }
        let y = ys.iter().sum::<f32>() / ys.len() as f32;
        let level = levels.iter().sum::<f32>() / levels.len() as f32 + 0.05; // top particle center + d/2
        let draft = level - (y - 0.2);
        println!(
            "density {density}: body y {y:.3}, water level {level:.3}, draft {draft:.3} (expect {:.3}), {:.1} ms/step",
            (density / 1000.0f32).min(1.0) * 0.4,
            t0.elapsed().as_secs_f64() * 1000.0 / 720.0
        );
    }
}
