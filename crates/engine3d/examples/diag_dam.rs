use engine3d::presets;
use std::time::Instant;
fn main() {
    let mut w = presets::dam_break();
    let n0 = w.fluid.len();
    println!("dam break: {n0} particles");
    let t0 = Instant::now();
    let mut max_v = 0.0f32;
    for step in 0..600 {
        w.step(1.0 / 120.0);
        max_v = max_v.max(
            w.fluid
                .velocities
                .iter()
                .map(|v| v.length())
                .fold(0.0, f32::max),
        );
        if step % 60 == 59 {
            let front = w
                .fluid
                .positions
                .iter()
                .map(|p| p.x)
                .fold(-9.0f32, f32::max);
            let top = w.fluid.positions.iter().map(|p| p.y).fold(0.0f32, f32::max);
            let out = w
                .fluid
                .positions
                .iter()
                .filter(|p| p.x.abs() > 1.5 || p.z.abs() > 0.6 || p.y < 0.0)
                .count();
            println!(
                "t={:.1} n={} front={:.2} top={:.2} out={} KE={:.2} substeps={} maxc={:.3} vmax={:.2}",
                w.time,
                w.fluid.len(),
                front,
                top,
                out,
                w.fluid.kinetic_energy(),
                w.fluid.stats.substeps,
                w.fluid.stats.max_compression,
                max_v
            );
        }
    }
    println!("{:.1} ms/step", t0.elapsed().as_secs_f64() * 1000.0 / 600.0);
}
