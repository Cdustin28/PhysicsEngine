//! Criterion benchmarks for the engine's hot paths.
//!
//! ```sh
//! cargo bench -p engine                     # single-threaded
//! cargo bench -p engine --features parallel # multithreaded fluid
//! ```
//!
//! Criterion keeps results in `target/criterion` and reports changes against
//! the previous run, so a regression shows up as "Performance has regressed".

use criterion::{Criterion, criterion_group, criterion_main};
use engine::fluid::grid::NeighborGrid;
use engine::{Vec2, World, presets};
use std::hint::black_box;

/// About 5,000 points spread like a settled pool of water.
fn pool_points() -> Vec<Vec2> {
    let d = 0.05;
    (0..5_000)
        .map(|i| {
            let (x, y) = ((i % 100) as f32, (i / 100) as f32);
            // Slight jitter so it isn't a perfect lattice.
            let jitter = ((i * 7919) % 97) as f32 / 97.0 - 0.5;
            Vec2::new(x * d + 0.1 * d * jitter, y * d - 0.1 * d * jitter)
        })
        .collect()
}

fn neighbor_search(c: &mut Criterion) {
    let points = pool_points();
    let h = 0.1;
    c.bench_function("grid build, 5k points", |b| {
        let mut grid = NeighborGrid::default();
        b.iter(|| grid.rebuild(black_box(&points), h));
    });
    let grid = NeighborGrid::build(&points, h);
    c.bench_function("grid query all, 5k points", |b| {
        b.iter(|| {
            let mut found = 0_usize;
            for &p in &points {
                grid.for_each_candidate(p, |j| {
                    if (points[j] - p).length_squared() < h * h {
                        found += 1;
                    }
                });
            }
            black_box(found)
        });
    });
}

fn world_steps(c: &mut Criterion) {
    let dt = 1.0 / 120.0;

    // Fluid: the dam-break example after it has started moving.
    let mut dam = presets::dam_break();
    for _ in 0..30 {
        dam.step(dt);
    }
    c.bench_function("world step, dam break (1.5k particles)", |b| {
        b.iter_batched(
            || dam.clone(),
            |mut w: World| w.step(dt),
            criterion::BatchSize::SmallInput,
        );
    });

    // Rigid bodies: the 55-box pyramid at rest (warm-started contacts).
    let mut pyramid = presets::pyramid();
    for _ in 0..120 {
        pyramid.step(dt);
    }
    c.bench_function("world step, 55-box pyramid", |b| {
        b.iter_batched(
            || pyramid.clone(),
            |mut w: World| w.step(dt),
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, neighbor_search, world_steps);
criterion_main!(benches);
