//! Neighbor search for SPH: a uniform grid.
//!
//! Space is divided into square cells the size of the kernel support. Every
//! neighbor of a particle within that radius is then in its own cell or one
//! of the 8 around it, so a query only looks at 9 cells instead of every
//! particle.
//!
//! The grid is built with a counting sort: count particles per cell (or
//! bucket), prefix-sum the counts into start offsets, then drop each particle
//! into place. That gives one flat array, with no per-cell allocations.
//!
//! Two layouts:
//!
//! - **Dense** (the usual case): cells over the particles' bounding box,
//!   numbered row by row. The 3 cells of each row a query needs are
//!   adjacent, so a query reads 3 contiguous slices with no filtering.
//! - **Hashed** (fallback): when particles are spread so thinly that a dense
//!   grid would be mostly empty (say, one flung far away), cells are hashed
//!   into a table about twice the particle count instead.

use glam::Vec2;

/// A point as stored in the grid: a copy of its position, so queries read
/// one contiguous array instead of looking each candidate up elsewhere.
#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    p: Vec2,
    index: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum Layout {
    /// Row-major cells: cell (cx, cy) is slot (cy − y0) · nx + (cx − x0).
    Dense { x0: i32, y0: i32, nx: i32, ny: i32 },
    /// Hashed buckets; `mask` = table size − 1.
    #[default]
    Hashed,
}

#[derive(Debug, Clone, Default)]
pub struct NeighborGrid {
    inv_cell: f32,
    /// Cells searched on each side of a query's own cell: 1 for cells the
    /// size of the search radius, 2 for half that (see
    /// [`NeighborGrid::rebuild_fine`]).
    reach: i32,
    layout: Layout,
    mask: usize,
    /// `starts[s]..starts[s + 1]` is slot `s`'s range in `entries`.
    starts: Vec<u32>,
    entries: Vec<Entry>,
    cells: Vec<(i32, i32)>,
}

/// Use the dense layout when it needs at most this many cells per point.
const MAX_DENSE_CELLS_PER_POINT: usize = 4;

/// ⌊x⌋ as an integer, saturating for huge values (so a far-flung particle
/// shares an edge cell instead of breaking the grid). Written out because
/// `f32::floor` is a library call on baseline x86-64, and this is on the
/// hottest path.
#[inline]
fn floor_i32(x: f32) -> i32 {
    let i = x as i32;
    if (i as f32) > x { i - 1 } else { i }
}

fn hash(cx: i32, cy: i32, mask: usize) -> usize {
    // Large primes from Teschner et al., "Optimized Spatial Hashing" (2003).
    ((cx as u32).wrapping_mul(73_856_093) ^ (cy as u32).wrapping_mul(19_349_663)) as usize & mask
}

impl NeighborGrid {
    /// Build a grid over `points` with square cells of side `cell`. A query
    /// finds every point within distance `cell`.
    pub fn build(points: &[Vec2], cell: f32) -> Self {
        let mut grid = Self::default();
        grid.rebuild(points, cell);
        grid
    }

    /// Rebuild in place, reusing allocations.
    pub fn rebuild(&mut self, points: &[Vec2], cell: f32) {
        self.rebuild_with_reach(points, cell, 1);
    }

    /// Rebuild with cells half the search `radius` across, searching the
    /// 5×5 block around a query instead of 3×3. The block covers 25
    /// (r/2)² = 6.25 r² instead of 9 r², so queries check about 30% fewer
    /// candidates; worth it when queries dominate (the fluid's neighbor
    /// search).
    pub fn rebuild_fine(&mut self, points: &[Vec2], radius: f32) {
        self.rebuild_with_reach(points, 0.5 * radius, 2);
    }

    fn rebuild_with_reach(&mut self, points: &[Vec2], cell: f32, reach: i32) {
        self.inv_cell = 1.0 / cell;
        self.reach = reach;
        let mut cells = std::mem::take(&mut self.cells);
        cells.clear();
        cells.extend(points.iter().map(|&p| self.cell_of(p)));

        // Choose the layout from the bounding box of occupied cells.
        let (mut lo, mut hi) = ((i32::MAX, i32::MAX), (i32::MIN, i32::MIN));
        for &(cx, cy) in &cells {
            lo = (lo.0.min(cx), lo.1.min(cy));
            hi = (hi.0.max(cx), hi.1.max(cy));
        }
        let (nx, ny) = (
            i64::from(hi.0) - i64::from(lo.0) + 1,
            i64::from(hi.1) - i64::from(lo.1) + 1,
        );
        let dense_cells = nx.saturating_mul(ny);
        let slots = if !cells.is_empty()
            && dense_cells
                <= (points.len() * MAX_DENSE_CELLS_PER_POINT * (reach * reach) as usize + 64) as i64
        {
            self.layout = Layout::Dense {
                x0: lo.0,
                y0: lo.1,
                nx: nx as i32,
                ny: ny as i32,
            };
            dense_cells as usize
        } else {
            self.layout = Layout::Hashed;
            let buckets = (points.len() * 2).next_power_of_two().max(16);
            self.mask = buckets - 1;
            buckets
        };

        self.starts.clear();
        self.starts.resize(slots + 1, 0);
        for &(cx, cy) in &cells {
            let s = self
                .slot(cx, cy)
                .expect("every point's cell is in the grid");
            self.starts[s + 1] += 1;
        }
        for s in 0..slots {
            self.starts[s + 1] += self.starts[s];
        }

        self.entries.clear();
        self.entries.resize(points.len(), Entry::default());
        let mut next: Vec<u32> = self.starts[..slots].to_vec();
        for (i, &(cx, cy)) in cells.iter().enumerate() {
            let s = self
                .slot(cx, cy)
                .expect("every point's cell is in the grid");
            self.entries[next[s] as usize] = Entry {
                p: points[i],
                index: i as u32,
            };
            next[s] += 1;
        }
        self.cells = cells;
    }

    /// Storage slot of cell (cx, cy), or `None` for a cell outside a dense
    /// grid (which therefore holds no points).
    fn slot(&self, cx: i32, cy: i32) -> Option<usize> {
        match self.layout {
            Layout::Dense { x0, y0, nx, ny } => {
                let (x, y) = (i64::from(cx) - i64::from(x0), i64::from(cy) - i64::from(y0));
                (x >= 0 && y >= 0 && x < i64::from(nx) && y < i64::from(ny))
                    .then(|| (y * i64::from(nx) + x) as usize)
            }
            Layout::Hashed => Some(hash(cx, cy, self.mask)),
        }
    }

    fn cell_of(&self, p: Vec2) -> (i32, i32) {
        (
            floor_i32(p.x * self.inv_cell),
            floor_i32(p.y * self.inv_cell),
        )
    }

    /// Call `f(index)` for every point in the block of cells around `p`
    /// (3×3, or 5×5 after [`Self::rebuild_fine`]). That is a superset of the
    /// points within the search radius of `p`; callers do the exact
    /// distance check.
    pub fn for_each_candidate(&self, p: Vec2, mut f: impl FnMut(usize)) {
        self.for_each_near(p, |j, _| f(j));
    }

    /// Like [`Self::for_each_candidate`], also passing each candidate's
    /// position (from the grid's own copy, which is faster to read).
    pub fn for_each_near(&self, p: Vec2, mut f: impl FnMut(usize, Vec2)) {
        if self.entries.is_empty() {
            return;
        }
        let (cx, cy) = self.cell_of(p);
        let reach = self.reach.max(1);
        if let Layout::Dense { x0, y0, nx, ny } = self.layout {
            // A few rows; in each, cells cx−reach..=cx+reach are adjacent
            // slots.
            let r = i64::from(reach);
            let x_lo = (i64::from(cx) - r - i64::from(x0)).max(0);
            let x_hi = (i64::from(cx) + r - i64::from(x0)).min(i64::from(nx) - 1);
            if x_lo > x_hi {
                return;
            }
            for dy in -r..=r {
                let y = i64::from(cy) + dy - i64::from(y0);
                if y < 0 || y >= i64::from(ny) {
                    continue;
                }
                let row = y * i64::from(nx);
                let range = self.starts[(row + x_lo) as usize] as usize
                    ..self.starts[(row + x_hi + 1) as usize] as usize;
                for e in &self.entries[range] {
                    f(e.index as usize, e.p);
                }
            }
            return;
        }
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (x, y) = (cx.saturating_add(dx), cy.saturating_add(dy));
                let b = hash(x, y, self.mask);
                let range = self.starts[b] as usize..self.starts[b + 1] as usize;
                for e in &self.entries[range] {
                    // Different cells can share a bucket; only take this one.
                    if self.cell_of(e.p) == (x, y) {
                        f(e.index as usize, e.p);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0 as f32 / u32::MAX as f32
        }
    }

    #[test]
    fn matches_brute_force() {
        let mut rng = Rng(0xDEAD_BEEF);
        let points: Vec<Vec2> = (0..2000)
            .map(|_| Vec2::new(rng.next() * 6.0 - 3.0, rng.next() * 4.0 - 1.0))
            .collect();
        let radius = 0.1;
        let grid = NeighborGrid::build(&points, radius);
        for (i, &p) in points.iter().enumerate().step_by(7) {
            let mut found = Vec::new();
            grid.for_each_candidate(p, |j| {
                if j != i && (points[j] - p).length_squared() < radius * radius {
                    found.push(j);
                }
            });
            found.sort_unstable();
            let expected: Vec<usize> = (0..points.len())
                .filter(|&j| j != i && (points[j] - p).length_squared() < radius * radius)
                .collect();
            assert_eq!(found, expected, "particle {i}");
        }
    }

    #[test]
    fn fine_grid_matches_brute_force() {
        let mut rng = Rng(0x1234_5678);
        let mut points: Vec<Vec2> = (0..2000)
            .map(|_| Vec2::new(rng.next() * 6.0 - 3.0, rng.next() * 4.0 - 1.0))
            .collect();
        let radius = 0.13;
        for outlier in [false, true] {
            if outlier {
                // One far-flung point forces the hashed layout.
                points.push(Vec2::new(900.0, -900.0));
            }
            let mut grid = NeighborGrid::default();
            grid.rebuild_fine(&points, radius);
            assert_eq!(grid.layout == Layout::Hashed, outlier);
            for (i, &p) in points.iter().enumerate().step_by(7) {
                let mut found = Vec::new();
                grid.for_each_near(p, |j, pj| {
                    assert_eq!(pj, points[j]);
                    if j != i && (pj - p).length_squared() < radius * radius {
                        found.push(j);
                    }
                });
                found.sort_unstable();
                let expected: Vec<usize> = (0..points.len())
                    .filter(|&j| j != i && (points[j] - p).length_squared() < radius * radius)
                    .collect();
                assert_eq!(found, expected, "particle {i}");
            }
        }
    }

    #[test]
    fn handles_far_away_and_negative_coordinates() {
        let points = vec![
            Vec2::new(-1000.05, -2000.0),
            Vec2::new(-1000.0, -2000.0),
            Vec2::new(1e9, 1e9),
        ];
        let grid = NeighborGrid::build(&points, 0.1);
        let mut found = Vec::new();
        grid.for_each_candidate(points[0], |j| found.push(j));
        found.sort_unstable();
        assert_eq!(found, vec![0, 1]);
    }

    #[test]
    fn dense_and_hashed_layouts_are_both_used() {
        let mut grid = NeighborGrid::default();
        let compact: Vec<Vec2> = (0..100)
            .map(|i| Vec2::new((i % 10) as f32, (i / 10) as f32) * 0.05)
            .collect();
        grid.rebuild(&compact, 0.1);
        assert!(matches!(grid.layout, Layout::Dense { .. }));
        let mut spread = compact.clone();
        spread.push(Vec2::new(500.0, 500.0));
        grid.rebuild(&spread, 0.1);
        assert_eq!(grid.layout, Layout::Hashed);
    }

    #[test]
    fn empty_grid_is_fine() {
        let grid = NeighborGrid::build(&[], 0.1);
        grid.for_each_candidate(Vec2::ZERO, |_| panic!("no points"));
    }
}
