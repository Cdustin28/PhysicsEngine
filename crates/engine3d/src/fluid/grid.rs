//! Neighbor search: a uniform grid of cubic cells. With cells the size of
//! the search radius, every neighbor is in the query point's own cell or
//! one of the 26 around it; with half-size cells ([`NeighborGrid::rebuild_fine`],
//! the fluid's choice) the 5×5×5 block around it, which covers 42% less
//! volume and so has fewer candidates to check.
//!
//! Built with a counting sort into one flat array, as in 2D, with a copy of
//! each point's position so queries read contiguous memory. Dense layout
//! (cells over the particles' bounding box, x fastest, so each query reads
//! runs of adjacent cells) when that box is compact; otherwise cells are
//! hashed into a table about twice the particle count.

use glam::Vec3;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum Layout {
    Dense {
        origin: [i32; 3],
        size: [i32; 3],
    },
    #[default]
    Hashed,
}

/// A point as stored in the grid: its position and index.
#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    p: Vec3,
    index: u32,
}

#[derive(Debug, Clone, Default)]
pub struct NeighborGrid {
    inv_cell: f32,
    /// Cells searched on each side of a query's own cell.
    reach: i32,
    layout: Layout,
    mask: usize,
    starts: Vec<u32>,
    /// Points grouped by slot.
    entries: Vec<Entry>,
    cells: Vec<[i32; 3]>,
}

const MAX_DENSE_CELLS_PER_POINT: usize = 8;

fn hash(c: [i32; 3], mask: usize) -> usize {
    // Teschner et al., "Optimized Spatial Hashing" (2003).
    ((c[0] as u32).wrapping_mul(73_856_093)
        ^ (c[1] as u32).wrapping_mul(19_349_663)
        ^ (c[2] as u32).wrapping_mul(83_492_791)) as usize
        & mask
}

/// ⌊x⌋ as an integer, saturating for huge values. Written out because
/// `f32::floor` is a library call on baseline x86-64.
#[inline]
fn floor_i32(x: f32) -> i32 {
    let i = x as i32;
    if (i as f32) > x { i - 1 } else { i }
}

impl NeighborGrid {
    pub fn build(points: &[Vec3], cell: f32) -> Self {
        let mut g = Self::default();
        g.rebuild(points, cell);
        g
    }

    /// Cells the size of the search radius `cell`; queries search 3×3×3.
    pub fn rebuild(&mut self, points: &[Vec3], cell: f32) {
        self.rebuild_with_reach(points, cell, 1);
    }

    /// Cells half the search `radius`; queries search 5×5×5.
    pub fn rebuild_fine(&mut self, points: &[Vec3], radius: f32) {
        self.rebuild_with_reach(points, 0.5 * radius, 2);
    }

    fn rebuild_with_reach(&mut self, points: &[Vec3], cell: f32, reach: i32) {
        self.inv_cell = 1.0 / cell;
        self.reach = reach;
        let mut cells = std::mem::take(&mut self.cells);
        cells.clear();
        cells.extend(points.iter().map(|&p| self.cell_of(p)));

        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for c in &cells {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        // Cell count of the bounding box, in i64 so far-flung points (or no
        // points: lo > hi) can't overflow.
        let size = [0, 1, 2].map(|k| (i64::from(hi[k]) - i64::from(lo[k]) + 1).max(1));
        let count = size.iter().product::<i64>();
        let per_point = MAX_DENSE_CELLS_PER_POINT * (reach * reach * reach) as usize;
        let dense_limit = (per_point * cells.len().max(64)) as i64;
        let slots = if !cells.is_empty() && count <= dense_limit {
            let size = size.map(|s| s as i32);
            self.layout = Layout::Dense { origin: lo, size };
            count as usize
        } else {
            self.layout = Layout::Hashed;
            let n = (2 * cells.len()).next_power_of_two().max(16);
            self.mask = n - 1;
            n
        };

        self.starts.clear();
        self.starts.resize(slots + 1, 0);
        for &c in &cells {
            let s = self.slot(c).expect("every point's own cell has a slot");
            self.starts[s + 1] += 1;
        }
        for s in 0..slots {
            self.starts[s + 1] += self.starts[s];
        }
        let mut fill = self.starts.clone();
        self.entries.clear();
        self.entries.resize(cells.len(), Entry::default());
        for (i, &c) in cells.iter().enumerate() {
            let s = self.slot(c).expect("checked above");
            let at = fill[s] as usize;
            self.entries[at] = Entry {
                p: points[i],
                index: i as u32,
            };
            fill[s] += 1;
        }
        self.cells = cells;
    }

    fn cell_of(&self, p: Vec3) -> [i32; 3] {
        let q = p * self.inv_cell;
        [floor_i32(q.x), floor_i32(q.y), floor_i32(q.z)]
    }

    fn slot(&self, c: [i32; 3]) -> Option<usize> {
        match self.layout {
            Layout::Dense { origin, size } => {
                let l = [0, 1, 2].map(|k| c[k] - origin[k]);
                if (0..3).any(|k| l[k] < 0 || l[k] >= size[k]) {
                    return None;
                }
                Some(((l[2] * size[1] + l[1]) * size[0] + l[0]) as usize)
            }
            Layout::Hashed => Some(hash(c, self.mask)),
        }
    }

    /// Call `f(index)` for every point in the block of cells around `p`: a
    /// superset of the points within the search radius of `p`.
    pub fn for_each_candidate(&self, p: Vec3, mut f: impl FnMut(usize)) {
        self.for_each_near(p, |j, _| f(j));
    }

    /// Like [`Self::for_each_candidate`], also passing each candidate's
    /// position.
    pub fn for_each_near(&self, p: Vec3, mut f: impl FnMut(usize, Vec3)) {
        if self.entries.is_empty() {
            return;
        }
        let c = self.cell_of(p);
        let r = self.reach.max(1);
        match self.layout {
            Layout::Dense { origin, size } => {
                let x0 = c[0].saturating_sub(r).saturating_sub(origin[0]).max(0);
                let x1 = c[0]
                    .saturating_add(r)
                    .saturating_sub(origin[0])
                    .min(size[0] - 1);
                if x0 > x1 {
                    return;
                }
                for dz in -r..=r {
                    let z = c[2].saturating_add(dz).saturating_sub(origin[2]);
                    if z < 0 || z >= size[2] {
                        continue;
                    }
                    for dy in -r..=r {
                        let y = c[1].saturating_add(dy).saturating_sub(origin[1]);
                        if y < 0 || y >= size[1] {
                            continue;
                        }
                        let row = ((z * size[1] + y) * size[0]) as usize;
                        let (a, b) = (
                            self.starts[row + x0 as usize] as usize,
                            self.starts[row + x1 as usize + 1] as usize,
                        );
                        for e in &self.entries[a..b] {
                            f(e.index as usize, e.p);
                        }
                    }
                }
            }
            Layout::Hashed => {
                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let cell = [
                                c[0].saturating_add(dx),
                                c[1].saturating_add(dy),
                                c[2].saturating_add(dz),
                            ];
                            let s = hash(cell, self.mask);
                            let (a, b) = (self.starts[s] as usize, self.starts[s + 1] as usize);
                            for e in &self.entries[a..b] {
                                if self.cell_of(e.p) == cell {
                                    f(e.index as usize, e.p);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> Vec<Vec3> {
        // A pseudo-random cloud (deterministic LCG).
        let mut s = 12345_u32;
        let mut r = || {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (s >> 8) as f32 / (1 << 24) as f32
        };
        (0..2000)
            .map(|_| Vec3::new(r(), r(), r()) * 2.0 - Vec3::ONE)
            .collect()
    }

    fn check(points: &[Vec3]) {
        check_with(points, false);
        check_with(points, true);
    }

    fn check_with(points: &[Vec3], fine: bool) {
        let cell = 0.15;
        let mut g = NeighborGrid::default();
        if fine {
            g.rebuild_fine(points, cell);
        } else {
            g.rebuild(points, cell);
        }
        for (i, &p) in points.iter().enumerate().step_by(37) {
            let mut found = Vec::new();
            g.for_each_candidate(p, |j| {
                if (points[j] - p).length() < cell {
                    found.push(j);
                }
            });
            found.sort_unstable();
            let brute: Vec<usize> = (0..points.len())
                .filter(|&j| (points[j] - p).length() < cell)
                .collect();
            assert_eq!(found, brute, "point {i}");
        }
    }

    #[test]
    fn dense_grid_matches_brute_force() {
        let p = points();
        check(&p);
        assert!(matches!(
            NeighborGrid::build(&p, 0.15).layout,
            Layout::Dense { .. }
        ));
    }

    #[test]
    fn hashed_grid_matches_brute_force() {
        let mut p = points();
        p.push(Vec3::splat(1000.0)); // one outlier forces hashing
        assert_eq!(NeighborGrid::build(&p, 0.15).layout, Layout::Hashed);
        check(&p);
    }
}
