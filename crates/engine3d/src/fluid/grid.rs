//! Neighbor search: a uniform grid of cubic cells the size of the kernel
//! support, so every neighbor within that radius is in the query point's own
//! cell or one of the 26 around it.
//!
//! Built with a counting sort into one flat array, as in 2D. Dense layout
//! (cells over the particles' bounding box, x fastest, so each query reads 9
//! runs of 3 adjacent cells) when that box is compact; otherwise cells are
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

#[derive(Debug, Clone, Default)]
pub struct NeighborGrid {
    inv_cell: f32,
    layout: Layout,
    mask: usize,
    starts: Vec<u32>,
    /// Point indices, grouped by slot.
    entries: Vec<u32>,
    /// Each entry's cell, to filter hash collisions.
    entry_cells: Vec<[i32; 3]>,
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

impl NeighborGrid {
    pub fn build(points: &[Vec3], cell: f32) -> Self {
        let mut g = Self::default();
        g.rebuild(points, cell);
        g
    }

    pub fn rebuild(&mut self, points: &[Vec3], cell: f32) {
        self.inv_cell = 1.0 / cell;
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
        let dense_limit = (MAX_DENSE_CELLS_PER_POINT * cells.len().max(64)) as i64;
        let slots = if !cells.is_empty() && count <= dense_limit {
            let size = size.map(|s| s as i32);
            let count = count as usize;
            self.layout = Layout::Dense { origin: lo, size };
            count
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
        self.entries.resize(cells.len(), 0);
        self.entry_cells.clear();
        self.entry_cells.resize(cells.len(), [0; 3]);
        for (i, &c) in cells.iter().enumerate() {
            let s = self.slot(c).expect("checked above");
            let at = fill[s] as usize;
            self.entries[at] = i as u32;
            self.entry_cells[at] = c;
            fill[s] += 1;
        }
        self.cells = cells;
    }

    fn cell_of(&self, p: Vec3) -> [i32; 3] {
        let c = (p * self.inv_cell).floor();
        [c.x as i32, c.y as i32, c.z as i32]
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

    /// Call `f(index)` for every point in the 27 cells around `p`: a
    /// superset of the points within one cell size of `p`.
    pub fn for_each_candidate(&self, p: Vec3, mut f: impl FnMut(usize)) {
        if self.entries.is_empty() {
            return;
        }
        let c = self.cell_of(p);
        match self.layout {
            Layout::Dense { origin, size } => {
                let x0 = (c[0] - 1 - origin[0]).max(0);
                let x1 = (c[0] + 1 - origin[0]).min(size[0] - 1);
                if x0 > x1 {
                    return;
                }
                for dz in -1..=1 {
                    let z = c[2] + dz - origin[2];
                    if z < 0 || z >= size[2] {
                        continue;
                    }
                    for dy in -1..=1 {
                        let y = c[1] + dy - origin[1];
                        if y < 0 || y >= size[1] {
                            continue;
                        }
                        let row = ((z * size[1] + y) * size[0]) as usize;
                        let (a, b) = (
                            self.starts[row + x0 as usize] as usize,
                            self.starts[row + x1 as usize + 1] as usize,
                        );
                        for &i in &self.entries[a..b] {
                            f(i as usize);
                        }
                    }
                }
            }
            Layout::Hashed => {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let cell = [c[0] + dx, c[1] + dy, c[2] + dz];
                            let s = hash(cell, self.mask);
                            let (a, b) = (self.starts[s] as usize, self.starts[s + 1] as usize);
                            for k in a..b {
                                if self.entry_cells[k] == cell {
                                    f(self.entries[k] as usize);
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
        let cell = 0.15;
        let g = NeighborGrid::build(points, cell);
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
