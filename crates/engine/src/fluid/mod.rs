//! Particle-based fluid: weakly compressible SPH (WCSPH).
//!
//! The fluid is a cloud of particles, each carrying a little mass of water.
//! Every substep:
//!
//! 1. **Neighbors.** Find each particle's neighbors within the kernel support
//!    `h` (a hash grid; see [`grid`]).
//! 2. **Density.** ρᵢ = Σⱼ m W(xᵢ − xⱼ) + Σ_b ψ_b W(xᵢ − x_b). Fluid
//!    neighbors plus boundary particles on nearby bodies (see [`boundary`]).
//! 3. **Pressure.** From the Tait equation of state (Becker & Teschner
//!    2007): pᵢ = B ((ρᵢ/ρ₀)⁷ − 1), with B = ρ₀ c² / 7. Water resists
//!    compression very strongly, so a 1% density increase already gives a
//!    large pressure. Pressure is clamped at zero, since real water doesn't
//!    pull on itself at a free surface (and allowing it causes clumping).
//! 4. **Forces.** Pressure gradient, artificial viscosity (Monaghan 1992),
//!    and gravity.
//! 5. **Integrate** with semi-implicit Euler, like the rigid bodies.
//!
//! "Weakly compressible" means we let the fluid compress a little (a few
//! percent) in exchange for a simple explicit solver. The price is small time
//! steps: information travels at the numerical speed of sound `c`, and a step
//! must not let it skip past a particle (the CFL condition,
//! dt ≤ λ·d / (c + v_max)). The world steps at 120 Hz, and the fluid takes as
//! many substeps as CFL requires inside each world step.
//!
//! # Units
//!
//! Like the rest of the engine this is 2D, so "density" is mass per area
//! (kg/m²) and pressure is force per length (N/m). Water is 1000 kg/m³; the
//! default `rest_density` of 10 kg/m² is water in a 1 cm thick slab, the
//! same convention as [`RigidBody::DEFAULT_DENSITY`](crate::RigidBody::DEFAULT_DENSITY).

pub mod boundary;
pub mod grid;
pub mod kernel;

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use boundary::Boundary;
use glam::Vec2;
use grid::NeighborGrid;
use kernel::CubicSpline;
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `par_iter_mut()` when the `parallel` feature is on (desktop), plain
/// `iter_mut()` otherwise (web, tests). Both support the same
/// `enumerate`/`zip`/`for_each` calls used below.
macro_rules! iter_mut {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter_mut();
        it
    }};
}

/// Largest value from a (parallel or sequential) iterator of non-negative
/// numbers; 0 if empty.
#[cfg(feature = "parallel")]
fn max_reduce(it: impl ParallelIterator<Item = f32>) -> f32 {
    it.reduce(|| 0.0, f32::max)
}

#[cfg(not(feature = "parallel"))]
fn max_reduce(it: impl Iterator<Item = f32>) -> f32 {
    it.fold(0.0, f32::max)
}

macro_rules! iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        let it = $e.par_iter();
        #[cfg(not(feature = "parallel"))]
        let it = $e.iter();
        it
    }};
}

/// CFL safety factor λ in dt ≤ λ·d / (c + v_max).
pub const CFL: f32 = 0.4;
/// Particles are never allowed to move faster than this fraction of the
/// speed of sound. Past that, WCSPH is meaningless anyway, and the cap
/// bounds the number of substeps per world step.
pub const MAX_SPEED_FRACTION: f32 = 0.5;
/// Exponent of the Tait equation; 7 is the standard value for water.
pub const TAIT_GAMMA: i32 = 7;
/// Boundary particle spacing as a fraction of fluid particle spacing.
/// Denser than the fluid so particles can't squeeze through gaps.
pub const BOUNDARY_SPACING_FACTOR: f32 = 0.5;
/// How far inside each body's surface its boundary particles sit, as a
/// fraction of particle spacing. Equal to the gap still water leaves next to
/// a boundary layer (measured: 0.59 at walls, 0.48 at the floor), so the
/// fluid's effective boundary is the real surface. See [`boundary`].
pub const BOUNDARY_INSET_FACTOR: f32 = 0.55;
/// Minimum distance between new (or projected) fluid particles and the
/// boundary layer, as a fraction of particle spacing. Boundary particles
/// over-count density very close by, so particles placed much closer than
/// this start out heavily compressed and get launched.
const BOUNDARY_GAP_FACTOR: f32 = 0.9;
/// The same minimum, measured from the body's real surface.
const SURFACE_GAP_FACTOR: f32 = BOUNDARY_GAP_FACTOR - BOUNDARY_INSET_FACTOR;
/// Fluid particles that fall below this height are deleted.
pub const KILL_DEPTH: f32 = -50.0;

/// Material and resolution settings for the fluid.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FluidParams {
    /// Distance between neighboring particles at rest, m. Smaller means more
    /// detail and many more particles (4× as many per halving).
    pub particle_spacing: f32,
    /// ρ₀, kg/m². Water is 10 (1000 kg/m³ × 1 cm).
    pub rest_density: f32,
    /// Numerical speed of sound c, m/s. Should be about 10× the fastest flow
    /// speed for ~1% compression. Higher is stiffer but needs more substeps.
    pub speed_of_sound: f32,
    /// Artificial viscosity α between fluid particles. 0.01–0.1 is typical.
    pub viscosity: f32,
    /// Artificial viscosity between fluid and bodies (wall friction).
    pub wall_viscosity: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            particle_spacing: 0.05,
            rest_density: 10.0,
            speed_of_sound: 20.0,
            viscosity: 0.05,
            // Higher than the fluid's own viscosity on purpose: it damps
            // particles before they reach the zone right at a wall where
            // single-layer boundary handling over-counts density, which
            // otherwise launches spray off walls and corners in hard impacts.
            wall_viscosity: 0.5,
        }
    }
}

impl FluidParams {
    /// Kernel support radius h = 2 × spacing (about 12 neighbors in 2D).
    pub fn support(&self) -> f32 {
        2.0 * self.particle_spacing
    }

    /// The fluid's view of `bodies`: boundary particles with weights.
    pub fn boundary(&self, bodies: &[RigidBody]) -> Boundary {
        Boundary::build(
            bodies,
            self.particle_spacing * BOUNDARY_SPACING_FACTOR,
            self.particle_spacing * BOUNDARY_INSET_FACTOR,
            &self.kernel(),
            self.rest_density,
            self.support(),
        )
    }

    pub fn kernel(&self) -> CubicSpline {
        CubicSpline::new(self.support())
    }

    /// Mass per particle, kg. Chosen so a particle in the middle of a
    /// perfect square lattice at `particle_spacing` has exactly `rest_density`.
    /// (Using ρ₀ d² instead is off by a percent or two because the kernel sum
    /// over a lattice isn't exactly the integral, and that error shows up as
    /// a startup jolt.)
    pub fn particle_mass(&self) -> f32 {
        let d = self.particle_spacing;
        let k = self.kernel();
        let mut sum = 0.0;
        for i in -3_i32..=3 {
            for j in -3_i32..=3 {
                sum += k.w(d * ((i * i + j * j) as f32).sqrt());
            }
        }
        self.rest_density / sum
    }

    /// The speed cap: particles never move faster than this, m/s.
    pub fn max_speed(&self) -> f32 {
        MAX_SPEED_FRACTION * self.speed_of_sound
    }

    /// Smallest number of equal substeps that satisfies the CFL condition
    /// for a `dt` world step even at the speed cap. Solvers that can't cheaply
    /// check the fastest particle every substep (the GPU) use this.
    pub fn worst_case_substeps(&self, dt: f32) -> u32 {
        let limit = CFL * self.particle_spacing / (self.speed_of_sound + self.max_speed());
        ((dt / limit).ceil() as u32).max(1)
    }

    /// Tait stiffness B = ρ₀ c² / γ, N/m.
    pub fn stiffness(&self) -> f32 {
        self.rest_density * self.speed_of_sound * self.speed_of_sound / TAIT_GAMMA as f32
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("particle_spacing", self.particle_spacing),
            ("rest_density", self.rest_density),
            ("speed_of_sound", self.speed_of_sound),
        ];
        for (name, v) in positive {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!("fluid {name} must be positive, got {v}"));
            }
        }
        for (name, v) in [
            ("viscosity", self.viscosity),
            ("wall_viscosity", self.wall_viscosity),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("fluid {name} must be non-negative, got {v}"));
            }
        }
        Ok(())
    }
}

/// Total push of the fluid on one body over a step: linear impulse (N·s) and
/// angular impulse about the body's center (N·m·s).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BodyImpulse {
    pub linear: Vec2,
    pub angular: f32,
}

/// Advances a [`Fluid`] by one world step. [`CpuSolver`] is the reference
/// implementation; the `gpu` crate provides a compute-shader one. They must
/// agree, which the GPU crate's tests check.
pub trait FluidSolver {
    /// Advance `fluid` by `dt` seconds with `boundary` as the bodies, and
    /// return the impulse the fluid exerted on each of the `body_count`
    /// bodies (see [`Fluid::step`]).
    fn step(
        &mut self,
        fluid: &mut Fluid,
        boundary: &Boundary,
        gravity: Vec2,
        dt: f32,
        body_count: usize,
    ) -> Vec<BodyImpulse>;

    /// Short name for display ("CPU", "GPU").
    fn name(&self) -> &str;
}

/// The CPU solver: [`Fluid::step`], multithreaded with the `parallel`
/// feature.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuSolver;

impl FluidSolver for CpuSolver {
    fn step(
        &mut self,
        fluid: &mut Fluid,
        boundary: &Boundary,
        gravity: Vec2,
        dt: f32,
        body_count: usize,
    ) -> Vec<BodyImpulse> {
        fluid.step(boundary, gravity, dt, body_count)
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

/// Diagnostics from the most recent step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FluidStats {
    /// Substeps taken in the last world step.
    pub substeps: u32,
    /// Largest (ρ − ρ₀)/ρ₀ in the last substep. A few percent is normal.
    pub max_compression: f32,
    /// Mean compression over particles with ρ > ρ₀.
    pub mean_compression: f32,
}

/// One particle's state for the force pass.
#[derive(Debug, Clone, Copy, Default)]
struct Packed {
    x: Vec2,
    v: Vec2,
    rho: f32,
    /// p / ρ², the factor the pressure force uses.
    term: f32,
}

/// Scratch arrays reused across substeps.
#[derive(Debug, Default)]
struct Scratch {
    densities: Vec<f32>,
    pressures: Vec<f32>,
    /// What the force pass reads about each neighbor, side by side so one
    /// lookup fetches it all.
    packed: Vec<Packed>,
    accelerations: Vec<Vec2>,
    fluid_neighbors: Vec<Vec<u32>>,
    boundary_neighbors: Vec<Vec<u32>>,
    grid: NeighborGrid,
}

impl Clone for Scratch {
    /// Scratch space is rebuilt every substep, so a copy of a fluid (an undo
    /// snapshot, the state saved when Play is pressed) starts with none
    /// rather than copying thousands of neighbor lists.
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// The fluid: particle positions and velocities, stored as separate arrays
/// (struct of arrays). Each pass reads only the arrays it needs, which is
/// cache-friendly and maps directly to GPU buffers later (M5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Fluid {
    #[serde(default)]
    pub params: FluidParams,
    #[serde(default)]
    pub positions: Vec<Vec2>,
    /// Omitted from scene files when every particle is at rest.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub velocities: Vec<Vec2>,
    #[serde(skip)]
    scratch: Scratch,
    #[serde(skip)]
    pub stats: FluidStats,
}

fn all_zero(v: &[Vec2]) -> bool {
    v.iter().all(|&x| x == Vec2::ZERO)
}

impl PartialEq for Fluid {
    /// Two fluids are equal if their saved state is: scratch space and
    /// statistics don't count.
    fn eq(&self, other: &Self) -> bool {
        let vel = |f: &Fluid, i: usize| f.velocities.get(i).copied().unwrap_or(Vec2::ZERO);
        self.params == other.params
            && self.positions == other.positions
            && (0..self.positions.len()).all(|i| vel(self, i) == vel(other, i))
    }
}

impl Fluid {
    pub fn new(params: FluidParams) -> Self {
        Self {
            params,
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// True when there's nothing worth saving: no particles and default
    /// settings. Scene files leave the fluid out in that case.
    pub fn is_unused(&self) -> bool {
        self.is_empty() && self.params == FluidParams::default()
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.stats = FluidStats::default();
    }

    /// Scene files may omit velocities; treat missing ones as zero.
    pub fn fill_missing_velocities(&mut self) {
        self.velocities.resize(self.positions.len(), Vec2::ZERO);
    }

    /// Density of each particle from the last substep, kg/m². Empty before
    /// the first step.
    pub fn densities(&self) -> &[f32] {
        &self.scratch.densities
    }

    /// Pressure of each particle from the last substep, N/m.
    pub fn pressures(&self) -> &[f32] {
        &self.scratch.pressures
    }

    /// For solvers outside this crate: store the densities and pressures a
    /// step computed (so coloring and diagnostics work) and update the
    /// statistics.
    pub fn record_external_step(
        &mut self,
        densities: Vec<f32>,
        pressures: Vec<f32>,
        substeps: u32,
    ) {
        self.scratch.densities = densities;
        self.scratch.pressures = pressures;
        self.update_stats(substeps);
    }

    pub fn validate(&self) -> Result<(), String> {
        self.params.validate()?;
        if !self.velocities.is_empty() && self.velocities.len() != self.positions.len() {
            return Err(format!(
                "fluid has {} positions but {} velocities",
                self.positions.len(),
                self.velocities.len()
            ));
        }
        if !self
            .positions
            .iter()
            .chain(&self.velocities)
            .all(|p| p.is_finite())
        {
            return Err("fluid positions and velocities must be finite".into());
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Adding and removing particles

    /// Lattice point (i, j): particles always sit on one global grid offset
    /// by half a spacing, so repeated fills and brush strokes line up
    /// instead of overlapping.
    fn lattice(&self, i: i32, j: i32) -> Vec2 {
        let d = self.params.particle_spacing;
        Vec2::new((i as f32 + 0.5) * d, (j as f32 + 0.5) * d)
    }

    /// Add particles at every free lattice point for which `inside` is true,
    /// within the lattice index ranges given. Returns how many were added.
    fn fill_where(
        &mut self,
        i_range: std::ops::RangeInclusive<i32>,
        j_range: std::ops::RangeInclusive<i32>,
        bodies: &[RigidBody],
        inside: impl Fn(Vec2) -> bool,
    ) -> usize {
        self.fill_missing_velocities();
        let d = self.params.particle_spacing;
        let existing = NeighborGrid::build(&self.positions, d);
        let before = self.positions.len();
        for j in j_range {
            for i in i_range.clone() {
                let p = self.lattice(i, j);
                if !inside(p) {
                    continue;
                }
                let mut crowded = false;
                existing.for_each_candidate(p, |k| {
                    crowded |= (self.positions[k] - p).length_squared() < (0.9 * d) * (0.9 * d);
                });
                // Keep clear of bodies' boundary layers: a fluid particle
                // much closer counts as heavily compressed and would be
                // thrown off at startup.
                let blocked = bodies.iter().any(|b| {
                    let (surface, _, inside_body) = b.closest_surface_point(p);
                    inside_body || (surface - p).length() < SURFACE_GAP_FACTOR * d
                });
                if !crowded && !blocked {
                    self.positions.push(p);
                    self.velocities.push(Vec2::ZERO);
                }
            }
        }
        self.positions.len() - before
    }

    /// Fill the rectangle between `a` and `b` with fluid at rest, skipping
    /// space taken by bodies or existing fluid.
    pub fn fill_rect(&mut self, a: Vec2, b: Vec2, bodies: &[RigidBody]) -> usize {
        let (min, max) = (a.min(b), a.max(b));
        let d = self.params.particle_spacing;
        let i0 = (min.x / d - 0.5).ceil() as i32;
        let i1 = (max.x / d - 0.5).floor() as i32;
        let j0 = (min.y / d - 0.5).ceil() as i32;
        let j1 = (max.y / d - 0.5).floor() as i32;
        self.fill_where(i0..=i1, j0..=j1, bodies, |_| true)
    }

    /// Fill a disc with fluid at rest (the brush tool).
    pub fn fill_disc(&mut self, center: Vec2, radius: f32, bodies: &[RigidBody]) -> usize {
        let d = self.params.particle_spacing;
        let i0 = ((center.x - radius) / d - 0.5).ceil() as i32;
        let i1 = ((center.x + radius) / d - 0.5).floor() as i32;
        let j0 = ((center.y - radius) / d - 0.5).ceil() as i32;
        let j1 = ((center.y + radius) / d - 0.5).floor() as i32;
        self.fill_where(i0..=i1, j0..=j1, bodies, |p| {
            (p - center).length_squared() <= radius * radius
        })
    }

    /// Remove every particle within `radius` of `center` (the eraser).
    pub fn remove_disc(&mut self, center: Vec2, radius: f32) -> usize {
        self.retain(|p| (p - center).length_squared() > radius * radius)
    }

    /// Keep only particles whose position satisfies `keep`.
    fn retain(&mut self, keep: impl Fn(Vec2) -> bool) -> usize {
        self.fill_missing_velocities();
        let before = self.positions.len();
        let mut w = 0;
        for r in 0..before {
            if keep(self.positions[r]) {
                self.positions[w] = self.positions[r];
                self.velocities[w] = self.velocities[r];
                w += 1;
            }
        }
        self.positions.truncate(w);
        self.velocities.truncate(w);
        before - w
    }

    // ---------------------------------------------------------------------
    // Simulation

    /// Advance the fluid by `dt` seconds, taking as many substeps as the CFL
    /// condition needs. `boundary` describes the bodies at the start of the
    /// step.
    ///
    /// Returns the impulse the fluid exerted on each body during the step
    /// (indexed like the bodies the boundary was built from, `body_count`
    /// long). This is the reaction to the boundary forces the fluid felt, so
    /// fluid and bodies exchange momentum exactly (Newton's third law).
    pub fn step(
        &mut self,
        boundary: &Boundary,
        gravity: Vec2,
        dt: f32,
        body_count: usize,
    ) -> Vec<BodyImpulse> {
        let mut impulses = vec![BodyImpulse::default(); body_count];
        if self.is_empty() {
            self.stats = FluidStats::default();
            return impulses;
        }
        self.fill_missing_velocities();
        self.sort_spatially();
        let c = self.params.speed_of_sound;
        let d = self.params.particle_spacing;
        let max_speed = MAX_SPEED_FRACTION * c;

        let mut elapsed = 0.0;
        let mut substeps = 0;
        let mut v_max = max_reduce(iter!(self.velocities).map(|v| v.length_squared())).sqrt();
        // The 1e-6 guard stops a sliver of round-off from costing a whole
        // extra substep.
        while dt - elapsed > 1e-6 {
            let h = (CFL * d / (c + v_max)).min(dt - elapsed);
            v_max = self.substep(boundary, gravity, h, max_speed, &mut impulses);
            elapsed += h;
            substeps += 1;
        }
        self.update_stats(substeps);
        impulses
    }

    /// Compression statistics from the last substep's densities.
    fn update_stats(&mut self, substeps: u32) {
        let rho0 = self.params.rest_density;
        let (mut max_c, mut sum_c, mut count) = (0.0_f32, 0.0_f32, 0_u32);
        for &rho in &self.scratch.densities {
            let compression = rho / rho0 - 1.0;
            if compression > 0.0 {
                max_c = max_c.max(compression);
                sum_c += compression;
                count += 1;
            }
        }
        self.stats = FluidStats {
            substeps,
            max_compression: max_c,
            mean_compression: if count > 0 { sum_c / count as f32 } else { 0.0 },
        };
    }

    /// Reorder particles so that ones close together in space are close
    /// together in memory (sorted by grid row, then column). Neighbor loops
    /// then touch a few contiguous chunks of memory instead of jumping all
    /// over the arrays. It costs little, doesn't change the physics, and
    /// matters once the particle count outgrows the CPU cache.
    pub fn sort_spatially(&mut self) {
        let inv = 1.0 / self.params.support();
        let key = |p: Vec2| ((p.y * inv).floor() as i32, (p.x * inv).floor() as i32);
        let mut order: Vec<u32> = (0..self.positions.len() as u32).collect();
        order.sort_unstable_by_key(|&i| key(self.positions[i as usize]));
        self.positions = order.iter().map(|&i| self.positions[i as usize]).collect();
        self.velocities = order.iter().map(|&i| self.velocities[i as usize]).collect();
    }

    /// One substep. Three parallel passes over the particles, so threads
    /// split up and rejoin as few times as possible: (1) neighbors, density,
    /// and pressure together; (2) accelerations; (3) integration, which also
    /// finds the fastest particle for the next substep's size. Returns that
    /// speed.
    fn substep(
        &mut self,
        boundary: &Boundary,
        gravity: Vec2,
        dt: f32,
        max_speed: f32,
        impulses: &mut [BodyImpulse],
    ) -> f32 {
        let params = self.params;
        let kernel = params.kernel();
        let h = params.support();
        let h2 = h * h;
        let mass = params.particle_mass();
        let rho0 = params.rest_density;
        let stiffness = params.stiffness();
        let c = params.speed_of_sound;
        // Monaghan viscosity uses the smoothing length (half our support).
        let h_visc = 0.5 * h;
        let eps = 0.01 * h_visc * h_visc;
        let n = self.positions.len();

        let Scratch {
            densities,
            pressures,
            packed,
            accelerations,
            fluid_neighbors,
            boundary_neighbors,
            grid,
        } = &mut self.scratch;
        let positions = &self.positions;
        let velocities = &self.velocities;

        // --- 1. Neighbors within the kernel support, density (the particle
        // itself counts too), and pressure from the Tait equation, clamped at
        // zero. All per particle, so one pass.
        grid.rebuild_fine(positions, h);
        let grid = &*grid;
        fluid_neighbors.resize_with(n, Vec::new);
        boundary_neighbors.resize_with(n, Vec::new);
        densities.resize(n, 0.0);
        pressures.resize(n, 0.0);
        packed.resize(n, Packed::default());
        let self_weight = mass * kernel.w(0.0);
        iter_mut!(fluid_neighbors)
            .zip(iter_mut!(boundary_neighbors))
            .zip(iter_mut!(densities))
            .zip(iter_mut!(pressures))
            .zip(iter_mut!(packed))
            .enumerate()
            .for_each(|(i, ((((fluid, walls), rho), p), packed))| {
                fluid.clear();
                walls.clear();
                let xi = positions[i];
                let mut sum = self_weight;
                grid.for_each_near(xi, |j, xj| {
                    let r2 = (xi - xj).length_squared();
                    if r2 < h2 && j != i {
                        fluid.push(j as u32);
                        sum += mass * kernel.w(r2.sqrt());
                    }
                });
                boundary.grid.for_each_near(xi, |b, xb| {
                    let r2 = (xi - xb).length_squared();
                    if r2 < h2 {
                        walls.push(b as u32);
                        sum += boundary.psi[b] * kernel.w(r2.sqrt());
                    }
                });
                *rho = sum;
                *p = (stiffness * ((sum / rho0).powi(TAIT_GAMMA) - 1.0)).max(0.0);
                *packed = Packed {
                    x: xi,
                    v: velocities[i],
                    rho: sum,
                    term: *p / (sum * sum),
                };
            });
        let fluid_neighbors = &*fluid_neighbors;
        let boundary_neighbors = &*boundary_neighbors;
        let packed = &*packed;

        // --- 2. Accelerations.
        accelerations.resize(n, Vec2::ZERO);
        iter_mut!(accelerations).enumerate().for_each(|(i, acc)| {
            let Packed {
                x: xi,
                v: vi,
                rho: rho_i,
                term: pi_term,
            } = packed[i];
            let mut a = gravity;

            for &j in &fluid_neighbors[i] {
                let pj = packed[j as usize];
                let r = xi - pj.x;
                let dist = r.length();
                let grad = kernel.grad(r, dist);
                // Symmetric pressure force: conserves momentum exactly.
                a -= mass * (pi_term + pj.term) * grad;
                // Monaghan artificial viscosity, only while approaching.
                let vr = (vi - pj.v).dot(r);
                if vr < 0.0 {
                    let nu = 2.0 * params.viscosity * h_visc * c / (rho_i + pj.rho);
                    let pi_visc = -nu * vr / (dist * dist + eps);
                    a -= mass * pi_visc * grad;
                }
            }

            for &b in &boundary_neighbors[i] {
                let b = b as usize;
                let r = xi - boundary.positions[b];
                let dist = r.length();
                let grad = kernel.grad(r, dist);
                // Akinci boundary pressure: the boundary "mirrors" this
                // particle's own pressure.
                a -= boundary.psi[b] * pi_term * grad;
                let vr = (vi - boundary.velocities[b]).dot(r);
                if vr < 0.0 {
                    let nu = params.wall_viscosity * h_visc * c / rho_i;
                    let pi_visc = -nu * vr / (dist * dist + eps);
                    a -= boundary.psi[b] * pi_visc * grad;
                }
            }
            *acc = a;
        });

        // --- 2b. Reaction on bodies: for every fluid-boundary pair, the
        // body gets exactly the opposite of the force the fluid particle got
        // from it above. Only particles near a body have boundary neighbors,
        // so this sequential pass is cheap.
        if !boundary.positions.is_empty() {
            for (i, list) in boundary_neighbors.iter().enumerate() {
                if list.is_empty() {
                    continue;
                }
                let Packed {
                    x: xi,
                    v: vi,
                    rho: rho_i,
                    term: pi_term,
                } = packed[i];
                for &b in list {
                    let b = b as usize;
                    let xb = boundary.positions[b];
                    let r = xi - xb;
                    let dist = r.length();
                    let mut coefficient = pi_term;
                    let vr = (vi - boundary.velocities[b]).dot(r);
                    if vr < 0.0 {
                        let nu = params.wall_viscosity * h_visc * c / rho_i;
                        coefficient += -nu * vr / (dist * dist + eps);
                    }
                    let force = mass * boundary.psi[b] * coefficient * kernel.grad(r, dist);
                    let body = boundary.body[b] as usize;
                    if let Some(impulse) = impulses.get_mut(body) {
                        impulse.linear += force * dt;
                        impulse.angular += (xb - boundary.body_centers[body]).perp_dot(force) * dt;
                    }
                }
            }
        }

        // --- 3. Integrate (semi-implicit Euler), noting the top speed.
        let accelerations = &self.scratch.accelerations;
        let top = max_reduce(
            iter_mut!(self.velocities)
                .zip(iter_mut!(self.positions))
                .zip(iter!(accelerations))
                .map(|((v, x), a)| {
                    *v = (*v + *a * dt).clamp_length_max(max_speed);
                    *x += *v * dt;
                    v.length_squared()
                }),
        );
        top.sqrt()
    }

    /// Safety net: push any particle that ended up inside a body back to its
    /// surface and remove its velocity into the body.
    ///
    /// Boundary pressure normally prevents this. It catches the rare
    /// particle that tunnels through at very high speed, or one a body moves
    /// on top of.
    pub fn project_out_of_bodies(&mut self, bodies: &[RigidBody]) {
        if self.is_empty() || bodies.is_empty() {
            return;
        }
        self.fill_missing_velocities();
        // Put it back where boundary pressure is moderate (see
        // BOUNDARY_GAP_FACTOR). Closer, and it would be launched.
        let margin = SURFACE_GAP_FACTOR * self.params.particle_spacing;
        let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
        iter_mut!(self.positions)
            .zip(iter_mut!(self.velocities))
            .for_each(|(x, v)| {
                for (body, bb) in bodies.iter().zip(&boxes) {
                    if x.cmplt(bb.min).any() || x.cmpgt(bb.max).any() {
                        continue;
                    }
                    let (surface, normal, inside) = body.closest_surface_point(*x);
                    if inside {
                        *x = surface + normal * margin;
                        let relative = *v - body.velocity_at(surface);
                        let vn = relative.dot(normal);
                        if vn < 0.0 {
                            *v -= vn * normal;
                        }
                    }
                }
            });
    }

    /// Delete particles that fell out of the world. Returns how many.
    pub fn remove_escaped(&mut self) -> usize {
        self.retain(|p| p.is_finite() && p.y > KILL_DEPTH)
    }

    /// Total kinetic energy of the fluid, J.
    pub fn kinetic_energy(&self) -> f32 {
        let m = self.params.particle_mass();
        self.velocities
            .iter()
            .map(|v| 0.5 * m * v.length_squared())
            .sum()
    }

    /// Total momentum of the fluid, kg·m/s.
    pub fn momentum(&self) -> Vec2 {
        let m = self.params.particle_mass();
        self.velocities.iter().map(|&v| m * v).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn lattice_interior_starts_at_rest_density() {
        let mut f = Fluid::default();
        f.fill_rect(Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), &[]);
        // One tiny step with no gravity just to compute densities.
        f.step(&Boundary::default(), Vec2::ZERO, 1e-5, 0);
        let center = f
            .positions
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.length().total_cmp(&b.1.length()))
            .unwrap()
            .0;
        let rho = f.densities()[center];
        assert!(
            (rho / f.params.rest_density - 1.0).abs() < 1e-4,
            "ρ = {rho}"
        );
    }

    #[test]
    fn fills_align_and_never_overlap() {
        let mut f = Fluid::default();
        let a = f.fill_rect(Vec2::ZERO, Vec2::new(1.0, 1.0), &[]);
        assert_eq!(a, 400); // 20 × 20 at 0.05 m
        // Filling the same area again adds nothing.
        assert_eq!(f.fill_rect(Vec2::ZERO, Vec2::new(1.0, 1.0), &[]), 0);
        // A brush stroke over part of it adds only the uncovered part.
        let added = f.fill_disc(Vec2::new(1.0, 0.5), 0.3, &[]);
        assert!(added > 0 && added < 150, "added {added}");
    }

    #[test]
    fn fill_skips_bodies() {
        let mut f = Fluid::default();
        let block = RigidBody::new(Shape::rect(0.5, 0.5), Vec2::new(0.5, 0.5));
        f.fill_rect(
            Vec2::ZERO,
            Vec2::new(1.0, 1.0),
            std::slice::from_ref(&block),
        );
        // The block covers 10 × 10 of the 20 × 20 lattice points.
        assert!(f.len() <= 400 - 100, "{} particles", f.len());
        assert!(f.positions.iter().all(|&p| !block.contains_point(p)));
    }

    #[test]
    fn eraser_removes_only_inside_radius() {
        let mut f = Fluid::default();
        f.fill_rect(Vec2::ZERO, Vec2::new(1.0, 1.0), &[]);
        let removed = f.remove_disc(Vec2::new(0.5, 0.5), 0.2);
        assert!(removed > 40 && removed < 60, "removed {removed}"); // π·0.2²/0.05² ≈ 50
        assert!(
            f.positions
                .iter()
                .all(|p| (*p - Vec2::new(0.5, 0.5)).length() > 0.2)
        );
        assert_eq!(f.velocities.len(), f.positions.len());
    }

    #[test]
    fn projection_pushes_particles_out() {
        let mut f = Fluid {
            positions: vec![Vec2::new(0.1, 0.45)],
            velocities: vec![Vec2::new(0.0, -2.0)],
            ..Fluid::default()
        };
        let block = RigidBody::new_static(Shape::rect(1.0, 1.0), Vec2::ZERO);
        f.project_out_of_bodies(std::slice::from_ref(&block));
        assert!(!block.contains_point(f.positions[0]));
        assert!(f.velocities[0].y >= 0.0);
    }
}
