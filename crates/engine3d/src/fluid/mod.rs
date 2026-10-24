//! Particle-based fluid in 3D: weakly compressible SPH (WCSPH).
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
//! This is the 3D version of the `engine` crate's 2D solver: the same
//! algorithm and constants, with a 3D kernel, grid, and boundary.
//!
//! # Units
//!
//! Real 3D SI units: density kg/m³ (water 1000), pressure Pa.

pub mod boundary;
pub mod grid;
pub mod kernel;

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use boundary::Boundary;
use glam::Vec3;
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
/// Denser than the fluid so particles can't squeeze through gaps. 2D uses
/// 0.5; in 3D that's 4× the fluid's surface density and the boundary
/// dominated the cost. Measured with 0.4 m cubes floating in a tank:
/// 0.5 gave drafts within 1.1 cm of Archimedes; 0.7 within 0.9 cm at 60% of
/// the cost; 1.0 was cheaper still, but a sunk cube rested on a 5.6 cm film
/// of trapped water.
pub const BOUNDARY_SPACING_FACTOR: f32 = 0.7;
/// How far inside each body's surface its boundary particles sit, as a
/// fraction of particle spacing, so the fluid's effective boundary is the
/// real surface. 0.55 was measured in 2D; in 3D it leaves floating drafts
/// within 1 cm (0.1 spacing) of Archimedes, so it carries over. See
/// [`boundary`].
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
    /// detail and many more particles (8× as many per halving).
    pub particle_spacing: f32,
    /// ρ₀, kg/m³. Water is 1000.
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
            // Coarser than 2D's 0.05 m: in 3D, halving the spacing means 8×
            // the particles. A 1 m cube of water is 1,000 particles.
            particle_spacing: 0.1,
            rest_density: 1000.0,
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
    /// Kernel support radius h = 2 × spacing (about 30 neighbors in 3D).
    pub fn support(&self) -> f32 {
        2.0 * self.particle_spacing
    }

    /// The fluid's view of `bodies`: boundary particles with weights,
    /// covering the parts of their surfaces within `region` (normally
    /// [`Fluid::boundary_region`]).
    pub fn boundary(&self, bodies: &[RigidBody], region: (Vec3, Vec3)) -> Boundary {
        Boundary::build(
            bodies,
            self.particle_spacing * BOUNDARY_SPACING_FACTOR,
            self.particle_spacing * BOUNDARY_INSET_FACTOR,
            &self.kernel(),
            self.rest_density,
            region,
        )
    }

    pub fn kernel(&self) -> CubicSpline {
        CubicSpline::new(self.support())
    }

    /// Mass per particle, kg. Chosen so a particle in the middle of a
    /// perfect cubic lattice at `particle_spacing` has exactly `rest_density`.
    /// (Using ρ₀ d³ instead is off by a percent or two because the kernel sum
    /// over a lattice isn't exactly the integral, and that error shows up as
    /// a startup jolt.)
    pub fn particle_mass(&self) -> f32 {
        let d = self.particle_spacing;
        let k = self.kernel();
        let mut sum = 0.0;
        for i in -3_i32..=3 {
            for j in -3_i32..=3 {
                for l in -3_i32..=3 {
                    sum += k.w(d * ((i * i + j * j + l * l) as f32).sqrt());
                }
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

    /// Tait stiffness B = ρ₀ c² / γ, Pa.
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
    pub linear: Vec3,
    pub angular: Vec3,
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
        gravity: Vec3,
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
        gravity: Vec3,
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

/// Scratch arrays reused across substeps.
#[derive(Debug, Default)]
struct Scratch {
    densities: Vec<f32>,
    pressures: Vec<f32>,
    accelerations: Vec<Vec3>,
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
    pub positions: Vec<Vec3>,
    /// Omitted from scene files when every particle is at rest.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub velocities: Vec<Vec3>,
    #[serde(skip)]
    scratch: Scratch,
    #[serde(skip)]
    pub stats: FluidStats,
}

fn all_zero(v: &[Vec3]) -> bool {
    v.iter().all(|&x| x == Vec3::ZERO)
}

impl PartialEq for Fluid {
    /// Two fluids are equal if their saved state is: scratch space and
    /// statistics don't count.
    fn eq(&self, other: &Self) -> bool {
        let vel = |f: &Fluid, i: usize| f.velocities.get(i).copied().unwrap_or(Vec3::ZERO);
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
        self.velocities.resize(self.positions.len(), Vec3::ZERO);
    }

    /// Density of each particle from the last substep, kg/m³. Empty before
    /// the first step.
    pub fn densities(&self) -> &[f32] {
        &self.scratch.densities
    }

    /// Pressure of each particle from the last substep, Pa.
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
        let rho0 = self.params.rest_density;
        let (mut max_c, mut sum_c, mut count) = (0.0_f32, 0.0_f32, 0_u32);
        for &rho in &densities {
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
        self.scratch.densities = densities;
        self.scratch.pressures = pressures;
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

    /// Lattice point (i, j, k): particles always sit on one global grid offset
    /// by half a spacing, so repeated fills line up instead of overlapping.
    fn lattice(&self, i: i32, j: i32, k: i32) -> Vec3 {
        let d = self.params.particle_spacing;
        (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * d
    }

    /// Lattice index range covering [lo, hi] along one axis.
    fn index_range(&self, lo: f32, hi: f32) -> std::ops::RangeInclusive<i32> {
        let d = self.params.particle_spacing;
        (lo / d - 0.5).ceil() as i32..=(hi / d - 0.5).floor() as i32
    }

    /// Add particles at every free lattice point between `min` and `max` for
    /// which `inside` is true. Returns how many were added.
    fn fill_where(
        &mut self,
        min: Vec3,
        max: Vec3,
        bodies: &[RigidBody],
        inside: impl Fn(Vec3) -> bool,
    ) -> usize {
        self.fill_missing_velocities();
        let d = self.params.particle_spacing;
        let existing = NeighborGrid::build(&self.positions, d);
        let before = self.positions.len();
        let boxes: Vec<Aabb> = bodies.iter().map(Aabb::of).collect();
        let gap = SURFACE_GAP_FACTOR * d;
        for k in self.index_range(min.z, max.z) {
            for j in self.index_range(min.y, max.y) {
                for i in self.index_range(min.x, max.x) {
                    let p = self.lattice(i, j, k);
                    if !inside(p) {
                        continue;
                    }
                    let mut crowded = false;
                    existing.for_each_candidate(p, |n| {
                        crowded |= (self.positions[n] - p).length_squared() < (0.9 * d) * (0.9 * d);
                    });
                    // Keep clear of bodies' boundary layers (see
                    // BOUNDARY_GAP_FACTOR).
                    let blocked = bodies.iter().zip(&boxes).any(|(b, bb)| {
                        if p.cmplt(bb.min - gap).any() || p.cmpgt(bb.max + gap).any() {
                            return false;
                        }
                        let (surface, _, inside_body) = b.closest_surface_point(p);
                        inside_body || (surface - p).length() < gap
                    });
                    if !crowded && !blocked {
                        self.positions.push(p);
                        self.velocities.push(Vec3::ZERO);
                    }
                }
            }
        }
        self.positions.len() - before
    }

    /// Fill the box between corners `a` and `b` with fluid at rest, skipping
    /// space taken by bodies or existing fluid.
    pub fn fill_box(&mut self, a: Vec3, b: Vec3, bodies: &[RigidBody]) -> usize {
        self.fill_where(a.min(b), a.max(b), bodies, |_| true)
    }

    /// Fill a ball with fluid at rest.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f32, bodies: &[RigidBody]) -> usize {
        self.fill_where(center - radius, center + radius, bodies, |p| {
            (p - center).length_squared() <= radius * radius
        })
    }

    /// Remove every particle within `radius` of `center`.
    pub fn remove_sphere(&mut self, center: Vec3, radius: f32) -> usize {
        self.retain(|p| (p - center).length_squared() > radius * radius)
    }

    /// Keep only particles whose position satisfies `keep`.
    fn retain(&mut self, keep: impl Fn(Vec3) -> bool) -> usize {
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

    /// The region whose body surfaces the fluid can touch this step: the
    /// particles' bounding box, grown by two kernel supports (one for the
    /// kernel's reach and one for motion and for the boundary weights near
    /// the region's edge to settle).
    pub fn boundary_region(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for &p in &self.positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let margin = Vec3::splat(2.0 * self.params.support());
        (lo - margin, hi + margin)
    }

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
        gravity: Vec3,
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
        // The 1e-6 guard stops a sliver of round-off from costing a whole
        // extra substep.
        while dt - elapsed > 1e-6 {
            let v_max = self
                .velocities
                .iter()
                .map(|v| v.length_squared())
                .fold(0.0_f32, f32::max)
                .sqrt();
            let h = (CFL * d / (c + v_max)).min(dt - elapsed);
            self.substep(boundary, gravity, h, max_speed, &mut impulses);
            elapsed += h;
            substeps += 1;
        }
        self.stats.substeps = substeps;
        impulses
    }

    /// Reorder particles so that ones close together in space are close
    /// together in memory (sorted by grid cell: z, then y, then x). Neighbor loops
    /// then touch a few contiguous chunks of memory instead of jumping all
    /// over the arrays. It costs little, doesn't change the physics, and
    /// matters once the particle count outgrows the CPU cache.
    pub fn sort_spatially(&mut self) {
        let inv = 1.0 / self.params.support();
        let key = |p: Vec3| {
            let c = (p * inv).floor();
            (c.z as i32, c.y as i32, c.x as i32)
        };
        let mut order: Vec<u32> = (0..self.positions.len() as u32).collect();
        order.sort_unstable_by_key(|&i| key(self.positions[i as usize]));
        self.positions = order.iter().map(|&i| self.positions[i as usize]).collect();
        self.velocities = order.iter().map(|&i| self.velocities[i as usize]).collect();
    }

    fn substep(
        &mut self,
        boundary: &Boundary,
        gravity: Vec3,
        dt: f32,
        max_speed: f32,
        impulses: &mut [BodyImpulse],
    ) {
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
            accelerations,
            fluid_neighbors,
            boundary_neighbors,
            grid,
        } = &mut self.scratch;
        let positions = &self.positions;
        let velocities = &self.velocities;

        // --- 1. Neighbors within the kernel support.
        grid.rebuild(positions, h);
        let grid = &*grid;
        fluid_neighbors.resize_with(n, Vec::new);
        boundary_neighbors.resize_with(n, Vec::new);
        iter_mut!(fluid_neighbors)
            .enumerate()
            .for_each(|(i, list)| {
                list.clear();
                let xi = positions[i];
                grid.for_each_candidate(xi, |j| {
                    if j != i && (xi - positions[j]).length_squared() < h2 {
                        list.push(j as u32);
                    }
                });
            });
        iter_mut!(boundary_neighbors)
            .enumerate()
            .for_each(|(i, list)| {
                list.clear();
                let xi = positions[i];
                boundary.grid.for_each_candidate(xi, |b| {
                    if (xi - boundary.positions[b]).length_squared() < h2 {
                        list.push(b as u32);
                    }
                });
            });
        let fluid_neighbors = &*fluid_neighbors;
        let boundary_neighbors = &*boundary_neighbors;

        // --- 2. Density (the particle itself counts too).
        densities.resize(n, 0.0);
        let self_weight = mass * kernel.w(0.0);
        iter_mut!(densities).enumerate().for_each(|(i, rho)| {
            let xi = positions[i];
            let mut sum = self_weight;
            for &j in &fluid_neighbors[i] {
                sum += mass * kernel.w((xi - positions[j as usize]).length());
            }
            for &b in &boundary_neighbors[i] {
                let b = b as usize;
                sum += boundary.psi[b] * kernel.w((xi - boundary.positions[b]).length());
            }
            *rho = sum;
        });
        let densities = &*densities;

        // --- 3. Pressure (Tait), clamped at zero.
        pressures.resize(n, 0.0);
        iter_mut!(pressures).enumerate().for_each(|(i, p)| {
            *p = (stiffness * ((densities[i] / rho0).powi(TAIT_GAMMA) - 1.0)).max(0.0);
        });
        let pressures = &*pressures;

        // --- 4. Accelerations.
        accelerations.resize(n, Vec3::ZERO);
        iter_mut!(accelerations).enumerate().for_each(|(i, acc)| {
            let (xi, vi) = (positions[i], velocities[i]);
            let (rho_i, p_i) = (densities[i], pressures[i]);
            let pi_term = p_i / (rho_i * rho_i);
            let mut a = gravity;

            for &j in &fluid_neighbors[i] {
                let j = j as usize;
                let r = xi - positions[j];
                let dist = r.length();
                let grad = kernel.grad(r, dist);
                let pj_term = pressures[j] / (densities[j] * densities[j]);
                // Symmetric pressure force: conserves momentum exactly.
                a -= mass * (pi_term + pj_term) * grad;
                // Monaghan artificial viscosity, only while approaching.
                let vr = (vi - velocities[j]).dot(r);
                if vr < 0.0 {
                    let nu = 2.0 * params.viscosity * h_visc * c / (rho_i + densities[j]);
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

        // --- 4b. Reaction on bodies: for every fluid-boundary pair, the
        // body gets exactly the opposite of the force the fluid particle got
        // from it above. Only particles near a body have boundary neighbors,
        // so this sequential pass is cheap.
        for (i, list) in boundary_neighbors.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            let (xi, vi, rho_i) = (positions[i], velocities[i], densities[i]);
            let pi_term = pressures[i] / (rho_i * rho_i);
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
                    impulse.angular += (xb - boundary.body_centers[body]).cross(force) * dt;
                }
            }
        }

        // Diagnostics.
        let (mut max_c, mut sum_c, mut count) = (0.0_f32, 0.0_f32, 0_u32);
        for &rho in densities {
            let compression = rho / rho0 - 1.0;
            if compression > 0.0 {
                max_c = max_c.max(compression);
                sum_c += compression;
                count += 1;
            }
        }
        self.stats.max_compression = max_c;
        self.stats.mean_compression = if count > 0 { sum_c / count as f32 } else { 0.0 };

        // --- 5. Integrate (semi-implicit Euler).
        let accelerations = &self.scratch.accelerations;
        iter_mut!(self.velocities)
            .zip(iter_mut!(self.positions))
            .zip(iter!(accelerations))
            .for_each(|((v, x), a)| {
                *v = (*v + *a * dt).clamp_length_max(max_speed);
                *x += *v * dt;
            });
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
    pub fn momentum(&self) -> Vec3 {
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
        f.fill_box(Vec3::splat(-0.5), Vec3::splat(0.5), &[]);
        assert_eq!(f.len(), 1000);
        f.step(&Boundary::default(), Vec3::ZERO, 1e-5, 0);
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
        assert_eq!(f.fill_box(Vec3::ZERO, Vec3::ONE, &[]), 1000);
        assert_eq!(f.fill_box(Vec3::ZERO, Vec3::ONE, &[]), 0);
        let added = f.fill_sphere(Vec3::new(1.0, 0.5, 0.5), 0.3, &[]);
        // Half the ball sticks out of the cube: about ½ · 4/3 π 0.3³ / 0.1³ = 57.
        assert!((30..90).contains(&added), "added {added}");
    }

    #[test]
    fn fill_skips_bodies() {
        let mut f = Fluid::default();
        let block = RigidBody::new(Shape::cube(0.5), Vec3::splat(0.5));
        f.fill_box(Vec3::ZERO, Vec3::ONE, std::slice::from_ref(&block));
        // The block covers 5³ = 125 of the 1000 lattice points (plus a gap).
        assert!(f.len() <= 1000 - 125, "{} particles", f.len());
        assert!(f.positions.iter().all(|&p| !block.contains_point(p)));
    }

    #[test]
    fn remove_sphere_removes_only_inside() {
        let mut f = Fluid::default();
        f.fill_box(Vec3::ZERO, Vec3::ONE, &[]);
        let c = Vec3::splat(0.5);
        let removed = f.remove_sphere(c, 0.25);
        // 4/3 π 0.25³ / 0.1³ ≈ 65
        assert!((50..80).contains(&removed), "removed {removed}");
        assert!(f.positions.iter().all(|p| (*p - c).length() > 0.25));
        assert_eq!(f.velocities.len(), f.positions.len());
    }

    #[test]
    fn projection_pushes_particles_out() {
        let mut f = Fluid {
            positions: vec![Vec3::new(0.1, 0.45, 0.0)],
            velocities: vec![Vec3::new(0.0, -2.0, 0.0)],
            ..Fluid::default()
        };
        let block = RigidBody::new_static(Shape::cube(1.0), Vec3::ZERO);
        f.project_out_of_bodies(std::slice::from_ref(&block));
        assert!(!block.contains_point(f.positions[0]));
        assert!(f.velocities[0].y >= 0.0);
    }
}
