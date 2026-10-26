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
use kernel::{CubicSpline, cohesion_normalization, cohesion_shape};
pub use materials::{Liquid, SEAWATER_SALINITY};
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
    /// Diffusivity of salt in water, m²/s. The real value
    /// ([`materials::SALT_DIFFUSIVITY`]) is so small that salt barely
    /// diffuses on simulation time scales; tests raise it to check the
    /// diffusion itself.
    pub salt_diffusivity: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            // Coarser than 2D's 0.05 m: in 3D, halving the spacing means 8×
            // the particles. A 1 m cube of water is 1,000 particles.
            particle_spacing: 0.1,
            rest_density: 1000.0,
            speed_of_sound: 20.0,
            // Low enough that small density differences (oil on water)
            // still drive flow; 0.05 froze a rising blob of oil in place.
            viscosity: 0.01,
            // Higher than the fluid's own viscosity on purpose: it damps
            // particles before they reach the zone right at a wall where
            // single-layer boundary handling over-counts density, which
            // otherwise launches spray off walls and corners in hard impacts.
            wall_viscosity: 0.5,
            salt_diffusivity: materials::SALT_DIFFUSIVITY,
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
            ("salt_diffusivity", self.salt_diffusivity),
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

/// Which optional physics a step includes (see [`Fluid::effects`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Effects {
    /// The liquids' real viscosity.
    pub viscosity: bool,
    /// Surface and interfacial tension.
    pub surface_tension: bool,
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
    x: Vec3,
    v: Vec3,
    /// Mass density ρ = m δ, kg/m³.
    rho: f32,
    /// Number density δ = Σ W, 1/m³.
    delta: f32,
    /// p / δ², the factor the pressure force uses.
    term: f32,
    /// Particle mass, kg.
    mass: f32,
    /// Dynamic viscosity, Pa·s.
    mu: f32,
    salinity: f32,
    liquid: u8,
}

/// A liquid's properties relative to the simulation's reference liquid.
#[derive(Debug, Clone, Copy)]
struct Scaled {
    /// ρ₀ / ρ₀(fresh water).
    density_ratio: f32,
    mu: f32,
}

fn scaled(liquid: Liquid, salinity: f32, scale: f32) -> Scaled {
    Scaled {
        density_ratio: liquid.density(salinity) / materials::WATER_DENSITY,
        mu: liquid.viscosity(salinity) * scale,
    }
}

/// Scratch arrays reused across substeps.
#[derive(Debug, Default)]
struct Scratch {
    densities: Vec<f32>,
    pressures: Vec<f32>,
    /// What the force pass reads about each neighbor, side by side so one
    /// lookup fetches it all.
    packed: Vec<Packed>,
    accelerations: Vec<Vec3>,
    /// d(salinity)/dt from diffusion.
    salinity_rates: Vec<f32>,
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
    /// Each particle's liquid. Omitted from scene files when all are water
    /// (and missing entries mean water).
    #[serde(default, skip_serializing_if = "all_water")]
    pub liquids: Vec<Liquid>,
    /// Each water particle's salinity, the mass fraction of dissolved salt
    /// (seawater: 0.035). Omitted when all are fresh.
    #[serde(default, skip_serializing_if = "all_fresh")]
    pub salinity: Vec<f32>,
    #[serde(skip)]
    scratch: Scratch,
    #[serde(skip)]
    pub stats: FluidStats,
}

fn all_zero(v: &[Vec3]) -> bool {
    v.iter().all(|&x| x == Vec3::ZERO)
}

fn all_water(v: &[Liquid]) -> bool {
    v.iter().all(|&l| l == Liquid::Water)
}

fn all_fresh(v: &[f32]) -> bool {
    v.iter().all(|&s| s == 0.0)
}

impl PartialEq for Fluid {
    /// Two fluids are equal if their saved state is: scratch space and
    /// statistics don't count.
    fn eq(&self, other: &Self) -> bool {
        let vel = |f: &Fluid, i: usize| f.velocities.get(i).copied().unwrap_or(Vec3::ZERO);
        let liquid = |f: &Fluid, i: usize| f.liquids.get(i).copied().unwrap_or_default();
        let salt = |f: &Fluid, i: usize| f.salinity.get(i).copied().unwrap_or(0.0);
        self.params == other.params
            && self.positions == other.positions
            && (0..self.positions.len()).all(|i| {
                vel(self, i) == vel(other, i)
                    && liquid(self, i) == liquid(other, i)
                    && salt(self, i) == salt(other, i)
            })
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
        self.liquids.clear();
        self.salinity.clear();
        self.stats = FluidStats::default();
    }

    /// Scene files may omit velocities, liquids, and salinities; treat
    /// missing ones as zero velocity and fresh water.
    pub fn fill_missing_velocities(&mut self) {
        let n = self.positions.len();
        self.velocities.resize(n, Vec3::ZERO);
        self.liquids.resize(n, Liquid::Water);
        self.salinity.resize(n, 0.0);
    }

    /// Particle `i`'s liquid.
    pub fn liquid(&self, i: usize) -> Liquid {
        self.liquids.get(i).copied().unwrap_or_default()
    }

    /// True if every particle is fresh water (the GPU solver handles only
    /// that so far).
    pub fn is_fresh_water(&self) -> bool {
        all_water(&self.liquids) && all_fresh(&self.salinity)
    }

    /// Volume of every particle at rest, m³: the reference mass over
    /// the reference density. All liquids use the same spacing, so a
    /// particle of liquid L weighs ρ₀(L) × this.
    pub fn particle_volume(&self) -> f32 {
        self.params.particle_mass() / self.params.rest_density
    }

    /// Particle `i`'s mass, kg.
    pub fn mass(&self, i: usize) -> f32 {
        let salinity = self.salinity.get(i).copied().unwrap_or(0.0);
        let ratio = self.liquid(i).density(salinity) / materials::WATER_DENSITY;
        self.params.particle_mass() * ratio
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
        let n = self.positions.len();
        if !self.liquids.is_empty() && self.liquids.len() != n {
            return Err(format!(
                "fluid has {n} positions but {} liquids",
                self.liquids.len()
            ));
        }
        if !self.salinity.is_empty() && self.salinity.len() != n {
            return Err(format!(
                "fluid has {n} positions but {} salinities",
                self.salinity.len()
            ));
        }
        if !self.salinity.iter().all(|s| (0.0..=0.4).contains(s)) {
            return Err("salinity must be between 0 and 0.4".into());
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
        liquid: Liquid,
        salinity: f32,
        inside: impl Fn(Vec3) -> bool,
    ) -> usize {
        let salinity = if liquid == Liquid::Water {
            salinity.clamp(0.0, 0.4)
        } else {
            0.0
        };
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
                        self.liquids.push(liquid);
                        self.salinity.push(salinity);
                    }
                }
            }
        }
        self.positions.len() - before
    }

    /// Fill the box between corners `a` and `b` with fresh water at rest,
    /// skipping space taken by bodies or existing fluid.
    pub fn fill_box(&mut self, a: Vec3, b: Vec3, bodies: &[RigidBody]) -> usize {
        self.fill_box_with(a, b, bodies, Liquid::Water, 0.0)
    }

    /// [`Self::fill_box`] with any liquid (salinity applies to water).
    pub fn fill_box_with(
        &mut self,
        a: Vec3,
        b: Vec3,
        bodies: &[RigidBody],
        liquid: Liquid,
        salinity: f32,
    ) -> usize {
        self.fill_where(a.min(b), a.max(b), bodies, liquid, salinity, |_| true)
    }

    /// Fill a ball with fresh water at rest.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f32, bodies: &[RigidBody]) -> usize {
        self.fill_sphere_with(center, radius, bodies, Liquid::Water, 0.0)
    }

    /// [`Self::fill_sphere`] with any liquid.
    pub fn fill_sphere_with(
        &mut self,
        center: Vec3,
        radius: f32,
        bodies: &[RigidBody],
        liquid: Liquid,
        salinity: f32,
    ) -> usize {
        self.fill_where(
            center - radius,
            center + radius,
            bodies,
            liquid,
            salinity,
            |p| (p - center).length_squared() <= radius * radius,
        )
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
                self.liquids[w] = self.liquids[r];
                self.salinity[w] = self.salinity[r];
                w += 1;
            }
        }
        self.positions.truncate(w);
        self.velocities.truncate(w);
        self.liquids.truncate(w);
        self.salinity.truncate(w);
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
        let mut v_max = max_reduce(iter!(self.velocities).map(|v| v.length_squared())).sqrt();
        let material_limit = self.material_dt_limit();
        // The 1e-6 guard stops a sliver of round-off from costing a whole
        // extra substep.
        while dt - elapsed > 1e-6 {
            let h = (CFL * d / (c + v_max))
                .min(material_limit)
                .min(dt - elapsed);
            v_max = self.substep(boundary, gravity, h, max_speed, &mut impulses);
            elapsed += h;
            substeps += 1;
        }
        self.update_stats(substeps);
        impulses
    }

    /// Which liquids are present, by [`Liquid::index`].
    fn present_liquids(&self) -> [bool; 3] {
        let mut present = [false; 3];
        for &l in &self.liquids {
            present[l.index()] = true;
        }
        if self.liquids.is_empty() && !self.positions.is_empty() {
            present[0] = true;
        }
        present
    }

    /// Whether the liquids' real viscosity and surface tension can have a
    /// visible effect at this resolution, and so are computed:
    ///
    /// - Viscosity, if the most viscous liquid present has a kinematic
    ///   viscosity at least 1% of the artificial viscosity every particle
    ///   already gets for stability (about α h c / 10 in 3D). Below that it's
    ///   lost in the numerical viscosity.
    /// - Surface tension, if the Bond number at the particle scale,
    ///   Bo = ρ g h² / σ, is at most 1 for some liquid: only then is a
    ///   single particle small enough for surface tension to beat its
    ///   weight (water: particles under 2.7 mm, the capillary length).
    ///   With coarser particles the pairwise cohesion forces that model
    ///   surface tension are far stronger than the real effect at that
    ///   scale and glue liquids in place (a 2 cm blob of oil at 4 mm
    ///   spacing, Bond number 12, never rises), so they're left out.
    ///
    /// Water at the default 5 cm spacing needs neither.
    pub fn effects(&self, gravity: Vec3) -> Effects {
        let present = self.present_liquids();
        let hs = 0.5 * self.params.support();
        let nu_art = self.params.viscosity * hs * self.params.speed_of_sound / 10.0;
        let g = gravity.length();
        let mut out = Effects::default();
        for l in Liquid::ALL.into_iter().filter(|l| present[l.index()]) {
            let rho = l.density(0.0);
            out.viscosity |= l.viscosity(0.0) / rho >= 0.01 * nu_art;
            out.surface_tension |= rho * g * hs * hs / l.surface_tension() <= 1.0;
        }
        out
    }

    /// The largest substep the liquids present allow for viscosity
    /// (Δt ≤ 0.125 h²ρ/μ) and surface tension (Δt ≤ 0.25 √(ρh³/2πσ),
    /// Brackbill et al. 1992), with h the smoothing length. Usually far
    /// above the CFL limit; it matters for thick oil and tiny particles.
    fn material_dt_limit(&self) -> f32 {
        let present = self.present_liquids();
        let hs = 0.5 * self.params.support();
        let mut limit = f32::INFINITY;
        for l in Liquid::ALL.into_iter().filter(|l| present[l.index()]) {
            let rho = l.density(0.0);
            limit = limit.min(0.125 * hs * hs * rho / l.viscosity(0.4));
            let sigma = l.surface_tension();
            limit = limit.min(0.25 * (rho * hs * hs * hs / (std::f32::consts::TAU * sigma)).sqrt());
        }
        limit
    }

    /// Compression statistics from the last substep's densities.
    fn update_stats(&mut self, substeps: u32) {
        let (mut max_c, mut sum_c, mut count) = (0.0_f32, 0.0_f32, 0_u32);
        for (i, &rho) in self.scratch.densities.iter().enumerate() {
            let rho0 = self.mass(i) / self.particle_volume();
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
        self.liquids = order.iter().map(|&i| self.liquids[i as usize]).collect();
        self.salinity = order.iter().map(|&i| self.salinity[i as usize]).collect();
    }

    /// One substep. Three parallel passes over the particles, so threads
    /// split up and rejoin as few times as possible: (1) neighbors, density,
    /// and pressure together; (2) accelerations; (3) integration, which also
    /// finds the fastest particle for the next substep's size. Returns that
    /// speed.
    ///
    /// Several liquids: densities differ up to 13.5× (mercury and water),
    /// which breaks the usual SPH density sum at interfaces (a water particle
    /// next to mercury would count mercury's mass as its own density). So
    /// this follows Solenthaler & Pajarola (2008): each particle sums the
    /// *number density* δ = Σⱼ W, its density is its own mass times that,
    /// ρ = m δ, and the pressure force is −Σⱼ (pᵢ/δᵢ² + pⱼ/δⱼ²) ∇W. With
    /// one liquid this is exactly the standard formulation.
    fn substep(
        &mut self,
        boundary: &Boundary,
        gravity: Vec3,
        dt: f32,
        max_speed: f32,
        impulses: &mut [BodyImpulse],
    ) -> f32 {
        let params = self.params;
        let kernel = params.kernel();
        let h = params.support();
        let h2 = h * h;
        let m_ref = params.particle_mass();
        let rho_ref = params.rest_density;
        // Every particle's volume at rest, and the reference liquid's
        // density relative to fresh water (1 unless rest_density is changed).
        let v0 = m_ref / rho_ref;
        let scale = rho_ref / materials::WATER_DENSITY;
        let c = params.speed_of_sound;
        let tait = c * c / TAIT_GAMMA as f32;
        // Monaghan viscosity uses the smoothing length (half our support).
        let h_visc = 0.5 * h;
        let eps = 0.01 * h_visc * h_visc;
        let n = self.positions.len();
        // Cohesion between liquids a and b: pair force k_ab s(r/h), with k
        // scaled so each surface and interface has its real tension (see
        // `kernel::cohesion_normalization` and `materials::work_of_adhesion`).
        let cohesion = v0 * v0 / cohesion_normalization(h);
        let mut k = [[0.0_f32; 3]; 3];
        for a in Liquid::ALL {
            for b in Liquid::ALL {
                k[a.index()][b.index()] =
                    0.5 * materials::work_of_adhesion(a, b) * scale * cohesion;
            }
        }
        let salt_d = params.salt_diffusivity;
        let water = Liquid::Water.index() as u8;
        let effects = self.effects(gravity);
        let (use_viscosity, use_cohesion) = (effects.viscosity, effects.surface_tension);
        let use_salt = salt_d > 0.0 && self.salinity.iter().any(|&s| s > 0.0);

        let Scratch {
            densities,
            pressures,
            packed,
            accelerations,
            salinity_rates,
            fluid_neighbors,
            boundary_neighbors,
            grid,
        } = &mut self.scratch;
        let positions = &self.positions;
        let velocities = &self.velocities;
        let liquids = &self.liquids;
        let salinity = &self.salinity;

        // --- 1. Neighbors within the kernel support, number density (the
        // particle itself counts too), density, and Tait pressure, clamped
        // at zero. All per particle, so one pass.
        grid.rebuild_fine(positions, h);
        let grid = &*grid;
        fluid_neighbors.resize_with(n, Vec::new);
        boundary_neighbors.resize_with(n, Vec::new);
        densities.resize(n, 0.0);
        pressures.resize(n, 0.0);
        packed.resize(n, Packed::default());
        let self_weight = kernel.w(0.0);
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
                let mut delta = self_weight;
                grid.for_each_near(xi, |j, xj| {
                    let r2 = (xi - xj).length_squared();
                    if r2 < h2 && j != i {
                        fluid.push(j as u32);
                        delta += kernel.w(r2.sqrt());
                    }
                });
                boundary.grid.for_each_near(xi, |b, xb| {
                    let r2 = (xi - xb).length_squared();
                    if r2 < h2 {
                        walls.push(b as u32);
                        // A boundary particle stands in for ψ/m_ref fluid
                        // particles.
                        delta += boundary.psi[b] / m_ref * kernel.w(r2.sqrt());
                    }
                });
                let (liquid, s) = (liquids[i], salinity[i]);
                let props = scaled(liquid, s, scale);
                let mass = m_ref * props.density_ratio;
                let rho0 = rho_ref * props.density_ratio;
                *rho = mass * delta;
                *p = (rho0 * tait * ((delta * v0).powi(TAIT_GAMMA) - 1.0)).max(0.0);
                *packed = Packed {
                    x: xi,
                    v: velocities[i],
                    rho: *rho,
                    delta,
                    term: *p / (delta * delta),
                    mass,
                    mu: props.mu,
                    salinity: s,
                    liquid: liquid.index() as u8,
                };
            });
        let fluid_neighbors = &*fluid_neighbors;
        let boundary_neighbors = &*boundary_neighbors;
        let packed = &*packed;

        // --- 2. Accelerations: pressure, artificial viscosity (for
        // stability), each liquid's real viscosity (Morris et al. 1997),
        // cohesion (surface and interfacial tension), and gravity. Also the
        // diffusion of salt between water particles.
        accelerations.resize(n, Vec3::ZERO);
        salinity_rates.resize(n, 0.0);
        iter_mut!(accelerations)
            .zip(iter_mut!(salinity_rates))
            .enumerate()
            .for_each(|(i, (acc, ds))| {
                let pi = packed[i];
                let inv_m = 1.0 / pi.mass;
                let li = pi.liquid as usize;
                let mut a = gravity;
                let mut salt = 0.0;

                for &j in &fluid_neighbors[i] {
                    let pj = packed[j as usize];
                    let r = pi.x - pj.x;
                    let dist = r.length();
                    let grad = kernel.grad(r, dist);
                    let r2e = dist * dist + eps;
                    // Pressure: antisymmetric, so momentum is conserved.
                    a -= inv_m * (pi.term + pj.term) * grad;
                    let v_ij = pi.v - pj.v;
                    let vr = v_ij.dot(r);
                    // Monaghan artificial viscosity, only while approaching.
                    if vr < 0.0 {
                        let nu = 2.0 * params.viscosity * h_visc * c / (pi.rho + pj.rho);
                        a -= pj.mass * (-nu * vr / r2e) * grad;
                    }
                    let rg = r.dot(grad);
                    // Physical viscosity: r·∇W < 0, so this pulls velocities
                    // together.
                    if use_viscosity {
                        a += inv_m * (pi.mu + pj.mu) * rg / (pi.delta * pj.delta * r2e) * v_ij;
                    }
                    // Cohesion: toward j where s > 0.
                    if use_cohesion && dist > 1e-9 {
                        let f = k[li][pj.liquid as usize] * cohesion_shape(dist / h);
                        a -= inv_m * f / dist * r;
                    }
                    if use_salt && pi.liquid == water && pj.liquid == water {
                        salt += 2.0 * salt_d * v0 * (pi.salinity - pj.salinity) * rg / r2e;
                    }
                }

                for &b in &boundary_neighbors[i] {
                    let b = b as usize;
                    let r = pi.x - boundary.positions[b];
                    let dist = r.length();
                    let grad = kernel.grad(r, dist);
                    let r2e = dist * dist + eps;
                    let w = boundary.psi[b] / m_ref;
                    // Akinci boundary pressure: the boundary "mirrors" this
                    // particle's own pressure.
                    a -= w * inv_m * pi.term * grad;
                    let v_ib = pi.v - boundary.velocities[b];
                    let vr = v_ib.dot(r);
                    if vr < 0.0 {
                        let nu = params.wall_viscosity * h_visc * c / pi.rho;
                        a -= w * pi.mass * (-nu * vr / r2e) * grad;
                    }
                    // No-slip physical viscosity against the wall.
                    if use_viscosity {
                        a += w * inv_m * 2.0 * pi.mu * r.dot(grad) / (pi.delta * pi.delta * r2e)
                            * v_ib;
                    }
                }
                *acc = a;
                *ds = salt;
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
                let pi = packed[i];
                for &b in list {
                    let b = b as usize;
                    let xb = boundary.positions[b];
                    let r = pi.x - xb;
                    let dist = r.length();
                    let grad = kernel.grad(r, dist);
                    let r2e = dist * dist + eps;
                    let w = boundary.psi[b] / m_ref;
                    let v_ib = pi.v - boundary.velocities[b];
                    let vr = v_ib.dot(r);
                    let mut force = w * pi.term * grad;
                    if vr < 0.0 {
                        let nu = params.wall_viscosity * h_visc * c / pi.rho;
                        force += w * pi.mass * pi.mass * (-nu * vr / r2e) * grad;
                    }
                    if use_viscosity {
                        force -= w * 2.0 * pi.mu * r.dot(grad) / (pi.delta * pi.delta * r2e) * v_ib;
                    }
                    let body = boundary.body[b] as usize;
                    if let Some(impulse) = impulses.get_mut(body) {
                        impulse.linear += force * dt;
                        impulse.angular += (xb - boundary.body_centers[body]).cross(force) * dt;
                    }
                }
            }
        }

        // --- 3. Integrate (semi-implicit Euler), noting the top speed.
        let accelerations = &self.scratch.accelerations;
        let salinity_rates = &self.scratch.salinity_rates;
        if use_salt {
            iter_mut!(self.salinity)
                .zip(iter!(salinity_rates))
                .for_each(|(s, ds)| *s = (*s + ds * dt).max(0.0));
        }
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
        self.velocities
            .iter()
            .enumerate()
            .map(|(i, v)| 0.5 * self.mass(i) * v.length_squared())
            .sum()
    }

    /// Total momentum of the fluid, kg·m/s.
    pub fn momentum(&self) -> Vec3 {
        self.velocities
            .iter()
            .enumerate()
            .map(|(i, &v)| self.mass(i) * v)
            .sum()
    }

    /// Total mass of each liquid, kg, indexed by [`Liquid::index`].
    pub fn mass_by_liquid(&self) -> [f32; 3] {
        let mut out = [0.0; 3];
        for i in 0..self.len() {
            out[self.liquid(i).index()] += self.mass(i);
        }
        out
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
