//! Smoke, dye, and heat: a grid-based (Eulerian) gas solver.
//!
//! Where the SPH fluid follows particles of water, this solver keeps the
//! air's velocity, a smoke (or dye) concentration, and the temperature on a
//! fixed grid covering a rectangle of the world. Each step:
//!
//! 1. **Obstacles.** Cells inside rigid bodies are solid and move with the
//!    body.
//! 2. **Sources.** Emitters add smoke and heat (and can blow).
//! 3. **Buoyancy** (Boussinesq approximation): warm air is lighter, smoke is
//!    heavier, so each parcel accelerates by a = −g (β ΔT − α s), with
//!    ΔT the temperature above ambient, β = 1/T_ambient the thermal
//!    expansion coefficient of an ideal gas, s the smoke concentration, and α
//!    `smoke_weight`.
//! 4. **Advection** (semi-Lagrangian, Stam 1999 "Stable Fluids"): each grid
//!    point looks back along the flow, with a second-order (midpoint)
//!    backtrace, and takes the value it finds there. Unconditionally stable.
//! 5. **Heat diffusion** (Fourier's law, ∂T/∂t = κ ∇²T), implicit so any
//!    κ and step size are stable, with insulated walls and obstacles.
//! 6. **Pressure projection.** Air is treated as incompressible at these
//!    speeds, so the velocity is made divergence-free by subtracting the
//!    gradient of a pressure found from a Poisson equation, solved with
//!    preconditioned conjugate gradients.
//!
//! The grid is staggered (a "MAC" grid, Harlow & Welch 1965): horizontal
//! velocities live on the left/right faces of each cell and vertical ones on
//! the bottom/top faces, which makes the divergence and pressure gradient
//! exact differences and avoids checkerboard pressure.
//!
//! Units: SI. Velocity m/s, temperature K, smoke concentration is
//! dimensionless (1 = thick smoke), cell size m.

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use glam::Vec2;
use serde::{Deserialize, Serialize};

/// Material settings for the gas.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GasParams {
    /// Temperature of still air, K (20 °C).
    pub ambient_temperature: f32,
    /// κ, m²/s. Air is 2.2e-5; at grid scale the flow itself mixes far more
    /// than that, so this matters mostly for still air.
    pub thermal_diffusivity: f32,
    /// α: how much heavier than air smoke of concentration 1 is, as a
    /// fraction (its buoyancy is −α g per unit concentration).
    /// 0 is weightless smoke (a passive dye).
    pub smoke_weight: f32,
    /// Fraction of smoke removed per second (it thins out).
    pub smoke_decay: f32,
    /// Newton cooling toward ambient, 1/s.
    pub cooling: f32,
    /// Let air (and smoke) leave through the top of the domain.
    pub open_top: bool,
    /// How often the world steps the gas, Hz. Smoke doesn't need the rigid
    /// bodies' 120 Hz, and its solver is stable at large steps, so the
    /// world steps it less often (with a correspondingly longer step).
    #[serde(default = "default_gas_hz")]
    pub step_hz: f32,
}

fn default_gas_hz() -> f32 {
    30.0
}

impl Default for GasParams {
    fn default() -> Self {
        Self {
            ambient_temperature: 293.15,
            thermal_diffusivity: 2.2e-5,
            smoke_weight: 0.0,
            smoke_decay: 0.0,
            cooling: 0.0,
            open_top: true,
            step_hz: default_gas_hz(),
        }
    }
}

/// A source of smoke and heat.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    pub center: Vec2,
    /// m
    pub radius: f32,
    /// Smoke concentration added per second.
    pub smoke_rate: f32,
    /// Temperature it holds the air at, K. Ambient or below means no heat.
    pub temperature: f32,
    /// Air velocity it blows, m/s (zero for none).
    #[serde(default)]
    pub velocity: Vec2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gas {
    /// World position of the grid's bottom-left corner, m.
    pub origin: Vec2,
    /// Side of each square cell, m.
    pub cell: f32,
    pub nx: usize,
    pub ny: usize,
    #[serde(default)]
    pub params: GasParams,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub emitters: Vec<Emitter>,
    /// Horizontal velocity on vertical faces, (nx + 1) × ny, m/s.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub u: Vec<f32>,
    /// Vertical velocity on horizontal faces, nx × (ny + 1), m/s.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub v: Vec<f32>,
    /// Smoke concentration per cell.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub smoke: Vec<f32>,
    /// Temperature above ambient per cell, K.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub heat: Vec<f32>,
    #[serde(skip)]
    solid: Vec<bool>,
    #[serde(skip)]
    solid_velocity: Vec<Vec2>,
    #[serde(skip)]
    pressure: Vec<f32>,
    #[serde(skip)]
    pub stats: GasStats,
}

fn all_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

impl PartialEq for Gas {
    fn eq(&self, other: &Self) -> bool {
        self.origin == other.origin
            && self.cell == other.cell
            && (self.nx, self.ny) == (other.nx, other.ny)
            && self.params == other.params
            && self.emitters == other.emitters
            && self.u == other.u
            && self.v == other.v
            && self.smoke == other.smoke
            && self.heat == other.heat
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GasStats {
    pub substeps: u32,
    /// Conjugate-gradient iterations in the last pressure solve.
    pub pressure_iterations: u32,
    /// Largest |∇·u| after projection, 1/s.
    pub max_divergence: f32,
}

/// The pressure solve stops when its residual drops below this fraction of
/// the right-hand side: divergence of order 1e-4 of what was there, far
/// below anything visible, at a third of the iterations 1e-6 takes.
const PRESSURE_TOLERANCE: f64 = 1e-4;
/// Heat diffusion converges quickly, so it's solved tightly (total heat is
/// conserved to about this fraction).
const DIFFUSION_TOLERANCE: f64 = 1e-6;
const MAX_SOLVER_ITERATIONS: usize = 2000;

/// What surrounds a cell on one side, for the Laplacian.
#[derive(Clone, Copy, PartialEq)]
enum Side {
    Fluid(usize),
    /// Wall or obstacle: no flow through (Neumann).
    Wall,
    /// Open boundary at zero pressure (Dirichlet).
    Open,
}

impl Gas {
    /// A still, ambient-temperature grid of nx × ny cells of side `cell`
    /// with its bottom-left corner at `origin`.
    pub fn new(origin: Vec2, cell: f32, nx: usize, ny: usize) -> Self {
        let (nx, ny) = (nx.max(2), ny.max(2));
        let mut g = Self {
            origin,
            cell,
            nx,
            ny,
            params: GasParams::default(),
            emitters: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            smoke: Vec::new(),
            heat: Vec::new(),
            solid: Vec::new(),
            solid_velocity: Vec::new(),
            pressure: Vec::new(),
            stats: GasStats::default(),
        };
        g.fill_missing();
        g
    }

    /// Resize any field a scene file left out.
    pub fn fill_missing(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        self.u.resize((nx + 1) * ny, 0.0);
        self.v.resize(nx * (ny + 1), 0.0);
        self.smoke.resize(nx * ny, 0.0);
        self.heat.resize(nx * ny, 0.0);
        self.solid.resize(nx * ny, false);
        self.solid_velocity.resize(nx * ny, Vec2::ZERO);
        self.pressure.resize(nx * ny, 0.0);
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.cell > 0.0 && self.cell.is_finite() && self.origin.is_finite()) {
            return Err("gas grid needs a positive cell size".into());
        }
        if self.nx < 2 || self.ny < 2 || self.nx * self.ny > 4_000_000 {
            return Err(format!(
                "gas grid size {}×{} is out of range",
                self.nx, self.ny
            ));
        }
        let n = self.nx * self.ny;
        let sized = |v: &[f32], len| v.is_empty() || v.len() == len;
        if !(sized(&self.u, (self.nx + 1) * self.ny)
            && sized(&self.v, self.nx * (self.ny + 1))
            && sized(&self.smoke, n)
            && sized(&self.heat, n))
        {
            return Err("gas field sizes don't match the grid".into());
        }
        if !(self.params.ambient_temperature > 0.0 && self.params.thermal_diffusivity >= 0.0) {
            return Err(
                "gas ambient temperature must be positive and diffusivity non-negative".into(),
            );
        }
        Ok(())
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.nx as f32, self.ny as f32) * self.cell
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    /// Center of cell (i, j) in world space.
    pub fn cell_center(&self, i: usize, j: usize) -> Vec2 {
        self.origin + (Vec2::new(i as f32, j as f32) + 0.5) * self.cell
    }

    /// The cell containing world point `p`, if inside the grid.
    pub fn cell_at(&self, p: Vec2) -> Option<(usize, usize)> {
        let q = (p - self.origin) / self.cell;
        (q.x >= 0.0 && q.y >= 0.0 && (q.x as usize) < self.nx && (q.y as usize) < self.ny)
            .then_some((q.x as usize, q.y as usize))
    }

    pub fn is_solid(&self, i: usize, j: usize) -> bool {
        self.solid.get(self.idx(i, j)).copied().unwrap_or(false)
    }

    /// Temperature at a cell, K.
    pub fn temperature(&self, i: usize, j: usize) -> f32 {
        self.params.ambient_temperature + self.heat[self.idx(i, j)]
    }

    /// Air velocity at cell (i, j)'s center (average of its faces), m/s.
    pub fn velocity_at_cell(&self, i: usize, j: usize) -> Vec2 {
        let (nx, _) = (self.nx, self.ny);
        Vec2::new(
            0.5 * (self.u[j * (nx + 1) + i] + self.u[j * (nx + 1) + i + 1]),
            0.5 * (self.v[j * nx + i] + self.v[(j + 1) * nx + i]),
        )
    }

    /// Total heat content above ambient, K·m² (proportional to energy).
    pub fn total_heat(&self) -> f32 {
        self.heat.iter().sum::<f32>() * self.cell * self.cell
    }

    pub fn total_smoke(&self) -> f32 {
        self.smoke.iter().sum::<f32>() * self.cell * self.cell
    }

    /// Add smoke and heat in a disc (the editor's brush): concentration
    /// `smoke` and temperature at least `temperature` K.
    pub fn paint(&mut self, center: Vec2, radius: f32, smoke: f32, temperature: f32) {
        self.fill_missing();
        let dt = temperature - self.params.ambient_temperature;
        for j in 0..self.ny {
            for i in 0..self.nx {
                if (self.cell_center(i, j) - center).length() <= radius {
                    let k = self.idx(i, j);
                    self.smoke[k] = self.smoke[k].max(smoke);
                    if dt > 0.0 {
                        self.heat[k] = self.heat[k].max(dt);
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        for f in [&mut self.u, &mut self.v, &mut self.smoke, &mut self.heat] {
            f.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    // ---------------------------------------------------------------------
    // Stepping

    /// Advance by `dt` with `bodies` as obstacles and `gravity` driving
    /// buoyancy. Takes substeps so the flow moves at most one cell per
    /// substep (more is stable but smears detail).
    pub fn step(&mut self, bodies: &[RigidBody], gravity: Vec2, dt: f32) {
        self.fill_missing();
        self.update_solids(bodies);
        let vmax = self
            .u
            .iter()
            .chain(&self.v)
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        let substeps = ((vmax * dt / self.cell).ceil() as u32).clamp(1, 8);
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.apply_emitters(h);
            self.apply_buoyancy(gravity, h);
            self.enforce_boundaries();
            self.advect_velocity(h);
            self.diffuse_heat(h);
            self.advect_scalars(h);
            self.decay(h);
            self.project(h);
        }
        self.stats.substeps = substeps;
    }

    fn update_solids(&mut self, bodies: &[RigidBody]) {
        self.solid.iter_mut().for_each(|s| *s = false);
        for body in bodies {
            let bb = Aabb::of(body);
            let lo = ((bb.min - self.origin) / self.cell).floor().max(Vec2::ZERO);
            let hi = ((bb.max - self.origin) / self.cell).ceil();
            let (i1, j1) = ((hi.x as usize).min(self.nx), (hi.y as usize).min(self.ny));
            for j in lo.y as usize..j1 {
                for i in lo.x as usize..i1 {
                    let c = self.cell_center(i, j);
                    if body.contains_point(c) {
                        let k = self.idx(i, j);
                        self.solid[k] = true;
                        self.solid_velocity[k] = body.velocity_at(c);
                        // Obstacles hold no smoke or heat.
                        self.smoke[k] = 0.0;
                        self.heat[k] = 0.0;
                    }
                }
            }
        }
    }

    fn apply_emitters(&mut self, h: f32) {
        let emitters = std::mem::take(&mut self.emitters);
        for e in &emitters {
            let r = e.radius.max(0.5 * self.cell);
            let dt = e.temperature - self.params.ambient_temperature;
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let k = self.idx(i, j);
                    if self.solid[k] || (self.cell_center(i, j) - e.center).length() > r {
                        continue;
                    }
                    self.smoke[k] += e.smoke_rate * h;
                    if dt > 0.0 {
                        self.heat[k] = self.heat[k].max(dt);
                    }
                    if e.velocity != Vec2::ZERO {
                        let nx = self.nx;
                        self.u[j * (nx + 1) + i] = e.velocity.x;
                        self.u[j * (nx + 1) + i + 1] = e.velocity.x;
                        self.v[j * nx + i] = e.velocity.y;
                        self.v[(j + 1) * nx + i] = e.velocity.y;
                    }
                }
            }
        }
        self.emitters = emitters;
    }

    fn apply_buoyancy(&mut self, gravity: Vec2, h: f32) {
        let beta = 1.0 / self.params.ambient_temperature;
        let alpha = self.params.smoke_weight;
        let (nx, ny) = (self.nx, self.ny);
        // b = β ΔT − α s, averaged onto each face from its two cells.
        let b = |g: &Gas, k: usize| beta * g.heat[k] - alpha * g.smoke[k];
        if gravity.y != 0.0 {
            for j in 1..ny {
                for i in 0..nx {
                    let force = 0.5 * (b(self, self.idx(i, j - 1)) + b(self, self.idx(i, j)));
                    self.v[j * nx + i] += -gravity.y * force * h;
                }
            }
        }
        if gravity.x != 0.0 {
            for j in 0..ny {
                for i in 1..nx {
                    let force = 0.5 * (b(self, self.idx(i - 1, j)) + b(self, self.idx(i, j)));
                    self.u[j * (nx + 1) + i] += -gravity.x * force * h;
                }
            }
        }
    }

    /// Walls and obstacles: no flow through them (faces touching a solid
    /// take its velocity); the domain's edges are walls except an open top.
    fn enforce_boundaries(&mut self) {
        let (nx, ny) = (self.nx, self.ny);
        for j in 0..ny {
            self.u[j * (nx + 1)] = 0.0;
            self.u[j * (nx + 1) + nx] = 0.0;
            for i in 1..nx {
                let (l, r) = (self.idx(i - 1, j), self.idx(i, j));
                if self.solid[l] || self.solid[r] {
                    let s = if self.solid[l] { l } else { r };
                    self.u[j * (nx + 1) + i] = self.solid_velocity[s].x;
                }
            }
        }
        for i in 0..nx {
            self.v[i] = 0.0;
            if !self.params.open_top {
                self.v[ny * nx + i] = 0.0;
            } else {
                // Open top: no inflow of outside air.
                self.v[ny * nx + i] = self.v[ny * nx + i].max(0.0);
            }
            for j in 1..ny {
                let (b, t) = (self.idx(i, j - 1), self.idx(i, j));
                if self.solid[b] || self.solid[t] {
                    let s = if self.solid[b] { b } else { t };
                    self.v[j * nx + i] = self.solid_velocity[s].y;
                }
            }
        }
    }

    /// Bilinear sample of a staggered field whose sample (i, j) sits at grid
    /// coordinates (i + ox, j + oy), clamped to the field.
    fn sample(field: &[f32], w: usize, h: usize, ox: f32, oy: f32, gx: f32, gy: f32) -> f32 {
        let x = (gx - ox).clamp(0.0, (w - 1) as f32);
        let y = (gy - oy).clamp(0.0, (h - 1) as f32);
        let (i0, j0) = ((x as usize).min(w - 2), (y as usize).min(h - 2));
        let (fx, fy) = (x - i0 as f32, y - j0 as f32);
        let at = |i: usize, j: usize| field[j * w + i];
        let bottom = at(i0, j0) * (1.0 - fx) + at(i0 + 1, j0) * fx;
        let top = at(i0, j0 + 1) * (1.0 - fx) + at(i0 + 1, j0 + 1) * fx;
        bottom * (1.0 - fy) + top * fy
    }

    /// Velocity at grid coordinates (cells; cell (i, j) spans [i, i+1]).
    fn velocity_grid(&self, gx: f32, gy: f32) -> Vec2 {
        let (nx, ny) = (self.nx, self.ny);
        Vec2::new(
            Self::sample(&self.u, nx + 1, ny, 0.0, 0.5, gx, gy),
            Self::sample(&self.v, nx, ny + 1, 0.5, 0.0, gx, gy),
        )
    }

    /// Midpoint (RK2) backtrace from grid point (gx, gy) over time h.
    fn backtrace(&self, gx: f32, gy: f32, h: f32) -> (f32, f32) {
        let s = h / self.cell;
        let v1 = self.velocity_grid(gx, gy);
        let mid = (gx - 0.5 * s * v1.x, gy - 0.5 * s * v1.y);
        let v2 = self.velocity_grid(mid.0, mid.1);
        (gx - s * v2.x, gy - s * v2.y)
    }

    fn advect_velocity(&mut self, h: f32) {
        let (nx, ny) = (self.nx, self.ny);
        let mut u = self.u.clone();
        let mut v = self.v.clone();
        for j in 0..ny {
            for i in 1..nx {
                let (x, y) = self.backtrace(i as f32, j as f32 + 0.5, h);
                u[j * (nx + 1) + i] = Self::sample(&self.u, nx + 1, ny, 0.0, 0.5, x, y);
            }
        }
        for j in 1..=ny {
            for i in 0..nx {
                if j == ny && !self.params.open_top {
                    continue;
                }
                let (x, y) = self.backtrace(i as f32 + 0.5, j as f32, h);
                v[j * nx + i] = Self::sample(&self.v, nx, ny + 1, 0.5, 0.0, x, y);
            }
        }
        self.u = u;
        self.v = v;
        self.enforce_boundaries();
    }

    fn advect_scalars(&mut self, h: f32) {
        let (nx, ny) = (self.nx, self.ny);
        let mut smoke = self.smoke.clone();
        let mut heat = self.heat.clone();
        for j in 0..ny {
            for i in 0..nx {
                let k = self.idx(i, j);
                if self.solid[k] {
                    continue;
                }
                let (x, y) = self.backtrace(i as f32 + 0.5, j as f32 + 0.5, h);
                smoke[k] = Self::sample(&self.smoke, nx, ny, 0.5, 0.5, x, y);
                heat[k] = Self::sample(&self.heat, nx, ny, 0.5, 0.5, x, y);
            }
        }
        self.smoke = smoke;
        self.heat = heat;
    }

    fn decay(&mut self, h: f32) {
        let s = (1.0 - self.params.smoke_decay * h).max(0.0);
        let c = (1.0 - self.params.cooling * h).max(0.0);
        if s < 1.0 {
            self.smoke.iter_mut().for_each(|x| *x *= s);
        }
        if c < 1.0 {
            self.heat.iter_mut().for_each(|x| *x *= c);
        }
    }

    /// What lies on each side of cell k: (left, right, bottom, top). For
    /// pressure the open top is `Open`; for heat every boundary is `Wall`
    /// (insulated).
    fn sides(&self, i: usize, j: usize, pressure: bool) -> [Side; 4] {
        let cell = |ii: usize, jj: usize| {
            let k = self.idx(ii, jj);
            if self.solid[k] {
                Side::Wall
            } else {
                Side::Fluid(k)
            }
        };
        [
            if i > 0 { cell(i - 1, j) } else { Side::Wall },
            if i + 1 < self.nx {
                cell(i + 1, j)
            } else {
                Side::Wall
            },
            if j > 0 { cell(i, j - 1) } else { Side::Wall },
            if j + 1 < self.ny {
                cell(i, j + 1)
            } else if pressure && self.params.open_top {
                Side::Open
            } else {
                Side::Wall
            },
        ]
    }

    /// Implicit heat diffusion: (1 − h κ ∇²) T' = T, solved with conjugate
    /// gradients. Walls and obstacles are insulating, so total heat is
    /// conserved exactly (up to the solver tolerance).
    fn diffuse_heat(&mut self, h: f32) {
        let kappa = self.params.thermal_diffusivity;
        if kappa <= 0.0 {
            return;
        }
        let c = f64::from(h * kappa / (self.cell * self.cell));
        let n = self.nx * self.ny;
        let sides: Vec<[Side; 4]> = (0..n)
            .map(|k| self.sides(k % self.nx, k / self.nx, false))
            .collect();
        let fluid: Vec<bool> = self.solid.iter().map(|s| !s).collect();
        let b: Vec<f64> = (0..n)
            .map(|k| {
                if fluid[k] {
                    f64::from(self.heat[k])
                } else {
                    0.0
                }
            })
            .collect();
        let apply = |x: &[f64], out: &mut [f64]| {
            fill(out, |k| {
                if !fluid[k] {
                    return x[k];
                }
                let mut lap = 0.0;
                for side in sides[k] {
                    if let Side::Fluid(m) = side {
                        lap += x[m] - x[k];
                    }
                }
                x[k] - c * lap
            });
        };
        let diag: Vec<f64> = (0..n)
            .map(|k| {
                if fluid[k] {
                    1.0 + c * sides[k]
                        .iter()
                        .filter(|s| matches!(s, Side::Fluid(_)))
                        .count() as f64
                } else {
                    1.0
                }
            })
            .collect();
        let mut x: Vec<f64> = b.clone();
        // Diagonally dominant: dividing by the diagonal is enough.
        let jacobi = |r: &[f64], z: &mut [f64]| fill(z, |k| r[k] / diag[k]);
        conjugate_gradient(&apply, &jacobi, &b, &mut x, DIFFUSION_TOLERANCE);
        for k in 0..n {
            if fluid[k] {
                self.heat[k] = x[k] as f32;
            }
        }
    }

    /// Make the velocity divergence-free. With q = p·h (kinematic pressure
    /// times the step), each fluid cell needs Σ_neighbors (q_c − q_n) =
    /// −(∇·u) dx², walls contributing nothing and the open top q = 0. Then
    /// each face's velocity drops by the q difference across it over dx.
    fn project(&mut self, h: f32) {
        self.enforce_boundaries();
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let dx = self.cell;
        let fluid: Vec<bool> = self.solid.iter().map(|s| !s).collect();
        let sides: Vec<[Side; 4]> = (0..n).map(|k| self.sides(k % nx, k / nx, true)).collect();
        let divergence = |g: &Gas, i: usize, j: usize| {
            (g.u[j * (nx + 1) + i + 1] - g.u[j * (nx + 1) + i] + g.v[(j + 1) * nx + i]
                - g.v[j * nx + i])
                / dx
        };
        let mut b: Vec<f64> = (0..n)
            .map(|k| {
                if fluid[k] {
                    -f64::from(divergence(self, k % nx, k / nx) * dx * dx)
                } else {
                    0.0
                }
            })
            .collect();
        // A closed domain's pressure is only defined up to a constant; the
        // right-hand side must then sum to zero (whatever flows in through
        // moving obstacles flows out), so remove any round-off mean.
        let open = sides.iter().flatten().any(|s| *s == Side::Open);
        if !open {
            let count = fluid.iter().filter(|&&f| f).count().max(1) as f64;
            let mean = b.iter().sum::<f64>() / count;
            for k in 0..n {
                if fluid[k] {
                    b[k] -= mean;
                }
            }
        }
        let apply = |x: &[f64], out: &mut [f64]| {
            fill(out, |k| {
                if !fluid[k] {
                    return 0.0;
                }
                let mut sum = 0.0;
                for side in sides[k] {
                    match side {
                        Side::Fluid(m) => sum += x[k] - x[m],
                        Side::Open => sum += x[k],
                        Side::Wall => {}
                    }
                }
                sum
            });
        };
        let diag: Vec<f64> = (0..n)
            .map(|k| {
                sides[k]
                    .iter()
                    .filter(|s| !matches!(s, Side::Wall))
                    .count()
                    .max(1) as f64
            })
            .collect();
        // Start from last step's pressure: it changes little from step to
        // step, so this saves most of the iterations.
        let mut q: Vec<f64> = (0..n)
            .map(|k| {
                if fluid[k] {
                    f64::from(self.pressure[k] * h)
                } else {
                    0.0
                }
            })
            .collect();
        let mic = Mic::new(&sides, &fluid, &diag);
        let precondition = |r: &[f64], z: &mut [f64]| mic.apply(r, z);
        let iterations = conjugate_gradient(&apply, &precondition, &b, &mut q, PRESSURE_TOLERANCE);
        // Subtract the gradient on faces between fluid cells (and at the
        // open top, where q = 0 outside).
        for j in 0..ny {
            for i in 1..nx {
                let (l, r) = (self.idx(i - 1, j), self.idx(i, j));
                if fluid[l] && fluid[r] {
                    self.u[j * (nx + 1) + i] -= ((q[r] - q[l]) as f32) / dx;
                }
            }
        }
        for j in 1..=ny {
            for i in 0..nx {
                if j == ny {
                    let below = self.idx(i, ny - 1);
                    if self.params.open_top && fluid[below] {
                        self.v[ny * nx + i] -= ((0.0 - q[below]) as f32) / dx;
                    }
                    continue;
                }
                let (bot, top) = (self.idx(i, j - 1), self.idx(i, j));
                if fluid[bot] && fluid[top] {
                    self.v[j * nx + i] -= ((q[top] - q[bot]) as f32) / dx;
                }
            }
        }
        for (p, &qk) in self.pressure.iter_mut().zip(&q) {
            *p = (qk / f64::from(h)) as f32;
        }
        self.stats.pressure_iterations = iterations as u32;
        self.stats.max_divergence = (0..n)
            .filter(|&k| fluid[k])
            .map(|k| divergence(self, k % nx, k / nx).abs())
            .fold(0.0, f32::max);
    }

    /// Kinematic pressure from the last projection, m²/s² (pressure / ρ).
    pub fn pressure(&self) -> &[f32] {
        &self.pressure
    }
}

/// Jacobi-preconditioned conjugate gradients for a symmetric positive
/// (semi-)definite operator. Returns the iteration count.
/// Modified incomplete Cholesky, MIC(0) (Bridson, "Fluid Simulation for
/// Computer Graphics", §5.3): an approximate factorization L Lᵀ ≈ A of the
/// pressure matrix with the same sparsity, applied by one forward and one
/// backward sweep. It cuts the conjugate-gradient iterations about 3–4×
/// compared with dividing by the diagonal.
struct Mic {
    /// 1 / L_cc per cell (0 for solid cells).
    precon: Vec<f64>,
    /// Off-diagonal coefficient to the next cell along each axis (−1
    /// between two fluid cells, else 0), and that cell's index.
    plus: Vec<[(f64, usize); 2]>,
}

impl Mic {
    /// τ: how much of the dropped fill-in to compensate (0.97 is standard);
    /// σ: safety floor against tiny pivots.
    const TAU: f64 = 0.97;
    const SIGMA: f64 = 0.25;

    fn new(sides: &[[Side; 4]], fluid: &[bool], diag: &[f64]) -> Self {
        let n = fluid.len();
        let plus: Vec<[(f64, usize); 2]> = (0..n)
            .map(|c| {
                [sides[c][1], sides[c][3]].map(|side| match side {
                    Side::Fluid(m) if fluid[c] => (-1.0, m),
                    _ => (0.0, c),
                })
            })
            .collect();
        // Cells before c that link to it, per axis: (coefficient, index).
        let mut minus: Vec<[(f64, usize); 2]> = vec![[(0.0, 0); 2]; n];
        for (c, links) in plus.iter().enumerate() {
            for (d, &(a, m)) in links.iter().enumerate() {
                if a != 0.0 {
                    minus[m][d] = (a, c);
                }
            }
        }
        let mut precon = vec![0.0; n];
        for c in 0..n {
            if !fluid[c] {
                continue;
            }
            let mut e = diag[c];
            for (d, &(a, m)) in minus[c].iter().enumerate() {
                if a == 0.0 {
                    continue;
                }
                let pm = precon[m];
                e -= (a * pm).powi(2);
                // Fill-in compensation: m's links along the other axes.
                let others: f64 = plus[m]
                    .iter()
                    .enumerate()
                    .filter(|&(dd, _)| dd != d)
                    .map(|(_, &(b, _))| b)
                    .sum();
                e -= Self::TAU * a * others * pm * pm;
            }
            if e < Self::SIGMA * diag[c] {
                e = diag[c];
            }
            precon[c] = 1.0 / e.sqrt();
        }
        Self { precon, plus }
    }

    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = r.len();
        let mut q = vec![0.0; n];
        // Forward: solve L q = r.
        for c in 0..n {
            if self.precon[c] == 0.0 {
                continue;
            }
            q[c] += r[c];
            q[c] *= self.precon[c];
            for &(a, m) in &self.plus[c] {
                if a != 0.0 {
                    q[m] -= a * self.precon[c] * q[c];
                }
            }
        }
        // Backward: solve Lᵀ z = q.
        for c in (0..n).rev() {
            if self.precon[c] == 0.0 {
                z[c] = 0.0;
                continue;
            }
            let mut t = q[c];
            for &(a, m) in &self.plus[c] {
                if a != 0.0 {
                    t -= a * self.precon[c] * z[m];
                }
            }
            z[c] = t * self.precon[c];
        }
    }
}

/// Run `f(k)` for every element of `out`, in parallel on desktop.
fn fill(out: &mut [f64], f: impl Fn(usize) -> f64 + Sync) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(k, o)| *o = f(k));
    }
    #[cfg(not(feature = "parallel"))]
    for (k, o) in out.iter_mut().enumerate() {
        *o = f(k);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        a.par_iter().zip(b).map(|(x, y)| x * y).sum()
    }
    #[cfg(not(feature = "parallel"))]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Preconditioned conjugate gradients for a symmetric positive
/// (semi-)definite operator. `apply(x, out)` writes A·x;
/// `precondition(r, z)` writes an approximation of A⁻¹·r. Returns the
/// iteration count.
fn conjugate_gradient(
    apply: &(dyn Fn(&[f64], &mut [f64]) + Sync),
    precondition: &dyn Fn(&[f64], &mut [f64]),
    b: &[f64],
    x: &mut [f64],
    tolerance: f64,
) -> usize {
    let n = b.len();
    let mut ax = vec![0.0; n];
    apply(x, &mut ax);
    let mut r: Vec<f64> = (0..n).map(|k| b[k] - ax[k]).collect();
    let b_norm = dot(b, b).sqrt();
    if b_norm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        return 0;
    }
    let mut z = vec![0.0; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    for it in 0..MAX_SOLVER_ITERATIONS {
        if dot(&r, &r).sqrt() <= tolerance * b_norm {
            return it;
        }
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-300 {
            return it;
        }
        let alpha = rz / pap;
        for k in 0..n {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }
        precondition(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        let p_old = std::mem::take(&mut p);
        p = vec![0.0; n];
        fill(&mut p, |k| z[k] + beta * p_old[k]);
    }
    MAX_SOLVER_ITERATIONS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_makes_flow_divergence_free() {
        let mut g = Gas::new(Vec2::ZERO, 0.1, 32, 24);
        g.params.open_top = false;
        // A swirl plus a source: plenty of divergence to remove.
        for j in 0..g.ny {
            for i in 1..g.nx {
                g.u[j * (g.nx + 1) + i] = ((i * 7 + j * 3) % 5) as f32 - 2.0;
            }
        }
        let nx = g.nx;
        let before = (0..g.ny)
            .flat_map(|j| (0..nx).map(move |i| (i, j)))
            .map(|(i, j)| {
                ((g.u[j * (nx + 1) + i + 1] - g.u[j * (nx + 1) + i] + g.v[(j + 1) * nx + i]
                    - g.v[j * nx + i])
                    / g.cell)
                    .abs()
            })
            .fold(0.0, f32::max);
        g.project(0.01);
        // At least 99.9% of the divergence removed (the solver stops at a
        // 1e-4 relative residual).
        assert!(
            g.stats.max_divergence < 1e-3 * before,
            "max ∇·u {} (was {before})",
            g.stats.max_divergence
        );
    }

    #[test]
    fn sampling_reproduces_a_linear_field() {
        let (w, h) = (5, 4);
        let field: Vec<f32> = (0..w * h)
            .map(|k| (k % w) as f32 * 2.0 + (k / w) as f32)
            .collect();
        // Sample (i, j) at (i + 0.5, j + 0.5): value at (2.25, 1.75) is
        // 2 (1.75) + 1.25 = 4.75.
        let v = Gas::sample(&field, w, h, 0.5, 0.5, 2.25, 1.75);
        assert!((v - 4.75).abs() < 1e-5, "{v}");
    }
}
