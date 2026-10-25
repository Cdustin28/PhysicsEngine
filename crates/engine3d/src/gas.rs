//! Smoke, dye, and heat in 3D: the 2D engine's grid gas solver (see the
//! `engine` crate's `gas` module for the method) on a 3D staggered grid.
//! Velocity components live on the faces normal to their axis; smoke and
//! temperature at cell centers. Advection is semi-Lagrangian with a midpoint
//! backtrace and trilinear interpolation; heat diffuses implicitly; the
//! pressure projection is solved with preconditioned conjugate gradients.

use crate::body::RigidBody;
use crate::broad_phase::Aabb;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GasParams {
    /// K
    pub ambient_temperature: f32,
    /// κ, m²/s
    pub thermal_diffusivity: f32,
    /// How much heavier than air smoke of concentration 1 is (fraction).
    pub smoke_weight: f32,
    /// 1/s
    pub smoke_decay: f32,
    /// 1/s
    pub cooling: f32,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Emitter {
    pub center: Vec3,
    /// m
    pub radius: f32,
    /// Smoke concentration added per second.
    pub smoke_rate: f32,
    /// K
    pub temperature: f32,
    #[serde(default)]
    pub velocity: Vec3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gas {
    /// World position of the grid's minimum corner, m.
    pub origin: Vec3,
    /// Cube cell side, m.
    pub cell: f32,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    #[serde(default)]
    pub params: GasParams,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub emitters: Vec<Emitter>,
    /// x velocity on x faces, (nx + 1) · ny · nz.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub u: Vec<f32>,
    /// y velocity on y faces, nx · (ny + 1) · nz.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub v: Vec<f32>,
    /// z velocity on z faces, nx · ny · (nz + 1).
    #[serde(default, skip_serializing_if = "all_zero")]
    pub w: Vec<f32>,
    #[serde(default, skip_serializing_if = "all_zero")]
    pub smoke: Vec<f32>,
    /// Temperature above ambient, K.
    #[serde(default, skip_serializing_if = "all_zero")]
    pub heat: Vec<f32>,
    #[serde(skip)]
    solid: Vec<bool>,
    #[serde(skip)]
    solid_velocity: Vec<Vec3>,
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
            && (self.nx, self.ny, self.nz) == (other.nx, other.ny, other.nz)
            && self.params == other.params
            && self.emitters == other.emitters
            && self.u == other.u
            && self.v == other.v
            && self.w == other.w
            && self.smoke == other.smoke
            && self.heat == other.heat
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GasStats {
    pub substeps: u32,
    pub pressure_iterations: u32,
    pub max_divergence: f32,
}

const PRESSURE_TOLERANCE: f64 = 1e-4;
const DIFFUSION_TOLERANCE: f64 = 1e-6;
const MAX_SOLVER_ITERATIONS: usize = 2000;

#[derive(Clone, Copy, PartialEq)]
enum Side {
    Fluid(usize),
    Wall,
    Open,
}

impl Gas {
    pub fn new(origin: Vec3, cell: f32, nx: usize, ny: usize, nz: usize) -> Self {
        let mut g = Self {
            origin,
            cell,
            nx: nx.max(2),
            ny: ny.max(2),
            nz: nz.max(2),
            params: GasParams::default(),
            emitters: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            w: Vec::new(),
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

    pub fn fill_missing(&mut self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n = nx * ny * nz;
        self.u.resize((nx + 1) * ny * nz, 0.0);
        self.v.resize(nx * (ny + 1) * nz, 0.0);
        self.w.resize(nx * ny * (nz + 1), 0.0);
        self.smoke.resize(n, 0.0);
        self.heat.resize(n, 0.0);
        self.solid.resize(n, false);
        self.solid_velocity.resize(n, Vec3::ZERO);
        self.pressure.resize(n, 0.0);
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.cell > 0.0 && self.cell.is_finite() && self.origin.is_finite()) {
            return Err("gas grid needs a positive cell size".into());
        }
        let n = self.nx * self.ny * self.nz;
        if self.nx < 2 || self.ny < 2 || self.nz < 2 || n > 8_000_000 {
            return Err("gas grid size is out of range".into());
        }
        let sized = |v: &[f32], len| v.is_empty() || v.len() == len;
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        if !(sized(&self.u, (nx + 1) * ny * nz)
            && sized(&self.v, nx * (ny + 1) * nz)
            && sized(&self.w, nx * ny * (nz + 1))
            && sized(&self.smoke, n)
            && sized(&self.heat, n))
        {
            return Err("gas field sizes don't match the grid".into());
        }
        if !(self.params.ambient_temperature > 0.0 && self.params.thermal_diffusivity >= 0.0) {
            return Err("gas ambient temperature must be positive".into());
        }
        Ok(())
    }

    pub fn size(&self) -> Vec3 {
        Vec3::new(self.nx as f32, self.ny as f32, self.nz as f32) * self.cell
    }

    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    fn ui(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * (self.nx + 1) + i
    }

    fn vi(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * self.nx + i
    }

    fn wi(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * self.cell
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        self.solid.get(self.idx(i, j, k)).copied().unwrap_or(false)
    }

    pub fn velocity_at_cell(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            0.5 * (self.u[self.ui(i, j, k)] + self.u[self.ui(i + 1, j, k)]),
            0.5 * (self.v[self.vi(i, j, k)] + self.v[self.vi(i, j + 1, k)]),
            0.5 * (self.w[self.wi(i, j, k)] + self.w[self.wi(i, j, k + 1)]),
        )
    }

    pub fn total_heat(&self) -> f32 {
        self.heat.iter().sum::<f32>() * self.cell.powi(3)
    }

    pub fn paint(&mut self, center: Vec3, radius: f32, smoke: f32, temperature: f32) {
        self.fill_missing();
        let dt = temperature - self.params.ambient_temperature;
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    if (self.cell_center(i, j, k) - center).length() <= radius {
                        let c = self.idx(i, j, k);
                        self.smoke[c] = self.smoke[c].max(smoke);
                        if dt > 0.0 {
                            self.heat[c] = self.heat[c].max(dt);
                        }
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        for f in [
            &mut self.u,
            &mut self.v,
            &mut self.w,
            &mut self.smoke,
            &mut self.heat,
        ] {
            f.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    pub fn step(&mut self, bodies: &[RigidBody], gravity: Vec3, dt: f32) {
        self.fill_missing();
        self.update_solids(bodies);
        let vmax = self
            .u
            .iter()
            .chain(&self.v)
            .chain(&self.w)
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
            let lo = ((bb.min - self.origin) / self.cell).floor().max(Vec3::ZERO);
            let hi = ((bb.max - self.origin) / self.cell).ceil();
            let hi = [
                (hi.x.max(0.0) as usize).min(self.nx),
                (hi.y.max(0.0) as usize).min(self.ny),
                (hi.z.max(0.0) as usize).min(self.nz),
            ];
            for k in lo.z as usize..hi[2] {
                for j in lo.y as usize..hi[1] {
                    for i in lo.x as usize..hi[0] {
                        let c = self.cell_center(i, j, k);
                        if body.contains_point(c) {
                            let n = self.idx(i, j, k);
                            self.solid[n] = true;
                            self.solid_velocity[n] = body.velocity_at(c);
                            self.smoke[n] = 0.0;
                            self.heat[n] = 0.0;
                        }
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
            for k in 0..self.nz {
                for j in 0..self.ny {
                    for i in 0..self.nx {
                        let c = self.idx(i, j, k);
                        if self.solid[c] || (self.cell_center(i, j, k) - e.center).length() > r {
                            continue;
                        }
                        self.smoke[c] += e.smoke_rate * h;
                        if dt > 0.0 {
                            self.heat[c] = self.heat[c].max(dt);
                        }
                        if e.velocity != Vec3::ZERO {
                            let (a, b) = (self.ui(i, j, k), self.ui(i + 1, j, k));
                            self.u[a] = e.velocity.x;
                            self.u[b] = e.velocity.x;
                            let (a, b) = (self.vi(i, j, k), self.vi(i, j + 1, k));
                            self.v[a] = e.velocity.y;
                            self.v[b] = e.velocity.y;
                            let (a, b) = (self.wi(i, j, k), self.wi(i, j, k + 1));
                            self.w[a] = e.velocity.z;
                            self.w[b] = e.velocity.z;
                        }
                    }
                }
            }
        }
        self.emitters = emitters;
    }

    fn apply_buoyancy(&mut self, gravity: Vec3, h: f32) {
        let beta = 1.0 / self.params.ambient_temperature;
        let alpha = self.params.smoke_weight;
        let b = |g: &Gas, c: usize| beta * g.heat[c] - alpha * g.smoke[c];
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    if i > 0 && gravity.x != 0.0 {
                        let f = 0.5 * (b(self, self.idx(i - 1, j, k)) + b(self, self.idx(i, j, k)));
                        let f_idx = self.ui(i, j, k);
                        self.u[f_idx] += -gravity.x * f * h;
                    }
                    if j > 0 && gravity.y != 0.0 {
                        let f = 0.5 * (b(self, self.idx(i, j - 1, k)) + b(self, self.idx(i, j, k)));
                        let f_idx = self.vi(i, j, k);
                        self.v[f_idx] += -gravity.y * f * h;
                    }
                    if k > 0 && gravity.z != 0.0 {
                        let f = 0.5 * (b(self, self.idx(i, j, k - 1)) + b(self, self.idx(i, j, k)));
                        let f_idx = self.wi(i, j, k);
                        self.w[f_idx] += -gravity.z * f * h;
                    }
                }
            }
        }
    }

    fn enforce_boundaries(&mut self) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let f = self.ui(i, j, k);
                    if i == 0 || i == nx {
                        self.u[f] = 0.0;
                    } else {
                        let (a, b) = (self.idx(i - 1, j, k), self.idx(i, j, k));
                        if self.solid[a] || self.solid[b] {
                            let s = if self.solid[a] { a } else { b };
                            self.u[f] = self.solid_velocity[s].x;
                        }
                    }
                }
            }
        }
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let f = self.vi(i, j, k);
                    if j == 0 {
                        self.v[f] = 0.0;
                    } else if j == ny {
                        self.v[f] = if self.params.open_top {
                            self.v[f].max(0.0)
                        } else {
                            0.0
                        };
                    } else {
                        let (a, b) = (self.idx(i, j - 1, k), self.idx(i, j, k));
                        if self.solid[a] || self.solid[b] {
                            let s = if self.solid[a] { a } else { b };
                            self.v[f] = self.solid_velocity[s].y;
                        }
                    }
                }
            }
        }
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let f = self.wi(i, j, k);
                    if k == 0 || k == nz {
                        self.w[f] = 0.0;
                    } else {
                        let (a, b) = (self.idx(i, j, k - 1), self.idx(i, j, k));
                        if self.solid[a] || self.solid[b] {
                            let s = if self.solid[a] { a } else { b };
                            self.w[f] = self.solid_velocity[s].z;
                        }
                    }
                }
            }
        }
    }

    /// Trilinear sample of a staggered field of size (sx, sy, sz) whose
    /// sample (i, j, k) sits at grid coordinates (i, j, k) + offset.
    #[allow(clippy::too_many_arguments)]
    fn sample(field: &[f32], sx: usize, sy: usize, sz: usize, off: Vec3, p: Vec3) -> f32 {
        let q = p - off;
        let x = q.x.clamp(0.0, (sx - 1) as f32);
        let y = q.y.clamp(0.0, (sy - 1) as f32);
        let z = q.z.clamp(0.0, (sz - 1) as f32);
        let (i0, j0, k0) = (
            (x as usize).min(sx - 2),
            (y as usize).min(sy - 2),
            (z as usize).min(sz - 2),
        );
        let (fx, fy, fz) = (x - i0 as f32, y - j0 as f32, z - k0 as f32);
        let at = |i: usize, j: usize, k: usize| field[(k * sy + j) * sx + i];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(i0, j0, k0), at(i0 + 1, j0, k0), fx);
        let c10 = lerp(at(i0, j0 + 1, k0), at(i0 + 1, j0 + 1, k0), fx);
        let c01 = lerp(at(i0, j0, k0 + 1), at(i0 + 1, j0, k0 + 1), fx);
        let c11 = lerp(at(i0, j0 + 1, k0 + 1), at(i0 + 1, j0 + 1, k0 + 1), fx);
        lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
    }

    fn velocity_grid(&self, p: Vec3) -> Vec3 {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        Vec3::new(
            Self::sample(&self.u, nx + 1, ny, nz, Vec3::new(0.0, 0.5, 0.5), p),
            Self::sample(&self.v, nx, ny + 1, nz, Vec3::new(0.5, 0.0, 0.5), p),
            Self::sample(&self.w, nx, ny, nz + 1, Vec3::new(0.5, 0.5, 0.0), p),
        )
    }

    fn backtrace(&self, p: Vec3, h: f32) -> Vec3 {
        let s = h / self.cell;
        let mid = p - 0.5 * s * self.velocity_grid(p);
        p - s * self.velocity_grid(mid)
    }

    fn advect_velocity(&mut self, h: f32) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (mut u, mut v, mut w) = (self.u.clone(), self.v.clone(), self.w.clone());
        let this = &*self;
        let open_top = self.params.open_top;
        // Every face looks back along the flow independently, so each
        // field is filled in parallel on desktop. Wall faces keep their
        // value (enforce_boundaries sets them).
        fill32(&mut u, |f| {
            let (i, j, k) = (f % (nx + 1), (f / (nx + 1)) % ny, f / ((nx + 1) * ny));
            if i == 0 || i == nx {
                return this.u[f];
            }
            let p = this.backtrace(Vec3::new(i as f32, j as f32 + 0.5, k as f32 + 0.5), h);
            Self::sample(&this.u, nx + 1, ny, nz, Vec3::new(0.0, 0.5, 0.5), p)
        });
        fill32(&mut v, |f| {
            let (i, j, k) = (f % nx, (f / nx) % (ny + 1), f / (nx * (ny + 1)));
            if j == 0 || (j == ny && !open_top) {
                return this.v[f];
            }
            let p = this.backtrace(Vec3::new(i as f32 + 0.5, j as f32, k as f32 + 0.5), h);
            Self::sample(&this.v, nx, ny + 1, nz, Vec3::new(0.5, 0.0, 0.5), p)
        });
        fill32(&mut w, |f| {
            let (i, j, k) = (f % nx, (f / nx) % ny, f / (nx * ny));
            if k == 0 || k == nz {
                return this.w[f];
            }
            let p = this.backtrace(Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32), h);
            Self::sample(&this.w, nx, ny, nz + 1, Vec3::new(0.5, 0.5, 0.0), p)
        });
        self.u = u;
        self.v = v;
        self.w = w;
        self.enforce_boundaries();
    }

    fn advect_scalars(&mut self, h: f32) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let off = Vec3::splat(0.5);
        let this = &*self;
        // Both fields are sampled at the same backtraced point.
        let mut points = vec![Vec3::ZERO; self.smoke.len()];
        fill_vec3(&mut points, |c| {
            let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
            this.backtrace(Vec3::new(i as f32, j as f32, k as f32) + off, h)
        });
        let (mut smoke, mut heat) = (self.smoke.clone(), self.heat.clone());
        fill32(&mut smoke, |c| {
            if this.solid[c] {
                this.smoke[c]
            } else {
                Self::sample(&this.smoke, nx, ny, nz, off, points[c])
            }
        });
        fill32(&mut heat, |c| {
            if this.solid[c] {
                this.heat[c]
            } else {
                Self::sample(&this.heat, nx, ny, nz, off, points[c])
            }
        });
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

    fn sides(&self, c: usize, pressure: bool) -> [Side; 6] {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
        let cell = |ii: usize, jj: usize, kk: usize| {
            let m = self.idx(ii, jj, kk);
            if self.solid[m] {
                Side::Wall
            } else {
                Side::Fluid(m)
            }
        };
        [
            if i > 0 { cell(i - 1, j, k) } else { Side::Wall },
            if i + 1 < nx {
                cell(i + 1, j, k)
            } else {
                Side::Wall
            },
            if j > 0 { cell(i, j - 1, k) } else { Side::Wall },
            if j + 1 < ny {
                cell(i, j + 1, k)
            } else if pressure && self.params.open_top {
                Side::Open
            } else {
                Side::Wall
            },
            if k > 0 { cell(i, j, k - 1) } else { Side::Wall },
            if k + 1 < nz {
                cell(i, j, k + 1)
            } else {
                Side::Wall
            },
        ]
    }

    fn diffuse_heat(&mut self, h: f32) {
        let kappa = self.params.thermal_diffusivity;
        if kappa <= 0.0 {
            return;
        }
        let c = f64::from(h * kappa / (self.cell * self.cell));
        let n = self.heat.len();
        let fluid: Vec<bool> = self.solid.iter().map(|s| !s).collect();
        let sides: Vec<[Side; 6]> = (0..n).map(|m| self.sides(m, false)).collect();
        let b: Vec<f64> = (0..n)
            .map(|m| {
                if fluid[m] {
                    f64::from(self.heat[m])
                } else {
                    0.0
                }
            })
            .collect();
        let apply = |x: &[f64], out: &mut [f64]| {
            fill(out, |m| {
                if !fluid[m] {
                    return x[m];
                }
                let mut lap = 0.0;
                for side in sides[m] {
                    if let Side::Fluid(q) = side {
                        lap += x[q] - x[m];
                    }
                }
                x[m] - c * lap
            });
        };
        let diag: Vec<f64> = (0..n)
            .map(|m| {
                if fluid[m] {
                    1.0 + c * sides[m]
                        .iter()
                        .filter(|s| matches!(s, Side::Fluid(_)))
                        .count() as f64
                } else {
                    1.0
                }
            })
            .collect();
        let mut x = b.clone();
        // Diagonally dominant: dividing by the diagonal is enough.
        let jacobi = |r: &[f64], z: &mut [f64]| fill(z, |k| r[k] / diag[k]);
        conjugate_gradient(&apply, &jacobi, &b, &mut x, DIFFUSION_TOLERANCE);
        for m in 0..n {
            if fluid[m] {
                self.heat[m] = x[m] as f32;
            }
        }
    }

    fn divergence(&self, i: usize, j: usize, k: usize) -> f32 {
        (self.u[self.ui(i + 1, j, k)] - self.u[self.ui(i, j, k)] + self.v[self.vi(i, j + 1, k)]
            - self.v[self.vi(i, j, k)]
            + self.w[self.wi(i, j, k + 1)]
            - self.w[self.wi(i, j, k)])
            / self.cell
    }

    fn project(&mut self, h: f32) {
        self.enforce_boundaries();
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n = nx * ny * nz;
        let dx = self.cell;
        let fluid: Vec<bool> = self.solid.iter().map(|s| !s).collect();
        let sides: Vec<[Side; 6]> = (0..n).map(|m| self.sides(m, true)).collect();
        let coords = |m: usize| (m % nx, (m / nx) % ny, m / (nx * ny));
        let mut b: Vec<f64> = (0..n)
            .map(|m| {
                if fluid[m] {
                    let (i, j, k) = coords(m);
                    -f64::from(self.divergence(i, j, k) * dx * dx)
                } else {
                    0.0
                }
            })
            .collect();
        let open = sides.iter().flatten().any(|s| *s == Side::Open);
        if !open {
            let count = fluid.iter().filter(|&&f| f).count().max(1) as f64;
            let mean = b.iter().sum::<f64>() / count;
            for m in 0..n {
                if fluid[m] {
                    b[m] -= mean;
                }
            }
        }
        let apply = |x: &[f64], out: &mut [f64]| {
            fill(out, |m| {
                if !fluid[m] {
                    return 0.0;
                }
                let mut sum = 0.0;
                for side in sides[m] {
                    match side {
                        Side::Fluid(q) => sum += x[m] - x[q],
                        Side::Open => sum += x[m],
                        Side::Wall => {}
                    }
                }
                sum
            });
        };
        let diag: Vec<f64> = (0..n)
            .map(|m| {
                sides[m]
                    .iter()
                    .filter(|s| !matches!(s, Side::Wall))
                    .count()
                    .max(1) as f64
            })
            .collect();
        let mut q: Vec<f64> = (0..n)
            .map(|m| {
                if fluid[m] {
                    f64::from(self.pressure[m] * h)
                } else {
                    0.0
                }
            })
            .collect();
        let mic = Mic::new(&sides, &fluid, &diag);
        let precondition = |r: &[f64], z: &mut [f64]| mic.apply(r, z);
        let iterations = conjugate_gradient(&apply, &precondition, &b, &mut q, PRESSURE_TOLERANCE);
        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    let (a, c) = (self.idx(i - 1, j, k), self.idx(i, j, k));
                    if fluid[a] && fluid[c] {
                        let f = self.ui(i, j, k);
                        self.u[f] -= ((q[c] - q[a]) as f32) / dx;
                    }
                }
            }
        }
        for k in 0..nz {
            for j in 1..=ny {
                for i in 0..nx {
                    let f = self.vi(i, j, k);
                    if j == ny {
                        let below = self.idx(i, ny - 1, k);
                        if self.params.open_top && fluid[below] {
                            self.v[f] += (q[below] as f32) / dx;
                        }
                        continue;
                    }
                    let (a, c) = (self.idx(i, j - 1, k), self.idx(i, j, k));
                    if fluid[a] && fluid[c] {
                        self.v[f] -= ((q[c] - q[a]) as f32) / dx;
                    }
                }
            }
        }
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (a, c) = (self.idx(i, j, k - 1), self.idx(i, j, k));
                    if fluid[a] && fluid[c] {
                        let f = self.wi(i, j, k);
                        self.w[f] -= ((q[c] - q[a]) as f32) / dx;
                    }
                }
            }
        }
        for (p, &qm) in self.pressure.iter_mut().zip(&q) {
            *p = (qm / f64::from(h)) as f32;
        }
        self.stats.pressure_iterations = iterations as u32;
        self.stats.max_divergence = (0..n)
            .filter(|&m| fluid[m])
            .map(|m| {
                let (i, j, k) = coords(m);
                self.divergence(i, j, k).abs()
            })
            .fold(0.0, f32::max);
    }
}

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
    plus: Vec<[(f64, usize); 3]>,
}

impl Mic {
    /// τ: how much of the dropped fill-in to compensate (0.97 is standard);
    /// σ: safety floor against tiny pivots.
    const TAU: f64 = 0.97;
    const SIGMA: f64 = 0.25;

    fn new(sides: &[[Side; 6]], fluid: &[bool], diag: &[f64]) -> Self {
        let n = fluid.len();
        let plus: Vec<[(f64, usize); 3]> = (0..n)
            .map(|c| {
                [sides[c][1], sides[c][3], sides[c][5]].map(|side| match side {
                    Side::Fluid(m) if fluid[c] => (-1.0, m),
                    _ => (0.0, c),
                })
            })
            .collect();
        // Cells before c that link to it, per axis: (coefficient, index).
        let mut minus: Vec<[(f64, usize); 3]> = vec![[(0.0, 0); 3]; n];
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

fn fill32(out: &mut [f32], f: impl Fn(usize) -> f32 + Sync) {
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

fn fill_vec3(out: &mut [Vec3], f: impl Fn(usize) -> Vec3 + Sync) {
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
    fn trilinear_sampling_reproduces_a_linear_field() {
        let (sx, sy, sz) = (4, 5, 3);
        let field: Vec<f32> = (0..sx * sy * sz)
            .map(|m| {
                let (i, j, k) = (m % sx, (m / sx) % sy, m / (sx * sy));
                i as f32 + 2.0 * j as f32 + 3.0 * k as f32
            })
            .collect();
        let v = Gas::sample(&field, sx, sy, sz, Vec3::ZERO, Vec3::new(1.25, 2.5, 0.75));
        assert!((v - (1.25 + 5.0 + 2.25)).abs() < 1e-5, "{v}");
    }
}
