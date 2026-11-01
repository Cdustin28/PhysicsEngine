//! Water changing phase, and heat crossing the liquid's surface:
//!
//! - **Boiling:** a water particle hotter than its boiling point at the
//!   pressure it's under (the surroundings plus the liquid above it, so
//!   deep water boils hotter, and water in vacuum boils at room
//!   temperature) turns the excess heat into vapor: m c (T − T_sat) = L dm.
//!   Bubbles aren't resolved; the vapor goes straight to the air above that
//!   spot (in vacuum it escapes).
//! - **Evaporation and condensation at the surface:** below boiling, a
//!   surface particle trades vapor with the air cell it's in at
//!   j = h_m (a ρ_sat(T) − ρ_v), the mass-transfer twin of natural
//!   convection (Lewis analogy), taking or giving the latent heat. Into
//!   vacuum, at most the Hertz–Knudsen rate.
//! - **Dew:** air touching a surface held colder than its dew point loses
//!   vapor onto it; once a particle's worth has collected, it appears as a
//!   drop of water on the surface.
//! - **Sensible heat** between the surface and the air (h = 10 W/m²·K), as
//!   before.
//!
//! Only water changes phase. Oil and mercury stop at their limits (oil's
//! smoke point, mercury's boiling point); any heat past that is counted in
//! [`WaterBudget::clamped_energy`].
//!
//! Every kilogram is accounted for: liquid (each particle's mass less what
//! has evaporated from it), vapor and mist in the air grid, dew waiting on
//! surfaces, and vapor vented (lost to vacuum or out of the grid).

use crate::body::RigidBody;
use crate::fluid::{Fluid, Liquid};
use crate::gas::Gas;
use glam::Vec3;
use materials::water;

/// Natural convection between the liquid surface and the air, W/(m²·K).
const SURFACE_H: f32 = 10.0;
/// Natural convection between a held surface and the air, W/(m²·K).
const WALL_H: f32 = 5.0;
/// Lewis number of water vapor in air to the 2/3 (Chilton–Colburn).
const LEWIS_FACTOR: f32 = 0.897;

/// Where the water that isn't liquid went.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WaterBudget {
    /// Vapor that escaped (into vacuum, or out past the air grid), kg.
    pub vented: f32,
    /// Water condensed on each body, waiting to become a drop, kg.
    pub dew: Vec<f32>,
    /// Heat beyond oil's and mercury's limits that went nowhere, J.
    pub clamped_energy: f64,
    /// Net latent heat taken from the liquid by evaporation and boiling
    /// (given back by condensation), J.
    pub latent: f64,
}

impl WaterBudget {
    /// Total dew waiting on surfaces, kg.
    pub fn total_dew(&self) -> f32 {
        self.dew.iter().sum()
    }

    /// Nothing recorded yet (left out of scene files then).
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Mass-transfer coefficient matching a heat-transfer coefficient h,
/// m/s.
fn mass_transfer(h: f32) -> f32 {
    h / (water::AIR_DENSITY * water::AIR_SPECIFIC_HEAT * LEWIS_FACTOR)
}

/// One step of phase change and surface heat exchange. `ambient_pressure`
/// is the pressure above the liquid (0 in vacuum), Pa; `wall_temperature`
/// the temperature walls without their own are held at, if any.
#[allow(clippy::too_many_arguments)]
pub fn step(
    fluid: &mut Fluid,
    mut gas: Option<&mut Gas>,
    bodies: &[RigidBody],
    gravity: Vec3,
    ambient_pressure: f32,
    wall_temperature: Option<f32>,
    budget: &mut WaterBudget,
    dt: f32,
) {
    budget.dew.resize(bodies.len(), 0.0);
    if let Some(gas) = gas.as_deref_mut() {
        gas.fill_missing();
        condense_on_surfaces(fluid, gas, bodies, wall_temperature, budget, dt);
    }
    if fluid.is_empty() {
        return;
    }
    fluid.fill_missing_velocities();
    let d = fluid.params.particle_spacing;
    let area = d * d;
    let pressures = hydrostatic_pressures(fluid, gravity);
    let surface: Vec<bool> = (0..fluid.len()).map(|i| fluid.is_surface(i)).collect();

    // The top of the liquid over each air column, where vapor from boiling
    // below comes out.
    let tops = gas.as_deref().map(|g| column_tops(fluid, g));

    for (i, &at_surface) in surface.iter().enumerate() {
        let liquid = fluid.liquid(i);
        let s = fluid.salinity[i];
        let m = fluid.mass(i);
        let cp = liquid.specific_heat(s);
        let capacity = m * cp;
        let mut t = fluid.temperatures[i] as f32;

        if liquid != Liquid::Water {
            let limit = liquid.max_temperature(s);
            if t > limit {
                budget.clamped_energy +=
                    f64::from(capacity) * (fluid.temperatures[i] - f64::from(limit));
                fluid.temperatures[i] = f64::from(limit);
                t = limit;
            }
        } else {
            let a = water::activity(s);
            let p_local = ambient_pressure + pressures[i];
            // Salt lowers the vapor pressure by a, so brine boils where
            // a · p_sat(T) reaches the pressure.
            let t_sat = water::saturation_temperature(p_local / a);
            let mut dm = 0.0;
            let mut into: Option<usize> = None;
            if t > t_sat {
                let l = water::latent_heat(t_sat);
                dm = capacity * (t - t_sat) / l;
                if at_surface {
                    // A surface can only lose molecules so fast.
                    dm = dm.min(water::vacuum_evaporation_flux(t, a) * area * dt);
                }
                into = gas.as_deref().and_then(|g| {
                    let tops = tops.as_ref()?;
                    let p = fluid.positions[i];
                    let (ci, ck) = column(g, p)?;
                    let top = tops[ck * g.nx + ci].max(p.y);
                    let j = (((top - g.origin.y) / g.cell) as usize + 1).min(g.ny - 1);
                    let c = g.idx(ci, j, ck);
                    (!g.is_solid(ci, j, ck)).then_some(c)
                });
            } else if at_surface
                && let Some(g) = gas.as_deref()
                && let Some(c) = g.cell_at(fluid.positions[i])
                && !g.vapor.is_empty()
            {
                // Evaporation (or condensation onto the liquid) into the
                // air cell it's in.
                let rho_s = a * water::saturation_vapor_density(t);
                let rho_v = g.vapor[c];
                let mut flux = mass_transfer(SURFACE_H) * (rho_s - rho_v);
                flux = flux.min(water::vacuum_evaporation_flux(t, a));
                dm = flux * area * dt;
                let volume = g.cell.powi(3);
                if dm < 0.0 {
                    // Never take more than half the excess from the cell.
                    dm = dm.max(-0.5 * (rho_v - rho_s).max(0.0) * volume);
                }
                into = Some(c);
            }
            if dm != 0.0 {
                let l = water::latent_heat(t);
                fluid.temperatures[i] -= f64::from(dm) * f64::from(l) / f64::from(capacity);
                t = fluid.temperatures[i] as f32;
                fluid.evaporated[i] += dm / m;
                budget.latent += f64::from(dm) * f64::from(l);
                match (into, gas.as_deref_mut()) {
                    (Some(c), Some(g)) => g.vapor[c] += dm / g.cell.powi(3),
                    _ => budget.vented += dm,
                }
            }
        }

        // Sensible heat between the surface and the air.
        if at_surface
            && let Some(g) = gas.as_deref_mut()
            && let Some(c) = g.cell_at(fluid.positions[i])
        {
            let cell_capacity = water::AIR_DENSITY * water::AIR_SPECIFIC_HEAT * g.cell.powi(3);
            let diff = g.temperature(c) - t;
            let equilibrium = diff * capacity * cell_capacity / (capacity + cell_capacity);
            let q = (SURFACE_H * area * diff * dt).clamp(-equilibrium.abs(), equilibrium.abs());
            fluid.temperatures[i] += f64::from(q / capacity);
            g.heat[c] -= q / cell_capacity;
        }
    }
    settle_evaporated(fluid);
}

/// The pressure of the liquid above each particle, Pa: the weight of the
/// particles stacked over it in its column (one particle spacing square),
/// plus half its own, per unit area. Unlike the SPH pressure this is right
/// from the first step (SPH pressure only builds up as the liquid
/// compresses under its weight) and doesn't carry the solver's noise.
fn hydrostatic_pressures(fluid: &Fluid, gravity: Vec3) -> Vec<f32> {
    let g = gravity.length();
    let up = if g > 0.0 { -gravity / g } else { Vec3::Y };
    let d = fluid.params.particle_spacing;
    // Columns: horizontal cells one spacing across.
    let side = up.any_orthonormal_vector();
    let other = up.cross(side);
    let mut columns: std::collections::HashMap<(i32, i32), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &p) in fluid.positions.iter().enumerate() {
        let key = (
            (p.dot(side) / d).floor() as i32,
            (p.dot(other) / d).floor() as i32,
        );
        columns.entry(key).or_default().push(i);
    }
    let mut out = vec![0.0; fluid.len()];
    for list in columns.values_mut() {
        list.sort_unstable_by(|&a, &b| {
            fluid.positions[b]
                .dot(up)
                .total_cmp(&fluid.positions[a].dot(up))
        });
        let mut above = 0.0;
        for &i in list.iter() {
            let m = fluid.liquid_mass(i).max(0.0);
            out[i] = g * (above + 0.5 * m) / (d * d);
            above += m;
        }
    }
    out
}

/// The air column (i, k) above `p`, if it's over the grid.
fn column(g: &Gas, p: Vec3) -> Option<(usize, usize)> {
    let rel = (p - g.origin) / g.cell;
    if rel.x < 0.0 || rel.z < 0.0 {
        return None;
    }
    let (i, k) = (rel.x as usize, rel.z as usize);
    (i < g.nx && k < g.nz).then_some((i, k))
}

/// Highest particle in each air column (nx × nz), or −∞.
fn column_tops(fluid: &Fluid, g: &Gas) -> Vec<f32> {
    let mut tops = vec![f32::NEG_INFINITY; g.nx * g.nz];
    for &p in &fluid.positions {
        if let Some((i, k)) = column(g, p) {
            let top = &mut tops[k * g.nx + i];
            *top = top.max(p.y);
        }
    }
    tops
}

/// Remove particles that have entirely evaporated (their salt, and any
/// overshoot, go to the nearest water particle), and add one beside any
/// particle that has gained a whole particle's worth by condensation.
fn settle_evaporated(fluid: &mut Fluid) {
    let d = fluid.params.particle_spacing;
    let gone: Vec<usize> = (0..fluid.len())
        .filter(|&i| fluid.evaporated[i] >= 1.0)
        .collect();
    for &i in &gone {
        let m = fluid.mass(i);
        let p = fluid.positions[i];
        let nearest = (0..fluid.len())
            .filter(|&j| j != i && fluid.liquid(j) == Liquid::Water && fluid.evaporated[j] < 1.0)
            .min_by(|&a, &b| {
                (fluid.positions[a] - p)
                    .length_squared()
                    .total_cmp(&(fluid.positions[b] - p).length_squared())
            });
        if let Some(j) = nearest {
            let mj = fluid.mass(j);
            let salt = fluid.salinity[i] * m;
            fluid.salinity[j] = ((fluid.salinity[j] * mj + salt) / mj).min(0.4);
            fluid.evaporated[j] += (fluid.evaporated[i] - 1.0) * m / mj;
        }
    }
    fluid.remove_particles(&gone);
    let n = fluid.len();
    for i in 0..n {
        if fluid.evaporated[i] <= -1.0 {
            fluid.evaporated[i] += 1.0;
            let p = fluid.positions[i] + Vec3::new(0.0, d, 0.0);
            let v = fluid.velocities[i];
            let t = fluid.temperatures[i];
            fluid.push_particle(p, v, Liquid::Water, 0.0, t);
        }
    }
}

/// Vapor condensing on (or dew evaporating from) bodies held at a
/// temperature, and sensible heat between them and the air.
fn condense_on_surfaces(
    fluid: &mut Fluid,
    gas: &mut Gas,
    bodies: &[RigidBody],
    wall_temperature: Option<f32>,
    budget: &mut WaterBudget,
    dt: f32,
) {
    let held: Vec<(usize, f32)> = bodies
        .iter()
        .enumerate()
        .filter_map(|(b, body)| {
            body.temperature
                .or(if body.is_static {
                    wall_temperature
                } else {
                    None
                })
                .map(|t| (b, t))
        })
        .collect();
    if held.is_empty() {
        return;
    }
    let cell = gas.cell;
    let face = cell * cell;
    let volume = cell.powi(3);
    let air_capacity = water::AIR_DENSITY * water::AIR_SPECIFIC_HEAT * volume;
    let h_m = mass_transfer(WALL_H);
    let drop_mass = fluid.params.particle_mass();
    for (b, t_wall) in held {
        let body = &bodies[b];
        let rho_wall = water::saturation_vapor_density(t_wall);
        for k in 0..gas.nz {
            for j in 0..gas.ny {
                for i in 0..gas.nx {
                    if gas.is_solid(i, j, k) {
                        continue;
                    }
                    let center = gas.cell_center(i, j, k);
                    let (surface, normal, inside) = body.closest_surface_point(center);
                    if inside || (surface - center).length() > cell {
                        continue;
                    }
                    let c = gas.idx(i, j, k);
                    // Sensible heat, stopping at the wall's temperature.
                    let diff = t_wall - gas.temperature(c);
                    let q = WALL_H * face * diff * dt;
                    gas.heat[c] += (q / air_capacity).clamp(-diff.abs(), diff.abs());
                    // Vapor onto the wall (dm > 0) or dew back into the air.
                    let excess = gas.vapor[c] - rho_wall;
                    let mut dm = h_m * excess * face * dt;
                    if dm > 0.0 {
                        dm = dm.min(0.5 * excess * volume);
                    } else {
                        dm = dm.max(-budget.dew[b]);
                    }
                    if dm == 0.0 {
                        continue;
                    }
                    gas.vapor[c] -= dm / volume;
                    budget.dew[b] += dm;
                    // A drop forms once there's a particle's worth.
                    if budget.dew[b] >= drop_mass {
                        budget.dew[b] -= drop_mass;
                        let at = surface + normal * (0.5 * fluid.params.particle_spacing);
                        fluid.push_particle(at, Vec3::ZERO, Liquid::Water, 0.0, f64::from(t_wall));
                    }
                }
            }
        }
    }
}
