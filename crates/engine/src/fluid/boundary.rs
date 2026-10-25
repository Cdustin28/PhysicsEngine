//! Rigid bodies as seen by the fluid: a layer of *boundary particles* on each
//! body's surface (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for
//! Incompressible SPH").
//!
//! Boundary particles count toward a nearby fluid particle's density, so
//! fluid pressure rises as it approaches a wall and pushes it back. A single
//! layer of particles has gaps and uneven spacing, especially at corners and
//! where bodies touch. Akinci's fix is to give each boundary particle a
//! weight ψ (a mass):
//!
//! ```text
//! ψ_b = ρ₀ / Σ_k W(x_b − x_k)     (sum over nearby boundary particles k)
//! ```
//!
//! Where boundary particles are crowded, each one counts for less. That
//! makes every wall look like the same amount of "stuff" to the fluid.
//!
//! Fluid comes to rest a little way off a layer of boundary particles
//! (about 0.5–0.6 particle spacings, measured with still water in a tank).
//! Sampled exactly on the surface, every body would therefore act about that
//! much bigger: floating bodies ride high and sunk ones rest on a film of
//! trapped water. The layer is sampled `inset` inside the surface to cancel
//! that, so the fluid's effective boundary is the body's real surface.

use super::grid::NeighborGrid;
use super::kernel::CubicSpline;
use crate::body::RigidBody;
use glam::Vec2;

#[derive(Debug, Clone, Default)]
pub struct Boundary {
    pub positions: Vec<Vec2>,
    /// Velocity of the body surface at each particle, m/s.
    pub velocities: Vec<Vec2>,
    /// Which body each particle belongs to (index into `World::bodies`).
    pub body: Vec<u32>,
    /// Center of mass of every body, for computing torques.
    pub body_centers: Vec<Vec2>,
    /// Akinci weight ψ: the mass this particle stands in for, in the same
    /// units as a fluid particle's mass.
    pub psi: Vec<f32>,
    pub grid: NeighborGrid,
    cell: f32,
}

impl Boundary {
    /// Sample every body's surface with spacing `spacing`, `inset` inside
    /// the surface, and compute ψ. The neighbor grid uses cells of
    /// `grid_cell` (at least the kernel support), so it can answer queries
    /// out to that radius.
    pub fn build(
        bodies: &[RigidBody],
        spacing: f32,
        inset: f32,
        kernel: &CubicSpline,
        rest_density: f32,
        grid_cell: f32,
    ) -> Self {
        let mut b = Boundary {
            cell: grid_cell.max(kernel.support()),
            ..Boundary::default()
        };
        let mut points = Vec::new();
        b.body_centers = bodies.iter().map(|body| body.position).collect();
        for (i, body) in bodies.iter().enumerate() {
            points.clear();
            body.sample_surface(spacing, inset, &mut points);
            b.velocities
                .extend(points.iter().map(|&p| body.velocity_at(p)));
            b.body.extend(std::iter::repeat_n(i as u32, points.len()));
            b.positions.extend_from_slice(&points);
        }
        b.grid.rebuild(&b.positions, b.cell);

        let h2 = kernel.support() * kernel.support();
        b.psi = b
            .positions
            .iter()
            .map(|&x| {
                let mut sum = 0.0;
                b.grid.for_each_candidate(x, |k| {
                    let d2 = (x - b.positions[k]).length_squared();
                    if d2 < h2 {
                        sum += kernel.w(d2.sqrt());
                    }
                });
                // `sum` includes the particle itself, so it is never zero.
                rest_density / sum
            })
            .collect();
        b
    }

    /// Side of the neighbor grid's cells, m.
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn crowded_boundary_particles_weigh_less() {
        let d = 0.05;
        let k = CubicSpline::new(2.0 * d);
        let wall = RigidBody::new_static(Shape::rect(4.0, 1.0), Vec2::ZERO);
        let alone = Boundary::build(std::slice::from_ref(&wall), d, 0.0, &k, 10.0, k.support());
        // Same wall with a second, overlapping copy: twice as many particles
        // in the same place, so each should weigh about half as much.
        let doubled = Boundary::build(&[wall.clone(), wall], d, 0.0, &k, 10.0, k.support());
        let mid = |b: &Boundary| {
            let i = b
                .positions
                .iter()
                .position(|p| p.x.abs() < 0.01 && p.y > 0.0)
                .unwrap();
            b.psi[i]
        };
        let ratio = mid(&doubled) / mid(&alone);
        assert!((ratio - 0.5).abs() < 0.01, "ratio {ratio}");
    }

    #[test]
    fn boundary_velocity_includes_rotation() {
        let d = 0.05;
        let k = CubicSpline::new(2.0 * d);
        let mut wheel = RigidBody::new(Shape::Circle { radius: 1.0 }, Vec2::ZERO);
        wheel.angular_velocity = 2.0;
        let b = Boundary::build(&[wheel], d, 0.0, &k, 10.0, k.support());
        for (p, v) in b.positions.iter().zip(&b.velocities) {
            // Rim speed ω r = 2 m/s, tangent to the circle.
            assert!((v.length() - 2.0).abs() < 1e-4);
            assert!(v.dot(*p).abs() < 1e-4);
        }
    }
}
