//! Rigid bodies as seen by the fluid: a layer of boundary particles on each
//! body's surface, weighted by Akinci et al. (2012):
//! ψ_b = ρ₀ / Σ_k W(x_b − x_k). Same scheme as 2D (see the `engine` crate's
//! `fluid::boundary` for the full explanation), with two 3D additions:
//!
//! - Only the part of each surface near the fluid is sampled (`region`).
//! - Torques are vectors: r × F.

use super::grid::NeighborGrid;
use super::kernel::CubicSpline;
use crate::body::RigidBody;
use glam::Vec3;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Boundary {
    pub positions: Vec<Vec3>,
    /// Velocity of the body surface at each particle, m/s.
    pub velocities: Vec<Vec3>,
    /// Which body each particle belongs to (index into `World::bodies`).
    pub body: Vec<u32>,
    /// Center of mass of every body, for torques.
    pub body_centers: Vec<Vec3>,
    /// Akinci weight ψ (kg): the mass this particle stands in for.
    pub psi: Vec<f32>,
    pub grid: NeighborGrid,
    cell: f32,
}

impl Boundary {
    /// Sample the bodies' surfaces inside `region` with spacing `spacing`,
    /// `inset` inside the surface, and compute ψ.
    pub fn build(
        bodies: &[RigidBody],
        spacing: f32,
        inset: f32,
        kernel: &CubicSpline,
        rest_density: f32,
        region: (Vec3, Vec3),
    ) -> Self {
        let mut b = Boundary {
            cell: kernel.support(),
            ..Boundary::default()
        };
        let mut points = Vec::new();
        b.body_centers = bodies.iter().map(|body| body.position).collect();
        for (i, body) in bodies.iter().enumerate() {
            points.clear();
            body.sample_surface(spacing, inset, region, &mut points);
            b.velocities
                .extend(points.iter().map(|&p| body.velocity_at(p)));
            b.body.extend(std::iter::repeat_n(i as u32, points.len()));
            b.positions.extend_from_slice(&points);
        }
        b.grid.rebuild(&b.positions, b.cell);
        let h2 = kernel.support() * kernel.support();
        #[cfg(feature = "parallel")]
        let points = b.positions.par_iter();
        #[cfg(not(feature = "parallel"))]
        let points = b.positions.iter();
        b.psi = points
            .map(|&x| {
                let mut sum = 0.0;
                b.grid.for_each_candidate(x, |k| {
                    let d2 = (x - b.positions[k]).length_squared();
                    if d2 < h2 {
                        sum += kernel.w(d2.sqrt());
                    }
                });
                rest_density / sum
            })
            .collect();
        b
    }

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

    const EVERYWHERE: (Vec3, Vec3) = (Vec3::splat(-100.0), Vec3::splat(100.0));

    #[test]
    fn crowded_boundary_particles_weigh_less() {
        let d = 0.05;
        let k = CubicSpline::new(2.0 * d);
        let wall = RigidBody::new_static(Shape::cuboid(2.0, 0.5, 2.0), Vec3::ZERO);
        let alone = Boundary::build(std::slice::from_ref(&wall), d, 0.0, &k, 1000.0, EVERYWHERE);
        let doubled = Boundary::build(&[wall.clone(), wall], d, 0.0, &k, 1000.0, EVERYWHERE);
        let top = |b: &Boundary| {
            let i = b
                .positions
                .iter()
                .position(|p| p.x.abs() < 0.01 && p.z.abs() < 0.01 && p.y > 0.0)
                .unwrap();
            b.psi[i]
        };
        let ratio = top(&doubled) / top(&alone);
        assert!((ratio - 0.5).abs() < 0.01, "ratio {ratio}");
    }

    #[test]
    fn boundary_velocity_includes_rotation() {
        let k = CubicSpline::new(0.1);
        let mut ball = RigidBody::new(Shape::Sphere { radius: 1.0 }, Vec3::ZERO);
        ball.angular_velocity = Vec3::new(0.0, 2.0, 0.0);
        let b = Boundary::build(&[ball], 0.1, 0.0, &k, 1000.0, EVERYWHERE);
        for (p, v) in b.positions.iter().zip(&b.velocities) {
            let expected = Vec3::new(0.0, 2.0, 0.0).cross(*p);
            assert!((*v - expected).length() < 1e-5);
        }
    }
}
