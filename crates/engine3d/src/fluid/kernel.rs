//! SPH smoothing kernel: the cubic spline (Monaghan 1992), as in 2D, with the
//! 3D normalization. With q = r / h and h the support radius:
//!
//! ```text
//! W(q) = σ · { 6(q³ − q²) + 1     0 ≤ q ≤ ½
//!            { 2(1 − q)³          ½ < q ≤ 1
//!            { 0                  q > 1
//! σ = 8 / (π h³)
//! ```
//!
//! Check of σ: ∫ W dV = 4π h³ σ ∫₀¹ f(q) q² dq, with
//! - inner piece: ∫₀^½ (6q⁵ − 6q⁴ + q²) dq = 1/64 − 6/160 + 1/24 = 0.019792
//! - outer piece (t = 1 − q): 2 ∫₀^½ t³(1 − t)² dt = 2 (1/64 − 2/160 + 1/384)
//!   = 0.011458
//! - sum 0.031250 = 1/32, so σ = 32 / (4π h³) = 8 / (π h³).
//!
//! The `kernel_integrates_to_one` test checks it numerically.

use glam::Vec3;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicSpline {
    support: f32,
    inv_support: f32,
    sigma: f32,
}

impl CubicSpline {
    /// A kernel that reaches zero at distance `support`, m.
    pub fn new(support: f32) -> Self {
        Self {
            support,
            inv_support: 1.0 / support,
            sigma: 8.0 / (PI * support * support * support),
        }
    }

    pub fn support(&self) -> f32 {
        self.support
    }

    /// W(r), units 1/m³.
    pub fn w(&self, r: f32) -> f32 {
        let q = r * self.inv_support;
        if q <= 0.5 {
            self.sigma * (6.0 * (q * q * q - q * q) + 1.0)
        } else if q <= 1.0 {
            let t = 1.0 - q;
            self.sigma * 2.0 * t * t * t
        } else {
            0.0
        }
    }

    /// ∇ᵢ W(xᵢ − xⱼ) for `r_vec` = xᵢ − xⱼ and `r` = |r_vec|, units 1/m⁴.
    pub fn grad(&self, r_vec: Vec3, r: f32) -> Vec3 {
        if r <= 1e-9 || r >= self.support {
            return Vec3::ZERO;
        }
        let q = r * self.inv_support;
        let dw_dq = if q <= 0.5 {
            6.0 * (3.0 * q * q - 2.0 * q)
        } else {
            let t = 1.0 - q;
            -6.0 * t * t
        };
        r_vec * (self.sigma * dw_dq * self.inv_support / r)
    }
}

/// The cohesion force shape of Akinci et al. (2013), "Versatile surface
/// tension and adhesion for SPH fluids", as a function of q = r / h:
/// attractive (positive) for q > ½, repulsive close in, so particles hold
/// together without clumping.
pub fn cohesion_shape(q: f32) -> f32 {
    if q >= 1.0 || q <= 0.0 {
        0.0
    } else if q > 0.5 {
        let t = (1.0 - q) * q;
        t * t * t
    } else {
        let t = (1.0 - q) * q;
        2.0 * t * t * t - 1.0 / 64.0
    }
}

/// Normalization for cohesion: with a pair force F(r) = k V² s(r/h) / J
/// between particles of volume V, a flat surface of uniformly spaced
/// particles has surface tension exactly k (mean-field theory: σ =
/// −(n²/2) ∫_(z>0) z φ(r) dV for pair potential φ and number density n).
/// Here Ψ(r) = ∫ᵣʰ s(t/h) dt and J = (π/2) ∫₀ʰ r³ Ψ(r) dr.
pub fn cohesion_normalization(h: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 * h * h * h * h * h * unit_integral(3)
}

/// ∫₀¹ q^(d) S(q) dq with S(q) = ∫_q¹ s(u) du, by Simpson's rule (once).
fn unit_integral(d: i32) -> f32 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<f32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let n = 4000;
        let du = 1.0 / n as f64;
        // S on the grid, integrating s from the top down.
        let mut big_s = vec![0.0_f64; n + 1];
        for k in (0..n).rev() {
            let (a, b) = (k as f64 * du, (k + 1) as f64 * du);
            let mid = 0.5 * (a + b);
            let f = |q: f64| f64::from(cohesion_shape(q as f32));
            big_s[k] = big_s[k + 1] + du / 6.0 * (f(a) + 4.0 * f(mid) + f(b));
        }
        let mut sum = 0.0;
        for k in 0..n {
            let (a, b) = (k as f64 * du, (k + 1) as f64 * du);
            let g = |q: f64, s: f64| q.powi(d) * s;
            sum += 0.5 * du * (g(a, big_s[k]) + g(b, big_s[k + 1]));
        }
        sum as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ∫ W dV = 1, summed over a fine grid.
    #[test]
    fn kernel_integrates_to_one() {
        let k = CubicSpline::new(0.1);
        let step = 0.0025_f32;
        let n = (k.support() / step).ceil() as i32 + 1;
        let mut total = 0.0_f64;
        for i in -n..=n {
            for j in -n..=n {
                for l in -n..=n {
                    let r = Vec3::new(i as f32, j as f32, l as f32).length() * step;
                    total += f64::from(k.w(r)) * f64::from(step).powi(3);
                }
            }
        }
        assert!((total - 1.0).abs() < 2e-3, "∫W = {total}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let k = CubicSpline::new(0.1);
        let eps = 1e-4;
        for &p in &[
            Vec3::new(0.01, 0.02, -0.01),
            Vec3::new(0.04, -0.01, 0.03),
            Vec3::new(-0.06, 0.05, 0.0),
        ] {
            let g = k.grad(p, p.length());
            let fd = Vec3::new(
                k.w((p + Vec3::X * eps).length()) - k.w((p - Vec3::X * eps).length()),
                k.w((p + Vec3::Y * eps).length()) - k.w((p - Vec3::Y * eps).length()),
                k.w((p + Vec3::Z * eps).length()) - k.w((p - Vec3::Z * eps).length()),
            ) / (2.0 * eps);
            assert!(
                (g - fd).length() < 1e-2 * fd.length().max(1.0),
                "at {p}: {g} vs {fd}"
            );
        }
    }
}
