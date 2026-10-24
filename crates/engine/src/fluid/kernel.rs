//! SPH smoothing kernel.
//!
//! SPH estimates any field at a point as a weighted sum over nearby
//! particles. The weight is the *kernel* W(r), a smooth bump that integrates
//! to 1 and falls to zero at the *support radius* h.
//!
//! We use the 2D cubic spline (Monaghan 1992), the kernel used in the weakly
//! compressible SPH paper this solver follows (Becker & Teschner 2007) and in
//! SPlisHSPlasH. With q = r / h:
//!
//! ```text
//! W(q) = σ · { 6(q³ − q²) + 1     0 ≤ q ≤ ½
//!            { 2(1 − q)³          ½ < q ≤ 1
//!            { 0                  q > 1
//! σ = 40 / (7π h²)
//! ```

use glam::Vec2;
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
            sigma: 40.0 / (7.0 * PI * support * support),
        }
    }

    pub fn support(&self) -> f32 {
        self.support
    }

    /// W(r), units 1/m².
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

    /// ∇ᵢ W(xᵢ − xⱼ) for `r_vec` = xᵢ − xⱼ and `r` = |r_vec|. Points from i
    /// toward j (the kernel decreases with distance), units 1/m³.
    pub fn grad(&self, r_vec: Vec2, r: f32) -> Vec2 {
        if r <= 1e-9 || r >= self.support {
            return Vec2::ZERO;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// ∫ W dA = 1, checked by summing over a fine grid.
    #[test]
    fn kernel_integrates_to_one() {
        let k = CubicSpline::new(0.1);
        let step = 0.001_f32;
        let n = (k.support() / step).ceil() as i32 + 1;
        let mut total = 0.0_f64;
        for i in -n..=n {
            for j in -n..=n {
                let r = Vec2::new(i as f32, j as f32).length() * step;
                total += f64::from(k.w(r)) * f64::from(step * step);
            }
        }
        assert!((total - 1.0).abs() < 1e-3, "∫W = {total}");
    }

    /// The analytic gradient matches a central finite difference.
    #[test]
    fn gradient_matches_finite_difference() {
        let k = CubicSpline::new(0.1);
        let eps = 1e-4;
        for &p in &[
            Vec2::new(0.01, 0.02),
            Vec2::new(0.04, -0.01),
            Vec2::new(-0.06, 0.05),
            Vec2::new(0.0, -0.09),
        ] {
            let g = k.grad(p, p.length());
            let fd = Vec2::new(
                (k.w((p + Vec2::X * eps).length()) - k.w((p - Vec2::X * eps).length()))
                    / (2.0 * eps),
                (k.w((p + Vec2::Y * eps).length()) - k.w((p - Vec2::Y * eps).length()))
                    / (2.0 * eps),
            );
            assert!(
                (g - fd).length() < 1e-2 * fd.length().max(1.0),
                "at {p}: {g} vs {fd}"
            );
        }
    }

    #[test]
    fn kernel_is_zero_outside_support() {
        let k = CubicSpline::new(0.1);
        assert_eq!(k.w(0.1001), 0.0);
        assert_eq!(k.grad(Vec2::new(0.2, 0.0), 0.2), Vec2::ZERO);
        assert!(k.w(0.0) > k.w(0.05));
    }
}
