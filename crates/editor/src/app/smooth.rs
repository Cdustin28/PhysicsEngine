//! Liquids in the 2D view drawn as one smooth body instead of dots: each
//! particle adds a soft bump to a density field on a fine grid, and the
//! liquid is wherever that field is above half its value deep inside
//! (a metaball surface). Colors blend the same way, so two liquids meet in
//! a soft line, and a thin lighter band traces the free surface.

use eframe::egui::{self, Color32};
use glam::Vec2;

/// Grid cells per particle spacing.
const CELLS_PER_SPACING: f32 = 3.0;
/// Bump radius in particle spacings.
const REACH: f32 = 1.5;
/// Largest image side, pixels (the cells grow past this).
const MAX_SIDE: usize = 1024;

/// The liquid as an image and the world rectangle it covers (min, max).
pub struct SmoothImage {
    pub image: egui::ColorImage,
    pub min: Vec2,
    pub max: Vec2,
}

/// Rasterize the particles at `positions` (with their `colors`) into a
/// smooth image. `None` when there are no particles.
pub fn rasterize(positions: &[Vec2], colors: &[Color32], spacing: f32) -> Option<SmoothImage> {
    let first = *positions.first()?;
    let (mut lo, mut hi) = (first, first);
    for &p in positions {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let reach = REACH * spacing;
    lo -= Vec2::splat(reach);
    hi += Vec2::splat(reach);
    let extent = hi - lo;
    let cell = (spacing / CELLS_PER_SPACING).max(extent.max_element() / MAX_SIDE as f32);
    let nx = ((extent.x / cell).ceil() as usize).max(1);
    let ny = ((extent.y / cell).ceil() as usize).max(1);

    let mut density = vec![0.0f32; nx * ny];
    let mut color = vec![[0.0f32; 3]; nx * ny];
    let r_cells = (reach / cell).ceil() as i64;
    let inv_r2 = 1.0 / (reach * reach);
    for (&p, c) in positions.iter().zip(colors) {
        let rgb = [f32::from(c.r()), f32::from(c.g()), f32::from(c.b())];
        let gx = ((p.x - lo.x) / cell) as i64;
        let gy = ((p.y - lo.y) / cell) as i64;
        for j in (gy - r_cells).max(0)..=(gy + r_cells).min(ny as i64 - 1) {
            let y = lo.y + (j as f32 + 0.5) * cell - p.y;
            for i in (gx - r_cells).max(0)..=(gx + r_cells).min(nx as i64 - 1) {
                let x = lo.x + (i as f32 + 0.5) * cell - p.x;
                let q = 1.0 - (x * x + y * y) * inv_r2;
                if q <= 0.0 {
                    continue;
                }
                let w = q * q * q;
                let k = j as usize * nx + i as usize;
                density[k] += w;
                for (sum, v) in color[k].iter_mut().zip(rgb) {
                    *sum += w * v;
                }
            }
        }
    }

    // Deep inside, with one particle per spacing², the field is the bump's
    // area over the area per particle: (πR²/4)/s².
    let interior = std::f32::consts::FRAC_PI_4 * REACH * REACH;
    let iso = 0.5 * interior;
    let soft = 0.12 * interior;
    let mut pixels = vec![Color32::TRANSPARENT; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            let f = density[k];
            let alpha = smoothstep(iso - soft, iso + soft, f);
            if alpha <= 0.0 {
                continue;
            }
            // A lighter band just inside the edge reads as the surface.
            let rim = 1.0 - smoothstep(iso, iso + 3.0 * soft, f);
            let lift = 0.25 * rim;
            let [r, g, b] = color[k].map(|s| {
                let v = s / f;
                v + (255.0 - v) * lift
            });
            pixels[(ny - 1 - j) * nx + i] =
                Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, (alpha * 255.0) as u8);
        }
    }
    Some(SmoothImage {
        image: egui::ColorImage::new([nx, ny], pixels),
        min: lo,
        max: lo + Vec2::new(nx as f32, ny as f32) * cell,
    })
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_block_fills_its_area() {
        let s = 0.05;
        let mut positions = Vec::new();
        for j in 0..20 {
            for i in 0..20 {
                positions.push(Vec2::new(i as f32 * s, j as f32 * s));
            }
        }
        let colors = vec![Color32::from_rgb(40, 90, 200); positions.len()];
        let img = rasterize(&positions, &colors, s).unwrap();
        let [nx, ny] = img.image.size;
        let cell = (img.max.x - img.min.x) / nx as f32;
        let covered: f32 = img
            .image
            .pixels
            .iter()
            .map(|p| f32::from(p.a()) / 255.0)
            .sum();
        let area = covered * cell * cell;
        // 400 particles stand for 400 s² = 1 m² of liquid.
        let expected = 400.0 * s * s;
        assert!(
            (area / expected - 1.0).abs() < 0.1,
            "area {area}, expected {expected}"
        );
        // The middle is opaque and keeps the color.
        let mid = img.image.pixels[(ny / 2) * nx + nx / 2];
        assert_eq!(mid.a(), 255);
        assert!(mid.b() > 190);
        assert!(ny > 0);
    }

    #[test]
    fn nothing_to_draw() {
        assert!(rasterize(&[], &[], 0.1).is_none());
    }
}
