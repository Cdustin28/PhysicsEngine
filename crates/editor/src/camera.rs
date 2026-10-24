//! Maps between world space (meters, +y up) and screen space (pixels, +y down).

use eframe::egui::{Pos2, Rect, vec2};
use glam::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// World point shown at the center of the viewport, m.
    pub center: Vec2,
    /// Zoom level.
    pub pixels_per_meter: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: Vec2::new(0.0, 3.0),
            pixels_per_meter: 60.0,
        }
    }
}

impl Camera {
    pub const MIN_ZOOM: f32 = 2.0;
    pub const MAX_ZOOM: f32 = 2000.0;

    pub fn world_to_screen(&self, viewport: Rect, p: Vec2) -> Pos2 {
        let d = (p - self.center) * self.pixels_per_meter;
        viewport.center() + vec2(d.x, -d.y)
    }

    pub fn screen_to_world(&self, viewport: Rect, p: Pos2) -> Vec2 {
        let d = p - viewport.center();
        self.center + Vec2::new(d.x, -d.y) / self.pixels_per_meter
    }

    /// Convert a length in meters to pixels.
    pub fn scale(&self, meters: f32) -> f32 {
        meters * self.pixels_per_meter
    }

    /// Pan by a screen-space drag delta (pixels).
    pub fn pan_pixels(&mut self, delta: eframe::egui::Vec2) {
        self.center -= Vec2::new(delta.x, -delta.y) / self.pixels_per_meter;
    }

    /// Zoom by `factor`, keeping the world point under `anchor` fixed on
    /// screen so zooming feels like it happens "at the cursor".
    pub fn zoom_about(&mut self, viewport: Rect, anchor: Pos2, factor: f32) {
        let before = self.screen_to_world(viewport, anchor);
        self.pixels_per_meter =
            (self.pixels_per_meter * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.screen_to_world(viewport, anchor);
        self.center += before - after;
    }

    /// World-space rectangle currently visible, as (min, max) corners.
    pub fn visible_world_bounds(&self, viewport: Rect) -> (Vec2, Vec2) {
        let a = self.screen_to_world(viewport, viewport.left_bottom());
        let b = self.screen_to_world(viewport, viewport.right_top());
        (a.min(b), a.max(b))
    }

    /// A "nice" grid spacing (1, 2, or 5 × 10ⁿ meters) that keeps grid lines
    /// roughly `target_px` pixels apart at the current zoom.
    pub fn grid_spacing(&self, target_px: f32) -> f32 {
        let raw = target_px / self.pixels_per_meter;
        let pow10 = 10f32.powf(raw.log10().floor());
        let mantissa = raw / pow10;
        let nice = if mantissa < 2.0 {
            1.0
        } else if mantissa < 5.0 {
            2.0
        } else {
            5.0
        };
        nice * pow10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::pos2;

    fn viewport() -> Rect {
        Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0))
    }

    #[test]
    fn round_trip() {
        let cam = Camera::default();
        let p = Vec2::new(1.25, -3.5);
        let back = cam.screen_to_world(viewport(), cam.world_to_screen(viewport(), p));
        assert!((back - p).length() < 1e-4);
    }

    #[test]
    fn y_axis_points_up_on_screen() {
        let cam = Camera::default();
        let low = cam.world_to_screen(viewport(), Vec2::new(0.0, 0.0));
        let high = cam.world_to_screen(viewport(), Vec2::new(0.0, 1.0));
        assert!(high.y < low.y);
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut cam = Camera::default();
        let anchor = pos2(123.0, 456.0);
        let before = cam.screen_to_world(viewport(), anchor);
        cam.zoom_about(viewport(), anchor, 1.7);
        let after = cam.screen_to_world(viewport(), anchor);
        assert!((before - after).length() < 1e-4);
    }

    #[test]
    fn grid_spacing_is_nice() {
        let cam = Camera {
            pixels_per_meter: 60.0,
            ..Default::default()
        };
        // 80 px / 60 px/m = 1.33 m -> 1 m
        assert_eq!(cam.grid_spacing(80.0), 1.0);
        let cam = Camera {
            pixels_per_meter: 20.0,
            ..Default::default()
        };
        // 80 / 20 = 4 m -> 2 m
        assert_eq!(cam.grid_spacing(80.0), 2.0);
    }
}
