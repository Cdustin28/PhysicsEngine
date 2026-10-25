//! Orbit camera for the 3D view: circles a target point. Right-drag orbits,
//! Shift+right-drag or middle-drag pans, the scroll wheel zooms.

use eframe::egui::{self, Pos2, Rect};
use glam::{Mat4, Vec3, Vec4Swizzles as _, camera};
use serde::{Deserialize, Serialize};

/// Vertical field of view, radians (45°).
const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
const NEAR: f32 = 0.05;
const FAR: f32 = 500.0;
/// Radians of orbit per pixel of mouse movement.
const ORBIT_SPEED: f32 = 0.008;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OrbitCamera {
    /// The point the camera looks at and circles, m.
    pub target: Vec3,
    /// Angle around the vertical axis, radians.
    pub yaw: f32,
    /// Angle above the horizon, radians.
    pub pitch: f32,
    /// Distance from the target, m.
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 1.5, 0.0),
            yaw: 0.6,
            pitch: 0.45,
            distance: 16.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cp * sy, sp, cp * cy)
    }

    /// World → clip space for a viewport of the given size.
    pub fn view_proj(&self, rect: Rect) -> Mat4 {
        let aspect = (rect.width() / rect.height().max(1.0)).max(1e-3);
        // wgpu's depth range is [0, 1], the DirectX convention.
        camera::rh::proj::directx::perspective(FOV_Y, aspect, NEAR, FAR)
            * camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// The ray from the eye through screen point `p`: (origin, unit direction).
    pub fn ray(&self, rect: Rect, p: Pos2) -> (Vec3, Vec3) {
        let x = (p.x - rect.left()) / rect.width() * 2.0 - 1.0;
        let y = 1.0 - (p.y - rect.top()) / rect.height() * 2.0;
        let inv = self.view_proj(rect).inverse();
        let near = inv.project_point3(Vec3::new(x, y, 0.0));
        let far = inv.project_point3(Vec3::new(x, y, 1.0));
        (near, (far - near).normalize())
    }

    /// Screen position of a world point, or `None` if it's behind the camera.
    pub fn project(&self, rect: Rect, p: Vec3) -> Option<Pos2> {
        let clip = self.view_proj(rect) * p.extend(1.0);
        if clip.w <= NEAR {
            return None;
        }
        let ndc = clip.xyz() / clip.w;
        Some(Pos2::new(
            rect.left() + (ndc.x + 1.0) * 0.5 * rect.width(),
            rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(),
        ))
    }

    pub fn orbit(&mut self, delta: egui::Vec2) {
        self.yaw -= delta.x * ORBIT_SPEED;
        self.pitch = (self.pitch + delta.y * ORBIT_SPEED).clamp(-1.5, 1.5);
    }

    /// Slide the target so the scene follows the mouse.
    pub fn pan(&mut self, rect: Rect, delta: egui::Vec2) {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        // Meters per pixel at the target's distance.
        let scale = 2.0 * self.distance * (FOV_Y * 0.5).tan() / rect.height().max(1.0);
        self.target += (-delta.x * right + delta.y * up) * scale;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.5, 200.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_size(Pos2::new(10.0, 20.0), egui::vec2(800.0, 600.0))
    }

    #[test]
    fn center_ray_hits_target() {
        let cam = OrbitCamera::default();
        let (o, d) = cam.ray(rect(), rect().center());
        let to_target = (cam.target - o).normalize();
        assert!(d.dot(to_target) > 0.9999);
    }

    #[test]
    fn project_inverts_ray() {
        let cam = OrbitCamera::default();
        let p = Pos2::new(200.0, 450.0);
        let (o, d) = cam.ray(rect(), p);
        let back = cam.project(rect(), o + 7.0 * d).unwrap();
        assert!((back - p).length() < 0.05, "{back:?}");
    }
}
