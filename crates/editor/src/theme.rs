//! Light and dark (dark grey) appearance.
//!
//! The theme is applied once to egui's visuals; drawing code that needs a
//! color that depends on it asks [`is_dark`] with the current `Ui` or
//! `Context`, so nothing has to be passed around.

use eframe::egui::{self, Color32, Visuals};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Theme {
    #[default]
    Light,
    Dark,
}

/// Dark grey surfaces, from the panels (lightest) to the canvas.
const PANEL: Color32 = Color32::from_rgb(0x2E, 0x2E, 0x30);
const WINDOW: Color32 = Color32::from_rgb(0x34, 0x34, 0x37);
const CANVAS: Color32 = Color32::from_rgb(0x24, 0x24, 0x26);
const FAINT: Color32 = Color32::from_rgb(0x38, 0x38, 0x3B);

impl Theme {
    pub fn visuals(self) -> Visuals {
        match self {
            Theme::Light => Visuals::light(),
            Theme::Dark => {
                let mut v = Visuals::dark();
                v.panel_fill = PANEL;
                v.window_fill = WINDOW;
                v.extreme_bg_color = CANVAS;
                v.faint_bg_color = FAINT;
                v.code_bg_color = FAINT;
                v.widgets.noninteractive.bg_fill = PANEL;
                v.widgets.noninteractive.weak_bg_fill = PANEL;
                v
            }
        }
    }

    /// Install both themes' visuals and switch to this one. (egui keeps a
    /// light and a dark style; choosing the theme explicitly also stops it
    /// following the operating system's setting.)
    pub fn apply(self, ctx: &egui::Context) {
        ctx.set_visuals_of(egui::Theme::Light, Theme::Light.visuals());
        ctx.set_visuals_of(egui::Theme::Dark, Theme::Dark.visuals());
        ctx.set_theme(match self {
            Theme::Light => egui::Theme::Light,
            Theme::Dark => egui::Theme::Dark,
        });
    }
}

/// Whether the dark theme is showing.
pub fn is_dark(ctx: &egui::Context) -> bool {
    ctx.global_style().visuals.dark_mode
}

/// Lines drawn over the scene (joints, ropes' outlines): near-black on the
/// light theme, near-white on the dark one.
pub fn ink(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(0xD8, 0xD8, 0xDC)
    } else {
        Color32::from_rgb(0x3C, 0x3C, 0x3C)
    }
}

/// Ropes: brown, lighter on the dark theme.
pub fn rope(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(0xC8, 0x94, 0x5E)
    } else {
        Color32::from_rgb(0x8B, 0x5A, 0x2B)
    }
}

/// Static bodies (floors, walls): mid grey, darker on the dark theme so
/// the floor doesn't glare.
pub fn static_fill(dark: bool, three_d: bool) -> Color32 {
    match (dark, three_d) {
        (false, false) => Color32::from_rgb(0x8A, 0x8D, 0x91),
        (false, true) => Color32::from_rgb(0xB4, 0xB6, 0xB9),
        (true, false) => Color32::from_rgb(0x5E, 0x61, 0x66),
        (true, true) => Color32::from_rgb(0x74, 0x77, 0x7C),
    }
}

/// Background of the 3D view.
pub fn sky(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(0x2A, 0x2B, 0x2E)
    } else {
        Color32::from_rgb(0xEE, 0xF1, 0xF5)
    }
}

/// Smoke's color: dark on the light theme, pale on the dark one (where dark
/// smoke would vanish into the background).
pub fn smoke_rgb(dark: bool) -> [u8; 3] {
    if dark { [205, 205, 210] } else { [60, 60, 64] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_theme_is_dark_grey_not_black() {
        let v = Theme::Dark.visuals();
        assert!(v.dark_mode);
        for c in [v.panel_fill, v.extreme_bg_color, v.window_fill] {
            assert!((0x20..0x40).contains(&c.r()), "{c:?}");
        }
        assert!(!Theme::Light.visuals().dark_mode);
    }
}
