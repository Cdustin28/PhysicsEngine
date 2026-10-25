//! The Orbits tab: a star, planets, and moons under N-body gravity
//! (`engine3d::orbits`), in real units.
//!
//! - `mod.rs` (this file): state, the clock, trails, editing, shortcuts
//! - `ui.rs`: toolbar, sidebar, status bar, and the 3D view
//!
//! The view works in AU relative to the body the camera follows, so `f32`
//! is precise wherever you're looking. Bodies are drawn at their true size
//! or a few pixels across, whichever is bigger, so they stay visible when
//! zoomed out and are true to scale up close.

mod ui;

use crate::view3d::orbit::{Limits, MAX_PITCH, OrbitCamera};
use eframe::egui::{self, Key};
use engine3d::orbits::{
    self, AU, Advance, DAY, Kind, M_EARTH, OrbitSpec, OrbitSystem, R_EARTH, YEAR,
};
use glam::{DVec3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Moon mass, kg.
const M_MOON: f64 = 7.342e22;
/// Points in a trail; spaced so a trail covers one orbit.
const TRAIL_POINTS: usize = 200;
/// Step budget per frame. Each step costs O(bodies²).
#[cfg(not(target_arch = "wasm32"))]
const MAX_STEPS_PER_FRAME: usize = 6000;
#[cfg(target_arch = "wasm32")]
const MAX_STEPS_PER_FRAME: usize = 2000;

/// Simulation speeds offered: (label, simulated seconds per real second).
const SPEEDS: [(&str, f64); 8] = [
    ("1 hour/s", 3600.0),
    ("6 hours/s", 6.0 * 3600.0),
    ("1 day/s", DAY),
    ("1 week/s", 7.0 * DAY),
    ("1 month/s", YEAR / 12.0),
    ("3 months/s", YEAR / 4.0),
    ("1 year/s", YEAR),
    ("10 years/s", 10.0 * YEAR),
];

/// The camera in AU, relative to the body it follows.
fn space_camera() -> OrbitCamera {
    OrbitCamera {
        target: Vec3::ZERO,
        yaw: 0.4,
        pitch: 0.5,
        distance: 45.0,
        limits: space_limits(),
    }
}

fn space_limits() -> Limits {
    Limits {
        min_distance: 1e-5,
        max_distance: 500.0,
        relative_clip: Some((0.01, 1e6)),
    }
}

/// Settings for the next planet added.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NewPlanet {
    distance_au: f64,
    mass_earths: f64,
    radius_earths: f64,
    color: [u8; 3],
}

impl Default for NewPlanet {
    fn default() -> Self {
        Self {
            distance_au: 1.5,
            mass_earths: 1.0,
            radius_earths: 1.0,
            color: [90, 160, 110],
        }
    }
}

/// Where a body has been, relative to its parent, so a trail traces the
/// orbit's shape even as the parent moves.
#[derive(Debug, Clone, Default)]
struct Trail {
    points: VecDeque<DVec3>,
    /// Simulation time of the last sample, s.
    last: f64,
    /// Time between samples, s: an orbit's period / [`TRAIL_POINTS`].
    spacing: f64,
}

/// Settings remembered between sessions.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedOrbits {
    system: Option<OrbitSystem>,
    camera: OrbitCamera,
    focus: usize,
    speed: usize,
    show_trails: bool,
    show_labels: bool,
    min_size_px: f32,
    top_down: bool,
    new_planet: NewPlanet,
}

impl Default for PersistedOrbits {
    fn default() -> Self {
        Self {
            system: None,
            camera: space_camera(),
            focus: 0,
            speed: 4,
            show_trails: true,
            show_labels: true,
            min_size_px: 5.0,
            top_down: false,
            new_planet: NewPlanet::default(),
        }
    }
}

pub struct OrbitEditor {
    system: OrbitSystem,
    /// The system when Play was pressed, restored by Reset.
    snapshot: Option<OrbitSystem>,
    running: bool,
    /// Index into [`SPEEDS`].
    speed: usize,
    camera: OrbitCamera,
    /// The body the camera follows.
    focus: usize,
    selected: Option<usize>,
    show_trails: bool,
    show_labels: bool,
    /// Smallest a planet is drawn, px (stars bigger, moons smaller).
    min_size_px: f32,
    /// Look straight down on the ecliptic; dragging only turns the view.
    top_down: bool,
    new_planet: NewPlanet,
    /// One per body, same order.
    trails: Vec<Trail>,
    /// Energy when Play was pressed, to show the integrator's error.
    energy_at_play: Option<f64>,
    /// Simulated seconds per real second achieved last frame.
    achieved_rate: f64,
    /// The step budget ran out last frame, so time ran slower than asked.
    rate_limited: bool,
    steps_last_frame: usize,
    /// Message for the app's status bar, taken each frame.
    status: Option<String>,
    gpu_render: bool,
}

impl OrbitEditor {
    pub fn new(mut p: PersistedOrbits, gpu_render: bool) -> Self {
        let fresh = p.system.is_none();
        let system = p.system.take().unwrap_or_else(orbits::solar_system);
        let mut camera = p.camera;
        camera.limits = space_limits();
        let n = system.bodies.len();
        let mut editor = Self {
            system,
            snapshot: None,
            running: false,
            speed: p.speed.min(SPEEDS.len() - 1),
            camera,
            focus: p.focus.min(n.saturating_sub(1)),
            selected: None,
            show_trails: p.show_trails,
            show_labels: p.show_labels,
            min_size_px: p.min_size_px.clamp(1.0, 30.0),
            top_down: p.top_down,
            new_planet: p.new_planet,
            trails: vec![Trail::default(); n],
            energy_at_play: None,
            achieved_rate: 0.0,
            rate_limited: false,
            steps_last_frame: 0,
            status: None,
            gpu_render,
        };
        if fresh {
            editor.frame_focus();
        }
        editor
    }

    pub fn set_gpu_render(&mut self, on: bool) {
        self.gpu_render = on;
    }

    pub fn persisted(&self) -> PersistedOrbits {
        PersistedOrbits {
            system: Some(self.system.clone()),
            camera: self.camera,
            focus: self.focus,
            speed: self.speed,
            show_trails: self.show_trails,
            show_labels: self.show_labels,
            min_size_px: self.min_size_px,
            top_down: self.top_down,
            new_planet: self.new_planet.clone(),
        }
    }

    /// The system as JSON.
    pub fn system_json(&self) -> String {
        serde_json::to_string_pretty(&self.system).unwrap_or_default()
    }

    /// The status message set since the last call, if any.
    pub fn take_status(&mut self) -> Option<String> {
        self.status.take()
    }

    // ---------------------------------------------------------------------
    // Clock

    pub fn play(&mut self) {
        if !self.running {
            self.snapshot = Some(self.system.clone());
            self.energy_at_play = Some(self.system.energy());
            self.running = true;
        }
    }

    pub fn pause(&mut self) {
        self.running = false;
    }

    pub fn reset(&mut self) {
        if let Some(s) = self.snapshot.take() {
            self.load(s, false);
            self.status = Some("Reset to when Play was pressed.".into());
        }
        self.running = false;
    }

    /// Replace the system. `reframe`: point the camera at the star and fit
    /// the planets in view.
    fn load(&mut self, system: OrbitSystem, reframe: bool) {
        self.system = system;
        self.trails = vec![Trail::default(); self.system.bodies.len()];
        self.selected = self.selected.filter(|&i| i < self.system.bodies.len());
        self.focus = self.focus.min(self.system.bodies.len().saturating_sub(1));
        self.energy_at_play = None;
        if reframe {
            self.focus = self.system.star().unwrap_or(0);
            self.selected = None;
            self.frame_focus();
        }
    }

    pub fn load_example(&mut self, solar: bool) {
        self.running = false;
        self.snapshot = None;
        if solar {
            self.load(orbits::solar_system(), true);
            self.status =
                Some("Loaded the Solar System at 1 January 2000. Press Space to play.".into());
        } else {
            self.load(orbits::sun_only(), true);
            self.camera.distance = 4.0;
            self.status = Some("A lone Sun. Add planets from the sidebar.".into());
        }
    }

    pub fn logic(&mut self, ctx: &egui::Context) {
        self.steps_last_frame = 0;
        if !self.running {
            return;
        }
        let frame = f64::from(ctx.input(|i| i.stable_dt).min(0.1));
        let wanted = SPEEDS[self.speed].1 * frame;
        self.prepare_trails();
        let trails = &mut self.trails;
        let advance = self
            .system
            .advance_observed(wanted, MAX_STEPS_PER_FRAME, |s| record_trails(trails, s));
        self.after_advance(&advance);
        self.steps_last_frame = advance.steps;
        self.achieved_rate = advance.simulated / frame.max(1e-6);
        self.rate_limited = advance.simulated < 0.99 * wanted;
        ctx.request_repaint();
    }

    /// Fix up selection, focus, and trails after collisions.
    fn after_advance(&mut self, advance: &Advance) {
        for &(lost, kept) in &advance.removed {
            self.on_removed(lost, Some(kept));
        }
        if let Some((a, b)) = advance.mergers.last() {
            self.status = Some(format!("{a} collided with {b} and merged into it."));
        }
    }

    /// Body `lost` is gone; `into` is where references to it should go.
    fn on_removed(&mut self, lost: usize, into: Option<usize>) {
        let fix = |i: usize| -> Option<usize> {
            match i.cmp(&lost) {
                std::cmp::Ordering::Less => Some(i),
                std::cmp::Ordering::Equal => into.map(|k| if k > lost { k - 1 } else { k }),
                std::cmp::Ordering::Greater => Some(i - 1),
            }
        };
        self.selected = self.selected.and_then(fix);
        self.focus = fix(self.focus).unwrap_or(0);
        if lost < self.trails.len() {
            self.trails.remove(lost);
        }
    }

    /// Set each trail's sample spacing from its body's current period.
    fn prepare_trails(&mut self) {
        self.trails
            .resize(self.system.bodies.len(), Trail::default());
        for (i, trail) in self.trails.iter_mut().enumerate() {
            trail.spacing = match self.system.elements(i) {
                Some(el) => match el.period {
                    Some(t) => t / TRAIL_POINTS as f64,
                    // Escaping: keep a few months of path.
                    None => 30.0 * DAY / TRAIL_POINTS as f64,
                },
                None => f64::INFINITY,
            };
        }
    }

    fn clear_trail(&mut self, i: usize) {
        if let Some(t) = self.trails.get_mut(i) {
            t.points.clear();
        }
        // A planet's moons are drawn relative to it, so their trails survive.
    }

    // ---------------------------------------------------------------------
    // Editing

    fn add_planet(&mut self) {
        if self.system.star().is_none() {
            self.status = Some("Load a system with a star first.".into());
            return;
        }
        let n = self
            .system
            .bodies
            .iter()
            .filter(|b| b.kind == Kind::Planet)
            .count();
        // Successive planets go around at the golden angle, so they spread out.
        let angle = 137.507_764 * n as f64;
        let p = &self.new_planet;
        let name = self.system.unique_name("Planet");
        let i = self.system.add_planet(
            &name,
            p.mass_earths * M_EARTH,
            p.radius_earths * R_EARTH,
            p.color,
            p.distance_au * AU,
            angle,
        );
        self.trails
            .resize(self.system.bodies.len(), Trail::default());
        self.selected = Some(i);
        self.status = Some(format!(
            "Added {name} {:.2} AU from the star.",
            p.distance_au
        ));
        self.edited();
    }

    /// The Hill radius of planet `i`: roughly how far out its gravity beats
    /// the star's, a (m / 3M)^⅓. Moons need to stay well inside it.
    fn hill_radius(&self, i: usize) -> f64 {
        let b = &self.system.bodies[i];
        let (Some(parent), Some(el)) = (b.parent, self.system.elements(i)) else {
            return f64::INFINITY;
        };
        let a = if el.semi_major_axis > 0.0 {
            el.semi_major_axis
        } else {
            el.distance
        };
        a * (b.mass / (3.0 * self.system.bodies[parent].mass)).cbrt()
    }

    fn add_moon(&mut self, planet: usize) {
        let p = &self.system.bodies[planet];
        let moon_radius = 1_737_400.0;
        // Outside the planet and any existing moons, well inside the Hill
        // sphere.
        let farthest = self
            .system
            .moons_of(planet)
            .iter()
            .filter_map(|&m| self.system.elements(m).map(|e| e.distance))
            .fold(0.0, f64::max);
        let hill = self.hill_radius(planet);
        let distance = (farthest * 1.6)
            .max(30.0 * p.radius)
            .max(p.radius + 3.0 * moon_radius)
            .min(0.3 * hill)
            .max(p.radius + 2.0 * moon_radius);
        let name = self.system.unique_name(&format!("{} moon", p.name));
        let angle = 137.507_764 * self.system.moons_of(planet).len() as f64;
        let i = self.system.add_moon(
            planet,
            &name,
            M_MOON,
            moon_radius,
            [190, 190, 186],
            OrbitSpec::circular(distance, angle),
        );
        self.trails
            .resize(self.system.bodies.len(), Trail::default());
        self.selected = Some(i);
        self.edited();
        self.status = Some(format!(
            "Added {name} {} km from {}.",
            ui::thousands(distance / 1e3),
            self.system.bodies[planet].name
        ));
    }

    fn remove(&mut self, i: usize) {
        if self.system.bodies[i].kind == Kind::Star {
            self.status = Some("The star stays; load the empty example to start over.".into());
            return;
        }
        let name = self.system.bodies[i].name.clone();
        let gone = self.system.remove(i);
        self.edited();
        for &k in gone.iter().rev() {
            self.on_removed(k, None);
        }
        self.status = Some(if gone.len() > 1 {
            format!("Removed {name} and its {} moons.", gone.len() - 1)
        } else {
            format!("Removed {name}.")
        });
    }

    fn set_distance(&mut self, i: usize, meters: f64) {
        self.system.set_distance(i, meters);
        self.clear_trail(i);
        self.edited();
    }

    fn set_mass(&mut self, i: usize, kg: f64) {
        self.system.set_mass(i, kg);
        self.edited();
    }

    /// After an edit, measure the integrator's energy error from here.
    fn edited(&mut self) {
        if self.running {
            self.energy_at_play = Some(self.system.energy());
        }
    }

    // ---------------------------------------------------------------------
    // Camera

    /// Fit the followed body and what orbits it in view.
    fn frame_focus(&mut self) {
        let i = self.focus;
        let Some(b) = self.system.bodies.get(i) else {
            return;
        };
        let children: Vec<usize> = (0..self.system.bodies.len())
            .filter(|&k| self.system.bodies[k].parent == Some(i))
            .collect();
        let reach = children
            .iter()
            .map(|&k| (self.system.bodies[k].position - b.position).length())
            .fold(0.0, f64::max);
        let size = if reach > 0.0 {
            2.6 * reach
        } else {
            60.0 * b.radius
        };
        self.camera.target = Vec3::ZERO;
        self.camera.distance = ((size / AU) as f32).clamp(
            self.camera.limits.min_distance,
            self.camera.limits.max_distance,
        );
    }

    fn follow(&mut self, i: usize) {
        self.focus = i;
        self.frame_focus();
        self.status = Some(format!("Following {}.", self.system.bodies[i].name));
    }

    /// Body `i`'s position in view coordinates: AU from the followed body.
    fn view_position(&self, i: usize) -> Vec3 {
        let origin = self
            .system
            .bodies
            .get(self.focus)
            .map_or(DVec3::ZERO, |b| b.position);
        ((self.system.bodies[i].position - origin) / AU).as_vec3()
    }

    // ---------------------------------------------------------------------
    // Keyboard

    pub fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let mods = ctx.input(|i| i.modifiers);
        if mods.command || mods.alt {
            return;
        }
        let pressed = |k| ctx.input(|i| i.key_pressed(k));
        if pressed(Key::Space) {
            if self.running {
                self.pause();
            } else {
                self.play();
            }
        }
        if pressed(Key::R) {
            self.reset();
        }
        if pressed(Key::Escape) {
            self.selected = None;
        }
        if pressed(Key::F)
            && let Some(i) = self.selected
        {
            self.follow(i);
        }
        if pressed(Key::T) {
            self.top_down = !self.top_down;
        }
        if (pressed(Key::Delete) || pressed(Key::Backspace))
            && let Some(i) = self.selected
        {
            self.remove(i);
        }
        if pressed(Key::Plus) || pressed(Key::Equals) {
            self.speed = (self.speed + 1).min(SPEEDS.len() - 1);
        }
        if pressed(Key::Minus) {
            self.speed = self.speed.saturating_sub(1);
        }
    }

    fn apply_top_down(&mut self) {
        if self.top_down {
            self.camera.pitch = MAX_PITCH;
        }
    }
}

/// Sample trails whose spacing has elapsed (called after every step).
fn record_trails(trails: &mut [Trail], s: &OrbitSystem) {
    for (i, trail) in trails.iter_mut().enumerate() {
        let Some(b) = s.bodies.get(i) else {
            break;
        };
        let Some(parent) = b.parent else {
            continue;
        };
        if s.time - trail.last < trail.spacing {
            continue;
        }
        trail.last = s.time;
        trail
            .points
            .push_back(b.position - s.bodies[parent].position);
        while trail.points.len() > TRAIL_POINTS {
            trail.points.pop_front();
        }
    }
}

/// "3.2 years", "41 days", "5.3 hours".
fn format_duration(seconds: f64) -> String {
    let s = seconds.abs();
    if s >= 2.0 * YEAR {
        format!("{:.1} years", seconds / YEAR)
    } else if s >= 2.0 * DAY {
        format!("{:.1} days", seconds / DAY)
    } else {
        format!("{:.1} hours", seconds / 3600.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_pick_sensible_units() {
        assert_eq!(format_duration(3.0 * YEAR), "3.0 years");
        assert_eq!(format_duration(27.3 * DAY), "27.3 days");
        assert_eq!(format_duration(7200.0), "2.0 hours");
    }

    #[test]
    fn trails_sample_every_spacing() {
        let mut s = orbits::solar_system();
        let mut trails = vec![
            Trail {
                spacing: DAY,
                ..Trail::default()
            };
            s.bodies.len()
        ];
        s.advance_observed(10.0 * DAY, 100_000, |s| record_trails(&mut trails, s));
        // The star has no parent and no trail; planets get ~10 samples.
        assert!(trails[0].points.is_empty());
        assert!(
            (9..=11).contains(&trails[3].points.len()),
            "{}",
            trails[3].points.len()
        );
    }

    #[test]
    fn removal_reindexes_selection_and_focus() {
        let mut e = OrbitEditor::new(PersistedOrbits::default(), false);
        let jupiter = e
            .system
            .bodies
            .iter()
            .position(|b| b.name == "Jupiter")
            .unwrap();
        let titan = e
            .system
            .bodies
            .iter()
            .position(|b| b.name == "Titan")
            .unwrap();
        e.selected = Some(titan);
        e.focus = jupiter;
        e.remove(jupiter);
        assert_eq!(e.focus, 0, "focus falls back to the star");
        let sel = e.selected.unwrap();
        assert_eq!(e.system.bodies[sel].name, "Titan");
        assert_eq!(e.trails.len(), e.system.bodies.len());
    }
}
