//! The Fluids tab: a glass box (in vacuum or in room-temperature air) to
//! pour liquids into, drop solids in, and heat or cool, with probes,
//! totals, and graphs. Liquids are drawn as smooth surfaces by default.
//!
//! - `mod.rs` (this file): state, the box, simulation control, tools'
//!   actions, shortcuts
//! - `ui.rs`: toolbar, sidebar, status bar, and the 3D view
//! - `graph.rs`: the history of totals and its plot

mod graph;
mod ui;

use crate::history::History;
use crate::liquid_ui::LiquidChoice;
use crate::view3d::orbit::OrbitCamera;
use eframe::egui::{self, Key, KeyboardShortcut, Modifiers};
use engine::FixedTimestep;
use engine3d::fluid::FluidParams;
use engine3d::gas::Gas;
use engine3d::{BodyHandle, BodyId, RigidBody, Scene, Shape, Vec3, World};
use serde::{Deserialize, Serialize};

const UNDO_LIMIT: usize = 50;
const STEP_HZ: f32 = 120.0;
/// Bodies 0..5 are the box: the floor, then four walls.
const BOX_BODIES: usize = 5;
/// Wall and floor thickness, m.
const WALL: f32 = 0.1;
/// Air cells along each side of the box.
const AIR_CELLS: usize = 16;
/// Room temperature, K.
const ROOM: f32 = 293.15;
/// One standard atmosphere, Pa.
const ATMOSPHERE: f32 = 101_325.0;
const KELVIN: f32 = 273.15;
/// Particle spacings offered, m (the box is filled on a lattice this fine).
const SPACINGS: [f32; 5] = [0.05, 0.04, 0.03, 0.025, 0.02];

/// What fills the box around the liquid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Environment {
    Vacuum,
    /// 1 atm of air at 20 °C, open at the top.
    #[default]
    Air,
}

impl Environment {
    /// Pressure above the liquid, Pa.
    fn pressure(self) -> f32 {
        match self {
            Environment::Vacuum => 0.0,
            Environment::Air => ATMOSPHERE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    /// Click a solid or plate to select it; drag to move it.
    Select,
    /// Drag a rectangle on the floor: a block of liquid on top of what's
    /// there.
    Block,
    /// Click: a ball of liquid dropped from above.
    Ball,
    /// Click: a floating or sinking solid.
    Solid,
    /// Click the floor: a heater or cooler plate.
    Plate,
    /// Hold over the liquid to heat (or cool) it.
    Heat,
    /// Hover to read the liquid; click to pin the probe.
    Probe,
}

/// How to color the liquid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorBy {
    /// Each liquid's own color.
    #[default]
    Liquid,
    Temperature,
    Speed,
    Pressure,
    Density,
}

/// Solids to drop in, with real densities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SolidMaterial {
    Cork,
    #[default]
    Pine,
    /// Polypropylene: floats on water, and only just on oil (900 vs 920).
    Plastic,
    Rubber,
    Aluminum,
    Steel,
    Gold,
}

impl SolidMaterial {
    const ALL: [SolidMaterial; 7] = [
        SolidMaterial::Cork,
        SolidMaterial::Pine,
        SolidMaterial::Plastic,
        SolidMaterial::Rubber,
        SolidMaterial::Aluminum,
        SolidMaterial::Steel,
        SolidMaterial::Gold,
    ];

    fn name(self) -> &'static str {
        match self {
            SolidMaterial::Cork => "Cork",
            SolidMaterial::Pine => "Pine",
            SolidMaterial::Plastic => "Plastic",
            SolidMaterial::Rubber => "Rubber",
            SolidMaterial::Aluminum => "Aluminum",
            SolidMaterial::Steel => "Steel",
            SolidMaterial::Gold => "Gold",
        }
    }

    /// kg/m³ (cork 240, pine 500, polypropylene 900, natural rubber 1100,
    /// aluminum 2700, steel 7850, gold 19,300).
    fn density(self) -> f32 {
        match self {
            SolidMaterial::Cork => 240.0,
            SolidMaterial::Pine => 500.0,
            SolidMaterial::Plastic => 900.0,
            SolidMaterial::Rubber => 1100.0,
            SolidMaterial::Aluminum => 2700.0,
            SolidMaterial::Steel => 7850.0,
            SolidMaterial::Gold => 19_300.0,
        }
    }

    fn color(self) -> [u8; 4] {
        match self {
            SolidMaterial::Cork => [0xB8, 0x8A, 0x5A, 255],
            SolidMaterial::Pine => [0xD9, 0xB3, 0x7A, 255],
            SolidMaterial::Plastic => [0xEE, 0xEE, 0xE8, 255],
            SolidMaterial::Rubber => [0x3A, 0x3A, 0x3E, 255],
            SolidMaterial::Aluminum => [0xC4, 0xC8, 0xCE, 255],
            SolidMaterial::Steel => [0x7C, 0x82, 0x8A, 255],
            SolidMaterial::Gold => [0xF1, 0xBE, 0x48, 255],
        }
    }

    /// The material whose density this is (the closest).
    fn of_density(rho: f32) -> SolidMaterial {
        Self::ALL
            .into_iter()
            .min_by(|a, b| {
                (a.density() - rho)
                    .abs()
                    .total_cmp(&(b.density() - rho).abs())
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
enum SolidShape {
    #[default]
    Ball,
    Cube,
}

/// Settings remembered between sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    /// Inner side of the (cubic) box, m.
    box_size: f32,
    spacing: f32,
    environment: Environment,
    /// Hold the walls and floor at `wall_celsius` (else insulated).
    hold_walls: bool,
    wall_celsius: f32,
    liquid: LiquidChoice,
    liquid_celsius: f32,
    /// Height of a block of liquid, m.
    block_height: f32,
    ball_radius: f32,
    /// How far above the surface a ball is dropped from, m.
    drop_height: f32,
    /// Fill level as a fraction of the box's height.
    fill_level: f32,
    solid: SolidMaterial,
    solid_shape: SolidShape,
    /// Diameter or side, m.
    solid_size: f32,
    /// Side of a square plate, m.
    plate_size: f32,
    plate_celsius: f32,
    /// Heat brush power, W (negative cools).
    heat_power: f32,
    heat_radius: f32,
    color_by: ColorBy,
    smooth: bool,
    time_scale: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            box_size: 1.0,
            spacing: 0.04,
            environment: Environment::Air,
            hold_walls: false,
            wall_celsius: 20.0,
            liquid: LiquidChoice::Water,
            liquid_celsius: 20.0,
            block_height: 0.2,
            ball_radius: 0.12,
            drop_height: 0.2,
            fill_level: 0.3,
            solid: SolidMaterial::Pine,
            solid_shape: SolidShape::Ball,
            solid_size: 0.15,
            plate_size: 0.3,
            plate_celsius: 80.0,
            heat_power: 2000.0,
            heat_radius: 0.1,
            color_by: ColorBy::Liquid,
            smooth: true,
            time_scale: 1.0,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedFluids {
    camera: OrbitCamera,
    settings: Settings,
    /// The box as it was when the app closed, as 3D scene JSON.
    scene: Option<String>,
}

impl Default for PersistedFluids {
    fn default() -> Self {
        let settings = Settings::default();
        Self {
            camera: default_camera(settings.box_size),
            settings,
            scene: None,
        }
    }
}

fn default_camera(size: f32) -> OrbitCamera {
    OrbitCamera {
        target: Vec3::new(0.0, 0.4 * size, 0.0),
        yaw: 0.55,
        pitch: 0.5,
        distance: 2.6 * size,
        ..OrbitCamera::default()
    }
}

/// Dragging a solid across a horizontal plane.
struct Grab {
    handle: BodyHandle,
    height: f32,
    offset: Vec3,
    prev: Vec3,
}

pub struct FluidsEditor {
    world: World,
    snapshot: Option<World>,
    running: bool,
    timestep: FixedTimestep,
    steps_last_frame: u32,
    /// Simulated seconds per real second, smoothed.
    speed: f32,
    history: History<World>,
    camera: OrbitCamera,
    settings: Settings,
    tool: Tool,
    selection: Option<BodyHandle>,
    grab: Option<Grab>,
    /// Where a Block drag started on the floor.
    block_start: Option<Vec3>,
    /// Where the heat brush is this frame (cleared every frame).
    heating: Option<Vec3>,
    /// Energy the heat brush has added since the box was made, J.
    brush_energy: f64,
    /// A pinned probe point.
    probe_pin: Option<Vec3>,
    /// The probe under the mouse this frame.
    probe_hover: Option<Vec3>,
    graph: graph::Graph,
    status: Option<String>,
    gpu_render: bool,
}

impl FluidsEditor {
    pub fn new(p: PersistedFluids) -> Self {
        let mut settings = p.settings;
        settings.box_size = settings.box_size.clamp(0.25, 2.0);
        let restored = p
            .scene
            .as_deref()
            .and_then(|json| Scene::from_json(json).ok())
            .map(|s| s.world)
            .filter(|w| w.bodies.len() >= BOX_BODIES);
        let world = restored.unwrap_or_else(|| build_box(&settings));
        let mut camera = p.camera;
        camera.limits.min_distance = 0.2;
        camera.limits.max_distance = 30.0;
        let mut editor = Self {
            world,
            snapshot: None,
            running: false,
            timestep: FixedTimestep::new(1.0 / STEP_HZ),
            steps_last_frame: 0,
            speed: 0.0,
            history: History::new(UNDO_LIMIT),
            camera,
            settings,
            tool: Tool::Block,
            selection: None,
            grab: None,
            block_start: None,
            heating: None,
            brush_energy: 0.0,
            probe_pin: None,
            probe_hover: None,
            graph: graph::Graph::default(),
            status: None,
            gpu_render: false,
        };
        // A fluid step can take tens of milliseconds; at most a few per
        // frame keeps the window responsive (time then runs slower).
        editor.timestep.max_steps_per_frame = 3;
        editor.history.rebase(&editor.world);
        editor
    }

    pub fn persisted(&self) -> PersistedFluids {
        PersistedFluids {
            camera: self.camera,
            settings: self.settings.clone(),
            scene: Some(Scene::from_world(&self.world).to_json()),
        }
    }

    /// The box as a 3D scene file.
    pub fn scene_json(&self) -> String {
        Scene::from_world(&self.world).to_json()
    }

    pub fn set_gpu_render(&mut self, on: bool) {
        self.gpu_render = on;
    }

    pub fn take_status(&mut self) -> Option<String> {
        self.status.take()
    }

    fn say(&mut self, s: impl Into<String>) {
        self.status = Some(s.into());
    }

    // ---------------------------------------------------------------------
    // Clock

    /// Run the simulation for this frame. Called before any UI is drawn.
    pub fn logic(&mut self, ctx: &egui::Context) {
        self.steps_last_frame = 0;
        if !self.running {
            self.speed = 0.0;
            return;
        }
        let (frame, real) = ctx.input(|i| (i.stable_dt.min(0.1), i.unstable_dt));
        let steps = self.timestep.advance(frame * self.settings.time_scale);
        for _ in 0..steps {
            self.step_world();
        }
        self.steps_last_frame = steps;
        let simulated = steps as f32 * self.timestep.dt;
        // About a second of smoothing.
        let a = (real / 1.0).min(1.0);
        self.speed += a * (simulated / real.max(1e-4) - self.speed);
    }

    fn step_world(&mut self) {
        let dt = self.timestep.dt;
        if let Some(c) = self.heating {
            let r = self.settings.heat_radius;
            self.brush_energy += self
                .world
                .fluid
                .add_heat(c, r, self.settings.heat_power, dt);
        }
        self.world.step(dt);
        self.graph.record(&self.world, self.brush_energy);
    }

    fn play(&mut self) {
        if !self.running {
            self.history.record(&self.world);
            self.snapshot = Some(self.world.clone());
            self.timestep.reset();
            self.running = true;
        }
    }

    pub fn pause(&mut self) {
        if self.running {
            self.running = false;
            self.history.rebase(&self.world);
        }
    }

    fn step_once(&mut self) {
        self.pause();
        self.history.record(&self.world);
        self.step_world();
        self.history.rebase(&self.world);
    }

    fn reset(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.running = false;
            self.world = snapshot;
            self.grab = None;
            self.timestep.reset();
            self.graph.clear();
            self.history.rebase(&self.world);
            self.say("Reset to the state before Play.");
        }
    }

    fn undo(&mut self) {
        self.pause();
        self.history.record(&self.world);
        if let Some(world) = self.history.undo() {
            self.restore(world);
            self.say("Undone.");
        }
    }

    fn redo(&mut self) {
        self.pause();
        if let Some(world) = self.history.redo() {
            self.restore(world);
            self.say("Redone.");
        }
    }

    fn restore(&mut self, world: World) {
        self.world = world;
        self.grab = None;
        self.snapshot = None;
        if let Some(h) = self.selection
            && self.world.find(h).is_none()
        {
            self.selection = None;
        }
    }

    /// Record an edit once the user has finished one (end of every frame).
    pub fn record_finished_edit(&mut self, ctx: &egui::Context) {
        let busy = self.running || self.grab.is_some() || ctx.input(|i| i.pointer.any_down());
        if !busy {
            self.history.record(&self.world);
        }
    }

    // ---------------------------------------------------------------------
    // The box

    /// Throw away everything and make a new, empty box from the settings.
    fn new_box(&mut self) {
        self.pause();
        self.history.record(&self.world);
        self.world = build_box(&self.settings);
        self.snapshot = None;
        self.selection = None;
        self.grab = None;
        self.probe_pin = None;
        self.brush_energy = 0.0;
        self.graph.clear();
        self.camera = OrbitCamera {
            limits: self.camera.limits,
            ..default_camera(self.settings.box_size)
        };
        self.say(format!(
            "New {:.2} m box, {}.",
            self.settings.box_size,
            match self.settings.environment {
                Environment::Vacuum => "in vacuum",
                Environment::Air => "in air at 1 atm and 20 °C",
            }
        ));
    }

    /// Switch between vacuum and air without touching the liquid.
    fn set_environment(&mut self, env: Environment) {
        self.settings.environment = env;
        self.world.gas = air_for(&self.settings);
    }

    fn apply_wall_temperature(&mut self) {
        self.world.fluid.params.wall_temperature = self
            .settings
            .hold_walls
            .then_some(self.settings.wall_celsius + KELVIN);
    }

    /// Inner corners of the box.
    fn inner(&self) -> (Vec3, Vec3) {
        let s = self.box_size();
        (
            Vec3::new(-0.5 * s, 0.0, -0.5 * s),
            Vec3::new(0.5 * s, s, 0.5 * s),
        )
    }

    /// The box's actual size (from its walls, in case the scene came from
    /// a file), m.
    fn box_size(&self) -> f32 {
        self.world
            .bodies
            .get(1)
            .and_then(|w| match w.shape {
                Shape::Cuboid { half_extents } => Some(2.0 * half_extents.y),
                Shape::Sphere { .. } => None,
            })
            .unwrap_or(self.settings.box_size)
    }

    /// Height of the liquid's top surface above (x, z), within a spacing
    /// or so; 0 if there's none.
    fn surface_height(&self, x: f32, z: f32, half_width: f32) -> f32 {
        let fluid = &self.world.fluid;
        let reach = half_width + fluid.params.particle_spacing;
        fluid
            .positions
            .iter()
            .filter(|p| (p.x - x).abs() <= reach && (p.z - z).abs() <= reach)
            .map(|p| p.y)
            .fold(0.0, f32::max)
    }

    /// A block of liquid over the floor rectangle `a`–`b`, standing on the
    /// liquid already there.
    fn pour_block(&mut self, a: Vec3, b: Vec3) {
        let (lo_box, hi_box) = self.inner();
        let d = self.world.fluid.params.particle_spacing;
        let (mut lo, mut hi) = (a.min(b), a.max(b));
        // A click (or tiny drag) pours a block the size of the ball.
        if (hi - lo).length() < 2.0 * d {
            let c = 0.5 * (lo + hi);
            let r = self.settings.ball_radius;
            lo = c - Vec3::new(r, 0.0, r);
            hi = c + Vec3::new(r, 0.0, r);
        }
        lo = lo.max(lo_box);
        hi = hi.min(hi_box);
        let c = 0.5 * (lo + hi);
        let half = 0.5 * (hi.x - lo.x).max(hi.z - lo.z);
        let bottom = self.surface_height(c.x, c.z, half);
        let top = (bottom + self.settings.block_height).min(hi_box.y);
        lo.y = bottom;
        hi.y = top;
        self.fill(lo, hi, None);
    }

    /// A ball of liquid dropped from `drop_height` above the surface.
    fn drop_ball(&mut self, at: Vec3) {
        let (lo_box, hi_box) = self.inner();
        let r = self.settings.ball_radius;
        let x = at.x.clamp(lo_box.x + r, hi_box.x - r);
        let z = at.z.clamp(lo_box.z + r, hi_box.z - r);
        let bottom = self.surface_height(x, z, r) + self.settings.drop_height;
        let y = (bottom + r).min(hi_box.y - r);
        let c = Vec3::new(x, y, z);
        self.fill(c - Vec3::splat(r), c + Vec3::splat(r), Some((c, r)));
    }

    /// Fill the box with the chosen liquid up to the fill level, wherever
    /// there's room.
    fn fill_to_level(&mut self) {
        let (lo, mut hi) = self.inner();
        hi.y = self.settings.fill_level * self.box_size();
        self.fill(lo, hi, None);
    }

    fn fill(&mut self, lo: Vec3, hi: Vec3, ball: Option<(Vec3, f32)>) {
        let c = self.settings.liquid;
        let spec = c.spec(self.settings.liquid_celsius);
        let bodies = self.world.bodies.clone();
        let fluid = &mut self.world.fluid;
        let added = match ball {
            Some((center, r)) => fluid.fill_sphere_as(center, r, &bodies, spec),
            None => fluid.fill_box_as(lo, hi, &bodies, spec),
        };
        let total = fluid.len();
        let litres = added as f32 * fluid.particle_volume() * 1000.0;
        self.say(if added == 0 {
            format!("No room for {} there.", c.noun())
        } else {
            format!(
                "Added {added} particles of {} ({litres:.1} L); {total} in all.",
                c.noun()
            )
        });
    }

    /// Drop a solid from above the surface at `at`.
    fn drop_solid(&mut self, at: Vec3) {
        let (lo_box, hi_box) = self.inner();
        let s = self.settings.solid_size;
        let half = 0.5 * s;
        let x = at.x.clamp(lo_box.x + half, hi_box.x - half);
        let z = at.z.clamp(lo_box.z + half, hi_box.z - half);
        let y = (self.surface_height(x, z, half) + half + 0.1).min(hi_box.y + half);
        let shape = match self.settings.solid_shape {
            SolidShape::Ball => Shape::Sphere { radius: half },
            SolidShape::Cube => Shape::cuboid(s, s, s),
        };
        let mut body = RigidBody::new(shape, Vec3::new(x, y, z));
        body.density = self.settings.solid.density();
        let id = self.world.add_body(body);
        self.selection = self.world.handle(id);
        self.say(format!(
            "Dropped {} ({:.0} kg/m³).",
            self.settings.solid.name().to_lowercase(),
            self.settings.solid.density()
        ));
    }

    /// A heater or cooler plate lying on the floor at `at`.
    fn place_plate(&mut self, at: Vec3) {
        let (lo_box, hi_box) = self.inner();
        let s = self.settings.plate_size.min(self.box_size());
        let half = 0.5 * s;
        let thickness = 0.02;
        let x = at.x.clamp(lo_box.x + half, hi_box.x - half);
        let z = at.z.clamp(lo_box.z + half, hi_box.z - half);
        let mut plate = RigidBody::new_static(
            Shape::cuboid(s, thickness, s),
            Vec3::new(x, 0.5 * thickness, z),
        );
        plate.temperature = Some(self.settings.plate_celsius + KELVIN);
        let id = self.world.add_body(plate);
        // Liquid where the plate now is would start inside it.
        let bodies = self.world.bodies.clone();
        self.world.fluid.project_out_of_bodies(&bodies);
        self.selection = self.world.handle(id);
        self.say(format!(
            "Plate at {:.0} °C. Select it to change the temperature.",
            self.settings.plate_celsius
        ));
    }

    fn selected(&self) -> Option<BodyId> {
        self.selection
            .and_then(|h| self.world.find(h))
            .filter(|id| id.0 >= BOX_BODIES)
    }

    fn delete_selected(&mut self) {
        if let Some(id) = self.selected() {
            self.world.remove_body(id);
            self.selection = None;
            self.grab = None;
            self.say("Deleted.");
        }
    }

    pub fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let shortcut =
            |mods, key| ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(mods, key)));
        if shortcut(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || shortcut(Modifiers::COMMAND, Key::Y)
        {
            self.redo();
        } else if shortcut(Modifiers::COMMAND, Key::Z) {
            self.undo();
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
        if pressed(Key::Period) {
            self.step_once();
        }
        if pressed(Key::R) {
            self.reset();
        }
        if pressed(Key::Escape) {
            self.tool = Tool::Select;
            self.selection = None;
            self.block_start = None;
        }
        for (key, tool) in [
            (Key::V, Tool::Select),
            (Key::B, Tool::Block),
            (Key::C, Tool::Ball),
            (Key::O, Tool::Solid),
            (Key::P, Tool::Plate),
            (Key::H, Tool::Heat),
            (Key::I, Tool::Probe),
        ] {
            if pressed(key) {
                self.tool = tool;
            }
        }
        if pressed(Key::Delete) || pressed(Key::Backspace) {
            self.delete_selected();
        }
    }
}

/// An empty box from the settings: floor and four walls (static), the
/// fluid's settings, and air if wanted.
fn build_box(s: &Settings) -> World {
    let size = s.box_size;
    let mut w = World::new();
    let outer = size + 2.0 * WALL;
    w.add_body(RigidBody::new_static(
        Shape::cuboid(outer, WALL, outer),
        Vec3::new(0.0, -0.5 * WALL, 0.0),
    ));
    let half = 0.5 * size;
    for sign in [-1.0, 1.0] {
        w.add_body(RigidBody::new_static(
            Shape::cuboid(WALL, size, outer),
            Vec3::new(sign * (half + 0.5 * WALL), 0.5 * size, 0.0),
        ));
        w.add_body(RigidBody::new_static(
            Shape::cuboid(size, size, WALL),
            Vec3::new(0.0, 0.5 * size, sign * (half + 0.5 * WALL)),
        ));
    }
    w.fluid.params = FluidParams {
        particle_spacing: s.spacing,
        // About 7× the fastest a liquid falling the box's height can go,
        // so it compresses about 2%.
        speed_of_sound: (7.0 * (2.0 * 9.81 * size).sqrt()).max(20.0),
        wall_temperature: s.hold_walls.then_some(s.wall_celsius + KELVIN),
        ..FluidParams::default()
    };
    w.gas = air_for(s);
    w
}

/// Air filling the box at 20 °C, or none in vacuum.
fn air_for(s: &Settings) -> Option<Gas> {
    match s.environment {
        Environment::Vacuum => None,
        Environment::Air => {
            let size = s.box_size;
            let cell = size / AIR_CELLS as f32;
            let mut gas = Gas::new(
                Vec3::new(-0.5 * size, 0.0, -0.5 * size),
                cell,
                AIR_CELLS,
                AIR_CELLS,
                AIR_CELLS,
            );
            gas.params.ambient_temperature = ROOM;
            gas.params.open_top = true;
            Some(gas)
        }
    }
}

/// Where the ray `origin + t·dir` crosses the plane, if in front.
fn ray_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-4 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    (t >= 0.0).then(|| origin + t * dir)
}

/// The nearest liquid particle along a ray, within `radius` of it.
fn pick_particle(positions: &[Vec3], origin: Vec3, dir: Vec3, radius: f32) -> Option<Vec3> {
    let r2 = radius * radius;
    positions
        .iter()
        .filter_map(|&p| {
            let to = p - origin;
            let t = to.dot(dir);
            (t > 0.0 && to.length_squared() - t * t <= r2).then_some((t, p))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
}

/// Thermal energy above room temperature, Σ m c (T − 20 °C), J.
fn thermal_energy_above_room(world: &World) -> f64 {
    let fluid = &world.fluid;
    let room: f64 = (0..fluid.len())
        .map(|i| {
            let s = fluid.salinity.get(i).copied().unwrap_or(0.0);
            f64::from(fluid.mass(i) * fluid.liquid(i).specific_heat(s)) * f64::from(ROOM)
        })
        .sum();
    fluid.thermal_energy() - room
}

/// Gravitational potential energy of the liquid above the floor, J.
fn potential_energy(world: &World) -> f64 {
    let g = f64::from(-world.gravity.y);
    let fluid = &world.fluid;
    (0..fluid.len())
        .map(|i| f64::from(fluid.mass(i)) * g * f64::from(fluid.positions[i].y))
        .sum()
}

/// Kelvin to Celsius.
fn celsius(k: f32) -> f32 {
    k - KELVIN
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine3d::fluid::Liquid;

    #[test]
    fn the_box_has_the_requested_inside() {
        let s = Settings::default();
        let w = build_box(&s);
        assert_eq!(w.bodies.len(), BOX_BODIES);
        let e = FluidsEditor::new(PersistedFluids {
            scene: Some(Scene::from_world(&w).to_json()),
            ..PersistedFluids::default()
        });
        assert!((e.box_size() - 1.0).abs() < 1e-6);
        assert!(e.world.gas.is_some());
    }

    #[test]
    fn filling_to_a_level_gives_that_volume() {
        let mut e = FluidsEditor::new(PersistedFluids::default());
        e.settings.fill_level = 0.25;
        e.fill_to_level();
        let volume = e.world.fluid.len() as f32 * e.world.fluid.particle_volume();
        // 1 m × 1 m × 0.25 m, less the gap kept from the walls.
        assert!(volume > 0.2 && volume < 0.26, "{volume} m³");
    }

    #[test]
    fn blocks_stack_instead_of_overlapping() {
        let mut e = FluidsEditor::new(PersistedFluids::default());
        e.pour_block(Vec3::new(-0.2, 0.0, -0.2), Vec3::new(0.2, 0.0, 0.2));
        let first = e.world.fluid.len();
        e.pour_block(Vec3::new(-0.2, 0.0, -0.2), Vec3::new(0.2, 0.0, 0.2));
        assert!(e.world.fluid.len() > first + first / 2);
        let top = e.surface_height(0.0, 0.0, 0.2);
        assert!(top > 0.35, "top {top} m");
    }

    #[test]
    fn steel_floats_on_mercury_and_gold_doesnt() {
        assert!(SolidMaterial::Steel.density() < Liquid::Mercury.density(0.0));
        assert!(SolidMaterial::Gold.density() > Liquid::Mercury.density(0.0));
        assert_eq!(SolidMaterial::of_density(7800.0), SolidMaterial::Steel);
    }

    #[test]
    fn picking_finds_the_nearest_particle_on_the_ray() {
        let ps = [
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(0.0, 0.0, -2.0),
            Vec3::new(1.0, 0.0, -1.0),
        ];
        let hit = pick_particle(&ps, Vec3::ZERO, -Vec3::Z, 0.1).unwrap();
        assert_eq!(hit, ps[1]);
    }
}
