# PhysicsEngine

A rigid-body and particle-fluid physics engine with an interactive editor, written in Rust, in 2D and 3D. It runs on Windows, macOS, Linux, and in the browser.

Website: [physicsengine.dev](https://physicsengine.dev) · Simulator in your browser: [physicsengine.dev/sim](https://physicsengine.dev/sim/) · Validation: [physicsengine.dev/validation](https://physicsengine.dev/validation/) · Plan: [ROADMAP.md](ROADMAP.md)

> **Status:** milestone M11. In 2D and 3D, rigid bodies (boxes and circles or spheres) collide, stack, bounce, roll, and slide with friction, and can be joined by pins, rods, ropes, springs, and chains; soft blobs, jelly, and cloth deform and push them around; smoke and hot air rise around them; SPH water flows around them, and bodies float, sink, and get pushed by it, on the CPU or (desktop) the GPU. The editor switches between 2D, 3D, and an Orbits tab (a star, planets, and moons under N-body gravity in real units, starting from the Solar System), has undo and a dark mode, and runs in the browser. A [validation page](https://physicsengine.dev/validation/) runs both engines against exact physics on every deploy.

## Running it

Install Rust from [rustup.rs](https://rustup.rs), then:

```sh
cargo run -p editor            # debug build (still reasonably fast)
cargo run -p editor --release  # full speed
cargo test --workspace         # run the physics tests
```

On Linux you may first need the GUI libraries:

```sh
sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev
```

### In the browser

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
cd crates/editor
trunk serve --release   # then open http://127.0.0.1:8080
```

## Editor controls

| Action | Input |
| --- | --- |
| Select / place | Left-click (tool chosen in the sidebar) |
| Select several | Shift+click to add or remove · drag on empty space to box-select · Ctrl+A for all |
| Move | Drag any selected body; the whole selection moves. Release while playing to throw. |
| Rotate / resize | Select one body, then drag its round (rotate) or square (resize) gold handle |
| Undo / redo | Ctrl+Z · Ctrl+Shift+Z or Ctrl+Y (or the toolbar buttons) |
| Snap to grid | G, or the Snap checkbox (positions and sizes to ¼ grid, angles to 15°) |
| Pan | Right- or middle-drag |
| Zoom | Scroll wheel or pinch |
| Play / pause | Space |
| Single step | `.` |
| Reset to before Play | R |
| Tools | V select · C circle · B box · F fluid · E erase · J joint · L chain · S soft body · K smoke · Esc deselect |
| Joints | J, then pick Pin (click where two bodies overlap, or on one to pin it to the world) or Rod / Rope / Spring (drag from one body to another, or to empty space) |
| Chain | L, then drag: each end is pinned to the body under it (or the world) |
| Soft body | S, choose Blob or Jelly, then click; drag soft bodies with Select |
| Smoke and heat | Add a grid in the Smoke and heat panel, then K: drag to paint smoke and heat, or click to place an emitter (heater/smoke source). Switch the view between Smoke and Temperature. |
| Delete selected | Delete / Backspace |
| Contact overlay | D, or the checkbox under World |
| Add water | F, then drag a rectangle (or switch to Brush and paint) |
| Erase water | E, then drag |
| Fluid force arrows | Fluid panel → Show forces on bodies (an arrow the length of the body's weight is 1 m) |

### 3D mode

Click **3D** at the left of the toolbar. Play/Pause, Step, Reset, Undo/Redo, Examples, and Open/Save work as in 2D.

| Action | Input |
| --- | --- |
| Select | Left-click a body |
| Place a sphere / box | C or B (or the sidebar), then click: it lands on whatever is under the mouse, or the ground |
| Add water | W, then click: drops a block of water (size set in the sidebar) onto whatever is under the mouse |
| Joints | J, then click a body and a second point (another body, or the ground for the world) |
| Chain | L, then click the two ends |
| Soft body | S, choose Ball, Cloth (optionally pinned), or Jelly, then click |
| Smoke and heat | Add a grid in the Smoke and heat panel, then K: click a surface for an emitter, or brush |
| Move | Left-drag a body across the ground plane · Shift+left-drag to raise or lower it · release while playing to throw |
| Orbit | Right-drag |
| Pan | Shift+right-drag or middle-drag |
| Zoom | Scroll wheel or pinch |
| Focus on selection | F |
| Contact overlay | D |

Scenes save as human-readable JSON. **Open…/Save…** use file dialogs on desktop; in the browser, Open uses the browser's file picker and Save downloads `scene.json`. **Copy JSON/Paste JSON…** work everywhere. Your settings and current scene are remembered between sessions. **Examples** loads the built-in scenes (demo, pyramid, friction ramps, bouncing balls, dam break, buoyancy, pendulums, rope bridge, Newton's cradle, springs, and in 3D a chain drape). The same scenes are in [`scenes/`](scenes) (3D ones in `scenes/3d/`); regenerate them with `cargo run -p engine --example make_scenes` and `cargo run -p engine3d --example make_scenes`. Opening a 3D scene switches the editor to 3D, and vice versa.

## Layout

```
crates/
  engine/   physics only: shapes, bodies, collision, contact solver, SPH fluid, scenes. No GUI deps.
    benches/           Criterion benchmarks
    tests/physics.rs   rigid-body validation against known answers
    tests/fluid.rs     fluid validation (hydrostatics, dam break, leaks)
    tests/coupling.rs  fluid ↔ body validation (Archimedes, momentum exchange)
    tests/joints.rs    joint validation (pendulum period, energy, springs, ropes, chains)
    tests/soft.rs      soft-body validation (Hooke's law, area, momentum, energy)
    tests/gas.rs       smoke/heat validation (heat diffusion, buoyancy, hydrostatics)
    examples/          make_scenes (regenerate scenes/), bench_fluid (performance)
  engine3d/ 3D: spheres and boxes, SAT contacts, patch friction, SPH water, ray casts, scenes,
            and planetary orbits (src/orbits.rs).
    tests/physics3d.rs validation against closed-form answers (sliding, rolling, bouncing, stacking)
    tests/fluid3d.rs   fluid validation (hydrostatics, leaks, Archimedes, momentum exchange)
    tests/joints3d.rs  joint validation (pendulums incl. conical, springs, chains)
    tests/soft3d.rs    soft-body validation (Hooke's law, volume, cloth draping, momentum)
    tests/gas3d.rs     smoke/heat validation (heat diffusion, buoyancy, hydrostatics)
    tests/orbits.rs    N-body orbits (Kepler's third law, Earth's year, the Moon's month, conservation)
  gpu/      the SPH solvers as wgpu compute shaders (sph.wgsl, sph3d.wgsl); tests check they agree with the CPU
  validation/ runs the engines against exact results and writes physicsengine.dev/validation
            (cargo run --release -p validation [-- OUTPUT_DIR]; exits non-zero if a check fails)
  editor/   eframe/egui app: canvas, tools, inspector, GPU particle renderer, and the 3D view
            (src/view3d: orbit camera, instanced mesh renderer; src/orbits: the Orbits tab;
            src/theme.rs: light and dark themes). Builds native and wasm.
scenes/     example scene files
index.html  physicsengine.dev landing page
.github/workflows/
  ci.yml      fmt, clippy, tests, desktop + web builds
  deploy.yml  publishes physicsengine.dev from main (landing page, /sim/, /scenes/, /validation/)
```

Units are SI throughout: meters, kilograms, seconds, radians. 3D density is kg/m³ (water 1000, default body 500, so a default body floats half-submerged). 2D density is kg/m². Water is 10 kg/m² (1000 kg/m³ × 1 cm), the same scale as body densities: a body of density 5 floats half-submerged, and anything above 10 sinks.

The desktop build runs the fluid on all CPU cores (the engine's `parallel` feature), or on the GPU with **Simulate on the GPU** (in the Fluid panel in 2D, the Water panel in 3D); the web build uses one CPU core. Fluid particles are drawn by the GPU everywhere. To measure performance on your machine:

```sh
cargo run --release -p gpu --example bench_gpu --features parallel   # CPU vs GPU at 5k / 20k / 100k particles
cargo run --release -p engine --example bench_fluid --features parallel
cargo bench -p engine                                                 # Criterion micro-benchmarks
```

The original C++ prototype lives on the [`legacy-cpp`](https://github.com/Cdustin28/PhysicsEngine/tree/legacy-cpp) branch.

## License

[MIT](LICENSE)
