# PhysicsEngine

A 2D rigid-body and particle-fluid physics engine with an interactive editor, written in Rust. It runs on Windows, macOS, Linux, and in the browser. 3D is planned once 2D is solid.

Website: [physicsengine.dev](https://physicsengine.dev) · Simulator in your browser: [physicsengine.dev/sim](https://physicsengine.dev/sim/) · Plan: [ROADMAP.md](ROADMAP.md)

> **Status:** milestone M4. Rigid boxes and circles collide, stack, bounce, and slide with friction. SPH water flows around them, and bodies float, sink, and get pushed by it. The editor has undo, multi-select, snapping, and runs in the browser. GPU acceleration (M5) is next.

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
| Tools | V select · C circle · B box · F fluid · E erase · Esc deselect |
| Delete selected | Delete / Backspace |
| Contact overlay | D, or the checkbox under World |
| Add water | F, then drag a rectangle (or switch to Brush and paint) |
| Erase water | E, then drag |
| Fluid force arrows | Fluid panel → Show forces on bodies (an arrow the length of the body's weight is 1 m) |

Scenes save as human-readable JSON. **Open…/Save…** use file dialogs on desktop; in the browser, Open uses the browser's file picker and Save downloads `scene.json`. **Copy JSON/Paste JSON…** work everywhere. Your settings and current scene are remembered between sessions. **Examples** loads the built-in scenes (demo, 55-box pyramid, friction ramps, bouncing balls, dam break, buoyancy). The same scenes are in [`scenes/`](scenes); regenerate them with `cargo run -p engine --example make_scenes`.

## Layout

```
crates/
  engine/   physics only: shapes, bodies, collision, contact solver, SPH fluid, scenes. No GUI deps.
    tests/physics.rs   rigid-body validation against known answers
    tests/fluid.rs     fluid validation (hydrostatics, dam break, leaks)
    tests/coupling.rs  fluid ↔ body validation (Archimedes, momentum exchange)
    examples/          make_scenes (regenerate scenes/), bench_fluid (performance)
  editor/   eframe/egui app: canvas, tools, inspector. Builds native and wasm.
scenes/     example scene files
index.html  physicsengine.dev landing page
.github/workflows/
  ci.yml      fmt, clippy, tests, desktop + web builds
  deploy.yml  publishes physicsengine.dev from main (landing page, /sim/, /scenes/)
```

Units are SI throughout: meters, kilograms, seconds, radians. 2D density is kg/m². Water is 10 kg/m² (1000 kg/m³ × 1 cm), the same scale as body densities: a body of density 5 floats half-submerged, and anything above 10 sinks.

The desktop build runs the fluid on all CPU cores (the engine's `parallel` feature); the web build is single-threaded. To measure fluid performance on your machine:

```sh
cargo run --release -p engine --example bench_fluid --features parallel
```

The original C++ prototype lives on the [`legacy-cpp`](https://github.com/Cdustin28/PhysicsEngine/tree/legacy-cpp) branch.

## License

[MIT](LICENSE)
