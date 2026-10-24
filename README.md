# PhysicsEngine

A 2D rigid-body and particle-fluid physics engine with an interactive editor, written in Rust. It runs on Windows, macOS, Linux, and in the browser. 3D is planned once 2D is solid.

Website: [physicsengine.dev](https://physicsengine.dev) · Plan: [ROADMAP.md](ROADMAP.md)

> **Status:** milestone M2. Rigid boxes and circles collide, stack, bounce, and slide with friction, and SPH water flows around them. Two-way coupling (floating and sinking, M3) is next.

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
| Move a body | Left-drag with the Select tool. Release while playing to throw it. |
| Rotate a body | Select it, then drag the gold handle |
| Pan | Right- or middle-drag |
| Zoom | Scroll wheel or pinch |
| Play / pause | Space |
| Single step | `.` |
| Reset to before Play | R |
| Tools | V select · C circle · B box · Esc back to select |
| Delete selected | Delete / Backspace |
| Contact overlay | D, or the checkbox under World |
| Add water | F, then drag a rectangle (or switch to Brush and paint) |
| Erase water | E, then drag |

Scenes save as human-readable JSON. **Save…/Open…** use file dialogs on desktop, and **Copy JSON/Paste JSON…** work everywhere. **Examples** loads the built-in scenes (demo, 55-box pyramid, friction ramps, bouncing balls, dam break). The same scenes are in [`scenes/`](scenes); regenerate them with `cargo run -p engine --example make_scenes`.

## Layout

```
crates/
  engine/   physics only: shapes, bodies, collision, contact solver, SPH fluid, scenes. No GUI deps.
    tests/physics.rs   rigid-body validation against known answers
    tests/fluid.rs     fluid validation (hydrostatics, dam break, leaks)
    examples/          make_scenes (regenerate scenes/), bench_fluid (performance)
  editor/   eframe/egui app: canvas, tools, inspector. Builds native and wasm.
scenes/     example scene files
index.html  physicsengine.dev landing page (served by GitHub Pages)
```

Units are SI throughout: meters, kilograms, seconds, radians. 2D density is kg/m². Water is 10 kg/m² (1000 kg/m³ × 1 cm), the same scale as body densities.

The desktop build runs the fluid on all CPU cores (the engine's `parallel` feature); the web build is single-threaded. To measure fluid performance on your machine:

```sh
cargo run --release -p engine --example bench_fluid --features parallel
```

The original C++ prototype lives on the [`legacy-cpp`](https://github.com/Cdustin28/PhysicsEngine/tree/legacy-cpp) branch.

## License

[MIT](LICENSE)
