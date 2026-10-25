# PhysicsEngine

A 2D rigid-body and particle-fluid physics engine with an interactive editor, written in Rust. It runs on Windows, macOS, Linux, and in the browser. 3D is planned once 2D is solid.

Website: [physicsengine.dev](https://physicsengine.dev) · Plan: [ROADMAP.md](ROADMAP.md)

> **Status:** early (milestone M0). Bodies fall under gravity and can be placed, dragged, thrown, saved, and loaded. Collisions (M1) and SPH fluids (M2) are next.

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
| Pan | Right- or middle-drag |
| Zoom | Scroll wheel or pinch |
| Play / pause | Space |
| Single step | `.` |
| Reset to before Play | R |
| Tools | V select · C circle · B box · Esc back to select |
| Delete selected | Delete / Backspace |

Scenes save as human-readable JSON. **Save…/Open…** use file dialogs on desktop, and **Copy JSON/Paste JSON…** work everywhere. There's an example in [`scenes/demo.json`](scenes/demo.json).

## Layout

```
crates/
  engine/   physics only: shapes, bodies, world stepping, scene files. No GUI deps.
  editor/   eframe/egui app: canvas, tools, inspector. Builds native and wasm.
scenes/     example scene files
index.html  physicsengine.dev landing page (served by GitHub Pages)
```

Units are SI throughout: meters, kilograms, seconds, radians. 2D density is kg/m².

The original C++ prototype lives on the [`legacy-cpp`](https://github.com/Cdustin28/PhysicsEngine/tree/legacy-cpp) branch.

## License

[MIT](LICENSE)
