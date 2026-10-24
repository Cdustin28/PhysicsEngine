# Roadmap

PhysicsEngine is a 2D-first rigid-body and particle-fluid simulator with an interactive editor. It is written in Rust and runs on Windows, macOS, Linux, and in the browser. 3D comes after the 2D engine is solid.

## Ground rules

These apply to every milestone.

- **SI units everywhere.** Meters, kilograms, seconds, radians. In 2D, density is kg/m². Convert to pixels only when drawing.
- **Fixed time step.** `World::step(dt)` always gets the same `dt`. Frame-rate differences are absorbed by `FixedTimestep`. This keeps results deterministic and stable.
- **`engine` stays pure.** It has no GUI or graphics dependencies, so it can be unit-tested, benchmarked, and later driven by a GPU backend. All drawing and input live in `editor`.
- **Every physics feature ships with a test against a known answer.** Examples: a closed-form free fall, momentum conserved in a collision, Archimedes' principle for floating bodies. A simulation that looks right but fails these is a bug.
- **CPU first, GPU later.** Get each algorithm correct and readable on the CPU, then port it once it is well understood.
- **Scene files are versioned.** Any incompatible change to the JSON format bumps `Scene::VERSION`, and `Scene::from_json` gets a migration so old scenes keep loading.

---

## M0: Scaffold ✅ (this PR)

**Goal:** a working app skeleton you can build on, on every platform.

- [x] Cargo workspace with `engine` (physics) and `editor` (eframe/egui app) crates
- [x] Rigid bodies with circle and box shapes; mass and moment of inertia derived from density
- [x] Semi-implicit Euler integration, gravity, force/torque accumulators
- [x] `FixedTimestep` accumulator with spiral-of-death protection
- [x] Editor: place circles and boxes, select, drag to move, throw while running, inspector, pan/zoom, play/pause/step/reset
- [x] JSON scene save/load (file dialogs on desktop, copy/paste JSON on both desktop and web)
- [x] CI: fmt, clippy, tests, desktop builds on Windows/macOS/Linux, and a web (Trunk) build
- [x] Old C++ prototype preserved on the `legacy-cpp` branch

**Rust to learn first:** *The Rust Book*, chapters 1–10. Work through the `rustlings` exercises alongside it. The code in this milestone exercises structs, enums with data (`Shape`), `match`, `Option`/`Result`, traits (`Default`, `Display`), modules, and unit tests.

---

## M1: 2D rigid-body collisions

**Goal:** stacks of boxes and circles that collide, bounce, slide, and come to rest without jitter.

- [ ] **Broad phase:** a uniform grid or sort-and-sweep on bounding circles/AABBs to find candidate pairs. Start with brute-force O(n²) as the reference implementation and test the faster one against it.
- [ ] **Narrow phase:**
  - [ ] circle–circle
  - [ ] circle–box (closest point on a rotated box)
  - [ ] box–box with the Separating Axis Theorem, plus clipping to get a two-point contact manifold
- [ ] **Contact solver:** sequential impulses with accumulated impulse clamping, restitution, and Coulomb friction
- [ ] **Stability:** position correction (Baumgarte or split impulses), warm starting, multiple solver iterations
- [ ] **Sleeping** (optional): bodies at rest stop integrating until something touches them
- [ ] **Editor:** debug overlay for contact points and normals, and a rotate handle on the selected body
- [ ] **Tests:**
  - momentum conserved in an elastic circle collision
  - a 10-box stack settles and stays put for 10 s
  - a body on a slope slides only when `tan θ > μ`

**Done when:** a pyramid of 50+ boxes stands stable at 120 Hz, and dropped objects bounce according to their restitution.

**References:** Erin Catto's GDC talks and the *Box2D Lite* source (a small, readable sequential-impulse solver); Christer Ericson, *Real-Time Collision Detection*; Ian Millington, *Game Physics Engine Development* (the same book your C++ prototype followed).

**Rust focus:** borrowing two bodies mutably at once. Use `split_at_mut` or index pairs, or compute impulses in one pass and apply them in a second.

---

## M2: SPH fluids (CPU)

**Goal:** a few thousand fluid particles splashing around a container at interactive frame rates.

- [ ] `Fluid` struct stored as struct-of-arrays (`positions: Vec<Vec2>`, `velocities`, `densities`, `pressures`). This layout is cache-friendly and maps directly to GPU buffers in M5.
- [ ] **Neighbor search:** a spatial hash grid with cell size equal to the smoothing radius *h*. This is the main performance lever.
- [ ] **Kernels:** poly6 for density; spiky gradient for pressure; viscosity Laplacian (Müller et al. 2003)
- [ ] **Pressure:** weakly compressible SPH using the Tait equation of state (Becker & Teschner 2007). Optionally move to PBF or DFSPH later for less compressibility.
- [ ] **Time step:** CFL-limited sub-stepping, since SPH usually needs smaller steps than rigid bodies
- [ ] **Static boundaries:** walls as boundary particles, or a signed-distance clamp
- [ ] **Editor:** fluid brush and rectangle-fill tools, particle spacing and rest density controls, coloring by speed or pressure
- [ ] **Scene format v2:** fluid particles and parameters, with a v1 → v2 migration
- [ ] **Tests:**
  - neighbor grid matches brute force
  - a resting column of fluid holds its density within a few percent of rest density
  - dam-break front position roughly tracks the Martin & Moyce (1952) experimental data

**Done when:** a 2D dam break with ~5,000 particles runs at 60 fps on a laptop in a desktop release build.

**References:** Müller, Charypar & Gross, "Particle-Based Fluid Simulation for Interactive Applications" (2003); Becker & Teschner, "Weakly Compressible SPH" (2007); Koschier et al., "SPH Techniques for the Physics-Based Simulation of Fluids and Solids" (Eurographics 2019 tutorial). The last one is the best single overview.

**Rust focus:** iterators, `Vec` layout, and profiling with `cargo build --release` plus a profiler such as `puffin` or `cargo flamegraph`. Add `rayon` (desktop only) for parallel loops.

---

## M3: Two-way fluid ↔ rigid-body coupling

**Goal:** fluid pushes bodies and bodies push fluid. Boxes float, sink, and get carried by currents.

- [ ] Sample rigid-body surfaces with boundary particles that move with the body (Akinci et al. 2012, "Versatile Rigid-Fluid Coupling")
- [ ] Fluid pressure on boundary particles → force and torque on the body
- [ ] Body velocity → boundary particle velocity → drag on the fluid
- [ ] Editor: show the buoyancy force vector on the selected body
- [ ] **Tests (Archimedes):**
  - a body with half the fluid's density floats about 50% submerged
  - a body denser than the fluid sinks
  - a uniform box floats level

**Done when:** you can drop a box into a tank and it bobs and settles at the waterline its density predicts.

---

## M4: Web release and editor polish

**Goal:** the simulator runs live at physicsengine.dev.

- [ ] GitHub Actions deploy: the landing page plus the Trunk build under `/sim/`. This needs GitHub Pages switched from "deploy from branch" to "GitHub Actions".
- [ ] Browser-native scene file open/save (download and upload), in addition to copy/paste
- [ ] Undo/redo using command history or world snapshots
- [ ] Generational body handles, so deleting a body never invalidates others
- [ ] Multi-select, rotate/resize handles, grid snapping
- [ ] An example scene gallery (dam break, stacking, floating boxes) loaded from `scenes/`
- [ ] Settings and last scene persisted between sessions (eframe `persistence`)

**Done when:** someone with only a browser can open physicsengine.dev, build a scene, and share its JSON.

---

## M5: Performance and GPU

**Goal:** move from thousands to 100k+ particles.

- [ ] Benchmarks with `criterion` for the neighbor search and the SPH step
- [ ] Port the SPH step to wgpu compute shaders (WGSL), with GPU neighbor search via counting sort on cell index
- [ ] Render particles with a custom wgpu pipeline through an egui paint callback (instanced quads), instead of one egui shape per particle
- [ ] Keep the CPU path as a fallback and as a correctness reference: GPU results must match within a tolerance
- [ ] Web: compute only where WebGPU is available, with automatic fallback to CPU

**Done when:** 100k particles at interactive rates on a discrete GPU, and the CPU and GPU paths agree in tests.

---

## M6: 3D

**Goal:** the same capabilities in three dimensions.

- [ ] Decide between a parallel `engine3d` module and a dimension-generic core. A parallel module is probably simpler for a first pass.
- [ ] `Vec3`/`Quat` from `glam`, inertia tensors, and angular velocity integration with quaternions
- [ ] Spheres and boxes; GJK + EPA for convex collision; 3D contact manifolds
- [ ] 3D SPH, which is a mostly mechanical extension of M2 and M5
- [ ] wgpu 3D renderer with an orbit camera; translate/rotate gizmos (e.g. `transform-gizmo-egui`)
- [ ] Scene format v3 with 3D bodies

---

## Stretch goals

- Joints and constraints (pin, hinge, spring), plus ropes and chains
- Soft bodies (mass–spring, or extended position-based dynamics)
- Grid-based (Eulerian) smoke and dye solver as an alternative fluid model
- Heat transfer and multiphase fluids
- A validation page comparing simulations against analytic and experimental results
