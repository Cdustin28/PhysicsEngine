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

## M0: Scaffold ✅

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

## M1: 2D rigid-body collisions ✅

**Goal:** stacks of boxes and circles that collide, bounce, slide, and come to rest without jitter.

- [x] **Broad phase:** sort-and-sweep on AABBs (`broad_phase.rs`), tested against a brute-force O(n²) reference on 50 random scenes
- [x] **Narrow phase** (`collision.rs`):
  - [x] circle–circle
  - [x] circle–box (closest point on a rotated box, including center-inside)
  - [x] box–box with the Separating Axis Theorem, plus reference-face clipping for a two-point manifold
- [x] **Contact solver** (`solver.rs`): sequential impulses with accumulated impulse clamping, restitution, and Coulomb friction
- [x] **Stability:**
  - [x] Warm starting, matched across steps by feature id
  - [x] Block solver for two-point contacts. Solving the two points one at a time left a residual twist that made boxes creep.
  - [x] Split-impulse position correction. Plain Baumgarte made overlapping bodies fly apart and dead (e = 0) balls bounce.
- [ ] **Sleeping:** deferred. Stacks already come fully to rest, so it's only a performance optimization now. Revisit in M5.
- [x] **Editor:** contact point/normal overlay (D), rotate handle on the selected body, solver settings panel, Examples menu
- [x] **Tests** (`crates/engine/tests/physics.rs`), all checked against known answers:
  - equal-mass elastic collision swaps velocities; unequal masses match the 1D elastic formulas
  - momentum is conserved in glancing box–circle collisions
  - a 10-box stack and the 55-box pyramid stand for 10 s (the pyramid's worst drift is about 2 mm)
  - a block holds on a 20° ramp and slides on a 35° ramp at g(sin θ − μ cos θ), within 5%
  - bounce height ≈ e²h for e = 0.5 and 0.9; e = 0 doesn't bounce
  - overlapping bodies separate without being launched

**Done when:** a pyramid of 50+ boxes stands stable at 120 Hz, and dropped objects bounce according to their restitution. ✅

**Scene format:** still version 1. `solver` is a new *optional* field, so older scenes load with default settings.

**References:** Erin Catto's GDC talks and the *Box2D Lite* source (a small, readable sequential-impulse solver); Box2D v2.4's `b2_contact_solver.cpp` (block solver); Christer Ericson, *Real-Time Collision Detection*; Ian Millington, *Game Physics Engine Development* (the same book your C++ prototype followed).

**Rust pattern used:** the solver copies velocities into flat `Vec`s indexed by body, so it can update two bodies at once without needing two `&mut` borrows into `bodies`. M2's struct-of-arrays fluid uses the same idea.

---

## M2: SPH fluids (CPU) ✅

**Goal:** a few thousand fluid particles splashing around a container at interactive frame rates.

- [x] `Fluid` stored as struct-of-arrays (`positions`, `velocities`, plus scratch `densities`, `pressures`, `accelerations`), the layout M5's GPU port will want
- [x] **Neighbor search** (`fluid/grid.rs`): a dense row-major cell grid over the particles' bounding box (3 contiguous reads per query), with a spatial hash as fallback when particles are spread thin. Tested against brute force.
- [x] **Kernel:** 2D cubic spline (Monaghan), the kernel the WCSPH paper and SPlisHSPlasH use. It replaces the poly6/spiky set originally listed here, so one kernel serves density and pressure. Tested for normalization (∫W = 1) and against a finite-difference gradient.
- [x] **Pressure:** weakly compressible SPH with the Tait equation (γ = 7, B = ρ₀c²/7), clamped at zero at the free surface; Monaghan artificial viscosity
- [x] **Time step:** CFL-limited substeps inside each 120 Hz world step (dt ≤ 0.4·d/(c + v_max)); particle speed is capped at c/2, which bounds the substep count
- [x] **Boundaries:** single-layer boundary particles with Akinci et al. (2012) ψ weights, sampled on *every* body. Bodies already push fluid in M2; M3 adds the force back on bodies. A projection pass is a safety net for anything that slips through.
- [x] **Editor:** Fluid tool (rectangle fill or brush), Eraser, color by speed or pressure, Fluid panel (spacing, density, stiffness, viscosity, wall friction, live substeps and compression), Dam break example
- [x] **Scene format v2:** optional `fluid` block; version 1 files load unchanged. Scene JSON now keeps vectors on one line (`[x, y]`), so a 1,500-particle scene is 42 KB.
- [x] **Multithreading:** `parallel` feature (rayon), on for desktop and off for web and `cargo test -p engine`
- [x] **Tests** (`crates/engine/tests/fluid.rs`):
  - hydrostatic pressure gradient = ρ₀g within 10% (time-averaged over 1 s; measured within 3%)
  - a settled column keeps its volume, and no particles are lost
  - dam-break front stays below the ideal Ritter speed 2√(gH) and covers more than 30% of it
  - no particle escapes a tank after a 1.5 m dam break slams into the far wall
  - a blob fired at the floor at 8 m/s doesn't tunnel in
  - a moving body pushes still water (and none ends up inside it)
- [ ] **Performance target (5,000 particles at 60 fps):** *not yet verified.* `cargo run --release -p engine --example bench_fluid` on the 2-core build machine: 4,900 particles take 32 ms per 1/120 s step single-threaded and 21 ms on 2 threads, against 8.3 ms for real time. Scaling suggests roughly 6 ms on an 8-core desktop, which would be real time. Run the benchmark with `--features parallel` on your machine to check. The 1,500-particle Dam break example runs at about real time on one thread. Closing the gap for sure is M5's job.

**Lessons worth remembering:**
- Single-layer boundary particles over-count density within about one spacing of a wall. Fluid placed (or projected) that close is effectively heavily compressed and gets launched. Fills and the projection safety net therefore keep a 0.9-spacing gap.
- Wall viscosity of 0.3 (versus 0.05 in the fluid) damps particles before they reach that zone. It stopped spray launching off walls in hard impacts, and the dam-break front barely changed (1.75 m → 1.67 m at 0.5 s).
- WCSPH pressure is noisy, so measure it averaged over time. A single snapshot read 12% low, but the 1 s average is within 3%.

**Martin & Moyce comparison:** the original plan compared the dam-break front with Martin & Moyce's 1952 experiments. That's replaced by the Ritter upper bound for now, since I didn't want to test against experimental values from memory. Adding the digitized data is a good follow-up.

**References:** Becker & Teschner, "Weakly Compressible SPH for Free Surface Flows" (2007); Monaghan, "Smoothed Particle Hydrodynamics" (1992); Akinci et al., "Versatile Rigid-Fluid Coupling for Incompressible SPH" (2012); Koschier et al., "SPH Techniques for the Physics-Based Simulation of Fluids and Solids" (Eurographics 2019 tutorial); Teschner et al., "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (2003).

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
