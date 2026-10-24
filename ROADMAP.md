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

## M3: Two-way fluid ↔ rigid-body coupling ✅

**Goal:** fluid pushes bodies and bodies push fluid. Boxes float, sink, and get carried by currents.

- [x] Rigid-body surfaces sampled with boundary particles that move with the body (Akinci et al. 2012). This landed in M2, where bodies already pushed fluid.
- [x] Fluid pressure and wall viscosity on boundary particles → force and torque on the body. It's the exact reaction to the boundary force each fluid particle feels, accumulated over the fluid substeps and applied as an external force in the rigid-body step.
- [x] Body velocity → boundary particle velocity → drag on the fluid
- [x] **Boundary inset:** boundary particles sit 0.55 particle spacings inside each surface. Still water rests that far off a boundary layer (measured 0.59 at walls, 0.48 at the floor). Before the inset, every body acted about 3 cm too big and floated high (submerged 0.16/0.35/0.58 where 0.25/0.50/0.75 was expected).
- [x] **Editor:** blue arrow from each body's center showing the fluid force (scaled so the body's weight would be 1 m, so a resting floater shows exactly 1 m); inspector shows force, force/weight, and torque; Buoyancy example
- [x] **Tests** (`crates/engine/tests/coupling.rs`):
  - boxes of density 2.5 / 5 / 7.5 float with 0.224 / 0.470 / 0.733 submerged (Archimedes: 0.25 / 0.50 / 0.75; the test allows ±0.08)
  - a density-20 box sinks to the bottom
  - sinking a box raises the water level by its area / tank width (0.112 m measured vs 0.100 m expected)
  - a wide, light box dropped in tilted at 0.5 rad rights itself (0.05 rad)
  - momentum is exchanged exactly between water and a coasting box (0.000% error over 0.5 s; the box slows from 2.0 to 1.0 m/s)

**Done when:** you can drop a box into a tank and it bobs and settles at the waterline its density predicts. ✅

**Known limitation:** a sunk body rests on a film of water about one particle thick (4 cm at the default spacing) instead of touching the floor. A single layer of particles squeezed between two boundaries can't drain in SPH. It shows in the level-rise test as about +10%. Possible fixes are letting rigid contact see through a thin film, or a dedicated squeeze-out term.

**Also tuned:** with the inset, fluid gets closer to walls, and hard dam-break impacts started hitting the speed cap again. Wall viscosity went from 0.3 to 0.5, which removed it; the dam-break front is unchanged.

---

## M4: Web release and editor polish ✅

**Goal:** the simulator runs live at physicsengine.dev.

- [x] **Deploy workflow** (`.github/workflows/deploy.yml`): on every push to `main`, builds the landing page at `/`, the simulator at `/sim/` (Trunk), and the example scenes at `/scenes/`, then publishes to GitHub Pages. **One-time setup:** Settings → Pages → *Source: GitHub Actions*.
- [x] Landing page: a *Launch the simulator* button; the GitHub link now points at the real repository
- [x] **Open/save in the browser:** *Open…* uses the browser's file picker; *Save…* downloads `scene.json`. Desktop keeps native dialogs. Copy/Paste JSON still works everywhere.
- [x] **Undo/redo** (Ctrl+Z, Ctrl+Shift+Z / Ctrl+Y, toolbar buttons): snapshot-based. After any frame where nothing is in progress (no button held, simulation paused), the scene is compared with the last recorded one, so a whole drag is one step and every kind of edit is undoable without per-edit code. Running the simulation isn't recorded as an edit; Reset covers that. 100 steps are kept. Copies are cheap because fluid scratch space isn't cloned.
- [x] **Stable body handles:** `BodyHandle`s are never reused and survive other bodies being removed (`World::find`). Selection and undo use them, so a selected body stays selected through undo. (Unique handles rather than slot+generation pairs, which gives the same guarantees without changing how the engine stores bodies.)
- [x] **Multi-select:** Shift+click to add or remove, drag on empty space to box-select, Ctrl+A for all; drag any selected body to move the group; Make static/dynamic and Delete for the whole selection
- [x] **Handles:** round = rotate (snaps to 15°), square = resize (boxes resize symmetrically, circles by radius)
- [x] **Grid snapping** (G or the toolbar checkbox): placement, moves, and sizes snap to a quarter of the visible grid; a group keeps its shape
- [x] **Example gallery:** the built-in Examples menu (works offline and on the web); the same scenes are published at `/scenes/`
- [x] **Persistence:** camera, tool settings, display options, step rate, and the current scene are saved between sessions (desktop: the OS data folder; web: local storage), autosaving every 30 s and on exit

**Done when:** someone with only a browser can open physicsengine.dev, build a scene, and share its JSON. This happens once the PR stack is merged and the Pages source is switched. The web build has passed CI since M0, but it hasn't been tried in a real browser from this environment.

---

## M5: Performance and GPU ✅ (performance target unverified)

**Goal:** move from thousands to 100k+ particles.

- [x] **Benchmarks** (`cargo bench -p engine`, Criterion): neighbor grid build and query (5k points), a dam-break world step, and a pyramid world step. Build machine, single-threaded: grid build 54 µs, grid query 327 µs, dam-break step (1.5k particles) 9.8 ms, pyramid step 151 µs.
- [x] **`FluidSolver` interface:** `World::step_with(dt, &mut solver)`. `CpuSolver` is the reference; the engine stays free of graphics dependencies.
- [x] **GPU SPH** (`crates/gpu`, `sph.wgsl`): six compute passes per substep (clear, insert, density, forces, boundary impulse, integrate). The neighbor grid is a hashed table of per-bucket linked lists built with `atomicExchange` in one pass, chosen over a counting sort because it needs no prefix sum and is simpler to verify. Buffers are packed into vec4s to fit WebGPU's limit of 8 storage buffers per stage. M3 coupling runs on the GPU too: each boundary particle accumulates its reaction impulse, and the CPU sums those into body forces and torques.
- [x] **CPU path kept as fallback and reference; GPU agrees with it.** `crates/gpu/tests/agreement.rs` checks one step from the same state (worst position difference 0.00003 m, 0.001 spacings; mean density identical to 4 digits), the hydrostatic gradient, Archimedes floating, and exact momentum exchange. They run in CI on lavapipe (software Vulkan) and skip only if no adapter exists at all.
- [x] **GPU particle rendering** (`crates/editor/src/particles.rs`): one instanced draw call through an egui paint callback, replacing per-particle egui circles. Works on every wgpu backend, including WebGL2.
- [x] **Editor:** Fluid panel → *Simulate on the GPU* (desktop, when the adapter supports compute shaders); the status bar shows which solver is running
- [ ] **Web GPU compute:** not yet. The GPU solver reads results back by blocking until the GPU finishes, which browsers forbid. The web build therefore uses the CPU solver (GPU rendering still works). It needs an asynchronous, one-frame-delayed readback.
- [ ] **Done when (100k particles interactive on a discrete GPU):** *not verified.* The build machine has no GPU; on lavapipe (a GPU emulated on 2 CPU cores) the GPU path is slower than the native CPU solver, which says nothing about real hardware. Run `cargo run --release -p gpu --example bench_gpu --features parallel` on your machine for real numbers at 5k, 20k, and 100k.

**Known costs to revisit if the benchmark disappoints:** 78 compute passes per world step (could be fused); a full upload and readback every world step (could stay on the GPU between steps when not editing); the worst-case substep count (13 vs 9–12 adaptive on the CPU).

**Sleeping** (deferred from M1) is still open; it matters most for large rigid-body scenes.

---

## M6: 3D

**Goal:** the same capabilities in three dimensions. Split in two: **M6a** rigid bodies and the 3D editor (done), **M6b** 3D fluid.

### M6a: 3D rigid bodies and editor ✅

- [x] A parallel `engine3d` crate rather than a dimension-generic core: the 2D code stays simple, and the two share structure, not types.
- [x] `Vec3`/`Quat` from `glam`; world-space inverse inertia tensors (R I⁻¹ Rᵀ); quaternion integration q += ½ ω q dt, renormalized each step
- [x] Spheres and boxes. Box–box uses SAT over all 15 axes (3 + 3 face normals, 9 edge cross products) with Sutherland–Hodgman face clipping (up to 4 contact points) and edge–edge contacts. GJK/EPA isn't needed until there are general convex shapes.
- [x] Sequential-impulse solver as in 2D: warm starting (contacts matched by position within 2 cm, since 3D manifolds don't carry stable feature ids), split-impulse position correction, restitution threshold
- [x] Patch friction: one friction impulse per manifold at its center of pressure, clamped to the Coulomb disc, plus twisting friction. Per-point friction made sliding boxes spin and slide 11% too far; friction at the geometric center (not weighted by load) made a 6-box tower sway forever.
- [x] Ray casts for mouse picking
- [x] 3D scene files (`"format": "physicsengine-scene-3d"`, version 1), separate from 2D's; `scenes/3d/`
- [x] Editor: 2D/3D switch; GPU renderer (instanced cube and sphere meshes with a depth buffer); orbit camera; click to select, drag to move in the ground plane (Shift: vertically), throw while running; Sphere/Box tools place on whatever is under the mouse; inspector; undo/redo, Play/Reset, Open/Save/Copy/Paste, examples
- [x] `tests/physics3d.rs`, against closed-form answers (120 Hz):

  | Test | Expected | Measured |
  | --- | --- | --- |
  | Elastic head-on, equal spheres | velocities swap | exact to 1e-8 |
  | Block on 35° ramp, μ = 0.5, 1 s | ½ g (sin θ − μ cos θ) t² = 0.804 m | 0.798 m |
  | Block on 20° ramp, μ = 0.5 | holds (tan 20° = 0.36 < μ) | 0.7 mm creep |
  | Ball rolling down 20°, 1 s | ½ (5/7) g sin θ t² = 1.198 m, ω r = v | 1.187 m, ω r / v = 1.001 |
  | Bounce from 3 m, e = 0.5 / 0.9 | e² h = 0.75 / 2.43 m | 0.751 / 2.485 m |
  | 6-box tower, 5 s | stands, KE → 0 | drift 2 mm, KE < 1e-6 J |
  | 30-cube pyramid, 5 s | holds | worst drift 0.7 mm |

- [ ] Translate/rotate gizmos and multi-select (the 2D editor has them; 3D has drag-to-move and the inspector for now)
- [ ] Shadows (they'd help judge depth)

### M6b: 3D fluid

- [ ] 3D SPH (CPU): cubic-spline kernel in 3D, 3D grid, boundary particles on box and sphere surfaces, water at 1000 kg/m³
- [ ] Two-way coupling with 3D bodies (buoyancy, drag, torque)
- [ ] GPU SPH in 3D (the M5 compute shaders with a 3D grid)
- [ ] 3D particle rendering (instanced sphere impostors)

---

## Stretch goals

- Joints and constraints (pin, hinge, spring), plus ropes and chains
- Soft bodies (mass–spring, or extended position-based dynamics)
- Grid-based (Eulerian) smoke and dye solver as an alternative fluid model
- Heat transfer and multiphase fluids
- A validation page comparing simulations against analytic and experimental results
