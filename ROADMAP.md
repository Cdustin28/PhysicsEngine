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
- [x] **Performance target (5,000 particles at 60 fps):** met on the GPU, not quite on the CPU. Measured on a desktop with 24 CPU threads and an RTX 3070 Ti, where real time needs ≤ 8.3 ms per 1/120 s step:
  - `bench_fluid --features parallel` (CPU, 24 threads): 5,040 particles, 11.6 ms per step (139% of real time), 12.7 substeps per step, 0.18 µs per particle-substep. Going from 2 threads (21 ms) to 24 gave only 1.8×: about 210 particles per thread is too little work to hide the per-pass threading overhead, and the neighbor-grid rebuild runs on one thread.
  - `bench_gpu --features parallel`, dam break: CPU 9.3 ms, **GPU 3.6 ms** at 5,041 particles.
  - CPU speed-up afterwards (same physics): the neighbor search, density, and pressure run as one pass, so each substep has three parallel passes instead of six; the grid uses half-size cells and a 5×5 (3D: 5×5×5) search, checking 30% (3D: 42%) fewer candidates; the grid keeps a copy of positions so a search reads one contiguous array; the force pass reads one packed record per neighbor; `floor` is done inline (it's a library call on baseline x86-64); statistics are computed once per step instead of every substep. On the 2-core build machine: **45.8 → 33.0 ms single-threaded (1.39×), 29.0 → 22 ms on 2 threads**. Re-run on the 24-thread desktop: pending.

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
- [x] **Dark mode** (toolbar toggle, remembered between sessions): dark grey panels and canvas in 2D and 3D. Colors that would vanish on dark grey switch with it: joints and ropes lighten, floors dim, and smoke turns pale and lightens what's behind it (screen blending) instead of darkening it.
- [x] **Persistence:** camera, tool settings, display options, step rate, and the current scene are saved between sessions (desktop: the OS data folder; web: local storage), autosaving every 30 s and on exit

**Done when:** someone with only a browser can open physicsengine.dev, build a scene, and share its JSON. This happens once the PR stack is merged and the Pages source is switched. The web build has passed CI since M0, but it hasn't been tried in a real browser from this environment.

---

## M5: Performance and GPU ✅

**Goal:** move from thousands to 100k+ particles.

- [x] **Benchmarks** (`cargo bench -p engine`, Criterion): neighbor grid build and query (5k points), a dam-break world step, and a pyramid world step. Build machine, single-threaded: grid build 54 µs, grid query 327 µs, dam-break step (1.5k particles) 9.8 ms, pyramid step 151 µs.
- [x] **`FluidSolver` interface:** `World::step_with(dt, &mut solver)`. `CpuSolver` is the reference; the engine stays free of graphics dependencies.
- [x] **GPU SPH** (`crates/gpu`, `sph.wgsl`): six compute passes per substep (clear, insert, density, forces, boundary impulse, integrate). The neighbor grid is a hashed table of per-bucket linked lists built with `atomicExchange` in one pass, chosen over a counting sort because it needs no prefix sum and is simpler to verify. Buffers are packed into vec4s to fit WebGPU's limit of 8 storage buffers per stage. M3 coupling runs on the GPU too: each boundary particle accumulates its reaction impulse, and the CPU sums those into body forces and torques.
- [x] **CPU path kept as fallback and reference; GPU agrees with it.** `crates/gpu/tests/agreement.rs` checks one step from the same state (worst position difference 0.00003 m, 0.001 spacings; mean density identical to 4 digits), the hydrostatic gradient, Archimedes floating, and exact momentum exchange. They run in CI on lavapipe (software Vulkan) and skip only if no adapter exists at all.
- [x] **GPU particle rendering** (`crates/editor/src/particles.rs`): one instanced draw call through an egui paint callback, replacing per-particle egui circles. Works on every wgpu backend, including WebGL2.
- [x] **Editor:** Fluid panel → *Simulate on the GPU* (desktop, when the adapter supports compute shaders); the status bar shows which solver is running
- [ ] **Web GPU compute:** not yet. The GPU solver reads results back by blocking until the GPU finishes, which browsers forbid. The web build therefore uses the CPU solver (GPU rendering still works). It needs an asynchronous, one-frame-delayed readback.
- [x] **Done when (100k particles interactive on a discrete GPU):** met. `bench_gpu --features parallel` on an RTX 3070 Ti (24-thread CPU for comparison):

  | Particles | CPU (all cores) | GPU | GPU speed-up |
  |---|---|---|---|
  | 5,041 | 9.3 ms | 3.6 ms | 2.6× |
  | 19,881 | 19.6 ms | 4.0 ms | 4.9× |
  | 99,856 | 54.6 ms | 11.6 ms | 4.7× |

  At 100k, 11.6 ms per step is about 86 steps a second: interactive, at 72% of real time. The GPU time barely moves from 5k to 20k (3.6 → 4.0 ms), so roughly 3.4 ms per step is fixed cost (dispatching and waiting for the results); each extra particle costs about 0.095 µs.

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

### M6b: 3D fluid ✅

- [x] 3D SPH on the CPU (`engine3d::fluid`): the 2D solver's algorithm and constants with the 3D cubic spline (σ = 8/(πh³)), a 27-cell neighbor grid, and water at 1000 kg/m³. Default particle spacing is 0.1 m (a cubic meter of water is 1,000 particles; halving the spacing means 8×).
- [x] Boundary particles on box and sphere surfaces, sampled only near the water (a 20 m floor at fluid resolution would otherwise be 300,000+ points). Boundary spacing is 0.7 × the fluid's (2D uses 0.5): within 1 cm of Archimedes at 60% of the cost; 1.0 left sunk bodies resting on a film of water.
- [x] Two-way coupling with 3D bodies: buoyancy, drag, and torque (r × F), with exact equal-and-opposite impulses
- [x] GPU solver in 3D (`gpu::GpuSolver3d`, `sph3d.wgsl`), checked against the CPU solver
- [x] Editor: Water tool (W) drops a block of water onto whatever is under the mouse; water panel (resolution, viscosity, GPU toggle, drain); particles drawn as instanced low-poly spheres with depth; camera frames each loaded scene; Dam break and Buoyancy examples
- [x] `tests/fluid3d.rs` (CPU) and `gpu/tests/agreement3d.rs` (GPU):

  | Test | Expected | Measured |
  | --- | --- | --- |
  | Pressure at depth 0.2 / 0.3 / 0.4 m | ρgh = 1961 / 2942 / 3923 Pa | 1880 / 2941 / 3819 Pa |
  | Still water, 3 s; dam break against a wall | no leaks | 0 particles escape |
  | Dam break settles: 1,080 L over 3.6 m² | mean particle height 0.150 m | 0.147 m |
  | 0.3 m cubes at 500 / 250 kg/m³ | draft 0.150 / 0.075 m | 0.151 / 0.074 m |
  | 0.3 m cube at 2000 kg/m³ | rests on the floor (center 0.15 m) | 0.145 m |
  | Ball into a free blob, no gravity | momentum conserved | relative error 1.6e-7 |
  | GPU vs CPU, one dam-break step | same positions, density, wall force | within 2 mm, 1%, 2% |

- [ ] Rendering water as a surface (screen-space fluid rendering) instead of balls
- [ ] Performance: about 7 ms per step for 1,000 particles on 2 CPU cores. Cache static bodies' boundary particles between steps, and keep the GPU's data resident instead of copying it every step.

---

## M7: Joints, ropes, chains ✅

**Goal:** tie bodies together, in 2D and 3D.

- [x] Pin (2D hinge; 3D ball-and-socket), distance (rod; rope with slack), and spring (N/m, N·s/m, implicit soft constraint) joints; `World::add_joint`; jointed bodies don't collide with each other; removing a body removes its joints
- [x] Solved in the contact solver's sweeps with warm starting. Each sweep aims at the joint's *predicted end-of-step* state (linear in velocity, plus a fixed curvature term for each lever arm's swing): a SHAKE-style scheme with no long-term energy drift. The textbook velocity-only constraint lost 23% of a 60° pendulum's energy in 10 s.
- [x] Rigid-body substeps (`SolverSettings::substeps`, default 1). Chains whip; 4 substeps keep their pins within about a millimeter (1 substep: up to 1 cm).
- [x] `add_chain` builder; presets: Pendulums, Rope bridge, Newton's cradle, Springs (2D); Pendulums, Newton's cradle, Chain drape (3D)
- [x] Editor: Joint tool (J) and Chain tool (L) in 2D and 3D; joints drawn as pins, rods, ropes (pale when slack), zigzag springs; "Remove joints" in the inspector; substeps in the Solver panel
- [x] Scene format v3 (2D and 3D) adds `joints`; older files load
- [x] `tests/joints.rs` (2D) and `tests/joints3d.rs` (3D):

  | Test | Expected | Result |
  | --- | --- | --- |
  | Pendulum, 5° (disk/ball on a 1 m rod) | T = 2π√((l² + I/m)/(g l)) | within 0.5% |
  | Pendulum, 60°, 30 s | energy conserved | error bounded at ±1.2%, no drift |
  | Conical pendulum (3D), 10 s | L_y conserved, stays on its cone | within 1%, within 1° |
  | 1 kg on a 40 N/m spring | stretch mg/k, T = 2π√(m/k) = 0.993 s | within 1% |
  | Slack rope | no force until taut | free fall exact to 1e-4 m |
  | Newton's cradle (2D) | last ball leaves at ≈ the striking speed | > 90% of 3.03 m/s |
  | Whipping chain, 4 substeps | pins hold, no energy gain | gaps < 2 mm (2D), < 3 mm (3D) |

- [ ] Hinge with an axis in 3D, joint limits and motors
- [ ] Breakable joints

## M8: Soft bodies ✅

**Goal:** deformable bodies that collide with everything else, in 2D and 3D.

- [x] XPBD (Macklin et al. 2016, "small steps" 2019): edge constraints with physical stiffness (N/m), area (2D) and volume (3D) constraints held like gas pressure, 20 substeps per world step
- [x] Shapes: blob and jelly block (2D); pressurized ball (icosphere), cloth with stretch/shear/bend springs and pinnable corners, and jelly cube (3D)
- [x] Two-way collisions with rigid bodies (corrections split by generalized inverse mass, inertia tensor in 3D) with positional friction, and between soft bodies
- [x] Damping toward each body's rigid motion (Müller et al. 2007), so wobbles die out while momentum and spin are kept; pinned bodies damp toward rest. Area/volume constraints get XPBD constraint damping.
- [x] The substep loop runs in f64: at 80 substeps, a particle at rest 3 m up moved less than an f32 can resolve, and a blob hung in midair.
- [x] Editor: Soft tool (S) in 2D (blob, jelly) and 3D (ball, cloth, jelly); select, drag, throw, delete; inspector; 3D surfaces drawn as dynamic meshes
- [x] Scene format v4 adds `soft_bodies`
- [x] `tests/soft.rs` (2D) and `tests/soft3d.rs` (3D):

  | Test | Expected | Measured |
  | --- | --- | --- |
  | 5 × 0.1 kg hanging on 200 N/m springs | stretch m g n(n+1)/(2k) = 0.0735 m, at any substep count | 0.0733 m (10 substeps), 0.0729 m (40) |
  | 0.5 kg on a 50 N/m spring | T = 2π√(m/k) = 0.628 s | within 1% |
  | Blob dropped 3 m (2D) / ball 2.5 m (3D) | keeps its area/volume, no tunneling | within 0.7% / 3.1%; no particle below the floor |
  | Cloth onto a 0.6 m ball | rests on top, no particle inside | middle at 1.2624 m (ball top + particle radius: 1.2625) |
  | Jelly into a free crate, no gravity | momentum conserved | relative error 2e-9 (2D), 3e-7 (3D) |
  | Jelly bouncing | never gains energy | ✓ |

- [ ] Soft bodies in water (coupling with SPH)
- [ ] Tearing cloth; self-collision within one body

## M9: Smoke, dye, and heat ✅

**Goal:** air you can see: smoke and dye carried by the flow, and heat that rises, in 2D and 3D.

- [x] Grid (Eulerian) gas on a staggered MAC grid: semi-Lagrangian advection with a midpoint backtrace (Stam 1999), pressure projection by conjugate gradients with a MIC(0) preconditioner (137 → 23 iterations on a 48,000-cell grid)
- [x] Heat: carried by the flow, diffused implicitly (Fourier's law; insulated walls and obstacles), and driving buoyancy by the Boussinesq approximation a = −g (β ΔT − α s) with β = 1/T_ambient (an ideal gas)
- [x] Smoke/dye concentration with optional weight and fading; Newton cooling; open or closed top
- [x] Rigid bodies are moving obstacles (cells inside them are solid and move with them)
- [x] Emitters (heaters and smoke sources, optionally blowing); the world steps the gas at its own rate (30 Hz by default); multithreaded on desktop
- [x] Editor: smoke grid panel, Smoke tool (K) with brush and emitters, Smoke and Temperature views. 2D draws the grid as a texture; 3D draws soft splats with multiplicative blending (order-independent, so no sorting, and it darkens like absorption)
- [x] Presets: Smoke plume and Convection (2D), Smoke plume (3D). Scene format v5 adds `gas`.
- [x] `tests/gas.rs` (2D) and `tests/gas3d.rs` (3D):

  | Test | Expected | Measured |
  | --- | --- | --- |
  | Gaussian hot spot spreading (κ = 0.01 m²/s), 2D, 1 s | peak ΔT₀ σ₀²/(σ₀² + 2κt) = 6.667 K | 6.696 K |
  | Same, 3D, 0.5 s | peak ΔT₀ (σ₀²/σ²)^{3/2} = 7.155 K | 7.191 K |
  | Heat in a sealed, insulated box | conserved | to 1e-6 (2D), 8e-6 (3D) |
  | Warm cylinder (2D) starts rising | ½ g β ΔT (added mass = its own) | 0.488 × g β ΔT |
  | Warm sphere (3D) starts rising | ⅔ g β ΔT (added mass = half its own) | 0.658 × g β ΔT |
  | Uniformly warm sealed box | stays still (hydrostatic balance) | < 1e-4 m/s |
  | Plume around an obstacle | rises, never enters the obstacle, divergence-free | ✓ |

- [ ] Gas pushing bodies (pressure forces on obstacles), and smoke interacting with SPH water and soft bodies
- [ ] Faster 3D: cheaper face-velocity sampling in advection, multigrid pressure solve, GPU compute

## M10: Validation page

- [x] `crates/validation`: eleven scenarios run the real engines and compare with closed-form answers, writing physicsengine.dev/validation (inline SVG charts, no JavaScript) on every deploy. It exits non-zero if any check fails, so a regression blocks the deploy.
- [x] Experimental data left out: everything on the page can be re-derived from the formula beside it.

| Scenario | Engine | Theory | Result |
|---|---|---|---|
| Pendulum period, 5°–80° | 2D joints | 4√(L/g) K(sin θ₀/2) | within 0.02% |
| Pendulum energy, 60°, 30 s | 2D joints | constant | ±1.24%, drift +0.09% |
| Ball rolling / box sliding, 20° | 3D rigid | (5/7) g sin θ; g(sin θ − μ cos θ) | +0.50%; +0.56% |
| Rebound height, e = 0.2–0.9 | 3D rigid | e² | within 0.018 h |
| Spring period, 0.25–4 kg | 2D joints | 2π√(m/k) | within 0.4% |
| XPBD chain stretch, 50–2000 N/m | 2D soft | m g n(n+1)/(2k) | +0.09% (10 substeps), +0.01% (40) |
| Hydrostatic gradient | 2D SPH | ρ₀ g | +2.2% |
| Floating cubes, 250–750 kg/m³ | 3D SPH + rigid | ρ/ρ_water | within 5.5% |
| Gaussian heat peak, t = 0.5–5 s | 2D gas | σ₀²/(σ₀² + 2κt) | within 0.5% |
| Warm patch initial acceleration | 2D / 3D gas | g′/2; 2g′/3 | −2.4%; −1.3% |
| Orbital period, 0.39–30 AU | N-body orbits | 2π√(a³/GM) | +0.02% |
| Solar System, 20 years | N-body orbits | energy, angular momentum constant | 1.7 × 10⁻⁸; 2 × 10⁻¹⁴ |

Restitution is applied to the velocity at the start of the contact step, which includes that step's gravity, so balls with e near 1 bounce about 2% above e².

## M11: Orbits ✅

**Goal:** a planetary-system tab: a star, planets, and their moons under real gravity, with distances you can change.

- [x] `engine3d::orbits`: full N-body Newtonian gravity in SI units and `f64` (30 AU is 4.5 × 10¹² m, held to a fraction of a millimeter). Kick-drift-kick leapfrog: symplectic, so energy errors stay bounded. The step is 1/256 of the circular period of the tightest pair, so moons and close passes get small steps automatically. Touching bodies merge, conserving mass and momentum.
- [x] Editing: add planets (circular orbits) and moons (inside the planet's Hill sphere); set a planet's distance from the star or a moon's from its planet (re-placed on a circular orbit in its current plane; a planet's moons come along, and moving a moon leaves its planet's orbit alone); change mass, radius, color; remove (a planet takes its moons).
- [x] **Solar System example:** the eight planets at their J2000 positions from JPL's mean orbital elements, plus the Moon, the Galilean moons, Titan, Titania, Oberon, and retrograde Triton. Mars's moons are left out: Phobos's 7.7-hour orbit would need steps 14× shorter than Io's.
- [x] **Orbits tab** in the editor: 3D view with a top-down option, following any body (double-click), trails drawn relative to each body's parent (so they trace the orbit's shape), decluttered name labels, a scale bar, a live readout (distance, speed, period, eccentricity, inclination), and speeds from 1 hour/s to 10 years/s. When the step budget can't keep up, the status bar says which pair is limiting it. Planets are lit by the star and drawn at true size or a few pixels, whichever is bigger.

| Check (tests/orbits.rs) | Result |
|---|---|
| Kepler's third law, 0.4–30 AU | +0.020% at every distance (leapfrog at 256 steps/orbit: (2π/256)²/3) |
| Earth's year, from the J2000 state | 365.25 d (sidereal 365.256), e = 0.0167 |
| Moon's month | 27.3 d; still bound after a year with everything pulling on it |
| Solar System energy / angular momentum, 20 years | < 10⁻⁶ / < 10⁻⁹ |
| Moving Jupiter to 2 AU | circular (e < 10⁻⁶), Io's orbit unchanged to 10⁻⁹ |
| Collision | mass and momentum conserved to 10⁻¹² |

Not yet: undo in the Orbits tab, saving systems as files (they're kept between sessions), Mars's moons in the example, and rings.

## M12: Liquids (in progress)

**Goal:** real liquids with heat, and a Fluids tab: a 3D box (vacuum or 1 atm air) to place water, salt water, oil, mercury, and steam in, drawn as smooth surfaces. Delivered in steps:

1. [x] CPU speed-up (#16)
2. [x] **Several liquids** in the 2D and 3D tabs (#17)
3. [x] **Heat in the liquids** (this step)
4. [ ] Smooth-surface rendering (a toggle in 2D/3D, off by default)
5. [ ] The Fluids tab: box, placing tools, solids, heat tools, readouts
6. [ ] Air, boiling, and condensing
7. [ ] Validation cases

**Several liquids** (`materials` crate for the properties; both engines):
- [x] Water, salt water (water with a salinity), canola oil, and mercury, with real densities (920–13,546 kg/m³) and viscosities (CRC Handbook; TEOS-10 for seawater).
- [x] Multiphase SPH after Solenthaler & Pajarola (2008): each particle sums *number* density and the pressure force is −Σ (pᵢ/δᵢ² + pⱼ/δⱼ²) ∇W, so a water particle next to mercury doesn't count mercury's mass as its own density (13.5× contrast). With one liquid this is exactly the old formulation.
- [x] Each liquid's real viscosity (Morris et al. 1997 Laplacian; checked against ∇²(y²) = 2 on the lattice), switched on only where it's resolvable: at least 1% of the artificial viscosity every particle gets.
- [x] Salt diffuses between water particles at its real rate (1.5 × 10⁻⁹ m²/s, so a brine layer stays sharp for hours) and changes density as it goes; total salt is conserved.
- [x] Surface and interfacial tension (Akinci et al. 2013 cohesion, normalized by mean-field theory so each surface's and interface's tension is the real one), used only for particles small enough that surface tension can beat a particle's weight (Bond number ≤ 1: water under 2.7 mm). At coarser spacing the pairwise forces glue liquids in place (a 2 cm oil blob at 4 mm spacing, Bond number 12, never rose), so they're left out.
- [x] The default artificial viscosity went from 0.05 to 0.01: at 0.05 a rising blob of oil stayed put.
- [x] Editor: a liquid picker in the 2D Fluid tool and the 3D Water tool; particles drawn in each liquid's color (oil amber, mercury silver, salt water teal). New "Liquids" example in 2D and 3D: four liquids poured in the wrong order sort themselves.
- [x] The GPU solvers still handle fresh water only; a scene with any other liquid steps on the CPU.

| Check (tests/liquids.rs, liquids3d.rs) | Result |
|---|---|
| Mercury slab on water (2D) | sinks to the bottom in 4 s |
| Inverted oil / water / mercury (3D) | sorts to mercury < water < oil in 5 s |
| Layered hydrostatics (2D) | dp/dy within 15% of ρg in water, 20% in mercury (WCSPH pressure noise) |
| Momentum, oil blob hitting mercury blob | conserved to 10⁻⁴ |
| Salt diffusion | total salt conserved to 10⁻³; spreads across the interface |
| Oil–water emulsion (checkerboard) | separates: oil's mean height ends > 0.25 m above water's in 6 s |

- [x] **Sub-resolution settling:** SPH particles settle into a lattice that resists the tiny buoyancy force on a single particle, so liquids stirred together particle by particle (oil caught under water after a splash) stayed mixed forever. Now a heavier particle right above a lighter one of a different liquid trades places with it at the rate a droplet that size would sink: the lesser of its Stokes velocity Δρ g d²/18μ and its inertial terminal velocity √(g d Δρ/ρ). Trading identities moves nothing, so pressure isn't disturbed and mass is conserved. The Liquids example (four liquids in the wrong order) sorts to within 0.02 m of the ideal layer heights.
- [x] All liquids share the heaviest one's Tait stiffness (and the CFL step uses the faster sound speed that gives the lighter ones): with each liquid's own, water under a slab of mercury compressed 25–30% and the rebound threw liquid out of the tank.

**Heat in the liquids** (both engines):
- [x] Every particle carries a temperature (stored in f64: in one substep conduction changes it by far less than an f32 can resolve at 300 K, and those changes were being rounded away).
- [x] Conduction after Cleary & Monaghan (1999), with each liquid's real conductivity and specific heat (water 0.598 W/m·K and 4182 J/kg·K; mercury 8.3 and 139.5; oil 0.17 and 1970) and the harmonic mean across interfaces. Energy is conserved pair by pair.
- [x] Thermal expansion: water follows Tanaka et al. (2001) (densest at 4 °C, 958 kg/m³ at 100 °C); oil and mercury expand linearly. Particle masses stay fixed; a warm particle's rest volume grows.
- [x] Viscosity with temperature: Vogel's equation for water (1.00 → 0.28 mPa·s from 20 to 100 °C), Andrade fits for oil (57 → 8.5) and mercury.
- [x] Sub-resolution settling now compares actual densities, so warm water below cold rises too, standing in for convection on scales smaller than a particle.
- [x] Walls insulated by default, or held at a temperature (`wall_temperature`).
- [x] The liquid's free surface trades heat with the smoke/heat grid by natural convection (h = 10 W/m²·K).
- [x] Editor: a temperature for the liquid you place (up to 99 °C for water: boiling comes in step 6), and color by temperature in 2D and 3D.

| Check (tests/heat.rs, heat3d.rs) | Result |
|---|---|
| Thermal energy, hot and cold halves (2D, 3D) | conserved to 10⁻⁶ |
| Sine wave of temperature in mercury | decays at the heat equation's rate e^(−αk²t), within 15% |
| Warm water at the bottom (2D blob, 3D layer) | rises |
| Hot walls | heat the water, never past the wall temperature |
| 80 °C water under the air grid | air gains heat; total energy conserved to 5% of what moved |

Known limit: buoyant blobs that are resolved (several particles across) rise more slowly than real ones; a perfectly flat oil layer under water waits for a disturbance before it overturns, as it would in reality. An implicit incompressible solver (DFSPH) was tried and didn't help; it's parked.

## Later

