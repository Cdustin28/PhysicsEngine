//! GPU fluid solvers: the engine's weakly compressible SPH as wgpu compute
//! shaders, in 2D (`sph.wgsl`, [`GpuSolver`]) and 3D (`sph3d.wgsl`,
//! [`GpuSolver3d`]). The 3D one is the same design; see [`sph3d`].
//!
//! [`GpuSolver`] implements [`engine::FluidSolver`], so it drops into
//! [`engine::World::step_with`] in place of the CPU solver. Each world step
//! it:
//!
//! 1. uploads the particles and the boundary (bodies) to the GPU,
//! 2. runs every substep there (six compute passes each),
//! 3. reads back positions, velocities, densities, pressures, and the
//!    impulse on each boundary particle, and sums those into per-body forces.
//!
//! The CPU keeps the authoritative copy, so the editor's tools, undo,
//! saving, and the safety-net projection keep working unchanged. Moving
//! a few megabytes per step over PCIe is cheap next to the substeps
//! themselves.
//!
//! Unlike the CPU solver, the GPU can't cheaply find the fastest particle
//! before every substep, so it always takes the worst-case CFL substep count
//! ([`engine::FluidParams::worst_case_substeps`]). That is a little more
//! work (typically 13 substeps instead of 9–12), but never unstable.
//!
//! Reading results back has to block until the GPU finishes, which a
//! browser can't do, so on the web this solver isn't available yet and the
//! editor uses the CPU solver.

pub mod sph3d;

pub use sph3d::GpuSolver3d;

use bytemuck::{Pod, Zeroable};
use engine::fluid::boundary::Boundary;
use engine::{BodyImpulse, Fluid, FluidSolver, Vec2};
use std::borrow::Cow;

const WORKGROUP: u32 = 64;

/// Uniform block; must match `Params` in `sph.wgsl` (80 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    nb: u32,
    table_mask: u32,
    btable_size: u32,
    h: f32,
    h2: f32,
    inv_cell: f32,
    mass: f32,
    rho0: f32,
    stiffness: f32,
    c: f32,
    alpha: f32,
    alpha_wall: f32,
    h_visc: f32,
    eps: f32,
    dt: f32,
    gravity: [f32; 2],
    max_speed: f32,
    sigma: f32,
}

/// Every buffer the shader binds, sized for at least `capacity` fluid and
/// `boundary_capacity` boundary particles.
struct Buffers {
    params: wgpu::Buffer,
    particles: wgpu::Buffer,
    scratch: wgpu::Buffer,
    // Only the shader touches these three (through `bind_group`); they're
    // kept here so ownership of every buffer is in one place.
    _next_velocity: wgpu::Buffer,
    _links: wgpu::Buffer,
    _heads: wgpu::Buffer,
    boundary: wgpu::Buffer,
    boundary_extra: wgpu::Buffer,
    boundary_grid: wgpu::Buffer,
    // Staging buffers for reading results back.
    read_particles: wgpu::Buffer,
    read_scratch: wgpu::Buffer,
    read_boundary_extra: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    capacity: usize,
    table_size: usize,
    boundary_capacity: usize,
    boundary_grid_capacity: usize,
}

struct Pipelines {
    layout: wgpu::BindGroupLayout,
    clear_heads: wgpu::ComputePipeline,
    insert: wgpu::ComputePipeline,
    density: wgpu::ComputePipeline,
    forces: wgpu::ComputePipeline,
    boundary_impulse: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
}

/// SPH on the GPU. Create one per device and reuse it; buffers grow as
/// needed.
pub struct GpuSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    buffers: Option<Buffers>,
    /// Re-sort particles for memory locality every this many steps.
    sort_interval: u32,
    steps: u32,
    adapter_name: String,
}

impl GpuSolver {
    /// Use an existing device (for example the one the editor renders with).
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, adapter_name: impl Into<String>) -> Self {
        let pipelines = Pipelines::new(&device);
        Self {
            device,
            queue,
            pipelines,
            buffers: None,
            sort_interval: 10,
            steps: 0,
            adapter_name: adapter_name.into(),
        }
    }

    /// Create a device of our own (tests, benchmarks). Fails if no GPU (or
    /// software rasterizer such as lavapipe) is available.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn headless() -> Result<Self, String> {
        let (device, queue, name) = headless_device()?;
        Ok(Self::new(device, queue, name))
    }

    /// Name of the GPU (or software renderer) in use.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Make sure the buffers can hold `n` fluid particles, `nb` boundary
    /// particles, and a boundary hash table of `btable` buckets.
    fn ensure_buffers(&mut self, n: usize, nb: usize, btable: usize) {
        let fits = self.buffers.as_ref().is_some_and(|b| {
            b.capacity >= n
                && b.boundary_capacity >= nb.max(1)
                && b.boundary_grid_capacity >= btable + nb.max(1)
        });
        if !fits {
            // Grow with headroom so adding fluid doesn't reallocate every step.
            let capacity = (n + n / 2).max(1024);
            let table_size = (capacity * 2).next_power_of_two();
            let boundary_capacity = (nb + nb / 2).max(256);
            let boundary_grid_capacity =
                (boundary_capacity * 2).next_power_of_two().max(16) + boundary_capacity;
            self.buffers = Some(Buffers::new(
                &self.device,
                &self.pipelines.layout,
                capacity,
                table_size,
                boundary_capacity,
                boundary_grid_capacity,
            ));
        }
    }
}

impl FluidSolver for GpuSolver {
    fn step(
        &mut self,
        fluid: &mut Fluid,
        boundary: &Boundary,
        gravity: Vec2,
        dt: f32,
        body_count: usize,
    ) -> Vec<BodyImpulse> {
        let mut impulses = vec![BodyImpulse::default(); body_count];
        if fluid.is_empty() {
            return impulses;
        }
        fluid.fill_missing_velocities();
        if self.steps.is_multiple_of(self.sort_interval) {
            fluid.sort_spatially();
        }
        self.steps = self.steps.wrapping_add(1);

        let params = fluid.params;
        let n = fluid.len();
        let nb = boundary.len();
        let h = params.support();
        let substeps = params.worst_case_substeps(dt);

        // Boundary hash grid, built here since the boundary is fixed for the
        // whole step: bucket heads, then one link per boundary particle.
        let btable = (nb * 2).next_power_of_two().max(16);
        let mut boundary_grid = vec![[-1_i32, 0, 0, 0]; btable + nb.max(1)];
        for (b, &x) in boundary.positions.iter().enumerate() {
            let c = cell(x, 1.0 / h);
            let bucket = hash(c, btable as u32 - 1) as usize;
            boundary_grid[btable + b] = [c.0, c.1, boundary_grid[bucket][0], 0];
            boundary_grid[bucket][0] = b as i32;
        }

        self.ensure_buffers(n, nb, btable);
        let buffers = self.buffers.as_ref().expect("buffers were just created");

        let h_visc = 0.5 * h;
        let uniform = Params {
            n: n as u32,
            nb: nb as u32,
            table_mask: buffers.table_size as u32 - 1,
            btable_size: btable as u32,
            h,
            h2: h * h,
            inv_cell: 1.0 / h,
            mass: params.particle_mass(),
            rho0: params.rest_density,
            stiffness: params.stiffness(),
            c: params.speed_of_sound,
            alpha: params.viscosity,
            alpha_wall: params.wall_viscosity,
            h_visc,
            eps: 0.01 * h_visc * h_visc,
            dt: dt / substeps as f32,
            gravity: gravity.to_array(),
            max_speed: params.max_speed(),
            sigma: 40.0 / (7.0 * std::f32::consts::PI * h * h),
        };

        // --- Upload.
        let particles: Vec<[f32; 4]> = fluid
            .positions
            .iter()
            .zip(&fluid.velocities)
            .map(|(p, v)| [p.x, p.y, v.x, v.y])
            .collect();
        let boundary_data: Vec<[f32; 4]> = boundary
            .positions
            .iter()
            .zip(&boundary.velocities)
            .map(|(p, v)| [p.x, p.y, v.x, v.y])
            .collect();
        let boundary_extra: Vec<[f32; 4]> = boundary
            .psi
            .iter()
            .map(|&psi| [psi, 0.0, 0.0, 0.0])
            .collect();
        let q = &self.queue;
        q.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&uniform));
        q.write_buffer(&buffers.particles, 0, bytemuck::cast_slice(&particles));
        if nb > 0 {
            q.write_buffer(&buffers.boundary, 0, bytemuck::cast_slice(&boundary_data));
            q.write_buffer(
                &buffers.boundary_extra,
                0,
                bytemuck::cast_slice(&boundary_extra),
            );
        }
        q.write_buffer(
            &buffers.boundary_grid,
            0,
            bytemuck::cast_slice(&boundary_grid),
        );

        // --- Substeps. One compute pass per stage keeps the ordering (and
        // memory visibility) between stages explicit.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sph step"),
            });
        let groups = |count: usize| (count as u32).div_ceil(WORKGROUP);
        let p = &self.pipelines;
        let stages: [(&wgpu::ComputePipeline, u32); 6] = [
            (&p.clear_heads, groups(buffers.table_size)),
            (&p.insert, groups(n)),
            (&p.density, groups(n)),
            (&p.forces, groups(n)),
            (&p.boundary_impulse, groups(nb)),
            (&p.integrate, groups(n)),
        ];
        for _ in 0..substeps {
            for (pipeline, count) in stages {
                if count == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &buffers.bind_group, &[]);
                pass.dispatch_workgroups(count, 1, 1);
            }
        }

        // --- Read back.
        let particle_bytes = (n * 16) as u64;
        let extra_bytes = (nb * 16) as u64;
        encoder.copy_buffer_to_buffer(
            &buffers.particles,
            0,
            &buffers.read_particles,
            0,
            particle_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &buffers.scratch,
            0,
            &buffers.read_scratch,
            0,
            particle_bytes,
        );
        if nb > 0 {
            encoder.copy_buffer_to_buffer(
                &buffers.boundary_extra,
                0,
                &buffers.read_boundary_extra,
                0,
                extra_bytes,
            );
        }
        self.queue.submit([encoder.finish()]);

        let particles: Vec<[f32; 4]> = read(&self.device, &buffers.read_particles, particle_bytes);
        let scratch: Vec<[f32; 4]> = read(&self.device, &buffers.read_scratch, particle_bytes);
        let extra: Vec<[f32; 4]> = if nb > 0 {
            read(&self.device, &buffers.read_boundary_extra, extra_bytes)
        } else {
            Vec::new()
        };

        for (i, p) in particles.iter().enumerate() {
            fluid.positions[i] = Vec2::new(p[0], p[1]);
            fluid.velocities[i] = Vec2::new(p[2], p[3]);
        }
        fluid.record_external_step(
            scratch.iter().map(|s| s[0]).collect(),
            scratch.iter().map(|s| s[1]).collect(),
            substeps,
        );

        // Sum boundary-particle impulses into each body's force and torque.
        for (b, e) in extra.iter().enumerate() {
            let body = boundary.body[b] as usize;
            if let Some(imp) = impulses.get_mut(body) {
                let j = Vec2::new(e[1], e[2]);
                imp.linear += j;
                imp.angular += (boundary.positions[b] - boundary.body_centers[body]).perp_dot(j);
            }
        }
        impulses
    }

    fn name(&self) -> &str {
        "GPU"
    }
}

/// A device of our own, for tests and benchmarks: (device, queue, adapter
/// name).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn headless_device() -> Result<(wgpu::Device, wgpu::Queue, String), String> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpu fluid"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("could not create a GPU device: {e}"))?;
        Ok((device, queue, name))
    })
}

/// Copy `bytes` out of a mappable staging buffer, blocking until the GPU is
/// done.
pub(crate) fn read<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer, bytes: u64) -> Vec<T> {
    let slice = buffer.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("mapping a GPU readback buffer failed");
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("waiting for the GPU failed");
    let data = {
        let view = slice
            .get_mapped_range()
            .expect("reading a mapped GPU buffer failed");
        bytemuck::cast_slice(&view).to_vec()
    };
    buffer.unmap();
    data
}

fn cell(x: Vec2, inv_cell: f32) -> (i32, i32) {
    let f = |v: f32| (v * inv_cell).floor().clamp(-1.0e9, 1.0e9) as i32;
    (f(x.x), f(x.y))
}

/// Must match `hash` in `sph.wgsl`.
fn hash(c: (i32, i32), mask: u32) -> u32 {
    ((c.0 as u32).wrapping_mul(73_856_093) ^ (c.1 as u32).wrapping_mul(19_349_663)) & mask
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sph.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sph.wgsl"))),
        });
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend((1..=8).map(storage));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sph bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sph"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            clear_heads: pipeline("clear_heads"),
            insert: pipeline("insert"),
            density: pipeline("density"),
            forces: pipeline("forces"),
            boundary_impulse: pipeline("boundary_impulse"),
            integrate: pipeline("integrate"),
            layout,
        }
    }
}

impl Buffers {
    fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        capacity: usize,
        table_size: usize,
        boundary_capacity: usize,
        boundary_grid_capacity: usize,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let storage = |label: &str, bytes: usize, extra: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage: U::STORAGE | U::COPY_DST | extra,
                mapped_at_creation: false,
            })
        };
        let staging = |label: &str, bytes: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes as u64,
                usage: U::MAP_READ | U::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: U::UNIFORM | U::COPY_DST,
            mapped_at_creation: false,
        });
        let particles = storage("particles", capacity * 16, U::COPY_SRC);
        let scratch = storage("scratch", capacity * 16, U::COPY_SRC);
        let next_velocity = storage("next velocity", capacity * 8, U::empty());
        let links = storage("links", capacity * 16, U::empty());
        let heads = storage("heads", table_size * 4, U::empty());
        let boundary = storage("boundary", boundary_capacity * 16, U::empty());
        let boundary_extra = storage("boundary extra", boundary_capacity * 16, U::COPY_SRC);
        let boundary_grid = storage("boundary grid", boundary_grid_capacity * 16, U::empty());
        let read_particles = staging("read particles", capacity * 16);
        let read_scratch = staging("read scratch", capacity * 16);
        let read_boundary_extra = staging("read boundary extra", boundary_capacity * 16);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sph"),
            layout,
            entries: &[
                params.as_entire_binding(),
                particles.as_entire_binding(),
                scratch.as_entire_binding(),
                next_velocity.as_entire_binding(),
                links.as_entire_binding(),
                heads.as_entire_binding(),
                boundary.as_entire_binding(),
                boundary_extra.as_entire_binding(),
                boundary_grid.as_entire_binding(),
            ]
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect::<Vec<_>>(),
        });
        Self {
            params,
            particles,
            scratch,
            _next_velocity: next_velocity,
            _links: links,
            _heads: heads,
            boundary,
            boundary_extra,
            boundary_grid,
            read_particles,
            read_scratch,
            read_boundary_extra,
            bind_group,
            capacity,
            table_size,
            boundary_capacity,
            boundary_grid_capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 80);
    }

    #[test]
    fn hash_matches_engine_grid_convention() {
        // Negative cells wrap the same way the WGSL bitcast does.
        assert_eq!(
            hash((-1, 0), u32::MAX),
            73_856_093_u32.wrapping_mul(u32::MAX)
        );
        assert_eq!(hash((0, 0), 1023), 0);
    }
}
