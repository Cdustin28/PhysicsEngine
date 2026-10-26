//! Draws 3D bodies and fluid particles with the GPU: a unit cube and a unit
//! sphere mesh, each drawn once per body (instanced) with its position,
//! rotation, and size. Particles use a low-poly sphere (96 triangles), so
//! 20,000 of them are about 2 million triangles: easy for any GPU, and they
//! sort correctly against bodies through the depth buffer.
//!
//! Like the particle renderer, it plugs into egui as a paint callback. The
//! editor asks eframe for a depth buffer (see `main.rs`), so bodies hide each
//! other correctly; egui clears it at the start of every frame, and egui's
//! own shapes ignore it.

use bytemuck::{Pod, Zeroable};
use eframe::egui::{self, Rect};
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use glam::{Mat4, Vec3};
use std::borrow::Cow;

/// Must match `depth_buffer: 24` in `main.rs`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Soft-body surfaces for one frame: world-space vertices and triangles,
/// plus one instance (color, highlight; identity transform) per body with
/// its index range.
#[derive(Default)]
pub struct SoftMeshes {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bodies: Vec<(std::ops::Range<u32>, Instance)>,
}

/// Per-body data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub position: [f32; 3],
    /// 1 = draw a 1 m checkerboard (static bodies, so the floor reads as one);
    /// 2 = emissive (a star: its own color, unlit).
    pub checker: f32,
    /// Box half extents, or the radius three times for a sphere.
    pub scale: [f32; 3],
    /// 1 = selected.
    pub highlight: f32,
    /// Unit quaternion [x, y, z, w].
    pub rotation: [f32; 4],
    /// sRGB.
    pub color: [u8; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    light: [f32; 4],
    /// Camera right and up directions, for camera-facing smoke splats.
    right: [f32; 4],
    up: [f32; 4],
}

/// One soft, camera-facing smoke/heat splat.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SplatInstance {
    pub position: [f32; 3],
    /// Radius, m.
    pub size: f32,
    /// Tint (rgb) and strength (a): the splat multiplies what's behind it
    /// by mix(white, tint, a × falloff).
    pub color: [u8; 4],
}

/// Smoke and heat drawn as overlapping soft splats with *multiplicative*
/// blending: each one scales the color behind it toward its tint. A product
/// doesn't depend on order, so thousands of splats need no sorting, and it
/// behaves like light absorption (smoke darkens what's behind it).
///
/// On a dark background, darkening is invisible, so the dark theme uses
/// `fs_glow` with *screen* blending instead: 1 − (1 − dst)(1 − src), which
/// lightens toward the tint. It's also order-independent.
const SPLAT_SHADER: &str = r"
struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @location(0) center: vec3<f32>,
    @location(1) size: f32,
    @location(2) color: vec4<f32>,
) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex];
    let world = center + (g.right.xyz * corner.x + g.up.xyz * corner.y) * size;
    var out: VertexOut;
    out.clip = g.view_proj * vec4<f32>(world, 1.0);
    out.corner = corner;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if (r2 > 1.0) {
        discard;
    }
    let falloff = (1.0 - r2) * (1.0 - r2);
    return vec4<f32>(mix(vec3<f32>(1.0), in.color.rgb, in.color.a * falloff), 1.0);
}

@fragment
fn fs_glow(in: VertexOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    if (r2 > 1.0) {
        discard;
    }
    let falloff = (1.0 - r2) * (1.0 - r2);
    return vec4<f32>(in.color.rgb * (in.color.a * falloff), 1.0);
}
";

const SHADER: &str = r"
struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) flags: vec2<f32>,
    @location(4) local: vec3<f32>,
    @location(5) scale: vec3<f32>,
};

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) center: vec3<f32>,
    @location(3) checker: f32,
    @location(4) scale: vec3<f32>,
    @location(5) highlight: f32,
    @location(6) rotation: vec4<f32>,
    @location(7) color: vec4<f32>,
) -> VertexOut {
    var out: VertexOut;
    let world = center + rotate(rotation, position * scale);
    out.clip = g.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    // Normals of a stretched shape: divide by the scale, then renormalize.
    out.normal = rotate(rotation, normal / scale);
    out.color = color;
    out.flags = vec2<f32>(checker, highlight);
    out.local = position;
    out.scale = scale;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    if (in.flags.x > 1.5) {
        // Emissive: a star lights itself.
        return vec4<f32>(in.color.rgb, 1.0);
    }
    // light.w = 0: a sun direction; 1: a point light at light.xyz (a star),
    // with no sky light, so night sides are dark.
    let point = g.light.w > 0.5;
    let l = select(normalize(g.light.xyz), normalize(g.light.xyz - in.world), point);
    let v = normalize(g.eye.xyz - in.world);
    // Two-sided lighting, so cloth seen from below isn't black.
    if (dot(n, v) < 0.0) {
        n = -n;
    }
    var base = in.color.rgb;
    if (in.flags.x > 0.5 && in.flags.x < 1.5) {
        let k = floor(in.world.x) + floor(in.world.z);
        if (abs(k - 2.0 * floor(k * 0.5)) > 0.5) {
            base = base * 0.88;
        }
    }
    // Thin dark edges on boxes, so stacked boxes read as separate. The
    // distances to the three face planes, in meters: on a face the smallest
    // is ~0 and the middle one is the distance to the nearest edge. (On a
    // sphere two coordinates are never both near ±1, so it has no edges.)
    let d = (1.0 - abs(in.local)) * in.scale;
    let middle = d.x + d.y + d.z - min(d.x, min(d.y, d.z)) - max(d.x, max(d.y, d.z));
    if (in.flags.x < 0.5 && middle < 0.012) {
        base = base * 0.55;
    }
    let sky = 0.5 + 0.5 * n.y;
    let diffuse = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let specular = pow(max(dot(n, h), 0.0), 40.0) * 0.2;
    var rgb = base * (0.22 + 0.25 * sky + 0.6 * diffuse) + vec3<f32>(specular);
    if (point) {
        // Enough ambient that a night side seen against the star reads as a
        // planet, not a hole.
        rgb = base * (0.2 + 0.85 * diffuse) + vec3<f32>(specular * 0.5);
    }
    if (in.flags.y > 0.5) {
        // Gold rim on the selected body.
        let rim = pow(1.0 - max(dot(n, v), 0.0), 2.0);
        rgb = mix(rgb, vec3<f32>(0.945, 0.745, 0.282), 0.25 + 0.6 * rim);
    }
    return vec4<f32>(rgb, 1.0);
}
";

struct Mesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

pub struct BodyRenderer {
    pipeline: wgpu::RenderPipeline,
    /// The same, for the offscreen scene pass of the smooth-liquid view.
    scene_pipeline: wgpu::RenderPipeline,
    /// Smooth liquid surfaces, if this GPU can render them.
    surface: Option<super::surface::SurfaceRenderer>,
    splat_pipeline: wgpu::RenderPipeline,
    glow_pipeline: wgpu::RenderPipeline,
    /// This frame's splats lighten (screen) instead of darken.
    glow: bool,
    splats: wgpu::Buffer,
    splat_capacity: usize,
    splat_count: u32,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    cube: Mesh,
    sphere: Mesh,
    particle: Mesh,
    soft_vertices: wgpu::Buffer,
    soft_indices: wgpu::Buffer,
    soft_vertex_capacity: usize,
    soft_index_capacity: usize,
    /// Index range and instance index of each soft body this frame.
    soft_draws: Vec<(std::ops::Range<u32>, u32)>,
    instances: wgpu::Buffer,
    capacity: usize,
    /// Instance ranges: cubes, then spheres, then particles.
    cubes: u32,
    spheres_end: u32,
    total: u32,
}

impl BodyRenderer {
    /// Set up the renderer and register it with egui. Returns false if eframe
    /// isn't using wgpu.
    pub fn install(cc: &eframe::CreationContext<'_>) -> bool {
        let Some(rs) = cc.wgpu_render_state.as_ref() else {
            return false;
        };
        let mut renderer = Self::new(&rs.device, rs.target_format);
        if super::surface::SurfaceRenderer::supported(&rs.adapter) {
            renderer.surface = Some(super::surface::SurfaceRenderer::new(
                &rs.device,
                rs.target_format,
            ));
        }
        rs.renderer.write().callback_resources.insert(renderer);
        true
    }

    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bodies3d"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bodies3d globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bodies3d"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let body_pipeline = |depth_format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bodies3d"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Vertex>() as u64,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                        }),
                        Some(wgpu::VertexBufferLayout {
                            array_stride: std::mem::size_of::<Instance>() as u64,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &wgpu::vertex_attr_array![
                                2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32,
                                6 => Float32x4, 7 => Unorm8x4
                            ],
                        }),
                    ],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = body_pipeline(DEPTH_FORMAT);
        let scene_pipeline = body_pipeline(wgpu::TextureFormat::Depth32Float);
        let splat_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splats3d"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SPLAT_SHADER)),
        });
        let splat_pipeline_with = |entry: &str, color: wgpu::BlendComponent| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("splats3d"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &splat_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SplatInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Unorm8x4],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &splat_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState {
                            color,
                            alpha: wgpu::BlendComponent::OVER,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                // Bodies in front hide smoke behind them; smoke doesn't hide
                // anything from the depth test.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // result = destination × source (multiplicative).
        let splat_pipeline = splat_pipeline_with(
            "fs_main",
            wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        );
        // result = source + destination × (1 − source) (screen).
        let glow_pipeline = splat_pipeline_with(
            "fs_glow",
            wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        );
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bodies3d globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bodies3d globals"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let (cv, ci) = cube_mesh();
        let (sv, si) = sphere_mesh(32, 16);
        let (pv, pi) = sphere_mesh(8, 6);
        let capacity = 256;
        Self {
            pipeline,
            scene_pipeline,
            surface: None,
            splat_pipeline,
            glow_pipeline,
            glow: false,
            splats: splat_buffer(device, 1024),
            splat_capacity: 1024,
            splat_count: 0,
            globals,
            bind_group,
            cube: Mesh::new(device, &cv, &ci),
            sphere: Mesh::new(device, &sv, &si),
            particle: Mesh::new(device, &pv, &pi),
            soft_vertices: soft_buffer(
                device,
                1024 * std::mem::size_of::<Vertex>(),
                wgpu::BufferUsages::VERTEX,
            ),
            soft_indices: soft_buffer(device, 4096 * 4, wgpu::BufferUsages::INDEX),
            soft_vertex_capacity: 1024,
            soft_index_capacity: 4096,
            soft_draws: Vec::new(),
            instances: instance_buffer(device, capacity),
            capacity,
            cubes: 0,
            spheres_end: 0,
            total: 0,
        }
    }
}

impl Mesh {
    fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt as _;
        Self {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: indices.len() as u32,
        }
    }
}

fn soft_buffer(device: &wgpu::Device, bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soft mesh"),
        size: bytes as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn splat_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("splats3d"),
        size: (capacity * std::mem::size_of::<SplatInstance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bodies3d instances"),
        size: (capacity * std::mem::size_of::<Instance>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The cube [-1, 1]³: 4 vertices per face so each face has a flat normal.
fn cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            let mut n = [0.0; 3];
            n[axis] = sign;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = vertices.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let mut p = [0.0; 3];
                p[axis] = sign;
                p[u] = a;
                p[v] = b;
                vertices.push(Vertex {
                    position: p,
                    normal: n,
                });
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    (vertices, indices)
}

/// The unit sphere as a latitude/longitude grid.
fn sphere_mesh(segments: u32, rings: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let theta = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let phi = std::f32::consts::TAU * s as f32 / segments as f32;
            let p = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            vertices.push(Vertex {
                position: p,
                normal: p,
            });
        }
    }
    let mut indices = Vec::new();
    let row = segments + 1;
    for r in 0..rings {
        for s in 0..segments {
            let (a, b) = (r * row + s, (r + 1) * row + s);
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (vertices, indices)
}

/// Where the camera is and which way it faces.
pub struct CameraView {
    pub view_proj: Mat4,
    pub eye: Vec3,
    /// Unit right and up directions on screen, in world space.
    pub right: Vec3,
    pub up: Vec3,
}

/// One frame's bodies and particles, handed to egui as a paint callback.
pub struct BodyCallback {
    /// Cubes, then spheres, then particles, then one per soft body.
    instances: Vec<Instance>,
    cubes: u32,
    spheres_end: u32,
    particles_end: u32,
    soft: SoftMeshes,
    splats: Vec<SplatInstance>,
    /// Splats lighten instead of darken (for dark backgrounds).
    glow: bool,
    globals: Globals,
    /// Draw the particles as a smooth liquid surface instead of balls.
    surface: Option<SurfaceFrame>,
}

/// What the smooth-surface passes need for one frame.
pub struct SurfaceFrame {
    pub particles: Vec<super::surface::SurfaceParticle>,
    pub view: super::surface::SurfaceView,
    /// The canvas, in points.
    pub rect: Rect,
}

impl BodyCallback {
    pub fn new(
        cubes: Vec<Instance>,
        spheres: Vec<Instance>,
        particles: Vec<Instance>,
        soft: SoftMeshes,
        splats: Vec<SplatInstance>,
        camera: CameraView,
    ) -> Self {
        let CameraView {
            view_proj,
            eye,
            right,
            up,
        } = camera;
        let n = cubes.len() as u32;
        let spheres_end = n + spheres.len() as u32;
        let particles_end = spheres_end + particles.len() as u32;
        let mut instances = cubes;
        instances.extend(spheres);
        instances.extend(particles);
        instances.extend(soft.bodies.iter().map(|(_, instance)| *instance));
        Self {
            instances,
            cubes: n,
            spheres_end,
            particles_end,
            soft,
            splats,
            glow: false,
            surface: None,
            globals: Globals {
                view_proj: view_proj.to_cols_array_2d(),
                eye: eye.extend(1.0).to_array(),
                // Sun from upper front-left.
                light: [0.45, 0.8, 0.35, 0.0],
                right: right.extend(0.0).to_array(),
                up: up.extend(0.0).to_array(),
            },
        }
    }

    /// Draw liquid as a smooth surface (if the GPU supports it); the
    /// `particles` passed to [`Self::new`] are then skipped.
    pub fn with_surface(mut self, frame: SurfaceFrame) -> Self {
        self.surface = Some(frame);
        self
    }

    /// Light the scene from a point (a star) instead of a distant sun, with
    /// no sky light.
    pub fn with_point_light(mut self, position: Vec3) -> Self {
        self.globals.light = position.extend(1.0).to_array();
        self
    }

    /// Draw the splats by lightening what's behind them (screen blending),
    /// for dark backgrounds, instead of darkening it.
    pub fn with_glowing_splats(mut self, glow: bool) -> Self {
        self.glow = glow;
        self
    }

    pub fn into_shape(self, rect: Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(rect, self))
    }
}

impl CallbackTrait for BodyCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<BodyRenderer>() else {
            return Vec::new();
        };
        if self.instances.len() > r.capacity {
            r.capacity = self.instances.len().next_power_of_two();
            r.instances = instance_buffer(device, r.capacity);
        }
        queue.write_buffer(&r.globals, 0, bytemuck::bytes_of(&self.globals));
        queue.write_buffer(&r.instances, 0, bytemuck::cast_slice(&self.instances));
        r.cubes = self.cubes;
        r.spheres_end = self.spheres_end;
        r.total = self.particles_end;
        // Soft-body meshes.
        let soft = &self.soft;
        if soft.vertices.len() > r.soft_vertex_capacity {
            r.soft_vertex_capacity = soft.vertices.len().next_power_of_two();
            r.soft_vertices = soft_buffer(
                device,
                r.soft_vertex_capacity * std::mem::size_of::<Vertex>(),
                wgpu::BufferUsages::VERTEX,
            );
        }
        if soft.indices.len() > r.soft_index_capacity {
            r.soft_index_capacity = soft.indices.len().next_power_of_two();
            r.soft_indices =
                soft_buffer(device, r.soft_index_capacity * 4, wgpu::BufferUsages::INDEX);
        }
        if !soft.indices.is_empty() {
            queue.write_buffer(&r.soft_vertices, 0, bytemuck::cast_slice(&soft.vertices));
            queue.write_buffer(&r.soft_indices, 0, bytemuck::cast_slice(&soft.indices));
        }
        if self.splats.len() > r.splat_capacity {
            r.splat_capacity = self.splats.len().next_power_of_two();
            r.splats = splat_buffer(device, r.splat_capacity);
        }
        if !self.splats.is_empty() {
            queue.write_buffer(&r.splats, 0, bytemuck::cast_slice(&self.splats));
        }
        r.splat_count = self.splats.len() as u32;
        r.glow = self.glow;
        r.soft_draws = soft
            .bodies
            .iter()
            .enumerate()
            .map(|(k, (range, _))| (range.clone(), self.particles_end + k as u32))
            .collect();

        // Smooth liquid: draw the opaque scene offscreen, then the surface
        // passes (see surface.rs).
        if let Some(mut surface) = r.surface.take() {
            surface.active = false;
            if let Some(frame) = &self.surface {
                let ppp = screen.pixels_per_point;
                let size = (
                    (frame.rect.width() * ppp).round() as u32,
                    (frame.rect.height() * ppp).round() as u32,
                );
                let origin = (frame.rect.min.x * ppp, frame.rect.min.y * ppp);
                {
                    let (color, depth) = surface.scene_targets(device, size);
                    let sky = frame.view.sky;
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("surface scene"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: f64::from(sky[0]),
                                    g: f64::from(sky[1]),
                                    b: f64::from(sky[2]),
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    draw_opaque(r, &mut pass, &r.scene_pipeline, false);
                }
                surface.render(
                    device,
                    queue,
                    encoder,
                    &frame.particles,
                    &frame.view,
                    origin,
                );
            }
            r.surface = Some(surface);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(r) = resources.get::<BodyRenderer>() else {
            return;
        };
        if r.total == 0 && r.soft_draws.is_empty() && r.splat_count == 0 {
            return;
        }
        let surface = r
            .surface
            .as_ref()
            .filter(|s| s.active && self.surface.is_some());
        draw_opaque(r, pass, &r.pipeline, surface.is_none());
        if let Some(surface) = surface {
            surface.composite(pass);
        }
        // Smoke last, over everything opaque.
        if r.splat_count > 0 {
            pass.set_pipeline(if r.glow {
                &r.glow_pipeline
            } else {
                &r.splat_pipeline
            });
            pass.set_bind_group(0, &r.bind_group, &[]);
            pass.set_vertex_buffer(0, r.splats.slice(..));
            pass.draw(0..6, 0..r.splat_count);
        }
    }
}

/// Bodies, soft bodies, and (if `particles`) fluid particles as balls.
fn draw_opaque(
    r: &BodyRenderer,
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    particles: bool,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &r.bind_group, &[]);
    pass.set_vertex_buffer(1, r.instances.slice(..));
    let particle_end = if particles { r.total } else { r.spheres_end };
    for (mesh, range) in [
        (&r.cube, 0..r.cubes),
        (&r.sphere, r.cubes..r.spheres_end),
        (&r.particle, r.spheres_end..particle_end),
    ] {
        if range.is_empty() {
            continue;
        }
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, range);
    }
    if !r.soft_draws.is_empty() {
        pass.set_vertex_buffer(0, r.soft_vertices.slice(..));
        pass.set_index_buffer(r.soft_indices.slice(..), wgpu::IndexFormat::Uint32);
        for (range, instance) in &r.soft_draws {
            pass.draw_indexed(range.clone(), 0, *instance..*instance + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshes_are_closed_and_in_range() {
        for (v, i) in [cube_mesh(), sphere_mesh(32, 16), sphere_mesh(8, 6)] {
            assert_eq!(i.len() % 3, 0);
            assert!(i.iter().all(|&k| (k as usize) < v.len()));
            for vertex in &v {
                let len = Vec3::from(vertex.normal).length();
                assert!((len - 1.0).abs() < 1e-5);
            }
        }
        assert_eq!(cube_mesh().1.len(), 36);
        assert_eq!(sphere_mesh(8, 6).1.len() / 3, 96);
    }

    #[test]
    fn instance_layout_matches_shader() {
        // 3 + 1 + 3 + 1 + 4 floats and 4 bytes of color.
        assert_eq!(std::mem::size_of::<Instance>(), 52);
        assert_eq!(std::mem::size_of::<Globals>(), 128);
        assert_eq!(std::mem::size_of::<SplatInstance>(), 20);
    }
}
