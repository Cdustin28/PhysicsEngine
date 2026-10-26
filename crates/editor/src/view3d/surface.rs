//! Liquids drawn as smooth, see-through surfaces instead of balls:
//! screen-space fluid rendering (van der Laan, Green & Sainz 2009).
//!
//! Each frame, inside the paint callback's `prepare`:
//!
//! 1. **Scene:** the opaque bodies are drawn into an offscreen color and
//!    depth texture (so the liquid can refract what's behind it).
//! 2. **Depth:** every particle is drawn as a camera-facing sphere
//!    imposter; the nearest surface's eye-space depth goes into a float
//!    texture. Particles behind a body are dropped.
//! 3. **Thickness:** the same imposters add up how much liquid each pixel
//!    looks through, weighted by each liquid's color, and how much of it is
//!    mercury (which reflects like a metal).
//! 4. **Smoothing:** a few passes of a depth-aware (bilateral) blur turn
//!    the bumpy union of spheres into a continuous surface.
//!
//! Then, in `paint`, one full-screen pass shades every pixel that has
//! liquid: normals from the smoothed depth, Fresnel reflection of the sky,
//! refraction of the scene behind, and Beer–Lambert absorption through the
//! thickness (so a thin film is clear and a deep pool takes on the
//! liquid's color).
//!
//! Needs float render targets; [`SurfaceRenderer::supported`] checks.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use glam::{Mat4, Vec3};
use std::borrow::Cow;

/// Eye-space depth of the nearest liquid surface.
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Color-weighted thickness (rgb) and thickness (a).
const THICKNESS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Thickness of metal (mercury) along the view (r) and that thickness
/// times its eye depth (g), for the metal's mean depth.
const METAL: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Offscreen depth for the scene and the particles' own depth test.
const ZBUFFER: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Smoothing passes (each a horizontal and a vertical blur).
const SMOOTHING_PASSES: usize = 4;

/// One particle for the surface: position, radius, color, and whether it's
/// a metal (mercury).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SurfaceParticle {
    pub position: [f32; 3],
    pub radius: f32,
    /// sRGB; alpha 255 = metal.
    pub color: [u8; 4],
}

/// What the surface passes need about the camera and the scene.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceView {
    pub view: Mat4,
    pub proj: Mat4,
    /// Background color behind everything (the 3D view's sky), sRGB 0–1.
    pub sky: [f32; 3],
    /// Direction toward the light, world space.
    pub light: Vec3,
    /// Particle radius, m (sets how far the smoothing reaches).
    pub radius: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    /// World-from-view rotation (inverse view), for world-space normals.
    inv_view: [[f32; 4]; 4],
    /// Texture size in pixels (xy), 1/size (zw).
    size: [f32; 4],
    /// Top-left of the viewport in framebuffer pixels (xy); blur direction
    /// (zw).
    origin: [f32; 4],
    sky: [f32; 4],
    light: [f32; 4],
}

const COMMON: &str = r"
struct U {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    size: vec4<f32>,
    origin: vec4<f32>,
    sky: vec4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
";

const SPLAT_SHADER: &str = r"
@group(1) @binding(0) var scene_depth: texture_depth_2d;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) center_view: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex];
    let c = (u.view * vec4<f32>(center, 1.0)).xyz;
    var out: VOut;
    out.clip = u.proj * vec4<f32>(c + vec3<f32>(corner * radius, 0.0), 1.0);
    out.corner = corner;
    out.center_view = c;
    out.radius = radius;
    out.color = color;
    return out;
}

struct Surface {
    // Point on the sphere facing the camera, view space.
    p: vec3<f32>,
    // Chord length through the sphere at this pixel.
    chord: f32,
    // Window-space depth of `p`.
    depth: f32,
};

fn surface(in: VOut) -> Surface {
    let r2 = dot(in.corner, in.corner);
    if (r2 > 1.0) {
        discard;
    }
    let h = sqrt(1.0 - r2);
    // View space looks down -z; the camera-facing point is at +h radius.
    let p = in.center_view + vec3<f32>(in.corner * in.radius, h * in.radius);
    let clip = u.proj * vec4<f32>(p, 1.0);
    let depth = clip.z / clip.w;
    // Hidden behind a body?
    let texel = vec2<i32>(in.clip.xy);
    if (depth >= textureLoad(scene_depth, texel, 0)) {
        discard;
    }
    return Surface(p, 2.0 * h * in.radius, depth);
}

struct DepthOut {
    @location(0) eye_depth: f32,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_depth(in: VOut) -> DepthOut {
    let s = surface(in);
    return DepthOut(-s.p.z, s.depth);
}

struct ThicknessOut {
    // rgb: color × thickness (the composite divides by the total);
    // a: thickness.
    @location(0) color: vec4<f32>,
    // Thickness of metal, and thickness × eye depth.
    @location(1) metal: vec4<f32>,
};

@fragment
fn fs_thickness(in: VOut) -> ThicknessOut {
    let s = surface(in);
    // Mercury goes only into the metal target, so the color target holds
    // just the see-through liquids.
    let clear = (1.0 - in.color.a) * s.chord;
    let metal = in.color.a * s.chord;
    return ThicknessOut(
        vec4<f32>(in.color.rgb * clear, clear),
        vec4<f32>(metal, metal * -in.center_view.z, 0.0, 0.0),
    );
}
";

const BLUR_SHADER: &str = r"
@group(1) @binding(0) var source: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

// Depth-aware Gaussian along u.origin.zw: neighbors whose depth differs a
// lot (another surface behind) don't blur across the silhouette.
@fragment
fn fs_blur(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    let p = vec2<i32>(pos.xy);
    let d0 = textureLoad(source, p, 0).r;
    if (d0 <= 0.0) {
        return 0.0;
    }
    // Blur radius: about two particle radii in world space, in pixels.
    let world = 2.5 * u.light.w;
    let pixels = clamp(world * u.proj[1][1] * u.size.y * 0.5 / d0, 2.0, 24.0);
    let sigma = pixels * 0.5;
    let dir = vec2<i32>(u.origin.zw);
    var sum = 0.0;
    var weight = 0.0;
    let n = i32(pixels);
    for (var k = -n; k <= n; k = k + 1) {
        let q = clamp(p + dir * k, vec2<i32>(0), vec2<i32>(u.size.xy) - vec2<i32>(1));
        let d = textureLoad(source, q, 0).r;
        if (d <= 0.0) {
            continue;
        }
        let fk = f32(k);
        let range = (d - d0) / world;
        let w = exp(-fk * fk / (2.0 * sigma * sigma)) * exp(-range * range);
        sum += d * w;
        weight += w;
    }
    return sum / max(weight, 1e-6);
}
";

const COMPOSITE_SHADER: &str = r"
@group(1) @binding(0) var depth_tex: texture_2d<f32>;
@group(1) @binding(1) var thick_tex: texture_2d<f32>;
@group(1) @binding(2) var scene_tex: texture_2d<f32>;
@group(1) @binding(3) var scene_sampler: sampler;
@group(1) @binding(4) var metal_tex: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

fn view_pos(p: vec2<i32>, depth: f32) -> vec3<f32> {
    // Pixel → NDC → view direction, scaled to eye depth.
    let uv = (vec2<f32>(p) + 0.5) * u.size.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
    let v = u.inv_proj * ndc;
    let dir = v.xyz / v.w;
    return dir * (depth / -dir.z);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy - u.origin.xy);
    let d = textureLoad(depth_tex, p, 0).r;
    if (d <= 0.0 || textureLoad(thick_tex, p, 0).a + textureLoad(metal_tex, p, 0).r <= 0.0) {
        discard;
    }
    // Thickness, averaged over about a particle's width so the individual
    // spheres' chords don't show as blotches.
    let reach = u.light.w * u.proj[1][1] * u.size.y * 0.5 / d;
    var t = vec4<f32>(0.0);
    var m = vec2<f32>(0.0);
    for (var ring = 0; ring <= 2; ring = ring + 1) {
        let taps = select(8, 1, ring == 0);
        for (var k = 0; k < taps; k = k + 1) {
            let a = 6.2831853 * (f32(k) + 0.5 * f32(ring)) / f32(taps);
            let o = vec2<f32>(cos(a), sin(a)) * reach * 0.6 * f32(ring);
            let st = (vec2<f32>(p) + 0.5 + o) * u.size.zw;
            t += textureSampleLevel(thick_tex, scene_sampler, st, 0.0);
            m += textureSampleLevel(metal_tex, scene_sampler, st, 0.0).rg;
        }
    }
    t = t / 17.0;
    m = m / 17.0;
    if (t.a + m.x <= 0.0) {
        discard;
    }
    // Normal from the smaller of the forward/backward depth differences, so
    // silhouettes don't smear.
    let c = view_pos(p, d);
    // Neighbors with no liquid (depth 0) or on another surface behind or
    // in front are skipped; with neither side usable, face the camera.
    let same_surface = 4.0 * u.light.w;
    let dr = textureLoad(depth_tex, p + vec2<i32>(1, 0), 0).r;
    let dl = textureLoad(depth_tex, p - vec2<i32>(1, 0), 0).r;
    let du = textureLoad(depth_tex, p + vec2<i32>(0, 1), 0).r;
    let dd = textureLoad(depth_tex, p - vec2<i32>(0, 1), 0).r;
    let ok_r = dr > 0.0 && abs(dr - d) < same_surface;
    let ok_l = dl > 0.0 && abs(dl - d) < same_surface;
    let ok_u = du > 0.0 && abs(du - d) < same_surface;
    let ok_d = dd > 0.0 && abs(dd - d) < same_surface;
    let dxp = view_pos(p + vec2<i32>(1, 0), dr) - c;
    let dxm = c - view_pos(p - vec2<i32>(1, 0), dl);
    let dyp = view_pos(p + vec2<i32>(0, 1), du) - c;
    let dym = c - view_pos(p - vec2<i32>(0, 1), dd);
    let ddx = select(select(dxm, dxp, abs(dxp.z) < abs(dxm.z)), select(dxm, dxp, ok_r), !(ok_r && ok_l));
    let ddy = select(select(dym, dyp, abs(dyp.z) < abs(dym.z)), select(dym, dyp, ok_u), !(ok_u && ok_d));
    var n = normalize(cross(ddy, ddx));
    if (!(ok_r || ok_l) || !(ok_u || ok_d)) {
        n = normalize(-c);
    }
    let v = normalize(-c);
    if (dot(n, v) < 0.0) {
        n = -n;
    }

    // The spheres overlap (each is about 3× the volume its particle stands
    // for), so the summed chords overstate the depth by that much.
    let column = t.a * 0.33;
    let albedo = srgb_to_linear(t.rgb / max(t.a, 1e-6));
    // How much of the view the mercury covers: a few millimetres of it are
    // already opaque.
    let metal = 1.0 - exp(-m.x * 0.33 * 60.0);
    // Clear liquid in front of the mercury (layers are what matter: oil
    // floating on mercury tints it; water under it doesn't).
    let metal_depth = m.y / max(m.x, 1e-6) - 0.5 * m.x * 0.33;
    let clear_front = clamp(metal_depth - d, 0.0, column);

    let world_n = normalize((u.inv_view * vec4<f32>(n, 0.0)).xyz);
    let facing = 1.0 - max(dot(n, v), 0.0);
    let fresnel = 0.04 + 0.96 * pow(facing, 5.0);

    // What's behind, bent by the surface (more through thicker liquid).
    let uv = (vec2<f32>(p) + 0.5) * u.size.zw;
    let offset = n.xy * vec2<f32>(1.0, -1.0) * 0.03 * clamp(column * 4.0, 0.0, 1.0);
    let behind = srgb_to_linear(textureSample(scene_tex, scene_sampler, clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0))).rgb);
    // Beer–Lambert: each liquid absorbs what its color lacks (several
    // times real water's absorption, so a tank of it reads as blue rather
    // than invisible).
    let absorb = (vec3<f32>(1.0) - albedo) * 4.0 + vec3<f32>(0.1);
    let transmit = exp(-absorb * column);
    // Light scattered back out of the liquid's body, lit like a matte
    // surface: what gives a deep pool its own color.
    let l = normalize(u.light.xyz);
    let seen = mix(column, clear_front, metal);
    let body = (0.35 + 0.65 * max(dot(world_n, l), 0.0)) * (1.0 - exp(-seen * 6.0));

    // Reflection of a simple sky (horizon color to a lighter zenith).
    let r = reflect(-normalize((u.inv_view * vec4<f32>(v, 0.0)).xyz), world_n);
    let sky = srgb_to_linear(u.sky.rgb);
    // A simple studio: grey ground below the horizon, the sky above
    // (paler at the horizon), and one big soft light overhead, so ripples
    // catch highlights the way a real liquid surface does.
    let ground = vec3<f32>(0.18);
    let above = mix(sky * 1.05, sky * 0.8, clamp(r.y, 0.0, 1.0));
    let horizon = smoothstep(-0.05, 0.05, r.y);
    let softbox = smoothstep(0.93, 0.985, dot(r, normalize(vec3<f32>(0.3, 1.0, 0.25))));
    let env = mix(ground, above, horizon) + vec3<f32>(2.5 * softbox);
    let h = normalize(l + normalize((u.inv_view * vec4<f32>(v, 0.0)).xyz));
    let spec = pow(max(dot(world_n, h), 0.0), 80.0);
    let reflection = env + vec3<f32>(spec * 1.5);
    // Mercury reflects like polished steel (about 75 %), hiding what's
    // behind it; any clear liquid in front tints it.
    let silver = vec3<f32>(0.62, 0.63, 0.66);
    let mercury = silver * (env * 0.8 + vec3<f32>(0.15)) + vec3<f32>(spec * 2.5);
    let through = mix(behind * transmit, mercury * exp(-absorb * clear_front), metal);
    let refracted = through + albedo * body * 0.35;

    let surface_color = mix(refracted, reflection, fresnel);
    return vec4<f32>(pow(surface_color, vec3<f32>(1.0 / 2.2)), 1.0);
}
";

struct Targets {
    size: (u32, u32),
    scene_color_view: wgpu::TextureView,
    scene_depth_view: wgpu::TextureView,
    fluid_z_view: wgpu::TextureView,
    depth_views: [wgpu::TextureView; 2],
    thick_view: wgpu::TextureView,
    metal_view: wgpu::TextureView,
    splat_group: wgpu::BindGroup,
    blur_groups: [wgpu::BindGroup; 2],
    composite_group: wgpu::BindGroup,
}

pub struct SurfaceRenderer {
    color_format: wgpu::TextureFormat,
    uniforms: wgpu::Buffer,
    blur_uniforms: [wgpu::Buffer; 2],
    uniform_group: wgpu::BindGroup,
    blur_uniform_groups: [wgpu::BindGroup; 2],
    splat_layout: wgpu::BindGroupLayout,
    tex1_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    depth_pipeline: wgpu::RenderPipeline,
    thickness_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    particles: wgpu::Buffer,
    capacity: usize,
    count: u32,
    targets: Option<Targets>,
    /// The last frame drew a surface (so `paint` composites it).
    pub active: bool,
}

impl SurfaceRenderer {
    /// Whether this adapter can render to the float textures this needs.
    pub fn supported(adapter: &wgpu::Adapter) -> bool {
        let renderable = |f: wgpu::TextureFormat| {
            adapter
                .get_texture_format_features(f)
                .allowed_usages
                .contains(
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                )
        };
        let blendable = |f: wgpu::TextureFormat| {
            adapter
                .get_texture_format_features(f)
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
        };
        renderable(DEPTH)
            && renderable(THICKNESS)
            && renderable(METAL)
            && blendable(THICKNESS)
            && blendable(METAL)
    }

    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface uniforms"),
            entries: &[uniform_entry],
        });
        let tex = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let unfilterable = wgpu::TextureSampleType::Float { filterable: false };
        let splat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface scene depth"),
            entries: &[tex(0, wgpu::TextureSampleType::Depth)],
        });
        let tex1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface blur source"),
            entries: &[tex(0, unfilterable)],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface composite"),
            entries: &[
                tex(0, unfilterable),
                tex(1, wgpu::TextureSampleType::Float { filterable: true }),
                tex(2, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex(4, wgpu::TextureSampleType::Float { filterable: true }),
            ],
        });
        let uniforms_buffer = || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("surface uniforms"),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniforms = uniforms_buffer();
        let blur_uniforms = [uniforms_buffer(), uniforms_buffer()];
        let group = |b: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface uniforms"),
                layout: &uniform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: b.as_entire_binding(),
                }],
            })
        };
        let uniform_group = group(&uniforms);
        let blur_uniform_groups = [group(&blur_uniforms[0]), group(&blur_uniforms[1])];

        let module = |label: &str, body: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(format!("{COMMON}{body}"))),
            })
        };
        let splat_module = module("surface splats", SPLAT_SHADER);
        let blur_module = module("surface blur", BLUR_SHADER);
        let composite_module = module("surface composite", COMPOSITE_SHADER);

        let splat_layout_full = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surface splats"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&splat_layout)],
            immediate_size: 0,
        });
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfaceParticle>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Unorm8x4],
        };
        let splat_pipeline = |entry: &str,
                              targets: &[Option<wgpu::ColorTargetState>],
                              depth: Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&splat_layout_full),
                vertex: wgpu::VertexState {
                    module: &splat_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(instance_layout.clone())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &splat_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let depth_pipeline = splat_pipeline(
            "fs_depth",
            &[Some(wgpu::ColorTargetState {
                format: DEPTH,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            Some(wgpu::DepthStencilState {
                format: ZBUFFER,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let additive = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: add,
                    alpha: add,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let thickness_pipeline = splat_pipeline(
            "fs_thickness",
            &[additive(THICKNESS), additive(METAL)],
            None,
        );
        let full_pipeline = |module: &wgpu::ShaderModule,
                             layouts: &[Option<&wgpu::BindGroupLayout>],
                             entry: &str,
                             target: wgpu::ColorTargetState| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_full"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(target)],
                }),
                primitive: wgpu::PrimitiveState::default(),
                // The composite runs inside egui's pass, which has a depth
                // buffer (see main.rs): don't test or write it.
                depth_stencil: (entry == "fs_main").then_some(wgpu::DepthStencilState {
                    format: super::render::DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let blur_pipeline = full_pipeline(
            &blur_module,
            &[Some(&uniform_layout), Some(&tex1_layout)],
            "fs_blur",
            wgpu::ColorTargetState {
                format: DEPTH,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            },
        );
        let composite_pipeline = full_pipeline(
            &composite_module,
            &[Some(&uniform_layout), Some(&composite_layout)],
            "fs_main",
            wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            },
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surface scene"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let capacity = 1024;
        Self {
            color_format,
            uniforms,
            blur_uniforms,
            uniform_group,
            blur_uniform_groups,
            splat_layout,
            tex1_layout,
            composite_layout,
            depth_pipeline,
            thickness_pipeline,
            blur_pipeline,
            composite_pipeline,
            sampler,
            particles: particle_buffer(device, capacity),
            capacity,
            count: 0,
            targets: None,
            active: false,
        }
    }

    /// Offscreen color and depth for the scene pass, sized for this frame;
    /// the body renderer draws into these before [`Self::render`].
    pub fn scene_targets(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
    ) -> (&wgpu::TextureView, &wgpu::TextureView) {
        let size = (size.0.max(1), size.1.max(1));
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            self.targets = Some(self.make_targets(device, size));
        }
        let t = self.targets.as_ref().expect("just made");
        (&t.scene_color_view, &t.scene_depth_view)
    }

    fn make_targets(&self, device: &wgpu::Device, size: (u32, u32)) -> Targets {
        let texture = |label: &str, format: wgpu::TextureFormat| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_color = texture("surface scene color", self.color_format);
        let scene_depth = texture("surface scene depth", ZBUFFER);
        let fluid_z = texture("surface particle z", ZBUFFER);
        let depth = [
            texture("surface depth a", DEPTH),
            texture("surface depth b", DEPTH),
        ];
        let thick = texture("surface thickness", THICKNESS);
        let metal = texture("surface metal", METAL);
        let metal_view = view(&metal);
        let scene_color_view = view(&scene_color);
        let scene_depth_view = view(&scene_depth);
        let depth_views = [view(&depth[0]), view(&depth[1])];
        let thick_view = view(&thick);
        let splat_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface scene depth"),
            layout: &self.splat_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&scene_depth_view),
            }],
        });
        let blur_group = |source: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surface blur"),
                layout: &self.tex1_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                }],
            })
        };
        let blur_groups = [blur_group(&depth_views[0]), blur_group(&depth_views[1])];
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface composite"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&thick_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&metal_view),
                },
            ],
        });
        Targets {
            size,
            scene_color_view,
            scene_depth_view,
            fluid_z_view: view(&fluid_z),
            depth_views,
            thick_view,
            metal_view,
            splat_group,
            blur_groups,
            composite_group,
        }
    }

    /// Passes 2–4 (the scene pass has already filled the scene targets).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        particles: &[SurfaceParticle],
        view: &SurfaceView,
        origin: (f32, f32),
    ) {
        self.active = false;
        let Some(t) = self.targets.as_ref() else {
            return;
        };
        if particles.is_empty() {
            return;
        }
        if particles.len() > self.capacity {
            self.capacity = particles.len().next_power_of_two();
            self.particles = particle_buffer(device, self.capacity);
        }
        queue.write_buffer(&self.particles, 0, bytemuck::cast_slice(particles));
        self.count = particles.len() as u32;
        let (w, h) = (t.size.0 as f32, t.size.1 as f32);
        let base = Uniforms {
            view: view.view.to_cols_array_2d(),
            proj: view.proj.to_cols_array_2d(),
            inv_proj: view.proj.inverse().to_cols_array_2d(),
            inv_view: view.view.inverse().to_cols_array_2d(),
            size: [w, h, 1.0 / w, 1.0 / h],
            origin: [origin.0, origin.1, 0.0, 0.0],
            sky: [view.sky[0], view.sky[1], view.sky[2], 1.0],
            light: view.light.normalize().extend(view.radius).to_array(),
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&base));
        for (k, dir) in [[1.0, 0.0], [0.0, 1.0]].into_iter().enumerate() {
            let mut u = base;
            u.origin[2] = dir[0];
            u.origin[3] = dir[1];
            queue.write_buffer(&self.blur_uniforms[k], 0, bytemuck::bytes_of(&u));
        }

        // 2. Nearest surface depth.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface depth"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.depth_views[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.fluid_z_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &self.uniform_group, &[]);
            pass.set_bind_group(1, &t.splat_group, &[]);
            pass.set_vertex_buffer(0, self.particles.slice(..));
            pass.draw(0..6, 0..self.count);
        }
        // 3. Thickness and color.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface thickness"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.thick_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.metal_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                ..Default::default()
            });
            pass.set_pipeline(&self.thickness_pipeline);
            pass.set_bind_group(0, &self.uniform_group, &[]);
            pass.set_bind_group(1, &t.splat_group, &[]);
            pass.set_vertex_buffer(0, self.particles.slice(..));
            pass.draw(0..6, 0..self.count);
        }
        // 4. Smooth: horizontal into b, vertical back into a.
        for _ in 0..SMOOTHING_PASSES {
            for (k, (src, dst)) in [(0usize, 1usize), (1, 0)].into_iter().enumerate() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("surface blur"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &t.depth_views[dst],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&self.blur_pipeline);
                pass.set_bind_group(0, &self.blur_uniform_groups[k], &[]);
                pass.set_bind_group(1, &t.blur_groups[src], &[]);
                pass.draw(0..3, 0..1);
            }
        }
        self.active = true;
    }

    /// The full-screen shading pass, inside egui's render pass.
    pub fn composite(&self, pass: &mut wgpu::RenderPass<'static>) {
        let Some(t) = self.targets.as_ref() else {
            return;
        };
        if !self.active {
            return;
        }
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.uniform_group, &[]);
        pass.set_bind_group(1, &t.composite_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// One [`SurfaceParticle`] per fluid particle, colored by its liquid.
pub fn particles(fluid: &engine3d::Fluid, radius: f32) -> Vec<SurfaceParticle> {
    particles_colored(fluid, radius, true, |i| {
        fluid
            .liquid(i)
            .color(fluid.salinity.get(i).copied().unwrap_or(0.0))
    })
}

/// One [`SurfaceParticle`] per fluid particle in the given colors; mercury
/// is drawn as a metal if `metal` (otherwise every liquid is see-through,
/// so a false-color view reads everywhere).
pub fn particles_colored(
    fluid: &engine3d::Fluid,
    radius: f32,
    metal: bool,
    color: impl Fn(usize) -> [u8; 3],
) -> Vec<SurfaceParticle> {
    use engine3d::fluid::Liquid;
    (0..fluid.len())
        .map(|i| {
            let [r, g, b] = color(i);
            let is_metal = metal && fluid.liquid(i) == Liquid::Mercury;
            SurfaceParticle {
                position: fluid.positions[i].to_array(),
                radius,
                color: [r, g, b, if is_metal { 255 } else { 0 }],
            }
        })
        .collect()
}

fn particle_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface particles"),
        size: (capacity * std::mem::size_of::<SurfaceParticle>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_shaders() {
        assert_eq!(std::mem::size_of::<SurfaceParticle>(), 20);
        assert_eq!(std::mem::size_of::<Uniforms>(), 4 * 64 + 4 * 16);
    }

    #[test]
    fn shaders_compile() {
        for (name, body) in [
            ("splat", SPLAT_SHADER),
            ("blur", BLUR_SHADER),
            ("composite", COMPOSITE_SHADER),
        ] {
            let source = format!("{COMMON}{body}");
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::default(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
