//! Draws fluid particles with the GPU: one instanced quad per particle,
//! cut to a circle in the fragment shader.
//!
//! egui tessellates each `circle_filled` into a few dozen triangles on the
//! CPU, which is fine for a few thousand particles and hopeless for 100k.
//! Here each particle is just a position and a color in an instance buffer,
//! drawn in a single call.
//!
//! It plugs into egui as a paint callback: `prepare` uploads this frame's
//! instances and view, then `paint` issues the draw inside egui's render
//! pass, with the viewport set to the canvas rectangle. Works with every wgpu
//! backend, including WebGL2 in the browser.

use bytemuck::{Pod, Zeroable};
use eframe::egui::{self, Color32, Rect};
use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use engine::Vec2;
use std::borrow::Cow;

/// Per-particle data: world position and sRGB color.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub position: [f32; 2],
    pub color: [u8; 4],
}

/// Must match `View` in the shader below (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct View {
    /// World point at the center of the canvas, m.
    center: [f32; 2],
    /// Meters → clip-space units, per axis.
    scale: [f32; 2],
    /// Particle radius in clip-space units, per axis.
    radius: [f32; 2],
    _padding: [f32; 2],
}

const SHADER: &str = r"
struct View {
    center: vec2<f32>,
    scale: vec2<f32>,
    radius: vec2<f32>,
    _padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> view: View;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @location(0) center: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> VertexOut {
    // Two triangles covering the square [-1, 1]^2.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vertex];
    var out: VertexOut;
    out.position = vec4<f32>((center - view.center) * view.scale + corner * view.radius, 0.0, 1.0);
    out.corner = corner;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if (dot(in.corner, in.corner) > 1.0) {
        discard;
    }
    return in.color;
}
";

/// GPU resources, stored in egui's callback resources so paint callbacks
/// can find them.
pub struct ParticleRenderer {
    pipeline: wgpu::RenderPipeline,
    view_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    capacity: usize,
    /// How many instances the last `prepare` uploaded.
    count: u32,
}

impl ParticleRenderer {
    /// Set up the renderer and register it with egui. Returns false if eframe
    /// isn't using wgpu (then the editor falls back to egui circles).
    pub fn install(cc: &eframe::CreationContext<'_>) -> bool {
        let Some(rs) = cc.wgpu_render_state.as_ref() else {
            return false;
        };
        let renderer = Self::new(&rs.device, rs.target_format);
        rs.renderer.write().callback_resources.insert(renderer);
        true
    }

    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles view"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Instance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Unorm8x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // The render pass has a depth buffer (for the 3D view); particles
            // ignore it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::view3d::render::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            // eframe renders without multisampling by default.
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles view"),
            size: std::mem::size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particles view"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });
        let capacity = 1024;
        Self {
            pipeline,
            view_buffer,
            bind_group,
            instances: Self::instance_buffer(device, capacity),
            capacity,
            count: 0,
        }
    }

    fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle instances"),
            size: (capacity * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

/// One frame's particles, handed to egui as a paint callback.
pub struct ParticleCallback {
    pub instances: Vec<Instance>,
    view: View,
}

impl ParticleCallback {
    /// `rect` is the canvas in points; `center` and `pixels_per_meter`
    /// describe the camera; `radius_points` is each particle's drawn radius.
    pub fn new(
        instances: Vec<Instance>,
        rect: Rect,
        center: Vec2,
        pixels_per_meter: f32,
        radius_points: f32,
    ) -> Self {
        let half = egui::vec2(rect.width(), rect.height()) * 0.5;
        Self {
            instances,
            view: View {
                center: center.to_array(),
                scale: [pixels_per_meter / half.x, pixels_per_meter / half.y],
                radius: [radius_points / half.x, radius_points / half.y],
                _padding: [0.0; 2],
            },
        }
    }

    pub fn into_shape(self, rect: Rect) -> egui::Shape {
        egui::Shape::Callback(egui_wgpu::Callback::new_paint_callback(rect, self))
    }
}

impl CallbackTrait for ParticleCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<ParticleRenderer>() else {
            return Vec::new();
        };
        if self.instances.len() > r.capacity {
            r.capacity = self.instances.len().next_power_of_two();
            r.instances = ParticleRenderer::instance_buffer(device, r.capacity);
        }
        queue.write_buffer(&r.view_buffer, 0, bytemuck::bytes_of(&self.view));
        queue.write_buffer(&r.instances, 0, bytemuck::cast_slice(&self.instances));
        r.count = self.instances.len() as u32;
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(r) = resources.get::<ParticleRenderer>() else {
            return;
        };
        if r.count == 0 {
            return;
        }
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, &r.bind_group, &[]);
        pass.set_vertex_buffer(0, r.instances.slice(..));
        pass.draw(0..6, 0..r.count);
    }
}

/// sRGB color as the 4 bytes the instance buffer wants.
pub fn color_bytes(c: Color32) -> [u8; 4] {
    c.to_array()
}
