//! The canvas: mouse interaction and drawing.

use super::{
    CARDINAL, CONTACT_COLOR, ColorBy, Drag, EditorApp, FluidShape, GOLD, Grab, HANDLE_RADIUS_PX,
    MAX_THROW_SPEED, MIN_HALF_SIZE, PRESSURE_COLOR_DEPTH, ROTATE_HANDLE_OFFSET_PX,
    SPEED_COLOR_SCALE, STATIC_FILL, Tool, WATER_DEEP, WATER_LIGHT,
};
use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Sense, Stroke, Ui};
use engine::{BodyId, RigidBody, Shape, Vec2};

/// How close (px) the pointer must be to a handle to grab it.
const HANDLE_GRAB_PX: f32 = HANDLE_RADIUS_PX + 4.0;

impl EditorApp {
    /// Draws the world and handles mouse interaction. Returns the cursor's
    /// world position, if it's over the canvas.
    pub(super) fn canvas(&mut self, ui: &mut Ui) -> Option<Vec2> {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let viewport = response.rect;
        let cursor_screen = response.hover_pos();
        let cursor_world = cursor_screen.map(|p| self.camera.screen_to_world(viewport, p));

        // --- Camera: zoom with scroll/pinch, pan with right or middle drag.
        if let Some(anchor) = cursor_screen {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll * 0.0015).exp();
            if factor != 1.0 {
                self.camera.zoom_about(viewport, anchor, factor);
            }
        }
        if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
        {
            self.camera.pan_pixels(response.drag_delta());
        }

        // --- Tools.
        match self.tool {
            Tool::Select => self.select_tool_input(ui, &response, viewport, cursor_world),
            Tool::Fluid | Tool::Erase => self.fluid_tool_input(ui, &response, viewport),
            Tool::Circle | Tool::Box => {
                if response.clicked_by(PointerButton::Primary)
                    && let Some(p) = cursor_world
                    && let Some(body) = self.new_body.build(self.tool, self.snap_point(p))
                {
                    let id = self.world.add_body(body);
                    self.select_only(id);
                    self.status = "Placed. Esc or V returns to Select.".into();
                }
            }
        }

        // --- Drawing.
        let visuals = ui.visuals();
        painter.rect_filled(viewport, 0.0, visuals.extreme_bg_color);
        self.draw_grid(&painter, viewport, visuals.weak_text_color());
        self.draw_fluid(&painter, viewport);

        let selected: Vec<BodyId> = self.selected_ids();
        for (i, body) in self.world.bodies.iter().enumerate() {
            if !selected.contains(&BodyId(i)) {
                self.draw_body(&painter, viewport, body, false, None);
            }
        }
        // Selected bodies last, so their outlines aren't hidden by neighbors.
        for &id in &selected {
            if let Some(body) = self.world.body(id) {
                self.draw_body(&painter, viewport, body, true, None);
            }
        }
        if self.show_contacts {
            self.draw_contacts(&painter, viewport);
        }
        if self.show_fluid_forces && !self.world.fluid.is_empty() {
            self.draw_fluid_forces(&painter, viewport);
        }
        if self.tool == Tool::Select {
            self.draw_handles(&painter, viewport);
            if let Some(Drag::Marquee { start, end, .. }) = &self.drag {
                let rect = Rect::from_two_pos(
                    self.camera.world_to_screen(viewport, *start),
                    self.camera.world_to_screen(viewport, *end),
                );
                painter.rect_filled(rect, 0.0, GOLD.gamma_multiply(0.15));
                painter.rect_stroke(rect, 0.0, Stroke::new(1.5, GOLD), egui::StrokeKind::Middle);
            }
        }

        self.draw_fluid_tool_preview(&painter, viewport, cursor_world);

        // Ghost preview of the object about to be placed.
        if let Some(p) = cursor_world
            && let Some(ghost) = self.new_body.build(self.tool, self.snap_point(p))
        {
            self.draw_body(
                &painter,
                viewport,
                &ghost,
                false,
                Some(GOLD.gamma_multiply(0.45)),
            );
        }

        if self.running || response.dragged() {
            ui.ctx().request_repaint();
        }
        cursor_world
    }

    // ---------------------------------------------------------------------
    // Select tool

    fn select_tool_input(
        &mut self,
        ui: &Ui,
        response: &egui::Response,
        viewport: Rect,
        cursor_world: Option<Vec2>,
    ) {
        let shift = ui.input(|i| i.modifiers.shift);
        if response.drag_started_by(PointerButton::Primary) {
            // Use where the button went down, not where the drag was
            // recognized a few pixels later.
            let press = ui
                .input(|i| i.pointer.press_origin())
                .or(response.hover_pos());
            self.drag = press.map(|p| self.start_drag(viewport, p, shift));
        }
        if response.dragged_by(PointerButton::Primary)
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let pointer = self.camera.screen_to_world(viewport, pointer);
            let dt = ui.input(|i| i.stable_dt).max(1e-4);
            self.continue_drag(pointer, dt);
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            self.finish_drag();
        }
        if response.clicked_by(PointerButton::Primary) {
            match cursor_world.and_then(|p| self.world.body_at_point(p)) {
                Some(id) if shift => self.toggle_selected(id),
                Some(id) => self.select_only(id),
                None if !shift => self.selection.clear(),
                None => {}
            }
        }
    }

    /// World position of the rotate handle: above the body, turning with it
    /// so you can see which way is "up" for it.
    fn rotate_handle(&self, body: &RigidBody) -> Vec2 {
        let reach =
            body.shape.bounding_radius() + ROTATE_HANDLE_OFFSET_PX / self.camera.pixels_per_meter;
        body.local_to_world(Vec2::new(0.0, reach))
    }

    /// World position of the resize handle: a box's top-right corner, or a
    /// circle's rim at the end of its spoke.
    fn resize_handle(&self, body: &RigidBody) -> Vec2 {
        match body.shape {
            Shape::Box { half_extents } => body.local_to_world(half_extents),
            Shape::Circle { radius } => body.local_to_world(Vec2::new(radius, 0.0)),
        }
    }

    /// Decide what a left-drag starting at screen point `press` does.
    fn start_drag(&mut self, viewport: Rect, press: Pos2, shift: bool) -> Drag {
        let grab = self.camera.screen_to_world(viewport, press);

        // Handles only appear (and work) with exactly one body selected.
        if let Some(id) = self.single_selected()
            && let Some(body) = self.world.body(id)
            && let Some(handle) = body.handle()
        {
            let near = |p: Vec2| {
                self.camera.world_to_screen(viewport, p).distance(press) <= HANDLE_GRAB_PX
            };
            if near(self.rotate_handle(body)) {
                let to_grab = grab - body.position;
                return Drag::Rotate {
                    handle,
                    offset: body.angle - to_grab.to_angle(),
                };
            }
            if near(self.resize_handle(body)) {
                return Drag::Resize { handle };
            }
        }

        let Some(id) = self.world.body_at_point(grab) else {
            return Drag::Marquee {
                start: grab,
                end: grab,
                additive: shift,
            };
        };
        if !self.is_selected(id) {
            if shift {
                self.add_to_selection(id);
            } else {
                self.select_only(id);
            }
        }
        let grabs = self
            .selected_ids()
            .into_iter()
            .filter_map(|id| {
                let body = self.world.body(id)?;
                Some(Grab {
                    handle: body.handle()?,
                    offset: body.position - grab,
                    prev_position: body.position,
                })
            })
            .collect();
        Drag::Move { grabs }
    }

    /// Apply an in-progress drag with the cursor at world point `pointer`.
    fn continue_drag(&mut self, pointer: Vec2, dt: f32) {
        let Some(mut drag) = self.drag.take() else {
            return;
        };
        match &mut drag {
            Drag::Move { grabs } => {
                // Snap the first body and move the rest by the same amount,
                // so the group keeps its shape.
                let correction = grabs.first().map_or(Vec2::ZERO, |g| {
                    let target = pointer + g.offset;
                    self.snap_point(target) - target
                });
                for g in grabs.iter_mut() {
                    let target = pointer + g.offset + correction;
                    if let Some(body) = self
                        .world
                        .find(g.handle)
                        .and_then(|id| self.world.body_mut(id))
                    {
                        body.position = target;
                        body.angular_velocity = 0.0;
                        // While playing, bodies leave the mouse with the
                        // speed they were dragged at, so you can throw them.
                        body.velocity = if self.running {
                            ((target - g.prev_position) / dt).clamp_length_max(MAX_THROW_SPEED)
                        } else {
                            Vec2::ZERO
                        };
                    }
                    g.prev_position = target;
                }
            }
            Drag::Rotate { handle, offset } => {
                let snap = self.snapper();
                if let Some(body) = self
                    .world
                    .find(*handle)
                    .and_then(|id| self.world.body_mut(id))
                {
                    let to_cursor = pointer - body.position;
                    if to_cursor.length_squared() > 1e-8 {
                        body.angle = snap.angle(to_cursor.to_angle() + *offset);
                    }
                    // Hold it still while rotating so it doesn't fall away.
                    body.velocity = Vec2::ZERO;
                    body.angular_velocity = 0.0;
                }
            }
            Drag::Resize { handle } => {
                let snap = self.snapper();
                if let Some(body) = self
                    .world
                    .find(*handle)
                    .and_then(|id| self.world.body_mut(id))
                {
                    let local = body.world_to_local(pointer);
                    match &mut body.shape {
                        Shape::Box { half_extents } => {
                            // Resizes symmetrically about the center.
                            let full = Vec2::new(
                                snap.length(2.0 * local.x.abs()),
                                snap.length(2.0 * local.y.abs()),
                            );
                            *half_extents = (0.5 * full).max(Vec2::splat(MIN_HALF_SIZE));
                        }
                        Shape::Circle { radius } => {
                            *radius = (0.5 * snap.length(2.0 * local.length())).max(MIN_HALF_SIZE);
                        }
                    }
                    body.velocity = Vec2::ZERO;
                    body.angular_velocity = 0.0;
                }
            }
            Drag::Marquee { end, .. } => *end = pointer,
        }
        self.drag = Some(drag);
    }

    fn finish_drag(&mut self) {
        if let Some(Drag::Marquee {
            start,
            end,
            additive,
        }) = self.drag.take()
        {
            let (min, max) = (start.min(end), start.max(end));
            if !additive {
                self.selection.clear();
            }
            let inside: Vec<BodyId> = (0..self.world.bodies.len())
                .map(BodyId)
                .filter(|&id| {
                    self.world
                        .body(id)
                        .is_some_and(|b| b.position.cmpge(min).all() && b.position.cmple(max).all())
                })
                .collect();
            for id in inside {
                self.add_to_selection(id);
            }
        }
        self.drag = None;
    }

    fn draw_handles(&self, painter: &egui::Painter, viewport: Rect) {
        let Some(body) = self.single_selected().and_then(|id| self.world.body(id)) else {
            return;
        };
        let to_screen = |p: Vec2| self.camera.world_to_screen(viewport, p);
        let outline = Stroke::new(1.0, Color32::from_black_alpha(160));

        let center = to_screen(body.position);
        let rotate = to_screen(self.rotate_handle(body));
        painter.line_segment([center, rotate], Stroke::new(1.5, GOLD));
        painter.circle_filled(rotate, HANDLE_RADIUS_PX, GOLD);
        painter.circle_stroke(rotate, HANDLE_RADIUS_PX, outline);

        let resize = Rect::from_center_size(
            to_screen(self.resize_handle(body)),
            egui::vec2(2.0 * HANDLE_RADIUS_PX, 2.0 * HANDLE_RADIUS_PX),
        );
        painter.rect_filled(resize, 1.0, GOLD);
        painter.rect_stroke(resize, 1.0, outline, egui::StrokeKind::Middle);
    }

    // ---------------------------------------------------------------------
    // Fluid tools

    /// Mouse handling for the Fluid and Erase tools.
    fn fluid_tool_input(&mut self, ui: &Ui, response: &egui::Response, viewport: Rect) {
        let to_world = |p: Pos2| self.camera.screen_to_world(viewport, p);
        let pointer = response.interact_pointer_pos().map(to_world);
        let painting = response.dragged_by(PointerButton::Primary)
            || response.clicked_by(PointerButton::Primary);

        match (self.tool, self.fluid_shape) {
            (Tool::Fluid, FluidShape::Rectangle) => {
                if response.drag_started_by(PointerButton::Primary) {
                    self.fluid_rect_start = ui.input(|i| i.pointer.press_origin()).map(to_world);
                }
                if response.drag_stopped_by(PointerButton::Primary)
                    && let (Some(a), Some(b)) = (self.fluid_rect_start.take(), pointer)
                {
                    let added = self.world.fluid.fill_rect(a, b, &self.world.bodies);
                    self.status = format!("Added {added} particles.");
                }
            }
            (Tool::Fluid, FluidShape::Brush) => {
                if painting && let Some(p) = pointer {
                    self.world
                        .fluid
                        .fill_disc(p, self.brush_radius, &self.world.bodies);
                }
            }
            (Tool::Erase, _) => {
                if painting && let Some(p) = pointer {
                    self.world.fluid.remove_disc(p, self.brush_radius);
                }
            }
            _ => {}
        }
    }

    fn draw_fluid_tool_preview(
        &self,
        painter: &egui::Painter,
        viewport: Rect,
        cursor: Option<Vec2>,
    ) {
        let stroke = Stroke::new(1.5, WATER_DEEP);
        match (self.tool, self.fluid_shape) {
            (Tool::Fluid, FluidShape::Rectangle) => {
                if let (Some(a), Some(b)) = (self.fluid_rect_start, cursor) {
                    let rect = Rect::from_two_pos(
                        self.camera.world_to_screen(viewport, a),
                        self.camera.world_to_screen(viewport, b),
                    );
                    painter.rect_filled(rect, 0.0, WATER_LIGHT.gamma_multiply(0.35));
                    painter.rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Middle);
                }
            }
            (Tool::Fluid, FluidShape::Brush) | (Tool::Erase, _) => {
                if let Some(c) = cursor {
                    let at = self.camera.world_to_screen(viewport, c);
                    let r = self.camera.scale(self.brush_radius);
                    if self.tool == Tool::Fluid {
                        painter.circle_filled(at, r, WATER_LIGHT.gamma_multiply(0.3));
                    }
                    painter.circle_stroke(at, r, stroke);
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------------
    // Drawing

    fn draw_fluid(&self, painter: &egui::Painter, viewport: Rect) {
        let fluid = &self.world.fluid;
        if fluid.is_empty() {
            return;
        }
        let radius = (self.camera.scale(fluid.params.particle_spacing) * 0.6).max(1.5);
        let pressures = fluid.pressures();
        let full_pressure =
            fluid.params.rest_density * engine::STANDARD_GRAVITY * PRESSURE_COLOR_DEPTH;
        let visible = viewport.expand(radius);
        for (i, &p) in fluid.positions.iter().enumerate() {
            let at = self.camera.world_to_screen(viewport, p);
            if !visible.contains(at) {
                continue;
            }
            let t = match self.color_by {
                ColorBy::Speed => {
                    fluid.velocities.get(i).map_or(0.0, |v| v.length()) / SPEED_COLOR_SCALE
                }
                ColorBy::Pressure => pressures.get(i).map_or(0.0, |&q| q / full_pressure),
            };
            painter.circle_filled(
                at,
                radius,
                lerp_color(WATER_DEEP, WATER_LIGHT, t.clamp(0.0, 1.0)),
            );
        }
    }

    /// An arrow from each dynamic body's center showing the fluid's force on
    /// it, scaled so the body's own weight would be a 1 m arrow. A floating
    /// body at rest shows an arrow exactly that long, pointing up.
    fn draw_fluid_forces(&self, painter: &egui::Painter, viewport: Rect) {
        for (i, body) in self.world.bodies.iter().enumerate() {
            if body.is_static {
                continue;
            }
            let (force, _) = self.world.fluid_force(BodyId(i));
            if force == Vec2::ZERO {
                continue;
            }
            let weight = body.mass() * engine::STANDARD_GRAVITY;
            let meters = force / weight;
            let from = self.camera.world_to_screen(viewport, body.position);
            let to = self
                .camera
                .world_to_screen(viewport, body.position + meters);
            painter.arrow(from, to - from, Stroke::new(2.5, WATER_DEEP));
        }
    }

    /// Debug overlay: a dot at each contact point and a short line along the
    /// contact normal (pointing from body A toward body B).
    fn draw_contacts(&self, painter: &egui::Painter, viewport: Rect) {
        let normal_px = 18.0;
        for m in self.world.contacts() {
            for p in &m.points {
                let at = self.camera.world_to_screen(viewport, p.position);
                let tip = self.camera.world_to_screen(
                    viewport,
                    p.position + m.normal * (normal_px / self.camera.pixels_per_meter),
                );
                painter.line_segment([at, tip], Stroke::new(2.0, CONTACT_COLOR));
                painter.circle_filled(at, 3.5, CONTACT_COLOR);
            }
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, viewport: Rect, color: Color32) {
        let spacing = self.camera.grid_spacing(80.0);
        let (min, max) = self.camera.visible_world_bounds(viewport);
        let minor = Stroke::new(1.0, color.gamma_multiply(0.18));
        let axis = Stroke::new(1.5, color.gamma_multiply(0.6));

        let to_screen = |p: Vec2| self.camera.world_to_screen(viewport, p);

        let mut x = (min.x / spacing).floor() * spacing;
        while x <= max.x {
            let stroke = if x.abs() < spacing * 0.5 { axis } else { minor };
            painter.line_segment(
                [
                    to_screen(Vec2::new(x, min.y)),
                    to_screen(Vec2::new(x, max.y)),
                ],
                stroke,
            );
            x += spacing;
        }
        let mut y = (min.y / spacing).floor() * spacing;
        while y <= max.y {
            let stroke = if y.abs() < spacing * 0.5 { axis } else { minor };
            painter.line_segment(
                [
                    to_screen(Vec2::new(min.x, y)),
                    to_screen(Vec2::new(max.x, y)),
                ],
                stroke,
            );
            y += spacing;
        }

        painter.text(
            viewport.left_bottom() + egui::vec2(8.0, -8.0),
            egui::Align2::LEFT_BOTTOM,
            format!("grid {spacing} m"),
            egui::FontId::proportional(12.0),
            color,
        );
    }

    fn draw_body(
        &self,
        painter: &egui::Painter,
        viewport: Rect,
        body: &RigidBody,
        selected: bool,
        override_fill: Option<Color32>,
    ) {
        let fill = override_fill.unwrap_or(if body.is_static {
            STATIC_FILL
        } else {
            CARDINAL
        });
        let outline = if selected {
            Stroke::new(3.0, GOLD)
        } else {
            Stroke::new(1.0, Color32::from_black_alpha(140))
        };
        let to_screen = |p: Vec2| -> Pos2 { self.camera.world_to_screen(viewport, p) };

        match body.shape {
            Shape::Circle { radius } => {
                let center = to_screen(body.position);
                let r = self.camera.scale(radius);
                painter.circle_filled(center, r, fill);
                painter.circle_stroke(center, r, outline);
                // A spoke so you can see it rotate.
                let rim = to_screen(body.local_to_world(Vec2::new(radius, 0.0)));
                painter.line_segment(
                    [center, rim],
                    Stroke::new(1.5, Color32::from_white_alpha(200)),
                );
            }
            Shape::Box { .. } => {
                if let Some(corners) = body.box_corners() {
                    let points = corners.iter().map(|&c| to_screen(c)).collect();
                    painter.add(egui::Shape::convex_polygon(points, fill, outline));
                }
            }
        }
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}
