// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, HitboxId, Hsla,
    NativeViewId, Path, PathTiles, Pixels, Point, Radians, ScaledPixels, Size, TimedTransformation,
    bounds_tree::BoundsTree, point,
};
use smallvec::SmallVec;
use std::{
    collections::VecDeque, fmt::Debug, iter::Peekable, ops::Range, rc::Rc, slice, time::Duration,
};

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// The largest layer group texture made to the group's size, in device
/// pixels: the texture size every wgpu device supports, and less than any
/// Metal device's. A larger group takes the window's size.
const MAX_GROUP_EXTENT: f32 = 8192.0;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

impl PaddedBool32 {
    /// The wrapped boolean.
    pub fn as_bool(self) -> bool {
        self.0 != 0
    }
}

struct Layer {
    primitives: SmallVec<[Range<usize>; 1]>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
    /// Whether a native view was painted in the layer, which then can't be
    /// drawn as one: what was painted after the view has to cover it.
    has_native_view: bool,
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<usize>,
    layers: VecDeque<Layer>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<PaintedPath>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub native_views: Vec<PaintNativeView>,
    animations: Vec<SpriteAnimation>,
    pub shaders: Vec<crate::PaintShader>,
    pub backdrop_blurs: Vec<BackdropBlur>,
    /// Which drawn frame this scene holds, so a renderer can tell a
    /// re-present of the same frame from a new one.
    pub generation: u64,
    /// When the present being made of this scene is expected to reach the
    /// screen: what its shaders reckon their time at. `None` for now.
    pub present_at: Option<scheduler::Instant>,
    /// The scene as layers its shaders are drawn between, so a re-present
    /// composes them rather than drawing the whole scene; `None` where the
    /// shaders cannot be drawn apart from the rest.
    pub shader_layers: Option<Box<ShaderLayers>>,
    /// What was painted inside each layer group, as a scene of its own that
    /// a renderer draws into a texture for the group's shader to compose.
    pub layer_groups: Vec<Scene>,
    /// Where each group's content lies, when its texture is the size of the
    /// shader composing it and its content is moved to its origin; `None`
    /// for one drawn the window's size, as a group holding paths is (their
    /// intermediate texture is the window's).
    pub layer_group_frames: Vec<Option<Bounds<ScaledPixels>>>,
    group_stack: Vec<usize>,
    /// The dynamic colours this scene's primitives are drawn in, layer
    /// groups' included, so a renderer draws again what holds one that
    /// changed.
    pub dynamic_colors: crate::DynamicColorSet,
}

/// A scene with shaders, split where they are drawn: `layers[0]` first, then
/// the shaders of `groups[0]` (a range of `Scene::shaders`), then
/// `layers[1]`, and so on, `layers` one longer than `groups`. Each layer
/// holds the scene's other primitives drawn between those shaders, with
/// their orders, ready for a renderer to draw into a texture of its own.
///
/// A backdrop blur is a group of its own, with no shaders: what is composed
/// by then is blurred under it at each present, so it follows the shaders
/// drawn before it.
#[derive(Default)]
pub struct ShaderLayers {
    /// The primitives drawn before, between and after the groups.
    pub layers: Vec<Scene>,
    /// Each group's shaders, as a range of `Scene::shaders`.
    pub groups: Vec<Range<usize>>,
    /// The backdrop blur each group is, as an index into
    /// `Scene::backdrop_blurs`; `None` for a group of shaders.
    pub blurs: Vec<Option<usize>>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.layers.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.shaders.clear();
        self.backdrop_blurs.clear();
        self.layer_groups.clear();
        self.layer_group_frames.clear();
        self.group_stack.clear();
        self.dynamic_colors = Default::default();
        self.native_views.clear();
        self.animations.clear();
    }

    /// Inserts a sprite that [`Self::advance_animations`] keeps moving. The
    /// sprite's color is taken as its color before the animation's alpha.
    pub(crate) fn insert_animated_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        mut animation: SpriteAnimation,
    ) {
        sprite.animation = SpriteAnimationId(self.animations.len() as u32);
        animation.base_color = sprite.color;
        // A layer group's texture is drawn again only when it changes, so a
        // sprite in one is drawn where it stands.
        if !self.group_stack.is_empty() {
            self.insert_primitive(sprite);
            return;
        }
        if self.try_insert_primitive(sprite.into(), PaintOperation::Primitive) {
            self.animations.push(animation);
        }
    }

    /// How long the window may wait between presents of this scene while only
    /// its animations need a frame: `None` when nothing in it animates, and
    /// zero when an animation wants every frame.
    pub(crate) fn animation_frame_interval(&self) -> Option<Duration> {
        self.animations
            .iter()
            .map(|animation| {
                animation
                    .transformation
                    .min_frame_interval()
                    .unwrap_or_default()
            })
            .min()
    }

    /// Moves every animated sprite to its pose at `elapsed` time on the app's
    /// synced animation clock, without the scene being rebuilt.
    pub(crate) fn advance_animations(&mut self, elapsed: Duration) {
        for animation in &self.animations {
            let Some(sprite) = self.monochrome_sprites.get_mut(animation.sprite_index) else {
                continue;
            };
            let pose = animation.transformation.at(elapsed);
            sprite.transformation = pose.into_matrix(animation.center, animation.scale_factor);
            sprite.color = animation.base_color.opacity(pose.alpha_factor());
        }
        // The sprites moved: a new frame to a renderer, its shader layers
        // split again from them.
        if !self.animations.is_empty() {
            self.generation = next_generation();
            if self.shader_layers.is_some() {
                self.shader_layers = self.plan_shader_layers().map(Box::new);
            }
        }
    }

    /// The dynamic colours the primitives are drawn in.
    fn collect_dynamic_colors(&self) -> crate::DynamicColorSet {
        let mut set = crate::DynamicColorSet::default();
        for quad in &self.quads {
            set.note(&quad.border_color);
            set.note(&quad.background.solid);
            for stop in &quad.background.colors {
                set.note(&stop.color);
            }
        }
        for shadow in &self.shadows {
            set.note(&shadow.color);
        }
        for underline in &self.underlines {
            set.note(&underline.color);
        }
        for sprite in &self.monochrome_sprites {
            set.note(&sprite.color);
        }
        for sprite in &self.subpixel_sprites {
            set.note(&sprite.color);
        }
        for path in &self.paths {
            set.note(&path.paint.color.solid);
        }
        set
    }

    /// When the present being made of this scene is expected to reach the
    /// screen: what its shaders reckon their time at.
    pub fn present_time(&self) -> scheduler::Instant {
        self.present_at.unwrap_or_else(scheduler::Instant::now)
    }

    /// Whether a shader on screen asks for the window to present every
    /// refresh at `now`.
    pub fn has_live_shaders(&self, now: scheduler::Instant) -> bool {
        self.shaders
            .iter()
            .any(|shader| shader.uniforms.is_live(now))
            || self
                .layer_groups
                .iter()
                .any(|group| group.has_live_shaders(now))
    }

    /// Whether group `group`, held from an earlier present, is drawn again
    /// at the present at `now`: while a shader in it is live, once more after
    /// they stop so the last value lands, when a dynamic colour it is drawn
    /// in changed (`changed`), and when a group nested in it was drawn
    /// again. `was_moving` is the group's from the last present, updated
    /// here; `redrawn` holds the groups drawn this present.
    pub fn layer_group_stale(
        &self,
        group: usize,
        now: scheduler::Instant,
        changed: &crate::DynamicColorSet,
        was_moving: &mut bool,
        redrawn: &[bool],
    ) -> bool {
        let Some(group) = self.layer_groups.get(group) else {
            return false;
        };
        let moving = group.has_live_shaders(now);
        let landing = std::mem::replace(was_moving, moving);
        moving
            || landing
            || group.dynamic_colors.intersects(changed)
            || group.shaders.iter().any(|shader| {
                shader
                    .group
                    .is_some_and(|nested| redrawn.get(nested).copied().unwrap_or(false))
            })
    }

    /// Every primitive moved by `(dx, dy)`, its clip with it: a layer
    /// group's content to its texture's origin.
    pub fn offset(&mut self, dx: ScaledPixels, dy: ScaledPixels) {
        let by = |b: &mut Bounds<ScaledPixels>| {
            b.origin.x += dx;
            b.origin.y += dy;
        };
        for quad in &mut self.quads {
            by(&mut quad.bounds);
            by(&mut quad.content_mask.bounds);
        }
        for shadow in &mut self.shadows {
            by(&mut shadow.bounds);
            by(&mut shadow.content_mask.bounds);
            by(&mut shadow.shape_bounds);
        }
        for underline in &mut self.underlines {
            by(&mut underline.bounds);
            by(&mut underline.content_mask.bounds);
        }
        for sprite in &mut self.monochrome_sprites {
            by(&mut sprite.bounds);
            by(&mut sprite.content_mask.bounds);
        }
        for sprite in &mut self.subpixel_sprites {
            by(&mut sprite.bounds);
            by(&mut sprite.content_mask.bounds);
        }
        for sprite in &mut self.polychrome_sprites {
            by(&mut sprite.bounds);
            by(&mut sprite.content_mask.bounds);
        }
        for surface in &mut self.surfaces {
            by(&mut surface.bounds);
            by(&mut surface.content_mask.bounds);
        }
        for shader in &mut self.shaders {
            by(&mut shader.bounds);
            by(&mut shader.content_mask.bounds);
        }
        for blur in &mut self.backdrop_blurs {
            by(&mut blur.bounds);
            by(&mut blur.content_mask.bounds);
        }
        // A group holding paths is drawn the window's size, never moved.
        debug_assert!(self.paths.is_empty(), "paths are not offset");
    }

    /// Whether this refresh at `now` should present again: a shader's
    /// uniforms changed, or one moves with time. Asks every shader, so each
    /// one's change is taken once.
    pub fn take_shader_present(&self, now: scheduler::Instant) -> bool {
        let groups = self
            .layer_groups
            .iter()
            .fold(false, |any, group| group.take_shader_present(now) | any);
        self.shaders
            .iter()
            .fold(false, |any, shader| shader.uniforms.take_present(now) | any)
            | groups
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self) {
        self.layer_stack.push(self.layers.len());
        self.layers.push_back(Layer {
            primitives: SmallVec::new(),
            bounds: Bounds::default(),
            max_kind: PrimitiveKind::Shadow,
            max_shared_kind: PrimitiveKind::NativeView,
            has_native_view: false,
        });
        self.paint_operations.push(PaintOperation::StartLayer);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        if self.layer_stack.is_empty() {
            while let Some(layer) = self.layers.pop_front() {
                self.insert_layer(layer);
            }
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() {
            return;
        }
        if layer.has_native_view {
            self.insert_layer_around_native_views(layer);
            return;
        }
        let order =
            self.primitive_bounds
                .insert(layer.bounds, layer.max_kind, layer.max_shared_kind);
        for index in layer.primitives.into_iter().flatten() {
            if let Some(primitive) = self.paint_operations[index].primitive() {
                self.push_primitive(primitive, order);
            }
        }
    }

    /// Inserts a layer that a native view was painted in.
    ///
    /// A layer's primitives share one place in the drawing order, where they
    /// are drawn by kind and not in the order they were painted. A native
    /// view erases what is drawn before it, so the layer is inserted in parts:
    /// what was painted before each view, the view, and what was painted after.
    fn insert_layer_around_native_views(&mut self, layer: Layer) {
        let mut part = Vec::new();
        for index in layer.primitives.into_iter().flatten() {
            let Some(primitive) = self.paint_operations[index].primitive() else {
                continue;
            };
            if primitive.kind() == PrimitiveKind::NativeView {
                self.insert_layer_part(std::mem::take(&mut part));
                self.insert_layer_part(vec![primitive]);
            } else {
                part.push(primitive);
            }
        }
        self.insert_layer_part(part);
    }

    fn insert_layer_part(&mut self, primitives: Vec<Primitive>) {
        let Some(first) = primitives.first() else {
            return;
        };
        let mut bounds = first.ordered_bounds();
        let mut max_kind = PrimitiveKind::Shadow;
        let mut max_shared_kind = PrimitiveKind::NativeView;
        for primitive in &primitives {
            let kind = primitive.kind();
            bounds = bounds.join(&primitive.ordered_bounds());
            max_kind = max_kind.max(kind);
            max_shared_kind = max_shared_kind.min(kind.max_shared_kind());
        }
        let order = self
            .primitive_bounds
            .insert(bounds, max_kind, max_shared_kind);
        for primitive in primitives {
            self.push_primitive(primitive, order);
        }
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.try_insert_primitive(primitive.into(), PaintOperation::Primitive);
    }

    /// Returns whether the primitive was kept, rather than clipped away.
    /// `operation` records it for a cached view's replay.
    fn try_insert_primitive(
        &mut self,
        primitive: Primitive,
        operation: impl FnOnce(Primitive) -> PaintOperation,
    ) -> bool {
        // Painted inside a layer group: into its scene, recorded here so a
        // cached view replays it.
        if let Some(&group) = self.group_stack.last() {
            self.layer_groups[group].insert_primitive(primitive.clone());
            self.paint_operations.push(operation(primitive));
            return true;
        }
        let clipped_bounds = primitive.clipped_bounds();
        if clipped_bounds.is_empty() {
            return false;
        }

        let kind = primitive.kind();
        let ordered_bounds = primitive.ordered_bounds();
        if let Some(&layer) = self.layer_stack.last()
            && let Some(layer) = self.layers.get_mut(layer)
        {
            layer.bounds = layer.bounds.join(&ordered_bounds);
            layer.max_kind = layer.max_kind.max(kind);
            layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
            layer.has_native_view |= kind == PrimitiveKind::NativeView;
            let index = self.paint_operations.len();
            match layer.primitives.last_mut() {
                Some(range) if range.end == index => range.end += 1,
                _ => layer.primitives.push(index..index + 1),
            }
        } else {
            let order = self
                .primitive_bounds
                .insert(ordered_bounds, kind, kind.max_shared_kind());
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations.push(operation(primitive));
        true
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
            }
            Primitive::NativeView(native_view) => self.native_views.push(PaintNativeView {
                order,
                ..native_view
            }),
            Primitive::Shader(mut shader) => {
                shader.order = order;
                self.shaders.push(shader);
            }
            Primitive::BackdropBlur(blur) => {
                self.backdrop_blurs.push(BackdropBlur { order, ..blur })
            }
        }
    }

    /// Opens a layer group: what is painted until `pop_group` is drawn into
    /// a texture of its own.
    pub fn push_group(&mut self) -> usize {
        let group = self.layer_groups.len();
        self.layer_groups.push(Scene::default());
        self.layer_group_frames.push(None);
        self.group_stack.push(group);
        self.paint_operations.push(PaintOperation::StartGroup);
        group
    }

    /// Closes the open group, composed by `shader` sampling its texture.
    pub fn pop_group(&mut self, mut shader: crate::PaintShader) {
        let Some(group) = self.group_stack.pop() else {
            return;
        };
        shader.group = Some(group);
        // The texture is the visible part of the group's bounds, its content
        // at its origin, unless it holds paths or would pass the largest
        // texture a device is sure to make. Content outside the mask is
        // clipped by its own masks already.
        let frame = shader.bounds.intersect(&shader.content_mask.bounds);
        let local = self.layer_groups[group].paths.is_empty()
            && frame.size.width.0 <= MAX_GROUP_EXTENT
            && frame.size.height.0 <= MAX_GROUP_EXTENT;
        let origin = if local {
            frame.origin
        } else {
            Point::default()
        };
        shader.group_inset = point(
            ScaledPixels(shader.bounds.origin.x.0 - origin.x.0),
            ScaledPixels(shader.bounds.origin.y.0 - origin.y.0),
        );
        self.layer_group_frames[group] = local.then_some(frame);
        // Into the enclosing group's scene, or this one, recorded as the
        // group's end so a replay closes the group, drawn or clipped away.
        let end = PaintOperation::EndGroup(shader.clone());
        if !self.try_insert_primitive(shader.into(), |_| end.clone()) {
            self.paint_operations.push(end);
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    // Animation ids index the scene they were inserted into.
                    if let Primitive::MonochromeSprite(sprite) = primitive
                        && let Some(animation) =
                            prev_scene.animations.get(sprite.animation.0 as usize)
                    {
                        self.insert_animated_sprite(*sprite, *animation);
                    } else {
                        self.insert_primitive(primitive.clone());
                    }
                }
                PaintOperation::StartLayer => self.push_layer(),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::StartGroup => {
                    self.push_group();
                }
                PaintOperation::EndGroup(shader) => self.pop_group(shader.clone()),
            }
        }
    }

    pub fn finish(&mut self) {
        debug_assert!(self.layer_stack.is_empty(), "unbalanced scene layers");
        let keys = &mut Vec::new();
        sort_in_place(&mut self.shadows, keys, |shadow| (shadow.order, 0));
        sort_in_place(&mut self.quads, keys, |quad| (quad.order, 0));
        self.paths.sort_by_key(|path| path.order);
        sort_in_place(&mut self.underlines, keys, |underline| (underline.order, 0));
        sort_in_place(&mut self.monochrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.subpixel_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.polychrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        self.surfaces.sort_by_key(|surface| surface.order);
        self.backdrop_blurs.sort_by_key(|blur| blur.order);
        // Kept together by program and image within an order.
        self.shaders.sort_by_key(|shader| {
            (
                shader.order,
                shader.program as *const crate::ShaderProgram as usize,
                shader.image.as_ref().map(|tile| tile.texture_id.index),
            )
        });
        self.native_views
            .sort_by_key(|native_view| native_view.order);
        self.keep_first_painted_native_views();

        // Sorting moved the sprites, so animations learn where theirs ended up.
        if !self.animations.is_empty() {
            for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
                if let Some(animation) = self.animations.get_mut(sprite.animation.0 as usize) {
                    animation.sprite_index = index;
                }
            }
        }

        self.generation = next_generation();
        // Each group drawn whole into its texture, moved to its texture's
        // origin where it is the group's size.
        for (group, frame) in self.layer_groups.iter_mut().zip(&self.layer_group_frames) {
            if let Some(frame) = frame {
                group.offset(
                    ScaledPixels(-frame.origin.x.0),
                    ScaledPixels(-frame.origin.y.0),
                );
            }
            group.finish();
            group.shader_layers = None;
        }
        self.dynamic_colors = self.collect_dynamic_colors();
        for group in &self.layer_groups {
            self.dynamic_colors.extend(&group.dynamic_colors);
        }
        self.shader_layers = self.plan_shader_layers().map(Box::new);
    }

    /// Where this scene's primitives can draw: the union of each one's
    /// bounds, a shadow's with its blur's reach and a transformed sprite's
    /// as transformed, within its content mask.
    /// `None` for a scene that draws nothing. A renderer composing the scene
    /// as a layer need touch no pixel outside it.
    pub fn painted_bounds(&self) -> Option<Bounds<ScaledPixels>> {
        let mut painted: Option<Bounds<ScaledPixels>> = None;
        let mut add =
            |bounds: Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>, reach: f32| {
                let drawn = bounds.dilate(ScaledPixels(reach)).intersect(&mask.bounds);
                if drawn.is_empty() {
                    return;
                }
                painted = Some(painted.map_or(drawn, |p| p.union(&drawn)));
            };
        // A pixel either way for antialiased edges.
        for s in &self.shadows {
            add(s.bounds, &s.content_mask, s.blur_radius.0 * 3.0 + 1.0);
        }
        macro_rules! each {
            ($($list:ident),*) => {$(
                for p in &self.$list {
                    add(p.bounds, &p.content_mask, 1.0);
                }
            )*};
        }
        // Drawn where its transformation takes its corners, and clipped
        // there, as the sprite program does.
        for sprite in &self.monochrome_sprites {
            add(
                sprite.transformation.bounds_of(sprite.bounds),
                &sprite.content_mask,
                1.0,
            );
        }
        for path in &self.paths {
            add(path.bounds, &path.paint.content_mask, 1.0);
        }
        each!(
            quads,
            underlines,
            subpixel_sprites,
            polychrome_sprites,
            surfaces,
            shaders
        );
        for b in &self.backdrop_blurs {
            add(b.bounds, &b.content_mask, 1.0);
        }
        painted
    }

    /// Whether this scene holds no primitives.
    pub fn is_blank(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.paths.is_empty()
            && self.underlines.is_empty()
            && self.monochrome_sprites.is_empty()
            && self.subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
            && self.surfaces.is_empty()
            && self.shaders.is_empty()
            && self.backdrop_blurs.is_empty()
            && self.native_views.is_empty()
    }

    // Splits the scene where its shaders are drawn.
    //
    // Shaders in draw order are grouped, a new group starting at a shader
    // that overlaps one already in the group, so shaders side by side are
    // drawn as one group. Every other primitive must be drawn after
    // each shader it overlaps with a lower order (at equal orders the shader
    // is drawn last), after every primitive drawn before it that it
    // overlaps, and before each shader it overlaps with an equal or higher
    // order. It goes in the lowest layer meeting the first two; if that is
    // past a group it must precede, the scene cannot be split and is drawn
    // whole. Layers then hold what is drawn between groups, each in its own
    // draw order, so a composed present draws everything in the order a
    // whole one would wherever two things overlap.
    fn plan_shader_layers(&self) -> Option<ShaderLayers> {
        // A native view is composited by the platform where it is ordered,
        // which a split scene would lose.
        if self.shaders.is_empty() || !self.native_views.is_empty() {
            return None;
        }
        let visible = |bounds: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>| {
            bounds.intersect(&mask.bounds)
        };
        // What is drawn at each present, in draw order: the shaders, and the
        // backdrop blurs, which read what is composed under them. A blur is
        // drawn before what shares its order and a shader after it, as the
        // batches draw them. Each holds whether it is a shader, and its
        // shader index or blur index.
        let mut presented: Vec<(DrawOrder, bool, usize, Bounds<ScaledPixels>)> = self
            .shaders
            .iter()
            .enumerate()
            .map(|(i, s)| (s.order, true, i, visible(&s.bounds, &s.content_mask)))
            .chain(
                self.backdrop_blurs
                    .iter()
                    .enumerate()
                    .map(|(i, blur)| (blur.order, false, i, blur.backdrop())),
            )
            .collect();
        // Stable, so the shaders keep their place in `Scene::shaders`.
        presented.sort_by_key(|(order, is_shader, _, _)| (*order, *is_shader));
        let mut groups = Vec::new();
        let mut blurs = Vec::new();
        let mut group_of = Vec::with_capacity(presented.len());
        // The open group of shaders: its first shader and what it covers.
        let mut open: Option<usize> = None;
        let mut members: Vec<Bounds<ScaledPixels>> = Vec::new();
        let mut next_shader = 0;
        for (_, is_shader, i, bounds) in &presented {
            if *is_shader {
                if open.is_some() && members.iter().any(|b| b.intersects(bounds)) {
                    groups.push(open.take().unwrap_or(*i)..*i);
                    blurs.push(None);
                    members.clear();
                }
                open.get_or_insert(*i);
                members.push(*bounds);
                group_of.push(groups.len());
                next_shader = i + 1;
            } else {
                if let Some(start) = open.take() {
                    groups.push(start..next_shader);
                    blurs.push(None);
                    members.clear();
                }
                group_of.push(groups.len());
                groups.push(next_shader..next_shader);
                blurs.push(Some(*i));
            }
        }
        if let Some(start) = open {
            groups.push(start..next_shader);
            blurs.push(None);
        }

        // Each other primitive in the order the renderer draws them: by
        // order, then kind, then place in its list.
        let mut drawn: Vec<(DrawOrder, PrimitiveKind, usize, Bounds<ScaledPixels>)> = Vec::new();
        macro_rules! collect {
            ($($list:ident => $kind:ident),*) => {$(
                for (i, p) in self.$list.iter().enumerate() {
                    drawn.push((p.order, PrimitiveKind::$kind, i, visible(&p.bounds, &p.content_mask)));
                }
            )*};
        }
        for (i, path) in self.paths.iter().enumerate() {
            drawn.push((
                path.order,
                PrimitiveKind::Path,
                i,
                visible(&path.bounds, &path.paint.content_mask),
            ));
        }
        collect!(
            shadows => Shadow,
            quads => Quad,
            underlines => Underline,
            monochrome_sprites => MonochromeSprite,
            subpixel_sprites => SubpixelSprite,
            polychrome_sprites => PolychromeSprite,
            surfaces => Surface
        );
        drawn.sort_unstable_by_key(|(order, kind, i, _)| (*order, *kind, *i));

        // From the shaders and blurs alone: the layer after the last group
        // holding one drawn before it that it overlaps, and the first group
        // holding one it overlaps that is drawn after it.
        let by_shaders = |order: DrawOrder, bounds: &Bounds<ScaledPixels>| {
            let (mut after, mut before) = (0, usize::MAX);
            for (k, (o, is_shader, _, s)) in presented.iter().enumerate() {
                if s.intersects(bounds) {
                    let drawn_before = if *is_shader { *o < order } else { *o <= order };
                    if drawn_before {
                        after = after.max(group_of[k] + 1);
                    } else {
                        before = before.min(group_of[k]);
                    }
                }
            }
            (after, before)
        };
        let placed: Vec<(usize, usize)> =
            drawn.iter().map(|(o, _, _, b)| by_shaders(*o, b)).collect();
        // A primitive also follows every earlier one it overlaps, so a menu's
        // text stays over the menu's ground wherever each is placed. Only
        // what sits above the lowest layer anything was placed in can raise
        // another, which is the little drawn over a shader: those are kept.
        let floor = placed.iter().map(|(after, _)| *after).min().unwrap_or(0);
        let mut raised: Vec<(Bounds<ScaledPixels>, usize)> = Vec::new();
        let mut layers_by_kind: Vec<Vec<usize>> = vec![Vec::new(); PrimitiveKind::COUNT];
        macro_rules! size_layers {
            ($($list:ident => $kind:ident),*) => {$(
                layers_by_kind[PrimitiveKind::$kind as usize] = vec![0; self.$list.len()];
            )*};
        }
        size_layers!(
            shadows => Shadow,
            quads => Quad,
            paths => Path,
            underlines => Underline,
            monochrome_sprites => MonochromeSprite,
            subpixel_sprites => SubpixelSprite,
            polychrome_sprites => PolychromeSprite,
            surfaces => Surface
        );
        for ((_, kind, i, bounds), (after, before)) in drawn.iter().zip(&placed) {
            let mut layer = *after;
            for (b, l) in &raised {
                if *l > layer && b.intersects(bounds) {
                    layer = *l;
                }
            }
            if layer > *before {
                return None;
            }
            if layer > floor {
                raised.push((*bounds, layer));
            }
            layers_by_kind[*kind as usize][*i] = layer;
        }

        let mut layers: Vec<Scene> = (0..=groups.len()).map(|_| Scene::default()).collect();
        macro_rules! split {
            ($($list:ident => $kind:ident),*) => {$(
                for (p, layer) in self.$list.iter().zip(&layers_by_kind[PrimitiveKind::$kind as usize]) {
                    layers[*layer].$list.push(p.clone());
                }
            )*};
        }
        split!(
            shadows => Shadow,
            quads => Quad,
            underlines => Underline,
            monochrome_sprites => MonochromeSprite,
            subpixel_sprites => SubpixelSprite,
            polychrome_sprites => PolychromeSprite,
            surfaces => Surface
        );
        for (path, layer) in self
            .paths
            .iter()
            .zip(&layers_by_kind[PrimitiveKind::Path as usize])
        {
            layers[*layer].paths.push(path.clone());
        }
        for layer in &mut layers {
            layer.dynamic_colors = layer.collect_dynamic_colors();
        }
        Some(ShaderLayers {
            layers,
            groups,
            blurs,
        })
    }

    /// Drops every occurrence of a native view but the one painted first,
    /// because a platform view can only be in one place.
    ///
    /// That is the occurrence whose hitbox takes the view's mouse input. It
    /// isn't always the first in the drawing order, which depends on what each
    /// occurrence overlaps.
    fn keep_first_painted_native_views(&mut self) {
        let native_views = &self.native_views;
        let is_repeated = native_views.iter().enumerate().any(|(index, native_view)| {
            native_views[..index]
                .iter()
                .any(|earlier| earlier.id == native_view.id)
        });
        if !is_repeated {
            return;
        }
        let mut first_painted: Vec<(NativeViewId, HitboxId)> = Vec::new();
        for operation in &self.paint_operations {
            if let PaintOperation::Primitive(Primitive::NativeView(native_view)) = operation
                && !first_painted.iter().any(|(id, _)| *id == native_view.id)
            {
                first_painted.push((native_view.id, native_view.hitbox_id));
            }
        }
        self.native_views
            .retain(|native_view| first_painted.contains(&(native_view.id, native_view.hitbox_id)));
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            shaders_start: 0,
            shaders_iter: self.shaders.iter().peekable(),
            backdrop_blurs_start: 0,
            backdrop_blurs_iter: self.backdrop_blurs.iter().peekable(),
            native_views_start: 0,
            native_views_iter: self.native_views.iter().peekable(),
        }
    }
}

/// A number no scene drawn before has had.
fn next_generation() -> u64 {
    static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Sorts small `(key, index)` pairs instead of the primitives themselves, then moves each
/// primitive into place once. Primitives are large, so a comparison sort that shuffles them
/// directly spends most of its time copying. Ties keep insertion order, as `sort_by_key`
/// does, because the index is part of what is sorted.
fn sort_in_place<T: Copy>(
    primitives: &mut [T],
    keys: &mut Vec<(u64, u32)>,
    key: impl Fn(&T) -> (u32, u32),
) {
    if primitives.is_sorted_by_key(&key) {
        return;
    }
    keys.clear();
    keys.extend(primitives.iter().enumerate().map(|(index, primitive)| {
        let (major, minor) = key(primitive);
        ((major as u64) << 32 | minor as u64, index as u32)
    }));
    keys.sort_unstable();

    // `keys[target].1` is the source index for `target`. Follow each cycle of that
    // permutation, marking visited slots by pointing them at themselves.
    for start in 0..primitives.len() {
        if keys[start].1 as usize == start {
            continue;
        }
        let displaced = primitives[start];
        let mut target = start;
        loop {
            let source = keys[target].1 as usize;
            keys[target].1 = target as u32;
            if source == start {
                primitives[target] = displaced;
                break;
            }
            primitives[target] = primitives[source];
            target = source;
        }
    }
}

/// The depth step between consecutive quads in a frame with `quad_count` quads.
pub fn quad_depth_scale(quad_count: usize) -> f32 {
    let denominator = (quad_count.min((1 << 24) - 1) + 1).next_power_of_two();
    1.0 / denominator as f32
}

/// Maps a quad's index in [`Scene::quads`] to depth, with greater values closer.
pub fn quad_depth(quad_id: u32, scale: f32) -> f32 {
    ((quad_id + 1) as f32 * scale).min(1.0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    /// First, so a blur reads none of what shares its order.
    BackdropBlur,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    Shader,
    NativeView,
}

impl PrimitiveKind {
    const COUNT: usize = PrimitiveKind::NativeView as usize + 1;

    /// We guarantee that overlapping primitives are drawn in paint order.
    /// But, to improve GPU draw batching, the renderer sorts primitives that
    /// share a draw order by kind, then by atlas texture, and only then in
    /// paint order. So, a primitive can only share the draw order of the
    /// topmost primitive it overlaps if that primitive's kind is at most the
    /// one returned here. Otherwise, it needs a new draw order.
    fn max_shared_kind(self) -> Self {
        match self {
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            // Shaders sharing an order are sorted by program and image.
            Self::Shader => Self::Surface,
            kind => kind,
        }
    }
}

#[derive(Clone)]
pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer,
    EndLayer,
    /// A layer group's start.
    StartGroup,
    /// A layer group's end, with the shader that composes it.
    EndGroup(crate::PaintShader),
}

impl PaintOperation {
    /// The primitive this operation inserted, if any: a group's end inserts
    /// the shader composing it.
    fn primitive(&self) -> Option<Primitive> {
        match self {
            PaintOperation::Primitive(primitive) => Some(primitive.clone()),
            PaintOperation::EndGroup(shader) => Some(Primitive::Shader(shader.clone())),
            _ => None,
        }
    }
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    Shader(crate::PaintShader),
    BackdropBlur(BackdropBlur),
    NativeView(PaintNativeView),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
            Primitive::NativeView(_) => PrimitiveKind::NativeView,
            Primitive::Shader(_) => PrimitiveKind::Shader,
            Primitive::BackdropBlur(_) => PrimitiveKind::BackdropBlur,
        }
    }

    /// Where the primitive is ordered against what it overlaps: a backdrop
    /// blur reads past its bounds, so it is drawn after whatever was painted
    /// before it under any of what it reads.
    fn ordered_bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Primitive::BackdropBlur(blur) => blur.backdrop(),
            _ => self.clipped_bounds(),
        }
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::Shader(shader) => &shader.bounds,
            Primitive::BackdropBlur(blur) => &blur.bounds,
            Primitive::NativeView(native_view) => &native_view.bounds,
        }
    }

    fn transformation(&self) -> TransformationMatrix {
        match self {
            Primitive::MonochromeSprite(sprite) => sprite.transformation,
            Primitive::SubpixelSprite(sprite) => sprite.transformation,
            _ => TransformationMatrix::unit(),
        }
    }

    fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::Shader(shader) => &shader.content_mask,
            Primitive::BackdropBlur(blur) => &blur.content_mask,
            Primitive::NativeView(native_view) => &native_view.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    shaders_start: usize,
    shaders_iter: Peekable<slice::Iter<'a, crate::PaintShader>>,
    backdrop_blurs_start: usize,
    backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
    native_views_start: usize,
    native_views_iter: Peekable<slice::Iter<'a, PaintNativeView>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.paths_iter.peek().map(|path| path.order),
                PrimitiveKind::Path,
            ),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.shaders_iter.peek().map(|s| s.order),
                PrimitiveKind::Shader,
            ),
            (
                self.backdrop_blurs_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlur,
            ),
            (
                self.native_views_iter
                    .peek()
                    .map(|native_view| native_view.order),
                PrimitiveKind::NativeView,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::BackdropBlur => {
                let start = self.backdrop_blurs_start;
                let mut end = start + 1;
                self.backdrop_blurs_iter.next();
                while self
                    .backdrop_blurs_iter
                    .next_if(|blur| (blur.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    end += 1;
                }
                self.backdrop_blurs_start = end;
                Some(PrimitiveBatch::BackdropBlurs(start..end))
            }
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            // One batch per run of one program and image.
            PrimitiveKind::Shader => {
                let first = self.shaders_iter.peek().unwrap();
                let (program, texture, group) = (
                    first.program,
                    first.image.as_ref().map(|t| t.texture_id),
                    first.group,
                );
                let shaders_start = self.shaders_start;
                let mut shaders_end = shaders_start + 1;
                self.shaders_iter.next();
                // A layer group's shader is drawn alone, with its texture.
                while self
                    .shaders_iter
                    .next_if(|shader| {
                        (shader.order, batch_kind) < max_order_and_kind
                            && std::ptr::eq(shader.program, program)
                            && shader.image.as_ref().map(|t| t.texture_id) == texture
                            && group.is_none()
                            && shader.group.is_none()
                    })
                    .is_some()
                {
                    shaders_end += 1;
                }
                self.shaders_start = shaders_end;
                Some(PrimitiveBatch::Shaders {
                    program,
                    texture,
                    group,
                    range: shaders_start..shaders_end,
                })
            }
            PrimitiveKind::NativeView => {
                let native_views_start = self.native_views_start;
                let mut native_views_end = native_views_start + 1;
                self.native_views_iter.next();
                while self
                    .native_views_iter
                    .next_if(|native_view| (native_view.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    native_views_end += 1;
                }
                self.native_views_start = native_views_end;
                Some(PrimitiveBatch::NativeViews(
                    native_views_start..native_views_end,
                ))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    Shaders {
        program: &'static crate::ShaderProgram,
        texture: Option<AtlasTextureId>,
        /// The layer group the shader composes, as its image.
        group: Option<usize>,
        range: Range<usize>,
    },
    /// Backdrop blurs, each drawn in turn over what was drawn before it.
    BackdropBlurs(Range<usize>),
    NativeViews(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::Shaders { program, range, .. } => {
                format!("shaders ({}) of {}", range.len(), program.name)
            }
            Self::BackdropBlurs(range) => format!("backdrop blurs ({})", range.len()),
            Self::NativeViews(range) => format!("native views ({})", range.len()),
        }
    }
}

/// A region drawn as a blur of what was drawn under it before it, clipped
/// to its bounds, its rounded corners and its content mask: a frosted
/// backdrop for a bar or panel, drawn under the element's own background.
///
/// The blur reads past its bounds by three times its radius, so content
/// next to the region blurs into it. Inside a layer group it reads only what
/// the group painted. Renderers without backdrop blurs draw nothing for it.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BackdropBlur {
    /// Where the blur is drawn among the scene's primitives.
    pub order: DrawOrder,
    /// The region drawn.
    pub bounds: Bounds<ScaledPixels>,
    /// What the region is clipped to.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The region's corner radii.
    pub corner_radii: Corners<ScaledPixels>,
    /// The blur's standard deviation, as CSS's `blur()` takes it.
    pub radius: ScaledPixels,
    /// How much of the blur is drawn over the backdrop, from 0 to 1.
    pub opacity: f32,
}

impl BackdropBlur {
    /// How far past its bounds the blur reads.
    pub fn reach(&self) -> ScaledPixels {
        ScaledPixels((self.radius.0.max(0.) * 3.0).ceil())
    }

    /// The region the blur reads: its visible part and its reach around it.
    pub fn backdrop(&self) -> Bounds<ScaledPixels> {
        self.bounds
            .intersect(&self.content_mask.bounds)
            .dilate(self.reach())
    }
}

impl From<BackdropBlur> for Primitive {
    fn from(blur: BackdropBlur) -> Self {
        Primitive::BackdropBlur(blur)
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    pub inset: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// The bounds of `bounds` with its corners transformed.
    pub fn bounds_of(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        let [[a, b], [c, d]] = self.rotation_scale;
        let [tx, ty] = self.translation;
        let corners = [
            (bounds.origin.x.0, bounds.origin.y.0),
            (bounds.origin.x.0 + bounds.size.width.0, bounds.origin.y.0),
            (bounds.origin.x.0, bounds.origin.y.0 + bounds.size.height.0),
            (
                bounds.origin.x.0 + bounds.size.width.0,
                bounds.origin.y.0 + bounds.size.height.0,
            ),
        ]
        .map(|(x, y)| (a * x + b * y + tx, c * x + d * y + ty));
        let min_x = corners.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
        let max_x = corners
            .iter()
            .map(|p| p.0)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = corners.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let max_y = corners
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);
        Bounds {
            origin: Point::new(ScaledPixels(min_x), ScaledPixels(min_y)),
            size: Size {
                width: ScaledPixels(max_x - min_x),
                height: ScaledPixels(max_y - min_y),
            },
        }
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub animation: SpriteAnimationId,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

/// Refers to an animation of the scene a sprite is inserted into.
///
/// A single `u32` so it fits the sprite's existing padding and leaves the
/// instance layout unchanged on every backend; renderers ignore it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpriteAnimationId(u32);

impl SpriteAnimationId {
    /// No animation.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for SpriteAnimationId {
    fn default() -> Self {
        Self::NONE
    }
}

/// A [`TimedTransformation`] the window applies to a sprite right before
/// every present, so the scene can be presented again without being redrawn.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpriteAnimation {
    transformation: TimedTransformation,
    center: Point<Pixels>,
    scale_factor: f32,
    base_color: Hsla,
    sprite_index: usize,
}

impl SpriteAnimation {
    pub(crate) fn new(
        transformation: TimedTransformation,
        center: Point<Pixels>,
        scale_factor: f32,
    ) -> Self {
        Self {
            transformation,
            center,
            scale_factor,
            base_color: Hsla::default(),
            sprite_index: usize::MAX,
        }
    }
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

/// Where a platform view shows through the scene.
///
/// The view sits beneath the window's rendered content, so the renderer
/// erases everything painted earlier within these bounds, and everything
/// painted later covers the view. The view is therefore treated as opaque.
#[derive(Clone, Copy, Debug)]
pub struct PaintNativeView {
    /// The view's position among the scene's other primitives.
    pub order: DrawOrder,
    /// The view these bounds belong to.
    pub id: NativeViewId,
    /// The hitbox that decides whether a mouse position reaches the view.
    pub hitbox_id: HitboxId,
    /// Where the view is laid out, before clipping.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip applied to the view.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The color to show where the view has not caught up with these bounds,
    /// or a transparent color for none.
    pub background: Hsla,
}

impl From<PaintNativeView> for Primitive {
    fn from(native_view: PaintNativeView) -> Self {
        Primitive::NativeView(native_view)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct PaintedPath {
    order: DrawOrder,
    bounds: Bounds<ScaledPixels>,
    tiles: Rc<PathTiles>,
    paint: PathPaint,
}

#[expect(missing_docs)]
impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        let content_mask = content_mask.scale(scale_factor);
        let mut tiles = path.bin(scale_factor);
        tiles.retain_visible(content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            pad: 0,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles: Rc::new(tiles),
            paint,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        }
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            ..Default::default()
        }
    }

    fn underline(x: f32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    static UNDER: crate::ShaderProgram = crate::ShaderProgram {
        name: "under",
        msl: "",
        wgsl: "",
        prepare: None,
    };

    static OVER: crate::ShaderProgram = crate::ShaderProgram {
        name: "over",
        msl: "",
        wgsl: "",
        prepare: None,
    };

    fn shader(program: &'static crate::ShaderProgram, x: f32) -> crate::PaintShader {
        crate::PaintShader {
            order: 0,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            program,
            uniforms: std::sync::Arc::new(crate::ShaderUniforms::default()),
            image: None,
            group: None,
            group_inset: Default::default(),
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    /// Shaders overlapping are drawn in the order they were painted, whatever
    /// their programs: one sharing another's order would be sorted by program.
    #[test]
    fn overlapping_shaders_are_drawn_in_paint_order() {
        for (first, second) in [(&UNDER, &OVER), (&OVER, &UNDER)] {
            let mut scene = Scene::default();
            scene.insert_primitive(shader(first, 0.));
            scene.insert_primitive(shader(second, 5.));
            assert_eq!(
                batch_labels(&mut scene),
                [
                    format!("shaders (1) of {}", first.name),
                    format!("shaders (1) of {}", second.name),
                ]
            );
        }
    }

    /// A shader still shares the order of the quad it overlaps: shaders are
    /// drawn after quads within an order.
    #[test]
    fn a_shader_over_a_quad_shares_its_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(shader(&OVER, 5.));
        scene.finish();
        assert_eq!(scene.quads[0].order, scene.shaders[0].order);
    }

    #[test]
    fn a_native_view_in_a_layer_is_covered_by_what_the_layer_paints_after_it() {
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id: HitboxId::placeholder(),
            bounds: square(0.),
            content_mask: ContentMask { bounds: square(0.) },
            background: Hsla::default(),
        });
        scene.insert_primitive(quad(0.));
        // A quad beside the view shares nothing with it, so it isn't held back.
        scene.insert_primitive(quad(20.));
        scene.pop_layer();
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "native views (1)", "quads (2)"]
        );
    }

    #[test]
    fn a_native_view_painted_twice_is_kept_where_it_was_painted_first() {
        let native_view = |hitbox_id: HitboxId, x: f32| PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id,
            bounds: square(x),
            content_mask: ContentMask { bounds: square(x) },
            background: Hsla::default(),
        };
        let first_hitbox = HitboxId::placeholder();
        let second_hitbox = first_hitbox.next();
        let mut scene = Scene::default();
        // The quad can't be drawn with the underline it covers, so the first
        // occurrence, painted over both, is drawn after the second, which
        // overlaps nothing.
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(quad(0.));
        scene.insert_primitive(native_view(first_hitbox, 0.));
        scene.insert_primitive(native_view(second_hitbox, 20.));
        scene.finish();

        let shown = scene
            .native_views
            .iter()
            .map(|native_view| native_view.bounds)
            .collect::<Vec<_>>();
        assert_eq!(shown, [square(0.)]);
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }
}

#[cfg(test)]
mod layer_group_tests {
    use crate::{
        Bounds, ContentMask, Path, Pixels, Quad, ScaledPixels, Scene, ShaderProgram,
        ShaderUniforms, Size, point, px,
    };

    static PROGRAM: ShaderProgram = ShaderProgram {
        name: "test",
        msl: "",
        wgsl: "",
        prepare: None,
    };

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: Size {
                width: ScaledPixels(w),
                height: ScaledPixels(h),
            },
        }
    }

    fn quad(at: Bounds<ScaledPixels>) -> Quad {
        Quad {
            bounds: at,
            content_mask: ContentMask { bounds: at },
            ..Default::default()
        }
    }

    fn shader(at: Bounds<ScaledPixels>) -> crate::PaintShader {
        crate::PaintShader {
            order: 0,
            bounds: at,
            content_mask: ContentMask { bounds: at },
            program: &PROGRAM,
            uniforms: std::sync::Arc::new(ShaderUniforms::default()),
            image: None,
            group: None,
            group_inset: Default::default(),
        }
    }

    /// What is painted in a group, shaders included, is the group's: drawn
    /// into its texture, so it moves and is clipped with the group. The
    /// window's scene has only the shader composing it.
    #[test]
    fn a_group_holds_what_is_painted_in_it_shaders_too() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.insert_primitive(quad(bounds(110.0, 60.0, 20.0, 20.0)));
        scene.insert_primitive(shader(bounds(120.0, 70.0, 10.0, 10.0)));
        scene.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        scene.finish();
        assert!(scene.quads.is_empty());
        assert_eq!(scene.shaders.len(), 1);
        assert_eq!(scene.shaders[0].group, Some(0));
        let group = &scene.layer_groups[0];
        assert_eq!((group.quads.len(), group.shaders.len()), (1, 1));
    }

    fn stale(scene: &Scene, was_moving: &mut bool, redrawn: &[bool]) -> bool {
        scene.layer_group_stale(
            0,
            scheduler::Instant::now(),
            &Default::default(),
            was_moving,
            redrawn,
        )
    }

    /// A group is drawn again while a shader in it is live, once more after,
    /// and not at all while its shaders rest: a group that never goes away
    /// costs nothing between frames.
    #[test]
    fn a_group_is_drawn_again_while_its_shaders_move_and_once_after() {
        let mut scene = Scene::default();
        scene.push_group();
        let inner = shader(bounds(120.0, 70.0, 10.0, 10.0));
        let uniforms = inner.uniforms.clone();
        scene.insert_primitive(inner);
        scene.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        scene.finish();
        let mut was_moving = false;
        assert!(!stale(&scene, &mut was_moving, &[false]));
        uniforms.present_continuously();
        assert!(stale(&scene, &mut was_moving, &[false]));
        uniforms.present_until(None);
        assert!(stale(&scene, &mut was_moving, &[false]), "the landing");
        assert!(!stale(&scene, &mut was_moving, &[false]));
    }

    /// A group composing one drawn again this present is drawn again too.
    #[test]
    fn a_group_follows_a_group_nested_in_it() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.push_group();
        scene.insert_primitive(quad(bounds(10.0, 10.0, 5.0, 5.0)));
        scene.pop_group(shader(bounds(0.0, 0.0, 50.0, 50.0)));
        scene.pop_group(shader(bounds(0.0, 0.0, 100.0, 100.0)));
        scene.finish();
        let mut was_moving = false;
        assert!(!stale(&scene, &mut was_moving, &[false, false]));
        assert!(stale(&scene, &mut was_moving, &[false, true]));
    }

    /// A group with no paths draws into a texture its own size: its content
    /// moved to the texture's origin, the composing shader's tile the whole
    /// texture.
    #[test]
    fn a_group_without_paths_is_drawn_at_its_own_size() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.insert_primitive(quad(bounds(110.0, 60.0, 20.0, 20.0)));
        scene.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        scene.finish();
        assert_eq!(
            scene.layer_group_frames[0],
            Some(bounds(100.0, 50.0, 200.0, 100.0))
        );
        let moved = &scene.layer_groups[0].quads[0];
        assert_eq!(moved.bounds, bounds(10.0, 10.0, 20.0, 20.0));
        assert_eq!(moved.content_mask.bounds, bounds(10.0, 10.0, 20.0, 20.0));
        let tile = crate::ShaderInstance::of(&scene.shaders[0], scheduler::Instant::now()).tile;
        assert_eq!(tile, [0.0, 0.0, 200.0, 100.0]);
    }

    /// Its texture is only the part of it inside its mask, the tile reaching
    /// back to the bounds' origin; one too large for a device takes the
    /// window's size.
    #[test]
    fn a_group_is_drawn_at_its_visible_size() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.insert_primitive(quad(bounds(110.0, 60.0, 20.0, 20.0)));
        let mut compose = shader(bounds(100.0, -950.0, 200.0, 20_000.0));
        compose.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 800.0, 600.0),
        };
        scene.pop_group(compose);
        scene.push_group();
        scene.insert_primitive(quad(bounds(110.0, 60.0, 20.0, 20.0)));
        let mut huge = shader(bounds(0.0, 0.0, 20_000.0, 100.0));
        huge.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 10_000.0, 100.0),
        };
        scene.pop_group(huge);
        scene.finish();
        assert_eq!(
            scene.layer_group_frames[0],
            Some(bounds(100.0, 0.0, 200.0, 600.0))
        );
        assert_eq!(
            scene.layer_groups[0].quads[0].bounds,
            bounds(10.0, 60.0, 20.0, 20.0)
        );
        let tile = crate::ShaderInstance::of(&scene.shaders[0], scheduler::Instant::now()).tile;
        assert_eq!(tile, [0.0, -950.0, 200.0, 20_000.0]);
        assert_eq!(scene.layer_group_frames[1], None);
        let tile = crate::ShaderInstance::of(&scene.shaders[1], scheduler::Instant::now()).tile;
        assert_eq!(tile, [0.0, 0.0, 20_000.0, 100.0]);
    }

    /// A group at the window's size inside one at its own keeps its tile in
    /// window coordinates as the outer group's content moves.
    #[test]
    fn a_nested_group_with_a_path_keeps_its_tile() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.push_group();
        let mut builder = crate::PathBuilder::fill();
        builder.add_polygon(
            &[
                point(px(110.0), px(60.0)),
                point(px(150.0), px(60.0)),
                point(px(150.0), px(90.0)),
            ],
            true,
        );
        scene.insert_primitive(crate::PaintedPath::new(
            builder.build().unwrap(),
            1.0,
            ContentMask {
                bounds: crate::Bounds::new(
                    point(px(0.0), px(0.0)),
                    crate::size(px(1000.0), px(1000.0)),
                ),
            },
            crate::Hsla::default().into(),
        ));
        scene.pop_group(shader(bounds(105.0, 55.0, 60.0, 40.0)));
        scene.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        scene.finish();
        assert_eq!(scene.layer_group_frames[1], None);
        let inner = &scene.layer_groups[0].shaders[0];
        assert_eq!(
            inner.bounds.origin,
            point(ScaledPixels(5.0), ScaledPixels(5.0))
        );
        let tile = crate::ShaderInstance::of(inner, scheduler::Instant::now()).tile;
        assert_eq!(tile, [105.0, 55.0, 60.0, 40.0]);
    }

    /// One holding a path stays the window's size, its content where it was
    /// painted: a path's tiles are binned in window coordinates.
    #[test]
    fn a_group_with_a_path_is_drawn_at_the_windows_size() {
        let mut scene = Scene::default();
        scene.push_group();
        let mut builder = crate::PathBuilder::fill();
        builder.add_polygon(
            &[
                point(px(110.0), px(60.0)),
                point(px(150.0), px(60.0)),
                point(px(150.0), px(90.0)),
            ],
            true,
        );
        scene.insert_primitive(crate::PaintedPath::new(
            builder.build().unwrap(),
            1.0,
            ContentMask {
                bounds: crate::Bounds::new(
                    point(px(0.0), px(0.0)),
                    crate::size(px(1000.0), px(1000.0)),
                ),
            },
            crate::Hsla::default().into(),
        ));
        scene.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        scene.finish();
        assert_eq!(scene.layer_group_frames[0], None);
        let tile = crate::ShaderInstance::of(&scene.shaders[0], scheduler::Instant::now()).tile;
        assert_eq!(tile, [100.0, 50.0, 200.0, 100.0]);
    }

    /// A group inside another is composed in the outer group's scene, so the
    /// outer group carries it as it moves.
    #[test]
    fn a_nested_group_is_composed_inside_the_outer_one() {
        let mut scene = Scene::default();
        scene.push_group();
        scene.push_group();
        scene.insert_primitive(quad(bounds(10.0, 10.0, 5.0, 5.0)));
        scene.pop_group(shader(bounds(0.0, 0.0, 50.0, 50.0)));
        scene.pop_group(shader(bounds(0.0, 0.0, 100.0, 100.0)));
        scene.finish();
        assert_eq!(scene.layer_groups.len(), 2);
        assert_eq!(scene.shaders.len(), 1);
        assert_eq!(scene.shaders[0].group, Some(0));
        assert_eq!(scene.layer_groups[0].shaders[0].group, Some(1));
        assert_eq!(scene.layer_groups[1].quads.len(), 1);
    }

    /// A cached view's replay builds the same groups.
    #[test]
    fn a_replay_rebuilds_the_groups() {
        let mut first = Scene::default();
        first.push_group();
        first.insert_primitive(quad(bounds(110.0, 60.0, 20.0, 20.0)));
        first.insert_primitive(shader(bounds(120.0, 70.0, 10.0, 10.0)));
        first.pop_group(shader(bounds(100.0, 50.0, 200.0, 100.0)));
        let ops = first.len();
        let mut again = Scene::default();
        again.replay(0..ops, &first);
        again.finish();
        assert_eq!(again.layer_groups.len(), 1);
        assert_eq!(again.layer_groups[0].quads.len(), 1);
        assert_eq!(again.layer_groups[0].shaders.len(), 1);
        assert_eq!(again.shaders.len(), 1);
        assert_eq!(
            again.layer_group_frames[0],
            Some(bounds(100.0, 50.0, 200.0, 100.0))
        );
    }
}

#[cfg(test)]
mod painted_bounds_tests {
    use crate::{
        AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, ContentMask, MonochromeSprite, Point,
        Radians, ScaledPixels, Scene, Size, TileId, TransformationMatrix,
    };

    fn bounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: Size {
                width: ScaledPixels(w),
                height: ScaledPixels(h),
            },
        }
    }

    // A sprite turned a quarter about a point draws where its corners turn
    // to, and its layer's painted bounds hold that, within its mask.
    #[test]
    fn a_transformed_sprite_is_painted_where_it_is_turned() {
        let turn = TransformationMatrix::unit()
            .translate(Point::new(ScaledPixels(50.), ScaledPixels(50.)))
            .rotate(Radians(std::f32::consts::FRAC_PI_2))
            .translate(Point::new(ScaledPixels(-50.), ScaledPixels(-50.)));
        let drawn = turn.bounds_of(bounds(40., 0., 20., 10.));
        assert!((drawn.origin.x.0 - 90.).abs() < 1e-3, "{drawn:?}");
        assert!((drawn.origin.y.0 - 40.).abs() < 1e-3, "{drawn:?}");
        assert!((drawn.size.width.0 - 10.).abs() < 1e-3, "{drawn:?}");
        assert!((drawn.size.height.0 - 20.).abs() < 1e-3, "{drawn:?}");

        let mut scene = Scene::default();
        scene.insert_primitive(MonochromeSprite {
            bounds: bounds(40., 0., 20., 10.),
            content_mask: ContentMask {
                bounds: bounds(0., 0., 200., 200.),
            },
            transformation: turn,
            order: 0,
            animation: crate::SpriteAnimationId::NONE,
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(0),
                padding: 0,
                bounds: Bounds::default(),
            },
            color: crate::black(),
        });
        let painted = scene.painted_bounds().expect("the sprite");
        assert!(painted.origin.x.0 <= 89. && painted.origin.x.0 + painted.size.width.0 >= 100.);
        assert!(painted.origin.y.0 <= 40. && painted.origin.y.0 + painted.size.height.0 >= 60.);
    }
}
