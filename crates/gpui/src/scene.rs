// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

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

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
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
            set.note(&path.color.solid);
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
            by(&mut shadow.element_bounds);
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
        for path in &mut self.paths {
            by(&mut path.bounds);
            by(&mut path.content_mask.bounds);
            for vertex in &mut path.vertices {
                vertex.xy_position.x += dx;
                vertex.xy_position.y += dy;
                by(&mut vertex.content_mask.bounds);
            }
        }
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

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        // Painted inside a layer group: into its scene, recorded here so a
        // cached view replays it.
        if let Some(&group) = self.group_stack.last() {
            self.layer_groups[group].insert_primitive(primitive.clone());
            self.paint_operations
                .push(PaintOperation::Primitive(primitive));
            return;
        }
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        // A backdrop blur reads past its bounds, so it is drawn after
        // whatever was painted before it under any of what it reads.
        let ordered_bounds = match &primitive {
            Primitive::BackdropBlur(blur) => blur.backdrop(),
            _ => clipped_bounds,
        };
        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(ordered_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
            Primitive::Shader(shader) => {
                shader.order = order;
                self.shaders.push(shader.clone());
            }
            Primitive::BackdropBlur(blur) => {
                blur.order = order;
                self.backdrop_blurs.push(*blur);
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
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
        self.paint_operations
            .push(PaintOperation::EndGroup(shader.clone()));
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
        // Into the enclosing group's scene, or this one. Its operation is
        // already recorded, as the EndGroup's: not twice.
        let ops = self.paint_operations.len();
        self.insert_primitive(shader);
        self.paint_operations.truncate(ops);
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::StartGroup => {
                    self.push_group();
                }
                PaintOperation::EndGroup(shader) => self.pop_group(shader.clone()),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.texture_id.index));
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
        static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        self.generation = GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        if self.shaders.is_empty() {
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
        collect!(
            shadows => Shadow,
            quads => Quad,
            paths => Path,
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
            let paths = &mut layers[*layer].paths;
            let mut path = path.clone();
            path.id = PathId(paths.len());
            paths.push(path);
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
        }
    }
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
}

impl PrimitiveKind {
    const COUNT: usize = PrimitiveKind::Shader as usize + 1;
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// A layer group's start.
    StartGroup,
    /// A layer group's end, with the shader that composes it.
    EndGroup(crate::PaintShader),
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    Shader(crate::PaintShader),
    BackdropBlur(BackdropBlur),
}

#[expect(missing_docs)]
impl Primitive {
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
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::Shader(shader) => &shader.content_mask,
            Primitive::BackdropBlur(blur) => &blur.content_mask,
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
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
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
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
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
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
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
    pub pad: u32,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
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
        let mut path: Path<Pixels> = Path::new(point(px(110.0), px(60.0)));
        path.line_to(point(px(150.0), px(60.0)));
        path.line_to(point(px(150.0), px(90.0)));
        let mut path = path.scale(1.0);
        path.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 1000.0, 1000.0),
        };
        scene.insert_primitive(path);
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
    /// painted: paths are drawn through a texture the window's size.
    #[test]
    fn a_group_with_a_path_is_drawn_at_the_windows_size() {
        let mut scene = Scene::default();
        scene.push_group();
        let mut path: Path<Pixels> = Path::new(point(px(110.0), px(60.0)));
        path.line_to(point(px(150.0), px(60.0)));
        path.line_to(point(px(150.0), px(90.0)));
        let mut path = path.scale(1.0);
        path.content_mask = ContentMask {
            bounds: bounds(0.0, 0.0, 1000.0, 1000.0),
        };
        scene.insert_primitive(path);
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
