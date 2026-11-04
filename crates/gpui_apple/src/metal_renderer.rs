use crate::metal_atlas::MetalAtlas;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use gpui::{
    AtlasTextureId, Background, Bounds, ContentMask, DevicePixels, PaintSurface, Path, Point,
    PrimitiveBatch, ScaledPixels, Scene, Size, point, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_graphics::{CGColorSpace, kCGColorSpaceDisplayP3};
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use objc2_foundation::{NSRange, NSString};
#[cfg(target_os = "macos")]
use objc2_metal::MTLCopyAllDevices;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLDrawable,
    MTLGPUFamily, MTLLibrary, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResourceOptions, MTLStorageMode, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage, MTLViewport,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use objc2_metal::{MTLBlitCommandEncoder, MTLOrigin, MTLRegion, MTLSize};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::Mutex;

use std::{cell::Cell, mem, mem::MaybeUninit, ops::Range, ptr, slice, sync::Arc};

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub fn new_renderer(
    context: self::Context,
    transparent: bool,
    use_display_p3: bool,
) -> Result<Renderer> {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<InstanceBuffer>,
}

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: 2 * 1024 * 1024,
            buffers: Vec::new(),
        }
    }
}

struct InstanceBuffer {
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    size: usize,
}

// SAFETY: Each buffer has one CPU writer and returns to the mutex-protected pool only after GPU completion.
unsafe impl Send for InstanceBuffer {}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
    }

    fn acquire(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        unified_memory: bool,
    ) -> Result<InstanceBuffer> {
        if let Some(buffer) = self.buffers.pop() {
            return Ok(buffer);
        }
        let options = if unified_memory {
            MTLResourceOptions::StorageModeShared
                // Buffers are write only which can benefit from the combined cache
                // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                | MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            MTLResourceOptions::StorageModeManaged
        };
        let buffer = device
            .newBufferWithLength_options(self.buffer_size, options)
            .with_context(|| format!("allocating {}-byte instance buffer", self.buffer_size))?;
        Ok(InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        })
    }

    fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer)
        }
    }
}

enum DisplayLinkMode {
    Detached,
    Attached {
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    },
}

pub struct MetalRenderer {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    layer: Option<Retained<CAMetalLayer>>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    display_link_mode: DisplayLinkMode,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    paths_rasterization_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    path_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    shadows_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    quads_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    underlines_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    monochrome_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    polychrome_sprites_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    surfaces_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Each shader program's pipeline, by the program's address, compiled
    /// the first time it is drawn; `None` for one that would not compile,
    /// which is not tried again.
    shader_pipelines: std::cell::RefCell<std::collections::HashMap<usize, Option<Pipeline>>>,
    /// The last frame drawn as the layers its shaders are drawn between,
    /// which each present composes with the shaders.
    shader_layers: ShaderLayerTextures,
    /// Each layer group's texture, the frame they were drawn for, and
    /// whether each group moved when last drawn.
    group_textures: Vec<Option<Texture>>,
    group_generation: u64,
    group_moving: Vec<bool>,
    /// The dynamic colour table as the last present uploaded it, and the
    /// entries that changed at this present.
    dynamic_colors: Vec<f32>,
    dynamic_changed: gpui::DynamicColorSet,
    compose_pipelines: Option<ComposePipelines>,
    /// The last frame drawn with shaders, as another thread can present it
    /// again; `None` after a frame without.
    present_plan: Option<Arc<PresentPlan>>,
    /// What a shader with no image is given to sample.
    blank_texture: Option<Texture>,
    unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    path_intermediate_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    path_intermediate_msaa_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = CAMetalLayer::new();
        Self::configure_layer(&layer, &device, transparent, use_display_p3);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view, such as
    /// the backing layer UIKit creates for a view whose `layerClass` is
    /// `CAMetalLayer`. The renderer retains the layer.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &CAMetalLayer,
        transparent: bool,
    ) -> Result<Self> {
        let device = Self::create_device()?;
        let layer = layer.retain();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool)
    }

    fn configure_layer(
        layer: &CAMetalLayer,
        device: &ProtocolObject<dyn MTLDevice>,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.setDevice(Some(device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::with_name(Some(display_p3_name)) {
                layer.setColorspace(Some(&display_p3_color_space));
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.setOpaque(!transparent);
        layer.setMaximumDrawableCount(3);
        // Allow texture reading for visual tests and UI automation screenshots
        // (captures without ScreenCaptureKit). Only debug builds pay the
        // presentation cost, even when `test-support` is compiled in.
        #[cfg(all(feature = "test-support", debug_assertions))]
        layer.setFramebufferOnly(false);
        layer.setAllowsNextDrawableTimeout(false);
        layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Result<Self> {
        let device = Self::create_device()?;
        Self::new_internal(device, None, true, instance_buffer_pool)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(device) = MTLCopyAllDevices()
            .iter()
            .min_by_key(|device| (device.isRemovable(), !device.isLowPower()))
        {
            Ok(device)
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            Self::system_default_device()
        }
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        Self::system_default_device()
    }

    fn system_default_device() -> Result<Retained<ProtocolObject<dyn MTLDevice>>> {
        MTLCreateSystemDefaultDevice().context("accessing a compatible Metal device")
    }

    fn new_internal(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        layer: Option<Retained<CAMetalLayer>>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    ) -> Result<Self> {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(SHADERS_SOURCE_FILE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .context("building Metal library")?;
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .newLibraryWithData_error(&dispatch2::DispatchData::from_static_bytes(
                SHADERS_METALLIB,
            ))
            .context("loading Metal library")?;

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.hasUnifiedMemory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supportsFamily(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        // SAFETY: Metal copies the live array using its exact byte length.
        let unit_vertices = unsafe {
            device.newBufferWithBytes_length_options(
                ptr::NonNull::from(&unit_vertices).cast(),
                mem::size_of_val(&unit_vertices),
                if is_unified_memory {
                    MTLResourceOptions::StorageModeShared
                        | MTLResourceOptions::CPUCacheModeWriteCombined
                } else {
                    MTLResourceOptions::StorageModeManaged
                },
            )
        }
        .context("allocating unit vertex buffer")?;

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        )?;
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        )?;

        let command_queue = device.newCommandQueue().context("creating command queue")?;
        let supports_shared_storage = cfg!(target_os = "ios") || is_apple_gpu;
        #[expect(
            clippy::arc_with_non_send_sync,
            reason = "PlatformWindow::sprite_atlas requires Arc, not Send or Sync"
        )]
        let sprite_atlas = Arc::new(MetalAtlas::new(device.clone(), supports_shared_storage));
        let mut cache = ptr::null_mut();
        // SAFETY: The output pointer is writable and both optional attribute dictionaries are absent.
        let status = unsafe {
            CVMetalTextureCache::create(None, None, &device, None, ptr::NonNull::from(&mut cache))
        };
        anyhow::ensure!(
            status == kCVReturnSuccess,
            "creating CoreVideo Metal texture cache: status {status}"
        );
        let cache = ptr::NonNull::new(cache).context("retrieving CoreVideo Metal texture cache")?;
        // SAFETY: The successful create call transfers ownership of this non-null cache.
        let core_video_texture_cache = unsafe { CFRetained::from_raw(cache) };

        Ok(Self {
            device,
            layer,
            presents_with_transaction: false,
            display_link_mode: DisplayLinkMode::Detached,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            shader_pipelines: Default::default(),
            shader_layers: Default::default(),
            group_textures: Vec::new(),
            group_generation: 0,
            group_moving: Vec::new(),
            dynamic_colors: Vec::new(),
            dynamic_changed: Default::default(),
            compose_pipelines: None,
            present_plan: None,
            blank_texture: None,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        })
    }

    pub fn layer(&self) -> Option<&CAMetalLayer> {
        self.layer.as_deref()
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|layer| Retained::as_ptr(layer).cast_mut())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    /// Draw the next frame into `drawable` instead of asking the layer for one.
    ///
    /// A `CAMetalDisplayLink` hands out each frame's drawable early enough for
    /// the frame to reach the display on time; `nextDrawable` would instead
    /// wait for the compositor to release one. Pass `None` once the frame is
    /// done, so a drawable the frame did not use goes back to the layer.
    ///
    /// Returns the drawable this replaces: after a frame, `Some` means the
    /// frame neither drew nor presented.
    ///
    /// This is a setter rather than a guard scoped to one frame because the
    /// window releases its lock on the renderer while GPUI draws.
    pub fn set_frame_drawable(
        &mut self,
        drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    ) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable: current } => mem::replace(current, drawable),
            DisplayLinkMode::Detached => {
                // Only a live link hands out drawables, and a link is detached
                // only after it has been invalidated.
                debug_assert!(
                    drawable.is_none(),
                    "received a display link drawable while detached"
                );
                None
            }
        }
    }

    /// Record whether a `CAMetalDisplayLink` drives the layer. While one
    /// does, frames are drawn only into the drawables it hands out.
    pub fn set_display_link_attached(&mut self, attached: bool) {
        self.display_link_mode = if attached {
            DisplayLinkMode::Attached { drawable: None }
        } else {
            DisplayLinkMode::Detached
        };
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.setPresentsWithTransaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        if let Some(layer) = &self.layer {
            layer.setDrawableSize(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        if let Err(error) = self.update_path_intermediate_textures(size) {
            log::error!("failed to resize path textures: {error:#}");
        }
    }

    fn update_path_intermediate_textures(&mut self, size: Size<DevicePixels>) -> Result<()> {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            return Ok(());
        }

        if self
            .path_intermediate_texture
            .as_ref()
            .is_some_and(|texture| {
                texture.width() == size.width.0 as usize
                    && texture.height() == size.height.0 as usize
            })
        {
            return Ok(());
        }

        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setStorageMode(MTLStorageMode::Private);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let intermediate_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating path intermediate texture")?;

        let msaa_texture = if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                MTLStorageMode::Memoryless
            } else {
                MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.setTextureType(MTLTextureType::Type2DMultisample);
            msaa_descriptor.setStorageMode(storage_mode);
            // SAFETY: The multisample texture uses the same sample count as its render pipeline.
            unsafe {
                msaa_descriptor.setSampleCount(self.path_sample_count as _);
            }
            Some(
                self.device
                    .newTextureWithDescriptor(&msaa_descriptor)
                    .context("allocating multisample path texture")?,
            )
        } else {
            None
        };
        self.path_intermediate_texture = Some(intermediate_texture);
        self.path_intermediate_msaa_texture = msaa_texture;
        Ok(())
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        if let Some(layer) = &self.layer {
            layer.setOpaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    pub fn draw(&mut self, scene: &Scene) {
        // Whatever this draw leaves, the last frame's plan no longer shows
        // the window: a draw that fails presents nothing in its place.
        self.present_plan = None;
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return;
            }
        };
        let drawable = match &mut self.display_link_mode {
            DisplayLinkMode::Attached { drawable } => {
                let Some(drawable) = drawable.take() else {
                    log::error!("skipped a frame drawn outside its display link's update");
                    return;
                };
                drawable
            }
            DisplayLinkMode::Detached => {
                let Some(drawable) = layer.nextDrawable() else {
                    log::error!(
                        "failed to retrieve next drawable, drawable size: {:?}",
                        layer.drawableSize()
                    );
                    return;
                };
                drawable
            }
        };
        // A display link's drawable was sized when the link handed it out,
        // which can be before a resize the layer has since taken.
        let texture = drawable.texture();
        let viewport_size: Size<DevicePixels> = size(
            (texture.width() as i32).into(),
            (texture.height() as i32).into(),
        );

        self.refresh_dynamic_colors(scene);
        // Each layer group drawn into its texture first.
        if let Err(error) = self.draw_layer_groups(scene, viewport_size) {
            log::error!("failed to draw layer groups: {error:#}");
        }
        // A scene with shaders is drawn once as the layers they are drawn
        // between, and each present composes the layers with the shaders, so
        // re-presenting a moving shader costs the shaders and a few
        // full-screen copies, not the scene.
        let rendered = if scene.shader_layers.is_some() {
            self.compose_frame(scene, &texture, viewport_size)
        } else {
            self.shader_layers.release();
            self.render_frame(scene, &texture, viewport_size, false)
        };
        let command_buffer = match rendered {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return;
            }
        };

        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            drawable.present();
        } else {
            command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
            command_buffer.commit();
        }
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        transparent: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        if self.path_intermediate_texture.is_none() {
            self.update_path_intermediate_textures(viewport_size)?;
        }
        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        )?;
        let instance_bindings = write_instances(scene, &mut writer).with_context(|| {
            format!(
                "writing instance data: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let mut surface_textures = Vec::new();
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instance_bindings,
            &mut writer,
            texture,
            viewport_size,
            &mut surface_textures,
            transparent,
        )?;

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let frame_resources = Cell::new(Some((writer.finish(), surface_textures)));
        let block = RcBlock::new(
            move |buffer: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                note_gpu_time(buffer);
                if let Some((instance_buffer, _surface_textures)) = frame_resources.take() {
                    instance_buffer_pool.lock().release(instance_buffer);
                }
            },
        );
        // SAFETY: Metal copies the live block and invokes it once after this frame completes.
        // Its captures own the frame resources and use the pool mutex when returning the buffer.
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
        }

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawableSize();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        // CAMetalDisplayLink owns the drawable, so render into a texture we own.
        self.render_offscreen(scene, viewport_size)
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;
            self.render_offscreen(scene, size)
        })
    }

    /// Renders a scene into a new offscreen texture of `size` and reads it
    /// back, without presenting anything.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    fn render_offscreen(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_offscreen: {:?}", size);
        }
        self.refresh_dynamic_colors(scene);
        self.draw_layer_groups(scene, size)?;

        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        // Like the atlas, only Apple GPUs can create shared textures on macOS;
        // Intel Macs cannot, even with unified memory. iOS has no managed storage.
        let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
        texture_descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating offscreen render target")?;

        let command_buffer = self.render_frame(scene, &target_texture, size, false)?;

        // Managed textures require an explicit blit synchronize before the CPU
        // can read back the rendered data. Without this, get_bytes returns
        // stale zeros.
        if !uses_shared_storage {
            let blit = command_buffer
                .blitCommandEncoder()
                .context("creating blit command encoder")?;
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }

        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        read_texture_to_image(&target_texture)
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as usize
                    || texture.height() != size.height.0 as usize
            });
            if needs_new_target {
                let texture_descriptor = MTLTextureDescriptor::new();
                unsafe {
                    texture_descriptor.setWidth(size.width.0 as usize);
                    texture_descriptor.setHeight(size.height.0 as usize);
                }
                texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor
                    .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                texture_descriptor.setStorageMode(MTLStorageMode::Private);
                self.headless_render_target = Some(
                    self.device
                        .newTextureWithDescriptor(&texture_descriptor)
                        .context("allocating headless render target")?,
                );
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .context("retrieving headless render target")?;

            let command_buffer = self.render_frame(scene, &target_texture, size, false)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instance_bindings: &InstanceBindings,
        writer: &mut InstanceBufferWriter,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
        transparent: bool,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue
            .commandBuffer()
            .context("creating command buffer")?;
        let alpha = if self.opaque && !transparent { 1. } else { 0. };

        let mut command_encoder = new_command_encoder_for_texture(
            &command_buffer,
            texture,
            viewport_size,
            Some(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha,
            }),
        )?;

        for batch in scene.batches() {
            match batch {
                PrimitiveBatch::Shadows(range) => {
                    self.draw_shadows(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::Quads(range) => {
                    self.draw_quads(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    command_encoder.endEncoding();

                    let did_draw = self.draw_paths_to_intermediate(
                        paths,
                        writer,
                        &instance_bindings.dynamic_colors,
                        viewport_size,
                        &command_buffer,
                    )?;

                    command_encoder = new_command_encoder_for_texture(
                        &command_buffer,
                        texture,
                        viewport_size,
                        None,
                    )?;

                    if did_draw {
                        if let Err(error) = self.draw_paths_from_intermediate(
                            paths,
                            writer,
                            viewport_size,
                            &command_encoder,
                        ) {
                            command_encoder.endEncoding();
                            return Err(error);
                        }
                    }
                }
                PrimitiveBatch::Underlines(range) => {
                    self.draw_underlines(range, instance_bindings, viewport_size, &command_encoder)
                }
                PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                    .draw_monochrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                    ),
                PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                    .draw_polychrome_sprites(
                        texture_id,
                        range,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                    ),
                PrimitiveBatch::Shaders {
                    program,
                    texture,
                    group,
                    range,
                } => self.draw_shaders(
                    program,
                    texture,
                    group,
                    range,
                    &instance_bindings.shaders,
                    viewport_size,
                    &command_encoder,
                ),
                PrimitiveBatch::Surfaces(range) => {
                    if let Err(error) = self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        range.start,
                        instance_bindings,
                        viewport_size,
                        &command_encoder,
                        surface_textures,
                    ) {
                        command_encoder.endEncoding();
                        return Err(error);
                    }
                }
                PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
            }
        }

        command_encoder.endEncoding();

        Ok(command_buffer)
    }

    fn draw_paths_to_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        dynamic_colors: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instance_bindings = writer.write(&vertices)?;

        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(MTLClearColor {
            red: 0.,
            green: 0.,
            blue: 0.,
            alpha: 0.,
        });

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.setTexture(Some(msaa_texture));
            color_attachment.setResolveTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.setTexture(Some(intermediate_texture));
            color_attachment.setStoreAction(MTLStoreAction::Store);
        }

        let command_encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating path render command encoder")?;
        command_encoder.setRenderPipelineState(&self.paths_rasterization_pipeline_state);
        // SAFETY: The binding contains vertices.len() shader-compatible vertices at an aligned offset.
        // Metal retains the buffer, and writer.finish publishes CPU writes before submission.
        // The viewport bytes are copied from a live value into the matching shader slot.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                PathRasterizationInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&vertex_instance_bindings.buffer),
                vertex_instance_bindings.offset,
                PathRasterizationInputIndex::Vertices as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&dynamic_colors.buffer),
                dynamic_colors.offset,
                DYNAMIC_COLORS_INDEX,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount(
                MTLPrimitiveType::Triangle,
                0,
                vertices.len(),
            );
        }

        command_encoder.endEncoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.shadows_pipeline_state);
        // SAFETY: the table was written with the frame's instances and lives as long.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.dynamic_colors.buffer),
                instance_bindings.dynamic_colors.offset,
                DYNAMIC_COLORS_INDEX,
            );
        }
        // SAFETY: Six unit vertices and the scene's shadow range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                ShadowInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.shadows.buffer),
                instance_bindings.shadows.offset,
                ShadowInputIndex::Shadows as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                ShadowInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                shadows.len(),
                shadows.start,
            );
        }
    }

    /// Renders a scene as a window with shaders presents it, the layers
    /// first and the shaders composed over them, `presents` times, for
    /// checking against `render_scene_to_image`.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn compose_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
        presents: usize,
    ) -> Result<RgbaImage> {
        objc2::rc::autoreleasepool(|_| {
            self.update_path_intermediate_textures(size)?;
            self.refresh_dynamic_colors(scene);
            self.draw_layer_groups(scene, size)?;
            let texture_descriptor = MTLTextureDescriptor::new();
            // SAFETY: The dimensions are positive and this descriptor creates a 2D texture.
            unsafe {
                texture_descriptor.setWidth(size.width.0 as usize);
                texture_descriptor.setHeight(size.height.0 as usize);
            }
            texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor
                .setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.setStorageMode(if uses_shared_storage {
                MTLStorageMode::Shared
            } else {
                MTLStorageMode::Managed
            });
            let target_texture = self
                .device
                .newTextureWithDescriptor(&texture_descriptor)
                .context("allocating compose target")?;
            let mut last = None;
            for _ in 0..presents.max(1) {
                let command_buffer = self.compose_frame(scene, &target_texture, size)?;
                command_buffer.commit();
                last = Some(command_buffer);
            }
            let command_buffer = last.context("no present")?;
            command_buffer.waitUntilCompleted();
            if !uses_shared_storage {
                let command_buffer = self
                    .command_queue
                    .commandBuffer()
                    .context("creating command buffer")?;
                let blit = command_buffer
                    .blitCommandEncoder()
                    .context("creating blit command encoder")?;
                blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
                blit.endEncoding();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
            }
            read_texture_to_image(&target_texture)
        })
    }

    /// A texture the size of the frame, to draw a layer into and read back
    /// texel for texel.
    fn layer_texture(&self, width: usize, height: usize) -> Result<Texture> {
        let descriptor = MTLTextureDescriptor::new();
        // SAFETY: The dimensions are the drawable's and this descriptor creates a 2D texture.
        unsafe {
            descriptor.setWidth(width);
            descriptor.setHeight(height);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Private);
        self.device
            .newTextureWithDescriptor(&descriptor)
            .context("allocating a layer texture")
    }

    /// Composes the scene's layers and shaders into `texture`, drawing the
    /// layers first if this scene's are not drawn yet, with the shaders'
    /// uniforms as they stand when the present reaches the screen.
    fn compose_frame(
        &mut self,
        scene: &Scene,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let at = scene.present_time();
        let plan = scene.shader_layers.as_deref().context("no shader layers")?;
        let width = i32::from(viewport_size.width) as usize;
        let height = i32::from(viewport_size.height) as usize;
        if self.path_intermediate_texture.is_none() {
            self.update_path_intermediate_textures(viewport_size)?;
        }
        if self.blank_texture.is_none() {
            let descriptor = MTLTextureDescriptor::new();
            // SAFETY: a 1×1 2D texture.
            unsafe {
                descriptor.setWidth(1);
                descriptor.setHeight(1);
            }
            descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            descriptor.setStorageMode(MTLStorageMode::Private);
            self.blank_texture = Some(
                self.device
                    .newTextureWithDescriptor(&descriptor)
                    .context("allocating the blank texture")?,
            );
        }
        if self.compose_pipelines.is_none() {
            self.compose_pipelines = Some(ComposePipelines::new(&self.device)?);
        }
        let held = self.shader_layers.holds(scene.generation, width, height)
            && self.shader_layers.layers.len() == plan.layers.len();
        if !held {
            let mut pool = self.shader_layers.take(width, height);
            let mut drawn = Vec::with_capacity(plan.layers.len());
            for (k, layer) in plan.layers.iter().enumerate() {
                if layer.is_blank() {
                    drawn.push(None);
                    continue;
                }
                let target = match pool.pop() {
                    Some(target) => target,
                    None => self.layer_texture(width, height)?,
                };
                // The same queue runs these before the compose that reads them.
                self.render_frame(layer, &target, viewport_size, k > 0)?
                    .commit();
                drawn.push(Some(target));
            }
            self.shader_layers = ShaderLayerTextures {
                generation: scene.generation,
                width,
                height,
                layers: drawn,
            };
        } else {
            // A held layer drawn in a dynamic colour that changed is drawn
            // again; the rest are kept.
            for (k, layer) in plan.layers.iter().enumerate() {
                if !layer.dynamic_colors.intersects(&self.dynamic_changed) {
                    continue;
                }
                if let Some(target) = self.shader_layers.layers[k].clone() {
                    // After the last compose that read it: the same queue.
                    self.render_frame(layer, &target, viewport_size, k > 0)?
                        .commit();
                }
            }
        }
        let present_plan = Arc::new(self.make_present_plan(scene, viewport_size)?);
        let command_buffer = present_plan.encode(texture, at)?;
        self.present_plan = Some(present_plan);
        Ok(command_buffer)
    }

    /// What presenting `scene` again needs once its layers are drawn: the
    /// layer textures, each run of shaders with its pipeline and image, and
    /// what the layers were drawn with, to tell when they no longer hold.
    fn make_present_plan(
        &self,
        scene: &Scene,
        viewport_size: Size<DevicePixels>,
    ) -> Result<PresentPlan> {
        let plan = scene.shader_layers.as_deref().context("no shader layers")?;
        let pipelines = self
            .compose_pipelines
            .clone()
            .context("missing compose pipelines")?;
        let blank = self.blank_texture.clone().context("no blank texture")?;
        let mut groups = Vec::with_capacity(plan.groups.len());
        for (k, group) in plan.groups.iter().enumerate() {
            let mut runs = Vec::new();
            let mut start = group.start;
            while start < group.end {
                let head = &scene.shaders[start];
                let key = |s: &gpui::PaintShader| {
                    (
                        s.program as *const gpui::ShaderProgram,
                        s.image.as_ref().map(|t| t.texture_id),
                        s.group,
                    )
                };
                // A layer group's shader is drawn alone, with its texture.
                let end = scene.shaders[start..group.end]
                    .iter()
                    .position(|s| key(s) != key(head) || s.group.is_some())
                    .map_or(group.end, |n| (start + n).max(start + 1));
                if let Some(pipeline) = self.shader_pipeline(head.program) {
                    let image = head
                        .group
                        .and_then(|group| self.group_textures.get(group).cloned().flatten())
                        .or_else(|| {
                            head.image
                                .as_ref()
                                .and_then(|tile| self.sprite_atlas.metal_texture(tile.texture_id))
                        })
                        .unwrap_or_else(|| blank.clone());
                    runs.push(ShaderRun {
                        range: start..end,
                        pipeline,
                        image,
                    });
                }
                start = end;
            }
            groups.push(PlanGroup {
                runs,
                over: self.shader_layers.layers.get(k + 1).cloned().flatten(),
            });
        }
        let mut drawn_colors = gpui::DynamicColorSet::default();
        for layer in plan.layers.iter().chain(&scene.layer_groups) {
            drawn_colors.extend(&layer.dynamic_colors);
        }
        Ok(PresentPlan {
            viewport: viewport_size,
            device: self.device.clone(),
            queue: self.command_queue.clone(),
            unified_memory: self.is_unified_memory,
            opaque: self.opaque,
            with_transaction: self.presents_with_transaction,
            pool: self.instance_buffer_pool.clone(),
            pipelines,
            unit_vertices: self.unit_vertices.clone(),
            first: self.shader_layers.layers.first().cloned().flatten(),
            groups,
            shaders: scene.shaders.clone(),
            group_uniforms: scene
                .layer_groups
                .iter()
                .flat_map(|group| group.shaders.iter().map(|shader| shader.uniforms.clone()))
                .collect(),
            groups_moving: self.group_moving.iter().any(|moving| *moving),
            drawn_colors,
            colors: self.dynamic_colors.clone(),
        })
    }

    /// The last frame drawn with shaders, as another thread can present it
    /// again while only its shaders move. `None` after a frame without.
    pub fn present_plan(&self) -> Option<Arc<PresentPlan>> {
        self.present_plan.clone()
    }

    /// Fills the dynamic colour table for this present and notes which
    /// entries changed since the last.
    fn refresh_dynamic_colors(&mut self, scene: &Scene) {
        gpui::refresh_dynamic_colors(scene.present_time());
        let mut table = Vec::new();
        gpui::dynamic_colors_frame(&mut table);
        self.dynamic_changed = gpui::changed_dynamic_colors(&self.dynamic_colors, &table);
        self.dynamic_colors = table;
    }

    /// Each layer group's scene drawn into a texture of its own: the
    /// visible part of the shader composing it, its content moved to the
    /// texture's origin, or the window's size for one holding paths or too
    /// large. A group is drawn again when the frame is new or when
    /// `Scene::layer_group_stale` says so. Groups nested in others are drawn
    /// first, as their composing shader is in the enclosing group's scene.
    fn draw_layer_groups(
        &mut self,
        scene: &Scene,
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        if scene.layer_groups.is_empty() {
            self.group_textures.clear();
            self.group_moving.clear();
            return Ok(());
        }
        if self.path_intermediate_texture.is_none() {
            self.update_path_intermediate_textures(viewport_size)?;
        }
        let now = scene.present_time();
        let new_frame = self.group_generation != scene.generation;
        let count = scene.layer_groups.len();
        if new_frame {
            self.group_moving.clear();
        }
        self.group_moving.resize(count, false);
        let mut redrawn = vec![false; count];
        let sizes: Vec<Size<DevicePixels>> = scene
            .layer_group_frames
            .iter()
            .map(|frame| match frame {
                Some(frame) => size(
                    DevicePixels((frame.size.width.0.ceil() as i32).max(1)),
                    DevicePixels((frame.size.height.0.ceil() as i32).max(1)),
                ),
                None => viewport_size,
            })
            .collect();
        let mut held = std::mem::take(&mut self.group_textures);
        held.resize(count, None);
        for index in (0..count).rev() {
            let group = &scene.layer_groups[index];
            if group.is_blank() {
                held[index] = None;
                continue;
            }
            let extent = sizes[index];
            let width = i32::from(extent.width) as usize;
            let height = i32::from(extent.height) as usize;
            let fits = held[index]
                .as_ref()
                .is_some_and(|t| t.width() == width && t.height() == height);
            let stale = scene.layer_group_stale(
                index,
                now,
                &self.dynamic_changed,
                &mut self.group_moving[index],
                &redrawn,
            );
            if fits && !new_frame && !stale {
                continue;
            }
            redrawn[index] = true;
            let target = match held[index].take().filter(|_| fits) {
                Some(target) => target,
                None => self.layer_texture(width, height)?,
            };
            // Drawn with the groups so far in place, which a group nested in
            // this one is (higher indices first). The same queue runs these
            // before whatever samples them.
            self.group_textures = held.clone();
            self.render_frame(group, &target, extent, true)?.commit();
            held[index] = Some(target);
        }
        self.group_textures = held;
        self.group_generation = scene.generation;
        Ok(())
    }

    /// A program's pipeline, compiled from its source the first time it is
    /// drawn: the app's `shade` between a prelude declaring the instance and
    /// a vertex and fragment function calling it.
    fn shader_pipeline(&self, program: &'static gpui::ShaderProgram) -> Option<Pipeline> {
        let mut pipelines = self.shader_pipelines.borrow_mut();
        pipelines
            .entry(program as *const gpui::ShaderProgram as usize)
            .or_insert_with(|| {
                let source = format!("{SHADER_PRELUDE}\n{}\n{SHADER_MAIN}", program.msl);
                let library = match self.device.newLibraryWithSource_options_error(
                    &NSString::from_str(&source),
                    Some(&objc2_metal::MTLCompileOptions::new()),
                ) {
                    Ok(library) => library,
                    Err(error) => {
                        log::error!(
                            "shader {} did not compile: {}",
                            program.name,
                            error.localizedDescription()
                        );
                        return None;
                    }
                };
                build_pipeline_state(
                    &self.device,
                    &library,
                    program.name,
                    "shader_vertex",
                    "shader_fragment",
                    MTLPixelFormat::BGRA8Unorm,
                )
                .inspect_err(|error| log::error!("shader {}: {error:#}", program.name))
                .ok()
            })
            .clone()
    }

    fn draw_shaders(
        &self,
        program: &'static gpui::ShaderProgram,
        texture: Option<AtlasTextureId>,
        group: Option<usize>,
        shaders: Range<usize>,
        binding: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if shaders.is_empty() {
            return;
        }
        let Some(pipeline) = self.shader_pipeline(program) else {
            return;
        };
        command_encoder.setRenderPipelineState(&pipeline);
        let image = group
            .and_then(|group| self.group_textures.get(group).cloned().flatten())
            .or_else(|| texture.and_then(|id| self.sprite_atlas.metal_texture(id)))
            .or_else(|| self.blank_texture.clone());
        let viewport = [
            i32::from(viewport_size.width) as f32,
            i32::from(viewport_size.height) as f32,
        ];
        // SAFETY: six unit vertices and the shader range match the instances
        // written for this frame; the viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(Some(&self.unit_vertices), 0, 0);
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&binding.buffer),
                binding.offset,
                1,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&binding.buffer),
                binding.offset,
                1,
            );
            command_encoder.setFragmentTexture_atIndex(image.as_deref(), 0);
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport).cast(),
                mem::size_of_val(&viewport),
                2,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                shaders.len(),
                shaders.start,
            );
        }
    }

    fn draw_quads(
        &self,
        quads: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if quads.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.quads_pipeline_state);
        // SAFETY: the table was written with the frame's instances and lives as long.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.dynamic_colors.buffer),
                instance_bindings.dynamic_colors.offset,
                DYNAMIC_COLORS_INDEX,
            );
        }
        // SAFETY: Six unit vertices and the scene's quad range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                QuadInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.quads.buffer),
                instance_bindings.quads.offset,
                QuadInputIndex::Quads as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                QuadInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                quads.len(),
                quads.start,
            );
        }
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) -> Result<()> {
        let (Some(first_path), Some(last_path)) = (paths.first(), paths.last()) else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.setRenderPipelineState(&self.path_sprites_pipeline_state);
        // SAFETY: The shader slots match the unit vertices, copied viewport, and retained path texture.
        // The preceding render pass writes this tracked texture before it is sampled.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(intermediate_texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
        }

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if last_path.order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite { bounds }];
        }

        let sprite_instance_bindings = writer.write(&sprites)?;
        // SAFETY: The draw uses six unit vertices and exactly the sprites written at this aligned binding.
        // Metal retains the buffer, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&sprite_instance_bindings.buffer),
                sprite_instance_bindings.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
            );
        }
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.setRenderPipelineState(&self.underlines_pipeline_state);
        // SAFETY: the table was written with the frame's instances and lives as long.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.dynamic_colors.buffer),
                instance_bindings.dynamic_colors.offset,
                DYNAMIC_COLORS_INDEX,
            );
        }
        // SAFETY: Six unit vertices and the scene's underline range match the uploaded shader layouts and slots.
        // Metal retains the buffers, writer.finish publishes their writes, and viewport bytes are copied.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                UnderlineInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.underlines.buffer),
                instance_bindings.underlines.offset,
                UnderlineInputIndex::Underlines as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                UnderlineInputIndex::ViewportSize as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                underlines.len(),
                underlines.start,
            );
        }
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.monochrome_sprites_pipeline_state);
        // SAFETY: the table was written with the frame's instances and lives as long.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.dynamic_colors.buffer),
                instance_bindings.dynamic_colors.offset,
                DYNAMIC_COLORS_INDEX,
            );
        }
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.monochrome_sprites.buffer),
                instance_bindings.monochrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.setRenderPipelineState(&self.polychrome_sprites_pipeline_state);
        // SAFETY: The scene's sprite range, six unit vertices, and copied sizes match the shader layouts and slots.
        // Metal retains the buffers and texture, and writer.finish publishes the instance writes.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SpriteInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SpriteInputIndex::ViewportSize as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&texture_size).cast(),
                mem::size_of_val(&texture_size),
                SpriteInputIndex::AtlasTextureSize as usize,
            );
            command_encoder.setFragmentBuffer_offset_atIndex(
                Some(&instance_bindings.polychrome_sprites.buffer),
                instance_bindings.polychrome_sprites.offset,
                SpriteInputIndex::Sprites as usize,
            );
            command_encoder.setFragmentTexture_atIndex(
                Some(&texture),
                SpriteInputIndex::AtlasTexture as usize,
            );
            command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::Triangle,
                0,
                6,
                sprites.len(),
                sprites.start,
            );
        }
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instance_bindings: &InstanceBindings,
        viewport_size: Size<DevicePixels>,
        command_encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }

        command_encoder.setRenderPipelineState(&self.surfaces_pipeline_state);
        // SAFETY: The unit vertices, SurfaceBounds instances, and copied viewport match the shader layouts and slots.
        // Metal retains the buffers, and writer.finish publishes CPU writes before submission.
        unsafe {
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&self.unit_vertices),
                0,
                SurfaceInputIndex::Vertices as usize,
            );
            command_encoder.setVertexBuffer_offset_atIndex(
                Some(&instance_bindings.surfaces.buffer),
                instance_bindings.surfaces.offset,
                SurfaceInputIndex::Surfaces as usize,
            );
            command_encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                SurfaceInputIndex::ViewportSize as usize,
            );
        }

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            let pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
            anyhow::ensure!(
                pixel_format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                "unsupported surface pixel format: {pixel_format:#x}"
            );

            let plane_texture = |plane, format| -> Result<CFRetained<CVMetalTexture>> {
                let width = CVPixelBufferGetWidthOfPlane(image_buffer, plane);
                let height = CVPixelBufferGetHeightOfPlane(image_buffer, plane);
                let mut texture = ptr::null_mut();
                // SAFETY: The buffer has two planes matching these formats and dimensions, and the output pointer is writable.
                let status = unsafe {
                    CVMetalTextureCache::create_texture_from_image(
                        None,
                        &self.core_video_texture_cache,
                        image_buffer,
                        None,
                        format,
                        width,
                        height,
                        plane,
                        ptr::NonNull::from(&mut texture),
                    )
                };
                anyhow::ensure!(
                    status == kCVReturnSuccess,
                    "creating CoreVideo Metal texture for plane {plane}: status {status}"
                );
                let texture =
                    ptr::NonNull::new(texture).context("retrieving CoreVideo Metal texture")?;
                // SAFETY: The successful create call transfers ownership of this non-null texture.
                Ok(unsafe { CFRetained::from_raw(texture) })
            };
            let y_texture = plane_texture(0, MTLPixelFormat::R8Unorm)?;
            let cb_cr_texture = plane_texture(1, MTLPixelFormat::RG8Unorm)?;
            let y_metal_texture =
                CVMetalTextureGetTexture(&y_texture).context("retrieving luma Metal texture")?;
            let cb_cr_metal_texture = CVMetalTextureGetTexture(&cb_cr_texture)
                .context("retrieving chroma Metal texture")?;

            // SAFETY: The size is copied from a live value, and both planes match their shader texture slots.
            // The frame completion handler retains the CoreVideo wrappers until GPU reads finish.
            unsafe {
                command_encoder.setVertexBytes_length_atIndex(
                    ptr::NonNull::from(&texture_size).cast(),
                    mem::size_of_val(&texture_size),
                    SurfaceInputIndex::TextureSize as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&y_metal_texture),
                    SurfaceInputIndex::YTexture as usize,
                );
                command_encoder.setFragmentTexture_atIndex(
                    Some(&cb_cr_metal_texture),
                    SurfaceInputIndex::CbCrTexture as usize,
                );
            }
            surface_textures.extend([y_texture, cb_cr_texture]);

            // SAFETY: Six unit vertices draw one instance within the uploaded scene's surface range.
            unsafe {
                command_encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    MTLPrimitiveType::Triangle,
                    0,
                    6,
                    1,
                    first_surface + index,
                );
            }
        }
        Ok(())
    }
}

fn new_command_encoder_for_texture(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<MTLClearColor>,
) -> Result<Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>> {
    let render_pass_descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe {
        render_pass_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
    };
    color_attachment.setTexture(Some(texture));
    color_attachment.setStoreAction(MTLStoreAction::Store);
    if let Some(clear_color) = clear_color {
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setClearColor(clear_color);
    } else {
        color_attachment.setLoadAction(MTLLoadAction::Load);
    }

    let command_encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
        .context("creating render command encoder")?;
    command_encoder.setViewport(MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    Ok(command_encoder)
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &ProtocolObject<dyn MTLTexture>) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    unsafe {
        texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            ptr::NonNull::from(pixels.as_mut_slice()).cast(),
            bytes_per_row,
            region,
            0,
        );
    }

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("creating RGBA image from pixel data")
}

fn build_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_path_sprite_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn build_path_rasterization_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: MTLPixelFormat,
    path_sample_count: u32,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vertex_fn = library
        .newFunctionWithName(&NSString::from_str(vertex_fn_name))
        .with_context(|| format!("locating vertex function {vertex_fn_name}"))?;
    let fragment_fn = library
        .newFunctionWithName(&NSString::from_str(fragment_fn_name))
        .with_context(|| format!("locating fragment function {fragment_fn_name}"))?;

    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex_fn.as_ref()));
    descriptor.setFragmentFunction(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.setRasterSampleCount(path_sample_count as _);
        descriptor.setAlphaToCoverageEnabled(false);
    }
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(pixel_format);
    color_attachment.setBlendingEnabled(true);
    color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
    color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
    color_attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
    color_attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
    color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);

    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

#[derive(Clone)]
struct InstanceBinding {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: usize,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
    surfaces: InstanceBinding,
    shaders: InstanceBinding,
    dynamic_colors: InstanceBinding,
}

/// The buffer index the stock shaders read the dynamic colour table from.
const DYNAMIC_COLORS_INDEX: usize = 20;

/// A finished command buffer's time on the GPU, for `gpui::take_gpu_times`,
/// while a measurement keeps them.
fn note_gpu_time(buffer: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>) {
    if !gpui::gpu_timing_on() {
        return;
    }
    // SAFETY: Metal hands the completed handler its live command buffer,
    // an MTLCommandBuffer, which answers both selectors with a time.
    let (start, end): (f64, f64) = unsafe {
        let buffer = buffer.as_ref();
        (
            objc2::msg_send![buffer, GPUStartTime],
            objc2::msg_send![buffer, GPUEndTime],
        )
    };
    if end > start {
        gpui::record_gpu_time(end - start);
    }
}

/// The Metal objects the shader-quad code holds.
type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;

/// A frame drawn with shaders, presented again from any thread: the layers
/// its shaders are drawn between, each run of shaders with its pipeline and
/// image, and the shaders, whose uniforms each present reads as they stand.
///
/// A plan shows the frame only while what its layers were drawn with holds:
/// no layer group moving and no dynamic colour drawn into a layer changed.
/// Otherwise the window's thread draws the frame again, making a new plan.
pub struct PresentPlan {
    viewport: Size<DevicePixels>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    unified_memory: bool,
    opaque: bool,
    with_transaction: bool,
    pool: Arc<Mutex<InstanceBufferPool>>,
    pipelines: ComposePipelines,
    unit_vertices: Retained<ProtocolObject<dyn MTLBuffer>>,
    first: Option<Texture>,
    groups: Vec<PlanGroup>,
    shaders: Vec<gpui::PaintShader>,
    /// The uniforms of the shaders inside layer groups, whose textures only
    /// the window's thread draws.
    group_uniforms: Vec<Arc<gpui::ShaderUniforms>>,
    /// Whether a layer group moved at the frame, so the present landing it
    /// is still to be drawn.
    groups_moving: bool,
    /// The dynamic colours the layers and layer groups are drawn in, and the
    /// table as they were drawn with it.
    drawn_colors: gpui::DynamicColorSet,
    colors: Vec<f32>,
}

// SAFETY: Metal's devices, queues, buffers, textures and pipeline states may
// be used from any thread, the instance buffer pool is behind its lock, and
// the shaders' uniforms are atomics read without waiting on a writer.
unsafe impl Send for PresentPlan {}
// SAFETY: as for `Send`; nothing in a plan changes once it is made.
unsafe impl Sync for PresentPlan {}

struct PlanGroup {
    runs: Vec<ShaderRun>,
    /// The layer drawn over the group's shaders.
    over: Option<Texture>,
}

struct ShaderRun {
    range: Range<usize>,
    pipeline: Pipeline,
    image: Texture,
}

impl PresentPlan {
    /// Whether presenting the plan at `at` shows something new: a shader on
    /// it moves, changed, or has its landing present due.
    pub fn moving(&self, at: std::time::Instant) -> bool {
        self.shaders
            .iter()
            .any(|shader| shader.uniforms.is_live(at))
    }

    /// Whether the plan alone can show the present at `at` in a drawable of
    /// `width` by `height`: the layers it holds are still the frame's. Runs
    /// the app's colour source for that present when the layers are drawn
    /// in dynamic colours.
    pub fn holds(&self, width: usize, height: usize, at: std::time::Instant) -> bool {
        if self.with_transaction
            || width != i32::from(self.viewport.width) as usize
            || height != i32::from(self.viewport.height) as usize
            || self.groups_moving
            || self
                .group_uniforms
                .iter()
                .any(|uniforms| uniforms.is_live(at))
        {
            return false;
        }
        if self.drawn_colors.is_empty() {
            return true;
        }
        gpui::refresh_dynamic_colors(at);
        let mut table = Vec::new();
        gpui::dynamic_colors_frame(&mut table);
        !gpui::changed_dynamic_colors(&self.colors, &table).intersects(&self.drawn_colors)
    }

    /// Presents the plan into `drawable` for the present at `at`, from any
    /// thread, if the plan holds for it. Whether it did; a drawable not
    /// presented is the caller's to use.
    pub fn present(
        &self,
        drawable: &ProtocolObject<dyn CAMetalDrawable>,
        at: std::time::Instant,
    ) -> bool {
        let texture = drawable.texture();
        if !self.holds(texture.width(), texture.height(), at) {
            return false;
        }
        match self.encode_taken(&texture, at) {
            Ok(command_buffer) => {
                command_buffer.presentDrawable(ProtocolObject::from_ref(drawable));
                command_buffer.commit();
                true
            }
            Err(error) => {
                log::error!("failed to present shaders: {error:#}");
                false
            }
        }
    }

    /// As `encode`, taking each shader's change as shown, as the window's
    /// thread takes it when it presents.
    fn encode_taken(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        at: std::time::Instant,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        for shader in &self.shaders {
            shader.uniforms.take_present(at);
        }
        self.encode(texture, at)
    }

    /// The layers and the shaders composed into `texture`, the shaders'
    /// uniforms as they stand when the present at `at` reaches the screen.
    fn encode(
        &self,
        texture: &ProtocolObject<dyn MTLTexture>,
        at: std::time::Instant,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let width = i32::from(self.viewport.width) as f64;
        let height = i32::from(self.viewport.height) as f64;
        let mut writer = InstanceBufferWriter::new(&self.device, &self.pool, self.unified_memory)?;
        let shaders = writer.write_iter(
            self.shaders
                .iter()
                .map(|shader| gpui::ShaderInstance::of(shader, at)),
        )?;
        let command_buffer = self
            .queue
            .commandBuffer()
            .context("creating command buffer")?;
        let render_pass_descriptor = MTLRenderPassDescriptor::new();
        // SAFETY: Index zero is within the render pass's color attachment array.
        let color_attachment = unsafe {
            render_pass_descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
        };
        color_attachment.setTexture(Some(texture));
        color_attachment.setStoreAction(MTLStoreAction::Store);
        if self.first.is_some() {
            color_attachment.setLoadAction(MTLLoadAction::DontCare);
        } else {
            let alpha = if self.opaque { 1. } else { 0. };
            color_attachment.setLoadAction(MTLLoadAction::Clear);
            color_attachment.setClearColor(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha,
            });
        }
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&render_pass_descriptor)
            .context("creating render command encoder")?;
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width,
            height,
            znear: 0.0,
            zfar: 1.0,
        });
        let layer = |pipeline: &Pipeline, texture: &Texture| {
            encoder.setRenderPipelineState(pipeline);
            // SAFETY: the compose program reads texture 0 texel for texel and
            // draws one triangle covering the target.
            unsafe {
                encoder.setFragmentTexture_atIndex(Some(texture), 0);
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
        };
        if let Some(first) = &self.first {
            layer(&self.pipelines.copy, first);
        }
        let viewport = [width as f32, height as f32];
        for group in &self.groups {
            for run in &group.runs {
                encoder.setRenderPipelineState(&run.pipeline);
                // SAFETY: six unit vertices and the run's range match the
                // instances written above; the viewport bytes are copied.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(Some(&self.unit_vertices), 0, 0);
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(&shaders.buffer),
                        shaders.offset,
                        1,
                    );
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(&shaders.buffer),
                        shaders.offset,
                        1,
                    );
                    encoder.setFragmentTexture_atIndex(Some(&run.image), 0);
                    encoder.setVertexBytes_length_atIndex(
                        ptr::NonNull::from(&viewport).cast(),
                        mem::size_of_val(&viewport),
                        2,
                    );
                    encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        MTLPrimitiveType::Triangle,
                        0,
                        6,
                        run.range.len(),
                        run.range.start,
                    );
                }
            }
            if let Some(over) = &group.over {
                layer(&self.pipelines.over, over);
            }
        }
        encoder.endEncoding();

        let pool = self.pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let block = RcBlock::new(
            move |buffer: ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                note_gpu_time(buffer);
                if let Some(instance_buffer) = instance_buffer.take() {
                    pool.lock().release(instance_buffer);
                }
            },
        );
        // SAFETY: as in `render_frame`.
        unsafe {
            command_buffer.addCompletedHandler(RcBlock::as_ptr(&block));
        }
        Ok(command_buffer)
    }
}

/// The layers of the last frame drawn with shaders, `None` for a blank one,
/// kept while presents compose them.
#[derive(Default)]
struct ShaderLayerTextures {
    generation: u64,
    width: usize,
    height: usize,
    layers: Vec<Option<Texture>>,
}

impl ShaderLayerTextures {
    fn holds(&self, generation: u64, width: usize, height: usize) -> bool {
        self.generation == generation
            && self.width == width
            && self.height == height
            && !self.layers.is_empty()
    }

    /// The textures, to be drawn again, where they are the size wanted.
    fn take(&mut self, width: usize, height: usize) -> Vec<Texture> {
        let same = self.width == width && self.height == height;
        let layers = std::mem::take(&mut self.layers);
        if same {
            layers.into_iter().flatten().collect()
        } else {
            Vec::new()
        }
    }

    fn release(&mut self) {
        *self = Self::default();
    }
}

/// Draws a layer into the drawable as one triangle covering it, read texel
/// for texel: unblended for the first, and over what is there for the rest,
/// whose colours are premultiplied, as drawing over a clear layer leaves
/// them.
const COMPOSE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

vertex float4 compose_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * 2.0 - 1.0, 0.0, 1.0);
}

fragment float4 compose_fragment(
    float4 position [[position]],
    texture2d<float, access::read> layer [[texture(0)]]
) {
    return layer.read(uint2(position.xy));
}
"#;

#[derive(Clone)]
struct ComposePipelines {
    copy: Pipeline,
    over: Pipeline,
}

impl ComposePipelines {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(COMPOSE_SOURCE),
                Some(&objc2_metal::MTLCompileOptions::new()),
            )
            .map_err(|error| anyhow::anyhow!("compose shader: {}", error.localizedDescription()))?;
        let vertex = library
            .newFunctionWithName(&NSString::from_str("compose_vertex"))
            .context("locating compose_vertex")?;
        let fragment = library
            .newFunctionWithName(&NSString::from_str("compose_fragment"))
            .context("locating compose_fragment")?;
        let pipeline = |over: bool| {
            let descriptor = MTLRenderPipelineDescriptor::new();
            descriptor.setLabel(Some(&NSString::from_str(if over {
                "compose_over"
            } else {
                "compose_copy"
            })));
            descriptor.setVertexFunction(Some(vertex.as_ref()));
            descriptor.setFragmentFunction(Some(fragment.as_ref()));
            // SAFETY: Index zero is within the pipeline's color attachment array.
            let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
            color.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            color.setBlendingEnabled(over);
            if over {
                color.setRgbBlendOperation(MTLBlendOperation::Add);
                color.setAlphaBlendOperation(MTLBlendOperation::Add);
                color.setSourceRGBBlendFactor(MTLBlendFactor::One);
                color.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                color.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                color.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            }
            device
                .newRenderPipelineStateWithDescriptor_error(&descriptor)
                .map_err(|error| {
                    anyhow::anyhow!("compose pipeline: {}", error.localizedDescription())
                })
        };
        Ok(Self {
            copy: pipeline(false)?,
            over: pipeline(true)?,
        })
    }
}

/// What a program's `shade` is compiled between: the instance, the image
/// sampler, and the vertex and fragment functions calling `shade`.
const SHADER_PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ShaderInstance {
    float4 bounds;
    float4 clip;
    float4 uniforms[20];
    float4 time;
    float4 tile;
};

// The quad's image at `uv`, 0 to 1 across it, filtered and kept off the
// texels around its tile in the atlas. Past the texture's edge it is clear,
// as a layer group's texture holds only its visible part.
float4 sample_image(texture2d<float> image, float4 tile, float2 uv) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float2 extent = float2(image.get_width(), image.get_height());
    float2 raw = tile.xy + uv * tile.zw;
    if (any(raw < 0.0) || any(raw > extent)) {
        return float4(0.0);
    }
    float2 at = clamp(raw, tile.xy + 0.5, tile.xy + tile.zw - 0.5);
    return image.sample(linear, at / extent);
}

struct ShaderVarying {
    float4 position [[position]];
    float2 local;
    uint instance [[flat]];
};
"#;

const SHADER_MAIN: &str = r#"
vertex ShaderVarying shader_vertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    constant float2 *unit [[buffer(0)]],
    constant ShaderInstance *instances [[buffer(1)]],
    constant float2 *viewport [[buffer(2)]]
) {
    ShaderInstance s = instances[instance_id];
    float2 u = unit[vertex_id];
    float2 p = s.bounds.xy + u * s.bounds.zw;
    float2 ndc = p / *viewport * float2(2.0, -2.0) + float2(-1.0, 1.0);
    ShaderVarying out;
    out.position = float4(ndc, 0.0, 1.0);
    out.local = u * s.bounds.zw;
    out.instance = instance_id;
    return out;
}

fragment float4 shader_fragment(
    ShaderVarying in [[stage_in]],
    constant ShaderInstance *instances [[buffer(1)]],
    texture2d<float> image [[texture(0)]]
) {
    constant ShaderInstance &s = instances[in.instance];
    float2 p = in.position.xy;
    if (any(p < s.clip.xy) || any(p > s.clip.xy + s.clip.zw)) {
        discard_fragment();
    }
    return shade(in.local, s.bounds.zw, s.time.x, (constant float *)s.uniforms, image, s.tile);
}
"#;

fn write_instances(scene: &Scene, writer: &mut InstanceBufferWriter) -> Result<InstanceBindings> {
    Ok(InstanceBindings {
        quads: writer.write(&scene.quads)?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
        }))?,
        // The dynamic colours as they stand for this present.
        dynamic_colors: {
            thread_local! {
                static TABLE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            TABLE.with(|table| {
                let mut table = table.borrow_mut();
                gpui::dynamic_colors_frame(&mut table);
                writer.write(&table)
            })?
        },
        // The shaders' uniforms as they are for this present.
        shaders: {
            let now = scene.present_time();
            writer.write_iter(
                scene
                    .shaders
                    .iter()
                    .map(|shader| gpui::ShaderInstance::of(shader, now)),
            )?
        },
    })
}

struct InstanceBufferWriter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Result<Self> {
        let current = pool.lock().acquire(device, unified_memory)?;
        Ok(Self {
            device: device.retain(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        })
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(InstanceBinding, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let binding = InstanceBinding {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // SAFETY: The aligned reservation lies within this frame's exclusively owned buffer and does not overlap prior writes.
        let values = unsafe {
            let start = self
                .current
                .metal_buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((binding, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        // SAFETY: The disjoint source and destination hold the same number of T values, and the destination stays MaybeUninit.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(binding)
    }

    fn write_iter<T>(
        &mut self,
        values: impl ExactSizeIterator<Item = T>,
    ) -> Result<InstanceBinding> {
        let (binding, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(binding)
    }

    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        anyhow::ensure!(
            buffer_size > self.current.size,
            "frame instance data exceeds the {MAX_INSTANCE_BUFFER_SIZE}-byte maximum"
        );
        if buffer_size != pool.buffer_size {
            log::info!("increased instance buffer size to {buffer_size}");
            pool.reset(buffer_size);
        }
        let buffer = pool.acquire(&self.device, self.unified_memory)?;
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    fn finish(self) -> InstanceBuffer {
        let Self {
            unified_memory,
            filled,
            current,
            offset,
            ..
        } = self;

        if !unified_memory {
            for (buffer, written) in &filled {
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: *written,
                });
            }
            if offset > 0 {
                current.metal_buffer.didModifyRange(NSRange {
                    location: 0,
                    length: offset,
                });
            }
        }

        // Metal retains encoded resources until the command buffer completes.
        // Only the final, largest buffer is worth keeping in the pool.
        drop(filled);
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Result<Self> {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool)?;
        Ok(Self { renderer })
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };
    use objc2_metal::MTLCommandBufferStatus;

    #[test]
    fn path_intermediate_textures_are_reused_until_resize() -> Result<()> {
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
        let initial_size = size(DevicePixels(64), DevicePixels(64));
        renderer.update_path_intermediate_textures(initial_size)?;
        let initial_texture = renderer
            .path_intermediate_texture
            .clone()
            .context("missing path intermediate texture")?;
        let initial_msaa_texture = renderer.path_intermediate_msaa_texture.clone();

        renderer.update_path_intermediate_textures(initial_size)?;
        assert!(ptr::eq(
            &*initial_texture,
            renderer
                .path_intermediate_texture
                .as_deref()
                .context("missing reused texture")?,
        ));
        assert_eq!(
            initial_msaa_texture.as_ref().map(Retained::as_ptr),
            renderer
                .path_intermediate_msaa_texture
                .as_ref()
                .map(Retained::as_ptr),
        );

        renderer.update_path_intermediate_textures(size(DevicePixels(128), DevicePixels(64)))?;
        let resized_texture = renderer
            .path_intermediate_texture
            .as_deref()
            .context("missing resized texture")?;
        assert_eq!(
            (resized_texture.width(), resized_texture.height()),
            (128, 64)
        );
        assert!(!ptr::eq(&*initial_texture, resized_texture));
        if let Some(texture) = &renderer.path_intermediate_msaa_texture {
            assert_eq!((texture.width(), texture.height()), (128, 64));
        }

        for (width, height) in [(0, 64), (64, 0), (-1, 64), (64, -1)] {
            renderer.update_path_intermediate_textures(size(
                DevicePixels(width),
                DevicePixels(height),
            ))?;
            assert!(renderer.path_intermediate_texture.is_none());
            assert!(renderer.path_intermediate_msaa_texture.is_none());
            renderer.update_path_intermediate_textures(initial_size)?;
            assert!(renderer.path_intermediate_texture.is_some());
        }
        Ok(())
    }

    #[test]
    fn render_to_image_rejects_zero_sized_layer() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        for (width, height) in [(0, 0), (0, 64), (64, 0)] {
            renderer.update_drawable_size(size(DevicePixels(width), DevicePixels(height)));
            assert!(renderer.render_to_image(&Scene::default()).is_err());
        }
        Ok(())
    }

    #[test]
    fn display_link_mode_consumes_and_clears_drawables() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let drawable = renderer
            .layer()
            .and_then(|layer| layer.nextDrawable())
            .context("window renderer must supply a drawable")?;

        renderer.set_display_link_attached(true);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: Some(_) }
        ));
        renderer.draw(&Scene::default());
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Attached { drawable: None }
        ));
        // A frame that drew consumed its drawable; one that didn't hands it back.
        assert!(
            renderer
                .set_frame_drawable(Some(drawable.clone()))
                .is_none()
        );
        assert!(renderer.set_frame_drawable(None).is_some());

        renderer.set_frame_drawable(Some(drawable));
        renderer.set_display_link_attached(false);
        assert!(matches!(
            renderer.display_link_mode,
            DisplayLinkMode::Detached
        ));
        Ok(())
    }

    #[test]
    fn render_core_video_surface() {
        let surface_properties_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
        let attributes = CFDictionary::from_slices(
            &[surface_properties_key],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer = ptr::NonNull::new(image_buffer).unwrap();
        let image_buffer = unsafe { CFRetained::from_raw(image_buffer) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
                .unwrap();
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }

        let descriptor = MTLTextureDescriptor::new();
        unsafe {
            descriptor.setWidth(16);
            descriptor.setHeight(16);
        }
        descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        descriptor.setUsage(MTLTextureUsage::RenderTarget);
        let uses_shared_storage = cfg!(target_os = "ios") || renderer.is_apple_gpu;
        descriptor.setStorageMode(if uses_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let target_texture = renderer
            .device
            .newTextureWithDescriptor(&descriptor)
            .unwrap();
        let command_buffer = renderer
            .render_frame(
                &scene,
                &target_texture,
                size(DevicePixels(16), DevicePixels(16)),
                false,
            )
            .unwrap();
        if !uses_shared_storage {
            let blit = command_buffer.blitCommandEncoder().unwrap();
            blit.synchronizeResource(ProtocolObject::from_ref(&*target_texture));
            blit.endEncoding();
        }
        drop(scene);
        drop(renderer);
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(read_texture_to_image(&target_texture).unwrap(), image);
    }

    // A scene with a shader nothing covers, presented as a
    // window presents it (the base kept, the shaders composed over it), is
    // the same picture as the whole scene drawn at once, and a re-present
    // reads the uniforms afresh.
    #[test]
    fn a_composed_present_matches_a_whole_render() {
        static PROGRAM: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-half",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return position.x < size.x * u[0] ? float4(1.0, 0.0, 0.0, 0.5) : float4(0.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let rect = |x: f32, y: f32, w: f32, h: f32| Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        };
        let uniforms = Arc::new(gpui::ShaderUniforms::new());
        uniforms.set(0, 0.5);
        let scene = |cover: bool| {
            let mut scene = Scene::default();
            let whole = rect(0.0, 0.0, 64.0, 32.0);
            scene.insert_primitive(gpui::Quad {
                bounds: whole,
                content_mask: ContentMask { bounds: whole },
                background: gpui::hsla(0.6, 0.5, 0.5, 1.0).into(),
                ..Default::default()
            });
            let shader = rect(8.0, 8.0, 32.0, 16.0);
            scene.insert_primitive(gpui::PaintShader {
                order: 0,
                bounds: shader,
                content_mask: ContentMask { bounds: shader },
                program: &PROGRAM,
                uniforms: uniforms.clone(),
                image: None,
                group: None,
                group_inset: Default::default(),
            });
            let over = if cover {
                rect(30.0, 10.0, 20.0, 4.0)
            } else {
                rect(50.0, 10.0, 10.0, 4.0)
            };
            scene.insert_primitive(gpui::Quad {
                bounds: over,
                content_mask: ContentMask { bounds: over },
                background: gpui::hsla(0.1, 0.8, 0.5, 1.0).into(),
                ..Default::default()
            });
            scene.finish();
            scene
        };
        let viewport = size(DevicePixels(64), DevicePixels(32));
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default()))).unwrap();
        // Layers over a shader are composed premultiplied in 8 bits, so a
        // translucent edge may round a level apart.
        let close = |a: &RgbaImage, b: &RgbaImage| {
            a.as_raw()
                .iter()
                .zip(b.as_raw())
                .all(|(x, y)| x.abs_diff(*y) <= 1)
        };
        for covered in [false, true] {
            let scene = scene(covered);
            let plan = scene.shader_layers.as_deref().expect("a plan");
            assert_eq!(plan.groups.len(), 1);
            assert_eq!(plan.layers[1].is_blank(), !covered);
            uniforms.set(0, 0.5);
            let whole = renderer.render_scene_to_image(&scene, viewport).unwrap();
            let composed = renderer
                .compose_scene_to_image(&scene, viewport, 2)
                .unwrap();
            assert!(close(&whole, &composed), "covered: {covered}");

            uniforms.set(0, 1.0);
            let moved = renderer
                .compose_scene_to_image(&scene, viewport, 1)
                .unwrap();
            assert_ne!(moved.as_raw(), composed.as_raw());
            let whole = renderer.render_scene_to_image(&scene, viewport).unwrap();
            assert!(close(&whole, &moved), "covered: {covered}, moved");
        }
    }

    // The present plan a composed frame leaves shows that frame again with
    // no scene: the same picture as the compose, its shaders read afresh at
    // each present, and it holds only for a drawable of the frame's size.
    #[test]
    fn a_present_plan_presents_the_frame_again_without_its_scene() {
        static PROGRAM: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-plan-half",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return position.x < size.x * u[0] ? float4(1.0, 0.0, 0.0, 0.5) : float4(0.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let rect = |x: f32, y: f32, w: f32, h: f32| Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        };
        let uniforms = Arc::new(gpui::ShaderUniforms::new());
        uniforms.set(0, 0.5);
        let mut scene = Scene::default();
        let whole = rect(0.0, 0.0, 64.0, 32.0);
        scene.insert_primitive(gpui::Quad {
            bounds: whole,
            content_mask: ContentMask { bounds: whole },
            background: gpui::hsla(0.6, 0.5, 0.5, 1.0).into(),
            ..Default::default()
        });
        let shader = rect(8.0, 8.0, 32.0, 16.0);
        scene.insert_primitive(gpui::PaintShader {
            order: 0,
            bounds: shader,
            content_mask: ContentMask { bounds: shader },
            program: &PROGRAM,
            uniforms: uniforms.clone(),
            image: None,
            group: None,
            group_inset: Default::default(),
        });
        let over = rect(30.0, 10.0, 20.0, 4.0);
        scene.insert_primitive(gpui::Quad {
            bounds: over,
            content_mask: ContentMask { bounds: over },
            background: gpui::hsla(0.1, 0.8, 0.5, 1.0).into(),
            ..Default::default()
        });
        scene.finish();
        let viewport = size(DevicePixels(64), DevicePixels(32));
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default()))).unwrap();
        assert!(renderer.present_plan().is_none());
        let composed = renderer
            .compose_scene_to_image(&scene, viewport, 1)
            .unwrap();
        let plan = renderer
            .present_plan()
            .expect("a composed frame leaves a plan");
        let at = std::time::Instant::now();

        let present = |plan: &PresentPlan| {
            let texture = renderer.layer_texture(64, 32).unwrap();
            let command_buffer = plan.encode_taken(&texture, at).unwrap();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            read_texture_to_image(&texture).unwrap()
        };
        assert_eq!(present(&plan).as_raw(), composed.as_raw());

        uniforms.set(0, 1.0);
        let moved = present(&plan);
        assert_ne!(moved.as_raw(), composed.as_raw());
        let whole = renderer.render_scene_to_image(&scene, viewport).unwrap();
        assert!(
            whole
                .as_raw()
                .iter()
                .zip(moved.as_raw())
                .all(|(a, b)| a.abs_diff(*b) <= 1),
            "the plan presents the moved shader as a whole render draws it"
        );

        assert!(plan.holds(64, 32, at));
        assert!(!plan.holds(128, 64, at), "a resized drawable needs a frame");
        assert!(!plan.moving(at), "a shader at rest asks for no presents");
        uniforms.present_continuously();
        assert!(plan.moving(at));
    }

    // A draw that presents nothing, here one outside its display link's
    // update, leaves no plan: the frame it replaced is no longer the
    // window's, so nothing may present it again.
    #[test]
    fn a_draw_that_presents_nothing_leaves_no_present_plan() -> Result<()> {
        static PROGRAM: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-plan-dropped",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        renderer.update_drawable_size(size(DevicePixels(64), DevicePixels(64)));
        let bounds = Bounds {
            origin: Point::new(ScaledPixels(8.0), ScaledPixels(8.0)),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(gpui::PaintShader {
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            program: &PROGRAM,
            uniforms: Arc::new(gpui::ShaderUniforms::new()),
            image: None,
            group: None,
            group_inset: Default::default(),
        });
        scene.finish();
        renderer.draw(&scene);
        assert!(renderer.present_plan().is_some());
        renderer.set_display_link_attached(true);
        renderer.draw(&scene);
        assert!(renderer.present_plan().is_none());
        Ok(())
    }

    // Something drawn over a shader keeps what is drawn over
    // it in turn on top, where that sits clear of the shader: a menu over a
    // playing row keeps its text.
    #[test]
    fn what_covers_a_shader_keeps_its_own_order() {
        static PROGRAM: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-solid",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return float4(0.0, 1.0, 0.0, 1.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let rect = |x: f32, y: f32, w: f32, h: f32| Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        };
        let quad = |b: Bounds<ScaledPixels>, hue: f32| gpui::Quad {
            bounds: b,
            content_mask: ContentMask { bounds: b },
            background: gpui::hsla(hue, 0.8, 0.5, 1.0).into(),
            ..Default::default()
        };
        let mut scene = Scene::default();
        let shader = rect(4.0, 4.0, 16.0, 16.0);
        scene.insert_primitive(gpui::PaintShader {
            order: 0,
            bounds: shader,
            content_mask: ContentMask { bounds: shader },
            program: &PROGRAM,
            uniforms: Arc::new(gpui::ShaderUniforms::new()),
            image: None,
            group: None,
            group_inset: Default::default(),
        });
        // A menu over the shader, and its text over the menu, clear of it.
        scene.insert_primitive(quad(rect(12.0, 8.0, 40.0, 16.0), 0.1));
        scene.insert_primitive(quad(rect(32.0, 12.0, 12.0, 8.0), 0.6));
        scene.finish();
        let plan = scene.shader_layers.as_deref().expect("a plan");
        assert!(plan.layers[1].quads.len() == 2, "the text follows the menu");

        let viewport = size(DevicePixels(64), DevicePixels(32));
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default()))).unwrap();
        let whole = renderer.render_scene_to_image(&scene, viewport).unwrap();
        let composed = renderer
            .compose_scene_to_image(&scene, viewport, 1)
            .unwrap();
        assert_eq!(
            whole.get_pixel(36, 16),
            composed.get_pixel(36, 16),
            "the text is on top"
        );
        assert!(
            whole
                .as_raw()
                .iter()
                .zip(composed.as_raw())
                .all(|(a, b)| a.abs_diff(*b) <= 1)
        );
    }

    // A layer group at rest, partly outside the window, composed through its
    // texture gives the pixels its content does drawn without a group: the
    // offset, the clipped frame, the tile and the texture's viewport agree.
    #[test]
    fn a_group_at_rest_matches_its_content() {
        static HALF: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-group-half",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return position.x < size.x * u[0] ? float4(1.0, 0.0, 0.0, 0.5) : float4(0.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        static IDENTITY: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-group-identity",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    float4 c = sample_image(image, tile, position / size);
    float3 rgb = c.a > 0.0 ? c.rgb / c.a : float3(0.0);
    return float4(rgb, c.a);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let rect = |x: f32, y: f32, w: f32, h: f32| Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        };
        let quad = |at: Bounds<ScaledPixels>, hue: f32| gpui::Quad {
            bounds: at,
            content_mask: ContentMask { bounds: at },
            background: gpui::hsla(hue, 0.5, 0.5, 1.0).into(),
            ..Default::default()
        };
        let shader = |at: Bounds<ScaledPixels>, program| gpui::PaintShader {
            order: 0,
            bounds: at,
            content_mask: ContentMask { bounds: at },
            program,
            uniforms: Arc::new(gpui::ShaderUniforms::new()),
            image: None,
            group: None,
            group_inset: Default::default(),
        };
        let scene = |grouped: bool| {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(rect(0.0, 0.0, 64.0, 32.0), 0.6));
            if grouped {
                scene.push_group();
            }
            scene.insert_primitive(quad(rect(10.0, 10.0, 20.0, 10.0), 0.3));
            let half = shader(rect(30.0, 8.0, 24.0, 16.0), &HALF);
            half.uniforms.set(0, 0.5);
            scene.insert_primitive(half);
            if grouped {
                let mut compose = shader(rect(4.0, 4.0, 80.0, 40.0), &IDENTITY);
                compose.content_mask = ContentMask {
                    bounds: rect(0.0, 0.0, 64.0, 32.0),
                };
                scene.pop_group(compose);
            }
            scene.finish();
            scene
        };
        let grouped = scene(true);
        assert_eq!(
            grouped.layer_group_frames[0],
            Some(rect(4.0, 4.0, 60.0, 28.0))
        );
        let viewport = size(DevicePixels(64), DevicePixels(32));
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default()))).unwrap();
        let plain = renderer
            .compose_scene_to_image(&scene(false), viewport, 1)
            .unwrap();
        let composed = renderer
            .compose_scene_to_image(&grouped, viewport, 1)
            .unwrap();
        let worst = plain
            .as_raw()
            .iter()
            .zip(composed.as_raw())
            .map(|(x, y)| x.abs_diff(*y))
            .max();
        assert!(worst <= Some(2), "worst channel difference {worst:?}");
    }

    // A held layer drawn in a dynamic colour follows the colour source at
    // each present, with no frame drawn: the layer is drawn again for the
    // entry that changed, and a shader over it is composed as before.
    #[test]
    fn a_dynamic_colour_follows_its_source_with_no_frame_drawn() {
        static SOLID: gpui::ShaderProgram = gpui::ShaderProgram {
            name: "test-dynamic-solid",
            msl: r#"
float4 shade(float2 position, float2 size, float time, constant float *u, texture2d<float> image, float4 tile) {
    return float4(0.0, 0.0, 1.0, 1.0);
}
"#,
            wgsl: "",
            prepare: None,
        };
        let rect = |x: f32, y: f32, w: f32, h: f32| Bounds {
            origin: Point::new(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        };
        let red = gpui::Rgba {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let handle = gpui::ColorHandle::claim(red).expect("a free entry");
        let index = handle.index();
        let green = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let source_green = green.clone();
        gpui::set_color_source(move |_, colors| {
            if source_green.load(std::sync::atomic::Ordering::Relaxed) {
                colors[index] = gpui::Rgba {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                };
            }
            None
        });
        let mut scene = Scene::default();
        let whole = rect(0.0, 0.0, 32.0, 16.0);
        scene.insert_primitive(gpui::Quad {
            bounds: whole,
            content_mask: ContentMask { bounds: whole },
            background: handle.hsla(1.0).into(),
            ..Default::default()
        });
        let corner = rect(24.0, 0.0, 8.0, 8.0);
        scene.insert_primitive(gpui::PaintShader {
            order: 0,
            bounds: corner,
            content_mask: ContentMask { bounds: corner },
            program: &SOLID,
            uniforms: Arc::new(gpui::ShaderUniforms::new()),
            image: None,
            group: None,
            group_inset: Default::default(),
        });
        scene.finish();
        let viewport = size(DevicePixels(32), DevicePixels(16));
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default()))).unwrap();
        let before = renderer
            .compose_scene_to_image(&scene, viewport, 1)
            .unwrap();
        assert!(before.get_pixel(4, 8).0[0] > 200, "red");
        green.store(true, std::sync::atomic::Ordering::Relaxed);
        let after = renderer
            .compose_scene_to_image(&scene, viewport, 1)
            .unwrap();
        assert!(
            after.get_pixel(4, 8).0[1] > 200,
            "green, from the same scene"
        );
        assert!(
            after.get_pixel(28, 4).0[2] > 200,
            "the shader composed over it"
        );
    }
}
