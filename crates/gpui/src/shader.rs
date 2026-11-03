//! Shader quads: a quad drawn by a fragment program the app supplies, whose
//! uniforms the app writes from any thread and the renderer reads as each
//! frame is presented.
//!
//! A window presents the last frame again, with no layout or paint, while a
//! shader on screen asks it to: when its uniforms change, and at every
//! refresh until the deadline its uniforms set ([`ShaderUniforms::present_until`]).
//! So motion that a program works out from `time` costs a present per
//! refresh and nothing more, and a window at rest presents nothing.

use scheduler::Instant;
use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};

use parking_lot::{Mutex, ReentrantMutex, ReentrantMutexGuard};

use crate::{AtlasTile, Bounds, ContentMask, DrawOrder, Primitive, ScaledPixels};

/// How many floats a shader's uniforms hold: twenty blocks of four.
pub const SHADER_UNIFORMS: usize = 80;

/// A fragment program drawing a quad.
///
/// `msl` and `wgsl` each define `shade`, given the pixel's position within
/// the quad and the quad's size in device pixels, `time` in seconds, the
/// uniforms, and the quad's image:
///
/// ```text
/// float4 shade(float2 position, float2 size, float time, constant float *u,
///              texture2d<float> image, float4 tile)
/// fn shade(position: vec2<f32>, size: vec2<f32>, time: f32,
///          u: array<vec4<f32>, 20>, tile: vec4<f32>) -> vec4<f32>
/// ```
///
/// The colour returned is straight, not premultiplied. `sample_image(image,
/// tile, uv)` in MSL and `sample_image(tile, uv)` in WGSL read the quad's
/// image at `uv`, 0 to 1 across it.
///
/// A program is compiled the first time it is drawn and kept for as long as
/// the renderer lives, by the address of its `static`.
#[derive(Debug)]
pub struct ShaderProgram {
    /// What the compiled program is labelled with in a GPU debugger.
    pub name: &'static str,
    /// The program's `shade` in the Metal Shading Language.
    pub msl: &'static str,
    /// The program's `shade` in WGSL.
    pub wgsl: &'static str,
    /// Run on the CPU once per quad as each frame is presented, on whichever
    /// thread presents it, with a copy of the uniforms and the seconds from
    /// the uniforms' origin to the frame's presentation, in double precision.
    /// It may rewrite the copy, and returns the `time` the program sees, so
    /// an app can keep `time` small and exact however long it runs. Without
    /// it, the program sees that time as an `f32`.
    pub prepare: Option<fn(&mut [f32; SHADER_UNIFORMS], f64) -> f32>,
}

/// The instant shader time counts from when the uniforms set no origin.
pub fn shader_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Signed nanoseconds from [`shader_epoch`] to `at`.
fn since_epoch(at: Instant) -> i64 {
    let epoch = shader_epoch();
    match at.checked_duration_since(epoch) {
        Some(after) => after.as_nanos() as i64,
        None => -(epoch.duration_since(at).as_nanos() as i64),
    }
}

/// A shader's uniforms, shared between the app and the renderer: written by
/// the app from any thread, read as each frame is presented.
///
/// Writes that belong together are made one with [`publish`](Self::publish):
/// a present sees all of them or none. A present never waits for a writer: a
/// block a writer holds is shown as the last present took it, and the
/// window presents again at its next refresh.
#[derive(Debug)]
pub struct ShaderUniforms {
    values: [AtomicU32; SHADER_UNIFORMS],
    /// Nanoseconds from [`shader_epoch`] to the origin `time` counts from,
    /// negative for an origin before it.
    origin: AtomicI64,
    changed: AtomicBool,
    /// Nanoseconds from [`shader_epoch`] until which the window presents at
    /// every refresh; 0 for none, `u64::MAX` for no end.
    until: AtomicU64,
    /// The present after `until` passes, not yet made.
    landing: AtomicBool,
    /// Odd while a publish is being made.
    sequence: AtomicU64,
    /// Held for a whole publish, reentrantly, with the publish's depth.
    writer: ReentrantMutex<Cell<u32>>,
    /// The thread publishing, by [`thread_number`]; 0 for none.
    owner: AtomicU64,
    /// The block as the last present took it, shown again while a writer
    /// holds the block. Touched only by presents.
    presented: Mutex<Option<Taken>>,
    /// The last present showed `presented` in place of a held block.
    stale: AtomicBool,
}

/// The origin and every value, as one publish left them.
type Taken = (i64, [f32; SHADER_UNIFORMS]);

/// How many times a present looks again at a publish in progress before it
/// shows the last copy instead.
const READ_SPINS: u32 = 64;

/// A number for each thread, from 1.
fn thread_number() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static NUMBER: u64 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    NUMBER.with(|number| *number)
}

/// Closes a publish however its closure ends, unwinding included.
struct Published<'a> {
    uniforms: &'a ShaderUniforms,
    held: ReentrantMutexGuard<'a, Cell<u32>>,
    depth: u32,
}

impl Drop for Published<'_> {
    fn drop(&mut self) {
        self.held.set(self.depth);
        if self.depth == 0 {
            self.uniforms.owner.store(0, Ordering::Relaxed);
            self.uniforms.sequence.fetch_add(1, Ordering::Release);
        }
    }
}

impl Default for ShaderUniforms {
    fn default() -> Self {
        Self {
            values: std::array::from_fn(|_| AtomicU32::new(0)),
            origin: AtomicI64::new(0),
            changed: AtomicBool::new(false),
            until: AtomicU64::new(0),
            landing: AtomicBool::new(false),
            sequence: AtomicU64::new(0),
            writer: ReentrantMutex::new(Cell::new(0)),
            owner: AtomicU64::new(0),
            presented: Mutex::new(None),
            stale: AtomicBool::new(false),
        }
    }
}

impl ShaderUniforms {
    /// Uniforms all zero, counting time from [`shader_epoch`], at rest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs `write` as one publish: a present sees what it writes all at
    /// once, and no other thread writes in between. Reentrant: a write
    /// inside it joins it.
    pub fn publish<R>(&self, write: impl FnOnce() -> R) -> R {
        let held = self.writer.lock();
        let depth = held.get();
        if depth == 0 {
            // Odd before any value is stored; the fence keeps the stores
            // that follow from being seen ahead of it.
            self.sequence.fetch_add(1, Ordering::Acquire);
            std::sync::atomic::fence(Ordering::Release);
            self.owner.store(thread_number(), Ordering::Relaxed);
        }
        held.set(depth + 1);
        let _published = Published {
            uniforms: self,
            held,
            depth,
        };
        write()
    }

    /// Runs `read` until it saw one whole publish, waiting for a writer if
    /// it must: for the app, to read several values together. Within a
    /// publish on this thread it reads as it stands.
    pub fn read<R>(&self, read: impl Fn() -> R) -> R {
        if let Some(result) = self.try_read(&read) {
            return result;
        }
        let _held = self.writer.lock();
        read()
    }

    /// `read` without waiting: `None` when a writer still holds the block
    /// after a few looks.
    fn try_read<R>(&self, read: impl Fn() -> R) -> Option<R> {
        if self.owner.load(Ordering::Relaxed) == thread_number() {
            return Some(read());
        }
        for _ in 0..READ_SPINS {
            let before = self.sequence.load(Ordering::Acquire);
            if before % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let result = read();
            std::sync::atomic::fence(Ordering::Acquire);
            if self.sequence.load(Ordering::Acquire) == before {
                return Some(result);
            }
        }
        let _held = self.writer.try_lock()?;
        Some(read())
    }

    /// Writes `value` at `index`. Whether it changed.
    pub fn set(&self, index: usize, value: f32) -> bool {
        self.publish(|| {
            let bits = value.to_bits();
            let changed = self.values[index].swap(bits, Ordering::Relaxed) != bits;
            if changed {
                self.changed.store(true, Ordering::Relaxed);
            }
            changed
        })
    }

    /// The value at `index` as it stands.
    pub fn get(&self, index: usize) -> f32 {
        f32::from_bits(self.values[index].load(Ordering::Relaxed))
    }

    /// The moment `time` counts from. Whether it moved.
    pub fn set_origin(&self, origin: Instant) -> bool {
        self.publish(|| {
            let nanos = since_epoch(origin);
            let changed = self.origin.swap(nanos, Ordering::Relaxed) != nanos;
            if changed {
                self.changed.store(true, Ordering::Relaxed);
            }
            changed
        })
    }

    /// Seconds from the origin to `at`.
    pub fn time_at(&self, at: Instant) -> f64 {
        let origin = self.read(|| self.origin.load(Ordering::Relaxed));
        (since_epoch(at) - origin) as f64 / 1e9
    }

    /// Presented at every refresh until `deadline`, and once after it, so a
    /// motion's last value lands; `None` to stop. A deadline of
    /// [`Instant`]'s far future runs until replaced. Replaces the last.
    pub fn present_until(&self, deadline: Option<Instant>) {
        let nanos = deadline.map_or(0, |at| since_epoch(at).max(1) as u64);
        self.until.store(nanos, Ordering::Relaxed);
        self.landing.store(nanos != 0, Ordering::Relaxed);
        self.changed.store(true, Ordering::Relaxed);
    }

    /// Presents at every refresh with no end, until `present_until` is set
    /// again.
    pub fn present_continuously(&self) {
        self.until.store(u64::MAX, Ordering::Relaxed);
        self.landing.store(true, Ordering::Relaxed);
        self.changed.store(true, Ordering::Relaxed);
    }

    fn moving(&self, now: Instant) -> bool {
        let until = self.until.load(Ordering::Relaxed);
        until != 0 && (since_epoch(now).max(0) as u64) < until
    }

    /// Whether the window should keep its refreshes coming for this shader
    /// at `now`: before its deadline, at the present that lands it, and
    /// after a present that could not read it whole.
    pub fn is_live(&self, now: Instant) -> bool {
        self.moving(now)
            || self.landing.load(Ordering::Relaxed)
            || self.stale.load(Ordering::Relaxed)
    }

    /// Whether the refresh at `now` should present: the uniforms changed
    /// since this was last asked, the shader moves, the present landing its
    /// deadline is due, or the last present was stale.
    pub fn take_present(&self, now: Instant) -> bool {
        let changed = self.changed.swap(false, Ordering::Relaxed);
        let moving = self.moving(now);
        let landed = !moving && self.landing.swap(false, Ordering::Relaxed);
        changed || moving || landed || self.stale.load(Ordering::Relaxed)
    }

    /// Every value, as one publish left them.
    pub fn snapshot(&self) -> [f32; SHADER_UNIFORMS] {
        self.read(|| self.take().1)
    }

    fn take(&self) -> Taken {
        (
            self.origin.load(Ordering::Relaxed),
            std::array::from_fn(|i| f32::from_bits(self.values[i].load(Ordering::Relaxed))),
        )
    }

    /// The seconds from the origin to `at` and every value, for a present:
    /// never waits on a writer. With the block held, the last present's copy,
    /// marked stale so the window presents again.
    pub fn presented_at(&self, at: Instant) -> (f64, [f32; SHADER_UNIFORMS]) {
        let mut presented = self.presented.lock();
        let taken = match self.try_read(|| self.take()) {
            Some(taken) => {
                self.stale.store(false, Ordering::Relaxed);
                taken
            }
            None => {
                self.stale.store(true, Ordering::Relaxed);
                match *presented {
                    Some(taken) => taken,
                    // Never presented: nothing to show in its place.
                    None => self.read(|| self.take()),
                }
            }
        };
        *presented = Some(taken);
        let (origin, values) = taken;
        ((since_epoch(at) - origin) as f64 / 1e9, values)
    }
}

/// A shader as the scene holds it.
#[derive(Clone, Debug)]
pub struct PaintShader {
    /// Where it is drawn among the scene's primitives.
    pub order: DrawOrder,
    /// The quad, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// What of the quad is drawn.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The program drawing it.
    pub program: &'static ShaderProgram,
    /// The uniforms the program reads, shared with the app.
    pub uniforms: std::sync::Arc<ShaderUniforms>,
    /// The image the program samples, in the sprite atlas.
    pub image: Option<AtlasTile>,
}

impl From<PaintShader> for Primitive {
    fn from(shader: PaintShader) -> Self {
        Primitive::Shader(shader)
    }
}

/// One shader as a renderer hands it to the GPU: its quad, its clip, and
/// its uniforms and time as they are for the frame being presented. Laid out
/// for Metal and WGSL alike: blocks of four 16-byte aligned floats.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ShaderInstance {
    /// x, y, width, height.
    pub bounds: [f32; 4],
    /// The content mask: x, y, width, height.
    pub clip: [f32; 4],
    /// The uniforms as the frame shows them.
    pub uniforms: [f32; SHADER_UNIFORMS],
    /// `time` first; the rest pads the block.
    pub time: [f32; 4],
    /// The image's tile in its atlas texture, in texels: x, y, width, height.
    pub tile: [f32; 4],
}

impl Default for ShaderInstance {
    fn default() -> Self {
        Self {
            bounds: [0.0; 4],
            clip: [0.0; 4],
            uniforms: [0.0; SHADER_UNIFORMS],
            time: [0.0; 4],
            tile: [0.0; 4],
        }
    }
}

impl ShaderInstance {
    /// `shader` as the frame presented at `at` shows it.
    pub fn of(shader: &PaintShader, at: Instant) -> Self {
        let (seconds, mut uniforms) = shader.uniforms.presented_at(at);
        let time = match shader.program.prepare {
            Some(prepare) => prepare(&mut uniforms, seconds),
            None => seconds as f32,
        };
        let rect = |b: &Bounds<ScaledPixels>| {
            [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0]
        };
        Self {
            bounds: rect(&shader.bounds),
            clip: rect(&shader.content_mask.bounds),
            uniforms,
            time: [time, 0.0, 0.0, 0.0],
            tile: shader.image.as_ref().map_or([0.0; 4], |tile| {
                let b = tile.bounds;
                [
                    b.origin.x.0 as f32,
                    b.origin.y.0 as f32,
                    b.size.width.0 as f32,
                    b.size.height.0 as f32,
                ]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn a_change_presents_once() {
        let uniforms = ShaderUniforms::new();
        let now = Instant::now();
        assert!(uniforms.set(3, 0.25));
        assert_eq!(uniforms.get(3), 0.25);
        assert!(uniforms.take_present(now));
        assert!(!uniforms.take_present(now));
        assert!(!uniforms.set(3, 0.25), "the same value is no change");
        assert!(!uniforms.take_present(now));
        assert!(!uniforms.is_live(now));
    }

    #[test]
    fn a_deadline_presents_until_it_passes_and_once_after() {
        let uniforms = ShaderUniforms::new();
        let now = Instant::now();
        uniforms.present_until(Some(now + Duration::from_millis(50)));
        assert!(uniforms.is_live(now));
        assert!(uniforms.take_present(now));
        assert!(uniforms.take_present(now + Duration::from_millis(10)));
        let after = now + Duration::from_millis(60);
        assert!(uniforms.take_present(after), "the present that lands it");
        assert!(!uniforms.take_present(after));
        assert!(!uniforms.is_live(after), "then the window parks");
    }

    #[test]
    fn time_counts_from_the_origin() {
        let uniforms = ShaderUniforms::new();
        let origin = shader_epoch() + Duration::from_secs(1_000_000);
        uniforms.set_origin(origin);
        let (seconds, _) = uniforms.presented_at(origin + Duration::from_millis(1500));
        assert!((seconds - 1.5).abs() < 1e-9, "{seconds}");
    }

    #[test]
    fn a_panic_mid_publish_leaves_the_block_readable() {
        let uniforms = Arc::new(ShaderUniforms::new());
        let writer = uniforms.clone();
        let panicked = std::thread::spawn(move || {
            writer.publish(|| {
                writer.set(0, 1.0);
                panic!("a writer fails mid-publish");
            })
        })
        .join();
        assert!(panicked.is_err());
        assert_eq!(uniforms.snapshot()[0], 1.0);
        uniforms.set(0, 2.0);
        assert_eq!(uniforms.snapshot()[0], 2.0);
    }

    #[test]
    fn a_present_never_waits_on_a_writer() {
        let uniforms = Arc::new(ShaderUniforms::new());
        uniforms.set(0, 1.0);
        let now = Instant::now();
        assert_eq!(uniforms.presented_at(now).1[0], 1.0);
        uniforms.take_present(now);

        let (entered, inside) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let writer = {
            let uniforms = uniforms.clone();
            std::thread::spawn(move || {
                uniforms.publish(|| {
                    uniforms.set(0, 2.0);
                    entered.send(()).unwrap();
                    released.recv().unwrap();
                })
            })
        };
        inside.recv().unwrap();
        let started = Instant::now();
        let (_, values) = uniforms.presented_at(now);
        assert!(started.elapsed() < Duration::from_millis(50));
        assert_eq!(values[0], 1.0, "the last present's copy");
        assert!(
            uniforms.is_live(now),
            "a stale present keeps the frames coming"
        );
        release.send(()).unwrap();
        writer.join().unwrap();
        assert!(uniforms.take_present(now));
        assert_eq!(uniforms.presented_at(now).1[0], 2.0);
        uniforms.take_present(now);
        assert!(!uniforms.is_live(now), "whole again: at rest");
    }

    #[test]
    fn writes_in_one_publish_are_seen_together() {
        let uniforms = Arc::new(ShaderUniforms::new());
        let reader = {
            let uniforms = uniforms.clone();
            std::thread::spawn(move || {
                for _ in 0..20_000 {
                    let values = uniforms.snapshot();
                    assert_eq!(values[0], values[1], "half a publish");
                }
            })
        };
        for i in 0..20_000 {
            let v = i as f32;
            uniforms.publish(|| {
                uniforms.set(0, v);
                uniforms.set(1, v);
            });
        }
        reader.join().unwrap();
    }

    #[test]
    fn the_instance_is_aligned_blocks() {
        assert_eq!(
            std::mem::size_of::<ShaderInstance>(),
            4 * (4 + 4 + SHADER_UNIFORMS + 4 + 4)
        );
    }
}
