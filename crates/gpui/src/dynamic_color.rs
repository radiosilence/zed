//! Dynamic colours: colours whose values an app supplies as each frame is
//! presented, so a colour can change, a fill fading or a theme colour
//! moving, with no frame drawn.
//!
//! A [`ColorHandle`] is an entry in an app-wide table of colours. It gives
//! out an [`Hsla`] that names the entry, which any quad, border, shadow,
//! underline, monochrome sprite or path may be drawn in, and a shader's
//! uniforms may hold ([`ShaderUniforms::set_color`](crate::ShaderUniforms::set_color)).
//! The table is filled by the app's colour source ([`set_color_source`]),
//! called as each frame is presented, on whichever thread presents it, with
//! the time the frame reaches the screen. The source says until when each
//! entry moves: a window, or a shader's uniforms, presents while an entry it
//! draws with moves, and a renderer draws again only what is drawn in an
//! entry that changed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use scheduler::Instant;

use crate::{Hsla, Rgba};

/// How many entries the table holds: with a block of four floats each, as
/// many as fit in the 4 KiB of uniforms every renderer guarantees.
pub const DYNAMIC_COLORS: usize = 255;

/// The hue at which a dynamic colour's hue starts: a colour whose hue is
/// this or more names entry `hue - DYNAMIC_HUE`. Real hues are 0 to 1.
const DYNAMIC_HUE: f32 = 2.0;

/// What fills the table at each present: given the time the frame reaches
/// the screen and the entries as the last present left them, writes each
/// entry's colour, and until when each entry moves into the second slice,
/// which starts with every entry at rest (`None`).
pub type ColorSource = dyn Fn(Instant, &mut [Rgba], &mut [Option<Instant>]) + Send + Sync;

struct Table {
    values: [Rgba; DYNAMIC_COLORS],
    used: [bool; DYNAMIC_COLORS],
    source: Option<Arc<ColorSource>>,
}

static TABLE: Mutex<Table> = Mutex::new(Table {
    values: [Rgba {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    }; DYNAMIC_COLORS],
    used: [false; DYNAMIC_COLORS],
    source: None,
});

/// Moves whenever an entry's value changes.
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// Nanoseconds from the first read of [`base`] until which each entry
/// moves; 0 for at rest.
static MOVING_UNTIL: [AtomicU64; DYNAMIC_COLORS] = [const { AtomicU64::new(0) }; DYNAMIC_COLORS];

/// The latest of [`MOVING_UNTIL`]: until when any entry moves.
static ANY_MOVING_UNTIL: AtomicU64 = AtomicU64::new(0);

fn base() -> Instant {
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

fn nanos(at: Instant) -> u64 {
    at.saturating_duration_since(base()).as_nanos() as u64
}

/// An entry of the colour table, held while something draws with it and
/// freed when dropped.
#[derive(Debug)]
pub struct ColorHandle(u16);

impl ColorHandle {
    /// A free entry standing at `color`, or `None` when every entry is
    /// taken.
    pub fn claim(color: Rgba) -> Option<Self> {
        let mut table = TABLE.lock();
        let index = table.used.iter().position(|used| !used)?;
        table.used[index] = true;
        table.values[index] = color;
        // At rest until the source says otherwise: the deadline of whoever
        // held the entry before is not this holder's.
        MOVING_UNTIL[index].store(0, Ordering::Relaxed);
        GENERATION.fetch_add(1, Ordering::Relaxed);
        Some(Self(index as u16))
    }

    /// The entry's place in the table, as the colour source sees it.
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// The colour to draw with: this entry, its alpha scaled by `alpha`.
    pub fn hsla(&self, alpha: f32) -> Hsla {
        Hsla {
            h: DYNAMIC_HUE + self.0 as f32,
            s: 0.0,
            l: 0.0,
            a: alpha,
        }
    }

    /// The entry's colour as the last present left it.
    pub fn rgba(&self) -> Rgba {
        TABLE.lock().values[self.0 as usize]
    }
}

impl Drop for ColorHandle {
    fn drop(&mut self) {
        TABLE.lock().used[self.0 as usize] = false;
    }
}

/// Sets what fills the colour table at each present, replacing any source
/// set before. The source must not block: it runs as frames are presented.
///
/// A window at rest presents nothing, so an app whose colours start to move
/// asks the windows drawing them for a present
/// ([`Window::request_present`](crate::Window::request_present)); from then
/// on, the deadlines the source writes keep them presenting while an entry
/// they draw with moves.
pub fn set_color_source(
    source: impl Fn(Instant, &mut [Rgba], &mut [Option<Instant>]) + Send + Sync + 'static,
) {
    TABLE.lock().source = Some(Arc::new(source));
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// The entry a colour names, if it names one.
pub fn dynamic_color_of(color: &Hsla) -> Option<usize> {
    (color.h >= DYNAMIC_HUE).then(|| ((color.h - DYNAMIC_HUE) as usize).min(DYNAMIC_COLORS - 1))
}

/// The colour entry `index` holds as the last present left it, its alpha
/// scaled by `alpha`: what a colour naming it is wherever the CPU reads it.
pub fn dynamic_color_rgba(index: usize, alpha: f32) -> Rgba {
    let mut rgba = TABLE.lock().values[index.min(DYNAMIC_COLORS - 1)];
    rgba.a *= alpha;
    rgba
}

/// The present the table was last filled for. Held across a whole refresh,
/// so refreshes from different threads run one at a time.
static REFRESHED: Mutex<Option<Instant>> = Mutex::new(None);

/// Fills the table for a present reaching the screen at `at`: runs the
/// colour source and notes until when each entry moves. Called by
/// renderers as they present, from any thread. A present reaching the
/// screen before the one the table was last filled for fills it for that
/// later present again, so the colours never step back and a source set
/// meanwhile is still heard.
pub fn refresh_dynamic_colors(at: Instant) {
    let mut refreshed = REFRESHED.lock();
    let at = refreshed.map_or(at, |last| at.max(last));
    let (source, before) = {
        let table = TABLE.lock();
        (table.source.clone(), table.values)
    };
    let Some(source) = source else {
        return;
    };
    *refreshed = Some(at);
    let mut values = before;
    let mut until = [None; DYNAMIC_COLORS];
    source(at, &mut values, &mut until);
    let mut any = 0;
    for (entry, until) in MOVING_UNTIL.iter().zip(until) {
        let nanos = until.map_or(0, |until| nanos(until).max(1));
        entry.store(nanos, Ordering::Relaxed);
        any = any.max(nanos);
    }
    // Released after the entries: a reader that sees this refresh's latest
    // deadline sees its entries' too.
    ANY_MOVING_UNTIL.store(any, Ordering::Release);
    if values != before {
        // Only the entries the source changed: one claimed meanwhile keeps
        // the colour it was claimed at.
        let mut table = TABLE.lock();
        for (index, value) in values.iter().enumerate() {
            if *value != before[index] {
                table.values[index] = *value;
            }
        }
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

fn moving(until: &AtomicU64, at: Instant) -> bool {
    let until = until.load(Ordering::Relaxed);
    until != 0 && nanos(at) < until
}

/// Whether the colour source said at the last present that any entry moves
/// at `at`.
pub fn dynamic_colors_moving(at: Instant) -> bool {
    let until = ANY_MOVING_UNTIL.load(Ordering::Acquire);
    until != 0 && nanos(at) < until
}

/// Whether the colour source said at the last present that entry `index`
/// moves at `at`.
pub fn dynamic_color_moving(index: usize, at: Instant) -> bool {
    moving(&MOVING_UNTIL[index.min(DYNAMIC_COLORS - 1)], at)
}

/// Moves whenever an entry's value changes.
pub fn dynamic_colors_generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

/// The table as a renderer uploads it: four floats per entry, red, green,
/// blue and alpha. Written into `out`, which is cleared first.
pub fn dynamic_colors_frame(out: &mut Vec<f32>) {
    out.clear();
    let table = TABLE.lock();
    for color in &table.values {
        out.extend_from_slice(&[color.r, color.g, color.b, color.a]);
    }
}

/// The entries of the table, as a set: those a scene draws with, or those
/// that changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DynamicColorSet([u64; DYNAMIC_COLORS.div_ceil(64)]);

impl DynamicColorSet {
    /// Notes the entry `color` names, if any.
    pub fn note(&mut self, color: &Hsla) {
        if let Some(index) = dynamic_color_of(color) {
            self.insert(index);
        }
    }

    /// Adds entry `index`.
    pub fn insert(&mut self, index: usize) {
        self.0[index / 64] |= 1 << (index % 64);
    }

    /// Whether the set holds no entry.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    /// Whether the two sets share an entry.
    pub fn intersects(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).any(|(a, b)| a & b != 0)
    }

    /// Whether the colour source said at the last present that an entry in
    /// the set moves at `at`.
    pub fn moving(&self, at: Instant) -> bool {
        dynamic_colors_moving(at)
            && self.0.iter().enumerate().any(|(word, bits)| {
                let mut bits = *bits;
                while bits != 0 {
                    let index = word * 64 + bits.trailing_zeros() as usize;
                    if dynamic_color_moving(index, at) {
                        return true;
                    }
                    bits &= bits - 1;
                }
                false
            })
    }

    /// Adds every entry `other` holds.
    pub fn extend(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a |= *b;
        }
    }
}

/// The entries whose colour differs between two uploaded tables, as
/// [`dynamic_colors_frame`] writes them.
pub fn changed_dynamic_colors(before: &[f32], after: &[f32]) -> DynamicColorSet {
    let mut changed = DynamicColorSet::default();
    for (index, (a, b)) in before.chunks(4).zip(after.chunks(4)).enumerate() {
        if a != b {
            changed.insert(index);
        }
    }
    if before.len() != after.len() {
        for index in 0..DYNAMIC_COLORS {
            changed.insert(index);
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    static SERIAL: Mutex<()> = Mutex::new(());

    /// One colour test at a time, each starting with no refresh behind it.
    fn serial() -> parking_lot::MutexGuard<'static, ()> {
        let serial = SERIAL.lock();
        *REFRESHED.lock() = None;
        for until in &MOVING_UNTIL {
            until.store(0, Ordering::Relaxed);
        }
        ANY_MOVING_UNTIL.store(0, Ordering::Relaxed);
        serial
    }

    fn red() -> Rgba {
        Rgba {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    #[test]
    fn a_handle_names_its_entry_and_frees_it() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let named = handle.hsla(0.5);
        assert_eq!(dynamic_color_of(&named), Some(handle.index()));
        assert_eq!(dynamic_color_rgba(handle.index(), 0.5).a, 0.5);
        assert_eq!(Rgba::from(named), Rgba { a: 0.5, ..red() });
        let index = handle.index();
        drop(handle);
        let again = ColorHandle::claim(red()).expect("the freed entry");
        assert_eq!(again.index(), index);
    }

    #[test]
    fn the_source_fills_the_table_and_says_until_when_it_moves() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let index = handle.index();
        let start = Instant::now();
        let end = start + Duration::from_millis(100);
        set_color_source(move |at, colors, until| {
            let t = (at.saturating_duration_since(start).as_secs_f32() / 0.1).min(1.0);
            colors[index] = Rgba {
                r: 1.0 - t,
                g: t,
                b: 0.0,
                a: 1.0,
            };
            until[index] = (at < end).then_some(end);
        });
        let before = dynamic_colors_generation();
        refresh_dynamic_colors(start + Duration::from_millis(50));
        assert!(dynamic_colors_generation() > before);
        assert!((handle.rgba().g - 0.5).abs() < 1e-3);
        assert!(dynamic_colors_moving(start + Duration::from_millis(60)));
        assert!(dynamic_color_moving(
            index,
            start + Duration::from_millis(60)
        ));
        refresh_dynamic_colors(end);
        assert_eq!(handle.rgba().g, 1.0);
        assert!(!dynamic_colors_moving(end));
        assert!(!dynamic_color_moving(index, end));
        TABLE.lock().source = None;
    }

    #[test]
    fn a_present_behind_the_last_refresh_leaves_the_table() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let index = handle.index();
        let start = Instant::now();
        set_color_source(move |at, colors, _| {
            let t = at.saturating_duration_since(start).as_secs_f32();
            colors[index] = Rgba {
                r: t,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            };
        });
        refresh_dynamic_colors(start + Duration::from_millis(500));
        assert!((handle.rgba().r - 0.5).abs() < 1e-3);
        refresh_dynamic_colors(start + Duration::from_millis(250));
        assert!(
            (handle.rgba().r - 0.5).abs() < 1e-3,
            "an older present does not step the colours back"
        );
        refresh_dynamic_colors(start + Duration::from_millis(750));
        assert!((handle.rgba().r - 0.75).abs() < 1e-3);
        TABLE.lock().source = None;
    }

    #[test]
    fn a_present_behind_the_last_refresh_hears_a_new_source() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let index = handle.index();
        let start = Instant::now();
        let source = move |level: f32| {
            move |at: Instant, colors: &mut [Rgba], _: &mut [Option<Instant>]| {
                let t = at.saturating_duration_since(start).as_secs_f32();
                colors[index] = Rgba {
                    r: t,
                    g: level,
                    b: 0.0,
                    a: 1.0,
                };
            }
        };
        set_color_source(source(0.0));
        refresh_dynamic_colors(start + Duration::from_millis(500));
        set_color_source(source(1.0));
        refresh_dynamic_colors(start + Duration::from_millis(250));
        let rgba = handle.rgba();
        assert_eq!(rgba.g, 1.0, "the new source is heard");
        assert!(
            (rgba.r - 0.5).abs() < 1e-3,
            "at the later present, not the older one"
        );
        TABLE.lock().source = None;
    }

    #[test]
    fn concurrent_refreshes_keep_the_latest_present() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let index = handle.index();
        let start = Instant::now();
        set_color_source(move |at, colors, _| {
            let t = at.saturating_duration_since(start).as_secs_f32();
            // Long enough for another refresh to start meanwhile.
            std::thread::yield_now();
            colors[index] = Rgba {
                r: t,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            };
        });
        let threads: Vec<_> = (1..=8u64)
            .map(|thread| {
                std::thread::spawn(move || {
                    for step in 0..50u64 {
                        refresh_dynamic_colors(start + Duration::from_millis(step * 8 + thread));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let last = *REFRESHED.lock();
        let last = last
            .expect("refreshed")
            .saturating_duration_since(start)
            .as_secs_f32();
        assert!(
            (handle.rgba().r - last).abs() < 1e-6,
            "the table holds the latest present's colours"
        );
        TABLE.lock().source = None;
    }

    #[test]
    fn a_claimed_entry_starts_at_rest() {
        let _serial = serial();
        let handle = ColorHandle::claim(red()).expect("a free entry");
        let index = handle.index();
        let start = Instant::now();
        let end = start + Duration::from_secs(3600);
        set_color_source(move |_, _, until| until[index] = Some(end));
        refresh_dynamic_colors(start);
        TABLE.lock().source = None;
        drop(handle);
        let again = ColorHandle::claim(red()).expect("the freed entry");
        assert_eq!(again.index(), index);
        assert!(!dynamic_color_moving(index, start));
    }

    #[test]
    fn a_set_moves_only_while_an_entry_it_holds_moves() {
        let _serial = serial();
        let still = ColorHandle::claim(red()).expect("a free entry");
        let turning = ColorHandle::claim(red()).expect("a free entry");
        let turning_index = turning.index();
        let start = Instant::now();
        let end = start + Duration::from_millis(100);
        set_color_source(move |at, _, until| {
            until[turning_index] = (at < end).then_some(end);
        });
        let mut held_still = DynamicColorSet::default();
        held_still.note(&still.hsla(1.0));
        let mut held_turning = held_still.clone();
        held_turning.note(&turning.hsla(1.0));
        let during = start + Duration::from_millis(50);
        refresh_dynamic_colors(during);
        assert!(dynamic_colors_moving(during));
        assert!(!dynamic_color_moving(still.index(), during));
        assert!(
            !held_still.moving(during),
            "a set of entries at rest stays at rest while another entry moves"
        );
        assert!(held_turning.moving(during));
        refresh_dynamic_colors(end);
        assert!(!held_turning.moving(end));
        TABLE.lock().source = None;
    }

    #[test]
    fn uniforms_stay_at_rest_while_an_entry_they_do_not_name_moves() {
        let _serial = serial();
        let still = ColorHandle::claim(red()).expect("a free entry");
        let turning = ColorHandle::claim(red()).expect("a free entry");
        let turning_index = turning.index();
        let start = Instant::now();
        let end = start + Duration::from_millis(100);
        set_color_source(move |at, _, until| {
            until[turning_index] = (at < end).then_some(end);
        });
        let uniforms = crate::ShaderUniforms::new();
        uniforms.set_color(0, still.hsla(1.0));
        let during = start + Duration::from_millis(50);
        refresh_dynamic_colors(during);
        assert!(uniforms.take_present(during), "the colour was just set");
        assert!(!uniforms.is_live(during));
        assert!(!uniforms.take_present(during));
        uniforms.set_color(1, turning.hsla(1.0));
        assert!(uniforms.take_present(during), "the colour was just set");
        assert!(uniforms.is_live(during), "an entry the uniforms name moves");
        assert!(
            uniforms.take_present(during),
            "presented while an entry it names moves, with nothing written"
        );
        refresh_dynamic_colors(end);
        assert!(uniforms.take_present(end), "the present landing it");
        assert!(!uniforms.is_live(end));
        TABLE.lock().source = None;
    }

    #[test]
    fn a_changed_entry_is_found() {
        let mut before = Vec::new();
        dynamic_colors_frame(&mut before);
        let mut after = before.clone();
        after[4 * 7 + 1] += 0.5;
        let changed = changed_dynamic_colors(&before, &after);
        let mut seven = DynamicColorSet::default();
        seven.insert(7);
        assert!(changed.intersects(&seven));
        let mut eight = DynamicColorSet::default();
        eight.insert(8);
        assert!(!changed.intersects(&eight));
    }
}
