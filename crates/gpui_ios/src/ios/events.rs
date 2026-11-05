//! Conversion from UIKit events to GPUI input events.

use gpui::{
    BackGestureEvent, BackGesturePhase, Keystroke, Modifiers, Pixels, Point, ScrollDelta,
    ScrollEvent, ScrollSource, TouchId, TouchPhase, px,
};
use objc2_core_foundation::CGPoint;
use objc2_ui_kit::{
    UIGestureRecognizerState, UIKeyModifierFlags, UIKeyboardHIDUsage, UIPanGestureRecognizer,
    UIRectEdge, UIScreenEdgePanGestureRecognizer, UITouch, UITouchPhase, UIView,
};

/// A release faster than this, in points per second toward going back, commits
/// a back gesture regardless of how far it travelled.
const BACK_COMMIT_VELOCITY: f64 = 300.;

/// Converts a step of the screen-edge back gesture, whose `Started` phase was
/// already delivered when the recognizer asked to begin.
///
/// Like `UINavigationController`, a release commits when the gesture is past
/// halfway without flicking back, or when it flicks forward quickly. Returns
/// `None` for states that carry no step.
pub fn back_event(
    recognizer: &UIScreenEdgePanGestureRecognizer,
    view: &UIView,
    start_position: Point<Pixels>,
) -> Option<BackGestureEvent> {
    // Gestures from the right edge move content leftward toward going back.
    let direction = if recognizer.edges() == UIRectEdge::Right {
        -1.
    } else {
        1.
    };
    let width = view.bounds().size.width.max(1.);
    let travel = recognizer.translationInView(Some(view)).x * direction;
    let progress = (travel / width).clamp(0., 1.);
    let phase = match recognizer.state() {
        UIGestureRecognizerState::Began | UIGestureRecognizerState::Changed => {
            BackGesturePhase::Progressed
        }
        UIGestureRecognizerState::Ended => {
            let velocity = recognizer.velocityInView(Some(view)).x * direction;
            if velocity > BACK_COMMIT_VELOCITY
                || (progress > 0.5 && velocity > -BACK_COMMIT_VELOCITY)
            {
                BackGesturePhase::Committed
            } else {
                BackGesturePhase::Cancelled
            }
        }
        UIGestureRecognizerState::Cancelled | UIGestureRecognizerState::Failed => {
            BackGesturePhase::Cancelled
        }
        _ => return None,
    };
    Some(BackGestureEvent {
        phase,
        start_position,
        progress: progress as f32,
    })
}

pub fn touch_location_in_view(touch: &UITouch, view: &UIView) -> Point<Pixels> {
    let location = touch.locationInView(Some(view));
    Point::new(px(location.x as f32), px(location.y as f32))
}

pub fn touch_phase(touch: &UITouch) -> TouchPhase {
    match touch.phase() {
        UITouchPhase::Began => TouchPhase::Started,
        UITouchPhase::Moved | UITouchPhase::Stationary => TouchPhase::Moved,
        UITouchPhase::Ended => TouchPhase::Ended,
        _ => TouchPhase::Cancelled,
    }
}

pub fn touch_id(touch: &UITouch) -> TouchId {
    TouchId(touch as *const UITouch as usize as u64)
}

/// Converts one step of a trackpad or mouse-wheel scroll gesture.
///
/// The recognizer's translation is reset after each step, so every event carries
/// only the movement since the previous one. UIKit reports the translation in
/// the direction the content should move, which matches GPUI's pixel deltas.
/// Returns `None` for states that carry no scroll.
pub fn scroll_event(recognizer: &UIPanGestureRecognizer, view: &UIView) -> Option<ScrollEvent> {
    let touch_phase = match recognizer.state() {
        UIGestureRecognizerState::Began => TouchPhase::Started,
        UIGestureRecognizerState::Changed => TouchPhase::Moved,
        UIGestureRecognizerState::Ended => TouchPhase::Ended,
        UIGestureRecognizerState::Cancelled | UIGestureRecognizerState::Failed => {
            TouchPhase::Cancelled
        }
        _ => return None,
    };
    let translation = recognizer.translationInView(Some(view));
    recognizer.setTranslation_inView(CGPoint::ZERO, Some(view));
    let location = recognizer.locationInView(Some(view));
    Some(ScrollEvent {
        position: Point::new(px(location.x as f32), px(location.y as f32)),
        delta: ScrollDelta::Pixels(Point::new(
            px(translation.x as f32),
            px(translation.y as f32),
        )),
        modifiers: modifiers(recognizer.modifierFlags()),
        touch_phase,
        source: ScrollSource::Pointer,
    })
}

pub(crate) fn modifiers(flags: UIKeyModifierFlags) -> Modifiers {
    Modifiers {
        control: flags.contains(UIKeyModifierFlags::Control),
        alt: flags.contains(UIKeyModifierFlags::Alternate),
        shift: flags.contains(UIKeyModifierFlags::Shift),
        platform: flags.contains(UIKeyModifierFlags::Command),
        function: false,
    }
}

pub fn touch_force(touch: &UITouch) -> Option<f32> {
    let maximum_force = touch.maximumPossibleForce();
    if maximum_force <= 0.0 {
        return None;
    }
    Some((touch.force() / maximum_force).clamp(0.0, 1.0) as f32)
}

/// The keystroke a hardware key press is to GPUI's bindings, from the key's
/// HID usage, the characters it types with and without its modifiers, and
/// the modifiers held. A modifier key alone is none.
pub fn keystroke(
    code: UIKeyboardHIDUsage,
    characters: &str,
    characters_ignoring_modifiers: &str,
    flags: UIKeyModifierFlags,
) -> Option<Keystroke> {
    let named = match code {
        UIKeyboardHIDUsage::KeyboardReturnOrEnter | UIKeyboardHIDUsage::KeypadEnter => "enter",
        UIKeyboardHIDUsage::KeyboardEscape => "escape",
        UIKeyboardHIDUsage::KeyboardDeleteOrBackspace => "backspace",
        UIKeyboardHIDUsage::KeyboardDeleteForward => "delete",
        UIKeyboardHIDUsage::KeyboardTab => "tab",
        UIKeyboardHIDUsage::KeyboardSpacebar => "space",
        UIKeyboardHIDUsage::KeyboardLeftArrow => "left",
        UIKeyboardHIDUsage::KeyboardRightArrow => "right",
        UIKeyboardHIDUsage::KeyboardUpArrow => "up",
        UIKeyboardHIDUsage::KeyboardDownArrow => "down",
        UIKeyboardHIDUsage::KeyboardHome => "home",
        UIKeyboardHIDUsage::KeyboardEnd => "end",
        UIKeyboardHIDUsage::KeyboardPageUp => "pageup",
        UIKeyboardHIDUsage::KeyboardPageDown => "pagedown",
        _ => "",
    };
    let function = (UIKeyboardHIDUsage::KeyboardF1.0..=UIKeyboardHIDUsage::KeyboardF12.0)
        .contains(&code.0)
        .then(|| format!("f{}", code.0 - UIKeyboardHIDUsage::KeyboardF1.0 + 1));
    let modifier_key = (UIKeyboardHIDUsage::KeyboardLeftControl.0
        ..=UIKeyboardHIDUsage::KeyboardRightGUI.0)
        .contains(&code.0);
    let mut modifiers = modifiers(flags);
    let key = if !named.is_empty() {
        named.to_string()
    } else if let Some(function) = function {
        function
    } else if modifier_key || characters_ignoring_modifiers.is_empty() {
        return None;
    } else if modifiers.shift
        && !characters_ignoring_modifiers
            .chars()
            .all(|c| c.is_ascii_alphabetic())
    {
        // As macOS reports it: a shifted symbol is the symbol, without shift
        // (`?`, not `shift-/`), so bindings written as the symbol match.
        modifiers.shift = false;
        let shifted = if characters.is_empty() {
            characters_ignoring_modifiers
        } else {
            characters
        };
        shifted.to_string()
    } else {
        characters_ignoring_modifiers.to_lowercase()
    };
    let types = !modifiers.control && !modifiers.platform;
    let key_char = (types && (named.is_empty() || named == "space"))
        .then(|| characters.to_string())
        .filter(|typed| !typed.is_empty() && !typed.chars().any(char::is_control));
    Some(Keystroke {
        modifiers,
        key,
        key_char,
    })
}

#[cfg(test)]
mod key_tests {
    use super::*;

    #[test]
    fn named_keys_and_shortcuts_become_gpui_keystrokes() {
        let none = UIKeyModifierFlags::empty();
        let enter =
            keystroke(UIKeyboardHIDUsage::KeyboardReturnOrEnter, "\r", "\r", none).expect("a key");
        assert_eq!((enter.key.as_str(), enter.key_char), ("enter", None));

        let find = keystroke(
            UIKeyboardHIDUsage::KeyboardK,
            "k",
            "k",
            UIKeyModifierFlags::Command,
        )
        .expect("a key");
        assert_eq!(find.key, "k");
        assert!(find.modifiers.platform);
        assert_eq!(find.key_char, None, "a shortcut types nothing");

        let shifted = keystroke(
            UIKeyboardHIDUsage::KeyboardA,
            "A",
            "A",
            UIKeyModifierFlags::Shift,
        )
        .expect("a key");
        assert_eq!(shifted.key, "a");
        assert!(shifted.modifiers.shift);
        assert_eq!(shifted.key_char.as_deref(), Some("A"));

        let question = keystroke(
            UIKeyboardHIDUsage::KeyboardSlash,
            "?",
            "/",
            UIKeyModifierFlags::Shift,
        )
        .expect("a key");
        assert_eq!(question.key, "?", "a shifted symbol is the symbol");
        assert!(!question.modifiers.shift);

        let f5 = keystroke(UIKeyboardHIDUsage::KeyboardF5, "", "", none).expect("a key");
        assert_eq!(f5.key, "f5");

        assert!(
            keystroke(
                UIKeyboardHIDUsage::KeyboardLeftShift,
                "",
                "",
                UIKeyModifierFlags::Shift
            )
            .is_none(),
            "a modifier alone is no keystroke"
        );
    }
}

/// When the hardware sampled `touch`, as an [`Instant`](std::time::Instant).
/// UIKit times touches in seconds of system uptime, the clock `Instant`
/// reads on Apple platforms (`CLOCK_UPTIME_RAW`).
pub fn touch_timestamp(touch: &UITouch) -> Option<std::time::Instant> {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to write.
    if unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut now) } != 0 {
        return None;
    }
    let uptime = now.tv_sec as f64 + now.tv_nsec as f64 / 1e9;
    let age = (uptime - touch.timestamp()).max(0.0);
    std::time::Instant::now().checked_sub(std::time::Duration::from_secs_f64(age))
}
