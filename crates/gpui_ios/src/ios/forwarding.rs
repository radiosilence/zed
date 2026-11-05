//! The application's own `UIApplicationDelegate`, which receives the delegate
//! messages GPUI's delegate does not implement.

use objc2::{
    Message,
    rc::Retained,
    runtime::{AnyObject, NSObjectProtocol, ProtocolObject, Sel},
};
use objc2_ui_kit::UIApplicationDelegate;
use std::cell::RefCell;

thread_local! {
    static FORWARDED: RefCell<Option<Retained<ProtocolObject<dyn UIApplicationDelegate>>>> =
        const { RefCell::new(None) };
}

/// Sends `delegate` every `UIApplicationDelegate` message that GPUI's
/// application delegate does not implement itself, such as remote
/// notification registration and background fetches, and the launch options
/// GPUI receives in `application:didFinishLaunchingWithOptions:`.
///
/// Call it on the main thread before running the application: UIKit asks
/// its delegate which messages it responds to when the delegate is installed,
/// and does not ask again.
pub fn forward_application_delegate(delegate: &ProtocolObject<dyn UIApplicationDelegate>) {
    FORWARDED.with_borrow_mut(|forwarded| *forwarded = Some(delegate.retain()));
}

/// The forwarded delegate, when it responds to `selector`.
pub(crate) fn forwarded_for(
    selector: Sel,
) -> Option<Retained<ProtocolObject<dyn UIApplicationDelegate>>> {
    FORWARDED.with_borrow(|forwarded| {
        forwarded
            .as_ref()
            .filter(|delegate| delegate.respondsToSelector(selector))
            .cloned()
    })
}

/// The object a message GPUI does not handle goes to, or null to let the
/// runtime raise as it would without forwarding.
pub(crate) fn forwarding_target(selector: Sel) -> *mut AnyObject {
    forwarded_for(selector).map_or(std::ptr::null_mut(), |delegate| {
        Retained::as_ptr(&delegate).cast::<AnyObject>().cast_mut()
    })
}
