use block2::RcBlock;
use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, Message, define_class, msg_send,
    rc::Retained,
    runtime::{AnyClass, ProtocolObject},
};
use objc2_core_foundation::{CFRetained, CFRunLoop, kCFRunLoopCommonModes};
use objc2_foundation::{
    NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSPort, NSRunLoop, NSRunLoopCommonModes,
};
use objc2_quartz_core::{
    CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate, CAMetalLayer,
};
use std::{cell::Cell, marker::PhantomData, sync::OnceLock};

struct MetalDisplayLinkDelegateIvars {
    callback: Box<dyn Fn(&CAMetalDisplayLinkUpdate)>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = MetalDisplayLinkDelegateIvars]
    struct MetalDisplayLinkDelegate;

    unsafe impl NSObjectProtocol for MetalDisplayLinkDelegate {}

    unsafe impl CAMetalDisplayLinkDelegate for MetalDisplayLinkDelegate {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn needs_update(&self, _link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            // The callback may destroy its frame source, releasing our owner's retain.
            let delegate = self.retain();
            (delegate.ivars().callback)(update);
        }
    }
);

impl MetalDisplayLinkDelegate {
    fn new(
        main_thread: MainThreadMarker,
        callback: impl Fn(&CAMetalDisplayLinkUpdate) + 'static,
    ) -> Retained<Self> {
        let this = Self::alloc(main_thread).set_ivars(MetalDisplayLinkDelegateIvars {
            callback: Box::new(callback),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }
}

struct PresentThreadDelegateIvars {
    callback: Box<dyn Fn(&CAMetalDisplayLinkUpdate) + Send + Sync>,
}

define_class!(
    // SAFETY: `NSObject` has no subclassing requirements.
    #[unsafe(super(NSObject))]
    #[ivars = PresentThreadDelegateIvars]
    struct PresentThreadDelegate;

    unsafe impl NSObjectProtocol for PresentThreadDelegate {}

    unsafe impl CAMetalDisplayLinkDelegate for PresentThreadDelegate {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn needs_update(&self, _link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            // The frame source may be dropped while this runs.
            let delegate = self.retain();
            (delegate.ivars().callback)(update);
        }
    }
);

impl PresentThreadDelegate {
    fn new(callback: impl Fn(&CAMetalDisplayLinkUpdate) + Send + Sync + 'static) -> Retained<Self> {
        let this = Self::alloc().set_ivars(PresentThreadDelegateIvars {
            callback: Box::new(callback),
        });
        // SAFETY: `NSObject`'s `init` is its designated initializer.
        unsafe { msg_send![super(this), init] }
    }
}

/// The thread that presents frames off the main thread, started the first
/// time a link is put on it and kept for the process.
struct PresentThread {
    run_loop: Retained<NSRunLoop>,
    cf_run_loop: CFRetained<CFRunLoop>,
}

// SAFETY: the run loop is only handed to `addToRunLoop:forMode:`, which adds
// the link as a source of it, and to `CFRunLoopPerformBlock` and
// `CFRunLoopWakeUp`; all three may be called from any thread.
unsafe impl Send for PresentThread {}
// SAFETY: as for `Send`.
unsafe impl Sync for PresentThread {}

impl PresentThread {
    fn get() -> &'static Self {
        static THREAD: OnceLock<PresentThread> = OnceLock::new();
        THREAD.get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            std::thread::Builder::new()
                .name("gpui-present".into())
                .spawn(move || {
                    let run_loop = NSRunLoop::currentRunLoop();
                    let cf_run_loop = CFRunLoop::current().expect("the thread's run loop");
                    // A port keeps the run loop waiting while no link is on it.
                    let port = NSPort::port();
                    // SAFETY: the port and mode are valid, and this is the
                    // run loop's own thread.
                    unsafe { run_loop.addPort_forMode(&port, NSDefaultRunLoopMode) };
                    sender
                        .send(PresentThread {
                            run_loop: run_loop.clone(),
                            cf_run_loop,
                        })
                        .ok();
                    run_loop.run();
                })
                .expect("spawning the present thread");
            receiver.recv().expect("the present thread's run loop")
        })
    }

    /// Runs `work` on the present thread as its run loop next turns, after
    /// any work asked for before it.
    fn perform(&self, work: impl FnOnce() + 'static) {
        let work = Cell::new(Some(work));
        let block = RcBlock::new(move || {
            if let Some(work) = work.take() {
                work();
            }
        });
        // SAFETY: the mode is the run loop's common modes, which its default
        // mode is among, and the run loop copies the block. The block runs
        // once, on the present thread, which is the only place its captures
        // are used.
        unsafe {
            self.cf_run_loop.perform_block(
                kCFRunLoopCommonModes.map(|mode| mode.as_ref()),
                Some(&block),
            );
        }
        self.cf_run_loop.wake_up();
    }
}

/// A source of Metal drawables on macOS 14+ and iOS/iPadOS 17+.
///
/// Callbacks run on the run loop the link is registered with:
/// <https://developer.apple.com/documentation/quartzcore/cametaldisplaylink>.
/// [`MetalFrameSource::new`] uses the main run loop, for a renderer drawing
/// each frame on the foreground thread; [`MetalFrameSource::off_main`] a
/// thread of its own, for presenting frames there.
pub struct MetalFrameSource {
    link: Retained<CAMetalDisplayLink>,
    // The link holds its delegate weakly.
    delegate: Retained<NSObject>,
    /// Whether the link runs on the present thread, where it is then paused
    /// and invalidated, never from the main thread under its updates.
    on_present_thread: bool,
    _not_send: PhantomData<*const ()>,
}

impl MetalFrameSource {
    /// Whether this system has `CAMetalDisplayLink` (macOS 14+, iOS/iPadOS 17+).
    pub fn is_available() -> bool {
        AnyClass::get(c"CAMetalDisplayLink").is_some()
    }

    /// Starts delivering updates for `layer` on the main thread, or returns
    /// `None` where `CAMetalDisplayLink` is unavailable.
    pub fn new(
        layer: &CAMetalLayer,
        main_thread: MainThreadMarker,
        callback: impl Fn(&CAMetalDisplayLinkUpdate) + 'static,
    ) -> Option<Self> {
        if !Self::is_available() {
            return None;
        }
        let delegate = MetalDisplayLinkDelegate::new(main_thread, callback);
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
        link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        // The default range follows the display and system power policy:
        // https://developer.apple.com/documentation/quartzcore/cametaldisplaylink/preferredframeraterange
        // SAFETY: the retained marker confines the source to the main thread,
        // and the main run loop outlives the source.
        unsafe { link.addToRunLoop_forMode(&NSRunLoop::mainRunLoop(), NSRunLoopCommonModes) };
        Some(Self {
            link,
            delegate: Retained::into_super(delegate),
            on_present_thread: false,
            _not_send: PhantomData,
        })
    }

    /// Starts delivering updates for `layer` on a thread presenting frames
    /// off the main thread, shared by every such link, or returns `None`
    /// where `CAMetalDisplayLink` is unavailable. The callback hands the
    /// main thread whatever it does not present itself.
    pub fn off_main(
        layer: &CAMetalLayer,
        callback: impl Fn(&CAMetalDisplayLinkUpdate) + Send + Sync + 'static,
    ) -> Option<Self> {
        if !Self::is_available() {
            return None;
        }
        let delegate = PresentThreadDelegate::new(callback);
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), layer);
        link.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        // SAFETY: the present thread's run loop runs for the life of the
        // process, and its sources may be added from any thread.
        unsafe { link.addToRunLoop_forMode(&PresentThread::get().run_loop, NSRunLoopCommonModes) };
        Some(Self {
            link,
            delegate: Retained::into_super(delegate),
            on_present_thread: true,
            _not_send: PhantomData,
        })
    }

    /// A paused link delivers no updates and acquires no drawables. A link
    /// on the present thread is paused there, in the order asked.
    pub fn set_paused(&self, paused: bool) {
        if self.on_present_thread {
            let link = self.link.clone();
            PresentThread::get().perform(move || link.setPaused(paused));
        } else {
            self.link.setPaused(paused);
        }
    }
}

impl Drop for MetalFrameSource {
    fn drop(&mut self) {
        if self.on_present_thread {
            // The delegate lives until the link is invalidated, on the
            // thread its updates run on, so none runs without it.
            let link = self.link.clone();
            let delegate = self.delegate.clone();
            PresentThread::get().perform(move || {
                link.invalidate();
                drop(delegate);
            });
        } else {
            self.link.invalidate();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    /// Runs `work` on the present thread and waits for it.
    fn on_present_thread<R: Send + 'static>(work: impl FnOnce() -> R + Send + 'static) -> R {
        let (sender, receiver) = std::sync::mpsc::channel();
        PresentThread::get().perform(move || sender.send(work()).unwrap());
        receiver.recv().unwrap()
    }

    // A link on the present thread is paused there, after the work asked of
    // the thread before it, not from the thread asking.
    #[test]
    fn a_present_thread_link_is_paused_on_the_present_thread() {
        if !MetalFrameSource::is_available() {
            return;
        }
        let layer = CAMetalLayer::new();
        layer.setDevice(MTLCreateSystemDefaultDevice().as_deref());
        let source = MetalFrameSource::off_main(&layer, |_| {}).expect("a link");
        // What the present thread was asked before this has run.
        let settled = || on_present_thread(|| ());
        settled();
        assert!(!source.link.isPaused());

        let (release, wait) = std::sync::mpsc::channel::<()>();
        PresentThread::get().perform(move || wait.recv().unwrap());
        source.set_paused(true);
        assert!(
            !source.link.isPaused(),
            "the pause waits for the present thread"
        );
        release.send(()).unwrap();
        settled();
        assert!(source.link.isPaused());
        source.set_paused(false);
        settled();
        assert!(!source.link.isPaused());
        assert_eq!(
            on_present_thread(|| std::thread::current().name().map(str::to_owned)),
            Some("gpui-present".to_owned())
        );
        drop(source);
    }
}
