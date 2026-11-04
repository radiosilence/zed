use objc2::{
    AnyThread, DefinedClass, MainThreadMarker, MainThreadOnly, Message, define_class, msg_send,
    rc::Retained,
    runtime::{AnyClass, ProtocolObject},
};
use objc2_foundation::{
    NSDefaultRunLoopMode, NSObject, NSObjectProtocol, NSPort, NSRunLoop, NSRunLoopCommonModes,
};
use objc2_quartz_core::{
    CAMetalDisplayLink, CAMetalDisplayLinkDelegate, CAMetalDisplayLinkUpdate, CAMetalLayer,
};
use std::sync::OnceLock;

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

/// The run loop of the thread that presents frames off the main thread,
/// started the first time a link is put on it and kept for the process.
fn present_run_loop() -> &'static NSRunLoop {
    struct RunLoop(Retained<NSRunLoop>);
    // SAFETY: the run loop is only handed to `addToRunLoop:forMode:`, which
    // adds the link as a source of it; run loop sources may be added from
    // any thread.
    unsafe impl Send for RunLoop {}
    // SAFETY: as for `Send`.
    unsafe impl Sync for RunLoop {}

    static RUN_LOOP: OnceLock<RunLoop> = OnceLock::new();
    &RUN_LOOP
        .get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            std::thread::Builder::new()
                .name("gpui-present".into())
                .spawn(move || {
                    let run_loop = NSRunLoop::currentRunLoop();
                    // A port keeps the run loop waiting while no link is on it.
                    let port = NSPort::port();
                    // SAFETY: the port and mode are valid, and this is the
                    // run loop's own thread.
                    unsafe { run_loop.addPort_forMode(&port, NSDefaultRunLoopMode) };
                    sender.send(RunLoop(run_loop.clone())).ok();
                    run_loop.run();
                })
                .expect("spawning the present thread");
            receiver.recv().expect("the present thread's run loop")
        })
        .0
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
    _delegate: Retained<NSObject>,
    _main_thread: MainThreadMarker,
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
            _delegate: Retained::into_super(delegate),
            _main_thread: main_thread,
        })
    }

    /// Starts delivering updates for `layer` on a thread presenting frames
    /// off the main thread, shared by every such link, or returns `None`
    /// where `CAMetalDisplayLink` is unavailable. The callback hands the
    /// main thread whatever it does not present itself.
    pub fn off_main(
        layer: &CAMetalLayer,
        main_thread: MainThreadMarker,
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
        unsafe { link.addToRunLoop_forMode(present_run_loop(), NSRunLoopCommonModes) };
        Some(Self {
            link,
            _delegate: Retained::into_super(delegate),
            _main_thread: main_thread,
        })
    }

    /// A paused link delivers no updates and acquires no drawables.
    pub fn set_paused(&self, paused: bool) {
        self.link.setPaused(paused);
    }
}

impl Drop for MetalFrameSource {
    fn drop(&mut self) {
        self.link.invalidate();
    }
}
