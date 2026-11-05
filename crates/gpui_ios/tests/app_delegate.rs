#[cfg(target_os = "ios")]
#[path = "../src/ios/forwarding.rs"]
mod forwarding;

#[cfg(target_os = "ios")]
mod fixture {
    use objc2::{
        MainThreadMarker, MainThreadOnly, define_class, msg_send,
        rc::Retained,
        runtime::{NSObject, NSObjectProtocol, ProtocolObject},
        sel,
    };
    use objc2_foundation::NSData;
    use objc2_ui_kit::{UIApplication, UIApplicationDelegate};

    define_class!(
        #[unsafe(super = NSObject)]
        #[thread_kind = MainThreadOnly]
        #[name = "GPUIIosForwardedDelegateFixture"]
        pub(crate) struct Forwarded;

        unsafe impl NSObjectProtocol for Forwarded {}

        unsafe impl UIApplicationDelegate for Forwarded {
            #[unsafe(method(application:didRegisterForRemoteNotificationsWithDeviceToken:))]
            fn registered(&self, _application: &UIApplication, _token: &NSData) {}
        }
    );

    pub(crate) fn forwards_only_what_the_delegate_implements(main_thread: MainThreadMarker) {
        let registered = sel!(application:didRegisterForRemoteNotificationsWithDeviceToken:);
        let fetched = sel!(application:didReceiveRemoteNotification:fetchCompletionHandler:);
        assert!(super::forwarding::forwarded_for(registered).is_none());
        assert!(super::forwarding::forwarding_target(registered).is_null());

        let delegate: Retained<Forwarded> =
            unsafe { msg_send![Forwarded::alloc(main_thread), init] };
        super::forwarding::forward_application_delegate(ProtocolObject::from_ref(&*delegate));

        let target = super::forwarding::forwarding_target(registered);
        assert_eq!(target.cast_const(), Retained::as_ptr(&delegate).cast());
        assert!(super::forwarding::forwarded_for(fetched).is_none());
        assert!(super::forwarding::forwarding_target(fetched).is_null());
    }
}

fn main() {
    #[cfg(target_os = "ios")]
    {
        let main_thread = objc2::MainThreadMarker::new().expect("native test runs on main thread");
        fixture::forwards_only_what_the_delegate_implements(main_thread);
        println!("application delegate forwarding test passed");
    }
}
