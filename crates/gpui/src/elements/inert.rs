use crate::{
    AnyElement, App, Bounds, Element, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Window,
};

/// Builds an `Inert` element, which draws its child as it is but lets it take
/// no input.
pub fn inert(child: impl IntoElement) -> Inert {
    Inert {
        child: child.into_any_element(),
    }
}

/// An element drawn inert (see `Window::with_inert`): laid out and painted as
/// it is, inserting no hitboxes and no tab stops. The pointer passes through
/// it to what lies beneath, occluding elements inside it included, and the
/// keyboard never tabs into it. For something seen on its way out, such as a
/// dialog fading after it was dismissed, whose controls must not act again.
pub struct Inert {
    child: AnyElement,
}

impl Element for Inert {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_inert(|window| self.child.prepaint(window, cx));
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_inert(|window| self.child.paint(window, cx));
    }
}

impl IntoElement for Inert {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::{
        Context, FocusHandle, InteractiveElement, IntoElement, Modifiers, ParentElement, Render,
        StatefulInteractiveElement, Styled, TestAppContext, Window, div, inert, point, px,
    };

    /// A clickable, focusable element under an overlay that occludes the
    /// whole window and holds a focusable element of its own, drawn inert
    /// or not.
    struct Covered {
        under: FocusHandle,
        over: FocusHandle,
        inert: bool,
        clicks: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Render for Covered {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let (under_clicks, over_clicks) = (self.clicks.clone(), self.clicks.clone());
            let over_keys = self.clicks.clone();
            let overlay = div().id("overlay").absolute().inset_0().occlude().child(
                div()
                    .id("over")
                    .size(px(50.))
                    .track_focus(&self.over)
                    .on_key_down(move |_, _, _| over_keys.borrow_mut().push("key"))
                    .on_click(move |_, _, _| over_clicks.borrow_mut().push("over")),
            );
            let overlay = if self.inert {
                inert(overlay).into_any_element()
            } else {
                overlay.into_any_element()
            };
            div()
                .size_full()
                .child(
                    div()
                        .id("under")
                        .size(px(50.))
                        .track_focus(&self.under)
                        .on_click(move |_, _, _| under_clicks.borrow_mut().push("under")),
                )
                .child(overlay)
        }
    }

    fn covered(
        inert: bool,
    ) -> (
        crate::VisualTestContext,
        Rc<RefCell<Vec<&'static str>>>,
        FocusHandle,
        FocusHandle,
    ) {
        let mut cx = TestAppContext::single();
        let (under, over) = cx.update(|cx| {
            (
                cx.focus_handle().tab_stop(true),
                cx.focus_handle().tab_stop(true),
            )
        });
        let clicks = Rc::new(RefCell::new(Vec::new()));
        let window = cx.add_window({
            let (under, over, clicks) = (under.clone(), over.clone(), clicks.clone());
            move |_, _| Covered {
                under,
                over,
                inert,
                clicks,
            }
        });
        let cx = crate::VisualTestContext::from_window(window.into(), &cx);
        cx.run_until_parked();
        (cx, clicks, under, over)
    }

    #[test]
    fn the_pointer_passes_through_an_inert_overlay() {
        let (mut cx, clicks, _, _) = covered(true);
        cx.simulate_click(point(px(25.), px(25.)), Modifiers::default());
        assert_eq!(*clicks.borrow(), vec!["under"]);
    }

    #[test]
    fn a_live_overlay_takes_the_pointer() {
        let (mut cx, clicks, _, _) = covered(false);
        cx.simulate_click(point(px(25.), px(25.)), Modifiers::default());
        assert_eq!(*clicks.borrow(), vec!["over"]);
    }

    /// Tabbing from nothing round the window's stops twice.
    fn tabbed_to(cx: &mut crate::VisualTestContext, handle: &FocusHandle) -> bool {
        (0..4).any(|_| {
            cx.update(|window, cx| window.focus_next(cx));
            cx.run_until_parked();
            cx.update(|window, _| handle.is_focused(window))
        })
    }

    #[test]
    fn tab_never_enters_an_inert_overlay() {
        let (mut cx, _, under, over) = covered(true);
        assert!(!tabbed_to(&mut cx, &over), "tab entered the inert overlay");
        assert!(tabbed_to(&mut cx, &under), "tab skipped what is beneath");
    }

    /// Focus left in an overlay when it went inert takes no keys.
    fn typed_into_over(inert: bool) -> Vec<&'static str> {
        let (mut cx, clicks, _, over) = covered(inert);
        cx.update(|window, cx| window.focus(&over, cx));
        cx.run_until_parked();
        cx.simulate_keystrokes("a");
        clicks.borrow().clone()
    }

    #[test]
    fn a_focused_element_in_an_inert_overlay_takes_no_keys() {
        assert_eq!(typed_into_over(true), Vec::<&str>::new());
        assert_eq!(typed_into_over(false), vec!["key"]);
    }

    #[test]
    fn tab_enters_a_live_overlay() {
        let (mut cx, _, _, over) = covered(false);
        assert!(tabbed_to(&mut cx, &over));
    }
}
