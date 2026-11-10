//! The system's document picker, behind `Platform::prompt_for_paths`.
//!
//! Each file is handed over as a copy in the app's own temporary storage, so
//! its path stays readable without a security-scoped grant. UIKit picks either
//! files or folders, never both at once, and a folder copied whole is not on
//! offer, so a prompt for files shows files, and a prompt for folders alone is
//! refused.

use anyhow::anyhow;
use futures::channel::oneshot;
use gpui::{PathPromptOptions, Result};
use objc2::rc::Retained;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_foundation::{NSArray, NSObject, NSObjectProtocol, NSURL};
use objc2_ui_kit::{UIDocumentPickerDelegate, UIDocumentPickerViewController, UIViewController};
use objc2_uniform_type_identifiers::UTTypeData;
use std::{cell::RefCell, path::PathBuf};

type Answer = oneshot::Sender<Result<Option<Vec<PathBuf>>>>;

thread_local! {
    /// The picker's delegate while it is up: UIKit holds delegates weakly.
    /// A later prompt replaces it, which cancels the earlier one's answer.
    static SHOWN: RefCell<Option<Retained<PickerDelegate>>> = const { RefCell::new(None) };
}

struct PickerIvars {
    answer: RefCell<Option<Answer>>,
}

define_class!(
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    #[name = "GPUIDocumentPickerDelegate"]
    #[ivars = PickerIvars]
    struct PickerDelegate;

    unsafe impl NSObjectProtocol for PickerDelegate {}

    unsafe impl UIDocumentPickerDelegate for PickerDelegate {
        #[unsafe(method(documentPicker:didPickDocumentsAtURLs:))]
        fn did_pick(&self, _: &UIDocumentPickerViewController, urls: &NSArray<NSURL>) {
            let paths = urls
                .iter()
                .filter_map(|url| url.path().map(|path| PathBuf::from(path.to_string())))
                .collect();
            self.answer(Some(paths));
        }

        #[unsafe(method(documentPickerWasCancelled:))]
        fn was_cancelled(&self, _: &UIDocumentPickerViewController) {
            self.answer(None);
        }
    }
);

impl PickerDelegate {
    fn answer(&self, paths: Option<Vec<PathBuf>>) {
        if let Some(answer) = self.ivars().answer.take()
            && answer.send(Ok(paths)).is_err()
        {
            log::debug!("GPUI iOS: file picker receiver was dropped");
        }
        // Held to the end of this call, which is running on it.
        let _shown = SHOWN.with(|shown| shown.take());
    }
}

pub(super) fn prompt(
    options: PathPromptOptions,
    presenter: Option<Retained<UIViewController>>,
) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
    let (answer, answered) = oneshot::channel();
    match presenter {
        Some(presenter) if options.files => show(options.multiple, presenter, answer),
        _ => {
            let why = if options.files {
                "no view controller to present the file picker"
            } else {
                "iOS has no picker for folders alone"
            };
            answer.send(Err(anyhow!(why))).ok();
        }
    }
    answered
}

fn show(multiple: bool, presenter: Retained<UIViewController>, answer: Answer) {
    let main_thread = presenter.mtm();
    let delegate = PickerDelegate::alloc(main_thread).set_ivars(PickerIvars {
        answer: RefCell::new(Some(answer)),
    });
    let delegate: Retained<PickerDelegate> = unsafe { msg_send![super(delegate), init] };
    let types = NSArray::from_slice(&[unsafe { UTTypeData }]);
    let picker = UIDocumentPickerViewController::initForOpeningContentTypes_asCopy(
        UIDocumentPickerViewController::alloc(main_thread),
        &types,
        true,
    );
    picker.setAllowsMultipleSelection(multiple);
    picker.setDelegate(Some(objc2::runtime::ProtocolObject::from_ref(&*delegate)));
    SHOWN.with(|shown| shown.replace(Some(delegate)));
    presenter.presentViewController_animated_completion(&picker, true, None);
}
