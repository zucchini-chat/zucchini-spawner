//! macOS power-management glue.
//!
//! Wake watcher fires on **full wake only** (display on) — `IOPMConnection` lets us
//! filter out dark wake / Power Nap so we don't spawn agents that get killed seconds
//! later when the Mac sleeps again.
//!
//! Lid-closed-on-battery still forces sleep — the `PreventUserIdleSystemSleep`
//! assertion only covers idle sleep.

use std::sync::Arc;

use tokio::sync::Notify;

pub type WakeSignal = Arc<Notify>;

pub fn start_wake_watcher() -> WakeSignal {
    let notify = Arc::new(Notify::new());
    #[cfg(target_os = "macos")]
    macos::spawn_watcher(Arc::clone(&notify));
    notify
}

pub struct AgentPowerAssertion {
    #[cfg(target_os = "macos")]
    id: u32,
}

impl AgentPowerAssertion {
    pub fn acquire() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            macos::acquire_assertion()
        }
        #[cfg(not(target_os = "macos"))]
        {
            Some(AgentPowerAssertion {})
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for AgentPowerAssertion {
    fn drop(&mut self) {
        // SAFETY: takes no pointers; `self.id` comes from a successful
        // `IOPMAssertionCreateWithName` (the only constructor) and Drop
        // releases it once.
        unsafe {
            macos::IOPMAssertionRelease(self.id);
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::CString;
    use std::os::raw::{c_int, c_void};
    use std::ptr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, OnceLock};

    use core_foundation_sys::base::CFRelease;
    use core_foundation_sys::runloop::{kCFRunLoopDefaultMode, CFRunLoopGetCurrent, CFRunLoopRun};
    use core_foundation_sys::string::{
        kCFStringEncodingUTF8, CFStringCreateWithCString, CFStringRef,
    };
    use tokio::sync::Notify;
    use tracing::{error, info, warn};

    use super::AgentPowerAssertion;

    type IOReturn = c_int;
    pub(super) type IOPMAssertionID = u32;
    type IOPMAssertionLevel = u32;
    type IOPMConnection = *mut c_void;
    type IOPMConnectionMessageToken = u32;
    type IOPMSystemPowerStateCapabilities = u32;

    const KERN_SUCCESS: IOReturn = 0;
    const IOPM_ASSERTION_LEVEL_ON: IOPMAssertionLevel = 255;
    const CAPABILITY_CPU: IOPMSystemPowerStateCapabilities = 0x01;
    const CAPABILITY_VIDEO: IOPMSystemPowerStateCapabilities = 0x02;

    type IOPMEventHandlerType = extern "C" fn(
        param: *mut c_void,
        connection: IOPMConnection,
        token: IOPMConnectionMessageToken,
        capabilities: IOPMSystemPowerStateCapabilities,
    );

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMConnectionCreate(
            name: CFStringRef,
            interest_capabilities: IOPMSystemPowerStateCapabilities,
            connection: *mut IOPMConnection,
        ) -> IOReturn;
        fn IOPMConnectionSetNotification(
            connection: IOPMConnection,
            param: *mut c_void,
            handler: IOPMEventHandlerType,
        ) -> IOReturn;
        fn IOPMConnectionScheduleWithRunLoop(
            connection: IOPMConnection,
            run_loop: *mut c_void,
            run_loop_mode: CFStringRef,
        ) -> IOReturn;
        fn IOPMConnectionAcknowledgeEvent(
            connection: IOPMConnection,
            token: IOPMConnectionMessageToken,
        ) -> IOReturn;
        fn IOPMConnectionRelease(connection: IOPMConnection) -> IOReturn;
        fn IOPMAssertionCreateWithName(
            assertion_type: CFStringRef,
            assertion_level: IOPMAssertionLevel,
            assertion_name: CFStringRef,
            assertion_id: *mut IOPMAssertionID,
        ) -> IOReturn;
        pub(super) fn IOPMAssertionRelease(assertion_id: IOPMAssertionID) -> IOReturn;
    }

    fn cfstring(s: &str) -> Option<CFStringRef> {
        let c = CString::new(s).ok()?;
        // SAFETY: a null allocator means the default one; `c` is
        // NUL-terminated UTF-8 and outlives the call, which copies it.
        let r =
            unsafe { CFStringCreateWithCString(ptr::null(), c.as_ptr(), kCFStringEncodingUTF8) };
        if r.is_null() {
            None
        } else {
            Some(r)
        }
    }

    /// Returns a process-lifetime CFString for `s`. The first call allocates;
    /// subsequent calls return the same handle. Stored as `usize` because raw
    /// pointers aren't `Send`. The handle is intentionally never `CFRelease`d.
    fn cfstring_static(slot: &OnceLock<Option<usize>>, s: &'static str) -> Option<CFStringRef> {
        slot.get_or_init(|| cfstring(s).map(|r| r as usize))
            .map(|v| v as CFStringRef)
    }

    struct WatcherContext {
        notify: Arc<Notify>,
        /// Init `true` to suppress the synthetic first event reporting current state.
        had_video: AtomicBool,
    }

    extern "C" fn power_callback(
        param: *mut c_void,
        connection: IOPMConnection,
        token: IOPMConnectionMessageToken,
        capabilities: IOPMSystemPowerStateCapabilities,
    ) {
        // SAFETY: `param` is the `ctx_ptr` registered in `spawn_watcher`. IOKit
        // calls us only from that thread's run loop, which runs only after
        // setup succeeded and leaked the context, so it lives for the process;
        // it is only ever read through `&` (atomics and an `Arc`).
        let ctx = unsafe { &*(param as *const WatcherContext) };
        let video_now = (capabilities & CAPABILITY_VIDEO) != 0;
        let had_video = ctx.had_video.swap(video_now, Ordering::Relaxed);
        if !had_video && video_now {
            info!(
                caps = format!("0x{:x}", capabilities),
                "full wake (display on), signaling sync reconnect"
            );
            ctx.notify.notify_one();
        }
        // SAFETY: acknowledges the event IOKit just delivered, on the
        // connection it delivered it on, which is never released once set up.
        unsafe {
            IOPMConnectionAcknowledgeEvent(connection, token);
        }
    }

    struct SetupErr {
        conn: Option<IOPMConnection>,
        step: &'static str,
        code: IOReturn,
    }

    pub fn spawn_watcher(notify: Arc<Notify>) {
        let result = std::thread::Builder::new()
            .name("power-watcher".into())
            .spawn(move || {
                // Box stays owned through setup; on Err it drops normally (no unsafe),
                // on Ok we leak it so the IOKit registration's raw pointer
                // remains valid for the process lifetime (CFRunLoopRun never returns).
                let setup = || -> Result<(IOPMConnection, Box<WatcherContext>), SetupErr> {
                    let cf_name = cfstring("zucchini-spawner").ok_or(SetupErr {
                        conn: None,
                        step: "CFStringCreateWithCString",
                        code: 0,
                    })?;

                    let ctx_box = Box::new(WatcherContext {
                        notify,
                        had_video: AtomicBool::new(true),
                    });
                    let ctx_ptr = &raw const *ctx_box as *mut c_void;

                    let mut conn: IOPMConnection = ptr::null_mut();
                    let interest = CAPABILITY_CPU | CAPABILITY_VIDEO;
                    // SAFETY: `cf_name` is a live CFString; `conn` is a writable
                    // local the call fills in.
                    let create_ret =
                        unsafe { IOPMConnectionCreate(cf_name, interest, &raw mut conn) };
                    // SAFETY: `cf_name` is non-null and its one reference is ours
                    // (Create rule), released once and unused after; a callee that
                    // keeps a CF object retains it.
                    unsafe { CFRelease(cf_name.cast::<c_void>()) };
                    if create_ret != KERN_SUCCESS || conn.is_null() {
                        return Err(SetupErr {
                            conn: None,
                            step: "IOPMConnectionCreate",
                            code: create_ret,
                        });
                    }

                    // SAFETY: `conn` is the live connection just created. `ctx_ptr`
                    // is valid whenever the handler runs: only on this thread's run
                    // loop, which runs only after setup succeeded and leaked the box.
                    let notify_ret =
                        unsafe { IOPMConnectionSetNotification(conn, ctx_ptr, power_callback) };
                    if notify_ret != KERN_SUCCESS {
                        return Err(SetupErr {
                            conn: Some(conn),
                            step: "IOPMConnectionSetNotification",
                            code: notify_ret,
                        });
                    }

                    // SAFETY: `conn` is live; `CFRunLoopGetCurrent` returns this
                    // thread's run loop (Get rule: not ours to release) and
                    // `kCFRunLoopDefaultMode` is a constant CFString.
                    let schedule_ret = unsafe {
                        IOPMConnectionScheduleWithRunLoop(
                            conn,
                            CFRunLoopGetCurrent().cast::<c_void>(),
                            kCFRunLoopDefaultMode,
                        )
                    };
                    if schedule_ret != KERN_SUCCESS {
                        return Err(SetupErr {
                            conn: Some(conn),
                            step: "IOPMConnectionScheduleWithRunLoop",
                            code: schedule_ret,
                        });
                    }

                    Ok((conn, ctx_box))
                };

                match setup() {
                    Ok((_conn, ctx_box)) => {
                        info!("power watcher registered (full-wake only)");
                        Box::leak(ctx_box);
                        // SAFETY: no arguments; runs this thread's run loop, which
                        // delivers `power_callback`, for the rest of the process.
                        unsafe { CFRunLoopRun() };
                    }
                    Err(e) => {
                        if let Some(c) = e.conn {
                            // SAFETY: `c` is the connection `IOPMConnectionCreate`
                            // returned, released once here. Its handler never ran
                            // (this thread's run loop didn't), so the dropped
                            // context was never read.
                            unsafe { IOPMConnectionRelease(c) };
                        }
                        error!(
                            step = e.step,
                            code = e.code,
                            "power watcher setup failed; disabled"
                        );
                    }
                }
            });
        if let Err(e) = result {
            error!(error = %e, "failed to spawn power-watcher thread; wake reconnect disabled");
        }
    }

    pub fn acquire_assertion() -> Option<AgentPowerAssertion> {
        // kIOPMAssertionTypePreventUserIdleSystemSleep is a CFSTR(...) macro, not a
        // linkable symbol — build it from the underlying literal. Cached for the
        // process lifetime so per-spawn cost is one IOPMAssertionCreateWithName call.
        static CF_TYPE: OnceLock<Option<usize>> = OnceLock::new();
        static CF_NAME: OnceLock<Option<usize>> = OnceLock::new();
        let cf_type = cfstring_static(&CF_TYPE, "PreventUserIdleSystemSleep")?;
        let cf_name = cfstring_static(&CF_NAME, "zucchini-spawner agent running")?;
        let mut id: IOPMAssertionID = 0;
        // SAFETY: `cf_type` and `cf_name` are process-lifetime CFStrings (never
        // released); `id` is a writable local the call fills in.
        let r = unsafe {
            IOPMAssertionCreateWithName(cf_type, IOPM_ASSERTION_LEVEL_ON, cf_name, &raw mut id)
        };
        if r != KERN_SUCCESS {
            warn!(code = r, "IOPMAssertionCreateWithName failed");
            return None;
        }
        Some(AgentPowerAssertion { id })
    }
}
