//! `PIE_BOUNDARY_TRACE=1`: where the host spends the frame boundary.
//!
//! The device is idle from the moment a step lands until the next step's
//! first kernel starts, and everything in between is host work on the engine
//! thread: the rest of `prepare` after the descriptor-port read (which is
//! where the wait for the previous step returns), the prologue guests, the
//! staging copies, the route (body pick + graph launch) and the readback
//! bookkeeping. This module stamps those phases and prints one line per
//! frame, so the split can be read off a serve log instead of guessed at.
//!
//! Off (the default) it is one relaxed atomic load per mark.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_BOUNDARY_TRACE").is_some())
}

thread_local! {
    static MARKS: RefCell<Vec<(&'static str, Instant)>> = const { RefCell::new(Vec::new()) };
}

/// Stamp `label` now. The first mark of a frame is its origin.
pub(crate) fn mark(label: &'static str) {
    if !on() {
        return;
    }
    MARKS.with(|marks| marks.borrow_mut().push((label, Instant::now())));
}

/// Print the frame's marks as deltas from the previous mark, then clear.
pub(crate) fn flush(seq: u64) {
    if !on() {
        return;
    }
    MARKS.with(|marks| {
        let mut marks = marks.borrow_mut();
        if marks.is_empty() {
            return;
        }
        let origin = marks[0].1;
        let mut line = format!("[boundary] seq={seq}");
        let mut prev = origin;
        for (label, at) in marks.iter().skip(1) {
            let us = at.duration_since(prev).as_micros();
            line.push_str(&format!(" {label}={us}"));
            prev = *at;
        }
        let total = prev.duration_since(origin).as_micros();
        line.push_str(&format!(" total={total}us"));
        eprintln!("{line}");
        marks.clear();
    });
}
