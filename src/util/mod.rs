//! Generic storage primitives beneath every backend.

use std::sync::atomic::{AtomicU64, Ordering};

#[macro_use]
pub mod dag;

/// Allocates a process-unique identity, reserving zero.
pub fn fresh_id() -> u64 {
    static IDS: AtomicU64 = AtomicU64::new(1);
    IDS.fetch_add(1, Ordering::Relaxed)
}
