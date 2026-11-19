//! Generic storage primitives and exact counts beneath every backend.

use std::sync::atomic::{AtomicU64, Ordering};

pub mod count;
pub mod dag;

/// Allocates a process-unique identity, reserving zero and `u64::MAX`.
///
/// # Panics
///
/// Panics if all identities have been allocated. Every later call panics as
/// well.
pub fn fresh_id() -> u64 {
    static IDS: AtomicU64 = AtomicU64::new(1);
    IDS.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("storage identity overflow")
}
