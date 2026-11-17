//! Generic storage primitives beneath every backend.

use std::sync::atomic::{AtomicU64, Ordering};

pub mod dag;

/// Takes the next identity, leaving an exhausted counter unchanged.
fn take_id(ids: &AtomicU64) -> u64 {
    ids.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("storage identity overflow")
}

/// Allocates a process-unique identity, reserving zero and `u64::MAX`.
///
/// # Panics
///
/// Panics if all identities have been allocated. Exhaustion is permanent.
pub fn fresh_id() -> u64 {
    static IDS: AtomicU64 = AtomicU64::new(1);
    take_id(&IDS)
}

#[cfg(test)]
#[path = "../../tests/unit/util.rs"]
mod tests;
