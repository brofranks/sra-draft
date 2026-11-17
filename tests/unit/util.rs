//! Identity exhaustion tests using an isolated counter.

use std::panic;
use std::sync::atomic::{AtomicU64, Ordering};

use super::take_id;

#[test]
fn identity_exhaustion_permanently_preserves_the_counter() {
    let ids = AtomicU64::new(u64::MAX - 2);
    assert_eq!(take_id(&ids), u64::MAX - 2);
    assert_eq!(take_id(&ids), u64::MAX - 1);
    for _ in 0..2 {
        let error = panic::catch_unwind(|| take_id(&ids)).unwrap_err();
        assert!(
            error
                .downcast_ref::<String>()
                .unwrap()
                .starts_with("storage identity overflow")
        );
        assert_eq!(ids.load(Ordering::Relaxed), u64::MAX);
    }
}
