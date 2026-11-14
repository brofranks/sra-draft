//! The example run, and chunk claims at an exhausted counter, including
//! usize::MAX, without enumerating the full source universe.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::{claim, main};

#[test]
fn main_passes() {
    main();
}

// --- bounded claims --------------------------------------------------------

#[test]
fn exhausted_chunk_counters_stay_exhausted() {
    for chunks in [usize::MAX, 3, 0] {
        let start = chunks.saturating_sub(2);
        let next = AtomicUsize::new(start);
        for i in start..chunks {
            assert_eq!(claim(&next, chunks), Some(i));
        }
        assert_eq!(claim(&next, chunks), None);
        assert_eq!(claim(&next, chunks), None);
        assert_eq!(next.load(Ordering::Relaxed), chunks);
    }
}

#[test]
fn concurrent_claims_do_not_reissue_chunks_after_exhaustion() {
    let next = AtomicUsize::new(usize::MAX - 2);
    let mut claimed = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| claim(&next, usize::MAX)))
            .collect();
        workers
            .into_iter()
            .filter_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    claimed.sort_unstable();
    assert_eq!(claimed, [usize::MAX - 2, usize::MAX - 1]);
    assert_eq!(next.load(Ordering::Relaxed), usize::MAX);
}
