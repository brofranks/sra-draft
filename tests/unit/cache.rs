//! Cache bounds, invalidation, and collision handling under one admission.

use super::{Cache, Function, NeedGrowth, align, quotient, restart};

#[test]
fn a_store_cache_restarts_on_a_new_store_epoch_or_bound_overflow() {
    let mut cache = Cache::default();
    assert_eq!(cache.bound, 1 << 20);
    let cache_entry = |cache: &mut Cache| {
        cache.aligned.insert(
            (Function::new(2), Function::new(3)),
            (Function::new(4), 0, 1),
        );
        cache.labels.push((0, 0));
    };
    restart(&mut cache, 1, 1);
    cache_entry(&mut cache);
    restart(&mut cache, 1, 1);
    assert_eq!((cache.aligned.len(), cache.labels.len()), (1, 1));
    restart(&mut cache, 2, 1);
    assert!(cache.aligned.is_empty() && cache.labels.is_empty());
    assert_eq!((cache.store, cache.epoch), (2, 1));
    cache_entry(&mut cache);
    restart(&mut cache, 2, 2);
    assert!(cache.aligned.is_empty());
    // A cache at its bound keeps its entries, and one over it clears them while
    // retaining its allocations.
    let mut cache = Cache::with_bound(1);
    restart(&mut cache, 1, 1);
    cache_entry(&mut cache);
    restart(&mut cache, 1, 1);
    assert_eq!(cache.aligned.len(), 1);
    cache
        .quotients
        .insert((Function::new(2), 0), (Function::new(4), 0, 0));
    restart(&mut cache, 1, 1);
    assert!(cache.aligned.is_empty() && cache.quotients.is_empty() && cache.labels.is_empty());
    assert!(cache.aligned.capacity() > 0 && cache.labels.capacity() > 0);
}

#[test]
fn a_zero_bound_keeps_a_completed_entry_until_the_next_attempt() {
    let mut cache = Cache::with_bound(0);
    let pair = (Function::new(2), Function::new(3));
    let root = Function::new(4);
    restart(&mut cache, 1, 1);
    align(&mut cache, pair, &mut Vec::new(), |out, _| {
        out.push((0, 1));
        Ok(root)
    })
    .unwrap();
    assert_eq!(cache.aligned[&pair], (root, 0, 1));
    assert_eq!(cache.labels, [(0, 1)]);
    restart(&mut cache, 1, 1);
    assert!(cache.aligned.is_empty() && cache.labels.is_empty());
    let key = (root, 17);
    quotient(&mut cache, key, &[0, 1, 0], || Ok(root)).unwrap();
    assert_eq!(cache.quotients[&key], (root, 0, 3));
    assert_eq!(cache.classes, [0, 1, 0]);
    restart(&mut cache, 1, 1);
    assert!(cache.quotients.is_empty() && cache.classes.is_empty());
}

#[test]
fn complete_entries_preserve_payloads_across_hits_and_roll_back_failed_builds() {
    let mut cache = Cache::default();
    let key = (Function::new(2), Function::new(3));
    let root = Function::new(4);
    let mut labels = vec![(9, 9)];
    let failed = align(&mut cache, key, &mut labels, |out, _| {
        out.push((1, 1));
        Err(NeedGrowth::Edges(2))
    });
    assert_eq!(failed, Err(NeedGrowth::Edges(2)));
    assert_eq!(labels, [(9, 9)]);
    assert!(cache.aligned.is_empty() && cache.labels.is_empty());

    align(&mut cache, key, &mut labels, |out, start| {
        assert_eq!(start, 1);
        out.extend([(0, 0), (1, 1)]);
        Ok(root)
    })
    .unwrap();
    assert_eq!(
        align(&mut cache, key, &mut labels, |_, _| panic!(
            "alignment cache miss"
        )),
        Ok(root)
    );
    assert_eq!(labels, [(9, 9), (0, 0), (1, 1), (0, 0), (1, 1)]);
    assert_eq!(cache.labels.len(), 2);

    let key = (root, 17);
    assert_eq!(quotient(&mut cache, key, &[0, 1, 0], || Ok(root)), Ok(root));
    assert_eq!(
        quotient(&mut cache, key, &[0, 1, 0], || panic!(
            "quotient cache miss"
        )),
        Ok(root)
    );
    assert_eq!(cache.classes.len(), 3);
    cache.clear();
    assert!(cache.aligned.is_empty() && cache.labels.is_empty());
    assert!(cache.quotients.is_empty() && cache.classes.is_empty());
}
