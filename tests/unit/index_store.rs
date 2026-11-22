//! Partition ownership and memo validity across native index maintenance.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;
use crate::backend::Backend;
use crate::bdd::BddStore;
use crate::scalar::ScalarSort;
use crate::sparse::SparseStore;

fn point(store: &Store, value: u64) -> Id {
    store.values(0, &[value], [Id::TRUE].into_iter())
}

fn assert_released(algebra: &dyn Algebra, f: Function) {
    // The caller checks before any scalar allocation can reuse the token.
    let panic = catch_unwind(AssertUnwindSafe(|| algebra.dec_ref(f)))
        .expect_err("a partition reference leaked");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap();
    assert!(
        message.contains("cannot decrement an unreferenced or reclaimed DAG node"),
        "unexpected panic: {message}"
    );
}

#[test]
fn collection_and_destruction_release_one_partition_per_node() {
    let backends: [Arc<dyn Backend>; 2] = [
        Arc::new(BddStore::with_capacity(32, 64, 1)),
        Arc::new(SparseStore::with_capacity(32, 64, 1)),
    ];
    for backend in backends {
        let algebra: Arc<dyn Algebra> = Arc::from(backend.bind(ScalarSort::bv(2)).unwrap());
        let store = Store::new(Box::new([(0, Arc::clone(&algebra))]), 3, 4);
        let old = point(&store, 1);
        let partition = store.read().node(old).partition;
        let duplicate = point(&store, 1);
        assert_eq!(old, duplicate);
        store.dec_ref(old);
        store.dec_ref(duplicate);
        assert_eq!(algebra.evaluate(partition, &[1]), 1);

        // The only nonterminal slot is occupied by an unreferenced node.
        // The new scalar partition is allocated before collection releases
        // the old one, and the retry reuses that new partition.
        let epoch = store.shared.epoch();
        let next = point(&store, 2);
        assert!(store.shared.epoch() > epoch);
        assert_released(&*algebra, partition);

        let partition = store.read().node(next).partition;
        assert_eq!(algebra.evaluate(partition, &[2]), 1);
        store.dec_ref(next);
        drop(store);
        assert_released(&*algebra, partition);
    }
}

#[test]
fn boolean_memo_expires_when_index_maintenance_recycles_a_node() {
    let algebra = Arc::new(SparseStore::with_capacity(32, 64, 1))
        .bind(ScalarSort::bv(2))
        .unwrap();
    let store = Store::new(Box::new([(0, Arc::from(algebra))]), 5, 8);
    let a = point(&store, 0);
    let b = point(&store, 1);
    let old = store.apply(BoolOp::Or, a, b);
    store.dec_ref(old);

    // The union is the store's sole reclaimable node. Its slot now names
    // a different set, while both operands are still live.
    let other = point(&store, 2);
    assert_eq!(other, old);
    let union = store.apply(BoolOp::Or, a, b);
    assert!(store.contains(union, &[0]));
    assert!(store.contains(union, &[1]));
    assert!(!store.contains(union, &[2]));
    assert_eq!(store.count(union), 2);
    for id in [a, b, other, union] {
        store.dec_ref(id);
    }
}
