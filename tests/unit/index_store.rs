//! Partition ownership and memo validity in the native index store.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;
use crate::backend::Backend;
use crate::bdd::BddStore;
use crate::scalar::ScalarSort;
use crate::sparse::SparseStore;

fn point(store: &Store, value: u64) -> Id {
    store.values(0, &[value], [Id::TRUE].into_iter())
}

fn rejects(expected: &str, run: impl FnOnce()) {
    let panic = catch_unwind(AssertUnwindSafe(run)).expect_err("operation was accepted");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap();
    assert!(message.contains(expected), "unexpected panic: {message}");
}

fn assert_released(algebra: &dyn Algebra, f: Function) {
    // The caller checks before any scalar allocation can reuse the token.
    rejects(
        "cannot decrement an unreferenced or reclaimed DAG node",
        || algebra.dec_ref(f),
    );
}

fn algebras() -> [Arc<dyn Algebra>; 2] {
    let backends: [Arc<dyn Backend>; 2] = [
        Arc::new(BddStore::with_capacity(32, 64, 1)),
        Arc::new(SparseStore::with_capacity(32, 64, 1)),
    ];
    backends.map(|backend| Arc::from(backend.bind(ScalarSort::bv(2)).unwrap()))
}

#[test]
fn rejecting_a_backward_child_releases_its_partition() {
    fn check(algebra: &Arc<dyn Algebra>, build: impl FnOnce(&Store, Id, Function)) {
        let layout = Box::new([(0, Arc::clone(algebra)), (0, Arc::clone(algebra))]);
        let store = Store::new(layout, 8, 16);
        let child = point(&store, 1);
        let partition = store.read().node(child).partition;
        rejects("index child must follow its parent level", || {
            build(&store, child, partition)
        });
        store.dec_ref(child);
        drop(store);
        assert_released(algebra.as_ref(), partition);
    }

    for algebra in algebras() {
        check(&algebra, |store, child, p| {
            store.cases(1, Id::FALSE, [(p, child)].into_iter());
        });
        check(&algebra, |store, child, _| {
            store.values(1, &[1], [child].into_iter());
        });
    }
}

#[test]
fn collection_and_destruction_release_one_partition_per_node() {
    for algebra in algebras() {
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
fn boolean_memo_expires_when_switching_stores() {
    let algebra = Arc::new(SparseStore::with_capacity(32, 64, 1))
        .bind(ScalarSort::bv(2))
        .unwrap();
    let algebra: Arc<dyn Algebra> = Arc::from(algebra);
    let first = Store::new(Box::new([(0, Arc::clone(&algebra))]), 5, 8);
    let second = Store::new(Box::new([(0, algebra)]), 5, 8);

    let x = point(&first, 0);
    let y = point(&first, 1);
    let old = first.apply(BoolOp::Or, x, y);
    let p = point(&second, 2);
    let q = point(&second, 3);
    let decoy = point(&second, 0);
    // Only the store ID distinguishes the two memo entries.
    assert_eq!((x, y, old), (p, q, decoy));
    assert_eq!(first.shared.epoch(), second.shared.epoch());

    let union = second.apply(BoolOp::Or, p, q);
    assert_eq!(
        [0, 1, 2, 3].map(|value| second.contains(union, &[value])),
        [false, false, true, true]
    );
    for id in [x, y, old] {
        first.dec_ref(id);
    }
    for id in [p, q, decoy, union] {
        second.dec_ref(id);
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
