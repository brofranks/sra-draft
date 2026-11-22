//! Public contracts for native indexes over bound scalar algebras.
//!
//! The `tiny` fixture binds one two-bit algebra on each built-in backend,
//! so every tuple can be checked against a closure over `0..4`. Most indexes
//! start with one slot beside the two constants to exercise maintenance.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Barrier;
use std::thread;

use sra::backend::BoolOp;
use sra::bdd::BddStore;
use sra::context::{AlgebraId, Context};
use sra::index::{Index, IndexRoot};
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;
use sra::util::count::Count;

#[path = "support/reentrant.rs"]
mod reentrant;

use reentrant::Reentrant;

const BV2: ScalarSort = ScalarSort::bv(2);

fn tiny() -> (Context, [AlgebraId; 2]) {
    let mut cx = Context::empty();
    cx.register_backend("bdd", BddStore::with_capacity(8, 16, 1));
    cx.register_backend("sparse", SparseStore::with_capacity(8, 16, 1));
    let ids = ["bdd", "sparse"].map(|name| cx.bind(name, BV2));
    (cx, ids)
}

fn reentrant_index() -> Index {
    let (other, [_, a]) = tiny();
    let index = Index::with_capacity(&other, &[a], 8, 16);
    let mut cx = Context::empty();
    cx.register_backend("reentrant", Reentrant::new(index));
    let a = cx.bind("reentrant", BV2);
    Index::with_capacity(&cx, &[a], 8, 16)
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

fn assert_members(root: &IndexRoot, holds: impl Fn(u64, u64) -> bool) {
    let mut n = 0;
    for x in 0..4 {
        for y in 0..4 {
            let expected = holds(x, y);
            assert_eq!(root.contains(&[x, y]), expected, "at ({x}, {y})");
            n += u64::from(expected);
        }
    }
    assert_eq!(root.count(), n);
    assert_eq!(root.is_empty(), n == 0);
}

// --- layouts ----

#[test]
fn indexes_share_storage_by_exact_ordered_layout() {
    let (cx, [a, b]) = tiny();
    let (_, [foreign, _]) = tiny();
    let index = Index::with_capacity(&cx, &[a, b], 3, 2);
    let other = Index::with_capacity(&cx, &[b, a], 3, 2);
    assert_eq!(index, Index::new(&cx, &[a, b]));
    assert_ne!(index, other);
    assert_eq!(index.context(), &cx);
    assert_eq!(index.arity(), 2);
    assert_eq!([index.algebra(0), index.algebra(1)], [a, b]);
    let message = "index roots must share a physical layout";
    rejects(message, || {
        drop(index.full().apply_bool(BoolOp::And, &other.full()))
    });
    rejects(message, || drop(index.values(0, &[1], &[other.full()])));
    rejects(message, || drop(index.cases(0, &other.empty(), &[])));
    rejects(message, || {
        drop(index.cases(0, &index.empty(), &[(&cx.t(a), &other.full())]));
    });
    rejects("case must use the level's algebra", || {
        drop(index.cases(0, &index.empty(), &[(&cx.t(b), &index.full())]));
    });
    rejects("algebra belongs to another context", || {
        drop(Index::new(&cx, &[foreign]));
    });
}

#[test]
fn rejected_capacity_leaves_the_index_registry_usable() {
    let cx = Context::empty();
    rejects("index node capacity must fit", || {
        drop(Index::with_capacity(&cx, &[], 1, 0));
    });
    #[cfg(target_pointer_width = "32")]
    rejects("unique-table capacity overflow", || {
        drop(Index::with_capacity(&cx, &[], u32::MAX as usize, 0));
    });
    assert_eq!(Index::with_capacity(&cx, &[], 2, 0).full().count(), 1);
}

#[test]
fn concurrent_first_requests_share_an_index() {
    let cx = Context::empty();
    let barrier = Barrier::new(2);
    let open = || {
        barrier.wait();
        Index::with_capacity(&cx, &[], 2, 0)
    };
    thread::scope(|s| {
        let a = s.spawn(open);
        let b = open();
        assert_eq!(a.join().unwrap_or_else(|p| resume_unwind(p)), b);
    });
}

// --- construction ----

#[test]
fn cases_check_explicit_default_overlaps_in_both_orders() {
    let (cx, algebras) = tiny();
    for a in algebras {
        let index = Index::with_capacity(&cx, &[a], 3, 2);
        let (empty, full) = (index.empty(), index.full());
        let (p, q) = (cx.set(a, &[1, 2]), cx.set(a, &[2, 3]));
        let (a, b) = ((&p, &full), (&q, &empty));
        for cases in [[a, b], [b, a]] {
            rejects("cases with different labels overlap", || {
                drop(index.cases(0, &empty, &cases));
            });
        }
        let agrees = index.cases(0, &empty, &[(&p, &full), (&q, &full)]);
        assert_eq!(agrees.count(), 3);
        assert!(!agrees.contains(&[0]));
        assert!(agrees.contains(&[2]));
    }
}

#[test]
fn only_retained_children_must_follow_their_parent() {
    let (cx, algebras) = tiny();
    for a in algebras {
        let index = Index::with_capacity(&cx, &[a, a], 3, 2);
        let (empty, full) = (index.empty(), index.full());
        let earlier = index.values(0, &[1], std::slice::from_ref(&full));
        let p = cx.set(a, &[1]);
        let collapsed = index.cases(1, &earlier, &[(&p, &earlier)]);
        assert_members(&collapsed, |x, _| x == 1);
        let covered = index.cases(1, &earlier, &[(&cx.t(a), &full)]);
        assert_members(&covered, |_, _| true);
        let message = "index child must follow its parent level";
        rejects(message, || {
            drop(index.cases(0, &empty, &[(&p, &earlier)]));
        });
        rejects(message, || {
            drop(index.cases(1, &empty, &[(&p, &earlier)]));
        });
        rejects(message, || {
            drop(index.values(1, &[1], &[earlier]));
        });
    }
}

#[test]
fn values_require_sorted_packed_keys_and_nonempty_children() {
    let (cx, algebras) = tiny();
    for a in algebras {
        let index = Index::with_capacity(&cx, &[a], 3, 2);
        let full = index.full();
        let children = [full.clone(), full.clone()];
        for keys in [[2, 1], [1, 1]] {
            rejects("values are not sorted and distinct", || {
                drop(index.values(0, &keys, &children));
            });
        }
        rejects("one case label per key", || {
            drop(index.values(0, &[1], &children));
        });
        rejects("values require nonempty continuations", || {
            drop(index.values(0, &[1], &[index.empty()]));
        });
        assert!(index.values(0, &[], &[]).is_empty());
        assert_eq!(
            index
                .values(
                    0,
                    &[0, 1, 2, 3],
                    &[full.clone(), full.clone(), full.clone(), full]
                )
                .count(),
            4,
        );
    }
}

#[test]
fn construction_can_build_an_independent_index() {
    let index = reentrant_index();
    let root = index.values(0, &[0, 3], &[index.full(), index.full()]);
    assert_eq!(
        [0, 1, 2, 3].map(|x| root.contains(&[x])),
        [true, false, false, true]
    );
}

// --- membership and counting ----

#[test]
fn contains_and_count_cover_multiword_and_skipped_levels() {
    let (mut cx, [a, _]) = tiny();
    let wide = cx.bind("sparse", ScalarSort::bv(66));
    let index = Index::with_capacity(&cx, &[a, wide, a], 3, 2);
    let root = index.values(1, &[7, 1], &[index.full()]);
    assert!(root.contains(&[3, 7, 1, 0]));
    assert!(!root.contains(&[3, 7, 0, 0]));
    assert_eq!(root.count(), 16);
    assert_eq!(index.full().count(), Count::from_digits(vec![0, 64]));
    let last = index.values(2, &[1], &[index.full()]);
    assert_eq!(last.count(), Count::from_digits(vec![0, 16]));

    for words in [&[3, 7, 1][..], &[3, 7, 1, 0, 0]] {
        rejects("one encoded value is required per index level", || {
            root.contains(words);
        });
    }
    rejects("outside its index level's sort", || {
        last.contains(&[0, 7, 4, 1]);
    });
    rejects("outside its index level's sort", || {
        index.empty().contains(&[0, 7, 1, 4]);
    });
    let unit = Index::with_capacity(&cx, &[], 2, 0);
    assert_eq!(unit.full().count(), 1);
    assert_eq!(unit.empty().count(), 0);
    assert!(unit.full().contains(&[]));
    assert!(!unit.empty().contains(&[]));
}

#[test]
fn count_adjusts_a_shared_suffix_for_skipped_levels() {
    let (cx, [a, b]) = tiny();
    let index = Index::with_capacity(&cx, &[a, b, a, b], 8, 16);
    let last = index.values(3, &[2], &[index.full()]);
    let middle = index.values(1, &[1], std::slice::from_ref(&last));
    let root = index.values(0, &[0, 1], &[last, middle]);
    // The first key leaves two positions free. The second leaves one free.
    assert_eq!(root.count(), 16 + 4);
}

// --- Boolean operations ----

#[test]
fn alignment_can_apply_a_boolean_operation_in_an_independent_index() {
    let index = reentrant_index();
    let cx = index.context();
    let algebra = index.algebra(0);
    let (empty, full) = (index.empty(), index.full());
    let at = |key| index.cases(0, &empty, &[(&cx.set(algebra, &[key]), &full)]);
    let union = at(1).apply_bool(BoolOp::Or, &at(3));
    assert_eq!(
        [0, 1, 2, 3].map(|x| union.contains(&[x])),
        [false, true, false, true]
    );
}

#[test]
fn mixed_boolean_operations_preserve_tuple_correlations() {
    let (cx, [a, b]) = tiny();
    for [a, b] in [[a, b], [b, a]] {
        let index = Index::with_capacity(&cx, &[a, b], 3, 4);
        let (empty, full) = (index.empty(), index.full());
        let end = index.values(1, &[0, 3], &[full.clone(), full.clone()]);
        let middle = index.values(1, &[1, 2], &[full.clone(), full.clone()]);
        let p = index.cases(
            0,
            &empty,
            &[(&cx.set(a, &[0, 2]), &end), (&cx.set(a, &[1]), &middle)],
        );
        let q = index.values(0, &[0, 1], &[middle, full.clone()]);
        let r = index.values(1, &[1, 3], &[full.clone(), full]);
        let in_p = |x, y| match x {
            0 | 2 => y == 0 || y == 3,
            1 => y == 1 || y == 2,
            _ => false,
        };
        let in_q = |x, y| x == 1 || (x == 0 && (y == 1 || y == 2));
        assert_members(&p, in_p);
        assert_members(&q, in_q);
        for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
            assert_members(&p.apply_bool(op, &q), |x, y| match op {
                BoolOp::And => in_p(x, y) && in_q(x, y),
                BoolOp::Or => in_p(x, y) || in_q(x, y),
                BoolOp::Xor => in_p(x, y) != in_q(x, y),
                BoolOp::AndNot => in_p(x, y) && !in_q(x, y),
            });
        }
        assert_members(&p.not(), |x, y| !in_p(x, y));
        assert_members(&r.apply_bool(BoolOp::AndNot, &p), |x, y| {
            (y == 1 || y == 3) && !in_p(x, y)
        });
    }
}

// --- ownership ----

#[test]
fn a_cloned_root_keeps_its_context_and_descendants_through_growth() {
    let saved = {
        let (cx, [a, b]) = tiny();
        let index = Index::with_capacity(&cx, &[a, b], 3, 2);
        let suffix = index.values(1, &[2], &[index.full()]);
        let parent = index.values(0, &[1], &[suffix]);
        parent.clone()
    };
    let index = saved.index();
    let other = index.values(1, &[3], &[index.full()]);
    assert_members(&saved, |x, y| x == 1 && y == 2);
    assert_members(&other, |_, y| y == 3);
}
