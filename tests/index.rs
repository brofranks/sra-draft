//! Public contracts for native indexes over bound scalar algebras.
//!
//! The `tiny` fixture binds one two-bit algebra on each built-in backend,
//! so every tuple can be checked against a closure over `0..4`. Most indexes
//! start with one slot beside the two constants to exercise maintenance.
//! The reentrant backends expose partition ownership and memo behavior
//! through the public index and scalar interfaces.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Barrier};
use std::thread;

use sra::backend::{Backend, BoolOp};
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

fn reentrant_backends() -> [Reentrant; 2] {
    let (other, [_, a]) = tiny();
    let nested = Index::with_capacity(&other, &[a], 8, 16);
    let stores: [Arc<dyn Backend>; 2] = [
        Arc::new(BddStore::with_capacity(8, 16, 1)),
        Arc::new(SparseStore::with_capacity(8, 16, 1)),
    ];
    stores.map(|store| Reentrant::new(nested.clone(), store.bind(BV2).unwrap()))
}

fn reentrant_indexes(arity: usize, nodes: usize, edges: usize) -> [(Index, Reentrant); 2] {
    reentrant_backends().map(|backend| {
        let mut cx = Context::empty();
        cx.register_backend("reentrant", backend.clone());
        let a = cx.bind("reentrant", BV2);
        (
            Index::with_capacity(&cx, &vec![a; arity], nodes, edges),
            backend,
        )
    })
}

fn point(index: &Index, value: u64) -> IndexRoot {
    index.values(0, &[value], &[index.full()])
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
    let (other_cx, foreign) = tiny();
    let index = Index::with_capacity(&cx, &[a, b], 3, 2);
    let other = Index::with_capacity(&cx, &[b, a], 3, 2);
    assert_eq!(index, Index::new(&cx, &[a, b]));
    assert_ne!(index, other);
    assert_eq!(index.context(), &cx);
    assert_eq!(index.arity(), 2);
    assert_eq!([index.algebra(0), index.algebra(1)], [a, b]);
    let message = "index roots must share a physical layout";
    let foreign_index = Index::with_capacity(&other_cx, &foreign, 3, 2);
    rejects(message, || {
        drop(index.full().apply(BoolOp::And, &foreign_index.full()));
    });
    rejects(message, || {
        drop(index.full().apply(BoolOp::And, &other.full()));
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
        drop(Index::new(&cx, &foreign));
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
    rejects("index node capacity must fit", || {
        drop(Index::with_capacity(&cx, &[], 1, 0));
    });
    #[cfg(target_pointer_width = "64")]
    {
        let overflow = u32::MAX as usize + 1;
        rejects("index node capacity must fit", || {
            drop(Index::with_capacity(&cx, &[], overflow, 0));
        });
        rejects("index edge capacity overflow", || {
            drop(Index::with_capacity(&cx, &[], 2, overflow));
        });
    }
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
    for (index, _) in reentrant_indexes(1, 8, 16) {
        let root = index.values(0, &[0, 3], &[index.full(), index.full()]);
        assert_eq!(
            [0, 1, 2, 3].map(|x| root.contains(&[x])),
            [true, false, false, true]
        );
    }
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

#[test]
fn a_shared_suffix_is_counted_once_across_independent_positions() {
    for (index, backend) in reentrant_indexes(3, 8, 16) {
        let last = index.values(2, &[2], &[index.full()]);
        let middle = index.values(1, &[1], std::slice::from_ref(&last));
        let root = index.values(0, &[0, 1], &[last, middle]);
        let before = backend.calls().refine;
        assert_eq!(root.count(), 5);
        assert_eq!(backend.calls().refine - before, 3);
        let complement = root.not();
        assert_eq!(complement.count(), 59);
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    let expected = z == 2 && (x == 0 || (x == 1 && y == 1));
                    assert_eq!(root.contains(&[x, y, z]), expected);
                    assert_eq!(complement.contains(&[x, y, z]), !expected);
                }
            }
        }
    }
}

#[test]
fn counting_survives_same_index_collection_during_cell_construction() {
    for (index, backend) in reentrant_indexes(2, 5, 8) {
        // The first nonterminal is garbage. The retained parent and its
        // suffix fill the other two slots, so the callback forces collection.
        drop(point(&index, 3));
        let suffix = index.values(1, &[2], &[index.full()]);
        let root = index.values(0, &[1], &[suffix]);
        let nested = index.clone();
        let observer = backend.clone();
        reentrant::ON_CELL.set(Some(Box::new(move || {
            let before = observer.calls().dec_refs;
            let other = point(&nested, 0);
            assert!(observer.calls().dec_refs > before);
            assert!(other.contains(&[0, 3]));
        })));
        assert_eq!(root.count(), 1);
        assert!(reentrant::ON_CELL.take().is_none());
        assert_members(&root, |x, y| x == 1 && y == 2);
        drop((root, index));
        assert_eq!(backend.refs(), []);
    }
}

// --- Boolean operations ----

#[test]
fn alignment_can_apply_a_boolean_operation_in_an_independent_index() {
    for (index, backend) in reentrant_indexes(1, 8, 16) {
        let cx = index.context();
        let algebra = index.algebra(0);
        let (empty, full) = (index.empty(), index.full());
        let at = |key| index.cases(0, &empty, &[(&cx.set(algebra, &[key]), &full)]);
        let (a, b) = (at(1), at(3));
        let before = backend.calls().align;
        for _ in 0..2 {
            let union = a.apply(BoolOp::Or, &b);
            assert_eq!(backend.calls().align - before, 1);
            assert_eq!(
                [0, 1, 2, 3].map(|x| union.contains(&[x])),
                [false, true, false, true]
            );
        }
    }
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
            assert_members(&p.apply(op, &q), |x, y| match op {
                BoolOp::And => in_p(x, y) && in_q(x, y),
                BoolOp::Or => in_p(x, y) || in_q(x, y),
                BoolOp::Xor => in_p(x, y) != in_q(x, y),
                BoolOp::AndNot => in_p(x, y) && !in_q(x, y),
            });
        }
        assert!(p.apply(BoolOp::Xor, &p).is_empty());
        assert_members(&p.not(), |x, y| !in_p(x, y));
        assert_members(&p.apply(BoolOp::AndNot, &r), |x, y| {
            in_p(x, y) && y != 1 && y != 3
        });
        assert_members(&r.apply(BoolOp::AndNot, &p), |x, y| {
            (y == 1 || y == 3) && !in_p(x, y)
        });
    }
}

#[test]
fn boolean_memo_expires_when_switching_stores() {
    let (cx, [_, a]) = tiny();
    let first = Index::with_capacity(&cx, &[a], 5, 8);
    let second = Index::with_capacity(&cx, &[a, a], 5, 8);
    let x = point(&first, 0);
    let y = point(&first, 1);
    let _old = x.apply(BoolOp::Or, &y);

    // The stores share their first algebra and have three nonterminal
    // slots each. Only the index store distinguishes these memo keys.
    let p = point(&second, 2);
    let q = point(&second, 3);
    let _decoy = point(&second, 0);
    let union = p.apply(BoolOp::Or, &q);
    assert_members(&union, |x, _| x >= 2);
}

#[test]
fn boolean_memo_expires_when_index_maintenance_recycles_a_node() {
    let (cx, [_, a]) = tiny();
    let index = Index::with_capacity(&cx, &[a], 5, 8);
    let x = point(&index, 0);
    let y = point(&index, 1);
    drop(x.apply(BoolOp::Or, &y));

    // The union is the sole reclaimable node. Its slot now names another
    // set while both operands are still live.
    let _other = point(&index, 2);
    let union = x.apply(BoolOp::Or, &y);
    assert_eq!(
        [0, 1, 2, 3].map(|v| union.contains(&[v])),
        [true, true, false, false]
    );
    assert_eq!(union.count(), 2);
}

#[test]
fn deep_counts_and_complements_fit_a_two_mib_stack() {
    let levels = if cfg!(miri) { 8 } else { 8192 };
    let mut cx = Context::empty();
    cx.register_backend("sparse", SparseStore::with_capacity(8, 16, 1));
    let a = cx.bind("sparse", ScalarSort::bool());
    let index = Index::with_capacity(&cx, &vec![a; levels], 2 + 2 * levels, 4 * levels);
    let mut root = index.full();
    for i in (0..levels).rev() {
        root = index.values(i, &[1], &[root]);
    }
    thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            assert_eq!(root.count(), 1);
            let complement = root.not();
            let mut tuple = vec![1; levels];
            assert!(!complement.contains(&tuple));
            tuple[levels - 1] = 0;
            assert!(complement.contains(&tuple));
            assert_eq!(
                complement.count() + Count::from(1),
                Count::from(1).shift(levels as u64)
            );
        })
        .unwrap()
        .join()
        .unwrap_or_else(|p| resume_unwind(p));
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

#[test]
fn rejecting_a_backward_child_releases_its_partition() {
    for (index, backend) in reentrant_indexes(2, 8, 16) {
        let child = point(&index, 1);
        let p = index.context().set(index.algebra(1), &[1]);
        let before = backend.refs();
        rejects("index child must follow its parent level", || {
            drop(index.cases(1, &index.empty(), &[(&p, &child)]));
        });
        assert_eq!(backend.refs(), before);
        rejects("index child must follow its parent level", || {
            drop(index.values(1, &[1], std::slice::from_ref(&child)));
        });
        assert_eq!(backend.refs(), before);
    }
}

#[test]
fn boolean_operations_release_their_temporary_partitions() {
    for (index, backend) in reentrant_indexes(2, 8, 16) {
        let full = index.full();
        let suffix = index.values(1, &[2], std::slice::from_ref(&full));
        let a = index.values(0, &[1], &[suffix]);
        let b = index.values(0, &[3], &[full]);
        // And collapses, Or keeps the aligned partition, Xor finds the
        // published union, and AndNot coarsens. The complement reuses
        // each operand partition.
        for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
            drop(a.apply(op, &b));
        }
        drop(a.not());
        drop((a, b, index));
        assert_eq!(backend.refs(), []);
    }
}

#[test]
fn counting_releases_each_selected_scalar_cell() {
    for (index, backend) in reentrant_indexes(1, 3, 4) {
        let root = point(&index, 1);
        let before = backend.refs();
        assert_eq!(root.count(), 1);
        assert_eq!(backend.refs(), before);
    }
}

#[test]
fn collection_and_destruction_release_one_partition_per_node() {
    for (index, backend) in reentrant_indexes(1, 3, 4) {
        let old = point(&index, 1);
        let first = backend.refs();
        assert_eq!(first.len(), 1);
        let duplicate = point(&index, 1);
        assert_eq!(backend.refs(), first);
        // Overlapping cases agree only when both designate the same root.
        let all = index.context().t(index.algebra(0));
        let common = index.cases(0, &old, &[(&all, &old), (&all, &duplicate)]);
        assert!(common.contains(&[1]));
        drop((old, duplicate, all, common));
        assert_eq!(backend.refs(), first);

        // The only nonterminal slot holds the unreferenced node. Its new
        // partition is built before collection releases the old one.
        let next = point(&index, 2);
        let second = backend.refs();
        assert_eq!(second.len(), 1);
        assert_ne!(first, second);
        assert!(next.contains(&[2]));
        drop((next, index));
        assert_eq!(backend.refs(), []);
    }
}

#[test]
fn concurrent_equal_nodes_retain_the_winner_through_collection() {
    for (index, backend) in reentrant_indexes(1, 4, 8) {
        let barrier = Barrier::new(2);
        let build = || {
            barrier.wait();
            point(&index, 1)
        };
        let (a, b) = thread::scope(|scope| {
            let worker = scope.spawn(build);
            let b = build();
            (worker.join().unwrap_or_else(|p| resume_unwind(p)), b)
        });
        // A sequential hit owns one partition. A second publication owns
        // the same partition until collection removes the losing candidate.
        let refs = backend.refs();
        assert!((1..=2).contains(&refs.len()));
        assert!(refs.iter().all(|&f| f == refs[0]));
        let all = index.context().t(index.algebra(0));
        drop(index.cases(0, &a, &[(&all, &a), (&all, &b)]));
        let p = point(&index, 2);
        let q = point(&index, 3);
        assert_eq!(backend.refs().len(), 3);
        assert!(a.contains(&[1]));
        assert!(b.contains(&[1]));
        drop((a, b, p, q, all, index));
        assert_eq!(backend.refs(), []);
    }
}

#[test]
fn partition_releases_batch_by_binding_and_preserve_repeated_tokens() {
    for collect in [false, true] {
        let [left, right] = reentrant_backends();
        let mut cx = Context::empty();
        cx.register_backend("left", left.clone());
        cx.register_backend("right", right.clone());
        let a = cx.bind("left", BV2);
        let b = cx.bind("right", BV2);
        let index = Index::with_capacity(&cx, &[b, a, b], 5, 8);
        let mut root = index.full();
        for i in (0..3).rev() {
            root = index.values(i, &[1], &[root]);
        }
        let refs = right.refs();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0], refs[1]);
        assert_eq!(left.refs().len(), 1);
        drop(root);

        if collect {
            let next = point(&index, 2);
            assert_eq!((left.calls().dec_refs, right.calls().dec_refs), (1, 1));
            assert_eq!(left.refs(), []);
            assert_eq!(right.refs().len(), 1);
            assert!(next.contains(&[2, 0, 3]));
        }
        drop((index, cx));
        assert_eq!(
            (left.calls().dec_refs, right.calls().dec_refs),
            (1, 1 + usize::from(collect))
        );
        assert_eq!(left.refs(), []);
        assert_eq!(right.refs(), []);
    }
}
