//! White-box tests for the sparse backend.
//!
//! Tiny stores force maintenance on the first publication. Other fixtures
//! force collection, inspect reclamation, or inject unique-table collisions.

use std::ops::ControlFlow;
use std::sync::{Arc, Barrier};
use std::thread;

use super::{NeedGrowth, Node, SparseStore};
use crate::backend::{Algebra, Align, Backend, BoolOp, Function, Refine, Values};
use crate::scalar::ScalarSort;
use crate::util::count::Count;
use crate::util::dag;

#[path = "../support/sparse.rs"]
mod fixtures;

use fixtures::{BV4, BV6, bind, holds, members, parts, set};

fn epoch(store: &SparseStore) -> u64 {
    store.shared.epoch()
}

fn collect(store: &SparseStore) {
    let mut first = true;
    store.shared.retry(store.threads, |_| {
        if first {
            first = false;
            Err(NeedGrowth::Nodes)
        } else {
            Ok(())
        }
    });
}

/// A table with room for neither a node nor an edge word beyond its constants.
fn tiny() -> Arc<SparseStore> {
    Arc::new(SparseStore::with_capacity(2, 0, 1))
}

// --- sets, cuts, and counts --------------------------------------------------

#[test]
fn sets_cuts_and_counts_survive_growth() {
    let store = tiny();
    let alg = bind(&store, BV6);
    let values: Vec<u64> = (0..12).map(|v| v * 3).collect();
    let before = epoch(&store);
    let p = alg.set(&values);
    assert!(
        epoch(&store) > before,
        "the set must retry after maintenance"
    );
    assert_eq!(members(&*alg, p), values);
    assert_eq!(alg.count(p), Count::from(12));
    // Adjacent points coalesce into one run rather than one transition each.
    let run = set(&*alg, 40..48);
    assert_eq!(members(&*alg, run), (40..48).collect::<Vec<_>>());
    let lt = alg.lt(&[12]);
    assert_eq!(members(&*alg, lt), (0..12).collect::<Vec<_>>());
    // The final run reaches the sort's exact boundary, which no transition
    // marks.
    let ge = alg.apply(BoolOp::AndNot, Function::TRUE, lt);
    assert_eq!(alg.count(ge), 64 - 12);
    // A cut at zero is the empty map, which is the constant rather than a run.
    assert_eq!(alg.lt(&[0]), Function::FALSE);
    assert_eq!(alg.set(&[]), Function::FALSE);
    for f in [p, run, lt, ge] {
        alg.dec_ref(f);
    }
}

// --- conditionals ------------------------------------------------------------

#[test]
fn trivial_conditionals_retain_the_selected_root_after_collection() {
    let store = tiny();
    let alg = bind(&store, BV6);
    let p = set(&*alg, [10, 20]);
    let q = set(&*alg, (0..64).step_by(2));
    let results = [
        alg.ite(Function::TRUE, p, q),
        alg.ite(Function::FALSE, q, p),
        alg.ite(q, p, p),
        alg.ite(p, Function::TRUE, Function::FALSE),
    ];
    alg.dec_ref(p);
    alg.dec_ref(q);
    collect(&store);
    for f in results {
        assert_eq!(members(&*alg, f), [10, 20]);
        alg.dec_ref(f);
    }
}

// --- alignment ---------------------------------------------------------------

#[test]
fn alignment_appends_one_label_suffix_after_a_retry() {
    let store = Arc::new(SparseStore::with_capacity(8, 7, 1));
    let alg = bind(&store, BV4);
    let f = alg.set(&[1]);
    let g = alg.set(&[3]);
    let mut labels = vec![(9, 9)];
    // Four words are live and the alignment needs four more.
    let before = epoch(&store);
    let h = alg.align(f, g, &mut labels, &mut Align::default());
    assert!(epoch(&store) > before);
    assert_eq!(labels, [(9, 9), (0, 0), (1, 0), (0, 1)]);
    for f in [f, g, h] {
        alg.dec_ref(f);
    }
}

// --- quotient ----------------------------------------------------------------

#[test]
fn a_quotient_merges_classes_after_a_retry() {
    let store = Arc::new(SparseStore::with_capacity(8, 7, 1));
    let alg = bind(&store, BV4);
    let f = alg.values(0, &[1, 3], &[1, 2], &mut Vec::new(), &mut Values::default());
    // Four words are live and the quotient needs four more.
    let before = epoch(&store);
    let q = alg.quotient(f, &[0, 1, 1], &mut Align::default());
    assert!(epoch(&store) > before);
    for x in 0..16 {
        assert_eq!(alg.evaluate(q, &[x]), u32::from(x == 1 || x == 3), "at {x}");
    }
    alg.dec_ref(f);
    alg.dec_ref(q);
}

// --- cases -------------------------------------------------------------------

#[test]
fn cases_append_one_label_suffix_after_a_retry() {
    let store = Arc::new(SparseStore::with_capacity(8, 7, 1));
    let alg = bind(&store, BV4);
    let a = alg.set(&[1]);
    let b = alg.set(&[3]);
    let mut labels = vec![99];
    // The two retained inputs and their partition need eight edge words.
    let before = epoch(&store);
    let f = alg.cases(0, &[(a, 4), (b, 5)], &mut labels);
    assert!(epoch(&store) > before);
    assert_eq!(labels, [99, 0, 4, 5]);
    for root in [a, b, f] {
        alg.dec_ref(root);
    }
}

// --- values ------------------------------------------------------------------

#[test]
fn point_labels_become_first_occurrence_classes_after_a_retry() {
    let store = tiny();
    let alg = bind(&store, BV6);
    let before = epoch(&store);
    let (f, labels) = parts(&*alg, 4, [7, 8, 7, 9]);
    assert!(epoch(&store) > before);
    assert_eq!(labels, [4, 7, 8, 9]);
    for x in 0..64 {
        let expect = match x {
            10 | 30 => 1,
            20 => 2,
            40 => 3,
            _ => 0,
        };
        assert_eq!(alg.evaluate(f, &[x]), expect, "at {x}");
    }
    alg.dec_ref(f);
}

// --- refinement --------------------------------------------------------------

#[test]
fn breaking_before_build_leaves_cells_unpublished() {
    let store = Arc::new(SparseStore::with_capacity(4, 32, 1));
    let alg = bind(&store, BV4);
    let f = alg.set(&[1, 2]);
    let p = alg.set(&[2, 3]);
    let mut scratch = Refine::default();
    // The retained roots and constants fill the node arena. Both cells would
    // require a new node.
    let before = epoch(&store);
    let stopped = alg.refine(f, p, &mut scratch, &mut |_| true, &mut |_, _| {
        ControlFlow::Break(())
    });
    assert!(stopped.is_break());
    assert_eq!(epoch(&store), before);

    let stopped = alg.refine(f, p, &mut scratch, &mut |_| true, &mut |_, build| {
        alg.dec_ref(build());
        ControlFlow::Break(())
    });
    assert!(stopped.is_break());
    assert!(epoch(&store) > before);
    alg.dec_ref(f);
    alg.dec_ref(p);
}

#[test]
fn refinement_delivery_can_collect_and_reenter_the_store() {
    let store = tiny();
    let alg = bind(&store, BV6);
    let (f, _) = parts(&*alg, 0, [1, 2, 1, 3]);
    let p = set(&*alg, [10, 20, 25, 30]);
    let mut seen = Vec::new();
    let result = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |_| true,
        &mut |class, build| {
            collect(&store);
            let a = build();
            let b = build();
            alg.dec_ref(a);
            collect(&store);
            let mut nested = Vec::new();
            let nested_result = alg.refine(
                f,
                b,
                &mut Refine::default(),
                &mut |_| true,
                &mut |inner, make| {
                    let selected = make();
                    nested.push(inner);
                    assert_eq!(members(&*alg, selected), members(&*alg, b));
                    alg.dec_ref(selected);
                    ControlFlow::Continue(())
                },
            );
            assert!(nested_result.is_continue());
            assert_eq!(nested, [class]);
            seen.push((class, members(&*alg, b)));
            alg.dec_ref(b);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]);
    alg.dec_ref(f);
    alg.dec_ref(p);
}

// --- collection and concurrency ----------------------------------------------

#[test]
fn retained_roots_and_member_cursors_survive_collection() {
    let store = tiny();
    let alg = bind(&store, ScalarSort::bv(12));
    let p = set(&*alg, [1, 65, 129, 2049, 4095]);
    alg.inc_ref(p);
    let mut cursor = alg.members(p);
    let mut suspended = alg.members(p);
    alg.dec_ref(p);
    alg.dec_ref(p);
    assert_eq!(suspended.next().unwrap(), [1]);
    drop(suspended);
    assert_eq!(cursor.next().unwrap(), [1]);
    collect(&store);
    let mut rest = Vec::new();
    while let Some(point) = cursor.next() {
        rest.push(point[0]);
    }
    assert_eq!(rest, [65, 129, 2049, 4095]);
    drop(cursor);
    collect(&store);
    assert!(store.shared.read(store.threads).get(p).is_none());
    assert_eq!(alg.evaluate(Function::FALSE, &[0]), 0);
    assert_eq!(alg.evaluate(Function::TRUE, &[4095]), 1);
}

#[test]
fn concurrent_operations_preserve_canonical_roots_across_collection() {
    let store = Arc::new(SparseStore::with_capacity(2, 0, 2));
    let alg = bind(&store, BV6);
    let barrier = Barrier::new(4);
    let roots = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let alg = &*alg;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let a = set(alg, (0..64).filter(|x| x & 1 != 0));
                    let b = set(alg, (0..64).filter(|x| x & 16 != 0));
                    let roots: Vec<_> = (0..8).map(|_| alg.apply(BoolOp::Xor, a, b)).collect();
                    alg.dec_ref(a);
                    alg.dec_ref(b);
                    roots
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    collect(&store);
    for &root in &roots {
        assert_eq!(root, roots[0]);
        for x in [0, 1, 16, 17, 62, 63] {
            assert_eq!(holds(&*alg, root, x), (x & 1 != 0) != (x & 16 != 0));
        }
        alg.dec_ref(root);
    }
    collect(&store);
    assert!(store.shared.read(store.threads).get(roots[0]).is_none());
}

// --- unique table ------------------------------------------------------------

#[test]
fn different_layouts_share_edge_words_across_collection() {
    let store = Arc::new(SparseStore::with_capacity(16, 32, 1));
    let a = bind(&store, ScalarSort::bv(64));
    let b = bind(&store, ScalarSort::bv(320));
    // The same six words encode transitions at 1, 2, and 3 in a, and one
    // five-word key followed by payload 1 in b. Both start at payload zero.
    let holes = a.set(&[0, 2]);
    let f = a.apply(BoolOp::AndNot, Function::TRUE, holes);
    let key = [1, 1, 2, 0, 3];
    let cut = b.lt(&key);
    let g = b.apply(BoolOp::AndNot, Function::TRUE, cut);
    assert_eq!(f, g);
    assert_eq!([0, 1, 2, 3].map(|v| a.evaluate(f, &[v])), [0, 1, 0, 1]);
    a.dec_ref(holes);
    a.dec_ref(f);
    b.dec_ref(cut);
    collect(&store);
    assert_eq!(b.evaluate(g, &[0, 1, 2, 0, 3]), 0);
    assert_eq!(b.evaluate(g, &key), 1);
    let cut = b.lt(&key);
    let again = b.apply(BoolOp::AndNot, Function::TRUE, cut);
    assert_eq!(again, g);
    for root in [g, cut, again] {
        b.dec_ref(root);
    }
}

/// Checks that a collision differing in its initial payload or edges stays
/// distinct.
///
/// The raw candidate deliberately uses the requested map's hash, forcing the
/// interning equality check to examine it. These are storage fixtures, not
/// interpreted partitions. The table has room for both nodes and never runs
/// maintenance, which would recompute the candidate's structural hash.
fn collision(initial: u32, last: u32) {
    let store = SparseStore::with_capacity(8, 64, 1);
    let state = store.shared.read(store.threads);
    let entries = [1_u64 << 32 | 2, 2_u64 << 32 | 3];
    let p = state
        .intern(
            dag::structural_hash((0_u32, entries.as_slice())),
            |_, _| false,
            || Node { initial },
            &[entries[0], 2_u64 << 32 | u64::from(last)],
        )
        .unwrap();
    state.inc_ref(p);
    let q = state.intern_referenced(0, &entries).unwrap();
    assert_ne!(p, q);
    let node = state.node(q);
    assert_eq!(node.initial, 0);
    assert_eq!(node.edges, entries);
    let r = state.intern_referenced(0, &entries).unwrap();
    assert_eq!(q, r);
    state.dec_ref(p);
    state.dec_ref(q);
    state.dec_ref(r);
    assert_eq!(epoch(&store), 0);
}

#[test]
fn a_unique_table_collision_compares_the_initial_payload() {
    collision(1, 3);
}

#[test]
fn a_unique_table_collision_compares_the_edge_words() {
    collision(0, 4);
}
