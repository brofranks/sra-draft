//! White-box tests for the sparse backend.
//!
//! Tiny stores force maintenance on the first publication. Other fixtures
//! force collection, inspect reclamation, or inject unique-table collisions.

use std::ops::ControlFlow;
use std::sync::{Arc, Barrier};
use std::thread;

use super::{NeedGrowth, Node, Packed, SparseStore};
use crate::backend::{Alignment, Backend, BoolOp, Column, Function, Refinement, Values};
use crate::scalar::ScalarSort;
use crate::util::count::Count;
use crate::util::dag;

#[path = "../support/sparse.rs"]
mod fixtures;

use fixtures::{BV4, BV6, col, holds, members, parts, set};

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
    let c = col(&store, BV6);
    let values: Vec<u64> = (0..12).map(|v| v * 3).collect();
    let before = epoch(&store);
    let p = c.set(&values);
    assert!(
        epoch(&store) > before,
        "the set must retry after maintenance"
    );
    assert_eq!(members(&*c, p), values);
    assert_eq!(c.count(p), Count::from(12));
    // Adjacent points coalesce into one run rather than one transition each.
    let run = set(&*c, 40..48);
    assert_eq!(members(&*c, run), (40..48).collect::<Vec<_>>());
    let lt = c.less_than(&[12]);
    assert_eq!(members(&*c, lt), (0..12).collect::<Vec<_>>());
    // The final run reaches the sort's exact boundary, which no transition
    // marks.
    let ge = c.apply_bool(BoolOp::AndNot, Function::TRUE, lt);
    assert_eq!(c.count(ge), 64 - 12);
    // A cut at zero is the empty map, which is the constant rather than a run.
    assert_eq!(c.less_than(&[0]), Function::FALSE);
    assert_eq!(c.set(&[]), Function::FALSE);
    for f in [p, run, lt, ge] {
        c.dec_ref(f);
    }
}

// --- conditionals ------------------------------------------------------------

#[test]
fn trivial_conditionals_retain_the_selected_root_after_collection() {
    let store = tiny();
    let c = col(&store, BV6);
    let p = set(&*c, [10, 20]);
    let q = set(&*c, (0..64).step_by(2));
    let results = [
        c.ite(Function::TRUE, p, q),
        c.ite(Function::FALSE, q, p),
        c.ite(q, p, p),
        c.ite(p, Function::TRUE, Function::FALSE),
    ];
    c.dec_ref(p);
    c.dec_ref(q);
    collect(&store);
    for f in results {
        assert_eq!(members(&*c, f), [10, 20]);
        c.dec_ref(f);
    }
}

// --- alignment --------------------------------------------------------------

#[test]
fn alignment_appends_one_label_suffix_after_a_retry() {
    let store = Arc::new(SparseStore::with_capacity(8, 7, 1));
    let c = col(&store, BV4);
    let f = c.set(&[1]);
    let g = c.set(&[3]);
    let mut labels = vec![(9, 9)];
    // Four words are live and the alignment needs four more.
    let before = epoch(&store);
    let h = c.align(f, g, &mut labels, &mut Alignment::default());
    assert!(epoch(&store) > before);
    assert_eq!(labels, [(9, 9), (0, 0), (1, 0), (0, 1)]);
    for f in [f, g, h] {
        c.dec_ref(f);
    }
}

// --- values ------------------------------------------------------------------

#[test]
fn point_labels_become_first_occurrence_classes_after_a_retry() {
    let store = tiny();
    let c = col(&store, BV6);
    let before = epoch(&store);
    let (f, labels) = parts(&*c, 4, [7, 8, 7, 9]);
    assert!(epoch(&store) > before);
    assert_eq!(labels, [4, 7, 8, 9]);
    for x in 0..64 {
        let expect = match x {
            10 | 30 => 1,
            20 => 2,
            40 => 3,
            _ => 0,
        };
        assert_eq!(c.evaluate(f, &[x]), expect, "at {x}");
    }
    c.dec_ref(f);
}

// --- refinement --------------------------------------------------------------

#[test]
fn breaking_before_build_leaves_cells_unpublished() {
    let store = Arc::new(SparseStore::with_capacity(4, 32, 1));
    let c = col(&store, BV4);
    let f = c.set(&[1, 2]);
    let p = c.set(&[2, 3]);
    let mut scratch = Refinement::default();
    // The retained roots and constants fill the node arena. Both cells would
    // require a new node.
    let before = epoch(&store);
    let stopped = c.refine(f, p, &mut scratch, &mut |_| true, &mut |_, _| {
        ControlFlow::Break(())
    });
    assert!(stopped.is_break());
    assert_eq!(epoch(&store), before);

    let stopped = c.refine(f, p, &mut scratch, &mut |_| true, &mut |_, build| {
        c.dec_ref(build());
        ControlFlow::Break(())
    });
    assert!(stopped.is_break());
    assert!(epoch(&store) > before);
    c.dec_ref(f);
    c.dec_ref(p);
}

#[test]
fn refinement_delivery_can_collect_and_reenter_the_store() {
    let store = tiny();
    let c = col(&store, BV6);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let p = set(&*c, [10, 20, 25, 30]);
    let mut seen = Vec::new();
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |class, build| {
            collect(&store);
            let a = build();
            let b = build();
            c.dec_ref(a);
            collect(&store);
            let mut nested = Vec::new();
            let nested_result = c.refine(
                f,
                b,
                &mut Refinement::default(),
                &mut |_| true,
                &mut |inner, make| {
                    let selected = make();
                    nested.push(inner);
                    assert_eq!(members(&*c, selected), members(&*c, b));
                    c.dec_ref(selected);
                    ControlFlow::Continue(())
                },
            );
            assert!(nested_result.is_continue());
            assert_eq!(nested, [class]);
            seen.push((class, members(&*c, b)));
            c.dec_ref(b);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]);
    c.dec_ref(f);
    c.dec_ref(p);
}

// --- collection and concurrency ----------------------------------------------

#[test]
fn retained_roots_and_member_cursors_survive_collection() {
    let store = tiny();
    let c = col(&store, ScalarSort::bit_vec(12));
    let p = set(&*c, [1, 65, 129, 2049, 4095]);
    c.inc_ref(p);
    let mut cursor = c.members(p);
    let mut suspended = c.members(p);
    c.dec_ref(p);
    c.dec_ref(p);
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
    assert_eq!(c.evaluate(Function::FALSE, &[0]), 0);
    assert_eq!(c.evaluate(Function::TRUE, &[4095]), 1);
}

#[test]
fn concurrent_operations_preserve_canonical_roots_across_collection() {
    let store = Arc::new(SparseStore::with_capacity(2, 0, 2));
    let c = col(&store, BV6);
    let barrier = Barrier::new(4);
    let roots = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let c = &*c;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let a = set(c, (0..64).filter(|x| x & 1 != 0));
                    let b = set(c, (0..64).filter(|x| x & 16 != 0));
                    let roots: Vec<_> = (0..8).map(|_| c.apply_bool(BoolOp::Xor, a, b)).collect();
                    c.dec_ref(a);
                    c.dec_ref(b);
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
            assert_eq!(holds(&*c, root, x), (x & 1 != 0) != (x & 16 != 0));
        }
        c.dec_ref(root);
    }
    collect(&store);
    assert!(store.shared.read(store.threads).get(roots[0]).is_none());
}

// --- unique table ------------------------------------------------------------

/// Checks that a collision differing only in one node field stays distinct.
///
/// The raw candidate deliberately uses the requested map's hash, forcing the
/// interning equality check to examine it. These are storage fixtures, not
/// interpreted partitions. The table has room for both nodes and never runs
/// maintenance, which would recompute the candidate's structural hash.
fn collision(candidate: Node) {
    let store = SparseStore::with_capacity(8, 64, 1);
    let state = store.shared.read(store.threads);
    let entries = [1_u64 << 32 | 2, 2_u64 << 32 | 3];
    let p = state
        .intern(
            dag::structural_hash((0_u32, 1_u32, entries.as_slice())),
            |_, _| false,
            || candidate,
            &entries,
        )
        .unwrap();
    state.inc_ref(p);
    let q = state.intern_referenced(Packed, 0, &entries).unwrap();
    assert_ne!(p, q);
    let node = state.node(q);
    assert_eq!((node.initial, node.stride), (0, 1));
    assert_eq!(node.edges, entries);
    let r = state.intern_referenced(Packed, 0, &entries).unwrap();
    assert_eq!(q, r);
    state.dec_ref(p);
    state.dec_ref(q);
    state.dec_ref(r);
    assert_eq!(epoch(&store), 0);
}

#[test]
fn a_unique_table_collision_compares_the_initial_payload() {
    collision(Node {
        initial: 1,
        stride: 1,
    });
}

#[test]
fn a_unique_table_collision_compares_the_stored_stride() {
    collision(Node {
        initial: 0,
        stride: 2,
    });
}
