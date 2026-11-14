//! Concrete checks for the sparse backend.
//!
//! The table is driven directly through [`Column`], since these are the raw
//! map contracts. Every result is checked against plain arithmetic over the
//! sort's values rather than against another map.
//!
//! A sort chooses the transition layout: at most 32 bits packs the key above
//! its slot in one word, wider sorts store key words followed by a slot word,
//! with the two-word width known statically. Every layout-dependent operation
//! is therefore exercised at all three widths.
//!
//! Most operations also run on a table built with no room for a node or an
//! edge word, so the first attempt fails, the table collects and grows, and
//! the retry path is the one under test.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use num_bigint::BigUint;

use super::{NeedGrowth, Node, Packed, SparseTable};
use crate::backend::{Alignment, Backend, BoolOp, Column, Function, MapSpec, Refinement, Values};
use crate::scalar::ScalarSort;
use crate::util::dag::structural_hash;

/// A one-word packed sort.
const BV4: ScalarSort = ScalarSort::bit_vec(4);
/// Keys packed above their slot in one word.
const BV12: ScalarSort = ScalarSort::bit_vec(12);
/// The widest packed sort.
const BV32: ScalarSort = ScalarSort::bit_vec(32);
/// The two wide layouts: a statically known pair of key words, and a run-time
/// key width.
const WIDE: [ScalarSort; 2] = [ScalarSort::bit_vec(100), ScalarSort::bit_vec(200)];

fn growths(store: &SparseTable) -> u64 {
    store.shared.epoch()
}

fn collect(store: &SparseTable) {
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
fn tiny() -> Arc<SparseTable> {
    Arc::new(SparseTable::with_capacity(2, 0, 1))
}

/// A roomy table.
fn roomy() -> Arc<SparseTable> {
    Arc::new(SparseTable::default())
}

/// A backend binding every column to one observed table.
struct SharedTable(Arc<SparseTable>);

impl Backend for SharedTable {
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        Arc::clone(&self.0).column(sort)
    }
}

/// The number of live referenced nodes in a table.
fn referenced(store: &SparseTable) -> usize {
    let state = store.shared.read(store.threads);
    (0..state.node_capacity())
        .map(Function::new)
        .filter(|&f| state.get(f).is_some() && state.has_refs(f))
        .count()
}

/// The table bound to one sort.
fn col(store: &Arc<SparseTable>, d: ScalarSort) -> Box<dyn Column> {
    Arc::clone(store).column(d).unwrap()
}

/// Every member of a predicate, one row of the sort's words each.
fn rows(c: &dyn Column, p: Function) -> Vec<Vec<u64>> {
    let mut cursor = c.members(p);
    let (mut out, mut v) = (Vec::new(), Vec::new());
    while cursor.next(&mut v) {
        out.push(v.clone());
    }
    out
}

/// Every member of a one-word sort.
fn members(c: &dyn Column, p: Function) -> Vec<u64> {
    rows(c, p).iter().map(|row| row[0]).collect()
}

/// The predicate for the sorted values of a one-word sort.
fn set(c: &dyn Column, values: impl IntoIterator<Item = u64>) -> Function {
    c.set(&values.into_iter().collect::<Vec<_>>())
}

/// A row of `d.words()` words holding `v`.
fn word(d: ScalarSort, v: u64) -> Vec<u64> {
    let mut row = vec![0; d.words()];
    row[0] = v;
    row
}

/// The rows of `d.words()` words holding each of `values`, laid end to end.
fn keys(d: ScalarSort, values: impl IntoIterator<Item = u64>) -> Vec<u64> {
    values.into_iter().flat_map(|v| word(d, v)).collect()
}

fn holds(c: &dyn Column, f: Function, x: u64) -> bool {
    c.evaluate(f, &[x]) != 0
}

/// A partition of a one-word sort with cells `{10, 30}`, `{20}`, `{40}` and
/// the rest, labeled with the given case labels.
fn parts(c: &dyn Column, default: u32, labels: [u32; 4]) -> (Function, Vec<u32>) {
    let mut slot_labels = Vec::new();
    let f = c.values(
        default,
        &keys(c.sort(), [10, 20, 30, 40]),
        &labels,
        &mut slot_labels,
        &mut Values::default(),
    );
    (f, slot_labels)
}

/// The slots a refinement reaches, with the first word of each member of the
/// delivered cell.
fn cells(
    c: &dyn Column,
    f: Function,
    p: Function,
    select: impl Fn(u32) -> bool,
) -> Vec<(u32, Vec<u64>)> {
    let mut out = Vec::new();
    let _ = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |slot| select(slot),
        &mut |slot, build| {
            let cell = build();
            out.push((slot, members(c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    out.sort();
    out
}

// --- columns and roots -------------------------------------------------------

#[test]
fn every_scalar_sort_binds_a_column() {
    let store: Arc<dyn Backend> = roomy();
    for sort in [
        ScalarSort::bool(),
        ScalarSort::bit_vec(1),
        BV4,
        BV32,
        ScalarSort::bit_vec(40),
        ScalarSort::bit_vec(70),
        ScalarSort::bit_vec(200),
        ScalarSort::symbol(),
    ] {
        let c = Arc::clone(&store).column(sort).unwrap();
        assert_eq!(c.sort(), sort);
    }
}

#[test]
fn repeated_bindings_share_live_roots() {
    let store = tiny();
    let a = col(&store, BV4);
    let b = col(&store, BV4);
    let p = a.set(&[1, 3]);
    let q = b.set(&[2, 3]);
    let r = a.apply_bool(BoolOp::Or, p, q);
    let s = b.set(&[1, 2, 3]);
    assert_eq!(r, s);
    for x in 0..16 {
        assert_eq!(b.evaluate(p, &[x]), u32::from(x == 1 || x == 3));
        assert_eq!(a.evaluate(q, &[x]), u32::from(x == 2 || x == 3));
        assert_eq!(b.evaluate(r, &[x]), u32::from((1..=3).contains(&x)));
    }
    b.dec_ref(p);
    a.dec_ref(q);
    b.dec_ref(r);
    a.dec_ref(s);
}

#[test]
fn dec_refs_releases_every_root_once() {
    let store = tiny();
    let c = col(&store, BV4);
    let (p, q) = (c.set(&[1, 3]), c.set(&[2]));
    c.inc_ref(p);
    // One batch releases one reference per entry; constants are skipped and
    // the second reference keeps `p` live.
    c.dec_refs(&[p, q, Function::TRUE, Function::FALSE, p]);
    let state = store.shared.read(store.threads);
    assert!(!state.has_refs(p) && !state.has_refs(q));
}

#[test]
fn outer_collection_releases_only_reclaimed_nodes_partitions() {
    use crate::context::Context;
    use crate::index::{Labels, Level, intern};
    use crate::util::dag::Idx;

    let store = roomy();
    let mut cx = Context::default();
    let backend = cx.register_backend("observed", SharedTable(Arc::clone(&store)));
    let column = cx.column(backend, BV32);
    let idx = intern(&cx, &[column]);
    let mut labels = Labels::default();
    let node = idx.values(Level::new(0), &[1], &[Function::TRUE], &mut labels);
    let p = idx.head(node.function).1.function;
    // Every node holds one reference to its partition. Dropping its root
    // leaves the node published until outer collection releases that reference.
    drop(node);
    let kept = idx.values(Level::new(0), &[7], &[Function::TRUE], &mut labels);
    let q = idx.head(kept.function).1.function;
    let data = cx.layouts.lock().unwrap()[0].clone();
    let mut first = true;
    data.shared.retry((&cx, &data.columns), |_| {
        if first {
            first = false;
            Err(NeedGrowth::Nodes)
        } else {
            Ok(())
        }
    });
    {
        let state = store.shared.read(store.threads);
        assert!(!state.has_refs(p));
        assert!(state.has_refs(q));
    }
    // Inspect ownership before backend collection can recycle either token,
    // then check that only the unreferenced partition is reclaimed.
    collect(&store);
    {
        let state = store.shared.read(store.threads);
        assert!(state.get(p).is_none());
        assert!(state.has_refs(q));
    }
    assert!(kept.contains(&[7]) && !kept.contains(&[1]));
}

#[test]
fn context_teardown_releases_index_partitions() {
    use crate::context::Context;
    use crate::index::{Labels, Level, intern};
    use crate::util::dag::Idx;

    let store = roomy();
    let p = {
        let mut cx = Context::default();
        let backend = cx.register_backend("observed", SharedTable(Arc::clone(&store)));
        let column = cx.column(backend, BV32);
        let idx = intern(&cx, &[column]);
        // Dropping every root leaves the node published until collection, so
        // the context releases its partition at teardown.
        let node = idx.values(
            Level::new(0),
            &[1],
            &[Function::TRUE],
            &mut Labels::default(),
        );
        idx.head(node.function).1.function
    };
    let state = store.shared.read(store.threads);
    assert!(state.get(p).is_some() && !state.has_refs(p));
}

#[test]
fn index_retries_release_every_partition_reference() {
    use std::collections::BTreeSet;

    use crate::context::Context;
    use crate::index::{IndexData, IndexNode, IndexRoot, Labels, Level, State, intern};
    use crate::util::dag::{Idx, Shared};

    /// A referenced root holding exactly the sorted, distinct `rows`.
    fn build(index: &IndexRoot, level: usize, rows: &[Vec<u64>], labels: &mut Labels) -> IndexRoot {
        if rows.is_empty() {
            return index.inc_ref(Function::FALSE);
        }
        if level == index.arity() {
            return index.inc_ref(Function::TRUE);
        }
        let (mut keys, mut children) = (Vec::new(), Vec::new());
        for group in rows.chunk_by(|x, y| x[level] == y[level]) {
            keys.push(group[0][level]);
            children.push(build(index, level + 1, group, labels));
        }
        let conts: Vec<_> = children.iter().map(|c| c.function).collect();
        index.values(Level::new(level), &keys, &conts, labels)
    }

    let store = roomy();
    let mut cx = Context::default();
    let backend = cx.register_backend("observed", SharedTable(Arc::clone(&store)));
    let columns: Vec<_> = (0..3).map(|_| cx.column(backend, BV4)).collect();
    // A tiny container fails interning at every depth, so operations retry
    // after releasing their frames' partitions.
    cx.layouts.lock().unwrap().push(Arc::new(IndexData {
        columns: columns.as_slice().into(),
        shared: Shared::new(State::with_capacity(
            4,
            4,
            (0..2).map(|_| IndexNode {
                level: Level::TERMINAL,
                partition: Function::FALSE,
                hole: u32::MAX,
            }),
        )),
    }));
    let base = referenced(&store);
    let idx = intern(&cx, &columns);
    let mut labels = Labels::default();
    let mut seed = 7_u64;
    let mut next = move |n: u64| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % n
    };
    let mut held = Vec::new();
    for _ in 0..40 {
        let mut sets: [BTreeSet<Vec<u64>>; 2] = Default::default();
        for set in &mut sets {
            for _ in 0..next(30) {
                set.insert(vec![next(16), next(16), next(16)]);
            }
        }
        let [l, r] = sets.each_ref().map(|set| {
            build(
                &idx,
                0,
                &set.iter().cloned().collect::<Vec<_>>(),
                &mut labels,
            )
        });
        for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
            held.push(idx.apply_bool(op, l.function, r.function));
        }
        held.push(l.min_at(Level::new(1)));
        held.extend([l, r]);
    }
    let data = cx.layouts.lock().unwrap()[0].clone();
    assert!(data.shared.epoch() > 0, "the container was maintained");
    drop(held);
    let mut first = true;
    data.shared.retry((&cx, &data.columns), |_| {
        if first {
            first = false;
            Err(NeedGrowth::Nodes)
        } else {
            Ok(())
        }
    });
    collect(&store);
    assert_eq!(referenced(&store), base);
}

#[test]
fn a_column_retains_the_store_and_roots_from_another_binding() {
    let store = tiny();
    let a = col(&store, BV12);
    let b = col(&store, BV12);
    let p = a.set(&[1, 65, 4095]);
    drop(a);
    drop(store);
    assert_eq!(b.evaluate(p, &[65]), 1);
    assert_eq!(b.evaluate(p, &[64]), 0);
    assert_eq!(members(&*b, p), [1, 65, 4095]);
    b.dec_ref(p);
}

// --- sets, cuts, and counts --------------------------------------------------

#[test]
fn sets_cuts_and_counts_survive_growth() {
    let store = tiny();
    let c = col(&store, BV12);
    let values: Vec<u64> = (0..300).map(|v| v * 13).collect();
    let p = c.set(&values);
    assert_eq!(members(&*c, p), values);
    assert_eq!(c.count(p), BigUint::from(300_u32));
    // Adjacent points coalesce into one run rather than one transition each.
    let run = set(&*c, 100..200);
    assert_eq!(members(&*c, run), (100..200).collect::<Vec<_>>());
    let lt = c.less_than(&[1000]);
    assert_eq!(members(&*c, lt), (0..1000).collect::<Vec<_>>());
    // The final run reaches the sort's exact boundary, which no transition
    // marks.
    let ge = c.apply_bool(BoolOp::AndNot, Function::TRUE, lt);
    assert_eq!(c.count(ge), BigUint::from(4096_u32 - 1000));
    // A cut at zero is the empty map, which is the constant rather than a run.
    assert_eq!(c.less_than(&[0]), Function::FALSE);
    assert_eq!(c.set(&[]), Function::FALSE);
    assert!(
        growths(&store) > 0,
        "a table this small cannot hold any of these results"
    );
}

#[test]
fn counts_reach_the_universe_boundary_at_every_layout() {
    // One-word sorts up to the full 64 bits, then the static and run-time wide
    // layouts, including a width whose universe count needs a word above the
    // key words.
    let sorts = [4, 32, 33, 64, 100, 128, 200].map(ScalarSort::bit_vec);
    for d in sorts {
        let c = col(&tiny(), d);
        let universe = BigUint::from(1_u32) << d.width();
        let ge = c.apply_bool(BoolOp::AndNot, Function::TRUE, c.less_than(&word(d, 10)));
        let p = c.apply_bool(BoolOp::Or, c.set(&keys(d, [1, 2, 3, 7])), ge);
        assert_eq!(c.count(p), &universe + 4_u32 - 10_u32, "{d:?}");
        let holes = c.apply_bool(BoolOp::AndNot, Function::TRUE, c.set(&keys(d, [0, 3])));
        assert_eq!(c.count(holes), &universe - 2_u32, "{d:?}");
        assert_eq!(c.count(Function::TRUE), universe, "{d:?}");
        assert_eq!(c.count(Function::FALSE), BigUint::ZERO, "{d:?}");
        // A predicate holding at zero opens its first run there.
        let low = c.apply_bool(BoolOp::Or, c.less_than(&word(d, 10)), c.set(&keys(d, [12])));
        assert_eq!(c.count(low), BigUint::from(11_u32), "{d:?}");
    }
}

#[test]
fn an_empty_count_stays_small_for_a_wide_universe() {
    let c = col(&tiny(), ScalarSort::bit_vec(1 << 30));
    assert_eq!(c.count(Function::FALSE), BigUint::ZERO);
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_with_more_than_two_slots_is_rejected() {
    let c = col(&roomy(), BV4);
    let cases = [(set(&*c, [1]), 2), (set(&*c, [5]), 3)];
    let f = c.cases(0, &cases, &mut Vec::new());
    c.count(f);
}

#[test]
#[should_panic(expected = "symbols have no numeric ordering")]
fn a_symbol_cut_is_rejected() {
    let _ = col(&roomy(), ScalarSort::symbol()).less_than(&[3]);
}

#[test]
#[should_panic(expected = "this backend cannot construct masked bit predicates")]
fn a_bit_test_is_rejected() {
    let _ = col(&roomy(), BV4).masked_eq(&[1], &[1]);
}

// --- Boolean operations ------------------------------------------------------

#[test]
fn boolean_operations_merge_runs_of_both_operands() {
    let store = tiny();
    let c = col(&store, BV12);
    let (a, b) = (set(&*c, (0..300).map(|v| v * 13)), set(&*c, 100..200));
    let (in_a, in_b) = (
        |x: u64| x.is_multiple_of(13) && x < 300 * 13,
        |x: u64| (100..200).contains(&x),
    );
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        let r = c.apply_bool(op, a, b);
        let expect: Vec<u64> = (0..4096)
            .filter(|&x| op.evaluate_bool(in_a(x), in_b(x)))
            .collect();
        assert_eq!(members(&*c, r), expect, "{op:?}");
        c.dec_ref(r);
    }
    // Equal operands and constants never reach the merge at all.
    assert_eq!(c.apply_bool(BoolOp::Xor, a, a), Function::FALSE);
    assert_eq!(c.apply_bool(BoolOp::And, a, Function::TRUE), a);
    assert!(growths(&store) > 0);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn an_unreferenced_boolean_operand_is_rejected() {
    let c = col(&roomy(), BV4);
    let (p, q) = (set(&*c, [1]), set(&*c, [2]));
    c.dec_ref(p);
    let _ = c.apply_bool(BoolOp::And, q, p);
}

// --- images and preimages ----------------------------------------------------

#[test]
fn maps_transfer_the_members_of_a_predicate() {
    let store = tiny();
    let c = col(&store, BV12);
    let (p, q) = (set(&*c, [3, 65, 2114, 4095]), set(&*c, [0, 21, 2070, 4051]));
    // `x - 44`, undefined below it, so the low member drops out of both
    // directions.
    let map = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = x[0].saturating_sub(44);
        x[0] >= 44
    }));
    assert_eq!(members(&*c, c.image(&map, p, q)), [21, 2070, 4051]);
    assert_eq!(members(&*c, c.preimage(&map, p, q)), [65, 2114, 4095]);
    // A bit map is enumerated the same way.
    let bits = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let moved = |x: u64| (x & !0x3F) ^ 0x55;
    let (in_p, in_q) = (members(&*c, p), members(&*c, q));
    let mut expect: Vec<u64> = in_p
        .iter()
        .map(|&x| moved(x))
        .filter(|y| in_q.contains(y))
        .collect();
    expect.sort_unstable();
    assert_eq!(members(&*c, c.image(&bits, p, q)), expect);
    let expect: Vec<u64> = in_p
        .iter()
        .copied()
        .filter(|&x| in_q.contains(&moved(x)))
        .collect();
    assert_eq!(members(&*c, c.preimage(&bits, p, q)), expect);
    // The identity needs no enumeration in either direction: it is a
    // conjunction.
    let id = MapSpec::identity(1);
    assert_eq!(c.image(&id, p, q), Function::FALSE);
    assert_eq!(c.preimage(&id, p, p), p);
    assert!(growths(&store) > 0);
}

#[test]
fn empty_transfer_targets_skip_source_enumeration_after_domain_validation() {
    let c = col(&roomy(), ScalarSort::bit_vec(8));
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let map = MapSpec::Rust(Arc::new(move |x, y| {
        count.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        true
    }));
    assert_eq!(
        c.image(&map, Function::TRUE, Function::FALSE),
        Function::FALSE
    );
    assert_eq!(
        c.preimage(&map, Function::TRUE, Function::FALSE),
        Function::FALSE
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let invalid = MapSpec::Bits {
        clear: vec![0, 0].into(),
        flip: vec![0, 0].into(),
    };
    for image in [true, false] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                if image {
                    c.image(&invalid, Function::TRUE, Function::FALSE)
                } else {
                    c.preimage(&invalid, Function::TRUE, Function::FALSE)
                }
            }))
            .is_err()
        );
    }
}

// --- conditionals ------------------------------------------------------------

#[test]
fn conditionals_advance_every_operand_over_shared_keys() {
    let store = tiny();
    let c = col(&store, BV12);
    // The branches share one transition point with the condition and one with
    // each other, so the sweep must advance several runs at the same key.
    let k = set(&*c, 100..200);
    let y = set(&*c, (100..300).step_by(2));
    let n = set(&*c, [50, 150, 250]);
    let r = c.ite(k, y, n);
    let expect: Vec<u64> = (0..4096)
        .filter(|&x| {
            if (100..200).contains(&x) {
                (100..300).contains(&x) && x % 2 == 0
            } else {
                [50, 150, 250].contains(&x)
            }
        })
        .collect();
    assert_eq!(members(&*c, r), expect);
    // A constant condition selects one branch outright, retaining its existing
    // root.
    assert_eq!(c.ite(Function::TRUE, y, n), y);
    assert_eq!(c.ite(Function::FALSE, y, n), n);
    assert!(growths(&store) > 0);
}

#[test]
fn trivial_conditionals_retain_the_selected_root_after_collection() {
    let store = tiny();
    let c = col(&store, BV12);
    let p = set(&*c, [10, 20]);
    let q = set(&*c, (0..4096).step_by(2));
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

// --- alignment and quotient --------------------------------------------------

#[test]
fn alignment_and_quotient_renumber_by_first_occurrence() {
    let store = tiny();
    let c = col(&store, BV12);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let g = set(&*c, [20, 30]);
    let mut labels = vec![(9, 9)];
    let mut scratch = Alignment::default();
    let h = c.align(f, g, &mut labels, &mut scratch);
    // The existing prefix is preserved and result slots index only the appended
    // suffix.
    assert_eq!(labels[0], (9, 9));
    for x in 0..4096 {
        let slot = c.evaluate(h, &[x]) as usize;
        let expect = (c.evaluate(f, &[x]), c.evaluate(g, &[x]));
        assert_eq!(labels[1..][slot], expect, "at {x}");
    }
    // Coarsening merges the two singleton cells, and the runs that become equal
    // coalesce.
    let q = c.quotient(f, &[0, 1, 1, 2], &mut scratch);
    for x in 0..4096 {
        let expect = match x {
            10 | 20 | 30 => 1,
            40 => 2,
            _ => 0,
        };
        assert_eq!(c.evaluate(q, &[x]), expect, "at {x}");
    }
    assert!(growths(&store) > 0);
}

// --- cases -------------------------------------------------------------------

#[test]
fn cases_fold_onto_labels_over_their_boundaries() {
    let store = tiny();
    let c = col(&store, BV12);
    let (a, b) = (set(&*c, 100..200), set(&*c, 150..300));
    let mut labels = Vec::new();
    // Overlapping cases agree on their label, so the first one ending leaves the
    // cell covered. The gap above them falls back to the default.
    let f = c.cases(9, &[(a, 4), (b, 4)], &mut labels);
    assert_eq!(labels, [9, 4]);
    for x in 0..4096 {
        assert_eq!(
            c.evaluate(f, &[x]),
            u32::from((100..300).contains(&x)),
            "at {x}"
        );
    }
    // A single case labeled like the default is one cell, and is swept like any
    // other.
    let mut labels = Vec::new();
    assert_eq!(c.cases(4, &[(a, 4)], &mut labels), Function::FALSE);
    assert_eq!(labels, [4]);
    assert!(growths(&store) > 0);
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn an_overlap_between_cases_with_different_labels_is_rejected() {
    let c = col(&roomy(), BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

#[test]
fn a_boolean_case_predicate_is_already_its_own_partition() {
    let c = col(&roomy(), BV12);
    let p = set(&*c, 100..200);
    let mut labels = Vec::new();
    // Zero is outside the case, so the predicate's own payloads are the
    // canonical slots.
    assert_eq!(c.cases(8, &[(p, 3)], &mut labels), p);
    assert_eq!(labels, [8, 3]);
    // One holding at zero is not, and is rebuilt.
    let q = c.less_than(&[200]);
    labels.clear();
    let f = c.cases(8, &[(q, 3)], &mut labels);
    assert_eq!(labels, [3, 8]);
    assert_eq!(
        (0..4096)
            .map(|x| c.evaluate(f, &[x]))
            .filter(|&s| s == 0)
            .count(),
        200
    );
    // Neither is a constant case, which has no transition to read at all.
    labels.clear();
    assert_eq!(
        c.cases(8, &[(Function::TRUE, 3)], &mut labels),
        Function::FALSE
    );
    assert_eq!(labels, [3]);
}

// --- values ------------------------------------------------------------------

#[test]
fn point_labels_become_first_occurrence_slots() {
    let store = tiny();
    let c = col(&store, BV12);
    let (f, labels) = parts(&*c, 4, [7, 8, 7, 9]);
    assert_eq!(labels, [4, 7, 8, 9]);
    for x in 0..4096 {
        let expect = match x {
            10 | 30 => 1,
            20 => 2,
            40 => 3,
            _ => 0,
        };
        assert_eq!(c.evaluate(f, &[x]), expect, "at {x}");
    }
    assert!(growths(&store) > 0);
}

// --- refinement --------------------------------------------------------------

#[test]
fn refinement_groups_the_accepted_runs_by_slot() {
    let store = tiny();
    let c = col(&store, BV12);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let p = set(&*c, [10, 20, 25, 30]);
    assert_eq!(
        cells(&*c, f, p, |_| true),
        [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]
    );
    // A declined slot is never delivered, and its boundaries leave no trace in
    // the others.
    assert_eq!(cells(&*c, f, p, |slot| slot == 1), [(1, vec![10, 30])]);
    // One reached slot means the predicate itself is that cell, whatever its
    // shape.
    assert_eq!(
        cells(&*c, f, set(&*c, [10, 30]), |_| true),
        [(1, vec![10, 30])]
    );
    // A false predicate reaches no slot at all.
    assert_eq!(
        cells(&*c, f, Function::FALSE, |_| panic!("no slot is reached")),
        []
    );
    // A wide interval spans whole cells and the gaps between them.
    assert_eq!(
        cells(&*c, f, set(&*c, 10..31), |slot| slot != 0),
        [(1, vec![10, 30]), (2, vec![20])]
    );
    assert!(growths(&store) > 0);
    // A visitor that breaks stops the walk immediately.
    let mut count = 0;
    let stopped = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |_, _| {
            count += 1;
            ControlFlow::Break(())
        },
    );
    assert!(stopped.is_break() && count == 1);
}

#[test]
fn refinement_of_a_boolean_map_reuses_the_map_itself() {
    let store = tiny();
    let c = col(&store, BV12);
    // A map whose payloads are already Boolean is the indicator of its own slot
    // one, so the unrestricted refinement is that map and its payload flip,
    // without rebuilding either.
    let f = set(&*c, [10, 20, 30]);
    assert_eq!(
        cells(&*c, f, Function::TRUE, |_| true)
            .into_iter()
            .map(|(slot, cell)| (slot, cell.len()))
            .collect::<Vec<_>>(),
        [(0, 4093), (1, 3)]
    );
    let flipped = cells(&*c, f, Function::TRUE, |slot| slot == 0);
    assert_eq!(flipped[0].1.len(), 4093);
    assert!(!flipped[0].1.contains(&20));
    // The trivial partition has no payload other than its default, so the whole
    // universe is one cell and the predicate is it.
    assert_eq!(
        cells(&*c, Function::FALSE, Function::TRUE, |_| true).len(),
        1
    );
    assert!(growths(&store) > 0);
}

#[test]
fn restricted_boolean_refinement_sizes_scratch_by_reached_slots() {
    let store = tiny();
    let c = col(&store, BV12);
    let f = set(&*c, (0..4096).step_by(4));
    let p = set(&*c, [4, 5]);
    let mut scratch = Refinement::default();
    let mut found = Vec::new();
    assert!(
        c.refine(f, p, &mut scratch, &mut |_| true, &mut |slot, build| {
            let cell = build();
            found.push((slot, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        })
        .is_continue()
    );
    found.sort();
    assert_eq!(found, [(0, vec![5]), (1, vec![4])]);
    assert_eq!(scratch.slots.len(), 2);
    c.dec_ref(f);
    c.dec_ref(p);
}

#[test]
fn refinement_delivery_can_collect_and_reenter_the_store() {
    let store = tiny();
    let c = col(&store, BV12);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let p = set(&*c, [10, 20, 25, 30]);
    let mut seen = Vec::new();
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |slot, build| {
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
            assert_eq!(nested, [slot]);
            seen.push((slot, members(&*c, b)));
            c.dec_ref(b);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn refinement_of_an_unreferenced_partition_is_rejected() {
    let c = col(&roomy(), BV4);
    let f = set(&*c, [1, 2]);
    c.dec_ref(f);
    let _ = c.refine(
        f,
        Function::TRUE,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |_, _| ControlFlow::Continue(()),
    );
}

// --- layouts -----------------------------------------------------------------

#[test]
fn refinement_over_wide_keys_recognizes_a_single_point() {
    for d in WIDE {
        let c = col(&roomy(), d);
        let mut labels = Vec::new();
        let f = c.values(
            0,
            &keys(d, [10, 20, 30, 40]),
            &[1, 2, 1, 3],
            &mut labels,
            &mut Values::default(),
        );
        assert_eq!(labels, [0, 1, 2, 3]);
        // A point predicate holds between two consecutive keys, so its cell is
        // itself.
        let p = c.set(&word(d, 20));
        assert_eq!(rows(&*c, p), [word(d, 20)]);
        assert_eq!(cells(&*c, f, p, |_| true), [(2, vec![20])]);
        assert_eq!(cells(&*c, f, p, |_| false), []);
        // Two adjacent points hold between keys that are two apart, so the sweep
        // runs instead.
        let q = c.set(&keys(d, [10, 11]));
        assert_eq!(cells(&*c, f, q, |_| true), [(0, vec![11]), (1, vec![10])]);
        // The layout also reads and writes the slot word of a wide quotient.
        let coarse = c.quotient(f, &[0, 1, 1, 2], &mut Alignment::default());
        assert_eq!(c.evaluate(coarse, &word(d, 20)), 1);
        assert_eq!(
            c.count(c.apply_bool(BoolOp::And, q, q)),
            BigUint::from(2_u32)
        );
    }
}

#[test]
fn wide_keys_drive_the_cut_the_conditional_and_the_folds() {
    for d in WIDE {
        let c = col(&roomy(), d);
        let (a, b) = (c.set(&keys(d, 100..200)), c.set(&keys(d, 150..300)));
        let lt = c.less_than(&word(d, 250));
        assert_eq!(
            c.count(c.apply_bool(BoolOp::And, b, lt)),
            BigUint::from(100_u32)
        );
        // Below the cut the first branch answers, above it the second.
        let r = c.ite(lt, a, b);
        let expect: Vec<Vec<u64>> = (100..200).chain(250..300).map(|v| word(d, v)).collect();
        assert_eq!(rows(&*c, r), expect);
        // A single case over a predicate that is false at zero is already its own
        // partition.
        let mut labels = Vec::new();
        assert_eq!(c.cases(9, &[(a, 4)], &mut labels), a);
        assert_eq!(labels, [9, 4]);
        // A case holding at zero opens its cell before the first boundary, so the
        // cell below the other case carries its label rather than the default.
        let low = c.less_than(&word(d, 100));
        labels.clear();
        let f = c.cases(9, &[(low, 4), (a, 5)], &mut labels);
        assert_eq!(labels, [4, 5, 9]);
        for v in [0, 99, 100, 199, 200] {
            let expect = if v < 100 {
                4
            } else if v < 200 {
                5
            } else {
                9
            };
            assert_eq!(
                labels[c.evaluate(f, &word(d, v)) as usize],
                expect,
                "at {v}"
            );
        }
        // Overlapping cases agree on their label, and the gap above them takes
        // the default.
        labels.clear();
        let f = c.cases(9, &[(a, 4), (b, 4)], &mut labels);
        assert_eq!(labels, [9, 4]);
        for v in [0, 100, 250, 300] {
            assert_eq!(
                c.evaluate(f, &word(d, v)),
                u32::from((100..300).contains(&v)),
                "at {v}"
            );
        }
        let mut pairs = Vec::new();
        let h = c.align(f, lt, &mut pairs, &mut Alignment::default());
        for v in [0, 100, 250, 300] {
            let slot = c.evaluate(h, &word(d, v)) as usize;
            assert_eq!(
                pairs[slot],
                (c.evaluate(f, &word(d, v)), c.evaluate(lt, &word(d, v))),
                "at {v}"
            );
        }
    }
}

#[test]
fn one_word_keys_above_32_bits_use_the_run_time_layout() {
    // A sort between 33 and 64 bits is one key word plus a slot word, the
    // run-time width arm with a single word, which neither packs nor fixes its
    // width statically.
    for d in [ScalarSort::bit_vec(40), ScalarSort::symbol()] {
        let store = tiny();
        let c = col(&store, d);
        let high = 1 << 36;
        let p = set(&*c, [3, high, high + 1, high + 2, u64::MAX >> 24]);
        assert_eq!(
            members(&*c, p),
            [3, high, high + 1, high + 2, u64::MAX >> 24]
        );
        assert_eq!(c.count(p), BigUint::from(5_u32));
        let q = set(&*c, [high + 1, high + 5]);
        assert_eq!(
            members(&*c, c.apply_bool(BoolOp::AndNot, p, q)),
            [3, high, high + 2, u64::MAX >> 24]
        );
        let mut labels = Vec::new();
        let f = c.cases(0, &[(p, 1), (q, 1)], &mut labels);
        assert_eq!(labels, [0, 1]);
        assert_eq!(
            members(&*c, f),
            [3, high, high + 1, high + 2, high + 5, u64::MAX >> 24]
        );
        assert_eq!(
            cells(&*c, f, set(&*c, [2, 3, high + 2, high + 3]), |_| true),
            [(0, vec![2, high + 3]), (1, vec![3, high + 2])]
        );
        assert!(growths(&store) > 0);
    }
    // The complement of a symbol set counts to the full 64-bit universe without
    // enumerating it.
    let c = col(&roomy(), ScalarSort::symbol());
    let p = set(&*c, [7, u64::MAX]);
    let n = c.apply_bool(BoolOp::AndNot, Function::TRUE, p);
    assert_eq!(c.count(n), (BigUint::from(1_u32) << 64) - 2_u32);
    assert!(holds(&*c, n, 0) && holds(&*c, n, 8) && !holds(&*c, n, u64::MAX));
}

// --- members -----------------------------------------------------------------

#[test]
fn member_cursors_cross_word_boundaries() {
    let c = col(&roomy(), ScalarSort::bit_vec(70));
    // A run ending at the word boundary steps the cursor across it.
    let sorted = [5, 0, u64::MAX, 0, 0, 1, 5, 1];
    let q = c.set(&sorted);
    let mut cursor = c.members(q);
    let (mut out, mut v) = (Vec::new(), Vec::new());
    while cursor.next(&mut v) {
        out.extend_from_slice(&v);
    }
    assert_eq!(out, sorted);
    assert!(rows(&*c, Function::FALSE).is_empty());
    assert_eq!(c.count(Function::TRUE), BigUint::from(1_u32) << 70);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn member_enumeration_of_an_unreferenced_predicate_is_rejected() {
    let c = col(&roomy(), BV4);
    let p = set(&*c, [1]);
    c.dec_ref(p);
    let _ = c.members(p);
}

// --- collection and concurrency ----------------------------------------------

#[test]
fn retained_roots_and_member_cursors_survive_collection() {
    let store = tiny();
    let c = col(&store, BV12);
    let p = set(&*c, [1, 65, 129, 2049, 4095]);
    c.inc_ref(p);
    let mut cursor = c.members(p);
    c.dec_ref(p);
    c.dec_ref(p);
    let mut point = Vec::new();
    assert!(cursor.next(&mut point));
    assert_eq!(point, [1]);
    collect(&store);
    let mut rest = Vec::new();
    while cursor.next(&mut point) {
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
    let store = Arc::new(SparseTable::with_capacity(2, 0, 2));
    let c = col(&store, BV12);
    let barrier = Barrier::new(4);
    let roots = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let c = &*c;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let a = set(c, (0..4096).filter(|x| x & 1 != 0));
                    let b = set(c, (0..4096).filter(|x| x & 64 != 0));
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
        for x in [0, 1, 64, 65, 4094, 4095] {
            assert_eq!(holds(&*c, root, x), (x & 1 != 0) != (x & 64 != 0));
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
    let store = SparseTable::with_capacity(8, 64, 1);
    let state = store.shared.read(store.threads);
    let entries = [1_u64 << 32 | 2, 2_u64 << 32 | 3];
    let p = state
        .intern(
            structural_hash((0_u32, 1_u32, entries.as_slice())),
            |_, _| false,
            || candidate,
            &entries,
        )
        .unwrap();
    state.inc_ref(p);
    let q = state.intern_referenced(Packed, 0, &entries).unwrap();
    assert_ne!(p, q);
    let node = state.node(q);
    assert_eq!((node.default, node.stride), (0, 1));
    assert_eq!(node.edges, entries);
    let r = state.intern_referenced(Packed, 0, &entries).unwrap();
    assert_eq!(q, r);
    state.dec_ref(p);
    state.dec_ref(q);
    state.dec_ref(r);
    assert_eq!(growths(&store), 0);
}

#[test]
fn a_unique_table_collision_compares_the_stored_default() {
    collision(Node {
        default: 1,
        stride: 1,
    });
}

#[test]
fn a_unique_table_collision_compares_the_stored_stride() {
    collision(Node {
        default: 0,
        stride: 2,
    });
}
