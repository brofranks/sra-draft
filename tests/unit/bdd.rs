//! White-box tests for `src/bdd.rs`.
//!
//! Tests inspect the node layout, level buffer, and memo tables. Tiny stores
//! trigger maintenance during construction, and forced collection checks which
//! roots survive.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::panic::{self, AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier};
use std::thread;

use super::{BddStore, LevelBuffer, Memo, NeedGrowth};
use crate::backend::{Alignment, Backend, BoolOp, Column, Function, MapSpec, Refinement, Values};
use crate::scalar::ScalarSort;

/// One six-bit level, so a diagram is a single node over the terminals.
const BV4: ScalarSort = ScalarSort::bit_vec(4);
/// Two levels.
const BV12: ScalarSort = ScalarSort::bit_vec(12);
/// Three levels, the shallowest sort whose traversals can reach one node pair
/// by two routes.
const BV18: ScalarSort = ScalarSort::bit_vec(18);

fn boolean(op: BoolOp, a: bool, b: bool) -> bool {
    match op {
        BoolOp::And => a && b,
        BoolOp::Or => a || b,
        BoolOp::Xor => a != b,
        BoolOp::AndNot => a && !b,
    }
}

fn epoch(store: &BddStore) -> u64 {
    store.shared.epoch()
}

fn collect(store: &BddStore) {
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

/// A store with room for neither a node nor an edge beyond its terminals.
fn tiny() -> Arc<BddStore> {
    Arc::new(BddStore::with_capacity(2, 0, 1))
}

/// A roomy store.
fn roomy() -> Arc<BddStore> {
    Arc::new(BddStore::default())
}

/// The store bound to one sort.
fn col(store: &Arc<BddStore>, d: ScalarSort) -> Box<dyn Column> {
    Arc::clone(store).column(d).unwrap()
}

/// Every member of a predicate over a one-word sort.
fn members(c: &dyn Column, p: Function) -> Vec<u64> {
    let mut cursor = c.members(p);
    let (mut out, mut v) = (Vec::new(), Vec::new());
    while cursor.next(&mut v) {
        out.push(v[0]);
    }
    out
}

/// The predicate for the sorted values of a one-word sort.
fn set(c: &dyn Column, values: impl IntoIterator<Item = u64>) -> Function {
    c.set(&values.into_iter().collect::<Vec<_>>())
}

/// The predicate testing `mask` bits of a one-word sort against `value`.
fn bit(c: &dyn Column, mask: u64, value: u64) -> Function {
    c.masked_eq(&[mask], &[value])
}

/// A three-level diagram whose two level-one nodes share both of their
/// level-two children.
///
/// A traversal reaches the same operand pair by two routes and must answer the
/// second from its memo. The `low` pair names the two accepted low-bit values.
fn shared(c: &dyn Column, low: (u64, u64)) -> Function {
    let (hi, mid) = (bit(c, 1 << 12, 0), bit(c, 1 << 6, 0));
    let (a, b) = (bit(c, 0x3F, low.0), bit(c, 0x3F, low.1));
    let (yes, no) = (c.ite(mid, a, b), c.ite(mid, b, a));
    let root = c.ite(hi, yes, no);
    for f in [hi, mid, a, b, yes, no] {
        c.dec_ref(f);
    }
    root
}

/// The concrete meaning of [`shared`], as plain integer arithmetic.
fn shared_holds(low: (u64, u64), x: u64) -> bool {
    let (hi, mid) = ((x >> 12) & 1 == 0, (x >> 6) & 1 == 0);
    x & 0x3F == if hi == mid { low.0 } else { low.1 }
}

/// Points spread over a three-level sort, for comparing a predicate with its
/// meaning.
fn samples() -> impl Iterator<Item = u64> {
    (0..1 << 18).step_by(701).chain(0..200)
}

fn holds(c: &dyn Column, f: Function, x: u64) -> bool {
    c.evaluate(f, &[x]) != 0
}

/// The class number of each four-bit value.
fn classes(c: &dyn Column, f: Function) -> Vec<u32> {
    (0..16).map(|v| c.evaluate(f, &[v])).collect()
}

/// The partition labeling values below 16 by their residue modulo `n`, with
/// label zero elsewhere.
fn residues(c: &dyn Column, n: u64) -> Function {
    let cases: Vec<(Function, u32)> = (1..n)
        .map(|r| (set(c, (0..16).filter(|v| v % n == r)), r as u32))
        .collect();
    let root = c.cases(0, &cases, &mut Vec::new());
    for (f, _) in cases {
        c.dec_ref(f);
    }
    root
}

// --- level buffers ----

#[test]
fn level_buffer_preserves_its_initialized_prefix_when_full() {
    let mut buffer = LevelBuffer::new();
    assert!(buffer.as_slice().is_empty());
    assert!(buffer.as_mut_slice().is_empty());

    let mut expected = [0_u64; 64];
    for (i, value) in expected.iter_mut().enumerate() {
        *value = i as u64 + 1;
        buffer.push(*value);
        assert_eq!(buffer.as_slice().len(), i + 1);
        assert_eq!(buffer.as_slice()[i], *value);
        assert!(buffer.as_slice().iter().copied().eq(1..=i as u64 + 1));
    }
    for value in buffer.as_mut_slice() {
        *value *= 3;
    }
    expected.iter_mut().for_each(|value| *value *= 3);
    assert_eq!(buffer.as_slice(), expected);

    let result = catch_unwind(AssertUnwindSafe(|| {
        buffer.push(u64::MAX);
    }));
    assert!(result.is_err());
    assert_eq!(buffer.as_slice(), expected);
    buffer.as_mut_slice()[63] = 999;
    expected[63] = 999;
    assert_eq!(buffer.as_slice(), expected);
}

// --- sets, cuts, and bit tests ----

#[test]
fn sets_cuts_and_bit_tests_survive_maintenance() {
    let store = tiny();
    let c = col(&store, BV12);
    // Values spread over both levels, so the construction recurses before it first
    // interns.
    let values: Vec<u64> = (0..300).map(|v| v * 13).collect();
    let p = c.set(&values);
    assert_eq!(members(&*c, p), values);
    assert_eq!(c.count(p), 300);
    let lt = c.less_than(&[1000]);
    assert_eq!(c.count(lt), 1000);
    // A bit test fixes the assigned bits at every level it reaches.
    let eq = bit(&*c, 0xF0F, 0x502);
    assert_eq!(
        members(&*c, eq),
        (0..4096).filter(|v| v & 0xF0F == 0x502).collect::<Vec<_>>()
    );
    // The empty and full results are the constants, without any construction at
    // all.
    assert_eq!(c.set(&[]), Function::FALSE);
    assert_eq!(c.less_than(&[0]), Function::FALSE);
    assert!(
        epoch(&store) > 0,
        "a store this small cannot hold any of these results"
    );
}

// --- boolean operations ----

#[test]
fn boolean_operations_on_shared_subgraphs_survive_maintenance() {
    let store = tiny();
    let c = col(&store, BV18);
    let (u, v) = ((1, 2), (3, 4));
    let (f, g) = (shared(&*c, u), shared(&*c, v));
    let before = epoch(&store);
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        let r = c.apply_bool(op, f, g);
        for x in samples() {
            assert_eq!(
                holds(&*c, r, x),
                boolean(op, shared_holds(u, x), shared_holds(v, x)),
                "{op:?} at {x}"
            );
        }
        c.dec_ref(r);
    }
    // Complementing is `true and-not p` or XOR with true in either order. Both
    // routes negate directly and memoize both directions.
    for (op, a, b) in [
        (BoolOp::AndNot, Function::TRUE, f),
        (BoolOp::Xor, Function::TRUE, f),
        (BoolOp::Xor, f, Function::TRUE),
    ] {
        let n = c.apply_bool(op, a, b);
        for x in samples() {
            assert_eq!(holds(&*c, n, x), !shared_holds(u, x), "{op:?} at {x}");
        }
        let again = c.apply_bool(op, Function::TRUE, n);
        assert_eq!(again, f);
        c.dec_ref(again);
        c.dec_ref(n);
    }
    assert!(epoch(&store) > before);
}

// --- images and preimages ----

#[test]
fn preimage_of_true_owns_its_input_predicate() {
    let store = tiny();
    let c = col(&store, BV4);
    let p = set(&*c, [2, 5]);
    let map = MapSpec::Bits {
        clear: vec![3].into(),
        flip: vec![5].into(),
    };
    let pre = c.preimage(&map, p, Function::TRUE);
    c.dec_ref(p);
    collect(&store);
    assert_eq!(members(&*c, pre), [2, 5]);
    c.dec_ref(pre);
    collect(&store);
    assert!(store.shared.read(store.threads).get(pre).is_none());
}

#[test]
fn images_and_preimages_match_a_bit_map_across_two_levels() {
    let store = tiny();
    let c = col(&store, BV12);
    let (p, q) = (
        set(&*c, [3, 0x41, 0x842, 0xFFF]),
        set(&*c, [0, 0x15, 0x855, 0xFBE]),
    );
    // The map assigns the low six bits and flips bit six on the other level.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let f = |x: u64| (x & !0x3F) ^ 0x55;
    let (members_p, members_q) = (members(&*c, p), members(&*c, q));
    let before = epoch(&store);
    let image = c.image(&map, p, q);
    let expect: Vec<u64> = members_p
        .iter()
        .map(|&x| f(x))
        .filter(|y| members_q.contains(y))
        .collect();
    assert_eq!(members(&*c, image), expect);
    let preimage = c.preimage(&map, p, q);
    let expect: Vec<u64> = members_p
        .iter()
        .copied()
        .filter(|&x| members_q.contains(&f(x)))
        .collect();
    assert_eq!(members(&*c, preimage), expect);
    // The identity is a conjunction, whichever direction it is applied in.
    let both = c.apply_bool(BoolOp::And, p, q);
    let id = MapSpec::identity(1);
    assert_eq!(c.image(&id, p, q), both);
    assert_eq!(c.preimage(&id, p, q), both);
    assert!(epoch(&store) > before);
}

// --- conditionals ----

#[test]
fn conditionals_resolve_the_constant_cases() {
    let store = tiny();
    let c = col(&store, BV18);
    let (u, v, w) = ((1, 2), (3, 4), (5, 6));
    let (k, y, n) = (shared(&*c, u), shared(&*c, v), shared(&*c, w));
    let before = epoch(&store);
    let r = c.ite(k, y, n);
    for x in samples() {
        let expect = if shared_holds(u, x) {
            shared_holds(v, x)
        } else {
            shared_holds(w, x)
        };
        assert_eq!(holds(&*c, r, x), expect, "at {x}");
    }
    // These cases return an operand directly.
    assert_eq!(c.ite(Function::TRUE, y, n), y);
    assert_eq!(c.ite(Function::FALSE, y, n), n);
    assert_eq!(c.ite(k, y, y), y);
    assert_eq!(c.ite(k, Function::TRUE, Function::FALSE), k);
    // Reversing the Boolean branches complements the condition.
    let not_k = c.ite(k, Function::FALSE, Function::TRUE);
    for x in samples() {
        assert_eq!(holds(&*c, not_k, x), !shared_holds(u, x), "not at {x}");
    }
    // One constant branch is a disjunction or a conjunction, which still needs the
    // traversal.
    let or = c.ite(k, Function::TRUE, n);
    let and = c.ite(k, Function::FALSE, n);
    for x in samples() {
        assert_eq!(
            holds(&*c, or, x),
            shared_holds(u, x) || shared_holds(w, x),
            "or at {x}"
        );
        assert_eq!(
            holds(&*c, and, x),
            !shared_holds(u, x) && shared_holds(w, x),
            "and at {x}"
        );
    }
    assert!(epoch(&store) > before);
}

#[test]
fn overlapping_masks_and_conditional_levels_match_concrete_values() {
    let store = tiny();
    let c = col(&store, BV12);
    let a = set(&*c, (0..4096).filter(|x| x % 5 < 2));
    let b = set(&*c, (0..4096).filter(|x| x % 7 < 3));
    let k = set(&*c, (0..4096).filter(|x| x % 11 < 4));
    let deeper = bit(&*c, 1, 1);
    let earlier = bit(&*c, 1 << 6, 1 << 6);
    let results: Vec<_> = [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot]
        .into_iter()
        .map(|op| (op, c.apply_bool(op, a, b)))
        .collect();
    let overlapping = c.ite(a, b, k);
    let reordered = c.ite(deeper, earlier, k);
    for x in 0..4096 {
        for &(op, root) in &results {
            assert_eq!(
                holds(&*c, root, x),
                boolean(op, x % 5 < 2, x % 7 < 3),
                "{op:?} at {x}"
            );
        }
        assert_eq!(
            holds(&*c, overlapping, x),
            if x % 5 < 2 { x % 7 < 3 } else { x % 11 < 4 }
        );
        assert_eq!(
            holds(&*c, reordered, x),
            if x & 1 == 1 { x & 64 != 0 } else { x % 11 < 4 }
        );
    }
    let equivalent = c.ite(a, b, Function::FALSE);
    assert_eq!(equivalent, results[0].1);
    let state = store.shared.read(store.threads);
    let mut visited = HashSet::new();
    let mut pending = vec![overlapping, reordered];
    while let Some(f) = pending.pop() {
        if !visited.insert(f) {
            continue;
        }
        let node = state.node(f);
        if node.is_terminal() {
            continue;
        }
        assert!(node.edges.len() > 1);
        let mut covered = 0;
        let mut children = HashSet::new();
        let mut previous = None;
        for edge in node.edges {
            assert_ne!(edge.mask, 0);
            assert_eq!(edge.mask & covered, 0);
            covered |= edge.mask;
            assert!(children.insert(edge.child));
            let first = edge.mask.trailing_zeros();
            assert!(previous.is_none_or(|last| last < first));
            previous = Some(first);
            assert!(state.node(edge.child).level() > node.level());
            pending.push(edge.child);
        }
        assert_eq!(covered, u64::MAX);
    }
}

#[test]
fn integer_conditionals_preserve_terminal_payloads_and_canonical_roots() {
    let store = tiny();
    let c = col(&store, BV12);
    let a = bit(&*c, 1, 1);
    let b = bit(&*c, 64, 64);
    let seven = store.owned(|state| state.intern_terminal(7));
    let max = store.owned(|state| state.intern_terminal(u32::MAX));
    let ite = |condition, yes, no| {
        store.owned(|state| state.try_ite(&mut Memo::default(), condition, yes, no))
    };
    let low = ite(b, seven, max);
    let high = ite(b, max, seven);
    let root = ite(a, high, low);
    let xor = c.apply_bool(BoolOp::Xor, a, b);
    let equivalent = ite(xor, seven, max);
    assert_eq!(root, equivalent);
    for f in [a, b, seven, max, low, high, xor, equivalent] {
        c.dec_ref(f);
    }
    collect(&store);
    for x in 0..4096 {
        assert_eq!(
            c.evaluate(root, &[x]),
            if (x & 1 != 0) != (x & 64 != 0) {
                7
            } else {
                u32::MAX
            }
        );
    }
    c.dec_ref(root);
}

// --- alignment ----

#[test]
fn alignment_numbers_reachable_pairs_in_discovery_order() {
    let store = tiny();
    let c = col(&store, BV4);
    let (f, g) = (residues(&*c, 3), residues(&*c, 2));
    let mut labels = vec![(9, 9)];
    let mut scratch = Alignment::default();
    let before = epoch(&store);
    let h = c.align(f, g, &mut labels, &mut scratch);
    // The existing prefix is preserved and result classes index only the appended
    // suffix.
    assert_eq!(labels[0], (9, 9));
    let pairs = labels[1..].to_vec();
    assert_eq!(pairs, [(0, 0), (1, 1), (2, 0), (0, 1), (1, 0), (2, 1)]);
    for v in 0..16 {
        let class = c.evaluate(h, &[v]) as usize;
        assert_eq!(pairs[class], ((v % 3) as u32, (v % 2) as u32), "at {v}");
    }
    assert!(
        epoch(&store) > before,
        "alignment in this store must advance the maintenance epoch"
    );
    // A repeated call answers from the whole-call cache, appending the same labels
    // afresh without recording them again.
    labels.truncate(1);
    let cached = scratch.cache.labels.len();
    let again = c.align(f, g, &mut labels, &mut scratch);
    assert_eq!(again, h);
    assert_eq!(&labels[1..], pairs);
    assert_eq!(scratch.cache.labels.len(), cached);
    // The hit carries its own reference, so its root survives releasing the first
    // result and collecting.
    for root in [h, f, g] {
        c.dec_ref(root);
    }
    collect(&store);
    assert!(store.shared.read(store.threads).get(again).is_some());
    for v in 0..16 {
        let class = c.evaluate(again, &[v]) as usize;
        assert_eq!(pairs[class], ((v % 3) as u32, (v % 2) as u32), "at {v}");
    }
    c.dec_ref(again);
    collect(&store);
    assert!(store.shared.read(store.threads).get(again).is_none());
}

#[test]
fn partition_caches_refresh_after_collection_and_between_stores() {
    let mut scratch = Alignment::default();
    for store in [tiny(), roomy()] {
        let c = col(&store, BV4);
        let f = residues(&*c, 3);
        let g = residues(&*c, 2);
        for _ in 0..2 {
            let mut labels = vec![(99, 99)];
            let h = c.align(f, g, &mut labels, &mut scratch);
            assert_eq!(labels[0], (99, 99));
            for x in 0..16 {
                let class = c.evaluate(h, &[x]) as usize;
                assert_eq!(labels[1 + class], ((x % 3) as u32, (x % 2) as u32));
            }
            c.dec_ref(h);
            collect(&store);
        }
        c.dec_ref(f);
        c.dec_ref(g);
    }
}

// --- quotients ----

#[test]
fn quotients_merge_classes_and_check_the_cached_map() {
    let store = tiny();
    let c = col(&store, BV4);
    let f = residues(&*c, 6);
    let mut scratch = Alignment::default();
    // The even residues form class zero and the odd residues form class one.
    let class_map = [0, 1, 0, 1, 0, 1];
    let q = c.quotient(f, &class_map, &mut scratch);
    assert_eq!(classes(&*c, q), (0..16).map(|v| v % 2).collect::<Vec<_>>());
    let cached = scratch.cache.classes.len();
    let hit = c.quotient(f, &class_map, &mut scratch);
    assert_eq!(hit, q, "the cached entry answers again");
    assert_eq!(
        scratch.cache.classes.len(),
        cached,
        "a cache hit records nothing"
    );
    // Changing the cached class map simulates a hash collision. The next
    // quotient must recompute the result and replace the cache entry.
    scratch.cache.classes[1] = 7;
    let recomputed = c.quotient(f, &class_map, &mut scratch);
    assert_eq!(recomputed, q);
    assert_eq!(
        scratch.cache.classes[scratch.cache.classes.len() - class_map.len()..],
        class_map
    );
    // The hit carries its own reference, so its root survives releasing the other
    // results and collecting.
    c.dec_ref(f);
    c.dec_ref(q);
    c.dec_ref(recomputed);
    collect(&store);
    assert!(store.shared.read(store.threads).get(hit).is_some());
    assert_eq!(
        classes(&*c, hit),
        (0..16).map(|v| v % 2).collect::<Vec<_>>()
    );
    c.dec_ref(hit);
    collect(&store);
    assert!(store.shared.read(store.threads).get(hit).is_none());
}

// --- cases ----

#[test]
fn cases_with_the_same_label_form_one_class() {
    let store = tiny();
    let c = col(&store, BV12);
    let (a, b) = (
        set(&*c, (0..300).map(|v| v * 13)),
        set(&*c, (0..300).map(|v| v * 7)),
    );
    let mut labels = Vec::new();
    let before = epoch(&store);
    let f = c.cases(9, &[(a, 4), (b, 4)], &mut labels);
    // Overlapping cases agree on their label, so the second one finds the class
    // already covered. Zero is a member, so the covered class is the one numbered
    // first.
    assert_eq!(labels, [4, 9]);
    for x in 0..4096 {
        let covered = x % 13 == 0 && x < 300 * 13 || x % 7 == 0 && x < 300 * 7;
        assert_eq!(c.evaluate(f, &[x]), u32::from(!covered), "at {x}");
    }
    assert!(epoch(&store) > before);
}

// --- values ----

#[test]
fn values_number_classes_in_order_of_first_appearance() {
    let store = tiny();
    let c = col(&store, BV12);
    let mut labels = Vec::new();
    let mut scratch = Values;
    // Labels above one are interned as terminals, so even the default needs a node
    // here. Zero takes the default label, so its class gets number zero.
    let keys: Vec<u64> = (1..300).map(|v| v * 13).collect();
    let case_labels: Vec<u32> = (0..299).map(|i| 20 + i % 3).collect();
    let f = c.values(9, &keys, &case_labels, &mut labels, &mut scratch);
    assert_eq!(labels, [9, 20, 21, 22]);
    for x in 0..4096 {
        let expect = keys
            .iter()
            .position(|&k| k == x)
            .map_or(0, |i| 1 + (i % 3) as u32);
        assert_eq!(c.evaluate(f, &[x]), expect, "at {x}");
    }
    assert!(epoch(&store) > 0);
}

// --- refinement ----

#[test]
fn refinement_delivers_only_the_selected_cells() {
    let store = tiny();
    let c = col(&store, BV12);
    let f = residues(&*c, 3);
    let p = set(&*c, [1, 2, 3, 4, 5]);
    let mut scratch = Refinement::default();
    let before = epoch(&store);
    let mut seen = Vec::new();
    let _ = c.refine(
        f,
        p,
        &mut scratch,
        &mut |class| class != 2,
        &mut |class, build| {
            let cell = build();
            seen.push((class, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    seen.sort();
    assert_eq!(seen, [(0, vec![3]), (1, vec![1, 4])]);
    // A false predicate reaches no class at all, and a break stops the visit
    // immediately.
    let _ = c.refine(
        f,
        Function::FALSE,
        &mut scratch,
        &mut |_| panic!("no class is reached"),
        &mut |_, _| panic!("no cell is delivered"),
    );
    let mut count = 0;
    let mut selected = 0;
    let stopped = c.refine(
        f,
        p,
        &mut scratch,
        &mut |_| {
            selected += 1;
            true
        },
        &mut |_, _| {
            count += 1;
            ControlFlow::Break(())
        },
    );
    assert!(stopped.is_break() && count == 1 && selected == 1);
    seen.clear();
    let resumed = c.refine(f, p, &mut scratch, &mut |_| true, &mut |class, _| {
        // Visiting without requesting a cell still reports every reached class.
        seen.push((class, Vec::new()));
        ControlFlow::Continue(())
    });
    seen.sort();
    assert!(resumed.is_continue());
    assert_eq!(seen, [(0, vec![]), (1, vec![]), (2, vec![])]);
    assert!(epoch(&store) > before);
    c.dec_ref(f);
    c.dec_ref(p);
}

#[test]
fn repeated_cell_builds_own_independent_references() {
    let store = tiny();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let p = c.set(&[1, 2, 3, 4, 5]);
    let mut cells = Vec::new();
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |class, build| {
            let a = build();
            let b = build();
            c.dec_ref(a);
            collect(&store);
            cells.push((class, members(&*c, b)));
            c.dec_ref(b);
            ControlFlow::Continue(())
        },
    );
    cells.sort();
    assert!(result.is_continue());
    assert_eq!(cells, [(0, vec![3]), (1, vec![1, 4]), (2, vec![2, 5])]);
    c.dec_ref(f);
    c.dec_ref(p);
}

#[test]
fn a_terminal_partition_selects_the_whole_predicate_or_nothing() {
    let store = tiny();
    let c = col(&store, BV4);
    let p = c.set(&[1, 3, 7]);
    let mut cells = Vec::new();
    let result = c.refine(
        Function::FALSE,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |class, build| {
            let cell = build();
            cells.push((class, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    assert_eq!(cells, [(0, vec![1, 3, 7])]);
    let absent =
        store.owned(|state| state.try_cell(&mut Default::default(), Function::FALSE, p, 1));
    assert_eq!(absent, Function::FALSE);
    c.dec_ref(absent);
    c.dec_ref(p);
}

#[test]
fn refinement_delivery_can_collect_and_reenter_the_store() {
    let store = tiny();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let p = set(&*c, [1, 2, 3, 4, 5]);
    let mut seen = Vec::new();
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |class, build| {
            collect(&store);
            let cell = build();
            let mut nested = Vec::new();
            let nested_result = c.refine(
                f,
                cell,
                &mut Refinement::default(),
                &mut |_| true,
                &mut |inner, make| {
                    let selected = make();
                    nested.push(inner);
                    assert_eq!(members(&*c, selected), members(&*c, cell));
                    c.dec_ref(selected);
                    ControlFlow::Continue(())
                },
            );
            assert!(nested_result.is_continue());
            assert_eq!(nested, [class]);
            seen.push((class, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![3]), (1, vec![1, 4]), (2, vec![2, 5])]);
}

// --- member cursors ----

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

// --- collection and growth ----

#[test]
fn repeated_constructions_preserve_results_across_maintenance() {
    // Each iteration releases its results so collection can reclaim prior work.
    // Repeated constructions exercise level tests, bit maps, complements, and
    // conjunctions as the small store collects and grows.
    let store = Arc::new(BddStore::with_capacity(4, 16, 1));
    let c = col(&store, BV12);
    // The map assigns the low six bits and flips bit six on the other level.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let eval = |x: u64| (x & !0x3F) ^ 0x55;
    for k in 1..24 {
        // The image of 2049 is 2133, so every round has a surviving pair.
        let p = set(&*c, (0..8).map(|i| i * k + 1).chain([2049]));
        let q = set(&*c, (0..8).map(|i| i * k * 7).chain([2133]));
        let (in_p, in_q) = (members(&*c, p), members(&*c, q));
        let mut expect: Vec<u64> = in_p
            .iter()
            .map(|&x| eval(x))
            .filter(|y| in_q.contains(y))
            .collect();
        expect.sort_unstable();
        expect.dedup();
        let image = c.image(&map, p, q);
        assert_eq!(members(&*c, image), expect, "image at {k}");
        let expect: Vec<u64> = in_p
            .iter()
            .copied()
            .filter(|&x| in_q.contains(&eval(x)))
            .collect();
        let preimage = c.preimage(&map, p, q);
        assert_eq!(members(&*c, preimage), expect, "preimage at {k}");
        // Zero is a member of `q`, so a single case over it is built by complementing
        // it.
        let mut labels = Vec::new();
        let f = c.cases(8, &[(q, 3)], &mut labels);
        assert_eq!(labels, [3, 8]);
        assert_eq!(
            (0..4096)
                .filter(|&x| c.evaluate(f, &[x]) == 0)
                .collect::<Vec<_>>(),
            in_q
        );
        let n = c.apply_bool(BoolOp::AndNot, Function::TRUE, p);
        let r = c.ite(p, q, n);
        let lt = c.less_than(&[k * 100]);
        for x in (0..4096).step_by(13) {
            let expect = !in_p.contains(&x) || in_q.contains(&x);
            assert_eq!(holds(&*c, r, x), expect, "at {x}");
            assert_eq!(holds(&*c, lt, x), x < k * 100, "at {x}");
        }
        for g in [p, q, image, preimage, f, n, r, lt] {
            c.dec_ref(g);
        }
    }
    assert!(epoch(&store) > 4);
}

#[test]
fn tight_and_roomy_stores_agree_on_three_level_construction() {
    let (tight, roomy) = (tiny(), roomy());
    let (t, r) = (col(&tight, BV18), col(&roomy, BV18));
    let build = |c: &dyn Column| {
        let a = shared(c, (1, 2));
        let b = shared(c, (2, 3));
        let k = shared(c, (3, 1));
        let xor = c.apply_bool(BoolOp::Xor, a, b);
        let result = c.ite(k, xor, a);
        for f in [a, b, k, xor] {
            c.dec_ref(f);
        }
        result
    };
    let a = build(&*t);
    let b = build(&*r);
    assert!(epoch(&tight) > 0);
    assert_eq!(epoch(&roomy), 0);
    collect(&tight);
    for x in samples() {
        let expected = if shared_holds((3, 1), x) {
            shared_holds((1, 2), x) != shared_holds((2, 3), x)
        } else {
            shared_holds((1, 2), x)
        };
        assert_eq!(holds(&*t, a, x), expected);
        assert_eq!(holds(&*r, b, x), expected);
    }
    assert_eq!(t.count(a), r.count(b));
    t.dec_ref(a);
    r.dec_ref(b);
}

#[test]
fn concurrent_operations_preserve_canonical_roots_across_collection() {
    let store = Arc::new(BddStore::with_capacity(2, 0, 2));
    let c = col(&store, BV12);
    let barrier = Barrier::new(4);
    let roots = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let c = &*c;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let a = bit(c, 1, 1);
                    let b = bit(c, 64, 64);
                    let roots: Vec<_> = (0..8).map(|_| c.apply_bool(BoolOp::Xor, a, b)).collect();
                    c.dec_ref(a);
                    c.dec_ref(b);
                    roots
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap_or_else(|e| panic::resume_unwind(e)))
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
