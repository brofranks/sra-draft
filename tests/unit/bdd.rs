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

use super::{BddStore, Edge, LevelBuffer, Memo, NeedGrowth, Node};
use crate::backend::{Algebra, Align, Backend, BoolOp, Function, MapSpec, Refine, Values};
use crate::scalar::ScalarSort;

#[path = "../support/bdd.rs"]
mod fixtures;

use fixtures::{
    BV4, BV12, BV18, bind, bit, classes, members, residues, roomy, samples, set, shared,
    shared_holds, tiny,
};

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

fn holds(alg: &dyn Algebra, f: Function, x: u64) -> bool {
    alg.evaluate(f, &[x]) != 0
}

// --- level buffers ----

#[test]
fn level_buffer_preserves_its_initialized_prefix_when_full() {
    let mut buffer = LevelBuffer::new();
    assert!(buffer.as_slice().is_empty());
    assert!(buffer.as_mut_slice().is_empty());

    for value in 1..=64 {
        buffer.push(value);
        assert!(
            buffer.as_slice().iter().copied().eq(1..=value),
            "prefix after pushing {value}"
        );
    }
    for value in buffer.as_mut_slice() {
        *value *= 3;
    }
    let mut expected: [u64; 64] = std::array::from_fn(|i| 3 * (i as u64 + 1));
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

// --- node interning ----

#[test]
fn colliding_nodes_are_distinguished_by_level_and_edges() {
    let store = roomy();
    let state = store.read();
    let level = 1_u32;
    let [edges, other] = [1_u64, 3].map(|mask| {
        [
            Edge {
                mask,
                child: Function::FALSE,
            },
            Edge {
                mask: !mask,
                child: Function::TRUE,
            },
        ]
    });
    // Give both candidates the target's hash so interning must compare them.
    // The read guard keeps the table from being rebuilt during the test.
    let hash = crate::util::dag::structural_hash((level, edges.as_slice()));
    let insert = |data, candidate: &[Edge]| {
        state
            .intern(hash, |_, _| false, || Node { data }, candidate)
            .unwrap()
    };
    let other_level = insert(0, &edges);
    let other_edges = insert(level, &other);
    let root = state.intern_node(level, &edges).unwrap();
    assert_ne!(root, other_level, "merged different levels");
    assert_ne!(root, other_edges, "merged different edges");
    assert_eq!(state.intern_node(level, &edges), Ok(root));
}

#[test]
fn edge_exhaustion_reports_the_span_of_the_branch() {
    let store = BddStore::with_capacity(3, 0, 1);
    let state = store.read();
    let span = super::Span { sort: BV4 };
    let need = state.try_masked_eq(span, &[1], &[1]).unwrap_err();
    assert_eq!(need, NeedGrowth::Edges(2));
}

// --- sets, cuts, and bit tests ----

#[test]
fn sets_cuts_and_bit_tests_survive_maintenance() {
    let store = tiny();
    let alg = bind(&store, BV12);
    // Values spread over both levels, so the construction recurses before it first
    // interns.
    let len = if cfg!(miri) { 16 } else { 300 };
    let values: Vec<u64> = (0..len).map(|v| v * 13).collect();
    let p = alg.set(&values);
    assert_eq!(members(&*alg, p), values);
    assert_eq!(alg.count(p), len);
    let lt = alg.lt(&[1000]);
    assert_eq!(alg.count(lt), 1000);
    // A bit test fixes the assigned bits at every level it reaches.
    let eq = bit(&*alg, 0xF0F, 0x502);
    assert_eq!(
        members(&*alg, eq),
        (0..4096).filter(|v| v & 0xF0F == 0x502).collect::<Vec<_>>()
    );
    // An empty set and a cut at zero both return the false constant.
    assert_eq!(alg.set(&[]), Function::FALSE);
    assert_eq!(alg.lt(&[0]), Function::FALSE);
    assert!(
        epoch(&store) > 0,
        "a store this small cannot hold any of these results"
    );
    for f in [p, lt, eq] {
        alg.dec_ref(f);
    }
}

// --- boolean operations ----

#[test]
fn boolean_operations_on_shared_subgraphs_survive_maintenance() {
    let store = tiny();
    let alg = bind(&store, BV18);
    let (u, v) = ((1, 2), (3, 4));
    let (f, g) = (shared(&*alg, u), shared(&*alg, v));
    let before = epoch(&store);
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        let r = alg.apply(op, f, g);
        for x in samples() {
            assert_eq!(
                holds(&*alg, r, x),
                boolean(op, shared_holds(u, x), shared_holds(v, x)),
                "{op:?} at {x}"
            );
        }
        alg.dec_ref(r);
    }
    // Complementing is `true and-not p` or XOR with true in either order. Both
    // routes negate directly and memoize both directions.
    for (op, a, b) in [
        (BoolOp::AndNot, Function::TRUE, f),
        (BoolOp::Xor, Function::TRUE, f),
        (BoolOp::Xor, f, Function::TRUE),
    ] {
        let n = alg.apply(op, a, b);
        for x in samples() {
            assert_eq!(holds(&*alg, n, x), !shared_holds(u, x), "{op:?} at {x}");
        }
        let again = alg.apply(op, Function::TRUE, n);
        assert_eq!(again, f);
        alg.dec_ref(again);
        alg.dec_ref(n);
    }
    assert!(epoch(&store) > before);
    alg.dec_ref(f);
    alg.dec_ref(g);
}

// --- images and preimages ----

#[test]
fn xor_of_the_full_domain_owns_the_target_without_maintenance() {
    let store = Arc::new(BddStore::with_capacity(3, 2, 1));
    let alg = bind(&store, BV4);
    let q = alg.set(&[3]);
    let map = MapSpec::Bits {
        clear: [0].into(),
        flip: [1].into(),
    };
    let before = epoch(&store);
    let image = alg.image(&map, Function::TRUE, q);
    assert_eq!(image, q);
    assert_eq!(epoch(&store), before);
    alg.dec_ref(q);
    collect(&store);
    assert_eq!(members(&*alg, image), [3]);
    alg.dec_ref(image);
    collect(&store);
    assert!(store.read().get(image).is_none());
}

#[test]
fn an_empty_filtered_image_needs_no_assigned_bit_nodes() {
    let store = Arc::new(BddStore::with_capacity(3, 2, 1));
    let alg = bind(&store, BV18);
    let odd = bit(&*alg, 1, 1);
    let map = MapSpec::Bits {
        clear: vec![(1 << 18) - 1].into(),
        flip: vec![0].into(),
    };
    let before = epoch(&store);
    assert_eq!(alg.image(&map, Function::TRUE, odd), Function::FALSE);
    assert_eq!(epoch(&store), before);
    alg.dec_ref(odd);
}

#[test]
fn a_preimage_skips_unreachable_output_branches() {
    let store = Arc::new(BddStore::with_capacity(4, 4, 1));
    let alg = bind(&store, BV12);
    // Flipping bit zero would rebuild the low node for 3. Only the high
    // branch that requires the cleared bit leads to that node.
    let q = bit(&*alg, 64 | 63, 64 | 3);
    let map = MapSpec::Bits {
        clear: vec![64].into(),
        flip: vec![1].into(),
    };
    let before = epoch(&store);
    assert_eq!(alg.preimage(&map, Function::TRUE, q), Function::FALSE);
    assert_eq!(epoch(&store), before);
    alg.dec_ref(q);
}

#[test]
fn a_selector_image_filters_before_merging_witnesses() {
    let store = Arc::new(BddStore::with_capacity(48, 256, 1));
    let alg = bind(&store, BV18);
    // Selector i witnesses a shared set bit of the two six-bit fields.
    // The unrestricted projection has 64 distinct residuals on the low field.
    let selector = 63 << 12;
    let mut p = Function::FALSE;
    for i in 0..6 {
        let bits = (1 << (6 + i)) | (1 << i);
        let term = bit(&*alg, selector | bits, (i << 12) | bits);
        let next = alg.apply(BoolOp::Or, p, term);
        alg.dec_ref(p);
        alg.dec_ref(term);
        p = next;
    }
    let map = MapSpec::Bits {
        clear: vec![selector].into(),
        flip: vec![0].into(),
    };
    for low in [0, 1] {
        let q = bit(&*alg, 63, low);
        let before = epoch(&store);
        let image = alg.image(&map, p, q);
        let expected: Vec<u64> = (0..64)
            .filter(|high| high & low != 0)
            .map(|high| (high << 6) | low)
            .collect();
        assert_eq!(members(&*alg, image), expected);
        if low == 0 {
            assert_eq!(epoch(&store), before);
        }
        alg.dec_ref(image);
        alg.dec_ref(q);
    }
    alg.dec_ref(p);
}

#[test]
fn accepted_overwrites_retain_the_singleton_image_and_the_source_preimage() {
    let store = tiny();
    let alg = bind(&store, BV12);
    let p = alg.set(&[7, 25]);
    let q = alg.set(&[131]);
    let map = MapSpec::Bits {
        clear: [4095].into(),
        flip: [131].into(),
    };
    let image = alg.image(&map, p, q);
    let preimage = alg.preimage(&map, p, q);
    alg.dec_ref(q);
    alg.dec_ref(p);
    collect(&store);
    assert_eq!(members(&*alg, image), [131]);
    assert_eq!(members(&*alg, preimage), [7, 25]);
    alg.dec_ref(image);
    alg.dec_ref(preimage);
}

#[test]
fn preimage_of_true_owns_its_input_predicate() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let p = set(&*alg, [2, 5]);
    let map = MapSpec::Bits {
        clear: vec![3].into(),
        flip: vec![5].into(),
    };
    let pre = alg.preimage(&map, p, Function::TRUE);
    alg.dec_ref(p);
    collect(&store);
    assert_eq!(members(&*alg, pre), [2, 5]);
    alg.dec_ref(pre);
    collect(&store);
    assert!(store.read().get(pre).is_none());
}

#[test]
fn images_and_preimages_match_a_bit_map_across_two_levels() {
    let store = tiny();
    let alg = bind(&store, BV12);
    let (in_p, in_q) = ([3, 0x41, 0x842, 0xFFF], [3, 0x15, 0x855, 0xFBE]);
    let (p, q) = (set(&*alg, in_p), set(&*alg, in_q));
    // The map assigns the low six bits and flips bit six on the other level.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let f = |x: u64| (x & !0x3F) ^ 0x55;
    let before = epoch(&store);
    let image = alg.image(&map, p, q);
    let mut expect: Vec<u64> = in_p
        .map(f)
        .into_iter()
        .filter(|y| in_q.contains(y))
        .collect();
    expect.sort_unstable();
    assert_eq!(members(&*alg, image), expect);
    let preimage = alg.preimage(&map, p, q);
    let expect: Vec<u64> = in_p.into_iter().filter(|&x| in_q.contains(&f(x))).collect();
    assert_eq!(members(&*alg, preimage), expect);
    // The identity is a conjunction, whichever direction it is applied in.
    let both = alg.apply(BoolOp::And, p, q);
    assert_eq!(members(&*alg, both), [3]);
    let id = MapSpec::identity(1);
    let (forward, backward) = (alg.image(&id, p, q), alg.preimage(&id, p, q));
    assert_eq!(forward, both);
    assert_eq!(backward, both);
    assert!(epoch(&store) > before);
    for f in [p, q, image, preimage, both, forward, backward] {
        alg.dec_ref(f);
    }
}

// --- conditionals ----

#[test]
fn conditionals_resolve_the_constant_cases() {
    let store = tiny();
    let alg = bind(&store, BV18);
    let (u, v, w) = ((1, 2), (3, 4), (5, 6));
    let (k, y, n) = (shared(&*alg, u), shared(&*alg, v), shared(&*alg, w));
    let before = epoch(&store);
    assert_eq!(alg.ite(k, k, Function::TRUE), Function::TRUE);
    assert_eq!(alg.ite(k, Function::FALSE, k), Function::FALSE);
    let r = alg.ite(k, y, n);
    for x in samples() {
        let expect = if shared_holds(u, x) {
            shared_holds(v, x)
        } else {
            shared_holds(w, x)
        };
        assert_eq!(holds(&*alg, r, x), expect, "at {x}");
    }
    // Reversing the Boolean branches complements the condition.
    let not_k = alg.ite(k, Function::FALSE, Function::TRUE);
    for x in samples() {
        assert_eq!(holds(&*alg, not_k, x), !shared_holds(u, x), "not at {x}");
    }
    // One constant branch is a disjunction or a conjunction, which still needs the
    // traversal.
    let or = alg.ite(k, Function::TRUE, n);
    let and = alg.ite(k, Function::FALSE, n);
    for x in samples() {
        assert_eq!(
            holds(&*alg, or, x),
            shared_holds(u, x) || shared_holds(w, x),
            "or at {x}"
        );
        assert_eq!(
            holds(&*alg, and, x),
            !shared_holds(u, x) && shared_holds(w, x),
            "and at {x}"
        );
    }
    assert!(epoch(&store) > before);
    for root in [k, y, n, r, not_k, or, and] {
        alg.dec_ref(root);
    }
}

#[test]
fn conditionals_returning_an_operand_acquire_a_reference() {
    let store = tiny();
    let alg = bind(&store, BV4);
    for case in 0..6 {
        let (p, o) = (set(&*alg, [1, 3]), set(&*alg, [2]));
        let r = match case {
            0 => alg.ite(Function::TRUE, p, o),
            1 => alg.ite(Function::FALSE, o, p),
            2 => alg.ite(o, p, p),
            3 => alg.ite(p, Function::TRUE, Function::FALSE),
            4 => alg.ite(p, p, Function::FALSE),
            _ => alg.ite(p, Function::TRUE, p),
        };
        assert_eq!(r, p, "case {case}");
        // The result outlives every input owner.
        alg.dec_ref(p);
        alg.dec_ref(o);
        collect(&store);
        assert!(store.read().get(r).is_some(), "case {case}");
        for x in 0..16 {
            assert_eq!(alg.evaluate(r, &[x]), u32::from(x == 1 || x == 3), "at {x}");
        }
        alg.dec_ref(r);
        collect(&store);
        assert!(store.read().get(r).is_none(), "case {case}");
    }
}

#[test]
fn overlapping_masks_and_conditional_levels_match_concrete_values() {
    let store = tiny();
    let width = if cfg!(miri) { 8 } else { 12 };
    let alg = bind(&store, ScalarSort::bv(width));
    let a = set(&*alg, (0..1 << width).filter(|x| x % 5 < 2));
    let b = set(&*alg, (0..1 << width).filter(|x| x % 7 < 3));
    let k = set(&*alg, (0..1 << width).filter(|x| x % 11 < 4));
    let deeper = bit(&*alg, 1, 1);
    let earlier = bit(&*alg, 1 << 6, 1 << 6);
    let results: Vec<_> = [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot]
        .into_iter()
        .map(|op| (op, alg.apply(op, a, b)))
        .collect();
    let overlapping = alg.ite(a, b, k);
    let reordered = alg.ite(deeper, earlier, k);
    for x in 0..1 << width {
        for &(op, root) in &results {
            assert_eq!(
                holds(&*alg, root, x),
                boolean(op, x % 5 < 2, x % 7 < 3),
                "{op:?} at {x}"
            );
        }
        assert_eq!(
            holds(&*alg, overlapping, x),
            if x % 5 < 2 { x % 7 < 3 } else { x % 11 < 4 }
        );
        assert_eq!(
            holds(&*alg, reordered, x),
            if x & 1 == 1 { x & 64 != 0 } else { x % 11 < 4 }
        );
    }
    let equivalent = alg.ite(a, b, Function::FALSE);
    assert_eq!(equivalent, results[0].1);
    let state = store.read();
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
    drop(state);
    for f in [a, b, k, deeper, earlier, overlapping, reordered, equivalent]
        .into_iter()
        .chain(results.into_iter().map(|(_, root)| root))
    {
        alg.dec_ref(f);
    }
}

#[test]
fn integer_conditionals_preserve_terminal_payloads_and_canonical_roots() {
    let store = tiny();
    let width = if cfg!(miri) { 7 } else { 12 };
    let alg = bind(&store, ScalarSort::bv(width));
    let a = bit(&*alg, 1, 1);
    let b = bit(&*alg, 64, 64);
    let seven = store.owned(|state| state.intern_terminal(7));
    let max = store.owned(|state| state.intern_terminal(u32::MAX));
    let ite = |condition, yes, no| {
        store.owned(|state| state.try_ite(&mut Memo::default(), condition, yes, no))
    };
    let low = ite(b, seven, max);
    let high = ite(b, max, seven);
    let root = ite(a, high, low);
    let xor = alg.apply(BoolOp::Xor, a, b);
    let equivalent = ite(xor, seven, max);
    assert_eq!(root, equivalent);
    for f in [a, b, seven, max, low, high, xor, equivalent] {
        alg.dec_ref(f);
    }
    collect(&store);
    for x in 0..1 << width {
        assert_eq!(
            alg.evaluate(root, &[x]),
            if (x & 1 != 0) != (x & 64 != 0) {
                7
            } else {
                u32::MAX
            }
        );
    }
    alg.dec_ref(root);
}

// --- alignment ----

#[test]
fn alignment_numbers_reachable_pairs_in_discovery_order() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let (f, g) = (residues(&*alg, 3), residues(&*alg, 2));
    let mut labels = vec![(9, 9)];
    let mut scratch = Align::default();
    let before = epoch(&store);
    let h = alg.align(f, g, &mut labels, &mut scratch);
    // The existing prefix is preserved and result classes index only the appended
    // suffix.
    assert_eq!(labels[0], (9, 9));
    let pairs = labels[1..].to_vec();
    assert_eq!(pairs, [(0, 0), (1, 1), (2, 0), (0, 1), (1, 0), (2, 1)]);
    for v in 0..16 {
        let class = alg.evaluate(h, &[v]) as usize;
        assert_eq!(pairs[class], ((v % 3) as u32, (v % 2) as u32), "at {v}");
    }
    assert!(
        epoch(&store) > before,
        "alignment in this store must advance the maintenance epoch"
    );
    // A repeated call answers from the whole-call cache, appending the same labels
    // afresh without recording them again or rebuilding.
    labels.truncate(1);
    scratch.memo.clear();
    let again = alg.align(f, g, &mut labels, &mut scratch);
    assert_eq!(again, h);
    assert_eq!(&labels[1..], pairs);
    assert!(scratch.memo.is_empty(), "the hit skips the build");
    // The hit carries its own reference, so its root survives releasing the first
    // result and collecting.
    for root in [h, f, g] {
        alg.dec_ref(root);
    }
    collect(&store);
    assert!(store.read().get(again).is_some());
    for v in 0..16 {
        let class = alg.evaluate(again, &[v]) as usize;
        assert_eq!(pairs[class], ((v % 3) as u32, (v % 2) as u32), "at {v}");
    }
    alg.dec_ref(again);
    collect(&store);
    assert!(store.read().get(again).is_none());
}

#[test]
fn partition_caches_refresh_after_collection_and_between_stores() {
    let stores = [roomy(), roomy()];
    let algebras = stores.each_ref().map(|store| bind(store, BV4));
    // Each predicate rejects zero, so its terminals already number the two
    // partition classes in order. The raw IDs coincide in both fresh stores.
    let pairs = [(&algebras[0], &[2][..]), (&algebras[1], &[1, 2][..])]
        .map(|(alg, keys)| (alg.set(&[1]), alg.set(keys)));
    assert_eq!(pairs[0], pairs[1]);
    assert!(stores.iter().all(|store| epoch(store) == 0));
    let mut scratch = Align::default();
    for i in [0, 1, 1] {
        let alg = &algebras[i];
        let (f, g) = pairs[i];
        let mut labels = vec![(99, 99)];
        let h = alg.align(f, g, &mut labels, &mut scratch);
        assert_eq!(labels, [(99, 99), (0, 0), (1, i as u32), (0, 1)]);
        for x in 0..16 {
            let class = alg.evaluate(h, &[x]) as usize;
            assert_eq!(
                labels[1 + class],
                (u32::from(x == 1), u32::from(x == 2 || i == 1 && x == 1))
            );
        }
        alg.dec_ref(h);
        collect(&stores[i]);
    }
    for (alg, (f, g)) in algebras.iter().zip(pairs) {
        alg.dec_ref(f);
        alg.dec_ref(g);
    }
}

// --- quotients ----

#[test]
fn quotients_merge_classes_and_check_the_cached_map() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 6);
    let mut scratch = Align::default();
    // The even residues form class zero and the odd residues form class one.
    let class_map = [0, 1, 0, 1, 0, 1];
    let parity: Vec<u32> = (0..16).map(|v| v % 2).collect();
    let q = alg.quotient(f, &class_map, &mut scratch);
    assert_eq!(classes(&*alg, q), parity);
    scratch.coarse.clear();
    let hit = alg.quotient(f, &class_map, &mut scratch);
    assert_eq!(hit, q, "the cached entry answers again");
    assert!(scratch.coarse.is_empty(), "the hit skips the build");
    // Cache a different class map under the same hash, so the next quotient must
    // compare the full map rather than trust the hash.
    {
        let _state = store.read();
        let key = (f, crate::util::dag::structural_hash(&class_map[..]));
        let other = crate::cache::quotient(&mut scratch.cache, key, &[0, 0, 1, 1, 0, 0], || {
            Ok(Function::FALSE)
        });
        assert_eq!(other, Ok(Function::FALSE));
    }
    let recomputed = alg.quotient(f, &class_map, &mut scratch);
    assert_eq!(recomputed, q);
    // The hit carries its own reference, so its root survives releasing the other
    // results and collecting.
    alg.dec_ref(q);
    alg.dec_ref(recomputed);
    collect(&store);
    assert!(store.read().get(hit).is_some());
    assert_eq!(classes(&*alg, hit), parity);
    alg.dec_ref(hit);
    collect(&store);
    assert!(store.read().get(hit).is_none());
    // The cached entry for the live input now names a reclaimed root, so the
    // advanced epoch must rebuild it.
    let rebuilt = alg.quotient(f, &class_map, &mut scratch);
    assert_eq!(classes(&*alg, rebuilt), parity);
    alg.dec_ref(rebuilt);
    alg.dec_ref(f);
}

// --- cases ----

#[test]
fn empty_partitions_append_only_the_default_without_maintenance() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let mut labels = vec![7];
    assert_eq!(alg.cases(42, &[], &mut labels), Function::FALSE);
    assert_eq!(
        alg.values(42, &[], &[], &mut labels, &mut Values),
        Function::FALSE
    );
    assert_eq!(labels, [7, 42, 42]);
    assert_eq!(epoch(&store), 0);
}

#[test]
fn cases_with_the_same_label_form_one_class() {
    let store = tiny();
    let (width, len) = if cfg!(miri) { (8, 18) } else { (12, 300) };
    let alg = bind(&store, ScalarSort::bv(width));
    let (a, b) = (
        set(&*alg, (0..len).map(|v| v * 13)),
        set(&*alg, (0..len).map(|v| v * 7)),
    );
    let mut labels = Vec::new();
    let before = epoch(&store);
    let f = alg.cases(9, &[(a, 4), (b, 4)], &mut labels);
    // Both predicates include zero and carry label 4. Their union therefore
    // forms the first class.
    assert_eq!(labels, [4, 9]);
    for x in 0..1 << width {
        let covered = x % 13 == 0 && x < len * 13 || x % 7 == 0 && x < len * 7;
        assert_eq!(alg.evaluate(f, &[x]), u32::from(!covered), "at {x}");
    }
    assert!(epoch(&store) > before);
    for g in [a, b, f] {
        alg.dec_ref(g);
    }
}

// --- values ----

#[test]
fn values_restore_the_label_prefix_when_construction_unwinds() {
    // The branch needs growth, and a zero thread budget makes maintenance
    // panic after the labels have been numbered.
    let store = Arc::new(BddStore {
        threads: 0,
        ..BddStore::with_capacity(2, 0, 1)
    });
    let alg = bind(&store, ScalarSort::bool());
    let mut labels = vec![77];
    let panic = catch_unwind(AssertUnwindSafe(|| {
        alg.values(9, &[0, 1], &[10, 20], &mut labels, &mut Values)
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"maintenance thread count must be nonzero")
    );
    assert_eq!(labels, [77]);
}

#[test]
fn values_number_classes_in_order_of_first_appearance() {
    let store = tiny();
    let (width, len) = if cfg!(miri) { (8, 18) } else { (12, 300) };
    let alg = bind(&store, ScalarSort::bv(width));
    let mut labels = Vec::new();
    let mut scratch = Values;
    // The raw labels lie above the class numbers. Zero takes the default
    // label, so its class gets the pinned zero terminal.
    let keys: Vec<u64> = (1..len).map(|v| v * 13).collect();
    let case_labels: Vec<u32> = (0..(len - 1) as u32).map(|i| 20 + i % 3).collect();
    let f = alg.values(9, &keys, &case_labels, &mut labels, &mut scratch);
    assert_eq!(labels, [9, 20, 21, 22]);
    for x in 0..1 << width {
        let i = x / 13;
        let expect = if x % 13 == 0 && (1..len).contains(&i) {
            1 + ((i - 1) % 3) as u32
        } else {
            0
        };
        assert_eq!(alg.evaluate(f, &[x]), expect, "at {x}");
    }
    assert!(epoch(&store) > 0);
    alg.dec_ref(f);
}

#[test]
fn full_key_coverage_interns_only_the_canonical_branch() {
    let store = Arc::new(BddStore::with_capacity(3, 2, 1));
    let alg = bind(&store, ScalarSort::bool());
    let mut labels = vec![8];
    let f = alg.values(7, &[0, 1], &[42, u32::MAX], &mut labels, &mut Values);
    assert_eq!(labels, [8, 42, u32::MAX]);
    assert_eq!([alg.evaluate(f, &[0]), alg.evaluate(f, &[1])], [0, 1]);
    assert_eq!(epoch(&store), 0);
    alg.dec_ref(f);
}

// --- refinement ----

#[test]
fn building_a_refinement_cell_retries_after_maintenance() {
    // Five initial edges make the build for class zero run out of space.
    let store = Arc::new(BddStore::with_capacity(2, 5, 1));
    let alg = bind(&store, BV12);
    let f = residues(&*alg, 3);
    let p = alg.set(&[1, 2, 3, 4, 5]);
    let result = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |class| class == 0,
        &mut |_, build| {
            let before = epoch(&store);
            let cell = build();
            assert!(epoch(&store) > before);
            assert_eq!(members(&*alg, cell), [3]);
            alg.dec_ref(cell);
            ControlFlow::Break(())
        },
    );
    assert!(result.is_break());
    alg.dec_ref(f);
    alg.dec_ref(p);
}

#[test]
fn repeated_cell_builds_own_independent_references() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    let p = alg.set(&[1, 2, 3, 4, 5]);
    let mut cells = Vec::new();
    let result = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |_| true,
        &mut |class, build| {
            let a = build();
            let b = build();
            alg.dec_ref(a);
            collect(&store);
            cells.push((class, members(&*alg, b)));
            alg.dec_ref(b);
            ControlFlow::Continue(())
        },
    );
    cells.sort();
    assert!(result.is_continue());
    assert_eq!(cells, [(0, vec![3]), (1, vec![1, 4]), (2, vec![2, 5])]);
    alg.dec_ref(f);
    alg.dec_ref(p);
}

#[test]
fn refinement_delivery_can_collect_and_reenter_the_store() {
    let store = tiny();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    let p = set(&*alg, [1, 2, 3, 4, 5]);
    let mut seen = Vec::new();
    let result = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |_| true,
        &mut |class, build| {
            collect(&store);
            let cell = build();
            let mut nested = Vec::new();
            let nested_result = alg.refine(
                f,
                cell,
                &mut Refine::default(),
                &mut |_| true,
                &mut |inner, make| {
                    let selected = make();
                    nested.push(inner);
                    assert_eq!(members(&*alg, selected), members(&*alg, cell));
                    alg.dec_ref(selected);
                    ControlFlow::Continue(())
                },
            );
            assert!(nested_result.is_continue());
            assert_eq!(nested, [class]);
            seen.push((class, members(&*alg, cell)));
            alg.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![3]), (1, vec![1, 4]), (2, vec![2, 5])]);
    alg.dec_ref(f);
    alg.dec_ref(p);
}

// --- member cursors ----

#[test]
fn a_cursor_resumes_interleaved_live_assignments_after_compaction() {
    let store = roomy();
    let alg = bind(&store, BV12);
    let junk = alg.lt(&[1234]);
    let points: Vec<_> = (0..64).filter(|i| i % 5 != 1).map(|i| 65 * i).collect();
    let p = alg.set(&points);
    let mut cursor = alg.members(p);
    alg.dec_ref(junk);
    alg.dec_ref(p);
    assert_eq!(cursor.next(), Some(&points[..1]));
    collect(&store);
    for value in &points[1..] {
        assert_eq!(cursor.next(), Some(std::slice::from_ref(value)));
    }
    assert_eq!(cursor.next(), None);
}

#[test]
fn retained_roots_and_member_cursors_survive_collection() {
    let store = roomy();
    let alg = bind(&store, ScalarSort::bv(70));
    let p = alg.set(&[1, 0, 65, 0, 129, 0, 5, 1, 7, 2]);
    alg.inc_ref(p);
    let mut cursor = alg.members(p);
    // A cursor dropped before exhaustion releases its root as well.
    let mut early = alg.members(p);
    assert_eq!(early.next(), Some(&[1, 0][..]));
    drop(early);
    alg.dec_ref(p);
    alg.dec_ref(p);
    let point = cursor.next().unwrap();
    collect(&store);
    assert_eq!(point, [1, 0]);
    let mut rest = Vec::new();
    while let Some(point) = cursor.next() {
        rest.extend_from_slice(point);
    }
    assert_eq!(rest, [65, 0, 129, 0, 5, 1, 7, 2]);
    drop(cursor);
    collect(&store);
    assert!(store.read().get(p).is_none());
    assert_eq!(alg.evaluate(Function::FALSE, &[0, 0]), 0);
    assert_eq!(alg.evaluate(Function::TRUE, &[4095, 0]), 1);
}

// --- collection and growth ----

#[test]
fn repeated_constructions_preserve_results_across_maintenance() {
    // Each iteration releases its results so collection can reclaim prior work.
    // Repeated constructions exercise level tests, bit maps, complements, and
    // conjunctions as the small store collects and grows.
    let store = Arc::new(BddStore::with_capacity(4, 16, 1));
    let alg = bind(&store, BV12);
    // The map assigns the low six bits and flips bit six on the other level.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let eval = |x: u64| (x & !0x3F) ^ 0x55;
    for k in 1..if cfg!(miri) { 3 } else { 24 } {
        // The image of 2049 is 2133, so every round has a surviving pair.
        let in_p: Vec<u64> = (0..8).map(|i| i * k + 1).chain([2049]).collect();
        let in_q: Vec<u64> = (0..8).map(|i| i * k * 7).chain([2133]).collect();
        let p = alg.set(&in_p);
        let q = alg.set(&in_q);
        let mut expect: Vec<u64> = in_p
            .iter()
            .map(|&x| eval(x))
            .filter(|y| in_q.contains(y))
            .collect();
        expect.sort_unstable();
        expect.dedup();
        let image = alg.image(&map, p, q);
        assert_eq!(members(&*alg, image), expect, "image at {k}");
        let expect: Vec<u64> = in_p
            .iter()
            .copied()
            .filter(|&x| in_q.contains(&eval(x)))
            .collect();
        let preimage = alg.preimage(&map, p, q);
        assert_eq!(members(&*alg, preimage), expect, "preimage at {k}");
        // Zero is a member of `q`, so a single case over it is built by complementing
        // it.
        let mut labels = Vec::new();
        let f = alg.cases(8, &[(q, 3)], &mut labels);
        assert_eq!(labels, [3, 8]);
        let step = if cfg!(miri) { 257 } else { 1 };
        for x in (0..4096).step_by(step).chain(in_q.iter().copied()) {
            assert_eq!(alg.evaluate(f, &[x]) == 0, in_q.contains(&x));
        }
        let n = alg.apply(BoolOp::AndNot, Function::TRUE, p);
        let r = alg.ite(p, q, n);
        let lt = alg.lt(&[k * 100]);
        for x in (0..4096).step_by(if cfg!(miri) { 257 } else { 13 }) {
            let expect = !in_p.contains(&x) || in_q.contains(&x);
            assert_eq!(holds(&*alg, r, x), expect, "at {x}");
            assert_eq!(holds(&*alg, lt, x), x < k * 100, "at {x}");
        }
        for g in [p, q, image, preimage, f, n, r, lt] {
            alg.dec_ref(g);
        }
    }
    assert!(epoch(&store) > 4);
}

#[test]
fn tight_and_roomy_stores_agree_on_three_level_construction() {
    let (tight, roomy) = (tiny(), roomy());
    let (t, r) = (bind(&tight, BV18), bind(&roomy, BV18));
    let build = |alg: &dyn Algebra| {
        let a = shared(alg, (1, 2));
        let b = shared(alg, (2, 3));
        let k = shared(alg, (3, 1));
        let xor = alg.apply(BoolOp::Xor, a, b);
        let result = alg.ite(k, xor, a);
        for f in [a, b, k, xor] {
            alg.dec_ref(f);
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
    let alg = bind(&store, BV12);
    let barrier = Barrier::new(4);
    let roots = thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let alg = &*alg;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let a = bit(alg, 1, 1);
                    let b = bit(alg, 64, 64);
                    let roots: Vec<_> = (0..8).map(|_| alg.apply(BoolOp::Xor, a, b)).collect();
                    alg.dec_ref(a);
                    alg.dec_ref(b);
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
            assert_eq!(holds(&*alg, root, x), (x & 1 != 0) != (x & 64 != 0));
        }
        alg.dec_ref(root);
    }
    collect(&store);
    assert!(store.read().get(roots[0]).is_none());
}
