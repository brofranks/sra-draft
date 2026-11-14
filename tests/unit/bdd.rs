//! Concrete checks for the BDD backend.
//!
//! The store is driven directly through [`Column`], since these are the raw
//! diagram contracts. Every symbolic result is checked against plain integer
//! arithmetic over the sort's values rather than against another diagram.
//!
//! Tiny stores start with room only for the pinned terminals, forcing
//! collection and growth during construction. Small capacity sweeps exercise
//! different allocation boundaries, while roomy stores provide comparison
//! results.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Barrier};
use std::thread;

use num_bigint::BigUint;

use super::{BddStore, LevelBuffer, Memo, NeedGrowth};
use crate::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, Values, transfer,
};
use crate::scalar::ScalarSort;

/// One six-bit level, so a diagram is a single node over the terminals.
const BV4: ScalarSort = ScalarSort::bit_vec(4);
/// Two levels.
const BV12: ScalarSort = ScalarSort::bit_vec(12);
/// Three levels, the shallowest sort whose traversals can reach one node pair
/// by two routes.
const BV18: ScalarSort = ScalarSort::bit_vec(18);

/// Literal encoded points for comparison with finite transfer.
struct Listed(std::vec::IntoIter<Vec<u64>>);

impl Members for Listed {
    fn next(&mut self, out: &mut Vec<u64>) -> bool {
        self.0.next().is_some_and(|value| {
            *out = value;
            true
        })
    }
}

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
/// level-two children, so a traversal reaches the same operand pair by two
/// routes and must answer the second from its memo. `low` names the two
/// accepted low-bit values.
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

/// The slot of every value of a four-bit map.
fn slots(c: &dyn Column, f: Function) -> Vec<u32> {
    (0..16).map(|v| c.evaluate(f, &[v])).collect()
}

/// The canonical partition of the four-bit universe by `x % n`, and its slot
/// labels.
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

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        buffer.push(u64::MAX);
    }));
    assert!(result.is_err());
    assert_eq!(buffer.as_slice(), expected);
    buffer.as_mut_slice()[63] = 999;
    expected[63] = 999;
    assert_eq!(buffer.as_slice(), expected);
}

// --- binding ----

#[test]
fn every_scalar_sort_binds_a_column() {
    let store: Arc<dyn Backend> = roomy();
    for sort in [
        ScalarSort::bool(),
        ScalarSort::bit_vec(1),
        BV4,
        ScalarSort::bit_vec(70),
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

// --- counting ----

#[test]
fn a_cut_over_a_short_first_level_counts_the_whole_universe() {
    // Twelve bits divide evenly, but four do not: level zero then reaches only part
    // of its 64 assignments and the unreachable ones must not be counted.
    let store = roomy();
    let columns = [
        col(&store, BV4),
        col(&store, BV12),
        col(&store, ScalarSort::bit_vec(70)),
    ];
    let cuts = [
        columns[0].less_than(&[7]),
        columns[1].less_than(&[7]),
        columns[2].less_than(&[7, 0]),
    ];
    let sizes = [
        BigUint::from(16_u32),
        BigUint::from(4096_u32),
        BigUint::from(1_u32) << 70,
    ];
    // Alternate retained bindings, including a multiword sort, to exercise each
    // column's own interpretation within the shared store.
    for i in [0, 2, 1, 0] {
        let c = &*columns[i];
        assert_eq!(c.count(cuts[i]), BigUint::from(7_u32));
        assert_eq!(members(c, cuts[i]), [0, 1, 2, 3, 4, 5, 6]);
        let other = c.apply_bool(BoolOp::AndNot, Function::TRUE, cuts[i]);
        let all = c.apply_bool(BoolOp::Or, cuts[i], other);
        assert_eq!(c.count(all), sizes[i]);
        c.dec_ref(other);
        c.dec_ref(all);
    }
    for (c, p) in columns.iter().zip(cuts) {
        c.dec_ref(p);
    }
}

#[test]
fn multiword_cuts_and_skipped_levels_have_exact_counts() {
    let c = col(&tiny(), ScalarSort::bit_vec(70));
    let cut = c.less_than(&[5, 1]);
    assert_eq!(c.count(cut), (BigUint::from(1_u32) << 64) + 5_u32);
    for (point, expected) in [
        ([u64::MAX, 0], true),
        ([4, 1], true),
        ([5, 1], false),
        ([0, 2], false),
    ] {
        assert_eq!(c.evaluate(cut, &point) != 0, expected);
    }
    let low = c.masked_eq(&[1, 0], &[1, 0]);
    assert_eq!(c.count(low), BigUint::from(1_u32) << 69);
    assert_eq!(c.count(Function::TRUE), BigUint::from(1_u32) << 70);
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let _ = c.count(f);
}

// --- sets, cuts, and bit tests ----

#[test]
fn sets_cuts_and_bit_tests_survive_growth() {
    let store = tiny();
    let c = col(&store, BV12);
    // Values spread over both levels, so the construction recurses before it first
    // interns.
    let values: Vec<u64> = (0..300).map(|v| v * 13).collect();
    let p = c.set(&values);
    assert_eq!(members(&*c, p), values);
    assert_eq!(c.count(p), BigUint::from(300_u32));
    let lt = c.less_than(&[1000]);
    assert_eq!(c.count(lt), BigUint::from(1000_u32));
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

#[test]
#[should_panic(expected = "symbols have no numeric ordering")]
fn a_cut_over_symbols_is_rejected() {
    let _ = col(&roomy(), ScalarSort::symbol()).less_than(&[5]);
}

#[test]
#[should_panic(expected = "symbols do not support bit predicates")]
fn a_bit_test_over_symbols_is_rejected() {
    let _ = col(&roomy(), ScalarSort::symbol()).masked_eq(&[1], &[1]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn a_set_of_unsorted_values_is_rejected() {
    let _ = col(&roomy(), BV4).set(&[3, 1, 2]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn a_set_of_repeated_values_is_rejected() {
    let _ = col(&roomy(), BV4).set(&[1, 1]);
}

#[test]
#[should_panic(expected = "value is outside the column sort")]
fn a_set_value_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).set(&[16]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn unsorted_partition_keys_are_rejected() {
    let c = col(&roomy(), BV4);
    let _ = c.values(0, &[2, 1], &[1, 2], &mut Vec::new(), &mut Values::default());
}

// --- boolean operations ----

#[test]
fn boolean_operations_reuse_their_memo_and_survive_growth() {
    let store = tiny();
    let c = col(&store, BV18);
    let (u, v) = ((1, 2), (3, 4));
    let (f, g) = (shared(&*c, u), shared(&*c, v));
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
    assert!(epoch(&store) > 0);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn an_unreferenced_left_boolean_operand_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (p, q) = (set(&*c, [1]), set(&*c, [2]));
    c.dec_ref(p);
    let _ = c.apply_bool(BoolOp::And, p, q);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn an_unreferenced_right_boolean_operand_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (p, q) = (set(&*c, [1]), set(&*c, [2]));
    c.dec_ref(q);
    let _ = c.apply_bool(BoolOp::And, p, q);
}

// --- images and preimages ----

#[test]
fn bit_maps_move_levels_for_images_and_preimages() {
    let store = tiny();
    let c = col(&store, BV12);
    let (p, q) = (
        set(&*c, [3, 0x41, 0x842, 0xFFF]),
        set(&*c, [0, 0x15, 0x855, 0xFBE]),
    );
    // Assign the low six bits and toggle bit six: both levels move.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let f = |x: u64| (x & !0x3F) ^ 0x55;
    let (members_p, members_q) = (members(&*c, p), members(&*c, q));
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
    assert!(epoch(&store) > 0);
}

#[test]
fn bit_maps_confined_to_one_level_agree_with_pointwise_maps() {
    let store = tiny();
    let c = col(&store, BV18);
    let points = [0, 5, 0x41, 0x7C0, 0x1003, 0x2FFFF, 0x3F040, 0x3FFFF];
    let p = set(&*c, points);
    // Each map touches a single level, or none, so moves stop above the
    // untouched levels below it.
    for (clear, flip) in [
        (0, 0),
        (0x3F, 0x15),
        (0xFC0, 0x240),
        (0x3F000, 0x15000),
        (0, 0x20000),
    ] {
        let map = MapSpec::Bits {
            clear: vec![clear].into(),
            flip: vec![flip].into(),
        };
        let f = |x: u64| (x & !clear) ^ flip;
        let mut expect: Vec<u64> = points.iter().map(|&x| f(x)).collect();
        expect.sort();
        expect.dedup();
        let image = c.image(&map, p, Function::TRUE);
        assert_eq!(members(&*c, image), expect);
        let q = set(&*c, expect.iter().copied().step_by(2));
        let in_q = members(&*c, q);
        let expect: Vec<u64> = points
            .into_iter()
            .filter(|&x| in_q.contains(&f(x)))
            .collect();
        let preimage = c.preimage(&map, p, q);
        assert_eq!(members(&*c, preimage), expect);
        for g in [image, q, preimage] {
            c.dec_ref(g);
        }
    }
    c.dec_ref(p);
}

#[test]
fn wide_bit_maps_agree_with_literal_finite_images_and_preimages() {
    let sort = ScalarSort::bit_vec(70);
    let c = col(&tiny(), sort);
    let points = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1], vec![8, 2]];
    let p = c.set(&[0, 0, 1, 0, 0, 1, 1, 1, 8, 2]);
    let q = c.set(&[4, 2]);
    let map = MapSpec::Bits {
        clear: vec![1, 1].into(),
        flip: vec![4, 2].into(),
    };
    let image = c.image(&map, p, Function::TRUE);
    let preimage = c.preimage(&map, p, q);
    for (forward, root, expected) in [
        (true, image, vec![12, 0, 4, 2]),
        (false, preimage, vec![0, 0, 1, 0, 0, 1, 1, 1]),
    ] {
        let mut cursor = c.members(root);
        let (mut words, mut point) = (Vec::new(), Vec::new());
        while cursor.next(&mut point) {
            words.extend_from_slice(&point);
        }
        assert_eq!(words, expected);
        assert_eq!(
            transfer(
                sort,
                &map,
                &mut Listed(points.clone().into_iter()),
                |value| forward || value == [4, 2],
                forward,
            ),
            expected
        );
    }
    for f in [p, q, image, preimage] {
        c.dec_ref(f);
    }
}

/// A twelve-bit column over a small initial capacity, varied to exercise
/// different allocation boundaries.
fn small(slack: usize) -> Box<dyn Column> {
    col(
        &Arc::new(BddStore::with_capacity(2 + slack, slack * 64, 1)),
        BV12,
    )
}

#[test]
#[expect(
    clippy::type_complexity,
    reason = "the paired spec and its reference implementation read better inline"
)]
fn moves_preserve_concrete_results_across_small_initial_capacities() {
    // A bit map abstracts its cleared bits and permutes its toggled ones. Clearing
    // the last level fuses terminals; clearing the first fuses whole subgraphs,
    // which is the disjunction the move must build. Either can be the point at
    // which the arena runs out.
    let maps: [(MapSpec, fn(u64) -> u64); 2] = [
        (
            MapSpec::Bits {
                clear: vec![0x3F].into(),
                flip: vec![0x55].into(),
            },
            |x| (x & !0x3F) ^ 0x55,
        ),
        (
            MapSpec::Bits {
                clear: vec![0xFC0].into(),
                flip: vec![0x009].into(),
            },
            |x| (x & !0xFC0) ^ 0x009,
        ),
    ];
    for (map, moved) in &maps {
        for slack in 0..8 {
            let build = |c: &dyn Column| {
                (
                    set(c, (0..40).map(|i| i * 101 + 1)),
                    set(c, (0..40).map(|i| i * 37)),
                )
            };
            let c = small(slack);
            let (p, q) = build(&*c);
            let (in_p, in_q) = (members(&*c, p), members(&*c, q));
            let mut expect: Vec<u64> = in_p
                .iter()
                .map(|&x| moved(x))
                .filter(|y| in_q.contains(y))
                .collect();
            expect.sort_unstable();
            expect.dedup();
            assert_eq!(members(&*c, c.image(map, p, q)), expect, "image at {slack}");
            // A fresh store gives preimage the same initial capacity as image.
            let c = small(slack);
            let (p, q) = build(&*c);
            let expect: Vec<u64> = in_p
                .iter()
                .copied()
                .filter(|&x| in_q.contains(&moved(x)))
                .collect();
            assert_eq!(
                members(&*c, c.preimage(map, p, q)),
                expect,
                "preimage at {slack}"
            );
        }
    }
}

#[test]
fn malformed_bit_maps_are_rejected_by_symbolic_and_finite_transfer() {
    let masks: &[(ScalarSort, &[u64], &[u64])] = &[
        (BV4, &[], &[]),
        (BV4, &[], &[0]),
        (BV4, &[0], &[]),
        (BV4, &[0, 0], &[0]),
        (BV4, &[0], &[0, 0]),
        (BV4, &[16], &[0]),
        (BV4, &[0], &[16]),
        (ScalarSort::bit_vec(70), &[0], &[0]),
        (ScalarSort::bit_vec(70), &[0, 64], &[0, 0]),
        (ScalarSort::bit_vec(70), &[0, 0], &[0, 64]),
        (ScalarSort::symbol(), &[], &[]),
    ];
    for &(sort, clear, flip) in masks {
        let c = col(&tiny(), sort);
        let map = MapSpec::Bits {
            clear: clear.into(),
            flip: flip.into(),
        };
        let zero = vec![0; sort.words()];
        let p = c.set(&zero);
        for predicate in [Function::FALSE, p] {
            for image in [false, true] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        let result = if image {
                            c.image(&map, predicate, Function::TRUE)
                        } else {
                            c.preimage(&map, predicate, Function::TRUE)
                        };
                        c.dec_ref(result);
                    }))
                    .is_err()
                );
                let points = if predicate.is_f() {
                    vec![]
                } else {
                    vec![zero.clone()]
                };
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        transfer(sort, &map, &mut Listed(points.into_iter()), |_| true, image)
                    }))
                    .is_err()
                );
            }
        }
        c.dec_ref(p);
    }
}

#[test]
#[should_panic(expected = "a BDD store cannot apply a Rust map")]
fn a_rust_map_on_a_bdd_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let map = MapSpec::Rust(std::sync::Arc::new(|_: &[u64], _: &mut [u64]| true));
    let _ = c.image(&map, Function::TRUE, Function::TRUE);
}

// --- conditionals ----

#[test]
fn conditionals_resolve_the_constant_cases() {
    let store = tiny();
    let c = col(&store, BV18);
    let (u, v, w) = ((1, 2), (3, 4), (5, 6));
    let (k, y, n) = (shared(&*c, u), shared(&*c, v), shared(&*c, w));
    let r = c.ite(k, y, n);
    for x in samples() {
        let expect = if shared_holds(u, x) {
            shared_holds(v, x)
        } else {
            shared_holds(w, x)
        };
        assert_eq!(holds(&*c, r, x), expect, "at {x}");
    }
    // A constant condition, equal branches, and the two Boolean branch pairs need
    // no traversal.
    assert_eq!(c.ite(Function::TRUE, y, n), y);
    assert_eq!(c.ite(Function::FALSE, y, n), n);
    assert_eq!(c.ite(k, y, y), y);
    assert_eq!(c.ite(k, Function::TRUE, Function::FALSE), k);
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
    assert!(epoch(&store) > 0);
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
    let h = c.align(f, g, &mut labels, &mut scratch);
    // The existing prefix is preserved and result slots index only the appended
    // suffix.
    assert_eq!(labels[0], (9, 9));
    let pairs = labels[1..].to_vec();
    assert_eq!(pairs, [(0, 0), (1, 1), (2, 0), (0, 1), (1, 0), (2, 1)]);
    for v in 0..16 {
        let slot = c.evaluate(h, &[v]) as usize;
        assert_eq!(pairs[slot], ((v % 3) as u32, (v % 2) as u32), "at {v}");
    }
    assert!(
        epoch(&store) > 0,
        "the cache must have been restarted at a new growth count"
    );
    // A repeated call answers from the whole-call cache, appending the same labels
    // afresh without recording them again.
    labels.truncate(1);
    let cached = scratch.cache.labels.len();
    let again = c.align(f, g, &mut labels, &mut scratch);
    assert_eq!(again, h);
    assert_eq!(&labels[1..], pairs);
    assert_eq!(scratch.cache.labels.len(), cached);
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
                let slot = c.evaluate(h, &[x]) as usize;
                assert_eq!(labels[1 + slot], ((x % 3) as u32, (x % 2) as u32));
            }
            c.dec_ref(h);
            collect(&store);
        }
        c.dec_ref(f);
        c.dec_ref(g);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a partition with itself")]
fn alignment_of_identical_partitions_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let _ = c.align(f, f, &mut Vec::new(), &mut Alignment::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a trivial partition")]
fn alignment_of_a_trivial_left_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let _ = c.align(
        Function::FALSE,
        f,
        &mut Vec::new(),
        &mut Alignment::default(),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a trivial partition")]
fn alignment_of_a_trivial_right_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    let _ = c.align(
        f,
        Function::FALSE,
        &mut Vec::new(),
        &mut Alignment::default(),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn alignment_of_an_unreferenced_left_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (f, g) = (residues(&*c, 3), residues(&*c, 2));
    c.dec_ref(f);
    let _ = c.align(f, g, &mut Vec::new(), &mut Alignment::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn alignment_of_an_unreferenced_right_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    // The fixture releases its case predicates, leaving only this root reference.
    let (f, g) = (residues(&*c, 3), residues(&*c, 4));
    c.dec_ref(g);
    let _ = c.align(f, g, &mut Vec::new(), &mut Alignment::default());
}

// --- quotients ----

#[test]
fn quotients_coarsen_and_recheck_their_cached_slots() {
    let store = tiny();
    let c = col(&store, BV4);
    let f = residues(&*c, 6);
    let mut scratch = Alignment::default();
    // Merge the residues into evens and odds; slot numbering stays
    // first-appearance.
    let slots_of = [0, 1, 0, 1, 0, 1];
    let q = c.quotient(f, &slots_of, &mut scratch);
    assert_eq!(slots(&*c, q), (0..16).map(|v| v % 2).collect::<Vec<_>>());
    let cached = scratch.cache.slots.len();
    assert_eq!(
        c.quotient(f, &slots_of, &mut scratch),
        q,
        "the cached entry answers again"
    );
    assert_eq!(
        scratch.cache.slots.len(),
        cached,
        "a cache hit records nothing"
    );
    // An entry reached by a hash collision carries different slots, so it is
    // recomputed and replaced rather than answered. Rewriting the cached slots is
    // that collision.
    scratch.cache.slots[1] = 7;
    assert_eq!(c.quotient(f, &slots_of, &mut scratch), q);
    assert_eq!(
        scratch.cache.slots[scratch.cache.slots.len() - slots_of.len()..],
        slots_of
    );
    assert!(epoch(&store) > 0);
}

// --- cases ----

#[test]
fn cases_fold_onto_labels_and_canonicalize_their_slots() {
    let store = tiny();
    let c = col(&store, BV12);
    let (a, b) = (
        set(&*c, (0..300).map(|v| v * 13)),
        set(&*c, (0..300).map(|v| v * 7)),
    );
    let mut labels = Vec::new();
    let f = c.cases(9, &[(a, 4), (b, 4)], &mut labels);
    // Overlapping cases agree on their label, so the second one finds the cell
    // already covered. Zero is a member, so the covered cell is the one numbered
    // first.
    assert_eq!(labels, [4, 9]);
    for x in 0..4096 {
        let covered = x % 13 == 0 && x < 300 * 13 || x % 7 == 0 && x < 300 * 7;
        assert_eq!(c.evaluate(f, &[x]), u32::from(!covered), "at {x}");
    }
    assert!(epoch(&store) > 0);
}

#[test]
fn cases_over_one_predicate_reuse_it_or_its_complement() {
    let store = roomy();
    let c = col(&store, BV4);
    let p = set(&*c, [1, 3, 5]);
    let mut labels = Vec::new();
    // A case labeled like the default, or over a constant, is the one-slot
    // partition.
    assert_eq!(c.cases(7, &[(p, 7)], &mut labels), Function::FALSE);
    assert_eq!(
        c.cases(7, &[(Function::TRUE, 8)], &mut labels),
        Function::FALSE
    );
    assert_eq!(
        c.cases(7, &[(Function::FALSE, 8)], &mut labels),
        Function::FALSE
    );
    assert_eq!(labels, [7, 8, 7]);
    labels.clear();
    // Zero is the first value, so a predicate rejecting it already numbers its
    // cells canonically.
    assert_eq!(c.cases(0, &[(p, 1)], &mut labels), p);
    assert_eq!(labels, [0, 1]);
    labels.clear();
    // One accepting zero does not, and is complemented instead.
    let q = set(&*c, [0, 2]);
    let f = c.cases(5, &[(q, 6)], &mut labels);
    assert_eq!(labels, [6, 5]);
    assert_eq!(
        slots(&*c, f),
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    );
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn overlapping_cases_with_different_labels_are_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

// --- values ----

#[test]
fn values_label_points_and_canonicalize_their_slots() {
    let store = tiny();
    let c = col(&store, BV12);
    let mut labels = Vec::new();
    let mut scratch = Values::default();
    // Labels above one are interned as terminals, so even the default needs a node
    // here. Zero is not a key, so the default is the slot numbered first.
    let keys: Vec<u64> = (1..300).map(|v| v * 13).collect();
    let case_labels: Vec<u32> = (0..299).map(|i| 20 + (i % 3) as u32).collect();
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
    let mut seen = Vec::new();
    let _ = c.refine(
        f,
        p,
        &mut scratch,
        &mut |slot| slot != 2,
        &mut |slot, build| {
            let cell = build();
            seen.push((slot, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    seen.sort();
    assert_eq!(seen, [(0, vec![3]), (1, vec![1, 4])]);
    // A false predicate reaches no slot at all, and a break stops the visit
    // immediately.
    let _ = c.refine(
        f,
        Function::FALSE,
        &mut scratch,
        &mut |_| panic!("no slot is reached"),
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
    let resumed = c.refine(f, p, &mut scratch, &mut |_| true, &mut |slot, _| {
        // Visiting without requesting a cell still reports every reached slot.
        seen.push((slot, Vec::new()));
        ControlFlow::Continue(())
    });
    seen.sort();
    assert!(resumed.is_continue());
    assert_eq!(seen, [(0, vec![]), (1, vec![]), (2, vec![])]);
    assert!(epoch(&store) > 0);
    c.dec_ref(f);
    c.dec_ref(p);
}

#[test]
fn a_shallow_partition_selects_each_slot_once_under_a_deep_predicate() {
    let store = tiny();
    let c = col(&store, BV18);
    let top = bit(&*c, 1 << 17, 1 << 17);
    let f = c.cases(0, &[(top, 1)], &mut Vec::new());
    let mut points: Vec<u64> = samples().filter(|x| x % 3 == 0).collect();
    points.sort();
    points.dedup();
    let p = set(&*c, points.iter().copied());
    let (mut selected, mut cells) = (Vec::new(), Vec::new());
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |slot| {
            selected.push(slot);
            true
        },
        &mut |slot, build| {
            let cell = build();
            cells.push((slot, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    selected.sort();
    cells.sort();
    assert!(result.is_continue());
    assert_eq!(selected, [0, 1]);
    let (high, low): (Vec<u64>, Vec<u64>) = points.into_iter().partition(|x| x >> 17 == 1);
    assert_eq!(cells, [(0, low), (1, high)]);
    for g in [top, f, p] {
        c.dec_ref(g);
    }
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
        &mut |slot, build| {
            let a = build();
            let b = build();
            c.dec_ref(a);
            collect(&store);
            cells.push((slot, members(&*c, b)));
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
        &mut |slot, build| {
            let cell = build();
            cells.push((slot, members(&*c, cell)));
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
        &mut |slot, build| {
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
            assert_eq!(nested, [slot]);
            seen.push((slot, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert!(result.is_continue());
    seen.sort();
    assert_eq!(seen, [(0, vec![3]), (1, vec![1, 4]), (2, vec![2, 5])]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn refinement_of_an_unreferenced_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (f, p) = (residues(&*c, 3), set(&*c, [1, 2]));
    c.dec_ref(f);
    let _ = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |_, _| ControlFlow::Continue(()),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn refinement_by_an_unreferenced_predicate_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (f, p) = (residues(&*c, 3), set(&*c, [1, 2]));
    c.dec_ref(p);
    let _ = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |_, _| ControlFlow::Continue(()),
    );
}

// --- member cursors ----

#[test]
fn member_cursors_skip_levels_and_cross_word_boundaries() {
    let store = roomy();
    let c = col(&store, BV12);
    // A predicate on the low bits alone tests no level-zero node, so the cursor
    // descends through the level it skips.
    let p = bit(&*c, 0x3F, 5);
    assert_eq!(
        members(&*c, p),
        (0..4096).filter(|v| v & 0x3F == 5).collect::<Vec<_>>()
    );
    assert!(members(&*c, Function::FALSE).is_empty());
    // A seventy-bit value straddles the word boundary: one level holds bits 60 to
    // 65.
    let w = col(&store, ScalarSort::bit_vec(70));
    let sorted = [5, 0, 1 << 60, 0, 5, 1];
    let q = w.set(&sorted);
    let mut cursor = w.members(q);
    let (mut out, mut v) = (Vec::new(), Vec::new());
    while cursor.next(&mut v) {
        out.extend_from_slice(&v);
    }
    assert_eq!(out, sorted);
}

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
fn repeated_constructions_preserve_results_across_collection_and_growth() {
    // Each iteration releases its results so collection can reclaim prior work.
    // Repeated constructions exercise level tests, moves, complements, and
    // conjunctions as the small store collects and grows.
    let store = Arc::new(BddStore::with_capacity(4, 16, 1));
    let c = col(&store, BV12);
    // Assign the low six bits and toggle bit six, so both levels move.
    let map = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let moved = |x: u64| (x & !0x3F) ^ 0x55;
    for k in 1..24 {
        let p = set(&*c, (0..8).map(|i| i * k + 1));
        let q = set(&*c, (0..8).map(|i| i * k * 7));
        let (in_p, in_q) = (members(&*c, p), members(&*c, q));
        let mut expect: Vec<u64> = in_p
            .iter()
            .map(|&x| moved(x))
            .filter(|y| in_q.contains(y))
            .collect();
        expect.sort_unstable();
        expect.dedup();
        let image = c.image(&map, p, q);
        assert_eq!(members(&*c, image), expect, "image at {k}");
        let expect: Vec<u64> = in_p
            .iter()
            .copied()
            .filter(|&x| in_q.contains(&moved(x)))
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
