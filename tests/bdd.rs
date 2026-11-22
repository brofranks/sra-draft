//! Public-contract tests for `sra::bdd`.
//!
//! Tests exercise the public `Algebra` operations using concrete scalar values
//! and check that equivalent diagrams have the same root. Small store
//! capacities force collection and growth at different construction steps.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread;

use sra::backend::{Algebra, Align, Backend, BoolOp, Function, MapSpec, Refine, transfer};
use sra::bdd::BddStore;
use sra::scalar::ScalarSort;
use sra::util::count::Count;

#[path = "support/bdd.rs"]
mod fixtures;
#[path = "support/listed.rs"]
mod listed;

use fixtures::{
    BV4, BV12, BV18, bind, bit, classes, members, residues, roomy, samples, set, shared,
    shared_holds, tiny,
};
use listed::Listed;

/// A partition with the shape of [`shared`]: values in `shared(alg, (1, 2))`
/// take one class, values in `shared(alg, (3, 4))` another, and every other
/// value a third.
fn diamond(alg: &dyn Algebra) -> Function {
    let (a, b) = (shared(alg, (1, 2)), shared(alg, (3, 4)));
    let root = alg.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
    alg.dec_ref(a);
    alg.dec_ref(b);
    root
}

/// The concrete meaning of [`diamond`] as its case labels.
fn diamond_case(x: u64) -> u32 {
    if shared_holds((1, 2), x) {
        1
    } else if shared_holds((3, 4), x) {
        2
    } else {
        0
    }
}

// --- construction and binding ----

#[test]
#[should_panic(expected = "maintenance thread count must be nonzero")]
fn a_store_with_zero_maintenance_threads_is_rejected() {
    BddStore::with_capacity(2, 0, 0);
}

#[test]
fn bdd_binds_every_scalar_sort() {
    let store: Arc<dyn Backend> = Arc::new(BddStore::default());
    for sort in [
        ScalarSort::bool(),
        ScalarSort::bv(1),
        BV4,
        ScalarSort::bv(70),
        ScalarSort::symbol(),
    ] {
        let alg = Arc::clone(&store).bind(sort).unwrap();
        assert_eq!(alg.sort(), sort);
    }
}

#[test]
fn repeated_bindings_share_live_roots() {
    let store = tiny();
    let a = bind(&store, BV4);
    let b = bind(&store, BV4);
    let p = a.set(&[1, 3]);
    let q = b.set(&[2, 3]);
    let r = a.apply(BoolOp::Or, p, q);
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
fn an_owned_root_outlives_its_creating_binding() {
    let store = tiny();
    let a = bind(&store, BV12);
    let b = bind(&store, BV12);
    let p = a.set(&[1, 65, 4095]);
    drop(a);
    drop(store);
    assert_eq!(b.evaluate(p, &[65]), 1);
    assert_eq!(b.evaluate(p, &[64]), 0);
    assert_eq!(members(&*b, p), [1, 65, 4095]);
    b.dec_ref(p);
}

#[test]
#[should_panic(expected = "cannot decrement an unreferenced or reclaimed DAG node")]
fn releasing_a_nonconstant_root_twice_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let p = alg.set(&[1]);
    alg.dec_ref(p);
    alg.dec_ref(p);
}

#[test]
fn evaluation_requires_an_exact_encoding_for_every_root() {
    let alg = bind(&tiny(), ScalarSort::bv(65));
    let max = [u64::MAX, 1];
    let p = alg.set(&max);
    for root in [Function::FALSE, Function::TRUE, p] {
        assert_eq!(alg.evaluate(root, &max), u32::from(!root.is_f()));
        for value in [&[0][..], &[0, 0, 0], &[0, 2]] {
            let panic = catch_unwind(AssertUnwindSafe(|| alg.evaluate(root, value)))
                .expect_err("evaluation must reject an invalid encoding");
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"value is outside the scalar sort")
            );
        }
    }
    alg.dec_ref(p);
}

// --- counting ----

#[test]
fn a_cut_and_its_complement_cover_each_bound_sort() {
    // Twelve bits divide evenly, but four do not: level zero then reaches only part
    // of its 64 assignments and the unreachable ones must not be counted.
    let store = roomy();
    let algebras = [
        bind(&store, BV4),
        bind(&store, BV12),
        bind(&store, ScalarSort::bv(70)),
    ];
    let cuts = [
        algebras[0].lt(&[7]),
        algebras[1].lt(&[7]),
        algebras[2].lt(&[7, 0]),
    ];
    let sizes = [Count::from(16), Count::from(4096), Count::from(1) << 70];
    // Alternate retained bindings to check that each sort, including the
    // multiword sort, keeps its own interpretation in the shared store.
    for i in [0, 2, 1, 0] {
        let alg = &*algebras[i];
        assert_eq!(alg.count(cuts[i]), 7);
        let mut cursor = alg.members(cuts[i]);
        for x in 0..7 {
            assert_eq!(cursor.next(), Some(&[x, 0][..alg.sort().words()]));
        }
        assert_eq!(cursor.next(), None);
        // The four- and seventy-bit complements pass through a short first level.
        let other = alg.apply(BoolOp::AndNot, Function::TRUE, cuts[i]);
        assert_eq!(alg.count(other) + Count::from(7), sizes[i]);
        alg.dec_ref(other);
    }
    for (alg, p) in algebras.iter().zip(cuts) {
        alg.dec_ref(p);
    }
}

#[test]
fn multiword_cuts_and_skipped_levels_have_exact_counts() {
    let alg = bind(&tiny(), ScalarSort::bv(70));
    for low in [0, 5] {
        let cut = alg.lt(&[low, 1]);
        assert_eq!(alg.count(cut), (Count::from(1) << 64) + Count::from(low));
        for (point, expected) in [
            ([u64::MAX, 0], true),
            ([4, 1], 4 < low),
            ([low, 1], false),
            ([64, 1], false),
            ([0, 2], false),
        ] {
            assert_eq!(alg.evaluate(cut, &point) != 0, expected);
        }
        alg.dec_ref(cut);
    }
    let low = alg.masked_eq(&[1, 0], &[1, 0]);
    assert_eq!(alg.count(low), Count::from(1) << 69);
    assert_eq!(alg.count(Function::TRUE), Count::from(1) << 70);
    alg.dec_ref(low);
}

#[test]
fn wide_singletons_and_complements_have_exact_counts() {
    let sort = ScalarSort::bv(if cfg!(miri) { 130 } else { 4096 });
    let alg = bind(&roomy(), sort);
    let p = alg.set(&vec![0; sort.words()]);
    assert_eq!(alg.count(p), 1);
    let q = alg.apply(BoolOp::AndNot, Function::TRUE, p);
    assert_eq!(
        alg.count(q) + Count::from(1),
        Count::from(1) << sort.width()
    );
    alg.dec_ref(q);
    alg.dec_ref(p);
}

#[test]
fn small_domain_counts_match_concrete_sets() {
    for width in [4, 6, 7, if cfg!(miri) { 8 } else { 12 }] {
        let alg = bind(&roomy(), ScalarSort::bv(width));
        let end = 1_u64 << width;
        assert_eq!(alg.count(Function::FALSE), 0);
        assert_eq!(alg.count(Function::TRUE), end);
        // Low-only and high-only masks skip levels on opposite sides. Widths
        // four and seven also exercise a partial first level.
        for mask in [0, 1, 1 << (width - 1), end - 1] {
            let value = mask & 0x555;
            let expected: Vec<_> = (0..end).filter(|&x| x & mask == value).collect();
            let p = bit(&*alg, mask, value);
            assert_eq!(alg.count(p), expected.len() as u64);
            assert_eq!(members(&*alg, p), expected);
            alg.dec_ref(p);
        }
        let expected: Vec<_> = (0..end).filter(|x| x % 7 == 2).collect();
        let p = alg.set(&expected);
        assert_eq!(alg.count(p), expected.len() as u64);
        alg.dec_ref(p);
    }
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    alg.count(f);
}

// --- sets, cuts, and bit tests ----

#[test]
#[should_panic(expected = "symbols have no numeric ordering")]
fn a_cut_over_symbols_is_rejected() {
    let _ = bind(&roomy(), ScalarSort::symbol()).lt(&[5]);
}

#[test]
#[should_panic(expected = "symbols do not support bit predicates")]
fn a_bit_test_over_symbols_is_rejected() {
    let _ = bind(&roomy(), ScalarSort::symbol()).masked_eq(&[1], &[1]);
}

#[test]
#[should_panic(expected = "bound is outside the scalar sort")]
fn a_cut_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), BV4).lt(&[0x10]);
}

#[test]
#[should_panic(expected = "bit test is outside the scalar sort")]
fn a_bit_test_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), BV4).masked_eq(&[0x10], &[0]);
}

#[test]
#[should_panic(expected = "bit test is outside the scalar sort")]
fn a_bit_test_value_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), BV4).masked_eq(&[0x1], &[0x11]);
}

#[test]
#[should_panic(expected = "bit test value has bits outside its mask")]
fn a_bit_test_value_outside_its_mask_is_rejected() {
    let _ = bind(&roomy(), BV4).masked_eq(&[0x1], &[0x3]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn unsorted_set_values_are_rejected() {
    let _ = bind(&roomy(), BV4).set(&[3, 1]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn repeated_set_values_are_rejected() {
    let _ = bind(&roomy(), ScalarSort::bool()).set(&[1, 1]);
}

#[test]
#[should_panic(expected = "value is outside the scalar sort")]
fn a_set_value_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), ScalarSort::bool()).set(&[0, 2]);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn set_values_off_the_word_stride_are_rejected() {
    let _ = bind(&roomy(), ScalarSort::bv(70)).set(&[1, 0, 2]);
}

// --- boolean operations ----

#[test]
fn xor_and_its_complement_respect_bit_positions() {
    let alg = bind(&roomy(), ScalarSort::bv(7));
    let high = bit(&*alg, 64, 64);
    let low = bit(&*alg, 1, 1);
    let parity = alg.apply(BoolOp::Xor, high, low);
    let canceled = alg.apply(BoolOp::Xor, parity, low);
    let complement = alg.apply(BoolOp::AndNot, Function::TRUE, parity);
    for (x, high_bit, bits_equal) in [(0, 0, 1), (1, 0, 0), (64, 1, 0), (65, 1, 1)] {
        assert_eq!(alg.evaluate(canceled, &[x]), high_bit, "at {x}");
        assert_eq!(alg.evaluate(complement, &[x]), bits_equal, "at {x}");
    }
    for f in [high, low, parity, canceled, complement] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn an_unreferenced_left_boolean_operand_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (p, q) = (set(&*alg, [1]), set(&*alg, [2]));
    alg.dec_ref(p);
    let _ = alg.apply(BoolOp::And, p, q);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn an_unreferenced_right_boolean_operand_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (p, q) = (set(&*alg, [1]), set(&*alg, [2]));
    alg.dec_ref(q);
    let _ = alg.apply(BoolOp::And, p, q);
}

// --- images and preimages ----

#[cfg(debug_assertions)]
#[track_caller]
fn rejects_unreferenced(build: impl FnOnce() -> Function) {
    let panic = catch_unwind(AssertUnwindSafe(build))
        .expect_err("the operation accepted an unreferenced BDD node");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"cannot use an unreferenced BDD node")
    );
}

#[test]
#[cfg(debug_assertions)]
fn bit_maps_check_both_input_roots() {
    let alg = bind(&roomy(), BV4);
    let released = alg.set(&[1]);
    let live = alg.set(&[2]);
    alg.dec_ref(released);
    for (clear, flip) in [(1, 0), (0, 1), (15, 3)] {
        let map = MapSpec::Bits {
            clear: [clear].into(),
            flip: [flip].into(),
        };
        for other in [Function::FALSE, Function::TRUE, live] {
            for (p, q) in [(released, other), (other, released)] {
                rejects_unreferenced(|| alg.image(&map, p, q));
                rejects_unreferenced(|| alg.preimage(&map, p, q));
            }
        }
    }
    alg.dec_ref(live);
}

#[test]
fn bit_maps_confined_to_one_level_agree_with_pointwise_maps() {
    let store = tiny();
    let alg = bind(&store, BV18);
    let points = [0, 5, 0x41, 0x7C0, 0x1003, 0x2FFFF, 0x3F040, 0x3FFFF];
    let p = set(&*alg, points);
    // Each map affects at most one level. Lower levels keep their assignments.
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
        expect.sort_unstable();
        expect.dedup();
        let image = alg.image(&map, p, Function::TRUE);
        assert_eq!(members(&*alg, image), expect);
        let in_q: Vec<_> = expect.into_iter().step_by(2).collect();
        let q = alg.set(&in_q);
        let expect: Vec<u64> = points
            .into_iter()
            .filter(|&x| in_q.contains(&f(x)))
            .collect();
        let preimage = alg.preimage(&map, p, q);
        assert_eq!(members(&*alg, preimage), expect);
        for g in [image, q, preimage] {
            alg.dec_ref(g);
        }
    }
    alg.dec_ref(p);
}

#[test]
fn clearing_a_low_bit_merges_the_branches_of_a_short_upper_level() {
    let alg = bind(&tiny(), ScalarSort::bv(7));
    // Both halves admit 2 modulo 4. Only the lower half also admits 3.
    let q = set(
        &*alg,
        (0..128).filter(|&x| x % 4 == 2 || x < 64 && x % 4 == 3),
    );
    let map = MapSpec::Bits {
        clear: [1].into(),
        flip: [0].into(),
    };
    let preimage = alg.preimage(&map, Function::TRUE, q);
    let expected = set(&*alg, (0..128).filter(|x| x % 4 >= 2));
    assert_eq!(preimage, expected);
    let image = alg.image(&map, Function::TRUE, q);
    assert_eq!(
        members(&*alg, image),
        (2..128).step_by(4).collect::<Vec<_>>()
    );
    for f in [q, preimage, expected, image] {
        alg.dec_ref(f);
    }
}

#[test]
fn wide_bit_maps_agree_with_literal_finite_images_and_preimages() {
    let sort = ScalarSort::bv(70);
    let alg = bind(&tiny(), sort);
    let points = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1], vec![8, 2]];
    let p = alg.set(&[0, 0, 1, 0, 0, 1, 1, 1, 8, 2]);
    let q = alg.set(&[4, 2]);
    let map = MapSpec::Bits {
        clear: vec![1, 1].into(),
        flip: vec![4, 2].into(),
    };
    let image = alg.image(&map, p, Function::TRUE);
    let preimage = alg.preimage(&map, p, q);
    for (forward, root, expected) in [
        (true, image, vec![12, 0, 4, 2]),
        (false, preimage, vec![0, 0, 1, 0, 0, 1, 1, 1]),
    ] {
        let mut cursor = alg.members(root);
        let mut words = Vec::new();
        while let Some(point) = cursor.next() {
            words.extend_from_slice(point);
        }
        assert_eq!(words, expected);
        assert_eq!(
            transfer(
                sort,
                &map,
                &mut Listed(&points),
                |value| forward || value == [4, 2],
                forward,
            ),
            expected
        );
    }
    for f in [p, q, image, preimage] {
        alg.dec_ref(f);
    }
}

#[test]
fn a_high_word_flip_reaches_a_level_spanning_two_words() {
    let alg = bind(&roomy(), ScalarSort::bv(70));
    // Bits 60 through 65 share a level. Only bit 64 changes.
    let mask = [0, 1];
    let zero = alg.masked_eq(&mask, &[0, 0]);
    let one = alg.masked_eq(&mask, &mask);
    let map = MapSpec::Bits {
        clear: [0, 0].into(),
        flip: mask.into(),
    };
    let image = alg.image(&map, zero, Function::TRUE);
    let preimage = alg.preimage(&map, Function::TRUE, zero);
    assert_eq!(image, one);
    assert_eq!(preimage, one);
    for f in [zero, one, image, preimage] {
        alg.dec_ref(f);
    }
}

#[test]
fn overwrites_respect_every_word_of_the_mask_and_the_target() {
    let alg = bind(&tiny(), ScalarSort::bv(65));
    let p = alg.set(&[0, 0, 2, 1]);
    let q = alg.set(&[4, 1, 5, 1]);
    let single = alg.set(&[4, 1]);
    let high = alg.set(&[2, 1]);
    for (clear, flip, expected) in [
        ([u64::MAX, 1], [4, 1], (single, p)),
        ([u64::MAX, 1], [4, 0], (Function::FALSE, Function::FALSE)),
        ([u64::MAX - 1, 1], [4, 1], (single, p)),
        ([u64::MAX, 0], [4, 0], (single, high)),
        ([0, 1], [6, 1], (single, high)),
    ] {
        let map = MapSpec::Bits {
            clear: clear.into(),
            flip: flip.into(),
        };
        let image = alg.image(&map, p, q);
        let preimage = alg.preimage(&map, p, q);
        assert_eq!((image, preimage), expected, "{map:?}");
        alg.dec_ref(image);
        alg.dec_ref(preimage);
    }
    for f in [p, q, single, high] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn bit_maps_preserve_constraints_on_untouched_levels() {
    let store = roomy();
    let alg = bind(&store, BV18);
    // The predicate tests levels zero and two. The map clears bit six on level
    // one, so the image and preimage must still satisfy both tests.
    let (hi, low) = (bit(&*alg, 1 << 12, 1 << 12), bit(&*alg, 1, 1));
    let p = alg.apply(BoolOp::And, hi, low);
    let map = MapSpec::Bits {
        clear: vec![1 << 6].into(),
        flip: vec![0].into(),
    };
    let image = alg.image(&map, p, Function::TRUE);
    let expect: Vec<u64> = (0..1 << 18)
        .filter(|x| x & (1 << 12) != 0 && x & 1 != 0 && x & (1 << 6) == 0)
        .collect();
    assert_eq!(members(&*alg, image), expect);
    let preimage = alg.preimage(&map, Function::TRUE, image);
    let expect: Vec<u64> = (0..1 << 18)
        .filter(|x| x & (1 << 12) != 0 && x & 1 != 0)
        .collect();
    assert_eq!(members(&*alg, preimage), expect);
    for f in [hi, low, p, image, preimage] {
        alg.dec_ref(f);
    }
}

/// A twelve-bit algebra whose store capacity varies to exercise different
/// allocation boundaries.
fn small(slack: usize) -> Box<dyn Algebra> {
    bind(
        &Arc::new(BddStore::with_capacity(2 + slack, slack * 64, 1)),
        BV12,
    )
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
#[expect(
    clippy::type_complexity,
    reason = "the paired spec and its reference implementation read better inline"
)]
fn bit_maps_match_concrete_results_across_small_capacities() {
    // Clearing the last level merges terminals. Clearing the first takes the
    // disjunction of whole subgraphs. Either can exhaust the arena.
    let maps: [(MapSpec, fn(u64) -> u64); 3] = [
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
        (
            MapSpec::Bits {
                clear: [1].into(),
                flip: [3].into(),
            },
            |x| (x & !1) ^ 3,
        ),
    ];
    let input: Vec<u64> = (0..40).map(|i| i * 101 + 1).collect();
    let allowed: Vec<u64> = (0..40).map(|i| i * 37).chain([3861]).collect();
    for ((map, eval), pair) in maps.iter().zip([(3940, 3861), (1516, 37), (405, 407)]) {
        let pairs: Vec<_> = input
            .iter()
            .map(|&x| (x, eval(x)))
            .filter(|(_, y)| allowed.contains(y))
            .collect();
        assert_eq!(pairs, [pair]);
        for slack in 0..8 {
            // A fresh store gives both directions the same initial capacity.
            for forward in [true, false] {
                let alg = small(slack);
                let p = alg.set(&input);
                let q = alg.set(&allowed);
                let (f, expected) = if forward {
                    (alg.image(map, p, q), pair.1)
                } else {
                    (alg.preimage(map, p, q), pair.0)
                };
                assert_eq!(
                    members(&*alg, f),
                    [expected],
                    "slack {slack}, image {forward}"
                );
                for root in [p, q, f] {
                    alg.dec_ref(root);
                }
            }
        }
    }
}

#[test]
fn malformed_bit_maps_are_rejected_by_symbolic_and_finite_transfer() {
    const MASKS: &str = "bit map masks must encode the scalar sort";
    const SYMBOLS: &str = "symbols do not support bitwise maps";
    let masks: &[(ScalarSort, &[u64], &[u64], &str)] = &[
        (BV4, &[], &[], MASKS),
        (BV4, &[], &[0], MASKS),
        (BV4, &[0], &[], MASKS),
        (BV4, &[0, 0], &[0], MASKS),
        (BV4, &[0], &[0, 0], MASKS),
        (BV4, &[16], &[0], MASKS),
        (BV4, &[0], &[16], MASKS),
        (ScalarSort::bv(70), &[0], &[0], MASKS),
        (ScalarSort::bv(70), &[0, 64], &[0, 0], MASKS),
        (ScalarSort::bv(70), &[0, 0], &[0, 64], MASKS),
        (ScalarSort::symbol(), &[], &[], MASKS),
        (ScalarSort::symbol(), &[0], &[1], SYMBOLS),
    ];
    // The sort check itself must reject each map, not a later failure.
    let rejects = |result: thread::Result<()>, message: &str| {
        let payload = result.expect_err("a malformed bit map must be rejected");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&message));
    };
    for &(sort, clear, flip, message) in masks {
        let alg = bind(&roomy(), sort);
        let map = MapSpec::Bits {
            clear: clear.into(),
            flip: flip.into(),
        };
        let zero = vec![0; sort.words()];
        let p = alg.set(&zero);
        for predicate in [Function::FALSE, p] {
            for image in [false, true] {
                rejects(
                    catch_unwind(AssertUnwindSafe(|| {
                        let _ = if image {
                            alg.image(&map, predicate, Function::TRUE)
                        } else {
                            alg.preimage(&map, predicate, Function::TRUE)
                        };
                    })),
                    message,
                );
                let points = if predicate.is_f() {
                    vec![]
                } else {
                    vec![zero.clone()]
                };
                rejects(
                    catch_unwind(AssertUnwindSafe(|| {
                        transfer(sort, &map, &mut Listed(&points), |_| true, image);
                    })),
                    message,
                );
            }
        }
        alg.dec_ref(p);
    }
}

#[test]
fn a_rust_map_on_a_bdd_is_rejected() {
    const MESSAGE: &str = "a BDD store cannot apply a Rust map, which needs an enumerating backend";
    let alg = bind(&tiny(), BV4);
    let map = MapSpec::Rust(Arc::new(|_, _| true));
    for p in [Function::FALSE, Function::TRUE] {
        for q in [Function::FALSE, Function::TRUE] {
            for image in [false, true] {
                let panic = catch_unwind(AssertUnwindSafe(|| {
                    if image {
                        alg.image(&map, p, q)
                    } else {
                        alg.preimage(&map, p, q)
                    }
                }))
                .expect_err("a BDD store must reject Rust maps");
                assert_eq!(panic.downcast_ref::<&str>(), Some(&MESSAGE));
            }
        }
    }
}

// --- alignment ----

#[test]
fn alignment_numbers_interleaved_masks_at_their_first_assignment() {
    let alg = bind(&roomy(), ScalarSort::bv(6));
    let keys: Vec<u64> = (0..64).collect();
    let left: Vec<u32> = (0..64).map(|x| x.min(63 - x)).collect();
    let right: Vec<u32> = (0..64)
        .map(|x| if x % 2 == 0 { x / 2 } else { 0 })
        .collect();
    let f = alg.values(100, &keys, &left, &mut Vec::new(), &mut Default::default());
    let g = alg.values(100, &keys, &right, &mut Vec::new(), &mut Default::default());
    let mut labels = Vec::new();
    let h = alg.align(f, g, &mut labels, &mut Align::default());
    let mut expected = Vec::new();
    for ((x, l), r) in keys.into_iter().zip(left).zip(right) {
        if !expected.contains(&(l, r)) {
            expected.push((l, r));
        }
        assert_eq!(labels[alg.evaluate(h, &[x]) as usize], (l, r));
    }
    assert_eq!(labels, expected);
    for p in [f, g, h] {
        alg.dec_ref(p);
    }
}

#[test]
fn a_later_alignment_cache_hit_returns_its_own_labels() {
    let alg = bind(&roomy(), BV4);
    let a = alg.set(&[1]);
    let b = alg.set(&[2]);
    let both = alg.set(&[1, 2]);
    let mut scratch = Align::default();
    let mut labels = Vec::new();
    let first = alg.align(a, b, &mut labels, &mut scratch);
    assert_eq!(labels, [(0, 0), (1, 0), (0, 1)]);
    labels.clear();
    let second = alg.align(a, both, &mut labels, &mut scratch);
    assert_eq!(labels, [(0, 0), (1, 1), (0, 1)]);
    let mut cached = vec![(9, 9)];
    let hit = alg.align(a, both, &mut cached, &mut scratch);
    assert_eq!(hit, second);
    assert_eq!(cached, [(9, 9), (0, 0), (1, 1), (0, 1)]);
    labels.clear();
    let reversed = alg.align(b, a, &mut labels, &mut scratch);
    for (x, pair) in [(0, (0, 0)), (1, (0, 1)), (2, (1, 0))] {
        let class = alg.evaluate(reversed, &[x]);
        assert_eq!(labels[class as usize], pair, "at {x}");
    }
    for f in [a, b, both, first, second, hit, reversed] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn alignment_shares_pairs_and_splits_below_a_terminal_side() {
    let store = roomy();
    let alg = bind(&store, BV18);
    let d = diamond(&*alg);
    let odd = bit(&*alg, 1, 1);
    let parity = alg.cases(0, &[(odd, 1)], &mut Vec::new());
    let top = bit(&*alg, 1 << 17, 1 << 17);
    let half = alg.cases(0, &[(top, 1)], &mut Vec::new());
    let mut scratch = Align::default();
    // The diamond reaches its level-two pairs by two routes, and below the top
    // level each half of `half` is a terminal beside a branching `parity`.
    let pairs = [
        (d, diamond_case as fn(u64) -> u32, 6),
        (half, |x| (x >> 17) as u32, 4),
    ];
    for (f, left, count) in pairs {
        let mut labels = Vec::new();
        let h = alg.align(f, parity, &mut labels, &mut scratch);
        let mut first = Vec::new();
        for x in 0..1 << 18 {
            let pair = (left(x), (x & 1) as u32);
            assert_eq!(labels[alg.evaluate(h, &[x]) as usize], pair, "at {x}");
            if !first.contains(&pair) {
                first.push(pair);
            }
        }
        assert_eq!(labels, first);
        // Build the same classes from concrete integer predicates, independently
        // of the aligned diagram and its class numbering.
        let label = |x| left(x) * 2 + (x & 1) as u32;
        let cases: Vec<_> = (1..count)
            .map(|i| (set(&*alg, (0..1 << 18).filter(|&x| label(x) == i)), i))
            .collect();
        let expected = alg.cases(0, &cases, &mut Vec::new());
        assert_eq!(h, expected);
        for (p, _) in cases {
            alg.dec_ref(p);
        }
        alg.dec_ref(expected);
        alg.dec_ref(h);
    }
    for f in [d, odd, parity, top, half] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a partition with itself")]
fn alignment_of_identical_partitions_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    let _ = alg.align(f, f, &mut Vec::new(), &mut Align::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a trivial partition")]
fn alignment_of_a_trivial_left_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    let _ = alg.align(Function::FALSE, f, &mut Vec::new(), &mut Align::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot align a trivial partition")]
fn alignment_of_a_trivial_right_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let f = residues(&*alg, 3);
    let _ = alg.align(f, Function::FALSE, &mut Vec::new(), &mut Align::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn alignment_of_an_unreferenced_left_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (f, g) = (residues(&*alg, 3), residues(&*alg, 2));
    alg.dec_ref(f);
    let _ = alg.align(f, g, &mut Vec::new(), &mut Align::default());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn alignment_of_an_unreferenced_right_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    // The fixture releases its case predicates, leaving only this root reference.
    let (f, g) = (residues(&*alg, 3), residues(&*alg, 4));
    alg.dec_ref(g);
    let _ = alg.align(f, g, &mut Vec::new(), &mut Align::default());
}

// --- quotients ----

#[test]
fn quotient_scratch_can_be_reused_across_stores_with_matching_root_ids() {
    let [a, b] = [roomy(), roomy()].map(|store| store.bind(BV4).unwrap());
    let [f, g] = [(&a, [1, 2]), (&b, [3, 4])]
        .map(|(alg, keys)| alg.values(0, &keys, &[7, 8], &mut Vec::new(), &mut Default::default()));
    assert_eq!(f.index(), g.index());
    let mut scratch = Align::default();
    let classes = [0, 1, 0];
    let first = a.quotient(f, &classes, &mut scratch);
    // A stale hit would return a live predicate for 8 in the second store.
    let other = b.set(&[8]);
    assert_eq!(first.index(), other.index());
    let second = b.quotient(g, &classes, &mut scratch);
    assert_eq!(members(&*a, first), [1]);
    assert_eq!(members(&*b, second), [3]);
    a.dec_ref(f);
    a.dec_ref(first);
    b.dec_ref(g);
    b.dec_ref(other);
    b.dec_ref(second);
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn a_quotient_merges_two_diamond_classes() {
    let store = roomy();
    let alg = bind(&store, BV18);
    let d = diamond(&*alg);
    // Inputs 0, 1, and 3 first reach the three classes in that order.
    // Merge the two shared cases and keep the rest apart.
    let q = alg.quotient(d, &[0, 1, 1], &mut Align::default());
    for x in samples() {
        let class = diamond_case(x);
        assert_eq!(alg.evaluate(d, &[x]), class, "at {x}");
        assert_eq!(alg.evaluate(q, &[x]), u32::from(class != 0), "at {x}");
    }
    // Construct the union from its concrete meaning rather than the quotient's
    // source classes.
    let p = set(&*alg, (0..1 << 18).filter(|&x| diamond_case(x) != 0));
    let expected = alg.cases(0, &[(p, 1)], &mut Vec::new());
    assert_eq!(q, expected);
    alg.dec_ref(expected);
    alg.dec_ref(p);
    alg.dec_ref(q);
    alg.dec_ref(d);
}

// --- cases ----

#[test]
fn cases_over_one_predicate_reuse_it_or_its_complement() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let p = set(&*alg, [1, 3, 5]);
    let mut labels = Vec::new();
    // A case labeled like the default, or over a constant, is the one-class
    // partition.
    assert_eq!(alg.cases(7, &[(p, 7)], &mut labels), Function::FALSE);
    assert_eq!(
        alg.cases(7, &[(Function::TRUE, 8)], &mut labels),
        Function::FALSE
    );
    assert_eq!(
        alg.cases(7, &[(Function::FALSE, 8)], &mut labels),
        Function::FALSE
    );
    assert_eq!(labels, [7, 8, 7]);
    labels.clear();
    // Zero is the first value, so a predicate rejecting it already numbers its
    // classes canonically.
    let reused = alg.cases(0, &[(p, 1)], &mut labels);
    assert_eq!(reused, p);
    alg.dec_ref(reused);
    assert_eq!(labels, [0, 1]);
    labels.clear();
    // One accepting zero does not, and is complemented instead.
    let q = set(&*alg, [0, 2]);
    let f = alg.cases(5, &[(q, 6)], &mut labels);
    assert_eq!(labels, [6, 5]);
    assert_eq!(
        classes(&*alg, f),
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    for root in [p, q, f] {
        alg.dec_ref(root);
    }
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn overlapping_cases_with_different_labels_are_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (a, b) = (set(&*alg, [1, 2, 3]), set(&*alg, [3, 4]));
    let _ = alg.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

#[test]
fn disjoint_cases_with_a_shared_child_keep_their_labels() {
    let alg = bind(&roomy(), BV12);
    let a = alg.set(&[1]);
    let b = alg.set(&[65]);
    let mut labels = Vec::new();
    let f = alg.cases(7, &[(a, 8), (b, 9)], &mut labels);
    for (x, label) in [(0, 7), (1, 8), (64, 7), (65, 9)] {
        assert_eq!(labels[alg.evaluate(f, &[x]) as usize], label, "at {x}");
    }
    for root in [a, b, f] {
        alg.dec_ref(root);
    }
}

#[test]
fn disjoint_default_labeled_cases_change_nothing() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (a, b) = (set(&*alg, [1, 2]), set(&*alg, [3, 4]));
    for cases in [[(a, 0), (b, 1)], [(b, 1), (a, 0)]] {
        let mut labels = Vec::new();
        let f = alg.cases(0, &cases, &mut labels);
        assert_eq!(labels, [0, 1]);
        assert_eq!(f, b);
        alg.dec_ref(f);
    }
    alg.dec_ref(a);
    alg.dec_ref(b);
}

// --- values ----

#[test]
fn full_key_sets_and_single_class_partitions_return_the_constants() {
    let alg = bind(&tiny(), ScalarSort::bv(7));
    let keys: Vec<_> = (0..128).collect();
    let mut labels = vec![77];
    assert_eq!(alg.set(&keys), Function::TRUE);
    let f = alg.values(9, &keys, &[42; 128], &mut labels, &mut Default::default());
    assert_eq!(f, Function::FALSE);
    assert_eq!(labels, [77, 42]);

    let f = alg.values(
        9,
        &keys[..127],
        &[42; 127],
        &mut labels,
        &mut Default::default(),
    );
    assert_eq!(labels, [77, 42, 42, 9]);
    assert_eq!(alg.evaluate(f, &[126]), 0);
    assert_eq!(alg.evaluate(f, &[127]), 1);
    alg.dec_ref(f);
}

#[test]
fn values_number_the_default_at_the_first_multiword_gap() {
    let alg = bind(&tiny(), ScalarSort::bv(65));
    for (keys, gap, expected) in [
        ([[0, 0], [1, 1], [2, 1]], [1, 0], [11, 9, 12]),
        ([[0, 0], [1, 0], [2, 0]], [3, 0], [11, 12, 9]),
    ] {
        let mut labels = Vec::new();
        let cases = [11, 12, 11];
        let f = alg.values(
            9,
            keys.as_flattened(),
            &cases,
            &mut labels,
            &mut Default::default(),
        );
        assert_eq!(labels, expected);
        for (key, label) in keys.iter().zip(cases) {
            assert_eq!(labels[alg.evaluate(f, key) as usize], label);
        }
        assert_eq!(labels[alg.evaluate(f, &gap) as usize], 9);
        alg.dec_ref(f);
    }
}

#[test]
fn shared_prefixes_preserve_wide_key_membership_and_label_offsets() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let keys = [[1, 0], [2, 0], [65, 0], [0, 1], [1, 1], [65, 1]];
    let cases = [8, 8, 7, 6, 8, 7];
    let p = alg.set(keys.as_flattened());
    let mut labels = vec![77];
    let f = alg.values(
        9,
        keys.as_flattened(),
        &cases,
        &mut labels,
        &mut Default::default(),
    );
    assert_eq!(labels, [77, 9, 8, 7, 6]);
    assert_eq!(alg.count(p), keys.len() as u64);
    for (key, label) in keys.iter().zip(cases) {
        assert_eq!(alg.evaluate(p, key), 1);
        assert_eq!(labels[1 + alg.evaluate(f, key) as usize], label);
    }
    for gap in [[0, 0], [64, 0], [2, 1]] {
        assert_eq!(alg.evaluate(p, &gap), 0);
        assert_eq!(alg.evaluate(f, &gap), 0);
    }
    alg.dec_ref(f);
    alg.dec_ref(p);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn value_keys_off_the_word_stride_are_rejected() {
    let alg = bind(&roomy(), ScalarSort::bv(70));
    let _ = alg.values(
        0,
        &[1, 0, 2],
        &[1],
        &mut Vec::new(),
        &mut Default::default(),
    );
}

#[test]
#[should_panic(expected = "value is outside the scalar sort")]
fn a_value_key_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), BV4).values(
        0,
        &[1, 0x10],
        &[1, 2],
        &mut Vec::new(),
        &mut Default::default(),
    );
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn unsorted_value_keys_are_rejected() {
    let _ = bind(&roomy(), BV4).values(
        0,
        &[2, 1],
        &[1, 2],
        &mut Vec::new(),
        &mut Default::default(),
    );
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn repeated_value_keys_are_rejected() {
    let alg = bind(&roomy(), ScalarSort::bool());
    let _ = alg.values(
        0,
        &[1, 1],
        &[7, 7],
        &mut Vec::new(),
        &mut Default::default(),
    );
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn values_with_too_few_labels_are_rejected() {
    let _ = bind(&roomy(), BV4).values(0, &[1, 2], &[1], &mut Vec::new(), &mut Default::default());
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn empty_values_with_a_case_label_are_rejected() {
    let _ = bind(&tiny(), BV4).values(42, &[], &[7], &mut Vec::new(), &mut Default::default());
}

// --- refinement ----

#[test]
fn empty_refinement_continues_without_visiting_a_class() {
    let alg = bind(&roomy(), BV4);
    let f = residues(&*alg, 3);
    let result = alg.refine(
        f,
        Function::FALSE,
        &mut Refine::default(),
        &mut |_| panic!("no class is reached"),
        &mut |_, _| panic!("no cell is delivered"),
    );
    assert_eq!(result, ControlFlow::Continue(()));
    alg.dec_ref(f);
}

#[test]
fn refinement_selects_cells_and_reuses_scratch_after_a_break() {
    let alg = bind(&roomy(), BV4);
    let f = residues(&*alg, 3);
    let p = alg.set(&[1, 2, 3, 4, 5]);
    let mut scratch = Refine::default();
    let mut cells = Vec::new();
    let result = alg.refine(
        f,
        p,
        &mut scratch,
        &mut |class| class != 2,
        &mut |class, build| {
            let cell = build();
            cells.push((class, members(&*alg, cell)));
            alg.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    cells.sort();
    assert!(result.is_continue());
    assert_eq!(cells, [(0, vec![3]), (1, vec![1, 4])]);

    // A break stops selection and delivery immediately.
    let mut selected = Vec::new();
    let mut delivered = Vec::new();
    let stopped = alg.refine(
        f,
        p,
        &mut scratch,
        &mut |class| {
            selected.push(class);
            true
        },
        &mut |class, _| {
            delivered.push(class);
            ControlFlow::Break(())
        },
    );
    assert!(stopped.is_break());
    assert_eq!(selected.len(), 1);
    assert_eq!(delivered, selected);

    // Visiting without requesting a cell still reports every reached class.
    delivered.clear();
    let resumed = alg.refine(f, p, &mut scratch, &mut |_| true, &mut |class, _| {
        delivered.push(class);
        ControlFlow::Continue(())
    });
    delivered.sort_unstable();
    assert!(resumed.is_continue());
    assert_eq!(delivered, [0, 1, 2]);
    alg.dec_ref(f);
    alg.dec_ref(p);
}

#[test]
fn a_one_class_partition_delivers_the_whole_predicate_only_when_selected() {
    let alg = bind(&roomy(), BV4);
    let p = alg.set(&[1, 3, 7]);
    for accept in [false, true] {
        let mut selected = Vec::new();
        let mut delivered = 0;
        let result = alg.refine(
            Function::FALSE,
            p,
            &mut Refine::default(),
            &mut |class| {
                selected.push(class);
                accept
            },
            &mut |class, build| {
                let cell = build();
                assert_eq!((class, cell), (0, p));
                alg.dec_ref(cell);
                delivered += 1;
                ControlFlow::Continue(())
            },
        );
        assert!(result.is_continue());
        assert_eq!(selected, [0]);
        assert_eq!(delivered, usize::from(accept));
    }
    alg.dec_ref(p);
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn refinement_cells_match_the_diamond_intersections() {
    let store = roomy();
    let alg = bind(&store, BV18);
    let d = diamond(&*alg);
    let p = bit(&*alg, 0x38, 0);
    let mut seen = Vec::new();
    let _ = alg.refine(
        d,
        p,
        &mut Refine::default(),
        &mut |_| true,
        &mut |class, build| {
            let cell = build();
            seen.push((class, members(&*alg, cell)));
            alg.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    let mut expected = [0, 1, 2].map(|class| (class, Vec::new()));
    for x in (0..1 << 18).filter(|x| x & 0x38 == 0) {
        expected[diamond_case(x) as usize].1.push(x);
    }
    seen.sort_unstable_by_key(|(class, _)| *class);
    assert_eq!(seen, expected);
    alg.dec_ref(d);
    alg.dec_ref(p);
}

#[test]
fn a_shallow_partition_selects_each_class_once_under_a_deep_predicate() {
    let store = tiny();
    let alg = bind(&store, BV18);
    let half = 1 << 17;
    let top = bit(&*alg, half, half);
    let f = alg.cases(0, &[(top, 1)], &mut Vec::new());
    // The multiples of three bordering the split reach both classes.
    let mut points: Vec<u64> = samples()
        .filter(|x| x % 3 == 0)
        .chain([half - 2, half + 1])
        .collect();
    points.sort_unstable();
    points.dedup();
    let p = set(&*alg, points.iter().copied());
    let (mut selected, mut cells) = (Vec::new(), Vec::new());
    let result = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |class| {
            selected.push(class);
            true
        },
        &mut |class, build| {
            let cell = build();
            cells.push((class, members(&*alg, cell)));
            alg.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    selected.sort_unstable();
    cells.sort();
    assert!(result.is_continue());
    assert_eq!(selected, [0, 1]);
    let (high, low): (Vec<u64>, Vec<u64>) = points.into_iter().partition(|x| x >> 17 == 1);
    assert_eq!(cells, [(0, low), (1, high)]);
    for g in [top, f, p] {
        alg.dec_ref(g);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn refinement_of_an_unreferenced_partition_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (f, p) = (residues(&*alg, 3), set(&*alg, [1, 2]));
    alg.dec_ref(f);
    let _ = alg.refine(f, p, &mut Refine::default(), &mut |_| true, &mut |_, _| {
        ControlFlow::Continue(())
    });
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn empty_refinement_of_an_unreferenced_partition_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let f = residues(&*alg, 3);
    alg.dec_ref(f);
    let _ = alg.refine(
        f,
        Function::FALSE,
        &mut Refine::default(),
        &mut |_| true,
        &mut |_, _| ControlFlow::Continue(()),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced BDD node")]
fn refinement_by_an_unreferenced_predicate_is_rejected() {
    let store = roomy();
    let alg = bind(&store, BV4);
    let (f, p) = (residues(&*alg, 3), set(&*alg, [1, 2]));
    alg.dec_ref(p);
    let _ = alg.refine(f, p, &mut Refine::default(), &mut |_| true, &mut |_, _| {
        ControlFlow::Continue(())
    });
}

// --- member cursors ----

#[test]
fn an_empty_wide_cursor_and_transfer_finish_without_visiting_a_value() {
    let sort = ScalarSort::bv(65_536);
    let alg = bind(&tiny(), sort);
    let mut cursor = alg.members(Function::FALSE);
    let map = MapSpec::Rust(Arc::new(|_, _| panic!("map received a value")));
    assert!(
        transfer(
            sort,
            &map,
            &mut *cursor,
            |_| panic!("filter received a value"),
            true
        )
        .is_empty()
    );
    assert_eq!(cursor.next(), None);
}

#[test]
fn member_cursors_skip_levels_and_cross_word_boundaries() {
    let store = roomy();
    let alg = bind(&store, BV12);
    // A predicate on the low bits alone tests no level-zero node, so the cursor
    // descends through the level it skips.
    let p = bit(&*alg, 0x3F, 5);
    assert_eq!(
        members(&*alg, p),
        (0..4096).filter(|v| v & 0x3F == 5).collect::<Vec<_>>()
    );
    assert!(members(&*alg, Function::FALSE).is_empty());
    // A seventy-bit value straddles the word boundary: one level holds bits 60 to
    // 65.
    let w = bind(&store, ScalarSort::bv(70));
    let sorted = [5, 0, 1 << 60, 0, 5, 1];
    let q = w.set(&sorted);
    let mut cursor = w.members(q);
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.extend_from_slice(v);
    }
    assert_eq!(out, sorted);
    alg.dec_ref(p);
    w.dec_ref(q);
}

#[test]
fn a_singleton_preserves_bits_on_both_sides_of_the_second_word_boundary() {
    let alg = bind(&roomy(), ScalarSort::bv(130));
    let key = [1, 1 << 62, 2];
    let p = alg.set(&key);
    assert_eq!(alg.evaluate(p, &[1, 1 << 62, 0]), 0);
    assert_eq!(alg.evaluate(p, &key), 1);
    let mut cursor = alg.members(p);
    assert_eq!(cursor.next(), Some(key.as_slice()));
    assert_eq!(cursor.next(), None);
    alg.dec_ref(p);
}

// --- bounded stacks ----

fn on_two_mib_stack(run: impl FnOnce() + Send + 'static) {
    thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn wide_boolean_products_conditionals_and_complements_fit_a_two_mib_stack() {
    let sort = ScalarSort::bv(if cfg!(miri) { 13 } else { 16_384 });
    let alg = bind(&roomy(), sort);
    let words = sort.words();
    let zero = vec![0; words];
    let mut one = zero.clone();
    one[0] = 1;
    let mut mask = vec![u64::MAX; words];
    mask[words - 1] = sort.top_mask();
    let a = alg.masked_eq(&mask, &zero);
    let b = alg.masked_eq(&mask, &one);
    mask[0] &= !1;
    let either = alg.masked_eq(&mask, &zero);

    on_two_mib_stack(move || {
        let union = alg.apply(BoolOp::Or, a, b);
        assert_eq!(union, either);
        let conditional = alg.ite(a, either, b);
        assert_eq!(conditional, either);
        let complement = alg.apply(BoolOp::AndNot, Function::TRUE, a);
        assert_eq!(alg.evaluate(complement, &zero), 0);
        assert_eq!(
            alg.count(complement) + Count::from(1),
            Count::from(1) << sort.width()
        );
        for root in [a, b, either, union, conditional, complement] {
            alg.dec_ref(root);
        }
    });
}

#[test]
fn wide_partition_products_and_filtered_transfers_fit_a_two_mib_stack() {
    let sort = ScalarSort::bv(if cfg!(miri) { 13 } else { 4096 });
    let alg = bind(&roomy(), sort);
    let words = sort.words();
    let mut keys = vec![0; 3 * words];
    keys[words] = 1;
    keys[2 * words] = 3;
    let (zero, one) = keys[..2 * words].split_at(words);
    let a = alg.set(zero);
    let b = alg.set(one);
    let p = alg.set(&keys[..2 * words]);
    let q = alg.set(&keys[words..]);
    let f = alg.values(1, zero, &[0], &mut Vec::new(), &mut Default::default());
    let partition = alg.values(
        17,
        &keys[..2 * words],
        &[4, 5],
        &mut Vec::new(),
        &mut Default::default(),
    );
    let map = MapSpec::Bits {
        clear: zero.into(),
        flip: one.into(),
    };

    on_two_mib_stack(move || {
        let mut labels = vec![77];
        let cases = alg.cases(17, &[(a, 4), (b, 5)], &mut labels);
        assert_eq!(cases, partition);
        assert_eq!(labels, [77, 4, 5, 17]);

        // Zero, one, and the remaining values each reach a different pair.
        let mut pairs = Vec::new();
        let aligned = alg.align(f, b, &mut pairs, &mut Align::default());
        assert_eq!(aligned, partition);
        assert_eq!(pairs, [(0, 0), (1, 1), (1, 0)]);
        let mut cells = Vec::new();
        let flow = alg.refine(
            f,
            p,
            &mut Refine::default(),
            &mut |_| true,
            &mut |class, build| {
                cells.push((class, build()));
                ControlFlow::Continue(())
            },
        );
        assert!(flow.is_continue());
        cells.sort_unstable_by_key(|&(class, _)| class);
        assert_eq!(cells, [(0, a), (1, b)]);
        for (_, cell) in cells {
            alg.dec_ref(cell);
        }

        // Flipping bit zero sends only zero from {0, 1} into {1, 3}.
        let image = alg.image(&map, p, q);
        let preimage = alg.preimage(&map, p, q);
        assert_eq!(image, b);
        assert_eq!(preimage, a);
        for root in [a, b, p, q, f, partition, cases, aligned, image, preimage] {
            alg.dec_ref(root);
        }
    });
}

#[test]
fn wide_singleton_transfers_and_a_three_class_quotient_fit_a_two_mib_stack() {
    let sort = ScalarSort::bv(if cfg!(miri) { 13 } else { 4096 });
    let alg = bind(&roomy(), sort);
    let words = sort.words();
    let mut keys = vec![0; 2 * words];
    keys[words] = 1;
    let (zero, one) = keys.split_at(words);
    let p = alg.set(zero);
    let expected = alg.set(one);
    let map = MapSpec::Bits {
        clear: zero.into(),
        flip: one.into(),
    };
    let mut labels = Vec::new();
    let f = alg.values(2, &keys, &[0, 1], &mut labels, &mut Default::default());
    assert_eq!(labels, [0, 1, 2]);

    // The flip exchanges zero and one. The quotient merges zero with the
    // default class, leaving only the value one in class 1.
    on_two_mib_stack(move || {
        let image = alg.image(&map, p, Function::TRUE);
        assert_eq!(image, expected);
        let preimage = alg.preimage(&map, Function::TRUE, p);
        assert_eq!(preimage, expected);
        let quotient = alg.quotient(f, &[0, 1, 0], &mut Align::default());
        assert_eq!(quotient, expected);
        for root in [p, f, expected, image, preimage, quotient] {
            alg.dec_ref(root);
        }
    });
}
