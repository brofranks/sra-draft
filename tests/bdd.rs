//! Public-contract tests for `sra::bdd`.
//!
//! Tests exercise the public `Column` operations using concrete scalar values
//! and check that equivalent diagrams have the same root. Small store
//! capacities force collection and growth at different construction steps.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread;

use sra::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Refinement, Values, transfer,
};
use sra::bdd::BddStore;
use sra::scalar::ScalarSort;
use sra::util::count::Count;

#[path = "support/bdd.rs"]
mod fixtures;
#[path = "support/listed.rs"]
mod listed;

use fixtures::{
    BV4, BV12, BV18, bit, classes, col, members, residues, roomy, samples, set, shared,
    shared_holds, tiny,
};
use listed::Listed;

/// A partition with the shape of [`shared`]: values in `shared(c, (1, 2))` take
/// one class, values in `shared(c, (3, 4))` another, and every other value a
/// third.
fn diamond(c: &dyn Column) -> Function {
    let (a, b) = (shared(c, (1, 2)), shared(c, (3, 4)));
    let root = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
    c.dec_ref(a);
    c.dec_ref(b);
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

// --- binding ----

#[test]
fn every_scalar_sort_binds_a_column() {
    let store: Arc<dyn Backend> = Arc::new(BddStore::default());
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
fn an_owned_root_survives_its_creating_column() {
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

#[test]
fn evaluation_requires_an_exact_encoding_for_every_root() {
    let c = col(&tiny(), ScalarSort::bit_vec(65));
    let max = [u64::MAX, 1];
    let p = c.set(&max);
    for root in [Function::FALSE, Function::TRUE, p] {
        assert_eq!(c.evaluate(root, &max), u32::from(!root.is_f()));
        for value in [&[0][..], &[0, 0, 0], &[0, 2]] {
            let panic = catch_unwind(AssertUnwindSafe(|| c.evaluate(root, value)))
                .expect_err("evaluation must reject an invalid encoding");
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"value is outside the column sort")
            );
        }
    }
    c.dec_ref(p);
}

// --- counting ----

#[test]
fn a_cut_and_its_complement_cover_each_bound_sort() {
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
    let sizes = [Count::from(16), Count::from(4096), Count::from(1) << 70];
    // Alternate retained bindings, including a multiword sort, to exercise each
    // column's own interpretation within the shared store.
    for i in [0, 2, 1, 0] {
        let c = &*columns[i];
        assert_eq!(c.count(cuts[i]), 7);
        assert_eq!(members(c, cuts[i]), [0, 1, 2, 3, 4, 5, 6]);
        // The complement is a real diagram through the short first level.
        let other = c.apply_bool(BoolOp::AndNot, Function::TRUE, cuts[i]);
        assert_eq!(c.count(other) + Count::from(7), sizes[i]);
        c.dec_ref(other);
    }
    for (c, p) in columns.iter().zip(cuts) {
        c.dec_ref(p);
    }
}

#[test]
fn multiword_cuts_and_skipped_levels_have_exact_counts() {
    let c = col(&tiny(), ScalarSort::bit_vec(70));
    let cut = c.less_than(&[5, 1]);
    assert_eq!(c.count(cut), (Count::from(1) << 64) + Count::from(5));
    for (point, expected) in [
        ([u64::MAX, 0], true),
        ([4, 1], true),
        ([5, 1], false),
        ([0, 2], false),
    ] {
        assert_eq!(c.evaluate(cut, &point) != 0, expected);
    }
    let low = c.masked_eq(&[1, 0], &[1, 0]);
    assert_eq!(c.count(low), Count::from(1) << 69);
    assert_eq!(c.count(Function::TRUE), Count::from(1) << 70);
}

#[test]
fn wide_singletons_and_complements_have_exact_counts() {
    let sort = ScalarSort::bit_vec(if cfg!(miri) { 130 } else { 4096 });
    let c = col(&roomy(), sort);
    let p = c.set(&vec![0; sort.words()]);
    assert_eq!(c.count(p), 1);
    let q = c.apply_bool(BoolOp::AndNot, Function::TRUE, p);
    assert_eq!(c.count(q) + Count::from(1), Count::from(1) << sort.width());
    c.dec_ref(q);
    c.dec_ref(p);
}

#[test]
fn small_domain_counts_match_concrete_sets() {
    for width in [4, 6, 7, if cfg!(miri) { 8 } else { 12 }] {
        let c = col(&roomy(), ScalarSort::bit_vec(width));
        let end = 1_u64 << width;
        assert_eq!(c.count(Function::FALSE), 0);
        assert_eq!(c.count(Function::TRUE), end);
        // Low-only and high-only masks skip levels on opposite sides. Widths
        // four and seven also exercise a partial first level.
        for mask in [0, 1, 1 << (width - 1), end - 1] {
            let value = mask & 0x555;
            let expected: Vec<_> = (0..end).filter(|&x| x & mask == value).collect();
            let p = bit(&*c, mask, value);
            assert_eq!(c.count(p), expected.len() as u64);
            assert_eq!(members(&*c, p), expected);
            c.dec_ref(p);
        }
        let expected: Vec<_> = (0..end).filter(|x| x % 7 == 2).collect();
        let p = c.set(&expected);
        assert_eq!(c.count(p), expected.len() as u64);
        c.dec_ref(p);
    }
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 3);
    c.count(f);
}

// --- sets, cuts, and bit tests ----

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
#[should_panic(expected = "bound is outside the column sort")]
fn a_cut_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).less_than(&[0x10]);
}

#[test]
#[should_panic(expected = "bit test is outside the column sort")]
fn a_bit_test_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).masked_eq(&[0x10], &[0x10]);
}

#[test]
#[should_panic(expected = "bit test is outside the column sort")]
fn a_bit_test_value_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).masked_eq(&[0x1], &[0x11]);
}

#[test]
#[should_panic(expected = "bit test value has bits outside its mask")]
fn a_bit_test_value_outside_its_mask_is_rejected() {
    let _ = col(&roomy(), BV4).masked_eq(&[0x1], &[0x3]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn unsorted_set_values_are_rejected() {
    let _ = col(&roomy(), BV4).set(&[3, 1]);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn repeated_set_values_are_rejected() {
    let _ = col(&roomy(), BV4).set(&[2, 2]);
}

#[test]
#[should_panic(expected = "value is outside the column sort")]
fn a_set_value_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).set(&[1, 0x10]);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn set_values_off_the_word_stride_are_rejected() {
    let _ = col(&roomy(), ScalarSort::bit_vec(70)).set(&[1, 0, 2]);
}

// --- boolean operations ----

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
#[cfg(debug_assertions)]
fn bit_map_shortcuts_check_both_input_roots() {
    let c = col(&tiny(), BV4);
    let released = set(&*c, [1]);
    c.dec_ref(released);
    let map = MapSpec::Bits {
        clear: vec![1].into(),
        flip: vec![0].into(),
    };
    for (p, q) in [(Function::FALSE, released), (released, Function::FALSE)] {
        for image in [false, true] {
            let panic = catch_unwind(AssertUnwindSafe(|| {
                if image {
                    c.image(&map, p, q)
                } else {
                    c.preimage(&map, p, q)
                }
            }))
            .expect_err("bit-map shortcuts must check root liveness");
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"cannot use an unreferenced BDD node")
            );
        }
    }
}

#[test]
fn bit_maps_confined_to_one_level_agree_with_pointwise_maps() {
    let store = tiny();
    let c = col(&store, BV18);
    let points = [0, 5, 0x41, 0x7C0, 0x1003, 0x2FFFF, 0x3F040, 0x3FFFF];
    let p = set(&*c, points);
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
        let image = c.image(&map, p, Function::TRUE);
        assert_eq!(members(&*c, image), expect);
        let in_q: Vec<_> = expect.into_iter().step_by(2).collect();
        let q = c.set(&in_q);
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
        c.dec_ref(f);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn bit_maps_preserve_constraints_on_untouched_levels() {
    let store = roomy();
    let c = col(&store, BV18);
    // The predicate tests levels zero and two. The map clears bit six on level
    // one, so the image and preimage must still satisfy both tests.
    let (hi, low) = (bit(&*c, 1 << 12, 1 << 12), bit(&*c, 1, 1));
    let p = c.apply_bool(BoolOp::And, hi, low);
    let map = MapSpec::Bits {
        clear: vec![1 << 6].into(),
        flip: vec![0].into(),
    };
    let image = c.image(&map, p, Function::TRUE);
    let expect: Vec<u64> = (0..1 << 18)
        .filter(|x| x & (1 << 12) != 0 && x & 1 != 0 && x & (1 << 6) == 0)
        .collect();
    assert_eq!(members(&*c, image), expect);
    let preimage = c.preimage(&map, Function::TRUE, image);
    let expect: Vec<u64> = (0..1 << 18)
        .filter(|x| x & (1 << 12) != 0 && x & 1 != 0)
        .collect();
    assert_eq!(members(&*c, preimage), expect);
    for f in [hi, low, p, image, preimage] {
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
#[cfg_attr(miri, ignore = "too slow under Miri")]
#[expect(
    clippy::type_complexity,
    reason = "the paired spec and its reference implementation read better inline"
)]
fn bit_maps_match_concrete_results_across_small_capacities() {
    // Clearing the last level merges terminals. Clearing the first takes the
    // disjunction of whole subgraphs. Either can exhaust the arena.
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
    let input: Vec<u64> = (0..40).map(|i| i * 101 + 1).collect();
    let allowed: Vec<u64> = (0..40).map(|i| i * 37).chain([3861]).collect();
    for ((map, eval), pair) in maps.iter().zip([(3940, 3861), (1516, 37)]) {
        let pairs: Vec<_> = input
            .iter()
            .map(|&x| (x, eval(x)))
            .filter(|(_, y)| allowed.contains(y))
            .collect();
        assert_eq!(pairs, [pair]);
        for slack in 0..8 {
            // A fresh store gives both directions the same initial capacity.
            for forward in [true, false] {
                let c = small(slack);
                let p = c.set(&input);
                let q = c.set(&allowed);
                let (f, expected) = if forward {
                    (c.image(map, p, q), pair.1)
                } else {
                    (c.preimage(map, p, q), pair.0)
                };
                assert_eq!(
                    members(&*c, f),
                    [expected],
                    "slack {slack}, image {forward}"
                );
                for root in [p, q, f] {
                    c.dec_ref(root);
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
        (ScalarSort::bit_vec(70), &[0], &[0], MASKS),
        (ScalarSort::bit_vec(70), &[0, 64], &[0, 0], MASKS),
        (ScalarSort::bit_vec(70), &[0, 0], &[0, 64], MASKS),
        (ScalarSort::symbol(), &[], &[], MASKS),
        (ScalarSort::symbol(), &[0], &[1], SYMBOLS),
    ];
    // The domain check itself must reject each map, not a later failure.
    let rejects = |result: thread::Result<()>, message: &str| {
        let payload = result.expect_err("a malformed bit map must be rejected");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&message));
    };
    for &(sort, clear, flip, message) in masks {
        let c = col(&roomy(), sort);
        let map = MapSpec::Bits {
            clear: clear.into(),
            flip: flip.into(),
        };
        let zero = vec![0; sort.words()];
        let p = c.set(&zero);
        for predicate in [Function::FALSE, p] {
            for image in [false, true] {
                rejects(
                    catch_unwind(AssertUnwindSafe(|| {
                        let _ = if image {
                            c.image(&map, predicate, Function::TRUE)
                        } else {
                            c.preimage(&map, predicate, Function::TRUE)
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
        c.dec_ref(p);
    }
}

#[test]
fn a_rust_map_on_a_bdd_is_rejected() {
    const MESSAGE: &str = "a BDD store cannot apply a Rust map, which needs an enumerating backend";
    let c = col(&tiny(), BV4);
    let map = MapSpec::Rust(Arc::new(|_, _| true));
    for p in [Function::FALSE, Function::TRUE] {
        for q in [Function::FALSE, Function::TRUE] {
            for image in [false, true] {
                let panic = catch_unwind(AssertUnwindSafe(|| {
                    if image {
                        c.image(&map, p, q)
                    } else {
                        c.preimage(&map, p, q)
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
fn a_later_alignment_cache_hit_returns_its_own_labels() {
    let c = col(&roomy(), BV4);
    let a = c.set(&[1]);
    let b = c.set(&[2]);
    let both = c.set(&[1, 2]);
    let mut scratch = Alignment::default();
    let mut labels = Vec::new();
    let first = c.align(a, b, &mut labels, &mut scratch);
    assert_eq!(labels, [(0, 0), (1, 0), (0, 1)]);
    labels.clear();
    let second = c.align(a, both, &mut labels, &mut scratch);
    assert_eq!(labels, [(0, 0), (1, 1), (0, 1)]);
    let mut cached = vec![(9, 9)];
    let hit = c.align(a, both, &mut cached, &mut scratch);
    assert_eq!(hit, second);
    assert_eq!(cached, [(9, 9), (0, 0), (1, 1), (0, 1)]);
    for f in [a, b, both, first, second, hit] {
        c.dec_ref(f);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn alignment_shares_pairs_and_splits_below_a_terminal_side() {
    let store = roomy();
    let c = col(&store, BV18);
    let d = diamond(&*c);
    let odd = bit(&*c, 1, 1);
    let parity = c.cases(0, &[(odd, 1)], &mut Vec::new());
    let top = bit(&*c, 1 << 17, 1 << 17);
    let half = c.cases(0, &[(top, 1)], &mut Vec::new());
    let mut scratch = Alignment::default();
    // The diamond reaches its level-two pairs by two routes, and below the top
    // level each half of `half` is a terminal beside a branching `parity`.
    let pairs = [
        (d, diamond_case as fn(u64) -> u32, 6),
        (half, |x| (x >> 17) as u32, 4),
    ];
    for (f, left, count) in pairs {
        let mut labels = Vec::new();
        let h = c.align(f, parity, &mut labels, &mut scratch);
        let mut first = Vec::new();
        for x in 0..1 << 18 {
            let pair = (left(x), (x & 1) as u32);
            assert_eq!(labels[c.evaluate(h, &[x]) as usize], pair, "at {x}");
            if !first.contains(&pair) {
                first.push(pair);
            }
        }
        assert_eq!(labels, first);
        // Build the same classes from concrete integer predicates, independently
        // of the aligned diagram and its class numbering.
        let label = |x| left(x) * 2 + (x & 1) as u32;
        let cases: Vec<_> = (1..count)
            .map(|i| (set(&*c, (0..1 << 18).filter(|&x| label(x) == i)), i))
            .collect();
        let expected = c.cases(0, &cases, &mut Vec::new());
        assert_eq!(h, expected);
        for (p, _) in cases {
            c.dec_ref(p);
        }
        c.dec_ref(expected);
        c.dec_ref(h);
    }
    for f in [d, odd, parity, top, half] {
        c.dec_ref(f);
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
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn a_quotient_merges_two_diamond_classes() {
    let store = roomy();
    let c = col(&store, BV18);
    let d = diamond(&*c);
    // Inputs 0, 1, and 3 first reach the three classes in that order.
    // Merge the two shared cases and keep the rest apart.
    let q = c.quotient(d, &[0, 1, 1], &mut Alignment::default());
    for x in samples() {
        let class = diamond_case(x);
        assert_eq!(c.evaluate(d, &[x]), class, "at {x}");
        assert_eq!(c.evaluate(q, &[x]), u32::from(class != 0), "at {x}");
    }
    // Construct the union from its concrete meaning rather than the quotient's
    // source classes.
    let p = set(&*c, (0..1 << 18).filter(|&x| diamond_case(x) != 0));
    let expected = c.cases(0, &[(p, 1)], &mut Vec::new());
    assert_eq!(q, expected);
    c.dec_ref(expected);
    c.dec_ref(p);
    c.dec_ref(q);
    c.dec_ref(d);
}

// --- cases ----

#[test]
fn cases_over_one_predicate_reuse_it_or_its_complement() {
    let store = roomy();
    let c = col(&store, BV4);
    let p = set(&*c, [1, 3, 5]);
    let mut labels = Vec::new();
    // A case labeled like the default, or over a constant, is the one-class
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
    // classes canonically.
    let reused = c.cases(0, &[(p, 1)], &mut labels);
    assert_eq!(reused, p);
    c.dec_ref(reused);
    assert_eq!(labels, [0, 1]);
    labels.clear();
    // One accepting zero does not, and is complemented instead.
    let q = set(&*c, [0, 2]);
    let f = c.cases(5, &[(q, 6)], &mut labels);
    assert_eq!(labels, [6, 5]);
    assert_eq!(
        classes(&*c, f),
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    for root in [p, q, f] {
        c.dec_ref(root);
    }
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn overlapping_cases_with_different_labels_are_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

#[test]
fn disjoint_default_labeled_cases_change_nothing() {
    let store = roomy();
    let c = col(&store, BV4);
    let (a, b) = (set(&*c, [1, 2]), set(&*c, [3, 4]));
    for cases in [[(a, 0), (b, 1)], [(b, 1), (a, 0)]] {
        let mut labels = Vec::new();
        let f = c.cases(0, &cases, &mut labels);
        assert_eq!(labels, [0, 1]);
        assert_eq!(f, b);
        c.dec_ref(f);
    }
    c.dec_ref(a);
    c.dec_ref(b);
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn a_default_labeled_case_overlapping_a_later_case_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(a, 0), (b, 1)], &mut Vec::new());
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn a_default_labeled_case_overlapping_an_earlier_case_is_rejected() {
    let store = roomy();
    let c = col(&store, BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(b, 1), (a, 0)], &mut Vec::new());
}

// --- values ----

#[test]
fn values_append_only_reachable_labels_after_the_prefix() {
    let store = roomy();
    let c = col(&store, ScalarSort::bool());
    // Both keys are listed, so the default is unreachable and not appended.
    let mut labels = vec![77];
    let f = c.values(9, &[0, 1], &[10, 20], &mut labels, &mut Values);
    assert_eq!(labels, [77, 10, 20]);
    assert_eq!([c.evaluate(f, &[0]), c.evaluate(f, &[1])], [0, 1]);
    c.dec_ref(f);
}

#[test]
fn wide_keys_take_one_label_per_record() {
    let c = col(&roomy(), ScalarSort::bit_vec(65));
    let mut labels = vec![77];
    let f = c.values(9, &[1, 0, 0, 1, 1, 1], &[8, 6, 8], &mut labels, &mut Values);
    assert_eq!(labels, [77, 9, 8, 6]);
    for (x, class) in [
        ([0, 0], 0),
        ([1, 0], 1),
        ([0, 1], 2),
        ([1, 1], 1),
        ([2, 1], 0),
    ] {
        assert_eq!(c.evaluate(f, &x), class, "at {x:?}");
    }
    c.dec_ref(f);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn value_keys_off_the_word_stride_are_rejected() {
    let c = col(&roomy(), ScalarSort::bit_vec(70));
    let _ = c.values(0, &[1, 0, 2], &[1], &mut Vec::new(), &mut Values);
}

#[test]
#[should_panic(expected = "value is outside the column sort")]
fn a_value_key_outside_the_sort_is_rejected() {
    let _ = col(&roomy(), BV4).values(0, &[1, 0x10], &[1, 2], &mut Vec::new(), &mut Values);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn unsorted_value_keys_are_rejected() {
    let _ = col(&roomy(), BV4).values(0, &[2, 1], &[1, 2], &mut Vec::new(), &mut Values);
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn repeated_value_keys_are_rejected() {
    let _ = col(&roomy(), BV4).values(0, &[1, 1], &[1, 2], &mut Vec::new(), &mut Values);
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn values_with_too_few_labels_are_rejected() {
    let _ = col(&roomy(), BV4).values(0, &[1, 2], &[1], &mut Vec::new(), &mut Values);
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn empty_values_with_a_case_label_are_rejected() {
    let _ = col(&tiny(), BV4).values(42, &[], &[7], &mut Vec::new(), &mut Values);
}

// --- refinement ----

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn refinement_cells_match_the_diamond_intersections() {
    let store = roomy();
    let c = col(&store, BV18);
    let d = diamond(&*c);
    let p = bit(&*c, 0x38, 0);
    let mut seen = Vec::new();
    let _ = c.refine(
        d,
        p,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |class, build| {
            let cell = build();
            seen.push((class, members(&*c, cell)));
            c.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert_eq!(seen.len(), 3);
    for (class, cell) in seen {
        let expect: Vec<u64> = (0..1 << 18)
            .filter(|&x| x & 0x38 == 0 && c.evaluate(d, &[x]) == class)
            .collect();
        assert_eq!(cell, expect, "class {class}");
    }
    c.dec_ref(d);
    c.dec_ref(p);
}

#[test]
fn a_shallow_partition_selects_each_class_once_under_a_deep_predicate() {
    let store = tiny();
    let c = col(&store, BV18);
    let half = 1 << 17;
    let top = bit(&*c, half, half);
    let f = c.cases(0, &[(top, 1)], &mut Vec::new());
    // The multiples of three bordering the split reach both classes.
    let mut points: Vec<u64> = samples()
        .filter(|x| x % 3 == 0)
        .chain([half - 2, half + 1])
        .collect();
    points.sort_unstable();
    points.dedup();
    let p = set(&*c, points.iter().copied());
    let (mut selected, mut cells) = (Vec::new(), Vec::new());
    let result = c.refine(
        f,
        p,
        &mut Refinement::default(),
        &mut |class| {
            selected.push(class);
            true
        },
        &mut |class, build| {
            let cell = build();
            cells.push((class, members(&*c, cell)));
            c.dec_ref(cell);
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
        c.dec_ref(g);
    }
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
fn empty_refinement_of_an_unreferenced_partition_is_rejected() {
    let c = col(&roomy(), BV4);
    let f = residues(&*c, 3);
    c.dec_ref(f);
    let _ = c.refine(
        f,
        Function::FALSE,
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
fn an_empty_wide_cursor_and_transfer_finish_without_visiting_a_value() {
    let sort = ScalarSort::bit_vec(65_536);
    let c = col(&tiny(), sort);
    let mut cursor = c.members(Function::FALSE);
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
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.extend_from_slice(v);
    }
    assert_eq!(out, sorted);
}
