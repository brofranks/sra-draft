//! Public-contract tests for the sparse backend.
//!
//! Concrete points and ranges provide expected values. Other assertions
//! check canonical roots and consistency between operations. The fixtures
//! cover packed, fixed two-word, and runtime-width keys.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sra::backend::{Algebra, Align, Backend, BoolOp, Function, MapSpec, Refine, Values};
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;
use sra::util::count::Count;

#[path = "support/sparse.rs"]
mod fixtures;

use fixtures::{BV4, BV6, bind, holds, members, parts, set};

/// The widest packed sort.
const BV32: ScalarSort = ScalarSort::bv(32);
/// The two wide layouts: a statically known pair of key words, and a run-time
/// key width.
const WIDE: [ScalarSort; 2] = [ScalarSort::bv(100), ScalarSort::bv(200)];

/// A store with room for ordinary test fixtures.
fn roomy() -> Arc<SparseStore> {
    Arc::new(SparseStore::with_capacity(128, 512, 1))
}

/// Every member of a predicate, one row of the sort's words each.
fn rows(alg: &dyn Algebra, p: Function) -> Vec<Vec<u64>> {
    let mut cursor = alg.members(p);
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.push(v.to_vec());
    }
    out
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

/// The classes a refinement reaches, with the first word of each member of the
/// delivered cell.
fn cells(
    alg: &dyn Algebra,
    f: Function,
    p: Function,
    scratch: &mut Refine,
    mut select: impl FnMut(u32) -> bool,
) -> Vec<(u32, Vec<u64>)> {
    let mut out = Vec::new();
    let result = alg.refine(f, p, scratch, &mut select, &mut |class, build| {
        let cell = build();
        out.push((class, members(alg, cell)));
        alg.dec_ref(cell);
        ControlFlow::Continue(())
    });
    assert_eq!(result, ControlFlow::Continue(()));
    out.sort();
    out
}

// --- bindings and roots ------------------------------------------------------

#[test]
fn every_scalar_sort_binds_to_the_store() {
    let store: Arc<dyn Backend> = roomy();
    for sort in [
        ScalarSort::bool(),
        ScalarSort::bv(1),
        BV4,
        BV32,
        ScalarSort::bv(40),
        ScalarSort::bv(70),
        ScalarSort::bv(200),
        ScalarSort::symbol(),
    ] {
        let alg = Arc::clone(&store).bind(sort).unwrap();
        assert_eq!(alg.sort(), sort);
    }
}

#[test]
fn repeated_bindings_share_live_roots() {
    let store = roomy();
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
fn a_binding_retains_the_store_and_roots_from_another_binding() {
    let store = roomy();
    let sort = ScalarSort::bv(12);
    let a = bind(&store, sort);
    let b = bind(&store, sort);
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
#[should_panic(expected = "value is outside the scalar sort")]
fn a_constant_evaluation_with_the_wrong_word_count_is_rejected() {
    let _ = bind(&roomy(), BV4).evaluate(Function::TRUE, &[0, 0]);
}

#[test]
#[should_panic(expected = "bound is outside the scalar sort")]
fn a_cut_with_the_wrong_word_count_is_rejected() {
    let _ = bind(&roomy(), BV4).lt(&[0, 0]);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn set_values_off_the_word_stride_are_rejected() {
    let _ = bind(&roomy(), ScalarSort::bv(65)).set(&[1]);
}

#[test]
#[should_panic(expected = "value is outside the scalar sort")]
fn a_set_key_outside_the_sort_is_rejected() {
    let _ = bind(&roomy(), BV4).set(&[16]);
}

#[test]
fn counts_reach_the_universe_boundary_at_every_layout() {
    // One-word sorts up to the full 64 bits, then the static and run-time wide
    // layouts, including a width whose universe count needs a word above the
    // key words.
    let sorts = [4, 32, 33, 64, 100, 128, 200].map(ScalarSort::bv);
    for d in sorts {
        let alg = bind(&roomy(), d);
        let mut digits = vec![0; (d.width() / u64::BITS + 1) as usize];
        *digits
            .last_mut()
            .expect("a universe needs at least one digit") = 1 << (d.width() % u64::BITS);
        let universe = Count::from_digits(digits);
        let without = |missing: u64| {
            let mut digits = vec![u64::MAX; d.words()];
            let high = d.width() % u64::BITS;
            if high != 0 {
                *digits.last_mut().unwrap() = (1_u64 << high) - 1;
            }
            digits[0] -= missing - 1;
            Count::from_digits(digits)
        };
        let lt = alg.lt(&word(d, 10));
        let ge = alg.apply(BoolOp::AndNot, Function::TRUE, lt);
        let points = alg.set(&keys(d, [1, 2, 3, 7]));
        let p = alg.apply(BoolOp::Or, points, ge);
        assert_eq!(alg.count(p), without(6), "{d:?}");
        let excluded = alg.set(&keys(d, [0, 3]));
        let holes = alg.apply(BoolOp::AndNot, Function::TRUE, excluded);
        assert_eq!(alg.count(holes), without(2), "{d:?}");
        assert_eq!(alg.count(Function::TRUE), universe, "{d:?}");
        assert_eq!(alg.count(Function::FALSE), Count::ZERO, "{d:?}");
        // A predicate holding at zero opens its first run there.
        let point = alg.set(&keys(d, [12]));
        let low = alg.apply(BoolOp::Or, lt, point);
        assert_eq!(alg.count(low), Count::from(11), "{d:?}");
        for root in [lt, ge, points, p, excluded, holes, point, low] {
            alg.dec_ref(root);
        }
    }
}

#[test]
fn an_empty_count_stays_small_for_a_wide_universe() {
    let alg = bind(&roomy(), ScalarSort::bv(1 << 30));
    assert_eq!(alg.count(Function::FALSE), Count::ZERO);
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_with_more_than_two_classes_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let cases = [(set(&*alg, [1]), 2), (set(&*alg, [5]), 3)];
    let f = alg.cases(0, &cases, &mut Vec::new());
    alg.count(f);
}

#[test]
#[should_panic(expected = "symbols have no numeric ordering")]
fn a_symbol_cut_is_rejected() {
    let _ = bind(&roomy(), ScalarSort::symbol()).lt(&[3]);
}

#[test]
#[should_panic(expected = "this backend cannot construct masked bit predicates")]
fn a_bit_test_is_rejected() {
    let _ = bind(&roomy(), BV4).masked_eq(&[1], &[1]);
}

// --- Boolean operations ------------------------------------------------------

#[test]
fn boolean_operations_merge_runs_of_both_operands() {
    let store = roomy();
    let alg = bind(&store, BV6);
    let (a, b) = (set(&*alg, (0..16).map(|v| v * 3)), set(&*alg, 10..28));
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        let r = alg.apply(op, a, b);
        let expect: Vec<u64> = (0_u64..64)
            .filter(|&x| {
                let left = x.is_multiple_of(3) && x < 48;
                let right = (10..28).contains(&x);
                match op {
                    BoolOp::And => left && right,
                    BoolOp::Or => left || right,
                    BoolOp::Xor => left ^ right,
                    BoolOp::AndNot => left && !right,
                }
            })
            .collect();
        assert_eq!(members(&*alg, r), expect, "{op:?}");
        alg.dec_ref(r);
    }
    // Equal operands and constants never reach the merge at all.
    assert_eq!(alg.apply(BoolOp::Xor, a, a), Function::FALSE);
    let r = alg.apply(BoolOp::And, a, Function::TRUE);
    assert_eq!(r, a);
    for f in [a, b, r] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg(debug_assertions)]
fn either_unreferenced_boolean_operand_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let p = alg.set(&[1]);
    alg.dec_ref(p);
    for (left, right) in [(p, Function::FALSE), (Function::FALSE, p)] {
        let panic =
            catch_unwind(AssertUnwindSafe(|| alg.apply(BoolOp::And, left, right))).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"cannot use an unreferenced sparse node")
        );
    }
}

// --- images and preimages ----------------------------------------------------

#[test]
fn maps_transfer_the_members_of_a_predicate() {
    let alg = bind(&roomy(), ScalarSort::bv(12));
    let in_p = [3, 65, 2114, 4095];
    let in_q = [0, 21, 2070, 4051];
    let (p, q) = (alg.set(&in_p), alg.set(&in_q));
    let read = |f| {
        let out = members(&*alg, f);
        alg.dec_ref(f);
        out
    };
    // `x - 44`, undefined below it, so the low member drops out of both
    // directions.
    let map = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = x[0].saturating_sub(44);
        x[0] >= 44
    }));
    assert_eq!(read(alg.image(&map, p, q)), [21, 2070, 4051]);
    assert_eq!(read(alg.preimage(&map, p, q)), [65, 2114, 4095]);
    assert_eq!(
        read(alg.preimage(&map, p, Function::TRUE)),
        [65, 2114, 4095]
    );
    // A bit map is enumerated the same way.
    let bits = MapSpec::Bits {
        clear: vec![0x3F].into(),
        flip: vec![0x55].into(),
    };
    let moved = |x: u64| (x & !0x3F) ^ 0x55;
    let mut expect: Vec<u64> = in_p
        .iter()
        .map(|&x| moved(x))
        .filter(|y| in_q.contains(y))
        .collect();
    expect.sort_unstable();
    assert_eq!(read(alg.image(&bits, p, q)), expect);
    let expect: Vec<u64> = in_p
        .iter()
        .copied()
        .filter(|&x| in_q.contains(&moved(x)))
        .collect();
    assert_eq!(read(alg.preimage(&bits, p, q)), expect);
    // The identity needs no enumeration in either direction: it is a
    // conjunction.
    let id = MapSpec::identity(1);
    assert_eq!(alg.image(&id, p, q), Function::FALSE);
    let r = alg.preimage(&id, p, p);
    assert_eq!(r, p);
    for f in [p, q, r] {
        alg.dec_ref(f);
    }
}

#[test]
fn bit_maps_respect_their_full_endpoints() {
    let alg = bind(&roomy(), BV4);
    let p = alg.set(&[1, 4]);
    let map = MapSpec::Bits {
        clear: vec![0].into(),
        flip: vec![1].into(),
    };
    let image = alg.image(&map, p, Function::TRUE);
    assert_eq!(members(&*alg, image), [0, 5]);
    alg.dec_ref(image);
    let pre = alg.preimage(&map, p, Function::TRUE);
    let full = alg.image(&map, Function::TRUE, p);
    assert_eq!(pre, p);
    assert_eq!(full, p);
    // Clearing a bit means the image can reach only part of the target.
    let clear = MapSpec::Bits {
        clear: Box::new([1]),
        flip: Box::new([0]),
    };
    let restricted = alg.image(&clear, Function::TRUE, p);
    assert_eq!(members(&*alg, restricted), [4]);
    alg.dec_ref(restricted);
    alg.dec_ref(p);
    assert_eq!(members(&*alg, pre), [1, 4]);
    alg.dec_ref(pre);
    assert_eq!(members(&*alg, full), [1, 4]);
    alg.dec_ref(full);
}

#[test]
fn a_constant_bit_map_tests_its_only_output() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let p = alg.set(&[1, 0, 3, 1]);
    let q = alg.set(&[5, 1, 6, 1]);
    let map = MapSpec::Bits {
        clear: Box::new([u64::MAX, 1]),
        flip: Box::new([5, 1]),
    };
    let image = alg.image(&map, p, q);
    let pre = alg.preimage(&map, p, q);
    assert_eq!(pre, p);
    assert_eq!(alg.image(&map, p, p), Function::FALSE);
    assert_eq!(alg.preimage(&map, p, p), Function::FALSE);
    assert_eq!(alg.image(&map, Function::FALSE, q), Function::FALSE);

    // Keeping bits in either word makes the output depend on the input.
    let partial = [([u64::MAX, 0], [5, 1]), ([0, 1], [4, 1])].map(|(clear, flip)| {
        let bits = MapSpec::Bits {
            clear: Box::new(clear),
            flip: Box::new(flip),
        };
        (alg.image(&bits, p, q), alg.preimage(&bits, p, q))
    });
    alg.dec_ref(p);
    alg.dec_ref(q);
    assert_eq!(rows(&*alg, image), [[5, 1]]);
    assert_eq!(rows(&*alg, pre), [[1, 0], [3, 1]]);
    for (r, s) in partial {
        assert_eq!(rows(&*alg, r), [[5, 1]]);
        assert_eq!(rows(&*alg, s), [[1, 0]]);
        alg.dec_ref(r);
        alg.dec_ref(s);
    }
    alg.dec_ref(image);
    alg.dec_ref(pre);
}

#[test]
#[should_panic(expected = "bit map masks must encode the scalar sort")]
fn a_full_target_preimage_with_an_invalid_mask_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let map = MapSpec::Bits {
        clear: vec![0].into(),
        flip: vec![16].into(),
    };
    let _ = alg.preimage(&map, Function::TRUE, Function::TRUE);
}

#[test]
#[cfg(debug_assertions)]
fn transfers_check_skipped_roots_when_the_map_yields_no_image() {
    let alg = bind(&roomy(), BV4);
    let p = alg.set(&[1]);
    let q = alg.set(&[2]);
    alg.dec_ref(q);
    let undefined = MapSpec::Rust(Arc::new(|_, _| false));
    let constant = MapSpec::Bits {
        clear: Box::new([15]),
        flip: Box::new([2]),
    };
    for (map, source, target) in [(undefined, p, q), (constant, q, p)] {
        for image in [true, false] {
            let panic = catch_unwind(AssertUnwindSafe(|| {
                if image {
                    alg.image(&map, source, target)
                } else {
                    alg.preimage(&map, source, target)
                }
            }))
            .unwrap_err();
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"cannot use an unreferenced sparse node")
            );
        }
    }
    alg.dec_ref(p);
}

#[test]
fn empty_transfer_targets_skip_source_enumeration_after_domain_validation() {
    let alg = bind(&roomy(), ScalarSort::bv(8));
    let calls = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&calls);
    let map = MapSpec::Rust(Arc::new(move |x, y| {
        count.fetch_add(1, Ordering::Relaxed);
        y.copy_from_slice(x);
        true
    }));
    assert_eq!(
        alg.image(&map, Function::TRUE, Function::FALSE),
        Function::FALSE
    );
    assert_eq!(
        alg.preimage(&map, Function::TRUE, Function::FALSE),
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
                    alg.image(&invalid, Function::TRUE, Function::FALSE)
                } else {
                    alg.preimage(&invalid, Function::TRUE, Function::FALSE)
                }
            }))
            .is_err()
        );
    }
}

// --- conditionals ------------------------------------------------------------

#[test]
fn conditionals_advance_every_operand_over_shared_keys() {
    let store = roomy();
    let alg = bind(&store, BV6);
    // The branches share one transition point with the condition and one with
    // each other, so the sweep must advance several runs at the same key.
    let k = set(&*alg, 10..20);
    let y = set(&*alg, (10..30).step_by(2));
    let n = alg.set(&[0, 5, 15, 25]);
    let r = alg.ite(k, y, n);
    let expect: Vec<u64> = (0..64)
        .filter(|&x| {
            if (10..20).contains(&x) {
                x % 2 == 0
            } else {
                [0, 5, 15, 25].contains(&x)
            }
        })
        .collect();
    assert_eq!(members(&*alg, r), expect);
    // A constant condition selects one branch outright, retaining its existing
    // root.
    let yes = alg.ite(Function::TRUE, y, n);
    let no = alg.ite(Function::FALSE, y, n);
    assert_eq!(yes, y);
    assert_eq!(no, n);
    // Choosing the condition and its complement in opposite arms gives
    // a full or empty result, depending on their order.
    let not = alg.apply(BoolOp::AndNot, Function::TRUE, k);
    assert_eq!(alg.ite(k, k, not), Function::TRUE);
    assert_eq!(alg.ite(k, not, k), Function::FALSE);
    for f in [k, y, n, r, yes, no, not] {
        alg.dec_ref(f);
    }
}

#[test]
fn a_full_true_branch_keeps_members_of_the_else_branch() {
    let alg = bind(&roomy(), BV4);
    let (k, no) = (alg.set(&[1]), alg.set(&[2]));
    let f = alg.ite(k, Function::TRUE, no);
    assert_eq!(members(&*alg, f), [1, 2]);
    for f in [k, no, f] {
        alg.dec_ref(f);
    }
}

// --- alignment and quotient --------------------------------------------------

#[test]
fn alignment_and_quotient_renumber_by_first_occurrence() {
    let store = roomy();
    let alg = bind(&store, BV6);
    let (f, _) = parts(&*alg, 0, [1, 2, 1, 3]);
    let g = set(&*alg, [20, 30]);
    let mut labels = vec![(9, 9)];
    let mut scratch = Align::default();
    let h = alg.align(f, g, &mut labels, &mut scratch);
    // Class numbers index the appended suffix after the existing prefix.
    assert_eq!(labels, [(9, 9), (0, 0), (1, 0), (2, 1), (1, 1), (3, 0)]);
    for x in 0..64 {
        let class = alg.evaluate(h, &[x]) as usize;
        let expect = match x {
            10 => (1, 0),
            20 => (2, 1),
            30 => (1, 1),
            40 => (3, 0),
            _ => (0, 0),
        };
        assert_eq!(labels[1..][class], expect, "at {x}");
    }
    // Coarsening merges the two singleton cells, and the runs that become equal
    // coalesce.
    let q = alg.quotient(f, &[0, 1, 1, 2], &mut scratch);
    for x in 0..64 {
        let expect = match x {
            10 | 20 | 30 => 1,
            40 => 2,
            _ => 0,
        };
        assert_eq!(alg.evaluate(q, &[x]), expect, "at {x}");
    }
    for f in [f, g, h, q] {
        alg.dec_ref(f);
    }
}

#[test]
#[cfg(debug_assertions)]
fn alignment_rejects_invalid_partitions() {
    let alg = bind(&roomy(), BV4);
    let (f, g) = (alg.set(&[1]), alg.set(&[2]));
    alg.dec_ref(g);
    let mut labels = Vec::new();
    let mut scratch = Align::default();
    let trivial = "cannot align a trivial partition";
    let released = "cannot use an unreferenced sparse node";
    for (left, right, expect) in [
        (f, f, "cannot align a partition with itself"),
        (Function::FALSE, f, trivial),
        (f, Function::FALSE, trivial),
        (f, g, released),
        (g, f, released),
    ] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            alg.align(left, right, &mut labels, &mut scratch)
        }))
        .expect_err("an invalid alignment must panic");
        assert_eq!(panic.downcast_ref::<&str>(), Some(&expect));
    }
    alg.dec_ref(f);
}

// --- cases -------------------------------------------------------------------

#[test]
fn cases_fold_onto_labels_over_their_boundaries() {
    let store = roomy();
    let alg = bind(&store, BV6);
    let (a, b) = (set(&*alg, 10..20), set(&*alg, 15..30));
    let mut labels = Vec::new();
    // Overlapping cases agree on their label, so the first one ending leaves the
    // cell covered. The gap above them falls back to the default.
    let f = alg.cases(9, &[(a, 4), (b, 4)], &mut labels);
    assert_eq!(labels, [9, 4]);
    for x in 0..64 {
        assert_eq!(
            alg.evaluate(f, &[x]),
            u32::from((10..30).contains(&x)),
            "at {x}"
        );
    }
    // A single case labeled like the default folds into the default cell, so the
    // result is the constant.
    let mut labels = Vec::new();
    assert_eq!(alg.cases(4, &[(a, 4)], &mut labels), Function::FALSE);
    assert_eq!(labels, [4]);
    for f in [a, b, f] {
        alg.dec_ref(f);
    }
}

#[test]
fn a_rejected_overlap_leaves_the_labels_unchanged() {
    let alg = bind(&roomy(), BV4);
    let (low, wide) = (alg.lt(&[2]), alg.lt(&[4]));
    let (a, b, c) = (set(&*alg, [1, 2]), set(&*alg, [4, 5]), set(&*alg, [5, 6]));
    let mut labels = vec![77];
    // The first overlap is found at zero, before any label is appended. The
    // second is found after the default and two case labels were appended.
    for cases in [&[(low, 1), (wide, 2)][..], &[(a, 1), (b, 2), (c, 3)]] {
        let panic =
            catch_unwind(AssertUnwindSafe(|| alg.cases(0, cases, &mut labels))).unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"cases with different labels overlap")
        );
        assert_eq!(labels, [77]);
    }
    for f in [low, wide, a, b, c] {
        alg.dec_ref(f);
    }
}

#[test]
fn a_boolean_case_predicate_is_already_its_own_partition() {
    let alg = bind(&roomy(), BV6);
    let p = set(&*alg, 10..20);
    let mut labels = Vec::new();
    // Zero is outside the case, so the predicate's own payloads are the
    // canonical classes.
    let f = alg.cases(8, &[(p, 3)], &mut labels);
    assert_eq!(f, p);
    assert_eq!(labels, [8, 3]);
    alg.dec_ref(p);
    alg.dec_ref(f);
    // A case holding at zero is rebuilt so its initial class is zero.
    let q = alg.lt(&[20]);
    labels.clear();
    let f = alg.cases(8, &[(q, 3)], &mut labels);
    assert_eq!(labels, [3, 8]);
    assert_eq!(
        (0..64)
            .map(|x| alg.evaluate(f, &[x]))
            .filter(|&class| class == 0)
            .count(),
        20
    );
    alg.dec_ref(q);
    alg.dec_ref(f);
    // Neither is a constant case, which has no transition to read at all.
    labels.clear();
    assert_eq!(
        alg.cases(8, &[(Function::TRUE, 3)], &mut labels),
        Function::FALSE
    );
    assert_eq!(labels, [3]);
    // A false case leaves the default cell alone, after any existing labels.
    assert_eq!(
        alg.cases(8, &[(Function::FALSE, 3)], &mut labels),
        Function::FALSE
    );
    assert_eq!(labels, [3, 8]);
}

// --- values ------------------------------------------------------------------

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn value_keys_off_the_word_stride_are_rejected() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let _ = alg.values(0, &[1, 0, 2], &[7], &mut Vec::new(), &mut Values::default());
}

#[test]
#[should_panic(expected = "value is outside the scalar sort")]
fn a_value_key_outside_the_sort_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let _ = alg.values(0, &[16], &[1], &mut Vec::new(), &mut Values::default());
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn values_with_too_few_labels_are_rejected() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let _ = alg.values(
        0,
        &[1, 0, 0, 1],
        &[7],
        &mut Vec::new(),
        &mut Values::default(),
    );
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn empty_values_with_a_case_label_are_rejected() {
    let alg = bind(&roomy(), BV4);
    let _ = alg.values(0, &[], &[7], &mut Vec::new(), &mut Values::default());
}

#[test]
fn values_reuse_buffers_and_number_classes_from_the_label_suffix() {
    let alg = bind(&roomy(), BV4);
    let mut scratch = Values::default();
    let mut first = Vec::new();
    let f = alg.values(9, &[1], &[3], &mut first, &mut scratch);
    assert_eq!(first, [9, 3]);
    alg.dec_ref(f);

    let mut labels = vec![77];
    assert_eq!(
        alg.values(8, &[], &[], &mut labels, &mut scratch),
        Function::FALSE
    );
    assert_eq!(labels, [77, 8]);

    let f = alg.values(9, &[1, 3], &[4, 3], &mut labels, &mut scratch);
    assert_eq!(labels, [77, 8, 9, 4, 3]);
    assert_eq!([0, 1, 3].map(|x| alg.evaluate(f, &[x])), [0, 1, 2]);
    alg.dec_ref(f);

    // Keys covering the whole universe with one label leave one class and an
    // unreachable default.
    let alg = bind(&roomy(), ScalarSort::bool());
    let mut labels = Vec::new();
    let f = alg.values(9, &[0, 1], &[5, 5], &mut labels, &mut scratch);
    assert_eq!(f, Function::FALSE);
    assert_eq!(labels, [5]);
}

#[test]
fn values_release_oversized_class_storage_after_a_small_build() {
    let alg = bind(&roomy(), ScalarSort::bv(8));
    let mut scratch = Values::default();
    let keys: Vec<_> = (1_u64..=128).collect();
    let case_labels: Vec<_> = (1_u32..=128).collect();
    let mut labels = Vec::new();
    let f = alg.values(0, &keys, &case_labels, &mut labels, &mut scratch);
    alg.dec_ref(f);
    assert_eq!(labels.len(), 129);
    let large = scratch.classes.capacity();

    // The first small build leaves a small entry count for the next reset.
    for label in [7, 8] {
        labels.clear();
        let f = alg.values(0, &[1], &[label], &mut labels, &mut scratch);
        assert_eq!(labels, [0, label]);
        assert_eq!([0, 1, 2].map(|x| alg.evaluate(f, &[x])), [0, 1, 0]);
        alg.dec_ref(f);
    }
    assert!(
        scratch.classes.capacity() < large,
        "small builds must release oversized class storage"
    );
}

#[test]
fn adjacent_point_labels_share_a_boundary_across_words() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let keys = [[u64::MAX, 0], [0, 1], [u64::MAX, 1]];
    let mut labels = Vec::new();
    let f = alg.values(
        9,
        keys.as_flattened(),
        &[7, 8, 7],
        &mut labels,
        &mut Values::default(),
    );

    let a = alg.set(&[u64::MAX, 0, u64::MAX, 1]);
    let b = alg.set(&keys[1]);
    let mut expected = Vec::new();
    let g = alg.cases(9, &[(a, 7), (b, 8)], &mut expected);
    assert_eq!(labels, [9, 7, 8]);
    assert_eq!(labels, expected);
    assert_eq!(f, g);
    for root in [f, g, a, b] {
        alg.dec_ref(root);
    }
}

// --- refinement --------------------------------------------------------------

#[test]
fn refinement_groups_the_accepted_runs_by_class() {
    let store = roomy();
    let alg = bind(&store, BV6);
    let (f, _) = parts(&*alg, 0, [1, 2, 1, 3]);
    let p = set(&*alg, [10, 20, 25, 30]);
    let mut scratch = Refine::default();
    assert_eq!(
        cells(&*alg, f, p, &mut scratch, |_| true),
        [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]
    );
    // Declining a class leaves the other cells unchanged.
    assert_eq!(
        cells(&*alg, f, p, &mut scratch, |class| class == 1),
        [(1, vec![10, 30])]
    );
    // One reached class means the predicate itself is that cell, whatever its
    // shape.
    let one = alg.set(&[10, 30]);
    assert_eq!(
        cells(&*alg, f, one, &mut scratch, |_| true),
        [(1, vec![10, 30])]
    );
    // A false predicate reaches no class.
    assert_eq!(
        cells(&*alg, f, Function::FALSE, &mut scratch, |_| {
            panic!("no class is reached")
        }),
        []
    );
    // A wide interval spans whole cells and the gaps between them.
    let window = set(&*alg, 10..31);
    assert_eq!(
        cells(&*alg, f, window, &mut scratch, |class| class != 0),
        [(1, vec![10, 30]), (2, vec![20])]
    );
    // A visitor that breaks stops the walk immediately.
    let mut count = 0;
    let stopped = alg.refine(f, p, &mut scratch, &mut |_| true, &mut |_, _| {
        count += 1;
        ControlFlow::Break(())
    });
    assert!(stopped.is_break() && count == 1);
    for f in [f, p, one, window] {
        alg.dec_ref(f);
    }
}

#[test]
fn sibling_refinements_reuse_scratch_across_zero_and_the_maximum() {
    let alg = bind(&roomy(), BV4);
    let f = alg.values(0, &[1, 3], &[1, 2], &mut Vec::new(), &mut Values::default());
    let p = alg.set(&[0, 1, 3, 15]);
    let q = alg.set(&[3, 5]);
    let mut scratch = Refine::default();
    assert_eq!(
        cells(&*alg, f, p, &mut scratch, |_| true),
        [(0, vec![0, 15]), (1, vec![1]), (2, vec![3])]
    );
    assert_eq!(
        cells(&*alg, f, q, &mut scratch, |_| true),
        [(0, vec![5]), (2, vec![3])]
    );
    // A single point is its own cell, at zero and at the maximum as well.
    for (x, class) in [(0, 0), (3, 2), (15, 0)] {
        let point = alg.set(&[x]);
        assert_eq!(
            cells(&*alg, f, point, &mut scratch, |_| true),
            [(class, vec![x])]
        );
        alg.dec_ref(point);
    }
    for f in [f, p, q] {
        alg.dec_ref(f);
    }
}

#[test]
fn refinement_of_a_predicate_uses_its_boolean_classes() {
    let store = roomy();
    let alg = bind(&store, BV6);
    // The predicate indicates class one, and its complement indicates class
    // zero.
    let included = [10, 20, 30];
    let f = set(&*alg, included);
    let expect = [
        (0, (0..64).filter(|x| !included.contains(x)).collect()),
        (1, included.to_vec()),
    ];
    let mut scratch = Refine::default();
    assert_eq!(
        cells(&*alg, f, Function::TRUE, &mut scratch, |_| true),
        expect
    );
    assert_eq!(
        cells(&*alg, f, Function::TRUE, &mut scratch, |class| class == 0),
        expect[..1]
    );
    // A predicate holding at zero indicates class one from the start.
    let low = alg.lt(&[5]);
    assert_eq!(
        cells(&*alg, low, Function::TRUE, &mut scratch, |_| true),
        [(0, (5..64).collect()), (1, (0..5).collect())]
    );
    assert_eq!(
        cells(&*alg, Function::TRUE, Function::TRUE, &mut scratch, |_| {
            true
        }),
        [(1, (0..64).collect())]
    );
    alg.dec_ref(low);
    // A one-class partition puts every member of the predicate in class zero.
    assert_eq!(
        cells(&*alg, Function::FALSE, f, &mut scratch, |class| class == 0),
        [(0, included.to_vec())]
    );
    alg.dec_ref(f);
}

#[test]
#[cfg(debug_assertions)]
fn refinement_with_an_unreferenced_operand_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let (live, released) = (alg.set(&[1, 2]), alg.set(&[2, 3]));
    alg.dec_ref(released);
    for (f, p) in [
        (released, Function::FALSE),
        (released, Function::TRUE),
        (live, released),
    ] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            alg.refine(f, p, &mut Refine::default(), &mut |_| true, &mut |_, _| {
                ControlFlow::Continue(())
            })
        }))
        .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"cannot use an unreferenced sparse node")
        );
    }
    alg.dec_ref(live);
}

// --- layouts -----------------------------------------------------------------

#[test]
fn refinement_over_wide_keys_recognizes_a_single_point() {
    for d in WIDE {
        let alg = bind(&roomy(), d);
        let mut labels = Vec::new();
        let f = alg.values(
            0,
            &keys(d, [10, 20, 30, 40]),
            &[1, 2, 1, 3],
            &mut labels,
            &mut Values::default(),
        );
        assert_eq!(labels, [0, 1, 2, 3]);
        // A point predicate holds between two consecutive keys, so its cell is
        // itself.
        let p = alg.set(&word(d, 20));
        assert_eq!(rows(&*alg, p), [word(d, 20)]);
        let mut scratch = Refine::default();
        assert_eq!(cells(&*alg, f, p, &mut scratch, |_| true), [(2, vec![20])]);
        assert_eq!(cells(&*alg, f, p, &mut scratch, |_| false), []);
        // Two adjacent points hold between keys that are two apart, so the sweep
        // runs instead.
        let q = alg.set(&keys(d, [10, 11]));
        assert_eq!(
            cells(&*alg, f, q, &mut scratch, |_| true),
            [(0, vec![11]), (1, vec![10])]
        );
        // The layout also reads and writes the payload word of a wide quotient.
        let coarse = alg.quotient(f, &[0, 1, 1, 2], &mut Align::default());
        assert_eq!(alg.evaluate(coarse, &word(d, 20)), 1);
        let same = alg.apply(BoolOp::And, q, q);
        assert_eq!(alg.count(same), Count::from(2));
        for root in [f, p, q, coarse, same] {
            alg.dec_ref(root);
        }
    }
}

#[test]
fn wide_refinement_distinguishes_an_interval_from_one_point() {
    let alg = bind(&roomy(), ScalarSort::bv(65));
    let lo = alg.lt(&[1, 0]);
    let hi = alg.lt(&[2, 1]);
    let p = alg.apply(BoolOp::AndNot, hi, lo);
    let f = alg.set(&[0, 1]);
    // The interval [1, 2^64 + 2) reaches the point 2^64 and its complement.
    // Its endpoints look consecutive if the upper words are ignored.
    let mut selected = Vec::new();
    let flow = alg.refine(
        f,
        p,
        &mut Refine::default(),
        &mut |_| true,
        &mut |class, build| {
            let cell = build();
            selected.push((
                class,
                [alg.evaluate(cell, &[1, 0]), alg.evaluate(cell, &[0, 1])],
            ));
            alg.dec_ref(cell);
            ControlFlow::Continue(())
        },
    );
    assert!(flow.is_continue());
    selected.sort_unstable();
    assert_eq!(selected, [(0, [1, 0]), (1, [0, 1])]);
    for root in [lo, hi, p, f] {
        alg.dec_ref(root);
    }
}

#[test]
fn wide_keys_drive_the_cut_the_conditional_and_the_folds() {
    for d in WIDE {
        let alg = bind(&roomy(), d);
        let (a, b) = (alg.set(&keys(d, 10..20)), alg.set(&keys(d, 15..30)));
        let lt = alg.lt(&word(d, 25));
        let cut = alg.apply(BoolOp::And, b, lt);
        assert_eq!(alg.count(cut), Count::from(10));
        // Below the cut the first branch answers, above it the second.
        let r = alg.ite(lt, a, b);
        let expect: Vec<Vec<u64>> = (10..20).chain(25..30).map(|v| word(d, v)).collect();
        assert_eq!(rows(&*alg, r), expect);
        // A single case over a predicate that is false at zero is already its own
        // partition.
        let mut labels = Vec::new();
        let single = alg.cases(9, &[(a, 4)], &mut labels);
        assert_eq!(single, a);
        assert_eq!(labels, [9, 4]);
        // A case holding at zero opens its cell before the first boundary, so the
        // cell below the other case carries its label rather than the default.
        let low = alg.lt(&word(d, 10));
        labels.clear();
        let f = alg.cases(9, &[(low, 4), (a, 5)], &mut labels);
        assert_eq!(labels, [4, 5, 9]);
        let mut reversed = Vec::new();
        let other = alg.cases(9, &[(a, 5), (low, 4)], &mut reversed);
        assert_eq!(other, f);
        assert_eq!(reversed, labels);
        alg.dec_ref(other);
        for v in [0, 9, 10, 19, 20] {
            let expect = if v < 10 {
                4
            } else if v < 20 {
                5
            } else {
                9
            };
            assert_eq!(
                labels[alg.evaluate(f, &word(d, v)) as usize],
                expect,
                "at {v}"
            );
        }
        // Overlapping cases agree on their label, and the gap above them takes
        // the default.
        labels.clear();
        let g = alg.cases(9, &[(a, 4), (b, 4)], &mut labels);
        assert_eq!(labels, [9, 4]);
        for v in [0, 10, 25, 30] {
            assert_eq!(
                alg.evaluate(g, &word(d, v)),
                u32::from((10..30).contains(&v)),
                "at {v}"
            );
        }
        let mut pairs = Vec::new();
        let h = alg.align(g, f, &mut pairs, &mut Align::default());
        for v in [0, 10, 25, 30] {
            let class = alg.evaluate(h, &word(d, v)) as usize;
            assert_eq!(
                pairs[class],
                (alg.evaluate(g, &word(d, v)), alg.evaluate(f, &word(d, v))),
                "at {v}"
            );
        }
        for root in [a, b, lt, cut, r, single, low, f, g, h] {
            alg.dec_ref(root);
        }
    }
}

#[test]
fn one_word_keys_above_32_bits_use_the_run_time_layout() {
    // A sort between 33 and 64 bits is one key word plus a payload word, the
    // run-time width arm with a single word, which neither packs nor fixes its
    // width statically.
    for d in [ScalarSort::bv(40), ScalarSort::symbol()] {
        let store = roomy();
        let alg = bind(&store, d);
        let high = 1 << 36;
        let p = set(&*alg, [3, high, high + 1, high + 2, u64::MAX >> 24]);
        assert_eq!(
            members(&*alg, p),
            [3, high, high + 1, high + 2, u64::MAX >> 24]
        );
        assert_eq!(alg.count(p), Count::from(5));
        let q = set(&*alg, [high + 1, high + 5]);
        let r = alg.apply(BoolOp::AndNot, p, q);
        assert_eq!(members(&*alg, r), [3, high, high + 2, u64::MAX >> 24]);
        let mut labels = Vec::new();
        let f = alg.cases(0, &[(p, 1), (q, 1)], &mut labels);
        assert_eq!(labels, [0, 1]);
        assert_eq!(
            members(&*alg, f),
            [3, high, high + 1, high + 2, high + 5, u64::MAX >> 24]
        );
        let selected = set(&*alg, [2, 3, high + 2, high + 3]);
        assert_eq!(
            cells(&*alg, f, selected, &mut Refine::default(), |_| true),
            [(0, vec![2, high + 3]), (1, vec![3, high + 2])]
        );
        for root in [p, q, r, f, selected] {
            alg.dec_ref(root);
        }
    }
    // The complement of a symbol set counts from the full 64-bit universe.
    let alg = bind(&roomy(), ScalarSort::symbol());
    let p = set(&*alg, [7, u64::MAX]);
    let n = alg.apply(BoolOp::AndNot, Function::TRUE, p);
    assert_eq!(alg.count(n), Count::from(u64::MAX - 1));
    assert!(holds(&*alg, n, 0) && holds(&*alg, n, 8) && !holds(&*alg, n, u64::MAX));
    alg.dec_ref(p);
    alg.dec_ref(n);
}

// --- members -----------------------------------------------------------------

#[test]
fn member_cursors_cross_word_boundaries() {
    let alg = bind(&roomy(), ScalarSort::bv(70));
    // The two adjacent values on opposite sides of the word boundary
    // form one run.
    let sorted = [5, 0, u64::MAX, 0, 0, 1, 5, 1];
    let q = alg.set(&sorted);
    let mut cursor = alg.members(q);
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.extend_from_slice(v);
    }
    assert_eq!(out, sorted);
    assert_eq!(alg.count(q), Count::from(4));
    assert!(rows(&*alg, Function::FALSE).is_empty());
    assert_eq!(alg.count(Function::TRUE), Count::from_digits(vec![0, 64]));
    alg.dec_ref(q);
    // The constants enumerate the whole of the smallest universe, or nothing.
    let alg = bind(&roomy(), ScalarSort::bool());
    assert_eq!(members(&*alg, Function::TRUE), [0, 1]);
    assert!(members(&*alg, Function::FALSE).is_empty());
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn member_enumeration_of_an_unreferenced_predicate_is_rejected() {
    let alg = bind(&roomy(), BV4);
    let p = set(&*alg, [1]);
    alg.dec_ref(p);
    let _ = alg.members(p);
}
