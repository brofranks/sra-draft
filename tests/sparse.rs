//! Public-contract tests for the sparse backend.
//!
//! Concrete points and ranges provide expected values. Other assertions
//! check canonical roots and consistency between operations. The fixtures
//! cover packed, fixed two-word, and runtime-width keys.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sra::backend::{Alignment, Backend, BoolOp, Column, Function, MapSpec, Refinement, Values};
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;
use sra::util::count::Count;

#[path = "support/sparse.rs"]
mod fixtures;

use fixtures::{BV4, BV6, col, holds, members, parts, set};

/// The widest packed sort.
const BV32: ScalarSort = ScalarSort::bit_vec(32);
/// The two wide layouts: a statically known pair of key words, and a run-time
/// key width.
const WIDE: [ScalarSort; 2] = [ScalarSort::bit_vec(100), ScalarSort::bit_vec(200)];

/// A store with room for ordinary test fixtures.
fn roomy() -> Arc<SparseStore> {
    Arc::new(SparseStore::with_capacity(128, 512, 1))
}

/// Every member of a predicate, one row of the sort's words each.
fn rows(c: &dyn Column, p: Function) -> Vec<Vec<u64>> {
    let mut cursor = c.members(p);
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
    c: &dyn Column,
    f: Function,
    p: Function,
    scratch: &mut Refinement,
    mut select: impl FnMut(u32) -> bool,
) -> Vec<(u32, Vec<u64>)> {
    let mut out = Vec::new();
    let _ = c.refine(f, p, scratch, &mut select, &mut |class, build| {
        let cell = build();
        out.push((class, members(c, cell)));
        c.dec_ref(cell);
        ControlFlow::Continue(())
    });
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
    let store = roomy();
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
fn a_column_retains_the_store_and_roots_from_another_binding() {
    let store = roomy();
    let sort = ScalarSort::bit_vec(12);
    let a = col(&store, sort);
    let b = col(&store, sort);
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
#[should_panic(expected = "value is outside the column sort")]
fn evaluating_a_constant_still_checks_the_value_encoding() {
    let _ = col(&roomy(), BV4).evaluate(Function::TRUE, &[0, 0]);
}

#[test]
#[should_panic(expected = "bound is outside the column sort")]
fn a_cut_with_the_wrong_word_count_is_rejected() {
    let _ = col(&roomy(), BV4).less_than(&[0, 0]);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn set_values_off_the_word_stride_are_rejected() {
    let _ = col(&roomy(), ScalarSort::bit_vec(65)).set(&[1]);
}

#[test]
fn counts_reach_the_universe_boundary_at_every_layout() {
    // One-word sorts up to the full 64 bits, then the static and run-time wide
    // layouts, including a width whose universe count needs a word above the
    // key words.
    let sorts = [4, 32, 33, 64, 100, 128, 200].map(ScalarSort::bit_vec);
    for d in sorts {
        let c = col(&roomy(), d);
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
        let ge = c.apply_bool(BoolOp::AndNot, Function::TRUE, c.less_than(&word(d, 10)));
        let p = c.apply_bool(BoolOp::Or, c.set(&keys(d, [1, 2, 3, 7])), ge);
        assert_eq!(c.count(p), without(6), "{d:?}");
        let holes = c.apply_bool(BoolOp::AndNot, Function::TRUE, c.set(&keys(d, [0, 3])));
        assert_eq!(c.count(holes), without(2), "{d:?}");
        assert_eq!(c.count(Function::TRUE), universe, "{d:?}");
        assert_eq!(c.count(Function::FALSE), Count::ZERO, "{d:?}");
        // A predicate holding at zero opens its first run there.
        let low = c.apply_bool(BoolOp::Or, c.less_than(&word(d, 10)), c.set(&keys(d, [12])));
        assert_eq!(c.count(low), Count::from(11), "{d:?}");
    }
}

#[test]
fn an_empty_count_stays_small_for_a_wide_universe() {
    let c = col(&roomy(), ScalarSort::bit_vec(1 << 30));
    assert_eq!(c.count(Function::FALSE), Count::ZERO);
}

#[test]
#[should_panic(expected = "count requires a Boolean predicate")]
fn counting_a_partition_with_more_than_two_classes_is_rejected() {
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
    let store = roomy();
    let c = col(&store, BV6);
    let (a, b) = (set(&*c, (0..16).map(|v| v * 3)), set(&*c, 10..28));
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        let r = c.apply_bool(op, a, b);
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
        assert_eq!(members(&*c, r), expect, "{op:?}");
        c.dec_ref(r);
    }
    // Equal operands and constants never reach the merge at all.
    assert_eq!(c.apply_bool(BoolOp::Xor, a, a), Function::FALSE);
    let r = c.apply_bool(BoolOp::And, a, Function::TRUE);
    assert_eq!(r, a);
    for f in [a, b, r] {
        c.dec_ref(f);
    }
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
    let c = col(&roomy(), ScalarSort::bit_vec(12));
    let in_p = [3, 65, 2114, 4095];
    let in_q = [0, 21, 2070, 4051];
    let (p, q) = (c.set(&in_p), c.set(&in_q));
    let read = |f| {
        let out = members(&*c, f);
        c.dec_ref(f);
        out
    };
    // `x - 44`, undefined below it, so the low member drops out of both
    // directions.
    let map = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = x[0].saturating_sub(44);
        x[0] >= 44
    }));
    assert_eq!(read(c.image(&map, p, q)), [21, 2070, 4051]);
    assert_eq!(read(c.preimage(&map, p, q)), [65, 2114, 4095]);
    assert_eq!(read(c.preimage(&map, p, Function::TRUE)), [65, 2114, 4095]);
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
    assert_eq!(read(c.image(&bits, p, q)), expect);
    let expect: Vec<u64> = in_p
        .iter()
        .copied()
        .filter(|&x| in_q.contains(&moved(x)))
        .collect();
    assert_eq!(read(c.preimage(&bits, p, q)), expect);
    // The identity needs no enumeration in either direction: it is a
    // conjunction.
    let id = MapSpec::identity(1);
    assert_eq!(c.image(&id, p, q), Function::FALSE);
    let r = c.preimage(&id, p, p);
    assert_eq!(r, p);
    for f in [p, q, r] {
        c.dec_ref(f);
    }
}

#[test]
fn a_bit_preimage_of_true_retains_the_source() {
    let c = col(&roomy(), BV4);
    let p = c.set(&[1, 4]);
    let map = MapSpec::Bits {
        clear: vec![0].into(),
        flip: vec![1].into(),
    };
    let image = c.image(&map, p, Function::TRUE);
    assert_eq!(members(&*c, image), [0, 5]);
    c.dec_ref(image);
    let f = c.preimage(&map, p, Function::TRUE);
    assert_eq!(f, p);
    c.dec_ref(p);
    assert_eq!(members(&*c, f), [1, 4]);
    c.dec_ref(f);
}

#[test]
#[should_panic(expected = "bit map masks must encode the scalar sort")]
fn a_bit_preimage_of_true_still_checks_its_masks() {
    let c = col(&roomy(), BV4);
    let map = MapSpec::Bits {
        clear: vec![0].into(),
        flip: vec![16].into(),
    };
    let _ = c.preimage(&map, Function::TRUE, Function::TRUE);
}

#[test]
#[cfg(debug_assertions)]
fn an_unreferenced_target_is_rejected_even_if_the_map_yields_no_image() {
    let c = col(&roomy(), BV4);
    let p = c.set(&[1]);
    let q = c.set(&[2]);
    c.dec_ref(q);
    let map = MapSpec::Rust(Arc::new(|_, _| false));
    for image in [true, false] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            if image {
                c.image(&map, p, q)
            } else {
                c.preimage(&map, p, q)
            }
        }))
        .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"cannot use an unreferenced sparse node")
        );
    }
    c.dec_ref(p);
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
    let store = roomy();
    let c = col(&store, BV6);
    // The branches share one transition point with the condition and one with
    // each other, so the sweep must advance several runs at the same key.
    let k = set(&*c, 10..20);
    let y = set(&*c, (10..30).step_by(2));
    let n = c.set(&[0, 5, 15, 25]);
    let r = c.ite(k, y, n);
    let expect: Vec<u64> = (0..64)
        .filter(|&x| {
            if (10..20).contains(&x) {
                x % 2 == 0
            } else {
                [0, 5, 15, 25].contains(&x)
            }
        })
        .collect();
    assert_eq!(members(&*c, r), expect);
    // A constant condition selects one branch outright, retaining its existing
    // root.
    let yes = c.ite(Function::TRUE, y, n);
    let no = c.ite(Function::FALSE, y, n);
    assert_eq!(yes, y);
    assert_eq!(no, n);
    for f in [k, y, n, r, yes, no] {
        c.dec_ref(f);
    }
}

// --- alignment and quotient --------------------------------------------------

#[test]
fn alignment_and_quotient_renumber_by_first_occurrence() {
    let store = roomy();
    let c = col(&store, BV6);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let g = set(&*c, [20, 30]);
    let mut labels = vec![(9, 9)];
    let mut scratch = Alignment::default();
    let h = c.align(f, g, &mut labels, &mut scratch);
    // Class numbers index the appended suffix after the existing prefix.
    assert_eq!(labels, [(9, 9), (0, 0), (1, 0), (2, 1), (1, 1), (3, 0)]);
    for x in 0..64 {
        let class = c.evaluate(h, &[x]) as usize;
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
    let q = c.quotient(f, &[0, 1, 1, 2], &mut scratch);
    for x in 0..64 {
        let expect = match x {
            10 | 20 | 30 => 1,
            40 => 2,
            _ => 0,
        };
        assert_eq!(c.evaluate(q, &[x]), expect, "at {x}");
    }
    for f in [f, g, h, q] {
        c.dec_ref(f);
    }
}

// --- cases -------------------------------------------------------------------

#[test]
fn cases_fold_onto_labels_over_their_boundaries() {
    let store = roomy();
    let c = col(&store, BV6);
    let (a, b) = (set(&*c, 10..20), set(&*c, 15..30));
    let mut labels = Vec::new();
    // Overlapping cases agree on their label, so the first one ending leaves the
    // cell covered. The gap above them falls back to the default.
    let f = c.cases(9, &[(a, 4), (b, 4)], &mut labels);
    assert_eq!(labels, [9, 4]);
    for x in 0..64 {
        assert_eq!(
            c.evaluate(f, &[x]),
            u32::from((10..30).contains(&x)),
            "at {x}"
        );
    }
    // A single case labeled like the default is one cell, and is swept like any
    // other.
    let mut labels = Vec::new();
    assert_eq!(c.cases(4, &[(a, 4)], &mut labels), Function::FALSE);
    assert_eq!(labels, [4]);
    for f in [a, b, f] {
        c.dec_ref(f);
    }
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn an_overlap_between_cases_with_different_labels_is_rejected() {
    let c = col(&roomy(), BV4);
    let (a, b) = (set(&*c, [1, 2, 3]), set(&*c, [3, 4]));
    let _ = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

#[test]
#[should_panic(expected = "cases with different labels overlap")]
fn an_overlap_starting_at_zero_is_rejected() {
    let c = col(&roomy(), BV4);
    let a = c.less_than(&[2]);
    let b = c.less_than(&[4]);
    let _ = c.cases(0, &[(a, 1), (b, 2)], &mut Vec::new());
}

#[test]
fn a_boolean_case_predicate_is_already_its_own_partition() {
    let c = col(&roomy(), BV6);
    let p = set(&*c, 10..20);
    let mut labels = Vec::new();
    // Zero is outside the case, so the predicate's own payloads are the
    // canonical classes.
    let f = c.cases(8, &[(p, 3)], &mut labels);
    assert_eq!(f, p);
    assert_eq!(labels, [8, 3]);
    c.dec_ref(p);
    c.dec_ref(f);
    // A case holding at zero is rebuilt so its initial class is zero.
    let q = c.less_than(&[20]);
    labels.clear();
    let f = c.cases(8, &[(q, 3)], &mut labels);
    assert_eq!(labels, [3, 8]);
    assert_eq!(
        (0..64)
            .map(|x| c.evaluate(f, &[x]))
            .filter(|&class| class == 0)
            .count(),
        20
    );
    c.dec_ref(q);
    c.dec_ref(f);
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
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn value_keys_off_the_word_stride_are_rejected() {
    let c = col(&roomy(), ScalarSort::bit_vec(65));
    let _ = c.values(0, &[1, 0, 2], &[7], &mut Vec::new(), &mut Values::default());
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn values_with_too_few_labels_are_rejected() {
    let c = col(&roomy(), ScalarSort::bit_vec(65));
    let _ = c.values(
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
    let c = col(&roomy(), BV4);
    let _ = c.values(0, &[], &[7], &mut Vec::new(), &mut Values::default());
}

#[test]
fn values_reuse_buffers_and_number_classes_from_the_label_suffix() {
    let c = col(&roomy(), BV4);
    let mut scratch = Values::default();
    let mut first = Vec::new();
    let f = c.values(9, &[1], &[3], &mut first, &mut scratch);
    assert_eq!(first, [9, 3]);
    c.dec_ref(f);

    let mut labels = vec![77];
    let f = c.values(9, &[1, 3], &[4, 3], &mut labels, &mut scratch);
    assert_eq!(labels, [77, 9, 4, 3]);
    assert_eq!([0, 1, 3].map(|x| c.evaluate(f, &[x])), [0, 1, 2]);
    c.dec_ref(f);
}

// --- refinement --------------------------------------------------------------

#[test]
fn refinement_groups_the_accepted_runs_by_class() {
    let store = roomy();
    let c = col(&store, BV6);
    let (f, _) = parts(&*c, 0, [1, 2, 1, 3]);
    let p = set(&*c, [10, 20, 25, 30]);
    let mut scratch = Refinement::default();
    assert_eq!(
        cells(&*c, f, p, &mut scratch, |_| true),
        [(0, vec![25]), (1, vec![10, 30]), (2, vec![20])]
    );
    // Declining a class leaves the other cells unchanged.
    assert_eq!(
        cells(&*c, f, p, &mut scratch, |class| class == 1),
        [(1, vec![10, 30])]
    );
    // One reached class means the predicate itself is that cell, whatever its
    // shape.
    let one = c.set(&[10, 30]);
    assert_eq!(
        cells(&*c, f, one, &mut scratch, |_| true),
        [(1, vec![10, 30])]
    );
    // A false predicate reaches no class.
    assert_eq!(
        cells(&*c, f, Function::FALSE, &mut scratch, |_| {
            panic!("no class is reached")
        }),
        []
    );
    // A wide interval spans whole cells and the gaps between them.
    let window = set(&*c, 10..31);
    assert_eq!(
        cells(&*c, f, window, &mut scratch, |class| class != 0),
        [(1, vec![10, 30]), (2, vec![20])]
    );
    // A visitor that breaks stops the walk immediately.
    let mut count = 0;
    let stopped = c.refine(f, p, &mut scratch, &mut |_| true, &mut |_, _| {
        count += 1;
        ControlFlow::Break(())
    });
    assert!(stopped.is_break() && count == 1);
    for f in [f, p, one, window] {
        c.dec_ref(f);
    }
}

#[test]
fn sibling_refinements_reuse_scratch_across_zero_and_the_maximum() {
    let c = col(&roomy(), BV4);
    let f = c.values(0, &[1, 3], &[1, 2], &mut Vec::new(), &mut Values::default());
    let p = c.set(&[0, 1, 3, 15]);
    let q = c.set(&[3, 5]);
    let mut scratch = Refinement::default();
    assert_eq!(
        cells(&*c, f, p, &mut scratch, |_| true),
        [(0, vec![0, 15]), (1, vec![1]), (2, vec![3])]
    );
    assert_eq!(
        cells(&*c, f, q, &mut scratch, |_| true),
        [(0, vec![5]), (2, vec![3])]
    );
    for f in [f, p, q] {
        c.dec_ref(f);
    }
}

#[test]
fn refinement_of_a_boolean_map_reuses_the_map_itself() {
    let store = roomy();
    let c = col(&store, BV6);
    // A map whose payloads are already Boolean is the indicator of its own class
    // one, so the unrestricted refinement is that map and its payload flip,
    // without rebuilding either.
    let f = set(&*c, [10, 20, 30]);
    let mut scratch = Refinement::default();
    assert_eq!(
        cells(&*c, f, Function::TRUE, &mut scratch, |_| true)
            .into_iter()
            .map(|(class, cell)| (class, cell.len()))
            .collect::<Vec<_>>(),
        [(0, 61), (1, 3)]
    );
    let flipped = cells(&*c, f, Function::TRUE, &mut scratch, |class| class == 0);
    assert_eq!(flipped[0].1.len(), 61);
    assert!(!flipped[0].1.contains(&20));
    // The trivial partition has no payload other than its default, so the whole
    // universe is one cell and the predicate is it.
    assert_eq!(
        cells(&*c, Function::FALSE, Function::TRUE, &mut scratch, |_| true).len(),
        1
    );
    c.dec_ref(f);
}

#[test]
#[cfg(debug_assertions)]
fn refinement_of_an_unreferenced_partition_is_rejected() {
    let c = col(&roomy(), BV4);
    let f = set(&*c, [1, 2]);
    c.dec_ref(f);
    for p in [Function::FALSE, Function::TRUE] {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            c.refine(
                f,
                p,
                &mut Refinement::default(),
                &mut |_| true,
                &mut |_, _| ControlFlow::Continue(()),
            )
        }))
        .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<&str>(),
            Some(&"cannot use an unreferenced sparse node")
        );
    }
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
        let mut scratch = Refinement::default();
        assert_eq!(cells(&*c, f, p, &mut scratch, |_| true), [(2, vec![20])]);
        assert_eq!(cells(&*c, f, p, &mut scratch, |_| false), []);
        // Two adjacent points hold between keys that are two apart, so the sweep
        // runs instead.
        let q = c.set(&keys(d, [10, 11]));
        assert_eq!(
            cells(&*c, f, q, &mut scratch, |_| true),
            [(0, vec![11]), (1, vec![10])]
        );
        // The layout also reads and writes the payload word of a wide quotient.
        let coarse = c.quotient(f, &[0, 1, 1, 2], &mut Alignment::default());
        assert_eq!(c.evaluate(coarse, &word(d, 20)), 1);
        assert_eq!(c.count(c.apply_bool(BoolOp::And, q, q)), Count::from(2));
    }
}

#[test]
fn wide_keys_drive_the_cut_the_conditional_and_the_folds() {
    for d in WIDE {
        let c = col(&roomy(), d);
        let (a, b) = (c.set(&keys(d, 10..20)), c.set(&keys(d, 15..30)));
        let lt = c.less_than(&word(d, 25));
        assert_eq!(c.count(c.apply_bool(BoolOp::And, b, lt)), Count::from(10));
        // Below the cut the first branch answers, above it the second.
        let r = c.ite(lt, a, b);
        let expect: Vec<Vec<u64>> = (10..20).chain(25..30).map(|v| word(d, v)).collect();
        assert_eq!(rows(&*c, r), expect);
        // A single case over a predicate that is false at zero is already its own
        // partition.
        let mut labels = Vec::new();
        assert_eq!(c.cases(9, &[(a, 4)], &mut labels), a);
        assert_eq!(labels, [9, 4]);
        // A case holding at zero opens its cell before the first boundary, so the
        // cell below the other case carries its label rather than the default.
        let low = c.less_than(&word(d, 10));
        labels.clear();
        let f = c.cases(9, &[(low, 4), (a, 5)], &mut labels);
        assert_eq!(labels, [4, 5, 9]);
        for v in [0, 9, 10, 19, 20] {
            let expect = if v < 10 {
                4
            } else if v < 20 {
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
        for v in [0, 10, 25, 30] {
            assert_eq!(
                c.evaluate(f, &word(d, v)),
                u32::from((10..30).contains(&v)),
                "at {v}"
            );
        }
        let mut pairs = Vec::new();
        let h = c.align(f, lt, &mut pairs, &mut Alignment::default());
        for v in [0, 10, 25, 30] {
            let class = c.evaluate(h, &word(d, v)) as usize;
            assert_eq!(
                pairs[class],
                (c.evaluate(f, &word(d, v)), c.evaluate(lt, &word(d, v))),
                "at {v}"
            );
        }
    }
}

#[test]
fn one_word_keys_above_32_bits_use_the_run_time_layout() {
    // A sort between 33 and 64 bits is one key word plus a payload word, the
    // run-time width arm with a single word, which neither packs nor fixes its
    // width statically.
    for d in [ScalarSort::bit_vec(40), ScalarSort::symbol()] {
        let store = roomy();
        let c = col(&store, d);
        let high = 1 << 36;
        let p = set(&*c, [3, high, high + 1, high + 2, u64::MAX >> 24]);
        assert_eq!(
            members(&*c, p),
            [3, high, high + 1, high + 2, u64::MAX >> 24]
        );
        assert_eq!(c.count(p), Count::from(5));
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
            cells(
                &*c,
                f,
                set(&*c, [2, 3, high + 2, high + 3]),
                &mut Refinement::default(),
                |_| true
            ),
            [(0, vec![2, high + 3]), (1, vec![3, high + 2])]
        );
    }
    // The complement of a symbol set counts to the full 64-bit universe without
    // enumerating it.
    let c = col(&roomy(), ScalarSort::symbol());
    let p = set(&*c, [7, u64::MAX]);
    let n = c.apply_bool(BoolOp::AndNot, Function::TRUE, p);
    assert_eq!(c.count(n), Count::from(u64::MAX - 1));
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
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.extend_from_slice(v);
    }
    assert_eq!(out, sorted);
    assert_eq!(c.count(q), Count::from(4));
    assert!(rows(&*c, Function::FALSE).is_empty());
    assert_eq!(c.count(Function::TRUE), Count::from_digits(vec![0, 64]));
    c.dec_ref(q);
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
