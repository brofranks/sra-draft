//! Public-contract tests for `sra::backend`.
//!
//! Covers tokens, Boolean operations, map descriptions, and transfer. A listed
//! cursor feeds transfer, so no backend's encoding shapes the expectations.
//! Both stores exercise the same public partition and conditional contracts.

use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sra::backend::{Algebra, Align, Backend, BoolOp, Function, Groups, MapSpec, Values, transfer};
use sra::bdd::BddStore;
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;

const BV4: ScalarSort = ScalarSort::bv(4);

/// A nonconstant token for testing Boolean identities without a store.
fn opaque() -> Function {
    Function::new(7)
}

#[path = "support/listed.rs"]
mod listed;

use listed::Listed;

// --- tokens ----

#[test]
fn function_tokens_report_the_store_constants() {
    assert!(Function::FALSE.is_f() && Function::FALSE.is_constant() && !Function::FALSE.is_t());
    assert!(Function::TRUE.is_t() && Function::TRUE.is_constant() && !Function::TRUE.is_f());
    assert_eq!(Function::new(7).index(), 7);
    assert!(!opaque().is_constant() && !opaque().is_f() && !opaque().is_t());
}

// --- boolean operations ----

#[test]
fn boolean_operations_evaluate_bitwise_and_by_value() {
    // `evaluate_bool` is `evaluate_mask` at one bit, so the truth tables are
    // spelled out for both.
    for (op, mask) in [
        (BoolOp::And, 0b1000),
        (BoolOp::Or, 0b1110),
        (BoolOp::Xor, 0b0110),
        (BoolOp::AndNot, 0b0100),
    ] {
        assert_eq!(op.evaluate_mask(0b1100, 0b1010), mask, "{op:?}");
    }
    let table = |op: BoolOp| {
        [(false, false), (false, true), (true, false), (true, true)]
            .map(|(l, r)| op.evaluate_bool(l, r))
    };
    assert_eq!(table(BoolOp::And), [false, false, false, true]);
    assert_eq!(table(BoolOp::Or), [false, true, true, true]);
    assert_eq!(table(BoolOp::Xor), [false, true, true, false]);
    assert_eq!(table(BoolOp::AndNot), [false, false, true, false]);
    assert!(BoolOp::And.is_commutative());
    assert!(BoolOp::Or.is_commutative());
    assert!(BoolOp::Xor.is_commutative());
    assert!(!BoolOp::AndNot.is_commutative());
}

#[test]
fn boolean_identities_simplify_without_a_store() {
    let (f, t, p, q) = (Function::FALSE, Function::TRUE, opaque(), Function::new(9));
    // Equal operands, then each constant on each side; the operands are never
    // dereferenced.
    assert_eq!(BoolOp::And.simplify(p, p), Some(p));
    assert_eq!(BoolOp::Or.simplify(p, p), Some(p));
    assert_eq!(BoolOp::Xor.simplify(p, p), Some(f));
    assert_eq!(BoolOp::AndNot.simplify(p, p), Some(f));
    assert_eq!(BoolOp::And.simplify(f, p), Some(f));
    assert_eq!(BoolOp::And.simplify(p, f), Some(f));
    assert_eq!(BoolOp::And.simplify(t, p), Some(p));
    assert_eq!(BoolOp::And.simplify(p, t), Some(p));
    assert_eq!(BoolOp::Or.simplify(f, p), Some(p));
    assert_eq!(BoolOp::Or.simplify(p, f), Some(p));
    assert_eq!(BoolOp::Or.simplify(t, p), Some(t));
    assert_eq!(BoolOp::Or.simplify(p, t), Some(t));
    assert_eq!(BoolOp::Xor.simplify(f, p), Some(p));
    assert_eq!(BoolOp::Xor.simplify(p, f), Some(p));
    assert_eq!(BoolOp::AndNot.simplify(f, p), Some(f));
    assert_eq!(BoolOp::AndNot.simplify(p, t), Some(f));
    assert_eq!(BoolOp::AndNot.simplify(p, f), Some(p));
    // Nothing determines these, so the caller must compute them.
    assert_eq!(BoolOp::Xor.simplify(p, t), None);
    assert_eq!(BoolOp::AndNot.simplify(t, p), None);
    assert_eq!(BoolOp::And.simplify(p, q), None);
}

// --- maps ----

#[test]
fn bit_maps_are_closed_data_every_backend_reads() {
    let id = MapSpec::identity(2);
    assert!(id.is_identity() && id.clone().is_identity());
    let mut y = vec![0; 2];
    assert!(id.evaluate(&[3, 5], &mut y) && y == [3, 5]);
    // This map clears bit 0 and flips bit 1.
    let m = MapSpec::Bits {
        clear: vec![0b01].into(),
        flip: vec![0b10].into(),
    };
    assert!(!m.is_identity());
    let mut y = vec![0];
    assert!(m.evaluate(&[0b01], &mut y) && y == [0b10]);
    assert!(m.evaluate(&[0b10], &mut y) && y == [0b00]);
    // A Rust map is partial: a rejected input has no image in either direction.
    let f = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = x[0] + 1;
        x[0] != 3
    }));
    assert!(!f.is_identity());
    assert!(f.evaluate(&[1], &mut y) && y == [2]);
    assert!(!f.evaluate(&[3], &mut y));
}

#[test]
fn symbol_maps_admit_only_the_identity_and_rust_kinds() {
    let symbol = ScalarSort::symbol();
    MapSpec::identity(1).check_sort(symbol);
    MapSpec::Rust(Arc::new(|_: &[u64], _: &mut [u64]| true)).check_sort(symbol);
    // Bit vectors keep their bitwise interpretation.
    MapSpec::Bits {
        clear: vec![1].into(),
        flip: vec![0].into(),
    }
    .check_sort(BV4);
}

#[test]
#[should_panic(expected = "symbols do not support bitwise maps")]
fn a_bitwise_map_over_symbols_is_rejected() {
    MapSpec::Bits {
        clear: vec![1].into(),
        flip: vec![0].into(),
    }
    .check_sort(ScalarSort::symbol());
}

#[test]
#[should_panic(expected = "bit map masks must encode the scalar sort")]
fn bit_map_masks_outside_the_sort_are_rejected() {
    MapSpec::Bits {
        clear: vec![0x10].into(),
        flip: vec![0].into(),
    }
    .check_sort(BV4);
}

// --- transfer ----

#[test]
fn transfer_keeps_the_members_or_the_images_the_filter_admits() {
    // The map adds four except at three, where its output may be outside BV4.
    let map = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = if x[0] == 3 { u64::MAX } else { x[0] + 4 };
        x[0] != 3
    }));
    let members = || Listed(&[[1], [2], [3]]);
    let image = transfer(BV4, &map, &mut members(), |y| y[0] != 5, true);
    assert_eq!(image, [6]);
    let preimage = transfer(BV4, &map, &mut members(), |y| y[0] != 5, false);
    assert_eq!(preimage, [2]);
    // Images arrive sorted and distinct even when the map merges members.
    let fold = MapSpec::Bits {
        clear: vec![0b0001].into(),
        flip: vec![0b1010].into(),
    };
    assert_eq!(
        transfer(BV4, &fold, &mut members(), |_| true, true),
        [8, 10]
    );
    assert_eq!(
        transfer(BV4, &fold, &mut members(), |_| true, false),
        [1, 2, 3]
    );
}

#[test]
fn repeated_images_use_bounded_storage_and_filter_each_member_in_order() {
    let points: Vec<_> = (0..1024).map(|x| [x]).collect();
    for keep in [0, 1] {
        let map = MapSpec::Bits {
            clear: [1023 ^ keep].into(),
            flip: [0].into(),
        };
        let mut cursor = Listed(&points);
        let mut calls = 0;
        let image = transfer(
            ScalarSort::bv(10),
            &map,
            &mut cursor,
            |y| {
                assert_eq!(y, [calls & keep]);
                calls += 1;
                calls > 2
            },
            true,
        );
        assert_eq!(calls, 1024);
        assert!(cursor.0.is_empty());
        assert_eq!(image, &[0, 1][..=keep as usize]);
        assert!(image.capacity() <= 128);
    }
}

#[test]
fn multiword_transfer_preserves_preimage_order_and_sorts_images() {
    let sort = ScalarSort::bv(65);
    let map = MapSpec::Bits {
        clear: vec![1, 0].into(),
        flip: vec![0, 1].into(),
    };
    let members = || Listed(&[[0, 0], [1, 0], [2, 0], [0, 1]]);
    assert_eq!(
        transfer(sort, &map, &mut members(), |_| true, true),
        [0, 0, 0, 1, 2, 1]
    );
    assert_eq!(
        transfer(sort, &map, &mut members(), |y| y[0] == 0, false),
        [0, 0, 1, 0, 0, 1]
    );
    assert_eq!(
        transfer(
            sort,
            &MapSpec::identity(2),
            &mut members(),
            |y| y[0] != 1,
            true
        ),
        [0, 0, 2, 0, 0, 1]
    );
    let points: Vec<_> = (0..128).map(|x| [x, 0]).collect();
    let fold = MapSpec::Rust(Arc::new(|x, y| {
        y.copy_from_slice(&[[0, 1], [u64::MAX, 0], [1, 0]][(x[0] % 3) as usize]);
        true
    }));
    assert_eq!(
        transfer(sort, &fold, &mut Listed(&points), |_| true, true),
        [1, 0, u64::MAX, 0, 0, 1]
    );
}

#[test]
#[should_panic(expected = "cursor member does not encode the scalar sort")]
fn a_preimage_member_with_the_wrong_stride_is_rejected() {
    let mut members = Listed(&[[5, 1]]);
    transfer(BV4, &MapSpec::identity(1), &mut members, |_| true, false);
}

#[test]
#[should_panic(expected = "cursor member does not encode the scalar sort")]
fn a_transfer_member_with_unused_high_bits_is_rejected() {
    let map = MapSpec::Rust(Arc::new(|_, _| panic!("map received an invalid member")));
    let mut members = Listed(&[[16]]);
    transfer(BV4, &map, &mut members, |_| true, true);
}

#[test]
#[should_panic(expected = "map image does not encode the scalar sort")]
fn a_transfer_image_outside_the_sort_is_rejected() {
    let map = MapSpec::Rust(Arc::new(|_: &[u64], y: &mut [u64]| {
        y[0] = 99;
        true
    }));
    let mut members = Listed(&[[1]]);
    transfer(
        BV4,
        &map,
        &mut members,
        |_| panic!("filter received an invalid image"),
        true,
    );
}

#[track_caller]
fn rejects<T: Debug>(expected: &str, build: impl FnOnce() -> T) {
    let panic = catch_unwind(AssertUnwindSafe(build)).expect_err("invalid input was accepted");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap();
    assert!(message.contains(expected), "unexpected panic: {message}");
}

// --- refinement grouping ----

#[test]
fn grouping_preserves_selected_classes_and_forgets_declined_ones() {
    let mut r = Groups::default();
    // Class 2 is declined once. Accepted classes retain their groups.
    let mut select = |class| class != 2;
    r.group(0, 5, &mut select).unwrap().push(8);
    assert!(r.group(2, 5, &mut select).is_none());
    assert!(
        r.group(2, 5, &mut |_| panic!(
            "a declined class is never asked again"
        ))
        .is_none()
    );
    assert!(r.group(3, 6, &mut select).unwrap().is_empty());
    assert_eq!(r.group(0, 9, &mut select).unwrap().as_slice(), [8]);
    assert_eq!(r.reached(), 3);
    assert!(
        r.selected()
            .map(|(class, initial, _)| (class, initial))
            .eq([(0, 5), (3, 6)])
    );

    // The next call clears the lookup and reuses the existing buffers.
    r.restart();
    assert!(r.group(3, 0, &mut select).unwrap().is_empty());
    assert_eq!(r.reached(), 1);
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((3, 0)));
    for class in [0, 2] {
        assert!(r.group(class, 0, &mut |_| true).unwrap().is_empty());
    }
}

#[test]
fn refinement_restart_clears_classes_after_a_selector_panic() {
    let mut r = Groups::default();
    r.group(3, 5, &mut |_| true).unwrap().extend([8, 9]);
    assert!(r.group(1, 5, &mut |_| false).is_none());
    rejects("selector failed", || {
        let _ = r.group(2, 5, &mut |_| panic!("selector failed"));
    });

    // Restart clears every class touched before the selector panicked.
    r.restart();
    assert_eq!(r.reached(), 0);
    assert!(r.selected().next().is_none());
    let mut asked = Vec::new();
    let mut select = |class| {
        asked.push(class);
        true
    };
    assert!(r.group(1, 7, &mut select).unwrap().is_empty());
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((1, 7)));
    assert_eq!(r.reached(), 1);

    r.restart();
    assert!(r.group(3, 9, &mut select).unwrap().is_empty());
    assert_eq!(asked, [1, 3]);
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((3, 9)));
}

#[test]
#[cfg(target_pointer_width = "32")]
fn an_unrepresentable_class_leaves_grouping_reusable() {
    let mut r = Groups::default();
    r.group(1, 1, &mut |_| true).unwrap().push(8);
    rejects("class lookup length exceeds usize", || {
        let _ = r.group(u32::MAX, 0, &mut |_| true);
    });
    r.restart();
    assert_eq!(r.reached(), 0);
    assert!(r.group(1, 0, &mut |_| true).unwrap().is_empty());
}

// --- store-backed operations ----

fn algebras(sort: ScalarSort, edges: usize) -> [Box<dyn Algebra>; 2] {
    let bdd = Arc::new(BddStore::with_capacity(16, edges, 1));
    let sparse = Arc::new(SparseStore::with_capacity(16, edges, 1));
    [bdd.bind(sort).unwrap(), sparse.bind(sort).unwrap()]
}

#[test]
fn batched_release_consumes_one_reference_per_entry() {
    for alg in algebras(BV4, 32) {
        let p = alg.set(&[1]);
        let q = alg.set(&[2]);
        alg.inc_ref(p);
        alg.dec_refs(&[p, Function::TRUE, q, p, Function::FALSE]);
        // Probe both tokens before any allocation can reuse them.
        for f in [p, q] {
            rejects(
                "cannot decrement an unreferenced or reclaimed DAG node",
                || alg.dec_ref(f),
            );
        }
    }
}

#[test]
#[cfg(debug_assertions)]
fn conditionals_check_all_three_input_roots() {
    for (alg, expected) in algebras(BV4, 32).into_iter().zip([
        "cannot use an unreferenced BDD node",
        "cannot use an unreferenced sparse node",
    ]) {
        let released = alg.set(&[1]);
        alg.dec_ref(released);
        rejects(expected, || {
            alg.ite(released, Function::TRUE, Function::TRUE)
        });
        rejects(expected, || {
            alg.ite(Function::FALSE, released, Function::TRUE)
        });
        rejects(expected, || {
            alg.ite(Function::TRUE, Function::FALSE, released)
        });
    }
}

fn quotient_inputs() -> [(Box<dyn Algebra>, Function); 2] {
    algebras(BV4, 32).map(|alg| {
        let f = alg.values(
            0,
            &[1, 3, 5],
            &[1, 2, 3],
            &mut Vec::new(),
            &mut Values::default(),
        );
        (alg, f)
    })
}

#[test]
fn quotient_merges_the_requested_classes() {
    for (alg, f) in quotient_inputs() {
        let mut scratch = Align::default();
        let classes = [0, 1, 0, 2];
        let q = alg.quotient(f, &classes, &mut scratch);
        let again = alg.quotient(f, &classes, &mut scratch);
        assert_eq!(again, q);
        for x in 0..16 {
            let expect = match x {
                1 => 1,
                5 => 2,
                _ => 0,
            };
            assert_eq!(alg.evaluate(again, &[x]), expect, "at {x}");
        }
        for root in [f, q, again] {
            alg.dec_ref(root);
        }
    }
}

#[test]
#[cfg(debug_assertions)]
fn quotient_rejects_invalid_class_maps() {
    let count = "quotient must merge input classes and leave at least two result classes";
    for (alg, f) in quotient_inputs() {
        let mut scratch = Align::default();
        for (classes, expect) in [
            (&[0, 0, 0, 0][..], count),
            (&[0, 1, 2, 3][..], count),
            (
                &[0, 2, 0, 1][..],
                "quotient classes must be numbered in first-appearance order",
            ),
            (
                &[0, 1, 2, 3, 3][..],
                "there must be one quotient class per input class",
            ),
        ] {
            rejects(expect, || alg.quotient(f, classes, &mut scratch));
        }
        alg.dec_ref(f);
    }
}

#[test]
fn an_overlapping_default_labeled_case_is_rejected() {
    for alg in algebras(BV4, 32) {
        let a = alg.set(&[1, 2, 3]);
        let b = alg.set(&[3, 4]);
        let mut labels = vec![77];
        for cases in [[(a, 0), (b, 1)], [(b, 1), (a, 0)]] {
            rejects("cases with different labels overlap", || {
                alg.cases(0, &cases, &mut labels)
            });
            assert_eq!(labels, [77]);
        }
        alg.dec_ref(a);
        alg.dec_ref(b);
    }
}

#[test]
fn values_append_only_reachable_labels_after_the_prefix() {
    // The BDD store starts without edge slots, so interning the canonical
    // branch forces a retry after both labels have been appended.
    for alg in algebras(ScalarSort::bool(), 0) {
        // Both keys are listed, so the default is unreachable.
        let mut labels = vec![77];
        let f = alg.values(9, &[0, 1], &[10, 20], &mut labels, &mut Default::default());
        assert_eq!(labels, [77, 10, 20]);
        assert_eq!([alg.evaluate(f, &[0]), alg.evaluate(f, &[1])], [0, 1]);
        alg.dec_ref(f);
    }
}
