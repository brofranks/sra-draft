//! Public-contract tests for `sra::backend`.
//!
//! The backend-independent pieces: tokens, the Boolean operation algebra, map
//! descriptions, the caller-owned scratch structures, and the enumerating
//! transfer. The transfer is fed by a listed cursor rather than a store, so no
//! backend's encoding takes part in the expectations.

use std::sync::Arc;

use sra::backend::{BoolOp, Function, MapSpec, Members, StoreCache, transfer};
use sra::scalar::ScalarSort;

const BV4: ScalarSort = ScalarSort::bit_vec(4);

/// A token that is neither constant, so the identities under test cannot fire
/// on it.
fn opaque() -> Function {
    Function::new(7)
}

/// A cursor over listed members, in the order given.
struct Listed<const N: usize>(std::vec::IntoIter<[u64; N]>);

impl<const N: usize> Members for Listed<N> {
    fn next(&mut self, out: &mut Vec<u64>) -> bool {
        self.0.next().is_some_and(|v| {
            out.clear();
            out.extend_from_slice(&v);
            true
        })
    }
}

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
    // A Rust map is partial: a rejected input has no image in either
    // direction.
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
    MapSpec::identity(1).check_domain(symbol);
    MapSpec::Rust(Arc::new(|_: &[u64], _: &mut [u64]| true)).check_domain(symbol);
    // Bit vectors keep their bitwise interpretation.
    MapSpec::Bits {
        clear: vec![1].into(),
        flip: vec![0].into(),
    }
    .check_domain(BV4);
}

#[test]
#[should_panic(expected = "symbols do not support bitwise maps")]
fn a_bitwise_map_over_symbols_is_rejected() {
    MapSpec::Bits {
        clear: vec![1].into(),
        flip: vec![0].into(),
    }
    .check_domain(ScalarSort::symbol());
}

#[test]
#[should_panic(expected = "bit map masks must encode the scalar sort")]
fn bit_map_masks_outside_the_sort_are_rejected() {
    MapSpec::Bits {
        clear: vec![0x10].into(),
        flip: vec![0].into(),
    }
    .check_domain(BV4);
}

// --- scratch ----

#[test]
fn a_store_cache_restarts_on_a_new_store_epoch_or_bound_overflow() {
    let mut cache = StoreCache::default();
    assert_eq!(cache.bound, 1 << 20);
    let cache_entry = |cache: &mut StoreCache| {
        cache.aligned.insert(
            (Function::new(2), Function::new(3)),
            (Function::new(4), 0, 1),
        );
        cache.labels.push((0, 0));
    };
    cache.restart(1, 1);
    cache_entry(&mut cache);
    cache.restart(1, 1);
    assert_eq!((cache.aligned.len(), cache.labels.len()), (1, 1));
    cache.restart(2, 1);
    assert!(cache.aligned.is_empty() && cache.labels.is_empty());
    assert_eq!((cache.store, cache.epoch), (2, 1));
    cache_entry(&mut cache);
    cache.restart(2, 2);
    assert!(cache.aligned.is_empty());
    // A cache at its bound keeps its entries, and one over it clears them while
    // retaining its allocations.
    let mut cache = StoreCache::with_bound(1);
    cache.restart(1, 1);
    cache_entry(&mut cache);
    cache.restart(1, 1);
    assert_eq!(cache.aligned.len(), 1);
    cache
        .quotients
        .insert((Function::new(2), 0), (Function::new(4), 0, 0));
    cache.restart(1, 1);
    assert!(cache.aligned.is_empty() && cache.quotients.is_empty() && cache.labels.is_empty());
    assert!(cache.aligned.capacity() > 0 && cache.labels.capacity() > 0);
}

// --- transfer ----

#[test]
fn transfer_keeps_the_members_or_the_images_the_filter_admits() {
    // `x + 4`, undefined at 3, so one member drops out of both directions.
    let map = MapSpec::Rust(Arc::new(|x: &[u64], y: &mut [u64]| {
        y[0] = x[0] + 4;
        x[0] != 3
    }));
    let members = || Listed(vec![[1], [2], [3]].into_iter());
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
fn multiword_transfer_preserves_preimage_order_and_sorts_images() {
    let sort = ScalarSort::bit_vec(65);
    let map = MapSpec::Bits {
        clear: vec![1, 0].into(),
        flip: vec![0, 1].into(),
    };
    let members = || Listed(vec![[0, 0], [1, 0], [2, 0], [0, 1]].into_iter());
    assert_eq!(
        transfer(sort, &map, &mut members(), |_| true, true),
        [0, 0, 0, 1, 2, 1]
    );
    assert_eq!(
        transfer(sort, &map, &mut members(), |y| y[0] == 0, false),
        [0, 0, 1, 0, 0, 1]
    );
}

#[test]
#[should_panic(expected = "cursor member does not encode the scalar sort")]
fn a_preimage_member_with_the_wrong_stride_is_rejected() {
    let mut members = Listed(vec![[5, 1]].into_iter());
    transfer(BV4, &MapSpec::identity(1), &mut members, |_| true, false);
}

#[test]
#[should_panic(expected = "cursor member does not encode the scalar sort")]
fn a_transfer_member_with_unused_high_bits_is_rejected() {
    let map = MapSpec::Rust(Arc::new(|_, _| panic!("map received an invalid member")));
    let mut members = Listed(vec![[16]].into_iter());
    transfer(BV4, &map, &mut members, |_| true, true);
}

#[test]
#[should_panic(expected = "map image does not encode the scalar sort")]
fn a_transfer_image_outside_the_sort_is_rejected() {
    let map = MapSpec::Rust(Arc::new(|_: &[u64], y: &mut [u64]| {
        y[0] = 99;
        true
    }));
    let mut members = Listed(vec![[1]].into_iter());
    transfer(BV4, &map, &mut members, |_| true, true);
}
