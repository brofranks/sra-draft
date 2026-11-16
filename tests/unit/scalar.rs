//! Unit tests for `src/scalar.rs`.
//!
//! Expected encodings are written out as literal words rather than derived
//! through the functions under test.

use std::cmp::Ordering;

use super::{ScalarKind, ScalarSort, compare, sort_records, successor};

// --- sorts ----

#[test]
fn scalar_sorts_describe_their_encoding() {
    assert_eq!(ScalarSort::bool().kind(), ScalarKind::Bool);
    assert_eq!(ScalarSort::bool().width(), 1);
    assert_eq!(ScalarSort::bool().bit_width(), None);
    assert_eq!(ScalarSort::symbol().kind(), ScalarKind::Symbol);
    assert_eq!(ScalarSort::symbol().width(), 64);
    assert_eq!(ScalarSort::symbol().bit_width(), None);
    assert_eq!(ScalarSort::bit_vec(70).kind(), ScalarKind::BitVec);
    assert_eq!(ScalarSort::bit_vec(70).bit_width(), Some(70));
    assert_eq!(ScalarSort::bit_vec(70).words(), 2);
    assert_eq!(ScalarSort::bit_vec(70).top_mask(), 0x3f);
    assert_eq!(ScalarSort::bit_vec(64).words(), 1);
    assert_eq!(ScalarSort::bit_vec(64).top_mask(), u64::MAX);
    assert_ne!(ScalarSort::bit_vec(1), ScalarSort::bool());
}

#[test]
fn sorts_accept_exactly_their_encodings() {
    assert!(ScalarSort::bit_vec(8).accepts(&[255]));
    assert!(!ScalarSort::bit_vec(8).accepts(&[256]));
    assert!(!ScalarSort::bit_vec(8).accepts(&[1, 0]));
    assert!(ScalarSort::bit_vec(70).accepts(&[u64::MAX, 0x3f]));
    assert!(!ScalarSort::bit_vec(70).accepts(&[u64::MAX, 0x40]));
    assert!(!ScalarSort::bit_vec(70).accepts(&[u64::MAX]));
    assert!(ScalarSort::bool().accepts(&[1]));
    assert!(!ScalarSort::bool().accepts(&[2]));
    assert!(ScalarSort::symbol().accepts(&[u64::MAX]));
}

#[test]
#[should_panic(expected = "bit-vector sorts must have nonzero width")]
fn zero_width_bit_vector_is_rejected() {
    ScalarSort::bit_vec(0);
}

// --- encodings ----

#[test]
fn encodings_compare_numerically() {
    assert_eq!(compare(&[2], &[3]), Ordering::Less);
    assert_eq!(compare(&[3], &[3]), Ordering::Equal);
    assert_eq!(compare(&[u64::MAX, 0], &[0, 1]), Ordering::Less);
    assert_eq!(compare(&[0, 2], &[u64::MAX, 1]), Ordering::Greater);
    assert_eq!(compare(&[1, 1], &[1, 1]), Ordering::Equal);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "compared encodings must have equal width")]
fn comparing_encodings_of_different_width_is_rejected() {
    compare(&[1], &[1, 0]);
}

#[test]
fn sorted_records_are_numeric_and_distinct() {
    let mut packed = vec![0, 1, u64::MAX, 0, 0, 1, 1, 0];
    sort_records(&mut packed, 2);
    assert_eq!(packed, [1, 0, u64::MAX, 0, 0, 1]);
    let mut packed = vec![5, 3, 5, 1];
    sort_records(&mut packed, 1);
    assert_eq!(packed, [1, 3, 5]);
    let mut packed = Vec::new();
    sort_records(&mut packed, 1);
    assert!(packed.is_empty());
    sort_records(&mut packed, 2);
    assert!(packed.is_empty());
}

#[test]
#[should_panic(expected = "record word stride must be nonzero")]
fn a_zero_record_stride_is_rejected() {
    sort_records(&mut Vec::new(), 0);
}

#[test]
#[should_panic(expected = "packed values must contain complete records")]
fn an_incomplete_record_is_rejected() {
    sort_records(&mut vec![1, 0, 2], 2);
}

#[test]
fn successors_carry_across_words_and_stop_at_the_width() {
    assert!(!successor(&mut [], ScalarSort::bool()));
    let mut value = vec![u64::MAX, 0];
    assert!(successor(&mut value, ScalarSort::bit_vec(65)));
    assert_eq!(value, [0, 1]);
    let mut value = vec![u64::MAX, 1];
    assert!(!successor(&mut value, ScalarSort::bit_vec(65)));
    let mut value = vec![u64::MAX];
    assert!(!successor(&mut value, ScalarSort::bit_vec(64)));
    let mut value = vec![6];
    assert!(successor(&mut value, ScalarSort::bit_vec(3)));
    assert_eq!(value, [7]);
    assert!(!successor(&mut value, ScalarSort::bit_vec(3)));
}
