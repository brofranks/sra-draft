//! Allocation invariants for count construction and addition.

use super::Count;

#[test]
fn zero_needs_no_digit_allocation_on_conversion_or_addition() {
    let zero = Count::from_digits(vec![0; 128]);
    assert_eq!(zero.0.capacity(), 0);
    let mut n = Count::from_digits(vec![1; 128]);
    let capacity = n.0.capacity();
    n += zero;
    assert_eq!(n.0, [1; 128]);
    assert_eq!(n.0.capacity(), capacity);
}

#[test]
fn addition_reuses_the_longer_owned_buffer() {
    let rhs = Count::from_digits(vec![u64::MAX, 1]);
    let digits = rhs.0.as_ptr();
    let sum = Count::from(1) + rhs;
    assert_eq!(sum.0, [0, 2]);
    assert_eq!(sum.0.capacity(), 2);
    assert_eq!(sum.0.as_ptr(), digits);
}
