//! Public-contract tests for `sra::util::count`.
//!
//! Fixed cases pin scaled accumulation, normalization, carries, products,
//! shifts across word boundaries, and decimal formatting.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use sra::util::count::{self, Count};

// --- scaled sums ----

#[test]
fn scaled_sums_carry_and_shift() {
    let max = u64::MAX;
    let mut sum = [0; 4];
    count::add_scaled(&mut sum, &[max], max, 0);
    // (2^64 - 1)^2 is 2^128 - 2^65 + 1.
    assert_eq!(sum, [1, max - 1, 0, 0]);
    count::add_scaled(&mut sum, &[max], 1, 0);
    assert_eq!(sum, [0, max, 0, 0]);
    count::add_scaled(&mut sum, &[1], 1, 64);
    assert_eq!(sum, [0, 0, 1, 0]);
    count::add_scaled(&mut sum, &[max, max], 0, 64);
    assert_eq!(sum, [0, 0, 1, 0]);
    for (shift, expected) in [
        (0, [3, 0, 0]),
        (63, [1 << 63, 1, 0]),
        (64, [0, 3, 0]),
        (65, [0, 6, 0]),
    ] {
        let mut sum = [0; 3];
        count::add_scaled(&mut sum, &[3], 1, shift);
        assert_eq!(sum, expected, "3 << {shift}");
    }
}

#[test]
fn scaled_sums_combine_digits_shift_and_carry() {
    let max = u64::MAX;
    // The first addition makes 2^256 - 2^128 + 48.
    // Adding (2^128 - 1) * 3 << 4 then leaves 2^256 + 47 * 2^128.
    let mut sum = [47, 0, max, max, 0];
    count::add_scaled(&mut sum, &[1], 1, 0);
    count::add_scaled(&mut sum, &[max, max], 3, 4);
    assert_eq!(sum, [0, 0, 47, 0, 1]);
}

#[test]
fn zero_terms_fit_any_buffer() {
    let mut sum = [7];
    count::add_scaled(&mut sum, &[1], 0, 128);
    count::add_scaled(&mut sum, &[0, 0], 5, 640);
    count::add_scaled(&mut sum, &[], 5, 64);
    assert_eq!(sum, [7]);
}

#[test]
fn scaled_sums_fit_exact_capacity() {
    let mut sum = [1];
    count::add_scaled(&mut sum, &[u64::MAX / 4], 2, 1);
    assert_eq!(sum, [u64::MAX - 2]);
    count::add_scaled(&mut sum, &[2], 1, 0);
    assert_eq!(sum, [u64::MAX]);
}

#[test]
#[should_panic(expected = "scaled sum exceeds its buffer")]
fn scaled_sum_overflow_is_rejected() {
    let mut sum = [u64::MAX; 2];
    let panic = catch_unwind(AssertUnwindSafe(|| count::add_scaled(&mut sum, &[2], 1, 0)))
        .expect_err("adding two to the maximum two-digit value must overflow");
    assert_eq!(sum, [1, 0]);
    resume_unwind(panic);
}

#[test]
#[should_panic(expected = "scaled sum exceeds its buffer")]
fn scaled_sum_multiplication_overflow_is_rejected() {
    count::add_scaled(&mut [0], &[u64::MAX], 2, 0);
}

#[test]
#[should_panic(expected = "scaled sum exceeds its buffer")]
fn scaled_sum_residual_shift_overflow_is_rejected() {
    count::add_scaled(&mut [0], &[1 << 63], 1, 1);
}

#[test]
#[should_panic(expected = "scaled sum exceeds its buffer")]
fn scaled_sum_shifted_beyond_its_buffer_is_rejected() {
    count::add_scaled(&mut [0; 2], &[1], 1, 192);
}

// --- counts ----

#[test]
fn zero_has_one_representation() {
    assert_eq!(Count::ZERO, Count::from(0));
    assert_eq!(Count::default(), Count::ZERO);
    assert_eq!(Count::ZERO, 0);
    assert_eq!(&Count::from(7) * &Count::ZERO, Count::ZERO);
    assert_eq!(&Count::from(u64::MAX) * &Count::ZERO, Count::ZERO);
    assert_eq!(Count::ZERO << 200, Count::ZERO);
    assert_eq!(Count::ZERO.shift(u64::MAX), Count::ZERO);
    assert_eq!(Count::ZERO.to_string(), "0");
}

#[test]
fn digits_ignore_leading_zeros() {
    assert_eq!(Count::from_digits(vec![]), Count::ZERO);
    assert_eq!(Count::from_digits(vec![0, 0]), Count::ZERO);
    assert_eq!(Count::from_digits(vec![5, 0, 0]), 5);
    assert_eq!(Count::from_digits(vec![0, 1, 0]), Count::from(1) << 64);
    assert_ne!(Count::from(1) << 64, 0);
}

#[test]
fn carries_cross_word_boundaries() {
    let max = Count::from(u64::MAX);
    assert_eq!(
        (max.clone() + Count::from(1)).to_string(),
        "18446744073709551616"
    );
    assert_eq!(
        (max.clone() + max.clone()).to_string(),
        "36893488147419103230"
    );
    assert_eq!(
        Count::from_digits(vec![u64::MAX; 3]) + Count::from_digits(vec![1, 1]),
        Count::from_digits(vec![0, 1, 0, 1])
    );
    assert_eq!(
        (&max * &max).to_string(),
        "340282366920938463426481119284349108225"
    );
}

#[test]
fn products_fold_from_one() {
    assert_eq!(std::iter::empty::<Count>().product::<Count>(), 1);
    let p: Count = [2, 3, 7].into_iter().map(Count::from).product();
    assert_eq!(p, 42);
}

#[test]
fn products_commute_across_lengths() {
    let long = (Count::from(u64::MAX) << 300) + Count::from(u64::MAX);
    let short = Count::from(u64::MAX);
    assert_eq!(&long * &short, &short * &long);
    // x * (2^64 - 1) + x is x << 64.
    assert_eq!(&long * &short + long.clone(), long << 64);
}

#[test]
fn wide_products_preserve_zero_and_sparse_digits() {
    let mut digits = vec![0; 65];
    digits[64] = 1;
    let n = Count::from_digits(digits);
    assert_eq!(&n * &Count::ZERO, Count::ZERO);
    assert_eq!(&Count::ZERO * &n, Count::ZERO);
    let mut expected = vec![0; 129];
    expected[128] = 1;
    assert_eq!(&n * &n, Count::from_digits(expected));
}

#[test]
fn dense_products_carry_across_multiple_words() {
    let n = Count::from_digits(vec![u64::MAX; 2]);
    // (2^128 - 1)^2 = 2^256 - 2^129 + 1.
    assert_eq!(
        &n * &n,
        Count::from_digits(vec![1, 0, u64::MAX - 1, u64::MAX])
    );
}

#[test]
fn arithmetic_matches_native_integers() {
    for a in [0, 1, 3, 1 << 63, u64::MAX] {
        for b in [0, 1, 7, 1 << 63, u64::MAX] {
            assert_eq!(
                (Count::from(a) + Count::from(b)).to_string(),
                (u128::from(a) + u128::from(b)).to_string()
            );
            assert_eq!(
                (&Count::from(a) * &Count::from(b)).to_string(),
                (u128::from(a) * u128::from(b)).to_string()
            );
        }
    }
    for (initial, value, factor, shift) in [
        (7, 0, u64::MAX, 65),
        (7, u64::MAX, 0, 65),
        (u64::MAX, u64::MAX, 1, 0),
        (u64::MAX, u64::MAX, u64::MAX, 0),
        (1, u64::MAX, 1, 64),
        (7, u64::MAX / 8, 3, 65),
    ] {
        let mut sum = [initial, 0];
        count::add_scaled(&mut sum, &[value], factor, shift);
        assert_eq!(
            u128::from(sum[0]) | (u128::from(sum[1]) << 64),
            u128::from(initial) + ((u128::from(value) * u128::from(factor)) << shift)
        );
    }
}

#[test]
fn shifts_move_bits_and_words() {
    for (n, s) in [
        (0, "3"),
        (63, "27670116110564327424"),
        (64, "55340232221128654848"),
        (65, "110680464442257309696"),
        (
            200,
            "4820814132776970826625886277023487807566608981348378505904128",
        ),
    ] {
        assert_eq!((Count::from(3) << n).to_string(), s, "3 << {n}");
    }
}

// --- formatting ----

#[test]
fn display_pads_like_an_integer() {
    assert_eq!(format!("{:>6}", Count::from(42)), "    42");
    assert_eq!(format!("{:06}", Count::from(42)), "000042");
    assert_eq!(
        format!("{:?}", Count::from(1) << 64),
        "18446744073709551616"
    );
    let tens = (0..40).fold(Count::from(1), |n, _| &n * &Count::from(10));
    assert_eq!(tens.to_string(), format!("1{}", "0".repeat(40)));
}
