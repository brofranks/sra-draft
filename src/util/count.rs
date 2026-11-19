//! Exact natural numbers for the sizes of relations and predicates.

use std::fmt::{self, Write};
use std::iter::Product;
use std::mem;
use std::ops::{Add, AddAssign, Mul, Shl};

/// Adds `(digits * factor) << shift` into `sum`, both little-endian digits.
///
/// The caller sizes `sum` to hold the result, so many terms can accumulate in
/// one buffer and convert once with [`Count::from_digits`]. Fusing the scale
/// and shift into the addition avoids building a temporary number per term.
/// The work is proportional to the term and its carry, whatever the size of
/// `sum`.
///
/// # Panics
///
/// Panics if the result exceeds `sum`, which then holds the low digits
/// written before the overflow was found.
pub fn add_scaled(sum: &mut [u64], digits: &[u64], factor: u64, shift: u32) {
    if factor == 0 {
        return;
    }
    let bits = shift % u64::BITS;
    // A shift past the buffer leaves no digits, which only a zero term fits.
    let sum = sum
        .get_mut((shift / u64::BITS) as usize..)
        .unwrap_or_default();
    let (mut mul, mut add, mut prev) = (0_u128, 0_u128, 0_u64);
    for (i, s) in sum.iter_mut().enumerate() {
        if i >= digits.len() && mul == 0 && add == 0 && prev == 0 {
            break;
        }
        let t = u128::from(digits.get(i).copied().unwrap_or(0)) * u128::from(factor) + mul;
        mul = t >> 64;
        let p = t as u64;
        let d = if bits == 0 {
            p
        } else {
            (p << bits) | (prev >> (u64::BITS - bits))
        };
        prev = p;
        let u = u128::from(*s) + u128::from(d) + add;
        *s = u as u64;
        add = u >> 64;
    }
    let rest = digits.get(sum.len()..).unwrap_or(&[]);
    assert!(
        mul == 0
            && add == 0
            && (bits == 0 || prev >> (u64::BITS - bits) == 0)
            && rest.iter().all(|&d| d == 0),
        "scaled sum exceeds its buffer"
    );
}

/// An exact natural number, the size of a relation or predicate.
///
/// Digits are little-endian `u64` values whose most significant digit is
/// nonzero, so each number has one representation and the derived equality and
/// hash are exact. Zero has no digits.
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct Count(Vec<u64>);

impl Count {
    /// The number zero.
    pub const ZERO: Self = Self(Vec::new());

    /// The number with little-endian `digits`, ignoring leading zero digits.
    #[must_use]
    pub fn from_digits(mut digits: Vec<u64>) -> Self {
        while digits.last() == Some(&0) {
            digits.pop();
        }
        Self(digits)
    }
}

impl From<u64> for Count {
    fn from(value: u64) -> Self {
        Self(if value == 0 { Vec::new() } else { vec![value] })
    }
}

impl PartialEq<u64> for Count {
    fn eq(&self, other: &u64) -> bool {
        match self.0[..] {
            [] => *other == 0,
            [d] => d == *other,
            _ => false,
        }
    }
}

impl AddAssign for Count {
    fn add_assign(&mut self, rhs: Self) {
        let mut sum = mem::take(&mut self.0);
        // One digit beyond the longer operand holds the carry.
        sum.resize(sum.len().max(rhs.0.len()) + 1, 0);
        add_scaled(&mut sum, &rhs.0, 1, 0);
        *self = Self::from_digits(sum);
    }
}

impl Add for Count {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl Mul for &Count {
    type Output = Count;

    /// Schoolbook multiplication, quadratic in the number of digits.
    fn mul(self, rhs: &Count) -> Count {
        if self.0.is_empty() || rhs.0.is_empty() {
            return Count::ZERO;
        }
        // The product of numbers with n and m digits has at most n + m.
        let mut out = vec![0; self.0.len() + rhs.0.len()];
        for (j, &d) in rhs.0.iter().enumerate() {
            add_scaled(&mut out[j..], &self.0, d, 0);
        }
        Count::from_digits(out)
    }
}

impl Product for Count {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| &a * &b).unwrap_or_else(|| Self::from(1))
    }
}

impl Shl<u32> for Count {
    type Output = Self;

    fn shl(self, n: u32) -> Self {
        if n == 0 || self.0.is_empty() {
            return self;
        }
        // One digit beyond the whole-word shift holds the shifted-out bits.
        let mut out = vec![0; self.0.len() + (n / u64::BITS) as usize + 1];
        add_scaled(&mut out, &self.0, 1, n);
        Self::from_digits(out)
    }
}

impl fmt::Display for Count {
    /// Formats the number in decimal, honoring width and fill like an integer.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The largest power of ten below 2^64, so each remainder is one group
        // of 19 decimal digits.
        const GROUP: u128 = 10_000_000_000_000_000_000;
        let mut digits = self.0.clone();
        let mut groups = Vec::new();
        while !digits.is_empty() {
            let mut rem = 0_u128;
            for d in digits.iter_mut().rev() {
                let cur = (rem << 64) | u128::from(*d);
                *d = (cur / GROUP) as u64;
                rem = cur % GROUP;
            }
            // Dividing by less than 2^64 removes at most one digit.
            if digits.last() == Some(&0) {
                digits.pop();
            }
            groups.push(rem as u64);
        }
        let mut s = groups.pop().unwrap_or(0).to_string();
        for g in groups.iter().rev() {
            write!(s, "{g:019}")?;
        }
        f.pad_integral(true, "", &s)
    }
}

impl fmt::Debug for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
