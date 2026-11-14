//! Concrete custom-width integers.
//!
//! Signedness is a Rust interpretation of
//! the same runtime bits. IPv4 addresses use the same scalar representation
//! contract without integer operations.

use std::cmp::Ordering;
use std::net::Ipv4Addr;

use super::{Cons, Integer, Nil, Sorted, View, expr};
use crate::ast::Sort;
use crate::context::Context;
use crate::row::{self, IntoRow, RowRef, RowSink};
use crate::scalar::ScalarSort;

impl Sorted for Ipv4Addr {
    type Shape = Cons<Self, Nil>;
    fn sort(cx: &Context) -> Sort {
        cx.bv_sort(32)
    }
}
impl View for Ipv4Addr {
    type Expr<D: Sorted> = expr::Expr<D, Self>;
    #[inline(always)]
    fn view<D: Sorted>(input: expr::Expr<D, Self>) -> Self::Expr<D> {
        input
    }
}
impl IntoRow for Ipv4Addr {
    fn passed(start: &Self, end: &Self) -> bool {
        start > end
    }

    #[inline(always)]
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert_eq!(
            row.sort.scalar(),
            ScalarSort::bit_vec(32),
            "IPv4 value requires 32 bits"
        );
        u32::from(*self).append(row, out);
    }
}

/// An unsigned value with exactly `N` canonical bits.
///
/// Construction checks
/// width and high bits. Words are available through immutable `AsRef<[u64]>`;
/// mutation cannot bypass validation.
///
/// ```compile_fail
/// use sra::typed::UInt;
/// let invalid = UInt::<8>(vec![256].into_boxed_slice());
/// ```
/// ```compile_fail
/// use sra::typed::UInt;
/// let mut value = UInt::<8>::new(1);
/// value.as_ref()[0] = 256;
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct UInt<const N: u32>(Box<[u64]>);
/// A two's-complement signed integer of exactly `N` bits, stored as its
/// unsigned bit pattern.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Int<const N: u32>(
    /// The unsigned bit pattern.
    pub UInt<N>,
);

fn words<const N: u32>(value: u128, negative: bool) -> Box<[u64]> {
    let sort = ScalarSort::bit_vec(N);
    let mut out = Vec::with_capacity(sort.words());
    row::integer(value, negative, sort, &mut out);
    out.into()
}

impl<const N: u32> UInt<N> {
    /// Packs `value`, which must fit in `N` bits.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero or `value` does not fit in `N` bits.
    pub fn new(value: u128) -> Self {
        Self(words::<N>(value, false))
    }
    /// The value with these exact canonical little-endian words, whose unused
    /// high bits are zero.
    ///
    /// # Panics
    ///
    /// Panics if the words do not encode a value of width `N`.
    pub fn from_words(words: impl Into<Box<[u64]>>) -> Self {
        let words = words.into();
        assert!(
            ScalarSort::bit_vec(N).accepts(&words),
            "words do not encode this integer width"
        );
        Self(words)
    }
}
impl<const N: u32> Int<N> {
    /// Packs `value` in two's complement, which must fit in `N` bits
    /// including the sign.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero or `value` does not fit in `N` signed bits.
    pub fn new(value: i128) -> Self {
        assert!(N > 0, "bit-vector sorts must have nonzero width");
        // The bits above the sign bit must all equal it.
        assert!(
            value.checked_shr(N - 1).unwrap_or(0).wrapping_add(1) as u128 <= 1,
            "signed integer exceeds its width"
        );
        Self(UInt(words::<N>(value as u128, value < 0)))
    }
    /// The value with these exact two's-complement little-endian words, whose
    /// unused high bits are zero.
    ///
    /// # Panics
    ///
    /// Panics if the words do not encode a value of width `N`.
    pub fn from_words(words: impl Into<Box<[u64]>>) -> Self {
        Self(UInt::from_words(words))
    }
}
impl<const N: u32> AsRef<[u64]> for UInt<N> {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}
impl<const N: u32> AsRef<[u64]> for Int<N> {
    fn as_ref(&self) -> &[u64] {
        self.0.as_ref()
    }
}
impl<const N: u32> Ord for UInt<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        crate::scalar::compare(&self.0, &other.0)
    }
}
impl<const N: u32> Ord for Int<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        let (w, bit) = ScalarSort::bit_vec(N).sign_bit();
        let negative = |v: &Self| v.0.0[w] & bit != 0;
        negative(other)
            .cmp(&negative(self))
            .then_with(|| self.0.cmp(&other.0))
    }
}
macro_rules! integers {
    ($($t:ident: $signed:expr),*) => {$(
        impl<const N: u32> Sorted for $t<N> {
            type Shape = Cons<Self, Nil>;
            fn sort(cx: &Context) -> Sort { cx.bv_sort(N) }
        }
        impl<const N: u32> View for $t<N> {
            type Expr<D: Sorted> = expr::Expr<D, Self>;
            #[inline(always)]
            fn view<D: Sorted>(input: expr::Expr<D, Self>) -> Self::Expr<D> { input }
        }
        impl<const N: u32> Integer for $t<N> { const SIGNED: bool = $signed; }
        impl<const N: u32> IntoRow for $t<N> {
            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                assert_eq!(
                    row.sort.scalar(),
                    ScalarSort::bit_vec(N),
                    "integer value requires its declared width"
                );
                self.as_ref().append(row, out);
            }
        }
        impl<const N: u32> PartialOrd for $t<N> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
    )*};
}
integers!(UInt: false, Int: true);

#[cfg(test)]
#[path = "../../tests/unit/typed/value.rs"]
mod tests;
