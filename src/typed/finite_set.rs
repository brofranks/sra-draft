//! One concrete subset of a fixed universe, represented by one membership
//! bit per member.
//!
//! A relation over these values represents possible sets; it
//! is not itself a `FiniteSet`.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use super::{Cons, Nil, Sorted, View, expr};
use crate::ast::Sort;
use crate::backend::BoolOp;
use crate::context::Context;
use crate::row::{IntoRow, RowRef, RowSink};
use crate::scalar::ScalarSort;

/// An application-defined, nonempty universe with a stable bijection onto
/// `0..LEN`.
///
/// Equal members must map to the same index, distinct members to
/// distinct indices, and every index must name a member. Return `None` for
/// values outside the universe. The mapping receives the identity of the
/// set's creation context, so members that carry a context identity can be
/// checked against it (see `Capacity`); a universe of plain values ignores
/// it. The universe identity is `Self` in query composition, and positions
/// must remain stable as dictionaries grow. Row packing checks the context and
/// width, so a same-width set of another universe packs as this universe's
/// mask. These semantic requirements belong to the implementation; call sites
/// check index bounds.
pub trait Universe {
    /// The Rust type of a member.
    type Member;
    /// The number of members, and the width of each membership mask.
    const LEN: u32;
    /// The position of `member` in `0..LEN`, or `None` outside the universe,
    /// for a set created in the context identified by `context`.
    fn index(context: u64, member: &Self::Member) -> Option<u32>;
}

/// A canonical membership mask, including widths above 64 bits.
///
/// Every mask
/// is a valid set. Use an application alias such as
/// `type CommunitySet = FiniteSet<CommunityUniverse>`. The universe defines
/// exactly the supported member positions; no implicit "other" class exists.
/// Constructors take their context, and concrete member operations resolve
/// members against that context. Raw words explicitly erase ownership.
pub struct FiniteSet<U> {
    words: Box<[u64]>,
    context: u64,
    universe: PhantomData<fn() -> U>,
}

impl<U: Universe> FiniteSet<U> {
    /// The set of `cx` with these canonical little-endian words, whose unused
    /// high bits are zero.
    ///
    /// # Panics
    ///
    /// Panics if the words do not encode a set of this universe's width.
    pub fn from_words(cx: &Context, words: impl Into<Box<[u64]>>) -> Self {
        let words = words.into();
        assert!(
            ScalarSort::bit_vec(U::LEN).accepts(&words),
            "words do not encode this set universe"
        );
        Self {
            words,
            context: cx.id,
            universe: PhantomData,
        }
    }

    /// The set with no members.
    ///
    /// # Panics
    ///
    /// Panics if the universe is empty.
    pub fn empty(cx: &Context) -> Self {
        Self::from_words(cx, vec![0; ScalarSort::bit_vec(U::LEN).words()])
    }

    /// The set holding every member of the universe.
    ///
    /// # Panics
    ///
    /// Panics if the universe is empty.
    pub fn full(cx: &Context) -> Self {
        let sort = ScalarSort::bit_vec(U::LEN);
        let mut words = vec![u64::MAX; sort.words()];
        *words.last_mut().unwrap() &= sort.top_mask();
        Self::from_words(cx, words)
    }

    /// The set holding exactly the given members.
    ///
    /// # Panics
    ///
    /// Panics if the universe is empty, a member is outside the universe, its
    /// index exceeds the declared length, or a [`super::Capacity`] member
    /// belongs to another context, as in [`Self::insert`].
    pub fn from_members(cx: &Context, members: impl IntoIterator<Item = U::Member>) -> Self {
        let mut set = Self::empty(cx);
        for member in members {
            set.insert(member);
        }
        set
    }

    fn combine(&self, other: &Self, op: BoolOp) -> Self {
        assert_eq!(
            self.context, other.context,
            "set belongs to another context"
        );
        let mut set = self.clone();
        for (a, b) in set.words.iter_mut().zip(other.words.iter()) {
            *a = op.evaluate_mask(*a, *b);
        }
        set
    }

    /// The members of either set.
    ///
    /// # Panics
    ///
    /// Panics if the sets belong to different contexts.
    pub fn union_with(&self, other: &Self) -> Self {
        self.combine(other, BoolOp::Or)
    }

    /// The members of both sets.
    ///
    /// # Panics
    ///
    /// Panics if the sets belong to different contexts.
    pub fn intersect_with(&self, other: &Self) -> Self {
        self.combine(other, BoolOp::And)
    }

    fn bit(&self, member: U::Member) -> (usize, u64) {
        let i = U::index(self.context, &member).expect("member must belong to this set universe");
        assert!(i < U::LEN, "universe index exceeds its declared length");
        ((i / 64) as usize, 1_u64 << (i % 64))
    }

    /// Whether `member` is in the set.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    pub fn contains(&self, member: U::Member) -> bool {
        let (i, mask) = self.bit(member);
        self.words[i] & mask != 0
    }

    /// Adds `member` to the set.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    pub fn insert(&mut self, member: U::Member) {
        let (i, mask) = self.bit(member);
        self.words[i] |= mask;
    }

    /// Removes `member` from the set.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    pub fn remove(&mut self, member: U::Member) {
        let (i, mask) = self.bit(member);
        self.words[i] &= !mask;
    }
}

impl<U> Clone for FiniteSet<U> {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
            context: self.context,
            universe: PhantomData,
        }
    }
}
impl<U> PartialEq for FiniteSet<U> {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.words == other.words
    }
}
impl<U> Eq for FiniteSet<U> {}
impl<U> Hash for FiniteSet<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.context, &self.words).hash(state);
    }
}
impl<U> fmt::Debug for FiniteSet<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteSet")
            .field("context", &self.context)
            .field("words", &self.words)
            .finish()
    }
}
impl<U> AsRef<[u64]> for FiniteSet<U> {
    fn as_ref(&self) -> &[u64] {
        &self.words
    }
}
impl<U: Universe> Sorted for FiniteSet<U> {
    type Shape = Cons<Self, Nil>;
    fn sort(cx: &Context) -> Sort {
        cx.bv_sort(U::LEN)
    }
}
impl<U: Universe> View for FiniteSet<U> {
    type Expr<D: Sorted> = expr::Expr<D, Self>;
    #[inline(always)]
    fn view<D: Sorted>(input: expr::Expr<D, Self>) -> Self::Expr<D> {
        input
    }
}
impl<U: Universe> IntoRow for FiniteSet<U> {
    #[inline(always)]
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert_eq!(self.context, row.cx.id, "set belongs to another context");
        assert_eq!(
            row.sort.scalar(),
            ScalarSort::bit_vec(U::LEN),
            "set value requires its universe width"
        );
        self.words.as_ref().append(row, out);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/typed/finite_set.rs"]
mod tests;
