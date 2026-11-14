//! Contextual predicate recipes and Boolean operands over a common input.
//!
//! Comparisons and ranges select signed or unsigned operations from the Rust
//! endpoint type during native lowering. Mixed predicate/query disjunction
//! hides witness ranges and resolves through the ordinary lazy query
//! combinators. Native `Predicate` consumers accept only composites whose
//! leaves are native predicates.
//!
//! Literals pack to the column's width, and the endpoint then reads the packed
//! bits. An ordering therefore compares against that reading of the bit
//! pattern: `lt(200_u8)` over `i8` compares against `-56`. A consumed inclusive
//! range keeps this reading too, so `127_u8..=128` emptied from both ends
//! stores `128..=127`, which over `i8` is the whole domain.

use std::marker::PhantomData;
use std::ops::{
    BitAnd, BitOr, Bound, Not, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};

use super::{FuncDecl, Integer, Operand, Query, Same, Sorted};
use crate::context::Context;
use crate::index::IndexRoot;
use crate::pred as raw;
use crate::query::{self, IntoQuery as RawQuery, Operand as RawOperand};
use crate::row::{IntoRow, Row};
use crate::run::Exec;

/// A predicate recipe over values of interpretation `T`, lowered to a raw
/// predicate on use.
pub trait Predicate<T: Sorted> {
    /// The raw predicate builder produced by lowering a borrowed recipe.
    type Raw<'a>: raw::Build
    where
        Self: 'a;
    /// Lowers the recipe to a raw filter, selecting operations from the
    /// interpretation `T`.
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>>;
}
impl<T: Sorted, P: Predicate<T> + ?Sized> Predicate<T> for &P {
    type Raw<'a>
        = P::Raw<'a>
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        (**self).lower()
    }
}

/// A predicate recipe usable as a query operand and combinable with `&`, `|`,
/// and `!`.
#[derive(Clone)]
pub struct Filter<P>(
    /// The wrapped predicate recipe.
    pub P,
);
impl<P> Filter<P> {
    /// Conjunction with another recipe or operand over the same input;
    /// equivalent to `&`.
    #[inline(always)]
    pub fn and<B>(self, b: B) -> Conj<Self, B> {
        self & b
    }
    /// Disjunction with another recipe or operand over the same input;
    /// equivalent to `|`.
    #[inline(always)]
    pub fn or<B>(self, b: B) -> Disj<Self, B> {
        self | b
    }
}
impl<T: Sorted, P: Predicate<T>> Predicate<T> for Filter<P> {
    type Raw<'a>
        = P::Raw<'a>
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        self.0.lower()
    }
}
// Equality needs no ordering interpretation, so the raw leaf serves directly.
pub use raw::Equal;
// Boolean operands share their syntax with the raw layer. Native lowering is
// available only when every child is itself a native predicate; query
// witnesses stay in CPS.
pub use raw::{Conj, Disj, Neg};

/// Equality with `value`.
#[inline(always)]
pub fn eq<V: IntoRow>(value: V) -> Filter<Equal<V>> {
    Filter(raw::eq(value).0)
}
/// Inequality with `value`.
#[inline(always)]
pub fn ne<V: IntoRow>(value: V) -> Filter<Neg<Equal<V>>> {
    Filter(Neg(raw::eq(value).0))
}
impl<T: Sorted, V: IntoRow> Predicate<T> for Equal<V> {
    type Raw<'a>
        = &'a Self
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        raw::Filter(self)
    }
}
impl<T: Sorted, P: Predicate<T>> Predicate<T> for Neg<P> {
    type Raw<'a>
        = raw::Neg<raw::Filter<P::Raw<'a>>>
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        raw::Filter(Neg(self.0.lower()))
    }
}
macro_rules! boolean {
    ($($node:ident, $trait:ident, $method:ident);*) => {$(
        impl<T: Sorted, A: Predicate<T>, B: Predicate<T>> Predicate<T> for $node<A, B> {
            type Raw<'a> = raw::$node<raw::Filter<A::Raw<'a>>, raw::Filter<B::Raw<'a>>>
            where Self: 'a;
            #[inline(always)]
            fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
                raw::Filter($node(self.0.lower(), self.1.lower()))
            }
        }
        impl<A, B> $trait<B> for Filter<A> {
            type Output = $node<Self, B>;
            #[inline(always)]
            fn $method(self, b: B) -> Self::Output { $node(self, b) }
        }
        impl<'a, A, B> $trait<B> for &'a Filter<A> {
            type Output = $node<Self, B>;
            #[inline(always)]
            fn $method(self, b: B) -> Self::Output { $node(self, b) }
        }
    )*};
}
boolean!(Conj, BitAnd, bitand; Disj, BitOr, bitor);
impl<P> Not for Filter<P> {
    type Output = Neg<Self>;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Neg(self)
    }
}
impl<P> Not for &Filter<P> {
    type Output = Neg<Self>;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Neg(self)
    }
}

/// A membership disjunction resolves both typed operands against the
/// consuming query and accepts the domain keys either one accepts.
impl<D: Sorted, A: Operand<D>, B: Operand<D>> Operand<D> for Disj<A, B> {
    type R = D;
    type Node = query::Union<query::Join<(), A::Node>, query::Join<(), B::Node>>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        RawOperand::resolve(
            raw::Disj(self.0.resolve(cx, input), self.1.resolve(cx, input)),
            cx,
            input,
        )
    }
}
impl<D: Sorted, A: Operand<D>, B: Operand<D>> Operand<D> for Conj<A, B> {
    type R = D;
    type Node = query::Join<query::Join<(), A::Node>, B::Node>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        let a = RawQuery::with(
            RawOperand::resolve(query::id(), cx, input),
            self.0.resolve(cx, input),
        );
        let b = self.1.resolve(cx, &a.range);
        RawQuery::with(a, b)
    }
}
macro_rules! borrowed_operands {
    ($($node:ident);*) => {$(
        impl<'a, D: Sorted, A, B> Operand<D> for &'a $node<A, B>
        where &'a A: Operand<D>, &'a B: Operand<D> {
            type R = D;
            type Node = <$node<&'a A, &'a B> as Operand<D>>::Node;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
                Operand::resolve($node(&self.0, &self.1), cx, input)
            }
        }
    )*};
}
borrowed_operands!(Disj; Conj);
impl<D: Sorted, A: Operand<D>> Operand<D> for Neg<A> {
    type R = D;
    type Node = query::Diff<(), A::Node>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        RawOperand::resolve(raw::Neg(self.0.resolve(cx, input)), cx, input)
    }
}
impl<'a, D: Sorted, A> Operand<D> for &'a Neg<A>
where
    &'a A: Operand<D>,
{
    type R = D;
    type Node = <Neg<&'a A> as Operand<D>>::Node;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        Operand::resolve(Neg(&self.0), cx, input)
    }
}

impl<T: Sorted, P: Predicate<T>> Operand<T> for Filter<P> {
    type R = T;
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        self.0.lower().resolve(cx, input)
    }
}
impl<T: Sorted, P: Predicate<T>> Operand<T> for &Filter<P> {
    type R = T;
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        self.0.lower().resolve(cx, input)
    }
}

macro_rules! borrowed_recipes {
    ($($name:ident$(<$($t:ident),+>)?);* $(;)?) => {$(
        impl<'a, D: Sorted, $($($t),+,)?> Operand<D> for &&'a $name$(<$($t),+>)?
        where &'a $name$(<$($t),+>)?: Operand<D> {
            type R = <&'a $name$(<$($t),+>)? as Operand<D>>::R;
            type Node = <&'a $name$(<$($t),+>)? as Operand<D>>::Node;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
                Operand::resolve(*self, cx, input)
            }
        }
    )*};
}
borrowed_recipes!(Filter<P>; Disj<A, B>; Conj<A, B>; Neg<P>; FuncDecl<A, B>;
    Range<V>; RangeFrom<V>; RangeInclusive<V>; RangeTo<V>; RangeToInclusive<V>; RangeFull);

/// One ordering comparison against a literal, lowered as signed or unsigned
/// on demand.
pub trait Comparison {
    /// The raw predicate builder for a literal of type `V`.
    type Raw<V: IntoRow>: raw::Build;
    /// Lowers the comparison against `value` with the given signedness.
    fn lower<V: IntoRow>(value: V, signed: bool) -> raw::Filter<Self::Raw<V>>;
}
/// An ordering comparison against a literal, interpreted by the endpoint.
#[derive(Clone)]
pub struct Compare<V, O> {
    value: V,
    op: PhantomData<O>,
}
macro_rules! comparisons {
    ($($op:ident, $plain:ident, $signed:ident, $unsigned:ident, $raw_signed:ident,
        $raw_unsigned:ident, $out:ty);*) => {$(
        #[doc = concat!("The `", stringify!($plain), "` comparison.")]
        #[derive(Clone)]
        pub struct $op;
        impl Comparison for $op {
            type Raw<V: IntoRow> = $out;
            #[inline(always)]
            fn lower<V: IntoRow>(value: V, signed: bool) -> raw::Filter<Self::Raw<V>> {
                if signed { raw::$raw_signed(value) } else { raw::$raw_unsigned(value) }
            }
        }
        #[doc = concat!("The `", stringify!($plain),
            "` comparison against `value`, with signedness taken from the endpoint type.")]
        ///
        /// See the [module documentation](self) for out-of-range literals.
        #[inline(always)]
        pub fn $plain<V: IntoRow>(value: V) -> Filter<Compare<V, $op>> {
            Filter(Compare { value, op: PhantomData })
        }
        #[doc = concat!("The signed `", stringify!($plain), "` comparison against `value`.")]
        #[inline(always)]
        pub fn $signed<V: IntoRow>(value: V) -> Filter<$out> {
            Filter(raw::$raw_signed(value).0)
        }
        #[doc = concat!("The unsigned `", stringify!($plain), "` comparison against `value`.")]
        #[inline(always)]
        pub fn $unsigned<V: IntoRow>(value: V) -> Filter<$out> {
            Filter(raw::$raw_unsigned(value).0)
        }
    )*};
}
comparisons!(
    Lt, lt, slt, ult, bvslt, bvult, raw::Compare<V>;
    Le, le, sle, ule, bvsle, bvule, raw::Compare<V>;
    Gt, gt, sgt, ugt, bvsgt, bvugt, raw::Neg<raw::Compare<V>>;
    Ge, ge, sge, uge, bvsge, bvuge, raw::Neg<raw::Compare<V>>
);
impl<T: Integer, V: IntoRow, O: Comparison> Predicate<T> for Compare<V, O> {
    type Raw<'a>
        = O::Raw<&'a V>
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        O::lower(&self.value, T::SIGNED)
    }
}
impl<T: Integer, V: IntoRow> Predicate<T> for raw::Compare<V> {
    type Raw<'a>
        = &'a Self
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        raw::Filter(self)
    }
}

/// A typed predicate carried through a bound raw projection until its input
/// row is known, retyped over the projection's domain `D`.
pub struct Lower<D, R, P> {
    p: P,
    types: PhantomData<fn() -> (D, R)>,
}
impl<D, R, P: Clone> Clone for Lower<D, R, P> {
    fn clone(&self) -> Self {
        Self {
            p: self.p.clone(),
            types: PhantomData,
        }
    }
}
impl<D, R: Sorted, P: Predicate<R>> raw::Build for Lower<D, R, P> {
    #[inline(always)]
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        raw::Build::build(&self.p.lower(), input, index, positions)
    }
}
impl<T: Sorted, D: Sorted, R: Sorted, S: raw::Selector, P: Predicate<R>> Predicate<T>
    for raw::On<S, Lower<D, R, P>>
where
    T: Same<D>,
{
    type Raw<'a>
        = &'a Self
    where
        Self: 'a;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        raw::Filter(self)
    }
}

/// Membership in a finite collection of literals or in an integer range.
#[derive(Clone)]
pub struct Member<C>(C);
/// Membership in `values`, a collection of literals or an integer range.
///
/// See the [module documentation](self) for out-of-range range literals.
#[inline(always)]
pub fn is_in<C>(values: C) -> Filter<Member<C>> {
    Filter(Member(values))
}
macro_rules! finite {
    ([$($generics:tt)*] $t:ty) => {
        impl<$($generics)*, T: Sorted> Predicate<T> for Member<$t> {
            type Raw<'a> = raw::Finite<&'a [V], V> where Self: 'a;
            #[inline(always)]
            fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
                raw::is_in::<&[V]>(self.0.as_ref())
            }
        }
    };
}
finite!([V: IntoRow, const N: usize] [V; N]);
finite!([V: IntoRow] Vec<V>);
finite!(['v, V: IntoRow] &'v [V]);
finite!(['v, V: IntoRow, const N: usize] &'v [V; N]);
finite!(['v, V: IntoRow] &'v Vec<V>);
// Range shape determines which comparisons are needed; ordering is resolved
// here just as it is for standalone comparisons, before any raw predicate
// reaches the runtime.
macro_rules! ranges {
    ($($t:ident, $out:ty, |$r:ident| $p:expr);* $(;)?) => {$(
        impl<T: Integer, V: IntoRow> Predicate<T> for $t<V> {
            type Raw<'a> = $out where Self: 'a;
            #[inline(always)]
            fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
                let $r = self;
                $p
            }
        }
        impl<T: Integer, V: IntoRow> Predicate<T> for Member<$t<V>> {
            type Raw<'a> = $out where Self: 'a;
            #[inline(always)]
            fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
                <$t<V> as Predicate<T>>::lower(&self.0)
            }
        }
        impl<T: Integer, V: IntoRow> Operand<T> for &$t<V> {
            type R = T;
            type Node = IndexRoot;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
                <$t<V> as Predicate<T>>::lower(self).resolve(cx, input)
            }
        }
        impl<T: Integer, V: IntoRow> Operand<T> for $t<V> {
            type R = T;
            type Node = IndexRoot;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
                Operand::<T>::resolve(&self, cx, input)
            }
        }
    )*};
}
ranges!(
    Range, raw::Conj<raw::Filter<raw::Neg<raw::Compare<&'a V>>>, raw::Filter<raw::Compare<&'a V>>>,
        |r| raw::Filter(Ge::lower(&r.start, T::SIGNED) & Lt::lower(&r.end, T::SIGNED));
    RangeFrom, raw::Neg<raw::Compare<&'a V>>, |r| Ge::lower(&r.start, T::SIGNED);
    RangeInclusive,
        raw::Conj<raw::Filter<raw::Neg<raw::Compare<&'a V>>>, raw::Filter<raw::Compare<&'a V>>>,
        // An iterated range can report an excluded end, and exhaustion leaves
        // the stored endpoints unspecified, so use that bound. Every other
        // inclusive range keeps its end, including one drained by `next` to a
        // start past its end, whose comparisons contradict when both stored
        // literals read in the same order under the endpoint.
        |r| raw::Filter(Ge::lower(r.start(), T::SIGNED)
            & match r.end_bound() {
                Bound::Excluded(e) => Lt::lower(e, T::SIGNED),
                _ => Le::lower(r.end(), T::SIGNED),
            });
    RangeTo, raw::Compare<&'a V>, |r| Lt::lower(&r.end, T::SIGNED);
    RangeToInclusive, raw::Compare<&'a V>, |r| Le::lower(&r.end, T::SIGNED);
);
impl<T: Integer> Predicate<T> for RangeFull {
    type Raw<'a> = RangeFull;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        raw::Filter(..)
    }
}
impl<T: Integer> Predicate<T> for Member<RangeFull> {
    type Raw<'a> = RangeFull;
    #[inline(always)]
    fn lower<'a>(&'a self) -> raw::Filter<Self::Raw<'a>> {
        <RangeFull as Predicate<T>>::lower(&self.0)
    }
}
impl<T: Integer> Operand<T> for &RangeFull {
    type R = T;
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        Operand::<T>::resolve(*self, cx, input)
    }
}
impl<T: Integer> Operand<T> for RangeFull {
    type R = T;
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        <Filter<Member<Self>> as Operand<T>>::resolve(is_in(self), cx, input)
    }
}

// Declarations and composed projections use the same typed comparisons. Raw
// test() checks that a query is a source-free structural projection and keeps
// the declaration's context identity. `$notes` extends the `test` documentation
// and `$panics` carries each method's construction-time `# Panics` section.
macro_rules! projection_methods {
    ($s:ty, $notes:literal, $panics:literal) => {
        /// Tests a structural projection's range, yielding a predicate over
        /// its domain.
        ///
        #[doc = $notes]
        #[doc = $panics]
        #[inline(always)]
        pub fn test<P: Predicate<R>>(&self, p: P) -> Filter<raw::On<$s, Lower<D, R, P>>> {
            Filter(self.raw.test(Lower { p, types: PhantomData }).0)
        }
        projection_methods!($s, $panics; eq: Equal<V>, ne: Neg<Equal<V>>,
            lt: Compare<V,Lt>, le: Compare<V,Le>, gt: Compare<V,Gt>, ge: Compare<V,Ge>,
            slt: raw::Compare<V>, sle: raw::Compare<V>, sgt: Neg<raw::Compare<V>>,
            sge: Neg<raw::Compare<V>>, ult: raw::Compare<V>, ule: raw::Compare<V>,
            ugt: Neg<raw::Compare<V>>, uge: Neg<raw::Compare<V>>);
        /// Tests the range for membership in `values` with [`is_in`].
        #[doc = $panics]
        #[inline(always)]
        pub fn is_in<C>(&self, values: C) -> Filter<raw::On<$s, Lower<D, R, Member<C>>>>
        where Member<C>: Predicate<R> { self.test(is_in(values).0) }
    };
    ($s:ty, $panics:literal; $($name:ident: $out:ty),*) => {$(
        #[doc = concat!("Tests the range with [`", stringify!($name), "`].")]
        #[doc = $panics]
        #[inline(always)]
        pub fn $name<V: IntoRow>(
            &self,
            value: V,
        ) -> Filter<raw::On<$s, Lower<D, R, $out>>>
        where $out: Predicate<R> { self.test($name(value).0) }
    )*};
}
impl<D: Sorted, R: Sorted> FuncDecl<D, R> {
    projection_methods!(
        crate::FuncDecl,
        "Selector `eq` and `ne` construct predicates. Declaration comparisons use \
         `==`, `!=`, or qualified [`PartialEq`] methods.",
        ""
    );
}
impl<D: Sorted, R: Sorted, A: Exec> Query<D, R, A> {
    projection_methods!(
        raw::Projection,
        "A stored or computed query restricts its range through `with` instead.\n\n\
         Selector `eq` and `ne` construct predicates. Semantic query comparisons \
         use `==`, `!=`, or qualified [`PartialEq`] methods.",
        "\n# Panics\n\nPanics if this query executes a source, filter, or map, or \
         its endpoint wiring does not project the complete input."
    );
}

#[cfg(test)]
#[path = "../../tests/unit/typed/pred.rs"]
mod tests;
