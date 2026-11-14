//! Functional expression sugar over existing typed query combinators.
//!
//! `q.map(|x| (x & 255) ^ 16)` transforms the range while retaining the key.
//! `q.filt(|x| (x | 16).gt(20))` tests a transformed value but retains the
//! original range. Clone an expression to reuse its input:
//! `q.filt(|x| x.clone().gt(10) & x.lt(20))`. Each branch is a function of
//! the same input; cloning does not bind another stored witness.
//!
//! These closures run once during construction rather than per row. Bitwise
//! operators build maps; comparisons use `.eq`, `.ne`, `.lt`, `.le`, `.gt`,
//! and `.ge` because Rust's comparison operators must return `bool`.
//! Predicate operators are `&`, `|`, and `!`. Conditions also accept
//! predicate recipes and query operands: `condition | pred::eq(value)`. A
//! recipe on the left, `pred::eq(value) | condition`, builds a `Disj` operand
//! instead of a condition. It composes and negates the same way and restricts
//! through `with`, but a `filt` closure must return a condition, so write
//! `condition | pred::eq(value)` there. See [`Restriction`] for the operands
//! a condition accepts.
//! Borrow conditions to reuse them, for example `&a | &b`. Rust binds `!`
//! before `&`, then `|`; use parentheses to make other groupings explicit.
//! Query `.and(...)` still pairs ranges. Value-dependent Rust `if`,
//! short-circuit Boolean operators, and arbitrary Rust functions are not
//! translated. Backend support and checked literal packing are unchanged, and
//! this syntax adds no execution path. Cost follows each field's backend: BDD
//! columns apply bit maps symbolically, and sparse columns enumerate admitted
//! inputs for any nonidentity map, so an unrestricted wide input takes time
//! proportional to its sort.
//!
//! `|` on conditions is a lazy union that runs the continuation once per
//! satisfied branch, so test one expression's alternatives together with
//! `x.test(pred::eq(1) | pred::eq(2))`. On a total expression, a single
//! negated test stays one native root as `x.test(!recipe)` or the dual
//! comparison. Keep `!` for composite conditions. A partial
//! `filter_by` or `map_by` needs it too: `!condition` keeps the inputs such a
//! map drops, while a negated test does not.
//!
//! Scalar maps currently support identity and bitwise operations against
//! literals. Comparisons also take literals. Record views written against
//! `View` expose field expressions through `Expr::field`. Expression-to-
//! expression arithmetic is unsupported. General relational operands remain
//! available through `select`/`with`. Fixed-set expressions additionally
//! support member tests and updates, and union/intersection with constant
//! sets. These lower to the same bitwise maps without integer interpretation.
//! Interned-value expressions support `filter_by` and `map_by` with Rust
//! closures over the decoded values; these are the one concrete callback,
//! applied by enumerating admitted IDs on sparse columns and rejected by BDD
//! columns.

use std::borrow::Borrow;
use std::ops::{BitAnd, BitOr, BitXor, Not, RangeFull};
use std::sync::Arc;

use super::{
    Domain, FiniteSet, FuncDecl, Ident, Integer, Interned, IntoQuery, Operand, Query, Rel, Same,
    Sorted, Universe, View, map, pred,
};
use crate::backend::MapSpec;
use crate::context::{Context, Interner};
use crate::index::IndexRoot;
use crate::map::Transform;
use crate::query::{self, Diff, IntoQuery as RawIntoQuery, Join, Operand as RawOperand, Union};
use crate::row::{self, IntoRow, Row};
use crate::run::Exec;

/// The node of a condition that tests a functional expression with node `A`.
pub type Test<A> = Join<(), Join<A, IndexRoot>>;

/// A subidentity on the closure input, including when a test observes a
/// transformed value.
///
/// Keeping construction private preserves the subidentity
/// required by Boolean complement.
pub struct Condition<D, A>(Query<D, D, A>);

impl<D, A: Clone> Clone for Condition<D, A> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<D: Sorted, A: Exec> IntoQuery for Condition<D, A> {
    type D = D;
    type R = D;
    type Node = A;
    #[inline(always)]
    fn into_query(self) -> Query<D, D, A> {
        self.0
    }
}

impl<'a, D: Sorted, A: Exec> IntoQuery for &'a Condition<D, A> {
    type D = D;
    type R = D;
    type Node = &'a A;
    #[inline(always)]
    fn into_query(self) -> Query<D, D, &'a A> {
        (&self.0).into_query()
    }
}

/// A right-hand operand that restricts a condition's input.
///
/// Conditions, queries, relations, and predicate recipes qualify: each admits
/// the inputs it holds or accepts. Expressions, declarations, and maps compute
/// values, so combining one with a condition would discard its value. Test the
/// value instead.
/// A value that should restrict by witness can still be converted explicitly
/// with [`IntoQuery::into_query`].
///
/// ```
/// use sra::context::Context;
/// use sra::typed::{IntoQuery, Rel, Sorted};
///
/// let mut cx = Context::default();
/// u8::bind(&mut cx);
/// let rel = Rel::<u8, u8>::new_set(&cx);
/// (&rel).filt(|x| x.clone().lt(2_u8) & x.eq(7_u8));
/// ```
///
/// ```compile_fail,E0277
/// use sra::context::Context;
/// use sra::typed::{IntoQuery, Rel, Sorted};
///
/// let mut cx = Context::default();
/// u8::bind(&mut cx);
/// let rel = Rel::<u8, u8>::new_set(&cx);
/// (&rel).filt(|x| x.clone().lt(2_u8) & x);
/// ```
pub trait Restriction<D: Sorted>: Operand<D> {}

impl<D: Sorted> Restriction<D> for Ident {}
impl<D: Sorted, A: Exec> Restriction<D> for Condition<D, A> {}
impl<D: Sorted, A: Exec> Restriction<D> for &Condition<D, A> {}
impl<X: Same<D>, D: Sorted, R: Sorted, A: Exec> Restriction<X> for Query<D, R, A> {}
impl<X: Same<D>, D: Sorted, R: Sorted, A: Exec> Restriction<X> for &Query<D, R, A> {}
impl<X: Same<D>, D: Sorted, R: Sorted> Restriction<X> for &Rel<D, R> {}
impl<T: Sorted, P: pred::Predicate<T>> Restriction<T> for pred::Filter<P> {}
impl<T: Sorted, P: pred::Predicate<T>> Restriction<T> for &pred::Filter<P> {}
impl<D: Sorted, A: Restriction<D>, B: Restriction<D>> Restriction<D> for pred::Disj<A, B> {}
impl<D: Sorted, A: Restriction<D>, B: Restriction<D>> Restriction<D> for pred::Conj<A, B> {}
impl<D: Sorted, A: Restriction<D>> Restriction<D> for pred::Neg<A> {}
impl<'a, D: Sorted, A, B> Restriction<D> for &'a pred::Disj<A, B>
where
    &'a A: Restriction<D>,
    &'a B: Restriction<D>,
{
}
impl<'a, D: Sorted, A, B> Restriction<D> for &'a pred::Conj<A, B>
where
    &'a A: Restriction<D>,
    &'a B: Restriction<D>,
{
}
impl<'a, D: Sorted, A> Restriction<D> for &'a pred::Neg<A> where &'a A: Restriction<D> {}
impl<'a, 'b, D: Sorted, A> Restriction<D> for &'b &'a A
where
    &'a A: Restriction<D>,
    &'b &'a A: Operand<D>,
{
}
impl<T: Integer> Restriction<T> for RangeFull {}
impl<T: Integer> Restriction<T> for &RangeFull {}
macro_rules! range_restrictions {
    ($($t:ident),*) => {$(
        impl<T: Integer, V: IntoRow> Restriction<T> for std::ops::$t<V> {}
        impl<T: Integer, V: IntoRow> Restriction<T> for &std::ops::$t<V> {}
    )*};
}
range_restrictions!(Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive);

/// A total symbolic function, built from expressions rather than arbitrary
/// relation witnesses.
///
/// Record views implement this contract by
/// pairing fields of the same closure input.
pub trait Functional: IntoQuery {
    /// Tests this expression's value while retaining the closure's original
    /// input.
    ///
    /// Accepts the same predicate recipes as projection tests,
    /// including borrowed recipes:
    /// `q.filt(|x| (x | 16).test(pred::is_in(20..40)))`.
    #[inline(always)]
    fn test<P: pred::Predicate<Self::R>>(self, p: P) -> Condition<Self::D, Test<Self::Node>> {
        let q = self.into_query();
        // Observe the transformed range only inside the restriction. The
        // exposed endpoints alias the original input, so filtering and
        // negation preserve its concrete meaning.
        Condition(q.with(pred::Filter(p)).keys())
    }

    /// Tests this expression's value for equality with `value`.
    #[inline(always)]
    fn eq(self, value: impl IntoRow) -> Condition<Self::D, Test<Self::Node>> {
        self.test(pred::eq(value))
    }

    /// Tests this expression's value for inequality with `value`.
    #[inline(always)]
    fn ne(self, value: impl IntoRow) -> Condition<Self::D, Test<Self::Node>> {
        self.test(pred::ne(value))
    }
}

// Output tuples pair functions of one input; anonymous tuple grouping is
// metadata only.
macro_rules! functional_tuples {
    (@node $a:ty; ) => { $a };
    (@node $a:ty; $b:ident $(, $rest:ident)*) => {
        functional_tuples!(@node Join<$a, $b::Node>; $($rest),*)
    };
    ($a:ident:$av:ident $(, $b:ident:$bv:ident)*) => {
        impl<$a: Functional, $($b: Functional<D = $a::D>),*> IntoQuery for ($a, $($b,)*)
        where ($a::R, $($b::R,)*): Sorted {
            type D = $a::D;
            type R = ($a::R, $($b::R,)*);
            type Node = functional_tuples!(@node $a::Node; $($b),*);
            #[inline(always)]
            fn into_query(self) -> Query<Self::D, Self::R, Self::Node> {
                let ($av, $($bv,)*) = self;
                let q = $av.into_query().raw;
                $(let q = RawIntoQuery::and(q, $bv.into_query().raw);)*
                let domain = q.domain.sort;
                let range = Self::R::sort(&q.cx);
                Query::from_raw(q.coerce(domain, range))
            }
        }
        impl<$a: Functional, $($b: Functional<D = $a::D>),*> Functional for ($a, $($b,)*)
        where ($a::R, $($b::R,)*): Sorted {}
        functional_tuples!($($b:$bv),*);
    };
    () => {};
}
functional_tuples!(A:a, B:b, C:c, D:d, E:e, F:f, G:g, H:h, I:i, J:j, K:k, L:l);

/// A total functional query from the closure input `D` to its current value
/// `R`.
///
/// Construction is restricted to functional operations; arbitrary
/// queries cannot be wrapped.
pub struct Expr<D, R> {
    // Sources have identity-ordered domain endpoints, so range endpoints
    // directly name positions in the original input.
    source: Query<D, R>,
    // None preserves projection aliases; an explicit operation stays Some even
    // if it cancels.
    map: Option<Arc<MapSpec>>,
}

impl<D, R> Clone for Expr<D, R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            map: self.map.clone(),
        }
    }
}

/// Sequential composition of two maps of one scalar sort.
///
/// Bit maps compose
/// into a single bit map, so a chain of literal operations stays symbolic; any
/// other pair evaluates in sequence.
fn compose(first: &MapSpec, second: MapSpec) -> MapSpec {
    match (first, second) {
        (
            MapSpec::Bits {
                clear: old_clear,
                flip: old_flip,
            },
            MapSpec::Bits { clear, flip },
        ) => MapSpec::Bits {
            clear: old_clear.iter().zip(&clear).map(|(a, b)| a | b).collect(),
            flip: old_flip
                .iter()
                .zip(clear.iter().zip(&flip))
                .map(|(a, (b, c))| (a & !b) ^ c)
                .collect(),
        },
        (a, b) => {
            let a = a.clone();
            MapSpec::Rust(Arc::new(move |x: &[u64], y: &mut [u64]| {
                // Only interned-ID chains reach this arm, with one word per value.
                let mut mid = [0_u64; 1];
                a.evaluate(x, &mut mid) && b.evaluate(&mid, y)
            }))
        }
    }
}

impl<T: Sorted> Expr<T, T> {
    /// Binds a symbolic input to this query's range, using its context and
    /// physical columns.
    #[inline(always)]
    pub fn input<D: Sorted, A: Exec>(q: &Query<D, T, A>) -> Self {
        Self {
            source: Query::wrap(RawOperand::resolve(query::id(), &q.raw.cx, &q.raw.range)),
            map: None,
        }
    }
}

impl<D: Sorted, R: Sorted> Expr<D, R> {
    /// Projects one declared field without binding another witness of the
    /// closure input.
    ///
    /// This low-level constructor serves [`View`] implementations. Like other
    /// raw conversions, it checks runtime sorts only, so the caller supplies
    /// the field's declared interpretation as `T`.
    ///
    /// # Panics
    ///
    /// Panics if this expression is mapped, its range has no field at `i`, or
    /// the selected field's sort differs from `T`'s declared sort.
    #[inline(always)]
    pub fn field<T: View>(&self, i: usize) -> T::Expr<D> {
        assert!(
            self.map.is_none(),
            "fields require a structural record expression"
        );
        let cx = &self.source.raw.cx;
        let part = row::children(cx, &self.source.raw.range)
            .nth(i)
            .expect("field index must name a declared field");
        let k = self.source.raw.domain.sort.len + part.start - self.source.raw.range.start;
        T::view(Expr {
            source: Query::from_raw(query::project(
                cx,
                &self.source.raw.domain,
                self.source.raw.ends[k..k + part.sort.len].iter().copied(),
                part,
            )),
            map: None,
        })
    }

    /// Empty result with the same closure input, used when assembling record
    /// arguments.
    ///
    /// # Panics
    ///
    /// Panics if this expression is mapped.
    #[inline(always)]
    pub fn unit(&self) -> Expr<D, ()> {
        assert!(
            self.map.is_none(),
            "units require a structural record expression"
        );
        let cx = &self.source.raw.cx;
        let input = &self.source.raw.domain;
        Expr {
            source: Query::wrap(query::project(
                cx,
                input,
                0..0,
                row::Domain::row(<() as Sorted>::sort(cx), cx),
            )),
            map: None,
        }
    }

    /// Tests the output while retaining the original input; also available on
    /// record views.
    #[inline(always)]
    pub fn test<P: pred::Predicate<R>>(
        self,
        p: P,
    ) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
        Functional::test(self, p)
    }

    /// Tests membership in a literal collection or range.
    #[inline(always)]
    pub fn is_in<C>(self, values: C) -> Condition<D, Test<<Self as IntoQuery>::Node>>
    where
        pred::Member<C>: pred::Predicate<R>,
    {
        self.test(pred::is_in(values))
    }

    /// Appends one transformation after any existing one, keeping a single
    /// pending map.
    ///
    /// Bit maps are total on `R`, while Rust maps are partial. Chained closures
    /// decode and re-intern at each step rather than composing on values.
    #[inline(always)]
    fn chain<F: FnOnce(&Context, &Row) -> MapSpec>(mut self, map: crate::map::Map<F>) -> Self {
        let spec = (map.0)(&self.source.raw.cx, &self.source.raw.range);
        self.map = Some(Arc::new(match self.map.take() {
            None => spec,
            Some(prev) => compose(&prev, spec),
        }));
        self
    }
}

impl<D: Sorted, U: Universe> FuncDecl<D, FiniteSet<U>> {
    /// Tests a set-valued field in `cx`, retaining its original domain.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    #[inline(always)]
    pub fn contains(
        &self,
        cx: &Context,
        member: impl Into<U::Member>,
    ) -> Condition<D, Test<<Expr<D, FiniteSet<U>> as IntoQuery>::Node>> {
        Expr {
            source: self.query(cx),
            map: None,
        }
        .contains(member)
    }
}

impl<D: Sorted, U: Universe> Expr<D, FiniteSet<U>> {
    /// Tests one membership bit, retaining the closure's original input.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    #[inline(always)]
    pub fn contains(
        self,
        member: impl Into<U::Member>,
    ) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
        let mask = FiniteSet::<U>::from_members(&self.source.raw.cx, [member.into()]);
        self.intersect_with(&mask).eq(mask)
    }

    /// Adds a member to every possible set.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    #[inline(always)]
    pub fn insert(self, member: impl Into<U::Member>) -> Self {
        let mask = FiniteSet::<U>::from_members(&self.source.raw.cx, [member.into()]);
        self.union_with(&mask)
    }

    /// Removes a member from every possible set.
    ///
    /// # Panics
    ///
    /// Panics if `member` is outside the universe, its index exceeds the
    /// declared length, or a [`super::Capacity`] member belongs to another
    /// context.
    #[inline(always)]
    pub fn remove(self, member: impl Into<U::Member>) -> Self {
        let mut keep = FiniteSet::<U>::full(&self.source.raw.cx);
        keep.remove(member.into());
        self.intersect_with(&keep)
    }

    /// Unions each possible set with a constant set from the same universe.
    ///
    /// # Panics
    ///
    /// Panics if `set` belongs to another context.
    #[inline(always)]
    pub fn union_with(self, set: &FiniteSet<U>) -> Self {
        self.chain(crate::map::bvor(set))
    }

    /// Intersects each possible set with a constant set from the same
    /// universe.
    ///
    /// # Panics
    ///
    /// Panics if `set` belongs to another context.
    #[inline(always)]
    pub fn intersect_with(self, set: &FiniteSet<U>) -> Self {
        self.chain(crate::map::bvand(set))
    }
}

impl<D: Sorted, T: Domain> Expr<D, Interned<T>> {
    fn table(&self) -> Arc<Interner<T::Value>> {
        super::interned::interner::<T>(&self.source.raw.cx)
    }

    /// Keeps the inputs whose decoded value satisfies `keep`, retaining the
    /// closure's original input.
    ///
    /// IDs with no dictionary entry have no value
    /// and are dropped. This runs by enumerating the admitted IDs of the
    /// expression's own context; a BDD-backed column rejects it at execution.
    #[inline(always)]
    pub fn filter_by(
        self,
        keep: impl Fn(&T::Value) -> bool + Send + Sync + 'static,
    ) -> Condition<D, Join<(), <Self as IntoQuery>::Node>> {
        let table = self.table();
        let q = self
            .chain(crate::map::func(move |x: &[u64], y: &mut [u64]| {
                y[0] = x[0];
                table.resolve(x[0]).is_some_and(|v| keep(&v))
            }))
            .into_query();
        // As in `Functional::test`, the exposed endpoints alias the original
        // input. The domain subidentity hides the mapped cell, so restricting
        // its preimage is the filter.
        Condition(q.keys())
    }

    /// Replaces each decoded value by `image` of it, interning the result in
    /// the expression's own context.
    ///
    /// Unassigned IDs are dropped. Enumerates like
    /// [`Self::filter_by`]. Interning happens during execution, for every
    /// admitted input including one a later test rejects, and each new ID
    /// counts toward a [`Capacity`](super::Capacity).
    #[inline(always)]
    pub fn map_by<O: Borrow<T::Value> + Into<Arc<T::Value>>>(
        self,
        image: impl Fn(&T::Value) -> O + Send + Sync + 'static,
    ) -> Self {
        let table = self.table();
        self.chain(crate::map::func(move |x: &[u64], y: &mut [u64]| {
            table.resolve(x[0]).is_some_and(|v| {
                y[0] = table.intern(image(&v));
                true
            })
        }))
    }
}

impl<D: Sorted, R: Sorted> IntoQuery for Expr<D, R> {
    type D = D;
    type R = R;
    type Node = Join<(), Transform>;
    #[inline(always)]
    fn into_query(self) -> Query<D, R, Self::Node> {
        let map: Query<R, R, _> = Query::wrap(query::transform(
            &self.source.raw.cx,
            &self.source.raw.range,
            self.map,
        ));
        self.source.select(map)
    }
}
impl<D: Sorted, R: Sorted> Functional for Expr<D, R> {}
macro_rules! bits {
    ($($trait:ident, $method:ident);* $(;)?) => {$(
        impl<D: Sorted, R: Integer, V: IntoRow> $trait<V> for Expr<D, R> {
            type Output = Self;
            #[inline(always)]
            fn $method(self, value: V) -> Self::Output {
                self.chain(map::$method(value).0)
            }
        }
    )*};
}
bits!(BitAnd, bitand; BitOr, bitor; BitXor, bitxor);

macro_rules! comparisons {
    ($bound:ident; $($method:ident),* $(,)?) => {$(
        impl<D: Sorted, R: $bound> Expr<D, R> {
            #[doc = concat!(
                "Tests this expression's value with [`pred::", stringify!($method), "`]."
            )]
            #[inline(always)]
            pub fn $method(
                self,
                value: impl IntoRow,
            ) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
                self.test(pred::$method(value))
            }
        }
    )*};
}
comparisons!(Sorted; eq, ne);
comparisons!(Integer; lt, le, gt, ge, slt, sle, sgt, sge, ult, ule, ugt, uge);

impl<D: Sorted, A: Exec, B: Restriction<D>> BitAnd<B> for Condition<D, A> {
    type Output = Condition<D, Join<A, B::Node>>;
    #[inline(always)]
    fn bitand(self, b: B) -> Self::Output {
        Condition(self.0.with(b))
    }
}

impl<D: Sorted, A: Exec, B: Restriction<D>> BitOr<B> for Condition<D, A> {
    type Output = Condition<D, Union<A, Join<(), B::Node>>>;
    #[inline(always)]
    fn bitor(self, b: B) -> Self::Output {
        let right = Expr::<D, D>::input(&self.0).source.with(b);
        Condition(self.0.union(right))
    }
}

macro_rules! borrowed_boolean {
    ($($trait:ident, $method:ident);* $(;)?) => {$(
        impl<'a, D: Sorted, A: Exec, B: Restriction<D>> $trait<B> for &'a Condition<D, A> {
            type Output = <Condition<D, &'a A> as $trait<B>>::Output;
            #[inline(always)]
            fn $method(self, b: B) -> Self::Output {
                Condition(self.into_query()).$method(b)
            }
        }
    )*};
}
borrowed_boolean!(BitAnd, bitand; BitOr, bitor);

impl<D: Sorted, A: Exec> Not for Condition<D, A> {
    type Output = Condition<D, Diff<(), A>>;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Condition(self.0.complement())
    }
}

impl<'a, D: Sorted, A: Exec> Not for &'a Condition<D, A> {
    type Output = Condition<D, Diff<(), &'a A>>;
    #[inline(always)]
    fn not(self) -> Self::Output {
        !Condition::<D, &A>(self.into_query())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/typed/expr.rs"]
mod tests;
