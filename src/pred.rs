//! Predicate recipes and lazy domain-membership conditions.
//!
//! Literals get their logical sort and physical columns from the consuming
//! query. `&`, `|`, and `!` operate on one row universe; accessor comparisons
//! test their parent row. Boolean operands resolve through lazy query
//! combinators; explicit `Build` consumers construct native predicates only
//! when all children are native builders. Predicate `.and()` is intersection,
//! while ordinary query `.and()` still pairs ranges: membership conjunction
//! never exposes a paired output.
//!
//! `bvult`/`bvule`/`bvugt`/`bvuge` and Rust ranges use unsigned bit-vector
//! order. The `bvs`-prefixed comparisons interpret the same encoded bits in
//! signed two's-complement order. Bounds must fit the declared width;
//! inclusive maximum or an unbounded endpoint reaches the universe's end.
//! Equality and finite membership also accept Boolean and complete row
//! literals.
//!
//! A typed frontend infers comparison interpretations through Rust endpoint
//! types and lowers them to these explicit operations. Runtime nominal sorts
//! remain independent; bit-vector sorts carry only width, and each comparison
//! selects its ordering. See <https://remy.wang/blog/value-polymorphism.html>.
//!
//! Selectors resolve lazily: a recipe built from a declaration or a
//! source-free projection carries no context and reads one from the index it
//! is built into, so the same recipe serves every query in that context.

use std::marker::PhantomData;
use std::ops::{
    BitAnd, BitOr, Bound, Not, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use std::sync::Arc;

use crate::ast::{FuncDecl, Sort};
use crate::backend::{BoolOp, Function};
use crate::context::Context;
use crate::index::{self, IndexRoot, Labels, Level};
use crate::query::{self, IntoQuery, Operand, Query};
use crate::rel::Universe;
use crate::row::{self, IntoRow, Row};
use crate::run::{Exec, Slot};
use crate::scalar::sort_records;
use crate::util::dag::Idx;

/// A reusable construction-time predicate.
///
/// Resolution owns its resulting source independently of the builder and its
/// literals, and execution never calls the builder again.
#[derive(Clone)]
pub struct Filter<P>(pub P);

/// Builds a predicate in one input index.
///
/// `positions` maps this node's logical row into that index, including
/// repeated positions. All Boolean children build into the same universe, and
/// every builder reads its context from the index.
pub trait Build {
    /// Constructs this predicate over `input`'s columns inside `index`, as one
    /// owned root.
    ///
    /// # Panics
    ///
    /// Panics if the input belongs to another context, the position count
    /// differs from the input's column count, a position exceeds the index,
    /// or a mapped column differs from the input's physical column. Leaf
    /// builders enforce these checks even for constant results; composite
    /// builders inherit them from their children. Leaf builders also panic if
    /// a comparison or interval tests an input without a bit-vector sort, a
    /// selector belongs to another context or requires an incompatible domain
    /// sort, or a literal does not match the input sort.
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot;
}

/// Checks the input row's complete physical mapping into the target index.
fn check(input: &Row, index: &IndexRoot, positions: &[usize]) {
    assert_eq!(
        input.sort.context,
        index.context().id,
        "predicate input belongs to another context"
    );
    assert_eq!(
        positions.len(),
        input.sort.len,
        "predicate positions must cover every input column"
    );
    for (&i, &column) in positions.iter().zip(row::columns(input)) {
        assert!(i < index.arity(), "predicate position exceeds the index");
        assert_eq!(
            column,
            index.column(i),
            "predicate input uses an incompatible column"
        );
    }
}

impl<P: Build + ?Sized> Build for &P {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        (**self).build(input, index, positions)
    }
}

impl<P: Build> Build for Filter<P> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        self.0.build(input, index, positions)
    }
}

impl<P: Build> Operand for &Filter<P> {
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        let root = self.build(
            input,
            &index::intern(cx, row::columns(input)),
            &(0..input.sort.len).collect::<Vec<_>>(),
        );
        // Only the columns the root constrains are read and fixed, so a
        // subidentity leaves its unconstrained cells to the consuming query.
        let slots = root
            .constrains()
            .into_iter()
            .enumerate()
            .map(|(i, read)| Slot {
                column: root.column(i),
                read,
                fixed: read,
            })
            .collect();
        let n = input.sort.len;
        query::make(
            cx,
            root,
            input.sort,
            input.sort,
            Universe::Set,
            slots,
            (0..n).chain(0..n).collect(),
        )
    }
}

impl<P: Build> Operand for Filter<P> {
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        (&self).resolve(cx, input)
    }
}

/// Domain-membership disjunction; the two witness ranges may differ.
///
/// Resolves to D -> D where either child has a witness, without exposing
/// either range. This is an operand, with no direct collection. Unlike
/// full-pair query `union`, it accepts different range sorts. Use `a.or(b)`
/// or predicate `a | b`; borrow reusable conditions as `&a | &b`. In `with`,
/// the enclosing query retains its original pairs.
///
/// Symbolic branches may cover different parts of the input: a left witness
/// does not suppress right-hand coverage. Normal terminal cancellation and
/// streaming duplicates follow the existing union executor.
#[derive(Clone)]
pub struct Disj<A, B>(pub A, pub B);
/// Domain-membership conjunction, with independent witnesses for each
/// operand.
#[derive(Clone)]
pub struct Conj<A, B>(pub A, pub B);
/// Domain-membership negation: the input rows with no witness in the operand.
#[derive(Clone)]
pub struct Neg<P>(pub P);

macro_rules! native_boolean {
    ($($name:ident, $op:ident);* $(;)?) => {$(
        impl<A: Build, B: Build> Build for $name<A, B> {
            fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
                let a = self.0.build(input, index, positions);
                let b = self.1.build(input, index, positions);
                index.apply_bool(BoolOp::$op, a.function, b.function)
            }
        }
    )*};
}
native_boolean!(Disj, Or; Conj, And);

impl<P: Build> Build for Neg<P> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        let p = self.0.build(input, index, positions);
        index.apply_bool(BoolOp::AndNot, Function::TRUE, p.function)
    }
}

impl<A: Operand, B: Operand> Operand for Disj<A, B> {
    type Node = query::Union<query::Join<(), A::Node>, query::Join<(), B::Node>>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        query::id()
            .resolve(cx, input)
            .with(self.0)
            .union(query::id().resolve(cx, input).with(self.1))
    }
}
impl<'a, A, B> Operand for &'a Disj<A, B>
where
    &'a A: Operand,
    &'a B: Operand,
{
    type Node = <Disj<&'a A, &'a B> as Operand>::Node;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        Disj(&self.0, &self.1).resolve(cx, input)
    }
}
impl<A: Operand, B: Operand> Operand for Conj<A, B> {
    type Node = query::Join<query::Join<(), A::Node>, B::Node>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        query::id().resolve(cx, input).with(self.0).with(self.1)
    }
}
impl<'a, A, B> Operand for &'a Conj<A, B>
where
    &'a A: Operand,
    &'a B: Operand,
{
    type Node = <Conj<&'a A, &'a B> as Operand>::Node;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        Conj(&self.0, &self.1).resolve(cx, input)
    }
}
impl<A: Operand> Operand for Neg<A> {
    type Node = query::Diff<(), A::Node>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        query::id().resolve(cx, input).minus(self.0)
    }
}
impl<'a, A> Operand for &'a Neg<A>
where
    &'a A: Operand,
{
    type Node = <Neg<&'a A> as Operand>::Node;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        Neg(&self.0).resolve(cx, input)
    }
}

// A borrowed Boolean tree can itself hold borrowed child recipes.
macro_rules! borrowed_operands {
    ($($name:ident $(<$($t:ident),+>)?);* $(;)?) => {$(
        impl<'a $(, $($t),+)?> Operand for &&'a $name$(<$($t),+>)?
        where &'a $name$(<$($t),+>)?: Operand {
            type Node = <&'a $name$(<$($t),+>)? as Operand>::Node;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
                (*self).resolve(cx, input)
            }
        }
    )*};
}
borrowed_operands!(
    Filter<P>; Disj<A, B>; Conj<A, B>; Neg<P>;
    Range<T>; RangeFrom<T>; RangeInclusive<T>; RangeTo<T>; RangeToInclusive<T>; RangeFull;
);

macro_rules! boolean {
    ($($name:ident<$($t:ident),+>);* $(;)?) => {$(
        impl<$($t),+> $name<$($t),+> {
            /// Membership conjunction with `b`, as for `&`.
            ///
            /// The method consumes this recipe. A borrowed recipe combines
            /// through `&` instead.
            #[inline(always)]
            pub fn and<B>(self, b: B) -> Conj<Self, B> { self & b }
            /// Membership disjunction with `b`, as for `|`.
            ///
            /// The method consumes this recipe. A borrowed recipe combines
            /// through `|` instead.
            #[inline(always)]
            pub fn or<B>(self, b: B) -> Disj<Self, B> { self | b }
        }
        impl<$($t),+, B> BitAnd<B> for $name<$($t),+> {
            type Output = Conj<Self, B>;
            #[inline(always)]
            fn bitand(self, b: B) -> Self::Output { Conj(self, b) }
        }
        impl<'a, $($t),+, B> BitAnd<B> for &'a $name<$($t),+> {
            type Output = Conj<Self, B>;
            #[inline(always)]
            fn bitand(self, b: B) -> Self::Output { Conj(self, b) }
        }
        impl<$($t),+, B> BitOr<B> for $name<$($t),+> {
            type Output = Disj<Self, B>;
            #[inline(always)]
            fn bitor(self, b: B) -> Self::Output { Disj(self, b) }
        }
        impl<'a, $($t),+, B> BitOr<B> for &'a $name<$($t),+> {
            type Output = Disj<Self, B>;
            #[inline(always)]
            fn bitor(self, b: B) -> Self::Output { Disj(self, b) }
        }
        impl<$($t),+> Not for $name<$($t),+> {
            type Output = Neg<Self>;
            #[inline(always)]
            fn not(self) -> Self::Output { Neg(self) }
        }
        impl<'a, $($t),+> Not for &'a $name<$($t),+> {
            type Output = Neg<Self>;
            #[inline(always)]
            fn not(self) -> Self::Output { Neg(self) }
        }
    )*};
}
boolean!(Filter<P>; Disj<P, Q>; Conj<P, Q>; Neg<P>);

/// Equality with one ground value; see [`eq`].
#[derive(Clone)]
pub struct Equal<T>(T);

/// Equality with one ground scalar, product, or nominal-record value.
#[inline(always)]
pub fn eq<T: IntoRow>(value: T) -> Filter<Equal<T>> {
    Filter(Equal(value))
}
/// Inequality with one ground value: the negation of [`eq`].
#[inline(always)]
pub fn ne<T: IntoRow>(value: T) -> Filter<Neg<Equal<T>>> {
    Filter(Neg(eq(value).0))
}

fn equal(value: &impl IntoRow, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
    check(input, index, positions);
    let cx = index.context();
    let mut packed = Vec::new();
    row::append(cx, value, input, &mut packed);
    let mut offset = 0;
    let mut fields: Vec<_> = positions
        .iter()
        .zip(row::columns(input))
        .map(|(&position, &column)| {
            let end = offset + cx.resolve(column).sort().words();
            let span = offset..end;
            offset = end;
            (position, span)
        })
        .collect();
    // Order constraints by physical level so each parent keeps the existing
    // suffix, including when the logical positions are permuted.
    fields.sort_unstable_by_key(|field| field.0);
    let mut root = index.inc_ref(Function::TRUE);
    let mut labels = Labels::default();
    for group in fields.chunk_by(|a, b| a.0 == b.0).rev() {
        let (position, span) = &group[0];
        let value = &packed[span.clone()];
        // Equal positions constrain one witness and must agree on its value.
        if group
            .iter()
            .skip(1)
            .any(|(_, span)| &packed[span.clone()] != value)
        {
            return index.inc_ref(Function::FALSE);
        }
        root = index.values(Level::new(*position), value, &[root.function], &mut labels);
    }
    root
}

impl<T: IntoRow> Build for Equal<T> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        equal(&self.0, input, index, positions)
    }
}

#[derive(Clone, Copy)]
enum Order {
    ULt,
    ULe,
    SLt,
    SLe,
}
/// A strict or inclusive lower-than comparison with one ground bit-vector
/// value.
#[derive(Clone)]
pub struct Compare<T> {
    value: T,
    order: Order,
}

macro_rules! comparison {
    ($name:ident, $op:ident, $doc:literal) => {
        #[doc = $doc]
        #[inline(always)]
        pub fn $name<T: IntoRow>(value: T) -> Filter<Compare<T>> {
            Filter(Compare {
                value,
                order: Order::$op,
            })
        }
    };
}
comparison!(
    bvult,
    ULt,
    "Strictly less than `value` in unsigned bit-vector order."
);
comparison!(
    bvule,
    ULe,
    "Less than or equal to `value` in unsigned bit-vector order."
);
comparison!(
    bvslt,
    SLt,
    "Strictly less than `value` in signed two's-complement order."
);
comparison!(
    bvsle,
    SLe,
    "Less than or equal to `value` in signed two's-complement order."
);
/// Greater than or equal to `value` in unsigned bit-vector order.
#[inline(always)]
pub fn bvuge<T: IntoRow>(value: T) -> Filter<Neg<Compare<T>>> {
    Filter(Neg(bvult(value).0))
}
/// Strictly greater than `value` in unsigned bit-vector order.
#[inline(always)]
pub fn bvugt<T: IntoRow>(value: T) -> Filter<Neg<Compare<T>>> {
    Filter(Neg(bvule(value).0))
}
/// Greater than or equal to `value` in signed two's-complement order.
#[inline(always)]
pub fn bvsge<T: IntoRow>(value: T) -> Filter<Neg<Compare<T>>> {
    Filter(Neg(bvslt(value).0))
}
/// Strictly greater than `value` in signed two's-complement order.
#[inline(always)]
pub fn bvsgt<T: IntoRow>(value: T) -> Filter<Neg<Compare<T>>> {
    Filter(Neg(bvsle(value).0))
}

impl<T: IntoRow> Build for Compare<T> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        check(input, index, positions);
        let cx = index.context();
        let column = row::bitvector_column(input);
        let width = cx.resolve(column).sort().width();
        let mut value = Vec::new();
        row::append(cx, &self.value, input, &mut value);
        let mut p = cx.less_than(column, &value);
        if matches!(self.order, Order::SLt | Order::SLe) {
            // Signed order flips the sign bit: a negative bound keeps only
            // negative values below it, and a nonnegative bound keeps every
            // negative value as well.
            let (w, bit) = (((width - 1) / 64) as usize, 1 << ((width - 1) % 64));
            let mut sign = vec![0; value.len()];
            sign[w] = bit;
            let negative = cx.t(column).difference(&cx.less_than(column, &sign));
            p = if value[w] & bit != 0 {
                p.and(&negative)
            } else {
                p.or(&negative)
            };
        }
        if matches!(self.order, Order::ULe | Order::SLe) {
            p = p.or(&cx.set(column, &value));
        }
        index.cases(
            Level::new(positions[0]),
            Function::FALSE,
            p.column(),
            &[(p.function(), Function::TRUE)],
        )
    }
}

/// A finite collection or symbolic interval usable as a membership predicate.
///
/// Input conversion distinguishes finite collections from symbolic intervals.
/// Iterators must be explicitly collected, so a large Rust range never enters
/// an enumerating blanket adapter.
pub trait Membership {
    /// The predicate recipe the collection or interval becomes.
    type Node: Build;
    /// Converts the collection or interval into its recipe.
    fn into_membership(self) -> Self::Node;
}

/// Membership in a finite collection of ground values or in a bit-vector
/// interval.
#[inline(always)]
pub fn is_in<T: Membership>(values: T) -> Filter<T::Node> {
    Filter(values.into_membership())
}

/// Membership in a finite collection of ground values, scalar or aggregate.
#[derive(Clone)]
pub struct Finite<C, T> {
    values: C,
    marker: PhantomData<fn() -> T>,
}

impl<C: AsRef<[T]>, T: IntoRow> Build for Finite<C, T> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        check(input, index, positions);
        let cx = index.context();
        if input.sort.kind.is_scalar() {
            let mut values = Vec::new();
            for value in self.values.as_ref() {
                row::append(cx, value, input, &mut values);
            }
            let words = input.sort.scalar().words();
            sort_records(&mut values, words);
            return index.values(
                Level::new(positions[0]),
                &values,
                &vec![Function::TRUE; values.len() / words],
                &mut Labels::default(),
            );
        }
        // A finite slice can be split before construction, so every union
        // combines similarly sized halves instead of rebuilding a growing
        // prefix. Each literal still uses the complete mapped-row checks.
        fn union<T: IntoRow>(
            values: &[T],
            input: &Row,
            index: &IndexRoot,
            positions: &[usize],
        ) -> IndexRoot {
            match values {
                [] => index.inc_ref(Function::FALSE),
                [value] => equal(value, input, index, positions),
                _ => {
                    let (a, b) = values.split_at(values.len() / 2);
                    let left = union(a, input, index, positions);
                    let right = union(b, input, index, positions);
                    index.apply_bool(BoolOp::Or, left.function, right.function)
                }
            }
        }
        union(self.values.as_ref(), input, index, positions)
    }
}

macro_rules! finite {
    ([$($generic:tt)*] $t:ty) => {
        impl<$($generic)*> Membership for $t {
            type Node = Finite<Self, T>;
            fn into_membership(self) -> Self::Node { Finite { values: self, marker: PhantomData } }
        }
    };
}
finite!([T: IntoRow, const N: usize] [T; N]);
finite!([T: IntoRow] Vec<T>);
finite!(['a, T: IntoRow] &'a [T]);
finite!(['a, T: IntoRow, const N: usize] &'a [T; N]);
finite!(['a, T: IntoRow] &'a Vec<T>);

// Ranges construct explicit unsigned comparisons. Language wrappers select
// their own bit-vector interpretation before passing comparison predicates to
// the runtime.
macro_rules! ranges {
    ($($t:ident, |$r:ident| $p:expr);* $(;)?) => {$(
        impl<T: IntoRow> Build for $t<T> {
            fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
                let $r = self;
                ($p).build(input, index, positions)
            }
        }
        impl<T: IntoRow> Membership for $t<T> {
            type Node = Self;
            fn into_membership(self) -> Self { self }
        }
        impl<T: IntoRow> Operand for $t<T> {
            type Node = IndexRoot;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
                Filter(self).resolve(cx, input)
            }
        }
        impl<T: IntoRow> Operand for &$t<T> {
            type Node = IndexRoot;
            #[inline(always)]
            fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
                Filter(self).resolve(cx, input)
            }
        }
    )*};
}
ranges!(
    Range, |r| bvuge(&r.start) & bvult(&r.end);
    RangeFrom, |r| bvuge(&r.start);
    // An iterated range can report an excluded end bound. Exhaustion leaves
    // `start()` and `end()` unspecified, so the strict comparison takes its
    // value from that bound. Every other inclusive range keeps its end.
    RangeInclusive, |r| bvuge(r.start())
        & match r.end_bound() { Bound::Excluded(e) => bvult(e), _ => bvule(r.end()) };
    RangeTo, |r| bvult(&r.end);
    RangeToInclusive, |r| bvule(&r.end);
);
impl Build for RangeFull {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        check(input, index, positions);
        row::bitvector_column(input);
        index.inc_ref(Function::TRUE)
    }
}
impl Membership for RangeFull {
    type Node = Self;
    fn into_membership(self) -> Self {
        self
    }
}
impl Operand for RangeFull {
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        Filter(self).resolve(cx, input)
    }
}
impl Operand for &RangeFull {
    type Node = IndexRoot;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node> {
        Filter(self).resolve(cx, input)
    }
}

/// A source-free structural map from a complete nominal domain onto the row
/// positions a predicate tests, resolved in the index's context.
pub trait Selector {
    /// The domain sort, the selected sort, and the selected positions in the
    /// domain row.
    fn select(&self, cx: &Context) -> (Sort, Sort, Vec<usize>);
}

/// A member accessor or constructor selects its range within its domain row.
impl Selector for FuncDecl {
    fn select(&self, cx: &Context) -> (Sort, Sort, Vec<usize>) {
        let info = (*self).resolve(cx);
        let start = match info.member {
            Some(i) => cx.members(info.domain)[..i]
                .iter()
                .map(|sort| sort.len)
                .sum(),
            None => 0,
        };
        (
            info.domain,
            info.range,
            (start..start + info.range.len).collect(),
        )
    }
}

/// The positions of a source-free projection query, fixed when the selector
/// is taken from the query.
#[derive(Clone)]
pub struct Projection {
    domain: Sort,
    range: Sort,
    positions: Arc<[usize]>,
}

impl Selector for Projection {
    fn select(&self, cx: &Context) -> (Sort, Sort, Vec<usize>) {
        assert!(
            self.domain.context == cx.id,
            "predicate selector belongs to another context"
        );
        (self.domain, self.range, self.positions.to_vec())
    }
}

/// A predicate on a structural selector's range, lifted into its complete
/// nominal domain.
#[derive(Clone)]
pub struct On<L, P> {
    selector: L,
    p: P,
}

impl<L: Selector, P: Build> Build for On<L, P> {
    fn build(&self, input: &Row, index: &IndexRoot, positions: &[usize]) -> IndexRoot {
        check(input, index, positions);
        let cx = index.context();
        let (domain, range, inner) = self.selector.select(cx);
        assert!(
            cx.compatible(domain, input.sort),
            "predicate selector requires a compatible input sort"
        );
        let mapped: Vec<_> = inner.iter().map(|&i| positions[i]).collect();
        // The selected logical row carries the consuming query's actual
        // physical columns.
        let selected = Row::new(
            cx,
            range,
            inner.iter().map(|&i| row::columns(input)[i]).collect(),
            0,
        );
        self.p.build(&selected, index, &mapped)
    }
}

// `$panics` carries each selector's construction-time `# Panics` section.
macro_rules! selector_methods {
    ($s:ty, $panics:literal) => {
        selector_methods!($s, $panics; eq: Equal<T>, ne: Neg<Equal<T>>, bvult: Compare<T>,
            bvule: Compare<T>, bvugt: Neg<Compare<T>>, bvuge: Neg<Compare<T>>,
            bvslt: Compare<T>, bvsle: Compare<T>, bvsgt: Neg<Compare<T>>,
            bvsge: Neg<Compare<T>>);
    };
    ($s:ty, $panics:literal; $($name:ident: $out:ty),*) => {
        $(/// Tests this source-free selector's result, yielding a predicate over
        /// its domain.
        #[doc = $panics]
        #[inline(always)]
        pub fn $name<T: IntoRow>(&self, value: T) -> Filter<On<$s, $out>> {
            self.test($name(value).0)
        })*
        /// Tests this source-free selector's result for membership, yielding a
        /// predicate over its domain.
        #[doc = $panics]
        #[inline(always)]
        pub fn is_in<T: Membership>(&self, values: T) -> Filter<On<$s, T::Node>> {
            self.test(is_in(values).0)
        }
    };
}

impl FuncDecl {
    /// Lifts `p` on this accessor's range into a predicate over its complete
    /// domain, resolved in the consuming index's context.
    ///
    /// Selector `eq` and `ne` construct predicates. Declaration comparisons use
    /// `==`, `!=`, or qualified [`PartialEq`] methods.
    #[inline(always)]
    pub fn test<P: Build>(&self, p: P) -> Filter<On<FuncDecl, P>> {
        Filter(On { selector: *self, p })
    }
    selector_methods!(FuncDecl, "");
}

impl<A: Exec> Query<A> {
    /// Lifts `p` on this source-free selector's range into a predicate over
    /// its domain.
    ///
    /// Selector `eq` and `ne` construct predicates. Semantic query comparisons
    /// use `==`, `!=`, or qualified [`PartialEq`] methods.
    ///
    /// # Panics
    ///
    /// Panics if this query executes a source, filter, or map, or its endpoint
    /// wiring does not project the complete input.
    #[inline(always)]
    pub fn test<P: Build>(&self, p: P) -> Filter<On<Projection, P>> {
        Filter(On {
            selector: Projection {
                domain: self.domain.sort,
                range: self.range.sort,
                positions: self.positions().into(),
            },
            p,
        })
    }
    selector_methods!(
        Projection,
        "\n# Panics\n\nPanics if this query executes a source, filter, or map, or its \
         endpoint wiring does not project the complete input."
    );
}

#[cfg(test)]
#[path = "../tests/unit/pred.rs"]
mod tests;
