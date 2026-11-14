//! Rust endpoint interpretations over the dynamically sorted runtime.
//!
//! A Rust type names a runtime sort through [`Sorted`], and a record type
//! names its declaration through [`Record`]. Typed queries forward every
//! combinator to the raw query they wrap; the Rust types only decide which
//! runtime sorts the endpoints must have, so no type registry or schema lookup
//! enters execution. Raw conversion is explicit and checks the runtime
//! endpoint sorts. Query operators accept a record or its declared argument
//! product, including within larger products. The complete-pair comparisons
//! `difference`, `is_subset`, `symmetric_difference`, and `==` require
//! identical endpoint types, and [`Except`] endpoints also agree at runtime.
//!
//! Every operation takes the context it works in. Storage is bound once,
//! through [`Sorted::bind`], before the context is shared, and later
//! constructors look the bound columns up:
//! ```
//! use sra::typed::{IntoQuery, Rel, Sorted};
//! use sra::Context;
//! let mut cx = Context::default();
//! u8::bind(&mut cx);
//! let mut edges = Rel::<u8, u8>::new(&cx);
//! edges.insert_rows([(0_u8, 1_u8), (1, 2)]);
//! assert_eq!((&edges).select(&edges).count(), 1_u32.into());
//! ```
//! Equal bit widths do not silently change the Rust interpretation:
//! ```compile_fail
//! use sra::typed::{Rel, IntoQuery};
//! use sra::Context;
//! let cx = Context::default();
//! let signed = Rel::<i32, i32>::new_set(&cx);
//! let unsigned = Rel::<u32, u32>::new_set(&cx);
//! let wrong = signed.select(&unsigned);
//! ```

//! The same interpretation check applies within normalized products:
//! ```compile_fail
//! use sra::typed::{IntoQuery, Rel, Sorted};
//! use sra::Context;
//! let mut cx = Context::default();
//! <(u8, i8)>::bind(&mut cx);
//! <(u8, u8)>::bind(&mut cx);
//! let signed = Rel::<(u8, i8), (u8, i8)>::new_set(&cx);
//! let unsigned = Rel::<(u8, u8), (u8, u8)>::new_set(&cx);
//! let wrong = signed.select(&unsigned);
//! ```
//! Distinct custom interpretations remain distinct even over one runtime sort:
//! ```compile_fail
//! use sra::typed::{Cons, IntoQuery, Nil, Rel, Sorted};
//! use sra::{Context, Sort};
//! struct A;
//! struct B;
//! impl Sorted for A {
//!     type Shape = Cons<Self, Nil>;
//!     fn sort(cx: &Context) -> Sort { cx.bv_sort(8) }
//! }
//! impl Sorted for B {
//!     type Shape = Cons<Self, Nil>;
//!     fn sort(cx: &Context) -> Sort { cx.bv_sort(8) }
//! }
//! let mut cx = Context::default();
//! A::bind(&mut cx);
//! B::bind(&mut cx);
//! let a = Rel::<A, A>::new_set(&cx);
//! let b = Rel::<B, B>::new_set(&cx);
//! let wrong = a.select(&b);
//! ```

use std::marker::PhantomData;
use std::ops::ControlFlow;

use crate::ast::Sort;
use crate::backend::BoolOp;
use crate::context::Context;
use crate::query::{self, IntoQuery as RawQuery, Operand as RawOperand};
use crate::row::{Domain, IntoRow, Row};
use crate::run::Exec;

/// Ordered scalar interpretations after flattening records and anonymous
/// products.
pub struct Nil;
/// One scalar interpretation `T` followed by the remaining list `L`.
pub struct Cons<T, L>(PhantomData<fn() -> (T, L)>);
/// Type-level concatenation of two flattened scalar lists.
pub trait Concat<T> {
    /// The list holding `Self`'s scalars followed by `T`'s.
    type Output;
}
impl<T> Concat<T> for Nil {
    type Output = T;
}
impl<H, L: Concat<T>, T> Concat<T> for Cons<H, L> {
    type Output = Cons<H, L::Output>;
}

/// Maps a Rust interpretation to an ordinary runtime sort of one context.
///
/// Custom scalars use `Shape = Cons<Self, Nil>` and implement `IntoRow` by
/// converting concrete values to that sort's representation. Every canonical
/// bit pattern must denote one distinct valid value: identity, complement,
/// and counting use the entire runtime sort. Restricted enums need a
/// valid-domain design before they can use this contract. Typed integer
/// operations require `Integer`, while explicit raw maps check only the runtime
/// bit-vector sort.
///
/// `Same` checks scalar interpretations in query composition, while runtime
/// compatibility preserves named records and their declared argument tuples.
/// Literals, mixed insertion, membership, and lookup retain contextual
/// `IntoRow` packing. Explicit raw conversion likewise checks runtime sorts,
/// not erased Rust interpretations.
pub trait Sorted: Sized {
    /// The flattened scalar interpretations, compared by [`Same`] during
    /// query composition.
    type Shape;

    /// The runtime sort that stores values of this interpretation.
    ///
    /// A record computes this from its declaration on every call. The
    /// declaration is hash-consed, so repeated calls in one context yield the
    /// same sort, and a second record type declaring the same name with
    /// different members panics instead of sharing the first type's sort.
    fn sort(cx: &Context) -> Sort;

    /// Binds each unbound scalar leaf of this interpretation to its configured
    /// or default storage and returns the bound row.
    ///
    /// Call this once per interpretation before the context is shared.
    ///
    /// # Panics
    ///
    /// Panics if a new binding is needed while `cx` is shared.
    fn bind(cx: &mut Context) -> Row {
        Self::sort(cx).bind(cx)
    }
}

/// Equal scalar interpretations; query construction additionally checks
/// nominal compatibility.
pub trait Same<T: Sorted>: Sorted {}
impl<A: Sorted, B: Sorted<Shape = A::Shape>> Same<B> for A {}

/// Integer endpoints for typed bitwise maps and contextual comparison
/// interpretation.
pub trait Integer: Sorted {
    /// Whether comparisons without an explicit signedness use signed
    /// interpretation.
    const SIGNED: bool;
}

macro_rules! integers {
    ($signed:expr; $($t:ty),* $(,)?) => {$(
        impl Sorted for $t {
            type Shape = Cons<Self, Nil>;
            fn sort(cx: &Context) -> Sort { cx.bv_sort(<$t>::BITS) }
        }
        impl Integer for $t { const SIGNED: bool = $signed; }
    )*};
}
integers!(false; u8, u16, u32, u64, u128, usize);
integers!(true; i8, i16, i32, i64, i128, isize);
impl Sorted for bool {
    type Shape = Cons<Self, Nil>;
    fn sort(cx: &Context) -> Sort {
        cx.bool_sort()
    }
}
impl Sorted for () {
    type Shape = Nil;
    fn sort(cx: &Context) -> Sort {
        cx.unit_sort()
    }
}
macro_rules! tuples {
    ($a:ident $(,$b:ident)* $(,)?) => {
        impl<$a: Sorted, $($b: Sorted),*> Sorted for ($a, $($b,)*)
        where ($($b,)*): Sorted, $a::Shape: Concat<<($($b,)*) as Sorted>::Shape> {
            type Shape = <$a::Shape as Concat<<($($b,)*) as Sorted>::Shape>>::Output;
            fn sort(cx: &Context) -> Sort {
                cx.product_sort([$a::sort(cx), $($b::sort(cx)),*])
            }
            fn bind(cx: &mut Context) -> Row {
                let sorts = [$a::bind(cx).sort, $($b::bind(cx).sort),*];
                cx.product_sort(sorts).bind(cx)
            }
        }
        tuples!($($b),*);
    };
    () => {};
}
tuples!(A, B, C, D, E, F, G, H, I, J, K, L);

/// The declarations generated by one concrete Rust record.
///
/// A record's `Sorted::bind` binds each member type before its own sort, as
/// tuple binding does, so members keep their configured storage.
pub trait Record: Sorted {
    /// The argument tuple whose flattened fields form the record.
    type Args: Sorted;
    /// The raw tuple declaration of this record in `cx`.
    fn declaration(cx: &Context) -> crate::TupleDecl;
}

/// An accessor or constructor descriptor with Rust endpoint interpretations.
///
/// Like its raw declaration it is a plain copyable name, resolved against the
/// context that it meets. Creating a declaration leaves its physical
/// representation configurable until it is bound.
pub struct FuncDecl<D, R> {
    /// The erased declaration this descriptor names.
    ///
    /// Replacing it must preserve the Rust endpoint interpretations. Use
    /// `from_raw` when attaching interpretations to a different declaration.
    pub raw: crate::FuncDecl,
    types: PhantomData<fn(D) -> R>,
}
impl<D: Sorted, R: Sorted> FuncDecl<D, R> {
    /// Attaches Rust endpoint interpretations to a raw declaration, checking
    /// both sorts in `cx`.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint sort differs from the Rust interpretation.
    pub fn from_raw(cx: &Context, raw: crate::FuncDecl) -> Self {
        let info = raw.resolve(cx);
        assert_eq!(
            info.domain,
            D::sort(cx),
            "typed declaration domain mismatch"
        );
        assert_eq!(info.range, R::sort(cx), "typed declaration range mismatch");
        Self {
            raw,
            types: PhantomData,
        }
    }
    /// The declared name, resolved in `cx`.
    pub fn name(&self, cx: &Context) -> String {
        self.raw.name(cx)
    }
    /// The declaration as a source-free query from the record to the member,
    /// for use as a left operand.
    ///
    /// # Panics
    ///
    /// Panics if the declaration belongs to another context or a required
    /// scalar leaf is unbound, as in [`query::Query::decl`].
    #[inline(always)]
    pub fn query(self, cx: &Context) -> Query<D, R> {
        Query::wrap(query::Query::decl(cx, self.raw))
    }
}
impl<D, R> Clone for FuncDecl<D, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D, R> Copy for FuncDecl<D, R> {}
impl<D: Sorted, R: Sorted> Domain for FuncDecl<D, R> {
    #[inline(always)]
    fn bind(self, cx: &mut Context) -> Row {
        self.raw.bind(cx)
    }
    #[inline(always)]
    fn row(self, cx: &Context) -> Row {
        self.raw.row(cx)
    }
}

/// A runtime query plus its Rust endpoint interpretations; the marker
/// occupies no storage.
pub struct Query<D, R, A = ()> {
    /// The erased runtime query.
    ///
    /// Direct changes must preserve the Rust endpoint interpretations. Use
    /// `from_raw` to check a replacement query's endpoint sorts.
    pub raw: query::Query<A>,
    types: PhantomData<fn(D) -> R>,
}
/// An owning runtime relation plus its Rust endpoint interpretations.
pub struct Rel<D, R> {
    /// The erased stored relation.
    ///
    /// Direct changes must preserve the Rust endpoint interpretations. Use
    /// `from_raw` to check a replacement relation's endpoint sorts.
    pub raw: crate::Rel,
    types: PhantomData<fn(D) -> R>,
}
/// A semi-naive variable with Rust endpoint interpretations over the shared
/// runtime protocol.
pub struct Variable<D, R> {
    /// The erased runtime variable.
    ///
    /// Direct changes must preserve the Rust endpoint interpretations and
    /// the raw variable's frontier protocol.
    pub raw: crate::Variable,
    types: PhantomData<fn(D) -> R>,
}
impl<D: Sorted, R: Sorted, A> Query<D, R, A> {
    #[inline(always)]
    fn wrap(raw: query::Query<A>) -> Self {
        Self {
            raw,
            types: PhantomData,
        }
    }
    /// Attaches Rust endpoint interpretations to a raw query, checking both
    /// sorts in its context.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint sort differs from the Rust interpretation.
    pub fn from_raw(raw: query::Query<A>) -> Self {
        assert_eq!(
            raw.domain.sort,
            D::sort(&raw.cx),
            "typed query domain mismatch"
        );
        assert_eq!(
            raw.range.sort,
            R::sort(&raw.cx),
            "typed query range mismatch"
        );
        Self::wrap(raw)
    }
}
impl<D: Sorted, R: Sorted, A: Exec> Query<D, R, A> {
    /// Counts distinct complete pairs without materializing the result
    /// relation.
    pub fn count(&self) -> num_bigint::BigUint {
        self.raw.count()
    }
    /// Materializes the complete pair set, retaining its field storage and
    /// endpoint universe.
    pub fn collect(&self) -> Rel<D, R> {
        Rel::wrap(self.raw.collect())
    }
    /// Executes until the first accepting branch; no concrete row enumeration
    /// is implied.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
    /// Visits borrowed symbolic branches, which may overlap or share predicate
    /// cells.
    ///
    /// These are not decoded Rust rows. The callback may stop early or invoke
    /// another query.
    pub fn drive<B>(
        &self,
        then: &mut impl for<'a> FnMut(crate::Binding<'a>, crate::Binding<'a>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        self.raw.drive(then)
    }
}
impl<T: Sorted> Query<T, T> {
    /// The identity relation over the whole runtime sort of `T`.
    ///
    /// # Panics
    ///
    /// Panics if a scalar leaf of `T` is not bound in `cx`.
    #[inline(always)]
    pub fn id(cx: &Context) -> Self {
        Self::wrap(query::Query::id(cx, T::sort(cx)))
    }
}

/// A record projected onto every field but one.
///
/// Its sort is fixed by the query that built it and is not a Rust value: it
/// composes, compares, and collects, but cannot be declared, inserted, or
/// decoded, so `sort` fails rather than guessing which field was removed.
/// Every one-field projection of `R` shares this type, so comparisons between
/// projections that remove different fields type-check, and runtime checks
/// reject them only when the remaining sorts differ.
pub struct Except<R>(PhantomData<fn() -> R>);
impl<R: Sorted> Sorted for Except<R> {
    type Shape = Cons<Self, Nil>;
    fn sort(_cx: &Context) -> Sort {
        panic!("projected record has no static sort; it is carried by the query that built it")
    }
}

impl<D: Record> Query<D, Except<D>> {
    /// Projects a record onto every field but `decl`, for comparisons that
    /// ignore one attribute.
    ///
    /// # Panics
    ///
    /// Panics if `decl` is the constructor or a scalar leaf of `D` is not bound
    /// in `cx`.
    pub fn except<T: Sorted>(cx: &Context, decl: FuncDecl<D, T>) -> Self {
        // `FuncDecl<D, T>` checks its domain against `D` when built. Direct
        // changes to its public raw declaration must preserve that invariant.
        // A constructor has no member position.
        let info = decl.raw.resolve(cx);
        Self::wrap(query::Query::except(
            cx,
            info.domain,
            info.member
                .expect("except declaration must be a member accessor"),
        ))
    }
}
impl<D, R, A: Clone> Clone for Query<D, R, A> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            types: PhantomData,
        }
    }
}

impl<D: Sorted, R: Sorted> Rel<D, R> {
    #[inline(always)]
    fn wrap(raw: crate::Rel) -> Self {
        Self {
            raw,
            types: PhantomData,
        }
    }

    /// An empty binary relation over the runtime sorts of `D` and `R` in
    /// `cx`.
    ///
    /// # Panics
    ///
    /// Panics if a scalar leaf of either endpoint is not bound in `cx`.
    pub fn new(cx: &Context) -> Self {
        Self::wrap(crate::Rel::new(cx, D::sort(cx), R::sort(cx)))
    }

    /// Attaches Rust endpoint interpretations to a raw relation, checking
    /// both sorts in its context.
    ///
    /// # Panics
    ///
    /// Panics if either endpoint sort differs from the Rust interpretation.
    pub fn from_raw(raw: crate::Rel) -> Self {
        let cx = raw.root.context();
        assert_eq!(
            raw.domain.sort,
            D::sort(cx),
            "typed relation domain mismatch"
        );
        assert_eq!(raw.range.sort, R::sort(cx), "typed relation range mismatch");
        Self::wrap(raw)
    }

    /// Retains this relation's owner, field backends, and set or binary
    /// universe, with no rows.
    pub fn empty_like(&self) -> Self {
        Self::wrap(self.raw.empty_like())
    }

    /// The exact cardinality of the stored pair set.
    pub fn count(&self) -> num_bigint::BigUint {
        self.raw.count()
    }

    /// Whether the stored pair set contains the given row or domain-range
    /// pair.
    ///
    /// # Panics
    ///
    /// Panics if `args` has an incompatible shape or sort, or holds a
    /// predicate, as in [`crate::Rel::contains`].
    pub fn contains(&self, args: impl IntoRow) -> bool {
        self.raw.contains(args)
    }

    /// Whether the stored pair set holds no pair.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Eager native Boolean algebra on the same stored index and endpoint
    /// universe.
    ///
    /// Relational operators such as `union` and `minus` instead build lazy
    /// queries.
    ///
    /// # Panics
    ///
    /// Panics if the operands differ in declared universe or physical columns.
    pub fn apply_bool(&self, op: BoolOp, other: &Self) -> Self {
        Self::wrap(self.raw.apply_bool(op, &other.raw))
    }

    /// Eager native complement within this relation's exact set or binary
    /// universe.
    pub fn not(&self) -> Self {
        Self::wrap(self.raw.not())
    }

    /// The pairs holding the least rank under each key, where the rank is the
    /// last field of `R` and compares as an unsigned integer.
    ///
    /// Every other field is a key, a symbolic domain included; see
    /// [`Rel::min_range`](crate::Rel::min_range). A signed last field orders
    /// as its unsigned encoding.
    ///
    /// # Panics
    ///
    /// Panics if the range has no scalar column or its last column has a
    /// symbol sort, as in [`crate::Rel::min_range`].
    pub fn min_range(&self) -> Self {
        Self::wrap(self.raw.min_range())
    }

    /// Inserts a row or domain-range pair, mixing concrete values and scalar
    /// predicates.
    ///
    /// Tuples may supply a record's fields; each predicate denotes independent
    /// column choices.
    ///
    /// # Panics
    ///
    /// Panics if a row has an incompatible shape or sort, or a predicate
    /// belongs to another context or column, as in [`crate::Rel::insert_rows`].
    pub fn insert(&mut self, args: impl IntoRow) {
        self.raw.insert(args);
    }

    /// Builds and unions a batch of concrete or symbolic rows, preserving
    /// each row's correlations.
    ///
    /// # Panics
    ///
    /// Panics if a row has an incompatible shape or sort, or a predicate
    /// belongs to another context or column, as in [`crate::Rel::insert_rows`].
    pub fn insert_rows(&mut self, rows: impl IntoIterator<Item = impl IntoRow>) {
        self.raw.insert_rows(rows);
    }
}
impl<R: Sorted> Rel<R, R> {
    /// An empty set over the runtime sort of `R`, whose universe is the
    /// identity relation.
    ///
    /// # Panics
    ///
    /// Panics if a scalar leaf of `R` is not bound in `cx`.
    pub fn new_set(cx: &Context) -> Self {
        Self::wrap(crate::Rel::new_set(cx, R::sort(cx)))
    }
}
impl<D, R> Clone for Rel<D, R> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            types: PhantomData,
        }
    }
}

impl<D: Sorted, R: Sorted> Variable<D, R> {
    /// Seeds candidates for the first `changed` barrier, preserving the
    /// initial relation's storage.
    #[inline(always)]
    pub fn new(initial: Rel<D, R>) -> Self {
        Self {
            raw: crate::Variable::new(initial.raw),
            types: PhantomData,
        }
    }
    /// An owning snapshot of the facts accepted before the current frontier.
    #[inline(always)]
    pub fn stable(&self) -> Rel<D, R> {
        Rel::wrap(self.raw.stable.clone())
    }
    /// An owning snapshot of the frontier for this round's positive linear
    /// rules.
    #[inline(always)]
    pub fn recent(&self) -> Rel<D, R> {
        Rel::wrap(self.raw.recent.clone())
    }
    /// Submits collected candidates; they remain invisible until `changed`.
    ///
    /// # Panics
    ///
    /// Panics if the candidate belongs to another context or differs in
    /// declared universe or physical index.
    #[inline(always)]
    pub fn insert(&mut self, rel: &Rel<D, R>) {
        self.raw.insert(&rel.raw);
    }
    /// An owning snapshot of all accepted facts, excluding pending
    /// candidates.
    #[inline(always)]
    pub fn current(&self) -> Rel<D, R> {
        Rel::wrap(self.raw.current())
    }
    /// Advances the barrier using exact complete-fact novelty and reports a
    /// nonempty frontier.
    #[inline(always)]
    pub fn changed(&mut self) -> bool {
        self.raw.changed()
    }
    /// Consumes the result after iteration has drained.
    ///
    /// # Panics
    ///
    /// Panics if any recent or pending facts remain.
    #[inline(always)]
    pub fn complete(self) -> Rel<D, R> {
        Rel::wrap(self.raw.complete())
    }
}

/// A right-hand construction recipe.
///
/// It resolves completely before runtime execution.
pub trait Operand<D: Sorted> {
    /// The range interpretation of the resolved query.
    type R: Sorted;
    /// The executable node type of the resolved query.
    type Node: Exec;
    /// Resolves the recipe against the consuming query's context and input
    /// row into a raw query.
    ///
    /// # Panics
    ///
    /// Panics if the input row or operand belongs to another context.
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node>;
}

/// Identity whose row type and physical columns come from the consuming
/// query.
#[derive(Clone, Copy)]
pub struct Ident;

/// The identity operand; its endpoints come from the consuming query.
#[inline(always)]
pub const fn id() -> Ident {
    Ident
}

impl Ident {
    /// Pairs each domain value with its witnesses in `a`, exposing complete
    /// pairs of `a` as the range.
    ///
    /// The result has domain `A::D` and range `(A::D, A::R)`.
    #[inline(always)]
    pub fn and<A: IntoQuery>(self, a: A) -> Query<A::D, (A::D, A::R), query::Join<(), A::Node>>
    where
        (A::D, A::R): Sorted,
    {
        Query::wrap(query::id().and(a.into_query().raw))
    }
}

impl<D: Sorted> Operand<D> for Ident {
    type R = D;
    type Node = ();
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<()> {
        RawOperand::resolve(query::id(), cx, input)
    }
}

impl<D: Same<T::D>, T: IntoQuery> Operand<D> for T {
    type R = T::R;
    type Node = T::Node;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        let q = self.into_query().raw;
        assert!(
            q.cx == *cx && input.sort.context == cx.id,
            "typed operand belongs to another context"
        );
        q
    }
}
/// A declaration resolves as the source-free query from its record to the
/// member, so it needs the consuming query's context.
impl<X: Same<D>, D: Sorted, R: Sorted> Operand<X> for FuncDecl<D, R> {
    type R = R;
    type Node = ();
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<()> {
        assert_eq!(
            input.sort.context, cx.id,
            "typed operand input belongs to another context"
        );
        RawOperand::resolve(self.raw, cx, input)
    }
}
impl<X: Same<D>, D: Sorted, R: Sorted> Operand<X> for &FuncDecl<D, R> {
    type R = R;
    type Node = ();
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<()> {
        Operand::<X>::resolve(*self, cx, input)
    }
}
/// A membership disjunction resolves both typed operands against the
/// consuming query and accepts the domain keys either one accepts.
impl<D: Sorted, A: Operand<D>, B: Operand<D>> Operand<D> for crate::pred::Disj<A, B> {
    type R = D;
    type Node = query::Union<query::Join<(), A::Node>, query::Join<(), B::Node>>;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        RawOperand::resolve(
            crate::pred::Disj(self.0.resolve(cx, input), self.1.resolve(cx, input)),
            cx,
            input,
        )
    }
}
/// Explicit raw maps preserve the input type and validate their sort during
/// resolution.
impl<D: Sorted, F: FnOnce(&Context, &Row) -> crate::MapSpec> Operand<D> for crate::map::Map<F> {
    type R = D;
    type Node = crate::map::Transform;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        RawOperand::resolve(self, cx, input)
    }
}

/// Typed versions of the existing combinators.
///
/// Only these signatures interpret Rust types.
pub trait IntoQuery: Sized {
    /// The domain interpretation.
    type D: Sorted;
    /// The range interpretation.
    type R: Sorted;
    /// The executable node type of the underlying raw query.
    type Node: Exec;
    /// Converts this value into a typed query.
    fn into_query(self) -> Query<Self::D, Self::R, Self::Node>;

    /// Membership disjunction on a shared domain; witness ranges may differ.
    ///
    /// The operand `b` must accept this query's domain `D`. The result is a
    /// predicate operand, not an enumerable relation. Use it in `with` to
    /// retain the original pairs, or compose explicitly to expose its accepted
    /// domain.
    #[inline(always)]
    fn or<B: Operand<Self::D>>(self, b: B) -> crate::pred::Disj<Self, B> {
        crate::pred::Disj(self, b)
    }

    /// Lazy full-pair union: a pair is in the result when either operand holds
    /// it.
    ///
    /// Both operands share domain `D` and range `R`, which the result keeps.
    /// Borrow a query to reuse its structure; stored operands retain snapshots.
    #[inline(always)]
    fn union<B: Operand<Self::D>>(
        self,
        b: B,
    ) -> Query<Self::D, Self::R, query::Union<Self::Node, B::Node>>
    where
        Self::R: Same<B::R>,
    {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.domain);
        Query::wrap(a.union(b))
    }

    /// Lazy intersection: a pair is in the result when both operands hold it,
    /// matching both endpoints rather than just domain keys.
    ///
    /// Both operands share domain `D` and range `R`, which the result keeps.
    #[inline(always)]
    fn intersection<B: Operand<Self::D>>(
        self,
        b: B,
    ) -> Query<Self::D, Self::R, query::Join<Self::Node, B::Node>>
    where
        Self::R: Same<B::R>,
    {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.domain);
        Query::wrap(a.intersection(b))
    }

    /// Lazy complement within the query's declared set or binary universe.
    ///
    /// The result contains the missing pairs within that universe. Equal Rust
    /// endpoint types alone do not imply a set universe. Domain and range are
    /// unchanged.
    #[inline(always)]
    fn complement(self) -> Query<Self::D, Self::R, query::Diff<(), Self::Node>> {
        Query::wrap(self.into_query().raw.complement())
    }

    /// The converse relation: each pair `(d, r)` becomes `(r, d)`.
    ///
    /// The result has domain `R` and range `D`; no rows are copied.
    #[inline(always)]
    fn inv(self) -> Query<Self::R, Self::D, Self::Node> {
        Query::wrap(self.into_query().raw.inv())
    }

    /// The domain subidentity `keys(e) = 1 - (1 - e)`: the identity over the
    /// bound domain, restricted to the keys that have a witness.
    ///
    /// Stays lazy.
    #[inline(always)]
    fn keys(self) -> Query<Self::D, Self::D, query::Join<(), Self::Node>> {
        Query::wrap(self.into_query().raw.keys())
    }

    /// The range subidentity `values(e) = keys(e^-1)`: the identity over the
    /// bound range, restricted to the values that have a witness.
    ///
    /// Stays lazy; the range keeps its Rust interpretation and its configured
    /// storage.
    #[inline(always)]
    fn values(self) -> Query<Self::R, Self::R, query::Join<(), Self::Node>> {
        Query::wrap(self.into_query().raw.values())
    }

    /// Lazy relational composition: `(d, x)` in this query and `(x, r)` in `b`
    /// yield `(d, r)`.
    ///
    /// The operand `b` must accept this query's range `R`; the result has
    /// domain `D` and range `B::R`.
    #[inline(always)]
    fn select<B: Operand<Self::R>>(
        self,
        b: B,
    ) -> Query<Self::D, B::R, query::Join<Self::Node, B::Node>> {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.range);
        Query::wrap(a.select(b))
    }
    /// Lazy pairing on a shared domain: `(d, r)` here and `(d, s)` in `b` yield
    /// `(d, (r, s))`.
    ///
    /// The operand `b` must accept this query's domain `D`; the result has
    /// domain `D` and range `(R, B::R)`, preserving each witness's
    /// correlations.
    #[inline(always)]
    fn and<B: Operand<Self::D>>(
        self,
        b: B,
    ) -> Query<Self::D, (Self::R, B::R), query::Join<Self::Node, B::Node>>
    where
        (Self::R, B::R): Sorted,
    {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.domain);
        Query::wrap(a.and(b))
    }
    /// Lazy range restriction: keeps `(d, r)` exactly when `b` accepts `r` as a
    /// domain key.
    ///
    /// The operand `b` must accept this query's range `R`; witnesses of `b` are
    /// hidden and the result keeps domain `D` and range `R`.
    #[inline(always)]
    fn with<B: Operand<Self::R>>(
        self,
        b: B,
    ) -> Query<Self::D, Self::R, query::Join<Self::Node, B::Node>> {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.range);
        Query::wrap(a.with(b))
    }
    /// Lazy keyed exclusion: remove a left pair whenever the right operand has
    /// its domain key.
    ///
    /// The operand `b` must accept this query's domain `D`; the result keeps
    /// domain `D` and range `R`. To compare complete pairs instead, use
    /// `difference`.
    #[inline(always)]
    fn minus<B: Operand<Self::D>>(
        self,
        b: B,
    ) -> Query<Self::D, Self::R, query::Diff<Self::Node, B::Node>> {
        let a = self.into_query().raw;
        let b = b.resolve(&a.cx, &a.domain);
        Query::wrap(a.minus(b))
    }
    /// Lazy set difference on complete pairs: the pairs here that `b` does not
    /// hold, keeping domain `D` and range `R`.
    ///
    /// Runtime checks still compare endpoint sorts, including the
    /// query-dependent sorts carried by `Except<D>`, which name the remaining
    /// sorts rather than the removed field. Stays lazy: `count`,
    /// `collect`, or `drive` the residual as counterexamples to containment,
    /// where `is_empty` reports containment alone.
    #[inline(always)]
    fn difference<B: IntoQuery<D = Self::D, R = Self::R>>(
        self,
        b: B,
    ) -> Query<Self::D, Self::R, query::Diff<Self::Node, B::Node>> {
        Query::wrap(self.into_query().raw.difference(b.into_query().raw))
    }
    /// Whether every complete pair here is also held by `b`.
    ///
    /// Executes a decision procedure and reports containment alone;
    /// `difference` keeps the pairs that violate it as counterexamples. Both
    /// operands share domain `D` and range `R`. To ask instead whether every
    /// domain key here is a key of `b`, use `minus` and `is_empty`, which hides
    /// the right operand's range.
    #[expect(
        clippy::wrong_self_convention,
        reason = "a query is consumed by every combinator, containment included"
    )]
    #[inline(always)]
    fn is_subset<B: IntoQuery<D = Self::D, R = Self::R>>(self, b: B) -> bool {
        self.difference(b).is_empty()
    }
    /// The complete pairs on which two relations disagree; empty exactly when
    /// they hold the same pairs.
    ///
    /// The union of both `difference`s. Stays lazy: `count`, `collect`, or
    /// `drive` the result for a counterexample, where `==` reports emptiness
    /// alone.
    #[inline(always)]
    fn symmetric_difference<B: IntoQuery<D = Self::D, R = Self::R>>(
        self,
        b: B,
    ) -> Query<
        Self::D,
        Self::R,
        query::Union<query::Diff<Self::Node, B::Node>, query::Diff<B::Node, Self::Node>>,
    >
    where
        Self::Node: Clone,
        B::Node: Clone,
    {
        Query::wrap(
            self.into_query()
                .raw
                .symmetric_difference(b.into_query().raw),
        )
    }
}

/// Semantic comparison under the same contract as the raw
/// [`Query`](crate::Query)'s `==`: two operands are equal when no complete
/// pair distinguishes them, which executes a decision procedure and reports
/// emptiness alone; `symmetric_difference` keeps the differing pairs.
impl<D: Sorted, R: Sorted, A: Exec, B: Exec> PartialEq<Query<D, R, B>> for Query<D, R, A> {
    #[inline(always)]
    fn eq(&self, other: &Query<D, R, B>) -> bool {
        self.raw == other.raw
    }
}

impl<D: Sorted, R: Sorted, A: Exec> PartialEq<Rel<D, R>> for Query<D, R, A> {
    #[inline(always)]
    fn eq(&self, other: &Rel<D, R>) -> bool {
        self.raw == other.raw
    }
}

impl<D: Sorted, R: Sorted, B: Exec> PartialEq<Query<D, R, B>> for Rel<D, R> {
    #[inline(always)]
    fn eq(&self, other: &Query<D, R, B>) -> bool {
        self.raw == other.raw
    }
}

impl<D: Sorted, R: Sorted> PartialEq for Rel<D, R> {
    #[inline(always)]
    fn eq(&self, other: &Rel<D, R>) -> bool {
        self.raw == other.raw
    }
}
impl<D: Sorted, R: Sorted, A: Exec> IntoQuery for Query<D, R, A> {
    type D = D;
    type R = R;
    type Node = A;
    #[inline(always)]
    fn into_query(self) -> Self {
        self
    }
}
impl<'a, D: Sorted, R: Sorted, A: Exec> IntoQuery for &'a Query<D, R, A> {
    type D = D;
    type R = R;
    type Node = &'a A;
    #[inline(always)]
    fn into_query(self) -> Query<D, R, &'a A> {
        Query::wrap((&self.raw).into_query())
    }
}
impl<'a, A> IntoQuery for &&'a A
where
    &'a A: IntoQuery,
{
    type D = <&'a A as IntoQuery>::D;
    type R = <&'a A as IntoQuery>::R;
    type Node = <&'a A as IntoQuery>::Node;
    #[inline(always)]
    fn into_query(self) -> Query<Self::D, Self::R, Self::Node> {
        (*self).into_query()
    }
}
impl<D: Sorted, R: Sorted> IntoQuery for &Rel<D, R> {
    type D = D;
    type R = R;
    type Node = crate::index::IndexRoot;
    #[inline(always)]
    fn into_query(self) -> Query<D, R, Self::Node> {
        Query::wrap((&self.raw).into_query())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/typed.rs"]
mod tests;
