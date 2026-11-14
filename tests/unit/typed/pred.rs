//! Checks for typed predicate recipes.
//!
//! Predicate recipes are contextless until they meet an input row. The Rust
//! endpoint type decides whether a comparison lowers to a signed or an
//! unsigned runtime operation unless an explicit `s`/`u` operation overrides
//! it, and the same recipes serve two roles here: Boolean operands over a
//! query's range, and projection tests attached to a declaration or a
//! structural projection query.

use std::collections::BTreeSet;

use super::{self as pred};
use crate::Schema;
use crate::context::Context;
use crate::run::Exec;
use crate::typed::{FuncDecl, IntoQuery, Query, Record, Rel, Sorted};

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &crate::rel::Rel) -> BTreeSet<Vec<u64>> {
    let cx = rel.root.context();
    let widths: Vec<u32> = (0..rel.root.arity())
        .map(|i| cx.resolve(rel.root.column(i)).sort().bit_width().unwrap())
        .collect();
    let mut out = BTreeSet::new();
    let mut row = vec![0; widths.len()];
    loop {
        if rel.root.contains(&row) {
            out.insert(row.clone());
        }
        let Some(i) = (0..row.len()).rev().find(|&i| row[i] + 1 < 1 << widths[i]) else {
            return out;
        };
        row[i] += 1;
        row[i + 1..].fill(0);
    }
}

/// A record with one unsigned and one signed member.
#[derive(Clone, Copy, Debug, Schema)]
struct Edge {
    from: u8,
    to: i8,
}

/// `{(0,0), ..., (7,7)}` over a bound 8-bit endpoint; a restriction of the
/// range shows exactly which values a recipe admits.
fn bytes(cx: &Context) -> Rel<u8, u8> {
    let mut rel = Rel::<u8, u8>::new_set(cx);
    rel.insert_rows(0_u8..8);
    rel
}

/// `{-4, -1, 0, 3}` as signed bytes, where signed and unsigned orderings
/// disagree.
fn signed(cx: &Context) -> Rel<i8, i8> {
    let mut rel = Rel::<i8, i8>::new_set(cx);
    rel.insert_rows([-4_i8, -1, 0, 3]);
    rel
}

/// The domain values a restriction admits, in order.
fn held<D: Sorted, R: Sorted, A: Exec>(q: Query<D, R, A>) -> Vec<u64> {
    rows(&q.collect().raw).into_iter().map(|r| r[0]).collect()
}

/// The same, reinterpreting the stored bits as signed bytes. The enumeration
/// order stays unsigned, so the negative values come last.
fn signed_held<A: Exec>(q: Query<i8, i8, A>) -> Vec<i8> {
    held(q).into_iter().map(|v| v as u8 as i8).collect()
}

// --- Boolean operands and reuse ----------------------------------------

#[test]
fn recipes_compose_as_operands() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    // As operands, composites resolve through the ordinary lazy combinators.
    assert_eq!(held((&rel).with(pred::eq(1_u8) | pred::eq(4_u8))), [1, 4]);
    assert_eq!(held((&rel).with(pred::ge(2_u8) & pred::lt(4_u8))), [2, 3]);
    assert_eq!(
        held((&rel).with(!pred::is_in([0_u8, 1, 2, 3, 4, 5]))),
        [6, 7]
    );
    assert_eq!(held((&rel).with(pred::eq(1_u8).or(pred::eq(4_u8)))), [1, 4]);
    assert_eq!(
        held((&rel).with(pred::ge(2_u8).and(pred::lt(4_u8)))),
        [2, 3]
    );

    // Borrowed recipes are operands too, through one or two layers of
    // reference.
    let single = pred::eq(1_u8);
    let disj = pred::eq(1_u8) | pred::eq(4_u8);
    let conj = pred::ge(2_u8) & pred::lt(4_u8);
    let neg = !pred::is_in([0_u8, 1, 2, 3, 4, 5]);
    assert_eq!(held((&rel).with(&single)), [1]);
    assert_eq!(held((&rel).with(&&single)), [1]);
    assert_eq!(held((&rel).with(&disj)), [1, 4]);
    assert_eq!(held((&rel).with(&&disj)), [1, 4]);
    assert_eq!(held((&rel).with(&conj)), [2, 3]);
    assert_eq!(held((&rel).with(&&conj)), [2, 3]);
    assert_eq!(held((&rel).with(&neg)), [6, 7]);
    assert_eq!(held((&rel).with(&&neg)), [6, 7]);
}

#[test]
fn borrowed_mixed_recipes_reuse_range_and_declaration_leaves() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    macro_rules! check {
        ($range:expr, $expected:expr) => {{
            let range = $range;
            let p = pred::eq(1_u8) | &range;
            for _ in 0..2 {
                assert_eq!(held((&rel).with(&p)), $expected);
                assert_eq!(held((&rel).with(&&p)), $expected);
            }
            let p = pred::eq(1_u8) | range;
            assert_eq!(held((&rel).with(&p)), $expected);
        }};
    }
    check!(2_u16..4, [1, 2, 3]);
    check!(6_u16.., [1, 6, 7]);
    check!(2_u16..=4, [1, 2, 3, 4]);
    check!(..2_u16, [0, 1]);
    check!(..=2_u16, [0, 1, 2]);
    check!(.., [0, 1, 2, 3, 4, 5, 6, 7]);
    let p = !(pred::ge(2_u8) & (2_u8..4));
    assert_eq!(held((&rel).with(&p)), [0, 1, 4, 5, 6, 7]);
    let neg = signed(&cx);
    let p = pred::eq(3_i8) | (-4_i8..0);
    assert_eq!(signed_held((&neg).with(&p)), [3, -4, -1]);
    assert_eq!(signed_held((&neg).with(&p)), [3, -4, -1]);
    drop((rel, neg));

    Edge::bind(&mut cx);
    let decl = Edge::declaration(&cx);
    let field = FuncDecl::<Edge, u8>::from_raw(&cx, decl.accessors[0]);
    let mut rel = Rel::<Edge, Edge>::new_set(&cx);
    rel.insert_rows([Edge { from: 1, to: -2 }, Edge { from: 4, to: 0 }]);
    let member = &field;
    let p = pred::eq((0_u8, 0_i8)) | member;
    for _ in 0..2 {
        assert_eq!((&rel).with(&p).collect().count(), 2_u32.into());
        assert_eq!((&rel).with(&&p).collect().count(), 2_u32.into());
    }
}

// --- comparisons -------------------------------------------------------

#[test]
fn comparisons_take_their_signedness_from_the_endpoint() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    // Unsigned endpoints choose unsigned runtime comparisons.
    assert_eq!(held((&rel).with(pred::lt(2_u8))), [0, 1]);
    assert_eq!(held((&rel).with(pred::le(2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).with(pred::gt(5_u8))), [6, 7]);
    assert_eq!(held((&rel).with(pred::ge(5_u8))), [5, 6, 7]);

    // A signed endpoint reads the same bits as two's complement, so
    // -4 < -1 < 0 < 3.
    let neg = signed(&cx);
    assert_eq!(signed_held((&neg).with(pred::lt(0_i8))), [-4, -1]);
    assert_eq!(signed_held((&neg).with(pred::le(-1_i8))), [-4, -1]);
    assert_eq!(signed_held((&neg).with(pred::gt(-1_i8))), [0, 3]);
    assert_eq!(signed_held((&neg).with(pred::ge(-1_i8))), [0, 3, -1]);

    // Explicit operations override the endpoint: unsigned readings put the
    // negatives last.
    assert_eq!(signed_held((&neg).with(pred::ult(1_i8))), [0]);
    assert_eq!(signed_held((&neg).with(pred::ule(0_i8))), [0]);
    assert_eq!(signed_held((&neg).with(pred::ugt(3_i8))), [-4, -1]);
    assert_eq!(signed_held((&neg).with(pred::uge(3_i8))), [3, -4, -1]);
    assert_eq!(held((&rel).with(pred::slt(2_u8))), [0, 1]);
    assert_eq!(held((&rel).with(pred::sle(2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).with(pred::sgt(5_u8))), [6, 7]);
    assert_eq!(held((&rel).with(pred::sge(5_u8))), [5, 6, 7]);
    // A signed reading of an unsigned endpoint makes the high byte negative.
    let mut high = Rel::<u8, u8>::new_set(&cx);
    high.insert_rows([1_u8, 200]);
    assert_eq!(held((&high).with(pred::lt(2_u8))), [1]);
    assert_eq!(held((&high).with(pred::slt(2_u8))), [1, 200]);
    assert!(held((&high).with(pred::sgt(1_u8))).is_empty());

    // Equality needs no ordering interpretation at all.
    assert_eq!(held((&rel).with(pred::eq(4_u8))), [4]);
    assert_eq!(held((&rel).with(pred::ne(4_u8))), [0, 1, 2, 3, 5, 6, 7]);
}

// --- membership and ranges ---------------------------------------------

#[test]
fn membership_recipes_accept_collections_and_ranges() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    // Any borrowed or owned slice-like collection of literals.
    assert_eq!(held((&rel).with(pred::is_in([1_u8, 4]))), [1, 4]);
    assert_eq!(held((&rel).with(pred::is_in(vec![1_u8, 4]))), [1, 4]);
    assert_eq!(held((&rel).with(pred::is_in(&[1_u8, 4][..]))), [1, 4]);
    assert_eq!(held((&rel).with(pred::is_in(&[1_u8, 4]))), [1, 4]);
    assert_eq!(held((&rel).with(pred::is_in(&vec![1_u8, 4]))), [1, 4]);

    // Range shape decides which comparisons are needed; a range is itself an
    // operand.
    assert_eq!(held((&rel).with(pred::is_in(2_u8..5))), [2, 3, 4]);
    assert_eq!(held((&rel).with(2_u8..5)), [2, 3, 4]);
    assert_eq!(held((&rel).with(pred::is_in(6_u8..))), [6, 7]);
    assert_eq!(held((&rel).with(6_u8..)), [6, 7]);
    assert_eq!(held((&rel).with(pred::is_in(2_u8..=4))), [2, 3, 4]);
    assert_eq!(held((&rel).with(2_u8..=4)), [2, 3, 4]);
    assert_eq!(held((&rel).with(pred::is_in(..2_u8))), [0, 1]);
    assert_eq!(held((&rel).with(..2_u8)), [0, 1]);
    assert_eq!(held((&rel).with(pred::is_in(..=2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).with(..=2_u8)), [0, 1, 2]);
    assert_eq!(held((&rel).with(pred::is_in(..))), [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(held((&rel).with(..)), [0, 1, 2, 3, 4, 5, 6, 7]);

    // Draining an iterated `RangeInclusive` with `next` leaves its start past
    // its end, so the inclusive comparison and the start comparison contradict
    // and the exhausted range admits nothing, as it should. The strict
    // comparison for a reported excluded bound has its own test below.
    let mut exhausted = 2_u8..=4;
    while exhausted.next().is_some() {}
    assert!(held((&rel).with(pred::is_in(exhausted))).is_empty());

    // Ranges over a signed endpoint use signed comparisons for both of their
    // bounds.
    let neg = signed(&cx);
    assert_eq!(signed_held((&neg).with(-2_i8..4)), [0, 3, -1]);
}

#[test]
fn exhausted_inclusive_ranges_use_the_reported_bounds() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    let mut r = 2_u8..=4;
    assert_eq!(r.nth(usize::MAX), None);
    assert!(held((&rel).with(pred::is_in(r))).is_empty());
    let mut r = 2_u8..=4;
    assert_eq!(r.nth_back(usize::MAX), None);
    assert!(held((&rel).with(r)).is_empty());

    let neg = signed(&cx);
    let mut r = -4_i16..=3;
    assert_eq!(r.nth(usize::MAX), None);
    assert!(signed_held((&neg).with(r)).is_empty());
    let mut r = -4_i16..=3;
    assert_eq!(r.nth_back(usize::MAX), None);
    assert!(signed_held((&neg).with(pred::is_in(r))).is_empty());

    // Literal ordering stays contextual to the endpoint interpretation.
    assert_eq!(
        signed_held((&neg).with(std::ops::RangeInclusive::new(250_u8, 3))),
        [0, 3, -4, -1]
    );
    let mut max = Rel::<u8, u8>::new_set(&cx);
    max.insert(u8::MAX);
    assert_eq!(held((&max).with(255_u16..=255)), [255]);
    let mut r = 255_u8..=255;
    assert_eq!(r.next(), Some(255));
    assert!(held((&max).with(r)).is_empty());
}

// --- projection predicates ---------------------------------------------

#[test]
fn projections_carry_typed_tests_to_their_declared_member() {
    let mut cx = Context::default();
    Edge::bind(&mut cx);
    let EdgeSchema { from, to, .. } = Edge::schema(&cx);
    let mut edges = Rel::<Edge, Edge>::new_set(&cx);
    edges.insert_rows([
        Edge { from: 1, to: -2 },
        Edge { from: 4, to: 0 },
        Edge { from: 6, to: 3 },
    ]);
    let admitted = |q: Rel<Edge, Edge>| q.count();
    macro_rules! check {
        ($s:expr) => {
            check!($s; eq(4_u8) => 1, ne(4_u8) => 2, lt(4_u8) => 1, le(4_u8) => 2);
            check!($s; gt(4_u8) => 1, ge(4_u8) => 2, ult(4_u8) => 1, ule(4_u8) => 2);
            check!($s; ugt(4_u8) => 1, uge(4_u8) => 2, slt(4_u8) => 1, sle(4_u8) => 2);
            check!($s; sgt(4_u8) => 1, sge(4_u8) => 2);
            check!($s; is_in([1_u8, 6]) => 2, test(pred::eq(1_u8)) => 1);
        };
        ($s:expr; $($m:ident($a:expr) => $n:expr),*) => {
            $(assert_eq!(
                admitted((&edges).with($s.$m($a)).collect()),
                u32::into($n),
                stringify!($m)
            );)*
        };
    }
    // A declaration tests its own member and yields a predicate over the
    // record.
    check!(from);
    // The signed member reads the same recipe with signed comparisons.
    assert_eq!(admitted((&edges).with(to.lt(0_i8)).collect()), 1_u32.into());
    // A resolved projection predicate is a plain value: cloning it reuses the
    // recipe.
    let recipe = from.eq(4_u8);
    assert_eq!(
        admitted((&edges).with(recipe.clone()).collect()),
        1_u32.into()
    );
    assert_eq!(admitted((&edges).with(&recipe).collect()), 1_u32.into());

    // The same methods exist on a structural projection query built from that
    // declaration.
    let projection: Query<Edge, u8, ()> = from.query(&cx);
    check!(projection);

    // Explicit operations override each member's interpretation.
    assert_eq!(admitted((&edges).with(to.gt(0_i8)).collect()), 1_u32.into());
    assert_eq!(
        admitted((&edges).with(to.ugt(0_i8)).collect()),
        2_u32.into()
    );
    let signed_to: Query<Edge, i8, ()> = to.query(&cx);
    assert_eq!(
        admitted((&edges).with(signed_to.ugt(0_i8)).collect()),
        2_u32.into()
    );
    let mut high = Rel::<Edge, Edge>::new_set(&cx);
    high.insert_rows([Edge { from: 200, to: 0 }]);
    assert_eq!(
        admitted((&high).with(from.lt(4_u8)).collect()),
        0_u32.into()
    );
    assert_eq!(
        admitted((&high).with(from.slt(4_u8)).collect()),
        1_u32.into()
    );
    assert_eq!(
        admitted((&high).with(projection.slt(4_u8)).collect()),
        1_u32.into()
    );

    // A full range is a native predicate at every consumer.
    assert_eq!(admitted((&edges).with(to.test(..)).collect()), 3_u32.into());
    assert_eq!(
        admitted((&edges).with(from.test(pred::eq(1_u8) | ..)).collect()),
        3_u32.into()
    );

    // A derived view member tests the same record member inside filt.
    assert_eq!(
        admitted((&edges).filt(|e| e.from.eq(4_u8)).collect()),
        1_u32.into()
    );
}
