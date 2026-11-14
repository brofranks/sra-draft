//! Checks for `src/typed/expr.rs`.
//!
//! Expression closures run once, during construction: each one receives a
//! symbolic view of the query's range and returns either a transformed value
//! or a condition on it. These tests pin that map composition stays in one
//! bit map, that conditions are subidentities on the original input (so
//! negation and Boolean composition preserve the concrete meaning), and that
//! record, fixed-set, and interned views expose their own operations over the
//! same input. `Packet` is a hand-written record, `bytes` is a small one-word
//! relation, and `rows` reads ground rows by membership alone.

use std::collections::BTreeSet;

use super::{Condition, Expr, Functional, Test};
use crate::ast::{Sort, TupleDecl};
use crate::backend::MapSpec;
use crate::context::Context;
use crate::query::Join;
use crate::row::{IntoRow, RowRef, RowSink};
use crate::run::Exec;
use crate::typed::{
    Dictionary, Domain, FiniteSet, FuncDecl, Interned, IntoQuery, Query, Record, Rel, Sorted,
    Universe, View, pred,
};

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

/// A two-field record, written by hand the way a derive would generate it.
#[derive(Clone, Copy, Debug)]
struct Packet {
    src: u8,
    dst: u8,
}

impl Sorted for Packet {
    type Shape = <(u8, u8) as Sorted>::Shape;
    fn sort(cx: &Context) -> Sort {
        Self::declaration(cx).sort
    }
}

impl Record for Packet {
    type Args = (u8, u8);
    fn declaration(cx: &Context) -> TupleDecl {
        cx.tuple_decl("Packet", [("src", u8::sort(cx)), ("dst", u8::sort(cx))])
    }
}

impl IntoRow for Packet {
    fn record_sort(&self, cx: &Context) -> Option<Sort> {
        Some(Packet::sort(cx))
    }

    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        (self.src, self.dst).append(row, out);
    }
}

/// The member expressions of one `Packet` input, written by hand the way a
/// derive would generate them: every field is a function of the same closure
/// input.
struct PacketView<D> {
    src: Expr<D, u8>,
    dst: Expr<D, u8>,
}

impl<D> Clone for PacketView<D> {
    fn clone(&self) -> Self {
        Self {
            src: self.src.clone(),
            dst: self.dst.clone(),
        }
    }
}

impl View for Packet {
    type Expr<D: Sorted> = PacketView<D>;
    fn view<D: Sorted>(input: Expr<D, Self>) -> PacketView<D> {
        PacketView {
            src: input.field::<u8>(0),
            dst: input.field::<u8>(1),
        }
    }
}

impl<D: Sorted> IntoQuery for PacketView<D> {
    type D = D;
    type R = Packet;
    type Node = Join<<Expr<D, u8> as IntoQuery>::Node, <Expr<D, u8> as IntoQuery>::Node>;
    fn into_query(self) -> Query<D, Packet, Self::Node> {
        let q = crate::query::IntoQuery::and(self.src.into_query().raw, self.dst.into_query().raw);
        let domain = q.domain.sort;
        let range = Packet::sort(&q.cx);
        Query::from_raw(q.coerce(domain, range))
    }
}

impl<D: Sorted> Functional for PacketView<D> {}

impl<D: Sorted> PacketView<D> {
    fn test<P: pred::Predicate<Packet>>(
        self,
        p: P,
    ) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
        Functional::test(self, p)
    }

    fn eq(self, value: impl IntoRow) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
        Functional::eq(self, value)
    }

    fn ne(self, value: impl IntoRow) -> Condition<D, Test<<Self as IntoQuery>::Node>> {
        Functional::ne(self, value)
    }
}

struct Flags;
impl Universe for Flags {
    type Member = u8;
    const LEN: u32 = 4;
    fn index(_: u64, member: &u8) -> Option<u32> {
        (*member < 4).then_some(u32::from(*member))
    }
}
type FlagSet = FiniteSet<Flags>;

/// A record with a set-valued member, declared without a view: its tests
/// reach the member through the accessor declaration.
#[derive(Clone, Debug)]
struct Host {
    id: u8,
    flags: FlagSet,
}

impl Sorted for Host {
    type Shape = <(u8, FlagSet) as Sorted>::Shape;
    fn sort(cx: &Context) -> Sort {
        Self::declaration(cx).sort
    }
}

impl Record for Host {
    type Args = (u8, FlagSet);
    fn declaration(cx: &Context) -> TupleDecl {
        cx.tuple_decl("Host", [("id", u8::sort(cx)), ("flags", FlagSet::sort(cx))])
    }
}

impl IntoRow for Host {
    fn record_sort(&self, cx: &Context) -> Option<Sort> {
        Some(Host::sort(cx))
    }

    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        (self.id, &self.flags).append(row, out);
    }
}

struct Names;
impl Domain for Names {
    type Value = str;
    const NAME: &'static str = "ExprName";
}
type Name = Interned<Names>;

/// The domain values a collected restriction holds; a subidentity stores one
/// column, and a transformed range stores the original value in the first
/// one.
fn held<T: Sorted, A: Exec>(q: Query<T, T, A>) -> Vec<u64> {
    rows(&q.collect().raw).into_iter().map(|r| r[0]).collect()
}

/// `{(0,0), (1,1), ..., (7,7)}` over a bound 8-bit endpoint, so a transformed
/// range is keyed by its original value.
fn bytes(cx: &Context) -> Rel<u8, u8> {
    let mut rel = Rel::<u8, u8>::new_set(cx);
    rel.insert_rows(0_u8..8);
    rel
}

// --- bit maps ----------------------------------------------------------

#[test]
fn expression_maps_compose_into_one_bit_map() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    // Each bitwise operator against a literal extends the same pending bit
    // map, so a chain of them is still a single transformation of the range.
    let image = |q: Rel<u8, u8>| {
        rows(&q.raw)
            .into_iter()
            .map(|r| (r[0], r[1]))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        image((&rel).map(|x| x | 0b1000_u8).collect()),
        (0..8).map(|v| (v, v | 8)).collect::<Vec<_>>()
    );
    assert_eq!(
        image((&rel).map(|x| x & 0b0110_u8).collect()),
        (0..8).map(|v| (v, v & 6)).collect::<Vec<_>>()
    );
    assert_eq!(
        image((&rel).map(|x| x ^ 0b0011_u8).collect()),
        (0..8).map(|v| (v, v ^ 3)).collect::<Vec<_>>()
    );
    assert_eq!(
        image(
            (&rel)
                .map(|x| ((x & 0b0110_u8) ^ 0b0001_u8) | 0b1000_u8)
                .collect()
        ),
        (0..8).map(|v| (v, ((v & 6) ^ 1) | 8)).collect::<Vec<_>>()
    );

    // The chain's pending state is one bit map, not a sequence of maps.
    let q = (&rel).into_query();
    let e = (Expr::input(&q) & 0b0110_u8) ^ 0b0001_u8;
    assert!(matches!(e.map.as_deref(), Some(MapSpec::Bits { .. })));

    // Cloning an expression reuses the same closure input rather than binding
    // a second witness: both branches below constrain one value, so the
    // conjunction is a band.
    assert_eq!(
        held((&rel).filt(|x| x.clone().ge(2_u8) & x.lt(5_u8))),
        [2, 3, 4]
    );
    // A test observes the transformed value but keeps the original range.
    assert_eq!(held((&rel).filt(|x| (x | 1_u8).eq(3_u8))), [2, 3]);
    // Negating that condition complements within the original universe, not
    // the image.
    assert_eq!((&rel).filt(|x| !(x | 1_u8).eq(3_u8)).count(), 6_u32.into());
}

#[test]
#[expect(
    clippy::needless_borrows_for_generic_args,
    reason = "the borrowed recipe is the counterpart of the owned case above it"
)]
fn comparisons_read_their_interpretation_from_the_endpoint() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    assert_eq!(held((&rel).filt(|x| x.eq(3_u8))), [3]);
    assert_eq!(held((&rel).filt(|x| x.ne(3_u8))), [0, 1, 2, 4, 5, 6, 7]);
    assert_eq!(held((&rel).filt(|x| x.lt(2_u8))), [0, 1]);
    assert_eq!(held((&rel).filt(|x| x.le(2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).filt(|x| x.gt(5_u8))), [6, 7]);
    assert_eq!(held((&rel).filt(|x| x.ge(5_u8))), [5, 6, 7]);
    // Explicit operations override the endpoint's interpretation; for these
    // small values the signed and unsigned readings agree.
    assert_eq!(held((&rel).filt(|x| x.ult(2_u8))), [0, 1]);
    assert_eq!(held((&rel).filt(|x| x.ule(2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).filt(|x| x.ugt(5_u8))), [6, 7]);
    assert_eq!(held((&rel).filt(|x| x.uge(5_u8))), [5, 6, 7]);
    assert_eq!(held((&rel).filt(|x| x.slt(2_u8))), [0, 1]);
    assert_eq!(held((&rel).filt(|x| x.sle(2_u8))), [0, 1, 2]);
    assert_eq!(held((&rel).filt(|x| x.sgt(5_u8))), [6, 7]);
    assert_eq!(held((&rel).filt(|x| x.sge(5_u8))), [5, 6, 7]);
    // Membership recipes reach expressions through the same test.
    assert_eq!(held((&rel).filt(|x| x.is_in([1_u8, 4]))), [1, 4]);
    assert_eq!(held((&rel).filt(|x| x.test(pred::is_in(2_u8..4)))), [2, 3]);
    assert_eq!(held((&rel).filt(|x| x.test(&pred::eq(6_u8)))), [6]);
}

#[test]
fn explicit_comparisons_override_the_endpoint_signedness() {
    let mut cx = Context::default();
    i8::bind(&mut cx);
    let mut signed = Rel::<i8, i8>::new_set(&cx);
    signed.insert_rows([-1_i8, 1]);
    // Rows read as raw bits, so -1 is 255.
    let (neg, pos) = (255, 1);
    assert_eq!(held((&signed).filt(|x| x.lt(0_i8))), [neg]);
    assert_eq!(held((&signed).filt(|x| x.le(0_i8))), [neg]);
    assert_eq!(held((&signed).filt(|x| x.gt(0_i8))), [pos]);
    assert_eq!(held((&signed).filt(|x| x.ge(0_i8))), [pos]);
    assert_eq!(held((&signed).filt(|x| x.slt(0_i8))), [neg]);
    assert_eq!(held((&signed).filt(|x| x.sle(0_i8))), [neg]);
    assert_eq!(held((&signed).filt(|x| x.sgt(0_i8))), [pos]);
    assert_eq!(held((&signed).filt(|x| x.sge(0_i8))), [pos]);
    assert_eq!(held((&signed).filt(|x| x.ult(1_i8))), []);
    assert_eq!(held((&signed).filt(|x| x.ule(1_i8))), [pos]);
    assert_eq!(held((&signed).filt(|x| x.ugt(1_i8))), [neg]);
    assert_eq!(held((&signed).filt(|x| x.uge(1_i8))), [pos, neg]);

    let mut cx = Context::default();
    u8::bind(&mut cx);
    let mut unsigned = Rel::<u8, u8>::new_set(&cx);
    unsigned.insert_rows([200_u8, 1]);
    assert_eq!(held((&unsigned).filt(|x| x.lt(100_u8))), [1]);
    assert_eq!(held((&unsigned).filt(|x| x.le(100_u8))), [1]);
    assert_eq!(held((&unsigned).filt(|x| x.gt(100_u8))), [200]);
    assert_eq!(held((&unsigned).filt(|x| x.ge(100_u8))), [200]);
    assert_eq!(held((&unsigned).filt(|x| x.ult(100_u8))), [1]);
    assert_eq!(held((&unsigned).filt(|x| x.ule(100_u8))), [1]);
    assert_eq!(held((&unsigned).filt(|x| x.ugt(100_u8))), [200]);
    assert_eq!(held((&unsigned).filt(|x| x.uge(100_u8))), [200]);
    assert_eq!(held((&unsigned).filt(|x| x.slt(100_u8))), [1, 200]);
    assert_eq!(held((&unsigned).filt(|x| x.sle(100_u8))), [1, 200]);
    assert_eq!(held((&unsigned).filt(|x| x.sgt(100_u8))), []);
    assert_eq!(held((&unsigned).filt(|x| x.sge(100_u8))), []);
}

#[test]
fn conditions_accept_every_restricting_operand() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    let mut odd = Rel::<u8, u8>::new_set(&cx);
    odd.insert_rows([1_u8, 3, 9]);
    let low = |x: Expr<u8, u8>| x.lt(4_u8);
    assert_eq!(held((&rel).filt(|x| low(x) & (1_u8..3))), [1, 2]);
    assert_eq!(held((&rel).filt(|x| low(x) & &(1_u8..3))), [1, 2]);
    assert_eq!(held((&rel).filt(|x| low(x) & (..))), [0, 1, 2, 3]);
    assert_eq!(held((&rel).filt(|x| low(x) & &odd)), [1, 3]);
    assert_eq!(held((&rel).filt(|x| low(x) & (&odd).into_query())), [1, 3]);
    assert_eq!(held((&rel).filt(|x| low(x) & !pred::eq(1_u8))), [0, 2, 3]);
    assert_eq!(held((&rel).filt(|x| low(x) | &odd)), [0, 1, 2, 3]);
    assert_eq!(held((&rel).filt(|x| low(x) | (6_u8..))), [0, 1, 2, 3, 6, 7]);
    assert_eq!(
        held((&rel).filt(|x| low(x) & crate::typed::id())),
        [0, 1, 2, 3]
    );
    assert_eq!(
        held((&rel).filt(|x| low(x) & (pred::eq(1_u8) | pred::eq(2_u8)))),
        [1, 2]
    );
    assert_eq!(
        held((&rel).filt(|x| low(x) & &(pred::ne(1_u8) & pred::ne(2_u8)))),
        [0, 3]
    );
}

#[test]
fn conditions_form_a_boolean_algebra_over_the_closure_input() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    assert_eq!(
        held((&rel).filt(|x| x.clone().lt(2_u8) | x.gt(5_u8))),
        [0, 1, 6, 7]
    );
    assert_eq!(
        held((&rel).filt(|x| x.clone().lt(3_u8) & !x.eq(0_u8))),
        [1, 2]
    );
    // Cloning a condition reuses the same restriction rather than rebuilding
    // it.
    assert_eq!(
        held((&rel).filt(|x| {
            let c = x.lt(1_u8);
            c.clone() | c
        })),
        [0]
    );
    // A condition can also be opened against a query's range outside any
    // closure, which is what `filt` does internally. Borrowed conditions then
    // compose without being rebuilt; `!` binds tighter than `&`, and `&`
    // tighter than `|`.
    let q = (&rel).into_query();
    let low = Expr::input(&q).lt(3_u8);
    let even = (Expr::input(&q) & 1_u8).eq(0_u8);
    assert_eq!(
        held(q.with((&low & &even) | (!&low & !&even))),
        [0, 2, 3, 5, 7]
    );
    // A condition also composes with a predicate recipe or a plain query
    // operand.
    assert_eq!(
        held((&rel).filt(|x| x.clone().lt(2_u8) | pred::eq(7_u8))),
        [0, 1, 7]
    );
    assert_eq!(held((&rel).filt(|x| x.lt(3_u8) & pred::ne(1_u8))), [0, 2]);
}

// --- record fields and reconstruction ----------------------------------

#[test]
fn record_views_project_fields_of_one_input() {
    let mut cx = Context::default();
    Packet::bind(&mut cx);
    let mut packets = Rel::<Packet, Packet>::new_set(&cx);
    packets.insert_rows([
        Packet { src: 1, dst: 2 },
        Packet { src: 1, dst: 5 },
        Packet { src: 4, dst: 2 },
    ]);
    // Every field of the view is a function of the same record input, so a
    // conjunction of field tests keeps their correlation rather than pairing
    // independent witnesses.
    assert_eq!(
        (&packets).filt(|p| p.src.eq(1_u8) & p.dst.gt(3_u8)).count(),
        1_u32.into()
    );
    assert_eq!(
        (&packets)
            .filt(|p| p.src.clone().eq(1_u8) | p.dst.eq(2_u8))
            .count(),
        3_u32.into()
    );
    // The view itself is a functional expression: mapping through it rebuilds
    // the record.
    assert_eq!((&packets).map(|p| p).count(), 3_u32.into());
    assert_eq!((&packets).map(|p| (p.src, p.dst)).count(), 3_u32.into());
    // Whole-record tests reuse the same lowering as scalar ones.
    assert_eq!(
        (&packets).filt(|p| p.eq(Packet { src: 4, dst: 2 })).count(),
        1_u32.into()
    );
    assert_eq!(
        (&packets).filt(|p| p.ne(Packet { src: 4, dst: 2 })).count(),
        2_u32.into()
    );
    assert_eq!(
        (&packets)
            .filt(|p| p.test(pred::eq(Packet { src: 1, dst: 5 })))
            .count(),
        1_u32.into()
    );
}

#[test]
#[should_panic(expected = "fields require a structural record expression")]
fn mapped_record_field_projection_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    (&rel).map(|x| (x | 1_u8).field::<u8>(0));
}

#[test]
fn units_pair_with_the_same_closure_input() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    assert_eq!((&rel).map(|x| x.unit()).count(), 8_u32.into());
    assert_eq!((&rel).map(|x| (x.clone(), x.unit())).count(), 8_u32.into());
}

#[test]
#[should_panic(expected = "units require a structural record expression")]
fn mapped_unit_projection_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    (&rel).map(|x| (x | 1_u8).unit());
}

// --- functional products -----------------------------------------------

#[test]
fn range_products_pair_functions_of_the_same_input() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let rel = bytes(&cx);
    // Anonymous products of expressions are metadata over one input: each
    // component is a function of the same value, so the pairing never
    // multiplies the row count.
    let one = (&rel).map(|x| (x | 1_u8,)).collect();
    assert_eq!(one.count(), 8_u32.into());
    assert_eq!(
        (&rel).map(|x| (x.clone() | 1_u8, x | 2_u8)).count(),
        8_u32.into()
    );
    let b = |x: &Expr<u8, u8>, n: u8| x.clone() | n;
    assert_eq!(
        (&rel).map(|x| (b(&x, 1), b(&x, 2), b(&x, 3))).count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (b(&x, 1), b(&x, 2), b(&x, 3), b(&x, 4)))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (b(&x, 1), b(&x, 2), b(&x, 3), b(&x, 4), b(&x, 5)))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (b(&x, 1), b(&x, 2), b(&x, 3), b(&x, 4), b(&x, 5), b(&x, 6)))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7)
            ))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7),
                b(&x, 8)
            ))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7),
                b(&x, 8),
                b(&x, 9)
            ))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7),
                b(&x, 8),
                b(&x, 9),
                b(&x, 10)
            ))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7),
                b(&x, 8),
                b(&x, 9),
                b(&x, 10),
                b(&x, 11)
            ))
            .count(),
        8_u32.into()
    );
    assert_eq!(
        (&rel)
            .map(|x| (
                b(&x, 1),
                b(&x, 2),
                b(&x, 3),
                b(&x, 4),
                b(&x, 5),
                b(&x, 6),
                b(&x, 7),
                b(&x, 8),
                b(&x, 9),
                b(&x, 10),
                b(&x, 11),
                b(&x, 12)
            ))
            .count(),
        8_u32.into()
    );
    // A product of expressions is itself functional, so it can be tested as a
    // whole.
    assert_eq!(
        (&rel)
            .filt(|x| (x.clone() | 1_u8, x | 2_u8).eq((3_u8, 2_u8)))
            .count(),
        1_u32.into()
    );
}

// --- fixed sets --------------------------------------------------------

#[test]
fn fixed_set_expressions_update_membership_bits() {
    let mut cx = Context::default();
    FlagSet::bind(&mut cx);
    Host::bind(&mut cx);
    let mut sets = Rel::<FlagSet, FlagSet>::new_set(&cx);
    let members = |m: &[u8]| FlagSet::from_members(&cx, m.iter().copied());
    sets.insert_rows([members(&[]), members(&[0]), members(&[1]), members(&[0, 1])]);
    // Membership tests and updates lower to the same bit maps as the integer
    // operators.
    assert_eq!((&sets).filt(|s| s.contains(0_u8)).count(), 2_u32.into());
    assert_eq!(
        (&sets)
            .map(|s| s.insert(2_u8))
            .filt(|s| s.contains(2_u8))
            .count(),
        4_u32.into()
    );
    assert_eq!(
        (&sets)
            .map(|s| s.remove(0_u8))
            .filt(|s| s.contains(0_u8))
            .count(),
        0_u32.into()
    );
    assert_eq!(
        (&sets)
            .map(|s| s.union_with(&members(&[3])))
            .filt(|s| s.contains(3_u8))
            .count(),
        4_u32.into()
    );
    assert_eq!(
        (&sets)
            .map(|s| s.intersect_with(&members(&[1])))
            .filt(|s| s.contains(1_u8))
            .count(),
        2_u32.into()
    );

    // A set-valued declaration tests one membership bit without opening a
    // closure.
    let flags = FuncDecl::<Host, FlagSet>::from_raw(&cx, Host::declaration(&cx).accessors[1]);
    let mut hosts = Rel::<Host, Host>::new_set(&cx);
    hosts.insert_rows([
        Host {
            id: 1,
            flags: members(&[0]),
        },
        Host {
            id: 2,
            flags: members(&[1]),
        },
    ]);
    assert_eq!(
        (&hosts).with(flags.contains(&cx, 0_u8)).count(),
        1_u32.into()
    );
}

// --- interned values ---------------------------------------------------

#[test]
fn interned_expressions_apply_rust_closures_by_enumeration() {
    let mut cx = Context::default();
    Name::bind(&mut cx);
    let names = Dictionary::<Names>::new(&cx);
    let (a, ab, b) = (names.intern("a"), names.intern("ab"), names.intern("b"));
    let mut rel = Rel::<Name, Name>::new_set(&cx);
    // The last handle names an ID the dictionary never assigned: every
    // closure drops it, including a later step of a chained map.
    rel.insert_rows([a, ab, b, names.interned(1000)]);
    // Decoded values drive ordinary Rust predicates; IDs with no entry are
    // dropped.
    assert_eq!(
        (&rel)
            .filt(|n| n.filter_by(|v: &str| v.starts_with('a')))
            .count(),
        2_u32.into()
    );
    // Mapping interns the result in the expression's own context, and chained
    // maps compose.
    let upper = (&rel)
        .map(|n| n.map_by(|v: &str| v.to_uppercase()))
        .collect();
    assert!(upper.contains((a, names.intern("A"))));
    let twice = (&rel)
        .map(|n| {
            n.map_by(|v: &str| format!("{v}{v}"))
                .map_by(|v: &str| v.to_uppercase())
        })
        .collect();
    assert!(twice.contains((b, names.intern("BB"))));
    // The unassigned ID has no image under either step, so the chained map
    // drops it.
    assert_eq!(twice.count(), 3_u32.into());
    // A map followed by a filter chains a Rust map behind an existing one as
    // well.
    assert_eq!(
        (&rel)
            .filt(|n| n
                .map_by(|v: &str| format!("{v}!"))
                .filter_by(|v: &str| v.len() == 2))
            .count(),
        2_u32.into()
    );
}
