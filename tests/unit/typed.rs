//! Checks for the typed core: every combinator forwards to the raw query it
//! wraps, the Rust types only decide which runtime sorts the endpoints must
//! have, and raw conversion checks those sorts.
//!
//! Fixtures are the hand-written `Edge` record, its two `u8` accessors, and
//! the three-pair `chain` relation over bound 8-bit endpoints.

use std::collections::BTreeSet;
use std::ops::ControlFlow;

use super::{Except, FuncDecl, IntoQuery, Query, Record, Rel, Sorted, Variable, id};
use crate::ast::{Sort, TupleDecl};
use crate::backend::BoolOp;
use crate::context::Context;
use crate::row::{self, Domain, IntoRow, Row, RowRef, RowSink};

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.bv_sort(w);
    sort.bind(cx)
}

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

/// The ground rows of a binary relation over 4-bit columns, as pairs.
fn pairs(rel: &crate::rel::Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

/// A two-field record, written by hand the way a derive would generate it.
#[derive(Clone, Copy, Debug)]
struct Edge {
    from: u8,
    to: u8,
}

impl Sorted for Edge {
    type Shape = <(u8, u8) as Sorted>::Shape;
    // Each call re-runs the declaration. The context returns the existing
    // declaration for an equal name and member list, so the sort is stable
    // within one context without a per-type memo.
    fn sort(cx: &Context) -> Sort {
        Self::declaration(cx).sort
    }
}

impl Record for Edge {
    type Args = (u8, u8);
    fn declaration(cx: &Context) -> TupleDecl {
        cx.tuple_decl("Edge", [("from", u8::sort(cx)), ("to", u8::sort(cx))])
    }
}

impl IntoRow for Edge {
    fn record_sort(&self, cx: &Context) -> Option<Sort> {
        Some(Edge::sort(cx))
    }

    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        (self.from, self.to).append(row, out);
    }
}

/// The typed `from` and `to` accessors of `Edge`.
fn accessors(cx: &Context) -> (FuncDecl<Edge, u8>, FuncDecl<Edge, u8>) {
    let decl = Edge::declaration(cx);
    (
        FuncDecl::from_raw(cx, decl.accessors[0]),
        FuncDecl::from_raw(cx, decl.accessors[1]),
    )
}

/// `{(0,1), (1,2), (2,3)}` over bound 8-bit endpoints.
fn chain(cx: &Context) -> Rel<u8, u8> {
    let mut rel = Rel::<u8, u8>::new(cx);
    rel.insert_rows([(0_u8, 1_u8), (1, 2), (2, 3)]);
    rel
}

// --- interpretations and declarations -------------------------------------

#[test]
fn rust_interpretations_select_runtime_sorts() {
    let cx = Context::default();
    assert_eq!(u8::sort(&cx), cx.bv_sort(8));
    assert_eq!(u16::sort(&cx), cx.bv_sort(16));
    assert_eq!(u32::sort(&cx), cx.bv_sort(32));
    assert_eq!(u64::sort(&cx), cx.bv_sort(64));
    assert_eq!(u128::sort(&cx), cx.bv_sort(128));
    assert_eq!(usize::sort(&cx), cx.bv_sort(usize::BITS));
    assert_eq!(i8::sort(&cx), cx.bv_sort(8));
    assert_eq!(i16::sort(&cx), cx.bv_sort(16));
    assert_eq!(i32::sort(&cx), cx.bv_sort(32));
    assert_eq!(i64::sort(&cx), cx.bv_sort(64));
    assert_eq!(i128::sort(&cx), cx.bv_sort(128));
    assert_eq!(isize::sort(&cx), cx.bv_sort(isize::BITS));
    assert_eq!(bool::sort(&cx), cx.bool_sort());
    assert_eq!(<()>::sort(&cx), cx.unit_sort());

    // Each tuple arity constructs its component product directly. Product
    // normalization also accepts equivalent nested and unit-terminated forms.
    assert_eq!(
        <(u8,)>::sort(&cx),
        cx.product_sort([cx.bv_sort(8), cx.unit_sort()])
    );
    assert_eq!(
        <(u16, u8)>::sort(&cx),
        cx.product_sort([cx.bv_sort(16), <(u8,)>::sort(&cx)])
    );
    assert_eq!(
        <(u8, u16, u32)>::sort(&cx),
        cx.product_sort([cx.bv_sort(8), cx.bv_sort(16), cx.bv_sort(32)])
    );
    <(u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);
    <(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8)>::sort(&cx);

    // A record takes its sort from its declaration.
    assert_eq!(Edge::sort(&cx), Edge::declaration(&cx).sort);
}

#[test]
fn tuple_binding_preserves_component_representation() {
    struct Packet;

    impl Sorted for Packet {
        type Shape = <(u8,) as Sorted>::Shape;

        fn sort(cx: &Context) -> Sort {
            cx.tuple_decl("Packet", [("byte", u8::sort(cx))]).sort
        }

        fn bind(cx: &mut Context) -> crate::Row {
            let decl = cx.tuple_decl("Packet", [("byte", u8::sort(cx))]);
            cx.set_representation(decl.accessors[0], cx.backend("bdd"));
            Self::sort(cx).bind(cx)
        }
    }

    for bind in [
        <(Packet, bool)>::bind as fn(&mut Context) -> crate::Row,
        <((Packet, bool), u16)>::bind,
    ] {
        let mut cx = Context::default();
        let bound = bind(&mut cx);
        let decl = cx.tuple_decl("Packet", [("byte", u8::sort(&cx))]);
        let actual = row::scalar_column(&decl.accessors[0].row(&cx));
        let expected = cx.tuple_decl("Expected", [("byte", u8::sort(&cx))]);
        let expected = cx.set_representation(expected.accessors[0], cx.backend("bdd"));
        assert_eq!(actual, expected);
        assert_eq!(bound.layout[bound.start], actual);
    }
}

#[test]
fn declarations_are_plain_names_resolved_in_their_context() {
    let mut cx = Context::default();
    let (from, to) = accessors(&cx);
    let decl = Edge::declaration(&cx);
    // Repeated declaration in one context yields the same nominal sort and
    // accessors.
    assert_eq!(decl.sort, Edge::sort(&cx));
    assert_eq!(from.name(&cx), "from");
    assert_eq!(to.raw, accessors(&cx).1.raw);
    // The constructor's domain is the declared argument product, not the
    // record itself.
    assert_eq!(decl.constructor.resolve(&cx).range, decl.sort);
    assert_eq!(decl.constructor.resolve(&cx).domain, <(u8, u8)>::sort(&cx));
    // A representation selected before binding is honored by the binding.
    let bdd = cx.backend("bdd");
    let column = cx.set_representation(from.raw, bdd);
    assert_eq!(row::scalar_column(&from.bind(&mut cx)), column);
    // Binding a declaration produces this context's row for its range.
    assert_eq!(to.bind(&mut cx).sort, cx.bv_sort(8));

    // A declaration is a query from the record to the member, owned or
    // borrowed, as a right operand or through `query`.
    let mut edges = Rel::<Edge, Edge>::new_set(&cx);
    edges.insert_rows([Edge { from: 0, to: 1 }, Edge { from: 1, to: 2 }]);
    let projected: Rel<Edge, u8> = (&edges).select(from).collect();
    let borrowed: Rel<Edge, u8> = (&edges).select(&from).collect();
    assert_eq!(projected.count(), 2_u32.into());
    assert!(projected == borrowed);
    assert_eq!(from.query(&cx).count(), (256_u32 * 256).into());
    assert!((&edges).select(from.query(&cx)) == projected);
    // `FuncDecl` is a copyable name; the copy denotes the same member.
    let copied = from;
    assert_eq!(copied.clone().name(&cx), "from");
    assert_eq!((&edges).select(copied).count(), projected.count());
}

// --- relations, queries, and variables ------------------------------------

#[test]
fn relations_own_rows_and_native_boolean_algebra() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let mut rel = chain(&cx);
    assert!(!rel.is_empty());
    assert_eq!(rel.count(), 3_u32.into());
    assert!(rel.contains((1_u8, 2_u8)) && !rel.contains((1_u8, 3_u8)));

    // `empty_like` keeps the storage and universe but no rows.
    let mut other = rel.empty_like();
    assert!(other.is_empty());
    other.insert((1_u8, 3_u8));
    assert_eq!(pairs(&other.raw), [(1, 3)].into_iter().collect());

    // Eager native Boolean algebra stays inside the same stored universe.
    assert_eq!(
        pairs(&rel.apply_bool(BoolOp::Or, &other).raw),
        [(0, 1), (1, 2), (1, 3), (2, 3)].into_iter().collect()
    );
    assert_eq!(rel.apply_bool(BoolOp::And, &other).count(), 0_u32.into());
    assert_eq!(rel.not().count(), (256_u32 * 256 - 3).into());
    assert!(rel.not().not() == rel);

    // Mixed insertion accepts a scalar predicate for one endpoint.
    rel.insert((9_u8, cx.set(row::scalar_column(&rel.raw.range), &[7])));
    assert!(rel.contains((9_u8, 7_u8)));

    // Raw conversion round-trips, and both directions agree on the same
    // pairs.
    let raw = rel.clone().raw;
    assert!(Rel::<u8, u8>::from_raw(raw) == rel);
    assert!(Rel::<u8, u8>::new(&cx).is_empty());
    assert!(Rel::<u8, u8>::new_set(&cx).is_empty());
}

#[test]
#[expect(
    clippy::needless_borrow,
    reason = "the explicit borrow keeps every combinator call on the borrowed-operand impl"
)]
#[expect(
    clippy::op_ref,
    reason = "comparing borrowed operands is itself under test"
)]
fn combinators_forward_to_the_raw_query_layer() {
    let mut cx = Context::default();
    let eight = bv(&mut cx, 8);
    let rel = chain(&cx);
    let mut step = Rel::<u8, u8>::new(&cx);
    step.insert_rows([(1_u8, 9_u8), (3, 9)]);

    // Lazy symbolic relation algebra: everything below builds a plan until it
    // is collected.
    assert_eq!(
        pairs(&(&rel).union(&step).collect().raw),
        [(0, 1), (1, 2), (1, 9), (2, 3), (3, 9)]
            .into_iter()
            .collect()
    );
    assert_eq!((&rel).intersection(&rel).count(), 3_u32.into());
    assert_eq!(
        pairs(&(&rel).minus(&step).collect().raw),
        [(0, 1), (2, 3)].into_iter().collect()
    );
    assert_eq!(
        pairs(&(&rel).inv().collect().raw),
        [(1, 0), (2, 1), (3, 2)].into_iter().collect()
    );
    assert_eq!((&rel).complement().count(), (256_u32 * 256 - 3).into());
    assert_eq!(
        pairs(&(&rel).select(&step).collect().raw),
        [(0, 9), (2, 9)].into_iter().collect()
    );
    assert_eq!(
        pairs(&(&rel).with(&step).collect().raw),
        [(0, 1), (2, 3)].into_iter().collect()
    );
    // `or` accepts a range value that either operand has as a domain key.
    let mut two = Rel::<u8, u8>::new(&cx);
    two.insert((2_u8, 0_u8));
    assert_eq!(
        pairs(&(&rel).with((&step).or(&two)).collect().raw),
        [(0, 1), (1, 2), (2, 3)].into_iter().collect()
    );
    // `and` pairs both ranges under the shared domain key.
    assert_eq!((&rel).and(&rel).count(), 3_u32.into());
    assert_eq!((&rel).and(&step).count(), 1_u32.into());

    // `id()` contributes the consuming query's own row type and columns.
    assert_eq!((&rel).select(id()).count(), 3_u32.into());
    assert_eq!((&rel).with(id()).count(), 3_u32.into());
    assert_eq!(id().and(&rel).count(), 3_u32.into());
    // A raw map operand keeps the typed endpoint interpretation.
    assert_eq!(
        pairs(&(&rel).select(crate::map::bvxor(1_u8)).collect().raw),
        [(0, 0), (1, 3), (2, 2)].into_iter().collect()
    );

    // Semantic equality: the symmetric difference of equal relations is
    // empty, and a differing pair survives as an explicit counterexample.
    let same = (&rel).union(&rel);
    assert!(same == rel && rel == same);
    assert!(&same == &(&rel).intersection(&rel));
    assert!(rel != step);
    let diff = (&rel).symmetric_difference(&step).collect();
    assert_eq!(diff.count(), 5_u32.into());
    assert!((&rel).is_subset(&same) && !(&rel).is_subset(&step));

    // Doubly borrowed operands resolve through the borrowed impl.
    let borrow = &&rel;
    assert_eq!((&rel).union(borrow).count(), 3_u32.into());

    // Raw conversion checks both endpoints; identity is the full universe of
    // its sort.
    let raw = (&rel).into_query().raw;
    assert_eq!(Query::<u8, u8, _>::from_raw(raw).count(), 3_u32.into());
    assert_eq!(Query::<u8, u8>::id(&cx).count(), 256_u32.into());
    assert_eq!(
        Query::<u8, u8, _>::from_raw(crate::Query::id(&cx, eight)).count(),
        256_u32.into()
    );
    assert!(!(&rel).is_empty() && (&rel).intersection(&step).is_empty());

    // `drive` visits symbolic branches rather than decoded rows, and may stop
    // early.
    let mut seen = 0;
    let flow = (&rel).into_query().drive(&mut |_, _| {
        seen += 1;
        ControlFlow::Break(())
    });
    assert!(flow.is_break() && seen == 1);
    let mut all = 0;
    let flow = (&rel).into_query().drive(&mut |_, _| {
        all += 1;
        ControlFlow::<()>::Continue(())
    });
    assert!(flow.is_continue() && all >= 1);
}

#[test]
fn variables_carry_endpoint_types_through_semi_naive_rounds() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let mut seed = Rel::<u8, u8>::new(&cx);
    seed.insert((0_u8, 0_u8));
    let edges = chain(&cx);
    let mut reach = Variable::new(seed);
    // The frontier is empty before the first barrier, and the seed is
    // pending.
    assert!(reach.recent().is_empty() && reach.stable().is_empty());
    let mut rounds = 0;
    while reach.changed() {
        rounds += 1;
        let recent = reach.recent();
        assert!(!recent.is_empty());
        reach.insert(&(&edges).inv().select(&recent).collect());
        assert_eq!(
            reach.current().count(),
            reach.stable().count() + recent.count()
        );
    }
    assert_eq!(rounds, 4);
    assert_eq!(
        pairs(&reach.complete().raw),
        [(0, 0), (1, 0), (2, 0), (3, 0)].into_iter().collect()
    );
}

#[test]
fn endpoint_subidentities_keep_their_rust_interpretation() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    Edge::bind(&mut cx);
    let rel = chain(&cx);
    // `values(e) = keys(e^-1)`. Both are subidentities, so each collects as a
    // set over its own endpoint, with that endpoint's Rust interpretation.
    let vals: Rel<u8, u8> = (&rel).values().collect();
    assert_eq!(rows(&vals.raw), [[1], [2], [3]].map(Vec::from).into());
    assert!((&rel).values() == Query::<u8, u8>::id(&cx).with((&rel).inv()));
    let keys: Rel<u8, u8> = (&rel).keys().collect();
    assert_eq!(rows(&keys.raw), [[0], [1], [2]].map(Vec::from).into());
    assert!((&rel).keys() == Query::<u8, u8>::id(&cx).with(&rel));
    // A record-valued range stays a record through the projection.
    let mut edges = Rel::<u8, Edge>::new(&cx);
    edges.insert_rows([(0_u8, Edge { from: 1, to: 2 })]);
    let seen: Rel<Edge, Edge> = (&edges).values().collect();
    assert!(seen.contains(Edge { from: 1, to: 2 }));
    assert!(!seen.contains(Edge { from: 2, to: 1 }));
}

#[test]
fn except_compares_records_around_one_field() {
    let mut cx = Context::default();
    Edge::bind(&mut cx);
    let (from, to) = accessors(&cx);
    let key = Query::<Edge, _>::except(&cx, to);
    let mut a = Rel::<Edge, Edge>::new_set(&cx);
    a.insert(Edge { from: 1, to: 2 });
    let mut b = Rel::<Edge, Edge>::new_set(&cx);
    b.insert(Edge { from: 1, to: 9 });
    let mut c = Rel::<Edge, Edge>::new_set(&cx);
    c.insert(Edge { from: 7, to: 2 });
    // Records that differ only in the dropped field share their key; a
    // differing kept field does not.
    assert!((&a).select(&key).values() == (&b).select(&key).values());
    assert!((&a).select(&key).values() != (&c).select(&key).values());
    // Dropping the other field swaps which comparison holds.
    let other = Query::<Edge, _>::except(&cx, from);
    assert!((&a).select(&other).values() == (&c).select(&other).values());
    assert!((&a).select(&other).values() != (&b).select(&other).values());
}

// --- rejected interpretations ---------------------------------------------

#[test]
#[should_panic(expected = "has no static sort")]
fn projected_record_sort_is_rejected() {
    Except::<Edge>::sort(&Context::default());
}

#[test]
#[should_panic(expected = "typed declaration domain mismatch")]
fn foreign_declaration_domain_interpretation_is_rejected() {
    let cx = Context::default();
    let raw = cx.tuple_decl("Pair", [("x", cx.bv_sort(8))]);
    FuncDecl::<u8, u8>::from_raw(&cx, raw.accessors[0]);
}

#[test]
#[should_panic(expected = "typed declaration range mismatch")]
fn foreign_declaration_range_interpretation_is_rejected() {
    let cx = Context::default();
    FuncDecl::<Edge, u16>::from_raw(&cx, Edge::declaration(&cx).accessors[0]);
}

#[test]
#[should_panic(expected = "typed query domain mismatch")]
fn query_domain_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let eight = bv(&mut cx, 8);
    Query::<u16, u16, _>::from_raw(crate::Query::id(&cx, eight));
}

#[test]
#[should_panic(expected = "typed query range mismatch")]
fn query_range_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let (eight, sixteen) = (bv(&mut cx, 8), bv(&mut cx, 16));
    let raw = crate::Rel::new(&cx, eight, sixteen);
    Query::<u8, u8, _>::from_raw(crate::IntoQuery::into_query(&raw));
}

#[test]
#[should_panic(expected = "typed relation domain mismatch")]
fn relation_domain_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let eight = bv(&mut cx, 8);
    Rel::<u16, u16>::from_raw(crate::Rel::new_set(&cx, eight));
}

#[test]
#[should_panic(expected = "typed relation range mismatch")]
fn relation_range_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let (eight, sixteen) = (bv(&mut cx, 8), bv(&mut cx, 16));
    Rel::<u8, u8>::from_raw(crate::Rel::new(&cx, eight, sixteen));
}

#[test]
#[should_panic(expected = "typed operand belongs to another context")]
fn operand_from_another_context_is_rejected() {
    let mut other = Context::default();
    u8::bind(&mut other);
    let mut foreign = Rel::<u8, u8>::new_set(&other);
    foreign.insert(1_u8);
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let local = chain(&cx);
    (&local).select(&foreign);
}

#[test]
fn direct_operand_resolution_rejects_mismatched_owners() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::Operand;

    fn rejected<T>(f: impl FnOnce() -> T) {
        assert!(catch_unwind(AssertUnwindSafe(f)).is_err());
    }

    let mut cx = Context::default();
    let row = Edge::bind(&mut cx);
    let mut other = Context::default();
    let foreign = Edge::bind(&mut other);
    let q = Query::<Edge, Edge>::id(&cx);
    let (from, _) = accessors(&cx);
    // The supplied context is an independent argument, even when the query
    // and input row agree with each other.
    rejected(|| Operand::<Edge>::resolve(q.clone(), &other, &row));
    rejected(|| Operand::<Edge>::resolve(&q, &other, &row));
    rejected(|| Operand::<Edge>::resolve(q.clone(), &cx, &foreign));
    rejected(|| Operand::<Edge>::resolve(&q, &cx, &foreign));
    rejected(|| Operand::<Edge>::resolve(from, &cx, &foreign));
    rejected(|| Operand::<Edge>::resolve(&from, &cx, &foreign));
    rejected(|| Operand::<Edge>::resolve(from, &other, &foreign));
    rejected(|| Operand::<Edge>::resolve(&from, &other, &foreign));
    assert_eq!(Operand::<Edge>::resolve(q.clone(), &cx, &row).cx.id, cx.id);
    assert_eq!(Operand::<Edge>::resolve(&q, &cx, &row).cx.id, cx.id);
    assert_eq!(
        Operand::<Edge>::resolve(from, &cx, &row).domain.sort,
        row.sort
    );
    assert_eq!(
        Operand::<Edge>::resolve(&from, &cx, &row).domain.sort,
        row.sort
    );
}

// --- records within products ----------------------------------------------

#[test]
fn records_compose_with_argument_tuples_inside_products() {
    let mut cx = Context::default();
    Edge::bind(&mut cx);
    <(u8, u8, bool)>::bind(&mut cx);
    let mut records = Rel::<Edge, Edge>::new_set(&cx);
    records.insert(Edge { from: 1, to: 2 });
    let args = (&records).select(Query::<(u8, u8), (u8, u8)>::id(&cx));
    let out = args.collect();
    assert_eq!(out.count(), 1_u32.into());
    assert!(out.contains((Edge { from: 1, to: 2 }, (1_u8, 2_u8))));
    let returned = args.select(Query::<Edge, Edge>::id(&cx));
    let out = returned.collect();
    assert_eq!(out.count(), 1_u32.into());
    assert!(out.contains((Edge { from: 1, to: 2 }, Edge { from: 1, to: 2 })));

    let mut products = Rel::<(Edge, bool), (Edge, bool)>::new_set(&cx);
    products.insert((Edge { from: 3, to: 4 }, true));
    let flat = (&products).select(Query::<(u8, u8, bool), (u8, u8, bool)>::id(&cx));
    let back = flat.select(Query::<(Edge, bool), (Edge, bool)>::id(&cx));
    assert!(back.collect().contains((
        (Edge { from: 3, to: 4 }, true),
        (Edge { from: 3, to: 4 }, true)
    )));
    assert_eq!(back.count(), 1_u32.into());
}

#[test]
#[should_panic(expected = "composition requires compatible whole endpoint sorts")]
fn differently_named_record_with_one_scalar_shape_is_rejected() {
    struct Other;
    impl Sorted for Other {
        type Shape = <Edge as Sorted>::Shape;
        fn sort(cx: &Context) -> Sort {
            cx.tuple_decl("Other", [("from", u8::sort(cx)), ("to", u8::sort(cx))])
                .sort
        }
    }
    let mut cx = Context::default();
    Edge::bind(&mut cx);
    Other::bind(&mut cx);
    Query::<Edge, Edge>::id(&cx).select(Query::<Other, Other>::id(&cx));
}

#[test]
fn nested_record_members_keep_their_nominal_boundaries() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct Envelope;
    impl Sorted for Envelope {
        type Shape = <(Edge, bool) as Sorted>::Shape;
        fn sort(cx: &Context) -> Sort {
            cx.tuple_decl(
                "Envelope",
                [("edge", Edge::sort(cx)), ("flag", bool::sort(cx))],
            )
            .sort
        }
    }
    let mut cx = Context::default();
    Envelope::bind(&mut cx);
    <(u8, u8, bool)>::bind(&mut cx);
    let q = Query::<Envelope, Envelope>::id(&cx);
    // One declared argument conversion exposes Edge as a whole member.
    assert_eq!(
        (&q).select(Query::<(Edge, bool), (Edge, bool)>::id(&cx))
            .count(),
        (256_u32 * 256 * 2).into()
    );
    // Flattened Rust scalar shapes alone do not authorize crossing that
    // remaining nominal boundary in the same conversion.
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            (&q).select(Query::<(u8, u8, bool), (u8, u8, bool)>::id(&cx))
        }))
        .is_err()
    );
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Edge")]
fn conflicting_record_declaration_is_rejected() {
    struct Conflict;
    impl Sorted for Conflict {
        type Shape = <(u16, u8) as Sorted>::Shape;
        fn sort(cx: &Context) -> Sort {
            cx.tuple_decl("Edge", [("from", u16::sort(cx)), ("to", u8::sort(cx))])
                .sort
        }
    }
    let cx = Context::default();
    assert_eq!(Edge::sort(&cx), Edge::sort(&cx));
    Conflict::sort(&cx);
}

#[test]
fn except_comparison_checks_the_runtime_shape() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct Mixed;
    impl Sorted for Mixed {
        type Shape = <(u8, u16) as Sorted>::Shape;
        fn sort(cx: &Context) -> Sort {
            Self::declaration(cx).sort
        }
    }
    impl Record for Mixed {
        type Args = (u8, u16);
        fn declaration(cx: &Context) -> TupleDecl {
            cx.tuple_decl("Mixed", [("a", u8::sort(cx)), ("b", u16::sort(cx))])
        }
    }
    let mut cx = Context::default();
    Mixed::bind(&mut cx);
    let decl = Mixed::declaration(&cx);
    let a = FuncDecl::<Mixed, u8>::from_raw(&cx, decl.accessors[0]);
    let b = FuncDecl::<Mixed, u16>::from_raw(&cx, decl.accessors[1]);
    let mut rows = Rel::<Mixed, Mixed>::new_set(&cx);
    rows.insert_rows([(1_u8, 2_u16), (3, 4)]);
    let left = (&rows).select(Query::<Mixed, _>::except(&cx, a));
    let right = (&rows).select(Query::<Mixed, _>::except(&cx, b));
    assert!((&left).difference(&left).is_empty());
    assert!(catch_unwind(AssertUnwindSafe(|| (&left).difference(&right))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| left == right)).is_err());
}
