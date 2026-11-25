//! Checks for combinator wiring, endpoint checks, terminals, and semantic
//! equality.
//!
//! The `relation` fixture builds a binary relation over one row from `u8`
//! pairs, and `values` reads the values of a one-column set. Local fixtures
//! bind bit-vector rows, build sets, and list ground pairs and rows.

#[path = "support/query.rs"]
mod fixture;

use fixture::{bv, tiny};

use std::collections::BTreeSet;
use std::ops::ControlFlow;

use sra::context::Context;
use sra::query::{self, IntoQuery, Query, id};
use sra::rel::Rel;
use sra::row::Row;

/// A binary relation over `four` holding `ps`.
fn relation(cx: &Context, four: &Row, ps: impl IntoIterator<Item = (u8, u8)>) -> Rel {
    let mut r = Rel::new(cx, four.clone(), four.clone());
    r.insert_rows(ps);
    r
}

/// The values of a set over one column.
fn values(rel: &Rel) -> BTreeSet<u64> {
    rows(rel)
        .into_iter()
        .map(|r| {
            assert_eq!(r.len(), 1);
            r[0]
        })
        .collect()
}

/// A 4-bit set holding `values`, over the already bound row `four`.
fn set(cx: &Context, four: &Row, values: impl IntoIterator<Item = u8>) -> Rel {
    let mut s = Rel::new_set(cx, four.clone());
    s.insert_rows(values);
    s
}

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &Rel) -> BTreeSet<Vec<u64>> {
    let cx = rel.root().index().context();
    let widths: Vec<u32> = (0..rel.root().index().arity())
        .map(|i| {
            cx.resolve(rel.root().index().algebra(i))
                .sort()
                .bv_width()
                .unwrap()
        })
        .collect();
    let mut out = BTreeSet::new();
    let mut row = vec![0; widths.len()];
    loop {
        if rel.root().contains(&row) {
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
fn pairs(rel: &Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

// --- union and operand checks ---------------------------------------------

#[test]
fn union_merges_sets_and_relations() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let (s, t) = (set(&cx, &four, [1, 2, 3]), set(&cx, &four, [2, 3, 4]));
    assert_eq!(values(&s.union(&t).collect()), [1, 2, 3, 4].into());
    assert_eq!(values(&s.union(&s).collect()), [1, 2, 3].into());
    let (r, r2) = (
        relation(&cx, &four, [(1, 2), (2, 3)]),
        relation(&cx, &four, [(2, 3), (3, 4)]),
    );
    assert_eq!(
        pairs(&r.union(&r2).collect()),
        [(1, 2), (2, 3), (3, 4)].into()
    );
    assert_eq!(r.union(&r2).count().to_string(), "3");
    // A break in the left branch stops before the right branch runs.
    let mut seen = 0;
    let stopped = r.union(&r2).drive(&mut |_, _| {
        seen += 1;
        ControlFlow::Break(())
    });
    assert!(stopped.is_break() && seen == 1);
}

#[test]
fn union_branches_restore_equalities_between_shared_cells() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    // Left: x -> (x, x) over a set, three exposed positions on one cell.
    // Right: d -> (r, r) over a relation. The union exposes three cells; the
    // left branch equates them on entry.
    let s = set(&cx, &four, [1, 2, 3]);
    let r = relation(&cx, &four, [(1, 2), (2, 3), (3, 1)]);
    let q = Query::id(&cx, four);
    let diag = q.clone().and(q);
    let u = id().and(&s).union((&r).select(&diag));
    assert_eq!(
        rows(&u.collect()),
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 2, 2],
            [2, 3, 3],
            [3, 1, 1]
        ]
        .map(Vec::from)
        .into()
    );
    assert_eq!(u.count().to_string(), "6");
    // Intersecting with the diagonal merges the union's cells before it
    // runs, so the left branch finds its equality already in place.
    let both = id()
        .and(&s)
        .union((&r).select(&diag))
        .intersection(id().and(&s));
    assert_eq!(
        rows(&both.collect()),
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]].map(Vec::from).into()
    );
    assert_eq!(both.count().to_string(), "3");
}

#[test]
#[should_panic(expected = "union requires compatible query endpoints")]
fn union_with_mismatched_domains_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    set(&cx, &four, [1]).union(&Rel::new_set(&cx, eight));
}

#[test]
#[should_panic(expected = "union requires compatible query endpoints")]
fn union_with_mismatched_ranges_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).union(&Rel::new(&cx, four.clone(), eight));
}

#[test]
#[should_panic(expected = "query operands belong to different contexts")]
fn mixing_operands_from_two_contexts_is_rejected() {
    let (mut cx, mut other) = (tiny(), tiny());
    let (four, four2) = (bv(&mut cx, 4), bv(&mut other, 4));
    set(&cx, &four, [1]).union(&set(&other, &four2, [1]));
}

#[test]
#[should_panic(expected = "query variable uses incompatible algebras")]
fn equating_stored_columns_with_different_storage_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let t = cx
        .sorts()
        .record("T", &[("x", four.sort()), ("y", four.sort())]);
    cx.bind_field(cx.sorts().project(t, 0), "bdd");
    Row::bind(&mut cx, t);
    Rel::new_set(&cx, cx.sorts().project(t, 0)).union(&Rel::new_set(&cx, cx.sorts().project(t, 1)));
}

// --- intersection and complement ------------------------------------------

#[test]
fn intersection_keeps_shared_pairs() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let (s, t) = (set(&cx, &four, [1, 2, 3]), set(&cx, &four, [2, 3, 4]));
    assert_eq!(values(&s.intersection(&t).collect()), [2, 3].into());
    let (r, r2) = (
        relation(&cx, &four, [(1, 2), (2, 3)]),
        relation(&cx, &four, [(2, 3), (3, 4)]),
    );
    assert_eq!(pairs(&r.intersection(&r2).collect()), [(2, 3)].into());
    assert!(r.intersection(&relation(&cx, &four, [(3, 4)])).is_empty());
}

#[test]
#[should_panic(expected = "intersection requires compatible query endpoints")]
fn intersection_with_mismatched_domains_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    set(&cx, &four, [1]).intersection(&Rel::new_set(&cx, eight));
}

#[test]
#[should_panic(expected = "intersection requires compatible query endpoints")]
fn intersection_with_mismatched_ranges_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).intersection(&Rel::new(&cx, four.clone(), eight));
}

#[test]
fn complement_uses_the_exact_universe() {
    let mut cx = tiny();
    let (four, one) = (bv(&mut cx, 4), bv(&mut cx, 1));
    let s = set(&cx, &four, [1, 2, 3]);
    assert_eq!(
        values(&s.complement().collect()),
        (0..16).filter(|v| !(1..=3).contains(v)).collect()
    );
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    assert_eq!(r.complement().count().to_string(), "254");
    assert!(!r.complement().collect().contains((1_u8, 2_u8)));
    assert!(r.complement().collect().contains((2_u8, 2_u8)));
    // A pair set built by duplicating one projection is still complemented
    // over the product.
    let mut pair = Rel::new(&cx, one.clone(), one.clone());
    pair.insert((0_u8, 0_u8));
    assert_eq!(
        pairs(&pair.complement().collect()),
        [(0, 1), (1, 0), (1, 1)].into()
    );
    let diagonal = Query::id(&cx, one.clone())
        .and(Query::id(&cx, one))
        .collect();
    assert_eq!(
        rows(&diagonal),
        [[0, 0, 0], [1, 1, 1]].map(Vec::from).into()
    );
    // A general relation is complemented over its domain-range product, even
    // when it was built from one repeated cell; a set is complemented within
    // its row universe.
    assert_eq!(diagonal.complement().count().to_string(), "6");
    assert_eq!(set(&cx, &four, []).complement().count().to_string(), "16");
}

// --- inverse and composition ----------------------------------------------

#[test]
fn inverse_swaps_endpoints_and_composition_chains_them() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    assert_eq!(pairs(&r.inv().collect()), [(2, 1), (3, 2)].into());
    assert_eq!(pairs(&r.inv().inv().collect()), [(1, 2), (2, 3)].into());
    let r2 = relation(&cx, &four, [(2, 5), (3, 6), (3, 7)]);
    assert_eq!(
        pairs(&r.select(&r2).collect()),
        [(1, 5), (2, 6), (2, 7)].into()
    );
    assert_eq!(pairs(&r.select(id()).collect()), [(1, 2), (2, 3)].into());
    // Borrowed queries and doubly borrowed operands share the same nodes.
    let q = (&r).into_query();
    let borrowed = &q;
    assert_eq!(
        pairs(&(&borrowed).select(&&(&q).inv()).collect()),
        [(1, 1), (2, 2)].into()
    );
    assert_eq!(pairs(&(&&q).union(&&q).collect()), [(1, 2), (2, 3)].into());
}

#[test]
#[should_panic(expected = "composition requires compatible whole endpoint sorts")]
fn composition_through_a_mismatched_endpoint_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).select(&Rel::new(&cx, eight, four.clone()));
}

// --- declarations and records ---------------------------------------------

#[test]
fn declarations_project_and_construct_records() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let t = cx
        .sorts()
        .record("T", &[("x", four.sort()), ("y", four.sort())]);
    let trow = Row::bind(&mut cx, t);
    let args_sort = cx.sorts().product([four.sort(), four.sort()]);
    let args_row = Row::bind(&mut cx, args_sort);
    let (x, y) = (cx.sorts().project(t, 0), cx.sorts().project(t, 1));
    let mut r = Rel::new_set(&cx, trow.clone());
    r.insert_rows([(1_u8, 2_u8), (3, 4)]);
    assert_eq!(
        rows(&r.select(x).collect()),
        [[1, 2, 1], [3, 4, 3]].map(Vec::from).into()
    );
    assert_eq!(
        rows(&r.select(y).collect()),
        [[1, 2, 2], [3, 4, 4]].map(Vec::from).into()
    );
    let swapped = r.select(Query::decl(&cx, y).and(x)).collect();
    assert_eq!(swapped.count(), 2);
    assert!(swapped.root().contains(&[1, 2, 2, 1]));
    assert!(swapped.root().contains(&[3, 4, 4, 3]));
    // A constructor keeps every column under the record sort; applying it
    // to an argument relation yields the records, and the constructor alone
    // is the identity between them.
    let mut args = Rel::new_set(&cx, args_row);
    args.insert_rows([(5_u8, 6_u8)]);
    let made = cx.sorts().constructor(t).apply(&args).collect();
    assert!(made.contains(((5_u8, 6_u8), (5_u8, 6_u8))));
    assert_eq!(made.range().sort(), t);
    let c = Query::decl(&cx, cx.sorts().constructor(t));
    assert_eq!((c.domain.sort(), c.range.sort()), (args.domain().sort(), t));
    assert_eq!(c.count().to_string(), "256");
    assert_eq!(
        Query::decl(&cx, cx.sorts().project(t, 1))
            .count()
            .to_string(),
        "256"
    );
    // Coercion swaps a record for its argument product without touching
    // cells.
    let coerced = Query::id(&cx, trow).coerce(args.domain().sort(), args.domain().sort());
    assert!(coerced.collect().contains((1_u8, 2_u8)));
    assert!(coerced == args.complement().union(&args));
}

#[test]
fn a_constructor_reads_the_record_fields_from_their_bound_algebras() {
    let mut cx = tiny();
    let two = cx.sorts().bv(2);
    let t = cx.sorts().record("T", &[("left", two), ("right", two)]);
    cx.bind_field(cx.sorts().project(t, 0), "bdd");
    let record = Row::bind(&mut cx, t);
    let query = Query::decl(&cx, cx.sorts().constructor(t));
    assert_eq!(query.domain.algebras(), record.algebras());
    assert_eq!(query.range.algebras(), record.algebras());
    let rel = query.collect();
    assert_eq!(rel.count(), 16);
    assert!(rel.contains(((1_u8, 2_u8), (1_u8, 2_u8))));
    assert!(!rel.contains(((1_u8, 2_u8), (2_u8, 1_u8))));
}

#[test]
fn except_projects_a_record_onto_its_other_members() {
    let mut cx = tiny();
    let two = bv(&mut cx, 2);
    let t = cx.sorts().record(
        "T",
        &[("x", two.sort()), ("y", two.sort()), ("z", two.sort())],
    );
    let trow = Row::bind(&mut cx, t);
    let p = cx
        .sorts()
        .record("P", &[("a", two.sort()), ("b", two.sort())]);
    let prow = Row::bind(&mut cx, p);
    let mut r = Rel::new_set(&cx, trow.clone());
    r.insert_rows([(0_u8, 1_u8, 2_u8), (1, 2, 3)]);
    let dropped = Query::except(&cx, trow, 1);
    assert_eq!(dropped.domain.sort(), t);
    assert_eq!(
        dropped.range.sort(),
        cx.sorts().product([two.sort(), two.sort()])
    );
    // The projection is exactly the pairing of the kept members' accessors.
    let kept = Query::decl(&cx, cx.sorts().project(t, 0)).and(cx.sorts().project(t, 2));
    assert_eq!(
        rows(&r.select(dropped.clone()).collect()),
        rows(&r.select(kept).collect())
    );
    assert_eq!(
        rows(&r.select(dropped).collect()),
        [[0, 1, 2, 0, 2], [1, 2, 3, 1, 3]].map(Vec::from).into()
    );
    // Dropping one member of a two-member record leaves a one-member
    // product.
    assert_eq!(
        Query::except(&cx, prow, 0).range.sort(),
        cx.sorts().product([two.sort()])
    );
}

#[test]
#[should_panic(expected = "member index exceeds the record")]
fn except_with_an_out_of_range_member_is_rejected() {
    let mut cx = tiny();
    let two = bv(&mut cx, 2);
    let t = cx
        .sorts()
        .record("T", &[("x", two.sort()), ("y", two.sort())]);
    let trow = Row::bind(&mut cx, t);
    Query::except(&cx, trow, 2);
}

// --- keys and values ------------------------------------------------------

#[test]
fn keys_and_values_are_the_witnessed_endpoint_subidentities() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    // Both are subidentities, so collecting one stores a set over that
    // endpoint.
    let r = relation(&cx, &four, [(1, 5), (2, 6), (2, 7)]);
    assert_eq!(values(&r.keys().collect()), [1, 2].into());
    assert_eq!(values(&r.values().collect()), [5, 6, 7].into());
    // `values` is `keys` of the converse, and both stay within their own
    // endpoint universe.
    assert!(r.values() == (&r).inv().keys());
    assert!(r.keys() == Query::id(&cx, four.clone()).with(&r));
    // Without a witness there is no diagonal at all.
    assert!(Rel::new(&cx, four.clone(), four.clone()).keys().is_empty());
    assert!(
        Rel::new(&cx, four.clone(), four.clone())
            .values()
            .is_empty()
    );
    // A set's two subidentities agree: its domain and range are the same
    // column, so the diagonal collapses back to the set itself.
    let s = set(&cx, &four, [1, 2, 3]);
    assert_eq!(values(&s.keys().collect()), [1, 2, 3].into());
    assert_eq!(values(&s.values().collect()), [1, 2, 3].into());
}

// --- subtraction ----------------------------------------------------------

#[test]
fn subtraction_keeps_equations_live_while_narrowing_keys() {
    for backend in ["sparse", "bdd"] {
        let mut cx = tiny();
        let sort = cx.sorts().bv(2);
        let column = cx.bind(backend, sort.scalar());
        fixture::indexes(&cx, column);
        let row = Row::new(&cx, sort, [column].into(), 0);
        for mask in 0..16_u8 {
            let excluded = set(&cx, &row, (0..4_u8).filter(|&x| mask & (1 << x) != 0));
            let query = Query::id(&cx, row.clone())
                .select(sra::map::copy())
                .minus(&excluded);
            let expected: BTreeSet<_> = (0..4_u64)
                .filter(|&x| mask & (1 << x) == 0)
                .map(|x| (x, x))
                .collect();
            assert_eq!(pairs(&query.collect()), expected);
            assert_eq!(query.count().to_string(), expected.len().to_string());
            assert_eq!(query.is_empty(), expected.is_empty());
            let mut visited = BTreeSet::new();
            let result = query.drive(&mut |d, r| {
                let (p, q) = (d.predicate(0), r.predicate(0));
                for x in 0..4_u64 {
                    for y in 0..4_u64 {
                        if p.contains(&[x]) && q.contains(&[y]) {
                            visited.insert((x, y));
                        }
                    }
                }
                ControlFlow::<()>::Continue(())
            });
            assert!(result.is_continue());
            assert_eq!(visited, expected);
        }
    }
}

#[test]
fn minus_subtracts_keys_and_pairs() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    let r = relation(&cx, &four, [(1, 2), (2, 3), (3, 4)]);
    assert_eq!(
        pairs(&r.minus(&set(&cx, &four, [2])).collect()),
        [(1, 2), (3, 4)].into()
    );
    // The right range is hidden: a relation with another range sort still
    // keys the domain.
    let mut other = Rel::new(&cx, four.clone(), eight);
    other.insert((3_u8, 200_u8));
    assert_eq!(pairs(&r.minus(&other).collect()), [(1, 2), (2, 3)].into());
    // Plain pair difference keys the complete pair and keeps the endpoints.
    let r2 = relation(&cx, &four, [(2, 3), (3, 9)]);
    assert_eq!(pairs(&r.difference(&r2).collect()), [(1, 2), (3, 4)].into());
    assert!(r.difference(&r).is_empty());
    assert!(r.minus(&set(&cx, &four, [1, 2, 3])).is_empty());
}

#[test]
#[should_panic(expected = "minus requires compatible whole domain sorts")]
fn minus_with_mismatched_domains_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).minus(&Rel::new_set(&cx, eight));
}

// --- pairing and restriction ----------------------------------------------

#[test]
fn pairing_and_restriction_share_the_domain() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    let r2 = relation(&cx, &four, [(1, 5), (1, 6), (3, 7)]);
    assert_eq!(
        rows(&r.and(&r2).collect()),
        [[1, 2, 5], [1, 2, 6]].map(Vec::from).into()
    );
    assert_eq!(
        pairs(&r.with(&set(&cx, &four, [3])).collect()),
        [(2, 3)].into()
    );
    assert_eq!(pairs(&r.with(&r2.inv()).collect()), BTreeSet::new());
    assert_eq!(
        pairs(&r.with(&relation(&cx, &four, [(3, 0)])).collect()),
        [(2, 3)].into()
    );
    // `id().and` keeps identity-first order, and an identity paired with
    // itself is the diagonal.
    assert_eq!(
        rows(&id().and(&r).collect()),
        [[1, 1, 2], [2, 2, 3]].map(Vec::from).into()
    );
}

#[test]
#[should_panic(expected = "pairing requires compatible whole domain sorts")]
fn pairing_with_mismatched_domains_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).and(&Rel::new_set(&cx, eight));
}

#[test]
#[should_panic(expected = "restriction requires compatible whole endpoint sorts")]
fn restriction_by_a_mismatched_range_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    relation(&cx, &four, [(1, 2)]).with(&Rel::new_set(&cx, eight));
}

// --- semantic equality ----------------------------------------------------

#[test]
fn semantic_equality_compares_complete_pairs() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let pair_sort = cx.sorts().product([four.sort(), four.sort()]);
    let pair = Row::bind(&mut cx, pair_sort);
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    let same = relation(&cx, &four, [(2, 3), (1, 2)]);
    let other = relation(&cx, &four, [(1, 2), (2, 4)]);
    assert!(r == same && r != other);
    assert!((&r).into_query() == same && (&r).into_query() != other);
    assert!(r == (&same).into_query() && r != (&other).into_query());
    assert!(
        (&r).into_query() == (&same).into_query() && (&r).into_query() != (&other).into_query()
    );
    assert_eq!(
        pairs(&r.symmetric_difference(&other).collect()),
        [(2, 3), (2, 4)].into()
    );
    assert!(r.symmetric_difference(&same).is_empty());
    // Equality sees the domain/range boundary, unlike pair-flattened keys.
    let mut flat = Rel::new(&cx, four.clone(), pair.clone());
    flat.insert((1_u8, 2_u8, 3_u8));
    let mut wide = Rel::new(&cx, pair, four);
    wide.insert((1_u8, 2_u8, 3_u8));
    assert!(flat.symmetric_difference(&flat).is_empty());
    assert!(wide.symmetric_difference(&wide).is_empty());
}

#[test]
#[should_panic(expected = "difference requires compatible query endpoints")]
fn symmetric_difference_with_mismatched_domains_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    let _ = set(&cx, &four, [1]) == Rel::new_set(&cx, eight);
}

#[test]
#[should_panic(expected = "difference requires compatible query endpoints")]
fn symmetric_difference_with_mismatched_ranges_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    let _ = relation(&cx, &four, [(1, 2)]) == Rel::new(&cx, four.clone(), eight);
}

// --- terminals ------------------------------------------------------------

#[test]
fn terminals_count_visit_and_collect() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let r = relation(&cx, &four, [(1, 2), (2, 3), (2, 4)]);
    let q = (&r).into_query();
    assert_eq!(q.count().to_string(), "3");
    // Counting the whole universe finds one branch covering every box.
    assert_eq!(Query::id(&cx, four.clone()).count().to_string(), "16");
    // Each delivered box is a Cartesian product of its endpoint cells;
    // together they cover the pairs exactly once.
    let mut seen = BTreeSet::new();
    let mut total = 0_u32;
    let done = q.drive(&mut |d, r| {
        assert_eq!((d.cells.len(), r.cells.len()), (1, 1));
        let (d, r) = (d.predicate(0), r.predicate(0));
        total += d.count().to_string().parse::<u32>().unwrap()
            * r.count().to_string().parse::<u32>().unwrap();
        for a in 1..5 {
            for b in 1..5 {
                if d.contains(&[a]) && r.contains(&[b]) {
                    seen.insert((a, b));
                }
            }
        }
        ControlFlow::<()>::Continue(())
    });
    assert!(done.is_continue());
    assert_eq!(total, 3);
    assert_eq!(seen, [(1, 2), (2, 3), (2, 4)].into());
    assert!(
        !q.is_empty()
            && Rel::new(&cx, four.clone(), four.clone())
                .into_query()
                .is_empty()
    );
    // Symbolic sources are collected exactly, and outputs report their
    // universe shape.
    let mut sym = Rel::new_set(&cx, four.clone());
    sym.insert(cx.set(four.scalar_algebra(), &[4, 6]));
    assert_eq!(values(&sym.into_query().collect()), [4, 6].into());
    let symbolic = sym.into_query();
    let (positions, ends) = symbolic.output();
    assert_eq!(
        (positions.to_vec(), ends.collect::<Vec<_>>()),
        (vec![0], vec![0, 0])
    );
    let (positions, ends) = q.output();
    assert_eq!(
        (positions.to_vec(), ends.collect::<Vec<_>>()),
        (vec![0, 1], vec![0, 1])
    );
}

#[test]
fn identity_carries_its_context() {
    let mut cx = tiny();
    let two = bv(&mut cx, 2);
    let t = cx.sorts().record("T", &[("x", two.sort())]);
    let trow = Row::bind(&mut cx, t);
    let q = Query::id(&cx, trow);
    assert_eq!(q.count().to_string(), "4");
    assert!(q.cx == cx);
}

// --- universes ------------------------------------------------------------

#[test]
fn zero_field_queries_preserve_declared_universes_and_argument_shapes() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use sra::rel::Universe;

    let mut cx = tiny();
    let empty = cx.sorts().record("Empty", &[]);
    let record = Row::bind(&mut cx, empty);
    let unit = Row::of(&cx, cx.sorts().unit());
    for populated in [false, true] {
        let mut s = Rel::new_set(&cx, record.clone());
        let mut r = Rel::new(&cx, record.clone(), record.clone());
        let mut u = Rel::new(&cx, unit.clone(), unit.clone());
        if populated {
            s.insert(());
            r.insert(((), ()));
            u.insert(());
        }
        for source in [&s, &r, &u] {
            let q = source.into_query();
            for out in [q.collect(), q.clone().collect(), (&q).inv().collect()] {
                assert_eq!(out.universe(), source.universe());
                assert_eq!(out.domain().sort(), source.domain().sort());
                assert_eq!(out.range().sort(), source.range().sort());
                assert_eq!(out.count(), source.count());
                assert_eq!(out.is_empty(), !populated);
                if out.domain().sort() == empty && out.universe() == Universe::Binary {
                    assert_eq!(out.contains(((), ())), populated);
                    assert!(catch_unwind(AssertUnwindSafe(|| out.contains(()))).is_err());
                } else {
                    assert_eq!(out.contains(()), populated);
                }
            }
            let out = q.complement().collect();
            assert_eq!(out.universe(), source.universe());
            assert_eq!(out.is_empty(), populated);
            assert_eq!(out.count(), source.not().count());
        }
        assert_eq!(r.union(&s).collect().universe(), Universe::Binary);
        assert_eq!(s.intersection(&r).collect().universe(), Universe::Binary);
        assert_eq!(r.select(&s).collect().universe(), Universe::Binary);
        assert_eq!(s.select(&r).collect().universe(), Universe::Binary);
        assert_eq!(s.with(&r).collect().universe(), Universe::Set);
        assert_eq!(r.keys().collect().universe(), Universe::Set);
        assert_eq!(r.values().collect().universe(), Universe::Set);
    }
}

#[test]
fn binary_universes_survive_aliases_and_mixed_set_operations() {
    use sra::rel::Universe;

    for backend in ["sparse", "bdd"] {
        let mut cx = tiny();
        let t = cx.sorts().record("T", &[("value", cx.sorts().bv(2))]);
        cx.bind_field(cx.sorts().project(t, 0), backend);
        let field = cx.sorts().project(t, 0);
        let two = Row::bind(&mut cx, field);
        fixture::indexes(&cx, two.scalar_algebra());
        let r = Rel::new(&cx, two.clone(), two.clone()).not();
        let s = set(&cx, &two, [0, 2]);
        let diag = r.intersection(&s);
        let expected: BTreeSet<_> = [(0, 0), (2, 2)].into();
        assert_eq!(diag.count().to_string(), "2");
        for out in [
            diag.collect(),
            diag.clone().collect(),
            (&diag).inv().collect(),
            s.intersection(&r).collect(),
            (&diag).union(&s).collect(),
            s.union(&diag).collect(),
            (&diag).select(id()).collect(),
            Query::id(&cx, two).select(&diag).collect(),
            (&diag).with(&s).collect(),
            (&diag).difference(&r.empty_like()).collect(),
            (&diag).minus(&s.empty_like()).collect(),
        ] {
            assert_eq!(out.universe(), Universe::Binary);
            assert_eq!(out.root().index().arity(), 2);
            assert_eq!(pairs(&out), expected);
        }
        let outside: BTreeSet<_> = (0..4)
            .flat_map(|x| (0..4).map(move |y| (x, y)))
            .filter(|p| !expected.contains(p))
            .collect();
        assert_eq!(pairs(&(&diag).complement().collect()), outside);
        assert_eq!((&diag).complement().count().to_string(), "14");
        for out in [
            (&diag).keys().collect(),
            (&diag).values().collect(),
            s.with(&r).collect(),
            s.select(&s).collect(),
            s.intersection(&s).collect(),
            s.union(&s).collect(),
            s.difference(&r.empty_like()).collect(),
            s.minus(&r.empty_like()).collect(),
        ] {
            assert_eq!(out.universe(), Universe::Set);
            assert_eq!(values(&out), [0, 2].into());
        }
        let paired = s.and(&s).collect();
        assert_eq!(paired.universe(), Universe::Binary);
        assert_eq!(rows(&paired), [vec![0, 0, 0], vec![2, 2, 2]].into());
        let copied = s.select(sra::map::copy());
        assert_eq!(copied.universe, Universe::Binary);
        assert_eq!(pairs(&copied.collect()), expected);
        assert_eq!(pairs(&copied.complement().collect()), outside);
    }
}

#[test]
fn nominal_coercion_and_structural_maps_keep_declared_products() {
    use sra::rel::Universe;

    let mut cx = tiny();
    let two = bv(&mut cx, 2);
    let t = cx.sorts().record("T", &[("x", two.sort())]);
    let record = Row::bind(&mut cx, t);
    let identity = Query::id(&cx, record.clone());
    let coerced = identity.clone().coerce(two.sort(), t);
    assert_eq!(coerced.universe, Universe::Binary);
    assert_eq!(coerced.collect().root().index().arity(), 2);
    assert_eq!(coerced.clone().coerce(t, t).universe, Universe::Binary);
    assert_eq!(coerced.complement().count().to_string(), "12");
    assert_eq!(
        identity.coerce(two.sort(), two.sort()).universe,
        Universe::Set
    );
    let constructor = Query::decl(&cx, cx.sorts().constructor(t));
    assert_eq!(constructor.universe, Universe::Binary);
    assert_eq!(
        constructor.coerce(two.sort(), two.sort()).universe,
        Universe::Binary
    );
    let projection = Query::decl(&cx, cx.sorts().project(t, 0));
    assert_eq!(projection.universe, Universe::Binary);
    assert_eq!(projection.clone().keys().universe, Universe::Set);
    assert_eq!(projection.values().universe, Universe::Set);
    assert_eq!(Query::id(&cx, record).universe, Universe::Set);
}

// --- projection positions -------------------------------------------------

#[test]
fn projection_positions_preserve_repeated_coordinates() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    let range = Row::product(&cx, &[&bit, &bit]);
    let query = query::project(&cx, &bit, [0, 0], range);
    assert_eq!(query.positions(), [0, 0]);
    assert_eq!(
        rows(&query.collect()),
        [vec![0, 0, 0], vec![1, 1, 1]].into()
    );
}

#[test]
fn projection_positions_follow_the_actual_input_order() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    let pair = Row::product(&cx, &[&bit, &bit]);
    let inverse = query::project(&cx, &pair, [1, 0], pair.clone()).inv();
    assert_eq!((&inverse).select(id()).positions(), [1, 0]);
    assert_eq!(inverse.positions(), [1, 0]);
    assert_eq!(
        rows(&inverse.collect()),
        [
            vec![0, 0, 0, 0],
            vec![0, 1, 1, 0],
            vec![1, 0, 0, 1],
            vec![1, 1, 1, 1]
        ]
        .into()
    );
    let repeat = query::project(&cx, &pair, [1, 0, 1], Row::product(&cx, &[&pair, &bit]));
    assert_eq!(repeat.positions(), [1, 0, 1]);
    let unit = Row::of(&cx, cx.sorts().unit());
    assert!(
        query::project(&cx, &pair, [], unit.clone())
            .positions()
            .is_empty()
    );
    assert!(Query::id(&cx, unit).positions().is_empty());
}

#[test]
#[should_panic(expected = "positions require a source-free projection")]
fn positions_of_an_inverse_duplication_are_rejected() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    query::project(&cx, &bit, [0, 0], Row::product(&cx, &[&bit, &bit]))
        .inv()
        .positions();
}

#[test]
#[should_panic(expected = "positions require a projection of the complete input")]
fn positions_of_an_inverse_partial_projection_are_rejected() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    let pair = Row::product(&cx, &[&bit, &bit]);
    query::project(&cx, &pair, [0], bit).inv().positions();
}

#[test]
#[should_panic(expected = "one projection position per range column")]
fn projection_with_surplus_positions_is_rejected() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    query::project(&cx, &bit, [0, 0], bit.clone());
}

#[test]
#[should_panic(expected = "positions require a source-free projection")]
fn positions_of_a_zero_column_filter_are_rejected() {
    let mut cx = tiny();
    let bit = bv(&mut cx, 1);
    let unit = Row::of(&cx, cx.sorts().unit());
    let empty = Rel::new_set(&cx, unit.clone());
    let constraint = query::project(&cx, &bit, [], unit).with(&empty);
    let query = Query::id(&cx, bit).with(constraint);
    assert!(query.is_empty());
    assert!(query.slots.iter().all(|slot| !slot.read));
    query.positions();
}

#[test]
#[should_panic(expected = "projection must preserve logical scalar sorts")]
fn projection_preserves_symbol_namespaces() {
    let mut cx = tiny();
    let x = cx.sorts().symbol("X");
    let y = cx.sorts().symbol("Y");
    let a = Row::bind(&mut cx, x);
    let b = Row::bind(&mut cx, y);
    assert_eq!(query::project(&cx, &a, [0], a.clone()).positions(), [0]);
    query::project(&cx, &a, [0], b);
}

#[test]
fn wide_sources_reach_the_first_callback_on_a_small_stack() {
    for (levels, constrained) in [(16_384, false), (4_096, true)] {
        let levels = if cfg!(miri) { 4 } else { levels };
        let mut cx = tiny();
        let sort = cx
            .sorts()
            .product(std::iter::repeat_n(cx.sorts().bool(), levels));
        let row = Row::bind(&mut cx, sort);
        let a = row.algebras()[0];
        sra::index::Index::with_capacity(&cx, row.algebras(), 32, 64);
        let p = if constrained {
            cx.set(a, &[1])
        } else {
            cx.t(a)
        };
        let mut rel = Rel::new_set(&cx, row);
        rel.insert(vec![p.clone(); levels].as_slice());
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                assert!(
                    query::source(&rel)
                        .drive(&mut |d, r| {
                            assert_eq!(d.cells, r.cells);
                            assert_eq!(d.cells.len(), levels);
                            for i in 0..levels {
                                assert_eq!(d.root(i), (a, p.function()));
                            }
                            ControlFlow::Break(())
                        })
                        .is_break()
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
