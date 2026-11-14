//! Checks for the semi-naive protocol: frontier visibility, novelty, and
//! completion.

use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::Variable;
use crate::context::Context;
use crate::query::IntoQuery;
use crate::rel::{Rel, Universe};
use crate::row::{self, Domain, Row};

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.bv_sort(w);
    sort.bind(cx)
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
fn pairs(rel: &Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

/// A binary relation over `four` holding `ps`.
fn relation(cx: &Context, four: &Row, ps: impl IntoIterator<Item = (u8, u8)>) -> Rel {
    let mut r = Rel::new(cx, four.clone(), four.clone());
    r.insert_rows(ps);
    r
}

/// A 4-bit edge relation with one cycle and one self-loop.
fn edges(cx: &Context, four: &Row) -> Rel {
    relation(cx, four, [(0, 1), (1, 2), (2, 0), (3, 3)])
}

// --- construction and universes ----------------------------------------------

#[test]
fn zero_field_sets_and_binary_relations_keep_their_universes() {
    let mut cx = Context::default();
    let empty = cx
        .tuple_decl("Empty", [] as [(&str, crate::ast::Sort); 0])
        .sort;
    let row = empty.bind(&mut cx);
    let mut s = Rel::new_set(&cx, row.clone());
    let mut r = Rel::new(&cx, row.clone(), row);
    s.insert(());
    r.insert(((), ()));
    for seed in [s, r] {
        for initial in [seed.empty_like(), seed.clone()] {
            let mut v = Variable::new(initial.clone());
            assert_eq!(v.current().universe, seed.universe);
            assert_eq!(v.changed(), !initial.is_empty());
            assert_eq!(v.recent.universe, seed.universe);
            v.insert(&seed);
            assert_eq!(v.changed(), initial.is_empty());
            assert_eq!(v.stable.universe, seed.universe);
            assert!(!v.changed());
            let out = v.complete();
            assert_eq!(out.universe, seed.universe);
            assert_eq!((out.domain.sort, out.range.sort), (empty, empty));
            assert_eq!(out.count(), 1_u32.into());
            match out.universe {
                Universe::Set => assert!(out.contains(())),
                Universe::Binary => assert!(out.contains(((), ()))),
            }
        }
    }
}

// --- current snapshots -------------------------------------------------------

#[test]
fn submissions_are_invisible_and_snapshots_outlive_the_variable() {
    let snapshot = {
        let mut cx = Context::default();
        let four = bv(&mut cx, 4);
        let mut v = Variable::new(set(&cx, &four, [1]));
        let empty = v.current();
        assert!(v.changed());
        let first = v.current();
        v.insert(&set(&cx, &four, [2]));
        v.insert(&set(&cx, &four, [3]));
        assert_eq!(rows(&v.current()), [vec![1]].into());
        assert!(empty.is_empty());
        assert!(v.changed());
        assert_eq!(rows(&v.current()), [vec![1], vec![2], vec![3]].into());
        assert_eq!(rows(&first), [vec![1]].into());
        assert!(!v.changed());
        assert_eq!(rows(&v.complete()), [vec![1], vec![2], vec![3]].into());
        first
    };
    assert_eq!(rows(&snapshot), [vec![1]].into());
}

// --- submissions -------------------------------------------------------------

#[test]
fn incompatible_submissions_are_rejected_without_changing_the_round() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let wide = bv(&mut cx, 5);
    let t = cx.tuple_decl("T", [("value", four.sort)]);
    let backend = cx.backend("bdd");
    cx.set_representation(t.accessors[0], backend);
    let bdd = t.accessors[0].bind(&mut cx);
    let mut foreign = Context::default();
    let other = bv(&mut foreign, 4);
    let mut wrong_context = Rel::new(&foreign, other.clone(), other);
    let mut wrong_domain = Rel::new(&cx, wide.clone(), four.clone());
    let mut wrong_range = Rel::new(&cx, four.clone(), wide);
    let mut wrong_columns = Rel::new(&cx, bdd.clone(), bdd);
    for r in [
        &mut wrong_context,
        &mut wrong_domain,
        &mut wrong_range,
        &mut wrong_columns,
    ] {
        r.insert((2_u8, 3_u8));
    }
    let wrong_universe = set(&cx, &four, [2]);
    let seed = relation(&cx, &four, [(0, 0)]);
    let pending = relation(&cx, &four, [(1, 1)]);
    for wrong in [
        wrong_context,
        wrong_domain,
        wrong_range,
        wrong_columns,
        wrong_universe,
    ] {
        for candidate in [wrong.empty_like(), wrong.clone()] {
            let mut v = Variable::new(seed.clone());
            assert!(v.changed());
            v.insert(&pending);
            assert!(catch_unwind(AssertUnwindSafe(|| v.insert(&candidate))).is_err());
            assert!(v.stable.is_empty());
            assert_eq!(pairs(&v.recent), [(0, 0)].into());
            assert!(v.changed());
            assert_eq!(pairs(&v.recent), [(1, 1)].into());
            assert!(!v.changed());
            assert_eq!(pairs(&v.complete()), [(0, 0), (1, 1)].into());
        }
    }
}

// --- barriers and novelty ----------------------------------------------------

#[test]
fn rounds_expose_only_new_facts() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let edges = edges(&cx, &four);
    let seed = relation(&cx, &four, [(0, 0)]);
    let mut reach = Variable::new(seed.clone());
    assert!(reach.recent.is_empty() && reach.current().is_empty());
    // An extra round indexes past the last frontier and panics instead of hanging.
    let frontiers = [(0, 0), (0, 1), (0, 2)];
    let mut rounds = 0;
    while reach.changed() {
        assert_eq!(pairs(&reach.recent), [frontiers[rounds]].into());
        rounds += 1;
        assert!(reach.current().contains((0_u64, 0_u64)));
        reach.insert(&reach.recent.select(&edges).collect());
        // Resubmitting accepted facts never reopens a round.
        reach.insert(&seed);
    }
    assert_eq!(rounds, 3);
    let closed = reach.complete();
    assert_eq!(pairs(&closed), [(0, 0), (0, 1), (0, 2)].into());
}

#[test]
fn symbolic_frontiers_preserve_complete_pairs_and_correlations() {
    for name in ["sparse", "bdd"] {
        let mut cx = Context::default();
        let t = cx.tuple_decl("T", [("value", cx.bv_sort(2))]);
        let backend = cx.backend(name);
        cx.set_representation(t.accessors[0], backend);
        let two = t.accessors[0].bind(&mut cx);
        let column = row::scalar_column(&two);
        let mut seed = Rel::new(&cx, two.clone(), two);
        seed.insert_rows([(0_u8, 0_u8), (1, 1)]);
        let mut v = Variable::new(seed.clone());
        assert!(v.changed());
        assert_eq!(pairs(&v.recent), [(0, 0), (1, 1)].into());
        // Separate rectangles share a boundary value but do not combine
        // their coordinates into one Cartesian product.
        let mut candidates = seed.empty_like();
        let low = cx.set(column, &[0, 1]);
        let high = cx.set(column, &[1, 2, 3]);
        candidates.insert((&low, &low));
        candidates.insert((&high, &high));
        v.insert(&candidates);
        v.insert(&candidates);
        let expected: BTreeSet<_> = (0..4)
            .flat_map(|x| (0..4).map(move |y| (x, y)))
            .filter(|&(x, y)| (x <= 1 && y <= 1) || (x >= 1 && y >= 1))
            .collect();
        assert!(v.changed());
        assert_eq!(pairs(&v.stable), [(0, 0), (1, 1)].into());
        assert_eq!(
            pairs(&v.recent),
            expected
                .difference(&[(0, 0), (1, 1)].into())
                .copied()
                .collect()
        );
        assert_eq!(pairs(&v.current()), expected);
        // Both old keys acquire new values. Repeating all candidates after
        // acceptance yields no further frontier.
        assert!(v.recent.contains((0_u8, 1_u8)) && v.recent.contains((1_u8, 0_u8)));
        v.insert(&candidates);
        assert!(!v.changed());
        assert!(!v.changed());
        assert_eq!(pairs(&v.complete()), expected);
    }
}

#[test]
fn empty_seeds_and_duplicate_candidates_drain_without_reopening() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let seed = set(&cx, &four, [1, 2]);
    let mut empty = Variable::new(seed.empty_like());
    empty.insert(&seed.empty_like());
    assert!(!empty.changed());
    assert!(!empty.changed());
    assert!(empty.complete().is_empty());
    assert!(Variable::new(seed.empty_like()).complete().is_empty());
    let mut v = Variable::new(seed.clone());
    assert!(v.changed());
    v.insert(&seed);
    assert!(!v.changed());
    v.insert(&seed);
    assert!(!v.changed());
    assert_eq!(rows(&v.complete()), [vec![1], vec![2]].into());
}

// --- completion --------------------------------------------------------------

#[test]
#[should_panic(expected = "cannot complete a variable with recent or pending facts")]
fn completion_with_a_live_frontier_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let mut reach = Variable::new(edges(&cx, &four));
    assert!(reach.changed());
    reach.complete();
}

#[test]
#[should_panic(expected = "cannot complete a variable with recent or pending facts")]
fn completion_with_pending_candidates_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    Variable::new(edges(&cx, &four)).complete();
}
