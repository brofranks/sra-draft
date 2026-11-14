//! Checks for stored relations: construction and lookup, the native Boolean
//! algebra on one index, the per-key range minimum, and batch loading.
//!
//! The `rows` fixture reads back every ground row of a relation over one-word
//! columns by exhaustive membership, and `pairs` is its binary view. The `set`
//! fixture builds a 4-bit set over a bound row, `Arg` mixes ground values and
//! predicates in one batch, and `key_minimum` is the independent reference for
//! `min_range`.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{Rel, Universe};
use crate::ast::Sort;
use crate::backend::BoolOp;
use crate::context::{Context, Predicate};
use crate::row::{self, Domain, IntoRow, Row, RowRef, RowSink};

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

/// One argument that is either a ground value or a scalar predicate, so one
/// batch mixes both.
enum Arg {
    Ground(u8),
    Sym(Predicate),
}

impl IntoRow for Arg {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        match self {
            Arg::Ground(v) => v.append(row, out),
            Arg::Sym(p) => p.append(row, out),
        }
    }
}

// --- construction and lookup ----------------------------------------------

#[test]
fn a_loaded_relation_reads_back_through_its_index() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let mut r = Rel::new(&cx, four.clone(), four);
    r.insert_rows([(1_u8, 2_u8), (2, 3), (3, 1)]);
    assert_eq!(rows(&r), [vec![1, 2], vec![2, 3], vec![3, 1]].into());
    assert_eq!(r.count().to_string(), "3");
    assert_eq!(r.not().count().to_string(), "253");
    assert_eq!(r.root.constrains(), [true, true]);
}

#[test]
fn insert_and_contains_go_through_bound_records() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.accessors[1], bdd);
    let four = bv(&mut cx, 4);
    let record = t.sort.bind(&mut cx);
    // A record set takes its members positionally.
    let mut s = Rel::new_set(&cx, t.sort);
    s.insert_rows([(1_u8, 200_u8), (2, 3)]);
    assert!(s.contains((1_u8, 200_u8)) && !s.contains((1_u8, 3_u8)));
    assert_eq!(s.count().to_string(), "2");
    // A record domain beside a scalar range flattens through a nested
    // tuple, and the constructor and the sort name the same columns.
    let mut r = Rel::new(&cx, t.constructor, four);
    r.insert(((1_u8, 2_u8), 3_u8));
    assert!(r.contains(((1_u8, 2_u8), 3_u8)) && !r.contains(((1_u8, 2_u8), 4_u8)));
    assert_eq!(row::columns(&r.domain), row::columns(&record));
    assert_eq!(r.root.arity(), 3);
}

#[test]
fn unit_relations_hold_at_most_the_empty_row() {
    let mut cx = Context::default();
    let unit = cx.unit_sort().bind(&mut cx);
    let mut u = Rel::new_set(&cx, unit);
    assert!(u.is_empty() && !u.contains(()));
    u.insert_rows(std::iter::empty::<()>());
    assert!(u.is_empty());
    u.insert(());
    assert!(!u.is_empty() && u.contains(()) && u.count().to_string() == "1");
}

#[test]
fn zero_column_batches_consume_and_validate_every_argument() {
    use std::cell::Cell;

    struct Unit(bool);

    impl IntoRow for Unit {
        fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
            if self.0 {
                ().append(row, out);
            } else {
                0_u8.append(row, out);
            }
        }
    }

    let mut cx = Context::default();
    let unit = cx.unit_sort().bind(&mut cx);
    let mut rel = Rel::new_set(&cx, unit);
    let calls = Cell::new(0);
    let rejected = catch_unwind(AssertUnwindSafe(|| {
        rel.insert_rows([true, true, false].into_iter().map(|valid| {
            calls.set(calls.get() + 1);
            Unit(valid)
        }));
    }));
    assert!(rejected.is_err());
    assert_eq!(calls.get(), 3);
    assert!(rel.is_empty());
    calls.set(0);
    rel.insert_rows((0..4).map(|_| {
        calls.set(calls.get() + 1);
    }));
    assert_eq!(calls.get(), 4);
    assert_eq!(rel.count().to_string(), "1");
}

#[test]
fn zero_column_relations_preserve_their_declared_universe() {
    let mut cx = Context::default();
    let empty = cx.tuple_decl("Empty", [] as [(&str, Sort); 0]).sort;
    let record = empty.bind(&mut cx);
    let unit = cx.unit_sort().bind(&mut cx);
    let mut set = Rel::new_set(&cx, record.clone());
    let mut binary = Rel::new(&cx, record.clone(), record);
    set.insert_rows(std::iter::empty::<()>());
    binary.insert_rows(std::iter::empty::<((), ())>());
    assert!(set.is_empty() && binary.is_empty());
    assert!(!set.contains(()) && !binary.contains(((), ())));
    set.insert(());
    binary.insert(((), ()));
    assert_eq!(set.count().to_string(), "1");
    assert_eq!(binary.count().to_string(), "1");
    assert!(set.not().is_empty() && binary.not().is_empty());
    assert!(catch_unwind(AssertUnwindSafe(|| set.contains(((), ())))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| binary.contains(()))).is_err());
    for r in [
        set.clone(),
        set.empty_like().not(),
        set.not().not(),
        set.apply_bool(BoolOp::Or, &set),
    ] {
        assert_eq!(r.universe, Universe::Set);
        assert!(r.contains(()));
    }
    for r in [
        binary.clone(),
        binary.empty_like().not(),
        binary.not().not(),
        binary.apply_bool(BoolOp::And, &binary),
    ] {
        assert_eq!(r.universe, Universe::Binary);
        assert!(r.contains(((), ())));
    }
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        assert!(catch_unwind(AssertUnwindSafe(|| set.apply_bool(op, &binary))).is_err());
    }
    // Anonymous units normalize out of the binary argument product.
    let mut units = Rel::new(&cx, unit.clone(), unit);
    assert_eq!(units.universe, Universe::Binary);
    assert!(units.is_empty());
    units.insert_rows([(), ()]);
    assert!(units.contains(()));
    assert_eq!(units.count().to_string(), "1");
    assert!(units.not().is_empty());
}

// --- Boolean operations ---------------------------------------------------

#[test]
fn boolean_algebra_combines_relations_over_one_index() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let (a, b) = (set(&cx, &four, [1, 2]), set(&cx, &four, [2, 3]));
    assert_eq!(rows(&a.apply_bool(BoolOp::And, &b)), [vec![2]].into());
    assert_eq!(
        rows(&a.apply_bool(BoolOp::Or, &b)),
        [vec![1], vec![2], vec![3]].into()
    );
    assert_eq!(rows(&a.apply_bool(BoolOp::AndNot, &b)), [vec![1]].into());
    assert_eq!(a.not().count().to_string(), "14");
    assert_eq!(rows(&a.not().not()), rows(&a));
    let none = a.empty_like();
    assert!(none.is_empty() && none.count().to_string() == "0" && !a.is_empty());
    assert_eq!(rows(&none.apply_bool(BoolOp::Or, &a)), rows(&a));
    assert!(a.contains(2_u8) && !a.contains(3_u8));
    let least = a.min_range();
    assert_eq!(least.universe, Universe::Set);
    assert_eq!(rows(&least), [vec![1]].into());
    let mut r = Rel::new(&cx, four.clone(), four.clone());
    r.insert((1_u8, 2_u8));
    assert!(r.contains((1_u8, 2_u8)) && !r.contains((2_u8, 1_u8)));
    assert_eq!(r.not().count().to_string(), "255");
}

#[test]
#[should_panic(expected = "relation endpoints have different logical sorts")]
fn domain_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    set(&cx, &four, [1]).apply_bool(BoolOp::Or, &Rel::new_set(&cx, eight));
}

#[test]
#[should_panic(expected = "relation endpoints have different logical sorts")]
fn range_sort_mismatch_is_rejected() {
    let mut cx = Context::default();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    let r = Rel::new(&cx, four.clone(), four.clone());
    r.apply_bool(BoolOp::Or, &Rel::new(&cx, four, eight));
}

#[test]
#[should_panic(expected = "relation operations require the same set or binary universe")]
fn universe_mismatch_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    set(&cx, &four, [1]).apply_bool(BoolOp::Or, &Rel::new(&cx, four.clone(), four.clone()));
}

#[test]
#[should_panic(expected = "relation operations require identical field columns")]
fn column_mismatch_is_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("x", cx.bv_sort(4)), ("y", cx.bv_sort(4))]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.accessors[0], bdd);
    let (x, y) = (t.accessors[0].bind(&mut cx), t.accessors[1].bind(&mut cx));
    Rel::new_set(&cx, x).apply_bool(BoolOp::Or, &Rel::new_set(&cx, y));
}

// --- per-key minimum ------------------------------------------------------

/// The per-key minimum of ground rows: every column but the last is the
/// key.
fn key_minimum(rows: &BTreeSet<Vec<u64>>) -> BTreeSet<Vec<u64>> {
    let mut least = BTreeMap::new();
    for row in rows {
        let (key, rank) = row.split_at(row.len() - 1);
        least
            .entry(key.to_vec())
            .and_modify(|r: &mut u64| *r = (*r).min(rank[0]))
            .or_insert(rank[0]);
    }
    least
        .into_iter()
        .map(|(mut k, r)| {
            k.push(r);
            k
        })
        .collect()
}

#[test]
#[should_panic(expected = "minimum requires a range column")]
fn min_range_of_a_unit_range_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let unit = cx.unit_sort().bind(&mut cx);
    let mut r = Rel::new(&cx, four, unit);
    r.insert_rows([1_u8, 2_u8]);
    r.min_range();
}

#[test]
fn min_range_rejects_every_zero_column_range_even_when_empty() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let unit = cx.unit_sort().bind(&mut cx);
    let empty = cx
        .tuple_decl("Empty", [] as [(&str, Sort); 0])
        .sort
        .bind(&mut cx);
    let mut named = Rel::new(&cx, four.clone(), empty.clone());
    named.insert_rows([(1_u8, ()), (2_u8, ())]);
    let mut plain = Rel::new(&cx, four, unit.clone());
    plain.insert_rows([1_u8, 2_u8]);
    let mut unit_set = Rel::new_set(&cx, unit);
    unit_set.insert(());
    let mut named_set = Rel::new_set(&cx, empty);
    named_set.insert(());
    for r in [named, plain, unit_set, named_set] {
        let count = r.count();
        for candidate in [r.empty_like(), r.clone()] {
            assert!(catch_unwind(AssertUnwindSafe(|| candidate.min_range())).is_err());
        }
        assert_eq!(r.count(), count);
    }
}

#[test]
fn min_range_agrees_with_the_per_key_minimum_on_random_relations() {
    let mut cx = Context::default();
    let three = cx.bv_sort(3);
    let four = cx.bv_sort(4);
    // A product domain and a scalar range, and a scalar domain with a product
    // range: in both shapes every stored column before the last is a key.
    let keys = cx.product_sort([three, three]).bind(&mut cx);
    let rank = four.bind(&mut cx);
    let key = three.bind(&mut cx);
    let ranks = cx.product_sort([three, four]).bind(&mut cx);
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..8 {
        let mut product = Rel::new(&cx, keys.clone(), rank.clone());
        let mut record = Rel::new(&cx, key.clone(), ranks.clone());
        assert!(product.min_range().is_empty() && record.min_range().is_empty());
        let drawn: Vec<(u8, u8, u8)> = (0..40)
            .map(|_| {
                let r = next();
                ((r & 7) as u8, (r >> 3 & 7) as u8, (r >> 6 & 15) as u8)
            })
            .collect();
        product.insert_rows(drawn.iter().map(|&(a, b, c)| ((a, b), c)));
        record.insert_rows(drawn.iter().map(|&(a, b, c)| (a, (b, c))));
        assert_eq!(rows(&product.min_range()), key_minimum(&rows(&product)));
        assert_eq!(rows(&record.min_range()), key_minimum(&rows(&record)));
    }
}

#[test]
fn min_range_is_independent_of_insertion_order_and_batching() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let drawn = [(1_u8, 5_u8), (1, 2), (1, 9), (2, 7), (3, 3), (3, 4)];
    let mut ascending = Rel::new(&cx, four.clone(), four.clone());
    ascending.insert_rows(drawn);
    let mut descending = Rel::new(&cx, four.clone(), four.clone());
    descending.insert_rows(drawn.iter().rev().copied());
    let mut single = Rel::new(&cx, four.clone(), four.clone());
    for pair in drawn {
        single.insert(pair);
    }
    let expected = [(1, 2), (2, 7), (3, 3)].into();
    assert_eq!(ascending.min_range().universe, Universe::Binary);
    assert_eq!(pairs(&ascending.min_range()), expected);
    assert_eq!(
        ascending.min_range().root.function,
        descending.min_range().root.function
    );
    assert_eq!(
        ascending.min_range().root.function,
        single.min_range().root.function
    );
    // A later, better rank replaces an earlier one; the candidates
    // themselves stay stored.
    let mut r = Rel::new(&cx, four.clone(), four);
    r.insert((1_u8, 5_u8));
    assert_eq!(pairs(&r.min_range()), [(1, 5)].into());
    r.insert((1_u8, 2_u8));
    assert_eq!(pairs(&r.min_range()), [(1, 2)].into());
    assert_eq!(pairs(&r), [(1, 2), (1, 5)].into());
}

// --- batch loading --------------------------------------------------------

#[test]
fn batches_sort_and_deduplicate_at_every_stride() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let seventy = bv(&mut cx, 70);
    let pair = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4)])
        .bind(&mut cx);
    let s = set(&cx, &four, [3, 1, 3, 2, 1]);
    assert_eq!(rows(&s), [vec![1], vec![2], vec![3]].into());
    // Two one-word columns sort as flat two-word records.
    let mut two = Rel::new(&cx, four.clone(), four.clone());
    two.insert_rows([(1_u8, 5_u8), (1, 2), (1, 5), (0, 9)]);
    assert_eq!(two.count().to_string(), "3");
    assert!(
        two.contains((1_u8, 2_u8)) && two.contains((1_u8, 5_u8)) && !two.contains((0_u8, 5_u8))
    );
    // One two-word column compares column by column.
    let mut wide = Rel::new_set(&cx, seventy.clone());
    wide.insert_rows([[5_u64, 32], [1, 0], [5, 32], [0, 1]]);
    assert_eq!(wide.count().to_string(), "3");
    assert!(wide.contains([5_u64, 32]) && wide.contains([0_u64, 1]) && !wide.contains([5_u64, 0]));
    // Three one-word columns, then four, which falls back to slice
    // comparison.
    let mut three = Rel::new(&cx, pair.clone(), four.clone());
    three.insert_rows([(1_u8, 2_u8, 3_u8), (1, 2, 4), (1, 2, 3)]);
    assert_eq!(three.count().to_string(), "2");
    assert!(three.contains(((1_u8, 2_u8), 4_u8)));
    let mut fourfold = Rel::new(&cx, pair.clone(), pair);
    fourfold.insert_rows([(1_u8, 2_u8, 3_u8, 4_u8), (0, 0, 0, 1), (1, 2, 3, 4)]);
    assert_eq!(fourfold.count().to_string(), "2");
    assert!(fourfold.contains((0_u8, 0_u8, 0_u8, 1_u8)));
    // A multi-word column beside a one-word column compares column by
    // column.
    let mut mixed = Rel::new(&cx, seventy, four.clone());
    mixed.insert_rows([([1_u64, 0], 2_u8), ([1, 0], 1), ([0, 1], 2), ([1, 0], 2)]);
    assert_eq!(mixed.count().to_string(), "3");
    assert!(mixed.contains(([0_u64, 1], 2_u8)) && !mixed.contains(([0_u64, 1], 1_u8)));
    // An empty batch contributes nothing.
    let mut none = Rel::new_set(&cx, four);
    none.insert_rows(std::iter::empty::<u8>());
    assert!(none.is_empty());
}

#[test]
fn symbolic_rows_keep_independent_coordinates_and_separate_correlations() {
    for backend in ["sparse", "bdd"] {
        let mut cx = Context::default();
        let t = cx.tuple_decl("T", [("value", cx.bv_sort(4))]);
        let backend = cx.backend(backend);
        cx.set_representation(t.accessors[0], backend);
        let four = t.accessors[0].bind(&mut cx);
        let column = row::scalar_column(&four);
        let p = cx.set(column, &[1, 2]);
        let q = cx.set(column, &[3, 4]);
        let a = cx.set(column, &[5, 6]);
        let b = cx.set(column, &[7, 8]);
        let mut same = Rel::new(&cx, four.clone(), four.clone());
        same.insert((&p, &p));
        assert_eq!(pairs(&same), [(1, 1), (1, 2), (2, 1), (2, 2)].into());
        let mut separate = Rel::new(&cx, four.clone(), four);
        separate.insert_rows([(&p, &q), (&a, &b)]);
        assert_eq!(
            pairs(&separate),
            [
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (5, 7),
                (5, 8),
                (6, 7),
                (6, 8)
            ]
            .into()
        );
    }
}

#[test]
fn symbolic_and_ground_rows_share_one_batch() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let column = row::scalar_column(&four);
    let some = |vs: &[u64]| Arg::Sym(cx.set(column, vs));
    let mut r = Rel::new(&cx, four.clone(), four.clone());
    r.insert_rows([
        (Arg::Ground(1), Arg::Ground(2)),
        (Arg::Ground(1), Arg::Ground(3)),
        // One predicate reused across rows offers the same independent
        // choices each time.
        (some(&[1, 2]), Arg::Ground(5)),
        (some(&[1, 2]), Arg::Ground(6)),
        (some(&[3]), Arg::Ground(7)),
        // False predicates drop their rows in either column.
        (some(&[]), Arg::Ground(8)),
        (Arg::Ground(4), some(&[])),
        (Arg::Ground(9), some(&[10, 11])),
    ]);
    assert_eq!(
        pairs(&r),
        [
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 6),
            (2, 5),
            (2, 6),
            (3, 7),
            (9, 10),
            (9, 11)
        ]
        .into()
    );
    // Symbolic rows alone.
    let mut s = Rel::new_set(&cx, four);
    s.insert_rows([some(&[4, 6]), some(&[6, 7])]);
    assert_eq!(rows(&s), [vec![4], vec![6], vec![7]].into());
}
