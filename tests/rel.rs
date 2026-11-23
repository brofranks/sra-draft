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

use sra::backend::BoolOp;
use sra::bdd::BddStore;
use sra::context::{Context, Predicate};
use sra::index::Index;
use sra::rel::{Rel, Universe};
use sra::row::{Domain, IntoRow, Row, RowRef, RowSink};
use sra::sparse::SparseStore;
use sra::util::count::Count;

/// A context with small scalar stores for loading and counting.
fn tiny() -> Context {
    let mut cx = Context::empty();
    cx.register_backend("bdd", BddStore::with_capacity(8, 16, 1));
    cx.register_backend("sparse", SparseStore::with_capacity(8, 16, 1));
    cx
}

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.sorts().bv(w);
    Row::bind(cx, sort)
}

/// Small native arenas exercise collection during relation operations.
fn new_set(cx: &Context, row: Row) -> Rel {
    Index::with_capacity(cx, row.algebras(), 8, 16);
    Rel::new_set(cx, row)
}

fn binary_rel(cx: &Context, domain: Row, range: Row) -> Rel {
    Index::with_capacity(cx, Row::product(cx, &[&domain, &range]).algebras(), 8, 16);
    Rel::new(cx, domain, range)
}

/// A 4-bit set holding `values`, over the already bound row `four`.
fn set(cx: &Context, four: &Row, values: impl IntoIterator<Item = u8>) -> Rel {
    let mut s = new_set(cx, four.clone());
    s.insert_rows(values);
    s
}

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &Rel) -> BTreeSet<Vec<u64>> {
    let index = rel.root().index();
    let widths: Vec<u32> = (0..index.arity())
        .map(|i| {
            index
                .context()
                .resolve(index.algebra(i))
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
#[should_panic(expected = "declaration belongs to another store")]
fn an_endpoint_with_a_foreign_empty_row_is_rejected() {
    struct Endpoint(Row);

    impl Domain for Endpoint {
        fn row(self, _: &Context) -> Row {
            self.0
        }
    }

    let other = tiny();
    let row = Row::of(&other, other.sorts().unit());
    let _ = Rel::new_set(&tiny(), Endpoint(row));
}

#[test]
fn a_loaded_relation_reads_back_through_its_index() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let mut r = binary_rel(&cx, four.clone(), four);
    r.insert_rows([(1_u8, 2_u8), (2, 3), (3, 1)]);
    assert_eq!(rows(&r), [vec![1, 2], vec![2, 3], vec![3, 1]].into());
    assert_eq!(r.count(), 3);
    assert_eq!(r.not().count(), 253);
}

#[test]
fn insert_and_contains_go_through_bound_records() {
    let mut cx = tiny();
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(8))]);
    cx.bind_field(cx.sorts().project(t, 1), "bdd");
    let four = bv(&mut cx, 4);
    let record = Row::bind(&mut cx, t);
    // A record set takes its members positionally.
    Index::with_capacity(&cx, record.algebras(), 8, 16);
    let mut s = Rel::new_set(&cx, t);
    s.insert_rows([(1_u8, 200_u8), (2, 3)]);
    assert!(s.contains((1_u8, 200_u8)) && !s.contains((1_u8, 3_u8)));
    assert_eq!(s.count(), 2);
    // A record domain beside a scalar range flattens through a nested
    // tuple, and the constructor and the sort name the same columns.
    Index::with_capacity(&cx, Row::product(&cx, &[&record, &four]).algebras(), 8, 16);
    let mut r = Rel::new(&cx, cx.sorts().constructor(t), four);
    r.insert(((1_u8, 2_u8), 3_u8));
    assert!(r.contains(((1_u8, 2_u8), 3_u8)) && !r.contains(((1_u8, 2_u8), 4_u8)));
    assert_eq!(r.domain().algebras(), record.algebras());
    assert_eq!(r.root().index().arity(), 3);
}

#[test]
fn unit_relations_hold_at_most_the_empty_row() {
    let mut cx = tiny();
    let sort = cx.sorts().unit();
    let unit = Row::bind(&mut cx, sort);
    let mut u = new_set(&cx, unit);
    assert!(u.is_empty() && !u.contains(()));
    u.insert_rows(std::iter::empty::<()>());
    assert!(u.is_empty());
    u.insert(());
    assert!(!u.is_empty() && u.contains(()) && u.count() == 1);
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

    let mut cx = tiny();
    let sort = cx.sorts().unit();
    let unit = Row::bind(&mut cx, sort);
    let mut rel = new_set(&cx, unit);
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
    assert_eq!(rel.count(), 1);
}

#[test]
fn zero_column_relations_preserve_their_declared_universe() {
    let mut cx = tiny();
    let empty = cx.sorts().record("Empty", &[]);
    let record = Row::bind(&mut cx, empty);
    let sort = cx.sorts().unit();
    let unit = Row::bind(&mut cx, sort);
    let mut set = new_set(&cx, record.clone());
    let mut binary = binary_rel(&cx, record.clone(), record);
    set.insert_rows(std::iter::empty::<()>());
    binary.insert_rows(std::iter::empty::<((), ())>());
    assert!(set.is_empty() && binary.is_empty());
    assert!(!set.contains(()) && !binary.contains(((), ())));
    set.insert(());
    binary.insert(((), ()));
    assert_eq!(set.count(), 1);
    assert_eq!(binary.count(), 1);
    assert!(set.not().is_empty() && binary.not().is_empty());
    assert!(catch_unwind(AssertUnwindSafe(|| set.contains(((), ())))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| binary.contains(()))).is_err());
    for r in [
        set.clone(),
        set.empty_like().not(),
        set.not().not(),
        set.apply_bool(BoolOp::Or, &set),
    ] {
        assert_eq!(r.universe(), Universe::Set);
        assert!(r.contains(()));
    }
    for r in [
        binary.clone(),
        binary.empty_like().not(),
        binary.not().not(),
        binary.apply_bool(BoolOp::And, &binary),
    ] {
        assert_eq!(r.universe(), Universe::Binary);
        assert!(r.contains(((), ())));
    }
    for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
        assert!(catch_unwind(AssertUnwindSafe(|| set.apply_bool(op, &binary))).is_err());
    }
    // Anonymous units normalize out of the binary argument product.
    let mut units = binary_rel(&cx, unit.clone(), unit);
    assert_eq!(units.universe(), Universe::Binary);
    assert!(units.is_empty());
    units.insert_rows([(), ()]);
    assert!(units.contains(()));
    assert_eq!(units.count(), 1);
    assert!(units.not().is_empty());
}

// --- Boolean operations ---------------------------------------------------

#[test]
fn a_diagonal_and_a_set_take_complements_in_their_declared_universes() {
    let mut cx = tiny();
    let row = bv(&mut cx, 2);
    let a = row.algebras()[0];
    Index::with_capacity(&cx, &[a], 8, 16);
    Index::with_capacity(&cx, &[a, a], 8, 16);
    let mut set = Rel::new_set(&cx, &row);
    let mut diagonal = Rel::new(&cx, &row, &row);
    set.insert_rows([0_u8, 1]);
    diagonal.insert_rows([(0_u8, 0_u8), (1, 1)]);
    let (set, diagonal) = (set.not(), diagonal.not());
    for x in 0_u8..4 {
        assert_eq!(set.contains(x), x >= 2);
        for y in 0_u8..4 {
            assert_eq!(diagonal.contains((x, y)), x != y || x >= 2);
        }
    }
    assert_eq!(set.count(), 2);
    assert_eq!(diagonal.count(), 14);
}

#[test]
fn boolean_algebra_combines_relations_over_one_index() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let (a, b) = (set(&cx, &four, [1, 2]), set(&cx, &four, [2, 3]));
    assert_eq!(rows(&a.apply_bool(BoolOp::And, &b)), [vec![2]].into());
    assert_eq!(
        rows(&a.apply_bool(BoolOp::Or, &b)),
        [vec![1], vec![2], vec![3]].into()
    );
    assert_eq!(rows(&a.apply_bool(BoolOp::AndNot, &b)), [vec![1]].into());
    assert_eq!(a.not().count(), 14);
    assert_eq!(rows(&a.not().not()), rows(&a));
    let none = a.empty_like();
    assert!(none.is_empty() && none.count() == 0 && !a.is_empty());
    assert_eq!(rows(&none.apply_bool(BoolOp::Or, &a)), rows(&a));
    assert!(a.contains(2_u8) && !a.contains(3_u8));
    let least = a.min_range();
    assert_eq!(least.universe(), Universe::Set);
    assert_eq!(rows(&least), [vec![1]].into());
    let mut r = binary_rel(&cx, four.clone(), four.clone());
    r.insert((1_u8, 2_u8));
    assert!(r.contains((1_u8, 2_u8)) && !r.contains((2_u8, 1_u8)));
    assert_eq!(r.not().count(), 255);
}

#[test]
#[should_panic(expected = "relation endpoints have different logical sorts")]
fn domain_sort_mismatch_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let [a, b] = ["A", "B"].map(|name| {
        let sort = cx.sorts().record(name, &[("x", four.sort())]);
        let domain = Row::new(&cx, sort, four.algebras().into(), 0);
        binary_rel(&cx, domain, four.clone())
    });
    let _ = a.apply_bool(BoolOp::Or, &b);
}

#[test]
#[should_panic(expected = "relation endpoints have different logical sorts")]
fn range_sort_mismatch_is_rejected() {
    let mut cx = tiny();
    let (four, eight) = (bv(&mut cx, 4), bv(&mut cx, 8));
    let r = binary_rel(&cx, four.clone(), four.clone());
    let _ = r.apply_bool(BoolOp::Or, &binary_rel(&cx, four, eight));
}

#[test]
#[should_panic(expected = "relation operations require the same set or binary universe")]
fn universe_mismatch_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let _ =
        set(&cx, &four, [1]).apply_bool(BoolOp::Or, &binary_rel(&cx, four.clone(), four.clone()));
}

#[test]
#[should_panic(expected = "index roots must share a physical layout")]
fn column_mismatch_is_rejected() {
    let mut cx = tiny();
    let t = cx
        .sorts()
        .record("T", &[("x", cx.sorts().bv(4)), ("y", cx.sorts().bv(4))]);
    cx.bind_field(cx.sorts().project(t, 0), "bdd");
    let x = Row::of(&cx, cx.sorts().project(t, 0));
    let y = cx.sorts().project(t, 1);
    let y = Row::bind(&mut cx, y);
    let _ = new_set(&cx, x).apply_bool(BoolOp::Or, &new_set(&cx, y));
}

// --- per-key minimum ------------------------------------------------------

#[test]
fn overlapping_symbolic_keys_preserve_each_keys_minimum() {
    for backend in ["sparse", "bdd"] {
        let mut cx = tiny();
        let sort = cx.sorts().bv(2);
        let a = cx.bind(backend, sort.scalar());
        let row = Row::new(&cx, sort, [a].into(), 0);
        let mut rel = binary_rel(&cx, row.clone(), row);
        let keys = cx.set(a, &[0, 1]);
        let one = cx.set(a, &[1]);
        let two = cx.set(a, &[2]);
        rel.insert_rows([(&keys, &two), (&one, &one)]);
        assert_eq!(pairs(&rel), [(0, 2), (1, 1), (1, 2)].into());
        assert_eq!(pairs(&rel.min_range()), [(0, 2), (1, 1)].into());
    }
}

#[test]
fn wide_ranks_compare_the_high_word_first_under_each_key() {
    for backend in ["sparse", "bdd"] {
        let mut cx = tiny();
        let key = bv(&mut cx, 2);
        let sort = cx.sorts().bv(65);
        let a = cx.bind(backend, sort.scalar());
        let rank = Row::new(&cx, sort, [a].into(), 0);
        let mut rel = binary_rel(&cx, key, rank);
        let low = [u64::MAX, 0];
        let high = [0, 1];
        rel.insert_rows([(0_u8, high), (1, high), (0, low)]);
        let min = rel.min_range();
        assert_eq!(min.count(), 2);
        assert!(min.contains((0_u8, low)) && min.contains((1_u8, high)));
        assert!(!min.contains((0_u8, high)));
    }
}

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
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let sort = cx.sorts().unit();
    let unit = Row::bind(&mut cx, sort);
    let mut r = binary_rel(&cx, four, unit);
    r.insert_rows([1_u8, 2_u8]);
    let _ = r.min_range();
}

#[test]
fn min_range_rejects_every_zero_column_range_even_when_empty() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let sort = cx.sorts().unit();
    let unit = Row::bind(&mut cx, sort);
    let sort = cx.sorts().record("Empty", &[]);
    let empty = Row::bind(&mut cx, sort);
    let mut named = binary_rel(&cx, four.clone(), empty.clone());
    named.insert_rows([(1_u8, ()), (2_u8, ())]);
    let mut plain = binary_rel(&cx, four, unit.clone());
    plain.insert_rows([1_u8, 2_u8]);
    let mut unit_set = new_set(&cx, unit);
    unit_set.insert(());
    let mut named_set = new_set(&cx, empty);
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
    let mut cx = tiny();
    // Exhaustive membership grows exponentially with the total bit width.
    let (key_bits, rank_bits, rounds, draws) = if cfg!(miri) {
        (2, 2, 1, 16)
    } else {
        (3, 4, 8, 40)
    };
    let key_sort = cx.sorts().bv(key_bits);
    let rank_sort = cx.sorts().bv(rank_bits);
    let key_mask = (1 << key_bits) - 1;
    let rank_mask = (1 << rank_bits) - 1;
    // A product domain and a scalar range, and a scalar domain with a product
    // range: in both shapes every stored column before the last is a key.
    let sort = cx.sorts().product([key_sort, key_sort]);
    let keys = Row::bind(&mut cx, sort);
    let rank = Row::bind(&mut cx, rank_sort);
    let key = Row::bind(&mut cx, key_sort);
    let sort = cx.sorts().product([key_sort, rank_sort]);
    let ranks = Row::bind(&mut cx, sort);
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..rounds {
        let mut product = binary_rel(&cx, keys.clone(), rank.clone());
        let mut record = binary_rel(&cx, key.clone(), ranks.clone());
        assert!(product.min_range().is_empty() && record.min_range().is_empty());
        let drawn: Vec<(u8, u8, u8)> = (0..draws)
            .map(|_| {
                let r = next();
                (
                    (r & key_mask) as u8,
                    (r >> key_bits & key_mask) as u8,
                    (r >> (2 * key_bits) & rank_mask) as u8,
                )
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
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let drawn = [(1_u8, 5_u8), (1, 2), (1, 9), (2, 7), (3, 3), (3, 4)];
    let mut ascending = binary_rel(&cx, four.clone(), four.clone());
    ascending.insert_rows(drawn);
    let mut descending = binary_rel(&cx, four.clone(), four.clone());
    descending.insert_rows(drawn.iter().rev().copied());
    let mut single = binary_rel(&cx, four.clone(), four.clone());
    for pair in drawn {
        single.insert(pair);
    }
    let expected = [(1, 2), (2, 7), (3, 3)].into();
    assert_eq!(ascending.min_range().universe(), Universe::Binary);
    assert_eq!(pairs(&ascending.min_range()), expected);
    assert_eq!(pairs(&descending.min_range()), expected);
    assert_eq!(pairs(&single.min_range()), expected);
    // A later, better rank replaces an earlier one; the candidates
    // themselves stay stored.
    let mut r = binary_rel(&cx, four.clone(), four);
    r.insert((1_u8, 5_u8));
    assert_eq!(pairs(&r.min_range()), [(1, 5)].into());
    r.insert((1_u8, 2_u8));
    assert_eq!(pairs(&r.min_range()), [(1, 2)].into());
    assert_eq!(pairs(&r), [(1, 2), (1, 5)].into());
}

// --- batch loading --------------------------------------------------------

#[test]
fn wide_symbolic_insertion_fits_a_two_mib_stack() {
    let levels = if cfg!(miri) { 8 } else { 16_384 };
    let mut cx = tiny();
    let boolean = cx.sorts().bool();
    let sort = cx.sorts().product(std::iter::repeat_n(boolean, levels));
    let row = Row::bind(&mut cx, sort);
    let args = vec![cx.t(row.algebras()[0]); levels];
    let mut relation = new_set(&cx, row);
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            relation.insert(args.as_slice());
            assert!(relation.root().contains(&vec![0; levels]));
            assert!(relation.root().contains(&vec![1; levels]));
            assert_eq!(relation.count(), Count::from(1).shift(levels as u64));
        })
        .unwrap()
        .join()
        .unwrap_or_else(|p| std::panic::resume_unwind(p));
}

#[test]
fn batches_sort_and_deduplicate_through_specialized_and_general_paths() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let seventy = bv(&mut cx, 70);
    let sort = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4)]);
    let pair = Row::bind(&mut cx, sort);
    let s = set(&cx, &four, [3, 1, 3, 2, 1]);
    assert_eq!(rows(&s), [vec![1], vec![2], vec![3]].into());
    // Two one-word columns sort as flat two-word records.
    let mut two = binary_rel(&cx, four.clone(), four.clone());
    two.insert_rows([(1_u8, 5_u8), (1, 2), (1, 5), (0, 9)]);
    assert_eq!(two.count(), 3);
    assert!(
        two.contains((1_u8, 2_u8)) && two.contains((1_u8, 5_u8)) && !two.contains((0_u8, 5_u8))
    );
    // One two-word column compares column by column.
    let mut wide = new_set(&cx, seventy.clone());
    wide.insert_rows([[5_u64, 32], [1, 0], [5, 32], [0, 1]]);
    assert_eq!(wide.count(), 3);
    assert!(wide.contains([5_u64, 32]) && wide.contains([0_u64, 1]) && !wide.contains([5_u64, 0]));
    // Three one-word columns, then four, which falls back to slice
    // comparison.
    let mut three = binary_rel(&cx, pair.clone(), four.clone());
    three.insert_rows([(1_u8, 2_u8, 3_u8), (1, 2, 4), (1, 2, 3)]);
    assert_eq!(three.count(), 2);
    assert!(three.contains(((1_u8, 2_u8), 4_u8)));
    let mut fourfold = binary_rel(&cx, pair.clone(), pair);
    fourfold.insert_rows([(1_u8, 2_u8, 3_u8, 4_u8), (0, 0, 0, 1), (1, 2, 3, 4)]);
    assert_eq!(fourfold.count(), 2);
    assert!(fourfold.contains((0_u8, 0_u8, 0_u8, 1_u8)));
    // A multi-word column beside a one-word column compares column by
    // column.
    let mut mixed = binary_rel(&cx, seventy, four.clone());
    mixed.insert_rows([([1_u64, 0], 2_u8), ([1, 0], 1), ([0, 1], 2), ([1, 0], 2)]);
    assert_eq!(mixed.count(), 3);
    assert!(mixed.contains(([0_u64, 1], 2_u8)) && !mixed.contains(([0_u64, 1], 1_u8)));
    // An empty batch contributes nothing.
    let mut none = new_set(&cx, four);
    none.insert_rows(std::iter::empty::<u8>());
    assert!(none.is_empty());
}

#[test]
fn symbolic_rows_keep_independent_coordinates_and_separate_correlations() {
    for backend in ["sparse", "bdd"] {
        let mut cx = tiny();
        let t = cx.sorts().record("T", &[("value", cx.sorts().bv(4))]);
        cx.bind_field(cx.sorts().project(t, 0), backend);
        let sort = cx.sorts().project(t, 0);
        let four = Row::bind(&mut cx, sort);
        let column = four.algebras()[0];
        let p = cx.set(column, &[1, 2]);
        let q = cx.set(column, &[3, 4]);
        let a = cx.set(column, &[5, 6]);
        let b = cx.set(column, &[7, 8]);
        let mut same = binary_rel(&cx, four.clone(), four.clone());
        same.insert((&p, &p));
        assert_eq!(pairs(&same), [(1, 1), (1, 2), (2, 1), (2, 2)].into());
        let mut separate = binary_rel(&cx, four.clone(), four);
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
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let column = four.algebras()[0];
    let some = |vs: &[u64]| Arg::Sym(cx.set(column, vs));
    let mut r = binary_rel(&cx, four.clone(), four.clone());
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
    // Three symbolic rows leave two pending unions, each with an exclusive value.
    let mut s = new_set(&cx, four);
    s.insert_rows([some(&[4, 6]), some(&[6, 7]), some(&[8])]);
    assert_eq!(rows(&s), [vec![4], vec![6], vec![7], vec![8]].into());
}
