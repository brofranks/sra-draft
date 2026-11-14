//! Checks for predicate recipes, comparisons, membership, Boolean recipes,
//! and selectors.
//!
//! Every symbolic primitive is checked against a ground truth computed with
//! plain integers, over the whole 4-bit universe or explicit wide rows, never
//! against the engine's own encoding.

use std::collections::BTreeSet;
use std::ops::RangeInclusive;

use super::{self as pred, Build, Filter};
use crate::context::Context;
use crate::index;
use crate::query::{IntoQuery, Operand, Query};
use crate::rel::Rel;
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

/// A four-bit field with the selected backend.
fn field(backend: &str) -> (Context, Row) {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("v", cx.bv_sort(4))]);
    let backend = cx.backend(backend);
    cx.set_representation(t.accessors[0], backend);
    let row = t.accessors[0].bind(&mut cx);
    (cx, row)
}

/// The values of `s` kept by `p`, through the lazy restriction and one
/// collection.
fn kept(s: &Rel, p: impl Operand) -> BTreeSet<u64> {
    rows(&s.with(p).collect())
        .into_iter()
        .map(|r| r[0])
        .collect()
}

fn expect(f: impl Fn(u64) -> bool) -> BTreeSet<u64> {
    (0..16).filter(|&v| f(v)).collect()
}

/// Signed reading of a 4-bit pattern.
fn signed(v: u64) -> i64 {
    if v >= 8 { v as i64 - 16 } else { v as i64 }
}

// --- comparisons and ranges ----

#[test]
fn unsigned_comparisons_match_integer_order() {
    for backend in ["sparse", "bdd"] {
        let (cx, four) = field(backend);
        let s = set(&cx, &four, 0..16);
        for c in 0_u8..16 {
            let v = u64::from(c);
            assert_eq!(kept(&s, pred::bvult(c)), expect(|x| x < v), "bvult {c}");
            assert_eq!(kept(&s, pred::bvule(c)), expect(|x| x <= v), "bvule {c}");
            assert_eq!(kept(&s, pred::bvugt(c)), expect(|x| x > v), "bvugt {c}");
            assert_eq!(kept(&s, pred::bvuge(c)), expect(|x| x >= v), "bvuge {c}");
        }
    }
}

#[test]
fn signed_comparisons_read_the_sign_bit() {
    for backend in ["sparse", "bdd"] {
        let (cx, four) = field(backend);
        let s = set(&cx, &four, 0..16);
        // A positive and a negative bound select the two sign-bit branches of
        // the builder; the negative bound is spelled both as a pattern and as a
        // negative literal.
        for c in 0_u8..16 {
            let v = signed(u64::from(c));
            assert_eq!(
                kept(&s, pred::bvslt(c)),
                expect(|x| signed(x) < v),
                "bvslt {c}"
            );
            assert_eq!(
                kept(&s, pred::bvsle(c)),
                expect(|x| signed(x) <= v),
                "bvsle {c}"
            );
            assert_eq!(
                kept(&s, pred::bvsgt(c)),
                expect(|x| signed(x) > v),
                "bvsgt {c}"
            );
            assert_eq!(
                kept(&s, pred::bvsge(c)),
                expect(|x| signed(x) >= v),
                "bvsge {c}"
            );
        }
        assert_eq!(kept(&s, pred::bvslt(-3_i8)), expect(|x| signed(x) < -3));
    }
}

#[test]
fn comparisons_on_wide_columns_place_the_sign_bit_in_the_high_word() {
    fn kept(s: &Rel, all: &[[u64; 2]], p: impl Operand) -> BTreeSet<[u64; 2]> {
        let c = s.with(p).collect();
        all.iter().copied().filter(|r| c.contains(&r[..])).collect()
    }
    // Widths 65 and 70 place the sign bit at different offsets of the second
    // word.
    for backend in ["sparse", "bdd"] {
        for width in [65, 70] {
            let mut cx = Context::default();
            let t = cx.tuple_decl("Wide", [("v", cx.bv_sort(width))]);
            let b = cx.backend(backend);
            cx.set_representation(t.accessors[0], b);
            let v = t.accessors[0].bind(&mut cx);
            let sign = 1_u64 << (width - 65);
            let all = [[1, 0], [5, sign], [7, 0], [0, sign >> 1], [9, sign | 1]];
            let mut s = Rel::new_set(&cx, v);
            s.insert_rows(all.iter().map(|r| &r[..]));
            let value = |r: [u64; 2]| i128::from(r[0]) | i128::from(r[1]) << 64;
            let twos = |r: [u64; 2]| {
                let x = value(r);
                if x >> (width - 1) & 1 == 1 {
                    x - (1 << width)
                } else {
                    x
                }
            };
            let expect = |f: &dyn Fn([u64; 2]) -> bool| -> BTreeSet<[u64; 2]> {
                all.iter().copied().filter(|&r| f(r)).collect()
            };
            let (zero, bound) = ([0_u64, 0], [6_u64, sign]);
            assert_eq!(
                kept(&s, &all, pred::bvslt(&zero[..])),
                expect(&|r| twos(r) < 0)
            );
            assert_eq!(
                kept(&s, &all, pred::bvsge(&zero[..])),
                expect(&|r| twos(r) >= 0)
            );
            // A negative bound.
            assert_eq!(
                kept(&s, &all, pred::bvsle(&bound[..])),
                expect(&|r| twos(r) <= twos(bound))
            );
            assert_eq!(
                kept(&s, &all, pred::bvult(&[7_u64, 0][..])),
                expect(&|r| value(r) < 7)
            );
            assert_eq!(
                kept(&s, &all, pred::bvugt(&bound[..])),
                expect(&|r| value(r) > value(bound))
            );
        }
    }
}

#[test]
#[expect(
    clippy::needless_borrows_for_generic_args,
    reason = "the doubled borrow exercises the recipe impl for references"
)]
fn ranges_lower_to_unsigned_comparisons() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let eight = bv(&mut cx, 8);
    let three = bv(&mut cx, 3);
    let s = set(&cx, &four, 0..16);
    assert_eq!(kept(&s, 3_u8..7), expect(|x| (3..7).contains(&x)));
    assert_eq!(kept(&s, 3_u8..), expect(|x| x >= 3));
    assert_eq!(kept(&s, ..7_u8), expect(|x| x < 7));
    assert_eq!(kept(&s, ..=7_u8), expect(|x| x <= 7));
    assert_eq!(kept(&s, 3_u8..=7), expect(|x| (3..=7).contains(&x)));
    assert_eq!(kept(&s, 3_u8..=15), expect(|x| x >= 3));
    assert_eq!(kept(&s, ..), expect(|_| true));
    // Membership accepts the same ranges, borrowed recipes, and the reusable
    // filter.
    assert_eq!(
        kept(&s, pred::is_in(3_u8..7)),
        expect(|x| (3..7).contains(&x))
    );
    assert_eq!(kept(&s, pred::is_in(..)), expect(|_| true));
    let f = pred::is_in(3_u8..=7);
    assert_eq!(kept(&s, &f), expect(|x| (3..=7).contains(&x)));
    assert_eq!(kept(&s, &&f), expect(|x| (3..=7).contains(&x)));
    // An exhausted inclusive range reports an excluded end and keeps nothing,
    // including one iterated to its type's limit, which leaves `start()` and
    // `end()` unspecified.
    let mut r = 3_u8..=5;
    r.by_ref().for_each(drop);
    assert_eq!(kept(&s, r), BTreeSet::new());
    let all = set(&cx, &eight, 0..=255);
    let mut r = 250_u8..=255;
    while r.next().is_some() {}
    assert!(r.is_empty() && all.with(r).is_empty());
    let mut r = 0_u8..=3;
    while r.next_back().is_some() {}
    assert!(r.is_empty() && all.with(r).is_empty());
    // Exhausting a range that ends at a narrower field's maximum can leave a
    // start outside the field.
    let all = set(&cx, &three, 0..8);
    let mut r = 6_u64..=7;
    while r.next().is_some() {}
    assert!(r.is_empty() && all.with(r).is_empty());
    assert_eq!(all.with(6_u64..=7).count(), 2_u32.into());
    // A signed range drained to its zero crossing keeps nothing, although its
    // bounds encode as the whole field.
    let mut r = -4_i8..=-1;
    while r.next().is_some() {}
    assert!(r.is_empty() && all.with(r).is_empty());
    // Literal and borrowed bounds in those states keep nothing the same way.
    assert!(all.with(RangeInclusive::new(8_u64, 7)).is_empty());
    assert!(all.with(RangeInclusive::new(&8_u64, &7)).is_empty());
    assert!(all.with(RangeInclusive::new(1_i8, -1)).is_empty());
    assert!(all.with(RangeInclusive::new(&1_i8, &-1)).is_empty());
    assert_eq!(all.with(&2_u64..=&3).count(), 2_u32.into());
}

#[test]
fn inclusive_word_ranges_follow_the_fields_unsigned_order() {
    let mut cx = Context::default();
    let wide = bv(&mut cx, 128);
    let mut s = Rel::new_set(&cx, wide);
    s.insert_rows([[7_u64, 0], [0, 1], [9, 0], [1, 1]]);
    // Little-endian words: 5 ..= 2^64 keeps 7, 9, and 2^64, while the array's
    // own order would call this range empty.
    assert_eq!(s.with([5_u64, 0]..=[0_u64, 1]).count(), 3_u32.into());
}

#[test]
#[should_panic(expected = "operation requires a bit-vector sort")]
fn full_interval_over_a_nonbitvector_is_rejected() {
    let mut cx = Context::default();
    let b = cx.bool_sort().bind(&mut cx);
    let mut r = Rel::new_set(&cx, b);
    r.insert(true);
    r.with(..).collect();
}

// --- equality and membership ----

#[test]
fn equality_and_finite_membership_take_scalars_and_rows() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let s = set(&cx, &four, 0..16);
    assert_eq!(kept(&s, pred::eq(5_u8)), expect(|x| x == 5));
    assert_eq!(kept(&s, pred::ne(5_u8)), expect(|x| x != 5));
    // Every finite collection form, scalar values in a scalar row.
    assert_eq!(
        kept(&s, pred::is_in([1_u8, 3])),
        expect(|x| x == 1 || x == 3)
    );
    assert_eq!(
        kept(&s, pred::is_in(vec![1_u8, 3, 3])),
        expect(|x| x == 1 || x == 3)
    );
    assert_eq!(
        kept(&s, pred::is_in(&[1_u8, 3][..])),
        expect(|x| x == 1 || x == 3)
    );
    assert_eq!(
        kept(&s, pred::is_in(&[1_u8, 3])),
        expect(|x| x == 1 || x == 3)
    );
    assert_eq!(
        kept(&s, pred::is_in(&vec![1_u8, 3])),
        expect(|x| x == 1 || x == 3)
    );
    assert_eq!(kept(&s, pred::is_in([0_u8; 0])), BTreeSet::new());
    // Aggregate rows: equality and membership over complete pairs.
    let mut r = Rel::new_set(&cx, cx.product_sort([four.sort, four.sort]));
    r.insert_rows([(1_u8, 2_u8), (3, 4), (5, 6)]);
    assert_eq!(
        pairs(&r.with(pred::eq((3_u8, 4_u8))).collect()),
        [(3, 4)].into()
    );
    assert_eq!(
        pairs(&r.with(pred::ne((3_u8, 4_u8))).collect()),
        [(1, 2), (5, 6)].into()
    );
    assert_eq!(
        pairs(
            &r.with(pred::is_in([(1_u8, 2_u8), (5, 6), (7, 8)]))
                .collect()
        ),
        [(1, 2), (5, 6)].into()
    );
    assert!(r.with(pred::is_in([(0_u8, 0_u8); 0])).is_empty());
}

#[test]
fn aggregate_equality_constructs_one_node_per_physical_level() {
    use crate::backend::Function;

    struct Zeroes(usize);

    impl crate::row::IntoRow for Zeroes {
        fn append(&self, _: crate::row::RowRef<'_>, out: &mut impl crate::row::RowSink) {
            out.words().extend(std::iter::repeat_n(0, self.0));
        }
    }

    for backend in ["bdd", "sparse"] {
        let (cx, scalar) = field(backend);
        let count = 16;
        let sort = cx.product_sort(std::iter::repeat_n(scalar.sort, count));
        let row = Row::new(
            &cx,
            sort,
            vec![row::scalar_column(&scalar); count].into(),
            0,
        );
        let index = index::intern(&cx, row::columns(&row));
        let root = pred::eq(Zeroes(count)).build(&row, &index, &(0..count).collect::<Vec<_>>());
        assert!(root.contains(&vec![0; count]));
        let mut other = vec![0; count];
        other[count - 1] = 1;
        assert!(!root.contains(&other));
        let data = cx
            .layouts
            .lock()
            .unwrap()
            .iter()
            .find(|data| data.columns.as_ref() == row::columns(&row))
            .unwrap()
            .clone();
        assert_eq!(data.shared.epoch(), 0);
        let state = data.shared.read((&cx, &data.columns));
        let nodes = (2..state.node_capacity())
            .filter(|&i| state.get(Function::new(i)).is_some())
            .count();
        assert_eq!(
            nodes, count,
            "one constructed native node per constrained level"
        );
    }
}

#[test]
fn equality_and_finite_membership_take_boolean_literals() {
    for backend in ["sparse", "bdd"] {
        let mut cx = Context::default();
        let t = cx.tuple_decl("B", [("b", cx.bool_sort())]);
        let b = cx.backend(backend);
        cx.set_representation(t.accessors[0], b);
        let flag = t.accessors[0].bind(&mut cx);
        let mut s = Rel::new_set(&cx, flag);
        s.insert_rows([false, true]);
        // Membership of `false` and `true` in a collected result.
        let holds = |r: Rel| [false, true].map(|v| r.contains(v));
        assert_eq!(holds(s.with(pred::eq(true)).collect()), [false, true]);
        assert_eq!(holds(s.with(pred::ne(true)).collect()), [true, false]);
        assert_eq!(holds(s.with(pred::is_in([false])).collect()), [true, false]);
        assert_eq!(
            holds(s.with(pred::is_in(vec![true, false, true])).collect()),
            [true, true]
        );
        assert_eq!(
            holds(s.with(pred::is_in([true; 0])).collect()),
            [false, false]
        );
        assert_eq!(
            holds(s.with(pred::eq(true) | pred::eq(false)).collect()),
            [true, true]
        );
        assert_eq!(holds(s.with(!pred::eq(false)).collect()), [false, true]);
    }
}

// --- boolean recipes ----

#[test]
fn borrowed_conditions_reuse_borrowed_range_leaves() {
    for backend in ["bdd", "sparse"] {
        let (cx, row) = field(backend);
        let s = set(&cx, &row, 0..16);
        macro_rules! check_range {
            ($range:expr, $contains:expr) => {{
                let range = $range;
                let contains = $contains;
                let disj = pred::eq(9_u8) | &range;
                let conj = pred::bvult(4_u8) & &range;
                let nested = &disj & &conj;
                let neg = !&disj;
                for _ in 0..2 {
                    assert_eq!(kept(&s, &disj), expect(|x| x == 9 || contains(x)));
                    assert_eq!(kept(&s, &conj), expect(|x| x < 4 && contains(x)));
                    assert_eq!(kept(&s, &nested), expect(|x| x < 4 && contains(x)));
                    assert_eq!(kept(&s, &neg), expect(|x| x != 9 && !contains(x)));
                }
            }};
        }
        check_range!(2_u8..5, |x| (2..5).contains(&x));
        check_range!(2_u8.., |x| x >= 2);
        check_range!(2_u8..=4, |x| (2..=4).contains(&x));
        check_range!(..5_u8, |x| x < 5);
        check_range!(..=4_u8, |x| x <= 4);
        check_range!(.., |_| true);
    }
}

#[test]
#[expect(
    clippy::needless_borrows_for_generic_args,
    reason = "the doubled borrow exercises the recipe impl for references"
)]
fn boolean_recipes_compose_natively_and_lazily() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let s = set(&cx, &four, 0..16);
    let (a, b) = (pred::bvult(6_u8), pred::is_in([2_u8, 9, 12]));
    let in_a = |x: u64| x < 6;
    let in_b = |x: u64| [2, 9, 12].contains(&x);
    // Lazy Boolean operands resolve through query combinators, owned and
    // borrowed.
    assert_eq!(kept(&s, &a & &b), expect(|x| in_a(x) && in_b(x)));
    assert_eq!(kept(&s, &a | &b), expect(|x| in_a(x) || in_b(x)));
    assert_eq!(kept(&s, !&a), expect(|x| !in_a(x)));
    assert_eq!(kept(&s, a.clone().and(&b)), expect(|x| in_a(x) && in_b(x)));
    assert_eq!(kept(&s, a.clone().or(&b)), expect(|x| in_a(x) || in_b(x)));
    assert_eq!(kept(&s, !a.clone()), expect(|x| !in_a(x)));
    let conj = &a & &b;
    let disj = &a | &b;
    let neg = !&a;
    assert_eq!(kept(&s, &conj), expect(|x| in_a(x) && in_b(x)));
    assert_eq!(kept(&s, &&conj), expect(|x| in_a(x) && in_b(x)));
    assert_eq!(kept(&s, &disj), expect(|x| in_a(x) || in_b(x)));
    assert_eq!(kept(&s, &&disj), expect(|x| in_a(x) || in_b(x)));
    assert_eq!(kept(&s, &neg), expect(|x| !in_a(x)));
    assert_eq!(kept(&s, &&neg), expect(|x| !in_a(x)));
    // Method and operator forms on every recipe shape, owned and borrowed.
    assert_eq!(
        kept(&s, conj.clone().or(&b)),
        expect(|x| in_a(x) && in_b(x) || in_b(x))
    );
    assert_eq!(
        kept(&s, conj.clone().and(!&b)),
        expect(|x| in_a(x) && in_b(x) && !in_b(x))
    );
    assert_eq!(
        kept(&s, disj.clone().and(!&b)),
        expect(|x| (in_a(x) || in_b(x)) && !in_b(x))
    );
    assert_eq!(kept(&s, disj.clone().or(!&a)), expect(|_| true));
    assert_eq!(
        kept(&s, neg.clone().and(&b)),
        expect(|x| !in_a(x) && in_b(x))
    );
    assert_eq!(
        kept(&s, neg.clone().or(&b)),
        expect(|x| !in_a(x) || in_b(x))
    );
    assert_eq!(
        kept(&s, &conj | &neg),
        expect(|x| in_a(x) && in_b(x) || !in_a(x))
    );
    assert_eq!(
        kept(&s, &disj & &neg),
        expect(|x| (in_a(x) || in_b(x)) && !in_a(x))
    );
    assert_eq!(kept(&s, &neg & &neg), expect(|x| !in_a(x)));
    assert_eq!(
        kept(&s, &neg | &conj),
        expect(|x| !in_a(x) || in_a(x) && in_b(x))
    );
    assert_eq!(kept(&s, !&conj), expect(|x| !(in_a(x) && in_b(x))));
    assert_eq!(kept(&s, !&disj), expect(|x| !(in_a(x) || in_b(x))));
    assert_eq!(kept(&s, !&neg), expect(in_a));
    assert_eq!(kept(&s, !conj.clone()), expect(|x| !(in_a(x) && in_b(x))));
    assert_eq!(kept(&s, !disj.clone()), expect(|x| !(in_a(x) || in_b(x))));
    assert_eq!(kept(&s, !neg.clone()), expect(in_a));
    // Wrapping a Boolean recipe in a filter builds it natively in one index.
    assert_eq!(kept(&s, Filter(&a & &b)), expect(|x| in_a(x) && in_b(x)));
    assert_eq!(kept(&s, Filter(&a | &b)), expect(|x| in_a(x) || in_b(x)));
    assert_eq!(kept(&s, Filter(!&a)), expect(|x| !in_a(x)));
    assert_eq!(
        kept(&s, Filter(!(&a | &b) & pred::bvuge(1_u8))),
        expect(|x| !(in_a(x) || in_b(x)) && x >= 1)
    );
    // Query operands take part in Boolean recipes: `or` on a stored set.
    let t = set(&cx, &four, [9, 10]);
    assert_eq!(
        kept(&s, (&t).or(&a)),
        expect(|x| x == 9 || x == 10 || in_a(x))
    );
    assert_eq!(
        kept(&s, (&t).or(&a).and(pred::ne(9_u8))),
        expect(|x| (x == 10 || in_a(x)) && x != 9)
    );
}

// --- selectors ----

#[test]
fn selectors_lift_member_predicates_to_the_record() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    let trow = t.sort.bind(&mut cx);
    let (x, y) = (t.accessors[0], t.accessors[1]);
    let mut r = Rel::new_set(&cx, trow.clone());
    let all: Vec<(u8, u8)> = (0..16).flat_map(|a| (0..16).map(move |b| (a, b))).collect();
    r.insert_rows(all.iter().copied());
    fn kept(r: &Rel, p: impl Operand) -> BTreeSet<(u64, u64)> {
        pairs(&r.with(p).collect())
    }
    let expect = |f: &dyn Fn(u64, u64) -> bool| -> BTreeSet<(u64, u64)> {
        all.iter()
            .map(|&(a, b)| (u64::from(a), u64::from(b)))
            .filter(|&(a, b)| f(a, b))
            .collect()
    };
    assert_eq!(kept(&r, x.eq(3_u8)), expect(&|a, _| a == 3));
    assert_eq!(kept(&r, y.ne(3_u8)), expect(&|_, b| b != 3));
    assert_eq!(kept(&r, x.bvult(3_u8)), expect(&|a, _| a < 3));
    assert_eq!(kept(&r, x.bvule(3_u8)), expect(&|a, _| a <= 3));
    assert_eq!(kept(&r, y.bvugt(12_u8)), expect(&|_, b| b > 12));
    assert_eq!(kept(&r, y.bvuge(12_u8)), expect(&|_, b| b >= 12));
    assert_eq!(kept(&r, x.bvslt(-2_i8)), expect(&|a, _| signed(a) < -2));
    assert_eq!(kept(&r, x.bvsle(-2_i8)), expect(&|a, _| signed(a) <= -2));
    assert_eq!(kept(&r, y.bvsgt(6_i8)), expect(&|_, b| signed(b) > 6));
    assert_eq!(kept(&r, y.bvsge(6_i8)), expect(&|_, b| signed(b) >= 6));
    assert_eq!(
        kept(&r, y.is_in([1_u8, 2])),
        expect(&|_, b| b == 1 || b == 2)
    );
    assert_eq!(kept(&r, x.is_in(4_u8..6)), expect(&|a, _| a == 4 || a == 5));
    assert_eq!(
        kept(&r, x.test(pred::eq(1_u8) | pred::eq(2_u8))),
        expect(&|a, _| a == 1 || a == 2)
    );
    // A projection query is a selector too; the whole-row constructor
    // selects every column.
    let xy = Query::decl(&cx, x).and(y);
    assert_eq!(
        kept(&r, xy.eq((1_u8, 2_u8))),
        expect(&|a, b| (a, b) == (1, 2))
    );
    assert_eq!(
        kept(&r, xy.is_in([(1_u8, 2_u8), (3, 4)])),
        expect(&|a, b| (a, b) == (1, 2) || (a, b) == (3, 4))
    );
    assert_eq!(
        kept(&r, Query::decl(&cx, y).and(x).ne((2_u8, 8_u8))),
        expect(&|a, b| (b, a) != (2, 8))
    );
    let mut args = Rel::new_set(&cx, cx.product_sort([four.sort, four.sort]));
    args.insert_rows([(1_u8, 2_u8), (3, 4)]);
    assert_eq!(
        pairs(&args.with(t.constructor.eq((3_u8, 4_u8))).collect()),
        [(3, 4)].into()
    );
    assert_eq!(
        pairs(
            &args
                .with(t.constructor.test(pred::ne((3_u8, 4_u8))))
                .collect()
        ),
        [(1, 2)].into()
    );
    // Selectors on symbolic rows keep every witness.
    let mut sym = Rel::new_set(&cx, trow.clone());
    sym.insert((&cx.set(row::columns(&trow)[0], &[1, 2]), 7_u8));
    assert_eq!(pairs(&sym.with(x.bvuge(2_u8)).collect()), [(2, 7)].into());
}

#[test]
#[should_panic(expected = "predicate selector requires a compatible input sort")]
fn selector_with_an_incompatible_input_sort_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    t.sort.bind(&mut cx);
    let u = cx.tuple_decl("Q", [("z", four.sort)]);
    let urow = u.sort.bind(&mut cx);
    Rel::new_set(&cx, urow)
        .with(t.accessors[0].eq(1_u8))
        .collect();
}

#[test]
#[should_panic(expected = "predicate selector belongs to another context")]
fn selector_from_another_context_is_rejected() {
    let f = {
        let mut other = Context::default();
        let four = bv(&mut other, 4);
        let t = other.tuple_decl("P", [("x", four.sort)]);
        t.sort.bind(&mut other);
        Query::decl(&other, t.accessors[0]).eq(1_u8)
    };
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort)]);
    let trow = t.sort.bind(&mut cx);
    Rel::new_set(&cx, trow).with(f).collect();
}

#[test]
#[should_panic(expected = "a selector must be a source-free projection")]
fn selector_reading_a_source_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let s = set(&cx, &four, 0..16);
    s.into_query().test(pred::eq(1_u8));
}

#[test]
#[should_panic(expected = "a selector must project its input")]
fn selector_without_the_complete_input_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // Inverting the second accessor leaves the record's first cell outside
    // the input, so the inverse cannot project that output coordinate even
    // though nothing reads a source.
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    t.sort.bind(&mut cx);
    Query::decl(&cx, t.accessors[1]).inv().test(pred::eq(1_u8));
}

#[test]
#[should_panic(expected = "a selector must project its input")]
fn selector_exposing_hidden_positions_is_rejected() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    t.sort.bind(&mut cx);
    Query::decl(&cx, t.accessors[0]).inv().test(pred::eq(1_u8));
}

#[test]
fn filters_resolve_with_fixed_read_slots() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // A resolved filter reads and fixes exactly the constrained positions.
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    let trow = t.sort.bind(&mut cx);
    let q = Query::id(&cx, trow).with(t.accessors[1].eq(2_u8));
    let (x, y) = (&q.slots[0], &q.slots[1]);
    assert!(!x.read && !x.fixed && y.read && y.fixed);
    assert_eq!(q.count(), 16_u32.into());
}

#[test]
fn constant_filters_and_zero_field_sources_are_rejected_as_selectors() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let empty = cx.tuple_decl("Empty", [] as [(&str, crate::ast::Sort); 0]);
    let row = empty.sort.bind(&mut cx);
    for values in [vec![], (0_u8..16).collect()] {
        let q = Query::id(&cx, four.clone()).with(pred::is_in(values));
        // Both constants leave every slot unread. Neither is a structural
        // selector, even when its current contents happen to be full.
        assert!(q.slots.iter().all(|slot| !slot.read));
        assert!(catch_unwind(AssertUnwindSafe(|| q.positions())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| q.test(pred::eq(3_u8)))).is_err());
    }
    let mut s = Rel::new_set(&cx, row);
    for populated in [false, true] {
        if populated {
            s.insert(());
        }
        let q = s.into_query();
        assert!(q.slots.is_empty());
        assert!(catch_unwind(AssertUnwindSafe(|| q.test(pred::eq(())))).is_err());
    }
}

#[test]
fn composed_and_borrowed_structural_queries_remain_selectors() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort), ("y", four.sort)]);
    let row = t.sort.bind(&mut cx);
    let q = Query::decl(&cx, t.accessors[1]).and(t.accessors[0]);
    let projection = (&q).select(crate::query::id());
    let p = projection.eq((2_u8, 1_u8));
    let mut r = Rel::new_set(&cx, row);
    r.insert_rows([(1_u8, 2_u8), (2, 1)]);
    assert_eq!(pairs(&r.with(p).collect()), [(1, 2)].into());
    let pass = crate::query::transform(&cx, &four, None);
    let s = set(&cx, &four, 0..16);
    assert_eq!(kept(&s, pass.eq(3_u8)), [3].into());
}

// --- direct builders ----

#[test]
fn direct_builders_reject_incomplete_or_incompatible_mappings() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let t = cx.tuple_decl("P", [("x", four.sort)]);
    let backend = cx.backend("bdd");
    let column = cx.set_representation(t.accessors[0], backend);
    let index = index::intern(&cx, row::columns(&four));
    let other = index::intern(&cx, &[column]);
    let mut foreign = Context::default();
    let foreign_row = bv(&mut foreign, 4);
    let foreign_index = index::intern(&foreign, row::columns(&foreign_row));
    let recipes: Vec<Box<dyn Build>> = vec![
        Box::new(pred::eq(2_u8)),
        Box::new(pred::ne(2_u8)),
        Box::new(pred::bvult(0_u8)),
        Box::new(pred::bvule(15_u8)),
        Box::new(pred::is_in(Vec::<u8>::new())),
        Box::new(pred::is_in((0_u8..16).collect::<Vec<_>>())),
        Box::new(pred::is_in(..)),
        Box::new(1_u8..3),
        Box::new(pred::eq(2_u8) & pred::ne(3_u8)),
        Box::new(pred::eq(2_u8) | pred::eq(3_u8)),
        Box::new(!pred::eq(2_u8)),
        Box::new(t.accessors[0].eq(2_u8)),
    ];
    for p in recipes {
        // Every recipe is valid on the complete mapping, including a
        // declaration whose own physical column differs from this input.
        p.build(&four, &index, &[0]);
        for positions in [&[][..], &[0, 0][..], &[1][..]] {
            assert!(catch_unwind(AssertUnwindSafe(|| p.build(&four, &index, positions))).is_err());
        }
        for target in [&other, &foreign_index] {
            assert!(catch_unwind(AssertUnwindSafe(|| p.build(&four, target, &[0]))).is_err());
        }
    }
}

#[test]
fn direct_builders_preserve_repeated_and_reordered_positions() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let six = bv(&mut cx, 6);
    let pair = cx.product_sort([four.sort, four.sort]).bind(&mut cx);
    let reversed = cx.product_sort([six.sort, four.sort]).bind(&mut cx);
    let single = index::intern(&cx, row::columns(&four));
    let same = pred::eq((2_u8, 2_u8)).build(&pair, &single, &[0, 0]);
    let different = pred::eq((2_u8, 3_u8)).build(&pair, &single, &[0, 0]);
    let outside = pred::ne((2_u8, 2_u8)).build(&pair, &single, &[0, 0]);
    for x in 0..16 {
        assert_eq!(same.contains(&[x]), x == 2);
        assert!(!different.contains(&[x]));
        assert_eq!(outside.contains(&[x]), x != 2);
    }
    let index = index::intern(&cx, &[row::columns(&four)[0], row::columns(&six)[0]]);
    let p = pred::eq((17_u8, 3_u8)).build(&reversed, &index, &[1, 0]);
    for x in 0..16 {
        for y in 0..64 {
            assert_eq!(p.contains(&[x, y]), x == 3 && y == 17);
        }
    }
}

// --- declaration selectors and composition ----

#[test]
fn declaration_selectors_use_consuming_columns_without_other_bindings() {
    for arguments in [false, true] {
        let mut cx = Context::default();
        let four = cx.bv_sort(4);
        let t = cx.tuple_decl("P", [("x", four), ("y", four)]);
        let backend = cx.backend("bdd");
        cx.set_representation(t.accessors[0], backend);
        // The argument row binds only the anonymous scalar, leaving y
        // unbound. The record row binds only the named fields, leaving the
        // anonymous scalar unbound. In either case the consumer supplies all
        // storage needed by both accessors and the constructor.
        let row = if arguments {
            cx.product_sort([four, four]).bind(&mut cx)
        } else {
            t.sort.bind(&mut cx)
        };
        let mut r = Rel::new_set(&cx, row);
        r.insert_rows([(1_u8, 2_u8), (3, 4), (1, 4)]);
        assert_eq!(
            pairs(&r.with(t.accessors[0].eq(1_u8)).collect()),
            [(1, 2), (1, 4)].into()
        );
        assert_eq!(
            pairs(&r.with(t.accessors[1].eq(4_u8)).collect()),
            [(3, 4), (1, 4)].into()
        );
        assert_eq!(
            pairs(&r.with(t.constructor.eq((1_u8, 2_u8))).collect()),
            [(1, 2)].into()
        );
    }
}

#[test]
fn a_projection_recipe_does_not_keep_the_context_shared() {
    let mut cx = Context::default();
    let four = cx.bv_sort(4);
    let t = cx.tuple_decl("P", [("x", four)]);
    let row = t.sort.bind(&mut cx);
    let p = Query::decl(&cx, t.accessors[0]).eq(3_u8);
    // The query temporary is gone. Keeping its selector recipe must still
    // allow configuring and binding another field before execution.
    let u = cx.tuple_decl("Q", [("y", four)]);
    let backend = cx.backend("bdd");
    cx.set_representation(u.accessors[0], backend);
    u.sort.bind(&mut cx);
    let mut r = Rel::new_set(&cx, row);
    r.insert_rows([(1_u8,), (3_u8,)]);
    assert_eq!(kept(&r, p), [3].into());
}

#[test]
fn membership_booleans_keep_independent_witnesses_and_symbolic_coverage() {
    for backend in ["sparse", "bdd"] {
        let (mut cx, four) = field(backend);
        let flag = cx.bool_sort().bind(&mut cx);
        let column = row::columns(&four)[0];
        let mut a = Rel::new(&cx, four.clone(), four.clone());
        let mut b = Rel::new(&cx, four.clone(), flag);
        a.insert((&cx.set(column, &[0, 1, 2]), 7_u8));
        a.insert((1_u8, 8_u8));
        b.insert((&cx.set(column, &[2, 3]), true));
        let mut c = Rel::new(&cx, four.clone(), four.clone());
        c.insert((2_u8, 9_u8));
        let s = set(&cx, &four, 0..16);
        assert_eq!(kept(&s, pred::Conj(&a, &c)), [2].into());
        assert_eq!(kept(&s, a.or(&b)), [0, 1, 2, 3].into());
        assert_eq!(kept(&s, pred::Conj(&a, &b)), [2].into());
        assert_eq!(kept(&s, !a.or(&b)), (4..16).collect());
        assert_eq!(kept(&s, a.or(pred::eq(3_u8))), [0, 1, 2, 3].into());
    }
}

#[test]
fn selector_recipes_coexist_with_semantic_query_comparisons() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let selector = Query::id(&cx, four.clone());
    let query = (&selector).with(selector.eq(3_u8));
    let expected = set(&cx, &four, [3]);
    let other = set(&cx, &four, [4]);
    assert!(query == expected);
    assert!(query != other);
    assert!(PartialEq::eq(&query, &expected));
    assert!(!PartialEq::ne(&query, &expected));
    assert!(PartialEq::ne(&query, &other));
    let outside = (&selector).with(selector.ne(3_u8)).collect();
    for value in 0..16_u8 {
        assert_eq!(outside.contains(value), value != 3);
    }
}
