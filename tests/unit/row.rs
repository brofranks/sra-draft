//! Checks for row binding, literal packing, and argument conversion.
//!
//! `bv` binds a bit-vector row of a given width, and `words` packs ground
//! arguments into a plain word vector. The `Symbolic` sink reserves each
//! predicate's words and records its offset, and `symbolic` packs arguments
//! through it, returning the words and predicate offsets.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::ast::Sort;
use crate::context::{Context, Predicate};
use crate::row::{self, Domain, IntoRow, Row, RowRef, RowSink};

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.bv_sort(w);
    sort.bind(cx)
}

/// The ground words `args` pack into for `row`.
fn words(cx: &Context, row: &Row, args: impl IntoRow) -> Vec<u64> {
    let mut out = Vec::new();
    row::append(cx, args, row, &mut out);
    out
}

/// A sink that reserves each predicate's span and retains it.
#[derive(Default)]
struct Symbolic {
    words: Vec<u64>,
    predicates: Vec<(usize, Predicate)>,
}

impl RowSink for Symbolic {
    fn words(&mut self) -> &mut Vec<u64> {
        &mut self.words
    }

    fn predicates(&self) -> &[(usize, Predicate)] {
        &self.predicates
    }

    fn predicate(&mut self, p: &Predicate) {
        self.predicates.push((self.words.len(), p.clone()));
        let n = p.context().resolve(p.column()).sort().words();
        self.words.resize(self.words.len() + n, 0);
    }
}

/// The words and predicate offsets `args` pack into for `row`.
fn symbolic(cx: &Context, row: &Row, args: impl IntoRow) -> (Vec<u64>, Vec<usize>) {
    let mut out = Symbolic::default();
    row::append(cx, args, row, &mut out);
    let offsets = out.predicates.iter().map(|(i, _)| *i).collect();
    (out.words, offsets)
}

// --- row layout -------------------------------------------------------------

#[test]
#[should_panic(expected = "row column must match its logical scalar sort")]
fn a_row_layout_with_a_wider_scalar_is_rejected() {
    let mut cx = Context::default();
    let wide = bv(&mut cx, 8);
    Row::new(&cx, cx.bv_sort(4), wide.layout, 0);
}

#[test]
#[should_panic(expected = "row column must match its logical scalar sort")]
fn a_row_layout_with_a_different_scalar_kind_is_rejected() {
    let mut cx = Context::default();
    let bits = bv(&mut cx, 1);
    Row::new(&cx, cx.bool_sort(), bits.layout, 0);
}

#[test]
#[should_panic(expected = "row column must match its logical scalar sort")]
fn a_row_layout_with_a_narrower_aggregate_member_is_rejected() {
    let mut cx = Context::default();
    let narrow = bv(&mut cx, 4);
    let sort = cx.product_sort([narrow.sort, cx.bv_sort(8)]);
    let column = narrow.layout[0];
    Row::new(&cx, sort, vec![column, column].into(), 0);
}

#[test]
#[should_panic(expected = "row layout must cover the logical sort")]
fn a_row_layout_with_too_few_columns_is_rejected() {
    let cx = Context::default();
    Row::new(&cx, cx.bv_sort(4), Vec::new().into(), 0);
}

#[test]
fn row_layout_checks_preserve_backends_offsets_aliases_and_empty_records() {
    let mut cx = Context::default();
    let sort = cx.bv_sort(4);
    let bdd = cx.backend("bdd");
    let column = cx.column(bdd, sort.scalar());
    let wide = bv(&mut cx, 8);
    let layout: Arc<[_]> = vec![wide.layout[0], column, column].into();
    let pair = Row::new(&cx, cx.product_sort([sort, sort]), layout.clone(), 1);
    assert_eq!(row::columns(&pair), [column, column]);
    assert_eq!(words(&cx, &pair, (3_u8, 7_u8)), [3, 7]);
    let empty = cx.tuple_decl("Empty", Vec::<(&str, Sort)>::new()).sort;
    let nested = cx.tuple_decl("Nested", [("empty", empty)]).sort;
    let end = layout.len();
    let r = Row::new(&cx, nested, layout, end);
    assert!(row::columns(&r).is_empty());
    assert_eq!(r.members.as_ref(), [empty]);
}

// --- rows, binding, and coercion --------------------------------------------

#[test]
fn declarations_bind_rows() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.accessors[1], bdd);
    // A sort, an accessor, a constructor, and an existing row all bind.
    let whole = t.sort.bind(&mut cx);
    assert_eq!(whole.sort, t.sort);
    assert_eq!(row::columns(&whole).len(), 2);
    assert_eq!(whole.members.as_ref(), cx.members(t.sort).as_slice());
    let b = t.accessors[1].bind(&mut cx);
    assert_eq!(b.sort, cx.bv_sort(8));
    assert_eq!(row::columns(&b)[0], row::columns(&whole)[1]);
    assert_eq!(
        row::scalar_column(&b),
        cx.column(bdd, crate::scalar::ScalarSort::bit_vec(8))
    );
    let made = t.constructor.bind(&mut cx);
    assert_eq!(made.sort, t.sort);
    assert_eq!(row::columns(&made), row::columns(&whole));
    let again = whole.clone().bind(&mut cx);
    assert_eq!(row::columns(&again), row::columns(&whole));
    // Positional projection of a product binds the sparse default.
    let p = cx.product_sort([cx.bv_sort(4), cx.bv_sort(8)]);
    let second = cx.project(p, 1).bind(&mut cx);
    let sparse = cx.backend("sparse");
    assert_eq!(
        row::scalar_column(&second),
        cx.column(sparse, crate::scalar::ScalarSort::bit_vec(8))
    );
    // The immediate children of an aggregate row share its layout.
    let parts: Vec<Row> = row::children(&cx, &whole).collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(row::columns(&parts[1]), row::columns(&b));
    assert_eq!(parts[1].view(&cx).columns, row::columns(&b));
    // A product of two rows concatenates their columns.
    let both = row::product(&cx, [&parts[0], &parts[1]]);
    assert_eq!(row::columns(&both), row::columns(&whole));
    assert_eq!(both.sort, p);
}

#[test]
fn rows_look_up_bound_leaves() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    let p = cx.product_sort([cx.bv_sort(4), cx.bv_sort(8)]);
    let whole = t.sort.bind(&mut cx);
    let second = cx.project(p, 1).bind(&mut cx);
    // Lookup works once the context is shared and finds the bound columns.
    let shared = cx.clone();
    assert_eq!(row::columns(&t.sort.row(&shared)), row::columns(&whole));
    assert_eq!(
        row::columns(&t.accessors[1].row(&shared)),
        &row::columns(&whole)[1..]
    );
    assert_eq!(row::columns(&t.constructor.row(&cx)), row::columns(&whole));
    assert_eq!(
        row::columns(&cx.project(p, 1).row(&cx)),
        row::columns(&second)
    );
    assert_eq!(row::columns(&whole.clone().row(&cx)), row::columns(&whole));
}

#[test]
#[should_panic(expected = "field must be bound")]
fn a_row_over_an_unbound_member_is_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    t.accessors[0].bind(&mut cx);
    t.sort.row(&cx);
}

#[test]
fn coercion_switches_record_wrappers() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    let p = cx.product_sort([cx.bv_sort(4), cx.bv_sort(8)]);
    let mut r = t.sort.bind(&mut cx);
    row::coerce(&cx, &mut r, p);
    assert_eq!(r.sort, p);
    assert_eq!(r.members.len(), 2);
    row::coerce(&cx, &mut r, t.sort);
    assert_eq!(r.sort, t.sort);
}

#[test]
#[should_panic(expected = "incompatible logical sorts")]
fn an_incompatible_coercion_is_rejected() {
    let mut cx = Context::default();
    let mut r = bv(&mut cx, 4);
    row::coerce(&cx, &mut r, cx.bv_sort(8));
}

#[test]
#[should_panic(expected = "aggregate arguments require a record or product")]
fn children_of_a_scalar_are_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    row::children(&cx, &r).count();
}

#[test]
#[should_panic(expected = "operation requires a bit-vector sort")]
fn a_boolean_bit_vector_column_is_rejected() {
    let mut cx = Context::default();
    row::bitvector_column(&cx.bool_sort().bind(&mut cx));
}

#[test]
#[should_panic(expected = "expected a scalar sort")]
fn a_scalar_column_of_an_aggregate_is_rejected() {
    let mut cx = Context::default();
    let p = cx.product_sort([cx.bv_sort(4), cx.bv_sort(4)]);
    row::scalar_column(&p.bind(&mut cx));
}

#[test]
#[should_panic(expected = "sort belongs to another context")]
fn a_sort_of_another_context_is_rejected() {
    let sort = Context::default().bv_sort(4);
    sort.bind(&mut Context::default());
}

#[test]
#[should_panic(expected = "row belongs to another context")]
fn a_row_of_another_context_is_rejected() {
    let r = bv(&mut Context::default(), 4);
    r.bind(&mut Context::default());
}

#[test]
#[should_panic(expected = "row belongs to another context")]
fn a_product_of_another_context_is_rejected() {
    let r = bv(&mut Context::default(), 4);
    row::product(&Context::default(), [&r, &r]);
}

// --- custom adapters are checked at the boundary ----------------------------

/// Appends whatever words it is given, so the boundary checks see malformed
/// output.
struct Raw(Vec<u64>);

impl IntoRow for Raw {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().extend_from_slice(&self.0);
    }
}

#[test]
#[should_panic(expected = "row arguments have too few words")]
fn an_adapter_short_of_words_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![]));
}

#[test]
#[should_panic(expected = "row arguments have too many words")]
fn an_adapter_with_extra_words_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![1, 2]));
}

#[test]
#[should_panic(expected = "row value does not encode its scalar sort")]
fn adapter_words_outside_the_sort_are_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![16]));
}

/// Emits a predicate in the middle of a two-word scalar.
struct Misplaced(Predicate);

impl IntoRow for Misplaced {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().push(0);
        out.predicate(&self.0);
        out.words().pop();
    }
}

#[test]
#[should_panic(expected = "predicate must start at a scalar boundary")]
fn a_predicate_inside_a_scalar_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 70);
    let p = cx.set(row::scalar_column(&r), &[1, 0]);
    symbolic(&cx, &r, Misplaced(p));
}

/// Emits a ground word, then a predicate past the end of the row.
struct Dangling(Predicate);

impl IntoRow for Dangling {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().push(1);
        let n = out.words().len();
        out.predicate(&self.0);
        out.words().truncate(n);
    }
}

#[test]
#[should_panic(expected = "row arguments have misplaced predicates")]
fn a_predicate_past_the_row_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    let p = cx.set(row::scalar_column(&r), &[1]);
    symbolic(&cx, &r, Dangling(p));
}

/// Hands a predicate straight to the sink, bypassing the predicate adapter's
/// own checks.
struct Direct(Predicate);

impl IntoRow for Direct {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.predicate(&self.0);
    }
}

#[test]
#[should_panic(expected = "predicate uses an incompatible column")]
fn a_direct_predicate_over_another_column_is_rejected() {
    let mut cx = Context::default();
    let c = row::scalar_column(&bv(&mut cx, 8));
    let r = bv(&mut cx, 4);
    let p = cx.set(c, &[1]);
    symbolic(&cx, &r, Direct(p));
}

#[test]
#[should_panic(expected = "predicate belongs to another context")]
fn a_direct_predicate_of_another_context_is_rejected() {
    let mut other = Context::default();
    let c = row::scalar_column(&bv(&mut other, 4));
    let p = other.set(c, &[1]);
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    symbolic(&cx, &r, Direct(p));
}

// --- literal packing --------------------------------------------------------

#[test]
fn unsigned_literals_pack_at_every_width() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 8);
    assert_eq!(words(&cx, &r, 5_u8), [5]);
    assert_eq!(words(&cx, &r, 6_u16), [6]);
    assert_eq!(words(&cx, &r, 7_u32), [7]);
    assert_eq!(words(&cx, &r, 8_u64), [8]);
    assert_eq!(words(&cx, &r, 9_u128), [9]);
    assert_eq!(words(&cx, &r, 10_usize), [10]);
    // Two words: the second word is the literal's high half.
    let wide = bv(&mut cx, 100);
    assert_eq!(words(&cx, &wide, (1_u128 << 70) | 3), [3, 1 << 6]);
    // Three words: words beyond the literal are zero-filled.
    let wider = bv(&mut cx, 130);
    assert_eq!(words(&cx, &wider, u128::MAX), [u64::MAX, u64::MAX, 0]);
    // A 128-bit sort accepts every u128 without a range check.
    let full = bv(&mut cx, 128);
    assert_eq!(words(&cx, &full, u128::MAX), [u64::MAX, u64::MAX]);
}

#[test]
fn signed_literals_use_twos_complement() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 8);
    assert_eq!(words(&cx, &r, -1_i8), [255]);
    assert_eq!(words(&cx, &r, -2_i16), [254]);
    assert_eq!(words(&cx, &r, -3_i32), [253]);
    assert_eq!(words(&cx, &r, -4_i64), [252]);
    assert_eq!(words(&cx, &r, -5_i128), [251]);
    assert_eq!(words(&cx, &r, -6_isize), [250]);
    assert_eq!(words(&cx, &r, 7_i8), [7]);
    // Sign extension fills every higher word, then the top word is masked to
    // the width.
    let wide = bv(&mut cx, 130);
    assert_eq!(words(&cx, &wide, -1_i8), [u64::MAX, u64::MAX, 3]);
    let w128 = bv(&mut cx, 128);
    assert_eq!(words(&cx, &w128, -1_i8), [u64::MAX, u64::MAX]);
}

#[test]
#[should_panic(expected = "integer literal exceeds its scalar sort")]
fn an_unsigned_literal_overflow_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 8);
    words(&cx, &r, 256_u16);
}

#[test]
#[should_panic(expected = "negative integer literal exceeds its signed width")]
fn a_negative_literal_overflow_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 8);
    words(&cx, &r, -129_i16);
}

#[test]
#[should_panic(expected = "integer literal sort must be a bit vector")]
fn an_integer_literal_without_a_bit_vector_is_rejected() {
    let mut cx = Context::default();
    let b = cx.bool_sort().bind(&mut cx);
    words(&cx, &b, 1_u8);
}

#[test]
#[expect(
    clippy::needless_borrows_for_generic_args,
    reason = "the doubled borrow exercises the row impl for references"
)]
fn booleans_words_and_arrays_pack() {
    let mut cx = Context::default();
    let b = cx.bool_sort().bind(&mut cx);
    assert_eq!(words(&cx, &b, true), [1]);
    assert_eq!(words(&cx, &b, false), [0]);
    let r = bv(&mut cx, 70);
    assert_eq!(words(&cx, &r, [1_u64, 2]), [1, 2]);
    assert_eq!(words(&cx, &r, &[3_u64, 4][..]), [3, 4]);
    // A borrowed literal forwards to the value.
    let s = bv(&mut cx, 8);
    assert_eq!(words(&cx, &s, &&9_u8), [9]);
}

#[test]
#[should_panic(expected = "a Boolean literal requires the Boolean sort")]
fn a_boolean_literal_without_the_boolean_sort_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 1);
    words(&cx, &r, true);
}

#[test]
#[should_panic(expected = "words do not encode the scalar sort")]
fn a_word_slice_outside_the_sort_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 8);
    words(&cx, &r, [256_u64]);
}

// --- predicate rows ---------------------------------------------------------

#[test]
fn predicate_slices_and_arrays_fill_rows() {
    let mut cx = Context::default();
    let c = row::scalar_column(&bv(&mut cx, 4));
    let pair = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4)])
        .bind(&mut cx);
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    let q = cx.product_sort([t.sort, cx.bv_sort(4)]).bind(&mut cx);
    let (a, b) = (cx.set(c, &[1, 2]), cx.set(c, &[7]));
    assert_eq!(
        symbolic(&cx, &pair, [a.clone(), b.clone()]),
        (vec![0, 0], vec![0, 1])
    );
    assert_eq!(
        symbolic(&cx, &pair, &[&a, &b][..]),
        (vec![0, 0], vec![0, 1])
    );
    assert_eq!(symbolic(&cx, &pair, [&b, &a]), (vec![0, 0], vec![0, 1]));
    assert_eq!(
        symbolic(&cx, &pair, vec![a.clone(), a.clone()].as_slice()),
        (vec![0, 0], vec![0, 1])
    );
    // Inside a tuple, a predicate slice spans the members its length covers.
    assert_eq!(
        symbolic(&cx, &q, ([&a, &b], 3_u8)),
        (vec![0, 0, 3], vec![0, 1])
    );
    assert_eq!(
        symbolic(&cx, &q, ([&b, &b, &b], [] as [&Predicate; 0])),
        (vec![0, 0, 0], vec![0, 1, 2])
    );
}

#[test]
fn empty_predicate_groups_pack_before_between_and_after_members() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    let predicates: &[Predicate] = &[];
    let borrowed: &[&Predicate] = &[];
    let owned: [Predicate; 0] = [];
    let refs: [&Predicate; 0] = [];
    assert_eq!(words(&cx, &r, (predicates, 7_u8)), [7]);
    assert_eq!(words(&cx, &r, (borrowed, 7_u8)), [7]);
    assert_eq!(words(&cx, &r, (owned, 7_u8)), [7]);
    assert_eq!(words(&cx, &r, (refs, 7_u8)), [7]);
    assert_eq!(words(&cx, &r, (7_u8, predicates)), [7]);
    let pair = cx.product_sort([r.sort, r.sort]).bind(&mut cx);
    assert_eq!(words(&cx, &pair, (3_u8, predicates, 7_u8)), [3, 7]);
    let empty = cx.tuple_decl("Empty", Vec::<(&str, Sort)>::new()).sort;
    let mixed = cx.product_sort([empty, r.sort]).bind(&mut cx);
    assert_eq!(words(&cx, &mixed, (predicates, 7_u8)), [7]);
    // A nonempty slice, like a scalar, needs a unit for an empty record on
    // either side, but spans one between its columns.
    let c = row::scalar_column(&r);
    let p = cx.set(c, &[1]);
    let trailing = cx.product_sort([r.sort, empty]).bind(&mut cx);
    let inner = cx.product_sort([r.sort, empty, r.sort]).bind(&mut cx);
    assert_eq!(symbolic(&cx, &inner, ([&p, &p],)), (vec![0, 0], vec![0, 1]));
    assert_eq!(symbolic(&cx, &mixed, ((), [&p])), (vec![0], vec![0]));
    assert_eq!(symbolic(&cx, &trailing, ([&p], ())), (vec![0], vec![0]));
    for (row, message) in [
        (&mixed, "row arguments must match the destination members"),
        (&trailing, "row arguments have the wrong member count"),
    ] {
        let e = catch_unwind(AssertUnwindSafe(|| symbolic(&cx, row, ([&p],)))).unwrap_err();
        let text = e.downcast_ref::<String>().map(String::as_str);
        assert!(
            text.or(e.downcast_ref::<&str>().copied())
                .unwrap()
                .contains(message)
        );
    }
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_predicate_slice_across_a_member_is_rejected() {
    let mut cx = Context::default();
    let c = row::scalar_column(&bv(&mut cx, 4));
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    let q = cx.product_sort([t.sort, cx.bv_sort(4)]).bind(&mut cx);
    let a = cx.set(c, &[1]);
    symbolic(&cx, &q, ([&a], &a, 3_u8));
}

#[test]
#[should_panic(expected = "predicate row has the wrong column count")]
fn a_short_predicate_row_is_rejected() {
    let mut cx = Context::default();
    let c = row::scalar_column(&bv(&mut cx, 4));
    let a = cx.set(c, &[1]);
    let pair = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4)])
        .bind(&mut cx);
    symbolic(&cx, &pair, [&a]);
}

#[test]
#[should_panic(expected = "row arguments must be ground values")]
fn a_predicate_in_a_ground_sink_is_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    let a = cx.set(row::scalar_column(&r), &[1]);
    words(&cx, &r, &a);
}

#[test]
#[should_panic(expected = "predicate uses an incompatible column")]
fn a_predicate_over_another_column_is_rejected() {
    let mut cx = Context::default();
    let c = row::scalar_column(&bv(&mut cx, 4));
    let r = bv(&mut cx, 8);
    let a = cx.set(c, &[1]);
    symbolic(&cx, &r, &a);
}

#[test]
#[should_panic(expected = "predicate belongs to another context")]
fn a_predicate_of_another_context_is_rejected() {
    let mut other = Context::default();
    let c = row::scalar_column(&bv(&mut other, 4));
    let a = other.set(c, &[1]);
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    symbolic(&cx, &r, &a);
}

#[test]
fn successive_appends_preserve_words_and_predicates() {
    let mut cx = Context::default();
    let wide = bv(&mut cx, 70);
    let narrow = bv(&mut cx, 4);
    let p = cx.set(row::scalar_column(&wide), &[1, 0]);
    let mut out = Symbolic::default();
    row::append(&cx, &p, &wide, &mut out);
    row::append(&cx, 7_u8, &narrow, &mut out);
    row::append(&cx, &p, &wide, &mut out);
    assert_eq!(out.words, [0, 0, 7, 0, 0]);
    assert_eq!(
        out.predicates.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
        [0, 3]
    );
    for (_, q) in &out.predicates {
        assert_eq!(q.column(), p.column());
        assert_eq!(q.function(), p.function());
    }
}

/// A sink reporting supplied offsets for newly appended predicates.
struct Offsets {
    out: Symbolic,
    offsets: std::array::IntoIter<usize, 2>,
}

impl RowSink for Offsets {
    fn words(&mut self) -> &mut Vec<u64> {
        &mut self.out.words
    }

    fn predicates(&self) -> &[(usize, Predicate)] {
        &self.out.predicates
    }

    fn predicate(&mut self, p: &Predicate) {
        self.out.predicate(p);
        self.out.predicates.last_mut().unwrap().0 = self.offsets.next().unwrap();
    }
}

#[test]
fn every_new_predicate_offset_is_checked_after_a_prefix() {
    let mut cx = Context::default();
    let n = cx.bv_sort(4);
    let pair = cx.product_sort([n, n]).bind(&mut cx);
    let p = cx.set(row::columns(&pair)[0], &[1]);
    // Before the prefix boundary, duplicated, descending, and past the row.
    for offsets in [[0, 2], [1, 1], [2, 1], [1, 3]] {
        let mut out = Offsets {
            out: Symbolic {
                words: vec![9],
                predicates: Vec::new(),
            },
            offsets: offsets.into_iter(),
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                row::append(&cx, [&p, &p], &pair, &mut out);
            }))
            .is_err()
        );
    }
}

// --- tuples, records, and products ------------------------------------------

#[test]
fn tuples_flatten_products_and_fill_records() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    // A 1-tuple into a scalar row and into a single-member record.
    let s = bv(&mut cx, 4);
    assert_eq!(words(&cx, &s, (3_u8,)), [3]);
    let one = cx.tuple_decl("One", [("x", cx.bv_sort(4))]);
    let o = one.sort.bind(&mut cx);
    assert_eq!(words(&cx, &o, (4_u8,)), [4]);
    // A record row takes its members positionally, nested products flatten.
    let r = t.sort.bind(&mut cx);
    assert_eq!(words(&cx, &r, (1_u8, 2_u8)), [1, 2]);
    // Product(T, c): a nested tuple supplies T's fields, so spans are inferred.
    let q = cx.product_sort([t.sort, cx.bv_sort(4)]).bind(&mut cx);
    assert_eq!(words(&cx, &q, ((1_u8, 2_u8), 3_u8)), [1, 2, 3]);
    assert_eq!(words(&cx, &q, ((5_u8, 6_u8), 7_u8)), [5, 6, 7]);
    // Product(a, b, T): a nested tuple spanning two scalars before the record.
    let q2 = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4), t.sort])
        .bind(&mut cx);
    assert_eq!(words(&cx, &q2, ((1_u8, 2_u8), (3_u8, 4_u8))), [1, 2, 3, 4]);
    assert_eq!(words(&cx, &q2, (1_u8, 2_u8, (3_u8, 4_u8))), [1, 2, 3, 4]);
    // A record whose member is itself a product takes a nested tuple for that
    // member.
    let ab = cx.product_sort([cx.bv_sort(4), cx.bv_sort(4)]);
    let u = cx.tuple_decl("U", [("ab", ab), ("c", cx.bv_sort(4))]);
    let q3 = cx.product_sort([u.sort, cx.bv_sort(4)]).bind(&mut cx);
    assert_eq!(words(&cx, &q3, (((1_u8, 2_u8), 3_u8), 4_u8)), [1, 2, 3, 4]);
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_tuple_span_short_of_the_members_is_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    let p = cx.product_sort([t.sort, cx.bv_sort(4)]).bind(&mut cx);
    words(&cx, &p, ((1_u8, 2_u8), 3_u8, 4_u8));
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_nested_group_span_failure_is_rejected() {
    let mut cx = Context::default();
    // The inner group has one component too many for T, so it cannot be the
    // record, and its own fallback then asks a scalar to span a record member:
    // that failure is what the outer group reports.
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    let p = cx.product_sort([t.sort, cx.bv_sort(4)]).bind(&mut cx);
    words(&cx, &p, ((1_u8, 2_u8, 3_u8), 4_u8));
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn extra_record_arguments_are_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(4))]);
    let r = t.sort.bind(&mut cx);
    words(&cx, &r, (1_u8, 2_u8, 3_u8));
}

#[test]
fn wrapped_record_values_fill_the_record() {
    let mut cx = Context::default();
    let n = cx.bv_sort(4);
    let t = cx.tuple_decl("T", [("a", n), ("b", n)]).sort;
    let r = t.bind(&mut cx);
    // A singleton tuple around the record's fields and a trailing unit are
    // anonymous groups around one record value.
    assert_eq!(words(&cx, &r, ((1_u8, 2_u8),)), [1, 2]);
    assert_eq!(words(&cx, &r, ((1_u8, 2_u8), ())), [1, 2]);
    assert_eq!(words(&cx, &r, (1_u8, 2_u8)), [1, 2]);
}

#[test]
#[should_panic(expected = "row arguments have the wrong member count")]
fn extra_product_arguments_are_rejected() {
    let mut cx = Context::default();
    // Spans are inferred (arity 3 != 4 members) and succeed, but leave a
    // member unfilled.
    let p = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4), cx.bv_sort(4), cx.bv_sort(4)])
        .bind(&mut cx);
    words(&cx, &p, ((1_u8, 2_u8), 3_u8));
}

#[test]
fn unit_arguments_fill_empty_aggregates() {
    let mut cx = Context::default();
    let u = cx.unit_sort().bind(&mut cx);
    assert_eq!(words(&cx, &u, ()), []);
    let e = cx.tuple_decl("E", [] as [(&str, Sort); 0]);
    let r = e.sort.bind(&mut cx);
    assert_eq!(words(&cx, &r, ()), []);
}

#[test]
fn empty_records_occupy_components_inside_products() {
    let mut cx = Context::default();
    let empty = cx.tuple_decl("Empty", [] as [(&str, Sort); 0]).sort;
    let n = cx.bv_sort(4);
    let before = cx.product_sort([empty, n]).bind(&mut cx);
    let after = cx.product_sort([n, empty]).bind(&mut cx);
    let both = cx.product_sort([empty, n, empty]).bind(&mut cx);
    let repeated = cx.product_sort([empty, empty]).bind(&mut cx);
    assert_eq!(words(&cx, &before, ((), 5_u8)), [5]);
    assert_eq!(words(&cx, &after, (5_u8, ())), [5]);
    assert_eq!(words(&cx, &both, ((), 5_u8, ())), [5]);
    assert_eq!(words(&cx, &repeated, ((), ())), []);
    let pair = cx.tuple_decl("Pair", [("a", n), ("b", n)]).sort;
    let nested = cx.product_sort([empty, pair]).bind(&mut cx);
    // Two input components and two destination members still require spans:
    // the empty record consumes one member and the pair consumes the other.
    assert_eq!(words(&cx, &nested, ((), (1_u8, 2_u8))), [1, 2]);
    let outer = cx
        .tuple_decl("Outer", [("empty", empty), ("pair", pair)])
        .sort;
    let row = cx.product_sort([outer, n]).bind(&mut cx);
    assert_eq!(words(&cx, &row, (((), (1_u8, 2_u8)), 3_u8)), [1, 2, 3]);
}

#[test]
fn unit_padding_preserves_anonymous_groups() {
    let mut cx = Context::default();
    let n = cx.bv_sort(4);
    let scalar = n.bind(&mut cx);
    let pair = cx.product_sort([n, n]).bind(&mut cx);
    assert_eq!(words(&cx, &scalar, ((), 5_u8, ())), [5]);
    assert_eq!(words(&cx, &pair, ((), (1_u8, ()), ((), 2_u8))), [1, 2]);
}

/// A nominal one-member record adapter.
struct Named {
    sort: Sort,
    value: u8,
}

impl IntoRow for Named {
    fn record_sort(&self, _: &Context) -> Option<Sort> {
        Some(self.sort)
    }

    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert_eq!(row.sort, self.sort);
        out.words().push(u64::from(self.value));
    }
}

#[test]
fn tuple_spans_preserve_nominal_adapters() {
    let mut cx = Context::default();
    let n = cx.bv_sort(4);
    let t = cx.tuple_decl("T", [("a", n)]).sort;
    let u = cx.tuple_decl("U", [("a", n)]).sort;
    let record = t.bind(&mut cx);
    let product = cx.product_sort([t, n]).bind(&mut cx);
    let value = Named { sort: t, value: 3 };
    assert_eq!(words(&cx, &record, (&value,)), [3]);
    assert_eq!(words(&cx, &product, ((&value,), 4_u8)), [3, 4]);
    let foreign = Named { sort: u, value: 3 };
    let e = catch_unwind(AssertUnwindSafe(|| {
        words(&cx, &product, (foreign, 4_u8));
    }))
    .unwrap_err();
    assert_eq!(
        e.downcast_ref::<String>().map(String::as_str),
        Some("row arguments must match the destination members")
    );
}

#[test]
#[should_panic(expected = "unit arguments require an empty aggregate")]
fn unit_arguments_into_a_scalar_are_rejected() {
    let mut cx = Context::default();
    let r = bv(&mut cx, 4);
    words(&cx, &r, ());
}

#[test]
#[should_panic(expected = "unit arguments require an empty aggregate")]
fn unit_arguments_into_a_product_are_rejected() {
    let mut cx = Context::default();
    let p = cx
        .product_sort([cx.bv_sort(4), cx.bv_sort(4)])
        .bind(&mut cx);
    words(&cx, &p, ());
}

#[test]
fn nested_groups_infer_spans_from_record_members() {
    let mut cx = Context::default();
    // U groups a record beside a scalar, so a flat-ish argument tuple has more
    // components than the destination has members and every component's span
    // has to be inferred.
    let t = cx.tuple_decl("T", [("x", cx.bv_sort(4)), ("y", cx.bv_sort(4))]);
    let u = cx.tuple_decl("U", [("p", t.sort), ("n", cx.bv_sort(4))]);
    let r = cx.product_sort([u.sort, cx.bv_sort(4)]).bind(&mut cx);
    assert_eq!(words(&cx, &r, (((1_u8, 2_u8), 3_u8), 4_u8)), [1, 2, 3, 4]);
    // A group with one component too many cannot be the record, and its own
    // fallback asks a scalar to span a record member: that component reports
    // the failure rather than guessing.
    let n = cx.bv_sort(4);
    assert_eq!((1_u8, 2_u8, 3_u8).span(&cx, &[u.sort, n]), None);
    assert_eq!((1_u8, 2_u8).span(&cx, &[u.sort, n]), None);
    // The failure can also fall on the last component, after earlier ones have
    // advanced.
    assert_eq!((1_u8, 2_u8).span(&cx, &[n, u.sort]), None);
    // A predicate slice reports its own length as its arity.
    let s = cx.product_sort([n, n, n]).bind(&mut cx);
    let p = cx.set(row::columns(&s)[0], &[1, 2]);
    assert_eq!(
        symbolic(&cx, &s, (&[p.clone(), p][..], 9_u8)),
        (vec![0, 0, 9], vec![0, 1])
    );
    // A unit component claims no members at all.
    let t = bv(&mut cx, 4);
    assert_eq!(words(&cx, &t, ((), 5_u8)), [5]);
}
