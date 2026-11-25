//! Public contracts for row layouts and argument packing.
//!
//! Small stores exercise explicit bindings. The packing fixtures check both
//! ground words and the word offsets of symbolic fields.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sra::ast::Sort;
use sra::bdd::BddStore;
use sra::context::{Context, Predicate};
use sra::row::{self, IntoRow, Row, RowRef, RowSink};
use sra::sparse::SparseStore;

/// A context with both backends initially holding only their constants.
fn tiny() -> Context {
    let mut cx = Context::empty();
    cx.register_backend("bdd", BddStore::with_capacity(2, 0, 1));
    cx.register_backend("sparse", SparseStore::with_capacity(2, 0, 1));
    cx
}

fn rejects(expected: &str, run: impl FnOnce()) {
    let panic = catch_unwind(AssertUnwindSafe(run)).expect_err("operation was accepted");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap();
    assert!(message.contains(expected), "unexpected panic: {message}");
}

// --- explicit layouts ----

#[test]
fn explicit_layouts_check_each_selected_algebra_and_the_slice_bounds() {
    let mut cx = tiny();
    let s = cx.sorts();
    let (a, b, flag, bit) = (s.bv(4), s.bv(8), s.bool(), s.bv(1));
    let pair = s.product([a, b]);
    let narrow = cx.bind("bdd", a.scalar());
    let wide = cx.bind("bdd", b.scalar());
    let bv1 = cx.bind("sparse", bit.scalar());
    let mismatch = "row algebra must match its logical scalar sort";
    rejects(mismatch, || {
        Row::new(&cx, a, Arc::from([wide]), 0);
    });
    rejects(mismatch, || {
        Row::new(&cx, flag, Arc::from([bv1]), 0);
    });
    rejects(mismatch, || {
        Row::new(&cx, pair, Arc::from([narrow, narrow]), 0);
    });
    let bounds = "row layout must cover the logical sort";
    rejects(bounds, || {
        Row::new(&cx, pair, Arc::from([narrow]), 0);
    });
    rejects(bounds, || {
        Row::new(&cx, cx.sorts().unit(), Arc::from([]), usize::MAX);
    });
    let foreign = tiny().bind("bdd", b.scalar());
    rejects("algebra belongs to another context", || {
        Row::new(&cx, pair, Arc::from([narrow, foreign]), 0);
    });
}

#[test]
fn nested_members_share_the_selected_slice_including_empty_members() {
    let mut cx = tiny();
    let s = cx.sorts();
    let a = s.bv(4);
    let empty = s.record("Empty", &[]);
    let r = s.record("R", &[("empty", empty), ("x", a), ("y", a)]);
    let p = s.product([a, r, empty]);
    let sparse = cx.bind("sparse", a.scalar());
    let bdd = cx.bind("bdd", a.scalar());
    let layout: Arc<[_]> = [sparse, bdd, sparse, sparse].into();
    let row = Row::new(&cx, p, layout.clone(), 1);
    assert_eq!(row.algebras(), [bdd, sparse, sparse]);
    let parts: Vec<_> = row.members(&cx).collect();
    assert_eq!(
        parts.iter().map(Row::sort).collect::<Vec<_>>(),
        [a, r, empty]
    );
    assert!(std::ptr::eq(parts[1].algebras(), &layout[2..]));
    assert!(std::ptr::eq(parts[2].algebras(), &layout[4..]));
    let reordered = Row::product(&cx, &[&parts[1], &parts[0], &parts[2]]);
    assert_eq!(reordered.sort(), cx.sorts().product([r, a, empty]));
    assert_eq!(reordered.algebras(), [sparse, sparse, bdd]);
    let mut members = parts[1].members(&cx);
    assert_eq!(members.len(), 3);
    assert_eq!(members.next().unwrap().sort(), empty);
    assert!(std::ptr::eq(
        members.next().unwrap().algebras(),
        &layout[2..3]
    ));
    assert!(std::ptr::eq(
        members.next().unwrap().algebras(),
        &layout[3..4]
    ));
    assert_eq!(members.len(), 0);
}

// --- setup and lookup ----

#[test]
fn binding_one_accessor_leaves_its_sibling_available_for_configuration() {
    let mut cx = tiny();
    let s = cx.sorts();
    let a = s.bv(8);
    let pair = s.record("Pair", &[("src", a), ("dst", a)]);
    let [src, dst] = [0, 1].map(|i| s.project(pair, i));
    let ctor = s.constructor(pair);
    let args = ctor.resolve(s).domain;
    let projection = s.project(args, 1);

    let first = Row::bind(&mut cx, src);
    let sparse = cx.bind("sparse", a.scalar());
    assert_eq!(first.sort(), a);
    assert_eq!(first.algebras(), [sparse]);
    assert_eq!(cx.field(dst), None);

    let shared = cx.clone();
    rejects("row field must be bound", || {
        Row::of(&shared, pair);
    });
    // The sparse algebra was created through a named field. Anonymous
    // occurrences can already use it while setup is closed.
    assert_eq!(Row::of(&shared, a).algebras(), [sparse]);
    assert_eq!(Row::bind(&mut cx, a).algebras(), [sparse]);
    assert_eq!(Row::of(&shared, projection).algebras(), [sparse]);
    assert_eq!(cx.field(dst), None);
    drop(shared);

    let bdd = cx.bind_field(dst, "bdd");
    let shared = cx.clone();
    let mut row = Row::bind(&mut cx, ctor);
    assert_eq!(row.sort(), pair);
    assert_eq!(row.algebras(), [sparse, bdd]);
    assert_eq!(Row::of(&shared, pair).algebras(), row.algebras());
    assert_eq!(Row::of(&shared, dst).algebras(), [bdd]);
    row.coerce(&shared, args);
    assert_eq!(row.sort(), args);
    assert_eq!(row.algebras(), [sparse, bdd]);
    assert_eq!(Row::of(&shared, args).algebras(), [sparse, sparse]);
    row.coerce(&shared, pair);
    assert_eq!(row.sort(), pair);
}

#[test]
fn aggregate_projections_bind_only_the_nested_leaves() {
    let mut cx = tiny();
    let s = cx.sorts();
    let a = s.bv(4);
    let inner = s.record("Inner", &[("x", a)]);
    let group = s.product([inner, a]);
    let outer = s.record("Outer", &[("group", group), ("y", a)]);
    let x = s.project(inner, 0);
    let part = s.project(outer, 0);
    let y = s.project(outer, 1);
    let bdd = cx.bind_field(x, "bdd");
    let row = Row::bind(&mut cx, part);
    let sparse = cx.bind("sparse", a.scalar());
    assert_eq!(row.sort(), group);
    assert_eq!(row.algebras(), [bdd, sparse]);
    assert_eq!(cx.field(y), None);
    let whole = Row::bind(&mut cx, outer);
    assert_eq!(whole.algebras(), [bdd, sparse, sparse]);
    assert_eq!(Row::of(&cx, part).algebras(), row.algebras());
}

#[test]
fn fully_configured_rows_need_only_the_selected_backend() {
    let mut cx = Context::empty();
    cx.register_backend("custom", BddStore::with_capacity(2, 0, 1));
    let s = cx.sorts();
    let a = s.bv(4);
    let r = s.record("R", &[("v", a)]);
    let field = s.project(r, 0);
    let id = cx.bind_field(field, "custom");
    assert_eq!(Row::bind(&mut cx, r).algebras(), [id]);
    assert_eq!(Row::of(&cx, r).algebras(), [id]);
    rejects("backend \"sparse\" is not registered", || {
        Row::bind(&mut cx, a);
    });
}

#[test]
fn lookup_of_an_unbound_anonymous_scalar_is_rejected() {
    let cx = tiny();
    let a = cx.sorts().bool();
    rejects("row field must be bound", || {
        Row::of(&cx, a);
    });
    assert_eq!(cx.algebra("sparse", a.scalar()), None);
}

// --- products and coercion ----

#[test]
fn products_use_the_member_layouts_and_preserve_nominal_symbol_sorts() {
    let mut cx = tiny();
    let s = cx.sorts();
    let empty = s.record("Empty", &[]);
    let (x, y) = (s.symbol("X"), s.symbol("Y"));
    let args = s.product([x, y, empty]);
    let unit = Row::product(&cx, &[]);
    let e = Row::bind(&mut cx, empty);
    let mut a = Row::bind(&mut cx, x);
    let b = Row::bind(&mut cx, y);
    assert_eq!(a.algebras(), b.algebras());
    let ab = Row::product(&cx, &[&unit, &a, &b, &e]);
    assert_eq!(ab.sort(), args);
    assert_eq!(ab.algebras(), [a.algebras()[0], b.algebras()[0]]);
    rejects("coercion requires", || {
        a.coerce(&cx, y);
    });
    assert_eq!(a.sort(), x);
}

#[test]
fn foreign_sorts_and_rows_are_rejected_even_without_scalar_leaves() {
    let mut cx = Context::empty();
    let mut other = Context::empty();
    let foreign = other.sorts().record("E", &[]);
    let mut row = Row::bind(&mut other, foreign);
    let ctor = other.sorts().constructor(foreign);
    let local = cx.sorts().record("E", &[]);
    let reason = "declaration belongs to another store";
    rejects(reason, || {
        row::Domain::row(&row, &cx);
    });
    rejects(reason, || {
        Row::new(&cx, foreign, Arc::from([]), 0);
    });
    rejects(reason, || {
        Row::bind(&mut cx, foreign);
    });
    rejects(reason, || {
        Row::of(&cx, foreign);
    });
    rejects(reason, || {
        Row::bind(&mut cx, ctor);
    });
    rejects(reason, || {
        Row::of(&cx, ctor);
    });
    rejects(reason, || {
        Row::product(&cx, &[&row]);
    });
    rejects(reason, || {
        row.members(&cx).count();
    });
    rejects(reason, || {
        row.coerce(&cx, local);
    });
    rejects(reason, || {
        row.coerce(&other, local);
    });
}

#[test]
#[should_panic(expected = "row members require a record or product")]
fn requesting_members_of_a_scalar_is_rejected() {
    let mut cx = tiny();
    let sort = cx.sorts().bool();
    Row::bind(&mut cx, sort).members(&cx).count();
}

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.sorts().bv(w);
    Row::bind(cx, sort)
}

/// The ground words `args` pack into for `row`.
fn words(cx: &Context, row: &Row, args: impl IntoRow) -> Vec<u64> {
    let mut out = Vec::new();
    row::append(cx, args, row, &mut out);
    out
}

/// A sink that retains predicates beside the words.
#[derive(Default)]
struct Symbolic {
    words: Vec<u64>,
    predicates: Vec<(usize, Predicate)>,
}

impl RowSink for Symbolic {
    fn words(&mut self) -> &mut Vec<u64> {
        &mut self.words
    }

    fn predicates(&mut self) -> Option<&mut Vec<(usize, Predicate)>> {
        Some(&mut self.predicates)
    }
}

/// The words and predicate offsets `args` pack into for `row`.
fn symbolic(cx: &Context, row: &Row, args: impl IntoRow) -> (Vec<u64>, Vec<usize>) {
    let mut out = Symbolic::default();
    row::append(cx, args, row, &mut out);
    let offsets = out.predicates.iter().map(|(i, _)| *i).collect();
    (out.words, offsets)
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
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![]));
}

#[test]
#[should_panic(expected = "row arguments have too many words")]
fn an_adapter_with_extra_words_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![1, 2]));
}

#[test]
#[should_panic(expected = "row value does not encode its scalar sort")]
fn adapter_words_outside_the_sort_are_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    words(&cx, &r, Raw(vec![16]));
}

/// Emits a predicate in the middle of a two-word scalar.
struct Misplaced(Predicate);

impl IntoRow for Misplaced {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().extend([0, 0]);
        out.predicates().unwrap().push((1, self.0.clone()));
    }
}

#[test]
#[should_panic(expected = "predicate must start at a scalar boundary")]
fn a_predicate_inside_a_scalar_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 70);
    let p = cx.set(r.algebras()[0], &[1, 0]);
    symbolic(&cx, &r, Misplaced(p));
}

/// Emits a ground word, then a predicate past the end of the row.
struct Dangling(Predicate);

impl IntoRow for Dangling {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().push(1);
        let n = out.words().len();
        out.predicates().unwrap().push((n, self.0.clone()));
    }
}

#[test]
#[should_panic(expected = "row arguments have misplaced predicates")]
fn a_predicate_past_the_row_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    let p = cx.set(r.algebras()[0], &[1]);
    symbolic(&cx, &r, Dangling(p));
}

/// Hands a predicate straight to the sink, bypassing the predicate adapter.
struct Direct(Predicate);

impl IntoRow for Direct {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().push(0);
        out.predicates().unwrap().push((0, self.0.clone()));
    }
}

#[test]
#[should_panic(expected = "predicate uses an incompatible algebra")]
fn a_direct_predicate_over_another_column_is_rejected() {
    let mut cx = tiny();
    let c = bv(&mut cx, 8).algebras()[0];
    let r = bv(&mut cx, 4);
    let p = cx.set(c, &[1]);
    symbolic(&cx, &r, Direct(p));
}

#[test]
#[should_panic(expected = "predicate belongs to another context")]
fn a_direct_predicate_of_another_context_is_rejected() {
    let mut other = tiny();
    let c = bv(&mut other, 4).algebras()[0];
    let p = other.set(c, &[1]);
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    symbolic(&cx, &r, Direct(p));
}

// --- literal packing --------------------------------------------------------

#[test]
fn unsigned_literals_pack_at_every_width() {
    let mut cx = tiny();
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
    // A 128-bit sort accepts every u128.
    let full = bv(&mut cx, 128);
    assert_eq!(words(&cx, &full, u128::MAX), [u64::MAX, u64::MAX]);
}

#[test]
fn signed_literals_use_twos_complement() {
    let mut cx = tiny();
    let r = bv(&mut cx, 8);
    assert_eq!(words(&cx, &r, -1_i8), [255]);
    assert_eq!(words(&cx, &r, i8::MIN), [128]);
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
    assert_eq!(words(&cx, &w128, i128::MIN), [0, 1 << 63]);
}

#[test]
#[should_panic(expected = "integer literal exceeds its scalar sort")]
fn an_unsigned_literal_overflow_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 8);
    words(&cx, &r, 256_u16);
}

#[test]
#[should_panic(expected = "negative integer literal exceeds its signed width")]
fn a_negative_literal_overflow_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 8);
    words(&cx, &r, -129_i16);
}

#[test]
#[should_panic(expected = "integer literal sort must be a bit vector")]
fn an_integer_literal_without_a_bit_vector_is_rejected() {
    let mut cx = tiny();
    let sort = cx.sorts().bool();
    let b = Row::bind(&mut cx, sort);
    words(&cx, &b, 1_u8);
}

#[test]
#[expect(
    clippy::needless_borrows_for_generic_args,
    reason = "the doubled borrow exercises the row impl for references"
)]
fn booleans_words_and_arrays_pack() {
    let mut cx = tiny();
    let sort = cx.sorts().bool();
    let b = Row::bind(&mut cx, sort);
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
    let mut cx = tiny();
    let r = bv(&mut cx, 1);
    words(&cx, &r, true);
}

// --- predicate rows ---------------------------------------------------------

#[test]
fn predicate_slices_and_arrays_fill_rows() {
    let mut cx = tiny();
    let c = bv(&mut cx, 4).algebras()[0];
    let sort = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4)]);
    let pair = Row::bind(&mut cx, sort);
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([t, cx.sorts().bv(4)]);
    let q = Row::bind(&mut cx, sort);
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
    let mut cx = tiny();
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
    let sort = cx.sorts().product([r.sort(), r.sort()]);
    let pair = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &pair, (3_u8, predicates, 7_u8)), [3, 7]);
    let empty = cx.sorts().record("Empty", &[]);
    let sort = cx.sorts().product([empty, r.sort()]);
    let mixed = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &mixed, (predicates, 7_u8)), [7]);
    // A nonempty slice, like a scalar, needs a unit for an empty record on
    // either side, but spans one between its columns.
    let c = r.algebras()[0];
    let p = cx.set(c, &[1]);
    let sort = cx.sorts().product([r.sort(), empty]);
    let trailing = Row::bind(&mut cx, sort);
    let sort = cx.sorts().product([r.sort(), empty, r.sort()]);
    let inner = Row::bind(&mut cx, sort);
    assert_eq!(symbolic(&cx, &inner, ([&p, &p],)), (vec![0, 0], vec![0, 1]));
    assert_eq!(symbolic(&cx, &mixed, ((), [&p])), (vec![0], vec![0]));
    assert_eq!(symbolic(&cx, &trailing, ([&p], ())), (vec![0], vec![0]));
    for (row, message) in [
        (&mixed, "row arguments must match the destination members"),
        (&trailing, "row arguments have the wrong member count"),
    ] {
        rejects(message, || {
            symbolic(&cx, row, ([&p],));
        });
    }
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_predicate_slice_across_a_member_is_rejected() {
    let mut cx = tiny();
    let c = bv(&mut cx, 4).algebras()[0];
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([t, cx.sorts().bv(4)]);
    let q = Row::bind(&mut cx, sort);
    let a = cx.set(c, &[1]);
    symbolic(&cx, &q, ([&a], &a, 3_u8));
}

#[test]
#[should_panic(expected = "predicate row has the wrong column count")]
fn a_short_predicate_row_is_rejected() {
    let mut cx = tiny();
    let c = bv(&mut cx, 4).algebras()[0];
    let a = cx.set(c, &[1]);
    let sort = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4)]);
    let pair = Row::bind(&mut cx, sort);
    symbolic(&cx, &pair, [&a]);
}

#[test]
#[should_panic(expected = "row arguments must be ground values")]
fn a_predicate_in_a_ground_sink_is_rejected() {
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    let a = cx.set(r.algebras()[0], &[1]);
    words(&cx, &r, &a);
}

#[test]
#[should_panic(expected = "predicate uses an incompatible algebra")]
fn a_predicate_over_another_column_is_rejected() {
    let mut cx = tiny();
    let c = bv(&mut cx, 4).algebras()[0];
    let r = bv(&mut cx, 8);
    let a = cx.set(c, &[1]);
    symbolic(&cx, &r, &a);
}

#[test]
#[should_panic(expected = "predicate belongs to another context")]
fn a_predicate_of_another_context_is_rejected() {
    let mut other = tiny();
    let c = bv(&mut other, 4).algebras()[0];
    let a = other.set(c, &[1]);
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    symbolic(&cx, &r, &a);
}

#[test]
fn predicate_adapters_check_the_binding_before_writing() {
    let mut cx = tiny();
    let row = bv(&mut cx, 4);
    let wide = bv(&mut cx, 8);
    let mut other = tiny();
    let foreign = bv(&mut other, 4);
    let view = RowRef {
        cx: &cx,
        sort: row.sort(),
        algebras: row.algebras(),
        members: &[],
    };
    for (p, message) in [
        (cx.set(wide.algebras()[0], &[1]), "incompatible algebra"),
        (other.set(foreign.algebras()[0], &[1]), "another context"),
    ] {
        let mut out = Symbolic::default();
        rejects(message, || p.append(view, &mut out));
        rejects(message, || [&p].append(view, &mut out));
        assert!(out.words.is_empty() && out.predicates.is_empty());
    }
}

#[test]
fn successive_appends_preserve_words_and_predicates() {
    let mut cx = tiny();
    let wide = bv(&mut cx, 70);
    let narrow = bv(&mut cx, 4);
    let p = cx.set(wide.algebras()[0], &[1, 0]);
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
        assert_eq!(q.algebra(), p.algebra());
        assert_eq!(q.function(), p.function());
    }
}

/// Emits two one-word predicates at the supplied offsets.
struct Offsets([usize; 2], Predicate);

impl IntoRow for Offsets {
    fn append(&self, _: RowRef<'_>, out: &mut impl RowSink) {
        out.words().extend([0, 0]);
        let v = out.predicates().unwrap();
        v.extend(self.0.map(|i| (i, self.1.clone())));
    }
}

#[test]
fn every_new_predicate_offset_is_checked_after_a_prefix() {
    let mut cx = tiny();
    let n = cx.sorts().bv(4);
    let sort = cx.sorts().product([n, n]);
    let pair = Row::bind(&mut cx, sort);
    let p = cx.set(pair.algebras()[0], &[1]);
    // Before the prefix boundary, duplicated, descending, and past the row.
    for (offsets, message) in [
        ([0, 2], "predicate must start at a scalar boundary"),
        ([1, 1], "predicate must start at a scalar boundary"),
        ([2, 1], "row arguments have misplaced predicates"),
        ([1, 3], "row arguments have misplaced predicates"),
    ] {
        let mut out = Symbolic {
            words: vec![9],
            predicates: Vec::new(),
        };
        rejects(message, || {
            row::append(&cx, Offsets(offsets, p.clone()), &pair, &mut out);
        });
    }
}

// --- tuples, records, and products ------------------------------------------

#[test]
fn tuples_flatten_products_and_fill_records() {
    let mut cx = tiny();
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    // A 1-tuple into a scalar row and into a single-member record.
    let s = bv(&mut cx, 4);
    assert_eq!(words(&cx, &s, (3_u8,)), [3]);
    let one = cx.sorts().record("One", &[("x", cx.sorts().bv(4))]);
    let o = Row::bind(&mut cx, one);
    assert_eq!(words(&cx, &o, (4_u8,)), [4]);
    // A record row takes its members positionally, nested products flatten.
    let r = Row::bind(&mut cx, t);
    assert_eq!(words(&cx, &r, (1_u8, 2_u8)), [1, 2]);
    // Product(T, c): a nested tuple supplies T's fields, so spans are inferred.
    let sort = cx.sorts().product([t, cx.sorts().bv(4)]);
    let q = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &q, ((1_u8, 2_u8), 3_u8)), [1, 2, 3]);
    assert_eq!(words(&cx, &q, ((5_u8, 6_u8), 7_u8)), [5, 6, 7]);
    // Product(a, b, T): a nested tuple spanning two scalars before the record.
    let sort = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4), t]);
    let q2 = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &q2, ((1_u8, 2_u8), (3_u8, 4_u8))), [1, 2, 3, 4]);
    assert_eq!(words(&cx, &q2, (1_u8, 2_u8, (3_u8, 4_u8))), [1, 2, 3, 4]);
    // A record whose member is itself a product takes a nested tuple for that
    // member.
    let ab = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4)]);
    let u = cx
        .sorts()
        .record("U", &[("ab", ab), ("c", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([u, cx.sorts().bv(4)]);
    let q3 = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &q3, (((1_u8, 2_u8), 3_u8), 4_u8)), [1, 2, 3, 4]);
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_tuple_span_short_of_the_members_is_rejected() {
    let mut cx = tiny();
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([t, cx.sorts().bv(4)]);
    let p = Row::bind(&mut cx, sort);
    words(&cx, &p, ((1_u8, 2_u8), 3_u8, 4_u8));
}

#[test]
#[should_panic(expected = "row arguments must match the destination members")]
fn a_nested_group_span_failure_is_rejected() {
    let mut cx = tiny();
    // The inner group has one component too many for T, so it cannot be the
    // record, and its own fallback then asks a scalar to span a record member:
    // that failure is what the outer group reports.
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([t, cx.sorts().bv(4)]);
    let p = Row::bind(&mut cx, sort);
    words(&cx, &p, ((1_u8, 2_u8, 3_u8), 4_u8));
}

#[test]
#[should_panic(expected = "row arguments have the wrong member count")]
fn extra_record_arguments_are_rejected() {
    let mut cx = tiny();
    let t = cx
        .sorts()
        .record("T", &[("a", cx.sorts().bv(4)), ("b", cx.sorts().bv(4))]);
    let r = Row::bind(&mut cx, t);
    words(&cx, &r, (1_u8, 2_u8, 3_u8));
}

#[test]
fn record_fields_require_complete_spans_directly_and_when_nested() {
    fn check<A: IntoRow + Clone>(
        cx: &Context,
        direct: &Row,
        binary: &Row,
        nested: &Row,
        arg: A,
        leading: bool,
    ) {
        assert!(catch_unwind(AssertUnwindSafe(|| symbolic(cx, direct, (arg.clone(),)))).is_err());
        assert!(
            catch_unwind(AssertUnwindSafe(|| symbolic(
                cx,
                binary,
                ((arg.clone(),), 0_u8)
            )))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| symbolic(cx, nested, ((arg.clone(),),)))).is_err()
        );
        if leading {
            assert_eq!(
                symbolic(cx, direct, (((), arg.clone()),)),
                (vec![0], vec![0])
            );
            assert_eq!(
                symbolic(cx, binary, ((((), arg.clone()),), 0_u8)),
                (vec![0, 0], vec![0])
            );
            assert_eq!(symbolic(cx, nested, ((((), arg),),)), (vec![0], vec![0]));
        } else {
            assert_eq!(
                symbolic(cx, direct, ((arg.clone(), ()),)),
                (vec![0], vec![0])
            );
            assert_eq!(
                symbolic(cx, binary, (((arg.clone(), ()),), 0_u8)),
                (vec![0, 0], vec![0])
            );
            assert_eq!(symbolic(cx, nested, (((arg, ()),),)), (vec![0], vec![0]));
        }
    }

    for backend in ["bdd", "sparse"] {
        for leading in [false, true] {
            let mut cx = tiny();
            let bits = cx.sorts().bv(1);
            let empty = cx.sorts().record("Empty", &[]);
            let field = cx.sorts().product(if leading {
                [empty, bits]
            } else {
                [bits, empty]
            });
            let r = cx.sorts().record("Record", &[("field", field)]);
            let outer = cx.sorts().record("Outer", &[("record", r)]);
            let direct = Row::bind(&mut cx, r);
            let sort = cx.sorts().product([r, bits]);
            let binary = Row::bind(&mut cx, sort);
            let nested = Row::bind(&mut cx, outer);
            let c = cx.bind(backend, bits.scalar());
            // Whole-field predicate arrays use the field's actual backend.
            let replace = |row: Row| {
                let mut layout = row.algebras().to_vec();
                layout[0] = c;
                Row::new(&cx, row.sort(), layout.into(), 0)
            };
            let (direct, binary, nested) = (replace(direct), replace(binary), replace(nested));
            let p = cx.set(c, &[0]);
            let owned = [p.clone()];
            let borrowed = [&p];
            check(&cx, &direct, &binary, &nested, owned.clone(), leading);
            check(&cx, &direct, &binary, &nested, borrowed, leading);
            check(&cx, &direct, &binary, &nested, owned.as_slice(), leading);
            check(&cx, &direct, &binary, &nested, borrowed.as_slice(), leading);
        }
    }
}

#[test]
fn record_destinations_take_their_members() {
    let mut cx = tiny();
    let n = cx.sorts().bv(4);
    let t = cx.sorts().record("T", &[("a", n), ("b", n)]);
    let r = Row::bind(&mut cx, t);
    let sort = cx.sorts().product([t, n]);
    let p = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &r, (1_u8, 2_u8)), [1, 2]);
    // A tuple aimed at a record supplies its members, so a singleton or
    // unit-padded group around the fields is rejected by `span` and `append`.
    let fails = |f: &dyn Fn()| catch_unwind(AssertUnwindSafe(f)).is_err();
    assert!(fails(&|| drop(words(&cx, &r, ((1_u8, 2_u8),)))));
    assert!(fails(&|| drop(words(&cx, &r, ((1_u8, 2_u8), ())))));
    assert!(fails(&|| drop(words(&cx, &p, (((1_u8, 2_u8),), 3_u8)))));
    assert!(fails(&|| drop(words(&cx, &p, (((1_u8, 2_u8), ()), 3_u8)))));
}

#[test]
#[should_panic(expected = "row arguments have the wrong member count")]
fn extra_product_arguments_are_rejected() {
    let mut cx = tiny();
    // Spans are inferred (arity 3 against 4 members) and succeed, but leave a
    // member unfilled.
    let sort = cx.sorts().product([
        cx.sorts().bv(4),
        cx.sorts().bv(4),
        cx.sorts().bv(4),
        cx.sorts().bv(4),
    ]);
    let p = Row::bind(&mut cx, sort);
    words(&cx, &p, ((1_u8, 2_u8), 3_u8));
}

#[test]
fn unit_arguments_fill_empty_aggregates() {
    let mut cx = tiny();
    let sort = cx.sorts().unit();
    let u = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &u, ()), []);
    let e = cx.sorts().record("E", &[]);
    let r = Row::bind(&mut cx, e);
    assert_eq!(words(&cx, &r, ()), []);
}

#[test]
fn empty_records_occupy_components_inside_products() {
    let mut cx = tiny();
    let empty = cx.sorts().record("Empty", &[]);
    let n = cx.sorts().bv(4);
    let sort = cx.sorts().product([empty, n]);
    let before = Row::bind(&mut cx, sort);
    let sort = cx.sorts().product([n, empty]);
    let after = Row::bind(&mut cx, sort);
    let sort = cx.sorts().product([empty, n, empty]);
    let both = Row::bind(&mut cx, sort);
    let sort = cx.sorts().product([empty, empty]);
    let repeated = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &before, ((), 5_u8)), [5]);
    assert_eq!(words(&cx, &after, (5_u8, ())), [5]);
    assert_eq!(words(&cx, &both, ((), 5_u8, ())), [5]);
    assert_eq!(words(&cx, &repeated, ((), ())), []);
    let pair = cx.sorts().record("Pair", &[("a", n), ("b", n)]);
    let sort = cx.sorts().product([empty, pair]);
    let nested = Row::bind(&mut cx, sort);
    // Two input components and two destination members still require spans:
    // the empty record consumes one member and the pair consumes the other.
    assert_eq!(words(&cx, &nested, ((), (1_u8, 2_u8))), [1, 2]);
    let outer = cx
        .sorts()
        .record("Outer", &[("empty", empty), ("pair", pair)]);
    let sort = cx.sorts().product([outer, n]);
    let row = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &row, (((), (1_u8, 2_u8)), 3_u8)), [1, 2, 3]);
}

#[test]
fn unit_padding_preserves_anonymous_groups() {
    let mut cx = tiny();
    let n = cx.sorts().bv(4);
    let scalar = Row::bind(&mut cx, n);
    let sort = cx.sorts().product([n, n]);
    let pair = Row::bind(&mut cx, sort);
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
    let mut cx = tiny();
    let n = cx.sorts().bv(4);
    let t = cx.sorts().record("T", &[("a", n)]);
    let u = cx.sorts().record("U", &[("a", n)]);
    let record = Row::bind(&mut cx, t);
    let sort = cx.sorts().product([t, n]);
    let product = Row::bind(&mut cx, sort);
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
    let mut cx = tiny();
    let r = bv(&mut cx, 4);
    words(&cx, &r, ());
}

#[test]
#[should_panic(expected = "unit arguments require an empty aggregate")]
fn unit_arguments_into_a_product_are_rejected() {
    let mut cx = tiny();
    let sort = cx.sorts().product([cx.sorts().bv(4), cx.sorts().bv(4)]);
    let p = Row::bind(&mut cx, sort);
    words(&cx, &p, ());
}

#[test]
fn nested_groups_infer_spans_from_record_members() {
    let mut cx = tiny();
    // U groups a record beside a scalar, so a flat argument tuple has more
    // components than the destination has members and every component's span
    // has to be inferred.
    let t = cx
        .sorts()
        .record("T", &[("x", cx.sorts().bv(4)), ("y", cx.sorts().bv(4))]);
    let u = cx.sorts().record("U", &[("p", t), ("n", cx.sorts().bv(4))]);
    let sort = cx.sorts().product([u, cx.sorts().bv(4)]);
    let r = Row::bind(&mut cx, sort);
    assert_eq!(words(&cx, &r, (((1_u8, 2_u8), 3_u8), 4_u8)), [1, 2, 3, 4]);
    // A group with one component too many cannot be the record, and its own
    // fallback asks a scalar to span a record member: that component reports
    // the failure rather than guessing.
    let n = cx.sorts().bv(4);
    assert_eq!((1_u8, 2_u8, 3_u8).span(&cx, &[u, n]), None);
    assert_eq!((1_u8, 2_u8).span(&cx, &[u, n]), None);
    // The failure can also fall on the last component, after earlier ones have
    // advanced.
    assert_eq!((1_u8, 2_u8).span(&cx, &[n, u]), None);
    // A predicate slice reports its own length as its arity.
    let sort = cx.sorts().product([n, n, n]);
    let s = Row::bind(&mut cx, sort);
    let p = cx.set(s.algebras()[0], &[1, 2]);
    assert_eq!(
        symbolic(&cx, &s, (&[p.clone(), p][..], 9_u8)),
        (vec![0, 0, 9], vec![0, 1])
    );
    // A unit component claims no members at all.
    let t = bv(&mut cx, 4);
    assert_eq!(words(&cx, &t, ((), 5_u8)), [5]);
}
