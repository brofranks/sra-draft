//! Public contracts for bound rows, using small stores and explicit layouts.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sra::bdd::BddStore;
use sra::context::Context;
use sra::row::Row;
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
