//! Public checks for logical declarations, interning, and signatures.
//!
//! Nested binary records reach scalar-count boundaries without enumerating
//! their leaves.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Barrier;
use std::thread;

use sra::ast::{FuncInfo, Sort, SortKind, Sorts};
use sra::context::Context;
use sra::scalar::ScalarSort;

fn largest_record(sorts: &Sorts) -> Sort {
    (0..usize::BITS - 1).fold(sorts.bool(), |sort, i| {
        sorts.record(&format!("T{i}"), &[("a", sort), ("b", sort)])
    })
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

// --- scalars ----

#[test]
fn sorts_carry_their_scalar_universe() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    assert_eq!(sorts.bool().scalar(), ScalarSort::bool());
    assert_eq!(sorts.symbol("Name").scalar(), ScalarSort::symbol());
    assert_eq!(sorts.symbol("Name"), sorts.symbol("Name"));
    assert_ne!(sorts.symbol("Name"), sorts.symbol("Other"));
    assert_eq!(sorts.bv(1).scalar(), ScalarSort::bv(1));
    assert_eq!(sorts.bv(1), sorts.bv(1));
    assert_ne!(sorts.bv(1), sorts.bool());
    assert_ne!(sorts.bv(1), sorts.bv(2));
    assert_eq!(sorts.bv(8).scalar_len(), 1);
}

#[test]
fn cloned_contexts_share_declarations() {
    let cx = Context::empty();
    let shared = cx.clone();
    let other = Context::empty();
    let empty = shared.sorts().record("Empty", &[]);
    assert_eq!(empty, cx.sorts().record("Empty", &[]));
    assert_ne!(empty, other.sorts().record("Empty", &[]));
}

#[test]
#[should_panic(expected = "bit-vector sorts must have nonzero width")]
fn zero_width_bit_vectors_are_rejected() {
    Context::empty().sorts().bv(0);
}

#[test]
#[should_panic(expected = "expected a scalar sort")]
fn the_scalar_of_an_aggregate_is_rejected() {
    Context::empty().sorts().unit().scalar();
}

// --- products ----

#[test]
fn products_flatten_and_collapse() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let (a, b) = (sorts.bv(8), sorts.bv(16));
    let unit = sorts.unit();
    assert_eq!(sorts.product([a]), a);
    assert_eq!(unit.scalar_len(), 0);
    assert_eq!(unit.kind(), SortKind::Product);
    let ab = sorts.product([a, b]);
    assert_eq!(ab.scalar_len(), 2);
    assert_eq!(sorts.product([unit, ab, unit]), ab);
    assert_eq!(sorts.product([ab, a]), sorts.product([a, b, a]));
    assert_eq!(sorts.members(ab), [a, b]);
    assert!(sorts.members(a).is_empty());

    let empty = sorts.record("Empty", &[]);
    let record = sorts.record("Pair", &[("a", a), ("b", b)]);
    assert_ne!(empty, unit);
    assert_ne!(empty, sorts.record("Other", &[]));
    assert_eq!(empty.scalar_len(), 0);
    assert_eq!(
        sorts.members(sorts.product([a, empty, record])),
        [a, empty, record]
    );
}

#[test]
#[should_panic(expected = "scalar leaf count overflow")]
fn product_scalar_leaf_count_overflow_is_rejected() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let sort = largest_record(sorts);
    assert_eq!(sort.scalar_len(), 1_usize << (usize::BITS - 1));
    sorts.product([sort, sort]);
}

#[test]
fn overflowing_record_is_rejected_without_reserving_the_name() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let sort = largest_record(sorts);
    assert_eq!(sort.scalar_len(), 1_usize << (usize::BITS - 1));
    rejects("scalar leaf count overflow", || {
        sorts.record("Overflow", &[("a", sort), ("b", sort)]);
    });
    assert_eq!(
        sorts.members(sorts.record("Overflow", &[("a", sort)])),
        [sort]
    );
}

#[test]
#[should_panic(expected = "declaration belongs to another store")]
fn foreign_product_components_are_rejected() {
    let foreign = Context::empty().sorts().unit();
    Context::empty().sorts().product([foreign]);
}

#[test]
#[should_panic(expected = "declaration belongs to another store")]
fn foreign_members_on_redeclaration_are_rejected() {
    let cx = Context::empty();
    let other = Context::empty();
    let sorts = cx.sorts();
    let local = sorts.bool();
    let foreign = other.sorts().bool();
    sorts.record("Record", &[("member", local)]);
    sorts.record("Record", &[("member", foreign)]);
}

// --- records and projections ----

#[test]
fn declarations_resolve_names_and_signatures() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let (a, b) = (sorts.bv(8), sorts.bv(16));
    let ab = sorts.product([a, b]);
    let members = [("pair", ab), ("tail", b)];
    let record = sorts.record("Record", &members);
    assert_eq!(record.kind(), SortKind::Record);
    assert_eq!(record.scalar_len(), 3);
    assert_eq!(sorts.members(record), [ab, b]);

    let ctor = sorts.constructor(record);
    assert_eq!(ctor.name(sorts), "Record");
    assert_eq!(
        ctor.resolve(sorts),
        FuncInfo {
            domain: sorts.product([a, b, b]),
            range: record,
            member: None,
        },
    );
    let first = sorts.project(record, 0);
    assert_eq!(first.name(sorts), "pair");
    assert_eq!(
        first.resolve(sorts),
        FuncInfo {
            domain: record,
            range: ab,
            member: Some(0),
        },
    );
    assert_eq!(sorts.constructor(sorts.record("Record", &members)), ctor);
    assert_eq!(sorts.project(sorts.record("Record", &members), 0), first);

    let second = sorts.project(ab, 1);
    assert_eq!(second.name(sorts), "project1");
    assert_eq!(second.resolve(sorts).range, b);

    let empty = sorts.record("Empty", &[]);
    assert!(sorts.members(empty).is_empty());
    assert_eq!(sorts.constructor(empty).resolve(sorts).domain, sorts.unit());
}

#[test]
#[should_panic(expected = "projection index must name a component of the sort")]
fn a_projection_past_the_declared_members_is_rejected() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    sorts.project(sorts.record("Record", &[("member", sorts.bool())]), 1);
}

#[test]
#[should_panic(expected = "projection requires an aggregate sort")]
fn projecting_a_scalar_is_rejected() {
    let cx = Context::empty();
    cx.sorts().project(cx.sorts().bool(), 0);
}

#[test]
#[should_panic(expected = "constructor requires a record")]
fn constructing_an_anonymous_product_is_rejected() {
    let cx = Context::empty();
    cx.sorts().constructor(cx.sorts().unit());
}

// --- conflicting definitions ----

#[test]
#[should_panic(expected = "conflicting definition for named sort Pair")]
fn conflicting_member_names_are_rejected() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    sorts.record("Pair", &[("a", sorts.bv(8))]);
    sorts.record("Pair", &[("b", sorts.bv(8))]);
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Pair")]
fn a_matching_prefix_with_different_arity_is_rejected() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let flag = sorts.bool();
    sorts.record("Pair", &[("a", flag)]);
    sorts.record("Pair", &[("a", flag), ("b", flag)]);
}

#[test]
fn conflicting_member_sorts_leave_the_original_definition_usable() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let members = [("a", sorts.bv(8))];
    let record = sorts.record("Record", &members);
    let other = [("a", sorts.bv(16))];
    rejects("conflicting definition for named sort Record", || {
        sorts.record("Record", &other);
    });
    assert_eq!(sorts.record("Record", &members), record);
    assert_ne!(sorts.record("Other", &other), record);
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Name")]
fn a_record_reusing_a_symbol_name_is_rejected() {
    let cx = Context::empty();
    cx.sorts().symbol("Name");
    cx.sorts().record("Name", &[]);
}

// --- ownership ----

#[test]
#[should_panic(expected = "declaration belongs to another store")]
fn resolving_in_another_context_is_rejected() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let decl = sorts.constructor(sorts.record("Empty", &[]));
    decl.resolve(Context::empty().sorts());
}

#[test]
fn foreign_records_and_declarations_are_rejected_at_each_entry_point() {
    let cx = Context::empty();
    let other = Context::empty();
    let sorts = cx.sorts();
    let foreign = other.sorts();
    // Give both records the same slot layout so a missing check can return
    // a result from the local store.
    sorts.record("Local", &[("flag", sorts.bool())]);
    let record = foreign.record("Remote", &[("flag", foreign.bool())]);
    let decl = foreign.constructor(record);

    let reason = "declaration belongs to another store";
    rejects(reason, || {
        sorts.members(record);
    });
    rejects(reason, || {
        sorts.constructor(record);
    });
    rejects(reason, || {
        sorts.project(record, 0);
    });
    rejects(reason, || {
        decl.name(sorts);
    });
}

// --- record and argument coercion ----

#[test]
fn records_convert_directly_to_their_declared_arguments() {
    let cx = Context::empty();
    let s = cx.sorts();
    let (a, b) = (s.bv(8), s.bv(16));
    let args = s.product([a, b]);
    let fields = [("a", a), ("b", b)];
    let pair = s.record("Pair", &fields);
    let other = s.record("Other", &fields);
    let nested = s.record("Nested", &[("p", pair)]);
    for (left, right, expected) in [
        (pair, pair, true),
        (pair, args, true),
        (other, args, true),
        (pair, other, false),
        (nested, pair, true),
        (nested, args, false),
        (s.product([a, pair]), s.product([a, a, b]), true),
        (pair, s.product([b, a]), false),
        (pair, a, false),
        (s.bool(), s.bv(1), false),
        (s.symbol("X"), s.symbol("Y"), false),
    ] {
        assert_eq!(s.coercible(left, right), expected);
        assert_eq!(s.coercible(right, left), expected);
    }
}

#[test]
fn zero_width_record_conversion_is_independent_of_component_order() {
    let cx = Context::empty();
    let s = cx.sorts();
    let a = s.bv(8);
    let e = s.record("E", &[]);
    let o = s.record("O", &[]);
    let w = s.record("W", &[("e", e)]);
    // An argumentless record disappears against nothing, but two different
    // ones never both disappear between the same corresponding components.
    for (left, right, expected) in [
        (s.product([e, o]), o, true),
        (s.product([o, e]), o, true),
        (s.product([e, w]), w, true),
        (s.product([e, o]), s.unit(), true),
        (s.product([e, a]), s.product([a, o]), true),
        (s.product([e, a]), s.product([o, a]), false),
        (e, o, false),
        (w, s.unit(), false),
        (s.product([w, a]), s.product([e, a]), true),
    ] {
        assert_eq!(s.coercible(left, right), expected);
        assert_eq!(s.coercible(right, left), expected);
    }
}

#[test]
fn repeated_records_check_the_final_scalar_after_the_shared_prefix() {
    let cx = Context::empty();
    let s = cx.sorts();
    let boolean = s.bool();
    let (members, repeats) = if cfg!(miri) { (32, 64) } else { (1024, 2048) };
    let record = s.record("Repeated", &vec![("x", boolean); members]);
    let product = |last| s.product(std::iter::repeat_n(record, repeats).chain([last]));
    let a = product(boolean);
    let b = product(s.bv(1));
    assert!(!s.coercible(a, b));
    assert!(!s.coercible(b, a));
}

#[test]
fn coercion_checks_both_owners_before_comparing_sorts() {
    let cx = Context::empty();
    let s = cx.sorts();
    let other = Context::empty();
    let foreign = other.sorts().bool();
    for (a, b) in [(foreign, foreign), (s.unit(), foreign), (foreign, s.unit())] {
        rejects("declaration belongs to another store", || {
            s.coercible(a, b);
        });
    }
}

// --- growth and concurrency ----

#[test]
fn declarations_survive_store_growth() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let a = sorts.bv(1);
    let b = sorts.bv(2);
    // Growth reallocates the store while the record's edges are live.
    let record = sorts.record("Pair", &[("a", a), ("b", b)]);
    let ctor = sorts.constructor(record);
    let product = sorts.product((1..600).map(|width| sorts.bv(width)));
    assert_eq!(sorts.members(product).len(), 599);
    assert_eq!(sorts.members(record), [a, b]);
    assert_eq!(ctor.resolve(sorts).domain, sorts.product([a, b]));
    assert_eq!(sorts.bv(1), a);
}

#[test]
fn concurrent_interning_shares_declarations() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let barrier = Barrier::new(4);
    let results = thread::scope(|s| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    (1..600).map(|w| sorts.bv(w)).collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(results.iter().all(|r| *r == results[0]));
    assert_eq!(sorts.bv(599), results[0][598]);
}

#[test]
fn concurrent_named_declarations_converge() {
    let cx = Context::empty();
    let sorts = cx.sorts();
    let barrier = Barrier::new(4);
    let results = thread::scope(|s| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    let pair = sorts.record("Pair", &[("a", sorts.bv(8))]);
                    (sorts.symbol("Name"), pair, sorts.project(pair, 0))
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(results.iter().all(|r| *r == results[0]));
}
