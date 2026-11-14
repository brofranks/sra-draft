//! Checks for logical declarations: sort interning, product flattening,
//! compatibility, generated functions, and the shared declaration store.
//!
//! Nested binary named records reach scalar-count boundaries without
//! enumerating their leaves.

use std::sync::Barrier;
use std::thread;

use super::{FuncDecl, NodeKind, Sort, SortKind};
use crate::context::Context;
use crate::scalar::ScalarSort;

fn cx() -> Context {
    Context::new()
}

/// A record whose scalar-leaf count is the largest power of two in `usize`.
fn largest_record(cx: &Context) -> Sort {
    (0..usize::BITS - 1).fold(cx.bool_sort(), |sort, i| {
        cx.tuple_decl(format!("T{i}"), [("a", sort), ("b", sort)])
            .sort
    })
}

// --- sorts -------------------------------------------------------------------

#[test]
fn sorts_carry_their_scalar_universe() {
    let cx = cx();
    assert_eq!(cx.bool_sort().scalar(), ScalarSort::bool());
    assert_eq!(cx.symbol_sort("Name").scalar(), ScalarSort::symbol());
    assert_eq!(cx.symbol_sort("Name"), cx.symbol_sort("Name"));
    assert_ne!(cx.symbol_sort("Name"), cx.symbol_sort("Other"));
    assert_eq!(cx.bv_sort(1).scalar(), ScalarSort::bit_vec(1));
    assert_eq!(cx.bv_sort(1), cx.bv_sort(1));
    assert_ne!(cx.bv_sort(1), cx.bool_sort());
    assert_ne!(cx.bv_sort(1), cx.bv_sort(2));
    assert_eq!(cx.bv_sort(8).len, 1);
    assert_eq!(cx.bool_sort().context, cx.id);
    assert_eq!(
        NodeKind::Sort(SortKind::Scalar(ScalarSort::bool())).sort(),
        SortKind::Scalar(ScalarSort::bool())
    );
    assert!(!SortKind::Tuple.is_scalar());
}

#[test]
#[should_panic(expected = "bit-vector sorts must have nonzero width")]
fn zero_width_bit_vector_is_rejected() {
    cx().bv_sort(0);
}

#[test]
#[should_panic(expected = "expected a scalar sort")]
fn the_scalar_of_an_aggregate_is_rejected() {
    cx().unit_sort().scalar();
}

#[test]
#[should_panic(expected = "expected a sort declaration")]
fn a_function_node_as_a_sort_is_rejected() {
    NodeKind::Accessor(0).sort();
}

// --- products and compatibility ----------------------------------------------

#[test]
fn products_flatten_and_collapse() {
    let cx = cx();
    let (a, b) = (cx.bv_sort(8), cx.bv_sort(16));
    assert_eq!(cx.product_sort([a]), a);
    assert_eq!(cx.unit_sort().len, 0);
    assert_eq!(cx.unit_sort().kind, SortKind::Product);
    let ab = cx.product_sort([a, b]);
    assert_eq!(ab.len, 2);
    assert_eq!(cx.product_sort([cx.unit_sort(), ab, cx.unit_sort()]), ab);
    assert_eq!(
        cx.product_sort([cx.product_sort([a, b]), a]),
        cx.product_sort([a, b, a])
    );
    assert_eq!(cx.members(ab), [a, b]);
    assert!(cx.members(a).is_empty());
}

#[test]
#[should_panic(expected = "scalar leaf count overflow")]
fn product_scalar_leaf_count_overflow_is_rejected() {
    let cx = cx();
    let sort = largest_record(&cx);
    assert_eq!(sort.len, 1_usize << (usize::BITS - 1));
    cx.product_sort([sort, sort]);
}

#[test]
#[should_panic(expected = "scalar leaf count overflow")]
fn tuple_scalar_leaf_count_overflow_is_rejected() {
    let cx = cx();
    let sort = largest_record(&cx);
    assert_eq!(sort.len, 1_usize << (usize::BITS - 1));
    cx.tuple_decl("Overflow", [("a", sort), ("b", sort)]);
}

#[test]
fn compatibility_converts_records_and_arguments() {
    let cx = cx();
    let (a, b) = (cx.bv_sort(8), cx.bv_sort(16));
    let pair = cx.tuple_decl("Pair", [("a", a), ("b", b)]).sort;
    let empty = cx.tuple_decl("Empty", Vec::<(&str, Sort)>::new()).sort;
    let other = cx.tuple_decl("Other", Vec::<(&str, Sort)>::new()).sort;
    assert!(cx.compatible(pair, pair));
    assert!(cx.compatible(cx.product_sort([a, pair]), cx.product_sort([a, a, b])));
    assert!(cx.compatible(cx.product_sort([a, b]), pair));
    assert!(!cx.compatible(pair, cx.product_sort([a, a])));
    assert!(!cx.compatible(a, b));
    assert!(cx.compatible(cx.unit_sort(), empty));
    assert!(cx.compatible(empty, cx.unit_sort()));
    assert!(!cx.compatible(empty, other));
    assert!(cx.compatible(cx.product_sort([empty, a]), a));
}

#[test]
fn repeated_records_keep_compatibility_storage_proportional_to_declarations() {
    let cx = cx();
    let boolean = cx.bool_sort();
    let record = cx
        .tuple_decl("Repeated", (0..1024).map(|i| (format!("x{i}"), boolean)))
        .sort;
    let product =
        |last| cx.product_sort(std::iter::repeat_n(record, 2048).chain(std::iter::once(last)));
    let a = product(boolean);
    let b = product(cx.bv_sort(1));
    assert!(!cx.compatible(a, b));
    assert!(!cx.compatible(b, a));
}

#[test]
fn compatibility_is_independent_of_component_order() {
    let cx = cx();
    let a = cx.bv_sort(8);
    let e = cx.tuple_decl("E", Vec::<(&str, Sort)>::new()).sort;
    let o = cx.tuple_decl("O", Vec::<(&str, Sort)>::new()).sort;
    let w = cx.tuple_decl("W", [("e", e)]).sort;
    let p = |sorts: &[Sort]| cx.product_sort(sorts.iter().copied());
    // An argumentless record disappears against nothing, but two different
    // ones never both disappear between the same corresponding components.
    for (x, y, expected) in [
        (p(&[e, o]), o, true),
        (p(&[o, e]), o, true),
        (o, p(&[e, o]), true),
        (p(&[e, w]), w, true),
        (p(&[e, o]), cx.unit_sort(), true),
        (p(&[e, a]), p(&[a, o]), true),
        (p(&[e, a]), p(&[o, a]), false),
        (e, o, false),
        (w, cx.unit_sort(), false),
        (p(&[w, a]), p(&[e, a]), true),
    ] {
        assert_eq!(cx.compatible(x, y), expected, "{x:?} ~ {y:?}");
        assert_eq!(cx.compatible(y, x), expected, "{y:?} ~ {x:?}");
    }
}

#[test]
#[should_panic(expected = "sort belongs to another context")]
fn comparing_a_sort_of_another_context_is_rejected() {
    let foreign = cx().bv_sort(8);
    cx().compatible(foreign, foreign);
}

// --- declarations ------------------------------------------------------------

#[test]
fn declarations_resolve_names_and_signatures() {
    let cx = cx();
    let (a, b) = (cx.bv_sort(8), cx.bv_sort(16));
    let pair = cx.tuple_decl("Pair", [("a", a), ("b", b)]);
    assert_eq!(pair.sort.kind, SortKind::Tuple);
    assert_eq!(pair.sort.len, 2);
    assert_eq!(pair.constructor.name(&cx), "Pair");
    assert_eq!(pair.accessors[1].name(&cx), "b");
    assert_eq!(pair.constructor.resolve(&cx).parent(), pair.sort);
    assert_eq!(
        pair.constructor.resolve(&cx).domain,
        cx.product_sort([a, b])
    );
    assert_eq!(pair.accessors[0].resolve(&cx).member, Some(0));
    assert_eq!(pair.accessors[0].resolve(&cx).parent(), pair.sort);
    assert_eq!(cx.project(pair.sort, 0), pair.accessors[0]);
    let second = cx.project(cx.product_sort([a, b]), 1);
    assert_eq!(second.name(&cx), "project1");
    assert_eq!(second.resolve(&cx).range, b);
    // Identical redeclaration reuses the record; the constructor stays interned.
    assert_eq!(
        cx.tuple_decl("Pair", [("a", a), ("b", b)]).constructor,
        pair.constructor
    );
    assert_eq!(
        cx.func_decl(cx.product_sort([a, b]), pair.sort, None),
        pair.constructor
    );
}

#[test]
#[should_panic(expected = "projection index must name a component of the sort")]
fn an_out_of_bounds_projection_is_rejected() {
    let cx = cx();
    cx.project(cx.product_sort([cx.bv_sort(8), cx.bv_sort(8)]), 2);
}

#[test]
#[should_panic(expected = "projection requires an aggregate sort")]
fn projecting_a_scalar_is_rejected() {
    let cx = cx();
    cx.project(cx.bv_sort(8), 0);
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Pair")]
fn conflicting_member_names_are_rejected() {
    let cx = cx();
    cx.tuple_decl("Pair", [("a", cx.bv_sort(8))]);
    cx.tuple_decl("Pair", [("b", cx.bv_sort(8))]);
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Pair")]
fn conflicting_member_sorts_are_rejected() {
    let cx = cx();
    cx.tuple_decl("Pair", [("a", cx.bv_sort(8))]);
    cx.tuple_decl("Pair", [("a", cx.bv_sort(16))]);
}

#[test]
#[should_panic(expected = "conflicting definition for named sort Name")]
fn a_tuple_reusing_a_symbol_name_is_rejected() {
    let cx = cx();
    cx.symbol_sort("Name");
    cx.tuple_decl("Name", Vec::<(&str, Sort)>::new());
}

#[test]
#[should_panic(expected = "invalid accessor signature")]
fn a_mismatched_accessor_signature_is_rejected() {
    let cx = cx();
    let pair = cx.tuple_decl("Pair", [("a", cx.bv_sort(8))]);
    cx.func_decl(pair.sort, cx.bv_sort(16), Some(0));
}

#[test]
#[should_panic(expected = "constructor requires a tuple sort")]
fn a_scalar_constructor_range_is_rejected() {
    let cx = cx();
    cx.func_decl(cx.bv_sort(8), cx.bv_sort(8), None);
}

#[test]
#[should_panic(expected = "invalid constructor signature")]
fn a_mismatched_constructor_domain_is_rejected() {
    let cx = cx();
    let pair = cx.tuple_decl("Pair", [("a", cx.bv_sort(8))]);
    cx.func_decl(cx.bv_sort(16), pair.sort, None);
}

#[test]
#[should_panic(expected = "declaration belongs to another context")]
fn resolving_in_another_context_is_rejected() {
    let foreign = cx();
    let decl = foreign
        .tuple_decl("Pair", [("a", foreign.bv_sort(8))])
        .constructor;
    decl.resolve(&cx());
}

#[test]
#[should_panic(expected = "expected a function declaration")]
fn forged_declaration_is_rejected() {
    let cx = cx();
    FuncDecl {
        context: cx.id,
        id: cx.bv_sort(8).id,
    }
    .resolve(&cx);
}

// --- store growth and concurrency --------------------------------------------

#[test]
fn declaration_store_grows_past_its_initial_capacity() {
    let cx = cx();
    // Growth reallocates the store while a product's edges are live.
    let pair = cx.product_sort([cx.bv_sort(1), cx.bv_sort(2)]);
    let sorts: Vec<_> = (1..600).map(|w| cx.bv_sort(w)).collect();
    assert_eq!(cx.members(pair), sorts[..2]);
    assert_eq!(
        cx.members(cx.product_sort(sorts.iter().copied())).len(),
        sorts.len()
    );
    assert_eq!(cx.bv_sort(1), sorts[0]);
}

#[test]
fn concurrent_interning_shares_declarations() {
    let cx = cx();
    // Racing threads publish losing candidates that the next growth reclaims.
    let barrier = Barrier::new(4);
    let results = thread::scope(|s| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    (1..600).map(|w| cx.bv_sort(w)).collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(results.iter().all(|r| *r == results[0]));
    assert_eq!(cx.bv_sort(599), results[0][598]);
}

#[test]
fn concurrent_named_declarations_converge() {
    let cx = cx();
    let barrier = Barrier::new(4);
    let results = thread::scope(|s| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    let pair = cx.tuple_decl("Pair", [("a", cx.bv_sort(8))]);
                    (cx.symbol_sort("Name"), pair.sort, pair.accessors)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(results.iter().all(|r| *r == results[0]));
}
