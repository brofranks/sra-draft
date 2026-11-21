//! Public contracts for context binding and owned predicates.
//!
//! The two-node stores start with only the Boolean constants, so producing
//! nonconstant roots exercises maintenance even on the four-bit fixtures.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sra::backend::{Algebra, Backend, BoolOp, Function};
use sra::bdd::BddStore;
use sra::context::{ColumnId, Context, Predicate};
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;
use sra::util::count::Count;

const BACKENDS: [&str; 2] = ["bdd", "sparse"];
const BV1: ScalarSort = ScalarSort::bv(1);
const BV4: ScalarSort = ScalarSort::bv(4);

/// A backend that rejects every sort.
struct Nothing;

impl Backend for Nothing {
    fn bind(self: Arc<Self>, _: ScalarSort) -> Option<Box<dyn Algebra>> {
        None
    }
}

/// A backend that always binds a Boolean algebra.
struct Boolean;

impl Backend for Boolean {
    fn bind(self: Arc<Self>, _: ScalarSort) -> Option<Box<dyn Algebra>> {
        Arc::new(BddStore::with_capacity(2, 0, 1)).bind(ScalarSort::bool())
    }
}

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

fn assert_members(p: &Predicate, holds: impl Fn(u64) -> bool) {
    let mut count = 0;
    for x in 0..16 {
        let expected = holds(x);
        assert_eq!(p.contains(&[x]), expected, "at {x}");
        count += u64::from(expected);
    }
    assert_eq!(p.count(), Count::from(count));
}

/// Probes a token immediately after releasing its last owner, before any
/// allocations can reuse it.
fn assert_released(cx: &Context, c: ColumnId, f: Function) {
    rejects(
        "cannot decrement an unreferenced or reclaimed DAG node",
        || cx.resolve(c).dec_ref(f),
    );
}

// --- registration and binding ----

#[test]
fn the_default_context_binds_both_backends() {
    let mut cx = Context::default();
    for sort in [
        BV4,
        ScalarSort::bool(),
        BV1,
        ScalarSort::bv(70),
        ScalarSort::symbol(),
    ] {
        let columns = BACKENDS.map(|name| cx.bind_column(name, sort));
        assert_ne!(columns[0], columns[1]);
        for c in columns {
            assert_eq!(cx.resolve(c).sort(), sort);
        }
    }
    let [bdd, sparse] = BACKENDS.map(|name| cx.bind_column(name, BV4));
    assert_members(&cx.masked_eq(bdd, &[1], &[1]), |x| x % 2 == 1);
    rejects(
        "this backend cannot construct masked bit predicates",
        || {
            let _ = cx.masked_eq(sparse, &[1], &[1]);
        },
    );
}

#[test]
#[should_panic(expected = "backend name \"bdd\" is already registered")]
fn a_duplicate_backend_name_is_rejected() {
    tiny().register_backend("bdd", Nothing);
}

#[test]
#[should_panic(expected = "backend \"sat\" is not registered")]
fn an_unknown_backend_name_is_rejected() {
    tiny().bind_column("sat", BV4);
}

#[test]
#[should_panic(expected = "cannot change bindings on a shared context")]
fn registering_a_backend_on_a_shared_context_is_rejected() {
    let mut cx = tiny();
    let _shared = cx.clone();
    cx.register_backend("nothing", Nothing);
}

#[test]
fn existing_columns_can_be_rebound_while_predicates_are_live() {
    let mut cx = tiny();
    let columns = BACKENDS.map(|name| cx.bind_column(name, BV4));
    assert_ne!(columns[0], columns[1]);
    assert_ne!(cx.bind_column("bdd", ScalarSort::bv(12)), columns[0]);
    let predicates = columns.map(|c| cx.set(c, &[4]));
    for ((name, c), p) in BACKENDS.into_iter().zip(columns).zip(predicates) {
        assert_eq!(cx.bind_column(name, BV4), c);
        assert_eq!(p.context(), &cx);
        assert_eq!(p.column(), c);
        assert!(p.contains(&[4]));
    }
    let sort = ScalarSort::bv(13);
    let c = cx.bind_column("bdd", sort);
    assert_eq!(cx.resolve(c).sort(), sort);
}

#[test]
#[should_panic(expected = "backend must support the column sort")]
fn an_unsupported_sort_is_rejected() {
    let mut cx = Context::empty();
    cx.register_backend("nothing", Nothing);
    cx.bind_column("nothing", BV4);
}

#[test]
#[should_panic(expected = "backend bound a different scalar sort")]
fn a_binding_reporting_the_wrong_sort_is_rejected() {
    let mut cx = Context::empty();
    cx.register_backend("boolean", Boolean);
    cx.bind_column("boolean", BV1);
}

#[test]
#[should_panic(expected = "cannot change bindings on a shared context")]
fn creating_a_column_on_a_shared_context_is_rejected() {
    let mut cx = tiny();
    let c = cx.bind_column("bdd", BV4);
    let _p = cx.t(c);
    cx.bind_column("bdd", ScalarSort::bv(12));
}

#[test]
fn constant_constructors_reject_a_foreign_column_before_returning() {
    let c = tiny().bind_column("bdd", BV4);
    let mut cx = tiny();
    cx.bind_column("bdd", BV4);
    // An invalid result would panic on drop and hide a missing constructor check.
    let expected = "column belongs to another context";
    rejects(expected, || std::mem::forget(cx.t(c)));
    rejects(expected, || std::mem::forget(cx.f(c)));
}

// --- predicate operations ----

#[test]
fn both_backends_match_concrete_predicates() {
    for name in BACKENDS {
        let mut cx = tiny();
        let c = cx.bind_column(name, BV4);
        let low = cx.lt(c, &[6]);
        let a = cx.set(c, &[1, 3, 7, 9]);
        let b = cx.set(c, &[2, 3, 6, 8]);
        let in_a = |x| [1, 3, 7, 9].contains(&x);
        let in_b = |x| [2, 3, 6, 8].contains(&x);
        assert_members(&cx.t(c), |_| true);
        assert_members(&cx.f(c), |_| false);
        assert_members(&low, |x| x < 6);
        assert_members(&a.and(&b), |x| in_a(x) && in_b(x));
        assert_members(&a.or(&b), |x| in_a(x) || in_b(x));
        assert_members(&a.difference(&b), |x| in_a(x) && !in_b(x));
        assert_members(&low.not(), |x| x >= 6);

        let xor = a.apply(BoolOp::Xor, &b);
        let conditional = low.ite(&a, &b);
        assert_members(&xor, |x| in_a(x) ^ in_b(x));
        assert_members(&conditional, |x| if x < 6 { in_a(x) } else { in_b(x) });
        for p in [xor, conditional] {
            let f = p.function();
            drop(p);
            assert_released(&cx, c, f);
        }
    }
}

#[test]
fn membership_rejects_invalid_encodings_even_for_constants() {
    for name in BACKENDS {
        let mut cx = tiny();
        let c = cx.bind_column(name, BV4);
        for p in [cx.f(c), cx.t(c), cx.set(c, &[1])] {
            for value in [&[16][..], &[]] {
                rejects("value is outside the scalar sort", || {
                    p.contains(value);
                });
            }
        }
    }
}

#[test]
fn predicates_reject_mixed_columns_in_every_operand_position() {
    let mut cx = tiny();
    let sort = ScalarSort::bool();
    let [a, b] = BACKENDS.map(|name| cx.bind_column(name, sort));
    let same_backend = cx.bind_column("bdd", BV1);
    let mut other = tiny();
    let c = other.bind_column("bdd", sort);
    assert_ne!(cx, other);
    let p = cx.set(a, &[1]);
    for q in [
        cx.set(b, &[0]),
        cx.set(same_backend, &[0]),
        other.set(c, &[0]),
    ] {
        let expected = "operands must belong to the same column";
        for (left, right) in [(&p, &q), (&q, &p)] {
            rejects(expected, || {
                let _ = left.or(right);
            });
        }
        for (condition, yes, no) in [(&q, &p, &p), (&p, &q, &p), (&p, &p, &q)] {
            rejects(expected, || {
                let _ = condition.ite(yes, no);
            });
        }
    }
}

// --- root ownership ----

#[test]
fn predicates_outlive_the_original_context_handle() {
    for name in BACKENDS {
        let mut cx = tiny();
        let c = cx.bind_column(name, BV4);
        let p = cx.set(c, &[4]);
        drop(cx);
        assert_members(&p, |x| x == 4);
        assert_eq!(p.context().resolve(c).evaluate(p.function(), &[4]), 1);
    }
}

#[test]
fn clones_and_shortcut_results_each_own_a_reference() {
    for name in BACKENDS {
        let mut cx = tiny();
        let c = cx.bind_column(name, BV4);
        let p = cx.set(c, &[4, 9]);
        let f = p.function();
        let copy = p.clone();
        let intersection = cx.t(c).and(&p);
        let conditional = cx.t(c).ite(&p, &cx.f(c));
        assert_eq!(intersection.function(), f);
        assert_eq!(conditional.function(), f);
        drop(p);

        // Keep distinct roots live to force maintenance while f has owners.
        let pressure: Vec<_> = (0..16).map(|x| cx.set(c, &[x])).collect();
        for p in [copy, intersection, conditional] {
            assert_members(&p, |x| [4, 9].contains(&x));
        }
        assert_released(&cx, c, f);
        drop(pressure);
    }
}
