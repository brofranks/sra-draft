//! Public contracts for context binding and owned predicates.
//!
//! The two-node stores start with only the Boolean constants, so producing
//! nonconstant roots exercises maintenance even on the four-bit fixtures.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use sra::backend::{Algebra, Backend, BoolOp, Function};
use sra::bdd::BddStore;
use sra::context::{AlgebraId, Context, Predicate};
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
fn assert_released(cx: &Context, id: AlgebraId, f: Function) {
    rejects(
        "cannot decrement an unreferenced or reclaimed DAG node",
        || cx.resolve(id).dec_ref(f),
    );
}

// --- registration and binding ----

#[test]
fn the_default_context_provides_both_backends() {
    let mut cx = Context::default();
    for sort in [
        BV4,
        ScalarSort::bool(),
        BV1,
        ScalarSort::bv(70),
        ScalarSort::symbol(),
    ] {
        let ids = BACKENDS.map(|name| cx.bind(name, sort));
        assert_ne!(ids[0], ids[1]);
        for id in ids {
            assert_eq!(cx.resolve(id).sort(), sort);
        }
    }
    let [bdd, sparse] = BACKENDS.map(|name| cx.bind(name, BV4));
    assert_members(&cx.masked_eq(bdd, &[3], &[1]), |x| x % 4 == 1);
    rejects(
        "this backend cannot construct masked bit predicates",
        || drop(cx.masked_eq(sparse, &[3], &[1])),
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
    tiny().bind("sat", BV4);
}

#[test]
#[should_panic(expected = "cannot change bindings on a shared context")]
fn registering_a_backend_on_a_shared_context_is_rejected() {
    let mut cx = tiny();
    let _shared = cx.clone();
    cx.register_backend("nothing", Nothing);
}

#[test]
fn existing_algebra_bindings_are_reused_while_predicates_are_live() {
    let mut cx = tiny();
    let ids = BACKENDS.map(|name| cx.bind(name, BV4));
    assert_ne!(ids[0], ids[1]);
    assert_ne!(cx.bind("bdd", ScalarSort::bv(12)), ids[0]);
    let predicates = ids.map(|id| cx.set(id, &[4]));
    for ((name, id), p) in BACKENDS.into_iter().zip(ids).zip(predicates) {
        assert_eq!(cx.bind(name, BV4), id);
        assert_eq!(p.context(), &cx);
        assert_eq!(p.algebra(), id);
        assert!(p.contains(&[4]));
    }
    let sort = ScalarSort::bv(13);
    let id = cx.bind("bdd", sort);
    assert_eq!(cx.resolve(id).sort(), sort);
}

#[test]
#[should_panic(expected = "backend must support the requested sort")]
fn an_unsupported_sort_is_rejected() {
    let mut cx = Context::empty();
    cx.register_backend("nothing", Nothing);
    cx.bind("nothing", BV4);
}

#[test]
#[should_panic(expected = "backend bound a different scalar sort")]
fn a_binding_reporting_the_wrong_sort_is_rejected() {
    let mut cx = Context::empty();
    cx.register_backend("boolean", Boolean);
    cx.bind("boolean", BV1);
}

#[test]
#[should_panic(expected = "cannot change bindings on a shared context")]
fn creating_an_algebra_on_a_shared_context_is_rejected() {
    let mut cx = tiny();
    let id = cx.bind("bdd", BV4);
    let _p = cx.t(id);
    cx.bind("bdd", ScalarSort::bv(12));
}

// --- field binding ----

#[test]
fn selections_belong_to_individual_record_members() {
    let mut cx = tiny();
    let sorts = cx.sorts();
    let members = [("x", sorts.bv(4)), ("x", sorts.bv(4))];
    let pair = sorts.record("Pair", &members);
    let a = sorts.project(pair, 0);
    let b = sorts.project(pair, 1);
    let other = sorts.project(sorts.record("Other", &members), 0);
    assert_eq!(cx.algebra("bdd", BV4), None);
    assert_eq!(cx.field(a), None);
    let bdd = cx.bind_field(a, "bdd");
    assert_eq!(cx.algebra("bdd", BV4), Some(bdd));
    assert_eq!(cx.field(b), None);
    assert_eq!(cx.field(other), None);
    rejects("already bound to a different backend", || {
        cx.bind_field(a, "sparse");
    });
    assert_eq!(cx.algebra("sparse", BV4), None);
    let sparse = cx.bind_field(b, "sparse");
    assert_ne!(bdd, sparse);
    assert_eq!(cx.bind_field(other, "bdd"), bdd);

    let p = cx.set(bdd, &[1, 3]);
    let again = cx.sorts().record("Pair", &members);
    assert_eq!(cx.bind_field(cx.sorts().project(again, 0), "bdd"), bdd);
    assert_eq!(cx.field(b), Some(sparse));
    assert!(p.contains(&[3]));
}

#[test]
fn fields_require_local_scalar_record_projections() {
    let mut cx = tiny();
    let sorts = cx.sorts();
    let a = sorts.bool();
    let r = sorts.record("R", &[("a", a)]);
    let nested = sorts.record("Nested", &[("r", r)]);
    for decl in [
        sorts.constructor(r),
        sorts.project(nested, 0),
        sorts.project(sorts.product([a, a]), 0),
    ] {
        let reason = "field must project a scalar record member";
        rejects(reason, || {
            cx.field(decl);
        });
        rejects(reason, || {
            cx.bind_field(decl, "sparse");
        });
    }

    let other = Context::empty();
    let sorts = other.sorts();
    let decl = sorts.project(sorts.record("R", &[("a", sorts.bool())]), 0);
    let reason = "declaration belongs to another store";
    rejects(reason, || {
        cx.field(decl);
    });
    rejects(reason, || {
        cx.bind_field(decl, "bdd");
    });
}

#[test]
fn a_new_field_needs_exclusive_access_even_when_its_algebra_exists() {
    let mut cx = tiny();
    let id = cx.bind("sparse", BV4);
    let sorts = cx.sorts();
    let field = sorts.project(sorts.record("R", &[("a", sorts.bv(4))]), 0);
    let p = cx.t(id);
    rejects("cannot change bindings on a shared context", || {
        cx.bind_field(field, "sparse");
    });
    assert_eq!(cx.field(field), None);
    drop(p);
    assert_eq!(cx.bind_field(field, "sparse"), id);
}

// --- predicate operations ----

#[test]
fn constant_constructors_reject_a_foreign_algebra_before_returning() {
    let id = tiny().bind("bdd", BV4);
    let mut cx = tiny();
    cx.bind("bdd", BV4);
    // An invalid result would panic on drop and hide a missing constructor check.
    let expected = "algebra belongs to another context";
    rejects(expected, || std::mem::forget(cx.t(id)));
    rejects(expected, || std::mem::forget(cx.f(id)));
}

#[test]
fn malformed_sets_are_rejected() {
    for name in BACKENDS {
        let mut cx = tiny();
        let narrow = cx.bind(name, BV4);
        let wide = cx.bind(name, ScalarSort::bv(70));
        rejects("values are not sorted and distinct", || {
            drop(cx.set(narrow, &[3, 1]));
        });
        rejects("value is outside the scalar sort", || {
            drop(cx.set(narrow, &[1, 16]));
        });
        rejects("values are not packed at the sort's word stride", || {
            drop(cx.set(wide, &[1, 0, 2]));
        });
    }
}

#[test]
fn invalid_numeric_cuts_are_rejected() {
    for name in BACKENDS {
        let mut cx = tiny();
        let id = cx.bind(name, BV4);
        let symbol = cx.bind(name, ScalarSort::symbol());
        rejects("bound is outside the scalar sort", || {
            drop(cx.lt(id, &[16]));
        });
        rejects("symbols have no numeric ordering", || {
            drop(cx.lt(symbol, &[0]));
        });
    }
}

#[test]
fn malformed_masked_equalities_are_rejected() {
    let mut cx = tiny();
    let id = cx.bind("bdd", BV4);
    let outside = "bit test is outside the scalar sort";
    rejects(outside, || drop(cx.masked_eq(id, &[16], &[0])));
    rejects(outside, || drop(cx.masked_eq(id, &[1], &[16])));
    rejects("bit test value has bits outside its mask", || {
        drop(cx.masked_eq(id, &[1], &[3]));
    });
}

#[test]
fn both_backends_match_concrete_predicates() {
    for name in BACKENDS {
        let mut cx = tiny();
        let id = cx.bind(name, BV4);
        let low = cx.lt(id, &[6]);
        let a = cx.set(id, &[1, 3, 7, 9]);
        let b = cx.set(id, &[2, 3, 6, 8]);
        let in_a = |x| [1, 3, 7, 9].contains(&x);
        let in_b = |x| [2, 3, 6, 8].contains(&x);
        assert_members(&cx.t(id), |_| true);
        assert_members(&cx.f(id), |_| false);
        assert_members(&low, |x| x < 6);
        assert_members(&a.and(&b), |x| in_a(x) && in_b(x));
        assert_members(&a.or(&b), |x| in_a(x) || in_b(x));
        assert_members(&a.difference(&b), |x| in_a(x) && !in_b(x));

        let high = low.not();
        let xor = a.apply(BoolOp::Xor, &b);
        let conditional = low.ite(&a, &b);
        assert_members(&high, |x| x >= 6);
        assert_members(&xor, |x| in_a(x) ^ in_b(x));
        assert_members(&conditional, |x| if x < 6 { in_a(x) } else { in_b(x) });
        for p in [high, xor, conditional] {
            let f = p.function();
            drop(p);
            assert_released(&cx, id, f);
        }
    }
}

#[test]
fn membership_rejects_invalid_encodings_even_for_constants() {
    for name in BACKENDS {
        let mut cx = tiny();
        let id = cx.bind(name, BV4);
        for p in [cx.f(id), cx.t(id), cx.set(id, &[1])] {
            for value in [&[16][..], &[]] {
                rejects("value is outside the scalar sort", || {
                    p.contains(value);
                });
            }
        }
    }
}

#[test]
fn predicates_reject_mixed_algebras_in_every_operand_position() {
    let mut cx = tiny();
    let sort = ScalarSort::bool();
    let [a, b] = BACKENDS.map(|name| cx.bind(name, sort));
    let same_backend = cx.bind("bdd", BV1);
    let mut other = tiny();
    let foreign = other.bind("bdd", sort);
    assert_ne!(cx, other);
    let p = cx.set(a, &[1]);
    for q in [
        cx.set(b, &[0]),
        cx.set(same_backend, &[0]),
        other.set(foreign, &[0]),
    ] {
        let expected = "operands must belong to the same algebra";
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
        let id = cx.bind(name, BV4);
        let p = cx.set(id, &[4]);
        drop(cx);
        assert_members(&p, |x| x == 4);
        assert_eq!(p.context().resolve(id).evaluate(p.function(), &[4]), 1);
    }
}

#[test]
fn clones_and_shortcut_results_each_own_a_reference() {
    for name in BACKENDS {
        let mut cx = tiny();
        let id = cx.bind(name, BV4);
        let p = cx.set(id, &[4, 9]);
        let f = p.function();
        let copy = p.clone();
        let intersection = cx.t(id).and(&p);
        let conditional = cx.t(id).ite(&p, &cx.f(id));
        assert_eq!(intersection.function(), f);
        assert_eq!(conditional.function(), f);
        drop(p);

        // Keep distinct roots live to force maintenance while f has owners.
        let pressure: Vec<_> = (0..16).map(|x| cx.set(id, &[x])).collect();
        for owner in [copy, intersection, conditional] {
            assert_members(&owner, |x| [4, 9].contains(&x));
        }
        assert_released(&cx, id, f);
        drop(pressure);
    }
}
