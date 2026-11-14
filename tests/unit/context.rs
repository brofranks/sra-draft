//! Checks for context ownership: backend registration, column interning, id
//! boundaries, and the reference balance of predicate handles.
//!
//! Most checks build a context through `cx()`, which registers a BDD store as
//! `bdd` with a 2-node arena, so nonconstant constructions also exercise arena
//! growth. `bv` declares bit-vector columns on it, `members` enumerates a 4-bit
//! predicate, and `release_sole_reference` checks that a predicate owns exactly
//! one root reference.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use super::{Context, Predicate};
use crate::backend::{Backend, BoolOp, Column};
use crate::bdd::BddStore;
use crate::scalar::ScalarSort;

const BV4: ScalarSort = ScalarSort::bit_vec(4);
const BV12: ScalarSort = ScalarSort::bit_vec(12);

/// A backend that binds no sort.
struct Nothing;

impl Backend for Nothing {
    fn column(self: Arc<Self>, _sort: ScalarSort) -> Option<Box<dyn Column>> {
        None
    }
}

fn cx() -> Context {
    let mut cx = Context::new();
    cx.register_backend("bdd", BddStore::with_capacity(2, 0, 1));
    cx
}

fn bv(cx: &mut Context, width: u32) -> super::ColumnId {
    let bdd = cx.backend("bdd");
    cx.column(bdd, ScalarSort::bit_vec(width))
}

fn members(p: &Predicate) -> Vec<u64> {
    (0..16).filter(|v| p.contains(&[*v])).collect()
}

/// Releases the one reference `p` owns, checking that no further one remains.
fn release_sole_reference(p: Predicate) {
    let (cx, column, f) = p.into_parts();
    cx.resolve(column).dec_ref(f);
    let again = catch_unwind(AssertUnwindSafe(|| cx.resolve(column).dec_ref(f)));
    assert!(again.is_err(), "predicate owned more than one reference");
}

// --- registration and resolution --------------------------------------------

#[test]
fn backends_register_under_unique_names() {
    let mut cx = Context::new();
    let a = cx.register_backend("a", BddStore::default());
    let b = cx.register_backend("b", Nothing);
    assert_eq!(cx.backend("a"), a);
    assert_eq!(cx.backend("b"), b);
    assert_ne!(a, b);
    assert_eq!(a.context, cx.id);
    assert_ne!(Context::new().id, cx.id);
}

#[test]
#[should_panic(expected = "backend \"sat\" is not registered")]
fn an_unknown_backend_name_is_rejected() {
    cx().backend("sat");
}

#[test]
#[should_panic(expected = "column belongs to another context")]
fn a_column_of_another_context_is_rejected() {
    let c = bv(&mut cx(), 4);
    cx().t(c);
}

#[test]
#[should_panic(expected = "backend name \"bdd\" is already registered")]
fn a_duplicate_backend_name_is_rejected() {
    cx().register_backend("bdd", Nothing);
}

// --- columns ----------------------------------------------------------------

#[test]
fn repeated_declarations_intern_one_column() {
    let mut cx = cx();
    let again = cx.register_backend("again", BddStore::default());
    let bdd = cx.backend("bdd");
    let c = cx.column(bdd, BV4);
    assert_eq!(cx.column(bdd, BV4), c);
    assert_ne!(cx.column(bdd, BV12), c);
    assert_ne!(cx.column(again, BV4), c);
    assert_eq!(cx.resolve(c).sort(), BV4);
}

#[test]
#[should_panic(expected = "backend belongs to another context")]
fn a_backend_of_another_context_is_rejected() {
    let bdd = cx().backend("bdd");
    cx().column(bdd, BV4);
}

#[test]
#[should_panic(expected = "backend must support the column sort")]
fn an_unsupported_sort_is_rejected() {
    let mut cx = Context::new();
    let nothing = cx.register_backend("nothing", Nothing);
    cx.column(nothing, BV4);
}

#[test]
#[should_panic(expected = "cannot declare backends or columns on a shared context")]
fn declaring_a_column_on_a_shared_context_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let _p = cx.t(c);
    bv(&mut cx, 12);
}

// --- constructors -----------------------------------------------------------

#[test]
fn constants_need_no_references() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    for _ in 0..8 {
        let t = cx.t(c);
        let f = cx.f(c);
        assert!(t.clone().contains(&[5]));
        assert!(!f.clone().contains(&[5]));
        assert!(cx.apply_bool(BoolOp::And, &t, &f).function.is_constant());
    }
}

#[test]
#[should_panic(expected = "values are not sorted and distinct")]
fn an_unsorted_set_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.set(c, &[3, 1]);
}

#[test]
#[should_panic(expected = "values are not packed at the sort's word stride")]
fn a_set_off_the_word_stride_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 70);
    cx.set(c, &[1, 0, 2]);
}

#[test]
#[should_panic(expected = "value is outside the column sort")]
fn a_set_value_outside_the_sort_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.set(c, &[1, 16]);
}

#[test]
#[should_panic(expected = "bound is outside the column sort")]
fn a_bound_outside_the_sort_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.less_than(c, &[16]);
}

#[test]
#[should_panic(expected = "mask is outside the column sort")]
fn a_mask_outside_the_sort_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.masked_eq(c, &[16], &[0]);
}

#[test]
#[should_panic(expected = "value is outside the column sort")]
fn a_masked_value_outside_the_sort_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.masked_eq(c, &[1], &[16]);
}

#[test]
#[should_panic(expected = "value has bits outside its mask")]
fn a_masked_value_outside_its_mask_is_rejected() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    cx.masked_eq(c, &[1], &[3]);
}

// --- Boolean operations -----------------------------------------------------

#[test]
fn boolean_operations_and_ite_adopt_fresh_references() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let low = cx.less_than(c, &[6]);
    let odd = cx.masked_eq(c, &[1], &[1]);
    let picked = cx.set(c, &[2, 7, 9]);
    assert_eq!(members(&low), [0, 1, 2, 3, 4, 5]);
    assert_eq!(members(&odd), [1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(members(&cx.apply_bool(BoolOp::And, &low, &odd)), [1, 3, 5]);
    assert_eq!(
        members(&cx.apply_bool(BoolOp::AndNot, &low, &odd)),
        [0, 2, 4]
    );
    assert_eq!(
        members(&cx.apply_bool(BoolOp::Xor, &picked, &odd)),
        [1, 2, 3, 5, 11, 13, 15]
    );
    assert_eq!(members(&cx.ite(&low, &odd, &picked)), [1, 3, 5, 7, 9]);
    release_sole_reference(cx.apply_bool(BoolOp::And, &low, &odd));
    release_sole_reference(cx.ite(&low, &odd, &picked));
    drop((low, odd, picked));
    let p = cx.set(c, &[2, 7, 9]);
    assert_eq!(members(&p), [2, 7, 9]);
}

#[test]
#[should_panic(expected = "operands belong to another column")]
fn operands_from_another_column_are_rejected() {
    let mut cx = cx();
    let (a, b) = (bv(&mut cx, 4), bv(&mut cx, 12));
    cx.apply_bool(BoolOp::Or, &cx.set(a, &[1]), &cx.set(b, &[1]));
}

#[test]
fn foreign_context_operands_are_rejected_in_every_position() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let p = cx.t(c);
    let mut other = Context::default();
    let d = bv(&mut other, 4);
    let q = other.t(d);
    for (left, right) in [(&p, &q), (&q, &p)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                cx.apply_bool(BoolOp::Or, left, right)
            }))
            .is_err()
        );
    }
    for (condition, yes, no) in [(&q, &p, &p), (&p, &q, &p), (&p, &p, &q)] {
        assert!(catch_unwind(AssertUnwindSafe(|| cx.ite(condition, yes, no))).is_err());
    }
}

#[test]
fn equal_context_stamps_preserve_operand_owner_checks() {
    let mut a = cx();
    let mut b = cx();
    Arc::get_mut(&mut b.data).unwrap().id = a.id;
    let ac = bv(&mut a, 1);
    let bc = bv(&mut b, 1);
    let p = a.set(ac, &[0]);
    let q = b.set(bc, &[1]);
    assert!(a != b);
    assert_eq!(ac, bc);
    assert_eq!(p.function(), q.function());
    assert!(p.contains(&[0]) && !p.contains(&[1]));
    assert!(!q.contains(&[0]) && q.contains(&[1]));
    let operands = [&p, &q];
    for mask in 1..4 {
        let left = operands[mask & 1];
        let right = operands[(mask >> 1) & 1];
        assert!(
            catch_unwind(AssertUnwindSafe(|| a.apply_bool(BoolOp::Or, left, right))).is_err(),
            "accepted foreign Boolean operand combination {mask}"
        );
    }
    for mask in 1..8 {
        let condition = operands[mask & 1];
        let yes = operands[(mask >> 1) & 1];
        let no = operands[(mask >> 2) & 1];
        assert!(
            catch_unwind(AssertUnwindSafe(|| a.ite(condition, yes, no))).is_err(),
            "accepted foreign conditional operand combination {mask}"
        );
    }
    let owner = a.clone();
    assert!(owner.apply_bool(BoolOp::Or, &p, &p).contains(&[0]));
    assert!(owner.ite(&p, &p, &p).contains(&[0]));
}

// --- predicates -------------------------------------------------------------

#[test]
fn predicates_outlive_the_context_handle() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let p = cx.set(c, &[4]);
    drop(cx);
    assert!(p.contains(&[4]));
    assert!(!p.contains(&[5]));
    drop(p);
}

#[test]
fn membership_rejects_invalid_encodings_even_for_constants() {
    let mut cx = cx();
    let bdd = cx.backend("bdd");
    let boolean = cx.column(bdd, ScalarSort::bool());
    let narrow = cx.column(bdd, BV4);
    let wide = cx.column(bdd, ScalarSort::bit_vec(70));
    let check = |column, valid: &[u64], invalid: &[&[u64]]| {
        for p in [cx.t(column), cx.set(column, valid)] {
            assert!(p.contains(valid));
            for &value in invalid {
                assert!(catch_unwind(AssertUnwindSafe(|| p.contains(value))).is_err());
            }
        }
    };
    check(boolean, &[1], &[&[], &[2], &[1, 0]]);
    check(narrow, &[1], &[&[], &[16], &[1, 0]]);
    check(wide, &[1, 1], &[&[], &[1], &[1, 64], &[1, 1, 0]]);
}

#[test]
fn raw_parts_retain_the_last_context_owner() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let p = cx.set(c, &[2, 7]);
    let owner = Arc::downgrade(&cx.data);
    assert!(p.context() == &cx);
    assert_eq!(p.column(), c);
    let f = p.function();
    drop(cx);
    let (cx, column, root) = p.into_parts();
    assert!(owner.upgrade().is_some());
    assert_eq!((column, root), (c, f));
    assert_eq!(cx.resolve(column).evaluate(root, &[7]), 1);
    assert_eq!(cx.resolve(column).evaluate(root, &[3]), 0);
    cx.resolve(column).dec_ref(root);
    drop(cx);
    assert!(owner.upgrade().is_none());
}

#[test]
fn clone_and_drop_balance_one_reference() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let p = cx.set(c, &[1, 3]);
    let q = p.clone();
    let (owner, column, f) = p.clone().into_parts();
    assert!(owner == cx);
    assert_eq!(column, c);
    drop(p);
    drop(q);
    assert_eq!(cx.resolve(c).evaluate(f, &[3]), 1);
    release_sole_reference(cx.adopt(c, f));
}
