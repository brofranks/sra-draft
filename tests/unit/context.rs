//! Checks for context ownership: backend registration, column interning, id
//! boundaries, the reference balance of predicate handles, scalar leaf
//! binding, and symbol interners.
//!
//! Most checks build a context through `cx()`, which registers a BDD store as
//! `bdd` with a 2-node arena, so nonconstant constructions also exercise arena
//! growth. `bv` declares bit-vector columns on it, `below` enumerates a
//! one-word predicate below a bound, and `release_sole_reference` checks that a
//! predicate owns exactly one root reference.
//!
//! The sparse and BDD agreement checks build a context through `both()`, which
//! registers a tiny sparse table and a tiny BDD store, so both collect and
//! grow. `pair` declares one column of a sort on each, and `slots` reads the
//! slot of every value. The checks compare the two backends against each other
//! and against plain arithmetic.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use super::{Context, Field, Predicate};
use crate::ast::{Sort, TupleDecl};
use crate::backend::{Alignment, Backend, BoolOp, Column, Function, Refinement, Values};
use crate::bdd::BddStore;
use crate::scalar::ScalarSort;
use crate::sparse::SparseTable;

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

/// The members of a one-word predicate below `size`.
fn below(p: &Predicate, size: u64) -> Vec<u64> {
    (0..size).filter(|v| p.contains(&[*v])).collect()
}

/// Releases the one reference `p` owns, checking that no further one remains.
fn release_sole_reference(p: Predicate) {
    let (cx, column, f) = p.into_parts();
    cx.resolve(column).dec_ref(f);
    let again = catch_unwind(AssertUnwindSafe(|| cx.resolve(column).dec_ref(f)));
    assert!(again.is_err(), "predicate owned more than one reference");
}

// --- registration and resolution ---------------------------------------------

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

// --- columns -----------------------------------------------------------------

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

// --- fields ------------------------------------------------------------------

const BV8: ScalarSort = ScalarSort::bit_vec(8);

/// A record with a sparse 4-bit member and a BDD 8-bit member.
fn record(cx: &mut Context) -> TupleDecl {
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.accessors[1], bdd);
    t
}

#[test]
fn members_bind_once_to_their_selected_backend() {
    let mut cx = Context::default();
    let t = record(&mut cx);
    let a = cx.bind_field(t.accessors[0]);
    let b = cx.bind_field(t.accessors[1]);
    assert_ne!(a, b);
    assert_eq!(cx.bind_field(t.accessors[0]), a);
    assert_eq!(cx.bind_field(t.accessors[1]), b);
    assert_eq!(cx.field(t.accessors[0]), a);
    assert_eq!(cx.field(t.accessors[1]), b);
    assert_eq!(a, cx.column(cx.backend("sparse"), BV4));
    assert_eq!(b, cx.column(cx.backend("bdd"), BV8));
    // An anonymous scalar keyed by its sort binds the sparse default.
    let sort = cx.bv_sort(4);
    assert_eq!(cx.bind_field(sort), a);
    assert_eq!(cx.field(sort), a);
}

#[test]
fn column_identity_is_separate_from_logical_identity() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4))]);
    let u = cx.tuple_decl("U", [("a", cx.bv_sort(4))]);
    assert_ne!(t.accessors[0], u.accessors[0]);
    assert!(!cx.compatible(t.sort, u.sort));
    assert_eq!(cx.bind_field(t.accessors[0]), cx.bind_field(u.accessors[0]));
}

#[test]
fn bound_fields_hold_predicates() {
    let mut cx = Context::default();
    let t = record(&mut cx);
    for (i, value) in [(0, 3), (1, 200)] {
        let c = cx.bind_field(t.accessors[i]);
        let p = cx.set(c, &[value]);
        assert_eq!(p.column(), c);
        assert!(p.contains(&[value]));
        assert!(!p.contains(&[value - 1]));
    }
}

#[test]
#[should_panic(expected = "field representation is already bound")]
fn representation_after_binding_is_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4))]);
    cx.bind_field(t.accessors[0]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.accessors[0], bdd);
}

#[test]
#[should_panic(expected = "field representation is already bound")]
fn a_second_representation_is_rejected() {
    let mut cx = Context::default();
    let t = record(&mut cx);
    let sparse = cx.backend("sparse");
    cx.set_representation(t.accessors[1], sparse);
}

#[test]
#[should_panic(expected = "representation requires a declared scalar member")]
fn a_constructor_representation_is_rejected() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4))]);
    let bdd = cx.backend("bdd");
    cx.set_representation(t.constructor, bdd);
}

#[test]
#[should_panic(expected = "representation requires a declared scalar member")]
fn a_nested_record_representation_is_rejected() {
    let mut cx = Context::default();
    let inner = cx.tuple_decl("Inner", [("a", cx.bv_sort(4))]);
    let outer = cx.tuple_decl("Outer", [("inner", inner.sort)]);
    let bdd = cx.backend("bdd");
    cx.set_representation(outer.accessors[0], bdd);
}

#[test]
fn field_binding_derives_the_declared_scalar_sort() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bool_sort())]);
    for (field, sort) in [
        (Field::from(t.accessors[0]), BV4),
        (t.accessors[1].into(), ScalarSort::bool()),
        (cx.bv_sort(8).into(), BV8),
        (cx.symbol_sort("Name").into(), ScalarSort::symbol()),
    ] {
        let column = cx.bind_field(field);
        assert_eq!(cx.resolve(column).sort(), sort);
        assert_eq!(cx.bind_field(field), column);
    }
}

#[test]
fn field_binding_rejects_nonleaf_declarations() {
    let mut cx = Context::default();
    let t = cx.tuple_decl("T", [("a", cx.bv_sort(4))]);
    let u = cx.tuple_decl("U", [("t", t.sort)]);
    let product = cx.product_sort([cx.bv_sort(4), cx.bv_sort(8)]);
    for field in [
        Field::from(t.sort),
        product.into(),
        t.constructor.into(),
        u.accessors[0].into(),
        cx.project(product, 0).into(),
    ] {
        let e = catch_unwind(AssertUnwindSafe(|| cx.bind_field(field))).unwrap_err();
        assert_eq!(
            e.downcast_ref::<&str>(),
            Some(&"field binding requires a scalar sort or record accessor")
        );
    }
    let column = cx.bind_field(t.accessors[0]);
    assert_eq!(cx.resolve(column).sort(), BV4);
}

#[test]
#[should_panic(expected = "field must be bound")]
fn an_unbound_field_is_rejected() {
    let cx = Context::default();
    cx.field(cx.bv_sort(4));
}

#[test]
fn redeclaration_and_sharing_keep_bound_members() {
    let mut cx = Context::default();
    let t = record(&mut cx);
    let (a, b) = (cx.bind_field(t.accessors[0]), cx.bind_field(t.accessors[1]));
    let sort = cx.bv_sort(4);
    let plain = cx.bind_field(sort);
    let again = cx.tuple_decl("T", [("a", cx.bv_sort(4)), ("b", cx.bv_sort(8))]);
    assert_eq!(again.accessors, t.accessors);
    let _shared = cx.clone();
    assert_eq!(cx.bind_field(again.accessors[0]), a);
    assert_eq!(cx.bind_field(again.accessors[1]), b);
    assert_eq!(cx.bind_field(cx.bv_sort(4)), plain);
}

#[test]
#[should_panic(expected = "field belongs to another context")]
fn a_field_of_another_context_is_rejected() {
    let mut cx = Context::default();
    let other = Context::default();
    cx.bind_field(other.bv_sort(4));
}

#[test]
#[should_panic(expected = "cannot declare backends or columns on a shared context")]
fn binding_on_a_shared_context_is_rejected() {
    let mut cx = Context::default();
    let _shared = cx.clone();
    let sort = cx.bv_sort(4);
    cx.bind_field(sort);
}

#[test]
#[should_panic(expected = "backend \"sparse\" is not registered")]
fn a_default_without_a_sparse_backend_is_rejected() {
    let mut cx = cx();
    let sort = cx.bv_sort(4);
    cx.bind_field(sort);
}

// --- interners ---------------------------------------------------------------

#[test]
fn interners_assign_stable_ids() {
    let cx = Context::new();
    let names = cx.interner::<str>(cx.symbol_sort("Name"));
    assert_eq!(names.get("a"), None);
    assert_eq!(names.intern("a"), 0);
    assert_eq!(names.intern("b"), 1);
    assert_eq!(names.intern(String::from("a")), 0);
    assert_eq!(names.get("b"), Some(1));
    assert_eq!(names.resolve(1).as_deref(), Some("b"));
    assert_eq!(names.resolve(2), None);
    assert_eq!(names.resolve(u64::MAX), None);
    let snapshot: Vec<_> = names.iter().map(|(id, v)| (id, v.to_string())).collect();
    assert_eq!(snapshot, [(0, "a".to_string()), (1, "b".to_string())]);
    assert_eq!(cx.interner::<str>(cx.symbol_sort("Name")).sort, names.sort);
}

#[test]
fn an_interner_type_mismatch_leaves_the_registry_usable() {
    let cx = Context::new();
    let sort = cx.symbol_sort("Name");
    let names = cx.interner::<str>(sort);
    let id = names.intern("a");
    let err = catch_unwind(AssertUnwindSafe(|| cx.interner::<String>(sort))).err();
    assert!(
        err.as_ref()
            .and_then(|e| e.downcast_ref::<String>())
            .is_some_and(|m| m.starts_with("symbol sort must intern one value type"))
    );
    let same = cx.interner::<str>(sort);
    assert!(Arc::ptr_eq(&names, &same));
    assert_eq!(same.resolve(id).as_deref(), Some("a"));
    let other = cx.interner::<str>(cx.symbol_sort("Other"));
    let id = other.intern("b");
    assert_eq!(other.resolve(id).as_deref(), Some("b"));
}

#[test]
#[should_panic(expected = "interned values require a symbol sort")]
fn an_interner_over_a_bit_vector_is_rejected() {
    let cx = Context::new();
    cx.interner::<str>(cx.bv_sort(8));
}

#[test]
#[should_panic(expected = "symbol sort belongs to another context")]
fn an_interner_of_another_context_is_rejected() {
    let sort: Sort = Context::new().symbol_sort("Name");
    Context::new().interner::<str>(sort);
}

// --- constructors ------------------------------------------------------------

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

// --- Boolean operations ------------------------------------------------------

#[test]
fn boolean_operations_and_ite_adopt_fresh_references() {
    let mut cx = cx();
    let c = bv(&mut cx, 4);
    let low = cx.less_than(c, &[6]);
    let odd = cx.masked_eq(c, &[1], &[1]);
    let picked = cx.set(c, &[2, 7, 9]);
    assert_eq!(below(&low, 16), [0, 1, 2, 3, 4, 5]);
    assert_eq!(below(&odd, 16), [1, 3, 5, 7, 9, 11, 13, 15]);
    assert_eq!(
        below(&cx.apply_bool(BoolOp::And, &low, &odd), 16),
        [1, 3, 5]
    );
    assert_eq!(
        below(&cx.apply_bool(BoolOp::AndNot, &low, &odd), 16),
        [0, 2, 4]
    );
    assert_eq!(
        below(&cx.apply_bool(BoolOp::Xor, &picked, &odd), 16),
        [1, 2, 3, 5, 11, 13, 15]
    );
    assert_eq!(below(&cx.ite(&low, &odd, &picked), 16), [1, 3, 5, 7, 9]);
    release_sole_reference(cx.apply_bool(BoolOp::And, &low, &odd));
    release_sole_reference(cx.ite(&low, &odd, &picked));
    drop((low, odd, picked));
    let p = cx.set(c, &[2, 7, 9]);
    assert_eq!(below(&p, 16), [2, 7, 9]);
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

// --- default backends and backend agreement ----------------------------------

#[test]
fn the_default_context_registers_sparse_before_bdd() {
    let mut cx = Context::default();
    let (sparse, bdd) = (cx.backend("sparse"), cx.backend("bdd"));
    assert_eq!((sparse.slot, bdd.slot), (0, 1));
    for sort in [
        BV4,
        ScalarSort::bool(),
        ScalarSort::bit_vec(70),
        ScalarSort::symbol(),
    ] {
        for backend in [sparse, bdd] {
            let column = cx.column(backend, sort);
            assert_eq!(cx.resolve(column).sort(), sort);
        }
    }
}

/// A context with a tiny sparse table and a tiny BDD store, so both backends
/// collect and grow under the same constructions.
fn both() -> Context {
    let mut cx = Context::new();
    cx.register_backend("sparse", SparseTable::with_capacity(2, 0, 1));
    cx.register_backend("bdd", BddStore::with_capacity(2, 0, 1));
    cx
}

/// The sparse and BDD columns of one sort.
fn pair(cx: &mut Context, sort: ScalarSort) -> [super::ColumnId; 2] {
    ["sparse", "bdd"].map(|name| {
        let backend = cx.backend(name);
        cx.column(backend, sort)
    })
}

#[test]
fn both_backends_agree_on_boolean_predicates() {
    let mut cx = both();
    for (sort, size) in [(BV4, 16), (BV12, 4096)] {
        let columns = pair(&mut cx, sort);
        let results = columns.map(|c| {
            let a = cx.set(c, &(0..size).filter(|v| v % 5 < 2).collect::<Vec<_>>());
            let b = cx.set(c, &(0..size).filter(|v| v % 7 < 3).collect::<Vec<_>>());
            let k = cx.less_than(c, &[size * 2 / 3]);
            let mut out = vec![cx.ite(&k, &a, &b)];
            out.extend(
                [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot]
                    .map(|op| cx.apply_bool(op, &a, &b)),
            );
            out.push(cx.apply_bool(BoolOp::AndNot, &cx.t(c), &k));
            out
        });
        for (p, q) in results[0].iter().zip(&results[1]) {
            let (sc, bc) = (cx.resolve(columns[0]), cx.resolve(columns[1]));
            assert_eq!(below(p, size), below(q, size));
            assert_eq!(sc.count(p.function()), bc.count(q.function()));
            let rows = [(sc, p), (bc, q)].map(|(d, r)| {
                let mut cursor = d.members(r.function());
                let (mut out, mut v) = (Vec::new(), Vec::new());
                while cursor.next(&mut v) {
                    out.push(v[0]);
                }
                out
            });
            assert_eq!(rows[0], rows[1]);
            assert_eq!(rows[0], below(p, size));
        }
    }
}

/// The slot of every value of a one-word sort below `size`.
fn slots(c: &dyn Column, f: Function, size: u64) -> Vec<u32> {
    (0..size).map(|x| c.evaluate(f, &[x])).collect()
}

#[test]
fn both_backends_agree_on_partition_slots() {
    let mut cx = both();
    for (sort, size) in [(BV4, 16), (BV12, 4096)] {
        let results = pair(&mut cx, sort).map(|column| {
            let c = cx.resolve(column);
            let residue = |n: u64| -> Vec<(Function, u32)> {
                (1..n)
                    .map(|r| {
                        let values: Vec<u64> = (0..size).filter(|v| v % n == r).collect();
                        (c.set(&values), r as u32)
                    })
                    .collect()
            };
            let mut labels = Vec::new();
            let f = c.cases(0, &residue(3), &mut labels);
            assert_eq!(labels, [0, 1, 2]);
            let g = c.cases(0, &residue(2), &mut labels);
            let keys: Vec<u64> = (1..size).step_by(3).collect();
            let case_labels: Vec<u32> = keys.iter().map(|k| 20 + (k % 4) as u32).collect();
            let mut value_labels = Vec::new();
            let v = c.values(
                9,
                &keys,
                &case_labels,
                &mut value_labels,
                &mut Values::default(),
            );
            let mut pairs = Vec::new();
            let mut scratch = Alignment::default();
            let h = c.align(f, g, &mut pairs, &mut scratch);
            let q = c.quotient(f, &[0, 1, 0], &mut scratch);
            let p = c.set(&(0..size).filter(|v| v % 5 == 1).collect::<Vec<_>>());
            let mut cells = Vec::new();
            let _ = c.refine(
                f,
                p,
                &mut Refinement::default(),
                &mut |slot| slot != 2,
                &mut |slot, build| {
                    let cell = build();
                    cells.push((slot, slots(c, cell, size)));
                    c.dec_ref(cell);
                    ControlFlow::Continue(())
                },
            );
            cells.sort();
            (
                [f, g, v, h, q].map(|x| slots(c, x, size)),
                value_labels,
                pairs,
                cells,
            )
        });
        assert_eq!(results[0], results[1]);
        let ([f, g, v, h, q], value_labels, pairs, cells) = &results[0];
        assert_eq!(value_labels, &[9, 21, 20, 23, 22]);
        assert_eq!(pairs.len(), 6);
        assert_eq!(cells.len(), 2);
        for x in 0..size as usize {
            assert_eq!(f[x], (x % 3) as u32);
            assert_eq!(g[x], (x % 2) as u32);
            assert_eq!(q[x], u32::from(x % 3 == 1));
            assert_eq!(pairs[h[x] as usize], (f[x], g[x]));
            let label = 20 + (x % 4) as u32;
            let expect = if x % 3 == 1 {
                value_labels.iter().position(|&l| l == label).unwrap()
            } else {
                0
            };
            assert_eq!(v[x] as usize, expect, "at {x}");
            for &(slot, ref cell) in cells {
                assert_eq!(cell[x], u32::from(x % 5 == 1 && x % 3 == slot as usize));
            }
        }
    }
}
