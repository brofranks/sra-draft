//! Checks for interned symbols: handles carry their context identity,
//! strings intern while packing rows, and capacity universes bound member
//! IDs within one context.

use std::collections::{BTreeSet, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{Capacity, Dictionary, Domain, Interned};
use crate::context::Context;
use crate::typed::{FiniteSet, IntoQuery, Rel, Sorted, Universe, pred};

struct Names;

impl Domain for Names {
    type Value = str;
    const NAME: &'static str = "Name";
}

type Name = Interned<Names>;
type Hops = FiniteSet<Capacity<Names, 4>>;

struct Paths;

impl Domain for Paths {
    type Value = Vec<u32>;
    const NAME: &'static str = "Path";
}

// --- handles and rows ---------------------------------------------------

#[test]
fn handles_carry_their_context_identity() {
    let cx = Context::default();
    let names = Dictionary::<Names>::new(&cx);
    let a = names.intern("a");
    let b = names.clone().intern("b");
    assert_eq!(a, names.interned(0));
    assert!(a < b);
    assert_eq!(BTreeSet::from([b, a, a]).len(), 2);
    assert_eq!(HashSet::from([b, a, a]).len(), 2);
    assert_eq!(
        format!("{a:?}"),
        format!("Interned {{ context: {}, id: 0 }}", cx.id)
    );
    assert_eq!(names.get("b"), Some(b));
    assert_eq!(names.get("c"), None);
    assert_eq!(names.resolve(b).as_deref(), Some("b"));
    assert_eq!(names.iter().map(|(x, _)| x).collect::<Vec<_>>(), [a, b]);
    let foreign = Dictionary::<Names>::new(&Context::default()).intern("a");
    assert_ne!(foreign, a);
    assert_eq!(Dictionary::<Names>::new(&cx).intern("a"), a);
    let paths = Dictionary::<Paths>::new(&cx);
    assert_eq!(
        paths.resolve(paths.intern(vec![1, 2])).as_deref(),
        Some(&vec![1, 2])
    );
}

#[test]
fn rows_intern_strings_on_the_fly() {
    let mut cx = Context::default();
    Name::bind(&mut cx);
    Hops::bind(&mut cx);
    let mut rel = Rel::<Name, Name>::new_set(&cx);
    rel.insert(String::from("x"));
    rel.insert("y");
    let names = Dictionary::<Names>::new(&cx);
    let x = names.get("x").unwrap();
    assert!(rel.contains(x));
    assert_eq!((&rel).with(pred::eq("y")).count(), 1_u32.into());
    let mut sets = Rel::<Hops, Hops>::new_set(&cx);
    sets.insert(Hops::from_members(
        &cx,
        [names.intern("x"), names.intern("z")],
    ));
    assert!(sets.contains(Hops::from_members(&cx, [x, names.get("z").unwrap()])));
}

#[test]
#[should_panic(expected = "interned value belongs to another context")]
fn dictionaries_resolve_only_their_own_handles_is_rejected() {
    let names = Dictionary::<Names>::new(&Context::default());
    let foreign = Dictionary::<Names>::new(&Context::default()).intern("a");
    names.resolve(foreign);
}

#[test]
#[should_panic(expected = "interned value belongs to another context")]
fn rows_reject_foreign_handles_is_rejected() {
    let mut cx = Context::default();
    Name::bind(&mut cx);
    let foreign = Dictionary::<Names>::new(&Context::default()).intern("a");
    Rel::<Name, Name>::new_set(&cx).insert(foreign);
}

#[test]
#[should_panic(expected = "interned value belongs to another namespace")]
fn rows_check_the_namespace_is_rejected() {
    let mut cx = Context::default();
    Interned::<Paths>::bind(&mut cx);
    let name = Dictionary::<Names>::new(&cx).intern("a");
    Rel::<Interned<Paths>, Interned<Paths>>::new_set(&cx).insert(name);
}

// --- capacity universes -------------------------------------------------

#[test]
#[should_panic(expected = "set member belongs to another context")]
fn set_members_convert_in_their_context_is_rejected() {
    let cx = Context::default();
    let foreign = Dictionary::<Names>::new(&Context::default()).intern("a");
    Hops::empty(&cx).contains(foreign);
}

#[test]
fn capacity_indices_include_unnamed_members_and_reject_outside_ids() {
    let cx = Context::default();
    let names = Dictionary::<Names>::new(&cx);
    assert_eq!(
        Capacity::<Names, 4>::index(cx.id, &names.interned(0)),
        Some(0)
    );
    assert_eq!(
        Capacity::<Names, 4>::index(cx.id, &names.interned(3)),
        Some(3)
    );
    assert_eq!(Capacity::<Names, 4>::index(cx.id, &names.interned(4)), None);
    assert_eq!(
        Capacity::<Names, 4>::index(cx.id, &names.interned(u64::MAX)),
        None
    );
}

#[test]
#[should_panic(expected = "member must belong to this set universe")]
fn out_of_capacity_set_members_is_rejected() {
    let cx = Context::default();
    let names = Dictionary::<Names>::new(&cx);
    FiniteSet::<Capacity<Names, 1>>::empty(&cx).contains(names.interned(1));
}

// --- namespace value types ----------------------------------------------

/// Runs `reject`, which must fail the one-value-type namespace check.
fn rejects_value_type(reject: impl FnOnce()) {
    let payload = catch_unwind(AssertUnwindSafe(reject)).unwrap_err();
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or(payload.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.contains("symbol sort must intern one value type"));
}

#[test]
fn typed_namespaces_reject_wrong_literal_types_before_dictionary_access() {
    for dictionary_first in [false, true] {
        let mut cx = Context::default();
        Interned::<Paths>::bind(&mut cx);
        let mut rel = Rel::<Interned<Paths>, Interned<Paths>>::new_set(&cx);
        if dictionary_first {
            Dictionary::<Paths>::new(&cx);
        }
        rejects_value_type(|| {
            rel.contains("not a path");
        });
        rejects_value_type(|| {
            rel.insert("not a path");
        });
        let paths = Dictionary::<Paths>::new(&cx);
        let path = paths.intern(vec![1, 2]);
        rel.insert(path);
        assert!(rel.contains(path));
        assert_eq!(paths.resolve(path).as_deref(), Some(&vec![1, 2]));
        assert!(!rel.contains(paths.interned(u64::MAX)));
    }
}

#[test]
fn typed_namespaces_check_existing_raw_dictionary_types() {
    let cx = Context::default();
    let sort = cx.symbol_sort(Paths::NAME);
    cx.interner::<str>(sort).intern("not a path");
    rejects_value_type(|| {
        Interned::<Paths>::sort(&cx);
    });

    struct AlsoNames;
    impl Domain for AlsoNames {
        type Value = str;
        const NAME: &'static str = Names::NAME;
    }
    let sort = cx.symbol_sort(Names::NAME);
    let id = cx.interner::<str>(sort).intern("a");
    assert_eq!(Interned::<AlsoNames>::sort(&cx), Name::sort(&cx));
    assert_eq!(Dictionary::<Names>::new(&cx).get("a").unwrap().id, id);
    assert_eq!(Dictionary::<AlsoNames>::new(&cx).get("a").unwrap().id, id);
}

#[test]
fn equal_namespace_names_require_equal_value_types() {
    struct OtherPaths;
    impl Domain for OtherPaths {
        type Value = str;
        const NAME: &'static str = Paths::NAME;
    }
    let cx = Context::default();
    Interned::<Paths>::sort(&cx);
    rejects_value_type(|| {
        Interned::<OtherPaths>::sort(&cx);
    });
    let paths = Dictionary::<Paths>::new(&cx);
    assert_eq!(paths.intern(vec![1]).id, 0);
}
