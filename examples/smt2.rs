//! Construct the same predicate directly and from a shared SMT-LIB guard
//! prelude.
//!
//! Run with `cargo run --locked --example smt2`.

use sra::row::Domain;
use sra::{Bv, Context, Rel, loader};

fn main() {
    let mut cx = Context::default();
    let schema = cx.tuple_decl("Route", [("world", cx.bv_sort(3))]);
    let backend = cx.backend("bdd");
    cx.set_representation(schema.accessors[0], backend);
    let world = schema.accessors[0].bind(&mut cx);

    // These views bind one existing column; extraction introduces no
    // independent variables.
    let x = Bv::new(&cx, world.clone());
    let prefix = x.extract(2, 2).eq(&cx, 1_u64);
    let a = x.extract(1, 1).eq(&cx, 1_u64);
    let b = x.extract(0, 0).eq(&cx, 1_u64);
    let permitted = prefix.ite(&b, &a);

    // The file's declaration binds to the same column. Repeated named roots
    // have independent owning handles, so dropping one or the loader's
    // temporaries does not invalidate another.
    let mut guards = loader::smt2(
        &cx,
        world.clone(),
        include_bytes!("guards.smt2").as_slice(),
        &["permitted", "permitted"],
    )
    .unwrap();
    drop(guards.pop());
    for value in 0..8 {
        let expected = if value & 4 != 0 {
            value & 1 != 0
        } else {
            value & 2 != 0
        };
        assert_eq!(permitted.contains(&[value]), expected);
        assert_eq!(guards[0].contains(&[value]), expected);
    }

    let mut routes = Rel::new_set(&cx, world);
    routes.insert(&guards[0]);
    drop(guards);
    assert!(routes.contains(2_u64) && !routes.contains(4_u64));
    println!("PASS: direct and SMT-LIB predicates agree in all eight worlds");
}

#[test]
fn main_passes() {
    main();
}
