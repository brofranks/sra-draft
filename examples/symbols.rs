//! Automatic interning, named symbol fields, explicit known-name relations,
//! and bounded sets over names discovered at runtime.
//!
//! Exact sets and symbolic
//! families use the same membership masks; BDD-backed updates transform them
//! without decoding set values.

use sra::{
    Context, Schema,
    typed::{Capacity, Dictionary, Domain, FiniteSet, Interned, IntoQuery, Rel, Sorted},
};
use std::ops::ControlFlow;

mod support;

struct RouterNames;
impl Domain for RouterNames {
    type Value = str;
    const NAME: &'static str = "RouterName";
}
type Router = Interned<RouterNames>;

/// Membership bits positioned by router ID; the first 64 interned routers are
/// set members.
type PathHops = FiniteSet<Capacity<RouterNames, 64>>;

#[derive(Schema)]
struct Route {
    router: Router,
    next_hop: Router,
    #[symbolic]
    path_hops: PathHops,
}

fn main() {
    let mut cx = Context::default();
    Route::bind(&mut cx);
    let names = Dictionary::<RouterNames>::new(&cx);
    let r1 = names.intern("r1");
    assert_eq!(names.intern("r1"), r1);
    let r2 = names.intern("r2");
    let r3 = names.intern("r3");
    let RouteSchema {
        router,
        next_hop,
        path_hops,
        ..
    } = Route::schema(&cx);
    let mut routes = Rel::<Route, Route>::new_set(&cx);
    routes.insert(("r1", "r2", PathHops::from_members(&cx, [r2])));
    routes.insert(("r2", "r1", PathHops::from_members(&cx, [r1])));
    routes.insert(("r3", "r1", PathHops::from_members(&cx, [r2])));
    routes.insert(("r1", "r3", PathHops::from_members(&cx, [r2])));
    assert_eq!(names.get("r2"), Some(r2));

    // Symbolic membership over the capacity bitmask needs no name list.
    let affected = router.eq("r1") | path_hops.contains(&cx, r1);
    assert_eq!(routes.with(affected).count(), 3_u32.into());

    // Expression operations add a member symbolically.
    let extended = routes.select(path_hops).map(|h| h.insert(r3)).collect();
    support::assert_rows(
        &extended,
        [
            (
                Route {
                    router: r1,
                    next_hop: r2,
                    path_hops: PathHops::from_members(&cx, [r2]),
                },
                PathHops::from_members(&cx, [r2, r3]),
            ),
            (
                Route {
                    router: r2,
                    next_hop: r1,
                    path_hops: PathHops::from_members(&cx, [r1]),
                },
                PathHops::from_members(&cx, [r1, r3]),
            ),
            (
                Route {
                    router: r3,
                    next_hop: r1,
                    path_hops: PathHops::from_members(&cx, [r2]),
                },
                PathHops::from_members(&cx, [r2, r3]),
            ),
            (
                Route {
                    router: r1,
                    next_hop: r3,
                    path_hops: PathHops::from_members(&cx, [r2]),
                },
                PathHops::from_members(&cx, [r2, r3]),
            ),
        ],
    );

    let mut known = Rel::<Router, Router>::new_set(&cx);
    known.insert_rows(names.iter().map(|(id, _)| id));
    let other = sra::typed::pred::ne("r1");
    // Construct the predicate before adding a name: its meaning does not
    // depend on the table.
    let allowed = sra::typed::Query::<Router, Router>::id(&cx)
        .with(&other)
        .collect();
    let r4 = names.intern("r4");
    assert!(allowed.contains(r4));
    assert!(!known.contains(r4));
    known.insert(r4);

    let mut reachable = Rel::<Router, Router>::new_set(&cx);
    reachable.insert_rows(["r1", "r2"]);
    let selected = routes
        .with(router.eq("r1"))
        .select(next_hop)
        .with(&reachable);
    // The callback decodes the interned IDs of the visited column back to
    // their names.
    let mut visited = Vec::new();
    let _: ControlFlow<()> = selected.drive(&mut |_, r| {
        visited.extend(
            support::members(&r.predicate(0))
                .iter()
                .map(|v| names.resolve(names.interned(v[0])).unwrap().to_string()),
        );
        ControlFlow::Continue(())
    });
    assert_eq!(visited, ["r2"]);
    assert_eq!(selected.count(), 1_u32.into());
    assert_eq!(known.with(&other).count(), 3_u32.into());
    assert_eq!(names.resolve(names.interned(u64::MAX)), None);
    println!("PASS: interned names, symbolic membership, and decoded next hops");
}

#[test]
fn main_passes() {
    main();
}
