//! Application-defined scalar interpretations and a fixed community universe.
//!
//! Prefix addresses are symbolic; next hops, AS numbers, and community sets
//! default to sparse.

use sra::row::{RowRef, RowSink};
use sra::typed::{Cons, FiniteSet, IntoQuery, Nil, Rel, Sorted, Universe};
use sra::{Context, IntoRow, ScalarSort, Schema, Sort};
use std::net::Ipv4Addr;

mod support;

/// A local scalar needs only its sort/interpretation and concrete packing.
///
/// Every u32 bit pattern denotes an Asn here; restrictions on acceptable AS
/// numbers are relation contents.
struct Asn(u32);
impl Sorted for Asn {
    type Shape = Cons<Self, Nil>;
    fn sort(cx: &Context) -> Sort {
        cx.bv_sort(32)
    }
}
impl sra::typed::View for Asn {
    type Expr<D: Sorted> = sra::typed::expr::Expr<D, Self>;
    fn view<D: Sorted>(input: Self::Expr<D>) -> Self::Expr<D> {
        input
    }
}
impl IntoRow for Asn {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert_eq!(
            row.sort.scalar(),
            ScalarSort::bit_vec(32),
            "asn requires 32 bits"
        );
        self.0.append(row, out);
    }
}

enum Community {
    Customer,
    Peer,
    Transit,
}
struct CommunityUniverse;
impl Universe for CommunityUniverse {
    type Member = Community;
    const LEN: u32 = 3;
    fn index(_: u64, member: &Community) -> Option<u32> {
        Some(match member {
            Community::Customer => 0,
            Community::Peer => 1,
            Community::Transit => 2,
        })
    }
}
type CommunitySet = FiniteSet<CommunityUniverse>;

#[derive(Schema)]
struct Prefix {
    #[symbolic]
    addr: Ipv4Addr,
    #[symbolic]
    len: u8,
}

#[derive(Schema)]
struct Route {
    pfx: Prefix,
    nh: Ipv4Addr,
    origin: Asn,
    comm: CommunitySet,
}

fn main() {
    let mut cx = Context::default();
    Route::bind(&mut cx);
    let RouteSchema {
        pfx,
        nh,
        origin,
        comm,
        ..
    } = Route::schema(&cx);
    let PrefixSchema { addr, .. } = Prefix::schema(&cx);
    let p10 = support::ip("10.0.0.0");
    let route = Route {
        pfx: Prefix { addr: p10, len: 24 },
        nh: support::ip("10.0.0.1"),
        origin: Asn(64512),
        comm: CommunitySet::from_members(&cx, [Community::Customer, Community::Peer]),
    };
    let mut routes = Rel::<Route, Route>::new_set(&cx);
    routes.insert(&route);

    // The one stored route is a customer route for 10.0.0.0, so the restriction
    // keeps it and its next hop. The custom scalars supply their runtime
    // representation and typed interpretation; these predicates compare the
    // canonical values packed in that representation.
    let selected = routes
        .with(pfx.query(&cx).select(addr).eq(p10))
        .with(comm.contains(&cx, Community::Customer));
    support::assert_rows(&selected.select(nh).collect(), [(&route, route.nh)]);
    assert!(
        routes
            .with(comm.contains(&cx, Community::Transit))
            .is_empty()
    );
    assert!(
        routes
            .with(pfx.query(&cx).select(addr).eq(support::ip("192.168.0.0")))
            .is_empty()
    );
    assert!(
        !routes
            .with(origin.query(&cx).filt(|a| a.eq(Asn(64512))))
            .is_empty()
    );
    assert!(
        routes
            .with(origin.query(&cx).filt(|a| a.eq(Asn(64513))))
            .is_empty()
    );

    // Set update on the range: Transit joins and Peer leaves, keyed by the
    // unchanged route.
    let sets = routes
        .select(comm)
        .map(|s| s.insert(Community::Transit).remove(Community::Peer))
        .collect();
    support::assert_rows(
        &sets,
        [(
            &route,
            CommunitySet::from_members(&cx, [Community::Customer, Community::Transit]),
        )],
    );
    println!("PASS: application scalars, a fixed community universe, and a set update");
}

#[test]
fn main_passes() {
    main();
}
