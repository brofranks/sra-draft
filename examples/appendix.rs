//! Examples drawn from the paper's Figure 1 and appendix, over handcrafted PRE
//! and POST route snapshots.
//!
//! Run from the crate directory with `cargo run --locked --example appendix`.
//! Each listing adapts the paper's to this API, and the paper's remaining
//! Internet2 listings are absent; `Route::sch()` there abbreviates
//! `Route::schema(&cx)`. The snapshots need no external data;
//! expected PASS/FAIL outcomes are derived by hand from the rows below,
//! independently of the engine, and semantic FAIL outcomes are intentional.
//! The RCL specifications come from Hoyan:
//! https://ennanzhai.github.io/pub/sigcomm25-hoyan2.pdf, section 4.3. S3
//! pools next hops across the selected routers and prefixes; S5 keeps the
//! symbolic advertisements in its key so coverage is checked per world. The
//! AS-path bound and the concrete attribute defaults are not RCL
//! specifications; they exercise the attributes Expresso keeps concrete.
//! `Route` follows Expresso's own route record; see the rule for its
//! declaration order there.

use sra::typed::{Capacity, Dictionary, Domain, FiniteSet, Interned, IntoQuery, Rel, Sorted, UInt};
use sra::{Bv, Context, Schema};
use std::net::Ipv4Addr;

mod support;

// Types used by the examples: names are interned on first use; sets are
// bounded by capacity.
struct AsPathNames;
impl Domain for AsPathNames {
    type Value = str;
    const NAME: &'static str = "AsPath";
}
struct RouterNames;
impl Domain for RouterNames {
    type Value = str;
    const NAME: &'static str = "Router";
}
struct VrfNames;
impl Domain for VrfNames {
    type Value = str;
    const NAME: &'static str = "Vrf";
}
struct PeerNames;
impl Domain for PeerNames {
    type Value = str;
    const NAME: &'static str = "Peer";
}
struct CommunityNames;
impl Domain for CommunityNames {
    type Value = str;
    const NAME: &'static str = "Community";
}
type Router = Interned<RouterNames>;
type Vrf = Interned<VrfNames>;
// The received-from peer and the advertisement bits share one dictionary, so
// the bit position of a peer in `ads` and its name in `peer` denote the same
// neighbor.
type Peer = Interned<PeerNames>;
// The whole AS path, interned: the engine has no sequence sort, so a path is an
// opaque identity shared by the routes that carry it, exactly as the RouteViews
// harness treats it.
type AsPath = Interned<AsPathNames>;
type PeerSet = FiniteSet<Capacity<PeerNames, 266>>;
type CommunitySet = FiniteSet<Capacity<CommunityNames, 64>>;
type HopSet = FiniteSet<Capacity<RouterNames, 64>>;

// 32 address bits and 6 length bits: the 38 symbolic prefix bits Expresso uses
// (its section 4.2).
#[derive(Schema)]
struct Prefix {
    #[symbolic]
    addr: Ipv4Addr,
    #[symbolic]
    len: UInt<6>,
}

/// A BGP route, modeled after Expresso's own record and the Batfish route it
/// descends from.
///
/// Only the prefix and the per-peer advertisements are symbolic. Expresso
/// makes the same choice and gives a reason that is not cost: a symbolic MED
/// or local preference would make best-route comparison nontotal, so those
/// attributes stay concrete: MED at its default and local preference as each
/// row sets it.
///
/// Declaration order is index level order, so it is a physical decision and is
/// fixed by a rule: location first, then the symbolic columns - which makes
/// S5's projection key a prefix of the declaration - then the
/// propagated attributes in BGP attribute-code order, then the router-local
/// attributes that are never advertised, then provenance. The rule is chosen
/// here for explainability over six rows, not measured; a layout for a real
/// workload is a separate question.
///
/// Omitted deliberately: the originator ID and cluster list, since no route
/// reflector appears in these snapshots; the route tag; and the AS path as an
/// ordered sequence, which no sort expresses.
#[derive(Schema)]
struct Route {
    // Location.
    router: Router,
    vrf: Vrf,
    // Symbolic frontier.
    pfx: Prefix,
    #[symbolic]
    ads: PeerSet,
    // Propagated attributes, in BGP attribute-code order.
    origin: UInt<2>,      // 1: IGP = 0, EGP = 1, INCOMPLETE = 2
    as_path: AsPath,      // 2: the path itself, interned
    as_path_len: UInt<8>, // 2: stored, since the engine computes no arithmetic over a path
    origin_as: u32,       // 2: the path's last AS
    nh: Ipv4Addr,         // 3
    med: u32,             // 4
    lp: u32,              // 5
    comm: CommunitySet,   // 8
    // Router-local; never advertised.
    weight: UInt<16>,
    admin: UInt<8>,
    // Provenance.
    peer: Peer,
    hops: HopSet,
}

// Concrete attribute defaults, taken from Expresso's `Attributes.java` and
// agreeing with Batfish where the two overlap. Local preference is not among
// them: it is set per route below.
const ORIGIN_IGP: u128 = 0;
const DEFAULT_MED: u32 = 0;
const DEFAULT_WEIGHT: u128 = 0;
const EBGP_ADMIN: u128 = 20;
const INTERNET2_AS: u32 = 11537;

fn prefix(s: &str) -> Prefix {
    let (addr, len) = s.split_once('/').unwrap();
    Prefix {
        addr: support::ip(addr),
        len: UInt::new(len.parse().unwrap()),
    }
}

fn main() {
    let mut cx = Context::default();
    Route::bind(&mut cx);
    let RouteSchema {
        router,
        vrf,
        pfx,
        ads,
        origin,
        as_path_len,
        nh,
        med,
        lp,
        comm,
        hops,
        ..
    } = Route::schema(&cx);
    let peers = Dictionary::<PeerNames>::new(&cx);
    let comms = Dictionary::<CommunityNames>::new(&cx);
    let routers = Dictionary::<RouterNames>::new(&cx);

    // Advertisement bits p and b stay correlated in one symbolic column; each
    // complete row associates its guard with the concrete route. Intern the
    // peers first so the bits are known.
    let input = Bv::new(&cx, ads);
    let bit = |name| {
        let i = peers.intern(name).id as u32;
        input.extract(i, i).eq(&cx, 1_u64)
    };
    let p = bit("p");
    let b = bit("b");
    let guards = [
        p.clone(),       // primary advertised (and any backup state)
        p.not().and(&b), // backup only
        p.or(&b),        // either advertisement
        p.and(&b.not()), // primary only
        b.clone(),       // backup advertised (and any primary state)
    ];

    // Each row supplies the columns that vary - (router, VRF, prefix, guard,
    // AS path, next hop, local preference, communities, received-from peer,
    // hops) - and the rest take the defaults above. The stored path length is
    // counted from the path.
    let load = |rows: &[(
        &str,
        &str,
        &str,
        usize,
        &str,
        &str,
        u32,
        &[&str],
        &str,
        &[&str],
    )]| {
        let mut rel = Rel::<Route, Route>::new_set(&cx);
        rel.insert_rows(
            rows.iter()
                .map(|&(r, v, x, i, asp, hop, pref, members, from, path)| {
                    (
                        r,
                        v,
                        prefix(x),
                        &guards[i],
                        UInt::<2>::new(ORIGIN_IGP),
                        asp,
                        UInt::<8>::new(asp.split(' ').count() as u128),
                        INTERNET2_AS,
                        support::ip(hop),
                        DEFAULT_MED,
                        pref,
                        CommunitySet::from_members(&cx, members.iter().map(|m| comms.intern(*m))),
                        UInt::<16>::new(DEFAULT_WEIGHT),
                        UInt::<8>::new(EBGP_ADMIN),
                        from,
                        HopSet::from_members(&cx, path.iter().map(|h| routers.intern(*h))),
                    )
                }),
        );
        rel
    };
    // AS paths are nearest-first, so every path here ends at the originating
    // AS 11537.
    #[rustfmt::skip]
    let pre = load(&[
        ("r1", "default", "10.0.0.0/24", 0, "11537",       "1.2.3.4", 300, &[],         "p", &[]),
        ("r1", "default", "10.0.0.0/24", 1, "11537",       "1.2.3.5", 300, &[],         "b", &[]),
        ("r2", "default", "10.0.0.0/24", 0, "11537",       "1.2.3.4", 300, &[],         "p", &["r1"]),
        ("r2", "default", "10.0.0.0/24", 1, "11537",       "1.2.3.6", 300, &[],         "b", &["r1"]),
        ("r3", "blue",    "20.0.0.0/24", 2, "20965 11537", "9.9.9.9", 100, &["100:1"],  "p", &[]),
        ("r2", "blue",    "20.0.0.0/24", 2, "20965 11537", "9.9.9.8", 100, &[],         "p", &["r3"]),
    ]);
    // POST adds 65000:100 at r1 and downstream of it (the first four rows);
    // nothing else changes.
    #[rustfmt::skip]
    let post = load(&[
        ("r1", "default", "10.0.0.0/24", 0, "11537",       "1.2.3.4", 300, &["65000:100"], "p", &[]),
        ("r1", "default", "10.0.0.0/24", 1, "11537",       "1.2.3.5", 300, &["65000:100"], "b", &[]),
        ("r2", "default", "10.0.0.0/24", 0, "11537",       "1.2.3.4", 300, &["65000:100"], "p", &["r1"]),
        ("r2", "default", "10.0.0.0/24", 1, "11537",       "1.2.3.6", 300, &["65000:100"], "b", &["r1"]),
        ("r3", "blue",    "20.0.0.0/24", 2, "20965 11537", "9.9.9.9", 100, &["100:1"],     "p", &[]),
        ("r2", "blue",    "20.0.0.0/24", 2, "20965 11537", "9.9.9.8", 100, &[],            "p", &["r3"]),
    ]);

    // Community addition (Figure 1).
    let community_addition = {
        let n = "r1";
        let scope = &(router.eq(n) | hops.contains(&cx, routers.intern(n)));
        let added = comms.intern("65000:100");
        let changed = pre.with(scope).map(|mut r| {
            r.comm = r.comm.insert(added);
            r
        });
        changed.values() == post.with(scope).values() && pre.with(!scope) == post.with(!scope)
    };

    // Locational equivalence.
    let locational_equivalence = {
        let loc = &router.query(&cx).and(vrf);
        let backup_only =
            pre.with(ads.contains(&cx, peers.intern("b")) & !ads.contains(&cx, peers.intern("p")));
        let primary = pre.with(ads.contains(&cx, peers.intern("p")));
        backup_only.select(loc).values() == primary.select(loc).values()
    };

    // Peer priority change.
    let peer_priority = {
        let n = "r1";
        let scope = &(router.eq(n) | hops.contains(&cx, routers.intern(n)));
        let p_id = peers.intern("p");
        let b_id = peers.intern("b");
        let backup_only = &(ads.contains(&cx, b_id) & !ads.contains(&cx, p_id));
        let key = &Route::schema(&cx).except(&cx, lp);
        let priority = |routes: &Rel<Route, Route>| {
            let before = &routes.with(scope);
            before
                .with(backup_only)
                .union(before.with(backup_only).map(|mut r| {
                    r.ads = r.ads.insert(p_id);
                    r
                }))
                .union(before.with(ads.contains(&cx, p_id)).map(|mut r| {
                    r.ads = r.ads.remove(b_id);
                    r
                }))
                .values()
                .collect()
        };
        // Matching communities isolate the peer rewrite: backup wins when
        // both advertise, while the primary-only world keeps its route.
        #[rustfmt::skip]
        let before = load(&[
            ("r1", "default", "10.0.0.0/24", 0, "11537", "1.2.3.4", 300, &[], "p", &[]),
            ("r1", "default", "10.0.0.0/24", 1, "11537", "1.2.3.5", 300, &[], "b", &[]),
        ]);
        #[rustfmt::skip]
        let expected = load(&[
            ("r1", "default", "10.0.0.0/24", 3, "11537", "1.2.3.4", 300, &[], "p", &[]),
            ("r1", "default", "10.0.0.0/24", 4, "11537", "1.2.3.5", 300, &[], "b", &[]),
        ]);
        assert!(priority(&before).select(key).values() == expected.select(key).values());
        assert!(before.select(key).values() != expected.select(key).values());
        priority(&pre).select(key).values() == post.with(scope).select(key).values()
    };

    // Preference pinning (RCL S1).
    let preference_pinning = {
        let selected = &post.with(pfx.eq(prefix("10.0.0.0/24")));
        !selected.is_empty() && selected.with(lp.ne(300_u32)).is_empty()
    };

    // Preservation outside a prefix (RCL S2).
    let preservation = {
        let scope = &pfx.eq(prefix("10.0.0.0/24"));
        pre.with(!scope) == post.with(!scope)
    };

    // Next-hop set stability (RCL S3).
    let next_hop_stability = {
        let scope = &(router.is_in(["r1", "r2"])
            & pfx.is_in([prefix("10.0.0.0/24"), prefix("20.0.0.0/24")]));
        support::assert_rows(
            &pre.with(scope).select(nh).values().collect(),
            ["1.2.3.4", "1.2.3.5", "1.2.3.6", "9.9.9.8"].map(support::ip),
        );
        pre.with(scope).select(nh).values() == post.with(scope).select(nh).values()
    };

    // Community absence (RCL S4).
    let community_absence = post
        .with(router.is_in(["r1", "r2"]))
        .with(comm.contains(&cx, comms.intern("100:1")))
        .is_empty();
    assert!(
        !post
            .with(router.eq("r3"))
            .with(comm.contains(&cx, comms.intern("100:1")))
            .is_empty()
    );

    // AS-path bound: the only ordering predicate here, and the reason the
    // length is stored alongside the interned path rather than computed from
    // it.
    let as_path_bound = post.with(as_path_len.gt(UInt::<8>::new(2))).is_empty();
    assert!(!post.with(as_path_len.gt(UInt::<8>::new(1))).is_empty());

    // Concrete attribute defaults: Expresso's concreteness boundary, checked
    // on the snapshot.
    let concrete_defaults = post.with(origin.ne(UInt::<2>::new(ORIGIN_IGP))).is_empty()
        && post.with(med.ne(DEFAULT_MED)).is_empty();
    assert!(!post.with(med.ne(1_u32)).is_empty());

    // Next-hop migration (RCL S5), also the body figure's query. The key is a
    // prefix of the declaration order, so it descends the index without
    // skipping a level.
    let next_hop_migration = {
        let key = &router.query(&cx).and(vrf).and(pfx).and(ads);
        let migrated = |l: &Rel<Route, Route>, r: &Rel<Route, Route>| {
            let before = l.with(nh.eq(support::ip("1.2.3.4"))).select(key);
            let after = r.with(nh.eq(support::ip("10.2.3.4"))).select(key);
            before.values().minus(after.values()).is_empty()
        };
        // A full move holds, while a narrower guard fails in the world where
        // both peers advertise, which only the per-world key can see.
        #[rustfmt::skip]
        let old = load(&[("r1", "default", "10.0.0.0/24", 0, "11537", "1.2.3.4", 300, &[], "p", &[])]);
        #[rustfmt::skip]
        let moved = load(&[("r1", "default", "10.0.0.0/24", 0, "11537", "10.2.3.4", 300, &[], "p", &[])]);
        #[rustfmt::skip]
        let narrowed = load(&[("r1", "default", "10.0.0.0/24", 3, "11537", "10.2.3.4", 300, &[], "p", &[])]);
        assert!(migrated(&old, &moved));
        assert!(!migrated(&old, &narrowed));
        migrated(&pre, &post)
    };

    println!("appendix examples: 6 PRE rows, 6 POST rows; semantic FAIL outcomes are intentional");
    for (name, holds, wanted, explanation) in [
        (
            "community addition",
            community_addition,
            true,
            "the four routes at or downstream of r1 gain exactly 65000:100",
        ),
        (
            "locational equivalence",
            locational_equivalence,
            true,
            "r1/default and r2/default hold both kinds; the either-advertised routes at r3/blue \
             and r2/blue are backup-only in the world b and not p",
        ),
        (
            "peer priority change",
            peer_priority,
            false,
            "with both peers advertised, POST keeps p/1.2.3.4 instead of b/1.2.3.5; \
             its communities also differ",
        ),
        (
            "preference pinning (S1)",
            preference_pinning,
            true,
            "every POST route for 10.0.0.0/24 has local preference 300",
        ),
        (
            "preservation outside a prefix (S2)",
            preservation,
            true,
            "the 20.0.0.0/24 routes are unchanged",
        ),
        (
            "next-hop set stability (S3)",
            next_hop_stability,
            true,
            "pooled next hops agree: {1.2.3.4, 1.2.3.5, 1.2.3.6, 9.9.9.8}",
        ),
        (
            "community absence (S4)",
            community_absence,
            true,
            "community 100:1 appears only on r3",
        ),
        (
            "AS-path bound",
            as_path_bound,
            true,
            "no route carries a path longer than two AS hops",
        ),
        (
            "concrete attribute defaults",
            concrete_defaults,
            true,
            "origin is IGP and MED is 0 on every route, the values best-route comparison relies on",
        ),
        (
            "next-hop migration (S5)",
            next_hop_migration,
            false,
            "1.2.3.4 is still in use and no route uses 10.2.3.4",
        ),
    ] {
        support::check(name, holds, wanted, explanation);
    }
}
