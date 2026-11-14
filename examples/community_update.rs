//! Add a community to guarded routes and compare community sets keyed by
//! world, router, and next hop.
//!
//! Manual version of the Internet2 community-insertion example; no external
//! data is needed. Run from the crate directory with
//! `cargo run --locked --example community_update`. Assertions check semantics
//! using five input rows and eight worlds, with no performance timing.

use std::collections::BTreeSet;

use sra::typed::{IntoQuery, Rel, Sorted, UInt};
use sra::{Bv, Context, Schema};

#[derive(Schema)]
struct Route {
    #[symbolic]
    world: UInt<3>,
    router: u32,
    next_hop: u32,
    // One concrete set over COMMUNITIES; zero is an empty set on an existing
    // route.
    communities: u64,
}

const COMMUNITIES: [&str; 3] = ["65000:1", "65000:2", "65000:100"];
const ADDED: &str = "65000:100";

fn bit(community: &str) -> u64 {
    1 << COMMUNITIES.iter().position(|&c| c == community).unwrap()
}

// (source ID, router, next hop, community members, guard index)
// The first two rows share an ID but different payloads. The first and third
// merge after rekeying and insertion. Repeated members, an empty set,
// overlapping guards, and false occur.
const ROWS: [(u64, u64, u64, &[&str], usize); 5] = [
    (1, 1, 10, &["65000:1", "65000:1"], 0),
    (1, 2, 20, &[], 1),
    (2, 1, 10, &["65000:1", ADDED], 0),
    (3, 3, 30, &["65000:2"], 2),
    (4, 4, 40, &[], 3),
];

// A second presentation keeps snapshots as loaded subidentities on complete
// routes. The reduced path representation records membership rather than path
// order.
struct CommunityUniverse;
impl sra::typed::Universe for CommunityUniverse {
    type Member = &'static str;
    const LEN: u32 = 3;
    fn index(_: u64, member: &&'static str) -> Option<u32> {
        COMMUNITIES
            .iter()
            .position(|c| c == member)
            .map(|i| i as u32)
    }
}
type CommunitySet = sra::typed::FiniteSet<CommunityUniverse>;
#[derive(Schema)]
struct SnapshotRoute {
    #[symbolic]
    world: UInt<3>,
    router: u32,
    visited: u64,
    next_hop: u32,
    communities: CommunitySet,
}

fn check_snapshot_spec(cx: &Context) {
    const R: u32 = 2;
    let SnapshotRouteSchema {
        router, visited, ..
    } = SnapshotRoute::schema(cx);
    let mut pre = Rel::<SnapshotRoute, SnapshotRoute>::new_set(cx);
    let mut post = Rel::<SnapshotRoute, SnapshotRoute>::new_set(cx);
    // Independent, explicitly supplied snapshots include insertion, an
    // already-present member, and an unaffected route. Worlds and all other
    // route fields remain identical.
    for (world, at, path, hop, before, after) in [
        (0, R, 0, 10, vec!["65000:1"], vec!["65000:1", ADDED]),
        (0, R, 0, 10, vec!["65000:1", ADDED], vec!["65000:1", ADDED]),
        (1, 3, 1 << R, 20, vec![], vec![ADDED]),
        (2, 1, 0, 30, vec!["65000:2"], vec!["65000:2"]),
    ] {
        pre.insert((
            UInt::<3>::new(world),
            at,
            path,
            hop,
            CommunitySet::from_members(cx, before),
        ));
        post.insert((
            UInt::<3>::new(world),
            at,
            path,
            hop,
            CommunitySet::from_members(cx, after),
        ));
    }
    let affected = router.eq(R).or(visited
        .query(cx)
        .filt(|path| (path & (1_u64 << R)).ne(0_u64)));
    let before = pre.with(&affected);
    let after = post.with(&affected);
    let changed = (&before).map(|mut r| {
        r.communities = r.communities.insert(ADDED);
        r
    });
    // changed retains each PRE route as its domain, so its values are the POST
    // routes that have a PRE witness; multiple PRE routes may merge into one
    // POST route.
    assert!(changed.values() == after);
    let mut unaffected = pre.empty_like();
    unaffected.insert((
        UInt::<3>::new(2),
        1,
        0,
        30,
        CommunitySet::from_members(cx, ["65000:2"]),
    ));
    assert!(pre.with(!&affected) == unaffected);
    assert!(pre.with(!&affected) == post.with(!&affected));
    println!(
        "PASS: loaded PRE/POST, whole-record community insertion, affected and unaffected equality"
    );
}

fn main() {
    let mut cx = Context::default();
    Route::bind(&mut cx);
    SnapshotRoute::bind(&mut cx);
    let RouteSchema {
        world,
        router,
        next_hop,
        communities,
        ..
    } = Route::schema(&cx);
    // Incidental source ID, outside the route. This remains a genuine binary
    // relation.
    let mut routes = Rel::<u32, Route>::new(&cx);

    // World bits: 2 selects a toy prefix; 1 and 0 mean peers A and B advertise
    // that prefix. Keep them in one predicate so the chosen prefix stays
    // correlated with advertisements.
    let input = Bv::new(&cx, world);
    let prefix = input.extract(2, 2).eq(&cx, 1_u64);
    let a = input.extract(1, 1).eq(&cx, 1_u64);
    let b = input.extract(0, 0).eq(&cx, 1_u64);
    let guards = [
        prefix.not().and(&a), // !prefix && A
        prefix.and(&b),       // prefix && B
        prefix.ite(&b, &a),   // Either permitted advertisement.
        cx.f(cx.field(world.raw).unwrap()),
    ];
    routes.insert_rows(ROWS.map(|(id, r, hop, members, i)| {
        (
            id,
            (
                &guards[i],
                r,
                hop,
                members.iter().fold(0, |s, c| s | bit(c)),
            ),
        )
    }));

    // Independent reference: insert into ordinary sets and enumerate all eight
    // worlds using ordinary Rust conditions, independently of the map equation
    // and constructed predicates.
    let mut before_rows = BTreeSet::new();
    let mut expected_rows = BTreeSet::new();
    for (_, r, hop, members, i) in ROWS {
        let mut set = members.iter().copied().collect::<BTreeSet<_>>();
        let loaded = set.iter().copied().map(bit).sum::<u64>();
        set.insert(ADDED);
        let mask = set.into_iter().map(bit).sum::<u64>();
        for value in 0..8 {
            let is_prefix = value & 4 != 0;
            let peer_a = value & 2 != 0;
            let peer_b = value & 1 != 0;
            let holds = match i {
                0 => !is_prefix && peer_a,
                1 => is_prefix && peer_b,
                2 => (!is_prefix && peer_a) || (is_prefix && peer_b),
                _ => false,
            };
            if holds {
                before_rows.insert([value, r, hop, loaded]);
                expected_rows.insert([value, r, hop, mask]);
            }
        }
    }
    assert_eq!(before_rows.len(), 10);
    assert_eq!(expected_rows.len(), 8);

    // Only source IDs disappear: unchanged route fields form the key,
    // communities the range. Restrict complete routes before projecting so
    // shared source IDs cannot mix payloads. The community update is a map on
    // the range, not a stored replacement relation.
    let key = world.query(&cx).and(router).and(next_hop);
    let before = (&key).inv().with((&routes).values()).select(communities);
    let changed = (&before).map(|c| c | bit(ADDED));
    let actual = changed.collect();

    // No updated route may lack the added community. The closure builds a
    // symbolic predicate; use .eq(...) because Rust's == returns bool.
    assert!(actual.filt(|c| (c & bit(ADDED)).eq(0_u64)).is_empty());

    // Keep the declared field backends when checking the independently
    // enumerated rows.
    let row = |[value, r, hop, mask]: [u64; 4]| {
        (((UInt::<3>::new(value.into()), r as u32), hop as u32), mask)
    };
    let mut expected = actual.empty_like();
    expected.insert_rows(expected_rows.iter().map(|&values| row(values)));
    assert!(changed == expected);
    assert!(actual == expected);

    let mut expected_before = actual.empty_like();
    expected_before.insert_rows(before_rows.iter().map(|&values| row(values)));
    assert!(before.collect() == expected_before);
    assert!(expected_before != expected);
    assert!(
        (&key)
            .inv()
            .with(routes.inv())
            .select(communities)
            .collect()
            == expected_before
    );
    let twice = actual.map(|c| c | bit(ADDED));
    assert!(twice == actual);
    assert!(
        (&key)
            .inv()
            .with(Rel::<u32, Route>::new(&cx).inv())
            .select(communities)
            .map(|c| c | bit(ADDED))
            .is_empty()
    );

    // An incorrect prefix or a missing community must remain observable after
    // dropping IDs.
    for (column, mask) in [(0, 4), (3, bit(ADDED))] {
        let mut wrong = actual.empty_like();
        wrong.insert_rows(expected_rows.iter().map(|values| {
            let mut values = *values;
            values[column] ^= mask;
            row(values)
        }));
        assert!(actual != wrong);
    }

    println!("PASS: add {ADDED}; 5 manual rows, 8 worlds, 8 distinct keyed community sets");
    println!(
        "PASS: equality, added-community presence, idempotence, empty input, and \
         incorrect-world/community checks"
    );
    check_snapshot_spec(&cx);
}

#[test]
fn main_passes() {
    main();
}
