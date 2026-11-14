//! Reachability preservation under symbolic link failures: a fault-tolerance
//! specification.
//!
//! Run from the crate directory with
//! `cargo run --locked --example fault_tolerance`. Link availability is one
//! symbolic column, exactly as the appendix keeps route advertisements: a set
//! of the links that are up, one bit per link, so a relation over it holds
//! every failure world at once. Reachability is the usual `Variable` fixpoint
//! over a concrete edge relation; each hop only guards the world column with
//! `with(up.contains(cx, link))`, so no two symbolic cells are ever joined.
//! Keeping the world in the key makes containment quantify over worlds: `pre`
//! is a subset of `post` says that in every world, whatever PRE connects, POST
//! connects too. Expected outcomes are derived by hand from the two topologies
//! below and checked against a concrete worklist per world; the unrestricted
//! FAIL is intentional.

use sra::typed::{
    Capacity, Dictionary, Domain, FiniteSet, Interned, IntoQuery, Rel, Sorted, Variable,
};
use sra::{Context, Schema};
use std::collections::BTreeSet;

mod support;

struct RouterNames;
impl Domain for RouterNames {
    type Value = str;
    const NAME: &'static str = "Router";
}
struct LinkNames;
impl Domain for LinkNames {
    type Value = str;
    const NAME: &'static str = "Link";
}
type Router = Interned<RouterNames>;
/// The links that are up: the failure world, bounded by the number of links
/// ever named.
type UpSet = FiniteSet<Capacity<LinkNames, { LINKS.len() as u32 }>>;

/// A source router together with the world it is reached from.
///
/// The world travels in the domain, so the reachability relation
/// `Origin -> Router` is keyed per world.
#[derive(Schema)]
struct Origin {
    src: Router,
    #[symbolic]
    up: UpSet,
}

const ROUTERS: [&str; 5] = ["r1", "r2", "r3", "r4", "r5"];
/// Every link either snapshot uses, interned in this order so bit `i` of a
/// world is `LINKS[i]`.
const LINKS: [&str; 7] = [
    "r1-r2", "r2-r3", "r3-r1", "r3-r4", "r4-r5", "r1-r4", "r5-r1",
];
/// Every failure world, as a bit set over `LINKS`.
const WORLDS: u64 = 1 << LINKS.len();
/// PRE: a triangle r1 r2 r3, two links into r4, and r4-r5 as the only way to
/// r5.
const PRE: [&str; 6] = ["r1-r2", "r2-r3", "r3-r1", "r3-r4", "r4-r5", "r1-r4"];
/// POST retires r1-r4 and adds r5-r1: a 5-cycle with the chord r3-r1, so no
/// single link cuts it.
const POST: [&str; 6] = ["r1-r2", "r2-r3", "r3-r1", "r3-r4", "r4-r5", "r5-r1"];

fn ends(link: &str) -> (&str, &str) {
    link.split_once('-').unwrap()
}

/// From every (source, world), the routers reachable over the links of `links`
/// that are up in that world, and the number of rounds the fixpoint took.
fn reach(cx: &Context, links: &[&str]) -> (Rel<Origin, Router>, usize) {
    let OriginSchema { up, .. } = Origin::schema(cx);
    let dictionary = Dictionary::<LinkNames>::new(cx);
    let hops: Vec<_> = links
        .iter()
        .map(|&link| {
            let (a, b) = ends(link);
            let mut hop = Rel::<Router, Router>::new(cx);
            hop.insert_rows([(a, b), (b, a)]);
            (up.contains(cx, dictionary.intern(link)), hop)
        })
        .collect();
    // Every router reaches itself in every world.
    let mut seed = Rel::<Origin, Router>::new(cx);
    let every_world = cx.t(cx.field(up.raw).unwrap());
    seed.insert_rows(ROUTERS.map(|r| ((r, &every_world), r)));
    let mut reach = Variable::new(seed);
    let mut rounds = 0;
    while reach.changed() {
        rounds += 1;
        let recent = reach.recent();
        for (guard, hop) in &hops {
            // `with` restricts the range, so two `inv()` bring the origin's
            // world under the guard; the world column is tested, never joined
            // against another symbolic cell.
            reach.insert(&(&recent).inv().with(guard).inv().select(hop).collect());
        }
    }
    (reach.complete(), rounds)
}

/// The links up in world `bits`, by position in `LINKS`.
fn links_up(links: &[&'static str], bits: u64) -> Vec<&'static str> {
    links
        .iter()
        .copied()
        .filter(|l| bits >> LINKS.iter().position(|k| k == l).unwrap() & 1 == 1)
        .collect()
}

/// Independent reference: the router pairs connected in world `bits`, by a
/// plain worklist over link names. No engine relation or predicate takes part.
fn connected(links: &[&'static str], bits: u64) -> BTreeSet<(&'static str, &'static str)> {
    let up = links_up(links, bits);
    ROUTERS
        .iter()
        .flat_map(|&s| {
            support::closure([s], |&r| {
                up.iter()
                    .filter_map(|l| {
                        let (a, b) = ends(l);
                        (a == r).then_some(b).or((b == r).then_some(a))
                    })
                    .collect()
            })
            .into_iter()
            .map(move |d| (s, d))
        })
        .collect()
}

fn failed(bits: u64) -> Vec<&'static str> {
    links_up(&LINKS, !bits)
}

fn main() {
    let mut cx = Context::default();
    Origin::bind(&mut cx);
    let cx = &cx;
    let OriginSchema { up, .. } = Origin::schema(cx);
    let dictionary = Dictionary::<LinkNames>::new(cx);
    for (i, link) in LINKS.iter().enumerate() {
        assert_eq!(dictionary.intern(*link).id, i as u64);
    }

    let (pre, pre_rounds) = reach(cx, &PRE);
    let (post, post_rounds) = reach(cx, &POST);

    // The worlds with at most one failed link, Hoyan's single-failure
    // assumption. Restricting the world column to them keeps the world in the
    // key, so containment still quantifies over them.
    let worlds: Vec<UpSet> = std::iter::once(UpSet::full(cx))
        .chain(LINKS.map(|link| {
            let mut world = UpSet::full(cx);
            world.remove(dictionary.intern(link));
            world
        }))
        .collect();
    let restrict = |r: &Rel<Origin, Router>| r.inv().with(up.is_in(&worlds)).inv().collect();

    // Reachability preservation under single failures: whatever PRE connects,
    // POST connects.
    let single_failure_preservation = (&restrict(&pre)).is_subset(&post);

    // The same without the failure bound. The residual is the counterexample:
    // the (source, world) pairs and destinations PRE connects but POST does
    // not.
    let exceeded = (&pre).difference(&post).collect();
    let unrestricted_preservation = exceeded.is_empty();

    // POST connectivity under single failures: 5 sources x 5 destinations x 8
    // worlds.
    let post_single_failure_connectivity = restrict(&post).count() == 200_u32.into();

    // Reference outcomes per world, from the concrete worklist. World `bits`
    // has bit `i` set exactly when `LINKS[i]` is up.
    let pre_ref: Vec<_> = (0..WORLDS).map(|bits| connected(&PRE, bits)).collect();
    let post_ref: Vec<_> = (0..WORLDS).map(|bits| connected(&POST, bits)).collect();
    let rows = |reference: &[BTreeSet<(&'static str, &'static str)>]| {
        reference
            .iter()
            .enumerate()
            .flat_map(|(bits, pairs)| {
                pairs
                    .iter()
                    .map(move |&(s, d)| ((s, UpSet::from_words(cx, [bits as u64])), d))
            })
            .collect::<Vec<_>>()
    };
    support::assert_rows(&pre, rows(&pre_ref));
    support::assert_rows(&post, rows(&post_ref));
    let single = |bits: u64| failed(bits).len() <= 1;
    let violated = |bits: u64| !pre_ref[bits as usize].is_subset(&post_ref[bits as usize]);
    assert_eq!(
        single_failure_preservation,
        !(0..WORLDS).any(|bits| single(bits) && violated(bits))
    );
    assert_eq!(unrestricted_preservation, !(0..WORLDS).any(violated));
    assert_eq!(
        post_single_failure_connectivity,
        (0..WORLDS).all(|bits| !single(bits) || post_ref[bits as usize].len() == 25)
    );
    // The engine's counterexample holds exactly the facts the reference loses.
    let failing: Vec<u64> = (0..WORLDS).filter(|&bits| violated(bits)).collect();
    let lost: Vec<_> = pre_ref.iter().zip(&post_ref).map(|(a, b)| a - b).collect();
    support::assert_rows(&exceeded, rows(&lost));
    // The two-failure worlds that violate preservation, as derived by hand.
    let smallest: BTreeSet<_> = failing
        .iter()
        .map(|&bits| failed(bits))
        .filter(|links| links.len() == 2)
        .collect();
    assert_eq!(
        smallest,
        [vec!["r3-r4", "r4-r5"], vec!["r3-r4", "r5-r1"]].into()
    );

    println!(
        "fault tolerance: 5 routers, 7 links, {WORLDS} failure worlds; PRE {} rows in {pre_rounds} \
         rounds, POST {} rows in {post_rounds} rounds",
        pre.count(),
        post.count()
    );
    let missing = format!(
        "{} (source, world, destination) facts of PRE are missing from POST across {} worlds; \
         the smallest ones fail {{r3-r4, r4-r5}} or {{r3-r4, r5-r1}}, where PRE still had r1-r4",
        exceeded.count(),
        failing.len()
    );
    for (name, holds, wanted, explanation) in [
        (
            "single-failure preservation",
            single_failure_preservation,
            true,
            "with at most one link down, every pair PRE connects POST connects; the inclusion is \
             strict, since PRE loses r5 when r4-r5 fails",
        ),
        (
            "unrestricted preservation",
            unrestricted_preservation,
            false,
            &*missing,
        ),
        (
            "POST single-failure connectivity",
            post_single_failure_connectivity,
            true,
            "the 5-cycle with chord r3-r1 stays connected under any single failure",
        ),
    ] {
        support::check(name, holds, wanted, explanation);
    }
}

#[test]
fn main_passes() {
    main();
}
