//! Reachability through a cycle with symbolic packet values and a guarded bit
//! rewrite.
//!
//! Run with `cargo run --locked --example iteration`. Each round reads only
//! the new frontier; ordinary query composition hides intermediate locations
//! and pre-rewrite packet values.

use sra::typed::{IntoQuery, Rel, Sorted, UInt, Variable};
use sra::{Bv, Context, Schema};

mod support;

#[derive(Schema)]
struct State {
    location: u8,
    #[symbolic]
    packet: UInt<3>,
}

fn main() {
    let mut cx = Context::default();
    State::bind(&mut cx);
    let StateSchema {
        location, packet, ..
    } = State::schema(&cx);
    // Explicit declarations keep the concrete location and symbolic packet
    // columns in every collected frontier. The relation stores current states,
    // not original/current packet pairs.
    let mut seed = Rel::<u8, UInt<3>>::from_raw(sra::Rel::new(&cx, location.raw, packet.raw));
    seed.insert((0_u8, Bv::new(&cx, packet.raw).extract(2, 2).eq(&cx, 0_u8)));
    let mut plain = Rel::<u8, u8>::new(&cx);
    plain.insert_rows([(0_u8, 1_u8), (2, 1)]);
    let mut rewrite = plain.empty_like();
    rewrite.insert((1_u8, 2_u8));

    // Independent concrete traversal: ordinary integer conditions and a
    // worklist, with no engine predicates or frontier algebra. Location 1 later
    // gains packets 4 and 6 under an old key.
    let expected = support::closure((0_u8..4).map(|p| (0_u8, p)), |&(loc, p)| match loc {
        0 | 2 => vec![(1, p)],
        1 if p % 2 == 0 => vec![(2, p | 4)],
        _ => Vec::new(),
    });
    assert_eq!(expected.len(), 12);

    let mut reach = Variable::new(seed);
    while reach.changed() {
        let recent = reach.recent();
        // Inverse edges compose destination -> previous location -> packet.
        // Only the rewrite edge requires even packets; setting bit 2 may
        // reproduce an already accepted packet.
        let forwarded = plain.inv().select(&recent);
        let rewritten = rewrite
            .inv()
            .select(&recent)
            .filt(|p| (p & 1_u8).eq(0_u8))
            .map(|p| p | 4_u8);
        reach.insert(&forwarded.union(rewritten).collect());
    }
    support::assert_rows(
        &reach.complete(),
        expected
            .iter()
            .map(|&(loc, p)| (loc, UInt::<3>::new(p.into()))),
    );
    println!("PASS: symbolic reachability matches all 12 concrete states");
}

#[test]
fn main_passes() {
    main();
}
