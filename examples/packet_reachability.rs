//! Forward and backward packet reachability over a small authored network.
//!
//! Run with `cargo run --locked --example packet_reachability`. Five symbolic
//! header fields retain their joint correlations; concrete locations name
//! pipeline stages. This adapts the workspace packet experiment, not a real
//! operator configuration. ACL priority, longest-prefix routing, ECMP, NAT, and
//! a cycle are specified below. There is no TTL, session tracking, control
//! plane, or loop-disposition analysis.

use sra::index::Level;
use sra::run::Exec;
use sra::typed::{IntoQuery, Query, Rel, Sorted};
use sra::util::dag::Idx;
use sra::{BoolOp, Context, Schema};
use std::collections::{BTreeSet, VecDeque};

mod support;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Schema)]
struct Packet {
    #[symbolic]
    dst_ip: u32,
    #[symbolic]
    src_ip: u32,
    #[symbolic]
    dst_port: u16,
    #[symbolic]
    src_port: u16,
    #[symbolic]
    protocol: u8,
}

#[derive(Schema)]
struct State {
    location: u8,
    packet: Packet,
}

// The ingress router splits traffic over two branches before delivery. A
// separate destination prefix traverses the cycle; one lap changes
// TCP/8080 to TCP/80, which may then leave the cycle.
const IN: u8 = 0;
const DNAT: u8 = 1;
const ROUTE: u8 = 2;
const LEFT: u8 = 3;
const LEFT_OUT: u8 = 4;
const RIGHT: u8 = 5;
const LOOP_A: u8 = 6;
const LOOP_B: u8 = 7;
const DELIVERED: u8 = 8;
const DENY_IN: u8 = 9;
const DENY_OUT: u8 = 10;
const NULL: u8 = 11;
const NO_ROUTE: u8 = 12;
/// 192.0.2.0/24
const VIP: u32 = 0xc0000200;
/// 10.64.4.0/24
const SERVICE: u32 = 0x0a400400;
/// 198.19.0.0/24
const CYCLE: u32 = 0xc6130000;
/// 198.18.0.0/24
const BLACKHOLE: u32 = 0xc6120000;
/// 198.51.100.7
const PUBLIC: u32 = 0xc6336407;

#[derive(Clone, Copy)]
enum Rewrite {
    Destination,
    Source,
    Port,
}

struct Rule {
    from: u8,
    to: u8,
    guard: Rel<Packet, Packet>,
    rewrite: Option<Rewrite>,
}

fn rules(cx: &Context) -> Vec<Rule> {
    let mut rules = Vec::new();
    let mut add = |from, to, guard, rewrite| {
        rules.push(Rule {
            from,
            to,
            guard,
            rewrite,
        });
    };
    let all = Query::<Packet, Packet>::id(cx).collect();
    let ssh = all
        .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(22_u16))
        .collect();
    let blocked = ssh
        .filt(|p| (p.src_ip & 0xffffff00_u32).eq(0x0a000000_u32))
        .collect();
    let exception = ssh.filt(|p| p.src_ip.eq(0x0a000001_u32)).collect();
    let service = all
        .filt(|p| {
            (p.protocol.clone().eq(6_u8) & p.dst_port.clone().is_in([80_u16, 443, 8080]))
                | (p.protocol.eq(17_u8) & p.dst_port.eq(53_u16))
        })
        .collect();
    // Ordered ACL: SSH exception, blocked source /24, then general
    // SSH/services.
    let allowed = exception
        .union(ssh.with(&blocked.not()))
        .union(&service)
        .collect();
    add(IN, DNAT, allowed.clone(), None);
    add(IN, DENY_IN, allowed.not(), None);

    let vip = all
        .filt(|p| {
            (p.dst_ip & 0xffffff00_u32).eq(VIP) & p.protocol.eq(6_u8) & p.dst_port.eq(8080_u16)
        })
        .collect();
    add(DNAT, ROUTE, vip.clone(), Some(Rewrite::Destination));
    add(DNAT, ROUTE, vip.not(), None);

    let dst = |mask: u32, net: u32| all.filt(|p| (p.dst_ip & mask).eq(net)).collect();
    let subnet = dst(0xffffff00, SERVICE);
    let specific = dst(0xffffff80, SERVICE + 128);
    let blackhole = dst(0xffffff00, BLACKHOLE);
    let cycle = dst(0xffffff00, CYCLE);
    // The /25 null route overrides the /24. Both ECMP branches see the
    // same accepted /24 packet set; a deny on one path does not cancel
    // the other.
    add(ROUTE, NULL, specific.clone(), None);
    add(ROUTE, NULL, blackhole.clone(), None);
    let accepted = subnet.with(&specific.not()).collect();
    add(ROUTE, LEFT, accepted.clone(), None);
    add(ROUTE, RIGHT, accepted, None);
    add(ROUTE, LOOP_A, cycle.clone(), None);
    add(
        ROUTE,
        NO_ROUTE,
        subnet.union(&blackhole).union(&cycle).collect().not(),
        None,
    );

    let tcp = all.filt(|p| p.protocol.eq(6_u8)).collect();
    add(LEFT, LEFT_OUT, tcp.clone(), Some(Rewrite::Source));
    add(LEFT, LEFT_OUT, tcp.not(), None);
    let http = all
        .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(80_u16))
        .collect();
    let untranslated = http.filt(|p| p.src_ip.ne(PUBLIC)).collect();
    // The left ACL observes the rewritten source; the right branch denies
    // HTTP.
    add(LEFT_OUT, DELIVERED, untranslated.not(), None);
    add(LEFT_OUT, DENY_OUT, untranslated, None);
    add(RIGHT, DENY_OUT, http.clone(), None);
    add(RIGHT, DELIVERED, http.not(), None);
    let remap = all
        .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(8080_u16))
        .collect();
    add(LOOP_A, LOOP_B, remap.clone(), Some(Rewrite::Port));
    add(LOOP_A, LOOP_B, remap.not(), None);
    add(LOOP_B, LOOP_A, all, None);
    add(LOOP_B, DELIVERED, http, None);
    rules
}

fn transition(cx: &Context, rewrite: Rewrite) -> Query<Packet, Packet, impl Exec> {
    Query::<Packet, Packet>::id(cx).map(move |mut p| {
        // Field expressions have no constant constructor, so a rewrite
        // that discards the old value masks it to zero and ors the
        // constant in, as the partial rewrites do.
        #[expect(
            clippy::erasing_op,
            reason = "masking to zero is how a constant field is written"
        )]
        match rewrite {
            Rewrite::Destination => {
                p.dst_ip = (p.dst_ip & 255_u32) | SERVICE;
                p.dst_port = (p.dst_port & 0_u16) | 80_u16;
            }
            Rewrite::Source => p.src_ip = (p.src_ip & 0_u32) | PUBLIC,
            Rewrite::Port => p.dst_port = (p.dst_port & 0_u16) | 80_u16,
        }
        p
    })
}

/// The locations and packets reachable from `seed` along `rules`.
///
/// With `backward` set, `seed` is a set of final states and the result holds
/// the states that can reach them. Guards always test pre-transformation
/// headers, in both directions.
fn reach(cx: &Context, seed: &Rel<u8, Packet>, rules: &[Rule], backward: bool) -> Rel<u8, Packet> {
    let locations: BTreeSet<_> = rules.iter().flat_map(|r| [r.from, r.to]).collect();
    let locations: Vec<_> = locations.into_iter().collect();
    let id = |at: u8| locations.binary_search(&at).unwrap();
    let empty = Rel::<Packet, Packet>::new_set(cx);
    let mut pending: Vec<_> = locations
        .iter()
        .map(|&at| seed.inv().filt(|p| p.eq(at)).keys().collect())
        .collect();
    let mut edges: Vec<_> = locations.iter().map(|_| Vec::new()).collect();
    for rule in rules {
        // Backward rules reverse locations but retain their
        // pre-transformation guards.
        let (from, to) = if backward {
            (rule.to, rule.from)
        } else {
            (rule.from, rule.to)
        };
        let transform = rule.rewrite.map(|r| transition(cx, r));
        edges[id(from)].push((id(to), &rule.guard, transform));
    }
    let mut known = vec![empty.clone(); locations.len()];
    let mut queue: VecDeque<_> = pending
        .iter()
        .enumerate()
        .filter_map(|(i, p)| (!p.is_empty()).then_some(i))
        .collect();
    while let Some(i) = queue.pop_front() {
        let delta = std::mem::replace(&mut pending[i], empty.clone());
        let next = known[i].apply_bool(BoolOp::Or, &delta);
        if next.raw.root.function == known[i].raw.root.function {
            continue;
        }
        known[i] = next;
        // Fixed transfers preserve unions, so a changed incoming batch can
        // propagate without subtracting known packets on every edge,
        // including cycle edges.
        for (j, guard, transform) in &edges[i] {
            // Forward hides pre-NAT headers; backward takes the inverse
            // transformation before matching the guarded rules, so guards
            // still apply to pre-NAT headers.
            let candidates = if backward {
                let preimage = match transform {
                    Some(t) => delta.select(t.inv()).values().collect(),
                    None => delta.clone(),
                };
                preimage.apply_bool(BoolOp::And, guard)
            } else {
                // The location selects its edges before traversing packet
                // partitions, avoiding work on irrelevant frontier
                // locations.
                let accepted = delta.apply_bool(BoolOp::And, guard);
                match transform {
                    Some(t) => accepted.select(t).values().collect(),
                    None => accepted,
                }
            };
            if candidates.is_empty() {
                continue;
            }
            if pending[*j].is_empty() {
                queue.push_back(*j);
            }
            pending[*j] = pending[*j].apply_bool(BoolOp::Or, &candidates);
        }
    }
    // Keep each complete symbolic product together; never enumerate
    // packet values or independently union its fields, which would lose
    // their correlations. Native suffix copying preserves shared packet
    // DAGs during final assembly.
    let (keys, sources): (Vec<_>, Vec<_>) = known
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_empty())
        .map(|(i, p)| (u64::from(locations[i]), &p.raw.root))
        .unzip();
    let mut result = seed.empty_like();
    let children = result.raw.root.copy_suffixes(&sources);
    let roots: Vec<_> = children.iter().map(|p| p.function).collect();
    result.raw.root = result
        .raw
        .root
        .values(Level::new(0), &keys, &roots, &mut Default::default());
    result
}

/// Independent ordinary integer execution: no symbolic guards, maps, or
/// rule table.
fn concrete(packet: Packet) -> BTreeSet<(u8, Packet)> {
    support::closure([(IN, packet)], |(at, current)| {
        let mut p = current.clone();
        let next: &[u8] = match *at {
            IN => {
                let permit = match (p.protocol, p.dst_port) {
                    (6, 22) => p.src_ip == 0x0a000001 || p.src_ip / 256 != 0x0a0000,
                    (6, 80 | 443 | 8080) | (17, 53) => true,
                    _ => false,
                };
                if permit { &[DNAT] } else { &[DENY_IN] }
            }
            DNAT => {
                if p.dst_ip / 256 == VIP / 256 && p.protocol == 6 && p.dst_port == 8080 {
                    p.dst_ip = SERVICE + p.dst_ip % 256;
                    p.dst_port = 80;
                }
                &[ROUTE]
            }
            ROUTE => {
                if p.dst_ip / 128 == (SERVICE + 128) / 128 || p.dst_ip / 256 == BLACKHOLE / 256 {
                    &[NULL]
                } else if p.dst_ip / 256 == SERVICE / 256 {
                    &[LEFT, RIGHT]
                } else if p.dst_ip / 256 == CYCLE / 256 {
                    &[LOOP_A]
                } else {
                    &[NO_ROUTE]
                }
            }
            LEFT => {
                if p.protocol == 6 {
                    p.src_ip = PUBLIC;
                }
                &[LEFT_OUT]
            }
            LEFT_OUT => {
                if p.protocol == 6 && p.dst_port == 80 && p.src_ip != PUBLIC {
                    &[DENY_OUT]
                } else {
                    &[DELIVERED]
                }
            }
            RIGHT => {
                if p.protocol == 6 && p.dst_port == 80 {
                    &[DENY_OUT]
                } else {
                    &[DELIVERED]
                }
            }
            LOOP_A => {
                if p.protocol == 6 && p.dst_port == 8080 {
                    p.dst_port = 80;
                }
                &[LOOP_B]
            }
            LOOP_B => {
                if p.protocol == 6 && p.dst_port == 80 {
                    &[LOOP_A, DELIVERED]
                } else {
                    &[LOOP_A]
                }
            }
            _ => &[],
        };
        next.iter().map(|&to| (to, p.clone())).collect()
    })
}

fn main() {
    let mut cx = Context::default();
    State::bind(&mut cx);
    let rules = rules(&cx);
    let mut seed = Rel::<u8, Packet>::new(&cx);
    let mut packets = Vec::new();
    for src_ip in [0x0a000001, 0x0a000002, 0x0a010101] {
        for dst_ip in [
            VIP + 1,
            VIP + 129,
            SERVICE + 1,
            SERVICE + 129,
            BLACKHOLE + 1,
            CYCLE + 1,
            0xcb007101,
        ] {
            for src_port in [0, 65535] {
                for dst_port in [22, 53, 80, 443, 8080, 81] {
                    for protocol in [6, 17] {
                        packets.push(Packet {
                            src_ip,
                            dst_ip,
                            src_port,
                            dst_port,
                            protocol,
                        });
                    }
                }
            }
        }
    }
    seed.insert_rows(packets.iter().map(|p| (IN, p)));
    let states: Vec<_> = packets.iter().map(|p| concrete(p.clone())).collect();
    let terminals: BTreeSet<_> = states
        .iter()
        .flatten()
        .filter_map(|&(at, _)| (at >= DELIVERED).then_some(at))
        .collect();
    let outcomes: Vec<_> = states
        .iter()
        .map(|s| s.iter().any(|(at, _)| *at == DELIVERED))
        .collect();
    assert_eq!(
        terminals,
        BTreeSet::from([DELIVERED, DENY_IN, DENY_OUT, NULL, NO_ROUTE])
    );
    assert!(outcomes.contains(&true) && outcomes.contains(&false));
    let forward = reach(&cx, &seed, &rules, false);
    support::assert_rows(&forward, states.iter().flatten().map(|(at, p)| (*at, p)));
    assert!(forward.contains((
        LOOP_A,
        &Packet {
            src_ip: 0x0a000001,
            dst_ip: CYCLE + 1,
            src_port: 0,
            dst_port: 80,
            protocol: 6
        }
    )));
    let delivered = seed
        .empty_like()
        .not()
        .inv()
        .filt(|at| at.eq(DELIVERED))
        .inv()
        .collect();
    let backward = reach(&cx, &delivered, &rules, true);
    for (p, expected) in packets.iter().zip(outcomes) {
        assert_eq!(
            backward.contains((IN, p)),
            expected,
            "delivery preimage differs for {p:?}"
        );
    }
    println!(
        "PASS: exact forward states and positive/negative backward delivery for {} packets",
        packets.len()
    );
}

#[test]
fn main_passes() {
    main();
}
