//! End-to-end scenarios, re-authored from the crate's examples as tests or
//! including an example directly.
//!
//! These sweep the whole stack - typed frontend, combinators, field
//! predicates, the outer index, the CPS executor, and both backends - in a
//! handful of tests, so the module-level tests can stay focused on the
//! branches these leave uncovered.

#[path = "../examples/concurrency.rs"]
mod concurrency_example;

// --- iteration ----
// Semi-naive iteration: reachability through a cycle with a guarded bit
// rewrite.

mod iteration {
    use std::collections::{BTreeSet, VecDeque};

    use sra::typed::{IntoQuery, Rel, Sorted, UInt, Variable};
    use sra::{Bv, Context, Schema};

    #[derive(Schema)]
    struct IterState {
        location: u8,
        #[symbolic]
        packet: UInt<3>,
    }

    #[test]
    fn iteration_reaches_every_concrete_state() {
        let mut cx = Context::default();
        IterState::bind(&mut cx);
        let IterStateSchema {
            location, packet, ..
        } = IterState::schema(&cx);
        // Explicit declarations keep the concrete location and symbolic packet
        // columns in every collected frontier. The relation stores current
        // states, not original/current pairs.
        let mut seed: Rel<u8, UInt<3>> =
            Rel::from_raw(sra::Rel::new(&cx, location.raw, packet.raw));
        seed.insert((0_u8, Bv::new(&cx, packet.raw).extract(2, 2).eq(&cx, 0_u8)));
        let mut plain: Rel<u8, u8> = Rel::from_raw(sra::Rel::new(&cx, location.raw, location.raw));
        plain.insert_rows([(0_u8, 1_u8), (2, 1)]);
        let mut rewrite = plain.empty_like();
        rewrite.insert((1_u8, 2_u8));

        // Independent concrete traversal: ordinary integer conditions and a
        // queue, with no engine predicates or frontier algebra. Location 1
        // later gains packets 4 and 6 under an old key.
        let mut expected_rows: BTreeSet<_> = (0_u8..4).map(|p| (0, p)).collect();
        let mut queue: VecDeque<_> = expected_rows.iter().copied().collect();
        while let Some((loc, p)) = queue.pop_front() {
            let next = match loc {
                0 | 2 => Some((1, p)),
                1 if p % 2 == 0 => Some((2, p | 4)),
                _ => None,
            };
            if let Some(pair) = next
                && expected_rows.insert(pair)
            {
                queue.push_back(pair);
            }
        }
        assert_eq!(expected_rows.len(), 12);

        let mut reach = Variable::new(seed);
        while reach.changed() {
            let recent = reach.recent();
            // Inverse edges compose destination -> previous location ->
            // packet. Only the rewrite edge requires even packets; setting
            // bit 2 may reproduce an already accepted packet.
            let forwarded = plain.inv().select(&recent);
            let rewritten = rewrite
                .inv()
                .select(&recent)
                .filt(|p| (p & 1_u8).eq(0_u8))
                .map(|p| p | 4_u8);
            reach.insert(&forwarded.union(rewritten).collect());
        }
        let actual = reach.complete();
        let mut expected = actual.empty_like();
        expected.insert_rows(
            expected_rows
                .iter()
                .map(|&(loc, p)| (loc, UInt::<3>::new(p.into()))),
        );
        assert!(actual == expected);
    }
}

// --- selection ----
// Per-key minimum under a symbolic key: the engine's `min_range` against the
// anti-join.

mod selection {
    use sra::typed::{IntoQuery, Rel, Sorted, UInt};
    use sra::{BoolOp, Bv, Context, Predicate, Schema};

    #[derive(Schema)]
    struct Key {
        #[symbolic]
        env: UInt<3>,
        router: u8,
    }

    #[test]
    fn min_range_matches_the_anti_join_oracle_per_env() {
        let mut cx = Context::default();
        Key::bind(&mut cx);
        let KeySchema { env, .. } = Key::schema(&cx);
        let bit = |i: u32, v: u64| Bv::new(&cx, env.raw).extract(i, i).eq(&cx, v);
        // Overlapping symbolic cells: each environment sees a different
        // candidate set.
        let cands_by_cell: Vec<(fn(u64) -> bool, Predicate, u8, u8)> = vec![
            (|_| true, cx.t(cx.field(env.raw).unwrap()), 0, 9),
            (|e| e & 1 == 1, bit(0, 1), 0, 4),
            (|e| e & 2 == 2, bit(1, 1), 0, 6),
            (|e| e & 1 == 0, bit(0, 0), 1, 3),
            (|e| e & 4 == 4, bit(2, 1), 1, 1),
            (|e| e & 2 == 0, bit(1, 0), 1, 8),
        ];
        let mut cands: Rel<Key, u8> = Rel::new(&cx);
        cands.insert_rows(cands_by_cell.iter().map(|(_, p, v, r)| ((p, *v), *r)));
        let mut lt: Rel<u8, u8> = Rel::new(&cx);
        lt.insert_rows((0..16_u8).flat_map(|r| (0..r).map(move |q| (q, r))));

        let anti_join = cands.apply_bool(BoolOp::AndNot, &(&cands).select(&lt).collect());
        let selected = cands.min_range();
        assert!(selected == anti_join);

        // A plain per-environment reference: the least rank each router is
        // offered.
        let mut expected = cands.empty_like();
        let cells = &cands_by_cell;
        expected.insert_rows((0..8_u64).flat_map(|e| {
            [0_u8, 1].into_iter().filter_map(move |v| {
                cells
                    .iter()
                    .filter(|(holds, _, w, _)| holds(e) && *w == v)
                    .map(|(_, _, _, r)| *r)
                    .min()
                    .map(|r| ((UInt::<3>::new(e.into()), v), r))
            })
        }));
        assert!(selected == expected);
        // Router 1 has no candidate in environment 3, and an absent key stays
        // absent.
        assert_eq!(selected.count(), 15_u32.into());
    }
}

// --- symbols ----
// Interned symbols, bounded membership masks, and symbolic set updates.

mod symbols {
    use std::ops::ControlFlow;

    use sra::typed::{
        Capacity, Dictionary, Domain, FiniteSet, Interned, IntoQuery, Query, Rel, Sorted,
    };
    use sra::{Context, Schema};

    struct RouterNames;
    impl Domain for RouterNames {
        type Value = str;
        const NAME: &'static str = "RouterName";
    }
    type Router = Interned<RouterNames>;

    /// Membership bits positioned by router ID; the first 64 interned routers
    /// are set members.
    type PathHops = FiniteSet<Capacity<RouterNames, 64>>;

    #[derive(Schema)]
    struct SymRoute {
        router: Router,
        next_hop: Router,
        #[symbolic]
        path_hops: PathHops,
    }

    #[test]
    fn symbols_intern_and_update_membership_masks() {
        let mut cx = Context::default();
        SymRoute::bind(&mut cx);
        let names = Dictionary::<RouterNames>::new(&cx);
        let r1 = names.intern("r1");
        assert_eq!(names.intern("r1"), r1);
        let r2 = names.intern("r2");
        let SymRouteSchema {
            router,
            next_hop,
            path_hops,
            ..
        } = SymRoute::schema(&cx);
        let mut routes = Rel::<SymRoute, SymRoute>::new_set(&cx);
        routes.insert((
            "r1",
            "r2",
            PathHops::from_members(&cx, [names.intern("r2")]),
        ));
        routes.insert((
            "r2",
            "r1",
            PathHops::from_members(&cx, [names.intern("r1")]),
        ));
        assert_eq!(names.get("r2"), Some(r2));
        let r3 = names.intern("r3");
        routes.insert(("r1", "r3", PathHops::from_members(&cx, [r3])));
        routes.insert(("r3", "r1", PathHops::from_members(&cx, [r2])));

        // Symbolic membership over the capacity bitmask needs no name list.
        let affected = router.eq("r1") | path_hops.contains(&cx, names.intern("r1"));
        assert_eq!((&routes).with(affected).count(), 3_u32.into());

        // Expression operations add a member symbolically, with the dictionary
        // interning its name.
        let extended = (&routes)
            .select(path_hops)
            .map(|h| h.insert(names.intern("r3")))
            .collect();
        let mut expected = extended.empty_like();
        for (r, hop, before, after) in [
            (r1, r2, vec![r2], vec![r2, r3]),
            (r2, r1, vec![r1], vec![r1, r3]),
            (r1, r3, vec![r3], vec![r3]),
            (r3, r1, vec![r2], vec![r2, r3]),
        ] {
            expected.insert((
                SymRoute {
                    router: r,
                    next_hop: hop,
                    path_hops: PathHops::from_members(&cx, before),
                },
                PathHops::from_members(&cx, after),
            ));
        }
        assert!(extended == expected);

        let mut known = Rel::<Router, Router>::new_set(&cx);
        known.insert_rows(names.iter().map(|(id, _)| id));
        let other = sra::typed::pred::ne("r1");
        // Construct the predicate before adding a name: its meaning does not
        // depend on the table.
        let allowed = Query::<Router, Router>::id(&cx).with(&other).collect();
        let r4 = names.intern("r4");
        assert!(allowed.contains(r4));
        assert!(!known.contains(r4));
        known.insert(r4);

        let mut reachable = Rel::<Router, Router>::new_set(&cx);
        reachable.insert("r2");
        let selected = routes
            .with(router.eq("r1"))
            .select(next_hop)
            .with(&reachable);
        let mut hops = Vec::new();
        let _: ControlFlow<()> = selected.drive(&mut |_, r| {
            let p = r.predicate(0);
            let mut cursor = p.context().resolve(p.column()).members(p.function());
            let mut value = Vec::new();
            while cursor.next(&mut value) {
                hops.push(names.resolve(names.interned(value[0])).unwrap().to_string());
            }
            ControlFlow::Continue(())
        });
        assert_eq!(hops, ["r2"]);
        assert_eq!(selected.count(), 1_u32.into());
        assert_eq!(known.with(&other).count(), 3_u32.into());
        assert_eq!(names.resolve(names.interned(u64::MAX)), None);
    }
}

// --- custom_types ----
// Application-defined scalars and a fixed community universe.

mod custom_types {
    use std::net::Ipv4Addr;

    use sra::row::{RowRef, RowSink};
    use sra::typed::{Cons, FiniteSet, IntoQuery, Nil, Rel, Sorted, Universe};
    use sra::{Context, IntoRow, ScalarSort, Schema, Sort};

    // A local scalar needs only its sort/interpretation and concrete packing.
    // Every u32 bit pattern denotes an Asn here; restrictions on acceptable AS
    // numbers are relation contents.
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
    struct CustomPrefix {
        #[symbolic]
        addr: Ipv4Addr,
        #[symbolic]
        len: u8,
    }

    #[derive(Schema)]
    struct CustomRoute {
        prefix: CustomPrefix,
        next_hop: Ipv4Addr,
        origin: Asn,
        communities: CommunitySet,
    }

    #[test]
    fn custom_types_select_per_field_storage() {
        let mut cx = Context::default();
        CustomRoute::bind(&mut cx);
        let CustomRouteSchema {
            prefix,
            next_hop,
            origin,
            communities,
            ..
        } = CustomRoute::schema(&cx);
        let CustomPrefixSchema { addr, len, .. } = CustomPrefix::schema(&cx);
        let columns = [
            (addr.raw, "bdd", 32),
            (len.raw, "bdd", 8),
            (next_hop.raw, "sparse", 32),
            (origin.raw, "sparse", 32),
            (communities.raw, "sparse", 3),
        ]
        .map(|(field, backend, width)| {
            let column = cx.column(cx.backend(backend), ScalarSort::bit_vec(width));
            assert_eq!(cx.field(field), Some(column));
            column
        });
        let p10 = Ipv4Addr::new(10, 0, 0, 0);
        let p11 = Ipv4Addr::new(11, 0, 0, 0);
        let hop = Ipv4Addr::new(10, 0, 0, 1);
        let other = Ipv4Addr::new(10, 0, 0, 2);
        let route = |a, h, o, c| CustomRoute {
            prefix: CustomPrefix { addr: a, len: 24 },
            next_hop: h,
            origin: Asn(o),
            communities: c,
        };
        let members = || CommunitySet::from_members(&cx, [Community::Customer, Community::Peer]);
        let mut routes = Rel::<CustomRoute, CustomRoute>::new_set(&cx);
        routes.insert(route(p10, hop, 64512, members()));
        routes.insert(route(p11, other, 64513, members()));
        routes.insert(route(
            p10,
            other,
            64514,
            CommunitySet::from_members(&cx, [Community::Peer]),
        ));
        for (i, column) in columns.into_iter().enumerate() {
            assert_eq!(routes.raw.root.column(i), column);
        }

        let selected = routes.with(prefix.query(&cx).select(addr).eq(p10)).with(
            communities
                .query(&cx)
                .filt(|s| s.contains(Community::Customer)),
        );
        let hops = selected.select(next_hop).collect();
        let sets = routes
            .select(communities)
            .map(|s| s.insert(Community::Transit).remove(Community::Peer))
            .collect();
        let mut expected_hops = Rel::<CustomRoute, Ipv4Addr>::new(&cx);
        expected_hops.insert((route(p10, hop, 64512, members()), hop));
        assert!(hops == expected_hops);
        let mut expected_sets = Rel::<CustomRoute, CommunitySet>::new(&cx);
        for (a, h, o, before, after) in [
            (p10, hop, 64512, 0b011, 0b101),
            (p11, other, 64513, 0b011, 0b101),
            (p10, other, 64514, 0b010, 0b100),
        ] {
            expected_sets.insert((
                route(a, h, o, CommunitySet::from_words(&cx, [before])),
                CommunitySet::from_words(&cx, [after]),
            ));
        }
        assert!(sets == expected_sets);
    }
}

// --- raw ----
// Shared context, source snapshots, and caller-partitioned collection (untyped
// API).

mod raw {
    use std::ops::ControlFlow;
    use std::thread;

    use sra::{BoolOp, Context, IntoQuery, Rel};

    use crate::concurrency_example::collect_partitioned;

    #[test]
    fn concurrency_shares_queries_across_scoped_workers() {
        let mut cx = Context::default();
        let node = cx.tuple_decl("Node", [("id", cx.bv_sort(3))]);
        let id = node.accessors[0];
        let backend = cx.backend("bdd");
        cx.set_representation(id, backend);
        let mut edges = Rel::new(&cx, id, id);
        edges.insert_rows([(0_u64, 1_u64), (0, 2), (1, 3), (2, 3), (4, 1), (5, 2)]);

        let query = edges.select(&edges);
        let expected = query.collect();
        let mut paths = Rel::new(&cx, id, id);
        paths.insert_rows([(0_u64, 3_u64), (4, 3), (5, 3)]);
        assert!(expected == paths);

        // The query owns source snapshots. All workers borrow the same context
        // and query, while each terminal creates its own state and each
        // worker owns its callback.
        thread::scope(|scope| {
            let workers = [0, 1].map(|_| {
                scope.spawn(|| {
                    assert_eq!(cx.bv_sort(3), id.resolve(&cx).range);
                    assert!(!query.is_empty());
                    let mut visits = 0;
                    let _ = query.drive(&mut |_, _| {
                        visits += 1;
                        // A nested invocation has independent state too.
                        assert!(!query.is_empty());
                        ControlFlow::Break(())
                    });
                    assert_eq!(visits, 1);
                    query.collect()
                })
            });
            for worker in workers {
                assert!(worker.join().unwrap() == expected);
            }
        });

        // Partition the source coordinate before the first occurrence, keeping
        // the second whole.
        const WORKERS: usize = 2;
        const CHUNKS_PER_WORKER: usize = 8;
        let result = collect_partitioned(&edges, &edges, WORKERS * CHUNKS_PER_WORKER, WORKERS);
        assert!(result == expected);
        // An uneven count yields multi-value ranges of different lengths.
        assert!(collect_partitioned(&edges, &edges, 3, WORKERS) == expected);

        // Separately restrict just the first occurrence by its range, using
        // symbolic restrictions. Keep the second occurrence whole: applying
        // the same range restriction to both loses the path 4 -> 1 -> 3.
        // These checks collect complete compositions, not input shards.
        let results = [
            edges.with(..2_u64).select(&edges).collect(),
            edges.with(2_u64..).select(&edges).collect(),
        ];
        // Both disjoint intermediate-key ranges connect 0 to 3. Composition
        // hides their witnesses, so merge by exact native union. This
        // decomposition uses positive operators.
        let mut low = Rel::new(&cx, id, id);
        low.insert_rows([(0_u64, 3_u64), (4, 3)]);
        let mut high = Rel::new(&cx, id, id);
        high.insert_rows([(0_u64, 3_u64), (5, 3)]);
        assert!(results[0] == low && results[1] == high);
        assert!(results[0].apply_bool(BoolOp::Or, &results[1]) == expected);

        // A later write leaves the built query's snapshot alone and reaches
        // the next query.
        edges.insert((3_u64, 5_u64));
        assert!(query.collect() == expected);
        assert!(edges.select(&edges).collect().contains((1_u64, 5_u64)));
    }
}

// --- packet_reachability ----
// Forward and backward packet reachability over a small authored network.

mod packet_reachability {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    use sra::index::Level;
    use sra::run::Exec;
    use sra::typed::{IntoQuery, Query, Rel, Sorted};
    use sra::util::dag::Idx;
    use sra::{BoolOp, Context, Schema};

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
    // 192.0.2.0/24
    const VIP: u32 = 0xc0000200;
    // 10.64.4.0/24
    const SERVICE: u32 = 0x0a400400;
    // 198.19.0.0/24
    const CYCLE: u32 = 0xc6130000;
    // 198.18.0.0/24
    const BLACKHOLE: u32 = 0xc6120000;
    // 198.51.100.7
    const PUBLIC: u32 = 0xc6336407;

    #[derive(Schema)]
    struct NetState {
        location: u8,
        packet: Packet,
    }

    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    enum Rewrite {
        Identity,
        Destination,
        Source,
        Port,
    }

    struct Rule {
        from: u8,
        to: u8,
        guard: Rel<Packet, Packet>,
        rewrite: Rewrite,
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
        let ssh = Query::<Packet, Packet>::id(cx)
            .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(22_u16))
            .collect();
        let blocked = (&ssh)
            .filt(|p| (p.src_ip & 0xffffff00_u32).eq(0x0a000000_u32))
            .collect();
        let exception = (&ssh).filt(|p| p.src_ip.eq(0x0a000001_u32)).collect();
        let service = Query::<Packet, Packet>::id(cx)
            .filt(|p| {
                (p.protocol.clone().eq(6_u8)
                    & (p.dst_port.clone().eq(80_u16)
                        | p.dst_port.clone().eq(443_u16)
                        | p.dst_port.clone().eq(8080_u16)))
                    | (p.protocol.eq(17_u8) & p.dst_port.eq(53_u16))
            })
            .collect();
        // Ordered ACL: SSH exception, blocked source /24, then general
        // SSH/services.
        let allowed = exception
            .union(ssh.with(&blocked.not()))
            .union(&service)
            .collect();
        add(IN, DNAT, allowed.clone(), Rewrite::Identity);
        add(IN, DENY_IN, allowed.not(), Rewrite::Identity);

        let vip = Query::<Packet, Packet>::id(cx)
            .filt(|p| {
                (p.dst_ip & 0xffffff00_u32).eq(VIP) & p.protocol.eq(6_u8) & p.dst_port.eq(8080_u16)
            })
            .collect();
        add(DNAT, ROUTE, vip.clone(), Rewrite::Destination);
        add(DNAT, ROUTE, vip.not(), Rewrite::Identity);

        let subnet = Query::<Packet, Packet>::id(cx)
            .filt(|p| (p.dst_ip & 0xffffff00_u32).eq(SERVICE))
            .collect();
        let specific = Query::<Packet, Packet>::id(cx)
            .filt(|p| (p.dst_ip & 0xffffff80_u32).eq(SERVICE + 128))
            .collect();
        let blackhole = Query::<Packet, Packet>::id(cx)
            .filt(|p| (p.dst_ip & 0xffffff00_u32).eq(BLACKHOLE))
            .collect();
        let cycle = Query::<Packet, Packet>::id(cx)
            .filt(|p| (p.dst_ip & 0xffffff00_u32).eq(CYCLE))
            .collect();
        // The /25 null route overrides the /24. Both ECMP branches see the
        // same accepted /24 packet set; a deny on one path does not cancel
        // the other.
        add(ROUTE, NULL, specific.clone(), Rewrite::Identity);
        add(ROUTE, NULL, blackhole.clone(), Rewrite::Identity);
        add(
            ROUTE,
            LEFT,
            (&subnet).with(&specific.not()).collect(),
            Rewrite::Identity,
        );
        add(
            ROUTE,
            RIGHT,
            (&subnet).with(&specific.not()).collect(),
            Rewrite::Identity,
        );
        add(ROUTE, LOOP_A, cycle.clone(), Rewrite::Identity);
        add(
            ROUTE,
            NO_ROUTE,
            subnet.union(&blackhole).union(&cycle).collect().not(),
            Rewrite::Identity,
        );

        let tcp = Query::<Packet, Packet>::id(cx)
            .filt(|p| p.protocol.eq(6_u8))
            .collect();
        add(LEFT, LEFT_OUT, tcp.clone(), Rewrite::Source);
        add(LEFT, LEFT_OUT, tcp.not(), Rewrite::Identity);
        let http = Query::<Packet, Packet>::id(cx)
            .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(80_u16))
            .collect();
        let translated = (&http).filt(|p| p.src_ip.eq(PUBLIC)).collect();
        // The left ACL observes the rewritten source; the right branch denies
        // HTTP.
        add(
            LEFT_OUT,
            DELIVERED,
            translated.union(&http.not()).collect(),
            Rewrite::Identity,
        );
        add(
            LEFT_OUT,
            DENY_OUT,
            (&http).filt(|p| p.src_ip.ne(PUBLIC)).collect(),
            Rewrite::Identity,
        );
        add(RIGHT, DENY_OUT, http.clone(), Rewrite::Identity);
        add(RIGHT, DELIVERED, http.not(), Rewrite::Identity);
        let remap = Query::<Packet, Packet>::id(cx)
            .filt(|p| p.protocol.eq(6_u8) & p.dst_port.eq(8080_u16))
            .collect();
        add(LOOP_A, LOOP_B, remap.clone(), Rewrite::Port);
        add(LOOP_A, LOOP_B, remap.not(), Rewrite::Identity);
        add(LOOP_B, LOOP_A, all, Rewrite::Identity);
        add(LOOP_B, DELIVERED, http, Rewrite::Identity);
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
                Rewrite::Identity => {}
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

    fn reach(
        cx: &Context,
        seed: &Rel<u8, Packet>,
        rules: &[Rule],
        backward: bool,
    ) -> Rel<u8, Packet> {
        let locations: BTreeSet<_> = rules.iter().flat_map(|r| [r.from, r.to]).collect();
        let locations: Vec<_> = locations.into_iter().collect();
        let id = |at: u8| locations.binary_search(&at).unwrap();
        let empty = Rel::<Packet, Packet>::new_set(cx);
        let mut pending: Vec<_> = locations
            .iter()
            .map(|&at| {
                Query::<Packet, Packet>::id(cx)
                    .with(seed.inv().filt(|p| p.eq(at)))
                    .collect()
            })
            .collect();
        let mut guards: BTreeMap<_, Rel<Packet, Packet>> = BTreeMap::new();
        for rule in rules {
            guards
                .entry((rule.rewrite, rule.from, rule.to))
                .and_modify(|p| *p = p.apply_bool(BoolOp::Or, &rule.guard))
                .or_insert_with(|| rule.guard.clone());
        }
        let mut transforms = BTreeMap::new();
        let mut edges = vec![Vec::new(); locations.len()];
        for ((rewrite, from, to), guard) in guards {
            // Backward rules reverse locations but retain their
            // pre-transformation guards.
            let (from, to) = if backward { (to, from) } else { (from, to) };
            if rewrite != Rewrite::Identity {
                transforms
                    .entry(rewrite)
                    .or_insert_with(|| transition(cx, rewrite));
            }
            edges[id(from)].push((id(to), guard, rewrite));
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
            for (j, guard, rewrite) in &edges[i] {
                // Forward hides pre-NAT headers; backward takes the inverse
                // transformation before matching the guarded rules, so guards
                // still apply to pre-NAT headers.
                let candidates = if backward {
                    let preimage = match transforms.get(rewrite) {
                        Some(t) => (&delta).select(t.inv()).values().collect(),
                        None => delta.clone(),
                    };
                    preimage.apply_bool(BoolOp::And, guard)
                } else {
                    // The location selects its edges before traversing packet
                    // partitions, avoiding work on irrelevant frontier
                    // locations.
                    let accepted = delta.apply_bool(BoolOp::And, guard);
                    match transforms.get(rewrite) {
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
        result.raw.root =
            result
                .raw
                .root
                .values(Level::new(0), &keys, &roots, &mut Default::default());
        result
    }

    // Independent ordinary integer execution: no symbolic guards, maps, or
    // rule table.
    fn concrete(packet: Packet) -> BTreeSet<(u8, Packet)> {
        let mut seen = BTreeSet::from([(IN, packet)]);
        let mut queue: VecDeque<_> = seen.iter().cloned().collect();
        while let Some((at, mut p)) = queue.pop_front() {
            let next: &[u8] = match at {
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
                    if p.dst_ip / 128 == (SERVICE + 128) / 128 || p.dst_ip / 256 == BLACKHOLE / 256
                    {
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
            for &to in next {
                if seen.insert((to, p.clone())) {
                    queue.push_back((to, p.clone()));
                }
            }
        }
        seen
    }

    #[test]
    fn packet_reachability_matches_concrete_traversal() {
        let mut cx = Context::default();
        NetState::bind(&mut cx);
        let rules = rules(&cx);
        let NetStateSchema {
            location, packet, ..
        } = NetState::schema(&cx);
        let mut seed = Rel::from_raw(sra::Rel::new(&cx, location.raw, packet.raw));
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
        let mut expected = seed.empty_like();
        let mut outcomes = Vec::new();
        let mut terminals = BTreeSet::new();
        for p in &packets {
            let states = concrete(p.clone());
            terminals.extend(
                states
                    .iter()
                    .filter_map(|(at, _)| (*at >= DELIVERED).then_some(*at)),
            );
            outcomes.push(states.iter().any(|(at, _)| *at == DELIVERED));
            expected.insert_rows(states.iter().map(|(at, q)| (*at, q)));
        }
        assert_eq!(
            terminals,
            BTreeSet::from([DELIVERED, DENY_IN, DENY_OUT, NULL, NO_ROUTE])
        );
        assert!(outcomes.contains(&true) && outcomes.contains(&false));
        assert!(reach(&cx, &seed, &rules, false) == expected);
        // Seeds cover most rewrite outputs in the aggregate, so a lost output
        // shows only per packet.
        for p in &packets {
            let mut single = seed.empty_like();
            single.insert((IN, p));
            let mut states = seed.empty_like();
            states.insert_rows(concrete(p.clone()).iter().map(|(at, q)| (*at, q)));
            assert!(
                reach(&cx, &single, &rules, false) == states,
                "forward states differ for {p:?}"
            );
        }
        let p = Packet::schema(&cx);
        let full = |f: sra::FuncDecl| cx.t(cx.field(f).unwrap());
        let mut delivered = seed.empty_like();
        delivered.insert((
            DELIVERED,
            (
                full(p.dst_ip.raw),
                full(p.src_ip.raw),
                full(p.dst_port.raw),
                full(p.src_port.raw),
                full(p.protocol.raw),
            ),
        ));
        let backward = reach(&cx, &delivered, &rules, true);
        for (q, want) in packets.iter().zip(outcomes) {
            assert_eq!(
                backward.contains((IN, q)),
                want,
                "delivery preimage differs for {q:?}"
            );
        }
    }
}
