//! White-box tests for `src/util/dag.rs`.
//!
//! These tests pin internal invariants that no public behavior shows: the
//! probe sequence, per-thread region caching, parallel and caller-side table
//! rebuilds, growth arithmetic, reader-counter overflow and admission, and the
//! merging of concurrent growth requests. The public contract is tested in
//! `tests/dag.rs`.
//!
//! Fixtures use a value word with opaque edge words. Arena sizes select the
//! batch and rebuild regimes each test reaches.

use std::cell::Cell;
use std::collections::HashSet;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread;

use super::{
    BATCH_MAX, COUNTER, Dag, Exhausted, GROW_EDGES, GROW_FORCE, GROW_NODES, GROW_UNIQUE, Grow, Idx,
    NeedGrowth, REGION_CELLS, REHASH_CHUNK, Shared, frontier_batch, growth_headroom, region_index,
    structural_hash, unpack_next,
};

idx! { struct Id; }

/// An arena over value words and opaque edge words.
type Arena<I> = Dag<I, u64, u64>;

/// The node and edge allocation frontiers.
fn frontiers<I: Idx, N, E: Copy>(dag: &Dag<I, N, E>) -> (usize, usize) {
    unpack_next(dag.next.load(Ordering::Relaxed))
}

/// The first positions `i < j` whose arenas share a region cache cell.
fn colliding<I: Idx, N, E: Copy>(arenas: &[Dag<I, N, E>]) -> (usize, usize) {
    (0..arenas.len())
        .flat_map(|i| (i + 1..arenas.len()).map(move |j| (i, j)))
        .find(|&(i, j)| region_index(arenas[i].id) == region_index(arenas[j].id))
        .expect("more arenas than cache cells must collide")
}

/// The number of published nodes below the frontier.
fn published<I: Idx>(dag: &Arena<I>) -> usize {
    (0..frontiers(dag).0)
        .filter(|&i| dag.get(I::new(i)).is_some())
        .count()
}

/// An arena and the values reclaimed during maintenance.
struct Store<I: Idx> {
    dag: Arena<I>,
    /// Values passed to the reclaim hook, in reclamation order.
    reclaimed: Vec<u64>,
}

impl<I: Idx> Store<I> {
    fn shared(nodes: usize, edges: usize) -> Shared<Self> {
        Shared::new(Self {
            dag: Dag::with_capacity(nodes, edges, []),
            reclaimed: Vec::new(),
        })
    }
}

impl<I: Idx> Grow for Store<I> {
    type Cx<'a> = ();

    fn grow(&mut self, exhausted: Exhausted, (): ()) {
        let Self { dag, reclaimed } = self;
        dag.maintain(
            exhausted,
            1,
            |_| None,
            |value| reclaimed.push(*value),
            |value, edges| node_hash(*value, edges),
        );
    }
}

/// A node hash using only the low half of its value.
///
/// Varying the high half produces distinct values with the same hash.
fn node_hash(value: u64, edges: &[u64]) -> u64 {
    structural_hash((value as u32, edges))
}

/// Interns a value with its edges under the hash the store's own rebuild
/// recomputes.
fn intern<I: Idx>(dag: &Arena<I>, value: u64, edges: &[u64]) -> Result<I, NeedGrowth> {
    dag.intern(
        node_hash(value, edges),
        |v, stored| *v == value && stored == edges,
        || value,
        edges,
    )
}

/// Interns and pins within one admitted attempt, so no collection can run
/// between them and the node the caller names stays published for the rest of
/// the test.
fn pinned<I: Idx>(shared: &Shared<Store<I>>, value: u64, edges: &[u64]) -> I {
    shared.retry((), |s| {
        intern(&s.dag, value, edges).inspect(|&id| s.dag.pin(id))
    })
}

// --- publication and batch reservation ---------------------------------------

#[test]
fn pinned_iterators_preserve_consecutive_ids_despite_cache_eviction() {
    let dag: Arena<Id> = Dag::with_capacity(
        8192,
        0,
        (0..3).inspect(|_| {
            // Model the cache eviction caused by iterator-side arena publications. Clearing
            // all cells makes this independent of concurrent ID allocation.
            super::REGIONS.with(|cells| {
                for cell in cells {
                    cell.set(super::Region::EMPTY);
                }
            });
        }),
    );
    for i in 0..3 {
        let id = Id::new(i);
        assert_eq!(*dag.node(id), i as u64);
        assert!(dag.has_refs(id));
        assert!(dag.node(id).edges.is_empty());
    }
    assert_eq!(frontiers(&dag), (3, 0));
}

#[test]
fn interleaved_arena_identities_spread_across_cache_cells() {
    // Each store allocates two identities, so its arena receives every other one.
    // Identity modulo the cell count would reach only half of the cells.
    let cells: HashSet<_> = (0..REGION_CELLS as u64)
        .map(|i| region_index(2 * i + 1))
        .collect();
    assert!(cells.len() > REGION_CELLS / 2);
}

#[test]
fn cached_batches_remain_owned_when_arena_cache_cells_collide() {
    // One more arena than cache cells guarantees a collision even when other tests
    // allocate identities concurrently. Only the input selection uses the cache
    // index; assertions check each arena's independent nodes and edges.
    let arenas: Vec<Arena<Id>> =
        Vec::from_fn(REGION_CELLS + 1, |_| Dag::with_capacity(4096, 4096, []));
    let (i, j) = colliding(&arenas);
    let (a, b) = (&arenas[i], &arenas[j]);
    let mut ids = Vec::new();
    for value in 0..4 {
        for dag in [a, b] {
            ids.push((dag, dag.publish(|| value, &[value]).unwrap(), value));
        }
    }
    for dag in [a, b] {
        assert_eq!(published(dag), 4);
        // Each switch evicted a partly unused two-slot region. This checks that the
        // selected arenas really collide, beyond their ID formula.
        assert_eq!(frontiers(dag), (8, 8));
    }
    for (dag, id, value) in ids {
        assert_eq!(*dag.node(id), value);
        assert_eq!(dag.node(id).edges, [value]);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn concurrent_publication_leaves_less_than_one_unused_node_batch_per_worker() {
    const WORKERS: usize = 4;
    const EACH: usize = 10_000;
    let dag: Arena<Id> = Dag::with_capacity(1 << 16, 1 << 17, []);
    thread::scope(|scope| {
        let arena = &dag;
        for w in 0..WORKERS {
            scope.spawn(move || {
                for i in 0..EACH {
                    arena
                        .publish(|| (w as u64) << 32 | i as u64, &[i as u64])
                        .unwrap();
                }
            });
        }
    });
    assert_eq!(published(&dag), WORKERS * EACH);
    // Each thread's final batch is partly unused, so the frontier runs past the
    // published count by at most one batch per worker.
    let (node, _) = frontiers(&dag);
    assert!(node - WORKERS * EACH < WORKERS * frontier_batch(dag.node_capacity()));
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn colliding_arenas_finish_more_than_2048_alternating_publications() {
    const EACH: usize = 2049;
    let capacity = (EACH * BATCH_MAX).next_power_of_two();
    for edge_len in [0, 1] {
        // Select real colliding identities cheaply, then enlarge just those two arenas.
        // Publication does not use their small unique tables. Unit edges keep the large
        // edge capacity free of backing allocations.
        let mut arenas: Vec<Dag<Id, u64, ()>> =
            Vec::from_fn(REGION_CELLS + 1, |_| Dag::with_capacity(1, capacity, []));
        let (i, j) = colliding(&arenas);
        let (left, right) = arenas.split_at_mut(j);
        let (a, b) = (&mut left[i], &mut right[0]);
        for dag in [&mut *a, &mut *b] {
            dag.nodes.resize_with(capacity, super::NodeSlot::new);
        }
        let mut ids = Vec::new();
        for value in 0..EACH {
            for dag in [&*a, &*b] {
                let id = dag.publish(|| value as u64, &[(); 1][..edge_len]).unwrap();
                ids.push((dag, id, value));
            }
        }
        for (dag, id, value) in ids {
            assert_eq!(*dag.node(id), value as u64);
            assert_eq!(dag.node(id).edges.len(), edge_len);
        }
        for dag in [&*a, &*b] {
            assert_eq!(
                frontiers(dag),
                (EACH * BATCH_MAX, EACH * BATCH_MAX * edge_len)
            );
        }
    }
}

#[test]
fn edge_spans_larger_than_the_speculative_cap_remain_contiguous() {
    // Native runs also saturate speculative batch sizing. Miri checks the same
    // oversized contiguous request with a smaller arena.
    let capacity = if cfg!(miri) { BATCH_MAX + 1 } else { 1 << 22 };
    let dag: Dag<Id, u64, ()> = Dag::with_capacity(2, capacity, []);
    let edges = vec![(); BATCH_MAX + 1];
    let id = dag.publish(|| 7, &edges).unwrap();
    assert_eq!(*dag.node(id), 7);
    assert_eq!(dag.node(id).edges.len(), edges.len());
}

#[test]
#[should_panic(
    expected = "cannot unpublish a reclaimed, externally referenced, or pinned DAG node"
)]
fn unpublishing_a_referenced_node_is_rejected() {
    // Collection only unpublishes unmarked slots, so the slot-level check is the
    // last guard against dropping a node a caller still holds.
    let mut dag: Arena<Id> = Dag::with_capacity(2, 0, []);
    let id = dag.publish(|| 1, &[]).unwrap();
    dag.inc_ref(id);
    dag.nodes[id.index()].unpublish();
}

// --- interning ---------------------------------------------------------------

#[test]
fn probing_visits_every_bucket_once() {
    for buckets in [8, 16, 64, 256] {
        for hash in [0, 1, 7, 8, 31, 1 << 32, (3 << 32) | 7, u64::MAX] {
            let mut seen: Vec<_> = super::Probe::new(hash, buckets)
                .map(|(group, bucket)| group * super::GROUP_WIDTH + bucket)
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..buckets).collect::<Vec<_>>());
        }
    }
}

// --- collection and maintenance ----------------------------------------------

#[test]
fn generation_exhaustion_is_permanent() {
    let mut dag: Arena<Id> = Dag::with_capacity(2, 2, []);
    dag.generation = u64::MAX;
    // A caught overflow must leave the generation exhausted, so a later
    // compaction cannot revalidate a stale reservation.
    for _ in 0..2 {
        let panic = panic::catch_unwind(AssertUnwindSafe(|| {
            dag.maintain(
                Exhausted::from(NeedGrowth::Nodes),
                1,
                |_| None,
                |_| {},
                |_, _| 0,
            );
        }))
        .unwrap_err();
        assert_eq!(
            panic.downcast_ref::<String>().map(String::as_str),
            Some("reservation generation overflow")
        );
        assert_eq!(dag.generation, u64::MAX);
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn rebuild_across_threads_keeps_every_retained_entry_findable() {
    // Two chunks of retained entries put helpers to work. Check ordinary hashes and
    // deliberate collisions, preserving canonical IDs and the entry count after
    // collection.
    const EACH: u64 = 3 * REHASH_CHUNK as u64;
    type Hash = fn(u64, &[u64]) -> u64;
    let hashes: [Hash; 2] = [node_hash, |value, _| {
        structural_hash(value % REHASH_CHUNK as u64)
    }];
    for hash in hashes {
        let mut dag: Arena<Id> = Dag::with_capacity(1 << 16, 1 << 17, []);
        let ids: Vec<Id> = (0..EACH)
            .map(|i| {
                let edges = [i];
                let id = dag
                    .intern(
                        hash(i, &edges),
                        |v, stored| *v == i && stored == edges,
                        || i,
                        &edges,
                    )
                    .unwrap();
                dag.inc_ref(id);
                id
            })
            .collect();
        for &id in ids.iter().step_by(3) {
            dag.dec_ref(id);
        }
        let workers = Mutex::new(HashSet::new());
        dag.maintain(
            Exhausted::from(NeedGrowth::Nodes),
            4,
            |_| None,
            |_| {},
            |value, edges| {
                workers.lock().unwrap().insert(thread::current().id());
                hash(*value, edges)
            },
        );
        let workers = workers.into_inner().unwrap();
        assert!(workers.contains(&thread::current().id()));
        assert!(workers.len() > 1);
        assert_eq!(
            dag.unique_len.load(Ordering::Relaxed),
            (EACH - EACH.div_ceil(3)) as usize
        );
        for (i, &id) in ids.iter().enumerate() {
            if i % 3 == 0 {
                assert!(dag.get(id).is_none());
            } else {
                let value = i as u64;
                let edges = [value];
                assert_eq!(
                    dag.intern(
                        hash(value, &edges),
                        |v, stored| *v == value && stored == edges,
                        || value,
                        &edges
                    ),
                    Ok(id)
                );
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "too slow under Miri")]
fn a_helper_panic_during_rebuild_reaches_the_maintenance_caller() {
    // Two chunks of retained entries give the insert phase one helper, which
    // panics while the caller's own chunk completes. The one unreferenced node
    // makes collection reclaim something, so the table is rebuilt.
    let mut dag: Arena<Id> = Dag::with_capacity(1 << 14, 1 << 14, []);
    for i in 0..2 * REHASH_CHUNK as u64 {
        let id = intern(&dag, i, &[]).unwrap();
        dag.inc_ref(id);
    }
    intern(&dag, u64::MAX, &[]).unwrap();
    let caller = thread::current().id();
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        dag.maintain(
            Exhausted::from(NeedGrowth::Nodes),
            2,
            |_| None,
            |_| {},
            |value, edges| {
                if thread::current().id() != caller {
                    panic!("hash interrupted");
                }
                node_hash(*value, edges)
            },
        );
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"hash interrupted"));
}

#[test]
fn small_table_rebuild_hashes_on_the_maintenance_caller() {
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    let retained = intern(&dag, 1, &[]).unwrap();
    dag.inc_ref(retained);
    let reclaimed = intern(&dag, 2, &[]).unwrap();
    let caller = thread::current().id();
    let calls = AtomicUsize::new(0);
    dag.maintain(
        Exhausted::from(NeedGrowth::Nodes),
        4,
        |_| None,
        |_| {},
        |value, edges| {
            assert_eq!(thread::current().id(), caller);
            calls.fetch_add(1, Ordering::Relaxed);
            node_hash(*value, edges)
        },
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(dag.get(reclaimed).is_none());
    assert_eq!(intern(&dag, 1, &[]), Ok(retained));
}

#[test]
fn reclaiming_published_nodes_preserves_an_empty_unique_table() {
    let mut dag: Arena<Id> = Dag::with_capacity(2, 0, []);
    let groups = dag.groups.as_ptr();
    let buckets = dag.buckets();
    let id = dag.publish(|| 7, &[]).unwrap();
    let mut reclaimed = Vec::new();
    dag.maintain(
        Exhausted::from(NeedGrowth::Nodes),
        1,
        |_| None,
        |value| reclaimed.push(*value),
        |value, edges| node_hash(*value, edges),
    );
    assert_eq!(reclaimed, [7]);
    assert!(dag.get(id).is_none());
    assert_eq!(dag.groups.as_ptr(), groups);
    assert_eq!(dag.buckets(), buckets);
    assert_eq!(dag.unique_len.load(Ordering::Relaxed), 0);

    let canonical = intern(&dag, 7, &[]).unwrap();
    assert_eq!(intern(&dag, 7, &[]), Ok(canonical));
    assert_eq!(*dag.node(canonical), 7);
    assert_eq!(dag.unique_len.load(Ordering::Relaxed), 1);
}

#[test]
#[should_panic(expected = "arena capacity overflow")]
fn forcing_growth_of_a_maximal_zero_sized_edge_arena_is_rejected() {
    let mut dag: Dag<Id, u64, ()> = Dag::with_capacity(1, 0, []);
    // SAFETY: Slot<()> is zero-sized and permits an uninitialized payload. Its
    // vector has capacity usize::MAX, so these elements require no storage.
    unsafe { dag.edges.set_len(u32::MAX as usize) };
    dag.maintain(
        Exhausted {
            flags: GROW_EDGES | GROW_FORCE,
            edge_len: 1,
        },
        1,
        |_| None,
        |_| {},
        |value, edges| structural_hash((value, edges)),
    );
}

// --- growth policy -----------------------------------------------------------

#[test]
fn growth_headroom_matches_three_quarters_without_overflow() {
    for capacity in [0, 1, 3, 4, 100, 100_663_296, u32::MAX as usize, usize::MAX] {
        assert_eq!(
            growth_headroom(capacity) as u128,
            capacity as u128 * 3 / 4 + 1
        );
    }
}

// --- concurrent admission ----------------------------------------------------

#[test]
fn admission_at_the_counter_limit_is_released_on_drop() {
    let shared = Store::<Id>::shared(2, 2);
    let counter = &shared.readers[super::counter_index()].0;
    counter.store(u32::MAX - 1, Ordering::Relaxed);
    let s = shared.read(());
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX);
    drop(s);
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX - 1);
    counter.store(0, Ordering::Relaxed);
}

#[test]
fn admission_overflow_preserves_a_retained_reader() {
    let shared = Store::<Id>::shared(2, 2);
    let id = pinned(&shared, 7, &[]);
    let s = shared.read(());
    let counter = &shared.readers[super::counter_index()].0;
    // Simulate forgotten guards while retaining one real borrow of the store.
    counter.store(u32::MAX, Ordering::Relaxed);
    let panic = panic::catch_unwind(AssertUnwindSafe(|| shared.read(())))
        .err()
        .expect("admission must reject overflow");
    assert_eq!(
        panic.downcast_ref::<String>().map(String::as_str),
        Some("reader count overflow: 4294967295"),
    );
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX);
    assert_eq!(*s.dag.node(id), 7);
    drop(s);
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX - 1);
    counter.store(0, Ordering::Relaxed);
}

#[test]
fn retry_overflow_rejects_admission_before_calling_the_attempt() {
    let shared = Store::<Id>::shared(2, 2);
    let counter = &shared.readers[super::counter_index()].0;
    counter.store(u32::MAX, Ordering::Relaxed);
    let called = Cell::new(false);
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        shared.retry((), |_| {
            called.set(true);
            Ok(())
        });
    }));
    assert!(panic.is_err());
    assert!(!called.get());
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX);
    assert_eq!(shared.requested.load(Ordering::Relaxed), 0);
    assert_eq!(shared.epoch(), 0);
    counter.store(0, Ordering::Relaxed);
}

#[test]
fn a_moved_guard_can_read_and_release_after_maintenance_is_requested() {
    let shared = &Store::<Id>::shared(2, 2);
    let id = pinned(shared, 7, &[7]);
    let guard = shared.read(());
    thread::scope(|scope| {
        scope.spawn(move || {
            // Observe only the request, leaving the admission counter as the
            // reader-to-writer synchronization for the store access below.
            while shared.requested.load(Ordering::Relaxed) == 0 {
                thread::yield_now();
            }
            assert_eq!(*guard.dag.node(id), 7);
            assert_eq!(guard.dag.node(id).edges, [7]);
            drop(guard);
        });
        let mut reason = Some(NeedGrowth::Nodes);
        shared.retry((), |_| reason.take().map_or(Ok(()), Err));
    });
    assert_eq!(shared.epoch(), 1);
}

#[test]
fn readers_sharing_a_counter_release_each_admission() {
    let shared = &Store::<Id>::shared(2, 2);
    let id = pinned(shared, 7, &[7]);
    let ready = &Barrier::new(3);
    let first = &Barrier::new(2);
    let second = &Barrier::new(2);
    thread::scope(|scope| {
        let a = scope.spawn(move || {
            COUNTER.set(0);
            let guard = shared.read(());
            ready.wait();
            first.wait();
            assert_eq!(*guard.dag.node(id), 7);
            drop(guard);
        });
        scope.spawn(move || {
            COUNTER.set(0);
            let guard = shared.read(());
            ready.wait();
            second.wait();
            assert_eq!(guard.dag.node(id).edges, [7]);
            drop(guard);
        });
        ready.wait();
        let initial = shared.readers[0].0.load(Ordering::Relaxed);
        scope.spawn(move || {
            // Only the two readers share counter zero in this fixture.
            COUNTER.set(1);
            let mut reason = Some(NeedGrowth::Nodes);
            shared.retry((), |_| reason.take().map_or(Ok(()), Err));
        });
        while shared.requested.load(Ordering::Relaxed) == 0 {
            thread::yield_now();
        }
        first.wait();
        let result = a.join();
        let remaining = shared.readers[0].0.load(Ordering::Relaxed);
        let epoch = shared.epoch();
        // Release the waiting reader before an assertion can unwind the scope.
        second.wait();
        result.unwrap();
        assert_eq!(initial, 2);
        assert_eq!(remaining, 1);
        assert_eq!(epoch, 0);
    });
    assert_eq!(shared.readers[0].0.load(Ordering::Relaxed), 0);
    assert_eq!(shared.epoch(), 1);
}

#[test]
fn concurrent_failures_merge_flags_and_the_largest_edge_span() {
    struct Passes(Vec<Exhausted>);

    impl Grow for Passes {
        type Cx<'a> = ();

        fn grow(&mut self, exhausted: Exhausted, (): ()) {
            self.0.push(exhausted);
        }
    }

    let mut shared = Shared::new(Passes(Vec::new()));
    let ready = &Barrier::new(4);
    thread::scope(|scope| {
        let store = &shared;
        for reason in [
            NeedGrowth::Nodes,
            NeedGrowth::Unique,
            NeedGrowth::Edges(3),
            NeedGrowth::Edges(9),
        ] {
            scope.spawn(move || {
                let mut failure = Some(reason);
                store.retry((), |_| {
                    let Some(exhausted) = failure.take() else {
                        return Ok(());
                    };
                    ready.wait();
                    Err(exhausted)
                });
            });
        }
    });
    assert_eq!(shared.epoch(), 1);
    let calls = &shared.get_mut().0;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].flags, GROW_NODES | GROW_UNIQUE | GROW_EDGES);
    assert_eq!(calls[0].edge_len, 9);
}
