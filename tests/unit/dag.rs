//! Tests for `src/util/dag.rs`.
//!
//! Fixtures use a value word and edges holding child links or opaque values.
//! They exercise publication, collection, compaction, and growth through the
//! arena and a small store with a caller-supplied growth policy.
//!
//! Small arenas force collect-and-grow retries. Roomier arenas isolate
//! publication and unique-table exhaustion.

use std::cell::Cell;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::mem::DropGuard;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::thread;

use super::{
    CHUNK_MAX, CHUNK_SHIFT, CHUNK_SLOTS, Dag, Exhausted, GROW_EDGES, GROW_FORCE, GROW_NODES, Grow,
    Idx, Map, NeedGrowth, REHASH_CHUNK, Set, Shared, StructuralHasher, chunk_cell, growth_headroom,
    structural_hash, unpack_next,
};

idx! { struct Id; }
idx! { struct Reserved < u32::MAX as usize; }

/// A handwritten index implementation with the same lifecycle as generated IDs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Manual(usize);

impl Idx for Manual {
    fn new(index: usize) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0
    }
}

/// One edge: a child link traced by collection, or an opaque value it skips.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Edge<I> {
    Child(I),
    Opaque(u64),
}

impl<I: Copy> Edge<I> {
    /// The physical child an edge names, as `Dag::maintain` asks for it.
    fn child(&self) -> Option<I> {
        match *self {
            Edge::Child(i) => Some(i),
            Edge::Opaque(_) => None,
        }
    }
}

/// An arena over value words and the test edges.
type Arena<I> = Dag<I, u64, Edge<I>>;

/// The node and edge allocation frontiers.
fn frontiers<I: Idx>(dag: &Arena<I>) -> (usize, usize) {
    unpack_next(dag.next.load(Ordering::Relaxed))
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
            Edge::child,
            |value| reclaimed.push(*value),
            |value, edges| node_hash(*value, edges),
        );
    }
}

/// A node hash using only the low half of its value.
///
/// Varying the high half produces distinct values with the same hash.
fn node_hash<I: Idx>(value: u64, edges: &[Edge<I>]) -> u64 {
    structural_hash((value as u32, edges))
}

/// Interns a value with its edges under the hash the store's own rebuild
/// recomputes.
fn intern<I: Idx>(dag: &Arena<I>, value: u64, edges: &[Edge<I>]) -> Result<I, NeedGrowth> {
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
fn pinned<I: Idx>(shared: &Shared<Store<I>>, value: u64, edges: &[Edge<I>]) -> I {
    shared.retry((), |s| {
        intern(&s.dag, value, edges).inspect(|&id| s.dag.pin(id))
    })
}

/// Runs one `retry` call whose attempt fails with each of `reasons` before
/// succeeding.
///
/// The store's growth policy runs once per failure, and a second failure within
/// the same call forces growth.
fn fail_with<I: Idx>(shared: &Shared<Store<I>>, reasons: &[NeedGrowth]) {
    let mut i = 0;
    shared.retry((), |_| {
        let reason = reasons.get(i).copied();
        i += 1;
        reason.map_or(Ok(()), Err)
    });
}

// --- typed indices -----------------------------------------------------------

#[test]
fn generated_indices_preserve_accepted_values() {
    for index in [0, 1, 255, 65_536, u32::MAX as usize - 1] {
        assert_eq!(Id::new(index).index(), index);
        assert_eq!(Reserved::new(index).index(), index);
    }
    assert_eq!(Id::new(u32::MAX as usize).index(), u32::MAX as usize);
}

#[test]
#[cfg(target_pointer_width = "64")]
#[should_panic(expected = "cannot wrap an index outside the accepted range")]
fn an_id_beyond_u32_is_rejected() {
    Id::new(u32::MAX as usize + 1);
}

#[test]
#[should_panic(expected = "cannot wrap an index outside the accepted range")]
fn an_id_at_a_reserved_sentinel_is_rejected() {
    // `Reserved` reserves its top value as a sentinel, so its accepted range stops
    // one below the range a plain `u32` newtype accepts.
    Reserved::new(u32::MAX as usize);
}

// --- structural hashing ------------------------------------------------------

#[test]
fn structural_hashing_handles_words_tails_and_fresh_state() {
    // Equal independently owned byte keys hash equally. These fixed fixtures also
    // exercise whole words, ragged tails, and zero-padding. Distinct keys may share
    // a hash.
    let whole = vec![1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let copied = whole.clone();
    assert_eq!(structural_hash(&whole), structural_hash(&copied));
    assert_ne!(structural_hash(&whole[..13]), structural_hash(&whole[..12]));
    assert_ne!(structural_hash(&whole[..13]), structural_hash(&whole[..16]));
    assert_ne!(structural_hash("abc"), structural_hash("abcd"));
    assert_ne!(
        structural_hash((1_u32, 2_u64, 3_usize)),
        structural_hash((1_u32, 2_u64, 4_usize)),
    );

    // The map's factory selects the structural algorithm and starts each key with
    // an independent hash state.
    let map = Map::<u64, ()>::default();
    let mut hasher = map.hasher().build_hasher();
    hasher.write_u64(7);
    assert_eq!(hasher.finish(), structural_hash(7_u64));
    assert_eq!(
        map.hasher().build_hasher().finish(),
        StructuralHasher::default().finish(),
    );
    // Leading zero words change the state.
    assert_ne!(structural_hash((0_u64, 5_u64)), structural_hash(5_u64));
}

#[test]
fn structural_hashing_distinguishes_zero_padded_strings() {
    // Each family differs only in zero padding, which the byte-length word
    // separates. The test pins these inputs and claims no general collision
    // freedom.
    for keys in [
        ["a", "a\0", "a\0\0"],
        [
            "a",
            "\0\0\0\0\0\0\0\0a",
            "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0a",
        ],
        ["", "\0", "\0\0\0\0\0\0\0\0"],
    ] {
        let hashes: HashSet<_> = keys.iter().map(structural_hash).collect();
        assert_eq!(hashes.len(), keys.len());
        let mut map = Map::default();
        let mut set = Set::default();
        for (i, key) in keys.into_iter().enumerate() {
            assert_eq!(map.hasher().hash_one(key), structural_hash(key));
            assert_eq!(set.hasher().hash_one(key), structural_hash(key));
            map.insert(key, i);
            assert!(set.insert(key));
        }
        for (i, key) in keys.into_iter().enumerate() {
            assert_eq!(map[key], i);
            assert!(set.contains(key));
        }
    }
}

// --- publication and the reference lifecycle ---------------------------------

/// Publication, reads, and reference counting at one index type.
///
/// Every `Idx` newtype the crate declares is a distinct instantiation of the
/// whole arena, including its ID arithmetic.
fn lifecycle<I: Idx>() {
    let dag: Arena<I> = Dag::with_capacity(4, 4, [10, 20]);
    // Pinned leaves are published in order, so they take the lowest IDs.
    let (a, b) = (I::new(0), I::new(1));
    assert_eq!((*dag.node(a), *dag.node(b)), (10, 20));
    assert!(dag.node(a).edges.is_empty());
    assert!(dag.has_refs(a) && dag.has_refs(b));

    let c = dag
        .publish(|| 30, &[Edge::Child(a), Edge::Opaque(5)])
        .unwrap();
    assert_eq!(c.index(), 2);
    assert_eq!(*dag.node(c), 30);
    assert_eq!(dag.node(c).edges, [Edge::Child(a), Edge::Opaque(5)]);

    // A published node starts unreferenced; counting is external and pinning
    // outranks it.
    assert!(!dag.has_refs(c));
    dag.inc_ref(c);
    dag.inc_ref(c);
    assert!(dag.has_refs(c));
    dag.dec_ref(c);
    assert!(dag.has_refs(c));
    dag.dec_ref(c);
    assert!(!dag.has_refs(c));
    dag.pin(c);
    dag.pin(c);
    dag.inc_ref(c);
    dag.dec_ref(c);
    assert!(dag.has_refs(c));

    // An ID within the arena but never published, and one past its end, are both
    // absent.
    assert!(dag.get(I::new(3)).is_none());
    assert!(dag.get(I::new(4)).is_none());
    assert_eq!(dag.node_capacity(), 4);

    assert_eq!(frontiers(&dag), (3, 2));
    assert_eq!((dag.nodes.len(), dag.edges.len()), (4, 4));
    assert_eq!(dag.buckets(), 8);
}

#[test]
fn a_dag_publishes_pinned_leaves_and_counts_references_at_every_index_type() {
    lifecycle::<Id>();
    lifecycle::<Reserved>();
    lifecycle::<Manual>();
}

#[test]
fn pinned_iterators_preserve_consecutive_ids_despite_cache_eviction() {
    let dag: Arena<Id> = Dag::with_capacity(
        8192,
        0,
        (0..3).inspect(|_| {
            // Model the cache eviction caused by iterator-side arena publications. Clearing
            // all cells makes this independent of concurrent ID allocation.
            super::CHUNKS.with(|chunks| {
                for cell in chunks {
                    cell.set(super::Chunk::EMPTY);
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
fn a_panicking_pinned_iterator_drops_initialized_values_once() {
    let drops = Cell::new(0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        Dag::<Id, _, ()>::with_capacity(
            8,
            0,
            (0..3).map(|i| {
                assert!(i < 2, "iterator interrupted");
                DropGuard::new(&drops, |c| c.update(|n| n + 1))
            }),
        );
    }));
    assert!(panic.is_err());
    assert_eq!(drops.get(), 2);
}

#[test]
fn publication_reports_the_arena_it_exhausted() {
    let dag: Arena<Id> = Dag::with_capacity(2, 2, [0_u64]);
    let builds = Cell::new(0);
    let build = || {
        builds.update(|n| n + 1);
        1
    };
    // A span larger than the remaining edge capacity reports its full length. The
    // node frontier stays at its current position after refusal.
    assert_eq!(
        dag.publish(build, &[Edge::Opaque(0), Edge::Opaque(1), Edge::Opaque(2)]),
        Err(NeedGrowth::Edges(3))
    );
    assert_eq!(frontiers(&dag).0, 1);
    assert_eq!(builds.get(), 0);
    assert!(
        dag.publish(build, &[Edge::Opaque(0), Edge::Opaque(1)])
            .is_ok()
    );
    // Both arenas are now full: the node frontier refuses first.
    assert_eq!(builds.get(), 1);
    assert_eq!(
        dag.publish(build, &[Edge::Opaque(2)]),
        Err(NeedGrowth::Nodes)
    );
    assert_eq!(builds.get(), 1);
}

#[test]
#[should_panic(expected = "pinned leaves must fit the node capacity")]
fn constructing_more_pinned_leaves_than_capacity_is_rejected() {
    // `with_capacity` publishes its leaves eagerly, so too little room is a
    // construction error rather than a growth request the caller could satisfy
    // later.
    Dag::<Id, u64, Edge<Id>>::with_capacity(1, 0, [1_u64, 2]);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "node capacity overflow")]
fn constructing_an_arena_beyond_u32_nodes_is_rejected() {
    // Node IDs pack into the low half of a unique-table entry, so the arena refuses
    // a capacity it could never address.
    Dag::<Id, u64, Edge<Id>>::with_capacity(u32::MAX as usize + 1, 0, []);
}

#[test]
fn interleaved_arena_identities_spread_across_cache_cells() {
    // Each store allocates two identities, so its arena receives every other one.
    // Identity modulo the cell count would reach only half of the cells.
    let cells: HashSet<_> = (0..CHUNK_SLOTS as u64)
        .map(|i| chunk_cell(2 * i + 1))
        .collect();
    assert!(cells.len() > CHUNK_SLOTS / 2);
}

#[test]
fn cached_batches_remain_owned_when_arena_cache_cells_collide() {
    // One more arena than cache cells guarantees a collision even when other tests
    // allocate identities concurrently. Only the input selection uses the cache
    // index; assertions check each arena's independent nodes and edges.
    let arenas: Vec<Arena<Id>> =
        Vec::from_fn(CHUNK_SLOTS + 1, |_| Dag::with_capacity(4096, 4096, []));
    let (a, b) = arenas
        .iter()
        .enumerate()
        .find_map(|(i, a)| {
            arenas[i + 1..]
                .iter()
                .find(|b| chunk_cell(a.id) == chunk_cell(b.id))
                .map(|b| (a, b))
        })
        .unwrap();
    let mut ids = Vec::new();
    for value in 0..4 {
        for dag in [a, b] {
            ids.push((
                dag,
                dag.publish(|| value, &[Edge::Opaque(value)]).unwrap(),
                value,
            ));
        }
    }
    for dag in [a, b] {
        assert_eq!(published(dag), 4);
        // Each switch evicted a partly unused two-slot reservation. This checks that
        // the selected arenas really collide, beyond their ID formula.
        assert_eq!(frontiers(dag), (8, 8));
    }
    for (dag, id, value) in ids {
        assert_eq!(*dag.node(id), value);
        assert_eq!(dag.node(id).edges, [Edge::Opaque(value)]);
    }
}

#[test]
fn colliding_arenas_finish_more_than_2048_alternating_publications() {
    const EACH: usize = 2049;
    let capacity = (EACH * CHUNK_MAX).next_power_of_two();
    for edge_len in [0, 1] {
        // Select real colliding identities cheaply, then enlarge just those two arenas.
        // Publication does not use their small unique tables. Unit edges keep the large
        // edge capacity free of backing allocations.
        let mut arenas: Vec<Dag<Id, u64, ()>> =
            Vec::from_fn(CHUNK_SLOTS + 1, |_| Dag::with_capacity(1, capacity, []));
        let (i, j) = arenas
            .iter()
            .enumerate()
            .find_map(|(i, a)| {
                arenas
                    .iter()
                    .enumerate()
                    .skip(i + 1)
                    .find(|(_, b)| chunk_cell(a.id) == chunk_cell(b.id))
                    .map(|(j, _)| (i, j))
            })
            .unwrap();
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
            let (node, edge) = unpack_next(dag.next.load(Ordering::Relaxed));
            assert_eq!(node, EACH * CHUNK_MAX);
            assert_eq!(edge, EACH * CHUNK_MAX * edge_len);
        }
    }
}

#[test]
fn edge_spans_larger_than_the_speculative_cap_remain_contiguous() {
    let dag: Dag<Id, u64, ()> = Dag::with_capacity(2, 1 << 22, []);
    let edges = vec![(); CHUNK_MAX + 1];
    let id = dag.publish(|| 7, &edges).unwrap();
    assert_eq!(*dag.node(id), 7);
    assert_eq!(dag.node(id).edges.len(), edges.len());
}

#[test]
fn independent_node_and_edge_batches_preserve_published_contents() {
    // Reverse the relative batch sizes, and mix empty, short, and oversized spans
    // so node and edge reservations must refill independently.
    for (nodes, edges) in [(8192, 4096), (4096, 8192)] {
        let dag: Arena<Id> = Dag::with_capacity(nodes, edges, []);
        let shapes: [&[Edge<Id>]; 6] = [
            &[],
            &[Edge::Opaque(11)],
            &[Edge::Opaque(21), Edge::Opaque(22), Edge::Opaque(23)],
            &[],
            &[Edge::Opaque(41), Edge::Opaque(42)],
            &[Edge::Opaque(51)],
        ];
        let ids: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(i, edges)| dag.publish(|| i as u64, edges).unwrap())
            .collect();
        assert_eq!(
            ids.iter().copied().collect::<HashSet<_>>().len(),
            shapes.len()
        );
        for (i, (&id, &edges)) in ids.iter().zip(&shapes).enumerate() {
            assert_eq!(*dag.node(id), i as u64);
            assert_eq!(dag.node(id).edges, edges);
        }
    }
}

/// A store whose one published node has been reclaimed by a collection.
fn with_reclaimed(body: impl FnOnce(&Shared<Store<Id>>, Id)) {
    let shared = Store::<Id>::shared(4, 0);
    let id = shared.read(()).dag.publish(|| 1, &[]).unwrap();
    fail_with(&shared, &[NeedGrowth::Nodes]);
    assert_eq!(shared.read(()).reclaimed, [1]);
    body(&shared, id);
}

#[test]
#[should_panic(expected = "cannot use a reclaimed DAG node")]
fn reading_a_reclaimed_node_is_rejected() {
    with_reclaimed(|shared, id| {
        shared.read(()).dag.node(id);
    })
}

#[test]
#[should_panic(expected = "cannot reference a reclaimed DAG node")]
fn referencing_a_reclaimed_node_is_rejected() {
    with_reclaimed(|shared, id| shared.read(()).dag.inc_ref(id))
}

#[test]
#[should_panic(expected = "cannot inspect a reclaimed DAG node")]
fn inspecting_a_reclaimed_node_is_rejected() {
    with_reclaimed(|shared, id| {
        shared.read(()).dag.has_refs(id);
    })
}

#[test]
#[should_panic(expected = "cannot decrement an unreferenced or reclaimed DAG node")]
fn decrementing_an_unreferenced_node_is_rejected() {
    let dag: Arena<Id> = Dag::with_capacity(2, 0, []);
    let id = dag.publish(|| 1, &[]).unwrap();
    dag.dec_ref(id);
}

#[test]
#[should_panic(expected = "cannot pin a reclaimed or externally referenced DAG node")]
fn pinning_a_referenced_node_is_rejected() {
    // Pinning requires the caller to release every external reference first,
    // preserving the obligation to decrement each one.
    let dag: Arena<Id> = Dag::with_capacity(2, 0, []);
    let id = dag.publish(|| 1, &[]).unwrap();
    dag.inc_ref(id);
    dag.pin(id);
}

#[test]
#[should_panic(expected = "cannot unpublish a reclaimed or referenced DAG node")]
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

#[test]
fn interning_returns_one_canonical_node_per_key_and_reports_a_full_table() {
    // A table sized for the arena always has room for every node the arena can
    // hold, so the only way to fill one is to grow the arena alone: growth expands
    // just the resources the failed attempts exhausted. These attempts request node
    // and edge growth while retaining the table capacity.
    let shared = Store::<Id>::shared(2, 2);
    for _ in 0..2 {
        fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    }
    fail_with(&shared, &[NeedGrowth::Edges(8), NeedGrowth::Edges(8)]);
    let s = shared.read(());
    let dag = &s.dag;
    // Eight buckets, so every key probes every bucket of the one group and
    // collisions are certain. Equality distinguishes the colliding entries.
    assert_eq!(dag.buckets(), 8);
    assert!(
        dag.node_capacity() > 6,
        "the arena must outgrow the table's capacity"
    );
    let a = intern(dag, 1, &[Edge::Opaque(1)]).unwrap();
    assert_eq!(intern(dag, 1, &[Edge::Opaque(1)]), Ok(a));
    // Equal values with different edges are different nodes.
    let b = intern(dag, 1, &[Edge::Opaque(2)]).unwrap();
    assert_ne!(b, a);

    // A distinct value that hashes the same shares the probe and the fingerprint;
    // only `eq` rejects it, and it is then interned into a different bucket of the
    // same sequence.
    let c = intern(dag, 1 << 32 | 1, &[Edge::Opaque(1)]).unwrap();
    assert_ne!(c, a);
    assert_eq!(
        node_hash(1 << 32 | 1, &[Edge::<Id>::Opaque(1)]),
        node_hash(1, &[Edge::<Id>::Opaque(1)])
    );
    assert_eq!(intern(dag, 1 << 32 | 1, &[Edge::Opaque(1)]), Ok(c));
    assert_eq!(*dag.node(c), 1 << 32 | 1);

    // The table refuses a new entry at its load-factor capacity rather than probing
    // on, and says so before any candidate is published: the node frontier is
    // unchanged by the refusal, even though the arena still has room.
    let mut value = 100;
    let refused = loop {
        value += 1;
        match intern(dag, value, &[]) {
            Ok(_) => {}
            Err(reason) => break reason,
        }
    };
    assert_eq!(refused, NeedGrowth::Unique);
    let nodes = frontiers(dag).0;
    assert_eq!(intern(dag, value, &[]), Err(NeedGrowth::Unique));
    assert_eq!(frontiers(dag).0, nodes);
    assert!(
        nodes < dag.node_capacity(),
        "the table filled before the arena did"
    );
}

// --- collection and maintenance ----------------------------------------------

#[test]
fn collection_reclaims_unreachable_nodes_compacts_edges_and_reuses_ids() {
    let shared = Store::<Id>::shared(8, 16);
    // One interned and one plainly published garbage node ahead of the live nodes,
    // so the live edge spans have a hole below them to be packed into, and the
    // table itself holds an entry whose node the collection reclaims.
    let (garbage, live) = {
        let s = shared.read(());
        let dag = &s.dag;
        let interned = intern(dag, 98, &[Edge::Opaque(1)]).unwrap();
        let published = dag
            .publish(|| 99, &[Edge::Opaque(2), Edge::Opaque(3)])
            .unwrap();
        // A diamond: the leaf is reached twice, so marking must recognize it the second
        // time. The collector skips the opaque edge while tracing links.
        let leaf = intern(dag, 1, &[]).unwrap();
        let left = intern(dag, 2, &[Edge::Child(leaf), Edge::Opaque(7)]).unwrap();
        let right = intern(dag, 3, &[Edge::Child(leaf)]).unwrap();
        let root = intern(dag, 4, &[Edge::Child(left), Edge::Child(right)]).unwrap();
        dag.pin(root);
        ((interned, published), (leaf, left, right, root))
    };
    let (leaf, left, right, root) = live;

    fail_with(&shared, &[NeedGrowth::Nodes]);

    let s = shared.read(());
    let dag = &s.dag;
    // Only the two unreachable nodes were reclaimed, and the reclaim hook saw their
    // values.
    assert_eq!(s.reclaimed, [98, 99]);
    assert!(dag.get(garbage.0).is_none() && dag.get(garbage.1).is_none());
    // Live nodes keep their IDs and their edges, though the edges moved: compaction
    // packs the surviving spans leftward over the hole the garbage left.
    assert_eq!(
        (
            *dag.node(leaf),
            *dag.node(left),
            *dag.node(right),
            *dag.node(root)
        ),
        (1, 2, 3, 4)
    );
    assert_eq!(dag.node(left).edges, [Edge::Child(leaf), Edge::Opaque(7)]);
    assert_eq!(dag.node(right).edges, [Edge::Child(leaf)]);
    assert_eq!(
        dag.node(root).edges,
        [Edge::Child(left), Edge::Child(right)]
    );
    assert!(dag.node(leaf).edges.is_empty());
    assert_eq!(frontiers(dag).1, 5);
    // The reclaimed entry left the table; the surviving ones were reinserted, so
    // interning a live key still answers with its canonical node rather than
    // publishing a second one.
    assert_eq!(
        intern(dag, 2, &[Edge::Child(leaf), Edge::Opaque(7)]),
        Ok(left)
    );
    assert_eq!(
        intern(dag, 4, &[Edge::Child(left), Edge::Child(right)]),
        Ok(root)
    );

    // Publication reuses reclaimed IDs, highest index first, while the node
    // frontier stays at its current position.
    let nodes = frontiers(dag).0;
    let reused = dag.publish(|| 55, &[]).unwrap();
    assert_eq!(reused, garbage.1);
    assert_eq!(frontiers(dag).0, nodes);
    drop(s);

    // A second collection: the one ID left free stays unpublished and is offered
    // again, the reused slot is reclaimed a second time, and nothing needs to move
    // because the live spans are already packed.
    fail_with(&shared, &[NeedGrowth::Nodes]);
    let s = shared.read(());
    assert_eq!(s.reclaimed, [98, 99, 55]);
    assert_eq!(
        s.dag.node(root).edges,
        [Edge::Child(left), Edge::Child(right)]
    );
    assert_eq!(frontiers(&s.dag).1, 5);
    assert_eq!(s.dag.publish(|| 56, &[]).unwrap(), garbage.1);
}

#[test]
fn publication_after_a_reclaim_panic_discards_reserved_free_list_entries() {
    // Two-slot batches leave an unused frontier slot, then an unused free-list
    // entry. The second collection clears that list before its callback panics.
    let mut dag: Arena<Id> = Dag::with_capacity(4096, 4096, []);
    let first = dag.publish(|| 0, &[Edge::Opaque(0)]).unwrap();
    dag.inc_ref(first);
    dag.publish(|| 1, &[Edge::Opaque(1)]).unwrap();
    dag.publish(|| 2, &[Edge::Opaque(2)]).unwrap();
    assert_eq!(dag.compact(Edge::child, |_| {}), 2);
    let held = dag.publish(|| 3, &[Edge::Opaque(3)]).unwrap();
    dag.inc_ref(held);
    dag.dec_ref(first);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        dag.compact(Edge::child, |_| panic!("reclaim interrupted"));
    }))
    .unwrap_err();
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"reclaim interrupted"));

    let a = dag.publish(|| 4, &[Edge::Opaque(4)]).unwrap();
    let b = dag.publish(|| 5, &[Edge::Opaque(5)]).unwrap();
    assert_eq!(HashSet::from([first, held, a, b]).len(), 4);
    for (id, value) in [(first, 0), (held, 3), (a, 4), (b, 5)] {
        assert_eq!(*dag.node(id), value);
        assert_eq!(dag.node(id).edges, [Edge::Opaque(value)]);
    }
    dag.dec_ref(held);
    assert_eq!(dag.compact(Edge::child, |_| {}), 4);
    assert_eq!(published(&dag), 0);
}

#[test]
fn reclaimed_batches_are_consumed_before_frontier_allocation() {
    // Several reclaimed nodes require both a complete free-list batch and a final
    // partial batch. A retained node keeps its identity throughout reuse.
    let mut dag: Arena<Id> = Dag::with_capacity(4096, 4096, []);
    let ids: Vec<_> = (0..6)
        .map(|value| dag.publish(|| value, &[Edge::Opaque(value)]).unwrap())
        .collect();
    dag.inc_ref(ids[0]);
    assert_eq!(dag.compact(Edge::child, |_| {}), 5);
    let frontier = frontiers(&dag).0;
    let mut reused = HashSet::new();
    for value in 10..15 {
        let id = dag.publish(|| value, &[Edge::Opaque(value)]).unwrap();
        assert!(ids[1..].contains(&id));
        assert!(reused.insert(id));
        assert_eq!(frontiers(&dag).0, frontier);
        assert_eq!(*dag.node(id), value);
        assert_eq!(dag.node(id).edges, [Edge::Opaque(value)]);
    }
    assert_eq!(reused, ids[1..].iter().copied().collect());
    let fresh = dag.publish(|| 20, &[Edge::Opaque(20)]).unwrap();
    assert!(!ids.contains(&fresh));
    assert_eq!(*dag.node(ids[0]), 0);
    assert_eq!(dag.node(ids[0]).edges, [Edge::Opaque(0)]);
    assert_eq!(*dag.node(fresh), 20);
    assert_eq!(dag.node(fresh).edges, [Edge::Opaque(20)]);
}

#[test]
fn rebuild_across_threads_keeps_every_retained_entry_findable() {
    // Two chunks of retained entries put helpers to work. Check ordinary hashes and
    // deliberate collisions, preserving canonical IDs and the entry count after
    // collection.
    const EACH: u64 = 3 * REHASH_CHUNK as u64;
    type Hash = fn(u64, &[Edge<Id>]) -> u64;
    let hashes: [Hash; 2] = [node_hash, |value, _| {
        structural_hash(value % REHASH_CHUNK as u64)
    }];
    for hash in hashes {
        let mut dag: Arena<Id> = Dag::with_capacity(1 << 16, 1 << 17, []);
        let ids: Vec<Id> = (0..EACH)
            .map(|i| {
                let edges = [Edge::Opaque(i)];
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
            Exhausted {
                flags: GROW_NODES,
                edge_span: 0,
            },
            4,
            Edge::child,
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
                let edges = [Edge::Opaque(value)];
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
fn collection_preserves_empty_roots_after_discarding_earlier_edges() {
    let mut dag: Arena<Id> = Dag::with_capacity(8, 8, []);
    dag.publish(|| 1, &[Edge::Opaque(1), Edge::Opaque(2)])
        .unwrap();
    let leaf = dag.publish(|| 7, &[]).unwrap();
    dag.pin(leaf);
    dag.maintain(
        Exhausted {
            flags: GROW_NODES,
            edge_span: 0,
        },
        1,
        Edge::child,
        |_| {},
        |value, edges| node_hash(*value, edges),
    );
    assert_eq!(frontiers(&dag).1, 0);
    assert_eq!(*dag.node(leaf), 7);
    assert!(dag.node(leaf).edges.is_empty());
    let next = dag.publish(|| 9, &[Edge::Opaque(9)]).unwrap();
    assert_eq!(dag.node(next).edges, [Edge::Opaque(9)]);
    assert!(dag.node(leaf).edges.is_empty());
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
        Exhausted {
            flags: GROW_NODES,
            edge_span: 0,
        },
        4,
        Edge::child,
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
    let mut dag: Arena<Id> = Dag::with_capacity(32768, 0, []);
    let groups = dag.groups.as_ptr();
    let buckets = dag.buckets();
    let id = dag.publish(|| 7, &[]).unwrap();
    let mut reclaimed = Vec::new();
    dag.maintain(
        Exhausted {
            flags: GROW_NODES,
            edge_span: 0,
        },
        1,
        Edge::child,
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
#[should_panic(expected = "maintenance thread count must be nonzero")]
fn maintenance_without_a_caller_thread_is_rejected() {
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    dag.maintain(
        Exhausted {
            flags: GROW_NODES,
            edge_span: 0,
        },
        0,
        Edge::child,
        |_| {},
        |value, edges| node_hash(*value, edges),
    );
}

#[test]
#[should_panic(expected = "arena capacity overflow")]
fn growing_an_arena_beyond_u32_slots_is_rejected() {
    // One pinned edge ahead of a forced span of `u32::MAX` edges asks for one more
    // slot than the edge frontier can address.
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    let leaf = dag.publish(|| 1, &[Edge::Opaque(1)]).unwrap();
    dag.pin(leaf);
    dag.maintain(
        Exhausted {
            flags: GROW_EDGES | GROW_FORCE,
            edge_span: u32::MAX as usize,
        },
        1,
        Edge::child,
        |_| {},
        |value, edges| node_hash(*value, edges),
    );
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
            edge_span: 1,
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

#[test]
fn exclusive_replacement_invalidates_cached_node_meanings() {
    let mut shared = Store::<Id>::shared(8, 0);
    let id = pinned(&shared, 7, &[]);
    let stamp = (shared.id, shared.growths());
    let mut cache = Map::default();
    cache.insert(id, 7);
    *shared.get_mut() = Store {
        dag: Dag::with_capacity(8, 0, [11]),
        reclaimed: Vec::new(),
    };
    let s = shared.read(());
    let current = (shared.id, shared.growths());
    assert_ne!(current.0, stamp.0);
    assert_eq!(current.1, stamp.1);
    if current != stamp {
        cache.clear();
    }
    assert_eq!(*cache.entry(id).or_insert_with(|| *s.dag.node(id)), 11);
}

#[test]
fn growth_expands_only_an_exhausted_resource_that_is_actually_short() {
    // An arena with plenty of free space: a single failure collects but must not
    // grow, since the free share is above the threshold that makes collection worth
    // its fixed cost.
    let shared = Store::<Id>::shared(64, 64);
    let capacity = (
        shared.read(()).dag.node_capacity(),
        shared.read(()).dag.edges.len(),
    );
    fail_with(&shared, &[NeedGrowth::Nodes]);
    fail_with(&shared, &[NeedGrowth::Edges(4)]);
    assert_eq!(shared.growths(), 2);
    let s = shared.read(());
    assert_eq!((s.dag.node_capacity(), s.dag.edges.len()), capacity);
    drop(s);

    // The same call failing twice forces growth regardless: collection alone would
    // otherwise reproduce the same failure forever.
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    fail_with(&shared, &[NeedGrowth::Edges(4), NeedGrowth::Edges(4)]);
    let s = shared.read(());
    assert!(s.dag.node_capacity() > capacity.0);
    assert!(s.dag.edges.len() > capacity.1);
    drop(s);

    // A span larger than the free edge capacity grows even unforced, and the growth
    // admits the whole span rather than merely closing the shortfall.
    let free = {
        let s = shared.read(());
        s.dag.edges.len() - frontiers(&s.dag).1
    };
    fail_with(&shared, &[NeedGrowth::Edges(free + 9)]);
    let s = shared.read(());
    assert!(s.dag.edges.len() - frontiers(&s.dag).1 >= free + 9);
    drop(s);

    // A forced unique-table growth exceeds the old capacity even though collection
    // retained nothing to reinsert, so the retry cannot fail the same way again.
    let buckets = shared.read(()).dag.buckets();
    fail_with(&shared, &[NeedGrowth::Unique, NeedGrowth::Unique]);
    assert!(shared.read(()).dag.buckets() > buckets);

    // A single unforced failure does grow once the free share is low, even when the
    // span it asked for would have fit: collection is proportional to the whole
    // arena, so a nearly full one must not pay a collection per allocation.
    let tight = Store::<Id>::shared(8, 8);
    for k in 0..3 {
        pinned(&tight, k, &[Edge::Opaque(k), Edge::Opaque(k + 1)]);
    }
    assert_eq!(frontiers(&tight.read(()).dag).1, 6);
    fail_with(&tight, &[NeedGrowth::Edges(1)]);
    let s = tight.read(());
    assert!(s.dag.edges.len() > 8);
    // Growth alone: the pinned nodes and their edges are untouched by the
    // collection.
    assert!(s.reclaimed.is_empty());
    assert_eq!(frontiers(&s.dag).1, 6);
}

#[test]
fn a_tiny_store_grows_through_retry_until_every_key_is_interned() {
    const KEYS: u64 = 40;
    // Room for one node and no edges: every arena and the table itself are
    // exhausted repeatedly, so this walks the whole failure-collect-grow-retry
    // cycle many times.
    let shared = Store::<Reserved>::shared(1, 0);
    let ids: Vec<Reserved> = (0..KEYS)
        .map(|k| pinned(&shared, k, &[Edge::Opaque(k)]))
        .collect();

    let s = shared.read(());
    for (k, &id) in ids.iter().enumerate() {
        assert_eq!(*s.dag.node(id), k as u64);
        assert_eq!(s.dag.node(id).edges, [Edge::Opaque(k as u64)]);
        // Growth rebuilt the table over the retained entries, so every key is still
        // canonical.
        assert_eq!(intern(&s.dag, k as u64, &[Edge::Opaque(k as u64)]), Ok(id));
    }
    assert_eq!(ids.iter().collect::<HashSet<_>>().len(), KEYS as usize);
    // Pinned nodes remain live, so growth alone absorbed every failure.
    assert!(s.reclaimed.is_empty());
    drop(s);

    // Exclusive access reaches the same state without admission.
    let mut shared = shared;
    assert_eq!(*shared.get_mut().dag.node(ids[0]), 0);
}

#[cfg(target_pointer_width = "64")]
#[test]
#[should_panic(expected = "edge span overflow")]
fn requesting_an_edge_span_beyond_u32_is_rejected() {
    // The aggregated request stores the span in 32 bits, so a failed attempt asking
    // for more is refused before maintenance runs.
    let shared = Store::<Id>::shared(4, 4);
    shared.retry((), |_| -> Result<(), NeedGrowth> {
        Err(NeedGrowth::Edges(u32::MAX as usize + 1))
    });
}

// --- concurrent admission ----------------------------------------------------

#[test]
fn admission_at_the_counter_limit_is_released_on_drop() {
    let shared = Store::<Id>::shared(2, 2);
    let counter = &shared.readers[super::counter()].0;
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
    let counter = &shared.readers[super::counter()].0;
    // Simulate forgotten guards while retaining one real borrow of the store.
    counter.store(u32::MAX, Ordering::Relaxed);
    let panic = catch_unwind(AssertUnwindSafe(|| shared.read(())))
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
    let counter = &shared.readers[super::counter()].0;
    counter.store(u32::MAX, Ordering::Relaxed);
    let called = Cell::new(false);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        shared.retry((), |_| {
            called.set(true);
            Ok(())
        });
    }));
    assert!(panic.is_err());
    assert!(!called.get());
    assert_eq!(counter.load(Ordering::Relaxed), u32::MAX);
    assert_eq!(shared.requested.load(Ordering::Relaxed), 0);
    assert_eq!(shared.growths(), 0);
    counter.store(0, Ordering::Relaxed);
}

#[test]
fn concurrent_interning_agrees_on_one_node_per_key_across_growth() {
    // Eight workers, a twelve-key table with eight buckets, and an arena with room
    // for almost nothing: admission, publication, bucket installation and growth
    // all collide. Each worker starts at a different key so no interleaving is
    // assumed. The test checks that every attempt eventually succeeds and all
    // workers agree on each key's canonical node. Repetitions sample schedules
    // without requiring a particular race in any round.
    const WORKERS: usize = 8;
    const KEYS: u64 = 12;
    for _round in 0..100 {
        let shared = Store::<Id>::shared(2, 2);
        let results: Vec<Vec<Id>> = thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS)
                .map(|w| {
                    let shared = &shared;
                    // Each worker borrows the same store directly, without a runtime context.
                    scope.spawn(move || {
                        let mut ids = vec![Id::new(0); KEYS as usize];
                        for step in 0..KEYS {
                            let k = (step + w as u64) % KEYS;
                            ids[k as usize] = pinned(shared, k, &[Edge::Opaque(k)]);
                        }
                        ids
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });

        let ids = &results[0];
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), KEYS as usize);
        for worker in &results {
            assert_eq!(worker, ids, "workers disagreed on the canonical nodes");
        }
        let s = shared.read(());
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(*s.dag.node(id), k as u64);
            assert_eq!(s.dag.node(id).edges, [Edge::Opaque(k as u64)]);
        }
        // Losing candidates stay published and unreferenced until collection reclaims
        // them. Callers receive the canonical nodes, so the arena may hold more nodes
        // than there are keys.
        assert!(frontiers(&s.dag).0 >= KEYS as usize);
    }
}

#[test]
fn a_candidate_that_loses_every_remaining_bucket_reports_a_full_table() {
    const WORKERS: usize = 8;
    // Eight buckets with five entries leave three buckets for eight candidates.
    // Grow just the node arena so every candidate has room for publication.
    let shared = Store::<Id>::shared(2, 2);
    while shared.read(()).dag.node_capacity() < 64 {
        fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    }
    assert_eq!(shared.read(()).dag.buckets(), 8);
    let filled: Vec<Id> = (0..5).map(|i| pinned(&shared, i << 32 | 5, &[])).collect();

    let growths = shared.growths();
    let barrier = Barrier::new(WORKERS);
    let attempts: Vec<_> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS as u64)
            .map(|w| {
                let (shared, barrier) = (&shared, &barrier);
                scope.spawn(move || {
                    let value = (w + 10) << 32 | 5;
                    let s = shared.read(());
                    // Construction follows admission but precedes installation. All candidates pass
                    // the load check while the count is still five.
                    let build = || {
                        barrier.wait();
                        value
                    };
                    s.dag
                        .intern(node_hash::<Id>(value, &[]), |v, _| *v == value, build, &[])
                        .inspect(|&id| s.dag.pin(id))
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    assert_eq!(attempts.iter().filter(|result| result.is_ok()).count(), 3);
    assert_eq!(
        attempts
            .iter()
            .filter(|&&result| result == Err(NeedGrowth::Unique))
            .count(),
        WORKERS - 3,
    );
    assert_eq!(shared.growths(), growths);

    // Retry the refused candidates. Growth preserves winners and the old entries.
    let ids: Vec<_> = (0..WORKERS as u64)
        .map(|w| pinned(&shared, (w + 10) << 32 | 5, &[]))
        .collect();
    let s = shared.read(());
    for (w, &id) in ids.iter().enumerate() {
        assert_eq!(*s.dag.node(id), (w as u64 + 10) << 32 | 5);
        if let Ok(winner) = attempts[w] {
            assert_eq!(id, winner);
        }
    }
    for (i, &id) in filled.iter().enumerate() {
        assert_eq!(*s.dag.node(id), (i as u64) << 32 | 5);
        assert_eq!(intern(&s.dag, (i as u64) << 32 | 5, &[]), Ok(id));
    }
    let all: HashSet<Id> = ids.iter().chain(&filled).copied().collect();
    assert_eq!(all.len(), WORKERS + filled.len());
}

#[test]
fn concurrent_publication_returns_each_workers_value() {
    const WORKERS: usize = 8;
    const EACH: u64 = 256;
    // Publication into a roomy arena: nothing here can fail, so every worker
    // contends for the shared frontier alone. The outcome is that the frontier
    // hands out each slot once.
    let dag: Arena<Id> = Dag::with_capacity(WORKERS * EACH as usize, 0, []);
    let published: Vec<(Id, u64)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS as u64)
            .map(|w| {
                let dag = &dag;
                scope.spawn(move || {
                    (0..EACH)
                        .map(|i| {
                            let value = w * EACH + i;
                            (dag.publish(|| value, &[]).unwrap(), value)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    assert_eq!(
        published
            .iter()
            .map(|&(id, _)| id)
            .collect::<HashSet<_>>()
            .len(),
        published.len()
    );
    for (id, value) in published {
        assert_eq!(*dag.node(id), value);
    }
}

#[test]
fn concurrent_colliding_interning_agrees_across_growth() {
    const WORKERS: usize = 8;
    const KEYS: u64 = 10;

    // Interning values that all hash alike, so every worker walks the same probe
    // sequence in the same order against a table of eight buckets: candidates lose
    // their bucket to a different value and have to be reoffered further along the
    // sequence, and a probe can run out of buckets while the table's own count
    // still reports room. Again only outcomes are asserted, over many rounds so no
    // round has to produce any particular interleaving.
    for _round in 0..100 {
        let shared = Store::<Id>::shared(2, 2);
        let results: Vec<Vec<Id>> = thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS)
                .map(|w| {
                    let shared = &shared;
                    scope.spawn(move || {
                        let mut ids = vec![Id::new(0); KEYS as usize];
                        for step in 0..KEYS {
                            let k = (step + w as u64) % KEYS;
                            // One hash family, distinct values.
                            ids[k as usize] = pinned(shared, k << 32 | 3, &[]);
                        }
                        ids
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });

        let ids = &results[0];
        assert_eq!(ids.iter().collect::<HashSet<_>>().len(), KEYS as usize);
        for worker in &results {
            assert_eq!(worker, ids, "workers disagreed on the canonical nodes");
        }
        let s = shared.read(());
        for (k, &id) in ids.iter().enumerate() {
            assert_eq!(*s.dag.node(id), (k as u64) << 32 | 3);
        }
    }
}

#[test]
fn workers_interning_shared_chains_with_reference_churn_stay_consistent() {
    // Thirty-two workers intern overlapping chains of child-linked nodes into an
    // arena that starts with room for almost nothing, so growth, collection, failed
    // attempts, and reference churn all interleave. Each worker keeps a window of
    // referenced roots and releases a randomly chosen one as it goes. Only outcomes
    // are checked: all retained roots agree with a fresh interning of their chain,
    // a sweep leaves exactly the nodes reachable from retained roots and no
    // duplicate keys, and the reference counts drain to zero.
    const WORKERS: u64 = 32;
    const KEYS: u64 = 24;
    const DEPTH: u64 = 5;
    const ROUNDS: usize = 3000;
    const WINDOW: usize = 8;
    type Roots = Vec<(u64, Id)>;

    let shared = Store::<Id>::shared(2, 2);
    let chain = |s: &Store<Id>, k: u64| -> Result<Id, NeedGrowth> {
        // Leaves are shared by every fourth key, so chains overlap across workers and
        // keys.
        let mut id = intern(&s.dag, k % 4, &[Edge::Opaque(k % 4)])?;
        for level in 1..=DEPTH {
            id = intern(&s.dag, level << 32 | k, &[Edge::Child(id)])?;
        }
        Ok(id)
    };
    let results: Vec<Roots> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS)
            .map(|w| {
                let (shared, chain) = (&shared, &chain);
                scope.spawn(move || {
                    let mut seed = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1;
                    let mut next = move || {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        seed >> 33
                    };
                    let mut roots: Roots = Vec::new();
                    for _ in 0..ROUNDS {
                        let k = next() % KEYS;
                        // Reference the root in the attempt that interned it, so no collection can
                        // reclaim it in between.
                        let id = shared.retry((), |s| chain(s, k).inspect(|&id| s.dag.inc_ref(id)));
                        roots.push((k, id));
                        if roots.len() > WINDOW {
                            let (_, old) = roots.swap_remove(next() as usize % roots.len());
                            shared.read(()).dag.dec_ref(old);
                        }
                    }
                    roots
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    // Every retained root is the canonical node for its chain.
    let held: Roots = results.into_iter().flatten().collect();
    for &(k, id) in &held {
        assert_eq!(shared.retry((), |s| chain(s, k)), id);
    }

    // After a forced sweep, exactly the nodes reachable from retained roots remain,
    // each key once.
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    {
        let s = shared.read(());
        let mut reachable = HashSet::new();
        let mut keys = HashSet::new();
        let mut stack: Vec<Id> = held.iter().map(|&(_, id)| id).collect();
        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            let node = s.dag.node(id);
            assert!(
                keys.insert((*node, node.edges.to_vec())),
                "duplicate key survived a sweep"
            );
            stack.extend(node.edges.iter().filter_map(Edge::child));
        }
        assert_eq!(published(&s.dag), reachable.len());
        for &(_, id) in &held {
            assert!(s.dag.has_refs(id));
        }
    }

    // Releasing every held reference drains each root's count to exactly zero, and
    // the next sweep reclaims everything.
    let mut counts: Map<Id, usize> = Map::default();
    for &(_, id) in &held {
        *counts.entry(id).or_default() += 1;
    }
    {
        let s = shared.read(());
        for (&id, &n) in &counts {
            for _ in 1..n {
                s.dag.dec_ref(id);
                assert!(s.dag.has_refs(id));
            }
            s.dag.dec_ref(id);
            assert!(!s.dag.has_refs(id));
        }
    }
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}

#[test]
fn concurrent_batched_publication_preserves_contents_and_reclaims_nodes() {
    // A large arena gives each thread batched reservations. Published IDs are
    // distinct, their contents are preserved, and collection reclaims every node
    // after its external references are released.
    const WORKERS: u64 = 4;
    const EACH: u64 = 10_000;
    let shared = Store::<Id>::shared(1 << 16, 1 << 17);
    let ids: Vec<Vec<Id>> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS)
            .map(|w| {
                let shared = &shared;
                scope.spawn(move || {
                    (0..EACH)
                        .map(|i| {
                            shared.retry((), |s| {
                                s.dag
                                    .publish(|| w << 32 | i, &[Edge::Opaque(i)])
                                    .inspect(|&id| s.dag.inc_ref(id))
                            })
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });

    let all: HashSet<Id> = ids.iter().flatten().copied().collect();
    assert_eq!(all.len(), (WORKERS * EACH) as usize);
    {
        let s = shared.read(());
        for (w, ids) in ids.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                assert_eq!(*s.dag.node(id), (w as u64) << 32 | i as u64);
                assert_eq!(s.dag.node(id).edges, [Edge::Opaque(i as u64)]);
            }
        }
        // Each thread's final batch is partly unused, so the frontier runs past the
        // published count by at most one batch per worker.
        let (node, _) = frontiers(&s.dag);
        let batch = (s.dag.node_capacity() >> CHUNK_SHIFT).clamp(1, CHUNK_MAX);
        assert!(node - all.len() < WORKERS as usize * batch);
        for &id in &all {
            s.dag.dec_ref(id);
        }
    }
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}

#[test]
fn workers_resume_reserved_batches_after_compaction() {
    // Workers remain alive across two collections. Each phase leaves a partial
    // two-slot batch, first in the frontier and then in the rebuilt free list.
    // Barriers keep every admission guard outside the maintenance phases.
    let shared = Store::<Id>::shared(4096, 4096);
    let barrier = Barrier::new(3);
    let ids = thread::scope(|scope| {
        let workers: Vec<_> = (0..2_u64)
            .map(|w| {
                let (shared, barrier) = (&shared, &barrier);
                scope.spawn(move || {
                    let mut held = Vec::new();
                    for phase in 0..3_u64 {
                        let value = w * 3 + phase;
                        let id = shared.retry((), |s| {
                            s.dag
                                .publish(|| value, &[Edge::Opaque(value)])
                                .inspect(|&id| s.dag.inc_ref(id))
                        });
                        held.push((id, value));
                        barrier.wait();
                        barrier.wait();
                        let s = shared.read(());
                        for &(id, value) in &held {
                            assert_eq!(*s.dag.node(id), value);
                            assert_eq!(s.dag.node(id).edges, [Edge::Opaque(value)]);
                        }
                    }
                    held
                })
            })
            .collect();
        for phase in 0..3 {
            barrier.wait();
            if phase < 2 {
                fail_with(&shared, &[NeedGrowth::Nodes]);
            }
            barrier.wait();
        }
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_eq!(
        ids.iter().map(|&(id, _)| id).collect::<HashSet<_>>().len(),
        6
    );
    {
        let s = shared.read(());
        for &(id, value) in &ids {
            assert_eq!(*s.dag.node(id), value);
            assert_eq!(s.dag.node(id).edges, [Edge::Opaque(value)]);
            s.dag.dec_ref(id);
        }
    }
    fail_with(&shared, &[NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}
