//! Public-contract tests for `sra::util::dag`.
//!
//! Fixtures use a value word and edges holding child links or opaque values.
//! They exercise publication, collection, compaction, and growth through an
//! exclusively owned arena and through a small store with a caller-supplied
//! growth policy.
//!
//! Small arenas force collect-and-grow retries. Roomier arenas isolate
//! publication and unique-table exhaustion.

use std::cell::Cell;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::mem::DropGuard;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::Barrier;
use std::thread;

use sra::util::dag::{
    Dag, Exhausted, Grow, Idx, Map, NeedGrowth, Set, Shared, StructuralHasher, structural_hash,
};

sra::idx! { struct Id; }
sra::idx! { struct Reserved < u32::MAX as usize; }

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
            Self::Child(i) => Some(i),
            Self::Opaque(_) => None,
        }
    }
}

/// An arena over value words and the test edges.
type Arena<I> = Dag<I, u64, Edge<I>>;

/// The number of published nodes in the arena.
fn published<I: Idx>(dag: &Arena<I>) -> usize {
    (0..dag.node_capacity())
        .filter(|&i| dag.get(I::new(i)).is_some())
        .count()
}

/// Asserts that exactly `free` edge slots remain, by refusing one span longer
/// than that and publishing one of that length.
fn assert_free_edges<I: Idx>(dag: &Arena<I>, free: usize) -> I {
    let span = vec![Edge::Opaque(0); free + 1];
    assert_eq!(dag.publish(|| 0, &span), Err(NeedGrowth::Edges(free + 1)));
    dag.publish(|| 0, &span[..free])
        .expect("the remaining edges must hold the span")
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
        self.reclaimed.extend(maintain(&mut self.dag, exhausted));
    }
}

/// A node hash using only the low half of its value.
///
/// Varying the high half produces distinct values with the same hash.
fn node_hash<I: Idx>(value: u64, edges: &[Edge<I>]) -> u64 {
    structural_hash((value as u32, edges))
}

/// Maintains an arena on one thread, returning the reclaimed values in
/// reclamation order.
fn maintain<I: Idx>(dag: &mut Arena<I>, exhausted: Exhausted) -> Vec<u64> {
    let mut reclaimed = Vec::new();
    dag.maintain(
        exhausted,
        1,
        Edge::child,
        |value| reclaimed.push(*value),
        |value, edges| node_hash(*value, edges),
    );
    reclaimed
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

#[cfg(target_pointer_width = "64")]
#[test]
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
fn structural_hashing_separates_word_tails_and_leading_zeros() {
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
    // Leading zero words change the state.
    assert_ne!(structural_hash((0_u64, 5_u64)), structural_hash(5_u64));
}

#[test]
fn structural_maps_hash_each_key_from_a_fresh_state() {
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
    assert_eq!((dag.node_capacity(), dag.edge_capacity()), (4, 4));
}

#[test]
fn a_dag_publishes_pinned_leaves_and_counts_references_at_every_index_type() {
    lifecycle::<Id>();
    lifecycle::<Reserved>();
    lifecycle::<Manual>();
}

#[test]
fn a_panicking_pinned_iterator_drops_initialized_values_once() {
    let drops = Cell::new(0);
    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
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
    // A span larger than the remaining edge capacity reports its full length and
    // builds nothing, so the next span that fits still publishes.
    assert_eq!(
        dag.publish(build, &[Edge::Opaque(0), Edge::Opaque(1), Edge::Opaque(2)]),
        Err(NeedGrowth::Edges(3))
    );
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
fn independent_node_and_edge_batches_preserve_published_contents() {
    // Reverse the relative batch sizes, and mix empty, short, and oversized spans
    // so node and edge ranges must refill independently.
    for (nodes, edge_capacity) in [(8192, 4096), (4096, 8192)] {
        let dag: Arena<Id> = Dag::with_capacity(nodes, edge_capacity, []);
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
#[should_panic(expected = "cannot wrap an index outside the accepted range")]
fn a_pinned_leaf_beyond_the_index_range_is_rejected() {
    // The node capacity fits all three leaves, but the index type names only two.
    sra::idx! { struct Pair < 2; }
    let _: Arena<Pair> = Dag::with_capacity(3, 0, [1, 2, 3]);
}

// --- interning ---------------------------------------------------------------

#[test]
fn interning_returns_one_canonical_node_per_key() {
    // The smallest table holds six entries in one eight-bucket group, so every
    // probe walks the same buckets and equality must tell apart the entries it
    // passes.
    let dag: Arena<Id> = Dag::with_capacity(6, 3, []);
    assert_eq!(dag.unique_capacity(), 6);
    let a = intern(&dag, 1, &[Edge::Opaque(1)]).unwrap();
    assert_eq!(intern(&dag, 1, &[Edge::Opaque(1)]), Ok(a));
    // Equal values with different edges are different nodes.
    let b = intern(&dag, 1, &[Edge::Opaque(2)]).unwrap();
    assert_ne!(b, a);

    // A distinct value that hashes the same shares the probe and the fingerprint;
    // only `eq` rejects it, and it is then interned into a different bucket of the
    // same sequence.
    let c = intern(&dag, 1 << 32 | 1, &[Edge::Opaque(1)]).unwrap();
    assert_ne!(c, a);
    assert_eq!(
        node_hash(1 << 32 | 1, &[Edge::<Id>::Opaque(1)]),
        node_hash(1, &[Edge::<Id>::Opaque(1)])
    );
    assert_eq!(intern(&dag, 1 << 32 | 1, &[Edge::Opaque(1)]), Ok(c));
    assert_eq!(*dag.node(c), 1 << 32 | 1);
}

#[test]
fn a_full_unique_table_refuses_before_publishing() {
    // A table sized for the arena always has room for every node the arena can
    // hold, so the only way to fill one is to grow the arena alone. Forced node
    // growth retains the table capacity.
    let shared = Store::<Id>::shared(6, 0);
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    let s = shared.read(());
    let dag = &s.dag;
    assert!(dag.node_capacity() > dag.unique_capacity());
    for value in 0..dag.unique_capacity() as u64 {
        assert!(intern(dag, value, &[]).is_ok());
    }

    // The table refuses a new entry at its capacity rather than probing on, and
    // says so before any candidate is built, even though the arena still has room.
    let built = Cell::new(false);
    assert_eq!(
        dag.intern(
            node_hash::<Id>(99, &[]),
            |_, _| false,
            || built.replace(true) as u64,
            &[]
        ),
        Err(NeedGrowth::Unique)
    );
    assert!(!built.get());
    assert!(dag.publish(|| 99, &[]).is_ok());
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

    // Publication reuses reclaimed IDs, highest index first.
    let reused = dag.publish(|| 55, &[]).unwrap();
    assert_eq!(reused, garbage.1);
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
    assert_eq!(s.dag.publish(|| 56, &[]).unwrap(), garbage.1);
    // The five live edges sit at the start of the arena.
    assert_free_edges(&s.dag, s.dag.edge_capacity() - 5);
}

#[test]
fn publication_after_a_reclaim_panic_discards_reserved_free_list_entries() {
    // Two-slot batches leave an unused frontier slot, then an unused free-list
    // entry. The second collection clears that list before its callback panics.
    let collect = Exhausted::from(NeedGrowth::Nodes);
    let mut dag: Arena<Id> = Dag::with_capacity(4096, 4096, []);
    let first = dag.publish(|| 0, &[Edge::Opaque(0)]).unwrap();
    dag.inc_ref(first);
    dag.publish(|| 1, &[Edge::Opaque(1)]).unwrap();
    dag.publish(|| 2, &[Edge::Opaque(2)]).unwrap();
    assert_eq!(maintain(&mut dag, collect), [1, 2]);
    let held = dag.publish(|| 3, &[Edge::Opaque(3)]).unwrap();
    dag.inc_ref(held);
    dag.dec_ref(first);

    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        dag.maintain(
            collect,
            1,
            Edge::child,
            |_| panic!("reclaim interrupted"),
            |value, edges| node_hash(*value, edges),
        );
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
    assert_eq!(maintain(&mut dag, collect).len(), 4);
    assert_eq!(published(&dag), 0);
}

#[test]
fn reclaimed_ids_are_reused_before_fresh_ones() {
    // Several reclaimed nodes require both a complete free-list batch and a final
    // partial batch. A retained node keeps its identity throughout reuse.
    let mut dag: Arena<Id> = Dag::with_capacity(4096, 4096, []);
    let ids: Vec<_> = (0..6)
        .map(|value| dag.publish(|| value, &[Edge::Opaque(value)]).unwrap())
        .collect();
    dag.inc_ref(ids[0]);
    assert_eq!(
        maintain(&mut dag, Exhausted::from(NeedGrowth::Nodes)).len(),
        5
    );
    let mut reused = HashSet::new();
    for value in 10..15 {
        let id = dag.publish(|| value, &[Edge::Opaque(value)]).unwrap();
        assert!(ids[1..].contains(&id));
        assert!(reused.insert(id));
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
fn collection_preserves_empty_roots_after_discarding_earlier_edges() {
    let mut dag: Arena<Id> = Dag::with_capacity(8, 8, []);
    dag.publish(|| 1, &[Edge::Opaque(1), Edge::Opaque(2)])
        .unwrap();
    let leaf = dag.publish(|| 7, &[]).unwrap();
    dag.pin(leaf);
    maintain(&mut dag, Exhausted::from(NeedGrowth::Nodes));
    assert_eq!(*dag.node(leaf), 7);
    assert!(dag.node(leaf).edges.is_empty());
    // Every edge slot is free again, and a span filling them leaves the empty root
    // empty.
    let next = assert_free_edges(&dag, 8);
    assert_eq!(dag.node(next).edges, [Edge::Opaque(0); 8]);
    assert!(dag.node(leaf).edges.is_empty());
}

#[test]
fn an_owned_dag_collects_and_grows_from_one_failure() {
    // Exclusive access needs no `Shared`: the failure itself names the exhausted
    // resource.
    let mut dag: Arena<Id> = Dag::with_capacity(2, 0, []);
    let kept = dag.publish(|| 1, &[]).unwrap();
    dag.inc_ref(kept);
    dag.publish(|| 2, &[]).unwrap();
    let reason = dag.publish(|| 3, &[]).unwrap_err();
    assert_eq!(reason, NeedGrowth::Nodes);

    // Collection frees one of two slots, which leaves too little free space, so
    // the node arena also grows.
    assert_eq!(maintain(&mut dag, reason.into()), [2]);
    assert!(dag.node_capacity() > 2);
    assert_eq!(*dag.node(kept), 1);
    assert!(dag.publish(|| 3, &[]).is_ok());
}

#[test]
#[should_panic(expected = "maintenance thread count must be nonzero")]
fn maintenance_without_a_caller_thread_is_rejected() {
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    dag.maintain(
        Exhausted::from(NeedGrowth::Nodes),
        0,
        Edge::child,
        |_| {},
        |value, edges| node_hash(*value, edges),
    );
}

#[test]
#[should_panic(expected = "arena capacity overflow")]
fn growing_an_arena_beyond_u32_slots_is_rejected() {
    // One pinned edge ahead of a span of `u32::MAX` edges asks for one more slot
    // than the edge frontier can address.
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    let leaf = dag.publish(|| 1, &[Edge::Opaque(1)]).unwrap();
    dag.pin(leaf);
    maintain(&mut dag, NeedGrowth::Edges(u32::MAX as usize).into());
}

// --- growth policy -----------------------------------------------------------

#[test]
fn exclusive_access_changes_the_store_identity() {
    let mut shared = Store::<Id>::shared(8, 0);
    let id = pinned(&shared, 7, &[]);
    let (store, epoch) = (shared.id, shared.epoch());
    *shared.get_mut() = Store {
        dag: Dag::with_capacity(8, 0, [11]),
        reclaimed: Vec::new(),
    };
    // Memo tables keyed by the old identity are stale: the same ID now names the
    // replacement's node, though no growth ran.
    assert_ne!(shared.id, store);
    assert_eq!(shared.epoch(), epoch);
    assert_eq!(*shared.read(()).dag.node(id), 11);
}

#[test]
fn a_single_failure_with_ample_free_space_collects_without_growing() {
    // The free share is above the threshold that makes collection worth its fixed
    // cost, so a single failure collects but must not grow.
    let shared = Store::<Id>::shared(64, 64);
    fail_with(&shared, &[NeedGrowth::Nodes]);
    fail_with(&shared, &[NeedGrowth::Edges(4)]);
    assert_eq!(shared.epoch(), 2);
    let s = shared.read(());
    assert_eq!((s.dag.node_capacity(), s.dag.edge_capacity()), (64, 64));
}

#[test]
fn an_owned_retry_terminates_by_forcing_growth() {
    // Each failure leaves an unreferenced partial chain, so collection frees the
    // whole arena and the threshold check alone never grows it. Forcing the second
    // pass grows the arena, so the third attempt succeeds.
    let mut dag: Arena<Id> = Dag::with_capacity(2, 8, []);
    let chain = |arena: &Arena<Id>| {
        let leaf = intern(arena, 1, &[])?;
        let mid = intern(arena, 2, &[Edge::Child(leaf)])?;
        intern(arena, 3, &[Edge::Child(mid)])
    };
    let first = chain(&dag).unwrap_err();
    maintain(&mut dag, Exhausted::from(first));
    assert_eq!(dag.node_capacity(), 2);
    let second = chain(&dag).unwrap_err();
    maintain(&mut dag, Exhausted::from(second).forced());
    let top = chain(&dag).unwrap();
    dag.inc_ref(top);
    assert!(dag.node_capacity() > 2);
}

#[test]
fn a_repeated_failure_grows_only_the_exhausted_resource() {
    // Collection alone would otherwise reproduce the same failure forever, so the
    // second failure in one retry forces growth. An empty store has all the free
    // space it can, so forcing alone grows the resource, and only that one.
    let capacities = |dag: &Arena<Id>| {
        [
            dag.node_capacity(),
            dag.edge_capacity(),
            dag.unique_capacity(),
        ]
    };
    for (reason, grown) in [
        (NeedGrowth::Nodes, [true, false, false]),
        (NeedGrowth::Edges(4), [false, true, false]),
        (NeedGrowth::Unique, [false, false, true]),
    ] {
        let shared = Store::<Id>::shared(64, 64);
        let before = capacities(&shared.read(()).dag);
        fail_with(&shared, &[reason, reason]);
        let after = capacities(&shared.read(()).dag);
        assert_eq!([0, 1, 2].map(|i| after[i] > before[i]), grown, "{reason:?}");
    }
}

#[test]
fn a_span_beyond_the_free_edges_grows_to_admit_it() {
    // The span grows the arena even unforced, and the grown arena admits the whole
    // span without another maintenance pass.
    let shared = Store::<Id>::shared(64, 64);
    let span = vec![Edge::Opaque(0); 64 + 9];
    fail_with(&shared, &[NeedGrowth::Edges(span.len())]);
    let epoch = shared.epoch();
    assert!(shared.read(()).dag.publish(|| 0, &span).is_ok());
    assert_eq!(shared.epoch(), epoch);
}

#[test]
fn a_nearly_full_arena_grows_on_one_failure() {
    // A single unforced failure does grow once the free share is low, even when the
    // span it asked for would have fit: collection is proportional to the whole
    // arena, so a nearly full one must not pay a collection per allocation.
    let shared = Store::<Id>::shared(8, 8);
    let ids: Vec<_> = (0..3)
        .map(|k| pinned(&shared, k, &[Edge::Opaque(k), Edge::Opaque(k + 1)]))
        .collect();
    fail_with(&shared, &[NeedGrowth::Edges(1)]);
    let s = shared.read(());
    assert!(s.dag.edge_capacity() > 8);
    // Growth alone: the pinned nodes and their edges are untouched by the
    // collection.
    assert!(s.reclaimed.is_empty());
    for (k, id) in (0..).zip(ids) {
        assert_eq!(s.dag.node(id).edges, [Edge::Opaque(k), Edge::Opaque(k + 1)]);
    }
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
fn concurrent_interning_agrees_on_one_node_per_key_across_growth() {
    // Eight workers, a six-entry table, and an arena with room for almost nothing:
    // admission, publication, bucket installation, and growth all collide. Each
    // worker starts at a different key so no interleaving is assumed. The second
    // family hashes every key alike, so every worker walks the same probe sequence
    // and candidates lose their bucket to a different value. The test checks that
    // every attempt eventually succeeds and all workers agree on each key's
    // canonical node. Repetitions sample schedules without requiring a particular
    // race in any round.
    const WORKERS: usize = 8;
    type Key = fn(u64) -> (u64, Vec<Edge<Id>>);
    let families: [(u64, Key); 2] = [
        (12, |k| (k, vec![Edge::Opaque(k)])),
        (10, |k| (k << 32 | 3, Vec::new())),
    ];
    for (keys, key) in families {
        for _round in 0..100 {
            let shared = &Store::<Id>::shared(2, 2);
            let results: Vec<Vec<Id>> = thread::scope(|scope| {
                let workers: Vec<_> = (0..WORKERS as u64)
                    .map(|w| {
                        scope.spawn(move || {
                            let mut ids = vec![Id::new(0); keys as usize];
                            for step in 0..keys {
                                let k = (step + w) % keys;
                                let (value, edges) = key(k);
                                ids[k as usize] = pinned(shared, value, &edges);
                            }
                            ids
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).collect()
            });

            let ids = &results[0];
            assert_eq!(ids.iter().collect::<HashSet<_>>().len(), keys as usize);
            for worker in &results {
                assert_eq!(worker, ids, "workers disagreed on the canonical nodes");
            }
            let s = shared.read(());
            for (k, &id) in (0..).zip(ids) {
                let (value, edges) = key(k);
                assert_eq!(*s.dag.node(id), value);
                assert_eq!(s.dag.node(id).edges, edges);
            }
        }
    }
}

#[test]
fn a_candidate_that_loses_every_remaining_bucket_reports_a_full_table() {
    const WORKERS: usize = 8;
    // Six entries with five filled and every key hashing alike leave the
    // remaining buckets to eight candidates. Grow just the node arena so every
    // candidate has room for publication.
    let shared = &Store::<Id>::shared(2, 2);
    while shared.read(()).dag.node_capacity() < 64 {
        fail_with(shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    }
    assert_eq!(shared.read(()).dag.unique_capacity(), 6);
    let filled: Vec<Id> = (0..5).map(|i| pinned(shared, i << 32 | 5, &[])).collect();

    let epoch = shared.epoch();
    let barrier = &Barrier::new(WORKERS);
    let attempts: Vec<_> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS as u64)
            .map(|w| {
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
    assert_eq!(shared.epoch(), epoch);

    // Retry the refused candidates. Growth preserves winners and the old entries.
    let ids: Vec<_> = (0..WORKERS as u64)
        .map(|w| pinned(shared, (w + 10) << 32 | 5, &[]))
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
    let dag: &Arena<Id> = &Dag::with_capacity(WORKERS * EACH as usize, 0, []);
    let published: Vec<(Id, u64)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS as u64)
            .map(|w| {
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

    let shared = &Store::<Id>::shared(2, 2);
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
                scope.spawn(move || {
                    let mut seed = w.wrapping_mul(0x9E37_79B9_7F4A_7C15) + 1;
                    let mut next = move || {
                        seed = seed
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
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
    fail_with(shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
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
    fail_with(shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}

#[test]
fn concurrent_batched_publication_preserves_contents_and_reclaims_nodes() {
    // A large arena gives each thread multi-slot regions. Published IDs are
    // distinct, their contents are preserved, and collection reclaims every node
    // after its external references are released.
    const WORKERS: u64 = 4;
    const EACH: u64 = 10_000;
    let shared = &Store::<Id>::shared(1 << 16, 1 << 17);
    let ids: Vec<Vec<Id>> = thread::scope(|scope| {
        let workers: Vec<_> = (0..WORKERS)
            .map(|w| {
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
        for (w, batch) in ids.iter().enumerate() {
            for (i, &id) in batch.iter().enumerate() {
                assert_eq!(*s.dag.node(id), (w as u64) << 32 | i as u64);
                assert_eq!(s.dag.node(id).edges, [Edge::Opaque(i as u64)]);
            }
        }
        for &id in &all {
            s.dag.dec_ref(id);
        }
    }
    fail_with(shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}

#[test]
fn workers_resume_reserved_batches_after_compaction() {
    // Workers remain alive across two collections. Each phase leaves a partial
    // two-slot batch, first in the frontier and then in the rebuilt free list.
    // Barriers keep every admission guard outside the maintenance phases.
    let shared = &Store::<Id>::shared(4096, 4096);
    let barrier = &Barrier::new(3);
    let ids = thread::scope(|scope| {
        let workers: Vec<_> = (0..2_u64)
            .map(|w| {
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
                        for &(kept, v) in &held {
                            assert_eq!(*s.dag.node(kept), v);
                            assert_eq!(s.dag.node(kept).edges, [Edge::Opaque(v)]);
                        }
                    }
                    held
                })
            })
            .collect();
        for phase in 0..3 {
            barrier.wait();
            if phase < 2 {
                fail_with(shared, &[NeedGrowth::Nodes]);
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
    fail_with(shared, &[NeedGrowth::Nodes]);
    assert_eq!(published(&shared.read(()).dag), 0);
}
