//! Small soundness fixtures for the DAG's unsafe operations.
//!
//! Heap-owning values expose leaks and double drops. Small arenas and a few
//! threads exercise publication ordering, compaction, and reader/maintenance
//! handoff. Index types that misreport their index or reenter the DAG show that
//! a wrong safe `Idx` implementation keeps memory safety. Miri can detect
//! safety violations in observed executions. Soundness requires reasoning about
//! all possible executions.
//!
//! Run under Miri with
//! `cargo +nightly miri test --locked --test soundness`. CI also enables strict
//! provenance, symbolic alignment checks, both borrow models, and a seed sweep.

use std::array;
use std::cell::Cell;
use std::hint;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use sra::util::dag::{Dag, Exhausted, Grow, Idx, NeedGrowth, Read, Shared, structural_hash};

sra::idx! { struct Id; }

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

/// An arena with heap-owning values, exposing omitted destruction as leaks and
/// double drops as use-after-free.
type Arena<I> = Dag<I, Box<u64>, Edge<I>>;

/// The number of published nodes in the arena.
fn published<I: Idx>(dag: &Arena<I>) -> usize {
    (0..dag.node_capacity())
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
            |value| reclaimed.push(**value),
            |value, edges| node_hash(**value, edges),
        );
    }
}

/// A node hash using only the low half of its value.
///
/// Varying the high half produces distinct values with the same hash.
fn node_hash<I: Idx>(value: u64, edges: &[Edge<I>]) -> u64 {
    structural_hash((value as u32, edges))
}

/// Interns a boxed value with its edges under the hash the store's own rebuild
/// recomputes.
fn intern<I: Idx>(dag: &Arena<I>, value: u64, edges: &[Edge<I>]) -> Result<I, NeedGrowth> {
    dag.intern(
        node_hash(value, edges),
        |v, stored| **v == value && stored == edges,
        || Box::new(value),
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

// --- index implementations ---------------------------------------------------

/// An index that reports the slot after the one it wraps.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Shifted(usize);

impl Idx for Shifted {
    fn new(index: usize) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0 + 1
    }
}

/// An index whose construction, once armed, publishes into this thread's arena,
/// as a conversion reentering the DAG would.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Reentrant(usize);

thread_local! {
    /// The arena that an armed `Reentrant::new` publishes into.
    static REENTERED: Arena<Reentrant> = Dag::with_capacity(4096, 0, []);
    /// Whether the next `Reentrant::new` publishes.
    static ARMED: Cell<bool> = const { Cell::new(false) };
    /// The index of the node an armed construction published.
    static INNER: Cell<Option<usize>> = const { Cell::new(None) };
}

impl Idx for Reentrant {
    fn new(index: usize) -> Self {
        if ARMED.replace(false) {
            let inner = REENTERED.with(|dag| dag.publish(|| Box::new(2), &[]).unwrap());
            INNER.set(Some(inner.0));
        }
        Self(index)
    }

    fn index(self) -> usize {
        self.0
    }
}

#[test]
#[should_panic(expected = "marked DAG node must remain published")]
fn a_misreporting_index_in_collection_is_rejected() {
    let shared = Store::<Shifted>::shared(4, 4);
    {
        let s = shared.read(());
        // The nodes occupy raw slots 0 and 1, but each ID names the next slot.
        let a = s.dag.publish(|| Box::new(1), &[]).unwrap();
        let b = s.dag.publish(|| Box::new(2), &[Edge::Child(a)]).unwrap();
        assert_eq!(s.dag.get(a).map(|n| **n), Some(2));
        assert!(s.dag.get(b).is_none());
        s.dag.pin(a);
    }

    // Marking finds the pinned slot 1 but looks it up through an ID naming the
    // unpublished slot 2. The panic precedes reclamation, and unwinding drops both
    // values once.
    fail_with(&shared, &[NeedGrowth::Nodes]);
}

#[test]
fn a_reentrant_conversion_reserves_a_distinct_slot() {
    // A thread of its own destroys the thread-local arena on exit, so Miri checks
    // its values for leaks and double drops.
    thread::spawn(|| {
        REENTERED.with(|dag| {
            // This capacity reserves two node slots per batch, so the warm-up leaves a
            // spare slot in this thread's cached range for the next publication.
            dag.publish(|| Box::new(0), &[]).unwrap();
            ARMED.set(true);
            let outer = dag.publish(|| Box::new(1), &[]).unwrap();
            let inner = Reentrant(INNER.get().unwrap());
            assert_ne!(outer, inner);
            assert_eq!((**dag.node(outer), **dag.node(inner)), (1, 2));
        });
    })
    .join()
    .unwrap();
}

// --- publication -------------------------------------------------------------

#[test]
fn publication_is_visible_through_acquire_alone() {
    let transitions: [fn(&Arena<Id>, Id); 4] = [
        |_, _| {},
        |dag, id| dag.pin(id),
        |dag, id| dag.inc_ref(id),
        |dag, id| {
            dag.inc_ref(id);
            dag.dec_ref(id);
        },
    ];
    for transition in transitions {
        let dag: &Arena<Id> = &Dag::with_capacity(8, 8, [Box::new(10)]);
        let leaf = Id::new(0);
        let shapes: [(u64, &[Edge<Id>]); 3] = [
            (1, &[]),
            (2, &[Edge::Child(leaf)]),
            (3, &[Edge::Opaque(4), Edge::Opaque(5)]),
        ];
        // The reader learns each ID through a Relaxed handoff, so the node's own
        // Release publication and Acquire read are the only ordering between the write
        // and the read.
        let handoff: &[AtomicUsize; 3] = &array::from_fn(|_| AtomicUsize::new(0));
        thread::scope(|scope| {
            scope.spawn(move || {
                for (cell, &(value, edges)) in handoff.iter().zip(&shapes) {
                    let id = dag.publish(|| Box::new(value), edges).unwrap();
                    // Each reference-state RMW must preserve the publication release sequence.
                    transition(dag, id);
                    cell.store(id.index() + 1, Ordering::Relaxed);
                }
            });
            scope.spawn(move || {
                // Publishing from the reader too contends for the frontier with the writer.
                let own = dag.publish(|| Box::new(6), &[Edge::Opaque(6)]).unwrap();
                for (cell, &(value, edges)) in handoff.iter().zip(&shapes) {
                    let id = loop {
                        match cell.load(Ordering::Relaxed) {
                            0 => hint::spin_loop(),
                            id => break Id::new(id - 1),
                        }
                    };
                    while dag.get(id).is_none() {
                        hint::spin_loop();
                    }
                    assert_eq!((**dag.node(id), dag.node(id).edges), (value, edges));
                }
                assert_eq!(
                    (**dag.node(own), dag.node(own).edges),
                    (6, &[Edge::Opaque(6)][..])
                );
            });
        });
        assert_eq!(published(dag), 5);
    }
}

#[test]
fn publication_with_no_edge_storage_reads_an_empty_span() {
    // No edge storage at all: the span of a leaf is a zero-length slice over a
    // dangling pointer.
    let bare: Arena<Id> = Dag::with_capacity(4, 0, []);
    let id = bare.publish(|| Box::new(7), &[]).unwrap();
    assert!(bare.node(id).edges.is_empty());
    assert_eq!(**bare.node(id), 7);
}

#[test]
fn over_aligned_zero_sized_edges_read_back() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(align(64))]
    struct Z;

    let bare: Dag<Id, Box<u64>, Z> = Dag::with_capacity(1, 0, []);
    let empty = bare.publish(|| Box::new(0), &[]).unwrap();
    assert!(bare.node(empty).edges.is_empty());
    assert!(bare.node(empty).edges.as_ptr().is_aligned());

    let mut dag: Dag<Id, Box<u64>, Z> = Dag::with_capacity(3, 4, []);
    let garbage = dag.publish(|| Box::new(1), &[Z]).unwrap();
    let span = dag.publish(|| Box::new(2), &[Z, Z, Z]).unwrap();
    let empty = dag.publish(|| Box::new(3), &[]).unwrap();
    dag.pin(span);
    dag.pin(empty);
    assert!(dag.node(empty).edges.is_empty());
    assert_eq!(dag.node(span).edges, [Z, Z, Z]);
    assert!(dag.node(span).edges.as_ptr().is_aligned());
    // The live span moves from 1..4 to 0..3. ZST growth increases logical
    // capacity without requiring a backing allocation to move.
    dag.maintain(
        Exhausted::from(NeedGrowth::Edges(1)).forced(),
        1,
        |_| None,
        |_| {},
        |_, _| 0,
    );
    assert!(dag.get(garbage).is_none());
    assert!(dag.edge_capacity() > 4);
    assert_eq!(dag.node(span).edges, [Z, Z, Z]);
    assert!(dag.node(span).edges.as_ptr().is_aligned());
    assert!(dag.node(empty).edges.is_empty());
    assert!(dag.node(empty).edges.as_ptr().is_aligned());
}

// --- concurrent interning ----------------------------------------------------

#[test]
fn equal_key_candidates_publish_one_canonical_node() {
    // Both threads observe the empty bucket and reach the value callback before
    // either publishes, so both publish a candidate for the same key and one
    // loses its bucket and reads the winner. Collection drops the loser, and
    // dropping the arena drops the pinned winner.
    let mut dag: Arena<Id> = Dag::with_capacity(4, 0, []);
    let barrier = &Barrier::new(2);
    let ids: Vec<Id> = thread::scope(|scope| {
        let dag = &dag;
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(move || {
                    let value = || {
                        barrier.wait();
                        Box::new(7)
                    };
                    let id = dag
                        .intern(node_hash::<Id>(7, &[]), |v, _| **v == 7, value, &[])
                        .unwrap();
                    dag.pin(id);
                    id
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    assert_eq!(ids[0], ids[1]);
    assert_eq!(published(&dag), 2);
    let mut reclaimed = Vec::new();
    dag.maintain(
        Exhausted::from(NeedGrowth::Nodes),
        1,
        Edge::child,
        |value| reclaimed.push(**value),
        |value, edges| node_hash(**value, edges),
    );
    assert_eq!(reclaimed, [7]);
    assert_eq!(intern(&dag, 7, &[]), Ok(ids[0]));
}

#[test]
fn equality_panic_after_lost_cas_leaves_candidate_owned_by_arena() {
    // Reentrant interning fills the bucket the outer call observed empty, so the
    // outer CAS loses and equality panics with the candidate already published.
    // Collection then drops the candidate, and dropping the arena drops the winner.
    let mut dag: Arena<Id> = Dag::with_capacity(4, 4, []);
    let hash = node_hash::<Id>(7, &[]);
    let mut winner = None;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let value = || {
            let inner = dag
                .intern(hash, |_, _| false, || Box::new(7), &[Edge::Opaque(7)])
                .unwrap();
            dag.pin(inner);
            winner = Some(inner);
            Box::new(20)
        };
        dag.intern(
            hash,
            |_, _| panic!("equality interrupted"),
            value,
            &[Edge::Opaque(20)],
        )
    }));
    assert!(result.is_err());
    let winner = winner.unwrap();
    assert_eq!(published(&dag), 2);
    let mut reclaimed = Vec::new();
    dag.maintain(
        Exhausted::from(NeedGrowth::Nodes),
        1,
        Edge::child,
        |value| reclaimed.push(**value),
        |_, _| hash,
    );
    assert_eq!(reclaimed, [20]);
    assert_eq!(
        (**dag.node(winner), dag.node(winner).edges),
        (7, &[Edge::Opaque(7)][..])
    );
}

// --- collection --------------------------------------------------------------

#[test]
fn collection_drops_reclaims_and_packs_overlapping_spans() {
    let shared = Store::<Id>::shared(8, 8);
    let (garbage, live, root) = {
        let s = shared.read(());
        let dag = &s.dag;
        // A one-edge hole directly below a three-edge live span: packing the span
        // leftward overlaps its own source. The interned garbage leaves a table entry
        // the rebuild drops.
        let published = dag.publish(|| Box::new(1), &[Edge::Opaque(0)]).unwrap();
        let live = intern(dag, 2, &[Edge::Opaque(1), Edge::Opaque(2), Edge::Opaque(3)]).unwrap();
        dag.pin(live);
        let interned = intern(dag, 3, &[]).unwrap();
        let root = dag.publish(|| Box::new(4), &[Edge::Child(live)]).unwrap();
        dag.pin(root);
        ((published, interned), live, root)
    };

    fail_with(&shared, &[NeedGrowth::Nodes]);
    let s = shared.read(());
    assert_eq!(s.reclaimed, [1, 3]);
    assert!(s.dag.get(garbage.0).is_none() && s.dag.get(garbage.1).is_none());
    assert_eq!(
        s.dag.node(live).edges,
        [Edge::Opaque(1), Edge::Opaque(2), Edge::Opaque(3)]
    );
    assert_eq!(s.dag.node(root).edges, [Edge::Child(live)]);
    assert_eq!(
        intern(
            &s.dag,
            2,
            &[Edge::Opaque(1), Edge::Opaque(2), Edge::Opaque(3)]
        ),
        Ok(live)
    );
    // Publication writes into the slot the collection dropped.
    let reused = s.dag.publish(|| Box::new(5), &[]).unwrap();
    assert_eq!(reused, garbage.1);
    // The four live edges sit at the start of the edge arena: a span filling the
    // rest publishes, and one edge more is refused unbuilt. Pinning keeps the
    // filler out of the next collection.
    let span = vec![Edge::Opaque(0); s.dag.edge_capacity() - 3];
    assert_eq!(
        s.dag.publish(|| unreachable!(), &span),
        Err(NeedGrowth::Edges(span.len()))
    );
    let filler = s.dag.publish(|| Box::new(6), &span[1..]).unwrap();
    s.dag.pin(filler);
    let capacity = s.dag.node_capacity();
    drop(s);

    // Collection reclaims the reused slot again, and the forced retry then moves
    // the node arena.
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    let s = shared.read(());
    assert_eq!(s.reclaimed, [1, 3, 5]);
    assert!(s.dag.node_capacity() > capacity);
    assert_eq!(**s.dag.node(live), 2);
    drop(s);
    // Dropping the arena drops the pinned values once and skips reclaimed and
    // never-published slots.
    drop(shared);
}

#[test]
fn reference_edges_survive_compaction_and_growth() {
    // Edges move as raw memory. References keep their provenance through an
    // overlapping compaction and the growth that follows it.
    let targets = [10, 11, 12, 13];
    let mut dag: Dag<Id, u64, &u64> = Dag::with_capacity(4, 4, []);
    let garbage = dag.publish(|| 0, &[&targets[0]]).unwrap();
    let live = dag
        .publish(|| 1, &[&targets[1], &targets[2], &targets[3]])
        .unwrap();
    dag.pin(live);
    let capacity = dag.edge_capacity();
    dag.maintain(
        Exhausted::from(NeedGrowth::Edges(1)),
        1,
        |_| None,
        |_| {},
        |v, edges| structural_hash((v, edges)),
    );
    assert!(dag.get(garbage).is_none());
    assert!(dag.edge_capacity() > capacity);
    let read: Vec<u64> = dag.node(live).edges.iter().map(|&&e| e).collect();
    assert_eq!(read, [11, 12, 13]);
}

#[test]
fn publication_after_a_destructor_panic_discards_reserved_batches() {
    // The first payload panics once when collected. Per-value counts reveal a
    // second drop, while heap ownership also exposes invalid reuse under Miri.
    struct Value<'a> {
        value: Box<u64>,
        drops: &'a [AtomicUsize; 6],
    }

    impl Drop for Value<'_> {
        fn drop(&mut self) {
            let previous = self.drops[*self.value as usize].fetch_add(1, Ordering::Relaxed);
            assert_eq!(previous, 0, "payload dropped twice");
            if *self.value == 0 {
                panic!("destructor interrupted");
            }
        }
    }

    let drops = array::from_fn(|_| AtomicUsize::new(0));
    let mut dag: Dag<Id, Value<'_>, u64> = Dag::with_capacity(4096, 4096, []);
    // A roomy arena collects without growing.
    let collect = |arena: &mut Dag<Id, Value<'_>, u64>| {
        let mut reclaimed = 0;
        arena.maintain(
            Exhausted::from(NeedGrowth::Nodes),
            1,
            |_| None,
            |_| reclaimed += 1,
            |v, edges| structural_hash((*v.value, edges)),
        );
        reclaimed
    };
    let value = |n| Value {
        value: Box::new(n),
        drops: &drops,
    };
    let first = dag.publish(|| value(0), &[0]).unwrap();
    dag.inc_ref(first);
    dag.publish(|| value(1), &[1]).unwrap();
    dag.publish(|| value(2), &[2]).unwrap();
    assert_eq!(collect(&mut dag), 2);
    let held = dag.publish(|| value(3), &[3]).unwrap();
    dag.inc_ref(held);
    dag.dec_ref(first);

    let panic = panic::catch_unwind(AssertUnwindSafe(|| {
        collect(&mut dag);
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"destructor interrupted")
    );
    assert!(dag.get(first).is_none());

    let a = dag.publish(|| value(4), &[4]).unwrap();
    let b = dag.publish(|| value(5), &[5]).unwrap();
    assert_ne!(a, b);
    assert_ne!(a, held);
    assert_ne!(b, held);
    for (id, expected) in [(held, 3), (a, 4), (b, 5)] {
        assert_eq!(*(*dag.node(id)).value, expected);
        assert_eq!(dag.node(id).edges, [expected]);
    }
    drop(dag);
    for count in drops {
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }
}

// --- reader and growth coordination ------------------------------------------

#[test]
fn a_guard_shared_and_released_on_other_threads_admits_growth() {
    // Two threads read through one guard admitted here, and a third releases it
    // as maintenance starts. The barrier samples overlap; the admission counter
    // alone orders release before exclusive access.
    let shared = &Store::<Id>::shared(4, 4);
    let id = pinned(shared, 1, &[Edge::Opaque(1)]);
    let s = shared.read(());
    thread::scope(|scope| {
        scope.spawn(|| assert_eq!(**s.dag.node(id), 1));
        assert_eq!(s.dag.node(id).edges, [Edge::Opaque(1)]);
    });
    let barrier = &Barrier::new(2);
    thread::scope(|scope| {
        scope.spawn(move || {
            barrier.wait();
            assert_eq!(**s.dag.node(id), 1);
            drop(s);
        });
        barrier.wait();
        fail_with(shared, &[NeedGrowth::Nodes]);
    });
    assert_eq!(shared.epoch(), 1);
    assert_eq!(**shared.read(()).dag.node(id), 1);
}

#[test]
fn concurrent_interning_survives_growth_with_owned_values() {
    const KEYS: u64 = 6;
    const ROUNDS: usize = 8;
    for _round in 0..ROUNDS {
        // Room for almost nothing, so each thread's interning fails and grows while the
        // other is admitted or failing itself. The outcome is one canonical node per
        // key.
        let shared = &Store::<Id>::shared(2, 2);
        let barrier = &Barrier::new(2);
        let ids: Vec<Vec<Id>> = thread::scope(|scope| {
            let workers: Vec<_> = (0..2_u64)
                .map(|w| {
                    scope.spawn(move || {
                        barrier.wait();
                        let mut ids = vec![Id::new(0); KEYS as usize];
                        for step in 0..KEYS {
                            let k = (step + w) % KEYS;
                            ids[k as usize] = pinned(shared, k, &[Edge::Opaque(k)]);
                        }
                        ids
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        assert_eq!(ids[0], ids[1]);
        let s = shared.read(());
        for (k, &id) in (0..).zip(&ids[0]) {
            assert_eq!(**s.dag.node(id), k);
            assert_eq!(s.dag.node(id).edges, [Edge::Opaque(k)]);
        }
    }
}

#[test]
fn reader_release_can_race_with_growth() {
    const ROUNDS: usize = 8;
    // Sample the last-reader release/growth race. The elected writer may be
    // draining this reader when its guard leaves. This is the handoff performed by
    // `retry` after a failure, where a reader's shared borrow must end before the
    // writer takes exclusive access. The barrier aligns the start and the spin loop
    // widens the window without guaranteeing a particular schedule.
    for _round in 0..ROUNDS {
        let shared = &Store::<Id>::shared(4, 0);
        let barrier = &Barrier::new(2);
        thread::scope(|scope| {
            scope.spawn(move || {
                let s = shared.read(());
                barrier.wait();
                // Widen the release/growth race. The writer may reach its drain loop before or
                // after the reader leaves.
                for _ in 0..256 {
                    hint::spin_loop();
                }
                drop(s);
            });
            scope.spawn(move || {
                barrier.wait();
                fail_with(shared, &[NeedGrowth::Nodes]);
            });
        });
        assert_eq!(shared.epoch(), 1);
    }
}

#[test]
fn a_guard_released_inside_its_callee_admits_growth_before_return() {
    // One thread: a reference stored in the by-value guard would remain
    // protected for the whole call. Releasing the guard must allow exclusive
    // maintenance before this call returns.
    fn release_then_grow(s: Read<'_, Store<Id>>, shared: &Shared<Store<Id>>) {
        drop(s);
        fail_with(shared, &[NeedGrowth::Nodes]);
    }

    let shared = Store::<Id>::shared(4, 0);
    release_then_grow(shared.read(()), &shared);
    assert_eq!(shared.epoch(), 1);
}

#[test]
fn maintenance_after_reader_scope_reclaims_owned_values() {
    // One thread: an admitted reader exhausts node storage, leaves by scope end,
    // and then runs maintenance alone. Collection can satisfy this request without
    // increasing capacity.
    let shared = Store::<Id>::shared(4, 0);
    {
        let s = shared.read(());
        let mut value = 0;
        while intern(&s.dag, value, &[]).is_ok() {
            value += 1;
        }
    }
    let id = pinned(&shared, 100, &[]);
    let s = shared.read(());
    assert_eq!(**s.dag.node(id), 100);
    assert_eq!(s.reclaimed, [0, 1, 2, 3]);
    assert_eq!(s.dag.node_capacity(), 4);
}
