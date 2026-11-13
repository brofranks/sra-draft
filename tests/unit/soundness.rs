//! Small soundness fixtures for the DAG's unsafe operations.
//!
//! Heap-owning values expose leaks and double drops. Small arenas and two
//! threads exercise publication ordering, compaction, and reader/maintenance
//! handoff. Miri can detect safety violations in observed executions. Soundness
//! requires reasoning about all possible executions.
//!
//! Run manually with
//! `cargo +nightly miri test --locked --lib util::dag::soundness::`.

use std::array::from_fn;
use std::hint::spin_loop;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use super::{Dag, Exhausted, Grow, Idx, NeedGrowth, Shared, structural_hash, unpack_next};

idx! { struct Id; }

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

/// An arena with heap-owning values, exposing omitted destruction as leaks and
/// double drops as use-after-free.
type Arena<I> = Dag<I, Box<u64>, Edge<I>>;

/// The node and edge allocation frontiers.
fn frontiers<I: Idx>(dag: &Arena<I>) -> (usize, usize) {
    unpack_next(dag.next.load(Ordering::Relaxed))
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

// --- publication -------------------------------------------------------------

#[test]
fn publication_is_visible_through_acquire_alone() {
    let dag: Arena<Id> = Dag::with_capacity(8, 8, [Box::new(10)]);
    let leaf = Id::new(0);
    let shapes: [(u64, &[Edge<Id>]); 3] = [
        (1, &[]),
        (2, &[Edge::Child(leaf)]),
        (3, &[Edge::Opaque(4), Edge::Opaque(5)]),
    ];
    // The reader learns each ID through a Relaxed handoff, so the node's own
    // Release publication and Acquire read are the only ordering between the write
    // and the read.
    let handoff: [AtomicUsize; 3] = from_fn(|_| AtomicUsize::new(0));
    thread::scope(|scope| {
        let (dag, handoff) = (&dag, &handoff);
        scope.spawn(move || {
            for (cell, &(value, edges)) in handoff.iter().zip(&shapes) {
                let id = dag.publish(|| Box::new(value), edges).unwrap();
                cell.store(id.index() + 1, Ordering::Relaxed);
            }
        });
        scope.spawn(move || {
            // Publishing from the reader too contends for the frontier with the writer.
            let own = dag.publish(|| Box::new(6), &[Edge::Opaque(6)]).unwrap();
            for (cell, &(value, edges)) in handoff.iter().zip(&shapes) {
                let id = loop {
                    match cell.load(Ordering::Relaxed) {
                        0 => spin_loop(),
                        id => break Id::new(id - 1),
                    }
                };
                while dag.get(id).is_none() {
                    spin_loop();
                }
                assert_eq!((**dag.node(id), dag.node(id).edges), (value, edges));
            }
            assert_eq!(
                (**dag.node(own), dag.node(own).edges),
                (6, &[Edge::Opaque(6)][..])
            );
        });
    });
    assert_eq!(frontiers(&dag).0, 5);
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
    assert_eq!(frontiers(&s.dag).1, 4);
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
    drop(s);

    // A forced growth reclaims the reused slot again and moves the node arena.
    fail_with(&shared, &[NeedGrowth::Nodes, NeedGrowth::Nodes]);
    let s = shared.read(());
    assert_eq!(s.reclaimed, [1, 3, 5]);
    assert!(s.dag.node_capacity() > 8);
    assert_eq!(**s.dag.node(live), 2);
    drop(s);
    // Dropping the arena drops the pinned values once and skips reclaimed and
    // never-published slots.
    drop(shared);
}

#[test]
fn publication_after_a_destructor_panic_discards_reserved_batches() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

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

    let drops = from_fn(|_| AtomicUsize::new(0));
    let mut dag: Dag<Id, Value<'_>, u64> = Dag::with_capacity(4096, 4096, []);
    let value = |n| Value {
        value: Box::new(n),
        drops: &drops,
    };
    let first = dag.publish(|| value(0), &[0]).unwrap();
    dag.inc_ref(first);
    dag.publish(|| value(1), &[1]).unwrap();
    dag.publish(|| value(2), &[2]).unwrap();
    assert_eq!(dag.compact(|_| None, |_| {}), 2);
    let held = dag.publish(|| value(3), &[3]).unwrap();
    dag.inc_ref(held);
    dag.dec_ref(first);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        dag.compact(|_| None, |_| {});
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
fn concurrent_interning_survives_growth_with_owned_values() {
    const KEYS: u64 = 6;
    const ROUNDS: usize = 8;
    for _round in 0..ROUNDS {
        // Room for almost nothing, so each thread's interning fails and grows while the
        // other is admitted or failing itself. The outcome is one canonical node per
        // key.
        let shared = Store::<Id>::shared(2, 2);
        let barrier = Barrier::new(2);
        let ids: Vec<Vec<Id>> = thread::scope(|scope| {
            let workers: Vec<_> = (0..2_u64)
                .map(|w| {
                    let (shared, barrier) = (&shared, &barrier);
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
        for (k, &id) in ids[0].iter().enumerate() {
            assert_eq!(**s.dag.node(id), k as u64);
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
        let shared = Store::<Id>::shared(4, 0);
        let barrier = Barrier::new(2);
        thread::scope(|scope| {
            let (shared, barrier) = (&shared, &barrier);
            scope.spawn(move || {
                let s = shared.read(());
                barrier.wait();
                // Widen the release/growth race. The writer may reach its drain loop before or
                // after the reader leaves.
                for _ in 0..256 {
                    spin_loop();
                }
                drop(s);
            });
            scope.spawn(move || {
                barrier.wait();
                fail_with(shared, &[NeedGrowth::Nodes]);
            });
        });
        assert_eq!(shared.growths(), 1);
    }
}

#[test]
fn maintenance_after_reader_scope_reclaims_owned_values() {
    // One thread: an admitted reader exhausts node storage, leaves by scope end,
    // and then runs maintenance alone. Collection can satisfy this request
    // without increasing capacity.
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
