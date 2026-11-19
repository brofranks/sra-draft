//! A hash-consed DAG arena with lock-free publication and caller-driven growth.
//!
//! A [`Dag`] stores nodes and contiguous edge spans in separate arenas, with a
//! unique table from structural hashes to node IDs. The [`Dag::intern`] method
//! consults the table first and returns an equal interned node when one exists.
//! It takes no locks itself. Its `eq` and `value` callbacks may block, which
//! stalls the calling thread and, once maintenance is requested, that
//! maintenance and every attempt awaiting admission. Leaves pinned by
//! [`Dag::with_capacity`] stay outside the table.
//!
//! Exhaustion returns [`NeedGrowth`], and the caller then runs
//! [`Dag::maintain`] with exclusive access. A node survives maintenance while
//! it is pinned, counted by [`Dag::inc_ref`], or reachable from such a node.
//! Any other ID is valid only until the next maintenance. The [`Shared`]
//! wrapper coordinates concurrent attempts with maintenance. Memo entries
//! belong to the operations built over the DAG, so callers own memo tables
//! keyed by node IDs and discard them when [`Shared::epoch`] or
//! [`Shared::id`] changes.
//!
//! A store wraps the DAG and supplies its maintenance policy. This one holds
//! lists, where each node carries a value and an optional edge to its tail.
//!
//! ```
//! use sra::util::dag::{Dag, Exhausted, Grow, Shared, structural_hash};
//!
//! sra::idx! {
//!     struct Id;
//! }
//!
//! struct Lists(Dag<Id, u64, Id>);
//!
//! impl Grow for Lists {
//!     type Cx<'a> = ();
//!
//!     fn grow(&mut self, exhausted: Exhausted, (): ()) {
//!         let hash = |value: &u64, edges: &[Id]| structural_hash((*value, edges));
//!         self.0.maintain(exhausted, 1, |&tail| Some(tail), |_| {}, hash);
//!     }
//! }
//!
//! // The small capacity makes the second list exhaust the arena, so `retry`
//! // runs maintenance and the attempt repeats.
//! let lists = Shared::new(Lists(Dag::with_capacity(1, 1, [])));
//! let cons = |value: u64, edges: &[Id]| {
//!     lists.retry((), |s| {
//!         let hash = structural_hash((value, edges));
//!         let id = s.0.intern(hash, |v, tail| *v == value && tail == edges, || value, edges)?;
//!         // The reference keeps the node across later maintenance.
//!         s.0.inc_ref(id);
//!         Ok(id)
//!     })
//! };
//!
//! let nil = cons(0, &[]);
//! let one = cons(1, &[nil]);
//! assert_eq!(cons(1, &[nil]), one);
//! assert_eq!(*lists.read(()).0.node(one), 1);
//! assert!(lists.epoch() > 0);
//!
//! // Each `cons` call took one reference, so `one` holds two.
//! let s = lists.read(());
//! for id in [one, one, nil] {
//!     s.0.dec_ref(id);
//! }
//! ```

use std::array;
use std::cell::{Cell, UnsafeCell};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Deref, Range};
use std::panic;
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread;

use super::fresh_id;

/// A typed index into a contiguous collection.
///
/// The index of `Self::new(i)` should equal `i`, and an ID should return the
/// same index on every call. The arena checks every access made through an ID,
/// so an implementation that breaks this causes panics or wrong lookups while
/// memory safety holds.
pub trait Idx: Copy + Debug + Eq + Hash + Send + Sync + 'static {
    /// Wraps an index.
    ///
    /// # Panics
    ///
    /// Panics if the index lies outside the type's accepted range.
    fn new(index: usize) -> Self;

    /// The wrapped index.
    fn index(self) -> usize;
}

/// Declares a `u32` newtype implementing [`Idx`].
///
/// An optional `< limit` narrows the accepted range below `u32::MAX`, for
/// domains that reserve the top value as a sentinel.
///
/// ```
/// use sra::util::dag::Idx;
///
/// sra::idx! { pub struct NodeId; }
/// sra::idx! { struct Bounded < 8; }
///
/// assert_eq!(NodeId::new(3).index(), 3);
/// assert_eq!(Bounded::new(7).index(), 7);
/// ```
#[macro_export]
macro_rules! idx {
    ($(#[$m:meta])* $vis:vis struct $name:ident $(< $limit:expr)?;) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        $vis struct $name(u32);

        impl $crate::util::dag::Idx for $name {
            #[inline(always)]
            fn new(index: usize) -> Self {
                assert!(
                    index <= u32::MAX as usize $(&& index < $limit as usize)?,
                    "cannot wrap an index outside the accepted range"
                );
                Self(index as u32)
            }

            #[inline(always)]
            fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

/// The odd multiplier folded into the state for each written word.
const HASH_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
/// The odd multiplier folded into the state once when finishing.
const HASH_FINAL: u64 = 0xd6e8_feb8_6659_fd93;

/// The two words mixed by folding the halves of their 128-bit product.
#[inline(always)]
fn folded_multiply(a: u64, b: u64) -> u64 {
    let product = u128::from(a).wrapping_mul(u128::from(b));
    product as u64 ^ (product >> u64::BITS) as u64
}

/// The running hash state for a structural key: a node's value and edges.
///
/// Every store hashes with this scheme, so [`Dag::intern`] and the `hash`
/// callback of [`Dag::maintain`] agree on equal keys.
#[derive(Debug)]
pub struct StructuralHasher(u64);

impl Default for StructuralHasher {
    /// A fresh state, nonzero so that leading zero words change the hash.
    #[inline(always)]
    fn default() -> Self {
        Self(HASH_MIX)
    }
}

impl Hasher for StructuralHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        folded_multiply(self.0, HASH_FINAL)
    }

    /// Mixes the byte length followed by eight-byte words and a zero-padded final
    /// word.
    ///
    /// The length distinguishes strings and other byte keys with zero-padded tails
    /// or leading zero words.
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.write_usize(bytes.len());
        let (words, tail) = bytes.as_chunks::<8>();
        for &word in words {
            self.write_u64(u64::from_ne_bytes(word));
        }
        if !tail.is_empty() {
            let mut last = [0_u8; 8];
            last[..tail.len()].copy_from_slice(tail);
            self.write_u64(u64::from_ne_bytes(last));
        }
    }

    #[inline(always)]
    fn write_u64(&mut self, value: u64) {
        self.0 = folded_multiply(self.0 ^ value, HASH_MIX);
    }

    #[inline(always)]
    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    #[inline(always)]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
}

/// The hasher factory for tables keyed by structural keys.
///
/// Every table built with it starts each key from the same initial
/// [`StructuralHasher`] state, so [`structural_hash`] reproduces its hashes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Structural;

impl BuildHasher for Structural {
    type Hasher = StructuralHasher;

    #[inline(always)]
    fn build_hasher(&self) -> StructuralHasher {
        StructuralHasher::default()
    }
}

/// A hash map over structural keys, built by [`Structural`].
pub type Map<K, V> = HashMap<K, V, Structural>;

/// A hash set over structural keys, built by [`Structural`].
pub type Set<T> = HashSet<T, Structural>;

/// The hash of `key` under [`StructuralHasher`], as the unique table and
/// structural memo tables compute it.
#[inline(always)]
pub fn structural_hash<H: Hash>(key: H) -> u64 {
    Structural.hash_one(key)
}

/// A cell whose owner tracks initialization and drops the contained value.
#[repr(transparent)]
struct Slot<T>(UnsafeCell<MaybeUninit<T>>);

// SAFETY: shared access to the contained value is available only through unsafe
// methods. Each owner supplies its own initialization, publication, and
// exclusive-write protocol.
unsafe impl<T: Send + Sync> Sync for Slot<T> {}

impl<T> Slot<T> {
    fn new() -> Self {
        Self(UnsafeCell::new(MaybeUninit::uninit()))
    }

    /// # Safety
    ///
    /// The slot must contain an initialized value that remains immutable for the
    /// returned reference's lifetime.
    #[inline(always)]
    unsafe fn get(&self) -> &T {
        // SAFETY: the caller guarantees initialization and immutability for this
        // borrow.
        unsafe { self.0.as_ref_unchecked().assume_init_ref() }
    }

    /// # Safety
    ///
    /// Every slot must contain an initialized value that remains immutable for the
    /// returned borrow.
    #[inline(always)]
    unsafe fn slice(slots: &[Self]) -> &[T] {
        // SAFETY: the representation is transparent over `UnsafeCell<MaybeUninit<T>>`,
        // whose layout is that of `T`, so the cast preserves length and alignment. The
        // standard library guarantees that the contents of an `UnsafeCell` can be
        // accessed without going through `UnsafeCell::get`. An empty slice yields a
        // dangling but aligned pointer, which is valid for a zero-length slice. The
        // caller guarantees initialization and immutability for this borrow.
        unsafe { slice::from_raw_parts(slots.as_ptr().cast::<T>(), slots.len()) }
    }

    /// # Safety
    ///
    /// The slot must contain an initialized value.
    unsafe fn get_mut(&mut self) -> &mut T {
        // SAFETY: the caller guarantees initialization, and this borrow is exclusive.
        unsafe { self.0.get_mut().assume_init_mut() }
    }

    /// # Safety
    ///
    /// The caller must exclusively own this slot.
    #[inline(always)]
    unsafe fn write(&self, value: T) {
        // SAFETY: the caller exclusively owns the slot while initializing it.
        unsafe { self.0.as_mut_unchecked().write(value) };
    }

    /// # Safety
    ///
    /// The slot must contain an initialized value owned exclusively by the caller.
    unsafe fn drop_value(&mut self) {
        // SAFETY: the caller owns the initialized value, and this borrow is exclusive.
        unsafe { self.0.get_mut().assume_init_drop() };
    }
}

/// A node's value and edge range, initialized together.
struct NodeData<N> {
    /// Caller-defined node value, interpreted by the layer using the DAG.
    value: N,
    /// The node's contiguous span in the DAG's separate edge arena, whose capacity
    /// never exceeds `u32::MAX`.
    span: Range<u32>,
}

impl<N> NodeData<N> {
    /// The edge arena positions of this node's span.
    #[inline(always)]
    fn range(&self) -> Range<usize> {
        self.span.start as usize..self.span.end as usize
    }
}

/// An unpublished node slot.
const UNPUBLISHED: u32 = 0;
/// A published node with zero external references.
const UNREFERENCED: u32 = 1;
/// A permanently retained node.
const PINNED: u32 = u32::MAX;

/// One node's publication state and data.
///
/// The slot is written once after its ID is reserved. Readers observe `state`
/// with Acquire ordering before reading the data, and only exclusive access
/// changes a live node's span or unpublishes it.
struct NodeSlot<N> {
    /// The reference state: zero when unpublished, one when published without
    /// external references, one plus the external reference count through
    /// `u32::MAX - 1`, and `u32::MAX` when permanently pinned.
    state: AtomicU32,
    /// The node's value and span, initialized before publication.
    data: Slot<NodeData<N>>,
}

impl<N> NodeSlot<N> {
    fn new() -> Self {
        Self {
            state: AtomicU32::new(UNPUBLISHED),
            data: Slot::new(),
        }
    }

    /// The published node data, or `None` for an unpublished slot.
    ///
    /// The Acquire load of `state` pairs with the Release store in `write`, so the
    /// data it reveals is initialized for this borrow.
    #[inline(always)]
    fn get(&self) -> Option<&NodeData<N>> {
        (self.state.load(Ordering::Acquire) != UNPUBLISHED).then(|| {
            // SAFETY: the Acquire load above observed the Release publication performed
            // after the complete node and edge span were initialized.
            unsafe { self.data.get() }
        })
    }

    /// Whether the slot holds a published node.
    ///
    /// This Relaxed load only observes the state. Reading node data goes through
    /// [`Self::get`], whose Acquire load orders it.
    #[inline(always)]
    fn is_published(&self) -> bool {
        self.state.load(Ordering::Relaxed) != UNPUBLISHED
    }

    /// Whether the published node has external references or is pinned, or `None`
    /// for an unpublished slot.
    #[inline(always)]
    fn has_refs(&self) -> Option<bool> {
        match self.state.load(Ordering::Relaxed) {
            UNPUBLISHED => None,
            state => Some(state > UNREFERENCED),
        }
    }

    /// The published node data, borrowed exclusively.
    ///
    /// # Panics
    ///
    /// Panics if the slot is unpublished.
    fn get_mut(&mut self) -> &mut NodeData<N> {
        assert!(self.is_published(), "cannot use a reclaimed DAG node");

        // SAFETY: the initialized slot is exclusively borrowed while the DAG is
        // mutable.
        unsafe { self.data.get_mut() }
    }

    /// # Safety
    ///
    /// The caller must exclusively own this unpublished slot.
    #[inline(always)]
    unsafe fn write(&self, data: NodeData<N>) {
        debug_assert_eq!(
            self.state.load(Ordering::Relaxed),
            UNPUBLISHED,
            "cannot write a published node slot"
        );
        // SAFETY: the region that claimed this slot, from the packed frontier or from a
        // thread's batch of reclaimed-node entries, gives this writer exclusive
        // ownership of the unpublished slot.
        unsafe { self.data.write(data) };
        self.state.store(UNREFERENCED, Ordering::Release);
    }

    // The `transition`, `pin`, `inc_ref`, and `dec_ref` lifecycle transitions must
    // remain atomic read-modify-write operations. Their relaxed RMWs extend the
    // release sequence established by `write`, so an Acquire `get` that observes a
    // later state also observes the initialized node and edge span.

    /// Applies one reference-state transition.
    ///
    /// The `step` callback returns the new state, or `None` to reject it.
    #[inline(always)]
    fn transition(&self, step: impl Fn(u32) -> Option<u32>, rejected: &str) {
        self.state
            .try_update(Ordering::Relaxed, Ordering::Relaxed, step)
            .expect(rejected);
    }

    /// Pins an unreferenced or already pinned node.
    ///
    /// # Panics
    ///
    /// Panics if the node is unpublished or externally referenced.
    #[inline(always)]
    fn pin(&self) {
        self.transition(
            |state| matches!(state, UNREFERENCED | PINNED).then_some(PINNED),
            "cannot pin a reclaimed or externally referenced DAG node",
        );
    }

    /// Increments the external reference count.
    ///
    /// Pinned nodes remain pinned.
    ///
    /// # Panics
    ///
    /// Panics if the node is unpublished or the count would reach the pinned state.
    #[inline(always)]
    fn inc_ref(&self) {
        self.transition(
            |state| match state {
                UNPUBLISHED => None,
                PINNED => Some(PINNED),
                count => {
                    assert!(count + 1 < PINNED, "reference count overflow");
                    Some(count + 1)
                }
            },
            "cannot reference a reclaimed DAG node",
        );
    }

    /// Decrements the external reference count.
    ///
    /// Pinned nodes remain pinned.
    ///
    /// # Panics
    ///
    /// Panics if the node is unpublished or has no external references.
    #[inline(always)]
    fn dec_ref(&self) {
        self.transition(
            |state| match state {
                UNPUBLISHED | UNREFERENCED => None,
                PINNED => Some(PINNED),
                count => Some(count - 1),
            },
            "cannot decrement an unreferenced or reclaimed DAG node",
        );
    }

    /// Unpublishes the node and drops its data.
    ///
    /// # Panics
    ///
    /// Panics if the node is unpublished, externally referenced, or pinned.
    fn unpublish(&mut self) {
        assert_eq!(
            *self.state.get_mut(),
            UNREFERENCED,
            "cannot unpublish a reclaimed, externally referenced, or pinned DAG node"
        );
        *self.state.get_mut() = UNPUBLISHED;
        // SAFETY: exclusive access excludes readers, and publication proves
        // initialization.
        unsafe { self.data.drop_value() };
    }
}

impl<N> Drop for NodeSlot<N> {
    fn drop(&mut self) {
        if self.is_published() {
            // SAFETY: a published slot is initialized and vector destruction owns it
            // exclusively.
            unsafe { self.data.drop_value() };
        }
    }
}

/// An unused unique-table bucket.
const EMPTY: u64 = 0;
/// Eight buckets per group, occupying one 64-byte cache line.
const GROUP_WIDTH: usize = 8;

/// One cache line of unique-table buckets, probed as a group.
#[repr(align(64))]
struct Group {
    /// Zero, or a nonzero 32-bit hash fingerprint followed by a node ID.
    buckets: [AtomicU64; GROUP_WIDTH],
}

impl Group {
    fn new() -> Self {
        Self {
            buckets: array::from_fn(|_| AtomicU64::new(EMPTY)),
        }
    }
}

const _: () = assert!(size_of::<Group>() == 64, "a group must fill one cache line");

/// Temporary iterator state for one hash's bucket walk.
///
/// The walk visits every bucket of a group, then advances by a hash-dependent
/// group stride. Each item is a group index and a bucket index within it.
struct Probe {
    /// The current group index.
    group: usize,
    /// The group count minus one, for wrapping group indices.
    group_mask: usize,
    /// The hash-derived step between groups, odd when there are multiple groups.
    stride: usize,
    /// The next bucket within the current group.
    offset: usize,
    /// The number of buckets left to probe across the table.
    ///
    /// Multiples of `GROUP_WIDTH` mark group boundaries.
    remaining: usize,
}

impl Probe {
    fn new(hash: u64, buckets: usize) -> Self {
        debug_assert!(
            buckets >= GROUP_WIDTH,
            "a probe must cover at least one whole group"
        );
        debug_assert!(
            buckets.is_power_of_two(),
            "the bucket count must be a power of two"
        );
        let groups = buckets / GROUP_WIDTH;
        let start = hash as usize & (buckets - 1);

        Self {
            group: start / GROUP_WIDTH,
            group_mask: groups - 1,
            stride: ((hash >> u32::BITS) as usize | 1) & (groups - 1),
            offset: start & (GROUP_WIDTH - 1),
            remaining: buckets,
        }
    }
}

impl Iterator for Probe {
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let at = (self.group, self.offset);
        self.offset = (self.offset + 1) & (GROUP_WIDTH - 1);
        self.remaining -= 1;

        if self.remaining & (GROUP_WIDTH - 1) == 0 {
            self.group = (self.group + self.stride) & self.group_mask;
        }

        Some(at)
    }
}

/// The number of entries the table accepts before it is full.
fn buckets_to_capacity(buckets: usize) -> usize {
    buckets / 4 * 3
}

/// The smallest bucket count whose load-factor capacity holds `capacity`
/// entries.
fn capacity_to_buckets(capacity: usize) -> usize {
    let mut buckets = GROUP_WIDTH;
    while buckets_to_capacity(buckets) < capacity {
        buckets = buckets
            .checked_mul(2)
            .expect("unique-table capacity overflow");
    }
    buckets
}

/// The nonzero high half of a hash, stored beside the node ID in a bucket.
#[inline(always)]
fn fingerprint(hash: u64) -> u32 {
    ((hash >> u32::BITS) as u32).max(1)
}

/// One bucket entry: the fingerprint in the high half and the ID in the low.
#[inline(always)]
fn pack<I: Idx>(fingerprint: u32, id: I) -> u64 {
    debug_assert!(
        id.index() <= u32::MAX as usize,
        "unique-table entry overflow"
    );
    let id = id.index() as u32;
    (u64::from(fingerprint) << u32::BITS) | u64::from(id)
}

/// The node ID stored in a bucket entry.
#[inline(always)]
fn unpack<I: Idx>(packed: u64) -> I {
    I::new(packed as u32 as usize)
}

/// The shift that sizes frontier batches from arena capacity.
const BATCH_SHIFT: u32 = 11;
/// The maximum speculative frontier reservation, bounding waste on cache
/// eviction.
const BATCH_MAX: usize = 512;
/// The maximum number of free-list entries claimed at once.
const FREE_BATCH: usize = 64;
/// Region cells per thread, indexed by arena identity.
const REGION_CELLS: usize = 16;

/// The region cell index for an arena identity.
///
/// Fibonacci hashing spreads consecutive identities, which other identity
/// allocations interleave, across the cells.
#[inline(always)]
fn region_index(id: u64) -> usize {
    (id.wrapping_mul(HASH_MIX) >> (u64::BITS - REGION_CELLS.ilog2())) as usize
}

/// The frontier slots one thread claims at once from an arena of `capacity`
/// slots.
#[inline(always)]
fn frontier_batch(capacity: usize) -> usize {
    (capacity >> BATCH_SHIFT).clamp(1, BATCH_MAX)
}

/// Slots one thread has claimed from an arena but not yet published: a batch of
/// free-list entries, a frontier node range, and a frontier edge range.
#[derive(Clone, Copy)]
struct Region {
    /// The identity of the arena the ranges belong to.
    id: u64,
    /// The arena's region generation when the ranges were claimed.
    generation: u64,
    /// The claimed free-list entries, consumed from the end downward.
    free: usize,
    free_end: usize,
    /// The claimed frontier node slots, consumed from the start upward.
    node: usize,
    node_end: usize,
    /// The claimed frontier edge slots, consumed from the start upward.
    edge: usize,
    edge_end: usize,
}

impl Region {
    /// A region with no claimed slots, naming no arena.
    const EMPTY: Self = Self {
        id: u64::MAX,
        generation: 0,
        free: 0,
        free_end: 0,
        node: 0,
        node_end: 0,
        edge: 0,
        edge_end: 0,
    };
}

thread_local! {
    /// This thread's regions, one cell per arena identity hash, as chosen by
    /// [`region_index`].
    static REGIONS: [Cell<Region>; REGION_CELLS] =
        const { [const { Cell::new(Region::EMPTY) }; REGION_CELLS] };
}

/// The node arena was exhausted by a failed attempt.
const GROW_NODES: u8 = 1 << 0;
/// The edge arena was exhausted by a failed attempt.
const GROW_EDGES: u8 = 1 << 1;
/// The unique table was exhausted by a failed attempt.
const GROW_UNIQUE: u8 = 1 << 2;
/// Forces growth when the same retry call fails again after collection.
const GROW_FORCE: u8 = 1 << 3;

/// The free-space percentage at or below which an exhausted arena grows.
const MIN_FREE_PERCENT: u64 = 75;
/// The minimum chunk size, in buckets scanned or entries inserted, of a
/// unique-table rebuild phase.
const REHASH_CHUNK: usize = 4096;

/// The results of `part` over consecutive chunks of `items`, each at least `min`
/// long except a shorter tail, in item order.
///
/// Up to `threads` threads run the chunks, counting the caller, which runs the
/// first.
fn fan_out<T: Sync, U: Send>(
    items: &[T],
    threads: usize,
    min: usize,
    part: &(impl Fn(&[T]) -> U + Sync),
) -> Vec<U> {
    thread::scope(|s| {
        let chunk = items.len().div_ceil(threads).max(min);
        let (first, rest) = items.split_at(chunk.min(items.len()));
        let parts: Vec<_> = rest
            .chunks(chunk)
            .map(|chunk| s.spawn(move || part(chunk)))
            .collect();
        let mut results = vec![part(first)];
        results.extend(parts.into_iter().map(|handle| {
            handle
                .join()
                .unwrap_or_else(|payload| panic::resume_unwind(payload))
        }));
        results
    })
}

/// Whether `free` slots of `capacity` are at or below the growth threshold.
fn low_free_space(free: usize, capacity: usize) -> bool {
    free as u64 * 100 <= capacity as u64 * MIN_FREE_PERCENT
}

/// The minimum additional entries reserved after unique-table exhaustion.
fn growth_headroom(capacity: usize) -> usize {
    let percent = MIN_FREE_PERCENT as usize;
    capacity / 100 * percent + capacity % 100 * percent / 100 + 1
}

/// Grows slot storage geometrically or to the required capacity, whichever is
/// larger, by at least one slot.
///
/// # Panics
///
/// Panics if the resulting capacity exceeds `u32::MAX`.
fn grow_slots<T>(slots: &mut Vec<T>, required: usize, slot: impl FnMut() -> T) {
    let floor = slots.len().checked_add(1).expect("arena capacity overflow");
    let required = required.max(floor);
    assert!(required <= u32::MAX as usize, "arena capacity overflow");
    let capacity = slots
        .len()
        .saturating_mul(2)
        .min(u32::MAX as usize)
        .max(required);
    slots.resize_with(capacity, slot);
}

/// The resource a failed operation exhausted.
///
/// The `Edges` variant carries the contiguous span it needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeedGrowth {
    /// An exhausted node arena, with no slot left for this attempt.
    Nodes,
    /// An exhausted edge arena, with no contiguous span of this many edges.
    Edges(usize),
    /// A full unique table, with no room for a new entry.
    Unique,
}

/// Every resource exhausted by attempts that failed since the last maintenance.
#[derive(Clone, Copy, Debug)]
pub struct Exhausted {
    /// The `GROW_*` bits of every exhausted resource, with `GROW_FORCE` when
    /// growth is forced.
    flags: u8,
    /// The longest edge span a failed attempt could not reserve, or zero.
    edge_len: usize,
}

impl From<NeedGrowth> for Exhausted {
    /// The exhaustion one failed attempt reports.
    fn from(reason: NeedGrowth) -> Self {
        let (flags, edge_len) = match reason {
            NeedGrowth::Nodes => (GROW_NODES, 0),
            NeedGrowth::Edges(edge_len) => (GROW_EDGES, edge_len),
            NeedGrowth::Unique => (GROW_UNIQUE, 0),
        };
        Self { flags, edge_len }
    }
}

impl Exhausted {
    /// The same exhaustion with growth forced.
    ///
    /// A caller owning the DAG passes this when an operation fails again after
    /// maintenance, so the next pass increases capacity even when collection
    /// frees enough space for the threshold check.
    #[inline(always)]
    pub fn forced(self) -> Self {
        Self {
            flags: self.flags | GROW_FORCE,
            ..self
        }
    }
}

/// The packed frontier: the next node ID in the high half and the next edge
/// offset in the low half.
const fn pack_next(node: usize, edge: usize) -> u64 {
    ((node as u64) << u32::BITS) | edge as u64
}

/// The node and edge frontiers of a packed value.
const fn unpack_next(packed: u64) -> (usize, usize) {
    ((packed >> u32::BITS) as usize, packed as u32 as usize)
}

/// A borrowed view of one published node's value and edge span.
///
/// Dereferencing the view also borrows its value. The public value reference
/// retains the DAG borrow's lifetime after the temporary view is gone.
///
/// ```
/// use sra::util::dag::{Dag, Idx};
///
/// fn value<I: Idx, N, E: Copy>(dag: &Dag<I, N, E>, id: I) -> &N {
///     dag.node(id).value
/// }
/// ```
#[derive(Debug)]
pub struct NodeRef<'a, N, E> {
    /// The node's value, borrowed from the DAG for the view's lifetime.
    pub value: &'a N,
    /// The node's edge span, borrowed from the DAG for the view's lifetime.
    pub edges: &'a [E],
}

impl<N, E> Deref for NodeRef<'_, N, E> {
    type Target = N;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.value
    }
}

/// Backing storage for atomically published nodes and their contiguous edge
/// spans, with a unique table for hash-consing.
///
/// Edge values are opaque, and callers enforce any acyclicity requirement.
/// Callers supply equality and hashing, and equal keys must hash equally.
pub struct Dag<I: Idx, N, E: Copy> {
    /// The node slots, indexed by node ID.
    nodes: Vec<NodeSlot<N>>,
    /// The edge slots, addressed through each node's span.
    edges: Vec<Slot<E>>,
    /// The node and edge frontiers, packed by [`pack_next`].
    next: AtomicU64,
    /// The indices of reclaimed nodes, rebuilt by compaction and claimed by shared
    /// publication through `free_len`.
    free_nodes: Vec<u32>,
    /// The length of the free list's unclaimed prefix, from whose end threads claim
    /// entries.
    free_len: AtomicUsize,
    /// Groups of buckets holding a fingerprint and node ID, or zero when empty.
    groups: Vec<Group>,
    /// The number of installed unique-table entries.
    unique_len: AtomicUsize,
    /// Process-unique identity, so a thread's reserved ranges name this arena.
    id: u64,
    /// Region generation, advanced before compaction changes the layout.
    generation: u64,
    /// The typed index naming this arena's nodes.
    marker: PhantomData<I>,
}

impl<I: Idx, N, E: Copy> Dag<I, N, E> {
    /// Creates a DAG with initial node and edge capacities, publishing the `pinned`
    /// leaves permanently in order so they receive the lowest IDs.
    ///
    /// The pinned leaves stay outside the unique table, so interning an equal
    /// value publishes a separate node. Callers name each pinned leaf by its ID.
    ///
    /// # Panics
    ///
    /// Panics if either capacity exceeds `u32::MAX`, the unique table's bucket
    /// count for the node capacity overflows `usize`, the pinned leaves outnumber
    /// the node slots, or the index type rejects a pinned leaf's index.
    pub fn with_capacity(
        node_capacity: usize,
        edge_capacity: usize,
        pinned: impl IntoIterator<Item = N>,
    ) -> Self {
        assert!(node_capacity <= u32::MAX as usize, "node capacity overflow");
        assert!(edge_capacity <= u32::MAX as usize, "edge capacity overflow");
        let buckets = capacity_to_buckets(node_capacity);
        let mut dag = Self {
            nodes: Vec::from_fn(node_capacity, |_| NodeSlot::new()),
            edges: Vec::from_fn(edge_capacity, |_| Slot::new()),
            next: AtomicU64::new(0),
            // Compaction makes reclaimed IDs and unpublished region holes available here.
            // Untouched slots remain beyond the frontier.
            free_nodes: Vec::new(),
            free_len: AtomicUsize::new(0),
            groups: Vec::from_fn(buckets / GROUP_WIDTH, |_| Group::new()),
            unique_len: AtomicUsize::new(0),
            id: fresh_id(),
            generation: 0,
            marker: PhantomData,
        };
        for (i, value) in pinned.into_iter().enumerate() {
            assert!(
                i < node_capacity,
                "pinned leaves must fit the node capacity"
            );
            // Reject an index type that cannot name this leaf before publishing it, as
            // collection would otherwise fail on every later pass.
            I::new(i);
            // SAFETY: construction exclusively owns every unpublished slot. Direct
            // initialization preserves consecutive IDs even when the iterator uses another
            // arena's thread-local region cell.
            unsafe { dag.nodes[i].write(NodeData { value, span: 0..0 }) };
            dag.nodes[i].pin();
            *dag.next.get_mut() = pack_next(i + 1, 0);
        }
        dag
    }

    /// The number of node slots in the arena.
    pub fn node_capacity(&self) -> usize {
        self.nodes.len()
    }

    /// The number of edge slots in the arena.
    pub fn edge_capacity(&self) -> usize {
        self.edges.len()
    }

    /// The number of unique-table entries at which interning reports
    /// [`NeedGrowth::Unique`].
    ///
    /// Concurrent insertions can pass it.
    pub fn unique_capacity(&self) -> usize {
        buckets_to_capacity(self.buckets())
    }

    /// The number of unique-table buckets, a power of two.
    #[inline(always)]
    fn buckets(&self) -> usize {
        self.groups.len() * GROUP_WIDTH
    }

    /// A published node, or `None` for an out-of-bounds or unpublished ID.
    ///
    /// The view borrows the node value and its edge span from separate arenas.
    #[inline(always)]
    pub fn get(&self, id: I) -> Option<NodeRef<'_, N, E>> {
        let data = self.nodes.get(id.index())?.get()?;
        // SAFETY: publication initialized this complete contiguous span before exposing
        // the node, and published edges remain immutable for this borrow.
        let edges = unsafe { Slot::slice(&self.edges[data.range()]) };
        Some(NodeRef {
            value: &data.value,
            edges,
        })
    }

    /// A published node that the caller holds live.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds or its node is unpublished or reclaimed.
    #[inline(always)]
    pub fn node(&self, id: I) -> NodeRef<'_, N, E> {
        self.get(id).expect("cannot use a reclaimed DAG node")
    }

    /// Whether a published node has external references or is permanently pinned.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds or its node is unpublished or reclaimed.
    #[inline(always)]
    pub fn has_refs(&self, id: I) -> bool {
        self.nodes[id.index()]
            .has_refs()
            .expect("cannot inspect a reclaimed DAG node")
    }

    /// Reserves one node slot and a contiguous span of `edge_len` edge slots from
    /// this thread's cached ranges, claiming more when they run out.
    #[inline(always)]
    fn reserve(&self, edge_len: usize) -> Result<(usize, usize), NeedGrowth> {
        let generation = self.generation;
        REGIONS.with(|cells| {
            let cell = &cells[region_index(self.id)];
            let mut r = cell.get();
            if r.id != self.id || r.generation != generation {
                r = Region {
                    id: self.id,
                    generation,
                    ..Region::EMPTY
                };
            }
            let batch = frontier_batch(self.nodes.len());
            // A nonempty frontier range shows that the free list was empty when this
            // generation's region began.
            if r.free == r.free_end && r.node == r.node_end {
                let free_batch = batch.min(FREE_BATCH);
                let claimed = self
                    .free_len
                    .try_update(Ordering::Relaxed, Ordering::Relaxed, |len| {
                        (len > 0).then(|| len.saturating_sub(free_batch))
                    })
                    .unwrap_or(0);
                // Exclusive compaction initialized every entry below `claimed`. The successful
                // decrement assigns these entries to this thread until the next exclusive
                // rebuild.
                r.free_end = claimed;
                r.free = claimed.saturating_sub(free_batch);
            }
            let need_nodes = r.free == r.free_end && r.node == r.node_end;
            let need_edges = r.edge_end - r.edge < edge_len;
            if need_nodes || need_edges {
                // This CAS only reserves existing frontier capacity. Exclusive growth extends
                // the backing vectors while mutable access excludes readers.
                let mut current = self.next.load(Ordering::Relaxed);
                let (node, edge, nodes, edges) = loop {
                    let (node, edge) = unpack_next(current);
                    let nodes = if need_nodes {
                        batch.min(self.nodes.len() - node)
                    } else {
                        0
                    };
                    if need_nodes && nodes == 0 {
                        cell.set(r);
                        return Err(NeedGrowth::Nodes);
                    }
                    let edges = if need_edges {
                        frontier_batch(self.edges.len())
                            .max(edge_len)
                            .min(self.edges.len() - edge)
                    } else {
                        0
                    };
                    if need_edges && edges < edge_len {
                        // Preserve claimed free-list entries until reuse or compaction.
                        cell.set(r);
                        return Err(NeedGrowth::Edges(edge_len));
                    }
                    match self.next.compare_exchange_weak(
                        current,
                        pack_next(node + nodes, edge + edges),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break (node, edge, nodes, edges),
                        Err(observed) => current = observed,
                    }
                };
                if need_nodes {
                    r.node = node;
                    r.node_end = node + nodes;
                }
                if need_edges {
                    r.edge = edge;
                    r.edge_end = edge + edges;
                }
            }
            let i = if r.free == r.free_end {
                r.node += 1;
                r.node - 1
            } else {
                r.free_end -= 1;
                self.free_nodes[r.free_end] as usize
            };
            let edge = r.edge;
            r.edge += edge_len;
            cell.set(r);
            Ok((i, edge))
        })
    }

    /// Publishes a node and its edges without deduplication, returning the physical
    /// node ID.
    ///
    /// The `value` callback runs only once storage has been reserved, so it may
    /// record that the node will be published.
    ///
    /// # Errors
    ///
    /// Returns [`NeedGrowth::Nodes`] or [`NeedGrowth::Edges`] when the
    /// corresponding arena cannot reserve storage.
    ///
    /// # Panics
    ///
    /// Panics if the index type rejects the reserved slot's index.
    #[inline(always)]
    pub fn publish(&self, value: impl FnOnce() -> N, edges: &[E]) -> Result<I, NeedGrowth> {
        let edge_len = edges.len();
        let (i, edge) = self.reserve(edge_len)?;
        let id = I::new(i);
        for (slot, &e) in self.edges[edge..edge + edge_len].iter().zip(edges) {
            // SAFETY: the region exclusively owns the complete edge span.
            unsafe { slot.write(e) };
        }
        // SAFETY: the region exclusively owns this unpublished node slot.
        unsafe {
            self.nodes[i].write(NodeData {
                value: value(),
                span: edge as u32..(edge + edge_len) as u32,
            })
        };
        Ok(id)
    }

    /// Returns an equal published node or atomically publishes and installs one
    /// candidate.
    ///
    /// Nodes that `eq` considers equal must share `hash`. The `eq` callback may run
    /// repeatedly, including after the candidate is published. The `value` callback
    /// runs at most once, when the candidate is published, and `edges` are copied
    /// on publication. A losing candidate stays unreferenced until collection
    /// reclaims it.
    ///
    /// # Errors
    ///
    /// Returns [`NeedGrowth`] when an arena or the unique table is exhausted. An
    /// equal node may already exist when [`NeedGrowth::Unique`] is returned.
    ///
    /// # Panics
    ///
    /// Panics if the index type rejects a reserved slot's index.
    #[inline(always)]
    pub fn intern(
        &self,
        hash: u64,
        mut eq: impl FnMut(&N, &[E]) -> bool,
        value: impl FnOnce() -> N,
        edges: &[E],
    ) -> Result<I, NeedGrowth> {
        let fingerprint = fingerprint(hash);
        let mut found = |packed: u64| {
            (packed >> u32::BITS == u64::from(fingerprint))
                .then(|| unpack(packed))
                .filter(|&id| {
                    let node = self
                        .get(id)
                        .expect("live unique-table entry must remain published");
                    eq(&node, node.edges)
                })
        };
        // The published candidate's ID, or the value callback while none is published.
        let mut candidate = Err(value);

        for (group, offset) in Probe::new(hash, self.buckets()) {
            let bucket = &self.groups[group].buckets[offset];
            let packed = bucket.load(Ordering::Acquire);

            if packed != EMPTY {
                if let Some(id) = found(packed) {
                    return Ok(id);
                }
                continue;
            }

            let id = match candidate {
                Ok(id) => id,
                Err(value) => {
                    if self.unique_len.load(Ordering::Relaxed) >= self.unique_capacity() {
                        return Err(NeedGrowth::Unique);
                    }
                    // Publish the candidate before its ID can become visible through the table.
                    let id = self.publish(value, edges)?;
                    candidate = Ok(id);
                    id
                }
            };

            match bucket.compare_exchange(
                EMPTY,
                pack(fingerprint, id),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.unique_len.fetch_add(1, Ordering::Relaxed);
                    return Ok(id);
                }
                Err(observed) if let Some(other) = found(observed) => return Ok(other),
                Err(_) => {}
            }
        }

        Err(NeedGrowth::Unique)
    }

    /// Permanently pins a published node without external references.
    ///
    /// Repeated pinning is idempotent while the node is unreferenced or already
    /// pinned, including when concurrent interning returns the same canonical node.
    /// Pinning must precede external reference acquisition.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds, its node is unpublished or reclaimed, or
    /// the node has external references. An already pinned node remains pinned.
    #[inline(always)]
    pub fn pin(&self, id: I) {
        self.nodes[id.index()].pin();
    }

    /// Increments the external reference count of a published node.
    ///
    /// Pinned nodes remain pinned.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds, its node is unpublished or reclaimed, or
    /// incrementing the reference count would reach the reserved pinned state.
    #[inline(always)]
    pub fn inc_ref(&self, id: I) {
        self.nodes[id.index()].inc_ref();
    }

    /// Decrements the external reference count of a published node.
    ///
    /// Pinned nodes remain pinned.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds, its node is unpublished or reclaimed, or
    /// an unpinned node has no external references.
    #[inline(always)]
    pub fn dec_ref(&self, id: I) {
        self.nodes[id.index()].dec_ref();
    }

    /// Reclaims nodes unreachable from externally referenced or pinned nodes and
    /// compacts live edges without changing any live node ID.
    ///
    /// Callback contracts are described by [`Self::maintain`]. Reclaimed IDs may be
    /// reused by later publications. If dropping `N` panics, nodes already
    /// reclaimed remain unpublished.
    fn compact(&mut self, target: impl Fn(&E) -> Option<I>, mut reclaim: impl FnMut(&N)) -> usize {
        let (limit, _) = unpack_next(*self.next.get_mut());
        let mut marked = vec![0_u64; limit.div_ceil(64)];
        // Marks node `i`, returning whether it was unmarked.
        let mut mark = |i: usize| {
            let bit = 1_u64 << (i % 64);
            let unmarked = marked[i / 64] & bit == 0;
            marked[i / 64] |= bit;
            unmarked
        };

        let mut pending = Vec::new();
        for (i, slot) in self.nodes[..limit].iter().enumerate() {
            if slot.has_refs() == Some(true) {
                mark(i);
                pending.push(I::new(i));
            }
        }

        while let Some(id) = pending.pop() {
            let node = self.get(id).expect("marked DAG node must remain published");
            for e in node.edges {
                let Some(child) = target(e) else {
                    continue;
                };
                // Each newly marked child is checked by `get` when popped. Marked children are
                // already queued or visited. Exclusive access keeps nodes stable until marking
                // finishes, before any reclamation begins.
                if mark(child.index()) {
                    pending.push(child);
                }
            }
        }

        // Invalidate reserved ranges before changing their free-list entries or edge
        // layout, including when a reclaim callback or node destructor panics.
        self.generation = self
            .generation
            .checked_add(1)
            .expect("reservation generation overflow");
        self.free_nodes.clear();
        *self.free_len.get_mut() = 0;
        let mut reclaimed = 0;
        let mut spans = Vec::new();
        for (i, slot) in self.nodes[..limit].iter_mut().enumerate() {
            if marked[i / 64] & (1_u64 << (i % 64)) != 0 {
                let span = slot.get_mut().range();
                if !span.is_empty() {
                    spans.push((span.start, span.end, i));
                }
                continue;
            }

            if slot.is_published() {
                reclaim(&slot.get_mut().value);
                slot.unpublish();
                reclaimed += 1;
            }
            self.free_nodes.push(i as u32);
        }
        spans.sort_unstable_by_key(|&(start, end, _)| (start, end));

        let mut edge = 0;
        for (start, end, i) in spans {
            let len = end - start;

            if edge != start {
                let slots = self.edges.as_mut_ptr();
                // SAFETY: live spans are initialized, nonoverlapping, and within `self.edges`.
                // Sorting and leftward packing keep the destination in bounds. Exclusive access
                // owns both ranges, `ptr::copy` permits overlap, and `E: Copy` permits
                // duplication.
                unsafe { ptr::copy(slots.add(start), slots.add(edge), len) };
            }

            self.nodes[i].get_mut().span = edge as u32..(edge + len) as u32;
            edge += len;
        }

        *self.free_len.get_mut() = self.free_nodes.len();

        // The node frontier stays at its high-water mark. Compaction only resets the
        // edge frontier. Stale edge values beyond it are ignored and overwritten by
        // later reservations.
        *self.next.get_mut() = pack_next(limit, edge);

        reclaimed
    }

    /// Rebuilds the unique table over its retained entries, with room for
    /// `additional` more and at least `min_buckets` buckets, sharing each phase
    /// with up to `threads` threads.
    ///
    /// # Panics
    ///
    /// Panics if the required capacity exceeds the representation limits.
    fn rebuild_unique(
        &mut self,
        additional: usize,
        min_buckets: usize,
        threads: usize,
        hash: impl Fn(&N, &[E]) -> u64 + Sync,
    ) where
        N: Send + Sync,
        E: Send + Sync,
    {
        let scan = |groups: &[Group]| -> Vec<I> {
            groups
                .iter()
                .flat_map(|group| &group.buckets)
                .map(|bucket| bucket.load(Ordering::Relaxed))
                .filter(|&packed| packed != EMPTY)
                .map(unpack)
                .filter(|&id| self.get(id).is_some())
                .collect()
        };
        let mut parts =
            fan_out(&self.groups, threads, REHASH_CHUNK / GROUP_WIDTH, &scan).into_iter();
        let mut retained = parts.next().expect("fan-out must return the caller's part");
        retained.reserve(parts.as_slice().iter().map(Vec::len).sum());
        retained.extend(parts.flatten());
        let entries = retained.len();
        let required = entries
            .checked_add(additional)
            .expect("unique-table capacity overflow");
        let buckets = capacity_to_buckets(required).max(min_buckets);
        let groups = Vec::from_fn(buckets / GROUP_WIDTH, |_| Group::new());
        let insert = |ids: &[I]| {
            for &id in ids {
                let node = self.node(id);
                let h = hash(&node, node.edges);
                let packed = pack(fingerprint(h), id);
                Probe::new(h, buckets)
                    .find(|&(group, offset)| {
                        groups[group].buckets[offset]
                            .compare_exchange(EMPTY, packed, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                    })
                    .expect("retained unique-table entries must fit during maintenance");
            }
        };
        fan_out(&retained, threads, REHASH_CHUNK, &insert);
        self.groups = groups;
        *self.unique_len.get_mut() = entries;
    }

    /// Collects unreachable nodes and grows exhausted resources when needed.
    ///
    /// Collection retains pinned and externally referenced nodes and everything
    /// reachable from them through `target`, which names an edge's child or `None`
    /// for an opaque edge. It calls `reclaim` on each unreachable node's value
    /// before dropping it and compacts live edges. When the unique table is
    /// exhausted, or holds entries and collection reclaimed a node, it rebuilds the
    /// table with `hash`, which must reproduce the interning hash and may run
    /// concurrently on up to `threads` threads, counting the caller. Forced growth
    /// in `exhausted` increases every exhausted resource's capacity.
    ///
    /// A caller owning the DAG builds `exhausted` from a failed operation's
    /// [`NeedGrowth`] with [`Exhausted::from`], and adds [`Exhausted::forced`]
    /// when the same operation fails again, so a retry loop terminates. The
    /// [`Shared`] wrapper instead accumulates every failure since the last
    /// maintenance.
    ///
    /// If a callback or node destructor panics, the DAG remains memory-safe,
    /// but its contents are unspecified. Callers should discard it.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero, a traced child does not name a published node,
    /// the reservation generation is exhausted, or the required capacity exceeds
    /// the representation limits.
    pub fn maintain(
        &mut self,
        exhausted: Exhausted,
        threads: usize,
        target: impl Fn(&E) -> Option<I>,
        reclaim: impl FnMut(&N),
        hash: impl Fn(&N, &[E]) -> u64 + Sync,
    ) where
        N: Send + Sync,
        E: Send + Sync,
    {
        assert!(threads != 0, "maintenance thread count must be nonzero");
        let forced = exhausted.flags & GROW_FORCE != 0;
        let unique = exhausted.flags & GROW_UNIQUE != 0;
        // Reserve the same minimum free share as the arenas, so the rebuilt table
        // absorbs that share of inserts before it fills again.
        let additional = if unique {
            growth_headroom(self.unique_capacity())
        } else {
            0
        };
        // Retain the table capacity so later inserts have room after collection. Forced
        // growth at least doubles it, exceeding the current capacity even when
        // compaction retains no interned entries.
        let min_buckets = self
            .buckets()
            .checked_mul(if unique && forced { 2 } else { 1 })
            .expect("unique-table capacity overflow");
        let reclaimed = self.compact(target, reclaim);
        if unique || (reclaimed != 0 && *self.unique_len.get_mut() != 0) {
            self.rebuild_unique(additional, min_buckets, threads, hash);
        }

        let (node, edge) = unpack_next(*self.next.get_mut());
        let capacity = self.nodes.len();
        let node_free = capacity - (node - *self.free_len.get_mut());
        if exhausted.flags & GROW_NODES != 0 && (forced || low_free_space(node_free, capacity)) {
            grow_slots(&mut self.nodes, 0, NodeSlot::new);
        }
        let edge_free = self.edges.len() - edge;
        if exhausted.flags & GROW_EDGES != 0
            && (forced
                || edge_free < exhausted.edge_len
                || low_free_space(edge_free, self.edges.len()))
        {
            // The `exhausted.edge_len` value is the complete pending span length.
            let required = edge
                .checked_add(exhausted.edge_len)
                .expect("arena capacity overflow");
            grow_slots(&mut self.edges, required, Slot::new);
        }
    }
}

impl<I: Idx, N, E: Copy> Debug for Dag<I, N, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dag")
            .field("node_capacity", &self.node_capacity())
            .field("edge_capacity", &self.edge_capacity())
            .field("unique_capacity", &self.unique_capacity())
            .finish_non_exhaustive()
    }
}

/// A store's own collection and growth policy, run by the elected writer under
/// the maintenance mutex.
pub trait Grow {
    /// The context the policy needs from the caller, such as a thread budget.
    ///
    /// This is `()` when maintenance needs nothing from outside the store.
    type Cx<'a>: Copy;

    /// Makes room for another attempt through collection or growth.
    ///
    /// The policy handles each exhausted resource in `exhausted`. Collection may
    /// suffice on the first failure. Forced growth must increase every exhausted
    /// resource's capacity to avoid repeating the same failure.
    fn grow(&mut self, exhausted: Exhausted, cx: Self::Cx<'_>);
}

/// The number of admission counters per [`Shared`].
///
/// Threads take counters round-robin, so threads beyond the first `READERS`
/// share one.
const READERS: usize = 64;

/// One admission counter aligned to 128 bytes.
///
/// The alignment keeps each counter on its own cache line, so admissions on
/// different threads avoid contending.
#[repr(align(128))]
struct Counter(AtomicU32);

/// The number of threads that have requested a counter so far.
static NEXT_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread's counter index, or `usize::MAX` before its first admission.
    static COUNTER: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// This thread's admission counter index, assigned on first use.
#[inline(always)]
fn counter_index() -> usize {
    COUNTER.with(|c| {
        let mut i = c.get();
        if i == usize::MAX {
            i = NEXT_COUNTER.fetch_add(1, Ordering::Relaxed) % READERS;
            c.set(i);
        }
        i
    })
}

/// One admitted reader.
///
/// Leaving releases its thread's admission counter.
#[must_use = "dropping the guard ends this thread's admission"]
pub struct Read<'a, S> {
    /// The admitted store, held as a pointer so no reference to it outlives this
    /// guard's release. A reference passed in a function argument, including one
    /// in a by-value guard, must stay valid until that call returns, while the
    /// release inside a call such as `drop` lets a writer take exclusive access.
    store: *const S,
    counter: &'a AtomicU32,
}

// SAFETY: the guard acts as a shared reference to the store and to its counter,
// which any thread may use and release.
unsafe impl<S: Sync> Send for Read<'_, S> {}

// SAFETY: a shared guard only lends shared references to the store.
unsafe impl<S: Sync> Sync for Read<'_, S> {}

impl<S> Deref for Read<'_, S> {
    type Target = S;

    #[inline(always)]
    fn deref(&self) -> &S {
        // SAFETY: admission excludes exclusive access until this guard's drop.
        unsafe { &*self.store }
    }
}

impl<S: Debug> Debug for Read<'_, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<S> Drop for Read<'_, S> {
    #[inline(always)]
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Release);
    }
}

/// One store shared between concurrent attempts and exclusive maintenance.
///
/// Attempts are admitted through [`Self::read`] or run to completion through
/// [`Self::retry`]. A failed attempt reports [`NeedGrowth`], after which one
/// elected writer runs the store's [`Grow`] policy once every admitted reader
/// has left. Admission blocks while maintenance is pending.
pub struct Shared<S> {
    /// Completed maintenance passes, used to invalidate caller-held memo tables.
    epoch: AtomicU64,
    /// Process-unique storage incarnation for caller-held memo tables.
    ///
    /// Exclusive access through [`Self::get_mut`] starts a new incarnation.
    pub id: u64,
    /// The resources failed attempts exhausted, nonzero while maintenance is
    /// pending and new admission is blocked.
    requested: AtomicU8,
    /// Largest contiguous edge span requested by a failed attempt.
    edge_len: AtomicU32,
    /// The mutex electing one maintenance writer to wait for the readers to drain.
    maintenance: Mutex<()>,
    /// The admission counters, indexed by [`counter_index`].
    readers: Box<[Counter; READERS]>,
    /// The store, borrowed by admitted readers or exclusively by the writer.
    store: UnsafeCell<S>,
}

// SAFETY: shared access to `store` is only through admitted readers, and
// exclusive access only once admission is blocked and every reader has left,
// under the `maintenance` mutex.
unsafe impl<S: Send + Sync> Sync for Shared<S> {}

impl<S: Grow> Shared<S> {
    /// Wraps a store with a fresh process-unique identity and no admitted readers.
    pub fn new(store: S) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            id: fresh_id(),
            requested: AtomicU8::new(0),
            edge_len: AtomicU32::new(0),
            maintenance: Mutex::new(()),
            readers: Box::new(array::from_fn(|_| Counter(AtomicU32::new(0)))),
            store: UnsafeCell::new(store),
        }
    }

    /// The maintenance epoch: the number of completed maintenance passes,
    /// including collection without growth.
    ///
    /// Callers compare it and [`Self::id`] with their memo table's recorded values
    /// while admitted, and clear stale entries before use.
    #[inline(always)]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Runs the store's maintenance policy once every admitted reader has left, if
    /// maintenance is still pending.
    fn finish_maintenance(&self, cx: S::Cx<'_>) {
        // Only the mutex holder waits for the readers. Other contenders wait here
        // instead.
        let _maintenance = self
            .maintenance
            .lock()
            .expect("maintenance must not have panicked");
        // This load joins the SeqCst admission order before the counter drains. A
        // reader admitted before the request must appear in those drains or have left.
        // A reader ordered after it observes the closed admission gate on its SeqCst
        // recheck before borrowing the store.
        if self.requested.load(Ordering::SeqCst) == 0 {
            return;
        }

        // No flag is lost between this load and the reset below: an attempt publishes
        // its flags while it is still admitted, and `read` admits no attempt while
        // `requested` is nonzero, so once every counter has drained every failed
        // attempt has already published.
        for counter in self.readers.iter() {
            while counter.0.load(Ordering::SeqCst) != 0 {
                thread::yield_now();
            }
        }
        // SAFETY: admission is blocked, every admitted reader has left, and
        // `maintenance` excludes other writers.
        unsafe { self.store.as_mut_unchecked() }.grow(
            Exhausted {
                flags: self.requested.load(Ordering::Acquire),
                edge_len: self.edge_len.load(Ordering::Acquire) as usize,
            },
            cx,
        );
        self.edge_len.store(0, Ordering::Relaxed);
        // Published before admission reopens, so no attempt observes a stale count and
        // keeps a memo table over the nodes maintenance reclaimed.
        self.epoch.fetch_add(1, Ordering::Release);
        self.requested.store(0, Ordering::Release);
    }

    /// Admits one shared attempt once no maintenance is pending, running pending
    /// maintenance itself with `cx` when needed.
    ///
    /// A thread must drop its [`Read`] before admitting another attempt to this
    /// store, or it can deadlock against maintenance requested meanwhile. Nested
    /// admissions to different stores must follow a consistent total order across
    /// threads, so pending maintenance cannot create a wait cycle.
    ///
    /// # Panics
    ///
    /// Panics if this thread's shared admission counter would exceed `u32::MAX`, or
    /// if an earlier [`Grow::grow`] call panicked and poisoned the maintenance
    /// mutex.
    #[inline(always)]
    pub fn read(&self, cx: S::Cx<'_>) -> Read<'_, S> {
        let counter = &self.readers[counter_index()].0;
        loop {
            if self.requested.load(Ordering::Acquire) != 0 {
                self.finish_maintenance(cx);
                continue;
            }

            // A request may race with this admission. SeqCst on both sides orders this
            // increment against the request and the writer's drain check against each
            // other, so either this attempt sees the request or the writer sees this
            // reader.
            counter
                .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_add(1))
                .expect("reader count overflow");
            if self.requested.load(Ordering::SeqCst) == 0 {
                return Read {
                    store: self.store.get(),
                    counter,
                };
            }

            // Leave this admission so the elected writer can reach quiescence.
            counter.fetch_sub(1, Ordering::Release);
        }
    }

    /// Runs `attempt` until it succeeds, maintaining the store with `cx` after each
    /// [`NeedGrowth`] failure.
    ///
    /// Each attempt is admitted as by [`Self::read`], so the same nesting rules
    /// apply.
    ///
    /// # Panics
    ///
    /// Panics if this thread's shared admission counter would exceed `u32::MAX`, if
    /// an attempt requests an edge span larger than `u32::MAX`, or if an earlier
    /// [`Grow::grow`] call panicked and poisoned the maintenance mutex.
    pub fn retry<T>(
        &self,
        cx: S::Cx<'_>,
        mut attempt: impl FnMut(&S) -> Result<T, NeedGrowth>,
    ) -> T {
        // Let the first failure try collection alone, but force growth if this call
        // fails again.
        let mut force = 0;
        loop {
            let store = self.read(cx);
            let reason = match attempt(&store) {
                Ok(value) => return value,
                Err(reason) => reason,
            };
            // Publish the reason before leaving. The writer sees every accumulated
            // reason after all admitted attempts have drained.
            if let NeedGrowth::Edges(len) = reason {
                let span = u32::try_from(len).expect("edge span overflow");
                self.edge_len.fetch_max(span, Ordering::AcqRel);
            }
            let flags = Exhausted::from(reason).flags | force;
            force = GROW_FORCE;
            self.requested.fetch_or(flags, Ordering::SeqCst);
            drop(store);
            self.finish_maintenance(cx);
        }
    }

    /// Refreshes [`Self::id`] and borrows the store exclusively without admission.
    ///
    /// The refresh precedes mutable access, invalidating caller-held memo tables
    /// even if the caller replaces the entire store.
    pub fn get_mut(&mut self) -> &mut S {
        self.id = fresh_id();
        self.store.get_mut()
    }
}

impl<S> Debug for Shared<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared")
            .field("id", &self.id)
            .field("epoch", &self.epoch.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/dag.rs"]
mod tests;
