//! Canonical maps represented by their numeric transition points.
//!
//! A map stores its payload at zero and sorted `(value, payload)` transitions.
//! Each transition changes the payload until the next transition, and the
//! final run ends at the exact logical universe boundary. Adjacent equal runs
//! coalesce. Singletons and large intervals use the same representation,
//! without enumerating a finite or cofinite side. Partition slots are numbered
//! by first numeric occurrence. Only this backend interprets transitions; query
//! execution uses the common exact field algebra. The [`SparseTable`] type
//! implements [`Backend`], binding each sort to the layout its width selects.

use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::ControlFlow;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, Values, transfer,
};
use crate::scalar::{ScalarKind, ScalarSort, check_packed, compare, successor};
use crate::util::dag::{Dag, Exhausted, Grow, Map, NeedGrowth, NodeRef, Shared, structural_hash};

/// Initial arena sizes, large enough that collection alone recovers most
/// transient results, so the arenas rarely grow and every collection pays its
/// fixed costs.
const NODE_CAPACITY: usize = 65_536;
const EDGE_CAPACITY: usize = 262_144;

#[derive(Debug)]
#[repr(C)]
struct Node {
    /// The payload at zero, before any transition.
    default: u32,
    /// Words per transition, from the layout the map's sort selects.
    stride: u32,
}

/// How a sort lays its transitions out in edge words, chosen by the sort
/// alone.
///
/// A sort of at most 32 bits packs each transition's key above its slot
/// in one word, so an identifier column costs eight bytes per transition; a
/// wider sort stores the key words followed by one slot word. Every reader and
/// writer of edge words goes through the layout, and the stride is interned
/// with the node, so maps of the two layouts never compare equal.
trait Layout: Copy + Send + Sync + 'static {
    /// One transition's key: the packed word's high half, or a borrowed
    /// key-word slice.
    type Key<'a>: Copy;
    fn stride(self) -> usize;
    fn key(self, entry: &[u64]) -> Self::Key<'_>;
    fn slot(self, entry: &[u64]) -> u32;
    /// A key from its row encoding of `sort.words()` little-endian words.
    fn decode(self, words: &[u64]) -> Self::Key<'_>;
    /// A key's row encoding, appended to `out`.
    fn encode(self, key: Self::Key<'_>, out: &mut Vec<u64>);
    fn compare(self, a: Self::Key<'_>, b: Self::Key<'_>) -> Ordering;
    fn is_zero(self, key: Self::Key<'_>) -> bool;
    /// Whether `b` is the value after `a`.
    fn follows(self, a: Self::Key<'_>, b: Self::Key<'_>) -> bool;
    fn push(self, out: &mut Vec<u64>, key: Self::Key<'_>, slot: u32);
    fn set_slot(self, entry: &mut [u64], slot: u32);
}

#[derive(Clone, Copy)]
struct Packed;

impl Layout for Packed {
    type Key<'a> = u64;
    #[inline(always)]
    fn stride(self) -> usize {
        1
    }
    #[inline(always)]
    fn key(self, entry: &[u64]) -> u64 {
        entry[0] >> 32
    }
    #[inline(always)]
    fn slot(self, entry: &[u64]) -> u32 {
        entry[0] as u32
    }
    #[inline(always)]
    fn decode(self, words: &[u64]) -> u64 {
        words[0]
    }
    #[inline(always)]
    fn encode(self, key: u64, out: &mut Vec<u64>) {
        out.push(key);
    }
    #[inline(always)]
    fn compare(self, a: u64, b: u64) -> Ordering {
        a.cmp(&b)
    }
    #[inline(always)]
    fn is_zero(self, key: u64) -> bool {
        key == 0
    }
    #[inline(always)]
    fn follows(self, a: u64, b: u64) -> bool {
        a + 1 == b
    }
    #[inline(always)]
    fn push(self, out: &mut Vec<u64>, key: u64, slot: u32) {
        out.push(key << 32 | u64::from(slot));
    }
    #[inline(always)]
    fn set_slot(self, entry: &mut [u64], slot: u32) {
        entry[0] = entry[0] & !0xFFFF_FFFF | u64::from(slot);
    }
}

/// Key words followed by one slot word.
///
/// The const parameter is the key width in words, so every copy and comparison
/// gets it as a constant, or zero to carry it at run time.
#[derive(Clone, Copy)]
struct Wide<const W: usize>(usize);

impl<const W: usize> Wide<W> {
    #[inline(always)]
    fn words(self) -> usize {
        if W == 0 { self.0 } else { W }
    }
}

impl<const W: usize> Layout for Wide<W> {
    type Key<'a> = &'a [u64];
    #[inline(always)]
    fn stride(self) -> usize {
        self.words() + 1
    }
    #[inline(always)]
    fn key(self, entry: &[u64]) -> &[u64] {
        &entry[..self.words()]
    }
    #[inline(always)]
    fn slot(self, entry: &[u64]) -> u32 {
        entry[self.words()] as u32
    }
    #[inline(always)]
    fn decode(self, words: &[u64]) -> &[u64] {
        words
    }
    #[inline(always)]
    fn encode(self, key: &[u64], out: &mut Vec<u64>) {
        out.extend_from_slice(key);
    }
    #[inline(always)]
    fn compare(self, a: &[u64], b: &[u64]) -> Ordering {
        compare(a, b)
    }
    #[inline(always)]
    fn is_zero(self, key: &[u64]) -> bool {
        key.iter().all(|&word| word == 0)
    }
    #[inline(always)]
    fn follows(self, a: &[u64], b: &[u64]) -> bool {
        let mut carry = true;
        a.iter().zip(b).all(|(&lo, &hi)| {
            let (next, overflow) = lo.overflowing_add(u64::from(carry));
            carry = overflow;
            next == hi
        }) && !carry
    }
    #[inline(always)]
    fn push(self, out: &mut Vec<u64>, key: &[u64], slot: u32) {
        out.extend_from_slice(key);
        out.push(u64::from(slot));
    }
    #[inline(always)]
    fn set_slot(self, entry: &mut [u64], slot: u32) {
        entry[self.words()] = u64::from(slot);
    }
}

#[inline(always)]
fn entries<'a, L: Layout>(
    l: L,
    node: &NodeRef<'a, Node, u64>,
) -> impl Iterator<Item = (L::Key<'a>, u32)> {
    node.edges
        .chunks_exact(l.stride())
        .map(move |entry| (l.key(entry), l.slot(entry)))
}

/// Binary search for `key` among a node's transitions: the index where it is
/// or would be.
#[inline(always)]
fn search<L: Layout>(l: L, edges: &[u64], key: L::Key<'_>) -> Result<usize, usize> {
    let stride = l.stride();
    let (mut low, mut high) = (0, edges.len() / stride);
    while low < high {
        let mid = low + (high - low) / 2;
        match l.compare(l.key(&edges[mid * stride..]), key) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Ok(mid),
        }
    }
    Err(low)
}

/// The payload after the first `i` of a node's transitions.
#[inline(always)]
fn run<L: Layout>(l: L, default: u32, edges: &[u64], i: usize) -> u32 {
    if i == 0 {
        default
    } else {
        l.slot(&edges[(i - 1) * l.stride()..])
    }
}

/// Appends a change in numeric order.
///
/// A second change at the same point replaces the first. This joins adjacent
/// point inputs without manufacturing a gap between them.
#[inline(always)]
fn change<L: Layout>(l: L, default: &mut u32, entries: &mut Vec<u64>, key: L::Key<'_>, value: u32) {
    if l.is_zero(key) {
        *default = value;
        return;
    }
    let stride = l.stride();
    if entries.len() >= stride
        && l.compare(l.key(&entries[entries.len() - stride..]), key) == Ordering::Equal
    {
        entries.truncate(entries.len() - stride);
    }
    if run(l, *default, entries, entries.len() / stride) != value {
        l.push(entries, key, value);
    }
}

/// Sweeps the union of transition points, retaining each map's current
/// payload between events.
fn merge_entries<'a, L: Layout>(
    l: L,
    left: NodeRef<'a, Node, u64>,
    right: NodeRef<'a, Node, u64>,
    mut emit: impl FnMut(L::Key<'a>, u32, u32),
) {
    let (mut lv, mut rv) = (left.default, right.default);
    let mut a = entries(l, &left).peekable();
    let mut b = entries(l, &right).peekable();
    while let (Some(&(x, p)), Some(&(y, q))) = (a.peek(), b.peek()) {
        match l.compare(x, y) {
            Ordering::Less => {
                lv = p;
                emit(x, lv, rv);
                a.next();
            }
            Ordering::Equal => {
                lv = p;
                rv = q;
                emit(x, lv, rv);
                a.next();
                b.next();
            }
            Ordering::Greater => {
                rv = q;
                emit(y, lv, rv);
                b.next();
            }
        }
    }
    for (key, value) in a {
        emit(key, value, rv);
    }
    for (key, value) in b {
        emit(key, lv, value);
    }
}

/// Converts sorted point assignments into runs, including zero and the last
/// domain value.
///
/// Returns the run at zero. The `out` buffer receives the rest, and `next`
/// holds one domain value.
#[inline(always)]
fn points<L: Layout>(
    l: L,
    sort: ScalarSort,
    default: u32,
    keys: &[u64],
    labels: impl IntoIterator<Item = u32>,
    out: &mut Vec<u64>,
    next: &mut Vec<u64>,
) -> u32 {
    let words = sort.words();
    let mut base = default;
    out.clear();
    next.clear();
    next.resize(words, 0);
    for (key, label) in keys.chunks_exact(words).zip(labels) {
        change(l, &mut base, out, l.decode(key), label);
        next.copy_from_slice(key);
        if successor(next, sort) {
            change(l, &mut base, out, l.decode(next), default);
        }
    }
    base
}

/// Returns the sink that assigns canonical slots to labeled runs.
///
/// Canonical slots follow the first run carrying each original label. There
/// is always a run at zero, so the initial slot is zero and no universe
/// cardinality or gap enumeration is needed. Appends each reached label to
/// `slot_labels` once; the runs in `entries` carry slots indexing that
/// appended suffix. The sink receives the labeled runs in numeric order.
fn partition<'s, L: Layout, T: Copy + Eq + Hash>(
    l: L,
    default: T,
    entries: &'s mut Vec<u64>,
    slots: &'s mut Map<T, u32>,
    slot_labels: &'s mut Vec<T>,
) -> impl FnMut(L::Key<'_>, T) + 's {
    let start = slot_labels.len();
    slot_labels.push(default);
    slots.clear();
    slots.insert(default, 0);
    entries.clear();
    move |key: L::Key<'_>, label: T| {
        let next = (slot_labels.len() - start) as u32;
        let slot = *slots.entry(label).or_insert(next);
        if slot == next {
            slot_labels.push(label);
        }
        change(l, &mut 0, entries, key, slot);
    }
}

/// Physical storage and canonical namespace for transition maps.
type State = Dag<Function, Node, u64>;

impl State {
    /// Interns one externally referenced map.
    ///
    /// Nonzero transition keys are strictly increasing. Each payload differs
    /// from the previous run's payload.
    fn intern_referenced(
        &self,
        l: impl Layout,
        default: u32,
        entries: &[u64],
    ) -> Result<Function, NeedGrowth> {
        let f = if entries.is_empty() {
            assert!(default <= 1, "constant sparse map must be Boolean");
            Function::new(default as usize)
        } else {
            let stride = l.stride() as u32;
            self.intern(
                structural_hash((default, stride, entries)),
                |node, edges| node.default == default && node.stride == stride && edges == entries,
                || Node { default, stride },
                entries,
            )?
        };
        self.inc_ref(f);
        Ok(f)
    }
}

impl Grow for State {
    type Cx<'a> = usize;

    fn grow(&mut self, exhausted: Exhausted, threads: usize) {
        self.maintain(
            exhausted,
            threads,
            |_| None,
            |_| {},
            |node, edges| structural_hash((node.default, node.stride, edges)),
        );
    }
}

/// A concurrent canonical table of transition maps over each sort's numeric
/// encoding.
///
/// Roots are [`Function`] tokens with manual external reference counts, which
/// the context pairs with this table through a column id. Columns of one
/// table share its root namespace; the layout interned with each map keeps
/// sorts of different word widths apart.
pub struct SparseTable {
    shared: Shared<State>,
    threads: usize,
}

impl SparseTable {
    /// An empty table containing the Boolean constants, with initial node and
    /// edge-word capacities and a maintenance thread budget.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero.
    pub fn with_capacity(nodes: usize, edges: usize, threads: usize) -> Self {
        assert!(threads != 0, "maintenance thread count must be nonzero");
        let state = Dag::with_capacity(
            nodes,
            edges,
            (0..2).map(|default| Node { default, stride: 0 }),
        );
        Self {
            shared: Shared::new(state),
            threads,
        }
    }
}

impl Default for SparseTable {
    /// Creates an empty table containing the canonical Boolean constants.
    fn default() -> Self {
        Self::with_capacity(NODE_CAPACITY, EDGE_CAPACITY, 1)
    }
}

impl Backend for SparseTable {
    /// Transition points encode every scalar sort at its exact word width.
    ///
    /// A sort of at most 32 bits packs each transition in one word, a two-word
    /// sort fixes its key width statically, and any other sort carries its key
    /// width at run time.
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        fn bind<L: Layout>(store: Arc<SparseTable>, sort: ScalarSort, l: L) -> Box<dyn Column> {
            Box::new(SparseColumn { store, sort, l })
        }
        Some(match sort.words() {
            _ if sort.width() <= 32 => bind(self, sort, Packed),
            2 => bind(self, sort, Wide::<2>(2)),
            words => bind(self, sort, Wide::<0>(words)),
        })
    }
}

/// A cursor over the members of a sparse map.
///
/// The cursor retains offsets rather than pointers into a table that can grow.
/// It jumps across excluded runs, and increments across included runs using
/// the complete logical width.
struct SparseMembers<'a, L> {
    store: &'a SparseTable,
    l: L,
    root: Function,
    sort: ScalarSort,
    entry: usize,
    value: Option<Vec<u64>>,
}

impl<L: Layout> Members for SparseMembers<'_, L> {
    fn next(&mut self, out: &mut Vec<u64>) -> bool {
        let state = self.store.shared.read(self.store.threads);
        let node = state.node(self.root);
        let (l, stride) = (self.l, self.l.stride());
        while let Some(value) = self.value.as_mut() {
            while self.entry < node.edges.len()
                && l.compare(l.key(&node.edges[self.entry..]), l.decode(value)) != Ordering::Greater
            {
                self.entry += stride;
            }
            if run(l, node.default, node.edges, self.entry / stride) == 0 {
                if self.entry == node.edges.len() {
                    self.value = None;
                    return false;
                }
                value.clear();
                l.encode(l.key(&node.edges[self.entry..]), value);
                continue;
            }
            out.clear();
            out.extend_from_slice(value);
            if !successor(value, self.sort) {
                self.value = None;
            }
            return true;
        }
        false
    }
}

impl<L> Drop for SparseMembers<'_, L> {
    fn drop(&mut self) {
        self.store
            .shared
            .read(self.store.threads)
            .dec_ref(self.root);
    }
}

/// A sparse table bound to one scalar sort, with the transition layout its
/// width selects.
struct SparseColumn<L> {
    store: Arc<SparseTable>,
    sort: ScalarSort,
    l: L,
}

impl<L: Layout> SparseColumn<L> {
    /// The images of `p` under `map` within `q`, or with `image` false, the
    /// members of `p` whose image lies in `q`.
    fn transfer(&self, map: &MapSpec, p: Function, q: Function, image: bool) -> Function {
        map.check_domain(self.sort);
        if p.is_f() || q.is_f() || map.is_identity() {
            return self.apply_bool(BoolOp::And, p, q);
        }
        let values = transfer(
            self.sort,
            map,
            &mut *self.members(p),
            |y| self.evaluate(q, y) != 0,
            image,
        );
        self.set(&values)
    }
}

impl<L: Layout> Column for SparseColumn<L> {
    /// The bound scalar sort.
    fn sort(&self) -> ScalarSort {
        self.sort
    }

    /// Acquires one external reference to a live root.
    ///
    /// A root with no external references remains live until collection.
    /// Reclamation is deferred to growth.
    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.shared.read(self.store.threads).inc_ref(f);
        }
    }

    /// Releases one external reference to a root.
    fn dec_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.shared.read(self.store.threads).dec_ref(f);
        }
    }

    /// Releases one external reference to each function under one guard.
    fn dec_refs(&self, fs: &[Function]) {
        let state = self.store.shared.read(self.store.threads);
        for &f in fs {
            if !f.is_constant() {
                state.dec_ref(f);
            }
        }
    }

    /// The payload of the run holding `value`.
    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        let l = self.l;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
        let node = state.node(f);
        let i = search(l, node.edges, l.decode(value)).map_or_else(|i| i, |i| i + 1);
        run(l, node.default, node.edges, i)
    }

    /// The number of members of predicate `p`, summed over its runs.
    ///
    /// A run holding to the universe boundary counts to the sort's full width,
    /// so a cofinite symbol predicate counts exactly.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a Boolean predicate.
    fn count(&self, p: Function) -> BigUint {
        if p.is_f() {
            return BigUint::ZERO;
        }
        let l = self.l;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(state.has_refs(p), "cannot use an unreferenced sparse node");
        let node = state.node(p);
        // Payloads alternate, so a transition to 1 starts a run and a transition
        // to 0 ends one. Each run adds its end and subtracts its start in two's
        // complement over one word above the sort's words, which holds the full
        // universe count.
        let words = self.sort.words();
        let mut sum = vec![0_u64; words + 1];
        let add = |sum: &mut [u64], key: &[u64], negate: bool| {
            let flip = if negate { u64::MAX } else { 0 };
            let mut carry = negate;
            for (i, word) in sum.iter_mut().enumerate() {
                let (a, x) = word.overflowing_add(key.get(i).copied().unwrap_or(0) ^ flip);
                let (b, y) = a.overflowing_add(u64::from(carry));
                *word = b;
                carry = x | y;
            }
        };
        let mut label = node.default;
        assert!(label <= 1, "count requires a Boolean predicate");
        let mut key_words = Vec::with_capacity(words);
        for (key, next) in entries(l, &node) {
            assert!(next <= 1, "count requires a Boolean predicate");
            key_words.clear();
            l.encode(key, &mut key_words);
            add(&mut sum, &key_words, next == 1);
            label = next;
        }
        if label == 1 {
            let width = self.sort.width() as usize;
            key_words.clear();
            key_words.resize(width / 64 + 1, 0);
            key_words[width / 64] = 1 << (width % 64);
            add(&mut sum, &key_words, false);
        }
        BigUint::new(
            sum.iter()
                .flat_map(|&v| [v as u32, (v >> 32) as u32])
                .collect(),
        )
    }

    /// The predicate holding exactly at the sorted, distinct `values`, with
    /// adjacent points coalesced into runs.
    fn set(&self, values: &[u64]) -> Function {
        check_packed(self.sort, values);
        let l = self.l;
        let (mut entries, mut next) = (Vec::new(), Vec::new());
        let default = points(
            l,
            self.sort,
            0,
            values,
            std::iter::repeat(1),
            &mut entries,
            &mut next,
        );
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, default, &entries)
        })
    }

    /// The predicate holding where the value is below `bound`: one run from
    /// zero to the bound.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering.
    fn less_than(&self, bound: &[u64]) -> Function {
        assert_ne!(
            self.sort.kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        let l = self.l;
        let mut default = 1;
        let mut entries = Vec::new();
        change(l, &mut default, &mut entries, l.decode(bound), 0);
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, default, &entries)
        })
    }

    /// The Boolean combination of two predicates, merging the runs of both.
    ///
    /// The `AndNot` operation is directional: it computes `left and !right`.
    fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> Function {
        let l = self.l;
        self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced sparse node"
            );
            if let Some(result) = op.simplify(left, right) {
                state.inc_ref(result);
                return Ok(result);
            }
            let (a, b) = (state.node(left), state.node(right));
            let mut default = u32::from(op.evaluate_bool(a.default != 0, b.default != 0));
            let mut entries = Vec::new();
            merge_entries(l, a, b, |key, x, y| {
                let value = u32::from(op.evaluate_bool(x != 0, y != 0));
                change(l, &mut default, &mut entries, key, value);
            });
            state.intern_referenced(l, default, &entries)
        })
    }

    /// The images of `p` under `map` within `q`.
    ///
    /// Empty endpoint predicates and identity maps use Boolean intersection.
    /// Every other map kind applies to the enumerated members of `p`. Nothing
    /// bounds that enumeration: a predicate holding over most of a wide sort
    /// transfers correctly but takes proportionally long.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.transfer(map, p, q, true)
    }

    /// The members of `p` whose image under `map` lies in `q`.
    ///
    /// Uses the same empty/identity shortcuts and enumeration as [`Self::image`].
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort.
    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.transfer(map, p, q, false)
    }

    /// The conditional of three predicates.
    ///
    /// Trivial conditionals retain an existing root. Others sweep the union of
    /// the operands' transition points.
    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        let l = self.l;
        self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(
                state.has_refs(condition) && state.has_refs(yes) && state.has_refs(no),
                "cannot use an unreferenced sparse node"
            );
            if let Some(f) = match condition {
                Function::TRUE => Some(yes),
                Function::FALSE => Some(no),
                _ if yes == no => Some(yes),
                _ if yes.is_t() && no.is_f() => Some(condition),
                _ => None,
            } {
                state.inc_ref(f);
                return Ok(f);
            }
            let nodes = [condition, yes, no].map(|f| state.node(f));
            let mut values = nodes.each_ref().map(|node| node.default);
            let mut default = if values[0] != 0 { values[1] } else { values[2] };
            let mut out = Vec::new();
            let mut runs = nodes.each_ref().map(|node| entries(l, node).peekable());
            while let Some(key) = runs
                .iter_mut()
                .filter_map(|run| run.peek().map(|&(key, _)| key))
                .min_by(|&a, &b| l.compare(a, b))
            {
                for (run, value) in runs.iter_mut().zip(&mut values) {
                    if run
                        .peek()
                        .is_some_and(|&(next, _)| l.compare(next, key) == Ordering::Equal)
                    {
                        *value = run.next().unwrap().1;
                    }
                }
                let value = if values[0] != 0 { values[1] } else { values[2] };
                change(l, &mut default, &mut out, key, value);
            }
            state.intern_referenced(l, default, &out)
        })
    }

    /// The common refinement of two distinct, nontrivial partitions.
    ///
    /// Common refinement appends each reachable source-slot pair exactly once.
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
    ) -> Function {
        let l = self.l;
        let Alignment { entries, pairs, .. } = scratch;
        let start = labels.len();
        self.store.shared.retry(self.store.threads, |state| {
            // A retried attempt appends its labels afresh.
            labels.truncate(start);
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced sparse node"
            );
            let (a, b) = (state.node(left), state.node(right));
            let mut emit = partition(l, (a.default, b.default), entries, pairs, labels);
            merge_entries(l, a, b, |key, x, y| emit(key, (x, y)));
            drop(emit);
            state.intern_referenced(l, 0, entries)
        })
    }

    /// The coarsening of `f` by `slots`.
    ///
    /// Coarsening preserves first-occurrence slot numbering and merges adjacent
    /// equal runs. The quotient slots already carry canonical labels, so only
    /// transition coalescing is needed.
    fn quotient(&self, f: Function, slots: &[u32], scratch: &mut Alignment) -> Function {
        let l = self.l;
        let out = &mut scratch.entries;
        self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
            let node = state.node(f);
            let mut default = slots[node.default as usize];
            out.clear();
            for (key, slot) in entries(l, &node) {
                change(l, &mut default, out, key, slots[slot as usize]);
            }
            state.intern_referenced(l, default, out)
        })
    }

    /// The partition labeling each case predicate's members by its label.
    ///
    /// Matching case predicates must agree on their label. Boundary events
    /// update the active labels together, so adjacent cases with different
    /// labels do not overlap.
    ///
    /// # Panics
    ///
    /// Panics if overlapping cases carry different labels.
    fn cases(
        &self,
        default: u32,
        cases: &[(Function, u32)],
        slot_labels: &mut Vec<u32>,
    ) -> Function {
        let l = self.l;
        let start = slot_labels.len();
        debug_assert!(
            {
                let state = self.store.shared.read(self.store.threads);
                cases.iter().all(|&(p, _)| state.has_refs(p))
            },
            "cannot use an unreferenced sparse node"
        );
        // Case inputs are Boolean predicates. A single nonconstant case that is
        // false at zero already has slots 0 then 1 in first-occurrence order.
        // Retaining it reproduces the canonical partition without scanning its
        // transitions, sorting, or interning a second copy.
        if let &[(p, label)] = cases
            && label != default
        {
            let state = self.store.shared.read(self.store.threads);
            let node = state.node(p);
            if node.default == 0 && !node.edges.is_empty() {
                slot_labels.extend([default, label]);
                state.inc_ref(p);
                return p;
            }
        }
        self.store.shared.retry(self.store.threads, |state| {
            // A retried attempt appends its labels afresh.
            slot_labels.truncate(start);
            let mut active = Map::<u32, usize>::default();
            let mut events = Vec::new();
            for &(p, label) in cases {
                let node = state.node(p);
                if node.default != 0 {
                    *active.entry(label).or_default() += 1;
                }
                events.extend(entries(l, &node).map(|(key, value)| (key, label, value != 0)));
            }
            let value = |counts: &Map<u32, usize>| {
                assert!(counts.len() <= 1, "cases with different labels overlap");
                counts.keys().next().copied().unwrap_or(default)
            };
            let initial = value(&active);
            events.sort_unstable_by(|a, b| l.compare(a.0, b.0));
            let changes = events
                .chunk_by(|a, b| l.compare(a.0, b.0) == Ordering::Equal)
                .map(|run| {
                    for &(_, label, holds) in run {
                        if holds {
                            *active.entry(label).or_default() += 1;
                        } else {
                            let count = active.get_mut(&label).unwrap();
                            *count -= 1;
                            if *count == 0 {
                                active.remove(&label);
                            }
                        }
                    }
                    (run[0].0, value(&active))
                });
            let (mut entries, mut slots) = (Vec::new(), Map::default());
            let mut emit = partition(l, initial, &mut entries, &mut slots, slot_labels);
            for (key, label) in changes {
                emit(key, label);
            }
            drop(emit);
            state.intern_referenced(l, 0, &entries)
        })
    }

    /// The partition labeling sorted keys by their labels.
    ///
    /// Point-labeled construction retains its public contract; only the
    /// internal map uses runs.
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        slot_labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Function {
        check_packed(self.sort, keys);
        let l = self.l;
        let Values {
            transitions,
            next,
            slots,
        } = scratch;
        let labels = case_labels.iter().copied();
        let initial = points(l, self.sort, default, keys, labels, transitions, next);
        // Canonical slots follow the first run carrying each label, as in
        // `partition`. Relabeling is injective, so the runs `points` merged stay
        // merged and the entries rewrite in place.
        let start = slot_labels.len();
        slot_labels.push(initial);
        slots.clear();
        slots.insert(initial, 0);
        for entry in transitions.chunks_exact_mut(l.stride()) {
            let label = l.slot(entry);
            let fresh = (slot_labels.len() - start) as u32;
            let slot = *slots.entry(label).or_insert(fresh);
            if slot == fresh {
                slot_labels.push(label);
            }
            l.set_slot(entry, slot);
        }
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, 0, transitions)
        })
    }

    /// Groups accepted runs by partition slot.
    ///
    /// The visitor runs outside the store guard, and a slot predicate is interned
    /// only when its thunk asks for it. All accepted runs are grouped and buffered
    /// before the first callback, so `Break` cannot avoid that work for unused
    /// slots.
    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refinement,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if p.is_f() {
            return ControlFlow::Continue(());
        }
        let l = self.l;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(
            state.has_refs(f) && state.has_refs(p),
            "cannot use an unreferenced sparse node"
        );
        let (a, b) = (state.node(f), state.node(p));
        let stride = l.stride();
        // A Boolean map's runs alternate, so two transitions after a false default
        // hold exactly between them.
        if b.default == 0
            && b.edges.len() == 2 * stride
            && l.follows(l.key(b.edges), l.key(&b.edges[stride..]))
        {
            // One concrete point reaches exactly one slot; its entire cell is the
            // borrowed p.
            let i = search(l, a.edges, l.key(b.edges)).map_or_else(|i| i, |i| i + 1);
            let slot = run(l, a.default, a.edges, i);
            drop(state);
            if !select(slot) {
                return ControlFlow::Continue(());
            }
            return then(slot, &mut || {
                self.inc_ref(p);
                p
            });
        }
        let n = a.edges.len() / stride;
        // A map whose payloads are already Boolean is the indicator of its own
        // slot 1, so every cell of its unrestricted refinement is `f` or the
        // payload flip of `f`. Discovering that by sweeping would rebuild both
        // cells and re-intern `f` back into itself. Every interned map's default
        // is Boolean: canonical partitions number the run at zero as slot 0.
        let mut other = false;
        let identity = p.is_t()
            && a.edges.chunks_exact(stride).all(|entry| {
                let slot = l.slot(entry);
                other |= slot != a.default;
                slot <= 1
            });
        // Slot numbering is dense. The scratch table grows to each reached slot,
        // independently of how many transitions the partition has.
        scratch.restart();
        let mut prior = (b.default != 0).then_some(a.default);
        if identity {
            scratch.group(a.default, 1, select);
            if other {
                scratch.group(1 - a.default, 0, select);
            }
        } else if let Some(slot) = prior {
            scratch.group(slot, 1, select);
        }
        // A declined slot costs its selection once; the sweep then drops its
        // boundaries, which on an index level is every gap between keys that the
        // predicate covers.
        let mut write = |slot: u32, key: L::Key<'_>, holds: u32| {
            let Some(i) = scratch.group(slot, 0, select) else {
                return;
            };
            let (_, _, entries) = &mut scratch.groups[i];
            // The Boolean interval sweep toggles each group at strictly increasing
            // nonzero keys.
            l.push(entries, key, holds);
        };
        // Consecutive events always change state: coalesced runs alternate slots
        // within an interval, and each interval's end clears the state before the
        // next one starts.
        let mut emit = |key: L::Key<'_>, slot, holds| {
            let next = (holds != 0).then_some(slot);
            if let Some(s) = prior {
                write(s, key, 0);
            }
            if let Some(s) = next {
                write(s, key, 1);
            }
            prior = next;
        };
        // Bound each accepted interval before visiting its transitions. Galloping
        // from the previous cursor skips rejected gaps without rescanning them or
        // charging a comparison against the predicate boundary to every accepted
        // transition. The layout compares a packed key as an integer without
        // forming a slice.
        let cmp = |i: usize, key: L::Key<'_>| l.compare(l.key(&a.edges[i * stride..]), key);
        let seek = |i: usize, key: L::Key<'_>, order: Ordering| {
            // The transitions before the answer are exactly those ordered below the
            // bound.
            let before = |j: usize| cmp(j, key) < order;
            if i == n || !before(i) {
                return i;
            }
            let (mut lo, hi) = (i + 1, n);
            // Gallop from the current lower bound to bracket the answer, then
            // binary-search that bracket. Nearby answers require only a short
            // doubling scan before the binary search.
            let (mut step, mut end) = (0, hi);
            while lo + step < hi {
                if !before(lo + step) {
                    end = lo + step;
                    break;
                }
                lo += step + 1;
                step = step * 2 + 1;
            }
            while lo < end {
                let mid = lo + (end - lo) / 2;
                if before(mid) {
                    lo = mid + 1;
                } else {
                    end = mid;
                }
            }
            lo
        };
        let mut i = 0;
        // None denotes the initial/final universe boundary without an encoded
        // zero allocation.
        let mut sweep = |lo: Option<L::Key<'_>>, hi: Option<L::Key<'_>>| {
            if let Some(lo) = lo {
                i = seek(i, lo, Ordering::Greater);
            }
            let end = hi.map_or(n, |hi| seek(i, hi, Ordering::Equal));
            let mut slot = run(l, a.default, a.edges, i);
            if let Some(lo) = lo {
                emit(lo, slot, 1);
            }
            for entry in a.edges[i * stride..end * stride].chunks_exact(stride) {
                slot = l.slot(entry);
                emit(l.key(entry), slot, 1);
            }
            i = end;
            if let Some(key) = hi {
                emit(key, slot, 0);
            }
        };
        if !identity {
            let mut lo = None;
            let mut holds = b.default != 0;
            for (key, value) in entries(l, &b) {
                if holds {
                    sweep(lo, Some(key));
                }
                lo = Some(key);
                holds = value != 0;
            }
            if holds {
                sweep(lo, None);
            }
        }
        drop(state);
        scratch.finish();
        // Only one reached slot means p itself is that cell.
        let (live, single) = (scratch.live, scratch.reached == 1);
        for group in scratch.groups.iter_mut().take(live) {
            let (slot, default, entries) = (group.0, group.1, &mut group.2);
            then(slot, &mut || {
                if single {
                    self.inc_ref(p);
                    return p;
                }
                if identity && slot == 1 {
                    self.inc_ref(f);
                    return f;
                }
                self.store.shared.retry(self.store.threads, |state| {
                    if identity {
                        // Flip the payloads of `f` under the guard that interns them,
                        // so a relocation between attempts rereads the relocated
                        // edges.
                        let node = state.node(f);
                        entries.clear();
                        for entry in node.edges.chunks_exact(stride) {
                            l.push(entries, l.key(entry), 1 - l.slot(entry));
                        }
                    }
                    state.intern_referenced(l, default, entries)
                })
            })?;
        }
        ControlFlow::Continue(())
    }

    /// A cursor over the members of `p`, holding its own reference to the root
    /// until dropped.
    ///
    /// It jumps across excluded runs and steps through included runs one
    /// value at a time, so a cofinite symbol predicate enumerates nearly the
    /// whole 64-bit universe.
    fn members(&self, p: Function) -> Box<dyn Members + '_> {
        debug_assert!(
            self.store.shared.read(self.store.threads).has_refs(p),
            "cannot use an unreferenced sparse node"
        );
        self.inc_ref(p);
        Box::new(SparseMembers {
            store: &self.store,
            l: self.l,
            root: p,
            sort: self.sort,
            entry: 0,
            value: Some(vec![0; self.sort.words()]),
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/sparse.rs"]
mod tests;
