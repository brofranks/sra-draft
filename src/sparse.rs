//! Canonical maps represented by their numeric transition points.
//!
//! A map stores its payload at zero and sorted `(value, payload)` transitions.
//! Each transition changes the payload until the next transition, and the
//! final run ends at the exact logical universe boundary. Adjacent equal runs
//! coalesce. Singletons and large intervals use the same representation,
//! without enumerating their members. Partition classes are numbered by first
//! occurrence in increasing unsigned encoding order. [`SparseStore`] implements
//! [`Backend`], binding each sort to the layout its width selects.

use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::backend::{
    self, Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, Values,
};
use crate::scalar::{self, ScalarKind, ScalarSort};
use crate::util::count::Count;
use crate::util::dag::{self, Dag, Exhausted, Grow, Map, NeedGrowth, NodeRef, Shared};

/// Initial arena sizes, large enough that collection alone recovers most
/// transient results, so the arenas rarely grow and every collection pays its
/// fixed costs.
const NODE_CAPACITY: usize = 65_536;
const EDGE_CAPACITY: usize = 262_144;

#[derive(Debug)]
#[repr(C)]
struct Node {
    /// The payload at zero, before any transition.
    initial: u32,
    /// Words per transition, from the layout the map's sort selects.
    stride: u32,
}

/// How a sort lays its transitions out in edge words, chosen by the sort
/// alone.
///
/// A sort of at most 32 bits packs each key above its payload in one word,
/// using eight bytes per transition. Wider sorts store the key words followed
/// by a payload word. Every reader and writer of edge words uses the layout.
/// The interned stride distinguishes maps of different layouts.
trait Layout: Copy + Send + Sync + 'static {
    /// One transition's key: the packed word's high half, or a borrowed
    /// key-word slice.
    type Key<'a>: Copy;
    fn stride(self) -> usize;
    fn key(self, entry: &[u64]) -> Self::Key<'_>;
    fn payload(self, entry: &[u64]) -> u32;
    /// A key from its row encoding of `sort.words()` little-endian words.
    fn decode(self, words: &[u64]) -> Self::Key<'_>;
    /// A key's row encoding, appended to `out`.
    fn encode(self, key: Self::Key<'_>, out: &mut Vec<u64>);
    fn compare(self, a: Self::Key<'_>, b: Self::Key<'_>) -> Ordering;
    fn is_zero(self, key: Self::Key<'_>) -> bool;
    /// Whether `b` is the value after `a`.
    fn follows(self, a: Self::Key<'_>, b: Self::Key<'_>) -> bool;
    fn push(self, out: &mut Vec<u64>, key: Self::Key<'_>, payload: u32);
    fn set_payload(self, entry: &mut [u64], payload: u32);
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
    fn payload(self, entry: &[u64]) -> u32 {
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
    fn push(self, out: &mut Vec<u64>, key: u64, payload: u32) {
        out.push(key << 32 | u64::from(payload));
    }
    #[inline(always)]
    fn set_payload(self, entry: &mut [u64], payload: u32) {
        entry[0] = entry[0] & !0xFFFF_FFFF | u64::from(payload);
    }
}

/// Key words followed by one payload word.
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
    fn payload(self, entry: &[u64]) -> u32 {
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
        scalar::compare(a, b)
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
    fn push(self, out: &mut Vec<u64>, key: &[u64], payload: u32) {
        out.extend_from_slice(key);
        out.push(u64::from(payload));
    }
    #[inline(always)]
    fn set_payload(self, entry: &mut [u64], payload: u32) {
        entry[self.words()] = u64::from(payload);
    }
}

#[inline(always)]
fn entries<'a, L: Layout>(
    l: L,
    node: &NodeRef<'a, Node, u64>,
) -> impl Iterator<Item = (L::Key<'a>, u32)> {
    node.edges
        .chunks_exact(l.stride())
        .map(move |entry| (l.key(entry), l.payload(entry)))
}

/// The number of transitions with keys at or below `key`, by binary search.
#[inline(always)]
fn search<L: Layout>(l: L, edges: &[u64], key: L::Key<'_>) -> usize {
    let stride = l.stride();
    let (mut low, mut high) = (0, edges.len() / stride);
    while low < high {
        let mid = low + (high - low) / 2;
        match l.compare(l.key(&edges[mid * stride..]), key) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return mid + 1,
        }
    }
    low
}

/// The payload after the first `i` of a node's transitions.
#[inline(always)]
fn run_payload<L: Layout>(l: L, initial: u32, edges: &[u64], i: usize) -> u32 {
    if i == 0 {
        initial
    } else {
        l.payload(&edges[(i - 1) * l.stride()..])
    }
}

/// Appends a change in numeric order.
///
/// A second change at the same point replaces the first. This joins adjacent
/// point inputs without manufacturing a gap between them.
#[inline(always)]
fn change<L: Layout>(l: L, initial: &mut u32, entries: &mut Vec<u64>, key: L::Key<'_>, value: u32) {
    if l.is_zero(key) {
        *initial = value;
        return;
    }
    let stride = l.stride();
    if entries.len() >= stride
        && l.compare(l.key(&entries[entries.len() - stride..]), key) == Ordering::Equal
    {
        entries.truncate(entries.len() - stride);
    }
    if run_payload(l, *initial, entries, entries.len() / stride) != value {
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
    let (mut lv, mut rv) = (left.initial, right.initial);
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
    let mut initial = default;
    out.clear();
    next.clear();
    next.resize(words, 0);
    for (key, label) in keys.chunks_exact(words).zip(labels) {
        change(l, &mut initial, out, l.decode(key), label);
        next.copy_from_slice(key);
        if scalar::successor(next, sort) {
            change(l, &mut initial, out, l.decode(next), default);
        }
    }
    initial
}

/// Returns the sink that assigns canonical classes to labeled runs.
///
/// Classes follow the first run carrying each label. The initial run gets
/// class zero without counting or enumerating any of the domain. The sink
/// receives labeled runs in numeric order and appends each reached label once.
/// The class numbers in `entries` index the appended suffix of `labels`.
fn partition<'s, L: Layout, T: Copy + Eq + Hash>(
    l: L,
    initial: T,
    entries: &'s mut Vec<u64>,
    classes: &'s mut Map<T, u32>,
    labels: &'s mut Vec<T>,
) -> impl FnMut(L::Key<'_>, T) + 's {
    let start = labels.len();
    labels.push(initial);
    classes.clear();
    classes.insert(initial, 0);
    entries.clear();
    move |key: L::Key<'_>, label: T| {
        let next = (labels.len() - start) as u32;
        let class = *classes.entry(label).or_insert(next);
        if class == next {
            labels.push(label);
        }
        change(l, &mut 0, entries, key, class);
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
        initial: u32,
        entries: &[u64],
    ) -> Result<Function, NeedGrowth> {
        let f = if entries.is_empty() {
            assert!(initial <= 1, "constant sparse map must be Boolean");
            Function::new(initial as usize)
        } else {
            let stride = l.stride() as u32;
            self.intern(
                dag::structural_hash((initial, stride, entries)),
                |node, edges| node.initial == initial && node.stride == stride && edges == entries,
                || Node { initial, stride },
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
            |node, edges| dag::structural_hash((node.initial, node.stride, edges)),
        );
    }
}

/// A concurrent canonical table of transition maps over each sort's numeric
/// encoding.
///
/// Roots are [`Function`] tokens with manual external reference counts.
/// Equal maps share a token within one store and sort. A live root can be
/// used through any column bound to that store and sort.
///
/// Numeric cuts operate on Boolean and bit-vector sorts. The backend rejects
/// [`Column::masked_eq`]. Transfers enumerate the source predicate's members
/// except for empty endpoints, identity maps, and bit-map preimages with a
/// full target. Enumeration time grows with the number of source members.
#[derive(Debug)]
pub struct SparseStore {
    shared: Shared<State>,
    threads: usize,
}

impl SparseStore {
    /// An empty table containing the Boolean constants, with initial node and
    /// edge-word capacities and a maintenance thread budget.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero, `nodes` is below two, either capacity
    /// exceeds `u32::MAX`, or the unique table's bucket count for the node
    /// capacity overflows `usize`.
    pub fn with_capacity(nodes: usize, edges: usize, threads: usize) -> Self {
        assert!(threads != 0, "maintenance thread count must be nonzero");
        let state = Dag::with_capacity(
            nodes,
            edges,
            (0..2).map(|initial| Node { initial, stride: 0 }),
        );
        Self {
            shared: Shared::new(state),
            threads,
        }
    }
}

impl Default for SparseStore {
    /// Creates an empty table containing the canonical Boolean constants.
    fn default() -> Self {
        Self::with_capacity(NODE_CAPACITY, EDGE_CAPACITY, 1)
    }
}

impl Backend for SparseStore {
    /// Transition points encode every scalar sort at its exact word width.
    ///
    /// A sort of at most 32 bits packs each transition in one word, a two-word
    /// sort fixes its key width statically, and any other sort carries its key
    /// width at run time.
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        fn bind<L: Layout>(store: Arc<SparseStore>, sort: ScalarSort, l: L) -> Box<dyn Column> {
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
struct SparseMembers<'a, L: Layout> {
    column: &'a SparseColumn<L>,
    root: Function,
    offset: usize,
    /// The current words, empty before the first advance and absent after
    /// exhaustion.
    value: Option<Vec<u64>>,
}

impl<L: Layout> Members for SparseMembers<'_, L> {
    fn next(&mut self) -> Option<&[u64]> {
        let col = self.column;
        let (sort, l) = (col.sort, col.l);
        let value = self.value.as_mut()?;
        if value.is_empty() {
            value.resize(sort.words(), 0);
        } else if !scalar::successor(value, sort) {
            self.value = None;
            return None;
        }
        let state = col.store.shared.read(col.store.threads);
        let node = state.node(self.root);
        let stride = l.stride();
        while self.offset < node.edges.len()
            && l.compare(l.key(&node.edges[self.offset..]), l.decode(value)) != Ordering::Greater
        {
            self.offset += stride;
        }
        if run_payload(l, node.initial, node.edges, self.offset / stride) == 0 {
            if self.offset == node.edges.len() {
                self.value = None;
                return None;
            }
            // The transition ending this excluded run opens an included one.
            value.clear();
            l.encode(l.key(&node.edges[self.offset..]), value);
            self.offset += stride;
        }
        drop(state);
        self.value.as_deref()
    }
}

impl<L: Layout> Drop for SparseMembers<'_, L> {
    fn drop(&mut self) {
        self.column.dec_ref(self.root);
    }
}

/// A sparse table bound to one scalar sort, with the transition layout its
/// width selects.
struct SparseColumn<L> {
    store: Arc<SparseStore>,
    sort: ScalarSort,
    l: L,
}

impl<L: Layout> SparseColumn<L> {
    /// The images of `p` under `map` within `q`, or with `image` false, the
    /// members of `p` whose image lies in `q`.
    fn transfer(&self, map: &MapSpec, p: Function, q: Function, image: bool) -> Function {
        map.check_domain(self.sort);
        if p.is_f()
            || q.is_f()
            || (!image && q.is_t() && matches!(map, MapSpec::Bits { .. }))
            || map.is_identity()
        {
            return self.apply_bool(BoolOp::And, p, q);
        }
        debug_assert!(
            self.store.shared.read(self.store.threads).has_refs(q),
            "cannot use an unreferenced sparse node"
        );
        let values = backend::transfer(
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
    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.shared.read(self.store.threads).inc_ref(f);
        }
    }

    /// Releases one external reference to a root.
    ///
    /// Maintenance can reclaim the map once its last reference is released.
    fn dec_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.shared.read(self.store.threads).dec_ref(f);
        }
    }

    /// The payload of the run holding `value`.
    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        assert!(self.sort.accepts(value), "value is outside the column sort");
        let l = self.l;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
        let node = state.node(f);
        let i = search(l, node.edges, l.decode(value));
        run_payload(l, node.initial, node.edges, i)
    }

    /// The number of members of predicate `p`, summed over its runs.
    ///
    /// A run holding to the universe boundary counts to the sort's full width,
    /// so a cofinite symbol predicate counts exactly.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a Boolean predicate.
    fn count(&self, p: Function) -> Count {
        if p.is_f() {
            return Count::ZERO;
        }
        if p.is_t() {
            return Count::from(1) << self.sort.width();
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
        let mut value = node.initial;
        assert!(value <= 1, "count requires a Boolean predicate");
        let mut key_words = Vec::with_capacity(words);
        for (key, next) in entries(l, &node) {
            assert!(next <= 1, "count requires a Boolean predicate");
            key_words.clear();
            l.encode(key, &mut key_words);
            add(&mut sum, &key_words, next == 1);
            value = next;
        }
        if value == 1 {
            let width = self.sort.width() as usize;
            key_words.clear();
            key_words.resize(width / 64 + 1, 0);
            key_words[width / 64] = 1 << (width % 64);
            add(&mut sum, &key_words, false);
        }
        Count::from_digits(sum)
    }

    /// The predicate holding exactly at the sorted, distinct `values`, with
    /// adjacent points coalesced into runs.
    fn set(&self, values: &[u64]) -> Function {
        scalar::check_packed(self.sort, values);
        if values.is_empty() {
            return Function::FALSE;
        }
        let l = self.l;
        let (mut entries, mut next) = (Vec::new(), Vec::new());
        let initial = points(
            l,
            self.sort,
            0,
            values,
            std::iter::repeat(1),
            &mut entries,
            &mut next,
        );
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, initial, &entries)
        })
    }

    /// The predicate holding where the value is below `bound`: one run from
    /// zero to the bound.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering, or
    /// `bound` is outside the sort.
    fn less_than(&self, bound: &[u64]) -> Function {
        assert_ne!(
            self.sort.kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        assert!(self.sort.accepts(bound), "bound is outside the column sort");
        let l = self.l;
        let mut initial = 1;
        let mut entries = Vec::new();
        change(l, &mut initial, &mut entries, l.decode(bound), 0);
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, initial, &entries)
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
            let mut initial = u32::from(op.evaluate_bool(a.initial != 0, b.initial != 0));
            let mut entries = Vec::new();
            merge_entries(l, a, b, |key, x, y| {
                let value = u32::from(op.evaluate_bool(x != 0, y != 0));
                change(l, &mut initial, &mut entries, key, value);
            });
            state.intern_referenced(l, initial, &entries)
        })
    }

    /// The images of `p` under `map` within `q`.
    ///
    /// Empty endpoint predicates and identity maps use Boolean intersection.
    /// Other valid maps apply to the enumerated members of `p`.
    ///
    /// # Panics
    ///
    /// Panics if the map or a defined Rust map output is outside the sort.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.transfer(map, p, q, true)
    }

    /// The members of `p` whose image under `map` lies in `q`.
    ///
    /// Also uses intersection for bit maps with a full target because every
    /// source member has a defined image.
    ///
    /// # Panics
    ///
    /// Panics if the map or a defined Rust map output is outside the sort.
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
            let mut values = nodes.each_ref().map(|node| node.initial);
            let mut initial = if values[0] != 0 { values[1] } else { values[2] };
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
                change(l, &mut initial, &mut out, key, value);
            }
            state.intern_referenced(l, initial, &out)
        })
    }

    /// The common refinement of two distinct, nontrivial partitions.
    ///
    /// Common refinement appends each reachable pair of input classes once.
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
    ) -> Function {
        let l = self.l;
        let Alignment {
            transitions, pairs, ..
        } = scratch;
        let start = labels.len();
        self.store.shared.retry(self.store.threads, |state| {
            // A retried attempt appends its labels afresh.
            labels.truncate(start);
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced sparse node"
            );
            let (a, b) = (state.node(left), state.node(right));
            let mut emit = partition(l, (a.initial, b.initial), transitions, pairs, labels);
            merge_entries(l, a, b, |key, x, y| emit(key, (x, y)));
            drop(emit);
            state.intern_referenced(l, 0, transitions)
        })
    }

    /// The coarsening of `f` by `classes`.
    ///
    /// The class map already specifies canonical result numbers. Coalescing
    /// adjacent runs that map to the same class preserves first-occurrence
    /// numbering.
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Alignment) -> Function {
        let l = self.l;
        let out = &mut scratch.transitions;
        self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
            let node = state.node(f);
            let mut initial = classes[node.initial as usize];
            out.clear();
            for (key, class) in entries(l, &node) {
                change(l, &mut initial, out, key, classes[class as usize]);
            }
            state.intern_referenced(l, initial, out)
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
    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        let l = self.l;
        let start = labels.len();
        debug_assert!(
            {
                let state = self.store.shared.read(self.store.threads);
                cases.iter().all(|&(p, _)| state.has_refs(p))
            },
            "cannot use an unreferenced sparse node"
        );
        // Case inputs are Boolean predicates. A single nonconstant case that is
        // false at zero already has classes 0 then 1 in first-occurrence order.
        // Retaining it reproduces the canonical partition without scanning its
        // transitions, sorting, or interning a second copy.
        if let &[(p, label)] = cases
            && label != default
        {
            let state = self.store.shared.read(self.store.threads);
            let node = state.node(p);
            if node.initial == 0 && !node.edges.is_empty() {
                labels.extend([default, label]);
                state.inc_ref(p);
                return p;
            }
        }
        self.store.shared.retry(self.store.threads, |state| {
            // A retried attempt appends its labels afresh.
            labels.truncate(start);
            let mut active = Map::<u32, usize>::default();
            let mut events = Vec::new();
            for &(p, label) in cases {
                let node = state.node(p);
                if node.initial != 0 {
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
            let (mut entries, mut classes) = (Vec::new(), Map::default());
            let mut emit = partition(l, initial, &mut entries, &mut classes, labels);
            for (key, label) in changes {
                emit(key, label);
            }
            drop(emit);
            state.intern_referenced(l, 0, &entries)
        })
    }

    /// The partition labeling sorted keys by their labels.
    ///
    /// Coalesces adjacent keys with the same label into one run.
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Function {
        scalar::check_packed(self.sort, keys);
        assert_eq!(
            case_labels.len(),
            keys.len() / self.sort.words(),
            "there must be one case label per key"
        );
        if keys.is_empty() {
            labels.push(default);
            return Function::FALSE;
        }
        let l = self.l;
        let Values {
            transitions,
            next,
            classes,
        } = scratch;
        let initial = points(
            l,
            self.sort,
            default,
            keys,
            case_labels.iter().copied(),
            transitions,
            next,
        );
        // Canonical classes follow the first run carrying each label, as in
        // `partition`. Relabeling is injective, so the runs `points` merged stay
        // merged and the entries rewrite in place.
        let start = labels.len();
        labels.push(initial);
        classes.clear();
        classes.insert(initial, 0);
        for entry in transitions.chunks_exact_mut(l.stride()) {
            let label = l.payload(entry);
            let fresh = (labels.len() - start) as u32;
            let class = *classes.entry(label).or_insert(fresh);
            if class == fresh {
                labels.push(label);
            }
            l.set_payload(entry, class);
        }
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(l, 0, transitions)
        })
    }

    /// Groups accepted runs by partition class.
    ///
    /// The general sweep calls `select` under the store guard and buffers the
    /// accepted runs before calling `then`. The guard is released for delivery.
    /// Cell predicates are built on demand, so `Break` skips later visits and
    /// their builds.
    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refinement,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if p.is_f() {
            debug_assert!(
                self.store.shared.read(self.store.threads).has_refs(f),
                "cannot use an unreferenced sparse node"
            );
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
        // A Boolean map's runs alternate, so two transitions from a false
        // initial payload bound the entire included run.
        if b.initial == 0
            && b.edges.len() == 2 * stride
            && l.follows(l.key(b.edges), l.key(&b.edges[stride..]))
        {
            // One concrete point reaches one class, whose cell is the input p.
            let i = search(l, a.edges, l.key(b.edges));
            let class = run_payload(l, a.initial, a.edges, i);
            drop(state);
            if !select(class) {
                return ControlFlow::Continue(());
            }
            return then(class, &mut || {
                self.inc_ref(p);
                p
            });
        }
        let n = a.edges.len() / stride;
        // A Boolean map indicates its own class 1. Its unrestricted refinement
        // uses f and its payload flip directly, avoiding a sweep and re-interning
        // f. The initial payload is Boolean because partitions start at class 0.
        let identity = p.is_t()
            && a.edges
                .chunks_exact(stride)
                .all(|entry| l.payload(entry) <= 1);
        // The dense lookup grows only to the classes reached by this refinement,
        // independently of the number of partition transitions.
        let groups = &mut scratch.groups;
        groups.restart();
        let mut holds = b.initial != 0;
        if holds {
            groups.group(a.initial, 1, select);
        }
        if identity && n != 0 {
            groups.group(1 - a.initial, 0, select);
        }
        // Selection is asked once per class. Declined boundaries are discarded.
        // Accepted intervals toggle each group at increasing nonzero keys.
        let mut write = |class, key: L::Key<'_>, value| {
            if let Some(transitions) = groups.group(class, 0, select) {
                l.push(transitions, key, value);
            }
        };
        // Bound each accepted interval before visiting its transitions. Galloping
        // from the previous cursor skips rejected gaps without rescanning them or
        // charging a comparison against the predicate boundary to every accepted
        // transition. The layout compares a packed key as an integer without
        // forming a slice.
        let cmp = |i: usize, key: L::Key<'_>| l.compare(l.key(&a.edges[i * stride..]), key);
        let seek = |i: usize, key: L::Key<'_>, order: Ordering| {
            // The lower bound consumes equal transitions. The upper bound
            // leaves them out of the accepted interval.
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
            let mut class = run_payload(l, a.initial, a.edges, i);
            if let Some(lo) = lo {
                write(class, lo, 1);
            }
            for entry in a.edges[i * stride..end * stride].chunks_exact(stride) {
                let key = l.key(entry);
                write(class, key, 0);
                class = l.payload(entry);
                write(class, key, 1);
            }
            i = end;
            if let Some(key) = hi {
                write(class, key, 0);
            }
        };
        if !identity {
            let mut lo = None;
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
        // If the predicate reaches one class, it is that class's cell.
        let single = groups.reached() == 1;
        for (class, initial, transitions) in groups.selected() {
            then(class, &mut || {
                if single {
                    self.inc_ref(p);
                    return p;
                }
                if identity && class == 1 {
                    self.inc_ref(f);
                    return f;
                }
                self.store.shared.retry(self.store.threads, |state| {
                    if identity {
                        // Flip the payloads of `f` under the guard that interns them,
                        // so a relocation between attempts rereads the relocated
                        // edges.
                        let node = state.node(f);
                        transitions.clear();
                        for entry in node.edges.chunks_exact(stride) {
                            l.push(transitions, l.key(entry), 1 - l.payload(entry));
                        }
                    }
                    state.intern_referenced(l, initial, transitions)
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
            column: self,
            root: p,
            offset: 0,
            value: (!p.is_f()).then(Vec::new),
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/sparse.rs"]
mod tests;
