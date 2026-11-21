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
use std::mem;
use std::ops::ControlFlow;
use std::slice;
use std::sync::Arc;

use crate::backend::{
    self, Algebra, Align, Backend, BoolOp, Cursor, Function, MapSpec, Refine, Values,
};
use crate::scalar::{self, ScalarKind, ScalarSort};
use crate::util::count::Count;
use crate::util::dag::{self, Dag, Exhausted, Grow, Map, NeedGrowth, NodeRef, Shared};

/// The initial node capacity of the default store.
///
/// The initial arena sizes are large enough that collection alone recovers
/// most transient results, so the arenas rarely grow.
const NODE_CAPACITY: usize = 65_536;
/// The initial edge capacity of the default store.
const EDGE_CAPACITY: usize = 262_144;

#[derive(Debug)]
struct Node {
    /// The payload at zero, before any transition.
    initial: u32,
}

/// How a sort lays its transitions out in edge words, chosen by the sort
/// alone.
///
/// A sort of at most 32 bits packs each key above its payload in one word,
/// using eight bytes per transition. Wider sorts store the key words followed
/// by a payload word. Every reader and writer of edge words uses the layout.
/// Each binding interprets shared edge words at its own stride.
trait Layout: Copy + Send + Sync + 'static {
    /// One transition's key: the packed word's high half, or a borrowed
    /// key-word slice.
    type Key<'a>: Copy;
    fn stride(self) -> usize;
    fn key(self, entry: &[u64]) -> Self::Key<'_>;
    fn payload(self, entry: &[u64]) -> u32;
    /// A key from its row encoding of `sort.words()` little-endian words.
    fn decode(self, words: &[u64]) -> Self::Key<'_>;
    /// A key's borrowed little-endian row encoding.
    fn key_words<'a>(self, key: &'a Self::Key<'_>) -> &'a [u64];
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
    fn key_words(self, key: &u64) -> &[u64] {
        slice::from_ref(key)
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
        out.push((key << 32) | u64::from(payload));
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
    fn key_words<'a>(self, key: &'a &[u64]) -> &'a [u64] {
        key
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

/// Appends a transition when its payload changes during an ordered sweep.
///
/// Each key arrives once, and `last` starts at the payload at zero.
#[inline(always)]
fn change<L: Layout>(l: L, out: &mut Vec<u64>, last: &mut u32, key: L::Key<'_>, value: u32) {
    if value != *last {
        l.push(out, key, value);
        *last = value;
    }
}

/// Sweeps the union of transition points, retaining each map's current
/// payload between events.
///
/// Emits each nonzero key once in increasing order.
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
    let mut points = keys.chunks_exact(words).zip(labels).peekable();
    let initial = match points.peek() {
        Some((key, label)) if l.is_zero(l.decode(key)) => *label,
        _ => default,
    };
    let mut last = initial;
    out.clear();
    next.resize(words, 0);
    while let Some((row, label)) = points.next() {
        let key = l.decode(row);
        change(l, out, &mut last, key, label);
        // Adjacent assignments meet at the next key. Insert a default
        // transition only across a gap.
        if points
            .peek()
            .is_some_and(|(following, _)| l.follows(key, l.decode(following)))
        {
            continue;
        }
        next.copy_from_slice(row);
        if scalar::successor(sort, next) {
            change(l, out, &mut last, l.decode(next), default);
        }
    }
    initial
}

/// Numbers labels by first appearance, appending each reached label once.
///
/// Classes follow the first run carrying each label. The initial run gets
/// class zero without enumerating the domain. Subsequent labels arrive in
/// numeric key order. Class numbers index the appended suffix of `labels`.
fn number<'a, T: Copy + Eq + Hash>(
    initial: T,
    classes: &'a mut Map<T, u32>,
    labels: &'a mut Vec<T>,
) -> impl FnMut(T) -> u32 + 'a {
    classes.clear();
    let mut class_of = move |label| {
        let next = classes.len() as u32;
        *classes.entry(label).or_insert_with(|| {
            labels.push(label);
            next
        })
    };
    class_of(initial);
    class_of
}

/// Physical storage and canonical namespace for transition maps.
type State = Dag<Function, Node, u64>;

impl State {
    /// Interns one externally referenced map.
    ///
    /// Nonzero transition keys are strictly increasing. Each payload differs
    /// from the previous run's payload.
    fn intern_referenced(&self, initial: u32, entries: &[u64]) -> Result<Function, NeedGrowth> {
        let f = if entries.is_empty() {
            assert!(initial <= 1, "constant sparse map must be Boolean");
            Function::new(initial as usize)
        } else {
            self.intern(
                dag::structural_hash((initial, entries)),
                |node, edges| node.initial == initial && edges == entries,
                || Node { initial },
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
            |node, edges| dag::structural_hash((node.initial, edges)),
        );
    }
}

/// A concurrent canonical table of transition maps over each sort's numeric
/// encoding.
///
/// Roots are [`Function`] tokens with manual external reference counts.
/// Equal maps share a token within one store and sort. A live root can be
/// used through any algebra bound to that store and sort. Different sorts
/// can share a token when their initial payload and edge words coincide.
///
/// Numeric cuts operate on Boolean and bit-vector sorts. The backend rejects
/// [`Algebra::masked_eq`]. Transfers handle empty endpoints, identity and
/// constant bit maps, bit-map preimages with a full target, and XOR images
/// of a full source directly. Other transfers enumerate the source members.
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
        let state = Dag::with_capacity(nodes, edges, (0..2).map(|initial| Node { initial }));
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
    fn bind(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Algebra>> {
        fn algebra<L: Layout>(store: Arc<SparseStore>, sort: ScalarSort, l: L) -> Box<dyn Algebra> {
            Box::new(SparseAlgebra { store, sort, l })
        }
        Some(match sort.words() {
            _ if sort.width() <= 32 => algebra(self, sort, Packed),
            2 => algebra(self, sort, Wide::<2>(2)),
            words => algebra(self, sort, Wide::<0>(words)),
        })
    }
}

/// A cursor over the members of a sparse map.
///
/// The cursor retains offsets rather than pointers into a table that can grow.
/// It jumps across excluded runs, and increments across included runs using
/// the complete logical width.
struct SparseCursor<'a, L: Layout> {
    algebra: &'a SparseAlgebra<L>,
    root: Function,
    offset: usize,
    /// The current words, empty before the first advance and absent after
    /// exhaustion.
    value: Option<Vec<u64>>,
}

impl<L: Layout> Cursor for SparseCursor<'_, L> {
    fn next(&mut self) -> Option<&[u64]> {
        let alg = self.algebra;
        let (sort, l) = (alg.sort, alg.l);
        let value = self.value.as_mut()?;
        if value.is_empty() {
            value.resize(sort.words(), 0);
        } else if !scalar::successor(sort, value) {
            self.value = None;
            return None;
        }
        let state = alg.store.shared.read(alg.store.threads);
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
            value.copy_from_slice(l.key_words(&l.key(&node.edges[self.offset..])));
            self.offset += stride;
        }
        drop(state);
        self.value.as_deref()
    }
}

impl<L: Layout> Drop for SparseCursor<'_, L> {
    fn drop(&mut self) {
        self.algebra.dec_ref(self.root);
    }
}

/// A sparse table bound to one scalar sort, with the transition layout its
/// width selects.
struct SparseAlgebra<L> {
    store: Arc<SparseStore>,
    sort: ScalarSort,
    l: L,
}

impl<L: Layout> SparseAlgebra<L> {
    /// The images of `p` under `map` within `q`, or with `image` false, the
    /// members of `p` whose image lies in `q`.
    fn transfer(&self, map: &MapSpec, p: Function, q: Function, image: bool) -> Function {
        map.check_sort(self.sort);
        debug_assert!(
            {
                let state = self.store.shared.read(self.store.threads);
                state.has_refs(p) && state.has_refs(q)
            },
            "cannot use an unreferenced sparse node"
        );
        if p.is_f() || q.is_f() {
            return Function::FALSE;
        }
        if let MapSpec::Bits { clear, flip } = map {
            // Every bit map is total, and XOR permutes the whole sort.
            if (!image && q.is_t())
                || (image && p.is_t() && clear.iter().all(|&word| word == 0))
                || map.is_identity()
            {
                return self.apply(BoolOp::And, p, q);
            }
            if let [lower @ .., top] = clear.as_ref()
                && *top == self.sort.top_mask()
                && lower.iter().all(|&word| word == u64::MAX)
            {
                // Every input maps to flip, and p has at least one member.
                if self.evaluate(q, flip) == 0 {
                    return Function::FALSE;
                }
                if image {
                    return self.set(flip);
                }
                self.inc_ref(p);
                return p;
            }
        }
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

impl<L: Layout> Algebra for SparseAlgebra<L> {
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
        assert!(self.sort.accepts(value), "value is outside the scalar sort");
        let l = self.l;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
        let node = state.node(f);
        let i = search(l, node.edges, l.decode(value));
        run_payload(l, node.initial, node.edges, i)
    }

    /// The number of members of predicate `p`, summed over its runs.
    ///
    /// An included final run has the exact universe boundary `2^width` as
    /// its exclusive upper bound.
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
        let add = |out: &mut [u64], key: &[u64], negate: bool| {
            let flip = if negate { u64::MAX } else { 0 };
            let mut carry = negate;
            for (i, word) in out.iter_mut().enumerate() {
                (*word, carry) = word.carrying_add(key.get(i).copied().unwrap_or(0) ^ flip, carry);
            }
        };
        let mut value = node.initial;
        assert!(value <= 1, "count requires a Boolean predicate");
        for (key, next) in entries(l, &node) {
            assert!(next <= 1, "count requires a Boolean predicate");
            add(&mut sum, l.key_words(&key), next == 1);
            value = next;
        }
        if value == 1 {
            let width = self.sort.width() as usize;
            add(&mut sum[width / 64..], &[1 << (width % 64)], false);
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
            state.intern_referenced(initial, &entries)
        })
    }

    /// The predicate holding where the value is below `bound`: one run from
    /// zero to the bound.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering, or
    /// `bound` is outside the sort.
    fn lt(&self, bound: &[u64]) -> Function {
        assert_ne!(
            self.sort.kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        assert!(self.sort.accepts(bound), "bound is outside the scalar sort");
        let l = self.l;
        let key = l.decode(bound);
        if l.is_zero(key) {
            return Function::FALSE;
        }
        let mut entries = Vec::with_capacity(l.stride());
        l.push(&mut entries, key, 0);
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(1, &entries)
        })
    }

    /// The Boolean combination of two predicates, merging the runs of both.
    ///
    /// The `AndNot` operation is directional: it computes `left and !right`.
    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function {
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
            let initial = u32::from(op.evaluate_bool(a.initial != 0, b.initial != 0));
            let mut last = initial;
            let mut entries = Vec::new();
            merge_entries(l, a, b, |key, x, y| {
                let value = u32::from(op.evaluate_bool(x != 0, y != 0));
                change(l, &mut entries, &mut last, key, value);
            });
            state.intern_referenced(initial, &entries)
        })
    }

    /// The images of `p` under `map` within `q`.
    ///
    /// Empty endpoints give an empty image. Identity bit maps intersect the
    /// endpoints, XOR maps take a full source onto all of `q`, and a constant
    /// bit map tests its one output against `q`. Other maps enumerate `p`.
    ///
    /// # Panics
    ///
    /// Panics if the map or a defined Rust map output is outside the sort.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.transfer(map, p, q, true)
    }

    /// The members of `p` whose image under `map` lies in `q`.
    ///
    /// A bit map with a full target retains `p` because every input has a
    /// defined image. A constant bit map tests its one output against `q`.
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
            let initial = if values[0] != 0 { values[1] } else { values[2] };
            let mut last = initial;
            let mut out = Vec::new();
            let mut runs = nodes.each_ref().map(|node| entries(l, node).peekable());
            while let Some(key) = runs
                .iter_mut()
                .filter_map(|run| run.peek().map(|&(key, _)| key))
                .min_by(|&a, &b| l.compare(a, b))
            {
                for (run, value) in runs.iter_mut().zip(&mut values) {
                    if let Some((_, v)) =
                        run.next_if(|&(next, _)| l.compare(next, key) == Ordering::Equal)
                    {
                        *value = v;
                    }
                }
                let value = if values[0] != 0 { values[1] } else { values[2] };
                change(l, &mut out, &mut last, key, value);
            }
            state.intern_referenced(initial, &out)
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
        scratch: &mut Align,
    ) -> Function {
        debug_assert!(left != right, "cannot align a partition with itself");
        debug_assert!(
            !left.is_f() && !right.is_f(),
            "cannot align a trivial partition"
        );
        let l = self.l;
        let Align {
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
            let mut class_of = number((a.initial, b.initial), pairs, labels);
            transitions.clear();
            // At least one input class changes at each key, so the pair's
            // output class changes as well.
            merge_entries(l, a, b, |key, x, y| {
                l.push(transitions, key, class_of((x, y)));
            });
            state.intern_referenced(0, transitions)
        })
    }

    /// The coarsening of `f` by `classes`.
    ///
    /// The class map already specifies canonical result numbers. Coalescing
    /// adjacent runs that map to the same class preserves first-occurrence
    /// numbering.
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Align) -> Function {
        if cfg!(debug_assertions) {
            backend::check_quotient(classes);
        }
        let l = self.l;
        let out = &mut scratch.transitions;
        self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(state.has_refs(f), "cannot use an unreferenced sparse node");
            let node = state.node(f);
            let initial = classes[node.initial as usize];
            let mut last = initial;
            #[cfg(debug_assertions)]
            let mut max = node.initial;
            out.clear();
            for (key, class) in entries(l, &node) {
                #[cfg(debug_assertions)]
                {
                    max = max.max(class);
                }
                change(l, out, &mut last, key, classes[class as usize]);
            }
            #[cfg(debug_assertions)]
            assert_eq!(
                classes.len(),
                max as usize + 1,
                "there must be one quotient class per input class"
            );
            state.intern_referenced(initial, out)
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
        // A later boundary can reveal an overlap after labels were appended.
        let mut labels = mem::DropGuard::new(labels, |out| out.truncate(start));
        let f = self.store.shared.retry(self.store.threads, |state| {
            debug_assert!(
                cases.iter().all(|&(p, _)| state.has_refs(p)),
                "cannot use an unreferenced sparse node"
            );
            // A retried attempt appends its labels afresh.
            labels.truncate(start);
            if let &[(p, label)] = cases {
                if label == default || p.is_constant() {
                    labels.push(if p.is_f() { default } else { label });
                    return Ok(Function::FALSE);
                }
                // A nonconstant case false at zero already has classes 0 then 1
                // in first-occurrence order. Retaining it skips the scan, sort,
                // and second interning.
                if state.node(p).initial == 0 {
                    labels.extend([default, label]);
                    state.inc_ref(p);
                    return Ok(p);
                }
            }
            let enter = |active, count, label| {
                assert!(
                    count == 0 || active == label,
                    "cases with different labels overlap"
                );
                (label, count + 1)
            };
            let (mut active, mut count) = (default, 0_usize);
            let mut events = Vec::new();
            for &(p, label) in cases {
                let node = state.node(p);
                if node.initial != 0 {
                    (active, count) = enter(active, count, label);
                }
                events.extend(entries(l, &node).map(|(key, value)| (key, label, value != 0)));
            }
            // Ends precede starts at the same key. Only the label after all
            // events at that key is reachable.
            events.sort_unstable_by(|a, b| l.compare(a.0, b.0).then_with(|| a.2.cmp(&b.2)));
            let (mut entries, mut classes) = (Vec::new(), Map::default());
            let mut class_of = number(active, &mut classes, &mut labels);
            let mut last = 0;
            for run in events.chunk_by(|a, b| l.compare(a.0, b.0) == Ordering::Equal) {
                for &(_, label, holds) in run {
                    if holds {
                        (active, count) = enter(active, count, label);
                    } else {
                        count -= 1;
                    }
                }
                let label = if count == 0 { default } else { active };
                change(l, &mut entries, &mut last, run[0].0, class_of(label));
            }
            state.intern_referenced(0, &entries)
        });
        mem::DropGuard::dismiss(labels);
        f
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
        // Relabeling by first occurrence is injective, so the runs that
        // `points` merged stay merged and the entries rewrite in place.
        let mut class_of = number(initial, classes, labels);
        for entry in transitions.chunks_exact_mut(l.stride()) {
            let class = class_of(l.payload(entry));
            l.set_payload(entry, class);
        }
        self.store.shared.retry(self.store.threads, |state| {
            state.intern_referenced(0, transitions)
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
        scratch: &mut Refine,
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
        // A one-class partition and a single point both give a cell equal to p.
        // A Boolean map's runs alternate, so two transitions from a false
        // initial payload bound the entire included run.
        let single = if f.is_f() {
            Some(0)
        } else if b.initial == 0
            && b.edges.len() == 2 * stride
            && l.follows(l.key(b.edges), l.key(&b.edges[stride..]))
        {
            let i = search(l, a.edges, l.key(b.edges));
            Some(run_payload(l, a.initial, a.edges, i))
        } else {
            None
        };
        if let Some(class) = single {
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
        // f. A predicate holding at zero is a Boolean map too, so the class
        // opposite its initial payload is the second one, and the full
        // predicate has none.
        let identity = p.is_t()
            && a.edges
                .chunks_exact(stride)
                .all(|entry| l.payload(entry) <= 1);
        // The dense lookup grows through the largest class number reached
        // and retains its length across calls, regardless of transition count.
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
                    state.intern_referenced(initial, transitions)
                })
            })?;
        }
        ControlFlow::Continue(())
    }

    /// A cursor over the members of `p`, holding its own reference to the root
    /// until dropped.
    ///
    /// It jumps across excluded runs and steps through included runs one
    /// value at a time. A symbol predicate excluding a few values still yields
    /// nearly all `2^64` members.
    fn members(&self, p: Function) -> Box<dyn Cursor + '_> {
        debug_assert!(
            self.store.shared.read(self.store.threads).has_refs(p),
            "cannot use an unreferenced sparse node"
        );
        self.inc_ref(p);
        Box::new(SparseCursor {
            algebra: self,
            root: p,
            offset: 0,
            value: (!p.is_f()).then(Vec::new),
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/sparse.rs"]
mod tests;
