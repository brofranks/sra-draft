//! Canonical multi-terminal decision diagrams with 6 bits per level.
//!
//! A branch tests one level. Its disjoint `(mask, child)` edges cover all 64
//! assignments, in order of each edge's first assignment. A terminal has no
//! edges and stores a `u32` value. Predicates use the terminals
//! [`Function::FALSE`] (0) and [`Function::TRUE`] (1). Partition terminals hold
//! dense class numbers, assigned by first occurrence in increasing unsigned
//! encoding order. The [`BddStore`] type implements [`Backend`], and `Span`
//! places an encoded scalar value on levels.
//!
//! Products and Boolean complement use an explicit stack. Construction from
//! sorted keys recurses once per scalar level. Bit-map composition, bit
//! remapping, and terminal relabeling recurse along diagram edges.

use std::array;
use std::cmp::Ordering;
use std::mem::{self, MaybeUninit};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::backend::{
    Algebra, Align, Backend, BoolOp, Cache, Cursor, Function, MapSpec, Refine, Values,
};
use crate::cache;
use crate::scalar::{self, ScalarKind, ScalarSort};
use crate::util::count::{self, Count};
use crate::util::dag::{
    self, Dag, Exhausted, Grow, Map, NeedGrowth, NodeRef as DagNodeRef, Read, Shared,
};

/// The number of value bits one level tests.
const LEVEL_BITS: u32 = 6;
/// The number of assignments at one level, one bit of a mask each.
const ASSIGNMENTS: usize = 1 << LEVEL_BITS;
/// The initial node capacity of the default store.
const NODE_CAPACITY: usize = 4_096;
/// The initial edge capacity of the default store.
const EDGE_CAPACITY: usize = 16_384;

const _: () = assert!(
    ASSIGNMENTS == u64::BITS as usize,
    "an edge mask must have exactly one bit per level assignment"
);

/// The six-bit levels occupied by one scalar sort, starting at level zero.
///
/// Bit 0 of the value sits at position 0 of `last_level`, and higher bits fill
/// higher positions and then earlier levels, so only level zero may be short.
/// Assignments beyond a short level's width alias the assignment obtained by
/// clearing its unused high bits, as normalized by [`State::intern_level`].
#[derive(Clone, Copy, Debug)]
struct Span {
    sort: ScalarSort,
}

impl Span {
    /// The level holding bit zero of the nonempty scalar encoding.
    #[inline(always)]
    const fn last_level(self) -> u32 {
        self.sort.width().div_ceil(LEVEL_BITS) - 1
    }

    /// The number of bits of the value that `level` holds.
    #[inline(always)]
    fn width(self, level: u32) -> u32 {
        (self.sort.width() - (self.last_level() - level) * LEVEL_BITS).min(LEVEL_BITS)
    }

    /// The mask of the assignments at `level` that some value reaches.
    #[inline(always)]
    fn valid_mask(self, level: u32) -> u64 {
        u64::MAX >> (u64::BITS - (1 << self.width(level)))
    }

    /// The `(word, shift, width)` of the bits `level` holds in an encoding.
    #[inline(always)]
    fn place(self, level: u32) -> (usize, u32, u32) {
        let low = (self.last_level() - level) * LEVEL_BITS;
        (
            (low / u64::BITS) as usize,
            low % u64::BITS,
            self.width(level),
        )
    }

    /// The assignment of `level` in `value`: its bits there, least significant
    /// first.
    #[inline(always)]
    fn assignment(self, level: u32, value: &[u64]) -> usize {
        let (word, shift, width) = self.place(level);
        let mut bits = value[word] >> shift;
        if shift + width > u64::BITS {
            bits |= value[word + 1] << (u64::BITS - shift);
        }
        (bits & ((1 << width) - 1)) as usize
    }

    /// Writes an assignment fitting the level's width, possibly across two words.
    #[inline(always)]
    fn assign(self, level: u32, assignment: u64, value: &mut [u64]) {
        let (word, shift, width) = self.place(level);
        value[word] = (value[word] & !(((1_u64 << width) - 1) << shift)) | (assignment << shift);
        if shift + width > u64::BITS {
            value[word + 1] = (value[word + 1] & !((1_u64 << (shift + width - u64::BITS)) - 1))
                | (assignment >> (u64::BITS - shift));
        }
    }
}

/// The level that orders terminals below every tested level.
///
/// A terminal may still store `u32::MAX` as its value, since its empty edge
/// span identifies it.
const TERMINAL_LEVEL: u32 = u32::MAX;

/// A diagram node: a terminal value or a tested level.
#[derive(Debug)]
struct Node {
    /// The terminal value of a zero-edge node, otherwise the tested level of a
    /// branch, read through [`NodeRef::terminal`] and [`NodeRef::level`].
    data: u32,
}

const _: () = assert!(
    size_of::<Node>() == size_of::<u32>(),
    "the edge count lets one u32 hold either a branch level or a terminal value"
);

/// An edge leading to a diagram node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Edge {
    /// The assignments at this level that traverse this edge.
    mask: u64,
    child: Function,
}

/// A borrowed diagram node with its stored edges.
type NodeRef<'a> = DagNodeRef<'a, Node, Edge>;

impl NodeRef<'_> {
    /// Whether this node is a terminal, which stores no edges.
    #[inline(always)]
    fn is_terminal(&self) -> bool {
        self.edges.is_empty()
    }

    /// The terminal value, or `None` for a branch.
    #[inline(always)]
    fn terminal(&self) -> Option<u32> {
        self.is_terminal().then_some(self.data)
    }

    /// The tested level of a branch, or [`TERMINAL_LEVEL`] for a terminal.
    #[inline(always)]
    fn level(&self) -> u32 {
        if self.is_terminal() {
            TERMINAL_LEVEL
        } else {
            self.data
        }
    }
}

/// The edge covering the assignment selected by the one-bit mask `bit`.
#[inline(always)]
fn covering_edge(edges: &[Edge], bit: u64) -> &Edge {
    edges
        .iter()
        .find(|edge| edge.mask & bit != 0)
        .expect("edges must cover every level assignment")
}

/// The single edge list of an operand below the level being combined, which
/// leads every assignment at that level to the operand itself.
#[inline(always)]
fn whole_edge(child: Function) -> [Edge; 1] {
    [Edge {
        mask: u64::MAX,
        child,
    }]
}

/// A fixed-capacity buffer holding at most one level's worth of items.
///
/// Items fill an initialized prefix. The unused tail stays uninitialized, so
/// constructing the buffer requires no clearing.
struct LevelBuffer<T: Copy> {
    items: [MaybeUninit<T>; ASSIGNMENTS],
    len: usize,
}

impl<T: Copy> LevelBuffer<T> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            items: [MaybeUninit::uninit(); ASSIGNMENTS],
            len: 0,
        }
    }

    /// Appends one item.
    ///
    /// # Panics
    ///
    /// Panics if the buffer already holds [`ASSIGNMENTS`] items.
    #[inline(always)]
    fn push(&mut self, item: T) {
        self.items[self.len].write(item);
        self.len += 1;
    }

    /// The initialized items.
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        // SAFETY: `push` initializes each entry before extending `len`.
        unsafe { self.items[..self.len].assume_init_ref() }
    }

    /// The initialized items, exclusively.
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the prefix is initialized as in `as_slice`, and the mutable
        // borrow grants exclusive access to it.
        unsafe { self.items[..self.len].assume_init_mut() }
    }
}

impl LevelBuffer<Edge> {
    /// Widens the edge to `child` by `mask`, or appends one.
    #[inline(always)]
    fn merge(&mut self, mask: u64, child: Function) {
        if let Some(edge) = self
            .as_mut_slice()
            .iter_mut()
            .find(|edge| edge.child == child)
        {
            edge.mask |= mask;
        } else {
            self.push(Edge { mask, child });
        }
    }
}

/// The backing storage and canonical namespace for all diagrams in one store.
type State = Dag<Function, Node, Edge>;

impl Grow for State {
    type Cx<'a> = usize;

    fn grow(&mut self, exhausted: Exhausted, threads: usize) {
        self.maintain(
            exhausted,
            threads,
            |edge| Some(edge.child),
            |_| {},
            |node, edges| dag::structural_hash((node.data, edges)),
        );
    }
}

/// Computed tables for one construction attempt.
///
/// Entries borrow their roots. Between attempts, collection can reclaim nodes
/// and leave their tokens available for reuse.
#[derive(Default)]
struct Memo {
    /// Boolean results keyed by operation and operands.
    apply: Map<(BoolOp, Function, Function), Function>,
    /// Conditional results keyed by `(condition, yes, no)`.
    ite: Map<(Function, Function, Function), Function>,
    /// Both directions of complementation within this attempt.
    not: Map<Function, Function>,
    /// Remapped roots for one span and one pair of clear and flip masks.
    remap: Map<Function, Function>,
}

/// Rebuilds each branch after visiting its cofactors in assignment order.
///
/// The `resolve` callback may canonicalize its key, and the branch is then
/// expanded, rebuilt, and memoized under that key. Each pending rebuild
/// records where its completed edges start. The first expansion reserves room
/// for a shallow walk, so a terminal or cached root returns before either
/// vector allocates.
fn try_walk<K: Copy, M>(
    root: K,
    memo: &mut M,
    mut resolve: impl FnMut(&mut M, &mut K) -> Result<Option<Function>, NeedGrowth>,
    mut expand: impl FnMut(K, &mut LevelBuffer<(u64, K)>) -> u32,
    mut rebuild: impl FnMut(&mut M, K, u32, &[Edge]) -> Result<Function, NeedGrowth>,
) -> Result<Function, NeedGrowth> {
    /// The entries reserved for each vector at the first expansion, which
    /// most shallow walks never exceed.
    const RESERVE: usize = 16;
    enum Task<K> {
        Visit(K),
        Rebuild(K, u32, usize),
    }

    let mut stack = Vec::new();
    let mut edges = Vec::new();
    let (mut mask, mut task) = (u64::MAX, Task::Visit(root));
    loop {
        let child = match task {
            Task::Visit(mut key) => {
                let Some(child) = resolve(memo, &mut key)? else {
                    let mut cofactors = LevelBuffer::new();
                    let level = expand(key, &mut cofactors);
                    // Each reservation requests additional space from the current
                    // length, so reserve only on the first expansion.
                    if stack.capacity() == 0 {
                        stack.reserve(RESERVE);
                        edges.reserve(RESERVE);
                    }
                    stack.push((mask, Task::Rebuild(key, level, edges.len())));
                    let (&(first_mask, first), rest) = cofactors
                        .as_slice()
                        .split_first()
                        .expect("a branch must have a cofactor");
                    stack.extend(rest.iter().rev().map(|&(m, k)| (m, Task::Visit(k))));
                    (mask, task) = (first_mask, Task::Visit(first));
                    continue;
                };
                child
            }
            Task::Rebuild(key, level, start) => {
                let child = rebuild(memo, key, level, &edges[start..])?;
                edges.truncate(start);
                child
            }
        };
        let Some(next) = stack.pop() else {
            return Ok(child);
        };
        edges.push(Edge { mask, child });
        (mask, task) = next;
    }
}

/// Whether `len` distinct valid keys cover the sort.
fn covers(sort: ScalarSort, len: usize) -> bool {
    1_usize.checked_shl(sort.width()) == Some(len)
}

/// Returns a label's class and appends the label on first use.
fn class(classes: &mut Map<u32, u32>, labels: &mut Vec<u32>, label: u32) -> u32 {
    let next = classes.len() as u32;
    *classes.entry(label).or_insert_with(|| {
        labels.push(label);
        next
    })
}

impl State {
    /// The terminal value reached by a valid scalar encoding.
    fn evaluate(&self, span: Span, mut f: Function, value: &[u64]) -> u32 {
        loop {
            let node = self.node(f);
            if let Some(terminal) = node.terminal() {
                return terminal;
            }
            let assignment = span.assignment(node.level(), value);
            f = covering_edge(node.edges, 1_u64 << assignment).child;
        }
    }

    /// Interns the canonical terminal for a value without acquiring an
    /// external reference.
    fn intern_terminal(&self, value: u32) -> Result<Function, NeedGrowth> {
        if value <= 1 {
            return Ok(Function::new(value as usize));
        }
        self.intern_node(value, &[])
    }

    /// The canonical node for a terminal or an ordered complete branch,
    /// reducing a one-child branch to its child.
    fn intern_node(&self, data: u32, edges: &[Edge]) -> Result<Function, NeedGrowth> {
        // A branch's nonempty, disjoint masks cover all 64 assignments in order
        // of first assignment. Each child is distinct and is either a terminal
        // or a branch at a later level.
        if let [edge] = edges {
            return Ok(edge.child);
        }

        self.intern(
            dag::structural_hash((data, edges)),
            |node, stored| node.data == data && stored == edges,
            || Node { data },
            edges,
        )
    }

    /// The complete level node whose assignment `a` leads to `child(a)`.
    ///
    /// Positions beyond a short first level are never reached and copy the
    /// reachable assignment they alias.
    fn intern_level(
        &self,
        span: Span,
        level: u32,
        child: impl Fn(usize) -> Function,
    ) -> Result<Function, NeedGrowth> {
        let mut edges = LevelBuffer::new();
        let mask = (1 << span.width(level)) - 1;
        for assignment in 0..ASSIGNMENTS {
            edges.merge(1 << assignment, child(assignment & mask));
        }
        self.intern_node(level, edges.as_slice())
    }

    /// Ordered nonempty product cofactors, with one full edge per deeper operand,
    /// in the same order as alignment.
    #[inline(always)]
    fn cofactors<const N: usize>(
        &self,
        operands: [Function; N],
        mut emit: impl FnMut(u64, [Function; N]),
    ) -> u32 {
        let nodes = operands.map(|f| self.node(f));
        let level = nodes
            .iter()
            .fold(TERMINAL_LEVEL, |l, node| l.min(node.level()));
        debug_assert_ne!(level, TERMINAL_LEVEL, "cofactors need a branch operand");
        let whole = operands.map(whole_edge);
        let parts: [&[Edge]; N] = array::from_fn(|i| {
            if nodes[i].level() == level {
                nodes[i].edges
            } else {
                &whole[i]
            }
        });
        // First appearances advance through each ordered edge list. A later
        // assignment of an interleaved mask belongs to an already seen edge.
        #[inline(always)]
        fn select<'a>(edges: &'a [Edge], seen: &mut usize, bit: u64) -> &'a Edge {
            if let Some(edge) = edges.get(*seen)
                && edge.mask & bit != 0
            {
                *seen += 1;
                return edge;
            }
            covering_edge(&edges[..*seen], bit)
        }
        // Emit each complete intersection at its first assignment so both result
        // edges and class numbers stay canonical.
        let mut seen = [0; N];
        let mut remaining = u64::MAX;
        while remaining != 0 {
            let bit = remaining.isolate_lowest_one();
            let selected: [_; N] = array::from_fn(|i| select(parts[i], &mut seen[i], bit));
            let mask = selected.iter().fold(u64::MAX, |m, edge| m & edge.mask);
            emit(mask, selected.map(|edge| edge.child));
            remaining &= !mask;
        }
        level
    }

    /// Interns the product at each earliest operand level, merging cofactor
    /// results in canonical order.
    fn try_product<const N: usize, M>(
        &self,
        operands: [Function; N],
        memo: &mut M,
        resolve: impl FnMut(&mut M, &mut [Function; N]) -> Result<Option<Function>, NeedGrowth>,
        mut remember: impl FnMut(&mut M, [Function; N], Function),
    ) -> Result<Function, NeedGrowth> {
        try_walk(
            operands,
            memo,
            resolve,
            |key, cofactors| self.cofactors(key, |mask, child| cofactors.push((mask, child))),
            |m, key, level, children| {
                let mut edges = LevelBuffer::new();
                for edge in children {
                    edges.merge(edge.mask, edge.child);
                }
                // Matching an operand's edges shares it without a unique-table probe.
                let edges = edges.as_slice();
                let result = if let Some(f) = key.into_iter().find(|&f| {
                    let node = self.node(f);
                    node.level() == level && node.edges == edges
                }) {
                    f
                } else {
                    self.intern_node(level, edges)?
                };
                remember(m, key, result);
                Ok(result)
            },
        )
    }

    /// Combines a Boolean condition and two branches at their earliest level.
    ///
    /// Each nonempty intersection of the three edge masks selects one child
    /// per operand, and deeper operands keep their root.
    fn try_ite(
        &self,
        memo: &mut Memo,
        condition: Function,
        yes: Function,
        no: Function,
    ) -> Result<Function, NeedGrowth> {
        self.try_product(
            [condition, yes, no],
            memo,
            |m, [c, y, n]| {
                if y == c {
                    *y = Function::TRUE;
                }
                if n == c {
                    *n = Function::FALSE;
                }
                if c.is_t() || y == n {
                    return Ok(Some(*y));
                }
                if c.is_f() {
                    return Ok(Some(*n));
                }
                if y.is_t() && n.is_f() {
                    return Ok(Some(*c));
                }
                if y.is_f() && n.is_t() {
                    return self.try_not(&mut m.not, *c).map(Some);
                }
                Ok(m.ite.get(&(*c, *y, *n)).copied())
            },
            |m, [c, y, n], f| {
                m.ite.insert((c, y, n), f);
            },
        )
    }

    /// Complements a Boolean predicate.
    ///
    /// Unlike `try_map` with `value ^ 1`, this skips the edge-merging scan and
    /// memoizes the involution in both directions.
    fn try_not(
        &self,
        memo: &mut Map<Function, Function>,
        p: Function,
    ) -> Result<Function, NeedGrowth> {
        try_walk(
            p,
            memo,
            |m, q| {
                Ok(match *q {
                    Function::FALSE => Some(Function::TRUE),
                    Function::TRUE => Some(Function::FALSE),
                    _ => m.get(q).copied(),
                })
            },
            |q, cofactors| {
                let node = self.node(q);
                for edge in node.edges {
                    cofactors.push((edge.mask, edge.child));
                }
                node.level()
            },
            |m, q, level, edges| {
                // Negation is bijective and preserves edge grouping and order.
                let result = self.intern_node(level, edges)?;
                m.insert(q, result);
                m.insert(result, q);
                Ok(result)
            },
        )
    }

    /// The Boolean combination `op` of two predicates, memoized per operation.
    ///
    /// Commutative operands are ordered so equal pairs share one entry.
    fn try_apply(
        &self,
        memo: &mut Memo,
        op: BoolOp,
        left: Function,
        right: Function,
    ) -> Result<Function, NeedGrowth> {
        self.try_product(
            [left, right],
            memo,
            |m, key| {
                // Canonicalize to increase odds of memo reuse.
                if op.is_commutative() && key[1] < key[0] {
                    key.swap(0, 1);
                }
                let [l, r] = *key;
                if let Some(f) = op.simplify(l, r) {
                    return Ok(Some(f));
                }
                if l.is_t() && matches!(op, BoolOp::AndNot | BoolOp::Xor) {
                    return self.try_not(&mut m.not, r).map(Some);
                }
                Ok(m.apply.get(&(op, l, r)).copied())
            },
            |m, [l, r], f| {
                m.apply.insert((op, l, r), f);
            },
        )
    }

    /// Interns a branch from one mapped child per stored edge, merging equal
    /// results and reusing the original root when every child stays the same.
    ///
    /// Callers collect the child tokens first, so the output edge buffer is
    /// live only while rebuilding this branch.
    fn intern_children(&self, f: Function, children: &[Function]) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        let pairs = node.edges.iter().zip(children);
        if pairs.clone().all(|(edge, &child)| edge.child == child) {
            return Ok(f);
        }
        let mut edges = LevelBuffer::new();
        for (edge, &child) in pairs {
            edges.merge(edge.mask, child);
        }
        self.intern_node(node.data, edges.as_slice())
    }

    /// Remaps assignments within each affected level.
    ///
    /// Assignments differing only in cleared bits share the disjunction of their
    /// children, making those bits unconstrained. Flipping any remaining bit
    /// exchanges the assignments that differ in that bit.
    ///
    /// Subgraphs below `last` hold only unaffected bits and are reused.
    fn try_remap(
        &self,
        memo: &mut Memo,
        span: Span,
        clear: &[u64],
        flip: &[u64],
        last: u32,
        f: Function,
    ) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        // A terminal orders below every level, so this also returns terminals.
        if node.level() > last {
            return Ok(f);
        }
        if let Some(&result) = memo.remap.get(&f) {
            return Ok(result);
        }
        let level = node.level();
        let level_clear = span.assignment(level, clear);
        let level_flip = span.assignment(level, flip);
        let mut children = [Function::FALSE; ASSIGNMENTS];
        let result = if level_clear == 0 && level_flip == 0 {
            // At an untouched level, each input edge's assignments stay
            // together. Edges whose mapped children coincide merge their masks.
            let children = &mut children[..node.edges.len()];
            for (child, edge) in children.iter_mut().zip(node.edges) {
                *child = self.try_remap(memo, span, clear, flip, last, edge.child)?;
            }
            self.intern_children(f, children)?
        } else {
            for edge in node.edges {
                if edge.child.is_f() {
                    continue;
                }
                let child = self.try_remap(memo, span, clear, flip, last, edge.child)?;
                let mut remaining = edge.mask & span.valid_mask(level);
                // Assignments of one edge that share a target contribute the
                // same child, which one union absorbs.
                let mut combined = 0_u64;
                while let Some(a) = remaining.lowest_one() {
                    remaining &= remaining - 1;
                    let i = (a as usize ^ level_flip) & !level_clear;
                    if combined & (1 << i) != 0 {
                        continue;
                    }
                    combined |= 1 << i;
                    children[i] = self.try_apply(memo, BoolOp::Or, children[i], child)?;
                }
            }
            self.intern_level(span, level, |a| children[a & !level_clear])?
        };
        memo.remap.insert(f, result);
        Ok(result)
    }

    /// The predicate holding at `x` where `f` holds at `(x & !clear) ^ flip`.
    ///
    /// Each assignment takes the child that `f` gives its image, so only the
    /// levels `f` tests are visited and cleared bits become unconstrained.
    ///
    /// Subgraphs below `last` hold only unaffected bits and are reused. The
    /// memo borrows composed roots for this span and pair of masks.
    fn try_compose(
        &self,
        memo: &mut Map<Function, Function>,
        span: Span,
        clear: &[u64],
        flip: &[u64],
        last: u32,
        f: Function,
    ) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        // A terminal orders below every level, so this also returns terminals.
        if node.level() > last {
            return Ok(f);
        }
        if let Some(&result) = memo.get(&f) {
            return Ok(result);
        }
        let level = node.level();
        let level_clear = span.assignment(level, clear);
        let level_flip = span.assignment(level, flip);
        let mut children = [Function::FALSE; ASSIGNMENTS];
        let result = if level_clear == 0 && level_flip == 0 {
            let children = &mut children[..node.edges.len()];
            for (child, edge) in children.iter_mut().zip(node.edges) {
                *child = self.try_compose(memo, span, clear, flip, last, edge.child)?;
            }
            self.intern_children(f, children)?
        } else {
            // Start with the fixed output bits. Each free bit adds the same
            // assignments with that bit set.
            let mut reachable = 1_u64 << (level_clear & level_flip);
            let mut free = ((1 << span.width(level)) - 1) & !level_clear;
            while free != 0 {
                reachable |= reachable << free.isolate_lowest_one();
                free &= free - 1;
            }
            for edge in node.edges {
                let mut remaining = edge.mask & reachable;
                if remaining == 0 || edge.child.is_f() {
                    continue;
                }
                let child = self.try_compose(memo, span, clear, flip, last, edge.child)?;
                while let Some(a) = remaining.lowest_one() {
                    remaining &= remaining - 1;
                    children[a as usize] = child;
                }
            }
            self.intern_level(span, level, |a| children[(a & !level_clear) ^ level_flip])?
        };
        memo.insert(f, result);
        Ok(result)
    }

    /// The predicate `(x & mask) == (value & mask)` over `span`, without an
    /// external reference.
    fn try_masked_eq(
        &self,
        span: Span,
        mask: &[u64],
        value: &[u64],
    ) -> Result<Function, NeedGrowth> {
        let mut f = Function::TRUE;
        for level in (0..=span.last_level()).rev() {
            let bits = span.assignment(level, mask);
            if bits == 0 {
                continue;
            }
            let expected = span.assignment(level, value) & bits;
            f = self.intern_level(span, level, |v| {
                if v & bits == expected {
                    f
                } else {
                    Function::FALSE
                }
            })?;
        }
        Ok(f)
    }

    /// The cell of partition `f` numbered `class`, restricted to `p`.
    fn try_cell(
        &self,
        memo: &mut Map<(Function, Function), Function>,
        f: Function,
        p: Function,
        class: u32,
    ) -> Result<Function, NeedGrowth> {
        self.try_product(
            [f, p],
            memo,
            |m, &mut [g, q]| {
                if q.is_f() {
                    return Ok(Some(Function::FALSE));
                }
                if let Some(value) = self.node(g).terminal() {
                    return Ok(Some(if value == class { q } else { Function::FALSE }));
                }
                Ok(m.get(&(g, q)).copied())
            },
            |m, [g, q], result| {
                m.insert((g, q), result);
            },
        )
    }

    /// Rebuilds `f` with each terminal `i` replaced by `map(i)`, merging edges
    /// whose children collapse and reusing unchanged subgraphs.
    fn try_map(
        &self,
        memo: &mut Map<Function, Function>,
        f: Function,
        map: &mut impl FnMut(u32) -> u32,
    ) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        if let Some(value) = node.terminal() {
            let mapped = map(value);
            return if mapped == value {
                Ok(f)
            } else {
                self.intern_terminal(mapped)
            };
        }
        if let Some(&result) = memo.get(&f) {
            return Ok(result);
        }

        let mut children = [Function::FALSE; ASSIGNMENTS];
        let children = &mut children[..node.edges.len()];
        for (child, edge) in children.iter_mut().zip(node.edges) {
            *child = self.try_map(memo, edge.child, map)?;
        }
        let result = self.intern_children(f, children)?;
        memo.insert(f, result);
        Ok(result)
    }

    /// Numbers the reachable pairs of input classes in order of first appearance.
    ///
    /// Each terminal holds a class number indexing its pair in `labels[start..]`.
    fn try_align(
        &self,
        memo: &mut Map<(Function, Function), Function>,
        labels: &mut Vec<(u32, u32)>,
        start: usize,
        left: Function,
        right: Function,
    ) -> Result<Function, NeedGrowth> {
        self.try_product(
            [left, right],
            memo,
            |m, &mut [l, r]| {
                if let Some(&result) = m.get(&(l, r)) {
                    return Ok(Some(result));
                }
                if let Some(a) = self.node(l).terminal()
                    && let Some(b) = self.node(r).terminal()
                {
                    let class = self.intern_terminal((labels.len() - start) as u32)?;
                    labels.push((a, b));
                    m.insert((l, r), class);
                    return Ok(Some(class));
                }
                Ok(None)
            },
            |m, [l, r], f| {
                m.insert((l, r), f);
            },
        )
    }

    /// Writes `label` into the accumulator `acc` where `p` holds, asserting it
    /// covers only `default` or `label`.
    fn try_select(
        &self,
        memo: &mut Map<(Function, Function), Function>,
        default: u32,
        p: Function,
        label: u32,
        acc: Function,
    ) -> Result<Function, NeedGrowth> {
        self.try_product(
            [p, acc],
            memo,
            |m, &mut [q, a]| {
                if q.is_f() {
                    return Ok(Some(a));
                }
                if let Some(covered) = self.node(a).terminal() {
                    if covered == label {
                        return Ok(Some(a));
                    }
                    if q.is_t() {
                        assert_eq!(covered, default, "cases with different labels overlap");
                        return self.intern_terminal(label).map(Some);
                    }
                }
                Ok(m.get(&(q, a)).copied())
            },
            |m, [q, a], f| {
                m.insert((q, a), f);
            },
        )
    }

    /// The map sending each of the sorted `keys` to its label terminal and
    /// other values to `default`, built from `level` down.
    ///
    /// The nonempty keys share the bits above `level`, and `offset` locates
    /// them in the sequence `label` indexes.
    fn try_values(
        &self,
        span: Span,
        level: u32,
        keys: &[u64],
        label: &impl Fn(usize) -> u32,
        offset: usize,
        default: Function,
    ) -> Result<Function, NeedGrowth> {
        let words = span.sort.words();
        let len = keys.len() / words;
        if level > span.last_level() {
            debug_assert_eq!(len, 1, "exactly one key must remain below the last level");
            return self.intern_terminal(label(offset));
        }
        let mut children = [default; ASSIGNMENTS];
        // Within this prefix the level's digits are sorted. Once a group
        // starts with the final digit, it extends to the end of the keys.
        let last = span.assignment(level, &keys[keys.len() - words..]);
        let mut start = 0;
        while start < len {
            let assignment = span.assignment(level, &keys[start * words..][..words]);
            let mut end = len;
            if assignment != last {
                end = start + 1;
                while span.assignment(level, &keys[end * words..][..words]) == assignment {
                    end += 1;
                }
            }
            children[assignment] = self.try_values(
                span,
                level + 1,
                &keys[start * words..end * words],
                label,
                offset + start,
                default,
            )?;
            start = end;
        }
        self.intern_level(span, level, |a| children[a])
    }

    /// Replaces label terminals with class numbers in order of first appearance.
    ///
    /// Appends one label per class, or on error or unwind truncates `labels`
    /// back to its original length.
    fn try_canonicalize(&self, f: Function, labels: &mut Vec<u32>) -> Result<Function, NeedGrowth> {
        // Depth-first traversal in stored edge order preserves the numbering of
        // an already canonical partition.
        let start = labels.len();
        let mut out = mem::DropGuard::new(labels, |out| out.truncate(start));
        let mut classes = Map::default();
        let result = self.try_map(&mut Map::default(), f, &mut |label| {
            class(&mut classes, &mut out, label)
        })?;
        mem::DropGuard::dismiss(out);
        Ok(result)
    }
}

/// The clear and flip masks of a bit map.
///
/// # Panics
///
/// Panics if `map` is a Rust closure, which has no symbolic form.
fn bit_map(map: &MapSpec) -> (&[u64], &[u64]) {
    let MapSpec::Bits { clear, flip } = map else {
        panic!("a BDD store cannot apply a Rust map, which needs an enumerating backend")
    };
    (clear, flip)
}

/// The deepest level holding a bit of `clear` or `flip`, or zero if none.
fn last_affected(span: Span, clear: &[u64], flip: &[u64]) -> u32 {
    // Bit zero belongs to the deepest level.
    for (i, (c, f)) in clear.iter().zip(flip).enumerate() {
        if let Some(bit) = (c | f).lowest_one() {
            return span.last_level() - (i as u32 * u64::BITS + bit) / LEVEL_BITS;
        }
    }
    0
}

/// A concurrent canonical store of predicates and partitions.
///
/// Nonconstant roots carry at most `u32::MAX - 2` external references.
/// Exceeding that limit or releasing a nonconstant root with no external
/// references panics. Equal maps of one sort share a token in this store.
/// Image and preimage apply bit maps symbolically and panic on Rust closures.
///
/// # Examples
///
/// A member cursor keeps the predicate live after its caller releases it.
///
/// ```
/// use std::sync::Arc;
///
/// use sra::backend::Backend;
/// use sra::bdd::BddStore;
/// use sra::scalar::ScalarSort;
///
/// let store = Arc::new(BddStore::default());
/// let alg = store.bind(ScalarSort::bv(8)).unwrap();
/// let p = alg.set(&[2, 5]);
/// let mut cursor = alg.members(p);
/// alg.dec_ref(p);
/// assert_eq!(cursor.next(), Some(&[2][..]));
/// assert_eq!(cursor.next(), Some(&[5][..]));
/// assert_eq!(cursor.next(), None);
/// ```
#[derive(Debug)]
pub struct BddStore {
    shared: Shared<State>,
    threads: usize,
}

impl BddStore {
    /// An empty store with initial capacities and a maintenance thread budget.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero, `nodes` is below two, either capacity
    /// exceeds `u32::MAX`, or the unique table size overflows `usize`.
    pub fn with_capacity(nodes: usize, edges: usize, threads: usize) -> Self {
        assert!(threads != 0, "maintenance thread count must be nonzero");
        let terminals = [Function::FALSE, Function::TRUE].map(|terminal| Node {
            data: terminal.index() as u32,
        });
        Self {
            shared: Shared::new(Dag::with_capacity(nodes, edges, terminals)),
            threads,
        }
    }

    /// Admits a reader with this store's maintenance thread budget.
    #[inline(always)]
    fn read(&self) -> Read<'_, State> {
        self.shared.read(self.threads)
    }

    /// Runs `build` until it completes without needing growth, then acquires
    /// one reference to its result.
    #[inline(always)]
    fn owned(&self, mut build: impl FnMut(&State) -> Result<Function, NeedGrowth>) -> Function {
        self.shared.retry(self.threads, |state| {
            let f = build(state)?;
            if !f.is_constant() {
                state.inc_ref(f);
            }
            Ok(f)
        })
    }

    /// Restarts `cache` unless it is current for this store and within bound.
    ///
    /// The caller holds admission through every use of its entries.
    #[inline(always)]
    fn restart_cache(&self, cache: &mut Cache) {
        cache::restart(cache, self.shared.id, self.shared.epoch());
    }
}

impl Default for BddStore {
    /// Creates an empty store containing the canonical Boolean terminals.
    fn default() -> Self {
        Self::with_capacity(NODE_CAPACITY, EDGE_CAPACITY, 1)
    }
}

/// A depth-first traversal of members, expanding skipped levels and terminal
/// suffixes.
///
/// The stack holds tokens below the owned root, and no node borrow or guard
/// survives `next`.
struct BddCursor<'a> {
    /// The algebra that interprets the diagram and retains its store.
    algebra: &'a BddAlgebra,
    /// The owned root, released when the cursor drops.
    root: Function,
    /// The encoded value assembled along the current path.
    value: Vec<u64>,
    /// Each frame holds its subtree, the next assignment (64 after assignment
    /// 63), and the false-edge index. The index is `u8::MAX` before lookup and
    /// when the level has no false edge.
    stack: Vec<(Function, u8, u8)>,
}

impl Cursor for BddCursor<'_> {
    fn next(&mut self) -> Option<&[u64]> {
        self.stack.last()?;
        let span = self.algebra.span;
        self.value.resize(span.sort.words(), 0);
        let state = self.algebra.store.read();
        while let Some(((f, next, empty), parents)) = self.stack.split_last_mut() {
            let level = parents.len() as u32;
            if *next == ASSIGNMENTS as u8 {
                self.stack.pop();
                continue;
            }
            let node = state.node(*f);
            // Only one edge can lead to false. Its index survives compaction,
            // which preserves stored edge order. Every assignment is live at
            // a skipped level.
            if *next == 0 && node.level() == level {
                *empty = node
                    .edges
                    .iter()
                    .position(|e| e.child.is_f())
                    .map_or(u8::MAX, |i| i as u8);
            }
            let live = node
                .edges
                .get(usize::from(*empty))
                .map_or(u64::MAX, |e| !e.mask);
            let rest = live & span.valid_mask(level) & (u64::MAX << *next);
            let Some(assignment) = rest.lowest_one() else {
                self.stack.pop();
                continue;
            };
            *next = assignment as u8 + 1;
            span.assign(level, u64::from(assignment), &mut self.value);
            if level == span.last_level() {
                return Some(&self.value);
            }
            let child = if node.level() == level {
                covering_edge(node.edges, 1 << assignment).child
            } else {
                *f
            };
            self.stack.push((child, 0, u8::MAX));
        }
        None
    }
}

impl Drop for BddCursor<'_> {
    fn drop(&mut self) {
        self.algebra.dec_ref(self.root);
    }
}

/// A BDD store bound to one scalar sort, with the level span its width needs.
struct BddAlgebra {
    store: Arc<BddStore>,
    span: Span,
}

impl BddAlgebra {
    /// Builds the image or preimage of nonempty operands under a checked bit map.
    fn transfer(
        &self,
        clear: &[u64],
        flip: &[u64],
        p: Function,
        q: Function,
        image: bool,
    ) -> Function {
        let span = self.span;
        let last = last_affected(span, clear, flip);
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(p) && state.has_refs(q),
                "cannot use an unreferenced BDD node"
            );
            if let [lower @ .., top] = clear
                && *top == span.sort.top_mask()
                && lower.iter().all(|&word| word == u64::MAX)
            {
                // Every input maps to flip, so one target lookup decides whether
                // the singleton image and its entire source survive.
                return if state.evaluate(span, q, flip) == 0 {
                    Ok(Function::FALSE)
                } else if image {
                    state.try_masked_eq(span, clear, flip)
                } else {
                    Ok(p)
                };
            }

            // Composition reads `q` only at images, whose assigned bits are
            // fixed, so an unreachable output in `q` admits no input.
            let f = state.try_compose(&mut Map::default(), span, clear, flip, last, q)?;
            let mut memo = Memo::default();
            let f = state.try_apply(&mut memo, BoolOp::And, p, f)?;
            if !image || f.is_f() {
                return Ok(f);
            }
            // Only surviving inputs contribute to the union. Remapping leaves
            // cleared bits unconstrained until the output mask fixes them.
            let f = state.try_remap(&mut memo, span, clear, flip, last, f)?;
            // A map that clears nothing has no output bits to fix.
            if clear.iter().all(|&word| word == 0) {
                return Ok(f);
            }
            state.try_apply(
                &mut memo,
                BoolOp::And,
                f,
                state.try_masked_eq(span, clear, flip)?,
            )
        })
    }
}

impl Backend for BddStore {
    /// Accepts every scalar sort.
    fn bind(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Algebra>> {
        Some(Box::new(BddAlgebra {
            span: Span { sort },
            store: self,
        }))
    }
}

impl Algebra for BddAlgebra {
    fn sort(&self) -> ScalarSort {
        self.span.sort
    }

    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.read().inc_ref(f);
        }
    }

    /// Defers reclamation to growth-triggered collection.
    fn dec_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.read().dec_ref(f);
        }
    }

    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        assert!(
            self.sort().accepts(value),
            "value is outside the scalar sort"
        );
        if f.is_constant() {
            return u32::from(f.is_t());
        }
        let state = self.store.read();
        debug_assert!(state.has_refs(f), "cannot use an unreferenced BDD node");
        state.evaluate(self.span, f, value)
    }

    fn count(&self, p: Function) -> Count {
        // The constants have store-independent counts.
        if p.is_f() {
            return Count::ZERO;
        }
        if p.is_t() {
            return Count::from(1) << self.sort().width();
        }
        let span = self.span;
        let state = self.store.read();
        debug_assert!(state.has_refs(p), "cannot use an unreferenced BDD node");
        // Counts start at a node's own level, and an incoming edge supplies the
        // skipped bits.
        let bits = |f| {
            let node = state.node(f);
            if node.is_terminal() {
                0
            } else {
                (span.last_level() - node.level()) * LEVEL_BITS + span.width(node.level())
            }
        };
        // Normalized counts share one digit pool, which starts with the digit of
        // `TRUE`, and one reusable accumulation buffer, so each count occupies
        // only its own digits.
        let mut pool = vec![1];
        let mut sum = Vec::new();
        let mut memo = Map::from_iter([(Function::FALSE, (0, 0)), (Function::TRUE, (0, 1))]);
        let mut stack = Vec::new();
        let (mut f, mut next) = (p, 0);
        'nodes: loop {
            let node = state.node(f);
            assert!(!node.is_terminal(), "count requires a Boolean predicate");
            // Every edge intersects the reachable assignments: `intern_level` aliases
            // positions beyond a short first level to reachable ones and products preserve
            // that closure.
            let valid_mask = span.valid_mask(node.level());
            // Edge masks count this level's assignments, so each shift spans only
            // the levels below it.
            let below = (span.last_level() - node.level()) * LEVEL_BITS;
            while let Some(edge) = node.edges.get(next) {
                next += 1;
                if !memo.contains_key(&edge.child) {
                    stack.push((f, next));
                    (f, next) = (edge.child, 0);
                    continue 'nodes;
                }
            }
            sum.clear();
            for edge in node.edges {
                let (i, len) = memo[&edge.child];
                if len == 0 {
                    continue;
                }
                let shift = below - bits(edge.child);
                // Two extra digits cover the residual shift, a factor of at most 64,
                // and the sum of at most 64 disjoint edges.
                sum.resize(sum.len().max(len + (shift / u64::BITS) as usize + 2), 0);
                count::add_scaled(
                    &mut sum,
                    &pool[i..i + len],
                    u64::from((edge.mask & valid_mask).count_ones()),
                    shift,
                );
            }
            while sum.last() == Some(&0) {
                sum.pop();
            }
            // The root's digits are already in the result buffer.
            let Some(parent) = stack.pop() else { break };
            memo.insert(f, (pool.len(), sum.len()));
            pool.extend_from_slice(&sum);
            (f, next) = parent;
        }
        Count::from_digits(sum) << (self.sort().width() - bits(p))
    }

    fn set(&self, values: &[u64]) -> Function {
        let sort = self.sort();
        scalar::check_packed(sort, values);
        if values.is_empty() {
            return Function::FALSE;
        }
        if covers(sort, values.len() / sort.words()) {
            return Function::TRUE;
        }
        self.store
            .owned(|state| state.try_values(self.span, 0, values, &|_| 1, 0, Function::FALSE))
    }

    fn lt(&self, bound: &[u64]) -> Function {
        assert_ne!(
            self.sort().kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        assert!(
            self.sort().accepts(bound),
            "bound is outside the scalar sort"
        );
        let span = self.span;
        self.store.owned(|state| {
            let mut f = Function::FALSE;
            for level in (0..=span.last_level()).rev() {
                let limit = span.assignment(level, bound);
                if limit == 0 && f.is_f() {
                    continue;
                }
                f = state.intern_level(span, level, |value| match value.cmp(&limit) {
                    Ordering::Less => Function::TRUE,
                    Ordering::Equal => f,
                    Ordering::Greater => Function::FALSE,
                })?;
            }
            Ok(f)
        })
    }

    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function {
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced BDD node"
            );
            state.try_apply(&mut Memo::default(), op, left, right)
        })
    }

    fn masked_eq(&self, mask: &[u64], value: &[u64]) -> Function {
        assert_ne!(
            self.sort().kind(),
            ScalarKind::Symbol,
            "symbols do not support bit predicates"
        );
        assert!(
            self.sort().accepts(mask) && self.sort().accepts(value),
            "bit test is outside the scalar sort"
        );
        assert!(
            mask.iter().zip(value).all(|(m, v)| v & !m == 0),
            "bit test value has bits outside its mask"
        );
        self.store
            .owned(|state| state.try_masked_eq(self.span, mask, value))
    }

    /// Restricts `p` to the inputs whose image `q` accepts and remaps the
    /// surviving assignments. A masked equality test enforces any bits the
    /// map sets to fixed values.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_sort(self.sort());
        let (clear, flip) = bit_map(map);
        // XOR permutes the full domain, and the identity preserves every source.
        if p.is_f()
            || q.is_f()
            || clear.iter().all(|&word| word == 0)
                && (p.is_t() || flip.iter().all(|&word| word == 0))
        {
            return self.apply(BoolOp::And, p, q);
        }
        self.transfer(clear, flip, p, q, true)
    }

    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_sort(self.sort());
        let (clear, flip) = bit_map(map);
        if p.is_f() || q.is_constant() || map.is_identity() {
            return self.apply(BoolOp::And, p, q);
        }
        self.transfer(clear, flip, p, q, false)
    }

    /// The kernel preserves arbitrary terminal labels without canonicalizing
    /// them as a partition.
    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(condition) && state.has_refs(yes) && state.has_refs(no),
                "cannot use an unreferenced BDD node"
            );
            state.try_ite(&mut Memo::default(), condition, yes, no)
        })
    }

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
        let cache = &mut scratch.cache;
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced BDD node"
            );
            self.store.restart_cache(cache);
            cache::align(cache, (left, right), labels, |out, start| {
                scratch.memo.clear();
                state.try_align(&mut scratch.memo, out, start, left, right)
            })
        })
    }

    /// Substituting `classes[i]` for each input class `i` preserves canonical
    /// numbering, since both partitions use order of first appearance.
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Align) -> Function {
        debug_assert!(
            classes
                .iter()
                .try_fold(0, |next, &i| (i <= next).then(|| next.max(i + 1)))
                .is_some(),
            "quotient classes must be numbered in first-appearance order"
        );
        debug_assert!(
            classes
                .iter()
                .enumerate()
                .any(|(i, &class)| class as usize != i),
            "quotient must merge at least two classes"
        );
        let cache = &mut scratch.cache;
        let key = (f, dag::structural_hash(classes));
        self.store.owned(|state| {
            debug_assert!(state.has_refs(f), "cannot use an unreferenced BDD node");
            self.store.restart_cache(cache);
            cache::quotient(cache, key, classes, || {
                scratch.coarse.clear();
                state.try_map(&mut scratch.coarse, f, &mut |i| classes[i as usize])
            })
        })
    }

    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        if cases.is_empty() {
            labels.push(default);
            return Function::FALSE;
        }
        // Each class number indexes its case label in the appended suffix.
        self.store.owned(|state| {
            debug_assert!(
                cases.iter().all(|&(p, _)| state.has_refs(p)),
                "cannot use an unreferenced BDD node",
            );
            if let &[(p, label)] = cases {
                if default == label || p.is_constant() {
                    labels.push(if p.is_f() { default } else { label });
                    return Ok(Function::FALSE);
                }
                // The first terminal determines whether the predicate's 0 and 1
                // already number its classes in order of first appearance.
                let mut first = p;
                while !first.is_constant() {
                    first = state.node(first).edges[0].child;
                }
                if first.is_f() {
                    labels.extend_from_slice(&[default, label]);
                    return Ok(p);
                }
                let result = state.try_not(&mut Map::default(), p)?;
                labels.extend_from_slice(&[label, default]);
                return Ok(result);
            }
            // Fold the cases onto label terminals, then assign class numbers. Check
            // explicit default-labeled cases last: before that, a default terminal also
            // denotes points no case has covered and cannot establish conflicting overlap.
            let mut memo = Map::default();
            let mut acc = state.intern_terminal(default)?;
            for &(p, label) in cases
                .iter()
                .filter(|&&(_, label)| label != default)
                .chain(cases.iter().filter(|&&(_, label)| label == default))
            {
                memo.clear();
                acc = state.try_select(&mut memo, default, p, label, acc)?;
            }
            state.try_canonicalize(acc, labels)
        })
    }

    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        labels: &mut Vec<u32>,
        _: &mut Values,
    ) -> Function {
        let sort = self.sort();
        let words = sort.words();
        scalar::check_packed(sort, keys);
        assert_eq!(
            case_labels.len(),
            keys.len() / words,
            "there must be one case label per key"
        );
        let Some((&first, rest)) = case_labels.split_first() else {
            labels.push(default);
            return Function::FALSE;
        };
        let full = covers(sort, case_labels.len());
        if full && rest.iter().all(|&label| label == first) {
            labels.push(first);
            return Function::FALSE;
        }
        let gap = if full {
            case_labels.len()
        } else {
            // Consecutive keys from zero equal their record indices. The default
            // first appears at the first gap, including a gap after the last key.
            keys.chunks_exact(words)
                .enumerate()
                .position(|(i, key)| key[0] != i as u64 || key[1..].iter().any(|&word| word != 0))
                .unwrap_or(case_labels.len())
        };
        let missing = (!full).then_some(&default);
        let (before, after) = case_labels.split_at(gap);
        // Retries reuse the classes. Keep their labels once the root is owned.
        let start = labels.len();
        let mut out = mem::DropGuard::new(labels, |out| out.truncate(start));
        let mut classes = Map::default();
        for &label in before.iter().chain(missing).chain(after) {
            class(&mut classes, &mut out, label);
        }
        let f = self.store.owned(|state| {
            state.try_values(
                self.span,
                0,
                keys,
                &|i| classes[&case_labels[i]],
                0,
                // On full coverage every assignment gets a key's class. Zero
                // is a pinned placeholder for the unused default child.
                state.intern_terminal(missing.map_or(0, |label| classes[label]))?,
            )
        });
        mem::DropGuard::dismiss(out);
        f
    }

    /// Discovery pauses at each selected class and releases its guard before
    /// calling `then`. Each `build` call enters the store separately and can
    /// retry from the original roots after maintenance.
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
                self.store.read().has_refs(f),
                "cannot use an unreferenced BDD node"
            );
            return ControlFlow::Continue(());
        }
        let Refine { stack, seen, memo } = scratch;
        stack.clear();
        stack.push((f, p));
        seen.clear();
        loop {
            let class = 'class: {
                let state = self.store.read();
                debug_assert!(
                    state.has_refs(f) && state.has_refs(p),
                    "cannot use an unreferenced BDD node"
                );
                while let Some((l, r)) = stack.pop() {
                    if r.is_f() {
                        continue;
                    }
                    // Every nonempty predicate reaches true, so a terminal partition side
                    // has reached its class without walking the rest of `r`.
                    if let Some(value) = state.node(l).terminal() {
                        if seen.insert((l, Function::TRUE)) && select(value) {
                            break 'class value;
                        }
                        continue;
                    }
                    if !seen.insert((l, r)) {
                        continue;
                    }
                    let start = stack.len();
                    state.cofactors([l, r], |_, [a, b]| stack.push((a, b)));
                    // Pop the new pairs in order of first assignment.
                    stack[start..].reverse();
                }
                return ControlFlow::Continue(());
            };
            then(class, &mut || {
                self.store.owned(|state| {
                    memo.clear();
                    state.try_cell(memo, f, p, class)
                })
            })?;
        }
    }

    fn members(&self, p: Function) -> Box<dyn Cursor + '_> {
        debug_assert!(
            self.store.read().has_refs(p),
            "cannot use an unreferenced BDD node"
        );
        self.inc_ref(p);
        Box::new(BddCursor {
            algebra: self,
            root: p,
            value: Vec::new(),
            stack: if p.is_f() {
                Vec::new()
            } else {
                vec![(p, 0, u8::MAX)]
            },
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/bdd.rs"]
mod tests;
