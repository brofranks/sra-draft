//! Canonical decision diagrams with 6 bits per level.
//!
//! A branch tests one level. Its disjoint `(mask, child)` edges cover all 64
//! assignments, in order of each edge's first assignment. A terminal has no
//! edges and stores a `u32` value. Predicates use the terminals
//! [`Function::FALSE`] (0) and [`Function::TRUE`] (1). Partition terminals hold
//! dense class numbers in order of first appearance. The [`BddStore`] type
//! implements [`Backend`], and `Span` places an encoded scalar value on levels.
//!
//! Construction and transformation recurse with diagram depth, so their stack
//! use grows with the number of levels traversed.

use std::array;
use std::cmp::Ordering;
use std::mem::{self, MaybeUninit};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, StoreCache, Values,
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
const LEVEL_VALUES: usize = 1 << LEVEL_BITS;
/// The initial node capacity of the default store.
const NODE_CAPACITY: usize = 4_096;
/// The initial edge capacity of the default store.
const EDGE_CAPACITY: usize = 16_384;

const _: () = assert!(LEVEL_VALUES == u64::BITS as usize);

/// The six-bit levels occupied by one scalar sort, starting at level zero.
///
/// Bit 0 of the value sits at position 0 of `last_level`, and higher bits fill
/// higher positions and then earlier levels, so only level zero may be short.
/// Assignments beyond a short level's width alias the assignment obtained by
/// clearing its unused high bits, as normalized by [`State::intern_level`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
    fn reach(self, level: u32) -> u32 {
        (self.sort.width() - (self.last_level() - level) * LEVEL_BITS).min(LEVEL_BITS)
    }

    /// The mask of the assignments at `level` that some value reaches.
    #[inline(always)]
    fn valid(self, level: u32) -> u64 {
        u64::MAX >> (u64::BITS - (1 << self.reach(level)))
    }

    /// The `(word, shift, reach)` locating the bits `level` holds inside an encoded
    /// value.
    #[inline(always)]
    fn place(self, level: u32) -> (usize, u32, u32) {
        let low = (self.last_level() - level) * LEVEL_BITS;
        (
            (low / u64::BITS) as usize,
            low % u64::BITS,
            self.reach(level),
        )
    }

    /// The assignment of `level` under the encoded `value`: its bits at this level,
    /// least significant first.
    #[inline(always)]
    fn assignment(self, level: u32, value: &[u64]) -> usize {
        let (word, shift, reach) = self.place(level);
        let mut bits = value[word] >> shift;
        if shift + reach > u64::BITS {
            bits |= value[word + 1] << (u64::BITS - shift);
        }
        (bits & ((1 << reach) - 1)) as usize
    }

    /// Writes `assignment` into the bits of the encoded `value` that `level` holds.
    ///
    /// Only reachable bits of a short first level are written, including across
    /// word bounds.
    #[inline(always)]
    fn assign(self, level: u32, assignment: u64, value: &mut [u64]) {
        let (word, shift, reach) = self.place(level);
        value[word] = (value[word] & !(((1_u64 << reach) - 1) << shift)) | (assignment << shift);
        if shift + reach > u64::BITS {
            value[word + 1] = (value[word + 1] & !((1_u64 << (shift + reach - u64::BITS)) - 1))
                | (assignment >> (u64::BITS - shift));
        }
    }
}

/// The level that orders terminals below every tested level.
///
/// Branch nodes always store a tested level.
///
/// A terminal may still store `u32::MAX` as its value because its empty edge
/// span identifies it.
const TERMINAL_LEVEL: u32 = u32::MAX;

/// A diagram node: a terminal value or a tested level.
#[derive(Debug)]
struct Node {
    /// The terminal value of a zero-edge node, otherwise the tested level of a
    /// branch, read through [`NodeRef::terminal`] and [`NodeRef::level`].
    data: u32,
}

const _: () = assert!(size_of::<Node>() == size_of::<u32>());

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

    /// The value of a terminal node.
    #[inline(always)]
    fn terminal(&self) -> u32 {
        debug_assert!(self.is_terminal(), "only a terminal stores a value");
        self.data
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
    items: [MaybeUninit<T>; LEVEL_VALUES],
    len: usize,
}

impl<T: Copy> LevelBuffer<T> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            items: [MaybeUninit::uninit(); LEVEL_VALUES],
            len: 0,
        }
    }

    /// Appends one item.
    ///
    /// # Panics
    ///
    /// Panics if the buffer already holds [`LEVEL_VALUES`] items.
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
    ite: Map<(Function, Function, Function), Function>,
    /// Both directions of complementation within this attempt.
    not: Map<Function, Function>,
    /// Remapped roots for one span and one pair of clear and flip masks.
    remap: Map<Function, Function>,
}

impl State {
    /// Interns the canonical terminal for a value without acquiring an
    /// external reference.
    fn intern_terminal(&self, value: u32) -> Result<Function, NeedGrowth> {
        if value <= 1 {
            return Ok(Function::new(value as usize));
        }
        self.intern_node(value, &[])
    }

    /// The canonical representation of a terminal or an already ordered complete
    /// branch, without an external reference.
    ///
    /// A branch with one distinct child reduces to that child.
    fn intern_node(&self, data: u32, edges: &[Edge]) -> Result<Function, NeedGrowth> {
        // A branch's nonempty, disjoint masks cover all 64 assignments in order
        // of first assignment. Each child is distinct and is either a terminal
        // or a branch at a later level.
        if edges.len() == 1 {
            return Ok(edges[0].child);
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
        let mask = (1 << span.reach(level)) - 1;
        for assignment in 0..LEVEL_VALUES {
            edges.merge(1 << assignment, child(assignment & mask));
        }
        self.intern_node(level, edges.as_slice())
    }

    /// Ordered nonempty product cofactors.
    ///
    /// Deeper operands contribute one full-domain edge. Discovery and
    /// materialization use the same level and edge-order convention as alignment.
    #[inline(always)]
    fn cofactors(
        &self,
        left: Function,
        right: Function,
        cofactors: &mut LevelBuffer<(u64, Function, Function)>,
    ) -> u32 {
        let nodes = [self.node(left), self.node(right)];
        let level = nodes[0].level().min(nodes[1].level());
        debug_assert_ne!(level, TERMINAL_LEVEL, "cofactors need a branch operand");
        let whole = [left, right].map(whole_edge);
        let [left_edges, right_edges]: [&[Edge]; 2] = array::from_fn(|i| {
            if nodes[i].level() == level {
                nodes[i].edges
            } else {
                &whole[i]
            }
        });
        for &l in left_edges {
            for &r in right_edges {
                let mask = l.mask & r.mask;
                if mask != 0 {
                    cofactors.push((mask, l.child, r.child));
                }
            }
        }
        // Intersecting edge lists can change their order. Sort by first assignment
        // so result edges and class numbers stay canonical. Already ordered
        // intersections, including those with a full mask, skip the sort.
        let by_first = |&(mask, _, _): &(u64, Function, Function)| mask.trailing_zeros();
        if !cofactors.as_slice().is_sorted_by_key(by_first) {
            cofactors.as_mut_slice().sort_unstable_by_key(by_first);
        }
        level
    }

    /// Interns the product of two nodes at their upper level, recursing on every
    /// nonempty cofactor in order of first assignment so result edges and class
    /// numbers are canonical.
    fn try_product(
        &self,
        left: Function,
        right: Function,
        mut recurse: impl FnMut(Function, Function) -> Result<Function, NeedGrowth>,
    ) -> Result<Function, NeedGrowth> {
        let mut cofactors = LevelBuffer::new();
        let level = self.cofactors(left, right, &mut cofactors);
        let mut edges = LevelBuffer::new();
        for &(mask, l, r) in cofactors.as_slice() {
            edges.merge(mask, recurse(l, r)?);
        }
        // An operand at this level whose edges the result reproduces is that result, so
        // unchanged subgraphs are shared without a unique-table probe.
        let same = |f: Function| {
            let node = self.node(f);
            node.level() == level && node.edges == edges.as_slice()
        };
        if same(left) {
            return Ok(left);
        }
        if same(right) {
            return Ok(right);
        }
        self.intern_node(level, edges.as_slice())
    }

    /// Combines a Boolean condition and two branches at their earliest level.
    ///
    /// Each nonempty intersection of the three edge masks selects one child
    /// from each operand. Deeper operands keep their current root at this level.
    fn try_ite(
        &self,
        memo: &mut Memo,
        condition: Function,
        yes: Function,
        no: Function,
    ) -> Result<Function, NeedGrowth> {
        if condition.is_t() || yes == no {
            return Ok(yes);
        }
        if condition.is_f() {
            return Ok(no);
        }
        if yes.is_t() && no.is_f() {
            return Ok(condition);
        }
        if yes.is_f() && no.is_t() {
            return self.try_not(memo, condition);
        }
        let key = (condition, yes, no);
        if let Some(&result) = memo.ite.get(&key) {
            return Ok(result);
        }
        let nodes = [self.node(condition), self.node(yes), self.node(no)];
        let level = nodes
            .iter()
            .map(NodeRef::level)
            .min()
            .expect("three nodes must have a minimum level");
        let whole = [condition, yes, no].map(whole_edge);
        let parts: [&[Edge]; 3] = array::from_fn(|i| {
            if nodes[i].level() == level {
                nodes[i].edges
            } else {
                &whole[i]
            }
        });
        let mut edges = LevelBuffer::new();
        let mut remaining = u64::MAX;
        while remaining != 0 {
            let bit = 1_u64 << remaining.trailing_zeros();
            let selected = parts.map(|part| covering_edge(part, bit));
            let mask = selected.iter().fold(u64::MAX, |acc, edge| acc & edge.mask);
            let child = self.try_ite(
                memo,
                selected[0].child,
                selected[1].child,
                selected[2].child,
            )?;
            edges.merge(mask, child);
            remaining &= !mask;
        }
        let result = self.intern_node(level, edges.as_slice())?;
        memo.ite.insert(key, result);
        Ok(result)
    }

    /// Complements a Boolean predicate without reclassifying its distinct children.
    ///
    /// Negation is expressible as `try_map(&mut memo.not, p, &mut |value| value ^
    /// 1)`, but this path skips the edge-merging scan a general terminal map needs
    /// and memoizes the involution in both directions.
    fn try_not(&self, memo: &mut Memo, p: Function) -> Result<Function, NeedGrowth> {
        match p {
            Function::FALSE => return Ok(Function::TRUE),
            Function::TRUE => return Ok(Function::FALSE),
            _ => {}
        }
        if let Some(&result) = memo.not.get(&p) {
            return Ok(result);
        }

        let node = self.node(p);
        let mut edges = LevelBuffer::new();
        for edge in node.edges {
            edges.push(Edge {
                mask: edge.mask,
                child: self.try_not(memo, edge.child)?,
            });
        }

        // Negation is bijective, so edge grouping, canonical order, and the level
        // remain unchanged.
        let result = self.intern_node(node.data, edges.as_slice())?;
        memo.not.insert(p, result);
        memo.not.insert(result, p);
        Ok(result)
    }

    /// The Boolean combination `op` of two predicates, memoized per operation.
    ///
    /// Commutative operands are ordered so equal pairs share one entry.
    fn try_apply_bool(
        &self,
        memo: &mut Memo,
        op: BoolOp,
        mut left: Function,
        mut right: Function,
    ) -> Result<Function, NeedGrowth> {
        // Canonicalize to increase odds of memo reuse.
        if op.is_commutative() && right < left {
            mem::swap(&mut left, &mut right);
        }
        if let Some(result) = op.simplify(left, right) {
            return Ok(result);
        }
        if left.is_t() && matches!(op, BoolOp::AndNot | BoolOp::Xor) {
            return self.try_not(memo, right);
        }
        let key = (op, left, right);
        if let Some(&result) = memo.apply.get(&key) {
            return Ok(result);
        }

        let result = self.try_product(left, right, |l, r| self.try_apply_bool(memo, op, l, r))?;
        memo.apply.insert(key, result);
        Ok(result)
    }

    /// Remaps assignments within each affected level.
    ///
    /// Assignments differing only in cleared bits share the disjunction of their
    /// children, making those bits unconstrained. Flipping any remaining bit
    /// exchanges the assignments that differ in that bit.
    ///
    /// Subgraphs below `last` contain only unaffected bits and can be reused.
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
        let result = if level_clear == 0 && level_flip == 0 {
            // An untouched level keeps its edges, merging those whose children
            // collapse together, as `try_map` does.
            let mut edges = LevelBuffer::new();
            for edge in node.edges {
                edges.merge(
                    edge.mask,
                    self.try_remap(memo, span, clear, flip, last, edge.child)?,
                );
            }
            if node.edges == edges.as_slice() {
                f
            } else {
                self.intern_node(level, edges.as_slice())?
            }
        } else {
            let mut children = [Function::FALSE; LEVEL_VALUES];
            for edge in node.edges {
                let child = self.try_remap(memo, span, clear, flip, last, edge.child)?;
                let mut remaining = edge.mask & span.valid(level);
                while remaining != 0 {
                    let a = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let target = &mut children[(a ^ level_flip) & !level_clear];
                    *target = self.try_apply_bool(memo, BoolOp::Or, *target, child)?;
                }
            }
            self.intern_level(span, level, |a| children[a & !level_clear])?
        };
        memo.remap.insert(f, result);
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
        if p.is_f() {
            return Ok(Function::FALSE);
        }
        let node = self.node(f);
        if node.is_terminal() {
            return Ok(if node.terminal() == class {
                p
            } else {
                Function::FALSE
            });
        }
        if let Some(&result) = memo.get(&(f, p)) {
            return Ok(result);
        }
        let result = self.try_product(f, p, |l, r| self.try_cell(memo, l, r, class))?;
        memo.insert((f, p), result);
        Ok(result)
    }

    /// Rebuilds `f` with every partition terminal `i` replaced by `map(i)`, merging
    /// edges whose children collapse together and reusing every subgraph the map
    /// leaves unchanged.
    ///
    /// A bijective map is a pure renumbering, and a non-injective one coarsens
    /// the partition.
    fn try_map(
        &self,
        memo: &mut Map<Function, Function>,
        f: Function,
        map: &mut impl FnMut(u32) -> u32,
    ) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        if node.is_terminal() {
            let value = node.terminal();
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

        let mut edges = LevelBuffer::new();
        for edge in node.edges {
            edges.merge(edge.mask, self.try_map(memo, edge.child, map)?);
        }
        let result = if node.edges == edges.as_slice() {
            f
        } else {
            self.intern_node(node.data, edges.as_slice())?
        };
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
        let key = (left, right);
        if let Some(&result) = memo.get(&key) {
            return Ok(result);
        }

        let left_node = self.node(left);
        let right_node = self.node(right);
        let result = if left_node.is_terminal() && right_node.is_terminal() {
            labels.push((left_node.terminal(), right_node.terminal()));
            self.intern_terminal((labels.len() - start) as u32 - 1)?
        } else {
            self.try_product(left, right, |l, r| {
                self.try_align(memo, labels, start, l, r)
            })?
        };
        memo.insert(key, result);
        Ok(result)
    }

    /// Overwrites the points where predicate `p` holds with the terminal for
    /// `label` inside the label-valued accumulator `acc`, asserting that it only
    /// covers `default` or same-label terminals.
    fn try_select(
        &self,
        memo: &mut Map<(Function, Function), Function>,
        default: u32,
        p: Function,
        label: u32,
        acc: Function,
    ) -> Result<Function, NeedGrowth> {
        if p.is_f() {
            return Ok(acc);
        }
        let acc_node = self.node(acc);
        if acc_node.is_terminal() {
            let covered = acc_node.terminal();
            if covered == label {
                return Ok(acc);
            }
            if p.is_t() {
                assert_eq!(covered, default, "cases with different labels overlap");
                return self.intern_terminal(label);
            }
        }

        let key = (p, acc);
        if let Some(&result) = memo.get(&key) {
            return Ok(result);
        }
        let result =
            self.try_product(p, acc, |q, a| self.try_select(memo, default, q, label, a))?;
        memo.insert(key, result);
        Ok(result)
    }

    /// The map over `span` sending each value of the sorted, distinct `keys` to its
    /// label terminal and every other value to the interned `default` terminal,
    /// built from `level` downwards without an external reference.
    ///
    /// Every case shares the bits above `level`. The `label` callback indexes the
    /// original key sequence, with `offset` locating this slice within it.
    fn try_values(
        &self,
        span: Span,
        level: u32,
        keys: &[u64],
        label: &impl Fn(usize) -> u32,
        offset: usize,
        default: Function,
    ) -> Result<Function, NeedGrowth> {
        if keys.is_empty() {
            return Ok(default);
        }
        let words = span.sort.words();
        let len = keys.len() / words;
        if level > span.last_level() {
            debug_assert_eq!(len, 1, "exactly one key must remain below the last level");
            return self.intern_terminal(label(offset));
        }
        let mut children = [default; LEVEL_VALUES];
        let mut start = 0;
        while start < len {
            let assignment = span.assignment(level, &keys[start * words..][..words]);
            let mut end = start + 1;
            while end < len && span.assignment(level, &keys[end * words..][..words]) == assignment {
                end += 1;
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
    /// Appends one label per class after construction succeeds and returns
    /// the root without acquiring an external reference.
    fn try_canonicalize(&self, f: Function, labels: &mut Vec<u32>) -> Result<Function, NeedGrowth> {
        // Depth-first traversal in stored edge order preserves the numbering of
        // an already canonical partition.
        let mut first = Vec::new();
        let mut class_of = Map::default();
        let result = self.try_map(&mut Map::default(), f, &mut |label| {
            *class_of.entry(label).or_insert_with(|| {
                first.push(label);
                first.len() as u32 - 1
            })
        })?;
        labels.extend_from_slice(&first);
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

/// The deepest level of `span` holding a bit of `clear` or `flip`, or level
/// zero when neither holds any.
fn last_affected(span: Span, clear: &[u64], flip: &[u64]) -> u32 {
    (0..=span.last_level())
        .rev()
        .find(|&level| span.assignment(level, clear) | span.assignment(level, flip) != 0)
        .unwrap_or(0)
}

/// A concurrent canonical store of predicates and partitions with 6 bits per
/// level.
///
/// Roots are [`Function`] tokens with manual external reference counts, which
/// callers pair with this store. Equal maps share one token, so token
/// identity decides equality within one column.
#[derive(Debug)]
pub struct BddStore {
    shared: Shared<State>,
    threads: usize,
}

impl BddStore {
    /// An empty store holding the Boolean terminals, with initial node and edge
    /// capacities and a maintenance thread budget.
    ///
    /// # Panics
    ///
    /// Panics if `threads` is zero, `nodes` is below two, either capacity
    /// exceeds `u32::MAX`, or the unique table's bucket count for the node
    /// capacity overflows `usize`.
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

    /// Restarts a caller-held cache unless it already belongs to this store at
    /// its current maintenance epoch and holds no more entries than its bound.
    ///
    /// The caller must hold admission through every use of the cache entries.
    #[inline(always)]
    fn restart_cache(&self, cache: &mut StoreCache) {
        cache.restart(self.shared.id, self.shared.epoch());
    }
}

impl Default for BddStore {
    /// Creates an empty store containing the canonical Boolean terminals.
    fn default() -> Self {
        Self::with_capacity(NODE_CAPACITY, EDGE_CAPACITY, 1)
    }
}

/// A depth-first traversal of members, including skipped levels and terminal
/// suffixes.
///
/// Stack entries hold tokens below the owned root, and no borrowed node or
/// store guard survives a call to `next`.
struct BddMembers<'a> {
    /// The column whose store holds the diagram.
    column: &'a BddColumn,
    /// The owned root, released when the cursor drops.
    root: Function,
    /// The encoded value assembled along the current path.
    value: Vec<u64>,
    /// Frames holding the level, subtree root, and next assignment to try. A
    /// next assignment of `u64::BITS` marks an exhausted frame.
    stack: Vec<(u32, Function, u64)>,
}

impl Members for BddMembers<'_> {
    fn next(&mut self) -> Option<&[u64]> {
        self.stack.last()?;
        self.value.resize(self.column.sort().words(), 0);
        let state = self.column.store.read();
        while let Some((level, f, next)) = self.stack.last_mut() {
            if *next >= u64::from(u64::BITS) {
                self.stack.pop();
                continue;
            }
            let node = state.node(*f);
            // Live edge masks skip assignments with empty children. At a skipped
            // level every valid assignment reaches the same subtree.
            let live = if node.level() == *level {
                node.edges
                    .iter()
                    .filter(|e| !e.child.is_f())
                    .fold(0, |m, e| m | e.mask)
            } else {
                u64::MAX
            };
            let rest = live & self.column.span.valid(*level) & (u64::MAX << *next);
            if rest == 0 {
                self.stack.pop();
                continue;
            }
            let assignment = u64::from(rest.trailing_zeros());
            *next = assignment + 1;
            let child = if node.level() == *level {
                covering_edge(node.edges, 1 << assignment).child
            } else {
                *f
            };
            self.column.span.assign(*level, assignment, &mut self.value);
            if *level == self.column.span.last_level() {
                return Some(&self.value);
            }
            let next_level = *level + 1;
            self.stack.push((next_level, child, 0));
        }
        None
    }
}

impl Drop for BddMembers<'_> {
    fn drop(&mut self) {
        self.column.dec_ref(self.root);
    }
}

/// A BDD store bound to one scalar sort, with the level span its width needs.
struct BddColumn {
    store: Arc<BddStore>,
    span: Span,
}

impl Backend for BddStore {
    /// Every scalar sort is a bit string over as many levels as its width needs.
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        Some(Box::new(BddColumn {
            span: Span { sort },
            store: self,
        }))
    }
}

impl Column for BddColumn {
    /// The bound scalar sort.
    fn sort(&self) -> ScalarSort {
        self.span.sort
    }

    /// Increments the manually managed external reference count of a live function.
    ///
    /// An existing owner must keep the root live while this call acquires another
    /// reference. The caller must balance each call with [`Self::dec_ref`].
    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.read().inc_ref(f);
        }
    }

    /// Decrements the manually managed external reference count of a function.
    ///
    /// Reclamation is deferred to growth-triggered collection.
    fn dec_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.read().dec_ref(f);
        }
    }

    /// The terminal of `f` at one value of the bound sort.
    fn evaluate(&self, mut f: Function, value: &[u64]) -> u32 {
        assert!(
            self.sort().accepts(value),
            "value is outside the column sort"
        );
        let span = self.span;
        let state = self.store.read();
        debug_assert!(state.has_refs(f), "cannot use an unreferenced BDD node");
        loop {
            let node = state.node(f);
            if node.is_terminal() {
                return node.terminal();
            }
            let assignment = span.assignment(node.level(), value);
            f = covering_edge(node.edges, 1_u64 << assignment).child;
        }
    }

    /// The number of members of predicate `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p` reaches a terminal other than the Boolean constants.
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
                (span.last_level() - node.level()) * LEVEL_BITS + span.reach(node.level())
            }
        };
        // Normalized counts share one digit pool, which starts with the digit of
        // `TRUE`, and one reusable accumulation buffer, so each count occupies
        // only its own digits.
        let mut pool = vec![1];
        let mut sum = Vec::new();
        let mut memo = Map::from_iter([(Function::FALSE, (0, 0)), (Function::TRUE, (0, 1))]);
        let mut stack = vec![(p, 0)];
        'nodes: while let Some((f, next)) = stack.last_mut() {
            let f = *f;
            let node = state.node(f);
            assert!(!node.is_terminal(), "count requires a Boolean predicate");
            // Every edge intersects the reachable assignments: `intern_level` aliases
            // positions beyond a short first level to reachable ones and products preserve
            // that closure.
            let valid = span.valid(node.level());
            // Edge masks count this level's assignments, so each shift spans only
            // the levels below it.
            let below = (span.last_level() - node.level()) * LEVEL_BITS;
            while let Some(edge) = node.edges.get(*next) {
                *next += 1;
                if !memo.contains_key(&edge.child) {
                    stack.push((edge.child, 0));
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
                // Two extra digits cover the residual shift, a factor of at
                // most 64, and the sum of at most 64 disjoint edges.
                sum.resize(sum.len().max(len + (shift / u64::BITS) as usize + 2), 0);
                count::add_scaled(
                    &mut sum,
                    &pool[i..i + len],
                    u64::from((edge.mask & valid).count_ones()),
                    shift,
                );
            }
            while sum.last() == Some(&0) {
                sum.pop();
            }
            // The root's digits are already in the result buffer.
            if f == p {
                break;
            }
            memo.insert(f, (pool.len(), sum.len()));
            pool.extend_from_slice(&sum);
            stack.pop();
        }
        Count::from_digits(sum) << (self.sort().width() - bits(p))
    }

    /// Returns one owned predicate holding exactly where the value takes one of
    /// the sorted, distinct `values`.
    fn set(&self, values: &[u64]) -> Function {
        scalar::check_packed(self.sort(), values);
        self.store
            .owned(|state| state.try_values(self.span, 0, values, &|_| 1, 0, Function::FALSE))
    }

    /// The predicate holding where the value is below `bound`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering, or
    /// `bound` is outside the sort.
    fn less_than(&self, bound: &[u64]) -> Function {
        assert_ne!(
            self.sort().kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        assert!(
            self.sort().accepts(bound),
            "bound is outside the column sort"
        );
        let span = self.span;
        self.store.owned(|state| {
            let mut f = Function::FALSE;
            for level in (0..=span.last_level()).rev() {
                let limit = span.assignment(level, bound);
                f = state.intern_level(span, level, |value| match value.cmp(&limit) {
                    Ordering::Less => Function::TRUE,
                    Ordering::Equal => f,
                    Ordering::Greater => Function::FALSE,
                })?;
            }
            Ok(f)
        })
    }

    /// Returns one owned Boolean combination of two predicates.
    ///
    /// The `AndNot` operation is directional: it computes `left and !right`.
    fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> Function {
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced BDD node"
            );
            state.try_apply_bool(&mut Memo::default(), op, left, right)
        })
    }

    /// The predicate `(x & mask) == value`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, whose values carry no bit structure,
    /// `mask` or `value` is outside the sort, or `value` has a bit outside
    /// `mask`.
    fn masked_eq(&self, mask: &[u64], value: &[u64]) -> Function {
        assert_ne!(
            self.sort().kind(),
            ScalarKind::Symbol,
            "symbols do not support bit predicates"
        );
        assert!(
            self.sort().accepts(mask) && self.sort().accepts(value),
            "bit test is outside the column sort"
        );
        assert!(
            mask.iter().zip(value).all(|(m, v)| v & !m == 0),
            "bit test value has bits outside its mask"
        );
        self.store
            .owned(|state| state.try_masked_eq(self.span, mask, value))
    }

    /// The image of `p` under the bit map, restricted to `q`.
    ///
    /// Remaps assignments within each affected level, then fixes the assigned
    /// output bits with a masked equality test.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort or is a Rust closure.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_domain(self.sort());
        let (clear, flip) = bit_map(map);
        if p.is_f() || q.is_f() || map.is_identity() {
            return self.apply_bool(BoolOp::And, p, q);
        }
        let span = self.span;
        let last = last_affected(span, clear, flip);
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(p) && state.has_refs(q),
                "cannot use an unreferenced BDD node"
            );
            let mut memo = Memo::default();
            let f = state.try_remap(&mut memo, span, clear, flip, last, p)?;
            let f = state.try_apply_bool(
                &mut memo,
                BoolOp::And,
                f,
                state.try_masked_eq(span, clear, flip)?,
            )?;
            state.try_apply_bool(&mut memo, BoolOp::And, f, q)
        })
    }

    /// The part of `p` whose image under the bit map satisfies `q`.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort or is a Rust closure.
    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_domain(self.sort());
        let (clear, flip) = bit_map(map);
        if p.is_f() || q.is_constant() || map.is_identity() {
            return self.apply_bool(BoolOp::And, p, q);
        }
        let span = self.span;
        let last = last_affected(span, clear, flip);
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(p) && state.has_refs(q),
                "cannot use an unreferenced BDD node"
            );
            let mut memo = Memo::default();
            // Fix the assigned output bits before merging assignments. Otherwise
            // an unreachable output in `q` could admit an input to the preimage.
            let f = state.try_apply_bool(
                &mut memo,
                BoolOp::And,
                q,
                state.try_masked_eq(span, clear, flip)?,
            )?;
            let f = state.try_remap(&mut memo, span, clear, flip, last, f)?;
            state.try_apply_bool(&mut memo, BoolOp::And, p, f)
        })
    }

    /// The Boolean selection between two predicates.
    ///
    /// The underlying kernel preserves arbitrary terminal labels. Producing a
    /// canonical partition from such labels requires separate canonicalization.
    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(condition) && state.has_refs(yes) && state.has_refs(no),
                "cannot use an unreferenced BDD node"
            );
            state.try_ite(&mut Memo::default(), condition, yes, no)
        })
    }

    /// Returns one owned common refinement of two distinct, nontrivial
    /// partitions.
    ///
    /// Appends the input class numbers for each result class to `labels`, as
    /// specified by [`Column::align`].
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
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
            cache::align(cache, (left, right), labels, |labels, start| {
                scratch.memo.clear();
                state.try_align(&mut scratch.memo, labels, start, left, right)
            })
        })
    }

    /// The coarsening of partition `f` by `classes`.
    ///
    /// Substituting `classes[i]` for each input class `i` preserves canonical
    /// numbering because both partitions use order of first appearance. See
    /// [`Column::quotient`] for the coarsening contract.
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Alignment) -> Function {
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
                scratch.map.clear();
                state.try_map(&mut scratch.map, f, &mut |i| classes[i as usize])
            })
        })
    }

    /// The partition built from case predicates, appending `labels` as
    /// specified by [`Column::cases`].
    ///
    /// # Panics
    ///
    /// Panics if overlapping cases carry different labels.
    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        // Each class number indexes its case label in the appended suffix.
        if cases.is_empty() {
            labels.push(default);
            return Function::FALSE;
        }
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
                let result = state.try_not(&mut Memo::default(), p)?;
                labels.extend_from_slice(&[label, default]);
                return Ok(result);
            }
            // Fold the cases onto label terminals, then assign class numbers.
            // Check explicit default-labeled cases last: before that, a default terminal
            // also denotes points no case has covered and cannot establish conflicting
            // overlap.
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

    /// The partition labeling the sorted, distinct `keys` by `case_labels`, which
    /// are free of `default`, the label unlisted values use.
    ///
    /// Appends each reachable label to `labels` as specified by [`Column::values`].
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        labels: &mut Vec<u32>,
        _: &mut Values,
    ) -> Function {
        scalar::check_packed(self.sort(), keys);
        assert_eq!(
            case_labels.len(),
            keys.len() / self.sort().words(),
            "there must be one case label per key"
        );
        if keys.is_empty() {
            labels.push(default);
            return Function::FALSE;
        }
        self.store.owned(|state| {
            let f = state.try_values(
                self.span,
                0,
                keys,
                &|i| case_labels[i],
                0,
                state.intern_terminal(default)?,
            )?;
            state.try_canonicalize(f, labels)
        })
    }

    /// Splits predicate `p` by partition `f`, as specified by [`Column::refine`].
    ///
    /// Discovery pauses at each selected class and releases its guard before
    /// calling `then`. A `Break` leaves pending pairs unvisited. Each call to
    /// `build` enters the store separately and can retry from the original roots
    /// after maintenance.
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
                self.store.read().has_refs(f),
                "cannot use an unreferenced BDD node"
            );
            return ControlFlow::Continue(());
        }
        let Refinement { stack, seen, memo } = scratch;
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
                    let node = state.node(l);
                    // Every nonempty predicate reaches true, so a terminal partition side
                    // has reached its class without walking the rest of `r`.
                    let key = if node.is_terminal() {
                        (l, Function::TRUE)
                    } else {
                        (l, r)
                    };
                    if !seen.insert(key) {
                        continue;
                    }
                    if node.is_terminal() {
                        if select(node.terminal()) {
                            break 'class Some(node.terminal());
                        }
                        continue;
                    }
                    let mut cofactors = LevelBuffer::new();
                    state.cofactors(l, r, &mut cofactors);
                    // Pop in the same assignment order as recursive discovery.
                    stack.extend(cofactors.as_slice().iter().rev().map(|&(_, a, b)| (a, b)));
                }
                None
            };
            let Some(class) = class else {
                break;
            };
            then(class, &mut || {
                self.store.owned(|state| {
                    memo.clear();
                    state.try_cell(memo, f, p, class)
                })
            })?;
        }
        ControlFlow::Continue(())
    }

    /// A cursor over the members of predicate `p`, holding its own reference to
    /// the root until dropped.
    fn members(&self, p: Function) -> Box<dyn Members + '_> {
        debug_assert!(
            self.store.read().has_refs(p),
            "cannot use an unreferenced BDD node"
        );
        self.inc_ref(p);
        Box::new(BddMembers {
            column: self,
            root: p,
            value: Vec::new(),
            stack: if p.is_f() {
                Vec::new()
            } else {
                vec![(0, p, 0)]
            },
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/bdd.rs"]
mod tests;
