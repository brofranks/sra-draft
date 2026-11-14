//! Canonical decision diagrams with 6 bits per level.
//!
//! A nonterminal node tests one level and stores disjoint `(mask, child)` edges
//! covering all 64 assignments at that level, ordered by each edge's first
//! value. A zero-edge node is a terminal storing its value: [`Function::FALSE`]
//! and [`Function::TRUE`] are terminals 0 and 1, a predicate uses only those,
//! and a partition uses densely numbered `u32` terminals in first-appearance
//! order. The [`BddStore`] type implements [`Backend`], and [`Span`] places a
//! sort's encoded value on levels.
//!
//! Construction and transformation recurse with diagram depth, so their stack
//! use grows with the number of levels traversed.

use std::array::from_fn;
use std::cmp::Ordering;
use std::mem::{MaybeUninit, swap};
use std::ops::ControlFlow;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;

use num_bigint::BigUint;

use crate::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, StoreCache, Values,
};
use crate::scalar::{ScalarKind, ScalarSort};
use crate::util::dag::{
    Dag, Exhausted, Grow, Map, NeedGrowth, NodeRef as DagNodeRef, Shared, structural_hash,
};

/// The number of value bits one level tests.
const LEVEL_BITS: usize = 6;
/// The number of assignments at one level, one bit of a mask each.
const LEVEL_VALUES: usize = 1 << LEVEL_BITS;
const NODE_CAPACITY: usize = 4_096;
const EDGE_CAPACITY: usize = 16_384;

const _: () = assert!(LEVEL_VALUES == u64::BITS as usize);

/// The six-bit levels occupied by one value of `width` bits, starting at level
/// zero.
///
/// Bit 0 of the value sits at position 0 of `last_level`, and higher bits fill
/// higher positions and then earlier levels, so only level zero may be short.
/// Assignments beyond a short level's width alias the assignment obtained by
/// clearing its unused high bits, as normalized by [`State::intern_level`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Span {
    width: u32,
}

impl Span {
    /// Places a value of `width` bits at the beginning of the BDD level order.
    ///
    /// # Panics
    ///
    /// Panics if `width` is zero.
    #[inline(always)]
    const fn new(width: u32) -> Self {
        assert!(width > 0, "span width must be nonzero");
        Self { width }
    }

    /// The level holding bit zero of the value.
    ///
    /// Values start at level zero.
    #[inline(always)]
    const fn last_level(self) -> u32 {
        self.width.div_ceil(LEVEL_BITS as u32) - 1
    }

    /// The number of bits of the value that `level` holds.
    ///
    /// Only the first level can be short.
    #[inline(always)]
    fn reach(self, level: u32) -> u32 {
        (self.width - (self.last_level() - level) * LEVEL_BITS as u32).min(LEVEL_BITS as u32)
    }

    /// The mask of the assignments at `level` that some value reaches.
    ///
    /// Only a short first level leaves assignments unreachable.
    #[inline(always)]
    fn valid(self, level: u32) -> u64 {
        u64::MAX >> (u64::BITS - (1 << self.reach(level)))
    }

    /// The `(word, shift, reach)` locating the bits `level` holds inside an encoded
    /// value.
    #[inline(always)]
    fn place(self, level: u32) -> (usize, u32, u32) {
        let low = (self.last_level() - level) * LEVEL_BITS as u32;
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

/// Orders terminals below every tested level; never stored as a branch level.
///
/// A terminal may still store `u32::MAX` as its value because its empty edge
/// span identifies it.
const TERMINAL_LEVEL: u32 = u32::MAX;

/// A diagram node: a terminal value or a tested level.
#[derive(Debug)]
struct Node {
    /// The terminal value of a zero-edge node, otherwise the tested level of a
    /// branch; read through [`NodeRef::terminal`] and [`NodeRef::level`].
    data: u32,
}

const _: () = assert!(size_of::<Node>() == size_of::<u32>());

/// An edge leading to a diagram node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Edge {
    /// The set of the 64 values at this level that traverse this edge.
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

/// The edge of a complete level whose mask contains the assignment `bit`.
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
        // SAFETY: `push` initializes each entry before extending `len`, so the
        // prefix contains valid values. `MaybeUninit<T>` has the layout of `T`,
        // and the slice borrows this buffer for its lifetime.
        unsafe { from_raw_parts(self.items.as_ptr() as *const T, self.len) }
    }

    /// The initialized items, exclusively.
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the initialized-prefix and layout invariants of `as_slice`
        // apply. The exclusive borrow prevents other access to the buffer
        // while the mutable slice exists.
        unsafe { from_raw_parts_mut(self.items.as_mut_ptr() as *mut T, self.len) }
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
            |node, edges| structural_hash((node.data, edges)),
        );
    }
}

/// Computed tables for one public store call, retaining no roots through
/// collection.
#[derive(Default)]
struct Memo {
    /// Results keyed by operation, since one call shares the table across
    /// operations.
    apply: Map<(BoolOp, Function, Function), Function>,
    ite: Map<(Function, Function, Function), Function>,
    /// Both directions of complementation within this attempt.
    not: Map<Function, Function>,
    /// One level move per table, with fixed mask and placement.
    moved: Map<Function, Function>,
}

impl State {
    /// The canonical terminal for a value, without an external reference.
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
        // A branch's edges are nonempty and disjoint, cover all 64 assignments, are
        // ordered by their first value, have distinct children, and have children that
        // are terminals or test later levels.
        if edges.len() == 1 {
            return Ok(edges[0].child);
        }

        self.intern(
            structural_hash((data, edges)),
            |node, stored| node.data == data && stored == edges,
            || Node { data },
            edges,
        )
        .map_err(|need| match need {
            // A branch may need a whole level of edges, so request that span rather than
            // this node's and let one growth serve the following nodes too.
            NeedGrowth::Edges(_) => NeedGrowth::Edges(LEVEL_VALUES),
            need => need,
        })
    }

    /// The complete level node whose assignment `a` leads to `child(a)`.
    ///
    /// Positions beyond a short first level are never reached and copy the
    /// reachable assignment they alias.
    fn intern_level(
        &self,
        level: u32,
        reach: u32,
        child: impl Fn(usize) -> Function,
    ) -> Result<Function, NeedGrowth> {
        let mut edges = LevelBuffer::new();
        for assignment in 0..LEVEL_VALUES {
            edges.merge(1 << assignment, child(assignment & ((1 << reach) - 1)));
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
        let [left_edges, right_edges]: [&[Edge]; 2] = from_fn(|i| {
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
        // Intersecting two ordered edge lists does not preserve first-value order;
        // restore it before recursion so result edges and first-discovered partition
        // slots remain canonical. A whole-domain edge against a node is already
        // ordered, so the common case skips the sort.
        let by_first = |&(mask, _, _): &(u64, Function, Function)| mask.trailing_zeros();
        if !cofactors.as_slice().is_sorted_by_key(by_first) {
            cofactors.as_mut_slice().sort_unstable_by_key(by_first);
        }
        level
    }

    /// Interns the product of two nodes at their upper level, recursing on every
    /// nonempty cofactor in ascending first-value order so result edges and
    /// first-discovered partition slots are canonical.
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

    /// One conditional traversal, including unordered source decisions.
    ///
    /// Ordered literals usually select their children immediately; overlapping
    /// levels are cofactored before interning.
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
        let level = nodes.iter().map(|node| node.level()).min().unwrap();
        let whole = [condition, yes, no].map(whole_edge);
        let parts: [&[Edge]; 3] = from_fn(|i| {
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
            swap(&mut left, &mut right);
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

    /// Abstracts cleared bits and permutes toggled bits.
    ///
    /// Assignments differing only in cleared bits share the disjunction of their
    /// children, making those bits unconstrained.
    ///
    /// No level below `last` holds a cleared or toggled bit, so subgraphs there
    /// are returned unchanged.
    fn try_move(
        &self,
        memo: &mut Memo,
        span: Span,
        clear: &[u64],
        toggle: &[u64],
        last: u32,
        f: Function,
    ) -> Result<Function, NeedGrowth> {
        let node = self.node(f);
        // A terminal orders below every level, so this also returns terminals.
        if node.level() > last {
            return Ok(f);
        }
        if let Some(&result) = memo.moved.get(&f) {
            return Ok(result);
        }
        let level = node.level();
        let bits = span.assignment(level, clear);
        let flip = span.assignment(level, toggle);
        let result = if bits == 0 && flip == 0 {
            // An untouched level keeps its edges, merging those whose children
            // collapse together, as `try_map` does.
            let mut edges = LevelBuffer::new();
            for edge in node.edges {
                edges.merge(
                    edge.mask,
                    self.try_move(memo, span, clear, toggle, last, edge.child)?,
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
                let child = self.try_move(memo, span, clear, toggle, last, edge.child)?;
                let mut remaining = edge.mask & span.valid(level);
                while remaining != 0 {
                    let a = remaining.trailing_zeros() as usize;
                    remaining &= remaining - 1;
                    let target = &mut children[(a ^ flip) & !bits];
                    *target = self.try_apply_bool(memo, BoolOp::Or, *target, child)?;
                }
            }
            self.intern_level(level, span.reach(level), |a| children[a & !bits])?
        };
        memo.moved.insert(f, result);
        Ok(result)
    }

    /// The predicate `(x & mask) == value` over `span`, without an external
    /// reference.
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
            let expected = span.assignment(level, value);
            f = self.intern_level(level, span.reach(level), |v| {
                if v & bits == expected {
                    f
                } else {
                    Function::FALSE
                }
            })?;
        }
        Ok(f)
    }

    /// The predicate for the points of `p` that `f` maps to `slot`.
    fn try_cell(
        &self,
        memo: &mut Map<(Function, Function), Function>,
        f: Function,
        p: Function,
        slot: u32,
    ) -> Result<Function, NeedGrowth> {
        if p.is_f() {
            return Ok(Function::FALSE);
        }
        let node = self.node(f);
        if node.is_terminal() {
            return Ok(if node.terminal() == slot {
                p
            } else {
                Function::FALSE
            });
        }
        if let Some(&result) = memo.get(&(f, p)) {
            return Ok(result);
        }
        let result = self.try_product(f, p, |l, r| self.try_cell(memo, l, r, slot))?;
        memo.insert((f, p), result);
        Ok(result)
    }

    /// Rebuilds `f` with every partition terminal `s` replaced by `map(s)`, merging
    /// edges whose children collapse together and reusing every subgraph the map
    /// leaves unchanged.
    ///
    /// A bijective `map` is a pure renumbering; a non-injective one coarsens the
    /// partition.
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

    /// Product traversal of two partitions onto label terminals.
    ///
    /// Each reachable `(left-slot, right-slot)` pair is discovered once and takes
    /// the next slot. `labels[start..]` holds the source-slot pair of each result
    /// slot found so far.
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
    /// label terminal and every other value to the `default` terminal, built from
    /// `level` downwards without an external reference.
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
        default: u32,
    ) -> Result<Function, NeedGrowth> {
        if keys.is_empty() {
            return self.intern_terminal(default);
        }
        let words = span.width.div_ceil(u64::BITS) as usize;
        let len = keys.len() / words;
        if level > span.last_level() {
            debug_assert_eq!(len, 1, "exactly one key must remain below the last level");
            return self.intern_terminal(label(offset));
        }
        let mut children = [self.intern_terminal(default)?; LEVEL_VALUES];
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
        self.intern_level(level, span.reach(level), |a| children[a])
    }

    /// Renumbers the label terminals of `f` to canonical slots without an external
    /// reference and appends each slot's label to `slot_labels` once the result
    /// exists.
    fn try_canonicalize(
        &self,
        f: Function,
        slot_labels: &mut Vec<u32>,
    ) -> Result<Function, NeedGrowth> {
        // Each distinct terminal value in depth-first first-appearance order over the
        // stored edge order, so an already canonical partition maps `0, 1, .., k - 1`
        // onto itself.
        let mut first = Vec::new();
        let mut slot_of = Map::default();
        let result = self.try_map(&mut Map::default(), f, &mut |label| {
            *slot_of.entry(label).or_insert_with(|| {
                first.push(label);
                first.len() as u32 - 1
            })
        })?;
        slot_labels.extend_from_slice(&first);
        Ok(result)
    }
}

/// The cleared bits of a bit map, the values assigned to them, and its flip
/// mask.
///
/// # Panics
///
/// Panics if `map` is a Rust closure, which has no symbolic form.
fn bit_map(map: &MapSpec) -> (&[u64], Vec<u64>, &[u64]) {
    let MapSpec::Bits { clear, flip } = map else {
        panic!("a BDD store cannot apply a Rust map, which needs an enumerating backend")
    };
    let set = clear.iter().zip(flip.iter()).map(|(c, v)| c & v).collect();
    (&clear[..], set, flip)
}

/// The deepest level of `span` holding a bit of `clear` or `flip`, or level
/// zero when neither holds any.
fn last_moved(span: Span, clear: &[u64], flip: &[u64]) -> u32 {
    (0..=span.last_level())
        .rev()
        .find(|&level| span.assignment(level, clear) | span.assignment(level, flip) != 0)
        .unwrap_or(0)
}

/// A concurrent canonical store of predicates and partitions with 6 bits per
/// level.
///
/// Roots are [`Function`] tokens with manual external reference counts, which
/// the context pairs with this store through a column id. Equal maps share one
/// token, so token identity decides equality within one column.
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
    /// Panics if `threads` is zero, `nodes` is below two, or either capacity
    /// exceeds `u32::MAX`.
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

    /// One referenced result of an attempt that retries after growth.
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
    /// its current maintenance epoch.
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
/// Stack entries hold tokens below the owned root; no borrowed node or store
/// guard survives a call to `next`.
struct BddMembers<'a> {
    store: &'a BddStore,
    root: Function,
    span: Span,
    value: Vec<u64>,
    stack: Vec<(u32, Function, u64)>,
}

impl Members for BddMembers<'_> {
    fn next(&mut self, out: &mut Vec<u64>) -> bool {
        let state = self.store.shared.read(self.store.threads);
        while let Some((level, f, next)) = self.stack.last_mut() {
            if *next >= u64::BITS as u64 {
                self.stack.pop();
                continue;
            }
            let node = state.node(*f);
            // The assignments at this level that reach a nonempty child: the live
            // edge masks of a node tested here, or every assignment of a skipped
            // level. Jumping to the next one costs one step per delivered value
            // instead of one per level value.
            let live = if node.level() == *level {
                node.edges
                    .iter()
                    .filter(|e| !e.child.is_f())
                    .fold(0, |m, e| m | e.mask)
            } else {
                u64::MAX
            };
            let rest = live & self.span.valid(*level) & (u64::MAX << *next);
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
            self.span.assign(*level, assignment, &mut self.value);
            if *level == self.span.last_level() {
                out.clear();
                out.extend_from_slice(&self.value);
                return true;
            }
            let next_level = *level + 1;
            self.stack.push((next_level, child, 0));
        }
        false
    }
}

impl Drop for BddMembers<'_> {
    fn drop(&mut self) {
        if !self.root.is_constant() {
            self.store
                .shared
                .read(self.store.threads)
                .dec_ref(self.root);
        }
    }
}

/// A BDD store bound to one scalar sort, with the level span its width needs.
struct BddColumn {
    store: Arc<BddStore>,
    sort: ScalarSort,
    span: Span,
}

impl Backend for BddStore {
    /// Every scalar sort is a bit string over as many levels as its width needs.
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        Some(Box::new(BddColumn {
            span: Span::new(sort.width()),
            sort,
            store: self,
        }))
    }
}

impl Column for BddColumn {
    /// The bound scalar sort.
    fn sort(&self) -> ScalarSort {
        self.sort
    }

    /// Increments the manually managed external reference count of a live function.
    ///
    /// An existing owner must keep the root live while this call acquires another
    /// reference. The caller must balance each call with [`Self::dec_ref`].
    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            self.store.shared.read(self.store.threads).inc_ref(f);
        }
    }

    /// Decrements the manually managed external reference count of a function.
    ///
    /// Reclamation is deferred to growth-triggered collection.
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

    /// The terminal of `f` at one value of the bound sort.
    fn evaluate(&self, mut f: Function, value: &[u64]) -> u32 {
        let span = self.span;
        let state = self.store.shared.read(self.store.threads);
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
    fn count(&self, p: Function) -> BigUint {
        let span = self.span;
        let state = self.store.shared.read(self.store.threads);
        debug_assert!(state.has_refs(p), "cannot use an unreferenced BDD node");
        // Counts start at a node's own level; an incoming edge supplies skipped bits.
        let bits = |f| {
            let node = state.node(f);
            if node.is_terminal() {
                0
            } else {
                (span.last_level() - node.level()) * LEVEL_BITS as u32 + span.reach(node.level())
            }
        };
        let mut memo = Map::from_iter([
            (Function::FALSE, BigUint::ZERO),
            (Function::TRUE, BigUint::ONE),
        ]);
        let mut stack = vec![(p, 0)];
        'nodes: while let Some((f, next)) = stack.last_mut() {
            let f = *f;
            if memo.contains_key(&f) {
                stack.pop();
                continue;
            }
            let node = state.node(f);
            assert!(!node.is_terminal(), "count requires a Boolean predicate");
            let reach = span.reach(node.level());
            // Every edge intersects the reachable assignments: `intern_level` aliases
            // positions beyond a short first level to reachable ones and products preserve
            // that closure.
            let valid = span.valid(node.level());
            while let Some(edge) = node.edges.get(*next) {
                *next += 1;
                if !memo.contains_key(&edge.child) {
                    stack.push((edge.child, 0));
                    continue 'nodes;
                }
            }
            let mut count = BigUint::ZERO;
            for edge in node.edges {
                count += (&memo[&edge.child] * (edge.mask & valid).count_ones())
                    << (bits(f) - reach - bits(edge.child));
            }
            memo.insert(f, count);
            stack.pop();
        }
        &memo[&p] << (self.sort.width() - bits(p))
    }

    /// The predicate holding exactly where the value takes one of the sorted,
    /// distinct `values`.
    fn set(&self, values: &[u64]) -> Function {
        self.store
            .owned(|state| state.try_values(self.span, 0, values, &|_| 1, 0, 0))
    }

    /// The predicate holding where the value is below `bound`.
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
        let span = self.span;
        self.store.owned(|state| {
            let mut f = Function::FALSE;
            for level in (0..=span.last_level()).rev() {
                let limit = span.assignment(level, bound);
                f = state.intern_level(level, span.reach(level), |value| {
                    match value.cmp(&limit) {
                        Ordering::Less => Function::TRUE,
                        Ordering::Equal => f,
                        Ordering::Greater => Function::FALSE,
                    }
                })?;
            }
            Ok(f)
        })
    }

    /// The Boolean combination of two predicates.
    ///
    /// The `AndNot` operation is directional: it computes `left and !right`.
    fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> Function {
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced BDD node"
            );
            // A failed attempt's results are unreferenced and collected before the retry,
            // so each attempt starts from an empty table.
            state.try_apply_bool(&mut Memo::default(), op, left, right)
        })
    }

    /// The predicate `(x & mask) == value`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, whose values carry no bit structure.
    fn masked_eq(&self, mask: &[u64], value: &[u64]) -> Function {
        assert_ne!(
            self.sort.kind(),
            ScalarKind::Symbol,
            "symbols do not support bit predicates"
        );
        self.store
            .owned(|state| state.try_masked_eq(self.span, mask, value))
    }

    /// The image under a bit map, which moves levels: its cleared bits are
    /// abstracted existentially and its toggled bits permute their assignments,
    /// while a masked test fixes the assigned bits of the output.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort or is a Rust closure.
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_domain(self.sort);
        if map.is_identity() {
            return self.apply_bool(BoolOp::And, p, q);
        }
        let (clear, set, flip) = bit_map(map);
        let span = self.span;
        let last = last_moved(span, clear, flip);
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(p) && state.has_refs(q),
                "cannot use an unreferenced BDD node"
            );
            let mut memo = Memo::default();
            let f = state.try_move(&mut memo, span, clear, flip, last, p)?;
            let f = state.try_apply_bool(
                &mut memo,
                BoolOp::And,
                f,
                state.try_masked_eq(span, clear, &set)?,
            )?;
            state.try_apply_bool(&mut memo, BoolOp::And, f, q)
        })
    }

    /// The moves of `image` in reverse: `q` restricted to the assigned bits,
    /// toggled back, and abstracted over the cleared bits, which the map forgets.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort or is a Rust closure.
    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_domain(self.sort);
        if map.is_identity() {
            return self.apply_bool(BoolOp::And, p, q);
        }
        let (clear, set, flip) = bit_map(map);
        let span = self.span;
        let last = last_moved(span, clear, flip);
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(p) && state.has_refs(q),
                "cannot use an unreferenced BDD node"
            );
            let mut memo = Memo::default();
            // The conjunction must precede the move: pinning the cleared bits is
            // what makes the move's disjunction over them exact. Mirroring the
            // order of `image` would silently widen the result.
            let f = state.try_apply_bool(
                &mut memo,
                BoolOp::And,
                q,
                state.try_masked_eq(span, clear, &set)?,
            )?;
            let f = state.try_move(&mut memo, span, clear, flip, last, f)?;
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

    /// The common refinement of two distinct, nontrivial partitions.
    ///
    /// The `labels` buffer is appended with the source slot pair of each result
    /// slot. See [`Column::align`].
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
        // Slots are taken in first-discovery order, so the result is canonical by
        // construction.
        let start = labels.len();
        let cache = &mut scratch.cache;
        self.store.owned(|state| {
            debug_assert!(
                state.has_refs(left) && state.has_refs(right),
                "cannot use an unreferenced BDD node"
            );
            self.store.restart_cache(cache);
            // A retried attempt appends its labels afresh.
            labels.truncate(start);
            if let Some(&(result, at, len)) = cache.aligned.get(&(left, right)) {
                labels.extend_from_slice(&cache.labels[at..at + len]);
                return Ok(result);
            }
            scratch.memo.clear();
            let result = state.try_align(&mut scratch.memo, labels, start, left, right)?;
            // Growth inside `owned` restarted the cache, so this insertion is against the
            // final epoch.
            let at = cache.labels.len();
            cache.labels.extend_from_slice(&labels[start..]);
            cache
                .aligned
                .insert((left, right), (result, at, labels.len() - start));
            Ok(result)
        })
    }

    /// The coarsening of partition `f` by `slots`.
    ///
    /// The `slots` slice is first-appearance numbered and merges at least one
    /// pair of slots. See [`Column::quotient`]. Slot substitution preserves
    /// canonical numbering because slot order is first-appearance order.
    fn quotient(&self, f: Function, slots: &[u32], scratch: &mut Alignment) -> Function {
        debug_assert_eq!(slots.first(), Some(&0), "quotient slots must start at zero");
        debug_assert!(
            slots
                .iter()
                .enumerate()
                .any(|(i, &slot)| slot as usize != i),
            "quotient must merge at least two slots"
        );
        let cache = &mut scratch.cache;
        let key = (f, structural_hash(slots));
        self.store.owned(|state| {
            debug_assert!(state.has_refs(f), "cannot use an unreferenced BDD node");
            self.store.restart_cache(cache);
            if let Some(&(result, at, len)) = cache.quotients.get(&key)
                && cache.slots[at..at + len] == *slots
            {
                return Ok(result);
            }
            scratch.map.clear();
            let result = state.try_map(&mut scratch.map, f, &mut |s| slots[s as usize])?;
            // A hash collision with different slots is simply replaced.
            let at = cache.slots.len();
            cache.slots.extend_from_slice(slots);
            cache.quotients.insert(key, (result, at, slots.len()));
            Ok(result)
        })
    }

    /// The partition built from case predicates, appending `slot_labels` as
    /// specified by [`Column::cases`].
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
        // `slot_labels` maps canonical partition slots to the requested case labels.
        self.store.owned(|state| {
            debug_assert!(
                cases.iter().all(|&(p, _)| state.has_refs(p)),
                "cannot use an unreferenced BDD node",
            );
            if let &[(p, label)] = cases {
                if default == label || p.is_constant() {
                    slot_labels.push(if p.is_f() { default } else { label });
                    return Ok(Function::FALSE);
                }
                // A Boolean predicate has only two labels. Its first stored terminal determines
                // whether it already numbers the cells in canonical first-appearance order.
                let mut first = p;
                while !first.is_constant() {
                    first = state.node(first).edges[0].child;
                }
                if first.is_f() {
                    slot_labels.extend_from_slice(&[default, label]);
                    return Ok(p);
                }
                let result = state.try_not(&mut Memo::default(), p)?;
                slot_labels.extend_from_slice(&[label, default]);
                return Ok(result);
            }
            // Fold each case onto label terminals, then renumber labels to canonical slots.
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
            state.try_canonicalize(acc, slot_labels)
        })
    }

    /// The partition labeling the sorted, distinct `keys` by `case_labels`, which
    /// are free of `default`, the label unlisted values use.
    ///
    /// The `slot_labels` buffer is appended as specified by [`Column::values`].
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        slot_labels: &mut Vec<u32>,
        _: &mut Values,
    ) -> Function {
        self.store.owned(|state| {
            let f = state.try_values(self.span, 0, keys, &|i| case_labels[i], 0, default)?;
            state.try_canonicalize(f, slot_labels)
        })
    }

    /// Splits predicate `p` by partition `f`, as specified by [`Column::refine`].
    ///
    /// Discovery pauses at each newly reached slot and keeps no guard or node
    /// borrow across a yield, so a `Break` leaves pending pairs unvisited. Each
    /// thunk builds its complete cell from the original roots, outside guards
    /// and retries.
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
        let Refinement {
            stack, seen, memo, ..
        } = scratch;
        stack.clear();
        stack.push((f, p));
        seen.clear();
        loop {
            let slot = {
                let state = self.store.shared.read(self.store.threads);
                debug_assert!(
                    state.has_refs(f) && state.has_refs(p),
                    "cannot use an unreferenced BDD node"
                );
                let mut reached = None;
                while let Some((l, r)) = stack.pop() {
                    if r.is_f() {
                        continue;
                    }
                    let node = state.node(l);
                    // Every nonempty predicate reaches true, so a terminal partition side
                    // has reached its slot without walking the rest of `r`.
                    let key = if node.is_terminal() {
                        (l, Function::TRUE)
                    } else {
                        (l, r)
                    };
                    if !seen.insert(key) {
                        continue;
                    }
                    if node.is_terminal() {
                        if !select(node.terminal()) {
                            continue;
                        }
                        reached = Some(node.terminal());
                        break;
                    }
                    let mut cofactors = LevelBuffer::new();
                    state.cofactors(l, r, &mut cofactors);
                    // Pop in the same first-value order as recursive discovery.
                    stack.extend(cofactors.as_slice().iter().rev().map(|&(_, a, b)| (a, b)));
                }
                reached
            };
            let Some(slot) = slot else {
                break;
            };
            then(slot, &mut || {
                self.store.owned(|state| {
                    memo.clear();
                    state.try_cell(memo, f, p, slot)
                })
            })?;
        }
        ControlFlow::Continue(())
    }

    /// A cursor over the members of predicate `p`, holding its own reference to
    /// the root until dropped.
    fn members(&self, p: Function) -> Box<dyn Members + '_> {
        debug_assert!(
            self.store.shared.read(self.store.threads).has_refs(p),
            "cannot use an unreferenced BDD node"
        );
        self.inc_ref(p);
        Box::new(BddMembers {
            store: &self.store,
            root: p,
            span: self.span,
            value: vec![0; self.sort.words()],
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
