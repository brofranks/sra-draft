//! The outer index: exact ordered-tuple maps whose nodes factor one level each
//! into a column partition and a slot vector of continuations.
//!
//! This module knows nothing about the stores behind a level; it names a level
//! by its [`ColumnId`] and reaches the storage only through the owning
//! [`Context`]. Lock order runs downward: an index guard may be held while a
//! column store runs, never the reverse.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::backend::{Alignment, BoolOp, Function, Refinement, Values};
use crate::context::{ColumnId, Context, ContextData};
use crate::partition::{self, Partition};
use crate::scalar::{self, ScalarKind};
use crate::util::dag::{
    Dag, Exhausted, Grow, Idx, Map, NeedGrowth, Read, Set, Shared, structural_hash,
};

/// Initial arena sizes.
///
/// Collection alone recovers most transient results on the gate workloads, so
/// the arenas rarely grow past their initial size and every collection pays its
/// fixed costs: at 4096 nodes the closed-graph gate collected 622 times.
pub const NODE_CAPACITY: usize = 65_536;
/// The initial edge arena size; see [`NODE_CAPACITY`].
pub const EDGE_CAPACITY: usize = 262_144;

/// The maintenance thread budget of every index container.
const THREADS: usize = 1;

idx! {
    /// A column position in an index; `u32::MAX` is reserved for `Level::TERMINAL`.
    pub struct Level < u32::MAX as usize;
}

impl Level {
    /// The level of both terminals, past every column.
    pub const TERMINAL: Self = Self(u32::MAX);
}

/// One level node, factored into a column partition and a slot vector.
///
/// `partition` is one referenced partition in the level's column; the node's
/// edge span carries the outer continuation of each partition slot. The column
/// owns the split of the level's sort and the index owns only continuations, so
/// nodes that split their level the same way share one partition regardless of
/// where they lead.
#[derive(Debug)]
#[repr(C)]
pub struct IndexNode {
    /// The position in the level order, whose column selects the store that
    /// interprets `partition`.
    pub level: Level,
    /// The partition of this node's level, with one external reference held until
    /// the node is reclaimed.
    pub partition: Function,
    /// The slot whose continuation is FALSE, or `u32::MAX`.
    ///
    /// Continuations are distinct, so at most one slot has none; refinement
    /// declines it without reading the node again.
    pub hole: u32,
}

/// Entries above which the persisting operation memo is dropped before the next
/// operation; one operation's memo is unbounded.
pub const MEMO_BOUND: usize = 4_000_000;

/// First-appearance classification of one frame's children.
///
/// A frame classifies only after every recursive call has returned, so one
/// table serves the whole operation without nesting, and it empties itself by
/// clearing exactly the buckets it filled rather than its capacity.
///
/// Nothing is allocated or rehashed per frame: the table is a flat probe over
/// interned tokens.
#[derive(Default)]
struct Classes {
    /// `(child, slot + 1)` per bucket; slot zero marks an empty bucket.
    buckets: Vec<(Function, u32)>,
    /// The buckets the current frame filled.
    filled: Vec<u32>,
}

impl Classes {
    /// Prepares the table for a frame of `n` children, keeping it at most half
    /// full.
    #[inline(always)]
    fn restart(&mut self, n: usize) {
        let want = (2 * n).next_power_of_two();
        if self.buckets.len() < want {
            self.buckets.resize(want, (Function::FALSE, 0));
        }
    }

    /// The slot of `child`, which is `next` where the child first appears in this
    /// frame.
    #[inline(always)]
    fn slot(&mut self, child: Function, next: u32) -> u32 {
        let mask = self.buckets.len() - 1;
        // Fibonacci hashing keeps the high bits of the product, which depend
        // on every bit of the token.
        let mut i = ((child.index() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            >> (u64::BITS - self.buckets.len().trailing_zeros())) as usize;
        loop {
            let (key, slot) = self.buckets[i];
            if slot == 0 {
                self.buckets[i] = (child, next + 1);
                self.filled.push(i as u32);
                return next;
            }
            if key == child {
                return slot - 1;
            }
            i = (i + 1) & mask;
        }
    }

    /// Empties the table after a frame.
    #[inline(always)]
    fn finish(&mut self) {
        for &i in &self.filled {
            self.buckets[i as usize].1 = 0;
        }
        self.filled.clear();
    }
}

/// Working memory for [`IndexRoot::cases`], which builds one level from the
/// disjoint cases of a whole union at once.
///
/// Its caller is the attachment path, which reaches the index through `&self`
/// and has nowhere to keep a buffer of its own, so this follows [`Scratch`]
/// rather than the caller-owned [`Labels`] that loading passes to
/// [`IndexRoot::values`].
#[derive(Default)]
struct Cases {
    /// The label of each distinct continuation beyond the first; label 0 is the
    /// default.
    of: Map<Function, u32>,
    /// The continuation carrying each label.
    conts: Vec<Function>,
    /// Each case's predicate under its continuation's label.
    labeled: Vec<(Function, u32)>,
    /// The label the store reports for each reachable slot, and that slot's
    /// continuation.
    labels: Vec<u32>,
    slots: Vec<Function>,
}

/// Working memory for [`State::try_apply_bool`], reused by each thread.
///
/// Workers never share these buffers. The frame windows below survive no
/// operation except in capacity; the memo survives operations on one container
/// (see [`Scratch::restart`]). Backend operations may allocate their own
/// working memory.
#[derive(Default)]
struct Scratch {
    /// Results of completed operations.
    ///
    /// Both results and released operands are unreferenced nodes, which only growth
    /// reclaims, so entries stay valid until the operated container grows or
    /// changes; a failed attempt is followed by a growth and so invalidates them
    /// too.
    memo: Map<(BoolOp, Function, Function), Function>,
    /// The container and its growth count these entries were computed against.
    container: u64,
    epoch: u64,
    /// Stacked per-frame windows of alignment labels, produced children, and slot
    /// renumberings.
    labels: Vec<(u32, u32)>,
    children: Vec<Function>,
    slots: Vec<u32>,
    classes: Classes,
    /// The backends' own working memory.
    alignment: Alignment,
    /// Working memory for [`IndexRoot::cases`].
    cases: Cases,
    /// Entries above which the memo is dropped; [`MEMO_BOUND`] by default.
    bound: usize,
}

impl Scratch {
    /// Begins one attempt on the container `container` at growth count `epoch`.
    ///
    /// The frame windows are already empty after a completed attempt; a failed one
    /// leaves them behind. The memo is kept across operations and dropped once its
    /// nodes can have been reclaimed, that is when the container or its growth
    /// count changes, or when it outgrows its bound. Dropping replaces the table so
    /// that its capacity is released too.
    #[inline(always)]
    fn restart(&mut self, container: u64, epoch: u64) {
        if self.container != container || self.epoch != epoch || self.memo.len() > self.bound {
            self.memo = Map::default();
            self.container = container;
            self.epoch = epoch;
        }
        self.labels.clear();
        self.children.clear();
        self.slots.clear();
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        bound: MEMO_BOUND,
        ..Scratch::default()
    });
}

/// Overrides this thread's operation-memo bound, as
/// [`crate::backend::StoreCache::with_bound`] does for a store cache.
///
/// Sizing the memo is a per-thread policy, not a container's property.
pub fn set_memo_bound(bound: usize) {
    SCRATCH.with_borrow_mut(|s| s.bound = bound);
}

/// Caller-owned working memory for [`IndexRoot::values`], like the label
/// buffers that [`Context::values`] and [`Context::cases`] already take.
///
/// Loading calls `values` once per interned node. Reusing the label table and
/// vectors retains their capacity across calls; backend operations may allocate
/// their own working memory.
#[derive(Default)]
pub struct Labels {
    /// The label of each distinct continuation; label 0 is FALSE.
    pub of: Map<Function, u32>,
    /// The continuation carrying each label.
    pub conts: Vec<Function>,
    /// The label of each case, in the caller's order.
    pub cases: Vec<u32>,
    /// The label the store reports for each reachable slot.
    pub labels: Vec<u32>,
    /// The continuation each reachable slot carries.
    pub slots: Vec<Function>,
    /// The backend's own working memory.
    pub values: Values,
}

/// Physical storage and structural interning for one ordered tuple schema.
///
/// Compact structural hashing serves both interning and rebuilding; exact
/// equality resolves collisions.
pub type State = Dag<Function, IndexNode, Function>;

impl Grow for State {
    type Cx<'a> = (&'a Context, &'a [ColumnId]);

    /// Collects reclaimed nodes and releases their partitions in the column stores,
    /// one batch per level.
    fn grow(&mut self, exhausted: Exhausted, (cx, columns): (&Context, &[ColumnId])) {
        let mut reclaimed = vec![Vec::new(); columns.len()];
        self.maintain(
            exhausted,
            THREADS,
            |&c| Some(c),
            |node| reclaimed[node.level.index()].push(node.partition),
            |node, edges| structural_hash((node.level, node.partition, edges)),
        );
        for (&column, fs) in columns.iter().zip(&reclaimed) {
            cx.resolve(column).dec_refs(fs);
        }
    }
}

impl State {
    /// Interns a structural node without an external reference, consuming the
    /// caller's reference to `partition` on success: a published candidate adopts
    /// it, and an existing node releases it.
    ///
    /// A failed attempt leaves the caller its reference for the retry; a candidate
    /// that was published before the failure holds its own until the next
    /// collection reclaims it. Rejection of a backward child releases the caller's
    /// reference before panicking.
    fn try_intern(
        &self,
        cx: &Context,
        column: ColumnId,
        level: Level,
        partition: Function,
        conts: &[Function],
    ) -> Result<Function, NeedGrowth> {
        debug_assert!(!partition.is_f(), "an interned node splits its level");
        debug_assert!(
            conts.len() >= 2,
            "an interned node has one continuation per slot"
        );
        if conts.iter().any(|&c| self.node(c).level <= level) {
            partition::dec_ref(cx, column, partition);
            panic!("outer-index child must follow its parent level");
        }
        debug_assert!(
            conts
                .iter()
                .enumerate()
                .all(|(t, c)| !conts[..t].contains(c)),
            "outer-index slots must have distinct continuations",
        );

        let mut adopted = false;
        let root = self.intern(
            structural_hash((level, partition, conts)),
            // One comparison of the whole structural key, as the hash above
            // takes it.
            |node, edges| (node.level, node.partition, edges) == (level, partition, conts),
            || {
                // The candidate adopts the caller's reference. A losing
                // candidate remains unreferenced and is reclaimed by the next
                // collection, releasing it then.
                adopted = true;
                let hole = conts
                    .iter()
                    .position(|c| c.is_f())
                    .map_or(u32::MAX, |t| t as u32);
                IndexNode {
                    level,
                    partition,
                    hole,
                }
            },
            conts,
        );
        match (adopted, &root) {
            (true, Err(_)) => partition::inc_ref(cx, column, partition),
            (false, Ok(_)) => partition::dec_ref(cx, column, partition),
            _ => {}
        }
        root
    }

    /// Aligns both nodes' partitions, recurses per refinement label, then quotients
    /// the refinement by which labels produced equal children.
    ///
    /// Identities never reach the store: equal partitions refine to themselves
    /// along the diagonal, a deeper operand is the trivial partition, and a
    /// refinement whose labels are all distinct is kept as is.
    fn try_apply_bool(
        &self,
        cx: &Context,
        columns: &[ColumnId],
        s: &mut Scratch,
        op: BoolOp,
        mut left: Function,
        mut right: Function,
    ) -> Result<Function, NeedGrowth> {
        if op.is_commutative() && right < left {
            std::mem::swap(&mut left, &mut right);
        }
        if let Some(result) = op.simplify(left, right) {
            return Ok(result);
        }

        let key = (op, left, right);
        if let Some(&result) = s.memo.get(&key) {
            return Ok(result);
        }

        let left_node = self.node(left);
        let right_node = self.node(right);
        let level = left_node.level.min(right_node.level);
        let column = columns[level.index()];
        let left_at = left_node.level == level;
        let right_at = right_node.level == level;
        let labels_start = s.labels.len();
        // Only an alignment owns its refinement; a node's own partition is
        // borrowed, and the attempt's read guard keeps that node alive.
        let aligned = left_at && right_at && left_node.partition != right_node.partition;
        let f = if aligned {
            // No alignment cache holds references here: whole partition
            // pairs rarely recur, and held references keep every operand
            // and result alive. The reference-free whole-call cache lives
            // in the backend scratch (`StoreCache`, per worker, dropped on
            // store growth), keyed by unreferenced operands with the label
            // window alongside.
            cx.align(
                cx.partition(column, left_node.partition, left_node.edges.len()),
                cx.partition(column, right_node.partition, right_node.edges.len()),
                &mut s.labels,
                &mut s.alignment,
            )
        } else {
            // One node splits this level, or both split it the same way: its
            // own partition is the refinement, and the operand that starts
            // deeper repeats its label. `level` is the minimum of the two, so
            // at least one operand is at it.
            let node = if left_at { &left_node } else { &right_node };
            s.labels.extend(
                (0..node.edges.len() as u32)
                    .map(|t| (t * u32::from(left_at), t * u32::from(right_at))),
            );
            cx.partition(column, node.partition, node.edges.len())
        };

        // Collect every child before classifying so this frame can compact
        // its window in place.
        let children_start = s.children.len();
        for t in labels_start..s.labels.len() {
            let (i, j) = s.labels[t];
            let left_child = if left_at {
                left_node.edges[i as usize]
            } else {
                left
            };
            let right_child = if right_at {
                right_node.edges[j as usize]
            } else {
                right
            };
            match self.try_apply_bool(cx, columns, s, op, left_child, right_child) {
                Ok(child) => s.children.push(child),
                Err(reason) => {
                    if aligned {
                        partition::dec_ref(cx, f.column, f.function);
                    }
                    return Err(reason);
                }
            }
        }
        s.labels.truncate(labels_start);

        // Number slots by first appearance and compact the distinct children
        // in place. The table in `Classes` replaces a linear scan over the
        // distinct prefix, which is O(c^2) in a frame's fanout and dominated
        // collection of wide symbolic results; a dense vector indexed by child
        // token would pay initialization through the largest ID, whereas this
        // probe pays only for the frame.
        let slots_start = s.slots.len();
        let mut k = 0;
        s.classes.restart(s.children.len() - children_start);
        for c in children_start..s.children.len() {
            let child = s.children[c];
            let slot = s.classes.slot(child, k as u32);
            if slot as usize == k {
                s.children[children_start + k] = child;
                k += 1;
            }
            s.slots.push(slot);
        }
        s.classes.finish();
        let distinct = &s.children[children_start..children_start + k];
        let slots = &s.slots[slots_start..];

        let result = if k == 1 {
            // Every label agrees, so the level is unconstrained and vanishes.
            if aligned {
                partition::dec_ref(cx, f.column, f.function);
            }
            distinct[0]
        } else {
            // Interning consumes one reference, which a borrowed partition
            // acquires only here.
            let f = if k == slots.len() {
                if !aligned {
                    partition::inc_ref(cx, column, f.function);
                }
                f
            } else {
                let g = cx.quotient(f, slots, &mut s.alignment);
                if aligned {
                    partition::dec_ref(cx, f.column, f.function);
                }
                g
            };
            let root = self.try_intern(cx, column, level, f.function, distinct);
            if root.is_err() {
                partition::dec_ref(cx, f.column, f.function);
            }
            root?
        };
        s.children.truncate(children_start);
        s.slots.truncate(slots_start);
        s.memo.insert(key, result);
        Ok(result)
    }
}

/// The label of `child`, numbering continuations beyond `conts[0]` by first
/// appearance.
#[inline(always)]
fn label(of: &mut Map<Function, u32>, conts: &mut Vec<Function>, child: Function) -> u32 {
    if conts.len() == 1 {
        conts.push(child);
        1
    } else if conts[1] == child {
        1
    } else {
        let next = conts.len() as u32;
        let label = *of.entry(child).or_insert(next);
        if label == next {
            conts.push(child);
        }
        label
    }
}

/// Physical layout and native storage, owned by the context that interned it
/// and shared by every root over it.
pub struct IndexData {
    /// The column held by each physical level.
    ///
    /// Level IDs are the normalization order.
    pub columns: Box<[ColumnId]>,
    /// The node storage.
    pub shared: Shared<State>,
}

/// Releases every store reference `data` owns, before its context's columns go
/// away.
///
/// Every published nonterminal owns a partition, including losing interning
/// candidates. Reclaimed nodes have already released theirs; the two pinned
/// terminals own none. DAG destruction then drops only payloads and buffers; it
/// does not repeat these releases. Like collection, this releases one batch per
/// level.
pub fn dec_partition_refs(data: &mut IndexData, cx: &ContextData) {
    let state = data.shared.get_mut();
    let mut live = vec![Vec::new(); data.columns.len()];
    for i in 2..state.node_capacity() {
        if let Some(node) = state.get(Function::new(i)) {
            live[node.level.index()].push(node.partition);
        }
    }
    for (&column, fs) in data.columns.iter().zip(&live) {
        cx.resolve(column).dec_refs(fs);
    }
}

impl IndexData {
    /// A read guard over the index state.
    ///
    /// Growth under this guard collects first, which releases the partitions of
    /// reclaimed nodes in the column stores. No guard may reach a user
    /// continuation.
    #[inline(always)]
    fn read<'a>(&'a self, cx: &'a Context) -> Read<'a, State> {
        self.shared.read((cx, &self.columns))
    }

    /// A referenced node from a column-scoped partition and its slot continuations,
    /// consuming the caller's reference to `partition`.
    ///
    /// A one-slot map is unconstrained: it collapses to its sole continuation and
    /// the trivial partition is dropped.
    fn branch(
        &self,
        cx: &Context,
        level: Level,
        partition: Partition,
        conts: &[Function],
    ) -> Function {
        assert_eq!(
            partition.column,
            self.columns[level.index()],
            "partition belongs to another column"
        );
        assert_eq!(
            partition.slots as usize,
            conts.len(),
            "one continuation per reachable slot"
        );
        if conts.len() == 1 {
            debug_assert!(
                partition.is_trivial(),
                "a one-continuation partition must be trivial"
            );
            self.read(cx).inc_ref(conts[0]);
            return conts[0];
        }

        // Interning consumes this reference on the attempt that succeeds and
        // keeps it across failed ones.
        self.shared.retry((cx, &self.columns), |state| {
            debug_assert!(
                conts.iter().all(|&c| state.has_refs(c)),
                "cannot use an unreferenced outer-index child",
            );
            let root = state.try_intern(
                cx,
                self.columns[level.index()],
                level,
                partition.function,
                conts,
            )?;
            state.inc_ref(root);
            Ok(root)
        })
    }
}

/// One owned native root and its container.
///
/// Retaining the context retains the column stores that interpret the
/// container.
///
/// The container is also this root's handle on its index: a method taking a
/// [`Function`] operates on any referenced node of the same container, while
/// the rest describe this root itself.
pub struct IndexRoot {
    // Release this container reference before the last context owner tears
    // down its layouts and releases every container's partition references.
    data: Arc<IndexData>,
    cx: Context,
    /// The native root token; this value owns one reference on it.
    pub function: Function,
}

/// The empty relation over one physical container, shared by these exact
/// columns and independent of logical endpoint shape.
///
/// The context owns every container it interns.
///
/// # Panics
///
/// Panics if a column belongs to another context.
pub fn intern(cx: &Context, columns: &[ColumnId]) -> IndexRoot {
    for &column in columns {
        cx.resolve(column);
    }
    let mut layouts = cx.layouts.lock().unwrap();
    let data = match layouts.iter().find(|held| &*held.columns == columns) {
        Some(data) => Arc::clone(data),
        None => {
            let data = Arc::new(IndexData {
                columns: columns.into(),
                shared: Shared::new(Dag::with_capacity(
                    NODE_CAPACITY,
                    EDGE_CAPACITY,
                    (0..2).map(|_| IndexNode {
                        level: Level::TERMINAL,
                        partition: Function::FALSE,
                        hole: u32::MAX,
                    }),
                )),
            });
            layouts.push(Arc::clone(&data));
            data
        }
    };
    drop(layouts);
    // FALSE is pinned, so this root holds no counted reference to acquire
    // here.
    IndexRoot {
        cx: cx.clone(),
        data,
        function: Function::FALSE,
    }
}

impl IndexRoot {
    /// The context owning this root's container.
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// Whether both roots share one container.
    #[inline(always)]
    pub fn same_index(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// The number of columns.
    #[inline(always)]
    pub fn arity(&self) -> usize {
        self.data.columns.len()
    }

    /// The column at position `i`.
    #[inline(always)]
    pub fn column(&self, i: usize) -> ColumnId {
        self.data.columns[i]
    }

    #[inline(always)]
    fn read(&self) -> Read<'_, State> {
        self.data.read(&self.cx)
    }

    /// The number of stored tuples, folded over disjoint partition cells and shared
    /// suffixes.
    ///
    /// Skipped columns contribute their full logical universes, including at a true
    /// root.
    pub fn count(&self) -> BigUint {
        fn count(
            index: &IndexRoot,
            root: Function,
            start: usize,
            bits: &[usize],
            memo: &mut Map<Function, BigUint>,
            scratch: &mut Vec<Refinement>,
        ) -> BigUint {
            if root.is_f() {
                return BigUint::ZERO;
            }
            if root.is_t() {
                return BigUint::ONE << bits[start];
            }
            let (at, partition, hole) = index.head(root);
            let level = at.index();
            if let Some(n) = memo.get(&root) {
                return n << (bits[start] - bits[level]);
            }
            let mut n = BigUint::ZERO;
            let store = index.cx.resolve(index.column(level));
            let mut current = scratch.pop().unwrap_or_default();
            let _ = index.refine_cells(
                root,
                partition,
                hole,
                Function::TRUE,
                &mut current,
                |p, child| {
                    n += store.count(p) * count(index, child, level + 1, bits, memo, scratch);
                    store.dec_ref(p);
                    ControlFlow::<()>::Continue(())
                },
            );
            scratch.push(current);
            memo.insert(root, n.clone());
            n << (bits[start] - bits[level])
        }

        let mut bits = vec![0; self.arity() + 1];
        for i in (0..self.arity()).rev() {
            bits[i] = bits[i + 1] + self.cx.resolve(self.column(i)).sort().width() as usize;
        }
        count(
            self,
            self.function,
            0,
            &bits,
            &mut Map::default(),
            &mut Vec::new(),
        )
    }

    /// The level of the top node of `root`.
    #[inline(always)]
    pub fn top(&self, root: Function) -> Level {
        self.read().node(root).level
    }

    /// The level, partition, and hole of the node `root`, read under one guard.
    ///
    /// The partition borrows the node's reference and stays valid while `root`
    /// stays reachable.
    #[inline(always)]
    pub fn head(&self, root: Function) -> (Level, Partition, u32) {
        let state = self.read();
        let node = state.node(root);
        (
            node.level,
            self.cx.partition(
                self.column(node.level.index()),
                node.partition,
                node.edges.len(),
            ),
            node.hole,
        )
    }

    /// The physical columns this root constrains.
    ///
    /// Predicate construction uses them so a whole-row subidentity does not retain
    /// unrelated prefix cells during suffix collection. The root stays owned
    /// throughout this read-only traversal.
    pub fn constrains(&self) -> Vec<bool> {
        let state = self.read();
        let mut columns = vec![false; self.arity()];
        let mut seen = Set::default();
        let mut stack = vec![self.function];
        while let Some(f) = stack.pop() {
            if f.is_constant() || !seen.insert(f) {
                continue;
            }
            let node = state.node(f);
            columns[node.level.index()] = true;
            stack.extend_from_slice(node.edges);
        }
        columns
    }

    /// The branches of the level node `root` where `p` holds: calls
    /// `then(q, child)` for each slot whose continuation `child` is not `FALSE`,
    /// with the column predicate `q` referenced once and materialized just before
    /// its call, and stops at the first `Break`.
    ///
    /// `root` must stay reachable from a referenced function, and `p` referenced,
    /// for the whole call.
    #[inline(always)]
    pub fn refine<B>(
        &self,
        root: Function,
        p: Function,
        scratch: &mut Refinement,
        then: impl FnMut(Function, Function) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let (_, partition, hole) = self.head(root);
        self.refine_cells(root, partition, hole, p, scratch, then)
    }

    /// [`Self::refine`] with the partition and hole of `root` already read by
    /// [`Self::head`].
    #[inline(always)]
    pub fn refine_cells<B>(
        &self,
        root: Function,
        partition: Partition,
        hole: u32,
        p: Function,
        scratch: &mut Refinement,
        mut then: impl FnMut(Function, Function) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        // The continuation is read under a fresh guard at delivery, released
        // before `then` runs: a callback may grow the store.
        self.cx.refine(
            partition,
            p,
            scratch,
            |t| t != hole,
            |q, t| {
                let child = self.read().node(root).edges[t as usize];
                then(q, child)
            },
        )
    }

    /// Whether this root holds at `words`, the encoded values of every index level
    /// in order.
    ///
    /// # Panics
    ///
    /// Panics if `words` holds other than one encoded value per level or a value
    /// outside its level's sort.
    #[inline(always)]
    pub fn contains(&self, words: &[u64]) -> bool {
        let mut root = self.function;
        let columns = &self.data.columns;
        let sort = |level: usize| self.cx.resolve(columns[level]).sort();
        let total: usize = (0..columns.len()).map(|level| sort(level).words()).sum();
        assert_eq!(
            words.len(),
            total,
            "one encoded value is required per index level"
        );
        let mut at = 0;
        for level in 0..columns.len() {
            let words_at = sort(level).words();
            assert!(
                sort(level).accepts(&words[at..at + words_at]),
                "encoded row contains a value outside its scalar sort"
            );
            at += words_at;
        }
        let state = self.read();
        // Levels only increase along a path, so one running offset follows
        // the visited nodes.
        let (mut level, mut start) = (0, 0);
        loop {
            let node = state.node(root);
            if node.level == Level::TERMINAL {
                return root.is_t();
            }
            while level < node.level.index() {
                start += sort(level).words();
                level += 1;
            }
            let store = self.cx.resolve(columns[level]);
            let width = store.sort().words();
            let slot = store.evaluate(node.partition, &words[start..start + width]);
            root = node.edges[slot as usize];
        }
    }

    /// The partition and continuation of `root` when it is the two-slot node at
    /// `level` whose first slot is FALSE, which is the shape [`Self::cases`]
    /// attaches one output coordinate with; any other node reports `None`.
    ///
    /// The continuation is returned owned, while the partition borrows `root`'s
    /// reference and stays valid while the caller holds `root`.
    #[inline(always)]
    pub fn attachment(&self, root: Function, level: Level) -> Option<(Function, Self)> {
        let state = self.read();
        let node = state.node(root);
        if node.level != level || node.hole != 0 || node.edges.len() != 2 {
            return None;
        }
        let child = node.edges[1];
        if !child.is_constant() {
            state.inc_ref(child);
        }
        Some((node.partition, self.adopt(child)))
    }

    /// Adopts one referenced result of a native operation in this root's container.
    #[inline(always)]
    fn adopt(&self, function: Function) -> Self {
        Self {
            cx: self.cx.clone(),
            data: Arc::clone(&self.data),
            function,
        }
    }

    /// A referenced node continuing with each case's `child` where its `p` holds on
    /// this level's column, and with `default` elsewhere.
    ///
    /// The column owns the split, the index owns the continuations. Equal
    /// continuations share one label, so a case leading to `default` contributes
    /// no additional region to the result. Every explicit case participates in the
    /// store's overlap check: overlapping cases must agree on their continuation.
    /// Each `p` is a borrowed root of `column`: the caller keeps it live across the
    /// call, so the attachment path pays no reference-count traffic for it.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not the column of `level`, overlapping cases disagree,
    /// or a retained node would have a continuation at or above `level`. A level
    /// eliminated by normalization imposes no continuation-order constraint.
    pub fn cases(
        &self,
        level: Level,
        default: Function,
        column: ColumnId,
        cases: &[(Function, Function)],
    ) -> IndexRoot {
        assert_eq!(
            column,
            self.column(level.index()),
            "a base-map root belongs to another column"
        );
        SCRATCH.with_borrow_mut(|s| {
            let Cases {
                of,
                conts,
                labeled,
                labels,
                slots,
            } = &mut s.cases;
            of.clear();
            conts.clear();
            conts.push(default);
            labeled.clear();
            // Labels follow the first appearance of each continuation, as in
            // `values`. Default-continuation cases keep label zero so the
            // backend can check them for conflicting overlaps.
            for &(p, child) in cases {
                let next = if child == default {
                    0
                } else {
                    label(of, conts, child)
                };
                labeled.push((p, next));
            }
            if labeled.is_empty() {
                return self.inc_ref(default);
            }
            labels.clear();
            let partition = self.cx.cases(column, 0, labeled, labels);
            slots.clear();
            slots.extend(labels.iter().map(|&label| conts[label as usize]));
            self.adopt(self.data.branch(&self.cx, level, partition, slots))
        })
    }

    /// A referenced node directly from sorted, distinct packed values and their
    /// nonfalse continuations, with FALSE elsewhere.
    ///
    /// The column places each value itself, so a symbolic level needs no point
    /// predicates here. Distinct continuations are numbered as labels (label 0 is
    /// FALSE, the rest by first appearance), the store reports the label of each
    /// reachable slot, and the slots carry those labels' continuations. `scratch`
    /// is the caller's working memory for this call; see [`Labels`].
    ///
    /// # Panics
    ///
    /// Panics if `keys` and `children` disagree in count, the keys are unsorted,
    /// repeated, or outside the level's sort, a continuation is FALSE, or a
    /// retained node would have a continuation at or above `level`. A level
    /// eliminated by normalization imposes no continuation-order constraint.
    pub fn values(
        &self,
        level: Level,
        keys: &[u64],
        children: &[Function],
        scratch: &mut Labels,
    ) -> IndexRoot {
        scratch.of.clear();
        scratch.conts.clear();
        scratch.conts.push(Function::FALSE);
        scratch.cases.clear();
        for &child in children {
            let label = if child.is_f() {
                0
            } else {
                label(&mut scratch.of, &mut scratch.conts, child)
            };
            scratch.cases.push(label);
        }
        scratch.labels.clear();
        let partition = self.cx.values(
            self.column(level.index()),
            0,
            keys,
            &scratch.cases,
            &mut scratch.labels,
            &mut scratch.values,
        );
        scratch.slots.clear();
        for i in 0..scratch.labels.len() {
            scratch
                .slots
                .push(scratch.conts[scratch.labels[i] as usize]);
        }
        self.adopt(self.data.branch(&self.cx, level, partition, &scratch.slots))
    }

    /// Copies roots of one index into this index's identical trailing columns,
    /// sharing their scalar partitions.
    ///
    /// Every returned root is owned; the invocation-local memo never escapes. This
    /// is an explicit native operation, not a query execution shortcut.
    ///
    /// # Panics
    ///
    /// Panics if a source is wider than this index, belongs to another context or a
    /// different index than the first source, or its columns differ from this
    /// index's trailing columns.
    pub fn copy_suffixes(&self, sources: &[&IndexRoot]) -> Vec<IndexRoot> {
        let Some(source) = sources.first() else {
            return Vec::new();
        };
        let offset = self
            .arity()
            .checked_sub(source.arity())
            .expect("source must be no wider than target");
        assert!(self.cx == source.cx, "native copying requires one context");
        assert!(
            sources.iter().all(|s| s.same_index(source)),
            "sources require one index"
        );
        assert!(
            (0..source.arity()).all(|i| source.column(i) == self.column(offset + i)),
            "suffix columns differ"
        );
        fn copy(
            source: &IndexRoot,
            target: &IndexRoot,
            f: Function,
            offset: usize,
            memo: &mut Map<Function, IndexRoot>,
        ) -> Function {
            if f.is_constant() {
                return f;
            }
            if let Some(root) = memo.get(&f) {
                return root.function;
            }
            let (level, partition, mut children) = {
                let state = source.read();
                let node = state.node(f);
                let partition = source.cx.partition(
                    source.column(node.level.index()),
                    node.partition,
                    node.edges.len(),
                );
                (node.level.index(), partition, node.edges.to_vec())
            };
            // Source roots retain their descendants and scalar partitions
            // across target growth. The memo owns every copied child until all
            // result roots have been retained.
            for child in &mut children {
                *child = copy(source, target, *child, offset, memo);
            }
            partition::inc_ref(&target.cx, partition.column, partition.function);
            let root = target.adopt(target.data.branch(
                &target.cx,
                Level::new(level + offset),
                partition,
                &children,
            ));
            let result = root.function;
            memo.insert(f, root);
            result
        }
        let mut memo = Map::default();
        sources
            .iter()
            .map(|s| self.inc_ref(copy(s, self, s.function, offset, &mut memo)))
            .collect()
    }

    /// The rows of this root that hold, at `level`, the least value stored under
    /// their prefix, keeping everything below that value, as one owned root.
    ///
    /// Every level above `level` is a key, one concrete key path at a time: a
    /// symbolic cell above `level` is one key class per concrete value and never
    /// merges with another cell. Everything below the winning value is kept whole.
    /// Values compare as unsigned bit-vectors, so lower is better. A key path with
    /// no rows stays absent. A path that skips `level`, including a TRUE root,
    /// admits every value of the column, so the least value of the sort wins there.
    /// This is an explicit native operation, not a query execution shortcut.
    ///
    /// # Panics
    ///
    /// Panics if `level` lies beyond the arity or names a symbol column, which has
    /// no numeric order.
    pub fn min_at(&self, level: Level) -> IndexRoot {
        assert!(
            level.index() < self.arity(),
            "level exceeds the index arity"
        );
        assert_ne!(
            self.cx.resolve(self.column(level.index())).sort().kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        fn min(
            index: &IndexRoot,
            root: Function,
            level: Level,
            memo: &mut Map<Function, IndexRoot>,
            labels: &mut Labels,
            scratch: &mut Vec<Refinement>,
        ) -> Function {
            if root.is_f() {
                return root;
            }
            if let Some(r) = memo.get(&root) {
                return r.function;
            }
            let at = index.top(root);
            let store = index.cx.resolve(index.column(level.index()));
            let mut current = scratch.pop().unwrap_or_default();
            let result = if at >= level {
                let (value, child) = if at > level {
                    // The level is skipped, so every value is admitted and the
                    // least one wins.
                    (vec![0; store.sort().words()], root)
                } else {
                    // Improvements swap buffers rather than clone the value.
                    let mut best: Option<Function> = None;
                    let mut least = Vec::new();
                    let mut value = Vec::new();
                    let _ = index.refine(root, Function::TRUE, &mut current, |q, child| {
                        // Members arrive in numeric order, so the first is the
                        // cell's least value. Slot order says nothing about
                        // value order.
                        assert!(
                            store.members(q).next(&mut value),
                            "a partition cell holds a member"
                        );
                        store.dec_ref(q);
                        if best.is_none() || scalar::compare(&value, &least).is_lt() {
                            std::mem::swap(&mut value, &mut least);
                            best = Some(child);
                        }
                        ControlFlow::<()>::Continue(())
                    });
                    (least, best.expect("a nonfalse node must have a cell"))
                };
                // The source root keeps this descendant live. Give construction
                // its own external reference across any collection and retry.
                index.read().inc_ref(child);
                let child = index.adopt(child);
                index.values(level, &value, &[child.function], labels)
            } else {
                // A key level: select under every cell independently, then
                // reattach the cells. `cases` borrows each cell's root, so the
                // cells are released after it.
                let mut cases = Vec::new();
                let _ = index.refine(root, Function::TRUE, &mut current, |q, child| {
                    cases.push((q, min(index, child, level, memo, labels, scratch)));
                    ControlFlow::<()>::Continue(())
                });
                let column = index.column(at.index());
                let node = index.cases(at, Function::FALSE, column, &cases);
                for (q, _) in cases {
                    index.cx.resolve(column).dec_ref(q);
                }
                node
            };
            scratch.push(current);
            let f = result.function;
            memo.insert(root, result);
            f
        }
        // The memo owns every produced node until the result root has been
        // retained.
        let mut memo = Map::default();
        self.inc_ref(min(
            self,
            self.function,
            level,
            &mut memo,
            &mut Labels::default(),
            &mut Vec::new(),
        ))
    }

    /// The owning Boolean combination of two referenced native functions in this
    /// index.
    pub fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> IndexRoot {
        let shared = &self.data.shared;
        let root = SCRATCH.with_borrow_mut(|s| {
            shared.retry((&self.cx, &self.data.columns), |state| {
                debug_assert!(
                    [left, right].into_iter().all(|f| state.has_refs(f)),
                    "cannot use an unreferenced outer-index node",
                );
                // A failed attempt's nodes are unreferenced and collected
                // before the retry, and the growth that collects them bumps
                // the count this memo is keyed against.
                s.restart(shared.id, shared.epoch());
                let root =
                    state.try_apply_bool(&self.cx, &self.data.columns, s, op, left, right)?;
                state.inc_ref(root);
                Ok(root)
            })
        });
        self.adopt(root)
    }

    /// Acquires one reference to a live externally referenced native root
    /// (including pinned constants), returning its owner.
    ///
    /// Dropping the owner decrements the count; callers must not also decrement it
    /// manually.
    ///
    /// # Panics
    ///
    /// Panics if `f` holds no external reference.
    #[must_use = "dropping the returned owner decrements the reference count"]
    #[inline(always)]
    pub fn inc_ref(&self, f: Function) -> Self {
        // Both terminals are pinned when the container is created, so counting
        // them cannot change their state. Skipping it keeps empty roots and
        // pruned branches off the container lock, which every worker shares.
        if !f.is_constant() {
            let state = self.read();
            assert!(
                state.has_refs(f),
                "cannot acquire an unreferenced native root"
            );
            state.inc_ref(f);
        }
        self.adopt(f)
    }
}

impl Clone for IndexRoot {
    #[inline(always)]
    fn clone(&self) -> Self {
        self.inc_ref(self.function)
    }
}

impl Drop for IndexRoot {
    fn drop(&mut self) {
        if !self.function.is_constant() {
            self.read().dec_ref(self.function);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/index.rs"]
mod tests;
