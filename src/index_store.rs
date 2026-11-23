//! Native tuple storage over bound scalar algebras.
//!
//! A node partitions one level's sort and follows a continuation for each
//! class. Nested admission goes from the index to its algebras. Cell
//! delivery takes place outside both stores' guards.

use std::cell::Cell;
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::backend::{Algebra, Align, BoolOp, Function, Refine, Values};
use crate::idx;
use crate::scalar::{self, ScalarKind};
use crate::util::count::Count;
use crate::util::dag::{Dag, Exhausted, Grow, Idx, Map, NeedGrowth, Read, Shared, structural_hash};

/// The level of the two constant nodes, which follows every layout level.
const TERMINAL: u32 = u32::MAX;
/// The memo size above which the next operation starts with an empty memo.
const MEMO_BOUND: usize = 4_000_000;

idx! {
    /// The ID of a tuple set in one index store.
    ///
    /// Equal live IDs in one store denote equal tuple sets. Interning shares
    /// nodes with the same level, partition token, and ordered children.
    pub struct Id;
}

impl Id {
    /// The empty tuple set, which needs no reference.
    pub const FALSE: Self = Self(0);
    /// The complete tuple universe, which needs no reference.
    pub const TRUE: Self = Self(1);

    #[inline(always)]
    fn is_constant(self) -> bool {
        self.0 <= Self::TRUE.0
    }
}

#[derive(Clone, Copy)]
struct Node {
    level: u32,
    partition: Function,
    /// The class that leads to the empty set, or `u32::MAX`.
    hole: u32,
}

impl Node {
    const TERMINAL: Self = Self {
        level: TERMINAL,
        partition: Function::FALSE,
        hole: u32::MAX,
    };
}

/// Owns a temporary partition or cell until it is transferred or released.
struct Held<'a> {
    algebra: &'a dyn Algebra,
    f: Option<Function>,
}

impl<'a> Held<'a> {
    fn new(algebra: &'a dyn Algebra, f: Function) -> Self {
        Self {
            algebra,
            f: Some(f),
        }
    }

    fn function(&self) -> Function {
        self.f.expect("temporary root must still be owned")
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if let Some(f) = self.f {
            self.algebra.dec_ref(f);
        }
    }
}

/// A half-full probe table that numbers children by first appearance.
///
/// A restart clears only the buckets the previous frame filled.
#[derive(Default)]
struct Classes {
    buckets: Vec<(Id, u32)>,
    filled: Vec<usize>,
}

impl Classes {
    fn restart(&mut self, len: usize) {
        for i in self.filled.drain(..) {
            self.buckets[i].1 = 0;
        }
        let want = len
            .max(1)
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .expect("too many index continuations");
        if self.buckets.len() < want {
            self.buckets.resize(want, (Id::FALSE, 0));
        }
    }

    #[inline(always)]
    fn slot(&mut self, child: Id, next: usize) -> u32 {
        let mask = self.buckets.len() - 1;
        // Fibonacci hashing keeps the high bits of the product to mix
        // every bit of the index token.
        let mut i = ((child.index() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            >> (u64::BITS - self.buckets.len().trailing_zeros())) as usize;
        loop {
            let (key, slot) = self.buckets[i];
            if slot == 0 {
                let next = u32::try_from(next + 1).expect("too many distinct continuations");
                self.buckets[i] = (child, next);
                self.filled.push(i);
                return next - 1;
            }
            if key == child {
                return slot - 1;
            }
            i = (i + 1) & mask;
        }
    }
}

#[derive(Default)]
struct Build {
    classes: Classes,
    children: Vec<Id>,
    cases: Vec<(Function, u32)>,
    key_labels: Vec<u32>,
    labels: Vec<u32>,
    slots: Vec<Id>,
    values: Values,
}

impl Build {
    fn restart(&mut self, default: Id, len: usize) {
        self.classes.restart(len + 1);
        self.children.clear();
        self.label(default);
        self.cases.clear();
        self.key_labels.clear();
        self.labels.clear();
        self.slots.clear();
    }

    fn label(&mut self, child: Id) -> u32 {
        let next = self.children.len();
        let label = self.classes.slot(child, next);
        if label as usize == next {
            self.children.push(child);
        }
        label
    }
}

/// Numbers transformed children in the original partition's class order.
#[derive(Default)]
struct Remap {
    classes: Classes,
    slots: Vec<u32>,
}

impl Remap {
    /// Compacts the children in place and returns their partition.
    ///
    /// A single child needs no partition. Otherwise, retains the original
    /// partition or quotients its merged classes. An owned alignment can be
    /// transferred to the new node.
    fn partition<'a, 'b>(
        &mut self,
        children: &'b mut [Id],
        algebra: &'a dyn Algebra,
        f: Function,
        owned: Option<Held<'a>>,
        align: &mut Align,
    ) -> (&'b [Id], Option<Held<'a>>) {
        self.classes.restart(children.len());
        self.slots.clear();
        let mut len = 0;
        for i in 0..children.len() {
            let child = children[i];
            let slot = self.classes.slot(child, len);
            if slot as usize == len {
                children[len] = child;
                len += 1;
            }
            self.slots.push(slot);
        }
        let partition = if len == 1 {
            None
        } else if len == children.len() {
            Some(owned.unwrap_or_else(|| {
                algebra.inc_ref(f);
                Held::new(algebra, f)
            }))
        } else {
            Some(Held::new(algebra, algebra.quotient(f, &self.slots, align)))
        };
        (&children[..len], partition)
    }
}

#[derive(Default)]
struct Scratch {
    memo: Map<(BoolOp, Id, Id), Id>,
    store: u64,
    epoch: u64,
    // Pending operand pairs and completed children occupy nested windows.
    // Each alignment and each remapping can reuse a single buffer.
    labels: Vec<(u32, u32)>,
    pending: Vec<(Id, Id)>,
    children: Vec<Id>,
    remap: Remap,
    align: Align,
}

impl Scratch {
    fn restart(&mut self, store: u64, epoch: u64) {
        if self.store != store || self.epoch != epoch || self.memo.len() > MEMO_BOUND {
            // A fresh map also drops the capacity of a memo that exceeded the bound.
            self.memo = Map::default();
            self.store = store;
            self.epoch = epoch;
        }
        self.pending.clear();
        self.children.clear();
    }
}

thread_local! {
    // Each call takes ownership of its scratch. An algebra can call into
    // another index on the same thread.
    static BUILD: Cell<Build> = Cell::default();
    static APPLY: Cell<Scratch> = Cell::default();
}

type Layout = [(usize, Arc<dyn Algebra>)];
type State = Dag<Id, Node, Id>;

/// One owned partition per released node, batched by bound algebra.
#[derive(Default)]
struct Batches<'a>(Map<usize, (&'a dyn Algebra, Vec<Function>)>);

impl<'a> Batches<'a> {
    fn push(&mut self, layout: &'a Layout, node: &Node) {
        let (id, algebra) = &layout[node.level as usize];
        let (_, roots) = self
            .0
            .entry(*id)
            .or_insert_with(|| (algebra.as_ref(), Vec::new()));
        roots.push(node.partition);
    }

    fn release(self) {
        for (algebra, roots) in self.0.into_values() {
            algebra.dec_refs(&roots);
        }
    }
}

impl Grow for State {
    type Cx<'a> = &'a Layout;

    fn grow(&mut self, exhausted: Exhausted, layout: &Layout) {
        let mut batches = Batches::default();
        self.maintain(
            exhausted,
            1,
            |&child| Some(child),
            |node| batches.push(layout, node),
            |node, edges| structural_hash((node.level, node.partition, edges)),
        );
        batches.release();
    }
}

impl State {
    /// Interns a node, transferring the partition reference on publication.
    ///
    /// A candidate that loses a race keeps its reference until collection.
    /// A hit before publication leaves the reference in `partition`. An
    /// error after publication reacquires a reference there for the retry.
    fn try_intern(
        &self,
        level: u32,
        partition: &mut Held<'_>,
        children: &[Id],
    ) -> Result<Id, NeedGrowth> {
        let f = partition.function();
        let id = self.intern(
            structural_hash((level, f, children)),
            |node, edges| (node.level, node.partition, edges) == (level, f, children),
            || {
                partition.f = None;
                Node {
                    level,
                    partition: f,
                    hole: children
                        .iter()
                        .position(|&child| child == Id::FALSE)
                        .map_or(u32::MAX, |i| i as u32),
                }
            },
            children,
        );
        if id.is_err() && partition.f.is_none() {
            partition.algebra.inc_ref(f);
            partition.f = Some(f);
        }
        id
    }

    /// Aligns partitions, applies the operation per cell, and coarsens the result.
    fn try_apply(
        &self,
        layout: &Layout,
        s: &mut Scratch,
        op: BoolOp,
        mut a: Id,
        mut b: Id,
    ) -> Result<Id, NeedGrowth> {
        struct Frame<'a> {
            key: (BoolOp, Id, Id),
            level: u32,
            f: Function,
            owned: Option<Held<'a>>,
            pending: usize,
            first: usize,
        }

        let mut stack = Vec::<Frame>::new();
        loop {
            if op.is_commutative() && b < a {
                std::mem::swap(&mut a, &mut b);
            }
            let key = (op, a, b);
            let Some(mut id) = op
                .simplify_roots(a, b, Id::FALSE, Id::TRUE)
                .or_else(|| s.memo.get(&key).copied())
            else {
                let left = self.node(a);
                let right = self.node(b);
                let level = left.level.min(right.level);
                let algebra = &*layout[level as usize].1;
                let left_at = left.level == level;
                let right_at = right.level == level;
                let pending = s.pending.len();
                let (f, owned) = if left_at && right_at && left.partition != right.partition {
                    s.labels.clear();
                    let f =
                        algebra.align(left.partition, right.partition, &mut s.labels, &mut s.align);
                    let owned = Held::new(algebra, f);
                    s.pending.extend(
                        s.labels
                            .iter()
                            .rev()
                            .map(|&(i, j)| (left.edges[i as usize], right.edges[j as usize])),
                    );
                    (f, Some(owned))
                } else {
                    // A deeper operand keeps its root at this level. Equal
                    // partitions align along their diagonal.
                    let node = if left_at { &left } else { &right };
                    s.pending.extend((0..node.edges.len()).rev().map(|i| {
                        (
                            if left_at { left.edges[i] } else { a },
                            if right_at { right.edges[i] } else { b },
                        )
                    }));
                    (node.partition, None)
                };
                stack.push(Frame {
                    key,
                    level,
                    f,
                    owned,
                    pending,
                    first: s.children.len(),
                });
                (a, b) = s
                    .pending
                    .pop()
                    .expect("a partition must have a reachable class");
                continue;
            };

            // Rebuild each completed parent before taking the next sibling.
            // The frame owns its alignment across child visits and releases
            // it if this attempt runs out of storage.
            while let Some(parent) = stack.pop_if(|p| p.pending == s.pending.len()) {
                s.children.push(id);
                let Frame {
                    key,
                    level,
                    f,
                    owned,
                    first,
                    ..
                } = parent;
                let algebra = &*layout[level as usize].1;
                let (children, partition) =
                    s.remap
                        .partition(&mut s.children[first..], algebra, f, owned, &mut s.align);
                id = if let Some(mut partition) = partition {
                    self.try_intern(level, &mut partition, children)?
                } else {
                    children[0]
                };
                s.children.truncate(first);
                s.memo.insert(key, id);
            }
            let Some(next) = s.pending.pop() else {
                return Ok(id);
            };
            s.children.push(id);
            (a, b) = next;
        }
    }
}

/// The decision DAG shared by every tuple set over one ordered layout.
///
/// Operations borrow their operand IDs, which the caller keeps referenced.
/// Each returned ID carries one reference for the caller to release with
/// [`Self::dec_ref`]. Constants are unaffected by reference counting.
///
/// Operations can enter the layout's algebras, including during
/// maintenance, so callers run them outside scalar-store guards.
pub struct Store {
    layout: Box<Layout>,
    shared: Shared<State>,
}

impl Store {
    /// Creates a store over the layout holding only the two constants.
    ///
    /// Each level pairs its algebra with the index that names it to the
    /// owner. The layout has at most `u32::MAX` levels.
    ///
    /// # Panics
    ///
    /// Panics if `nodes` is below two, a capacity exceeds `u32::MAX`, or the
    /// arena or unique-table sizes exceed the target's addressable capacity.
    pub fn new(layout: Box<Layout>, nodes: usize, edges: usize) -> Self {
        Self {
            layout,
            shared: Shared::new(Dag::with_capacity(nodes, edges, [Node::TERMINAL; 2])),
        }
    }

    /// The algebra of each level, in level order.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[inline(always)]
    fn read(&self) -> Read<'_, State> {
        self.shared.read(&self.layout)
    }

    /// Whether one packed value per level forms a member of `root`.
    ///
    /// # Panics
    ///
    /// Panics if `words` has the wrong length or a value is outside its
    /// level's sort, including at levels the root skips.
    pub fn contains(&self, mut root: Id, mut words: &[u64]) -> bool {
        let state = self.read();
        let mut node = state.node(root);
        for (i, (_, algebra)) in self.layout.iter().enumerate() {
            let sort = algebra.sort();
            let value = words
                .split_off(..sort.words())
                .expect("one encoded value is required per index level");
            if node.level as usize == i {
                let slot = algebra.evaluate(node.partition, value);
                root = node.edges[slot as usize];
                node = state.node(root);
            } else {
                assert!(
                    sort.accepts(value),
                    "value is outside its index level's sort"
                );
            }
        }
        assert!(
            words.is_empty(),
            "one encoded value is required per index level"
        );
        root == Id::TRUE
    }

    /// The exact number of tuples in `root`, including every choice at the
    /// levels it skips.
    pub fn count(&self, root: Id) -> Count {
        if root == Id::FALSE {
            return Count::ZERO;
        }
        let mut bits = vec![0; self.layout.len() + 1];
        for i in (0..self.layout.len()).rev() {
            bits[i] = bits[i + 1] + u64::from(self.layout[i].1.sort().width());
        }
        if root == Id::TRUE {
            return Count::from(1).shift(bits[0]);
        }

        // Count each child before refining its parent. A suspended parent
        // resumes at its next edge. The live root retains every descendant
        // and its partition between index admissions.
        let mut memo = Map::default();
        let mut stack = Vec::new();
        let mut scratch = Refine::default();
        let (mut id, mut next) = (root, 0);
        loop {
            let node = {
                let state = self.read();
                let node = state.node(id);
                let edges = &node.edges[next..];
                if let Some(i) = edges
                    .iter()
                    .position(|c| !c.is_constant() && !memo.contains_key(c))
                {
                    stack.push((id, next + i + 1));
                    (id, next) = (edges[i], 0);
                    continue;
                }
                *node.value
            };
            let level = node.level as usize;
            let algebra = &*self.layout[level].1;
            let mut n = Count::ZERO;
            let _ = algebra.refine(
                node.partition,
                Function::TRUE,
                &mut scratch,
                &mut |i| i != node.hole,
                &mut |i, build| {
                    let child = self.read().node(id).edges[i as usize];
                    let cell = Held::new(algebra, build());
                    let members = algebra.count(cell.function());
                    n += if child == Id::TRUE {
                        members.shift(bits[level + 1])
                    } else {
                        let (lower, count) = &memo[&child];
                        (&members * count).shift(bits[level + 1] - bits[*lower])
                    };
                    ControlFlow::Continue(())
                },
            );
            let Some(parent) = stack.pop() else {
                return n.shift(bits[0] - bits[level]);
            };
            memo.insert(id, (level, n));
            (id, next) = parent;
        }
    }

    /// The rows with the least unsigned value at `level` for each prefix.
    pub fn min_at(&self, root: Id, level: usize) -> Id {
        assert!(level < self.layout.len(), "level exceeds the index arity");
        let rank = self.layout[level].1.sort();
        assert_ne!(
            rank.kind(),
            ScalarKind::Symbol,
            "symbols have no numeric ordering"
        );
        if root == Id::FALSE {
            return root;
        }

        // Each completed result owns a reference across later construction
        // and collection. The source root retains its original descendants.
        struct Results<'a> {
            store: &'a Store,
            roots: Map<Id, Id>,
        }

        impl Drop for Results<'_> {
            fn drop(&mut self) {
                for &id in self.roots.values() {
                    self.store.dec_ref(id);
                }
            }
        }

        let mut memo = Results {
            store: self,
            roots: Map::default(),
        };
        let mut stack = Vec::new();
        let mut scratch = Refine::default();
        let mut least = Vec::new();
        let mut children = Vec::new();
        let mut remap = Remap::default();
        let mut align = Align::default();
        let (mut id, mut next) = (root, 0);
        loop {
            let node = {
                let state = self.read();
                let node = state.node(id);
                if (node.level as usize) < level {
                    if let Some(i) = node.edges[next..]
                        .iter()
                        .position(|c| *c != Id::FALSE && !memo.roots.contains_key(c))
                    {
                        stack.push((id, next + i + 1));
                        (id, next) = (node.edges[next + i], 0);
                        continue;
                    }
                    // The original partition also selects its empty class.
                    // Preserve that edge when transforming a partial key domain.
                    children.clear();
                    children.extend(node.edges.iter().map(|c| {
                        if *c == Id::FALSE {
                            Id::FALSE
                        } else {
                            memo.roots[c]
                        }
                    }));
                }
                *node.value
            };
            let at = node.level as usize;
            let result = if at >= level {
                least.clear();
                let child = if at > level {
                    // A skipped rank admits every value, including zero.
                    least.resize(rank.words(), 0);
                    id
                } else {
                    let algebra = &*self.layout[level].1;
                    let mut best = None;
                    let _ = algebra.refine(
                        node.partition,
                        Function::TRUE,
                        &mut scratch,
                        &mut |i| i != node.hole,
                        &mut |i, build| {
                            let cell = Held::new(algebra, build());
                            let mut members = algebra.members(cell.function());
                            // Members arrive in unsigned order. Class numbers
                            // do not determine which class has the least value.
                            let value =
                                members.next().expect("a partition cell must hold a member");
                            if best.is_none() || scalar::compare(value, &least).is_lt() {
                                least.clear();
                                least.extend_from_slice(value);
                                best = Some(i as usize);
                            }
                            ControlFlow::Continue(())
                        },
                    );
                    let i = best.expect("a nonempty node must have a cell");
                    self.read().node(id).edges[i]
                };
                self.values(level, &least, std::iter::once(child))
            } else {
                let (children, partition) = remap.partition(
                    &mut children,
                    &*self.layout[at].1,
                    node.partition,
                    None,
                    &mut align,
                );
                if let Some(f) = partition {
                    self.branch(at, f, children)
                } else {
                    self.inc_ref(children[0]);
                    children[0]
                }
            };
            let Some(parent) = stack.pop() else {
                return result;
            };
            memo.roots.insert(id, result);
            (id, next) = parent;
        }
    }

    fn branch(&self, level: usize, mut f: Held<'_>, children: &[Id]) -> Id {
        if let [id] = *children {
            self.inc_ref(id);
            return id;
        }
        assert!(
            !children.is_empty() && !f.function().is_constant(),
            "partition must supply one continuation per class"
        );
        self.shared.retry(&self.layout, |state| {
            let level = level as u32;
            assert!(
                children.iter().all(|&id| state.node(id).level > level),
                "index child must follow its parent level"
            );
            let id = state.try_intern(level, &mut f, children)?;
            state.inc_ref(id);
            Ok(id)
        })
    }

    fn build(
        &self,
        level: usize,
        default: Id,
        len: usize,
        build: impl FnOnce(&dyn Algebra, &mut Build) -> Function,
    ) -> Id {
        let algebra = &*self.layout[level].1;
        let mut s = BUILD.take();
        s.restart(default, len);
        let f = Held::new(algebra, build(algebra, &mut s));
        s.slots
            .extend(s.labels.iter().map(|&i| s.children[i as usize]));
        let id = self.branch(level, f, &s.slots);
        BUILD.set(s);
        id
    }

    /// Returns the set that follows each predicate of the level's algebra
    /// to its continuation, and `default` elsewhere.
    ///
    /// A level with one reachable continuation collapses to that child.
    ///
    /// # Panics
    ///
    /// Panics if `level` is outside the layout, overlapping cases lead to
    /// different root IDs, or a retained child starts at or before
    /// `level`.
    pub fn cases(
        &self,
        level: usize,
        default: Id,
        cases: impl ExactSizeIterator<Item = (Function, Id)>,
    ) -> Id {
        self.build(level, default, cases.len(), |algebra, s| {
            // Explicit default cases keep label zero so they still take
            // part in overlap checking.
            for (predicate, child) in cases {
                let label = s.label(child);
                s.cases.push((predicate, label));
            }
            algebra.cases(0, &s.cases, &mut s.labels)
        })
    }

    /// Returns the set that follows each packed key to its continuation,
    /// and is empty elsewhere.
    ///
    /// # Panics
    ///
    /// Panics if `level` is outside the layout, a child is `FALSE`, the keys
    /// break the packing of [`Algebra::set`] or differ in number from the
    /// children, or a retained child starts at or before `level`.
    pub fn values(
        &self,
        level: usize,
        keys: &[u64],
        children: impl ExactSizeIterator<Item = Id>,
    ) -> Id {
        self.build(level, Id::FALSE, children.len(), |algebra, s| {
            for child in children {
                assert!(child != Id::FALSE, "values require nonempty continuations");
                let label = s.label(child);
                s.key_labels.push(label);
            }
            algebra.values(0, keys, &s.key_labels, &mut s.labels, &mut s.values)
        })
    }

    /// Returns the set `op(a, b)`, where `AndNot` is `a` minus `b`.
    pub fn apply(&self, op: BoolOp, a: Id, b: Id) -> Id {
        let mut s = APPLY.take();
        let id = self.shared.retry(&self.layout, |state| {
            // Check the epoch while admitted. A failed attempt may
            // leave unreferenced results for maintenance to reclaim.
            s.restart(self.shared.id, self.shared.epoch());
            let id = state.try_apply(&self.layout, &mut s, op, a, b)?;
            if !id.is_constant() {
                state.inc_ref(id);
            }
            Ok(id)
        });
        APPLY.set(s);
        id
    }

    /// Acquires one reference to a live root.
    #[inline(always)]
    pub fn inc_ref(&self, id: Id) {
        if !id.is_constant() {
            self.read().inc_ref(id);
        }
    }

    /// Releases one owned reference to a root.
    ///
    /// Maintenance reclaims the nodes that no referenced root reaches.
    #[inline(always)]
    pub fn dec_ref(&self, id: Id) {
        if !id.is_constant() {
            self.read().dec_ref(id);
        }
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        let state = self.shared.get_mut();
        let mut batches = Batches::default();
        // Reclaimed nodes already released their partitions. Every
        // remaining nonterminal, including losing candidates, owns one.
        for i in 2..state.node_capacity() {
            if let Some(node) = state.get(Id::new(i)) {
                batches.push(&self.layout, node.value);
            }
        }
        batches.release();
    }
}
