//! Native tuple storage over bound scalar algebras.
//!
//! A node partitions one level's sort and follows a continuation for each
//! class. Nested admission goes from the index to its algebras. Cell
//! delivery takes place outside both stores' guards.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::backend::{Algebra, Align, BoolOp, Function, Refine, Values};
use crate::idx;
use crate::util::count::Count;
use crate::util::dag::{Dag, Exhausted, Grow, Idx, Map, NeedGrowth, Read, Shared, structural_hash};

/// The level of the two constant nodes, which follows every layout level.
const TERMINAL: u32 = u32::MAX;
/// Drop an oversized memo between operations.
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
    /// The class that leads to FALSE, or u32::MAX.
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

/// Number children by first appearance with a half-full probe table. Clear
/// only the buckets the previous frame filled.
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

#[derive(Default)]
struct Scratch {
    memo: Map<(BoolOp, Id, Id), Id>,
    container: u64,
    epoch: u64,
    // The label and child windows nest. Classing begins after every child
    // returns, so one remapping buffer suffices for all frames.
    labels: Vec<(u32, u32)>,
    children: Vec<Id>,
    slots: Vec<u32>,
    classes: Classes,
    align: Align,
}

impl Scratch {
    fn restart(&mut self, container: u64, epoch: u64) {
        if self.container != container || self.epoch != epoch || self.memo.len() > MEMO_BOUND {
            // Drop capacity as well as entries when the bound is exceeded.
            self.memo = Map::default();
            self.container = container;
            self.epoch = epoch;
        }
        self.labels.clear();
        self.children.clear();
    }
}

thread_local! {
    static BUILD: RefCell<Build> = RefCell::default();
    static APPLY: RefCell<Scratch> = RefCell::default();
}

type Layout = [(usize, Arc<dyn Algebra>)];
type State = Dag<Id, Node, Id>;

fn release(layout: &Layout, roots: Vec<Vec<Function>>) {
    for ((_, algebra), roots) in layout.iter().zip(roots) {
        if !roots.is_empty() {
            algebra.dec_refs(&roots);
        }
    }
}

impl Grow for State {
    type Cx<'a> = &'a Layout;

    fn grow(&mut self, exhausted: Exhausted, layout: &Layout) {
        let mut roots = vec![Vec::new(); layout.len()];
        self.maintain(
            exhausted,
            1,
            |&child| Some(child),
            |node| roots[node.level as usize].push(node.partition),
            |node, edges| structural_hash((node.level, node.partition, edges)),
        );
        release(layout, roots);
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
            || Node {
                level,
                partition: partition
                    .f
                    .take()
                    .expect("published partition must be owned"),
                hole: children
                    .iter()
                    .position(|&child| child == Id::FALSE)
                    .map_or(u32::MAX, |i| i as u32),
            },
            children,
        );
        if id.is_err() && partition.f.is_none() {
            partition.algebra.inc_ref(f);
            partition.f = Some(f);
        }
        id
    }

    /// Align partitions, apply per cell, then coarsen by equal continuations.
    fn try_apply(
        &self,
        layout: &Layout,
        s: &mut Scratch,
        op: BoolOp,
        mut a: Id,
        mut b: Id,
    ) -> Result<Id, NeedGrowth> {
        if op.is_commutative() && b < a {
            std::mem::swap(&mut a, &mut b);
        }
        if let Some(id) = op.simplify_roots(a, b, Id::FALSE, Id::TRUE) {
            return Ok(id);
        }
        let key = (op, a, b);
        if let Some(&id) = s.memo.get(&key) {
            return Ok(id);
        }

        let left = self.node(a);
        let right = self.node(b);
        let level = left.level.min(right.level);
        let algebra = &*layout[level as usize].1;
        let left_at = left.level == level;
        let right_at = right.level == level;
        let start = s.labels.len();
        let owned = (left_at && right_at && left.partition != right.partition).then(|| {
            Held::new(
                algebra,
                algebra.align(left.partition, right.partition, &mut s.labels, &mut s.align),
            )
        });
        let f = match &owned {
            Some(partition) => partition.function(),
            None => {
                // A deeper operand has one class at this level. Equal
                // partitions align along their diagonal.
                let node = if left_at { &left } else { &right };
                s.labels.extend(
                    (0..node.edges.len() as u32)
                        .map(|i| (i * u32::from(left_at), i * u32::from(right_at))),
                );
                node.partition
            }
        };

        let first = s.children.len();
        for i in start..s.labels.len() {
            let (j, k) = s.labels[i];
            let l = if left_at { left.edges[j as usize] } else { a };
            let r = if right_at { right.edges[k as usize] } else { b };
            let child = self.try_apply(layout, s, op, l, r)?;
            s.children.push(child);
        }
        s.labels.truncate(start);

        let mut len = 0;
        s.slots.clear();
        s.classes.restart(s.children.len() - first);
        // Compact just this frame's children after all recursion returns.
        for i in first..s.children.len() {
            let child = s.children[i];
            let slot = s.classes.slot(child, len);
            if slot as usize == len {
                s.children[first + len] = child;
                len += 1;
            }
            s.slots.push(slot);
        }
        let children = &s.children[first..first + len];
        let id = if len == 1 {
            children[0]
        } else {
            let mut partition = if len != s.slots.len() {
                let q = Held::new(algebra, algebra.quotient(f, &s.slots, &mut s.align));
                drop(owned);
                q
            } else {
                owned.unwrap_or_else(|| {
                    algebra.inc_ref(f);
                    Held::new(algebra, f)
                })
            };
            self.try_intern(level, &mut partition, children)?
        };
        s.children.truncate(first);
        s.memo.insert(key, id);
        Ok(id)
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
            let (value, rest) = words
                .split_at_checked(sort.words())
                .expect("one encoded value is required per index level");
            words = rest;
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
        fn count(
            store: &Store,
            root: Id,
            start: usize,
            bits: &[u64],
            memo: &mut Map<Id, (usize, Count)>,
            scratch: &mut Vec<Refine>,
        ) -> Count {
            if root == Id::TRUE {
                return Count::from(1).shift(bits[start]);
            }
            if let Some((level, n)) = memo.get(&root) {
                return n.clone().shift(bits[start] - bits[*level]);
            }
            let (level, partition, hole) = {
                let state = store.read();
                let node = state.node(root);
                (node.level as usize, node.partition, node.hole)
            };
            let algebra = &*store.layout[level].1;
            let mut local = scratch.pop().unwrap_or_default();
            let mut n = Count::ZERO;
            let _ = algebra.refine(
                partition,
                Function::TRUE,
                &mut local,
                &mut |i| i != hole,
                &mut |i, build| {
                    let child = store.read().node(root).edges[i as usize];
                    let cell = Held::new(algebra, build());
                    let members = algebra.count(cell.function());
                    n += &members * &count(store, child, level + 1, bits, memo, scratch);
                    ControlFlow::Continue(())
                },
            );
            scratch.push(local);
            memo.insert(root, (level, n.clone()));
            n.shift(bits[start] - bits[level])
        }

        if root == Id::FALSE {
            return Count::ZERO;
        }
        let mut bits = vec![0; self.layout.len() + 1];
        for i in (0..self.layout.len()).rev() {
            bits[i] = bits[i + 1] + u64::from(self.layout[i].1.sort().width());
        }
        count(self, root, 0, &bits, &mut Map::default(), &mut Vec::new())
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
        BUILD.with_borrow_mut(|s| {
            s.restart(default, len);
            let f = Held::new(algebra, build(algebra, s));
            s.slots
                .extend(s.labels.iter().map(|&i| s.children[i as usize]));
            self.branch(level, f, &s.slots)
        })
    }

    /// Returns the set that follows each predicate of the level's algebra
    /// to its continuation, and `default` elsewhere.
    ///
    /// A level with one reachable continuation collapses to that child.
    ///
    /// # Panics
    ///
    /// Panics if `level` is outside the layout, overlapping cases lead to
    /// different continuations, or a retained child starts at or before
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
        APPLY.with_borrow_mut(|s| {
            self.shared.retry(&self.layout, |state| {
                // Check the epoch while admitted. A failed attempt may
                // leave unreferenced results for maintenance to reclaim.
                s.restart(self.shared.id, self.shared.epoch());
                let id = state.try_apply(&self.layout, s, op, a, b)?;
                if !id.is_constant() {
                    state.inc_ref(id);
                }
                Ok(id)
            })
        })
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
        let mut roots = vec![Vec::new(); self.layout.len()];
        // Reclaimed nodes already released their partitions. Every
        // remaining nonterminal, including losing candidates, owns one.
        for i in 2..state.node_capacity() {
            if let Some(node) = state.get(Id::new(i)) {
                roots[node.level as usize].push(node.partition);
            }
        }
        release(&self.layout, roots);
    }
}

#[cfg(test)]
#[path = "../tests/unit/index_store.rs"]
mod tests;
