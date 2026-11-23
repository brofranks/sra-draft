//! Context-owned native indexes over ordered scalar positions.
//!
//! An index can share storage across roots with the same physical layout.
//! Each position remains independent even when several positions use the
//! same algebra. Logical declarations and rows live above this layer.

use std::fmt;
use std::sync::Arc;

use super::{AlgebraId, Context, Predicate};
use crate::backend::BoolOp;
use crate::index_store::{Id, Store};
use crate::util::count::Count;

// Amortize the fixed cost of collecting transient results.
const NODE_CAPACITY: usize = 65_536;
const EDGE_CAPACITY: usize = 262_144;

/// The physical storage shared by native roots over one ordered layout.
///
/// A context interns one index for each exact sequence of algebra IDs. An
/// index retains the context and gives its roots access to the bound algebras.
///
/// Index operations can enter scalar stores directly or during maintenance.
/// Dropping an index that is the final owner of its context can also enter
/// scalar stores to release partitions. Use and drop indexes outside
/// scalar-store guards.
///
/// Membership and Boolean operations can call a custom algebra while
/// holding index admission. Index maintenance can call its
/// [`dec_refs`](crate::backend::Algebra::dec_refs) method with exclusive
/// access. An algebra running under either guard must not reenter that
/// index, including by cloning or dropping a root. Nested calls to other
/// indexes must follow a consistent lock order across threads.
#[derive(Clone)]
pub struct Index {
    // The context retains the registry until this handle releases its store.
    store: Arc<Store>,
    cx: Context,
}

impl Index {
    /// The index over the specified algebras, in physical level order.
    ///
    /// # Panics
    ///
    /// Panics if an algebra belongs to another context or the layout has
    /// more levels than the native index can name.
    pub fn new(cx: &Context, algebras: &[AlgebraId]) -> Self {
        Self::with_capacity(cx, algebras, NODE_CAPACITY, EDGE_CAPACITY)
    }

    /// The index over the specified algebras with an initial arena capacity.
    ///
    /// The first request to register the layout sets its initial capacities.
    /// Subsequent requests for the layout use the same index.
    ///
    /// # Panics
    ///
    /// Panics if an algebra is foreign, fewer than two nodes are reserved,
    /// or the layout or a capacity exceeds the native index range. For a new
    /// index, panics if the arena or unique-table sizes exceed the target's
    /// addressable capacity.
    pub fn with_capacity(cx: &Context, algebras: &[AlgebraId], nodes: usize, edges: usize) -> Self {
        assert!(
            algebras.iter().all(|id| id.context == cx.sorts().id()),
            "algebra belongs to another context"
        );
        assert!(algebras.len() <= u32::MAX as usize, "too many index levels");
        assert!(
            (2..=u32::MAX as usize).contains(&nodes),
            "index node capacity must fit its terminals and node IDs"
        );
        assert!(edges <= u32::MAX as usize, "index edge capacity overflow");
        let registry = &cx.data.indexes;
        let lock = || registry.lock().expect("index registry must be usable");
        let find = |indexes: &[Arc<Store>]| {
            indexes
                .iter()
                .find(|store| {
                    store
                        .layout()
                        .iter()
                        .map(|&(index, _)| index)
                        .eq(algebras.iter().map(|id| id.index))
                })
                .cloned()
        };
        let existing = find(&lock());
        let store = existing.unwrap_or_else(|| {
            let layout = algebras
                .iter()
                .map(|id| (id.index, Arc::clone(&cx.data.algebras[id.index].1)))
                .collect();
            // Arena construction can panic on the target's capacity limits.
            let candidate = Arc::new(Store::new(layout, nodes, edges));
            let mut indexes = lock();
            if let Some(store) = find(&indexes) {
                return store;
            }
            indexes.push(Arc::clone(&candidate));
            candidate
        });
        Self {
            store,
            cx: cx.clone(),
        }
    }

    /// The context that owns this index.
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// The number of scalar positions in the physical layout.
    #[inline(always)]
    pub fn arity(&self) -> usize {
        self.store.layout().len()
    }

    /// The algebra interpreting the level.
    ///
    /// # Panics
    ///
    /// Panics if the level is outside the layout.
    #[inline(always)]
    pub fn algebra(&self, level: usize) -> AlgebraId {
        AlgebraId {
            context: self.cx.sorts().id(),
            index: self.store.layout()[level].0,
        }
    }

    /// The empty tuple set over this layout.
    pub fn empty(&self) -> IndexRoot {
        self.adopt(Id::FALSE)
    }

    /// The complete tuple universe over this layout.
    pub fn full(&self) -> IndexRoot {
        self.adopt(Id::TRUE)
    }

    fn adopt(&self, id: Id) -> IndexRoot {
        IndexRoot {
            index: self.clone(),
            id,
        }
    }

    fn id(&self, root: &IndexRoot) -> Id {
        assert!(
            self == &root.index,
            "index roots must share a physical layout"
        );
        root.id
    }

    /// Builds a level from scalar predicates and their continuations.
    ///
    /// Values outside the explicit cases follow `default`. Overlapping
    /// explicit cases must select the same native root, including cases
    /// that select `default`. A level with one reachable continuation
    /// collapses to that child.
    ///
    /// # Panics
    ///
    /// Panics if the level is outside the layout, an operand belongs to
    /// another index or algebra, overlapping cases select different roots,
    /// or a retained child starts at or before this level.
    pub fn cases(
        &self,
        level: usize,
        default: &IndexRoot,
        cases: &[(&Predicate, &IndexRoot)],
    ) -> IndexRoot {
        let algebra = self.algebra(level);
        let default = self.id(default);
        let cases = cases.iter().map(|&(p, child)| {
            assert!(p.algebra() == algebra, "case must use the level's algebra");
            (p.function(), self.id(child))
        });
        let id = self.store.cases(level, default, cases);
        self.adopt(id)
    }

    /// Builds a level from distinct keys in increasing numeric order.
    ///
    /// Keys use the packed encoding of
    /// [`Algebra::set`](crate::backend::Algebra::set). Each value has a
    /// nonempty continuation. Values outside the keys follow the empty
    /// continuation.
    ///
    /// # Panics
    ///
    /// Panics if the level is outside the layout, keys are malformed or
    /// disagree with the child count, a child is empty or belongs to another
    /// index, or a retained child starts at or before this level.
    pub fn values(&self, level: usize, keys: &[u64], children: &[IndexRoot]) -> IndexRoot {
        let id = self
            .store
            .values(level, keys, children.iter().map(|child| self.id(child)));
        self.adopt(id)
    }
}

impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.store, &other.store)
    }
}

impl Eq for Index {}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Index")
            .field("context", &self.cx)
            .field("arity", &self.arity())
            .finish_non_exhaustive()
    }
}

/// One owned reference to a native tuple set in its index.
///
/// Cloning retains the root and dropping releases it. The index traces its
/// continuations for as long as the root is retained.
///
/// Use, clone, and drop roots outside scalar-store guards. See [`Index`].
#[must_use = "native operations return a new index root"]
#[derive(Debug)]
pub struct IndexRoot {
    index: Index,
    id: Id,
}

impl IndexRoot {
    /// The physical index interpreting this root.
    #[inline(always)]
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Whether this root represents the empty tuple set.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.id == Id::FALSE
    }

    /// Whether one packed value per level forms a member of the root.
    ///
    /// # Panics
    ///
    /// Panics if the tuple has the wrong word count or a value is outside its
    /// level's sort, including at levels skipped by the root.
    pub fn contains(&self, words: &[u64]) -> bool {
        self.index.store.contains(self.id, words)
    }

    /// The exact number of tuples, including all choices at skipped levels.
    pub fn count(&self) -> Count {
        self.index.store.count(self.id)
    }

    /// The rows with the least unsigned value at `level` under each prefix.
    ///
    /// Each prefix keeps every suffix of its winning value. A prefix that
    /// skips the selected level admits its whole sort, so zero wins there.
    ///
    /// # Panics
    ///
    /// Panics if the level is outside the layout or has a symbol sort.
    pub fn min_at(&self, level: usize) -> Self {
        self.index.adopt(self.index.store.min_at(self.id, level))
    }

    /// The native Boolean combination of two tuple sets in the same index.
    ///
    /// # Panics
    ///
    /// Panics if the other root belongs to another index.
    pub fn apply(&self, op: BoolOp, other: &Self) -> Self {
        let id = self.index.store.apply(op, self.id, self.index.id(other));
        self.index.adopt(id)
    }

    /// The complement of this root over its complete tuple universe.
    pub fn not(&self) -> Self {
        let id = self.index.store.apply(BoolOp::AndNot, Id::TRUE, self.id);
        self.index.adopt(id)
    }
}

impl Clone for IndexRoot {
    fn clone(&self) -> Self {
        self.index.store.inc_ref(self.id);
        self.index.adopt(self.id)
    }
}

impl Drop for IndexRoot {
    fn drop(&mut self) {
        self.index.store.dec_ref(self.id);
    }
}
