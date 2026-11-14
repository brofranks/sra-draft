//! Context-owned value dictionaries over named 64-bit symbol ID universes.
//!
//! Dictionary growth never changes a sort or an existing predicate. Every u64
//! is a valid ID, but only assigned IDs have values. Use an explicit relation
//! of known values for application-domain complements and counts;
//! unrestricted operations use all 2^64 IDs.
//!
//! Any `Hash + Eq` Rust value can be interned: strings, sets of other
//! interned values, paths. Relations store only the ID, and symbolic
//! operations see an opaque scalar. Applications decode values through
//! [`Dictionary::resolve`] and intern results through [`Dictionary::intern`].
//!
//! Sets use `FiniteSet<Capacity<D, N>>`: exact values and symbolic families
//! share one bounded membership-mask representation. BDD updates use native
//! bit transformations; sparse updates enumerate admitted values. Interning a
//! Rust collection makes it an opaque value, without the membership semantics
//! of `FiniteSet`. Capacity changes require explicit reconstruction; a set
//! never widens or switches representation automatically.
//!
//! String literals intern wherever they pack, including `contains` and `eq`,
//! so a lookup miss still assigns an ID and counts toward a `Capacity`. Use
//! [`Dictionary::get`] to look up a value without adding it.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use super::{Cons, Nil, Sorted, Universe};
use crate::ast::{AstId, Sort};
use crate::context::{Context, Interner};
use crate::row::{IntoRow, RowRef, RowSink};

/// Names a symbol namespace and its interned Rust value type, shared by raw
/// and typed declarations in each context.
///
/// Equal names in one context share
/// one dictionary, including across different marker types; they must agree
/// on `Value`. Typed sort lookup establishes or checks that value type.
/// Explicit handles carry their context identity; crossing
/// contexts requires decoding the value and interning it in the destination
/// dictionary.
pub trait Domain: 'static {
    /// The interned Rust value type.
    type Value: ?Sized + Hash + Eq + Send + Sync + 'static;
    /// The dictionary name shared by every declaration of this namespace in
    /// a context.
    const NAME: &'static str;
}

/// One context-associated ID.
///
/// The value retains no dictionary or context
/// allocation; its identity includes its originating context. ID order is
/// not value order. Raw IDs and explicit word packing can name values with
/// no dictionary entry.
pub struct Interned<D> {
    /// The 64-bit symbol ID.
    pub id: u64,
    context: u64,
    sort: AstId,
    domain: PhantomData<fn() -> D>,
}

impl<D> Interned<D> {
    #[inline(always)]
    const fn new(id: u64, sort: Sort) -> Self {
        Self {
            id,
            context: sort.context,
            sort: sort.id,
            domain: PhantomData,
        }
    }
}

impl<D> Copy for Interned<D> {}
impl<D> Clone for Interned<D> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<D> PartialEq for Interned<D> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.id == other.id
    }
}
impl<D> Eq for Interned<D> {}
impl<D> PartialOrd for Interned<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<D> Ord for Interned<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.context, self.id).cmp(&(other.context, other.id))
    }
}
impl<D> Hash for Interned<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.context, self.id).hash(state);
    }
}
impl<D> fmt::Debug for Interned<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interned")
            .field("context", &self.context)
            .field("id", &self.id)
            .finish()
    }
}

fn interner<D: Domain>(cx: &Context) -> Arc<Interner<D::Value>> {
    cx.interner::<D::Value>(cx.symbol_sort(D::NAME))
}

impl<D: Domain> Sorted for Interned<D> {
    type Shape = Cons<Self, Nil>;
    #[inline(always)]
    fn sort(cx: &Context) -> Sort {
        interner::<D>(cx).sort
    }
}
impl<D: Domain> IntoRow for Interned<D> {
    #[inline(always)]
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert_eq!(
            self.context, row.cx.id,
            "interned value belongs to another context"
        );
        assert_eq!(
            row.sort.id, self.sort,
            "interned value belongs to another namespace"
        );
        out.words().push(self.id);
    }
}

/// String conversion in the consuming row's context.
///
/// Interning occurs while packing literals and rows, not during backend
/// traversal. Both `str` and `String` literals use the `str` table, so the
/// namespace must declare `Value = str`.
impl IntoRow for str {
    #[inline(always)]
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        out.words()
            .push(row.cx.interner::<str>(row.sort).intern(self));
    }
}
impl IntoRow for String {
    #[inline(always)]
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        self.as_str().append(row, out);
    }
}

/// A typed handle to a context's growing value dictionary.
///
/// Cloning shares the table.
///
/// The context and handles retain the same table, so there are no competing
/// dictionaries for one namespace. Interning is synchronized; decoded values
/// and iteration snapshots own their storage and hold no locks. Dropping
/// local handles does not remove values from a live context.
pub struct Dictionary<D: Domain> {
    table: Arc<Interner<D::Value>>,
    domain: PhantomData<fn() -> D>,
}

impl<D: Domain> Dictionary<D> {
    /// Gets this namespace's shared dictionary in `cx`.
    ///
    /// # Panics
    ///
    /// Panics if the name conflicts with another sort declaration or the
    /// namespace already holds another value type.
    pub fn new(cx: &Context) -> Self {
        Self {
            table: interner::<D>(cx),
            domain: PhantomData,
        }
    }

    /// The handle of `id` in this dictionary's context.
    ///
    /// Use it to reconstruct a handle, for example from a member cursor. This
    /// does not assert a value; every u64 remains a valid ID.
    #[inline(always)]
    pub fn interned(&self, id: u64) -> Interned<D> {
        Interned::new(id, self.table.sort)
    }

    /// The handle of `value`, without adding it to the dictionary.
    pub fn get(&self, value: &D::Value) -> Option<Interned<D>> {
        self.table.get(value).map(|id| self.interned(id))
    }

    /// The shared value of `x`, or None for a valid but unassigned ID.
    ///
    /// # Panics
    ///
    /// Panics if `x` belongs to another context, even when its numeric ID
    /// happens to match.
    pub fn resolve(&self, x: Interned<D>) -> Option<Arc<D::Value>> {
        assert_eq!(
            x.context, self.table.sort.context,
            "interned value belongs to another context"
        );
        self.table.resolve(x.id)
    }

    /// An owned snapshot of assigned handles in ID order.
    ///
    /// Collecting these into a relation creates a snapshot too; later
    /// interning does not insert rows into that relation.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Interned<D>, Arc<D::Value>)> + use<D> {
        let sort = self.table.sort;
        self.table
            .iter()
            .map(move |(id, v)| (Interned::new(id, sort), v))
    }

    /// Interns a value using the same table as automatic literal conversion.
    pub fn intern(&self, value: impl Borrow<D::Value> + Into<Arc<D::Value>>) -> Interned<D> {
        self.interned(self.table.intern(value))
    }
}

impl<D: Domain> Clone for Dictionary<D> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            domain: PhantomData,
        }
    }
}

/// A set universe over the first `N` IDs of a namespace: one membership bit
/// per interned value, positioned by its ID.
///
/// Every position is valid even
/// before it has a name, so dictionary growth never changes existing sets or
/// predicates. No name list is declared. `N` must be nonzero and bounds
/// possible members, not just the size of one set. Scalar interning can
/// continue beyond `N`, but using an ID at or above `N` as a member fails.
/// BDD updates stay symbolic; sparse updates enumerate admitted set values.
pub struct Capacity<D, const N: u32>(PhantomData<fn() -> D>);

impl<D: Domain, const N: u32> Universe for Capacity<D, N> {
    type Member = Interned<D>;
    const LEN: u32 = N;
    fn index(context: u64, m: &Interned<D>) -> Option<u32> {
        assert_eq!(m.context, context, "set member belongs to another context");
        (m.id < u64::from(N)).then_some(m.id as u32)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/typed/interned.rs"]
mod tests;
