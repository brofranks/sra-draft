//! Storage owner for one family of backends, columns, and predicate handles.
//!
//! A [`Context`] registers backends by name, binds each to scalar sorts as
//! columns, and hands out [`Predicate`] handles whose clone and drop balance
//! the root references of their column. Backends and columns live for the
//! whole context, so a handle addresses its column by a plain [`ColumnId`]
//! that checked constructors and operations validate. The context interns one
//! column per backend and scalar sort, so compatible operands share its ID.
//! The context also owns the logical declarations of its sorts and records
//! and binds each scalar leaf of a declared sort to one column, so logical
//! identity, physical identity, and query occurrences remain distinct.

use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem::{self, ManuallyDrop};
use std::ops::Deref;
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::ast::{AstId, AstStore, FuncDecl, NodeKind, Sort, SortKind};
use crate::backend::{Backend, BoolOp, Column, Function};
use crate::bdd::BddStore;
use crate::index::{IndexData, dec_partition_refs};
use crate::scalar::{self, ScalarSort};
use crate::sparse::SparseTable;
use crate::util::fresh_id;

/// One registered backend of one context.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendId {
    /// The owning context's identity.
    pub context: u64,
    /// The backend's registration slot.
    pub slot: usize,
}

/// One physical backend binding to one scalar sort.
///
/// Equal IDs share one root namespace, and checked operations combine roots
/// of equal IDs only.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColumnId {
    /// The owning context's identity.
    pub context: u64,
    /// The column's declaration slot.
    pub slot: usize,
}

/// A field: the binding key of a scalar leaf, which is the accessor of a
/// record member or the sort of an anonymous scalar.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Field {
    /// The owning context's identity.
    pub context: u64,
    /// The accessor or sort declaration.
    pub id: AstId,
}

impl From<Sort> for Field {
    fn from(sort: Sort) -> Field {
        Field {
            context: sort.context,
            id: sort.id,
        }
    }
}

impl From<FuncDecl> for Field {
    fn from(decl: FuncDecl) -> Field {
        Field {
            context: decl.context,
            id: decl.id,
        }
    }
}

/// The two-way mapping behind an [`Interner`].
struct Entries<V: ?Sized> {
    ids: HashMap<Arc<V>, u64>,
    values: Vec<Arc<V>>,
}

/// Context-owned value dictionary for one symbol sort.
///
/// Symbol IDs are dense `u64` values in interning order. The context keeps the
/// values alive for its own lifetime, so resolution succeeds for every ID
/// previously issued by this dictionary.
pub struct Interner<V: ?Sized> {
    /// The symbol sort whose IDs this dictionary names.
    pub sort: Sort,
    data: Mutex<Entries<V>>,
}

impl<V: ?Sized + Eq + Hash> Interner<V> {
    /// Interns `value`, returning its stable ID.
    pub fn intern(&self, value: impl Borrow<V> + Into<Arc<V>>) -> u64 {
        let mut data = self.data.lock().unwrap();
        if let Some(&id) = data.ids.get(value.borrow()) {
            return id;
        }
        let id = u64::try_from(data.values.len()).expect("symbol ID overflow");
        let value = value.into();
        data.values.push(Arc::clone(&value));
        data.ids.insert(value, id);
        id
    }

    /// The ID of an interned value.
    pub fn get(&self, value: &V) -> Option<u64> {
        self.data.lock().unwrap().ids.get(value).copied()
    }

    /// The value behind an ID.
    pub fn resolve(&self, id: u64) -> Option<Arc<V>> {
        let data = self.data.lock().unwrap();
        usize::try_from(id)
            .ok()
            .and_then(|i| data.values.get(i))
            .cloned()
    }

    /// A snapshot of every interned value with its ID.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u64, Arc<V>)> + use<V> {
        let values = self.data.lock().unwrap().values.clone();
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| (i as u64, value))
    }
}

/// Shared state of a [`Context`].
pub struct ContextData {
    /// The context's identity, stamped on every id it issues.
    pub id: u64,
    /// The logical declarations of this context.
    pub ast: AstStore,
    /// The bound column of each field, keyed by its declaration.
    fields: HashMap<AstId, ColumnId>,
    /// The lazily allocated value dictionary of each symbol sort.
    interners: Mutex<HashMap<AstId, Arc<dyn Any + Send + Sync>>>,
    /// The index container of each exact column list, interned on first use.
    pub layouts: Mutex<Vec<Arc<IndexData>>>,
    // Declared first so columns drop before the backends they bind.
    columns: Vec<(usize, Box<dyn Column>)>,
    backends: Vec<(Box<str>, Arc<dyn Backend>)>,
}

impl ContextData {
    /// The backend registered under `name`.
    ///
    /// # Panics
    ///
    /// Panics if no backend is registered under `name`.
    pub fn backend(&self, name: &str) -> BackendId {
        let slot = self.backends.iter().position(|(n, _)| &**n == name);
        let slot = slot.unwrap_or_else(|| panic!("backend {name:?} is not registered"));
        BackendId {
            context: self.id,
            slot,
        }
    }

    /// The bound column behind `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context.
    #[inline(always)]
    pub fn resolve(&self, column: ColumnId) -> &dyn Column {
        assert!(
            column.context == self.id,
            "column belongs to another context"
        );
        &*self.columns[column.slot].1
    }
}

impl Drop for ContextData {
    /// Releases the partition references every index container owns before
    /// the columns they live in drop.
    fn drop(&mut self) {
        for data in mem::take(self.layouts.get_mut().unwrap()) {
            let mut data = Arc::try_unwrap(data)
                .ok()
                .expect("index data must not outlive its context");
            dec_partition_refs(&mut data, self);
        }
    }
}

/// One owned root reference to a predicate over a column.
///
/// Cloning acquires a reference and dropping releases it.
pub struct Predicate {
    /// The owning context.
    cx: Context,
    /// The column the root lives in.
    column: ColumnId,
    /// The owned root.
    function: Function,
}

impl Predicate {
    /// The context retaining the predicate's storage.
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// The physical column interpreting the root.
    #[inline(always)]
    pub fn column(&self) -> ColumnId {
        self.column
    }

    /// The borrowed root token, without acquiring a reference.
    ///
    /// The caller keeps this predicate live while borrowing the token, or
    /// acquires a separate reference and retains the context for its use.
    #[inline(always)]
    pub fn function(&self) -> Function {
        self.function
    }

    /// Whether `value` satisfies the predicate.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not encode a value of the column's scalar sort.
    pub fn contains(&self, value: &[u64]) -> bool {
        let c = self.cx.resolve(self.column);
        assert!(c.sort().accepts(value), "value is outside the column sort");
        c.evaluate(self.function, value) != 0
    }

    /// Transfers the owned root reference with its context and physical column.
    ///
    /// The caller retains the returned context while using the token and
    /// releases the reference through that context and column.
    pub fn into_parts(self) -> (Context, ColumnId, Function) {
        let p = ManuallyDrop::new(self);
        // SAFETY: Suppressing Predicate's drop transfers its existing root reference.
        // The context moves exactly once, and the remaining fields are Copy.
        let cx = unsafe { ptr::read(&p.cx) };
        (cx, p.column, p.function)
    }
}

impl Clone for Predicate {
    fn clone(&self) -> Self {
        self.cx.resolve(self.column).inc_ref(self.function);
        self.cx.adopt(self.column, self.function)
    }
}

impl Drop for Predicate {
    fn drop(&mut self) {
        self.cx.resolve(self.column).dec_ref(self.function);
    }
}

/// Handle to one shared [`ContextData`].
///
/// Clones address the same storage and compare equal.
#[derive(Clone)]
pub struct Context {
    /// The shared storage.
    pub data: Arc<ContextData>,
}

/// Checks that `values` are sorted, distinct, and encoded in `sort`.
pub fn check_packed(sort: ScalarSort, values: &[u64]) {
    let words = sort.words();
    assert!(
        values.len().is_multiple_of(words),
        "values are not packed at the sort's word stride"
    );
    assert!(
        values.chunks_exact(words).all(|v| sort.accepts(v)),
        "value is outside the column sort"
    );
    assert!(
        values
            .chunks_exact(words)
            .is_sorted_by(|a, b| scalar::compare(a, b).is_lt()),
        "values are not sorted and distinct"
    );
}

impl Context {
    /// Creates an empty context with a fresh identity.
    pub fn new() -> Self {
        Self {
            data: Arc::new(ContextData {
                id: fresh_id(),
                ast: AstStore::default(),
                fields: HashMap::new(),
                interners: Mutex::default(),
                layouts: Mutex::default(),
                columns: Vec::new(),
                backends: Vec::new(),
            }),
        }
    }

    fn data_mut(&mut self) -> &mut ContextData {
        Arc::get_mut(&mut self.data)
            .expect("cannot declare backends or columns on a shared context")
    }

    /// Registers `backend` under `name`.
    ///
    /// # Panics
    ///
    /// Panics if `name` is already registered or the context is shared.
    pub fn register_backend(
        &mut self,
        name: impl Into<Box<str>>,
        backend: impl Backend + 'static,
    ) -> BackendId {
        let name = name.into();
        let data = self.data_mut();
        assert!(
            data.backends.iter().all(|(n, _)| *n != name),
            "backend name {name:?} is already registered"
        );
        data.backends.push((name, Arc::new(backend)));
        BackendId {
            context: data.id,
            slot: data.backends.len() - 1,
        }
    }

    /// Declares the column of `backend` over `sort`, returning the existing
    /// column when the pair was declared before.
    ///
    /// # Panics
    ///
    /// Panics if `backend` is not registered in this context, the backend does
    /// not support `sort`, or the context is shared when the column is new.
    pub fn column(&mut self, backend: BackendId, sort: ScalarSort) -> ColumnId {
        assert!(
            backend.context == self.id,
            "backend belongs to another context"
        );
        let id = self.id;
        let existing = self
            .columns
            .iter()
            .position(|(b, c)| *b == backend.slot && c.sort() == sort);
        let slot = existing.unwrap_or_else(|| {
            let data = self.data_mut();
            let column = Arc::clone(&data.backends[backend.slot].1).column(sort);
            data.columns.push((
                backend.slot,
                column.expect("backend must support the column sort"),
            ));
            data.columns.len() - 1
        });
        ColumnId { context: id, slot }
    }

    /// Binds the declared scalar member behind `accessor` to a column of
    /// `backend`.
    ///
    /// # Panics
    ///
    /// Panics if `accessor` belongs to another context or does not project a
    /// scalar member of a record, `backend` belongs to another context or does
    /// not support the member's sort, the member is already bound, or the
    /// context is shared.
    pub fn set_representation(&mut self, accessor: FuncDecl, backend: BackendId) -> ColumnId {
        let info = accessor.resolve(self);
        assert!(
            info.domain.kind == SortKind::Tuple && info.range.kind.is_scalar(),
            "representation requires a declared scalar member"
        );
        assert!(
            !self.fields.contains_key(&accessor.id),
            "field representation is already bound"
        );
        let column = self.column(backend, info.range.scalar());
        self.data_mut().fields.insert(accessor.id, column);
        column
    }

    /// Binds `field`, returning its existing column when it is bound.
    ///
    /// A record member binds to the backend selected by
    /// [`Context::set_representation`] or else to `sparse`. An anonymous
    /// scalar binds to `sparse`.
    ///
    /// # Panics
    ///
    /// Panics if `field` belongs to another context or is neither a scalar sort
    /// nor a scalar record accessor, no `sparse` backend is registered when the
    /// default is needed, or a new binding is needed while the context is
    /// shared.
    pub fn bind_field(&mut self, field: impl Into<Field>) -> ColumnId {
        let field = field.into();
        assert!(field.context == self.id, "field belongs to another context");
        if let Some(&column) = self.fields.get(&field.id) {
            return column;
        }
        let scalar = {
            let nodes = self.ast.shared.read(());
            let node = nodes.node(field.id);
            let kind = match node.kind {
                NodeKind::Accessor(_)
                    if nodes.node(node.edges[0]).kind == NodeKind::Sort(SortKind::Tuple) =>
                {
                    nodes.node(node.edges[1]).kind
                }
                kind => kind,
            };
            let NodeKind::Sort(SortKind::Scalar(scalar)) = kind else {
                panic!("field binding requires a scalar sort or record accessor");
            };
            scalar
        };
        let column = self.column(self.backend("sparse"), scalar);
        self.data_mut().fields.insert(field.id, column);
        column
    }

    /// The column bound to `field`.
    ///
    /// # Panics
    ///
    /// Panics if `field` belongs to another context or is not bound.
    pub fn field(&self, field: impl Into<Field>) -> ColumnId {
        let field = field.into();
        assert!(field.context == self.id, "field belongs to another context");
        *self.fields.get(&field.id).expect("field must be bound")
    }

    /// The value dictionary of the symbol sort `sort`, allocated on first use
    /// with the value type `V`.
    ///
    /// # Panics
    ///
    /// Panics if `sort` belongs to another context, is not a symbol sort, or
    /// already interns another value type.
    pub fn interner<V: ?Sized + Eq + Hash + Send + Sync + 'static>(
        &self,
        sort: Sort,
    ) -> Arc<Interner<V>> {
        assert!(
            sort.context == self.id,
            "symbol sort belongs to another context"
        );
        assert!(
            sort.kind == SortKind::Scalar(ScalarSort::symbol()),
            "interned values require a symbol sort"
        );
        // The registry lock is released before the downcast, so a type
        // mismatch panics without poisoning the registry.
        let entry = Arc::clone(
            self.interners
                .lock()
                .unwrap()
                .entry(sort.id)
                .or_insert_with(|| {
                    Arc::new(Interner::<V> {
                        sort,
                        data: Mutex::new(Entries {
                            ids: HashMap::new(),
                            values: Vec::new(),
                        }),
                    })
                }),
        );
        entry
            .downcast()
            .expect("symbol sort must intern one value type")
    }

    /// The predicate adopting the owned reference to `function` in `column`.
    fn adopt(&self, column: ColumnId, function: Function) -> Predicate {
        Predicate {
            cx: self.clone(),
            column,
            function,
        }
    }

    /// The empty predicate over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context.
    pub fn f(&self, column: ColumnId) -> Predicate {
        self.resolve(column);
        self.adopt(column, Function::FALSE)
    }

    /// The full predicate over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context.
    pub fn t(&self, column: ColumnId) -> Predicate {
        self.resolve(column);
        self.adopt(column, Function::TRUE)
    }

    /// The predicate denoting exactly `values`, packed at the column sort's word
    /// stride.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context or `values` are
    /// unsorted, repeated, or outside the sort.
    pub fn set(&self, column: ColumnId, values: &[u64]) -> Predicate {
        let c = self.resolve(column);
        check_packed(c.sort(), values);
        self.adopt(column, c.set(values))
    }

    /// The predicate `x < bound` over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context, `bound` is outside the
    /// sort, or the sort has no ordering.
    pub fn less_than(&self, column: ColumnId, bound: &[u64]) -> Predicate {
        let c = self.resolve(column);
        assert!(c.sort().accepts(bound), "bound is outside the column sort");
        self.adopt(column, c.less_than(bound))
    }

    /// The predicate `(x & mask) == value` over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` is not declared in this context, `mask` or `value` is
    /// outside the sort, `value` has bits outside `mask`, the sort is a symbol
    /// sort, or the backend has no bit predicates.
    pub fn masked_eq(&self, column: ColumnId, mask: &[u64], value: &[u64]) -> Predicate {
        let c = self.resolve(column);
        assert!(c.sort().accepts(mask), "mask is outside the column sort");
        assert!(c.sort().accepts(value), "value is outside the column sort");
        assert!(
            mask.iter().zip(value).all(|(m, v)| v & !m == 0),
            "value has bits outside its mask"
        );
        self.adopt(column, c.masked_eq(mask, value))
    }

    /// The Boolean combination of two predicates over one column.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different columns or contexts.
    pub fn apply_bool(&self, op: BoolOp, left: &Predicate, right: &Predicate) -> Predicate {
        assert!(
            left.cx == *self && right.cx == *self,
            "operands belong to another context"
        );
        let column = left.column;
        assert!(right.column == column, "operands belong to another column");
        let function = self
            .resolve(column)
            .apply_bool(op, left.function, right.function);
        self.adopt(column, function)
    }

    /// The predicate choosing `yes` where `condition` holds and `no` elsewhere.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different columns or contexts.
    pub fn ite(&self, condition: &Predicate, yes: &Predicate, no: &Predicate) -> Predicate {
        assert!(
            condition.cx == *self && yes.cx == *self && no.cx == *self,
            "operands belong to another context"
        );
        let column = condition.column;
        assert!(
            yes.column == column && no.column == column,
            "operands belong to another column"
        );
        let function = self
            .resolve(column)
            .ite(condition.function, yes.function, no.function);
        self.adopt(column, function)
    }
}

impl Default for Context {
    /// Creates a context with a default [`SparseTable`] registered as `sparse`
    /// and a default [`BddStore`] registered as `bdd`.
    fn default() -> Self {
        let mut cx = Self::new();
        cx.register_backend("sparse", SparseTable::default());
        cx.register_backend("bdd", BddStore::default());
        cx
    }
}

impl Deref for Context {
    type Target = ContextData;

    #[inline(always)]
    fn deref(&self) -> &ContextData {
        &self.data
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Context {}

#[cfg(test)]
#[path = "../tests/unit/context.rs"]
mod tests;
