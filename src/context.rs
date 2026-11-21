//! Shared ownership of scalar backends, columns, and predicates.
//!
//! A [`Context`] binds each registered backend to scalar sorts as columns.
//! Each backend and sort pair has one column for the life of the context.
//! A [`ColumnId`] identifies that physical binding, while a [`Predicate`]
//! retains the context and its root in that column.

use std::fmt;
use std::sync::Arc;

use crate::backend::{Algebra, Backend, BoolOp, Function};
use crate::bdd::BddStore;
use crate::scalar::ScalarSort;
use crate::sparse::SparseStore;
use crate::util;
use crate::util::count::Count;

/// One physical backend binding to one scalar sort.
///
/// Equal IDs identify the same root namespace. Predicate operations require
/// all operands to belong to the same column.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ColumnId {
    context: u64,
    index: usize,
}

struct ContextData {
    id: u64,
    // Each entry records the backend index alongside its bound algebra.
    columns: Vec<(usize, Box<dyn Algebra>)>,
    backends: Vec<(Box<str>, Arc<dyn Backend>)>,
}

/// One owned root reference to a predicate over a column.
///
/// Cloning acquires a reference and dropping releases it. Both may enter the
/// backend store and follow the selector's store access rules in
/// [`Algebra::refine`].
#[must_use = "predicate operations return a new predicate"]
#[derive(Debug)]
pub struct Predicate {
    cx: Context,
    column: ColumnId,
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
    /// Keep this predicate live while borrowing the token. To keep the token
    /// after dropping the predicate, retain its context, acquire a reference
    /// with [`Algebra::inc_ref`], and release it with [`Algebra::dec_ref`] on
    /// the same column before releasing the context.
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
        self.cx.resolve(self.column).evaluate(self.function, value) != 0
    }

    /// The exact number of accepted scalar values, without enumerating them.
    pub fn count(&self) -> Count {
        self.cx.resolve(self.column).count(self.function)
    }

    #[inline(always)]
    fn check_column(&self, other: &Self) {
        assert!(
            self.column == other.column,
            "operands must belong to the same column"
        );
    }

    /// The Boolean combination of this predicate and `other`.
    ///
    /// # Panics
    ///
    /// Panics if `other` belongs to a different column.
    pub fn apply(&self, op: BoolOp, other: &Self) -> Self {
        self.check_column(other);
        let f = self
            .cx
            .resolve(self.column)
            .apply(op, self.function, other.function);
        self.cx.adopt(self.column, f)
    }

    /// The intersection with `other`.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::apply`] does.
    pub fn and(&self, other: &Self) -> Self {
        self.apply(BoolOp::And, other)
    }

    /// The union with `other`.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::apply`] does.
    pub fn or(&self, other: &Self) -> Self {
        self.apply(BoolOp::Or, other)
    }

    /// The difference from `other`.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::apply`] does.
    pub fn difference(&self, other: &Self) -> Self {
        self.apply(BoolOp::AndNot, other)
    }

    /// The complement within the column's scalar universe.
    pub fn not(&self) -> Self {
        let f = self
            .cx
            .resolve(self.column)
            .apply(BoolOp::AndNot, Function::TRUE, self.function);
        self.cx.adopt(self.column, f)
    }

    /// The predicate selecting `yes` where this predicate holds and `no`
    /// elsewhere.
    ///
    /// # Panics
    ///
    /// Panics if either operand belongs to a different column.
    pub fn ite(&self, yes: &Self, no: &Self) -> Self {
        self.check_column(yes);
        self.check_column(no);
        let f = self
            .cx
            .resolve(self.column)
            .ite(self.function, yes.function, no.function);
        self.cx.adopt(self.column, f)
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

/// A shared owner of backends and their bound columns.
///
/// Clones address the same storage and compare equal. Backend registration
/// and new column bindings require exclusive access to keep column resolution
/// a direct lookup during execution. Each live [`Predicate`] shares its
/// context. New bindings can be added whenever the context has a sole owner.
#[derive(Clone)]
pub struct Context {
    data: Arc<ContextData>,
}

impl Context {
    /// An empty context with a fresh identity.
    pub fn empty() -> Self {
        Self {
            data: Arc::new(ContextData {
                id: util::fresh_id(),
                columns: Vec::new(),
                backends: Vec::new(),
            }),
        }
    }

    fn data_mut(&mut self) -> &mut ContextData {
        Arc::get_mut(&mut self.data).expect("cannot change bindings on a shared context")
    }

    /// Registers `backend` under `name`.
    ///
    /// # Panics
    ///
    /// Panics if `name` is already registered or the context is shared.
    pub fn register_backend(&mut self, name: impl Into<Box<str>>, backend: impl Backend + 'static) {
        let name = name.into();
        let data = self.data_mut();
        assert!(
            data.backends.iter().all(|(n, _)| *n != name),
            "backend name {name:?} is already registered"
        );
        data.backends.push((name, Arc::new(backend)));
    }

    /// Binds the named backend to `sort`, or returns its existing column.
    ///
    /// # Panics
    ///
    /// Panics if the backend is unregistered, rejects `sort`, binds a
    /// different sort, or needs a new column while the context is shared.
    pub fn bind_column(&mut self, name: &str, sort: ScalarSort) -> ColumnId {
        let backend = self
            .data
            .backends
            .iter()
            .position(|(n, _)| &**n == name)
            .unwrap_or_else(|| panic!("backend {name:?} is not registered"));
        let index = self
            .data
            .columns
            .iter()
            .position(|(b, alg)| *b == backend && alg.sort() == sort)
            .unwrap_or_else(|| {
                let data = self.data_mut();
                let alg = Arc::clone(&data.backends[backend].1)
                    .bind(sort)
                    .expect("backend must support the column sort");
                assert!(alg.sort() == sort, "backend bound a different scalar sort");
                let index = data.columns.len();
                data.columns.push((backend, alg));
                index
            });
        ColumnId {
            context: self.data.id,
            index,
        }
    }

    /// The bound algebra behind `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context.
    #[inline(always)]
    pub fn resolve(&self, column: ColumnId) -> &dyn Algebra {
        assert!(
            column.context == self.data.id,
            "column belongs to another context"
        );
        &*self.data.columns[column.index].1
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
    /// Panics if `column` belongs to another context.
    pub fn f(&self, column: ColumnId) -> Predicate {
        self.resolve(column);
        self.adopt(column, Function::FALSE)
    }

    /// The full predicate over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context.
    pub fn t(&self, column: ColumnId) -> Predicate {
        self.resolve(column);
        self.adopt(column, Function::TRUE)
    }

    /// The predicate denoting exactly the sorted, distinct `values` packed as
    /// in [`Algebra::set`].
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context or [`Algebra::set`]
    /// rejects `values`.
    pub fn set(&self, column: ColumnId, values: &[u64]) -> Predicate {
        self.adopt(column, self.resolve(column).set(values))
    }

    /// The predicate for the unsigned cut `x < bound` over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context or [`Algebra::lt`]
    /// rejects the bound or sort.
    pub fn lt(&self, column: ColumnId, bound: &[u64]) -> Predicate {
        self.adopt(column, self.resolve(column).lt(bound))
    }

    /// The predicate `(x & mask) == value` over `column`.
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context or [`Algebra::masked_eq`]
    /// rejects the bit test.
    pub fn masked_eq(&self, column: ColumnId, mask: &[u64], value: &[u64]) -> Predicate {
        self.adopt(column, self.resolve(column).masked_eq(mask, value))
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.data.id)
            .finish_non_exhaustive()
    }
}

impl Default for Context {
    /// A context with default sparse and BDD stores registered as `sparse`
    /// and `bdd`.
    fn default() -> Self {
        let mut cx = Self::empty();
        cx.register_backend("sparse", SparseStore::default());
        cx.register_backend("bdd", BddStore::default());
        cx
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.data.id == other.data.id
    }
}

impl Eq for Context {}
