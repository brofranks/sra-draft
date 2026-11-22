//! Shared ownership of logical sorts, backends, algebras, and predicates.
//!
//! A [`Context`] interns one bound algebra per registered backend and scalar
//! sort and identifies it by an [`AlgebraId`]. A [`Predicate`] retains the
//! context and its root in that algebra. A field binding records the algebra
//! selected for a scalar record member.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::ast::{FuncDecl, Sorts};
use crate::backend::{Algebra, Backend, BoolOp, Function};
use crate::bdd::BddStore;
use crate::index_store::Store;
use crate::scalar::ScalarSort;
use crate::sparse::SparseStore;
use crate::util::count::Count;

pub mod index;

/// The ID of a backend binding to one scalar sort within a context.
///
/// Equal IDs identify the same root namespace. Predicate operations require
/// all operands to belong to the same algebra.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AlgebraId {
    context: u64,
    index: usize,
}

struct ContextData {
    sorts: Sorts,
    fields: HashMap<FuncDecl, AlgebraId>,
    // Indexes release their partitions before algebras and backends drop.
    indexes: Mutex<Vec<Arc<Store>>>,
    // Each entry records the backend index alongside its bound algebra.
    // Fields drop in declaration order, so the registry retains its
    // backends while the algebras drop.
    algebras: Vec<(usize, Arc<dyn Algebra>)>,
    backends: Vec<(Box<str>, Arc<dyn Backend>)>,
}

/// One owned root reference to a predicate in a bound algebra.
///
/// Cloning acquires a reference and dropping releases it. Both may enter the
/// backend store, so within a selector they are subject to the store access
/// rules of [`Algebra::refine`].
#[must_use = "predicate operations return a new predicate"]
#[derive(Debug)]
pub struct Predicate {
    cx: Context,
    algebra: AlgebraId,
    function: Function,
}

impl Predicate {
    /// The context retaining the predicate's storage.
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.cx
    }

    /// The ID of the algebra interpreting this root.
    #[inline(always)]
    pub fn algebra(&self) -> AlgebraId {
        self.algebra
    }

    /// The borrowed root token, without acquiring a reference.
    ///
    /// Keep this predicate live while borrowing the token. To keep the token
    /// after dropping the predicate, retain its context, acquire a reference
    /// with [`Algebra::inc_ref`], and release it with [`Algebra::dec_ref`] on
    /// the same algebra before releasing the context.
    #[inline(always)]
    pub fn function(&self) -> Function {
        self.function
    }

    /// Whether `value` satisfies the predicate.
    ///
    /// # Panics
    ///
    /// Panics if `value` does not encode a value of the algebra's scalar sort.
    pub fn contains(&self, value: &[u64]) -> bool {
        self.cx.resolve(self.algebra).evaluate(self.function, value) != 0
    }

    /// The exact number of accepted scalar values, without enumerating them.
    pub fn count(&self) -> Count {
        self.cx.resolve(self.algebra).count(self.function)
    }

    #[inline(always)]
    fn check_algebra(&self, other: &Self) {
        assert!(
            self.algebra == other.algebra,
            "operands must belong to the same algebra"
        );
    }

    /// The Boolean combination of this predicate and `other`.
    ///
    /// # Panics
    ///
    /// Panics if `other` belongs to a different algebra.
    pub fn apply(&self, op: BoolOp, other: &Self) -> Self {
        self.check_algebra(other);
        let f = self
            .cx
            .resolve(self.algebra)
            .apply(op, self.function, other.function);
        self.cx.adopt(self.algebra, f)
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

    /// The complement within the algebra's scalar universe.
    pub fn not(&self) -> Self {
        let f = self
            .cx
            .resolve(self.algebra)
            .apply(BoolOp::AndNot, Function::TRUE, self.function);
        self.cx.adopt(self.algebra, f)
    }

    /// The predicate selecting `yes` where this predicate holds and `no`
    /// elsewhere.
    ///
    /// # Panics
    ///
    /// Panics if either operand belongs to a different algebra.
    pub fn ite(&self, yes: &Self, no: &Self) -> Self {
        self.check_algebra(yes);
        self.check_algebra(no);
        let f = self
            .cx
            .resolve(self.algebra)
            .ite(self.function, yes.function, no.function);
        self.cx.adopt(self.algebra, f)
    }
}

impl Clone for Predicate {
    fn clone(&self) -> Self {
        self.cx.resolve(self.algebra).inc_ref(self.function);
        self.cx.adopt(self.algebra, self.function)
    }
}

impl Drop for Predicate {
    fn drop(&mut self) {
        self.cx.resolve(self.algebra).dec_ref(self.function);
    }
}

/// A shared owner of logical sorts, backends, bound algebras, field bindings,
/// and native indexes.
///
/// Clones address the same storage and compare equal. Backend registration
/// and new bindings require exclusive access so algebra resolution stays a
/// direct lookup during execution. Each live [`Predicate`], [`index::Index`],
/// and [`index::IndexRoot`] shares its context.
/// New bindings can be added whenever the context has a sole owner.
///
/// Dropping the final owner can enter scalar stores to release partitions
/// held by the index registry. Drop the final owner outside scalar-store
/// guards.
#[derive(Clone)]
pub struct Context {
    data: Arc<ContextData>,
}

impl Context {
    /// An empty context with a fresh identity.
    pub fn empty() -> Self {
        Self {
            data: Arc::new(ContextData {
                sorts: Sorts::default(),
                fields: HashMap::new(),
                indexes: Mutex::default(),
                algebras: Vec::new(),
                backends: Vec::new(),
            }),
        }
    }

    /// The logical declaration store shared by this context and its clones.
    pub fn sorts(&self) -> &Sorts {
        &self.data.sorts
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

    fn backend(&self, name: &str) -> usize {
        self.data
            .backends
            .iter()
            .position(|(n, _)| &**n == name)
            .unwrap_or_else(|| panic!("backend {name:?} is not registered"))
    }

    fn bound(&self, backend: usize, sort: ScalarSort) -> Option<AlgebraId> {
        self.data
            .algebras
            .iter()
            .position(|(b, alg)| *b == backend && alg.sort() == sort)
            .map(|index| AlgebraId {
                context: self.sorts().id(),
                index,
            })
    }

    /// The existing algebra of the named backend over `sort`, if bound.
    ///
    /// # Panics
    ///
    /// Panics if the backend is unregistered.
    pub fn algebra(&self, name: &str, sort: ScalarSort) -> Option<AlgebraId> {
        self.bound(self.backend(name), sort)
    }

    /// Binds the named backend to `sort` or returns its existing algebra ID.
    ///
    /// # Panics
    ///
    /// Panics if the backend is unregistered, rejects `sort`, binds a
    /// different sort, or needs a new binding while the context is shared.
    pub fn bind(&mut self, name: &str, sort: ScalarSort) -> AlgebraId {
        let backend = self.backend(name);
        if let Some(id) = self.bound(backend, sort) {
            return id;
        }
        let data = self.data_mut();
        let alg = Arc::clone(&data.backends[backend].1)
            .bind(sort)
            .expect("backend must support the requested sort");
        assert!(alg.sort() == sort, "backend bound a different scalar sort");
        let index = data.algebras.len();
        data.algebras.push((backend, Arc::from(alg)));
        AlgebraId {
            context: data.sorts.id(),
            index,
        }
    }

    fn field_sort(&self, accessor: FuncDecl) -> ScalarSort {
        accessor
            .resolve(self.sorts())
            .field_sort()
            .expect("field must project a scalar record member")
    }

    /// The algebra bound to a scalar record member, if configured.
    ///
    /// # Panics
    ///
    /// Panics if `accessor` belongs to another context or does not project a
    /// scalar record member.
    pub fn field(&self, accessor: FuncDecl) -> Option<AlgebraId> {
        self.field_sort(accessor);
        self.data.fields.get(&accessor).copied()
    }

    /// Binds a scalar record member to the named backend.
    ///
    /// Repeating the same selection returns the original algebra, including
    /// while the context is shared.
    ///
    /// # Panics
    ///
    /// Panics if the accessor is invalid as in [`Self::field`], the field is
    /// bound to another backend, the context is shared when binding a new
    /// field, or [`Self::bind`] rejects the backend or sort.
    pub fn bind_field(&mut self, accessor: FuncDecl, name: &str) -> AlgebraId {
        let sort = self.field_sort(accessor);
        if let Some(&id) = self.data.fields.get(&accessor) {
            let backend = self.data.algebras[id.index].0;
            assert!(
                &*self.data.backends[backend].0 == name,
                "field is already bound to a different backend"
            );
            return id;
        }
        let id = self.bind(name, sort);
        self.data_mut().fields.insert(accessor, id);
        id
    }

    /// The bound algebra identified by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` belongs to another context.
    #[inline(always)]
    pub fn resolve(&self, id: AlgebraId) -> &dyn Algebra {
        assert!(
            id.context == self.data.sorts.id(),
            "algebra belongs to another context"
        );
        &*self.data.algebras[id.index].1
    }

    /// The predicate adopting the owned reference to `function` in `algebra`.
    fn adopt(&self, algebra: AlgebraId, function: Function) -> Predicate {
        Predicate {
            cx: self.clone(),
            algebra,
            function,
        }
    }

    /// The empty predicate over `algebra`.
    ///
    /// # Panics
    ///
    /// Panics if `algebra` belongs to another context.
    pub fn f(&self, algebra: AlgebraId) -> Predicate {
        self.resolve(algebra);
        self.adopt(algebra, Function::FALSE)
    }

    /// The full predicate over `algebra`.
    ///
    /// # Panics
    ///
    /// Panics if `algebra` belongs to another context.
    pub fn t(&self, algebra: AlgebraId) -> Predicate {
        self.resolve(algebra);
        self.adopt(algebra, Function::TRUE)
    }

    /// The predicate denoting exactly the sorted, distinct `values` packed as
    /// in [`Algebra::set`].
    ///
    /// # Panics
    ///
    /// Panics if `algebra` belongs to another context or [`Algebra::set`]
    /// rejects `values`.
    pub fn set(&self, algebra: AlgebraId, values: &[u64]) -> Predicate {
        self.adopt(algebra, self.resolve(algebra).set(values))
    }

    /// The predicate for the unsigned cut `x < bound` over `algebra`.
    ///
    /// # Panics
    ///
    /// Panics if `algebra` belongs to another context or [`Algebra::lt`]
    /// rejects the bound or sort.
    pub fn lt(&self, algebra: AlgebraId, bound: &[u64]) -> Predicate {
        self.adopt(algebra, self.resolve(algebra).lt(bound))
    }

    /// The predicate `(x & mask) == value` over `algebra`.
    ///
    /// # Panics
    ///
    /// Panics if `algebra` belongs to another context or [`Algebra::masked_eq`]
    /// rejects the bit test.
    pub fn masked_eq(&self, algebra: AlgebraId, mask: &[u64], value: &[u64]) -> Predicate {
        self.adopt(algebra, self.resolve(algebra).masked_eq(mask, value))
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.data.sorts.id())
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
        Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Eq for Context {}
