//! Storage and exact scalar algebra for symbolic and concrete relations.
//!
//! The crate is layered: [`util`] holds the shared hash-consed DAG, [`backend`]
//! the column abstraction that [`bdd`] and [`sparse`] implement, [`ast`] and
//! [`context`] the sorts, declarations, and store registry, [`row`],
//! [`partition`], and [`index`] the stored shape of a relation, and [`rel`],
//! [`query`], and [`run`] the relations, combinators, and their execution,
//! [`pred`] the field predicates that restrict them, [`iteration`] the
//! semi-naive fixed points over them, and [`typed`] the Rust endpoint
//! interpretations over all of it.
#![warn(missing_docs)]

#[macro_use]
pub mod util;

pub mod ast;
pub mod backend;
pub mod bdd;
pub mod context;
pub mod index;
pub mod iteration;
pub mod map;
pub mod partition;
pub mod pred;
pub mod query;
pub mod rel;
pub mod row;
pub mod run;
pub mod scalar;
pub mod sparse;
pub mod typed;

pub use ast::{FuncDecl, FuncInfo, Sort, SortKind, TupleDecl};
pub use backend::{Backend, BoolOp, Column, Function, MapSpec, Members};
pub use bdd::BddStore;
pub use context::{BackendId, ColumnId, Context, Predicate};
pub use iteration::Variable;
pub use pred::{Build, Filter, Membership};
pub use query::{IntoQuery, Operand, Query};
pub use rel::Rel;
pub use row::{IntoRow, Row};
pub use run::Binding;
pub use scalar::ScalarSort;
pub use sparse::SparseTable;
