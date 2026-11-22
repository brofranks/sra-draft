//! Storage, logical declarations, bound rows, scalar algebra, and native
//! indexes for symbolic and concrete relations.
#![warn(missing_docs)]

pub mod util;

pub mod ast;
pub mod backend;
pub mod bdd;
mod cache;
pub mod context;
pub use context::index;
mod index_store;
pub mod row;
pub mod scalar;
pub mod sparse;
