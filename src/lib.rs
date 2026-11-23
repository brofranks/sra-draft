//! Stored symbolic and concrete relations over bound rows and native indexes.
#![warn(missing_docs)]

pub mod util;

pub mod ast;
pub mod backend;
pub mod bdd;
mod cache;
pub mod context;
pub use context::index;
mod index_store;
pub mod rel;
pub mod row;
pub mod scalar;
pub mod sparse;
