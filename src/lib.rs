//! Statically composed queries over stored symbolic and concrete relations.
#![warn(missing_docs)]

pub mod util;

pub mod ast;
pub mod backend;
pub mod bdd;
mod cache;
pub mod context;
pub use context::index;
mod index_store;
pub mod map;
pub mod query;
pub mod rel;
pub mod row;
pub mod run;
pub mod scalar;
pub mod sparse;
