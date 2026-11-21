//! Storage and exact scalar algebra for symbolic and concrete relations.
#![warn(missing_docs)]

pub mod util;

pub mod backend;
pub mod bdd;
mod cache;
pub mod context;
pub mod scalar;
pub mod sparse;
