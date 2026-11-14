//! Integer map operands lowered to the runtime's explicit bit-vector
//! operations.
//!
//! Signedness does not change bitwise operations; literals use the runtime's
//! checked packing.
//!
//! Maps stay restricted to integer endpoints:
//! ```compile_fail
//! use sra::typed::{map, IntoQuery, Rel};
//! use sra::Context;
//! let cx = Context::default();
//! let flags = Rel::<bool, bool>::new_set(&cx);
//! let wrong = (&flags).select(map::bitor(true));
//! ```

use super::{Integer, Operand};
use crate::backend::MapSpec;
use crate::context::Context;
use crate::map as raw;
use crate::query::{self, Operand as RawOperand};
use crate::row::{IntoRow, Row};

/// A raw map recipe restricted to integer endpoints by the typed frontend.
pub struct Map<F>(pub raw::Map<F>);

/// Bitwise OR with a literal, preserving the endpoint's integer
/// interpretation.
#[inline(always)]
pub fn bitor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(raw::bvor(value))
}

/// Bitwise AND with a literal, preserving the endpoint's integer
/// interpretation.
#[inline(always)]
pub fn bitand(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(raw::bvand(value))
}

/// Bitwise XOR with a literal, preserving the endpoint's integer
/// interpretation.
#[inline(always)]
pub fn bitxor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(raw::bvxor(value))
}

impl<D: Integer, F: FnOnce(&Context, &Row) -> MapSpec> Operand<D> for Map<F> {
    type R = D;
    type Node = raw::Transform;
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> query::Query<Self::Node> {
        RawOperand::resolve(self.0, cx, input)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/typed/map.rs"]
mod tests;
