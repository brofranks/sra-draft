//! Maps, copies, and equalities as invocation-local equations between two
//! cells.
//!
//! A map operand reads its scalar input cell and exposes a fresh output cell
//! of the same sort. While its continuation runs, the equation between the
//! two transfers every narrowing of one cell to the other using native
//! image/preimage operations. At the terminal, observed cells connected
//! through equations are decomposed into exact branches, including
//! correlations through hidden witnesses. A component with at most one
//! observed cell stays symbolic.

use std::ops::Range;
use std::sync::Arc;

use crate::backend::MapSpec;
use crate::context::Context;
use crate::query::{self, Operand, Query};
use crate::row::{self, IntoRow, Row};
use crate::run::{Equation, Exec, Flow, Mode, Run};

/// Packs one literal against the scalar input row.
fn pack(cx: &Context, value: impl IntoRow, input: &Row) -> Box<[u64]> {
    let mut out = Vec::new();
    row::append(cx, value, input, &mut out);
    out.into()
}

/// A map operand, lowered against the input row it transforms.
pub struct Map<F>(pub F);

/// The output equals the input, in the input's column.
pub fn copy() -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(|_: &Context, input: &Row| MapSpec::identity(input.sort.scalar().words()))
}

/// `y = x | value` on a bit-vector input.
#[inline(always)]
pub fn bvor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        row::bitvector_column(input);
        let value = pack(cx, value, input);
        MapSpec::Bits {
            clear: value.clone(),
            flip: value,
        }
    })
}

/// `y = x & value` on a bit-vector input.
#[inline(always)]
pub fn bvand(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        row::bitvector_column(input);
        let keep = pack(cx, value, input);
        let mut clear: Box<[u64]> = keep.iter().map(|k| !k).collect();
        *clear.last_mut().unwrap() &= input.sort.scalar().top_mask();
        MapSpec::Bits {
            clear,
            flip: vec![0; keep.len()].into(),
        }
    })
}

/// `y = x ^ value` on a bit-vector input.
#[inline(always)]
pub fn bvxor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        row::bitvector_column(input);
        let flip = pack(cx, value, input);
        MapSpec::Bits {
            clear: vec![0; flip.len()].into(),
            flip,
        }
    })
}

/// `y = (x & !mask) | value` for a `value` within `mask`.
pub fn assign(
    mask: impl IntoRow,
    value: impl IntoRow,
) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        row::bitvector_column(input);
        let (clear, flip) = (pack(cx, mask, input), pack(cx, value, input));
        assert!(
            clear.iter().zip(flip.iter()).all(|(m, v)| v & !m == 0),
            "assigned value has bits outside its mask"
        );
        MapSpec::Bits { clear, flip }
    })
}

/// A partial, pure, deterministic Rust function that writes every output
/// word and returns whether the input has an image, applied by enumerating
/// admitted finite inputs; a symbolic column rejects it.
pub fn func(
    f: impl Fn(&[u64], &mut [u64]) -> bool + Send + Sync + 'static,
) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |_: &Context, _: &Row| MapSpec::Rust(Arc::new(f)))
}

impl<F: FnOnce(&Context, &Row) -> MapSpec> Operand for Map<F> {
    type Node = Transform;
    fn resolve(self, cx: &Context, input: &Row) -> Query<Transform> {
        query::transform(cx, input, Some(Arc::new((self.0)(cx, input))))
    }
}

/// A map between two boundary cells: the input at position 0 and the output
/// at position 1.
#[derive(Clone)]
pub struct Transform {
    /// The map, or `None` for structural pass-through.
    // None is structural pass-through, not an explicitly authored identity
    // map. Some(identity) keeps a fresh endpoint and the full relation
    // universe.
    pub map: Option<Arc<MapSpec>>,
}

impl Exec for Transform {
    type State<M: Mode> = ();

    #[inline(always)]
    fn is_projection(&self) -> bool {
        self.map.is_none()
    }

    /// Registers the equation, tightens both cells against each other once, and
    /// runs the continuation.
    ///
    /// The terminal decomposes correlations among observed cells.
    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        _state: &mut (),
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        let Some(spec) = &self.map else {
            return then(eval, mode);
        };
        let (x, y) = (eval.maps[map.start], eval.maps[map.start + 1]);
        assert!(
            !eval.connected(x, y),
            "equations between cells must not form a cycle"
        );
        eval.push_read(x);
        eval.push_read(y);
        eval.equations.push(Equation {
            x,
            y,
            map: spec.clone(),
        });
        let p = eval.transfer(spec, x, y, false);
        let result = eval.narrow(x, p, mode, |eval, mode| then(eval, mode));
        eval.equations.pop();
        eval.pop_reads(2);
        result
    }
}

#[cfg(test)]
#[path = "../tests/unit/map.rs"]
mod tests;
