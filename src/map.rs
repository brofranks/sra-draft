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
use crate::run::{Exec, Flow, Mode, Run};

/// Packs one literal against the bit-vector input row.
fn pack(cx: &Context, value: impl IntoRow, input: &Row) -> Box<[u64]> {
    assert!(
        input.sort().scalar().bv_width().is_some(),
        "operation requires a bit-vector sort"
    );
    let mut out = Vec::new();
    row::append(cx, value, input, &mut out);
    out.into()
}

/// A map operand, lowered against the input row it transforms.
pub struct Map<F>(pub F);

/// The output equals the input, in the input's column.
pub fn copy() -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(|_: &Context, input: &Row| MapSpec::identity(input.sort().scalar().words()))
}

/// `y = x | value` on a bit-vector input.
///
/// # Panics
///
/// The consuming combinator panics when it resolves this operand if the input
/// is not a bit-vector scalar or `value` does not pack into its width.
#[inline(always)]
pub fn bvor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        let value = pack(cx, value, input);
        MapSpec::Bits {
            clear: value.clone(),
            flip: value,
        }
    })
}

/// `y = x & value` on a bit-vector input.
///
/// # Panics
///
/// The consuming combinator panics when it resolves this operand if the input
/// is not a bit-vector scalar or `value` does not pack into its width.
#[inline(always)]
pub fn bvand(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        let keep = pack(cx, value, input);
        let mut clear: Box<[u64]> = keep.iter().map(|k| !k).collect();
        *clear.last_mut().unwrap() &= input.sort().scalar().top_mask();
        MapSpec::Bits {
            clear,
            flip: vec![0; keep.len()].into(),
        }
    })
}

/// `y = x ^ value` on a bit-vector input.
///
/// # Panics
///
/// The consuming combinator panics when it resolves this operand if the input
/// is not a bit-vector scalar or `value` does not pack into its width.
#[inline(always)]
pub fn bvxor(value: impl IntoRow) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
        let flip = pack(cx, value, input);
        MapSpec::Bits {
            clear: vec![0; flip.len()].into(),
            flip,
        }
    })
}

/// `y = (x & !mask) | value` for a `value` within `mask`.
///
/// # Panics
///
/// The consuming combinator panics when it resolves this operand if the input
/// is not a bit-vector scalar, `mask` or `value` does not pack into its width,
/// or `value` has bits outside `mask`.
pub fn assign(
    mask: impl IntoRow,
    value: impl IntoRow,
) -> Map<impl FnOnce(&Context, &Row) -> MapSpec> {
    Map(move |cx: &Context, input: &Row| {
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
        query::transform(cx, input, Arc::new((self.0)(cx, input)))
    }
}

/// A map between two boundary cells: the input at position 0 and the output
/// at position 1.
#[derive(Clone)]
pub struct Transform {
    /// The map from the input cell to the output cell.
    pub map: Arc<MapSpec>,
}

impl Exec for Transform {
    type State<M: Mode> = ();

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
        eval.equate(
            eval.maps[map.start],
            eval.maps[map.start + 1],
            &self.map,
            mode,
            then,
        )
    }
}
