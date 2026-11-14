//! Checks for the column-scoped partition wrapper: slot validation, the
//! alignment and quotient laws resolved before any backend call, case and
//! packed-value construction, refinement delivery, and the boundary checks
//! that catch a misbehaving backend.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use num_bigint::BigUint;

use super::Partition;
use crate::backend::{
    Alignment, Backend, BoolOp, Column, Function, MapSpec, Members, Refinement, Values,
};
use crate::context::{ColumnId, Context};
use crate::scalar::ScalarSort;

/// A `w`-bit column on the named registered backend.
fn col(cx: &mut Context, backend: &str, w: u32) -> ColumnId {
    let backend = cx.backend(backend);
    cx.column(backend, ScalarSort::bit_vec(w))
}

/// The slot of each value of a 4-bit column under a partition.
fn slots(cx: &Context, f: Partition) -> Vec<u32> {
    (0..16)
        .map(|v| cx.resolve(f.column).evaluate(f.function, &[v]))
        .collect()
}

/// The three-cell partition `{1, 3} -> 1`, `{5} -> 2`, everything else `0`,
/// with its slot labels.
fn cases(cx: &Context, c: ColumnId) -> (Partition, Vec<u32>) {
    let (a, b) = (cx.set(c, &[1, 3]), cx.set(c, &[5]));
    let mut labels = Vec::new();
    let f = cx.cases(c, 0, &[(a.function(), 1), (b.function(), 2)], &mut labels);
    (f, labels)
}

/// A backend that breaks one contract of the checked interface, so the
/// boundary checks that catch a misbehaving implementation have a caller.
enum Rogue {
    /// `align` labels a cell with a source slot its operand does not have.
    Labels,
    /// `refine` visits a slot outside the partition.
    Slot,
    /// `refine` breaks without delivering the cell that stopped it.
    Break,
}

/// The column of a rogue backend; its roots are raw tokens, never built by
/// it.
struct RogueColumn {
    rogue: Arc<Rogue>,
    sort: ScalarSort,
}

impl Backend for Rogue {
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
        Some(Box::new(RogueColumn { rogue: self, sort }))
    }
}

impl Column for RogueColumn {
    fn sort(&self) -> ScalarSort {
        self.sort
    }
    fn inc_ref(&self, _f: Function) {}
    fn dec_ref(&self, _f: Function) {}

    fn align(
        &self,
        _left: Function,
        _right: Function,
        labels: &mut Vec<(u32, u32)>,
        _scratch: &mut Alignment,
    ) -> Function {
        labels.push((9, 0));
        Function::new(4)
    }

    fn refine(
        &self,
        _f: Function,
        _p: Function,
        _scratch: &mut Refinement,
        select: &mut dyn FnMut(u32) -> bool,
        _then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        if let Rogue::Slot = *self.rogue {
            select(9);
        }
        ControlFlow::Break(())
    }

    fn evaluate(&self, _f: Function, _value: &[u64]) -> u32 {
        unreachable!()
    }
    fn count(&self, _p: Function) -> BigUint {
        unreachable!()
    }
    fn set(&self, _values: &[u64]) -> Function {
        unreachable!()
    }
    fn less_than(&self, _bound: &[u64]) -> Function {
        unreachable!()
    }
    fn apply_bool(&self, _op: BoolOp, _l: Function, _r: Function) -> Function {
        unreachable!()
    }
    fn image(&self, _map: &MapSpec, _p: Function, _q: Function) -> Function {
        unreachable!()
    }
    fn preimage(&self, _map: &MapSpec, _p: Function, _q: Function) -> Function {
        unreachable!()
    }
    fn ite(&self, _c: Function, _yes: Function, _no: Function) -> Function {
        unreachable!()
    }
    fn quotient(&self, _f: Function, _slots: &[u32], _s: &mut Alignment) -> Function {
        unreachable!()
    }
    fn cases(&self, _default: u32, _cases: &[(Function, u32)], _labels: &mut Vec<u32>) -> Function {
        unreachable!()
    }
    fn values(
        &self,
        _default: u32,
        _keys: &[u64],
        _case_labels: &[u32],
        _slot_labels: &mut Vec<u32>,
        _scratch: &mut Values,
    ) -> Function {
        unreachable!()
    }
    fn members(&self, _p: Function) -> Box<dyn Members + '_> {
        unreachable!()
    }
}

/// A context whose only backend is `rogue`, with one 4-bit column on it.
fn rogue(rogue: Rogue) -> (Context, ColumnId) {
    let mut cx = Context::new();
    let backend = cx.register_backend("rogue", rogue);
    let c = cx.column(backend, ScalarSort::bit_vec(4));
    (cx, c)
}

// --- references --------------------------------------------------------------

#[test]
fn reference_helpers_skip_constants() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let p = cx.set(c, &[1, 3]);
    // A counted root survives one extra acquisition and release; constants
    // take neither.
    super::inc_ref(&cx, c, p.function());
    super::dec_ref(&cx, c, p.function());
    super::inc_ref(&cx, c, Function::TRUE);
    super::dec_ref(&cx, c, Function::FALSE);
    assert!(p.contains(&[1]) && !p.contains(&[2]));
}

// --- partition construction --------------------------------------------------

#[test]
#[should_panic(expected = "partition must have reachable dense slots")]
fn partitions_have_at_least_one_slot_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.partition(c, Function::FALSE, 0);
}

#[test]
#[should_panic(expected = "partition must have reachable dense slots")]
fn partition_slots_are_dense_u32s_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.partition(c, Function::new(2), 1 << 32);
}

#[test]
#[should_panic(expected = "trivial partition must be slot-zero constant")]
fn one_slot_partitions_are_the_false_constant_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.partition(c, Function::new(2), 1);
}

#[test]
#[should_panic(expected = "trivial partition must be slot-zero constant")]
fn the_false_constant_has_one_slot_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.partition(c, Function::FALSE, 2);
}

#[test]
#[should_panic(expected = "slot-one constant is not a dense partition")]
fn the_true_constant_is_no_partition_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.partition(c, Function::TRUE, 2);
}

#[test]
#[should_panic(expected = "column belongs to another context")]
fn partitions_resolve_only_in_their_own_context_is_rejected() {
    let mut other = Context::default();
    let c = col(&mut other, "bdd", 4);
    Context::default().partition(c, Function::FALSE, 1);
}

// --- alignment ---------------------------------------------------------------

#[test]
fn alignment_applies_its_laws_before_the_backend() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, _) = cases(&cx, c);
    let unit = cx.partition(c, Function::FALSE, 1);
    let mut labels = Vec::new();
    // Identical, left-trivial, and right-trivial alignments are laws of the
    // abstraction.
    assert_eq!(cx.align(f, f, &mut labels, &mut Alignment::default()), f);
    assert_eq!(labels, [(0, 0), (1, 1), (2, 2)]);
    labels.clear();
    assert_eq!(cx.align(unit, f, &mut labels, &mut Alignment::default()), f);
    assert_eq!(labels, [(0, 0), (0, 1), (0, 2)]);
    labels.clear();
    assert_eq!(cx.align(f, unit, &mut labels, &mut Alignment::default()), f);
    assert_eq!(labels, [(0, 0), (1, 0), (2, 0)]);
    // A real alignment refines both operands; the labels name the source
    // cells.
    let mut sl = Vec::new();
    let g = cx.values(c, 0, &[3, 5], &[1, 1], &mut sl, &mut Values::default());
    labels.clear();
    let h = cx.align(f, g, &mut labels, &mut Alignment::default());
    let cell = |v: u64| labels[cx.resolve(c).evaluate(h.function, &[v]) as usize];
    assert_eq!(h.slots as usize, labels.len());
    assert_eq!(cell(0), (0, 0));
    assert_eq!(cell(1), (1, 0));
    assert_eq!(cell(3), (1, 1));
    assert_eq!(cell(5), (2, 1));
}

#[test]
#[should_panic(expected = "a base-map root belongs to another column")]
fn alignment_stays_in_one_column_is_rejected() {
    let mut cx = Context::default();
    let (c, d) = (col(&mut cx, "bdd", 4), col(&mut cx, "sparse", 4));
    let (f, _) = cases(&cx, c);
    let (g, _) = cases(&cx, d);
    cx.align(f, g, &mut Vec::new(), &mut Alignment::default());
}

#[test]
fn alignment_shortcuts_reject_a_foreign_context_before_appending_labels() {
    let mut other = Context::default();
    let c = col(&mut other, "sparse", 4);
    let (f, _) = cases(&other, c);
    let unit = other.partition(c, Function::FALSE, 1);
    let cx = Context::default();
    for (left, right) in [(unit, unit), (f, f), (unit, f), (f, unit)] {
        let mut labels = vec![(7, 9)];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                cx.align(left, right, &mut labels, &mut Alignment::default());
            }))
            .is_err()
        );
        assert_eq!(labels, [(7, 9)]);
    }
    super::dec_ref(&other, c, f.function);
}

#[test]
#[should_panic(expected = "backend labeled an unreachable source slot")]
fn alignment_labels_name_reachable_source_slots_is_rejected() {
    let (cx, c) = rogue(Rogue::Labels);
    let (f, g) = (
        cx.partition(c, Function::new(2), 2),
        cx.partition(c, Function::new(3), 2),
    );
    cx.align(f, g, &mut Vec::new(), &mut Alignment::default());
}

// --- quotient ----------------------------------------------------------------

#[test]
fn quotients_coarsen_dense_classes() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, _) = cases(&cx, c);
    let mut scratch = Alignment::default();
    // A constant quotient is the trivial partition and an identity quotient
    // is `f` itself.
    assert!(cx.quotient(f, &[0, 0, 0], &mut scratch).is_trivial());
    assert_eq!(cx.quotient(f, &[0, 1, 2], &mut scratch), f);
    let g = cx.quotient(f, &[0, 0, 1], &mut scratch);
    assert_eq!(g.slots, 2);
    assert_eq!(
        slots(&cx, g),
        (0..16).map(|v| u32::from(v == 5)).collect::<Vec<_>>()
    );
}

#[test]
#[should_panic(expected = "one quotient class per old slot")]
fn quotients_map_every_slot_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, _) = cases(&cx, c);
    cx.quotient(f, &[0, 1], &mut Alignment::default());
}

#[test]
#[should_panic(expected = "quotient classes must be dense in first-occurrence order")]
fn quotient_classes_are_dense_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, _) = cases(&cx, c);
    cx.quotient(f, &[0, 2, 1], &mut Alignment::default());
}

// --- cases and values --------------------------------------------------------

#[test]
fn partitions_label_their_cells() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, labels) = cases(&cx, c);
    assert_eq!(f.slots, 3);
    assert!(!f.is_trivial());
    assert_eq!(labels.len(), 3);
    assert_eq!(
        slots(&cx, f)
            .iter()
            .map(|&i| labels[i as usize])
            .collect::<Vec<_>>(),
        (0..16)
            .map(|v| match v {
                1 | 3 => 1,
                5 => 2,
                _ => 0,
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(HashSet::from([f, f]).len(), 1);
    assert!(format!("{f:?}").contains("Partition"));
    // Packed exceptions build the same shape without predicate roots.
    let mut sl = Vec::new();
    let g = cx.values(c, 0, &[2, 6], &[1, 1], &mut sl, &mut Values::default());
    assert_eq!(g.slots, 2);
    assert_eq!(
        slots(&cx, g)
            .iter()
            .map(|&i| sl[i as usize])
            .collect::<Vec<_>>(),
        (0..16)
            .map(|v| u32::from(v == 2 || v == 6))
            .collect::<Vec<_>>()
    );
    // The trivial partition is the slot-zero constant.
    assert!(cx.partition(c, Function::FALSE, 1).is_trivial());
}

#[test]
#[should_panic(expected = "one packed key per case label")]
fn packed_cases_pair_keys_with_labels_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.values(c, 0, &[2, 6], &[1], &mut Vec::new(), &mut Values::default());
}

#[test]
#[should_panic(expected = "an explicit case must differ from the default")]
fn packed_cases_are_nondefault_is_rejected() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    cx.values(c, 1, &[2], &[1], &mut Vec::new(), &mut Values::default());
}

// --- refinement --------------------------------------------------------------

#[test]
fn refinement_delivers_only_the_selected_cells() {
    let mut cx = Context::default();
    let c = col(&mut cx, "bdd", 4);
    let (f, _) = cases(&cx, c);
    let p = cx.set(c, &[0, 1, 3, 5]);
    let mut scratch = Refinement::default();
    // Slot zero is declined, so it costs no cell; the others deliver one
    // owned root each.
    let mut seen = Vec::new();
    let flow = cx.refine(
        f,
        p.function(),
        &mut scratch,
        |slot| slot != 0,
        |cell, slot| {
            seen.push((slot, cx.resolve(c).count(cell)));
            cx.resolve(c).dec_ref(cell);
            ControlFlow::<()>::Continue(())
        },
    );
    assert!(flow.is_continue());
    seen.sort();
    assert_eq!(seen, [(1, BigUint::from(2_u32)), (2, BigUint::from(1_u32))]);
    // A Break stops visitation and returns the caller's value unchanged.
    let stop = cx.refine(
        f,
        p.function(),
        &mut scratch,
        |_| true,
        |cell, slot| {
            cx.resolve(c).dec_ref(cell);
            ControlFlow::Break(slot)
        },
    );
    assert!(matches!(stop, ControlFlow::Break(_)));
}

#[test]
#[should_panic(expected = "backend visited an invalid partition slot")]
fn refinement_slots_belong_to_the_partition_is_rejected() {
    let (cx, c) = rogue(Rogue::Slot);
    let f = cx.partition(c, Function::new(2), 2);
    let _ = cx.refine(
        f,
        Function::TRUE,
        &mut Refinement::default(),
        |_| true,
        |_, _| ControlFlow::<()>::Continue(()),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "backend violated the refinement Break contract")]
fn refinement_breaks_only_through_the_caller_is_rejected() {
    let (cx, c) = rogue(Rogue::Break);
    let f = cx.partition(c, Function::new(2), 2);
    let _ = cx.refine(
        f,
        Function::TRUE,
        &mut Refinement::default(),
        |_| true,
        |_, _| ControlFlow::<()>::Continue(()),
    );
}
