//! Dense partitions of one column and the checked operations that combine them.
//!
//! A partition is a raw root token tagged with its column and reachable slot
//! count. Predicate owners account for their references through
//! [`crate::context::Predicate`]; partition reconstruction and reference
//! accounting remain separate trusted storage operations that require the
//! originating column and a live token. Raw functions are backend plumbing,
//! never comparisons across columns or contexts.

use std::ops::ControlFlow;

use crate::backend::{Alignment, Function, Refinement, Values};
use crate::context::{ColumnId, Context, check_packed};

/// A borrowed raw finite partition handle.
///
/// Its cells are nonempty, disjoint, cover the column's universe, and have
/// dense slot numbers `0..slots`. Slot numbers are not logical column values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Partition {
    /// The column the partition is over.
    pub column: ColumnId,
    /// The raw root; the reference belongs to whoever supplied the handle.
    pub function: Function,
    /// The number of reachable cells, numbered densely from zero.
    pub slots: u32,
}

impl Partition {
    /// Whether the partition has one cell, mapping every value to slot zero.
    pub const fn is_trivial(self) -> bool {
        self.slots == 1
    }
}

/// Acquires a partition reference for a native node.
///
/// Trusted storage lifecycle: `f` must be a live root of `column`. Unlike the
/// checked predicate API, native index nodes supply the column from their
/// immutable level.
#[inline(always)]
pub fn inc_ref(cx: &Context, column: ColumnId, f: Function) {
    // The backend contract makes both constants immortal, including the
    // trivial partition, so neither the column lookup nor the count is
    // needed for them.
    if !f.is_constant() {
        cx.resolve(column).inc_ref(f);
    }
}

/// Releases the partition reference owned by a native node, including losing
/// candidates.
///
/// The originating column is supplied by the container, never encoded in the
/// raw token.
#[inline(always)]
pub fn dec_ref(cx: &Context, column: ColumnId, f: Function) {
    if !f.is_constant() {
        cx.resolve(column).dec_ref(f);
    }
}

/// Checked partition construction over this context's columns.
impl Context {
    /// Tags a retained raw partition from trusted native index storage.
    ///
    /// The caller must supply its originating column and reachable slot count; this
    /// borrows the existing reference.
    ///
    /// # Panics
    ///
    /// Panics if `column` belongs to another context, `slots` is zero or exceeds
    /// `u32::MAX`, or the function and slot count disagree about triviality.
    pub fn partition(&self, column: ColumnId, function: Function, slots: usize) -> Partition {
        self.resolve(column);
        assert!(
            slots != 0 && slots <= u32::MAX as usize,
            "partition must have reachable dense slots"
        );
        assert_eq!(
            function.is_f(),
            slots == 1,
            "trivial partition must be slot-zero constant"
        );
        assert!(
            !function.is_t(),
            "slot-one constant is not a dense partition"
        );
        Partition {
            column,
            function,
            slots: slots as u32,
        }
    }

    /// The coarsest partition refining both operands, as one owned reference.
    ///
    /// Each result slot's `(left, right)` source slots are appended to `labels`.
    /// Equal and trivial operands are resolved here without a backend call.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different columns or another context, or
    /// the backend labels a slot its operand lacks.
    pub fn align(
        &self,
        left: Partition,
        right: Partition,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
    ) -> Partition {
        let column = left.column;
        assert_eq!(
            right.column, column,
            "a base-map root belongs to another column"
        );
        let backend = self.resolve(column);
        // These laws belong to the abstraction, not to a particular backend's
        // canonical form.
        if left == right {
            labels.extend((0..left.slots).map(|i| (i, i)));
            inc_ref(self, column, left.function);
            return left;
        }
        if left.is_trivial() {
            labels.extend((0..right.slots).map(|i| (0, i)));
            inc_ref(self, column, right.function);
            return right;
        }
        if right.is_trivial() {
            labels.extend((0..left.slots).map(|i| (i, 0)));
            inc_ref(self, column, left.function);
            return left;
        }
        let start = labels.len();
        let function = backend.align(left.function, right.function, labels, scratch);
        assert!(
            labels[start..]
                .iter()
                .all(|&(i, j)| i < left.slots && j < right.slots),
            "backend labeled an unreachable source slot"
        );
        self.partition(column, function, labels.len() - start)
    }

    /// Merges the cells of `f` by `slots`, which names each source slot's class in
    /// first-occurrence order, as one owned reference.
    ///
    /// A one-class result is the trivial partition and an identity map returns `f`
    /// itself.
    ///
    /// # Panics
    ///
    /// Panics if `slots` lacks one entry per old slot or its classes are not dense
    /// in first-occurrence order.
    pub fn quotient(&self, f: Partition, slots: &[u32], scratch: &mut Alignment) -> Partition {
        let column = f.column;
        let backend = self.resolve(column);
        assert_eq!(
            slots.len(),
            f.slots as usize,
            "one quotient class per old slot"
        );
        let mut count = 0_u32;
        for &slot in slots {
            assert!(
                slot <= count,
                "quotient classes must be dense in first-occurrence order"
            );
            if slot == count {
                count = count.checked_add(1).expect("quotient class count overflow");
            }
        }
        if count == 1 {
            // The one-cell partition maps every value to slot zero.
            return self.partition(column, Function::FALSE, 1);
        }
        if count as usize == slots.len() {
            inc_ref(self, column, f.function);
            return f;
        }
        let function = backend.quotient(f.function, slots, scratch);
        self.partition(column, function, count as usize)
    }

    /// Constructs the partition separating each case's predicate under its label
    /// from the `default` label; unreachable labels disappear.
    ///
    /// Labels append to the caller's buffer. Cases with different labels must not
    /// overlap; the backend checks that. The roots are borrowed rather than owned:
    /// the caller holds a live reference on each predicate for the duration of the
    /// call, so no reference is taken here.
    pub fn cases(
        &self,
        column: ColumnId,
        default: u32,
        cases: &[(Function, u32)],
        slot_labels: &mut Vec<u32>,
    ) -> Partition {
        let start = slot_labels.len();
        let function = self.resolve(column).cases(default, cases, slot_labels);
        self.partition(column, function, slot_labels.len() - start)
    }

    /// Constructs a labeled partition from sorted, distinct packed exceptions and
    /// nondefault labels.
    ///
    /// Ground-row construction has already normalized input order and repeated
    /// cases; this boundary checks that contract without copying the keys or
    /// labels.
    ///
    /// # Panics
    ///
    /// Panics if the keys and labels disagree in count, the keys are unsorted,
    /// repeated, or outside the sort, or a label equals `default`.
    pub fn values(
        &self,
        column: ColumnId,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        slot_labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Partition {
        let c = self.resolve(column);
        assert_eq!(
            keys.len(),
            case_labels.len() * c.sort().words(),
            "one packed key per case label"
        );
        check_packed(c.sort(), keys);
        assert!(
            case_labels.iter().all(|&label| label != default),
            "an explicit case must differ from the default"
        );
        let start = slot_labels.len();
        let function = c.values(default, keys, case_labels, slot_labels, scratch);
        self.partition(column, function, slot_labels.len() - start)
    }

    /// Visits the cells of `f` where `p` holds, keeping the caller's Break type.
    ///
    /// `select` accepts the slots the caller wants when they are first reached, so
    /// a declined slot costs no grouping or interning; `then` receives each
    /// accepted cell with its slot. Each delivered cell is one owned reference in
    /// `f`'s column, released by `then`. The object-safe bridge boxes nothing.
    /// Selection may run under the backend's guard, so `select` must not reenter
    /// that backend. Delivery through `then` runs outside all store guards.
    ///
    /// # Panics
    ///
    /// Panics if the backend visits a slot outside the partition.
    pub fn refine<B>(
        &self,
        f: Partition,
        p: Function,
        scratch: &mut Refinement,
        mut select: impl FnMut(u32) -> bool,
        mut then: impl FnMut(Function, u32) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let mut stopped = None;
        let flow = self.resolve(f.column).refine(
            f.function,
            p,
            scratch,
            &mut |slot| {
                assert!(slot < f.slots, "backend visited an invalid partition slot");
                select(slot)
            },
            &mut |slot, build| then(build(), slot).map_break(|value| stopped = Some(value)),
        );
        debug_assert_eq!(
            flow.is_break(),
            stopped.is_some(),
            "backend violated the refinement Break contract"
        );
        stopped.map_or(ControlFlow::Continue(()), ControlFlow::Break)
    }
}

#[cfg(test)]
#[path = "../tests/unit/partition.rs"]
mod tests;
