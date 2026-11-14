//! Exact base-index algebra and native predicate construction.
//!
//! Backends operate on store-local [`Function`] tokens. The context pairs
//! each bound column with an id and owns the roots that handles retain; the
//! backend itself checks neither.

// Keep callback signatures explicit at the backend boundary.
#![allow(clippy::type_complexity)]

use std::ops::ControlFlow;
use std::sync::Arc;

use num_bigint::BigUint;

use crate::scalar::{ScalarKind, ScalarSort, sort_records};
use crate::util::dag::{Idx, Map, Set};

idx! {
    /// One store-local map root.
    ///
    /// A root names either a **predicate**, a map to `{0, 1}`, or a
    /// **partition**, a surjective map to slot numbers `0..k` numbered by first
    /// appearance in the store's traversal order. The token records neither
    /// role. A root is meaningful only in the column that produced it, and
    /// copying one neither increments its reference count nor records
    /// ownership.
    ///
    /// Every backend upholds three laws within one column: equal tokens denote
    /// equal maps, every empty predicate and trivial partition is
    /// [`Self::FALSE`], and every full predicate is [`Self::TRUE`]. Whether
    /// equal maps also share one token is a property each backend documents.
    /// Callers needing semantic equality of distinct tokens use a backend
    /// operation. Unequal tokens alone establish neither semantic inequality
    /// nor progress toward a fixed point.
    pub struct Function;
}

impl Function {
    /// The immortal false predicate and trivial partition with the single reachable
    /// slot zero.
    ///
    /// Every false predicate result must use this token.
    pub const FALSE: Self = Self(0);

    /// The immortal true predicate, relative to the scalar universe.
    ///
    /// Every true predicate result must use this token; it is not a canonical
    /// partition.
    pub const TRUE: Self = Self(1);

    /// Constructs a token from a store index without allocating a root or acquiring
    /// a reference.
    #[inline(always)]
    pub fn new(index: usize) -> Self {
        <Self as Idx>::new(index)
    }

    /// The store index, unchecked for ownership or liveness of the root.
    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Whether this token is the false predicate or trivial partition constant.
    #[inline(always)]
    pub const fn is_f(self) -> bool {
        self.0 == Self::FALSE.0
    }

    /// Whether this token is the true predicate constant.
    #[inline(always)]
    pub const fn is_t(self) -> bool {
        self.0 == Self::TRUE.0
    }

    /// Whether this token is one of the immortal constants.
    ///
    /// Every store and container pins both, so their reference counts never change
    /// and counting them is a guaranteed no-op.
    #[inline(always)]
    pub const fn is_constant(self) -> bool {
        self.0 <= Self::TRUE.0
    }
}

/// A Boolean operation over two predicates.
///
/// The `AndNot` operation is directional. The others are commutative.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BoolOp {
    /// Intersection: both predicates hold.
    And,
    /// Union: at least one predicate holds.
    Or,
    /// Symmetric difference: exactly one predicate holds.
    Xor,
    /// Difference: the left predicate holds and the right does not.
    AndNot,
}

impl BoolOp {
    /// Whether exchanging the operands preserves the result.
    #[inline(always)]
    pub const fn is_commutative(self) -> bool {
        matches!(self, Self::And | Self::Or | Self::Xor)
    }

    /// The result of the operation applied independently at each bit.
    ///
    /// The `AndNot` operation means `left & !right`.
    #[inline(always)]
    pub const fn evaluate_mask(self, left: u64, right: u64) -> u64 {
        match self {
            Self::And => left & right,
            Self::Or => left | right,
            Self::Xor => left ^ right,
            Self::AndNot => left & !right,
        }
    }

    /// The result of the operation on two Boolean values.
    ///
    /// The `AndNot` operation means `left && !right`.
    #[inline(always)]
    pub const fn evaluate_bool(self, left: bool, right: bool) -> bool {
        self.evaluate_mask(left as u64, right as u64) != 0
    }

    /// The predicate token simplified using equality and the Boolean constants.
    ///
    /// Inputs must belong to the same column. Returns a borrowed token without
    /// acquiring a reference, or `None` when these identities do not determine a
    /// token.
    #[inline(always)]
    pub fn simplify(self, left: Function, right: Function) -> Option<Function> {
        if left == right {
            return Some(match self {
                Self::And | Self::Or => left,
                Self::Xor | Self::AndNot => Function::FALSE,
            });
        }

        match self {
            Self::And if left.is_f() || right.is_f() => Some(Function::FALSE),
            Self::And if left.is_t() => Some(right),
            Self::And if right.is_t() => Some(left),
            Self::Or if left.is_f() => Some(right),
            Self::Or if right.is_f() => Some(left),
            Self::Or if left.is_t() || right.is_t() => Some(Function::TRUE),
            Self::Xor if left.is_f() => Some(right),
            Self::Xor if right.is_f() => Some(left),
            Self::AndNot if left.is_f() || right.is_t() => Some(Function::FALSE),
            Self::AndNot if right.is_f() => Some(left),
            _ => None,
        }
    }
}

/// A unary map of one scalar sort onto itself, as closed data that every
/// backend interprets directly.
///
/// Each backend states which kinds it applies. [`Self::evaluate`] is the
/// concrete meaning that enumeration-based transfer shares.
#[derive(Clone)]
pub enum MapSpec {
    /// `y = (x & !clear) ^ flip`: every bit of `clear` takes the value of `flip`
    /// there, and every other bit is inverted where `flip` is set.
    ///
    /// Identity, masked assignment, and the bitwise operations with a constant are
    /// all instances.
    Bits {
        /// The bits assigned from `flip` rather than inverted by it.
        clear: Box<[u64]>,
        /// The bits inverted, or assigned where `clear` is set.
        flip: Box<[u64]>,
    },
    /// A partial, pure, deterministic Rust function applied only by enumerating
    /// admitted finite inputs.
    ///
    /// It writes every output word and returns `true`, or returns `false` for an
    /// input with no image, which transfer then drops in both directions.
    Rust(Arc<dyn Fn(&[u64], &mut [u64]) -> bool + Send + Sync>),
}

impl MapSpec {
    /// The identity over a sort of `words` words: nothing cleared, nothing flipped.
    pub fn identity(words: usize) -> Self {
        Self::Bits {
            clear: vec![0; words].into(),
            flip: vec![0; words].into(),
        }
    }

    /// Whether this map is structurally the identity.
    ///
    /// Operation shortcuts require [`Self::check_domain`] first.
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Bits { clear, flip } => clear.iter().chain(flip.iter()).all(|&v| v == 0),
            Self::Rust(_) => false,
        }
    }

    /// Asserts that the map is valid over `domain`.
    ///
    /// # Panics
    ///
    /// Panics if bit masks have the wrong word stride or bits outside `domain`,
    /// or `domain` is a symbol sort and the bitwise map is not the identity.
    pub fn check_domain(&self, domain: ScalarSort) {
        if let Self::Bits { clear, flip } = self {
            assert!(
                domain.accepts(clear) && domain.accepts(flip),
                "bit map masks must encode the scalar sort"
            );
        }
        assert!(
            domain.kind() != ScalarKind::Symbol
                || !matches!(self, Self::Bits { .. })
                || self.is_identity(),
            "symbols do not support bitwise maps"
        );
    }

    /// Writes the image of `x` into `y`, both encoded as the sort's little-endian
    /// words, and returns whether `x` has an image; bit maps are total.
    ///
    /// The caller checks the map with [`Self::check_domain`] before evaluation.
    /// Both buffers must have that domain's exact word stride, and `x` must
    /// encode a value of the domain.
    #[inline(always)]
    pub fn evaluate(&self, x: &[u64], y: &mut [u64]) -> bool {
        match self {
            Self::Bits { clear, flip } => {
                for i in 0..y.len() {
                    y[i] = (x[i] & !clear[i]) ^ flip[i];
                }
                true
            }
            Self::Rust(f) => f(x, y),
        }
    }
}

/// The slot table entry of a reached slot the caller declined.
const SKIPPED: u32 = u32::MAX - 1;

/// Caller-owned working memory for [`Column::refine`].
///
/// A delivered cell may start a nested refinement, so callers keep one
/// `Refinement` per nesting level and reuse it across sibling calls.
#[derive(Default)]
pub struct Refinement {
    /// `(slot, default payload, transition entries)` for each selected slot.
    ///
    /// Groups at and beyond `live` belong to finished calls and keep their entry
    /// buffers for the next one.
    pub groups: Vec<(u32, u32, Vec<u64>)>,
    /// How many groups the call in progress has selected.
    pub live: usize,
    /// How many distinct slots the call in progress has reached, selected or not.
    pub reached: usize,
    /// The group holding each partition slot, `SKIPPED` where the caller declined
    /// the slot, or `u32::MAX` where the slot is unreached.
    ///
    /// Slot numbering is dense, so indexing it directly costs less than hashing a
    /// slot per transition.
    pub slots: Vec<u32>,
    /// The declined slots, so `finish` can clear their table entries.
    pub skipped: Vec<u32>,
    /// Node pairs of the product still to discover.
    pub stack: Vec<(Function, Function)>,
    /// Node pairs of the product discovered already.
    pub seen: Set<(Function, Function)>,
    /// Product nodes already interned while building one cell.
    pub memo: Map<(Function, Function), Function>,
}

impl Refinement {
    /// Begins a call over a partition whose slot numbers are below `slots`.
    #[inline(always)]
    pub fn restart(&mut self, slots: usize) {
        self.finish();
        self.live = 0;
        self.reached = 0;
        self.skipped.clear();
        if self.slots.len() < slots {
            self.slots.resize(slots, u32::MAX);
        }
    }

    /// Finds or opens the group of `slot`, returning `None` once the caller has
    /// declined it.
    ///
    /// The first reach of a slot asks `select` and opens a group with payload
    /// `default` when it accepts.
    #[inline(always)]
    pub fn group(
        &mut self,
        slot: u32,
        default: u32,
        select: &mut dyn FnMut(u32) -> bool,
    ) -> Option<usize> {
        let found = self.slots[slot as usize];
        if found == SKIPPED {
            return None;
        }
        if found != u32::MAX {
            return Some(found as usize);
        }
        self.reached += 1;
        if !select(slot) {
            self.slots[slot as usize] = SKIPPED;
            self.skipped.push(slot);
            return None;
        }
        let i = self.live;
        self.slots[slot as usize] = i as u32;
        if i == self.groups.len() {
            self.groups.push((slot, default, Vec::new()));
        } else {
            let group = &mut self.groups[i];
            (group.0, group.1) = (slot, default);
            group.2.clear();
        }
        self.live = i + 1;
        Some(i)
    }

    /// Clears the slot table once grouping is complete.
    ///
    /// The groups stay live for delivery, so a nested refinement started by a
    /// delivered cell finds the table empty and the groups intact.
    #[inline(always)]
    pub fn finish(&mut self) {
        for i in 0..self.live {
            self.slots[self.groups[i].0 as usize] = u32::MAX;
        }
        for &slot in &self.skipped {
            self.slots[slot as usize] = u32::MAX;
        }
    }
}

/// Caller-owned working memory for [`Column::values`].
///
/// Reusing the buffers avoids allocations across calls with similar workloads.
#[derive(Default)]
pub struct Values {
    /// The point assignment as runs, relabeled in place with canonical slots.
    pub transitions: Vec<u64>,
    /// One domain value, for stepping a key to its successor.
    pub next: Vec<u64>,
    /// The slot carrying each label reached so far.
    pub slots: Map<u32, u32>,
}

/// Entries above which a store cache is cleared, retaining its allocations.
const CACHE_BOUND: usize = 1 << 20;

/// Whole-call alignment and quotient results cached on one store.
///
/// Entries borrow tokens without retaining them, so [`Self::restart`] clears
/// the cache whenever the store or its maintenance epoch changes.
pub struct StoreCache {
    /// The store whose results are cached.
    pub store: u64,
    /// The store's maintenance epoch when the cached results were computed.
    pub epoch: u64,
    /// Entries above which the cache is cleared; `CACHE_BOUND` by default.
    ///
    /// This bounds entries rather than retained bytes.
    pub bound: usize,
    /// `(left, right)` to `(result, labels start, labels length)` in `labels`.
    pub aligned: Map<(Function, Function), (Function, usize, usize)>,
    /// The slot-pair labels of every cached alignment, back to back.
    pub labels: Vec<(u32, u32)>,
    /// `(f, slots hash)` to `(result, slots start, slots length)` in `slots`; the
    /// stored slots are compared in full on a hit.
    pub quotients: Map<(Function, u64), (Function, usize, usize)>,
    /// The slot maps of every cached quotient, back to back.
    pub slots: Vec<u32>,
}

impl StoreCache {
    /// An empty cache cleared once it holds more than `bound` entries.
    #[inline(always)]
    pub fn with_bound(bound: usize) -> Self {
        Self {
            store: 0,
            epoch: 0,
            bound,
            aligned: Map::default(),
            labels: Vec::new(),
            quotients: Map::default(),
            slots: Vec::new(),
        }
    }

    /// Binds the cache to `store` at maintenance epoch `epoch`, clearing every
    /// entry when either differs from the previous call or the entry count exceeds
    /// the bound.
    ///
    /// Allocations remain available for reuse.
    #[inline(always)]
    pub fn restart(&mut self, store: u64, epoch: u64) {
        if self.store != store
            || self.epoch != epoch
            || self.aligned.len() + self.quotients.len() > self.bound
        {
            self.aligned.clear();
            self.labels.clear();
            self.quotients.clear();
            self.slots.clear();
            self.store = store;
            self.epoch = epoch;
        }
    }
}

impl Default for StoreCache {
    fn default() -> Self {
        Self::with_bound(CACHE_BOUND)
    }
}

/// Caller-owned working memory for [`Column::align`] and [`Column::quotient`].
#[derive(Default)]
pub struct Alignment {
    /// The result map as runs.
    pub entries: Vec<u64>,
    /// The result slot of each source-slot pair reached so far.
    pub pairs: Map<(u32, u32), u32>,
    /// Aligned node pairs already interned.
    pub memo: Map<(Function, Function), Function>,
    /// Coarsened nodes already interned.
    pub map: Map<Function, Function>,
    /// Whole-call results kept across calls; see [`StoreCache`].
    pub cache: StoreCache,
}

/// An owning cursor over a predicate's members.
///
/// Dropping a suspended cursor releases its root without visiting the remaining
/// values.
pub trait Members {
    /// Replaces `out` with the next member in numeric order, without duplicates,
    /// and returns `true`; returns `false` when exhausted.
    ///
    /// Each member occupies the column's word stride of little-endian words. All
    /// store guards are released before returning.
    fn next(&mut self, out: &mut Vec<u64>) -> bool;
}

/// Transfers the finite members of one predicate through `map`, keeping those
/// whose image `holds`.
///
/// The result holds the images when `image`, otherwise the members, sorted,
/// distinct, and packed at a stride of `domain.words()`. The whole input is
/// exhausted and sorted before returning.
///
/// # Panics
///
/// Panics if `map` is invalid over `domain` or writes an image outside its
/// encoding.
pub fn transfer(
    domain: ScalarSort,
    map: &MapSpec,
    members: &mut dyn Members,
    mut holds: impl FnMut(&[u64]) -> bool,
    image: bool,
) -> Vec<u64> {
    map.check_domain(domain);
    let (mut x, mut y, mut out) = (Vec::new(), vec![0; domain.words()], Vec::new());
    while members.next(&mut x) {
        if !map.evaluate(&x, &mut y) {
            continue;
        }
        assert!(
            domain.accepts(&y),
            "map image does not encode the scalar sort"
        );
        if holds(&y) {
            out.extend_from_slice(if image { &y } else { &x });
        }
    }
    sort_records(&mut out, domain.words());
    out
}

/// A store of exact predicates and partitions that binds to scalar sorts.
///
/// Implementations define which exact scalar domains they support. Every
/// field operation runs on a [`Column`], the store bound to one sort.
pub trait Backend: Send + Sync {
    /// Binds this store to one scalar sort, or `None` when the store cannot
    /// hold that sort.
    ///
    /// Columns of the same store instance and equal sort share a root namespace
    /// and accept each other's live roots. Operations with a documented domain
    /// restriction reject unsupported sorts individually.
    fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>>;
}

/// One store bound to one scalar sort: the object-safe interface for exact
/// field algebra.
///
/// Predicates denote subsets of the sort's universe `U` and partitions denote
/// surjections `U -> 0..k`. Inputs are live roots from a compatible column,
/// one bound to the same store instance and equal sort. Operations borrow
/// their inputs and return one owned reference per output root. Beyond the
/// [`Function`] laws, every implementation is exact: [`Self::align`] reports
/// exactly the reachable slot pairs, [`Self::cases`] and [`Self::values`] only
/// reachable labels, [`Self::refine`] only nonempty cells, and [`Self::count`]
/// the exact count.
pub trait Column: Send + Sync {
    /// The scalar sort this column is bound to.
    fn sort(&self) -> ScalarSort;

    /// Acquires one external reference to a live root in this store.
    ///
    /// Constants remain immortal.
    fn inc_ref(&self, f: Function);

    /// Releases one external reference to a root in this store.
    ///
    /// Constants remain immortal; other tokens may be reused only after all
    /// external and internal owners release them.
    fn dec_ref(&self, f: Function);

    /// The value of `f` at one valid domain value: a predicate's zero/one or a
    /// partition's slot number.
    ///
    /// The value occupies `self.sort().words()` little-endian words.
    fn evaluate(&self, f: Function, value: &[u64]) -> u32;

    /// The number of values accepted by a borrowed Boolean predicate over the
    /// exact domain.
    ///
    /// Backends count symbolically without enumerating members or acquiring an
    /// output root.
    fn count(&self, p: Function) -> BigUint;

    /// Returns one owned predicate reference denoting exactly the supplied values.
    ///
    /// Values are sorted, distinct, and in the exact logical universe, packed at a
    /// stride of `self.sort().words()` little-endian words. An empty/full set uses
    /// the corresponding constant.
    #[must_use]
    fn set(&self, values: &[u64]) -> Function;

    /// Returns one owned predicate for the unsigned cut `x < bound`, with `bound`
    /// one valid domain value.
    ///
    /// An empty/full cut uses the corresponding constant.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering.
    #[must_use]
    fn less_than(&self, bound: &[u64]) -> Function;

    /// Returns one owned predicate reference for the exact Boolean operation on
    /// `left` and `right`, relative to the domain's universe.
    ///
    /// Empty/full results use the corresponding constants; `AndNot` denotes `left`
    /// minus `right`.
    #[must_use]
    fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> Function;

    /// Returns one owned predicate for `(x & mask) == value`, where `value` has
    /// no bits outside `mask`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort or the backend has no symbolic bit
    /// tests. The default implementation always panics.
    #[must_use]
    fn masked_eq(&self, _mask: &[u64], _value: &[u64]) -> Function {
        panic!("this backend cannot construct masked bit predicates")
    }

    /// Returns one owned predicate for `{ f(x) | x in p, f(x) in q }`: the image of
    /// `p` under `map` within `q`.
    ///
    /// Both predicates are borrowed from this column. A backend applies each map
    /// kind natively or by enumerating `p`, or rejects it explicitly.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort, as
    /// [`MapSpec::check_domain`] specifies, or the backend rejects its kind.
    #[must_use]
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function;

    /// Returns one owned predicate for `{ x in p | f(x) in q }`: the preimage of
    /// `q` under `map` within `p`.
    ///
    /// Both predicates are borrowed from this column.
    ///
    /// # Panics
    ///
    /// Panics if `map` is invalid over the bound sort, as
    /// [`MapSpec::check_domain`] specifies, or the backend rejects its kind.
    #[must_use]
    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function;

    /// Returns one owned Boolean predicate for `condition(x) ? yes(x) : no(x)`.
    ///
    /// All three inputs are borrowed predicates in this column.
    #[must_use]
    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function;

    /// Returns one owned partition `h` aligning `left` and `right`.
    ///
    /// Appends each reachable pair of input slots exactly once to `labels`,
    /// preserving its existing prefix. With `start` its original length, `(left(x),
    /// right(x)) == labels[start + h(x)]`; result slots index only the appended
    /// suffix. Inputs are distinct nontrivial partitions with live references;
    /// callers handle identical and trivial partitions directly. `scratch` is the
    /// caller's working memory for this call; see [`Alignment`].
    #[must_use]
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
    ) -> Function;

    /// Returns one owned partition `h` with `h(x) == slots[f(x)]`.
    ///
    /// There is one class per input slot, dense in first-occurrence order. This
    /// primitive receives a proper coarsening with at least two result cells;
    /// callers handle constant and identity quotients directly. `scratch` is the
    /// caller's working memory for this call.
    #[must_use]
    fn quotient(&self, f: Function, slots: &[u32], scratch: &mut Alignment) -> Function;

    /// Returns one owned partition grouping values by their case label, or
    /// `default` when no predicate matches.
    ///
    /// Overlapping predicates must agree on their label.
    ///
    /// Appends each reachable label exactly once to `slot_labels`, preserving its
    /// existing prefix. With `start` its original length and `h` the result,
    /// `slot_labels[start + h(x)]` is the requested label at `x`. Result slots
    /// index only the appended suffix.
    #[must_use]
    fn cases(
        &self,
        default: u32,
        cases: &[(Function, u32)],
        slot_labels: &mut Vec<u32>,
    ) -> Function;

    /// Returns one owned partition grouping keys by `case_labels`, with `default`
    /// elsewhere.
    ///
    /// Keys are sorted, distinct, valid domain values packed at a stride of
    /// `self.sort().words()` little-endian words. There is one `case_labels[i]`
    /// per key, differing from `default`. Labels are appended to `slot_labels`
    /// under the same contract as [`Self::cases`]. The caller supplies working
    /// memory in `scratch`; see [`Values`].
    #[must_use]
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        slot_labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Function;

    /// Visits exactly the nonempty cells `p intersect f^-1(i)` the caller
    /// selects, once per reached slot `i`, in no promised order.
    ///
    /// Calls `select(i)` once when slot `i` is first reached, possibly under
    /// this store's guard, so it must not call into this backend. For each
    /// accepted slot, calls `then(i, build)`, where each call to `build`
    /// transfers one owned reference to the cell predicate. Neither runs under
    /// a guard or within a retryable attempt. A `Break` stops visitation and is
    /// returned unchanged. The `scratch` buffer is the caller's working memory;
    /// see [`Refinement`].
    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refinement,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()>;

    /// Returns a cursor over the members of `p`, holding its own root reference
    /// for its lifetime, which is tied only to the column.
    ///
    /// Cursors need not be `Send`, and no store guard survives a returned
    /// member, so other operations may run between calls.
    fn members(&self, p: Function) -> Box<dyn Members + '_>;
}

#[cfg(test)]
#[path = "../tests/unit/backend.rs"]
mod tests;
