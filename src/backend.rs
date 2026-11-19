//! Exact predicates and partitions over scalar fields.
//!
//! A [`Column`] binds a store to a scalar sort. Callers retain the [`Function`]
//! roots they need and use them through columns with that store and sort.

use std::fmt;
use std::ops::ControlFlow;
use std::sync::Arc;

pub use crate::cache::StoreCache;
use crate::idx;
use crate::scalar::{self, ScalarKind, ScalarSort};
use crate::util::count::Count;
use crate::util::dag::{Idx, Map, Set};

idx! {
    /// One store-local map root.
    ///
    /// A root names either a **predicate**, a map to `{0, 1}`, or a
    /// **partition**, a surjective map to class numbers `0..k`. The store
    /// numbers each class in order of first appearance during traversal.
    /// A live root can be used in any column bound to its originating store
    /// and sort. The token records neither its role nor its ownership. Copying
    /// it leaves the reference count unchanged.
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
    /// The false predicate and trivial partition mapping every value to class
    /// zero.
    ///
    /// Every false predicate result must use this token.
    pub const FALSE: Self = Self(0);

    /// The true predicate, relative to the scalar universe.
    ///
    /// Every true predicate result must use this token, which denotes no
    /// canonical partition.
    pub const TRUE: Self = Self(1);

    /// Constructs a token from a store index without allocating a root or acquiring
    /// a reference.
    ///
    /// # Panics
    ///
    /// Panics if `index` exceeds `u32::MAX`.
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

    /// Whether this token is one of the two constants.
    ///
    /// Every store and container pins both constants, so acquiring or releasing a
    /// reference leaves their counts unchanged.
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

/// A unary map whose inputs and outputs belong to the same scalar sort.
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
    #[expect(
        clippy::type_complexity,
        reason = "keep the map callback signature explicit at the backend boundary"
    )]
    Rust(Arc<dyn Fn(&[u64], &mut [u64]) -> bool + Send + Sync>),
}

impl fmt::Debug for MapSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits { clear, flip } => f
                .debug_struct("Bits")
                .field("clear", clear)
                .field("flip", flip)
                .finish(),
            Self::Rust(_) => f.debug_tuple("Rust").finish_non_exhaustive(),
        }
    }
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
            assert!(
                domain.kind() != ScalarKind::Symbol || self.is_identity(),
                "symbols do not support bitwise maps"
            );
        }
    }

    /// Writes the image of `x` into `y` and returns whether `x` has an image.
    ///
    /// Both use the sort's little-endian words. Bit maps are total.
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

/// Caller-owned working memory for [`Column::refine`].
///
/// A delivered cell may start a nested refinement, so callers keep one
/// `Refinement` per nesting level and reuse it across sibling calls.
#[derive(Debug, Default)]
pub struct Refinement {
    /// Node pairs of the product still to discover.
    pub stack: Vec<(Function, Function)>,
    /// Node pairs of the product discovered already.
    pub seen: Set<(Function, Function)>,
    /// Product nodes already interned while building one cell.
    pub memo: Map<(Function, Function), Function>,
}

/// Caller-owned working memory for [`Column::values`].
#[derive(Debug, Default)]
pub struct Values;

/// Caller-owned working memory for [`Column::align`] and [`Column::quotient`].
#[derive(Debug, Default)]
pub struct Alignment {
    /// Aligned node pairs already interned.
    pub memo: Map<(Function, Function), Function>,
    /// Coarsened nodes already interned.
    pub map: Map<Function, Function>,
    /// Whole-call results kept across calls, as [`StoreCache`] describes.
    pub cache: StoreCache,
}

/// An owning cursor over a predicate's members.
///
/// Dropping a suspended cursor releases its root without visiting the remaining
/// values.
pub trait Members {
    /// Lends the next member in numeric order, or returns `None` at exhaustion.
    ///
    /// Members are distinct. The slice holds the column's exact word stride of
    /// little-endian words and borrows the cursor until its next advance. All
    /// store guards are released before returning.
    fn next(&mut self) -> Option<&[u64]>;
}

/// Transfers the finite members of one predicate through `map`, keeping those
/// whose image `holds`.
///
/// The result holds the images when `image`, otherwise the members, sorted,
/// distinct, and packed at a stride of `domain.words()`. The whole input is
/// exhausted before returning.
///
/// # Panics
///
/// Panics if `map` is invalid over `domain`, or if a cursor member or a
/// defined map image does not encode a value of that sort.
pub fn transfer(
    domain: ScalarSort,
    map: &MapSpec,
    members: &mut dyn Members,
    mut holds: impl FnMut(&[u64]) -> bool,
    image: bool,
) -> Vec<u64> {
    map.check_domain(domain);
    let (mut y, mut out) = (Vec::new(), Vec::new());
    while let Some(x) = members.next() {
        assert!(
            domain.accepts(x),
            "cursor member does not encode the scalar sort"
        );
        y.resize(x.len(), 0);
        if !map.evaluate(x, &mut y) {
            continue;
        }
        assert!(
            domain.accepts(&y),
            "map image does not encode the scalar sort"
        );
        if holds(&y) {
            out.extend_from_slice(if image { &y } else { x });
        }
    }
    if image {
        scalar::sort_records(&mut out, domain.words());
    }
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
/// [`Function`] laws, [`Self::align`] reports exactly the reachable pairs of
/// classes, [`Self::cases`] and [`Self::values`] only reachable labels,
/// [`Self::refine`] only nonempty cells, and [`Self::count`] the exact count.
pub trait Column: Send + Sync {
    /// The scalar sort this column is bound to.
    fn sort(&self) -> ScalarSort;

    /// Acquires one external reference to a live root in this store.
    ///
    /// Constants are unaffected.
    fn inc_ref(&self, f: Function);

    /// Releases one external reference to a root in this store, which the
    /// caller must own.
    ///
    /// Constants are unaffected. Other tokens may be reused only after all
    /// external and internal owners release them.
    fn dec_ref(&self, f: Function);

    /// The value of `f` at one valid domain value: a predicate's zero/one or a
    /// partition's class number.
    ///
    /// The value occupies `self.sort().words()` little-endian words.
    ///
    /// # Panics
    ///
    /// Panics if the value has the wrong word count or bits outside the sort.
    fn evaluate(&self, f: Function, value: &[u64]) -> u32;

    /// The number of values accepted by a borrowed Boolean predicate over the
    /// exact domain.
    ///
    /// Backends count symbolically without enumerating members or acquiring an
    /// output root.
    ///
    /// # Panics
    ///
    /// Panics if `p` is a partition with a class number above one, wherever the
    /// backend can tell it from a predicate.
    fn count(&self, p: Function) -> Count;

    /// Returns one owned predicate reference denoting exactly the supplied values.
    ///
    /// Values are sorted, distinct, and in the exact logical universe, packed at a
    /// stride of `self.sort().words()` little-endian words. An empty/full set uses
    /// the corresponding constant.
    ///
    /// # Panics
    ///
    /// Panics if `values` breaks this packing contract. Every implementation
    /// enforces it, with [`check_packed`](crate::scalar::check_packed).
    #[must_use]
    fn set(&self, values: &[u64]) -> Function;

    /// Returns one owned predicate for the unsigned cut `x < bound`, with `bound`
    /// one valid domain value.
    ///
    /// An empty/full cut uses the corresponding constant.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering, or
    /// `bound` is outside the sort.
    #[must_use]
    fn less_than(&self, bound: &[u64]) -> Function;

    /// Returns one owned predicate reference for the exact Boolean operation on
    /// `left` and `right`, relative to the domain's universe.
    ///
    /// Empty/full results use the corresponding constants, and `AndNot` denotes
    /// `left` minus `right`.
    #[must_use]
    fn apply_bool(&self, op: BoolOp, left: Function, right: Function) -> Function;

    /// Returns one owned predicate for `(x & mask) == value`, where `value` has
    /// no bits outside `mask`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, `mask` or `value` is outside the
    /// sort, `value` has a bit outside `mask`, or the backend has no symbolic bit
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
    /// Appends each reachable pair of input class numbers to `labels` once.
    /// The result class number indexes that suffix. With `start` the original
    /// length, `labels[start + h(x)] == (left(x), right(x))`.
    ///
    /// Inputs are distinct nontrivial partitions with live references. Callers
    /// handle identical and trivial partitions directly and supply working
    /// memory in `scratch`, as described at [`Alignment`].
    #[must_use]
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Alignment,
    ) -> Function;

    /// Returns one owned partition `h` with `h(x) == classes[f(x)]`.
    ///
    /// The slice has one entry per input class. Entry `classes[i]` gives the
    /// output class number for input class `i`, so `[0, 1, 0]` merges classes 0
    /// and 2. Output classes are numbered from zero in order of first appearance.
    ///
    /// The caller passes a proper coarsening with at least two result classes,
    /// handles constant and identity quotients directly, and supplies working
    /// memory in `scratch`.
    #[must_use]
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Alignment) -> Function;

    /// Returns one owned partition grouping values by their case label, or
    /// `default` when no predicate matches.
    ///
    /// Overlapping predicates must agree on their label.
    ///
    /// Appends each reachable label exactly once to `labels`, preserving its
    /// existing prefix. With `start` its original length and `h` the result,
    /// `labels[start + h(x)]` is the requested label at `x`. The result class
    /// number indexes only the appended suffix.
    ///
    /// # Panics
    ///
    /// Panics if overlapping predicates carry different labels.
    #[must_use]
    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function;

    /// Returns one owned partition grouping keys by `case_labels`, with `default`
    /// elsewhere.
    ///
    /// Keys are sorted, distinct, valid domain values packed at a stride of
    /// `self.sort().words()` little-endian words. There is one `case_labels[i]`
    /// per key, differing from `default`. As in [`Self::cases`], the result
    /// class numbers index the suffix appended to `labels`. The caller supplies
    /// working memory in `scratch`, described at [`Values`].
    ///
    /// # Panics
    ///
    /// Panics if `keys` breaks this packing contract or `case_labels` has a
    /// length other than the number of keys. Every implementation enforces it,
    /// with [`check_packed`](crate::scalar::check_packed).
    #[must_use]
    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Function;

    /// Visits the selected nonempty cells `p intersect f^-1(i)` in unspecified
    /// order.
    ///
    /// Calls `select(i)` once per reached class, possibly under this store's
    /// guard. The selector must not reenter this store, even by dropping
    /// one of its [`Self::members`] cursors. To prevent deadlock, any store it
    /// enters must follow this one in a lock order shared by all threads. Use
    /// `then` for other backend work.
    ///
    /// Calls `then(i, build)` once per accepted class, outside this store's guard
    /// and any retryable attempt. Each call to `build` returns one owned reference
    /// to the cell predicate. A `Break` stops visitation and is returned unchanged.
    ///
    /// The caller keeps `f` and `p` referenced until this returns, including
    /// during callbacks, because traversal borrows their descendants. The caller
    /// supplies working memory in `scratch`, as described at [`Refinement`].
    #[expect(
        clippy::type_complexity,
        reason = "keep the cell builder and continuation signatures explicit"
    )]
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
