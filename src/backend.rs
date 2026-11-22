//! Exact predicates and partitions over scalar fields.
//!
//! A store exposes its [`Algebra`] for a scalar sort through [`Backend::bind`].
//! Callers retain the [`Function`] roots they need and use them through any
//! binding to the same store and sort.

use std::fmt;
use std::ops::ControlFlow;
use std::sync::Arc;

pub use crate::cache::Cache;
use crate::idx;
use crate::scalar::{self, ScalarKind, ScalarSort};
use crate::util::count::Count;
use crate::util::dag::{Idx, Map, Set};

idx! {
    /// A store-local root token for a scalar predicate or partition.
    ///
    /// Predicates return 0 or 1. Partitions number their classes from zero in
    /// order of first appearance during the store's traversal.
    ///
    /// Any [`Algebra`] for the same store and sort can use a live root. Within
    /// that store and sort, equal live tokens denote equal maps. A backend may
    /// give equal maps distinct tokens, so fixed-point checks compare the maps
    /// themselves. Callers track each token's role and ownership. Copying a
    /// token acquires no reference.
    pub struct Function;
}

impl Function {
    /// The empty predicate and one-class partition, used by every such result.
    pub const FALSE: Self = Self(0);

    /// The full predicate, used by every such result. It denotes no canonical
    /// partition.
    pub const TRUE: Self = Self(1);

    /// Constructs a token from a store index, acquiring no reference.
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

    /// Whether this token is one of the two constants, which every store pins.
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
    #[inline(always)]
    pub const fn evaluate_bool(self, left: bool, right: bool) -> bool {
        self.evaluate_mask(left as u64, right as u64) != 0
    }

    /// The predicate token simplified using equality and the Boolean constants.
    ///
    /// Inputs share a store and sort. The result is borrowed, or `None` when
    /// these identities do not decide it.
    #[inline(always)]
    pub fn simplify(self, left: Function, right: Function) -> Option<Function> {
        self.simplify_roots(left, right, Function::FALSE, Function::TRUE)
    }

    /// The same Boolean identities for a root type with its own constants.
    ///
    /// Both inputs share a root namespace. A simplified result borrows its
    /// reference from an input or is one of the supplied constants.
    #[inline(always)]
    pub fn simplify_roots<I: Copy + Eq>(self, left: I, right: I, f: I, t: I) -> Option<I> {
        if left == right {
            return Some(match self {
                Self::And | Self::Or => left,
                Self::Xor | Self::AndNot => f,
            });
        }

        match self {
            Self::And if left == f || right == f => Some(f),
            Self::And if left == t => Some(right),
            Self::And if right == t => Some(left),
            Self::Or if left == f => Some(right),
            Self::Or if right == f => Some(left),
            Self::Or if left == t || right == t => Some(t),
            Self::Xor if left == f => Some(right),
            Self::Xor if right == f => Some(left),
            Self::AndNot if left == f || right == t => Some(f),
            Self::AndNot if right == f => Some(left),
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
    /// `y = (x & !clear) ^ flip`: bits in `clear` take `flip`'s value, and
    /// other bits invert where `flip` is set.
    ///
    /// This covers identity, masked assignment, and bitwise constant operations.
    Bits {
        /// The bits assigned from `flip` rather than inverted by it.
        clear: Box<[u64]>,
        /// The bits inverted, or assigned where `clear` is set.
        flip: Box<[u64]>,
    },
    /// A partial, pure, deterministic Rust function, applied only by enumeration.
    ///
    /// It writes every output word and returns `true`, or returns `false` for
    /// an input with no image, which transfer drops.
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
    /// Operation shortcuts require [`Self::check_sort`] first.
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Bits { clear, flip } => clear.iter().chain(flip.iter()).all(|&v| v == 0),
            Self::Rust(_) => false,
        }
    }

    /// Asserts that the map is valid over `sort`.
    ///
    /// # Panics
    ///
    /// Panics if bit masks have the wrong word stride or bits outside `sort`,
    /// or `sort` is a symbol sort and the bitwise map is not the identity.
    pub fn check_sort(&self, sort: ScalarSort) {
        if let Self::Bits { clear, flip } = self {
            assert!(
                sort.accepts(clear) && sort.accepts(flip),
                "bit map masks must encode the scalar sort"
            );
            assert!(
                sort.kind() != ScalarKind::Symbol || self.is_identity(),
                "symbols do not support bitwise maps"
            );
        }
    }

    /// Writes the image of `x` into `y` and returns whether `x` has an image.
    ///
    /// The caller first applies [`Self::check_sort`], and `x` encodes a value of
    /// that sort with `y` at the same word stride. Bit maps are total.
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

/// Group lookup sentinels for classes not yet reached and classes declined.
const UNREACHED: u32 = u32::MAX;
const DECLINED: u32 = u32::MAX - 1;

/// Reusable transition buffers for a sparse refinement.
#[derive(Debug, Default)]
pub struct Groups {
    /// Class, initial payload, and encoded transitions for each selected class.
    /// Groups beyond the selected prefix keep their buffers for the next call.
    groups: Vec<(u32, u32, Vec<u64>)>,
    /// Length of the selected prefix.
    len: usize,
    /// The group index for each class, or a sentinel for unreached and declined
    /// classes. Dense class numbering makes this a direct lookup.
    classes: Vec<u32>,
    /// Declined classes whose lookup entries need clearing on restart.
    declined: Vec<u32>,
}

impl Groups {
    /// The number of distinct classes reached, including declined classes.
    #[inline(always)]
    pub fn reached(&self) -> usize {
        self.len + self.declined.len()
    }

    /// The selected classes and their initial payloads and transition buffers.
    #[inline(always)]
    pub fn selected(&mut self) -> impl Iterator<Item = (u32, u32, &mut Vec<u64>)> {
        self.groups[..self.len]
            .iter_mut()
            .map(|(class, initial, transitions)| (*class, *initial, transitions))
    }

    /// Returns the selected class's transition buffer.
    ///
    /// Asks `select` on the first reach of a class after restart and sets the
    /// accepted group's initial payload. Declined classes yield `None`.
    ///
    /// # Panics
    ///
    /// Panics if the class lookup exceeds the platform's vector capacity or
    /// selecting the class would exceed `u32::MAX - 1` selected classes.
    #[inline(always)]
    pub fn group(
        &mut self,
        class: u32,
        initial: u32,
        select: &mut dyn FnMut(u32) -> bool,
    ) -> Option<&mut Vec<u64>> {
        let idx = class as usize;
        if self.classes.len() <= idx {
            let len = idx
                .checked_add(1)
                .expect("class lookup length exceeds usize");
            self.classes.resize(len, UNREACHED);
        }
        let i = self.classes[idx];
        if i == DECLINED {
            return None;
        }
        if i != UNREACHED {
            return Some(&mut self.groups[i as usize].2);
        }
        if !select(class) {
            self.classes[idx] = DECLINED;
            self.declined.push(class);
            return None;
        }
        let i = self.len;
        assert!(i < DECLINED as usize, "selected group index overflow");
        self.classes[idx] = i as u32;
        if i == self.groups.len() {
            self.groups.push((class, initial, Vec::new()));
        } else {
            let group = &mut self.groups[i];
            (group.0, group.1) = (class, initial);
            group.2.clear();
        }
        self.len += 1;
        Some(&mut self.groups[i].2)
    }

    /// Begins grouping, retaining the buffers from the previous call.
    #[inline(always)]
    pub fn restart(&mut self) {
        for &(class, _, _) in &self.groups[..self.len] {
            self.classes[class as usize] = UNREACHED;
        }
        for class in self.declined.drain(..) {
            self.classes[class as usize] = UNREACHED;
        }
        self.len = 0;
    }
}

/// Caller-owned working memory for [`Algebra::refine`].
///
/// A delivered cell may start a nested refinement, so callers keep one `Refine`
/// per nesting level and reuse it across sibling calls.
#[derive(Debug, Default)]
pub struct Refine {
    /// Buffered transitions for selected classes.
    pub groups: Groups,
    /// Node pairs of the product still to discover.
    pub stack: Vec<(Function, Function)>,
    /// Node pairs of the product discovered already.
    pub seen: Set<(Function, Function)>,
    /// Product nodes already interned while building one cell.
    pub memo: Map<(Function, Function), Function>,
}

/// Caller-owned working memory for [`Algebra::values`].
#[derive(Debug, Default)]
pub struct Values {
    /// The point assignment as encoded transition words, relabeled in place.
    pub transitions: Vec<u64>,
    /// One domain value, for stepping a key to its successor.
    pub next: Vec<u64>,
    /// The class carrying each label reached so far.
    pub classes: Map<u32, u32>,
}

/// Caller-owned working memory for [`Algebra::align`] and
/// [`Algebra::quotient`].
#[derive(Debug, Default)]
pub struct Align {
    /// The result map as encoded transition words.
    pub transitions: Vec<u64>,
    /// The result class for each reached pair of input classes.
    pub pairs: Map<(u32, u32), u32>,
    /// Aligned node pairs already interned.
    pub memo: Map<(Function, Function), Function>,
    /// Coarsened nodes already interned.
    pub coarse: Map<Function, Function>,
    /// Whole-call results kept across calls, as [`Cache`] describes.
    pub cache: Cache,
}

/// Checks the result class numbering and count of a quotient map.
///
/// # Panics
///
/// Panics if result classes are numbered out of first-appearance order or
/// the map leaves fewer than two classes or merges no input classes.
pub fn check_quotient(classes: &[u32]) {
    let mut n = 0;
    for &class in classes {
        let i = class as usize;
        assert!(
            i <= n,
            "quotient classes must be numbered in first-appearance order"
        );
        n += usize::from(i == n);
    }
    assert!(
        (2..classes.len()).contains(&n),
        "quotient must merge input classes and leave at least two result classes"
    );
}

/// A cursor over a predicate's members, as returned by [`Algebra::members`].
pub trait Cursor {
    /// Lends the next distinct member in unsigned numeric order,
    /// or returns `None` at exhaustion.
    ///
    /// The slice holds the sort's words and lives until the next advance. No
    /// store guard is held on return.
    fn next(&mut self) -> Option<&[u64]>;
}

/// Transfers the finite members of one predicate through `map`, keeping those
/// whose image `holds`.
///
/// Returns the images when `image`, otherwise the members, packed as for
/// [`Algebra::set`]. Calls `holds` in cursor order for every member with a
/// defined image, including repeated images. The cursor is exhausted.
///
/// # Panics
///
/// Panics if `map` is invalid over `sort`, or a member or image is outside it.
pub fn transfer(
    sort: ScalarSort,
    map: &MapSpec,
    cursor: &mut dyn Cursor,
    mut holds: impl FnMut(&[u64]) -> bool,
    image: bool,
) -> Vec<u64> {
    map.check_sort(sort);
    let (mut y, mut out) = (Vec::new(), Vec::new());
    let mut sorted = 0;
    while let Some(x) = cursor.next() {
        assert!(
            sort.accepts(x),
            "cursor member does not encode the scalar sort"
        );
        y.resize(x.len(), 0);
        if !map.evaluate(x, &mut y) {
            continue;
        }
        assert!(
            sort.accepts(&y),
            "map image does not encode the scalar sort"
        );
        if !holds(&y) || image && out.ends_with(&y) {
            continue;
        }
        out.extend_from_slice(if image { &y } else { x });
        // A small batch amortizes sorting repeated images. As the distinct
        // prefix grows, double its length before sorting again.
        if image && out.len() >= (2 * sorted).max(64) {
            scalar::sort_dedup(&mut out, sort.words());
            sorted = out.len();
        }
    }
    if image && sorted != out.len() {
        scalar::sort_dedup(&mut out, sort.words());
    }
    out
}

/// A store of exact predicates and partitions that binds to scalar sorts.
pub trait Backend: Send + Sync {
    /// Binds this store to `sort`, returning `None` if the sort is unsupported.
    ///
    /// Bindings to one store and sort share roots. Individual operations may
    /// still reject a sort, as each documents.
    fn bind(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Algebra>>;
}

/// The object-safe interface for exact predicate and partition operations
/// over one scalar sort in one store.
///
/// Predicates denote subsets of the sort's universe `U` and partitions denote
/// surjections `U -> 0..k`. Operations borrow live roots from the same store
/// and sort and return one owned reference per output root. Empty and full
/// predicate results use the [`Function`] constants. Results are exact:
/// labels and class pairs are only those reached, and cells are nonempty.
pub trait Algebra: Send + Sync {
    /// The scalar sort of this algebra.
    fn sort(&self) -> ScalarSort;

    /// Acquires one external reference to a live root in this store.
    ///
    /// Constants are unaffected.
    fn inc_ref(&self, f: Function);

    /// Releases one owned external reference to a root in this store.
    ///
    /// Constants are unaffected. A token is reused only after all owners
    /// release it.
    fn dec_ref(&self, f: Function);

    /// Releases one owned external reference per entry in the slice.
    ///
    /// Repeated tokens represent separate references. Constants are unaffected.
    fn dec_refs(&self, roots: &[Function]) {
        for &f in roots {
            self.dec_ref(f);
        }
    }

    /// The value of `f` at `value`: a predicate's zero/one or a partition's
    /// class number.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not an encoding in the sort.
    fn evaluate(&self, f: Function, value: &[u64]) -> u32;

    /// The exact number of values a predicate accepts, counted symbolically.
    ///
    /// # Panics
    ///
    /// Panics if `p` is a partition with a class above one, wherever the
    /// backend can tell.
    fn count(&self, p: Function) -> Count;

    /// Returns the predicate holding exactly at `values`.
    ///
    /// Values are sorted, distinct encodings in the sort, packed at a stride
    /// of `self.sort().words()` little-endian words.
    ///
    /// # Panics
    ///
    /// Panics if `values` breaks this packing, as
    /// [`check_packed`](crate::scalar::check_packed) checks.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn set(&self, values: &[u64]) -> Function;

    /// Returns the predicate for the unsigned cut `x < bound`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, which has no numeric ordering, or
    /// `bound` is outside the sort.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn lt(&self, bound: &[u64]) -> Function;

    /// Returns the predicate `op(left, right)`, where `AndNot` is `left` minus
    /// `right`.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function;

    /// Returns the predicate `(x & mask) == value`.
    ///
    /// # Panics
    ///
    /// Panics if the sort is a symbol sort, `mask` or `value` is outside the sort,
    /// `value` has a bit outside `mask`, or the backend has no symbolic bit tests,
    /// as by default.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn masked_eq(&self, _mask: &[u64], _value: &[u64]) -> Function {
        panic!("this backend cannot construct masked bit predicates")
    }

    /// Returns the image `{ f(x) | x in p, f(x) in q }` of `p` under `map`.
    ///
    /// A backend applies each supported map kind natively or by enumerating `p`.
    ///
    /// # Panics
    ///
    /// Panics if [`MapSpec::check_sort`] rejects `map`, or the backend rejects
    /// its kind.
    /// Panics if an invoked [`MapSpec::Rust`] callback returns `true` with an
    /// output outside the bound sort.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function;

    /// Returns the preimage `{ x in p | f(x) in q }` of `q` under `map`.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::image`] does.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function;

    /// Returns the predicate `condition(x) ? yes(x) : no(x)`.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function;

    /// Returns the partition `h` aligning `left` and `right`.
    ///
    /// Appends each reachable pair of input classes to `labels` once, so that
    /// `labels[start + h(x)] == (left(x), right(x))` with `start` the original
    /// length. Inputs are distinct nontrivial partitions; callers handle the
    /// identical and trivial cases.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Align,
    ) -> Function;

    /// Returns the partition `h` with `h(x) == classes[f(x)]`.
    ///
    /// Entry `classes[i]` is the output class of input class `i`, so `[0, 1, 0]`
    /// merges classes 0 and 2. Outputs number in order of first appearance.
    /// The caller passes a proper coarsening with at least two result classes.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Align) -> Function;

    /// Returns the partition grouping values by case label, with `default`
    /// where no predicate holds.
    ///
    /// Appends each reachable label once to `labels`, so that
    /// `labels[start + h(x)]` is the label at `x` with `start` the original length.
    ///
    /// # Panics
    ///
    /// Panics if overlapping predicates carry different labels, leaving
    /// `labels` unchanged.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function;

    /// Returns the partition grouping keys by `case_labels`, with `default`
    /// elsewhere.
    ///
    /// Keys are packed as for [`Self::set`], with one `case_labels[i]` per key,
    /// differing from `default`. Labels are appended as in [`Self::cases`].
    ///
    /// # Panics
    ///
    /// Panics if the keys break the packing or the label count differs from
    /// the number of keys.
    #[must_use = "release the returned root with `Algebra::dec_ref`"]
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
    /// When `f` is a predicate, its false values have class 0 and its true
    /// values have class 1.
    ///
    /// Calls `select(i)` at most once per reached class and once for every
    /// reached class if visitation completes. The selector may run under this
    /// store's guard. It must not reenter this store, even by dropping a
    /// [`Self::members`] cursor. Any store it enters must follow this one in a
    /// lock order shared by all threads.
    ///
    /// Calls `then(i, build)` once per accepted class until the callback
    /// returns `Break`, which stops visitation and is returned unchanged.
    /// Each visit runs outside the guard and any retryable attempt. Each
    /// `build` call returns one owned reference to the cell. Completing
    /// visitation returns `Continue(())`, including when `p` is empty.
    ///
    /// The caller keeps `f` and `p` referenced until this returns.
    #[expect(
        clippy::type_complexity,
        reason = "keep the cell builder and continuation signatures explicit"
    )]
    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refine,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()>;

    /// Returns a cursor over the members of `p` that owns a root reference
    /// until dropped.
    ///
    /// Other operations may run between advances.
    fn members(&self, p: Function) -> Box<dyn Cursor + '_>;
}

#[cfg(test)]
#[path = "../tests/unit/backend.rs"]
mod tests;
