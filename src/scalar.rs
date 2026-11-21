//! Scalar universes and their packed little-endian value representation.

use std::cmp::Ordering;
use std::num::NonZeroU32;

/// The kind of scalar value interpreted by a backend.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ScalarKind {
    /// The Boolean sort, distinct from a one-bit bit-vector.
    Bool,
    /// A fixed-width bit-vector sort.
    Bv,
    /// Opaque 64-bit symbol IDs.
    Symbol,
}

/// The logical universe of one scalar sort.
///
/// Boolean and one-bit bit-vector sorts differ, as in SMT-LIB, and operations
/// choose signed or unsigned bit-vector meaning. A value uses exactly
/// [`Self::words`] little-endian `u64` words, with unused high bits zero.
/// Symbols span all 64-bit IDs, and callers own namespace identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScalarSort {
    /// The scalar kind: Boolean, bit-vector, or symbol.
    kind: ScalarKind,
    /// The number of value bits: one for Boolean and 64 for symbol sorts.
    width: NonZeroU32,
}

impl ScalarSort {
    /// The Boolean scalar sort, encoded in one bit.
    pub const fn bool() -> Self {
        Self {
            kind: ScalarKind::Bool,
            width: NonZeroU32::MIN,
        }
    }

    /// Constructs an exact fixed-width bit-vector sort.
    ///
    /// # Panics
    ///
    /// Panics if the width is zero.
    pub const fn bv(width: u32) -> Self {
        let Some(width) = NonZeroU32::new(width) else {
            panic!("bit-vector sorts must have nonzero width");
        };
        Self {
            kind: ScalarKind::Bv,
            width,
        }
    }

    /// The complete 64-bit symbol ID universe, including IDs outside any
    /// dictionary.
    pub const fn symbol() -> Self {
        Self {
            kind: ScalarKind::Symbol,
            width: NonZeroU32::new(64).expect("64 must be nonzero"),
        }
    }

    /// The kind of scalar value.
    pub const fn kind(self) -> ScalarKind {
        self.kind
    }

    /// The number of bits in one encoded value.
    pub const fn width(self) -> u32 {
        self.width.get()
    }

    /// The exact bit-vector width, or `None` for Boolean and symbols.
    pub const fn bv_width(self) -> Option<u32> {
        if matches!(self.kind, ScalarKind::Bv) {
            Some(self.width())
        } else {
            None
        }
    }

    /// The number of `u64` words in one encoded value.
    pub const fn words(self) -> usize {
        self.width().div_ceil(u64::BITS) as usize
    }

    /// The bits of the last encoded word that this sort uses.
    pub const fn top_mask(self) -> u64 {
        match self.width() % u64::BITS {
            0 => u64::MAX,
            bits => (1_u64 << bits) - 1,
        }
    }

    /// Whether `value` is [`Self::words`] words with no bit beyond the width.
    pub const fn accepts(self, value: &[u64]) -> bool {
        value.len() == self.words() && value[value.len() - 1] & !self.top_mask() == 0
    }
}

/// Checks that `values` are sorted, distinct, and encoded in `sort`.
///
/// # Panics
///
/// Panics if `values` is not a whole number of records, a record is outside
/// `sort`, or the records are not strictly increasing.
pub fn check_packed(sort: ScalarSort, values: &[u64]) {
    let words = sort.words();
    assert!(
        values.len().is_multiple_of(words),
        "values are not packed at the sort's word stride"
    );
    let records = values.chunks_exact(words);
    assert!(
        records.clone().is_sorted_by(|a, b| compare(a, b).is_lt()),
        "values are not sorted and distinct"
    );
    // Valid encodings form an initial interval, so the largest key suffices.
    assert!(
        records.last().is_none_or(|value| sort.accepts(value)),
        "value is outside the scalar sort"
    );
}

/// Numeric order of two equal-length little-endian encodings.
#[inline(always)]
pub fn compare(left: &[u64], right: &[u64]) -> Ordering {
    debug_assert_eq!(
        left.len(),
        right.len(),
        "compared encodings must have equal word counts"
    );
    // One-word keys, every sort up to 64 bits, compare as integers rather than
    // through the reversed iterator's length loop.
    if let ([l], [r]) = (left, right) {
        return l.cmp(r);
    }
    left.iter().rev().cmp(right.iter().rev())
}

/// Sorts the `words`-wide records of `packed` into numeric order and drops
/// repeats.
///
/// # Panics
///
/// Panics if the word stride is zero or `packed` ends with an incomplete
/// record.
pub fn sort_dedup(packed: &mut Vec<u64>, words: usize) {
    assert!(words != 0, "record word stride must be nonzero");
    assert!(
        packed.len().is_multiple_of(words),
        "packed values must contain complete records"
    );
    if words == 1 {
        packed.sort_unstable();
        packed.dedup();
        return;
    }
    if packed
        .chunks_exact(words)
        .is_sorted_by(|a, b| compare(a, b).is_lt())
    {
        return;
    }
    let record = |i: usize| &packed[i * words..][..words];
    let mut order: Vec<usize> = (0..packed.len() / words).collect();
    order.sort_unstable_by(|&a, &b| compare(record(a), record(b)));
    order.dedup_by(|a, b| record(*a) == record(*b));
    let mut out = Vec::with_capacity(order.len() * words);
    for &i in &order {
        out.extend_from_slice(record(i));
    }
    *packed = out;
}

/// Advances one valid encoding of `sort`.
///
/// Returns false, leaving `value` unspecified, past the end of `sort`.
pub fn successor(sort: ScalarSort, value: &mut [u64]) -> bool {
    // Words wrap to zero until one does not, ending the carry.
    value.iter_mut().any(|word| {
        *word = word.wrapping_add(1);
        *word != 0
    }) && sort.accepts(value)
}
