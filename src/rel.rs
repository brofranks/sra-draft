//! Stored sets and binary relations over native indexes.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use crate::backend::BoolOp;
use crate::context::{Context, Predicate};
use crate::index::{Index, IndexRoot};
use crate::row::{self, Domain, IntoRow, Row, RowSink};
use crate::scalar::compare;
use crate::util::count::Count;

/// The declared coordinate universe and argument shape of a relation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Universe {
    /// A row subidentity, with both endpoints sharing each row coordinate.
    Set,
    /// A product of independent domain and range coordinates.
    Binary,
}

/// A stored binary relation over one index.
///
/// A subidentity over an n-column row stores n columns shared by both
/// endpoints. A general relation stores its domain columns followed by its
/// range columns. Every row position has its own coordinate. Additional
/// equalities live in the root, never in storage aliases. Cloning shares the
/// immutable index root.
///
/// Insertion and membership take one row for a set or the normalized product
/// of the domain and range for a binary relation. Anonymous products flatten,
/// the empty product contributes no components, and named records keep their
/// member grouping. A relation from unit to a record with two four-bit fields
/// takes `(1, 2)`.
#[must_use = "relation operations return a new relation"]
#[derive(Clone, Debug)]
pub struct Rel {
    /// The relation's owned root, shared by clones.
    root: IndexRoot,
    /// The row used for packing and for the physical index layout.
    args: Row,
    /// The endpoint rows of a binary relation.
    ///
    /// A binary relation retains its two endpoints even when both are empty
    /// records. A set uses its argument row for both endpoints.
    endpoints: Option<Arc<(Row, Row)>>,
}

impl Rel {
    fn empty(cx: &Context, args: Row, endpoints: Option<(Row, Row)>) -> Self {
        Self {
            root: Index::new(cx, args.algebras()).empty(),
            args,
            endpoints: endpoints.map(Arc::new),
        }
    }

    /// The empty binary relation of `cx`, with independently bound domain
    /// and range occurrences.
    ///
    /// # Panics
    ///
    /// Panics if an endpoint belongs to another context or has an unbound
    /// leaf.
    pub fn new(cx: &Context, domain: impl Domain, range: impl Domain) -> Self {
        let (domain, range) = (domain.row(cx), range.row(cx));
        let args = Row::product(cx, &[&domain, &range]);
        Self::empty(cx, args, Some((domain, range)))
    }

    /// The empty subidentity over one bound row of `cx`.
    ///
    /// Both endpoints name the same columns.
    ///
    /// # Panics
    ///
    /// Panics if `row` belongs to another context or has an unbound leaf.
    pub fn new_set(cx: &Context, row: impl Domain) -> Self {
        let row = row.row(cx);
        cx.sorts().check(row.sort());
        Self::empty(cx, row, None)
    }

    /// The stored tuple set in its physical index.
    pub fn root(&self) -> &IndexRoot {
        &self.root
    }

    /// The bound domain row.
    pub fn domain(&self) -> &Row {
        self.endpoints.as_deref().map_or(&self.args, |(d, _)| d)
    }

    /// The bound range row.
    pub fn range(&self) -> &Row {
        self.endpoints.as_deref().map_or(&self.args, |(_, r)| r)
    }

    /// The declared coordinate universe.
    pub fn universe(&self) -> Universe {
        if self.endpoints.is_some() {
            Universe::Binary
        } else {
            Universe::Set
        }
    }

    #[inline(always)]
    fn with_root(&self, root: IndexRoot) -> Self {
        Self {
            root,
            args: self.args.clone(),
            endpoints: self.endpoints.clone(),
        }
    }

    /// An empty relation with the same owner, field storage, and endpoint
    /// universe.
    ///
    /// Retains the shared index and its layout.
    pub fn empty_like(&self) -> Self {
        self.with_root(self.root.index().empty())
    }

    /// The exact cardinality of the stored pair set.
    pub fn count(&self) -> Count {
        self.root.count()
    }

    /// Whether the relation holds nothing.
    ///
    /// This compares the stored root with the empty root.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Whether a ground row or pair belongs to this relation.
    ///
    /// The root checks any coordinate equalities.
    ///
    /// # Panics
    ///
    /// Panics if `args` does not match the argument row in shape or sorts, or
    /// holds a predicate.
    pub fn contains(&self, args: impl IntoRow) -> bool {
        let cx = self.root.index().context();
        let mut logical = Vec::new();
        row::append(cx, args, &self.args, &mut logical);
        self.root.contains(&logical)
    }

    /// The eager Boolean combination of two relations over one shared index.
    ///
    /// Here `AndNot` is plain set difference over the shared index.
    ///
    /// # Panics
    ///
    /// Panics if the endpoints differ in logical sort, declared universe, or
    /// physical columns.
    pub fn apply_bool(&self, op: BoolOp, other: &Self) -> Self {
        assert!(
            self.domain().sort() == other.domain().sort()
                && self.range().sort() == other.range().sort(),
            "relation endpoints have different logical sorts"
        );
        assert_eq!(
            self.universe(),
            other.universe(),
            "relation operations require the same set or binary universe"
        );
        self.with_root(self.root.apply(op, &other.root))
    }

    /// The pairs this relation does not hold, relative to its columns' exact
    /// logical universes.
    ///
    /// A subidentity uses the whole row universe. A general relation uses
    /// the domain-range product. Equalities between coordinates constrain
    /// the root and are negated along with its other contents.
    pub fn not(&self) -> Self {
        self.with_root(self.root.not())
    }

    /// The pairs whose last range column holds the least value stored under
    /// the columns before it, in unsigned order.
    ///
    /// Every preceding column in the index layout contributes to the key,
    /// including earlier range columns. The minimum is defined independently
    /// for each concrete key, even when many keys share a symbolic class.
    /// Keys with no pair stay absent. Like [`Self::apply_bool`], this is eager
    /// and uses all candidates stored for a key.
    ///
    /// # Panics
    ///
    /// Panics if the range has no scalar column or its last column is a symbol
    /// sort, which has no numeric ordering.
    pub fn min_range(&self) -> Self {
        assert!(
            self.range().sort().scalar_len() != 0,
            "minimum requires a range column"
        );
        self.with_root(self.root.min_at(self.root.index().arity() - 1))
    }

    /// Adds one row through [`Self::insert_rows`].
    ///
    /// # Panics
    ///
    /// Panics as [`Self::insert_rows`] does.
    #[inline(always)]
    pub fn insert(&mut self, args: impl IntoRow) {
        self.insert_rows(std::iter::once(args));
    }

    /// Adds rows built from values and scalar predicates.
    ///
    /// Subidentities contribute their row once; each scalar uses its exact
    /// word count and nested records flatten in order. A predicate supplies
    /// independent choices for its column, even when reused. Each row is a
    /// Cartesian product; complete products union without mixing separate
    /// rows' correlations. The batch shares sorting, duplicate removal, and
    /// equal prefixes before it is unioned once.
    ///
    /// # Panics
    ///
    /// Panics if a row does not match the argument row in shape or sorts.
    /// Each predicate must belong to this relation's context and use its
    /// destination column's algebra.
    pub fn insert_rows(&mut self, rows: impl IntoIterator<Item = impl IntoRow>) {
        #[derive(Default)]
        struct Batch {
            words: Vec<u64>,
            predicates: Vec<(usize, Predicate)>,
        }
        impl RowSink for Batch {
            fn words(&mut self) -> &mut Vec<u64> {
                &mut self.words
            }
            fn predicates(&self) -> &[(usize, Predicate)] {
                &self.predicates
            }
            fn push_predicate(&mut self, p: &Predicate) {
                self.predicates.push((self.words.len(), p.clone()));
                let n = p.context().resolve(p.algebra()).sort().words();
                self.words.resize(self.words.len() + n, 0);
            }
        }
        impl Batch {
            fn predicate_at(&self, offset: usize) -> Option<&Predicate> {
                self.predicates
                    .binary_search_by_key(&offset, |(i, _)| *i)
                    .ok()
                    .map(|i| &self.predicates[i].1)
            }

            fn compare(&self, a: usize, b: usize, c: &Range<usize>) -> Ordering {
                // A ground batch holds no predicate offsets, so it compares
                // the field words directly instead of searching an empty
                // list twice.
                if self.predicates.is_empty() {
                    return compare(
                        &self.words[a + c.start..a + c.end],
                        &self.words[b + c.start..b + c.end],
                    );
                }
                match (
                    self.predicate_at(a + c.start),
                    self.predicate_at(b + c.start),
                ) {
                    (None, None) => compare(
                        &self.words[a + c.start..a + c.end],
                        &self.words[b + c.start..b + c.end],
                    ),
                    (Some(p), Some(q)) => p.function().cmp(&q.function()),
                    (None, Some(_)) => Ordering::Less,
                    (Some(_), None) => Ordering::Greater,
                }
            }
        }

        let cx = self.root.index().context();
        let mut batch = Batch::default();
        if !row::append_rows(cx, rows, &self.args, &mut batch) {
            return;
        }
        let index = self.root.index();
        // The word range of each field; numeric order is per field, most
        // significant word first.
        let mut spans = Vec::with_capacity(index.arity());
        let mut stride = 0;
        for j in 0..index.arity() {
            let words = cx.resolve(index.algebra(j)).sort().words();
            spans.push(stride..stride + words);
            stride += words;
        }
        // Columns cover the row in order, so comparing them in turn compares
        // the packed words in turn; where every column is one word wide and
        // no column carries a predicate, that is one comparison of the rows
        // themselves instead of a per-column search and slice pair.
        let flat = batch.predicates.is_empty() && spans.iter().all(|c| c.len() == 1);

        /// The offsets of distinct flat `W`-word rows, sorted in place.
        ///
        /// Comparisons read each record directly instead of chasing offsets.
        fn sorted<const W: usize>(words: &mut [u64]) -> Vec<usize> {
            let records = words.as_chunks_mut::<W>().0;
            records.sort_unstable();
            records
                .iter()
                .enumerate()
                .filter(|&(i, record)| i == 0 || record != &records[i - 1])
                .map(|(i, _)| i * W)
                .collect()
        }
        let offsets: Vec<usize> = match (flat, stride) {
            (_, 0) => vec![0],
            (true, 1) => sorted::<1>(&mut batch.words),
            (true, 2) => sorted::<2>(&mut batch.words),
            (true, 3) => sorted::<3>(&mut batch.words),
            _ => {
                let compare_rows = |a: usize, b: usize| {
                    if flat {
                        batch.words[a..a + stride].cmp(&batch.words[b..b + stride])
                    } else {
                        spans
                            .iter()
                            .map(|c| batch.compare(a, b, c))
                            .find(|order| order.is_ne())
                            .unwrap_or(Ordering::Equal)
                    }
                };
                let mut distinct: Vec<usize> = (0..batch.words.len()).step_by(stride).collect();
                distinct.sort_unstable_by(|&a, &b| compare_rows(a, b));
                distinct.dedup_by(|a, b| compare_rows(*a, *b).is_eq());
                distinct
            }
        };

        /// A column's pending key and the buffers for its completed children.
        ///
        /// Each column reuses its buffers across prefixes. Its remaining rows
        /// resume after the child of the pending key has been built.
        #[derive(Default)]
        struct Frame<'a> {
            rest: &'a [usize],
            key: &'a [u64],
            predicate: Option<&'a Predicate>,
            keys: Vec<u64>,
            children: Vec<IndexRoot>,
            /// A binary counter of symbolic branches.
            ///
            /// Occupied slot `i` holds the union of `2^i` branches, so `k`
            /// branches of similar size cost `O(log k)` rounds of pairwise
            /// unions.
            branches: Vec<Option<IndexRoot>>,
        }

        impl Frame<'_> {
            /// Adds a symbolic branch to the balanced union.
            #[inline(always)]
            fn push_branch(&mut self, mut value: IndexRoot) {
                for slot in &mut self.branches {
                    match slot.take() {
                        Some(left) => value = left.apply(BoolOp::Or, &value),
                        None => {
                            *slot = Some(value);
                            return;
                        }
                    }
                }
                self.branches.push(Some(value));
            }

            /// Unions the pending branches and ground points, emptying their buffers.
            fn finish(&mut self, index: &Index, level: usize) -> IndexRoot {
                let root = self
                    .branches
                    .drain(..)
                    .rev()
                    .flatten()
                    .reduce(|a, b| a.apply(BoolOp::Or, &b))
                    .unwrap_or_else(|| index.empty());
                if self.keys.is_empty() {
                    return root;
                }
                let points = index.values(level, &self.keys, &self.children);
                // Keep the children owned until values has retained the result.
                self.keys.clear();
                self.children.clear();
                if root.is_empty() {
                    points
                } else {
                    root.apply(BoolOp::Or, &points)
                }
            }
        }

        let mut frames: Vec<Frame> = (0..spans.len()).map(|_| Frame::default()).collect();
        let mut level = 0;
        let mut rows = offsets.as_slice();
        let contribution = loop {
            let child = if level == spans.len() {
                index.full()
            } else {
                let span = &spans[level];
                let frame = &mut frames[level];
                if let Some((&first, tail)) = rows.split_first() {
                    let len = 1 + tail
                        .iter()
                        .take_while(|&&next| batch.compare(first, next, span).is_eq())
                        .count();
                    let (group, rest) = rows.split_at(len);
                    let start = first + span.start;
                    let predicate = batch.predicate_at(start);
                    if predicate.is_some_and(|p| p.function().is_f()) {
                        rows = rest;
                        continue;
                    }
                    frame.rest = rest;
                    frame.key = &batch.words[start..start + span.len()];
                    frame.predicate = predicate;
                    rows = group;
                    level += 1;
                    continue;
                }
                frame.finish(index, level)
            };
            if level == 0 {
                break child;
            }
            level -= 1;
            let frame = &mut frames[level];
            rows = frame.rest;
            if child.is_empty() {
                continue;
            }
            if let Some(p) = frame.predicate {
                frame.push_branch(index.cases(level, &index.empty(), &[(p, &child)]));
            } else {
                frame.keys.extend_from_slice(frame.key);
                frame.children.push(child);
            }
        };
        self.root = self.root.apply(BoolOp::Or, &contribution);
    }
}
