//! Stored sets and binary relations over native indexes.

use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use crate::backend::BoolOp;
use crate::context::{Context, Predicate};
use crate::index::{BalancedUnion, Index, IndexRoot};
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
    pub const fn root(&self) -> &IndexRoot {
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
    pub const fn universe(&self) -> Universe {
        if self.endpoints.is_some() {
            Universe::Binary
        } else {
            Universe::Set
        }
    }

    /// The relation with the same declared endpoints and the supplied root.
    ///
    /// # Panics
    ///
    /// Panics if the root belongs to another physical index.
    #[inline(always)]
    pub fn with_root(&self, root: IndexRoot) -> Self {
        assert_eq!(
            self.root.index(),
            root.index(),
            "index roots must share a physical layout"
        );
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
    pub fn apply(&self, op: BoolOp, other: &Self) -> Self {
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
    /// word count and record members pack in declaration order. A predicate
    /// supplies independent choices for its column, even when reused. Each row
    /// is a Cartesian product; complete products union without mixing
    /// separate rows' correlations. The batch shares sorting, duplicate
    /// removal, and equal prefixes before it is unioned once.
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
            fn predicates(&mut self) -> Option<&mut Vec<(usize, Predicate)>> {
                Some(&mut self.predicates)
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
        // The word range of each column.
        let mut spans = Vec::with_capacity(index.arity());
        let mut stride = 0;
        for j in 0..index.arity() {
            let words = cx.resolve(index.algebra(j)).sort().words();
            spans.push(stride..stride + words);
            stride += words;
        }
        // Each column compares its little-endian words from the most
        // significant. Where every column is one word wide and no column
        // carries a predicate, that order is the rows' word order, so one
        // slice comparison replaces the per-column search and slice pair.
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
            branches: BalancedUnion,
        }

        impl Frame<'_> {
            /// Unions the pending branches and ground points, emptying their buffers.
            fn finish(&mut self, index: &Index, level: usize) -> IndexRoot {
                let points = (!self.keys.is_empty()).then(|| {
                    let points = index.values(level, &self.keys, &self.children);
                    // Keep the children owned until values has retained the result.
                    self.keys.clear();
                    self.children.clear();
                    points
                });
                let out = self.branches.finish(index.empty());
                // Union the points after every symbolic branch.
                match points {
                    Some(p) => out.apply(BoolOp::Or, &p),
                    None => out,
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
                frame
                    .branches
                    .push(index.cases(level, &index.empty(), &[(p, &child)]));
            } else {
                frame.keys.extend_from_slice(frame.key);
                frame.children.push(child);
            }
        };
        self.root = self.root.apply(BoolOp::Or, &contribution);
    }
}
