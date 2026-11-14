//! Stored relations: a domain and range row over one native index, batch
//! loading, and the eager Boolean natives.

use std::cmp::Ordering;
use std::ops::Range;

use num_bigint::BigUint;

use crate::backend::{BoolOp, Function};
use crate::context::{Context, Predicate};
use crate::index::{self, BalancedUnion, IndexRoot, Labels, Level};
use crate::row::{self, Domain, IntoRow, Row, RowSink};
use crate::scalar::compare;
use crate::util::dag::Idx;

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
/// The universe determines the index columns: a subidentity over an n-column
/// row stores n columns shared by both endpoints, and a general relation
/// stores the d domain columns followed by the r range columns.
/// Every row position has its own coordinate; additional equalities live in
/// the root, never in storage aliases. Cloning shares the immutable index
/// root.
#[derive(Clone)]
pub struct Rel {
    /// The relation's owned root, shared by clones.
    pub root: IndexRoot,
    /// The domain row.
    pub domain: Row,
    /// The range row.
    pub range: Row,
    /// The declared universe, retained even when both endpoints have no columns.
    pub universe: Universe,
}

/// The argument row selected by `universe`.
///
/// Sets take one row and binary relations take the normalized domain-range
/// product, including named records with no scalar leaves. Both forms use the
/// same typed packing, and the row's columns are the index columns.
#[inline(always)]
fn arguments(cx: &Context, domain: &Row, range: &Row, universe: Universe) -> Row {
    if universe == Universe::Set {
        return range.clone();
    }
    row::product(cx, [domain, range])
}

impl Rel {
    fn empty(cx: &Context, domain: Row, range: Row, universe: Universe) -> Self {
        Self {
            root: index::intern(cx, row::columns(&arguments(cx, &domain, &range, universe))),
            domain,
            range,
            universe,
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
        Rel::empty(cx, domain, range, Universe::Binary)
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
        Rel::empty(cx, row.clone(), row, Universe::Set)
    }

    #[inline(always)]
    fn with_root(&self, root: IndexRoot) -> Self {
        Self {
            root,
            domain: self.domain.clone(),
            range: self.range.clone(),
            universe: self.universe,
        }
    }

    /// An empty relation with the same owner, field storage, and endpoint
    /// universe.
    ///
    /// This retains the shared index without executing a query or
    /// rebuilding its layout.
    pub fn empty_like(&self) -> Self {
        self.with_root(self.root.inc_ref(Function::FALSE))
    }

    /// The exact cardinality of the stored pair set, without running a
    /// query.
    pub fn count(&self) -> BigUint {
        self.root.count()
    }

    /// Whether the relation holds nothing.
    ///
    /// This is a root comparison that executes no query.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.root.function.is_f()
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
        let cx = self.root.context();
        let mut logical = Vec::new();
        row::append(
            cx,
            args,
            &arguments(cx, &self.domain, &self.range, self.universe),
            &mut logical,
        );
        self.root.contains(&logical)
    }

    /// Native Boolean algebra on stored relations over one shared index: `op`
    /// combines the two roots eagerly and materializes the result
    /// immediately.
    ///
    /// Here `AndNot` is plain set difference over the shared index.
    ///
    /// # Panics
    ///
    /// Panics if the endpoints differ in logical sort, declared universe, or
    /// physical columns.
    pub fn apply_bool(&self, op: BoolOp, other: &Self) -> Self {
        assert!(
            self.domain.sort == other.domain.sort && self.range.sort == other.range.sort,
            "relation endpoints have different logical sorts"
        );
        assert_eq!(
            self.universe, other.universe,
            "relation operations require the same set or binary universe"
        );
        assert!(
            self.root.same_index(&other.root),
            "relation operations require identical field columns"
        );
        self.with_root(
            self.root
                .apply_bool(op, self.root.function, other.root.function),
        )
    }

    /// The pairs this relation does not hold, relative to its columns' exact
    /// logical universes.
    ///
    /// A subidentity uses the whole row universe; a general relation uses the
    /// domain-range product. Equalities between
    /// coordinates constrain the root and are negated along with its other
    /// contents.
    pub fn not(&self) -> Self {
        self.with_root(
            self.root
                .apply_bool(BoolOp::AndNot, Function::TRUE, self.root.function),
        )
    }

    /// The pairs whose last range column holds the least value stored under
    /// the columns before it, in unsigned order.
    ///
    /// Every column before the last is a key: a symbolic domain cell is one
    /// key class per concrete value, so no minimum crosses the values it
    /// stands for, and a leading range field is part of the key as well.
    /// Keys with no pair stay absent, and lower is better. Like
    /// [`Self::apply_bool`] this is eager, since a key's minimum is known
    /// only once all of its candidates are stored. A stored relation keeps
    /// its last logical column as its last index column, so the rank column
    /// is the last level of the root.
    ///
    /// # Panics
    ///
    /// Panics if the range has no scalar column or its last column is a symbol
    /// sort, which has no numeric ordering.
    pub fn min_range(&self) -> Self {
        assert!(self.range.sort.len != 0, "minimum requires a range column");
        self.with_root(self.root.min_at(Level::new(self.root.arity() - 1)))
    }

    /// Adds one concrete or symbolic row/domain-range pair through the batch
    /// construction flow.
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
    /// Panics if a row does not match the argument row in shape or sorts, or
    /// a predicate belongs to another context or column.
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
            fn predicate(&mut self, p: &Predicate) {
                self.predicates.push((self.words.len(), p.clone()));
                let n = p.context().resolve(p.column()).sort().words();
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

        let cx = self.root.context();
        let row = arguments(cx, &self.domain, &self.range, self.universe);
        let mut batch = Batch::default();
        let mut count = 0;
        for args in rows {
            row::append(cx, args, &row, &mut batch);
            count += 1;
        }
        let index = &self.root;
        // The word range of each field; numeric order is per field, most
        // significant word first.
        let mut spans = Vec::with_capacity(index.arity());
        let mut stride = 0;
        for j in 0..index.arity() {
            let words = cx.resolve(index.column(j)).sort().words();
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
            (_, 0) => (0..usize::from(count != 0)).collect(),
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
                let mut distinct: Vec<usize> = (0..count).map(|i| i * stride).collect();
                distinct.sort_unstable_by(|&a, &b| compare_rows(a, b));
                distinct.dedup_by(|a, b| compare_rows(*a, *b).is_eq());
                distinct
            }
        };

        /// One column's working memory.
        ///
        /// The recursion descends a column per level and never re-enters a
        /// level while its keys and children are in use, so a level owns one
        /// set of buffers for the whole load instead of allocating them per
        /// interned node.
        #[derive(Default)]
        struct Frame {
            keys: Vec<u64>,
            children: Vec<IndexRoot>,
            branches: BalancedUnion,
        }

        /// Working memory shared by every column.
        ///
        /// A level fills it only after every deeper level returns, so one set
        /// serves the whole load.
        #[derive(Default)]
        struct Scratch {
            functions: Vec<Function>,
            labels: Labels,
        }

        /// The node of column `j` over rows sharing one prefix; equal
        /// subtrees intern once.
        ///
        /// The `frames` slice starts at column `j`, so the recursion hands the
        /// deeper levels their own.
        fn subtree(
            index: &IndexRoot,
            spans: &[Range<usize>],
            batch: &Batch,
            offsets: &[usize],
            j: usize,
            frames: &mut [Frame],
            scratch: &mut Scratch,
        ) -> IndexRoot {
            if offsets.is_empty() {
                return index.inc_ref(Function::FALSE);
            }
            if j == spans.len() {
                return index.inc_ref(Function::TRUE);
            }
            let c = &spans[j];
            let (frame, deeper) = frames
                .split_first_mut()
                .expect("frames must cover every column");
            frame.keys.clear();
            frame.children.clear();
            for run in offsets.chunk_by(|&a, &b| batch.compare(a, b, c).is_eq()) {
                let p = batch.predicate_at(run[0] + c.start);
                if p.is_some_and(|q| q.function().is_f()) {
                    continue;
                }
                let child = subtree(index, spans, batch, run, j + 1, deeper, scratch);
                if child.function.is_f() {
                    continue;
                }
                if let Some(p) = p {
                    let branch = index.cases(
                        Level::new(j),
                        Function::FALSE,
                        p.column(),
                        &[(p.function(), child.function)],
                    );
                    frame.branches.push(index, branch);
                } else {
                    frame
                        .keys
                        .extend_from_slice(&batch.words[run[0] + c.start..run[0] + c.end]);
                    frame.children.push(child);
                }
            }
            let root = frame.branches.finish(index, index.inc_ref(Function::FALSE));
            if frame.keys.is_empty() {
                return root;
            }
            scratch.functions.clear();
            scratch
                .functions
                .extend(frame.children.iter().map(|child| child.function));
            let points = index.values(
                Level::new(j),
                &frame.keys,
                &scratch.functions,
                &mut scratch.labels,
            );
            // The children are owned until `values` has taken its own
            // references.
            frame.children.clear();
            if root.function.is_f() {
                return points;
            }
            root.apply_bool(BoolOp::Or, root.function, points.function)
        }

        let mut frames: Vec<Frame> = (0..spans.len()).map(|_| Frame::default()).collect();
        let contribution = subtree(
            index,
            &spans,
            &batch,
            &offsets,
            0,
            &mut frames,
            &mut Scratch::default(),
        );
        self.root = self
            .root
            .apply_bool(BoolOp::Or, self.root.function, contribution.function);
    }
}

#[cfg(test)]
#[path = "../tests/unit/rel.rs"]
mod tests;
