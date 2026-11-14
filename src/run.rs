//! Source-order CPS refinement with invocation-local predicate cells.
//!
//! Pending continuations retain only the endpoint cells they can read.
//! Collection detaches finalized output cells before memoizing the remaining
//! computation, so unrelated prefixes share one suffix graph. Callback
//! traversal follows the same refinements without memoizing caller effects.
//! No whole-query source table or field schedule is constructed.
//!
//! A map between two cells is an equation active while its operand's
//! continuation runs. Every narrowing of one endpoint transfers to the other,
//! so the active equations, which must form a forest, keep every cell arc
//! consistent. The terminal decomposes observed correlations before exposing
//! exact branches.

use std::ops::{ControlFlow, Range};
use std::sync::Arc;

use crate::backend::{BoolOp, Function, MapSpec, Refinement};
use crate::context::{ColumnId, Context, Predicate};
use crate::index::{IndexRoot, Level};
use crate::partition;
use crate::util::dag::{Idx, Map, Set, structural_hash};

/// A boundary cell's column and pending dependency.
///
/// Stored sources fix the column; maps read both endpoints without fixing their
/// storage.
#[derive(Clone, Copy)]
pub struct Slot {
    /// The cell's column.
    pub column: ColumnId,
    /// Whether a stored source or map reads this cell.
    pub read: bool,
    /// Whether a stored source fixes the column.
    pub fixed: bool,
}

/// A borrowed endpoint.
///
/// Repeated cell entries identify the very same predicate cell; they are not
/// independent choices from equal sets.
#[derive(Clone, Copy)]
pub struct Binding<'a> {
    eval: &'a Run<'a>,
    /// The endpoint's cells, in position order.
    pub cells: &'a [usize],
}

impl<'a> Binding<'a> {
    /// An owned handle on the endpoint's current cell.
    #[inline(always)]
    pub fn predicate(self, i: usize) -> Predicate {
        self.eval.predicate(self.cells[i])
    }
    /// The column and current root of the endpoint's cell `i`, borrowed for
    /// the callback's duration: the run holds the reference, so a use that
    /// ends before the callback returns needs no handle of its own.
    #[inline(always)]
    pub fn root(self, i: usize) -> (ColumnId, Function) {
        self.eval.root(self.cells[i])
    }

    /// The context the run evaluates in.
    #[inline(always)]
    pub fn context(self) -> &'a Context {
        self.eval.cx
    }
}

/// One referenced root of `column`; the run owns the reference.
///
/// Cells hold raw tokens rather than [`Predicate`] handles so each delivered
/// refinement cell costs one reference count, not a context clone as well.
pub struct Cell {
    /// The column whose store interprets `function`.
    pub column: ColumnId,
    /// The current root; the run owns one reference on it.
    pub function: Function,
    reads: usize,
}

/// The active equation `y = map(x)` between two cells.
pub struct Equation {
    /// The input cell of the equation.
    pub x: usize,
    /// The output cell of the equation.
    pub y: usize,
    /// The map that carries narrowings between the two cells.
    pub map: Arc<MapSpec>,
}

/// The cells, pending read footprints, and operand cell maps of one terminal
/// invocation.
///
/// Shared cell IDs propagate every downstream restriction to all aliases,
/// without copying predicate rows. Each combinator pushes its operands' maps
/// and truncates them on return.
pub struct Run<'c> {
    /// The context owning every column and store.
    pub cx: &'c Context,
    /// Every cell of this invocation: the enclosing boundary first, then
    /// operands' local cells.
    pub cells: Vec<Cell>,
    /// The read footprint, one entry per pending read of a cell.
    pub reads: Vec<usize>,
    /// Operand boundary cells, pushed by each combinator and truncated on
    /// return.
    pub maps: Vec<usize>,
    /// The active equations, a forest over cells.
    pub equations: Vec<Equation>,
    /// Cell roots replaced by the narrowings in progress, restored on return.
    undo: Vec<(usize, Function)>,
    /// Idle refinement buffers.
    ///
    /// Refinements nest through their delivered cells, so each active level
    /// holds one. Returning it here lets sibling calls at that level reuse its
    /// buffers.
    refinements: Vec<Refinement>,
}

impl<'c> Run<'c> {
    /// An invocation over the query boundary `slots`, each cell starting at
    /// its full universe.
    pub fn new(cx: &'c Context, slots: &[Slot]) -> Self {
        let mut eval = Self {
            cx,
            cells: Vec::new(),
            reads: Vec::new(),
            maps: (0..slots.len()).collect(),
            equations: Vec::new(),
            undo: Vec::new(),
            refinements: Vec::new(),
        };
        eval.extend(slots);
        eval
    }

    /// Appends one cell per slot, each at its full universe.
    pub fn extend(&mut self, slots: &[Slot]) {
        self.cells.extend(slots.iter().map(|slot| {
            // Resolving checks the column has a backend; the full universe
            // holds no counted reference.
            self.cx.resolve(slot.column);
            Cell {
                column: slot.column,
                function: Function::TRUE,
                reads: 0,
            }
        }));
    }

    /// The cell's column and current root, borrowed.
    ///
    /// `Run` holds one reference per cell root for its whole lifetime
    /// (`narrow`/`set` retire the previous root only after adopting the new
    /// one), so a use that does not outlive this borrow needs no reference of
    /// its own.
    #[inline(always)]
    pub fn root(&self, cell: usize) -> (ColumnId, Function) {
        let c = &self.cells[cell];
        (c.column, c.function)
    }

    /// A handle on the current root of `cell`, with its own reference.
    #[inline(always)]
    pub fn predicate(&self, cell: usize) -> Predicate {
        let c = &self.cells[cell];
        self.cx.predicate(c.column, c.function)
    }

    /// Restricts `cell` to `p` outright, outside every narrowing.
    pub fn set(&mut self, cell: usize, p: &Predicate) {
        let c = &mut self.cells[cell];
        assert_eq!(
            c.column,
            p.column(),
            "predicate uses an incompatible column"
        );
        partition::inc_ref(self.cx, c.column, p.function());
        partition::dec_ref(
            self.cx,
            c.column,
            std::mem::replace(&mut c.function, p.function()),
        );
    }

    /// Takes a refinement buffer for one nesting level, returned by
    /// [`Self::recycle`].
    #[inline(always)]
    pub fn refinement(&mut self) -> Refinement {
        self.refinements.pop().unwrap_or_default()
    }

    /// Returns a refinement buffer taken by [`Self::refinement`].
    #[inline(always)]
    pub fn recycle(&mut self, scratch: Refinement) {
        self.refinements.push(scratch);
    }

    /// Records one pending read of `cell`.
    #[inline(always)]
    pub fn push_read(&mut self, cell: usize) {
        self.reads.push(cell);
        self.cells[cell].reads += 1;
    }

    /// Retains an operand's stored endpoints until it runs, so earlier
    /// operands cannot finalize a cell that it still refines.
    pub fn push_reads(&mut self, slots: &[Slot], map: Range<usize>) -> usize {
        let start = self.reads.len();
        for (slot, i) in slots.iter().zip(map) {
            if slot.read {
                self.push_read(self.maps[i]);
            }
        }
        self.reads.len() - start
    }

    /// Removes the `n` most recent pending reads.
    #[inline(always)]
    pub fn pop_reads(&mut self, n: usize) {
        self.remove_reads(self.reads.len() - n, n);
    }

    /// Removes the `n` pending reads starting at `start`.
    ///
    /// A pending operand can sit below equations installed by the currently
    /// running operand.
    #[inline(always)]
    pub fn remove_reads(&mut self, start: usize, n: usize) {
        for cell in self.reads.drain(start..start + n) {
            self.cells[cell].reads -= 1;
        }
    }

    /// Restores the read footprint even when the continuation terminates
    /// early.
    #[inline(always)]
    fn read<T>(&mut self, cell: usize, then: impl FnOnce(&mut Self) -> T) -> T {
        debug_assert_eq!(self.reads.last(), Some(&cell));
        self.pop_reads(1);
        let result = then(self);
        self.push_read(cell);
        result
    }

    /// Whether the active equations, a forest, already connect `a` to `b`.
    pub fn connected(&self, a: usize, b: usize) -> bool {
        self.linked(a, b, usize::MAX, &|_| true)
    }

    /// Whether active identity equations already connect `a` to `b`.
    ///
    /// Structural equalities already active through identity equations need
    /// no second edge.
    pub fn aliased(&self, a: usize, b: usize) -> bool {
        self.linked(a, b, usize::MAX, &|eq| eq.map.is_identity())
    }

    /// Walks the forest from `a` to `b` over equations accepted by `ok`,
    /// excluding `from`.
    #[inline(always)]
    fn linked(&self, a: usize, b: usize, from: usize, ok: &impl Fn(&Equation) -> bool) -> bool {
        a == b
            || self
                .equations
                .iter()
                .enumerate()
                .filter(|&(i, eq)| i != from && ok(eq))
                .any(|(i, eq)| {
                    let other = if eq.x == a {
                        eq.y
                    } else if eq.y == a {
                        eq.x
                    } else {
                        return false;
                    };
                    self.linked(other, b, i, ok)
                })
    }

    /// Sets `cell` to `p`, remembering the replaced predicate, then transfers
    /// the narrowing along every active equation at `cell` except `from`,
    /// outward through the forest.
    ///
    /// Each equation keeps its endpoints mutually consistent, so every cell the
    /// forest reaches from a nonempty `p` stays nonempty: a transferred
    /// predicate is empty only for the first assignment of an equation, which
    /// [`Self::narrow`] rejects before assigning.
    fn assign(&mut self, cell: usize, p: Function, from: usize) {
        let old = std::mem::replace(&mut self.cells[cell].function, p);
        self.undo.push((cell, old));
        for i in 0..self.equations.len() {
            let eq = &self.equations[i];
            let (x, y) = (eq.x, eq.y);
            if i == from || (x != cell && y != cell) {
                continue;
            }
            let other = if x == cell { y } else { x };
            let next = self.transfer(&eq.map, x, y, x == cell);
            if next == self.cells[other].function {
                partition::dec_ref(self.cx, self.cells[other].column, next);
                continue;
            }
            self.assign(other, next, i);
        }
    }

    /// One referenced root in `y`'s column when `image`, else in `x`'s; see
    /// [`Context::transfer`].
    #[inline(always)]
    pub fn transfer(&self, map: &MapSpec, x: usize, y: usize, image: bool) -> Function {
        let (p, q) = (&self.cells[x], &self.cells[y]);
        self.cx
            .transfer(map, (p.column, p.function), (q.column, q.function), image)
    }

    /// Decomposes correlations among the cells observed at this continuation
    /// boundary.
    ///
    /// The enumerated cell is the highest observed index in a component, not
    /// the one with the fewest members, so a noninjective map may enumerate
    /// its larger side. Leave one observed cell per connected component
    /// symbolic; singleton choices for the others propagate through hidden
    /// witnesses before the sink sees the branch.
    pub fn finish<M: Mode, K: FnMut(&mut Self, &mut M) -> Flow<M>>(
        &mut self,
        end: usize,
        observed: &[usize],
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        if self.equations.is_empty() {
            return then(self, mode);
        }
        let cell = (0..end).rev().find(|&i| {
            observed.contains(&i) && observed.iter().any(|&j| j < i && self.connected(i, j))
        });
        let Some(cell) = cell else {
            return then(self, mode);
        };
        let cx = self.cx;
        let column = self.cells[cell].column;
        let mut cursor = cx.resolve(column).members(self.cells[cell].function);
        let mut value = Vec::new();
        let mut out = mode.sum();
        while cursor.next(&mut value) {
            let point = cx.resolve(column).set(&value);
            let result = self.narrow(cell, point, mode, |eval, mode| {
                eval.finish(cell, observed, mode, then)
            })?;
            mode.push(&mut out, cell, result);
        }
        ControlFlow::Continue(mode.finish_sum(out))
    }

    /// Restricts `cell` to `p`, one owned reference in the cell's column, for
    /// the extent of `then`, tightening every cell an active equation connects
    /// to it.
    ///
    /// An empty cell skips the continuation.
    #[inline(always)]
    pub fn narrow<M: Mode>(
        &mut self,
        cell: usize,
        p: Function,
        mode: &mut M,
        then: impl FnOnce(&mut Self, &mut M) -> Flow<M>,
    ) -> Flow<M> {
        if p.is_f() {
            partition::dec_ref(self.cx, self.cells[cell].column, p);
            return ControlFlow::Continue(mode.identity());
        }
        if self.equations.is_empty() && p == self.cells[cell].function {
            partition::dec_ref(self.cx, self.cells[cell].column, p);
            return then(self, mode);
        }
        let mark = self.undo.len();
        self.assign(cell, p, usize::MAX);
        let result = then(self, mode);
        for (cell, old) in self.undo.drain(mark..).rev() {
            let c = &mut self.cells[cell];
            partition::dec_ref(self.cx, c.column, std::mem::replace(&mut c.function, old));
        }
        result
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        // Unwinding leaves replaced roots owned by pending narrowings.
        for &(cell, old) in &self.undo {
            partition::dec_ref(self.cx, self.cells[cell].column, old);
        }
        for c in &self.cells {
            partition::dec_ref(self.cx, c.column, c.function);
        }
    }
}

/// The control flow of one traversal in mode `M`: a partial result to
/// continue with, or an early exit value.
pub type Flow<M> = ControlFlow<<M as Mode>::Break, <M as Mode>::Output>;

/// Execution supplied by a query's concrete Rust structure.
///
/// `map` addresses this operand's boundary cells within `eval.maps`. State
/// holds occurrence caches, not a lowered program.
pub trait Exec {
    /// Per-invocation occurrence state of this operand, such as suffix caches.
    type State<M: Mode>: Default;

    /// Whether this node only passes through its structural connections.
    ///
    /// A projection has no source traversal, filtering, or map equations.
    /// Its endpoint wiring supplies its entire meaning. Query selectors use
    /// this property independently of the cells an execution would read.
    #[inline(always)]
    fn is_projection(&self) -> bool {
        false
    }

    /// Runs this operand over the cells that `map` names within `eval.maps`,
    /// calling `then` for each accepting branch.
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        state: &mut Self::State<M>,
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M>;
}

impl<A: Exec> Exec for &A {
    type State<M: Mode> = A::State<M>;

    #[inline(always)]
    fn is_projection(&self) -> bool {
        (**self).is_projection()
    }

    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        state: &mut Self::State<M>,
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        (**self).run(map, eval, state, mode, then)
    }
}

/// Identity, projection, and constructors only connect cells; they do not
/// traverse data.
impl Exec for () {
    type State<M: Mode> = ();

    #[inline(always)]
    fn is_projection(&self) -> bool {
        true
    }

    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        _map: Range<usize>,
        eval: &mut Run<'_>,
        _state: &mut (),
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        then(eval, mode)
    }
}

/// Runtime field recursion passes the same continuation type unchanged.
///
/// The invocation retains its native source root; raw child tokens stay
/// reachable throughout recursion.
fn walk<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
    source: &IndexRoot,
    root: Function,
    level: usize,
    map: &Range<usize>,
    stop: usize,
    eval: &mut Run<'_>,
    memo: &mut M::Memo,
    mode: &mut M,
    then: &mut K,
) -> Flow<M> {
    if root.is_f() {
        return ControlFlow::Continue(mode.identity());
    }
    // A stop below the arity leaves only free trailing levels, and the full
    // arity ends at the TRUE leaf: no source suffix is left unconsumed either
    // way.
    if level == stop {
        return then(eval, mode);
    }
    mode.level(memo, root, level, eval, |eval, mode, memo| {
        let cell = eval.maps[map.start + level];
        // One guard reads the level and the partition it refines by.
        let head = (!root.is_t())
            .then(|| source.head(root))
            .filter(|(top, ..)| top.index() <= level);
        let Some((_, partition, hole)) = head else {
            // A skipped source level imposes the full field universe.
            return eval.read(cell, |eval| {
                walk(source, root, level + 1, map, stop, eval, memo, mode, then)
            });
        };
        // Skipped levels impose no backend-specific constraint and may adopt
        // another source's column. Only a level that actually refines its
        // cell requires identical storage.
        debug_assert_eq!(
            eval.cells[cell].column,
            source.column(level),
            "query variable uses incompatible columns"
        );
        let mut out = mode.sum();
        let mut scratch = eval.refinement();
        let flow = source.refine_cells(
            root,
            partition,
            hole,
            eval.cells[cell].function,
            &mut scratch,
            |p, child| {
                let result = eval.narrow(cell, p, mode, |eval, mode| {
                    eval.read(cell, |eval| {
                        walk(source, child, level + 1, map, stop, eval, memo, mode, then)
                    })
                });
                mode.push(&mut out, cell, result?);
                ControlFlow::Continue(())
            },
        );
        eval.recycle(scratch);
        flow.map_continue(|()| mode.finish_sum(out))
    })
}

/// A stored source leaf.
///
/// Trailing levels that nobody else reads, that carry no incoming restriction,
/// and that the sink never observes cannot change the answer, so a nonfalse
/// residual there already witnesses one accepting branch.
impl Exec for IndexRoot {
    type State<M: Mode> = M::Memo;

    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        memo: &mut M::Memo,
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        let arity = self.arity();
        debug_assert_eq!(map.len(), arity);
        for i in map.clone().rev() {
            eval.push_read(eval.maps[i]);
        }
        let mut stop = arity;
        while stop > 0 {
            let cell = eval.maps[map.start + stop - 1];
            let own = (stop - 1..arity)
                .filter(|&i| eval.maps[map.start + i] == cell)
                .count();
            if eval.cells[cell].reads != own
                || !eval.cells[cell].function.is_t()
                || mode.needs(cell)
            {
                break;
            }
            stop -= 1;
        }
        let result = walk(self, self.function, 0, &map, stop, eval, memo, mode, then);
        eval.pop_reads(arity);
        result
    }
}

/// A fold policy for the common traversal.
///
/// Callback mode keeps no state and never touches a memo; collection returns
/// graph roots and caches only completed suffix computations.
pub trait Mode: Sized {
    /// The result of one traversal.
    type Output;
    /// The value carried by an early exit.
    type Break;
    /// Suffix cache state for one source occurrence.
    type Memo: Default;
    /// The accumulator over the refinement cells of one level.
    type Sum;
    /// A fresh or recycled sum for one level.
    fn sum(&mut self) -> Self::Sum;
    /// Adds one branch of the refinement of `cell` to the sum over its cells.
    fn push(&mut self, sum: &mut Self::Sum, cell: usize, value: Self::Output);
    /// Folds a completed sum into one output.
    fn finish_sum(&mut self, sum: Self::Sum) -> Self::Output;

    /// The output of a traversal with no accepting branch.
    fn identity(&mut self) -> Self::Output;
    /// Accumulates `value` into `out`.
    fn add(&mut self, out: &mut Self::Output, value: Self::Output);
    /// One accepting branch, with every cell this sink still observes.
    fn done(&mut self, eval: &Run<'_>) -> Flow<Self>;
    /// Whether this sink itself observes `cell` once every source has passed
    /// it.
    ///
    /// Invocation-local cells lie outside every sink's boundary.
    fn needs(&self, cell: usize) -> bool;
    /// Brackets one union alternative; callback traversal needs no cache
    /// namespace.
    #[inline(always)]
    fn branch(
        &mut self,
        _right: bool,
        eval: &mut Run<'_>,
        then: impl FnOnce(&mut Run<'_>, &mut Self) -> Flow<Self>,
    ) -> Flow<Self> {
        then(eval, self)
    }
    /// Wraps one source level: finalize unread outputs, then reuse or compute
    /// the suffix.
    fn level(
        &mut self,
        memo: &mut Self::Memo,
        root: Function,
        level: usize,
        eval: &mut Run<'_>,
        then: impl FnOnce(&mut Run<'_>, &mut Self, &mut Self::Memo) -> Flow<Self>,
    ) -> Flow<Self>;
}

/// Callback mode: delivers each accepting branch to `then` as a domain and a
/// range binding.
pub struct Visit<'a, F> {
    /// The callback receiving each accepting branch.
    pub then: &'a mut F,
    /// The observed cells, domain first.
    pub ends: &'a [usize],
    /// The number of leading `ends` that form the domain binding.
    pub domain: usize,
}

impl<B, F: for<'a> FnMut(Binding<'a>, Binding<'a>) -> ControlFlow<B>> Mode for Visit<'_, F> {
    type Output = ();
    type Break = B;
    type Memo = ();
    type Sum = ();
    #[inline(always)]
    fn sum(&mut self) {}
    #[inline(always)]
    fn push(&mut self, _: &mut (), _: usize, _: ()) {}
    #[inline(always)]
    fn finish_sum(&mut self, _: ()) {}

    #[inline(always)]
    fn identity(&mut self) {}
    #[inline(always)]
    fn add(&mut self, _: &mut (), _: ()) {}
    #[inline(always)]
    fn done(&mut self, eval: &Run<'_>) -> ControlFlow<B> {
        (self.then)(
            Binding {
                eval,
                cells: &self.ends[..self.domain],
            },
            Binding {
                eval,
                cells: &self.ends[self.domain..],
            },
        )
    }
    #[inline(always)]
    fn needs(&self, cell: usize) -> bool {
        self.ends.contains(&cell)
    }
    #[inline(always)]
    fn level(
        &mut self,
        memo: &mut (),
        _: Function,
        _: usize,
        eval: &mut Run<'_>,
        then: impl FnOnce(&mut Run<'_>, &mut Self, &mut ()) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        then(eval, self, memo)
    }
}

/// Suffix results for one source occurrence, keyed by residual root and the
/// live predicates in read order.
///
/// Read footprints, aliases, and unfinalized output positions are fixed for
/// each occurrence, level, and union branch path. The path prefixes the key to
/// keep alternatives separate. Retained clones keep the key's tokens live.
pub type Memo = Vec<MemoLevel>;

/// Suffix results at one source level.
#[derive(Default)]
pub struct MemoLevel {
    /// Cached suffix roots by key.
    pub entries: Map<Box<[usize]>, IndexRoot>,
    /// Retained predicates keeping the keys' tokens live.
    pub live: Vec<Predicate>,
    /// Fingerprints of the keys computed so far.
    ///
    /// A key is memoized only when it recurs, so a key seen once, the rule on
    /// concrete point joins, costs neither retained clones nor an entry; a
    /// collision merely memoizes a key one sighting early.
    pub seen: Set<u64>,
}

/// The sum over one level's refinement cells of one query cell.
///
/// A branch that ends in the attachment of that cell's own output coordinate
/// is one case of the level node the union would build anyway: the cells of
/// one refinement are disjoint, so the whole sum is one partition over that
/// coordinate and the store builds it in a single sweep instead of a chain of
/// pairwise unions, each of which aligns and interns its own partition.
/// Branches of any other shape, including those that finalize a lower
/// coordinate first, fall to the balanced union in `rest` and are unioned
/// with the case node at the end.
#[derive(Default)]
pub struct Sum {
    /// The partition and continuation of each case, in one slice for the
    /// builder.
    cases: Vec<(Function, Function)>,
    /// The attachment retaining each case's partition, beside a reference on
    /// its continuation.
    held: Vec<(IndexRoot, IndexRoot)>,
    /// The output coordinate the cases attach; meaningful once `cases` is
    /// nonempty.
    level: usize,
    /// Pairwise union of the remaining branches, a binary counter of powers
    /// of two.
    rest: Vec<Option<IndexRoot>>,
}

impl Sum {
    /// Adds an unclassified branch to the balanced union.
    #[inline(always)]
    fn push_rest(&mut self, output: &IndexRoot, mut value: IndexRoot) {
        for slot in &mut self.rest {
            match slot.take() {
                Some(left) => value = output.apply_bool(BoolOp::Or, left.function, value.function),
                None => {
                    *slot = Some(value);
                    return;
                }
            }
        }
        self.rest.push(Some(value));
    }

    /// Unions unclassified branches with `out` and empties their buffer.
    #[inline(always)]
    fn finish_rest(&mut self, output: &IndexRoot, mut out: IndexRoot) -> IndexRoot {
        for value in self.rest.drain(..).rev().flatten() {
            out = output.apply_bool(BoolOp::Or, out.function, value.function);
        }
        out
    }
}

/// Boolean multiplication handles authored field reordering.
///
/// The ordinary increasing-level case attaches directly to the shared child,
/// retaining the existing index's partition/child algebra.
fn attach(
    index: &IndexRoot,
    level: usize,
    column: ColumnId,
    p: Function,
    child: &IndexRoot,
) -> IndexRoot {
    if child.function.is_t() || index.top(child.function).index() > level {
        return index.cases(
            Level::new(level),
            Function::FALSE,
            column,
            &[(p, child.function)],
        );
    }
    let cell = index.cases(
        Level::new(level),
        Function::FALSE,
        column,
        &[(p, Function::TRUE)],
    );
    index.apply_bool(BoolOp::And, cell.function, child.function)
}

/// Collection mode: materializes accepting branches into one output index,
/// memoizing completed suffixes.
pub struct Collect {
    /// The empty root of the output container: it supplies the container,
    /// its owning context, and the identity of collection.
    output: IndexRoot,
    levels: Vec<Vec<usize>>,
    pending: Vec<bool>,
    retired: Vec<usize>,
    key: Vec<usize>,
    branches: Vec<bool>,
    /// Output coordinates no source has finalized yet, the multiplicity of
    /// the pending cells: one aliased cell can occupy several levels.
    coordinates: usize,
    /// Finished level sums, kept for the next level to accumulate into.
    sums: Vec<Sum>,
    /// Unmatched keys for an exclusion probe; suffix caches still hold
    /// ordinary coverage.
    pub remaining: Option<IndexRoot>,
}

impl Collect {
    /// A collector into the empty `output` whose level `i` is filled by cell
    /// `positions[i]`, over `cells` invocation cells.
    ///
    /// A `remaining` universe turns the collector into an exclusion probe.
    pub fn new(
        output: IndexRoot,
        positions: &[usize],
        cells: usize,
        remaining: Option<IndexRoot>,
    ) -> Self {
        let mut levels = vec![Vec::new(); cells];
        for (i, &cell) in positions.iter().enumerate() {
            levels[cell].push(i);
        }
        Self {
            remaining,
            output,
            pending: levels.iter().map(|p| !p.is_empty()).collect(),
            levels,
            retired: Vec::new(),
            key: Vec::new(),
            branches: Vec::new(),
            sums: Vec::new(),
            coordinates: positions.len(),
        }
    }

    /// Removes complete output contributions from an exclusion probe's
    /// remaining keys.
    ///
    /// Contributions include prefixes omitted from cached suffixes. Only
    /// terminal branches and cache hits report here, and unwinding must not
    /// count them again. An exclusion probe keeps its own suffix cache, and a
    /// cache hit may replay an empty suffix, which removes nothing.
    fn observe(&mut self, suffix: IndexRoot, eval: &Run<'_>) -> Flow<Self> {
        if self.remaining.is_none() {
            return ControlFlow::Continue(suffix);
        }
        let mut root = suffix.clone();
        for &cell in self.retired.iter().rev() {
            root = self.attach(cell, eval, root);
        }
        let remaining = self.remaining.as_mut().unwrap();
        *remaining = self
            .output
            .apply_bool(BoolOp::AndNot, remaining.function, root.function);
        if remaining.function.is_f() {
            return ControlFlow::Break(remaining.clone());
        }
        ControlFlow::Continue(suffix)
    }

    /// Attaches the output coordinates of `cell` above `child`.
    ///
    /// The cell is pending or retired, so it holds at least one output
    /// coordinate.
    fn attach(&self, cell: usize, eval: &Run<'_>, child: IndexRoot) -> IndexRoot {
        // `eval` owns a reference on the cell's root for its whole lifetime,
        // so the ordinary single-level attachment borrows it instead of
        // constructing an owned `Predicate`.
        let (column, function) = eval.root(cell);
        match self.levels[cell].as_slice() {
            &[level] => attach(&self.output, level, column, function, &child),
            levels => {
                // Stored coordinates are independent. Build the diagonal over
                // their shared residual instead of a Cartesian product, even
                // for different nominal endpoints.
                let cx = self.output.context();
                let mut cursor = cx.resolve(column).members(function);
                let mut value = Vec::new();
                let mut diagonal = Sum::default();
                while cursor.next(&mut value) {
                    let point = cx.set(column, &value);
                    let mut branch = child.clone();
                    for &level in levels.iter().rev() {
                        branch = attach(
                            &self.output,
                            level,
                            point.column(),
                            point.function(),
                            &branch,
                        );
                    }
                    diagonal.push_rest(&self.output, branch);
                }
                diagonal.finish_rest(&self.output, self.output.clone())
            }
        }
    }
}

impl Mode for Collect {
    type Output = IndexRoot;
    type Break = IndexRoot;
    type Memo = Memo;
    type Sum = Sum;
    #[inline(always)]
    fn sum(&mut self) -> Self::Sum {
        self.sums.pop().unwrap_or_default()
    }
    #[inline(always)]
    fn push(&mut self, sum: &mut Self::Sum, cell: usize, value: IndexRoot) {
        if value.function.is_f() {
            return;
        }
        // One output coordinate of `cell` is the only level of the output
        // this branch can have attached outermost, and its region lies within
        // this refinement cell, so the cases collected here are disjoint.
        if let Some(&[level]) = self.levels.get(cell).map(Vec::as_slice)
            && let Some((p, child)) = self.output.attachment(value.function, Level::new(level))
        {
            sum.level = level;
            sum.cases.push((p, child.function));
            sum.held.push((value, child));
            return;
        }
        sum.push_rest(&self.output, value);
    }
    #[inline(always)]
    fn finish_sum(&mut self, mut sum: Self::Sum) -> IndexRoot {
        let out = match sum.cases.len() {
            0 => self.identity(),
            // A lone case is the node the builder would rebuild: it was read
            // off that very node.
            1 => {
                sum.cases.clear();
                sum.held.pop().unwrap().0
            }
            _ => {
                let level = Level::new(sum.level);
                let out = self.output.cases(
                    level,
                    Function::FALSE,
                    self.output.column(sum.level),
                    &sum.cases,
                );
                // The k-ary union and pairwise chain denote the same relation.
                // Backends may give equal partitions distinct tokens, so
                // compare their contents where the cost is affordable, in
                // the debug build used by the exhaustive fixtures.
                #[cfg(debug_assertions)]
                {
                    let mut chain = self.output.inc_ref(Function::FALSE);
                    for (value, _) in sum.held.iter() {
                        chain = self
                            .output
                            .apply_bool(BoolOp::Or, chain.function, value.function);
                    }
                    assert!(
                        self.output
                            .apply_bool(BoolOp::Xor, chain.function, out.function)
                            .function
                            .is_f(),
                        "the k-ary union differs from the Or chain"
                    );
                }
                sum.cases.clear();
                sum.held.clear();
                out
            }
        };
        let out = sum.finish_rest(&self.output, out);
        self.sums.push(sum);
        out
    }

    #[inline(always)]
    fn branch(
        &mut self,
        right: bool,
        eval: &mut Run<'_>,
        then: impl FnOnce(&mut Run<'_>, &mut Self) -> Flow<Self>,
    ) -> Flow<Self> {
        self.branches.push(right);
        let result = then(eval, self);
        self.branches.pop();
        result
    }

    #[inline(always)]
    fn identity(&mut self) -> IndexRoot {
        self.output.clone()
    }

    #[inline(always)]
    fn add(&mut self, out: &mut IndexRoot, value: IndexRoot) {
        *out = self
            .output
            .apply_bool(BoolOp::Or, out.function, value.function);
    }

    /// Every output cell that no source has finalized is attached to this
    /// accepting branch.
    #[inline(always)]
    fn done(&mut self, eval: &Run<'_>) -> Flow<Self> {
        let mut root = self.output.inc_ref(Function::TRUE);
        for cell in (0..self.pending.len()).rev() {
            if self.pending[cell] {
                root = self.attach(cell, eval, root);
            }
        }
        self.observe(root, eval)
    }

    #[inline(always)]
    fn needs(&self, cell: usize) -> bool {
        self.levels
            .get(cell)
            .is_some_and(|levels| !levels.is_empty())
    }

    #[inline(always)]
    fn level(
        &mut self,
        memo: &mut Memo,
        root: Function,
        level: usize,
        eval: &mut Run<'_>,
        then: impl FnOnce(&mut Run<'_>, &mut Self, &mut Memo) -> Flow<Self>,
    ) -> Flow<Self> {
        let start = self.retired.len();
        for cell in 0..self.pending.len() {
            if self.pending[cell] && eval.cells[cell].reads == 0 {
                self.pending[cell] = false;
                self.coordinates -= self.levels[cell].len();
                self.retired.push(cell);
            }
        }
        // Only suffixes with two coordinates still to fill are keyed, looked
        // up, and retained. A suffix with one pending coordinate can recur,
        // because its key omits retired cells and upstream concrete values,
        // so composition chains recompute it once per upstream branch. The
        // gate trades that recomputation for cheaper keys and smaller caches,
        // measured as the faster choice on clique workloads.
        let mut result = if self.coordinates < 2 {
            then(eval, self, memo)
        } else {
            if memo.len() <= level {
                memo.resize_with(level + 1, MemoLevel::default);
            }
            self.key.clear();
            self.key.push(self.branches.len());
            self.key
                .extend(self.branches.iter().map(|&right| usize::from(right)));
            self.key.push(root.index());
            self.key.extend(
                eval.reads
                    .iter()
                    .map(|&cell| eval.cells[cell].function.index()),
            );
            let cached = memo[level].entries.get(self.key.as_slice()).cloned();
            match cached {
                Some(suffix) => self.observe(suffix, eval),
                None => {
                    let repeat = !memo[level]
                        .seen
                        .insert(structural_hash(self.key.as_slice()));
                    // Deeper levels reuse the scratch, so this key is fixed
                    // before descending.
                    let key: Option<Box<[usize]>> = repeat.then(|| self.key.as_slice().into());
                    let result = then(eval, self, memo);
                    if let (Some(key), ControlFlow::Continue(suffix)) = (key, &result) {
                        memo[level]
                            .live
                            .extend(eval.reads.iter().map(|&cell| eval.predicate(cell)));
                        memo[level].entries.insert(key, suffix.clone());
                    }
                    result
                }
            }
        };
        while self.retired.len() > start {
            let cell = self.retired.pop().unwrap();
            self.pending[cell] = true;
            self.coordinates += self.levels[cell].len();
            result = result.map_continue(|root| {
                if root.function.is_f() {
                    root
                } else {
                    self.attach(cell, eval, root)
                }
            });
        }
        result
    }
}

#[cfg(test)]
#[path = "../tests/unit/run.rs"]
mod tests;
