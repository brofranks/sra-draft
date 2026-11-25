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

use crate::backend::{BoolOp, Function, MapSpec, Refine};
use crate::context::{AlgebraId, Context, Predicate};
use crate::index::{BalancedUnion, Index, IndexRoot, Residual};
use crate::util::dag::{Map, Set, structural_hash};

/// A boundary cell's algebra and pending dependency.
///
/// Stored sources fix the column; maps read both endpoints without fixing their
/// storage.
#[derive(Clone, Copy)]
pub struct Slot {
    /// The algebra interpreting this cell.
    pub algebra: AlgebraId,
    /// Whether a running operand reads this cell: a stored source, map, union
    /// boundary, or subtraction key.
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
    /// The algebra and current root of the endpoint's cell `i`, borrowed for
    /// the callback's duration: the run holds the reference, so a use that
    /// ends before the callback returns needs no handle of its own.
    #[inline(always)]
    pub fn root(self, i: usize) -> (AlgebraId, Function) {
        self.eval.root(self.cells[i])
    }

    /// The context the run evaluates in.
    #[inline(always)]
    pub fn context(self) -> &'a Context {
        self.eval.cx
    }
}

/// One referenced root in a bound algebra; the run owns the reference.
///
/// Cells hold raw tokens rather than [`Predicate`] handles so each delivered
/// refinement cell costs one reference count, not a context clone as well.
pub struct Cell {
    /// The algebra interpreting `function`.
    pub algebra: AlgebraId,
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
    /// A level keeps its refinement suspended while its children run, so each
    /// open level takes one buffer and returns it when the level closes.
    refine: Vec<Refine>,
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
            refine: Vec::new(),
        };
        eval.extend(slots);
        eval
    }

    /// Appends one cell per slot, each at its full universe.
    pub fn extend(&mut self, slots: &[Slot]) {
        self.cells.extend(slots.iter().map(|slot| {
            // Resolving checks the column has a backend; the full universe
            // holds no counted reference.
            self.cx.resolve(slot.algebra);
            Cell {
                algebra: slot.algebra,
                function: Function::TRUE,
                reads: 0,
            }
        }));
    }

    /// The cell's algebra and current root, borrowed.
    ///
    /// `Run` holds one reference per cell root for its whole lifetime
    /// (`narrow`/`set` retire the previous root only after adopting the new
    /// one), so a use that does not outlive this borrow needs no reference of
    /// its own.
    #[inline(always)]
    pub fn root(&self, cell: usize) -> (AlgebraId, Function) {
        let c = &self.cells[cell];
        (c.algebra, c.function)
    }

    /// A handle on the current root of `cell`, with its own reference.
    #[inline(always)]
    pub fn predicate(&self, cell: usize) -> Predicate {
        let c = &self.cells[cell];
        self.cx.predicate(c.algebra, c.function)
    }

    /// Restricts `cell` to `p` outright, outside every narrowing.
    pub fn set(&mut self, cell: usize, p: &Predicate) {
        let c = &mut self.cells[cell];
        assert_eq!(
            c.algebra,
            p.algebra(),
            "predicate uses an incompatible algebra"
        );
        self.cx.resolve(c.algebra).inc_ref(p.function());
        self.cx
            .resolve(c.algebra)
            .dec_ref(std::mem::replace(&mut c.function, p.function()));
    }

    /// Records one pending read of `cell`.
    #[inline(always)]
    pub fn push_read(&mut self, cell: usize) {
        self.reads.push(cell);
        self.cells[cell].reads += 1;
    }

    /// Retains the cells an operand reads until it runs, so earlier operands
    /// cannot finalize a cell that it still refines.
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
                self.cx.resolve(self.cells[other].algebra).dec_ref(next);
                continue;
            }
            self.assign(other, next, i);
        }
    }

    /// One owned image or preimage under the map between these cells.
    ///
    /// # Panics
    ///
    /// Panics if the endpoints use different algebras or the map is unsupported.
    #[inline(always)]
    pub fn transfer(&self, map: &MapSpec, x: usize, y: usize, image: bool) -> Function {
        let (p, q) = (&self.cells[x], &self.cells[y]);
        assert_eq!(
            p.algebra, q.algebra,
            "cross-backend transformations are unsupported"
        );
        let algebra = self.cx.resolve(p.algebra);
        if image {
            algebra.image(map, p.function, q.function)
        } else {
            algebra.preimage(map, p.function, q.function)
        }
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
        let algebra = self.cells[cell].algebra;
        let mut cursor = cx.resolve(algebra).members(self.cells[cell].function);
        let mut out = mode.sum();
        while let Some(value) = cursor.next() {
            let point = cx.resolve(algebra).set(value);
            let result = self.narrow(cell, point, mode, |eval, mode| {
                eval.finish(cell, observed, mode, then)
            })?;
            mode.push(&mut out, cell, result);
        }
        ControlFlow::Continue(mode.finish_sum(out))
    }

    /// Registers the equation `spec` from `x` to `y`, tightens both cells
    /// against each other once, and runs `then`.
    ///
    /// # Panics
    ///
    /// Panics if the equation closes a cycle, its endpoints use different
    /// algebras, or the backend does not support the map.
    #[inline(always)]
    pub fn equate<M: Mode>(
        &mut self,
        x: usize,
        y: usize,
        spec: &Arc<MapSpec>,
        mode: &mut M,
        then: &mut impl FnMut(&mut Self, &mut M) -> Flow<M>,
    ) -> Flow<M> {
        assert!(
            !self.connected(x, y),
            "equations between cells must not form a cycle"
        );
        self.push_read(x);
        self.push_read(y);
        self.equations.push(Equation {
            x,
            y,
            map: spec.clone(),
        });
        let p = self.transfer(spec, x, y, false);
        let result = self.narrow(x, p, mode, |eval, mode| then(eval, mode));
        self.equations.pop();
        self.pop_reads(2);
        result
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
        let Some(mark) = self.tighten(cell, p) else {
            return ControlFlow::Continue(mode.identity());
        };
        let result = then(self, mode);
        self.restore(mark);
        result
    }

    /// The undo mark after restricting `cell` to `p`, one owned reference in
    /// the cell's column, and tightening every cell an active equation
    /// connects to it.
    ///
    /// An empty `p` changes nothing and returns `None`.
    #[inline(always)]
    pub fn tighten(&mut self, cell: usize, p: Function) -> Option<usize> {
        if p.is_f() {
            self.cx.resolve(self.cells[cell].algebra).dec_ref(p);
            return None;
        }
        let mark = self.undo.len();
        if self.equations.is_empty() && p == self.cells[cell].function {
            self.cx.resolve(self.cells[cell].algebra).dec_ref(p);
        } else {
            self.assign(cell, p, usize::MAX);
        }
        Some(mark)
    }

    /// Restores the cell roots replaced since the undo mark `mark`.
    #[inline(always)]
    pub fn restore(&mut self, mark: usize) {
        for (cell, old) in self.undo.drain(mark..).rev() {
            let c = &mut self.cells[cell];
            self.cx
                .resolve(c.algebra)
                .dec_ref(std::mem::replace(&mut c.function, old));
        }
    }
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        // Unwinding leaves replaced roots owned by pending narrowings.
        for &(cell, old) in &self.undo {
            self.cx.resolve(self.cells[cell].algebra).dec_ref(old);
        }
        for c in &self.cells {
            self.cx.resolve(c.algebra).dec_ref(c.function);
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
    /// Its endpoint wiring supplies its entire meaning.
    /// [`positions`](crate::query::Query::positions) uses this property
    /// independently of the cells an execution would read.
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

/// Refines the cells in `cells` against `root`, running `then` on each
/// accepting branch.
///
/// Each level keeps a heap frame rather than a native one, so the stack depth
/// stays independent of the row width. A level suspends its refinement while
/// a child is explored and resumes it only when the child continues, so it
/// builds one cell per explored child and nothing for siblings after a break.
/// The invocation retains its native source root throughout the walk, so
/// every yielded child stays reachable.
fn walk<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
    root: Residual<'_>,
    cells: Range<usize>,
    eval: &mut Run<'_>,
    memo: &mut M::Memo,
    mode: &mut M,
    then: &mut K,
) -> Flow<M> {
    /// One level whose suffix is in progress.
    struct Frame<'a, M: Mode> {
        cell: usize,
        level: usize,
        state: M::Level,
        root: Residual<'a>,
        /// The suspended refinement and the sum over the explored cells, or
        /// `None` for a skipped level.
        sum: Option<(Refine, M::Sum)>,
        /// The undo mark of the cell being explored.
        mark: usize,
    }
    let mut frames: Vec<Frame<'_, M>> = Vec::new();
    let mut next = Some((root, 0));
    let mut result = None;
    loop {
        if let Some((root, level)) = next.take() {
            // An unobserved, unrestricted trailing suffix already has a
            // witness in this nonempty residual.
            if level == cells.len() {
                result = Some(then(eval, mode));
            } else {
                let cell = eval.maps[cells.start + level];
                let (state, cached) = mode.enter(memo, root.key(), level, eval);
                if let Some(flow) = cached {
                    result = Some(mode.exit(memo, level, state, eval, flow));
                } else if root.level() > level {
                    // A skipped source level admits the full field universe
                    // and can adopt the other source's algebra.
                    debug_assert_eq!(eval.reads.last(), Some(&cell));
                    eval.pop_reads(1);
                    frames.push(Frame {
                        cell,
                        level,
                        state,
                        root,
                        sum: None,
                        mark: 0,
                    });
                    next = Some((root, level + 1));
                    continue;
                } else {
                    let mut scratch = eval.refine.pop().unwrap_or_default();
                    let Cell {
                        algebra, function, ..
                    } = eval.cells[cell];
                    root.refine(algebra, function, &mut scratch);
                    frames.push(Frame {
                        cell,
                        level,
                        state,
                        root,
                        sum: Some((scratch, mode.sum())),
                        mark: 0,
                    });
                }
            }
        }
        let Some(frame) = frames.last_mut() else {
            return result.expect("an empty walk yields its continuation's result");
        };
        if let Some(flow) = result.take() {
            eval.push_read(frame.cell);
            match (&mut frame.sum, flow) {
                (Some((_, sum)), ControlFlow::Continue(value)) => {
                    eval.restore(frame.mark);
                    mode.push(sum, frame.cell, value);
                }
                (sum, flow) => {
                    if sum.is_some() {
                        eval.restore(frame.mark);
                    }
                    let f = frames.pop().unwrap();
                    eval.refine.extend(f.sum.map(|(scratch, _)| scratch));
                    result = Some(mode.exit(memo, f.level, f.state, eval, flow));
                    continue;
                }
            }
        }
        let (scratch, sum) = frame.sum.as_mut().unwrap();
        let algebra = eval.cells[frame.cell].algebra;
        while let Some(child) = frame.root.next(algebra, scratch) {
            let p = eval.cx.resolve(algebra).cell(scratch);
            if let Some(mark) = eval.tighten(frame.cell, p) {
                frame.mark = mark;
                debug_assert_eq!(eval.reads.last(), Some(&frame.cell));
                eval.pop_reads(1);
                next = Some((child, frame.level + 1));
                break;
            }
            let empty = mode.identity();
            mode.push(sum, frame.cell, empty);
        }
        if next.is_none() {
            let f = frames.pop().unwrap();
            let (scratch, sum) = f.sum.unwrap();
            eval.refine.push(scratch);
            let out = mode.finish_sum(sum);
            result = Some(mode.exit(memo, f.level, f.state, eval, ControlFlow::Continue(out)));
        }
    }
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
        if self.is_empty() {
            return ControlFlow::Continue(mode.identity());
        }
        let arity = self.index().arity();
        debug_assert_eq!(map.len(), arity);
        if map
            .clone()
            .any(|i| eval.cells[eval.maps[i]].function.is_f())
        {
            return ControlFlow::Continue(mode.identity());
        }
        for i in map.clone().rev() {
            eval.push_read(eval.maps[i]);
        }
        let stop = (0..arity)
            .rposition(|i| {
                let cell = eval.maps[map.start + i];
                // A repeated cell stops the scan at its last occurrence.
                eval.cells[cell].reads != 1 || !eval.cells[cell].function.is_t() || mode.needs(cell)
            })
            .map_or(0, |i| i + 1);
        let result = walk(
            self.residual(),
            map.start..map.start + stop,
            eval,
            memo,
            mode,
            then,
        );
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
    /// The state one open source level carries to its exit.
    type Level;
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
    /// Opens one source level: finalizes unread outputs, then returns the
    /// state for [`exit`](Self::exit) and any reused suffix.
    fn enter(
        &mut self,
        memo: &mut Self::Memo,
        root: usize,
        level: usize,
        eval: &Run<'_>,
    ) -> (Self::Level, Option<Flow<Self>>);
    /// Closes one source level opened with `state` over its suffix `result`.
    fn exit(
        &mut self,
        memo: &mut Self::Memo,
        level: usize,
        state: Self::Level,
        eval: &Run<'_>,
        result: Flow<Self>,
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
    type Level = ();
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
    fn enter(&mut self, _: &mut (), _: usize, _: usize, _: &Run<'_>) -> ((), Option<Flow<Self>>) {
        ((), None)
    }
    #[inline(always)]
    fn exit(&mut self, _: &mut (), _: usize, _: (), _: &Run<'_>, result: Flow<Self>) -> Flow<Self> {
        result
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
    /// Each attached root, its scalar predicate, and its continuation.
    cases: Vec<(IndexRoot, Predicate, IndexRoot)>,
    /// The coordinate shared by the cases.
    level: usize,
    /// Pairwise union of branches with other shapes.
    rest: BalancedUnion,
}

/// The root holding `child` where coordinate `level` satisfies `p`.
///
/// Boolean multiplication handles authored field reordering. The ordinary
/// increasing-level case attaches directly to the shared child,
/// retaining the existing index's partition/child algebra.
fn attach(index: &Index, level: usize, p: &Predicate, child: &IndexRoot) -> IndexRoot {
    if child.residual().level() > level {
        return index.cases(level, &index.empty(), &[(p, child)]);
    }
    let cell = index.cases(level, &index.empty(), &[(p, &index.full())]);
    cell.apply(BoolOp::And, child)
}

/// Collection mode: materializes accepting branches into one output index,
/// memoizing completed suffixes.
pub struct Collect {
    /// The physical index for every collected branch.
    output: Index,
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
    /// A collector into the output index whose level `i` is filled by cell
    /// `positions[i]`, over `cells` invocation cells.
    ///
    /// A `remaining` universe turns the collector into an exclusion probe.
    pub fn new(
        output: Index,
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
        *remaining = remaining.apply(BoolOp::AndNot, &root);
        if remaining.is_empty() {
            return ControlFlow::Break(remaining.clone());
        }
        ControlFlow::Continue(suffix)
    }

    /// Attaches the output coordinates of `cell` above `child`.
    ///
    /// The cell is pending or retired, so it holds at least one output
    /// coordinate.
    fn attach(&self, cell: usize, eval: &Run<'_>, child: IndexRoot) -> IndexRoot {
        let p = eval.predicate(cell);
        match self.levels[cell].as_slice() {
            &[level] => attach(&self.output, level, &p, &child),
            levels => {
                // Stored coordinates are independent. Each point on the
                // diagonal fills every occurrence of its shared cell.
                let cx = self.output.context();
                let mut cursor = cx.resolve(p.algebra()).members(p.function());
                let mut diagonal = BalancedUnion::default();
                while let Some(value) = cursor.next() {
                    let point = cx.set(p.algebra(), value);
                    let mut branch = child.clone();
                    for &level in levels.iter().rev() {
                        branch = attach(&self.output, level, &point, &branch);
                    }
                    diagonal.push(branch);
                }
                diagonal
                    .drain()
                    .reduce(|a, b| a.apply(BoolOp::Or, &b))
                    .unwrap_or_else(|| self.output.empty())
            }
        }
    }
}

impl Mode for Collect {
    type Output = IndexRoot;
    type Break = IndexRoot;
    type Memo = Memo;
    type Sum = Sum;
    /// The first retirement of the level and the key of a suffix to store.
    type Level = (usize, Option<Box<[usize]>>);
    #[inline(always)]
    fn sum(&mut self) -> Self::Sum {
        self.sums.pop().unwrap_or_default()
    }
    #[inline(always)]
    fn push(&mut self, sum: &mut Self::Sum, cell: usize, value: IndexRoot) {
        if value.is_empty() {
            return;
        }
        // One output coordinate of `cell` is the only level of the output
        // this branch can have attached outermost, and its region lies within
        // this refinement cell, so the cases collected here are disjoint.
        if let Some(&[level]) = self.levels.get(cell).map(Vec::as_slice)
            && let Some((p, child)) = value.attachment(level)
        {
            sum.level = level;
            sum.cases.push((value, p, child));
            return;
        }
        sum.rest.push(value);
    }
    #[inline(always)]
    fn finish_sum(&mut self, mut sum: Self::Sum) -> IndexRoot {
        let out = match sum.cases.len() {
            0 => None,
            // A lone case already owns exactly the node we would rebuild.
            1 => Some(sum.cases.pop().unwrap().0),
            _ => {
                let cases: Vec<_> = sum.cases.iter().map(|(_, p, child)| (p, child)).collect();
                let out = self.output.cases(sum.level, &self.output.empty(), &cases);
                // Backends may give equal partitions distinct tokens. Compare
                // the k-ary union with a pairwise union by their contents.
                #[cfg(debug_assertions)]
                {
                    let chain = sum
                        .cases
                        .iter()
                        .fold(self.output.empty(), |a, (b, _, _)| a.apply(BoolOp::Or, b));
                    assert!(
                        chain.apply(BoolOp::Xor, &out).is_empty(),
                        "the k-ary union differs from the Or chain"
                    );
                }
                sum.cases.clear();
                Some(out)
            }
        };
        let out = out
            .into_iter()
            .chain(sum.rest.drain())
            .reduce(|a, b| a.apply(BoolOp::Or, &b))
            .unwrap_or_else(|| self.output.empty());
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
        self.output.empty()
    }

    #[inline(always)]
    fn add(&mut self, out: &mut IndexRoot, value: IndexRoot) {
        *out = out.apply(BoolOp::Or, &value);
    }

    /// Every output cell that no source has finalized is attached to this
    /// accepting branch.
    #[inline(always)]
    fn done(&mut self, eval: &Run<'_>) -> Flow<Self> {
        let mut root = self.output.full();
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
    fn enter(
        &mut self,
        memo: &mut Memo,
        root: usize,
        level: usize,
        eval: &Run<'_>,
    ) -> (Self::Level, Option<Flow<Self>>) {
        let start = self.retired.len();
        for cell in 0..self.pending.len() {
            if self.pending[cell] && eval.cells[cell].reads == 0 {
                self.pending[cell] = false;
                self.coordinates -= self.levels[cell].len();
                self.retired.push(cell);
            }
        }
        // Only suffixes with at least two coordinates still to fill are keyed,
        // looked up, and retained. A suffix with one pending coordinate can
        // recur, because its key omits retired cells and upstream concrete
        // values, so composition chains recompute it once per upstream branch.
        // The gate trades that recomputation for cheaper keys and smaller
        // caches.
        if self.coordinates < 2 {
            return ((start, None), None);
        }
        if memo.len() <= level {
            memo.resize_with(level + 1, MemoLevel::default);
        }
        self.key.clear();
        self.key.push(self.branches.len());
        self.key
            .extend(self.branches.iter().map(|&right| usize::from(right)));
        self.key.push(root);
        self.key.extend(
            eval.reads
                .iter()
                .map(|&cell| eval.cells[cell].function.index()),
        );
        if let Some(suffix) = memo[level].entries.get(self.key.as_slice()).cloned() {
            return ((start, None), Some(self.observe(suffix, eval)));
        }
        let repeat = !memo[level]
            .seen
            .insert(structural_hash(self.key.as_slice()));
        // Deeper levels reuse the scratch, so this key is fixed before
        // descending.
        ((start, repeat.then(|| self.key.as_slice().into())), None)
    }

    #[inline(always)]
    fn exit(
        &mut self,
        memo: &mut Memo,
        level: usize,
        (start, key): Self::Level,
        eval: &Run<'_>,
        mut result: Flow<Self>,
    ) -> Flow<Self> {
        if let (Some(key), ControlFlow::Continue(suffix)) = (key, &result) {
            memo[level]
                .live
                .extend(eval.reads.iter().map(|&cell| eval.predicate(cell)));
            memo[level].entries.insert(key, suffix.clone());
        }
        while self.retired.len() > start {
            let cell = self.retired.pop().unwrap();
            self.pending[cell] = true;
            self.coordinates += self.levels[cell].len();
            result = result.map_continue(|root| {
                if root.is_empty() {
                    root
                } else {
                    self.attach(cell, eval, root)
                }
            });
        }
        result
    }
}
