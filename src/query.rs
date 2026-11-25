//! Statically composed relational queries with runtime nominal endpoints.
//!
//! Each operator connects only its operands' endpoint cells. Hidden cells
//! belong to that operator's invocation; there is no whole-query binding pass
//! or source schedule. Equalities are structural wiring, while predicates are
//! narrowed and restored during CPS execution.
//!
//! A declaration binds its own logical sort to physical columns and becomes
//! an ordinary source-free projection. Combining it with a stored relation
//! adopts that relation's columns through structural equality. Maps and
//! contextual identity receive an expected row; declarations and existing
//! queries remain closed. Every query carries the context that owns its
//! columns, so constructors take the context explicitly.

use std::ops::{ControlFlow, Range};
use std::sync::Arc;

use crate::ast::{FuncDecl, Sort};
use crate::backend::MapSpec;
use crate::context::{AlgebraId, Context, Predicate};
use crate::index::{Index, IndexRoot};
use crate::map::Transform;
use crate::rel::{Rel, Universe};
use crate::row::{Domain, Row};
use crate::run::{Binding, Collect, Exec, Flow, Mode, Run, Slot, Visit};
use crate::util::count::Count;

/// A binary relational expression with runtime domain and range shapes.
///
/// Concrete Rust types carry the operator topology. Sources own their
/// immutable roots; borrowed subqueries share structure, while owned clones
/// recursively clone their operand handles.
///
/// A query can be shared across scoped threads when its node type is `Sync`.
/// Every terminal invocation owns its execution state, including nested calls
/// on the same thread. Source roots are snapshots taken at construction;
/// coordinating snapshots of several relations is up to the caller.
#[derive(Clone)]
pub struct Query<A> {
    /// The context owning every boundary column.
    pub cx: Context,
    /// The operator node supplying execution.
    pub node: A,
    /// The boundary cells, one per distinct endpoint cell.
    pub slots: Arc<[Slot]>,
    /// The domain and range endpoints, as indices into `slots`.
    pub ends: Arc<[usize]>,
    /// The declared universe used by collection and complement.
    ///
    /// Binary queries retain both endpoint coordinates even when execution
    /// aliases them. Combining operands preserves any binary universe, while
    /// restriction and subtraction retain the left universe. Identity, keys,
    /// and values explicitly select the set universe.
    pub universe: Universe,
    /// The domain row.
    pub domain: Row,
    /// The range row.
    pub range: Row,
}

/// Operand endpoints mapped to the enclosing boundary followed by
/// invocation-local cells.
///
/// This describes a single combinator's connections, never an executable list
/// of sources.
#[derive(Clone)]
pub struct Links {
    /// The invocation-local cells appended beyond the boundary.
    pub local: Box<[Slot]>,
    /// Each operand endpoint's cell: a boundary index, or a local index
    /// offset by the boundary length.
    pub map: Box<[usize]>,
    /// Identity equations between operand cells, active while the operand
    /// runs.
    pub equal: Box<[(usize, usize, Arc<MapSpec>)]>,
}

/// Temporary endpoint equivalences used only while constructing one
/// combinator.
fn connect(
    cx: &Context,
    slots: impl IntoIterator<Item = Slot>,
    equal: impl IntoIterator<Item = (usize, usize)>,
    output: &[usize],
) -> (Arc<[Slot]>, Arc<[usize]>, Links) {
    let mut slots: Vec<_> = slots.into_iter().collect();
    let mut map: Vec<_> = (0..slots.len()).collect();
    for (a, b) in equal {
        let (a, b) = (map[a].min(map[b]), map[a].max(map[b]));
        if a == b {
            continue;
        }
        assert_eq!(
            cx.resolve(slots[a].algebra).sort(),
            cx.resolve(slots[b].algebra).sort(),
            "incompatible variable sorts"
        );
        if slots[a].fixed && slots[b].fixed {
            // Layout sharing preserves structural equality but cannot remove
            // this compatibility check.
            assert_eq!(
                slots[a].algebra, slots[b].algebra,
                "query variable uses incompatible algebras"
            );
        }
        slots[a].read |= slots[b].read;
        if !slots[a].fixed && slots[b].fixed {
            slots[a].algebra = slots[b].algebra;
            slots[a].fixed = true;
        }
        for i in &mut map {
            if *i == b {
                *i = a;
            }
        }
    }
    let mut order = Vec::new();
    let mut labels = vec![usize::MAX; slots.len()];
    for &i in output.iter().chain(map.iter()) {
        let i = map[i];
        if labels[i] == usize::MAX {
            labels[i] = order.len();
            order.push(i);
        }
    }
    let ends: Arc<[usize]> = output.iter().map(|&i| labels[map[i]]).collect();
    let n = ends.iter().copied().max().map_or(0, |i| i + 1);
    let boundary = order[..n].iter().map(|&i| slots[i]).collect();
    let local = order[n..].iter().map(|&i| slots[i]).collect();
    let map = map.iter().map(|&i| labels[i]).collect();
    (
        boundary,
        ends,
        Links {
            local,
            map,
            equal: Box::new([]),
        },
    )
}

/// A join of two children sharing one source-order CPS flow.
///
/// Their connections distinguish composition, shared-domain pairing, and
/// restriction.
#[derive(Clone)]
pub struct Join<A, B> {
    /// The left operand.
    pub a: A,
    /// The right operand.
    pub b: B,
    /// The right operand's boundary slots, read while the left operand runs.
    pub b_slots: Arc<[Slot]>,
    /// The operands' connections to this operator's boundary.
    pub links: Links,
}

/// Left-first lazy union.
///
/// Each alternative owns its constraints and source state.
#[derive(Clone)]
pub struct Union<A, B> {
    /// The left alternative.
    pub a: A,
    /// The right alternative.
    pub b: B,
    /// Each alternative's connections to the shared boundary.
    pub branches: [Links; 2],
}

impl Links {
    /// Runs an operand connection with local cells and equalities restored on
    /// either exit.
    #[inline(always)]
    fn run<M: Mode>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        mode: &mut M,
        mut then: impl FnMut(Range<usize>, &mut Run<'_>, &mut M) -> Flow<M>,
    ) -> Flow<M> {
        fn equal<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
            pairs: &[(usize, usize, Arc<MapSpec>)],
            start: usize,
            eval: &mut Run<'_>,
            mode: &mut M,
            then: &mut K,
        ) -> Flow<M> {
            let [(a, b, spec), rest @ ..] = pairs else {
                return then(eval, mode);
            };
            let (x, y) = (eval.maps[start + a], eval.maps[start + b]);
            if eval.aliased(x, y) {
                return equal(rest, start, eval, mode, then);
            }
            eval.equate(x, y, spec, mode, &mut |eval, mode| {
                equal(rest, start, eval, mode, then)
            })
        }
        let base = eval.cells.len();
        eval.extend(&self.local);
        // Pushes this combinator's operand cells: enclosing boundary cells by
        // the caller's map, then the invocation-local cells appended just
        // above `base`.
        let start = eval.maps.len();
        for &i in &self.map {
            let cell = if i < map.len() {
                eval.maps[map.start + i]
            } else {
                base + i - map.len()
            };
            eval.maps.push(cell);
        }
        let cells = start..eval.maps.len();
        let result = equal(&self.equal, map.start, eval, mode, &mut |eval, mode| {
            then(cells.clone(), eval, mode)
        });
        eval.maps.truncate(start);
        eval.cells.truncate(base);
        result
    }
}

impl<A: Exec, B: Exec> Exec for Union<A, B> {
    type State<M: Mode> = (A::State<M>, B::State<M>);

    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        state: &mut Self::State<M>,
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        let mut out = mode.branch(false, eval, |eval, mode| {
            self.branches[0].run(map.clone(), eval, mode, |cells, eval, mode| {
                self.a.run(cells, eval, &mut state.0, mode, then)
            })
        })?;
        let right = mode.branch(true, eval, |eval, mode| {
            self.branches[1].run(map, eval, mode, |cells, eval, mode| {
                self.b.run(cells, eval, &mut state.1, mode, then)
            })
        })?;
        mode.add(&mut out, right);
        ControlFlow::Continue(out)
    }
}

/// The label of `x` in `v`, appending it when new.
fn label<T: PartialEq>(v: &mut Vec<T>, x: T) -> usize {
    v.iter().position(|y| *y == x).unwrap_or_else(|| {
        v.push(x);
        v.len() - 1
    })
}

/// Subtraction on selected keys, with symbolic exclusion scoped to each left
/// branch.
#[derive(Clone)]
pub struct Diff<A, B> {
    /// The left operand.
    pub a: A,
    /// The right operand, run as an exclusion probe.
    pub b: B,
    /// The left endpoints forming the subtraction key.
    pub keys: Box<[usize]>,
    /// The right endpoints matched against `keys`.
    pub b_keys: Box<[usize]>,
    /// The right operand's boundary slots.
    pub b_slots: Arc<[Slot]>,
}

impl<A: Exec, B: Exec> Exec for Diff<A, B> {
    type State<M: Mode> = (A::State<M>, B::State<Collect>);

    #[inline(always)]
    fn run<M: Mode, K: FnMut(&mut Run<'_>, &mut M) -> Flow<M>>(
        &self,
        map: Range<usize>,
        eval: &mut Run<'_>,
        state: &mut Self::State<M>,
        mode: &mut M,
        then: &mut K,
    ) -> Flow<M> {
        // Coordinates name unique left cells: pair-keying an identity adds no
        // diagonal.
        let mut keys = Vec::new();
        let positions: Vec<_> = self
            .keys
            .iter()
            .map(|&i| label(&mut keys, eval.maps[map.start + i]))
            .collect();
        let columns: Vec<_> = keys.iter().map(|&i| eval.cells[i].algebra).collect();
        let output = Index::new(eval.cx, &columns);
        let (slots, ends, links) = connect(
            eval.cx,
            columns
                .iter()
                .map(|&algebra| Slot {
                    algebra,
                    read: false,
                    fixed: true,
                })
                .chain(self.b_slots.iter().copied()),
            positions
                .iter()
                .zip(self.b_keys.iter())
                .map(|(&i, &j)| (i, keys.len() + j)),
            &(0..keys.len()).collect::<Vec<_>>(),
        );
        let mut probe = Run::new(eval.cx, &slots);
        probe.extend(&links.local);
        probe.maps = links.map.into_vec();
        let right = keys.len()..probe.maps.len();
        let start = eval.maps.len();
        eval.maps.extend_from_slice(&keys);
        let reads = eval.reads.len();
        for &cell in &keys {
            eval.push_read(cell);
        }
        let result = self
            .a
            .run(map, eval, &mut state.0, mode, &mut |eval, mode| {
                eval.remove_reads(reads, keys.len());
                let result = eval.finish(eval.cells.len(), &keys, mode, &mut |eval, mode| {
                    let mut bound = output.full();
                    for (i, &cell) in keys.iter().enumerate().rev() {
                        bound =
                            output.cases(i, &output.empty(), &[(&eval.predicate(cell), &bound)]);
                    }
                    // A residual may be emitted only once its absence from
                    // `b` is established, so the residual waits for the probe
                    // before downstream execution, even for `is_empty`.
                    let mut collect = Collect::new(
                        output.clone(),
                        &ends,
                        probe.cells.len(),
                        Some(bound.clone()),
                    );
                    let n = probe.push_reads(&self.b_slots, right.clone());
                    // A local break means the residual is empty; it must not
                    // stop the outer query.
                    let _ = bound.run(
                        0..keys.len(),
                        &mut probe,
                        &mut Default::default(),
                        &mut collect,
                        &mut |probe, collect| {
                            probe.pop_reads(n);
                            let result = self.b.run(
                                right.clone(),
                                probe,
                                &mut state.1,
                                collect,
                                &mut |probe, collect| {
                                    probe.finish(
                                        probe.cells.len(),
                                        &ends,
                                        collect,
                                        &mut |probe, collect| collect.done(probe),
                                    )
                                },
                            );
                            probe.push_reads(&self.b_slots, right.clone());
                            result
                        },
                    );
                    probe.pop_reads(n);
                    collect.remaining.take().unwrap().run(
                        start..start + keys.len(),
                        eval,
                        &mut Default::default(),
                        mode,
                        then,
                    )
                });
                for &cell in &keys {
                    eval.push_read(cell);
                }
                eval.reads[reads..].rotate_right(keys.len());
                result
            });
        eval.pop_reads(keys.len());
        eval.maps.truncate(start);
        result
    }
}

impl<A: Exec, B: Exec> Exec for Join<A, B> {
    type State<M: Mode> = (A::State<M>, B::State<M>);

    #[inline(always)]
    fn is_projection(&self) -> bool {
        self.a.is_projection() && self.b.is_projection()
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
        self.links.run(map, eval, mode, |cells, eval, mode| {
            let split = cells.end - self.b_slots.len();
            let (left, right) = (cells.start..split, split..cells.end);
            let reads = eval.reads.len();
            let n = eval.push_reads(&self.b_slots, right.clone());
            let result = self
                .a
                .run(left, eval, &mut state.0, mode, &mut |eval, mode| {
                    eval.remove_reads(reads, n);
                    let result = self.b.run(right.clone(), eval, &mut state.1, mode, then);
                    eval.push_reads(&self.b_slots, right.clone());
                    eval.reads[reads..].rotate_right(n);
                    result
                });
            eval.pop_reads(n);
            result
        })
    }
}

impl<A> Query<A> {
    /// Coerces each endpoint between a record and its declared argument sort,
    /// without execution.
    ///
    /// Cells, aliases, physical columns, and the
    /// domain/range boundary remain unchanged. Distinct endpoint sorts promote
    /// a set to a binary universe. Binary universes stay binary.
    ///
    /// # Panics
    ///
    /// Panics if either replacement sort belongs to another context or does not
    /// match its endpoint as a record or declared argument sort.
    #[inline(always)]
    pub fn coerce(mut self, domain: Sort, range: Sort) -> Self {
        self.domain.coerce(&self.cx, domain);
        self.range.coerce(&self.cx, range);
        if domain != range {
            self.universe = Universe::Binary;
        }
        self
    }

    /// The output cells and endpoint map of a materialized result: a
    /// declared set keeps its row universe once, and a binary relation keeps
    /// the domain-range product, including when endpoint cells alias.
    pub fn output(&self) -> (&[usize], impl Iterator<Item = usize>) {
        let n = if self.universe == Universe::Set {
            self.domain.sort().scalar_len()
        } else {
            0
        };
        let positions = &self.ends[n..];
        (positions, (0..n).chain(0..positions.len()))
    }
}

/// The combined universe, retaining every declared binary product.
fn universe(a: Universe, b: Universe, domain: Sort, range: Sort) -> Universe {
    if a == Universe::Set && b == Universe::Set && domain == range {
        Universe::Set
    } else {
        Universe::Binary
    }
}

fn make<A>(
    cx: &Context,
    node: A,
    domain: Sort,
    range: Sort,
    universe: Universe,
    slots: Arc<[Slot]>,
    ends: Arc<[usize]>,
) -> Query<A> {
    let layout: Arc<[AlgebraId]> = ends.iter().map(|&i| slots[i].algebra).collect();
    Query {
        cx: cx.clone(),
        node,
        universe,
        slots,
        ends,
        domain: Row::new(cx, domain, layout.clone(), 0),
        range: Row::new(cx, range, layout, domain.scalar_len()),
    }
}

/// Connects two operands' boundaries.
///
/// The endpoints of `a` and `b` at the positions in `equal` are paired as
/// structurally identical cells, and those at the positions in `output`,
/// `a`'s first, are exposed as the domain and range cells.
fn binary<A, B>(
    a: Query<A>,
    b: Query<B>,
    equal: (Range<usize>, Range<usize>),
    output: (Range<usize>, Range<usize>),
    domain: Sort,
    range: Sort,
    universe: Universe,
) -> Query<Join<A, B>> {
    assert!(a.cx == b.cx, "query operands belong to different contexts");
    let n = a.slots.len();
    let equal = a.ends[equal.0]
        .iter()
        .zip(&b.ends[equal.1])
        .map(|(&i, &j)| (i, n + j));
    let output: Vec<_> = a.ends[output.0]
        .iter()
        .copied()
        .chain(b.ends[output.1].iter().map(|&i| n + i))
        .collect();
    let (slots, ends, links) = connect(
        &a.cx,
        a.slots.iter().chain(b.slots.iter()).copied(),
        equal,
        &output,
    );
    make(
        &a.cx,
        Join {
            a: a.node,
            b: b.node,
            b_slots: b.slots,
            links,
        },
        domain,
        range,
        universe,
        slots,
        ends,
    )
}

/// Subtracts matching keys without exporting the probe's aliases into the
/// left operand.
#[inline(always)]
fn subtract<A, B>(
    a: Query<A>,
    b: Query<B>,
    keys: Box<[usize]>,
    b_keys: Box<[usize]>,
) -> Query<Diff<A, B>> {
    assert!(a.cx == b.cx, "query operands belong to different contexts");
    let (slots, ends, _) = connect(
        &a.cx,
        a.slots.iter().chain(b.slots.iter()).copied(),
        keys.iter()
            .zip(b_keys.iter())
            .map(|(&i, &j)| (i, a.slots.len() + j)),
        &(0..a.slots.len()).collect::<Vec<_>>(),
    );
    // Adopt/check storage without exporting the right operand's equalities.
    let slots = a
        .slots
        .iter()
        .enumerate()
        .map(|(i, slot)| Slot {
            algebra: slots[ends[i]].algebra,
            fixed: slots[ends[i]].fixed,
            read: slot.read || keys.contains(&i),
        })
        .collect();
    make(
        &a.cx,
        Diff {
            a: a.node,
            b: b.node,
            keys,
            b_keys,
            b_slots: b.slots,
        },
        a.domain.sort(),
        a.range.sort(),
        a.universe,
        slots,
        a.ends,
    )
}

impl FuncDecl {
    /// Applies this declaration to one query's range.
    ///
    /// Constructors consume
    /// one argument product.
    ///
    /// # Panics
    ///
    /// Panics if the declaration and query belong to different contexts, their
    /// matched endpoint scalar sorts differ, or fixed columns require incompatible
    /// storage.
    #[inline(always)]
    pub fn apply<A: IntoQuery>(&self, query: A) -> Query<Join<A::Node, ()>> {
        query.select(*self)
    }
}

/// A right-hand operand receives the consuming query's context, the expected
/// domain, and its actual physical columns.
///
/// Closed queries keep their signature; contextual builders resolve against
/// this input row.
pub trait Operand: Sized {
    /// The operator node of the resolved query.
    type Node: Exec;
    /// The closed query for this operand, given the consuming query's context
    /// and `input` row.
    fn resolve(self, cx: &Context, input: &Row) -> Query<Self::Node>;
}

impl<T: IntoQuery> Operand for T {
    type Node = T::Node;
    #[inline(always)]
    fn resolve(self, _cx: &Context, _input: &Row) -> Query<Self::Node> {
        self.into_query()
    }
}

/// A declaration is a source-free projection of its bound declared row,
/// closed over the consuming query's context; see [`Query::decl`].
impl Operand for FuncDecl {
    type Node = ();
    #[inline(always)]
    fn resolve(self, cx: &Context, _input: &Row) -> Query<()> {
        Query::decl(cx, self)
    }
}

/// Contextual identity, resolved from the consuming query's row and physical
/// columns.
#[derive(Clone, Copy)]
pub struct Ident;

/// The contextual identity operand; see [`Ident`].
#[inline(always)]
pub const fn id() -> Ident {
    Ident
}

impl Operand for Ident {
    type Node = ();
    #[inline(always)]
    fn resolve(self, cx: &Context, input: &Row) -> Query<()> {
        project(cx, input, 0..input.sort().scalar_len(), input.clone())
    }
}

impl Ident {
    /// Resolves identity from the right operand, preserving identity-first pair
    /// order.
    #[inline(always)]
    pub fn and<A: IntoQuery>(self, a: A) -> Query<Join<(), A::Node>> {
        let a = a.into_query();
        self.resolve(&a.cx, &a.domain).and(a)
    }
}

/// A closed binary relation.
///
/// Stored sources own their roots; reused subqueries can be borrowed.
pub trait IntoQuery: Sized {
    /// The operator node type.
    type Node: Exec;
    /// The closed query.
    fn into_query(self) -> Query<Self::Node>;

    /// Lazy set union: visit the left alternative before the right, stopping
    /// on a break.
    ///
    /// Overlapping branches may be delivered twice; explicit
    /// collection unions their results.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, either endpoint has
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    fn union<B: Operand>(self, b: B) -> Query<Union<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.domain);
        assert!(a.cx == b.cx, "query operands belong to different contexts");
        assert!(
            a.cx.sorts().coercible(a.domain.sort(), b.domain.sort())
                && a.cx.sorts().coercible(a.range.sort(), b.range.sort()),
            "union requires compatible query endpoints"
        );
        let (columns, positions, _) = connect(
            &a.cx,
            a.slots.iter().chain(b.slots.iter()).copied(),
            a.ends
                .iter()
                .zip(b.ends.iter())
                .map(|(&i, &j)| (i, a.slots.len() + j)),
            &a.ends,
        );
        let mut pairs = Vec::new();
        let ends: Arc<[usize]> = a
            .ends
            .iter()
            .zip(b.ends.iter())
            .map(|(&i, &j)| label(&mut pairs, (i, j)))
            .collect();
        let mut slots = Vec::new();
        for (i, &cell) in ends.iter().enumerate() {
            if cell == slots.len() {
                slots.push(Slot {
                    read: true,
                    ..columns[positions[i]]
                });
            }
        }
        let branches = [(&a.slots, &a.ends), (&b.slots, &b.ends)].map(|(input, output)| {
            let mut map = vec![usize::MAX; input.len()];
            let mut equal = Vec::new();
            for (&i, &j) in output.iter().zip(ends.iter()) {
                if map[i] == usize::MAX {
                    map[i] = j;
                } else if map[i] != j && !equal.iter().any(|&(x, y, _)| (x, y) == (map[i], j)) {
                    equal.push((
                        map[i],
                        j,
                        Arc::new(MapSpec::identity(
                            a.cx.resolve(slots[j].algebra).sort().words(),
                        )),
                    ));
                }
            }
            // A query's slots are exactly its distinct endpoint cells, so
            // every branch cell is named by that branch's output and none of
            // them stays local to the alternative.
            Links {
                local: Box::new([]),
                map: map.into(),
                equal: equal.into(),
            }
        });
        make(
            &a.cx,
            Union {
                a: a.node,
                b: b.node,
                branches,
            },
            a.domain.sort(),
            a.range.sort(),
            universe(a.universe, b.universe, a.domain.sort(), a.range.sort()),
            slots.into(),
            ends,
        )
    }

    /// The pairs held by both operands.
    ///
    /// Both complete endpoints must match.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, either endpoint has
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    fn intersection<B: Operand>(self, b: B) -> Query<Join<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.domain);
        assert!(
            a.cx.sorts().coercible(a.domain.sort(), b.domain.sort())
                && a.cx.sorts().coercible(a.range.sort(), b.range.sort()),
            "intersection requires compatible query endpoints"
        );
        let (m, k) = (a.ends.len(), b.ends.len());
        let (domain, range) = (a.domain.sort(), a.range.sort());
        let u = universe(a.universe, b.universe, domain, range);
        binary(a, b, (0..m, 0..k), (0..m, 0..0), domain, range, u)
    }

    /// The pairs this relation does not hold, relative to its columns' exact
    /// logical universes, as for [`Rel::not`].
    ///
    /// For example, the complement of the one-bit pair set `{(0,0)}` contains
    /// `(0,1)`, `(1,0)`, and `(1,1)`, including when the original pair came
    /// from duplicating a projection. A complemented subidentity can exclude
    /// keys through an explicit projection and `with`. Complementing a general
    /// binary relation does not implement keyed anti-join.
    #[inline(always)]
    fn complement(self) -> Query<Diff<(), Self::Node>> {
        let a = self.into_query();
        let (positions, ends) = a.output();
        let ends: Arc<[usize]> = ends.collect();
        let slots = positions
            .iter()
            .map(|&i| Slot {
                read: false,
                fixed: false,
                ..a.slots[i]
            })
            .collect();
        let keys = ends.as_ref().into();
        let b_keys = a.ends.as_ref().into();
        let universe = make(
            &a.cx,
            (),
            a.domain.sort(),
            a.range.sort(),
            a.universe,
            slots,
            ends,
        );
        subtract(universe, a, keys, b_keys)
    }

    /// The converse, which swaps exposed endpoints while preserving the node
    /// and its source traversal order.
    #[inline(always)]
    fn inv(self) -> Query<Self::Node> {
        let a = self.into_query();
        let ends = a.ends[a.domain.sort().scalar_len()..]
            .iter()
            .chain(&a.ends[..a.domain.sort().scalar_len()])
            .copied()
            .collect();
        Query {
            cx: a.cx,
            node: a.node,
            universe: a.universe,
            slots: a.slots,
            ends,
            domain: a.range,
            range: a.domain,
        }
    }

    /// Composition: A:D->M and B:M->R produce D->R.
    ///
    /// Direct record/argument
    /// conversions are checked after accessor resolution, before the
    /// endpoints are connected.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, the left range and
    /// right domain have incompatible scalar sorts, or matched fixed columns
    /// require incompatible storage.
    #[inline(always)]
    fn select<B: Operand>(self, b: B) -> Query<Join<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.range);
        assert!(
            a.cx.sorts().coercible(a.range.sort(), b.domain.sort()),
            "composition requires compatible whole endpoint sorts"
        );
        let (m, k) = (a.ends.len(), b.ends.len());
        let (d, e) = (a.domain.sort().scalar_len(), b.domain.sort().scalar_len());
        let (domain, range) = (a.domain.sort(), b.range.sort());
        let u = universe(a.universe, b.universe, domain, range);
        binary(a, b, (d..m, 0..e), (0..d, e..k), domain, range, u)
    }

    /// Keyed subtraction: retain A:D->R exactly where B:D->S has no witness.
    ///
    /// Each left branch excludes right-hand keys before emitting its
    /// residual. The right range is hidden; cached suffixes can materialize
    /// all right-hand key coverage. Plain set difference on complete pairs
    /// is [`difference`](Self::difference).
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, their domains have
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    fn minus<B: Operand>(self, b: B) -> Query<Diff<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.domain);
        assert!(
            a.cx.sorts().coercible(a.domain.sort(), b.domain.sort()),
            "minus requires compatible whole domain sorts"
        );
        let keys: Box<[usize]> = a.ends[..a.domain.sort().scalar_len()].into();
        let b_keys: Box<[usize]> = b.ends[..b.domain.sort().scalar_len()].into();
        subtract(a, b, keys, b_keys)
    }

    /// Set difference on complete pairs, as in `HashSet::difference`: the
    /// pairs of A:D->R that B:D->R does not hold, keeping domain D and range
    /// R.
    ///
    /// The whole pair is the subtraction key, as for
    /// [`complement`](Self::complement), so a symbolic cell keeps exactly the
    /// values `b` lacks at that key. Keys alone cannot see the domain/range
    /// boundary, since `X -> (Y, Z)` and `(X, Y) -> Z` flatten onto the same
    /// positions, so both endpoints are checked here.
    ///
    /// This stays lazy, so a caller may `count`, `collect`, or `drive` the
    /// residual pairs as counterexamples to containment; `is_empty` reports
    /// containment alone.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, either endpoint has
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    fn difference<B: IntoQuery>(self, b: B) -> Query<Diff<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.into_query();
        assert!(
            a.cx.sorts().coercible(a.domain.sort(), b.domain.sort())
                && a.cx.sorts().coercible(a.range.sort(), b.range.sort()),
            "difference requires compatible query endpoints"
        );
        let (keys, b_keys) = (a.ends.as_ref().into(), b.ends.as_ref().into());
        subtract(a, b, keys, b_keys)
    }

    /// The complete pairs on which two relations disagree; empty exactly when
    /// they hold the same pairs.
    ///
    /// The union of both
    /// [`difference`](Self::difference)s, so it stays lazy and a caller may
    /// `count`, `collect`, or `drive` the differing pairs for a
    /// counterexample; `==` discards them and reports emptiness alone.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, either endpoint has
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    #[expect(
        clippy::type_complexity,
        reason = "retain both directed differences in the fused query type"
    )]
    fn symmetric_difference<B: IntoQuery>(
        self,
        b: B,
    ) -> Query<Union<Diff<Self::Node, B::Node>, Diff<B::Node, Self::Node>>>
    where
        Self::Node: Clone,
        B::Node: Clone,
    {
        let a = self.into_query();
        let b = b.into_query();
        // Each operand is read in both directions, so neither can borrow a
        // local here.
        a.clone().difference(b.clone()).union(b.difference(a))
    }

    /// Shared-domain pairing: A:D->R and B:D->S produce D->(R,S), flattening
    /// anonymous products.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, their domains have
    /// incompatible scalar sorts, or matched fixed columns require incompatible
    /// storage.
    #[inline(always)]
    fn and<B: Operand>(self, b: B) -> Query<Join<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.domain);
        assert!(
            a.cx.sorts().coercible(a.domain.sort(), b.domain.sort()),
            "pairing requires compatible whole domain sorts"
        );
        let (m, k) = (a.ends.len(), b.ends.len());
        let (d, e) = (a.domain.sort().scalar_len(), b.domain.sort().scalar_len());
        let (domain, range) = (
            a.domain.sort(),
            a.cx.sorts().product([a.range.sort(), b.range.sort()]),
        );
        let u = universe(a.universe, b.universe, domain, range);
        binary(a, b, (0..d, 0..e), (0..m, e..k), domain, range, u)
    }

    /// Keeps ranges belonging to the operand's domain, without replacing
    /// them.
    ///
    /// Hidden source levels still enumerate: `drive` may revisit a
    /// visible pair from different witnesses, while `collect` unions them.
    /// Chained restrictions conjoin different field tests.
    ///
    /// # Panics
    ///
    /// Panics if the operands belong to different contexts, the left range and
    /// right domain have incompatible scalar sorts, or matched fixed columns
    /// require incompatible storage.
    #[inline(always)]
    fn with<B: Operand>(self, b: B) -> Query<Join<Self::Node, B::Node>> {
        let a = self.into_query();
        let b = b.resolve(&a.cx, &a.range);
        assert!(
            a.cx.sorts().coercible(a.range.sort(), b.domain.sort()),
            "restriction requires compatible whole endpoint sorts"
        );
        let (m, e) = (a.ends.len(), b.domain.sort().scalar_len());
        let d = a.domain.sort().scalar_len();
        let (domain, range) = (a.domain.sort(), a.range.sort());
        let u = a.universe;
        binary(a, b, (d..m, 0..e), (0..m, 0..0), domain, range, u)
    }

    /// The domain subidentity: the identity over this query's bound domain,
    /// restricted to the keys that have a witness.
    ///
    /// Stays lazy.
    #[inline(always)]
    fn keys(self) -> Query<Join<(), Self::Node>> {
        let a = self.into_query();
        Query::id(&a.cx, a.domain.clone()).with(a)
    }

    /// The range subidentity, [`keys`](Self::keys) of the inverse: the identity
    /// over this query's bound range, restricted to the values that have a
    /// witness.
    ///
    /// Stays lazy, and the identity is built from the bound range row, so the
    /// values keep their configured storage.
    #[inline(always)]
    fn values(self) -> Query<Join<(), Self::Node>> {
        let a = self.into_query();
        Query::id(&a.cx, a.range.clone()).with(a.inv())
    }
}

impl<A: Exec> IntoQuery for Query<A> {
    type Node = A;
    #[inline(always)]
    fn into_query(self) -> Self {
        self
    }
}

impl<'a, A: Exec> IntoQuery for &'a Query<A> {
    type Node = &'a A;
    #[inline(always)]
    fn into_query(self) -> Query<&'a A> {
        Query {
            cx: self.cx.clone(),
            node: &self.node,
            universe: self.universe,
            domain: self.domain.clone(),
            range: self.range.clone(),
            slots: self.slots.clone(),
            ends: self.ends.clone(),
        }
    }
}

// Borrowing a composite operand may add another reference to a stored/query
// child.
impl<'a, A> IntoQuery for &&'a A
where
    &'a A: IntoQuery,
{
    type Node = <&'a A as IntoQuery>::Node;
    #[inline(always)]
    fn into_query(self) -> Query<Self::Node> {
        (*self).into_query()
    }
}

impl IntoQuery for &Rel {
    type Node = IndexRoot;
    #[inline(always)]
    fn into_query(self) -> Query<Self::Node> {
        source(self)
    }
}

/// Two queries are equal when they contain the same complete pairs.
///
/// Equality tests whether the symmetric difference is empty. Use
/// [`IntoQuery::symmetric_difference`] to inspect the differing pairs.
/// The right operand is spelled out per type because a generic
/// `for<'x> &'x B: IntoQuery` bound recurses through the implementation
/// that makes `&&A` a query whenever `&A` is one.
///
/// # Panics
///
/// Panics if the operands belong to different contexts, either endpoint has
/// incompatible scalar sorts, or matched fixed columns require incompatible
/// storage, as [`symmetric_difference`](IntoQuery::symmetric_difference)
/// does, or if the comparison's map equations form a cycle or a map transfers
/// a predicate between endpoints in different columns.
impl<A: Exec, B: Exec> PartialEq<Query<B>> for Query<A> {
    #[inline(always)]
    fn eq(&self, other: &Query<B>) -> bool {
        self.symmetric_difference(other).is_empty()
    }
}

impl<A: Exec> PartialEq<Rel> for Query<A> {
    #[inline(always)]
    fn eq(&self, other: &Rel) -> bool {
        self.symmetric_difference(other).is_empty()
    }
}

impl<B: Exec> PartialEq<Query<B>> for Rel {
    #[inline(always)]
    fn eq(&self, other: &Query<B>) -> bool {
        self.symmetric_difference(other).is_empty()
    }
}

impl PartialEq for Rel {
    #[inline(always)]
    fn eq(&self, other: &Rel) -> bool {
        self.symmetric_difference(other).is_empty()
    }
}

/// Projects a component of a source-free identity.
///
/// The enclosing query
/// composes with it once, even when a larger accessor pairs several
/// projections of that same input row. Constructors use this same mapping
/// with every input column and a different nominal output sort.
///
/// # Panics
///
/// Panics if positions does not name exactly one input column per range scalar
/// leaf, an endpoint belongs to another context, or the projected scalar sorts
/// differ from the range.
#[inline(always)]
pub fn project(
    cx: &Context,
    input: &Row,
    positions: impl IntoIterator<Item = usize>,
    range: Row,
) -> Query<()> {
    let slots: Arc<[Slot]> = input
        .algebras()
        .iter()
        .map(|&algebra| Slot {
            algebra,
            read: false,
            fixed: false,
        })
        .collect();
    let ends: Arc<[usize]> = (0..input.sort().scalar_len()).chain(positions).collect();
    assert_eq!(
        ends[input.sort().scalar_len()..].len(),
        range.sort().scalar_len(),
        "one projection position per range column"
    );
    let universe = if input.sort() == range.sort()
        && ends[..input.sort().scalar_len()] == ends[input.sort().scalar_len()..]
    {
        Universe::Set
    } else {
        Universe::Binary
    };
    make(cx, (), input.sort(), range.sort(), universe, slots, ends)
}

/// Opens an owned source leaf.
///
/// Dropping the relation leaves its root alive; every occurrence receives its
/// own invocation state. Endpoint sharing comes only from the stored column
/// map.
#[inline(always)]
pub fn source(rel: &Rel) -> Query<IndexRoot> {
    let slots = (0..rel.root().index().arity())
        .map(|i| Slot {
            algebra: rel.root().index().algebra(i),
            read: true,
            fixed: true,
        })
        .collect();
    let n = rel.domain().sort().scalar_len();
    let ends: Arc<[usize]> = match rel.universe() {
        Universe::Set => (0..n).chain(0..n).collect(),
        Universe::Binary => (0..n + rel.range().sort().scalar_len()).collect(),
    };
    make(
        rel.root().index().context(),
        rel.root().clone(),
        rel.domain().sort(),
        rel.range().sort(),
        rel.universe(),
        slots,
        ends,
    )
}

/// A map from an input scalar cell to a fresh output cell.
///
/// Both endpoints stay live before registration, so earlier sources cannot
/// skip their restrictions or finalize their outputs. Stored sources choose
/// endpoint columns; transfers require the same backend instance. Execution
/// panics if the equations form a cycle or the endpoints of a transfer
/// adopt different columns.
///
/// # Panics
///
/// Panics if the input belongs to another context or the map has an aggregate
/// input.
// A map, including one whose operations cancel, must keep its fresh output
// endpoint.
pub fn transform(cx: &Context, input: &Row, map: Arc<MapSpec>) -> Query<Transform> {
    let algebra = input.scalar_algebra();
    let slots = [Slot {
        algebra,
        read: true,
        fixed: false,
    }; 2]
        .into();
    make(
        cx,
        Transform { map },
        input.sort(),
        input.sort(),
        Universe::Binary,
        slots,
        [0, 1].into(),
    )
}

impl<A: Exec> Query<A> {
    /// The range positions of a source-free structural map, including
    /// metadata-only constructors.
    ///
    /// Positions may repeat.
    ///
    /// # Panics
    ///
    /// Panics if the node executes a source, filter, or map, or the endpoint
    /// wiring does not project the complete input.
    pub fn positions(&self) -> Vec<usize> {
        assert!(
            self.node.is_projection() && self.slots.iter().all(|slot| !slot.read),
            "positions require a source-free projection"
        );
        let (domain, range) = self.ends.split_at(self.domain.sort().scalar_len());
        let mut input = vec![usize::MAX; self.slots.len()];
        for (i, &cell) in domain.iter().enumerate() {
            assert!(
                input[cell] == usize::MAX,
                "positions require a source-free projection"
            );
            input[cell] = i;
        }
        range
            .iter()
            .map(|&cell| {
                let i = input[cell];
                assert!(
                    i != usize::MAX,
                    "positions require a projection of the complete input"
                );
                i
            })
            .collect()
    }

    /// Runs this query over the `observed` cells.
    ///
    /// Every terminal uses the same execution and correlation finalization.
    /// Initial predicates in `eval` restrict sources before traversal;
    /// occurrence state belongs to this invocation.
    #[inline(always)]
    fn run<M: Mode>(&self, eval: &mut Run<'_>, observed: &[usize], mode: &mut M) -> Flow<M> {
        self.node.run(
            0..self.slots.len(),
            eval,
            &mut A::State::<M>::default(),
            mode,
            &mut |eval, mode| {
                eval.finish(eval.cells.len(), observed, mode, &mut |eval, mode| {
                    mode.done(eval)
                })
            },
        )
    }

    /// Runs one callback traversal that observes the cells in `ends`, the
    /// first `domain` of them as the domain binding.
    fn visit<B>(
        &self,
        ends: &[usize],
        domain: usize,
        then: &mut impl for<'a> FnMut(Binding<'a>, Binding<'a>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        let mut eval = Run::new(&self.cx, &self.slots);
        let mut mode = Visit { then, ends, domain };
        self.run(&mut eval, ends, &mut mode)
    }

    /// Visits exact symbolic branches; hidden witnesses may revisit visible
    /// results.
    ///
    /// Both endpoints borrow the same predicate cells, and no store guard
    /// crosses the callback. Each invocation owns its state and callback;
    /// callbacks need not be `Send` or `Sync` and may run nested queries.
    /// Concurrent callers construct their callbacks in their workers.
    ///
    /// # Panics
    ///
    /// Panics if the query's map equations form a cycle, or a map transfers
    /// a predicate between endpoints in different columns.
    pub fn drive<B>(
        &self,
        then: &mut impl for<'a> FnMut(Binding<'a>, Binding<'a>) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        self.visit(&self.ends, self.domain.sort().scalar_len(), then)
    }

    /// Whether this query holds no pairs.
    ///
    /// The first accepting symbolic branch decides emptiness. The terminal
    /// observes no cells; operators such as minus may still collect key
    /// coverage and decompose key correlations.
    ///
    /// # Panics
    ///
    /// Panics if the query's map equations form a cycle, or a map transfers
    /// a predicate between endpoints in different columns.
    pub fn is_empty(&self) -> bool {
        self.visit(&[], 0, &mut |_, _| ControlFlow::Break(()))
            .is_continue()
    }

    /// Counts distinct domain-range pairs without constructing the final
    /// relation.
    ///
    /// Each probe finds one exact residual inside a disjoint search box, then
    /// excludes that residual from future probes. Aliases count once;
    /// overlapping branches and hidden witnesses add no multiplicity.
    /// Subtraction and equation finalization retain their existing internal
    /// work.
    ///
    /// Every probe restarts the traversal, so the cost grows with the number
    /// of disjoint residuals times the branches each probe rejects before
    /// accepting one. Fragmented outputs can count more slowly than they
    /// collect.
    ///
    /// # Panics
    ///
    /// Panics if the query's map equations form a cycle, or a map transfers
    /// a predicate between endpoints in different columns.
    pub fn count(&self) -> Count {
        let mut cells = Vec::new();
        for &cell in self.ends.iter() {
            label(&mut cells, cell);
        }
        let mut pending = vec![
            cells
                .iter()
                .map(|&cell| self.cx.t(self.slots[cell].algebra))
                .collect::<Vec<_>>(),
        ];
        let mut count = Count::ZERO;
        while let Some(mut bounds) = pending.pop() {
            let mut eval = Run::new(&self.cx, &self.slots);
            for (&cell, p) in cells.iter().zip(&bounds) {
                eval.set(cell, p);
            }
            let mut first = |_: Binding<'_>, r: Binding<'_>| {
                ControlFlow::Break(
                    (0..r.cells.len())
                        .map(|i| r.predicate(i))
                        .collect::<Vec<_>>(),
                )
            };
            let mut mode = Visit {
                then: &mut first,
                ends: &cells,
                domain: 0,
            };
            // TODO: Benchmark repeated restricted traversal in more detail
            // against `collect().count()`, especially fragmented outputs and
            // repeated hidden-witness probes. Each probe restarts from the
            // root and re-walks every rejected branch inside its box, so a
            // select over n stored rows rejecting most of them can probe in
            // O(n^2). Investigate
            // continuation/suffix reuse only with measured benefit and
            // preserved distinct-output semantics.
            let ControlFlow::Break(branch) = self.run(&mut eval, &cells, &mut mode) else {
                continue;
            };
            count += branch.iter().map(Predicate::count).product::<Count>();
            // U \ B is the disjoint union of B[..i] x (U[i] \ B[i]) x U[i+1..].
            // Pending boxes exclude all previously counted pairs without
            // storing their union.
            for (i, p) in branch.into_iter().enumerate() {
                let outside = bounds[i].difference(&p);
                if !outside.function().is_f() {
                    let mut rest = bounds.clone();
                    rest[i] = outside;
                    pending.push(rest);
                }
                bounds[i] = p;
            }
        }
        count
    }

    /// Materializes the declared endpoint universe.
    ///
    /// Binary queries retain the full domain/range product even when their
    /// endpoint cells alias, including constructors that change nominal type.
    /// Each invocation owns its accumulator and suffix caches, so shared
    /// queries can collect independently.
    /// Caller-partitioned executions combine compatible results with lazy
    /// `union` and explicit collection.
    ///
    /// # Panics
    ///
    /// Panics if the query's map equations form a cycle, or a map transfers
    /// a predicate between endpoints in different columns.
    pub fn collect(&self) -> Rel {
        let (positions, _) = self.output();
        let rel = match self.universe {
            Universe::Set => Rel::new_set(&self.cx, &self.domain),
            Universe::Binary => Rel::new(&self.cx, &self.domain, &self.range),
        };
        let mut mode = Collect::new(
            rel.root().index().clone(),
            positions,
            self.slots.len(),
            None,
        );
        let mut eval = Run::new(&self.cx, &self.slots);
        let result = self.run(&mut eval, positions, &mut mode);
        let root = match result {
            ControlFlow::Continue(root) | ControlFlow::Break(root) => root,
        };
        rel.with_root(root)
    }
}

impl Query<()> {
    /// The source-free identity over one complete bound row of `cx`, with an
    /// explicit fixed domain.
    ///
    /// # Panics
    ///
    /// Panics if the row belongs to another context or has an unbound scalar leaf.
    #[inline(always)]
    pub fn id(cx: &Context, row: impl Domain) -> Self {
        let row = row.row(cx);
        project(cx, &row, 0..row.sort().scalar_len(), row.clone())
    }

    /// A declaration as a source-free projection of its bound declared row.
    ///
    /// A member selects its columns from the domain row. A constructor reads
    /// the record's configured columns as its argument product and writes
    /// the same columns under its record sort.
    ///
    /// # Panics
    ///
    /// Panics if the declaration belongs to another context or a required scalar
    /// leaf is unbound.
    pub fn decl(cx: &Context, decl: FuncDecl) -> Self {
        let info = decl.resolve(cx.sorts());
        match info.member {
            Some(i) => {
                let input = info.domain.row(cx);
                let mut members = input.members(cx);
                let start: usize = members
                    .by_ref()
                    .take(i)
                    .map(|r| r.sort().scalar_len())
                    .sum();
                let part = members.next().unwrap();
                project(cx, &input, start..start + part.sort().scalar_len(), part)
            }
            None => {
                let range = info.range.row(cx);
                let mut input = range.clone();
                input.coerce(cx, info.domain);
                project(cx, &input, 0..input.sort().scalar_len(), range)
            }
        }
    }

    /// The source-free projection of one record row onto every member but
    /// `i`.
    ///
    /// The output is the anonymous product of the kept members, sharing
    /// their configured storage.
    ///
    /// # Panics
    ///
    /// Panics if the member index is out of bounds, the row belongs to another
    /// context, or a required scalar leaf is unbound.
    pub fn except(cx: &Context, row: impl Domain, i: usize) -> Self {
        let row = row.row(cx);
        let members = row.members(cx);
        assert!(i < members.len(), "member index exceeds the record");
        let mut positions = Vec::new();
        let mut kept = Vec::new();
        let mut start = 0;
        for (j, member) in members.enumerate() {
            let end = start + member.sort().scalar_len();
            if i != j {
                positions.extend(start..end);
                kept.push(member);
            }
            start = end;
        }
        let range = Row::new(
            cx,
            cx.sorts().product(kept.iter().map(Row::sort)),
            kept.iter().flat_map(Row::algebras).copied().collect(),
            0,
        );
        project(cx, &row, positions, range)
    }
}
