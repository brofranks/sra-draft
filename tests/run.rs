//! Checks for cell equations, narrowing, traversal modes, and output
//! collection.
//!
//! The `Table` fixture is a four-value truth-table backend that counts the
//! references on each root it hands out, so tests can observe leaked or
//! multiply released roots and noncanonical tokens. Local fixtures construct
//! relations, bind bit-vector rows, build sets, and list ground pairs, rows,
//! and set values.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use sra::backend::{Algebra, Align, Backend, BoolOp, Function, MapSpec, Refine, Values};
use sra::context::Context;
use sra::map;
use sra::query::{IntoQuery, Query, id};
use sra::rel::Rel;
use sra::row::Row;
use sra::scalar::ScalarSort;

/// A bound row of the bit-vector sort of width `w`.
fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.sorts().bv(w);
    Row::bind(cx, sort)
}

/// A 4-bit set holding `values`, over the already bound row `four`.
fn set(cx: &Context, four: &Row, values: impl IntoIterator<Item = u8>) -> Rel {
    let mut s = Rel::new_set(cx, four.clone());
    s.insert_rows(values);
    s
}

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &Rel) -> BTreeSet<Vec<u64>> {
    let cx = rel.root().index().context();
    let widths: Vec<u32> = (0..rel.root().index().arity())
        .map(|i| {
            cx.resolve(rel.root().index().algebra(i))
                .sort()
                .bv_width()
                .unwrap()
        })
        .collect();
    let mut out = BTreeSet::new();
    let mut row = vec![0; widths.len()];
    loop {
        if rel.root().contains(&row) {
            out.insert(row.clone());
        }
        let Some(i) = (0..row.len()).rev().find(|&i| row[i] + 1 < 1 << widths[i]) else {
            return out;
        };
        row[i] += 1;
        row[i + 1..].fill(0);
    }
}

/// The ground rows of a binary relation over 4-bit columns, as pairs.
fn pairs(rel: &Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

/// A binary relation over `four` holding `ps`.
fn relation(cx: &Context, four: &Row, ps: impl IntoIterator<Item = (u8, u8)>) -> Rel {
    let mut r = Rel::new(cx, four.clone(), four.clone());
    r.insert_rows(ps);
    r
}

/// The values of a set over one column.
fn values(rel: &Rel) -> BTreeSet<u64> {
    rows(rel)
        .into_iter()
        .map(|r| {
            assert_eq!(r.len(), 1);
            r[0]
        })
        .collect()
}

// --- traversal and memoization --------------------------------------------

#[test]
fn chained_joins_traverse_every_source_in_order() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let a = relation(&cx, &four, [(1, 2), (1, 3), (2, 3)]);
    let b = relation(&cx, &four, [(2, 5), (3, 6)]);
    let c = relation(&cx, &four, [(5, 8), (6, 9)]);
    assert_eq!(
        pairs(&a.select(&b).select(&c).collect()),
        [(1, 8), (1, 9), (2, 9)].into()
    );
    // A self-join shares one stored relation between two independently
    // bound occurrences.
    assert_eq!(pairs(&a.select(&a).collect()), [(1, 3)].into());
    // Three classes of a hidden join cell accumulate outside the
    // single-level fast path, so the level sum folds them through its carry
    // slots.
    let wide = relation(&cx, &four, [(1, 5), (1, 6), (1, 7)]);
    let fan = relation(&cx, &four, [(5, 9), (6, 10), (7, 11)]);
    assert_eq!(
        pairs(&wide.select(&fan).collect()),
        [(1, 9), (1, 10), (1, 11)].into()
    );
    // Three output coordinates on one cell: the collector attaches a
    // diagonal, not a product.
    let s = set(&cx, &four, [1, 2, 3]);
    let d = id().and(&s).and(&s);
    let result = d.collect();
    assert_eq!(result.count(), 3);
    for x in 1..=3 {
        assert!(result.root().contains(&[x; 4]));
    }
}

#[test]
fn shared_range_subtrees_collect_and_exclude_exactly() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // Many domain values share one range subtree. Each such suffix has one
    // pending coordinate, so the traversal recomputes it rather than
    // consulting the suffix cache.
    let mut a = Rel::new(&cx, four.clone(), four.clone());
    a.insert_rows((0..16_u8).flat_map(|x| [(x, 1_u8), (x, 2)]));
    let b = relation(&cx, &four, [(1, 7), (2, 8)]);
    let joined = a.select(&b);
    assert_eq!(joined.count().to_string(), "32");
    assert_eq!(
        pairs(&joined.collect()),
        (0..16).flat_map(|x| [(x, 7), (x, 8)]).collect()
    );
    // The same shape under an exclusion probe, where coverage can also
    // complete early. `minus` subtracts whole keys, so excluding one pair
    // drops that domain value entirely.
    assert!((&joined).minus(&joined).is_empty());
    assert_eq!(
        pairs(&(&joined).minus(&relation(&cx, &four, [(0, 7)])).collect()),
        (1..16).flat_map(|x| [(x, 7), (x, 8)]).collect()
    );
}

#[test]
fn shared_suffixes_preserve_each_prefix_with_two_pending_coordinates() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let pair_sort = cx.sorts().product([four.sort(), four.sort()]);
    let pair = Row::bind(&mut cx, pair_sort);
    // Three left prefixes differ in their own columns but all finish with
    // the same middle value, and nothing reads them afterwards, so each
    // retires before the right-hand side runs and the suffix key is
    // identical in all three. The pairing keeps two output coordinates
    // pending throughout, which is what makes the suffix worth keying: the
    // second visit records it and the third observes it from the cache
    // without walking.
    let mut a = Rel::new(&cx, pair, four.clone());
    a.insert_rows([
        (1_u8, 1_u8, 5_u8),
        (1, 2, 5),
        (2, 1, 5),
        (3, 1, 5),
        (3, 2, 5),
        (3, 3, 5),
    ]);
    let b = relation(&cx, &four, [(5, 9)]);
    let c = relation(&cx, &four, [(5, 2)]);
    let paired = b.and(&c);
    let q = a.select(&paired);
    let result = q.collect();
    assert_eq!(result.count(), 6);
    for [x, y] in [[1, 1], [1, 2], [2, 1], [3, 1], [3, 2], [3, 3]] {
        assert!(result.root().contains(&[x, y, 9, 2]));
    }
}

// --- equations and narrowing ----------------------------------------------

#[test]
fn a_stored_restriction_travels_back_through_both_equations() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // Both maps are active when the stored relation refines the last cell
    // against its current image {13, 14, 15}, so keys outside that image,
    // such as 0, are never visited. Each surviving value travels back
    // through both equations to the source.
    let s = set(&cx, &four, [1, 2, 3]);
    let t = relation(&cx, &four, [(0, 1), (13, 2), (15, 4)]);
    let q = s.select(map::bvor(4_u8)).select(map::bvor(8_u8)).select(&t);
    assert_eq!(pairs(&q.collect()), [(1, 2), (3, 4)].into());
}

#[test]
fn union_branches_restore_equalities_and_hide_local_cells() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let pair = cx.sorts().product([four.sort(), four.sort()]);
    let triple_sort = cx.sorts().product([pair, four.sort()]);
    let triple = Row::bind(&mut cx, triple_sort);
    // The left branch reads one cell at three output positions, so it
    // restores identity equations between the union's distinct cells.
    let s = set(&cx, &four, [1, 2]);
    let mut wide = Rel::new(&cx, four.clone(), triple);
    wide.insert((5_u8, 6_u8, 7_u8, 8_u8));
    let u = id().and(&s).and(&s).union(&wide);
    let result = u.collect();
    assert_eq!(result.count(), 3);
    for row in [[1, 1, 1, 1], [2, 2, 2, 2], [5, 6, 7, 8]] {
        assert!(result.root().contains(&row));
    }
    // A branch with a join keeps its middle cell local to the branch.
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    assert_eq!(
        pairs(&r.select(&r).union(&r).collect()),
        [(1, 2), (1, 3), (2, 3)].into()
    );
    // A composed branch keeps no cell of its own at the union boundary: its
    // middle cell is created per invocation, so both alternatives map onto
    // the union's two cells.
    let composed = r.select(&r);
    assert_eq!(
        (composed.slots.len(), composed.ends.to_vec()),
        (2, vec![0, 1])
    );
    // Callback traversal brackets the same alternatives without a cache
    // namespace, and sums their contributions with nothing to accumulate.
    assert_eq!(r.select(&r).union(&r).count().to_string(), "3");
    assert!(!r.select(&r).union(&r).is_empty());
}

#[test]
fn equations_transfer_narrowings_along_the_whole_chain() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // Three cells joined by two maps: restricting the last one has to travel
    // back through both equations, and a restriction disjoint from the image
    // leaves the restricted cell empty.
    let s = set(&cx, &four, [1, 2, 3]);
    let q = s.select(map::bvor(4_u8)).select(map::bvor(8_u8));
    assert_eq!(pairs(&q.collect()), [(1, 13), (2, 14), (3, 15)].into());
    assert!((&q).with(&set(&cx, &four, [13])).count().to_string() == "1");
    assert!(q.with(&set(&cx, &four, [0])).is_empty());
    // An identity map keeps a fresh endpoint, which a union with the
    // unmapped relation reads beside the stored range cell.
    let r = relation(&cx, &four, [(1, 2), (2, 3)]);
    let u = r.select(map::copy()).union(&r);
    assert_eq!(pairs(&u.collect()), [(1, 2), (2, 3)].into());
    let both = r.select(map::copy()).union(&r).intersection(&r);
    assert_eq!(pairs(&both.collect()), [(1, 2), (2, 3)].into());
}

#[test]
fn unsatisfiable_restrictions_narrow_to_nothing() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    // No equations are active, and the restriction narrows the only cell to
    // the false predicate: a disjoint set leaves nothing, and an empty set
    // leaves no branch at all.
    let s = set(&cx, &four, [1, 2, 3]);
    let (nine, none) = (set(&cx, &four, [9]), set(&cx, &four, []));
    assert!(s.with(&nine).is_empty());
    assert_eq!(s.with(&nine).count().to_string(), "0");
    assert!(Query::id(&cx, four).with(&none).is_empty());
    let mut seen = BTreeSet::new();
    let _ = s.with(&none).drive(&mut |d, _| {
        seen.insert(d.predicate(0).count().to_string());
        std::ops::ControlFlow::<()>::Continue(())
    });
    assert!(seen.is_empty());
}

#[test]
fn borrowed_root_names_the_same_cell_as_the_handle() {
    let mut cx = Context::default();
    let four = bv(&mut cx, 4);
    let s = set(&cx, &four, [1, 2, 3]);
    let mut seen = Vec::new();
    let _ = (&s).into_query().drive(&mut |d, r| {
        let p = d.predicate(0);
        let (algebra, f) = r.root(0);
        assert_eq!((p.algebra(), p.function()), (algebra, f));
        assert!(d.context() == p.context());
        let mut cursor = d.context().resolve(algebra).members(f);
        while let Some(value) = cursor.next() {
            seen.push(value[0]);
        }
        std::ops::ControlFlow::<()>::Continue(())
    });
    seen.sort_unstable();
    assert_eq!(seen, [1, 2, 3]);
}

#[test]
fn narrowing_an_empty_cell_never_calls_the_continuation() {
    use sra::run::{Binding, Mode, Run, Slot, Visit};

    let mut cx = Context::default();
    let two = bv(&mut cx, 2);
    let algebra = two.scalar_algebra();
    let mut eval = Run::new(
        &cx,
        &[Slot {
            algebra,
            read: false,
            fixed: true,
        }],
    );
    eval.set(0, &cx.set(algebra, &[]));
    let mut called = 0;
    let mut then = |_: Binding<'_>, _: Binding<'_>| {
        called += 1;
        ControlFlow::<()>::Continue(())
    };
    let mut mode = Visit {
        then: &mut then,
        ends: &[0, 0],
        domain: 1,
    };
    assert!(
        eval.narrow(0, Function::FALSE, &mut mode, |a, m| m.done(a))
            .is_continue()
    );
    assert_eq!(called, 0);
}

// --- the Table backend ----------------------------------------------------

/// A four-value truth-table store with fresh tokens for nonconstant maps.
///
/// Explicit counts expose leaked or multiply released external roots.
#[derive(Clone)]
struct Table(Arc<Mutex<TableState>>);

struct TableState {
    entries: Vec<([u32; 4], usize)>,
    cases: usize,
}

impl Table {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(TableState {
            entries: vec![([0; 4], 0), ([1; 4], 0)],
            cases: 0,
        })))
    }

    fn get(&self, f: Function) -> [u32; 4] {
        let entries = &self.0.lock().unwrap().entries;
        let (values, refs) = entries[f.index()];
        assert!(f.is_constant() || refs > 0);
        values
    }

    fn put(&self, values: [u32; 4]) -> Function {
        if values == [0; 4] {
            return Function::FALSE;
        }
        if values == [1; 4] {
            return Function::TRUE;
        }
        let entries = &mut self.0.lock().unwrap().entries;
        let f = Function::new(entries.len());
        entries.push((values, 1));
        f
    }

    fn partition<T: Copy + PartialEq>(&self, values: [T; 4], labels: &mut Vec<T>) -> Function {
        let start = labels.len();
        self.put(values.map(|value| {
            let i = labels[start..].iter().position(|&v| v == value);
            i.unwrap_or_else(|| {
                labels.push(value);
                labels.len() - start - 1
            }) as u32
        }))
    }

    fn refs(&self) -> usize {
        self.0.lock().unwrap().entries.iter().map(|(_, n)| n).sum()
    }
}

impl Backend for Table {
    fn bind(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Algebra>> {
        (sort == self.sort()).then(|| Box::new((*self).clone()) as Box<dyn Algebra>)
    }
}

impl Algebra for Table {
    fn sort(&self) -> ScalarSort {
        ScalarSort::bv(2)
    }

    fn inc_ref(&self, f: Function) {
        if !f.is_constant() {
            let entries = &mut self.0.lock().unwrap().entries;
            assert!(entries[f.index()].1 > 0);
            entries[f.index()].1 += 1;
        }
    }

    fn dec_ref(&self, f: Function) {
        if !f.is_constant() {
            let entries = &mut self.0.lock().unwrap().entries;
            assert!(entries[f.index()].1 > 0);
            entries[f.index()].1 -= 1;
        }
    }

    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        self.get(f)[value[0] as usize]
    }

    fn count(&self, p: Function) -> sra::util::count::Count {
        self.get(p).into_iter().map(u64::from).sum::<u64>().into()
    }

    fn set(&self, values: &[u64]) -> Function {
        self.put(std::array::from_fn(|i| {
            u32::from(values.contains(&(i as u64)))
        }))
    }

    fn lt(&self, bound: &[u64]) -> Function {
        self.put(std::array::from_fn(|i| u32::from((i as u64) < bound[0])))
    }

    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function {
        let (a, b) = (self.get(left), self.get(right));
        self.put(std::array::from_fn(|i| {
            u32::from(op.evaluate_bool(a[i] != 0, b[i] != 0))
        }))
    }

    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_sort(self.sort());
        let (a, b) = (self.get(p), self.get(q));
        let mut values = [0; 4];
        for (i, &x) in a.iter().enumerate() {
            let mut y = [0];
            if x != 0 && map.evaluate(&[i as u64], &mut y) && b[y[0] as usize] != 0 {
                values[y[0] as usize] = 1;
            }
        }
        self.put(values)
    }

    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        map.check_sort(self.sort());
        let (a, b) = (self.get(p), self.get(q));
        self.put(std::array::from_fn(|i| {
            let mut y = [0];
            u32::from(a[i] != 0 && map.evaluate(&[i as u64], &mut y) && b[y[0] as usize] != 0)
        }))
    }

    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        let (p, a, b) = (self.get(condition), self.get(yes), self.get(no));
        self.put(std::array::from_fn(|i| if p[i] != 0 { a[i] } else { b[i] }))
    }

    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        _: &mut Align,
    ) -> Function {
        let (a, b) = (self.get(left), self.get(right));
        self.partition(std::array::from_fn(|i| (a[i], b[i])), labels)
    }

    fn quotient(&self, f: Function, slots: &[u32], _: &mut Align) -> Function {
        self.put(self.get(f).map(|i| slots[i as usize]))
    }

    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        self.0.lock().unwrap().cases += 1;
        let mut values = [default; 4];
        for &(p, label) in cases {
            for (i, value) in self.get(p).into_iter().enumerate() {
                if value != 0 {
                    values[i] = label;
                }
            }
        }
        self.partition(values, labels)
    }

    fn values(
        &self,
        default: u32,
        keys: &[u64],
        cases: &[u32],
        labels: &mut Vec<u32>,
        _: &mut Values,
    ) -> Function {
        let mut values = [default; 4];
        for (&key, &label) in keys.iter().zip(cases) {
            values[key as usize] = label;
        }
        self.partition(values, labels)
    }

    fn refine(
        &self,
        f: Function,
        p: Function,
        _: &mut Refine,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let (slots, values) = (self.get(f), self.get(p));
        for slot in 0..=slots.into_iter().max().unwrap() {
            let cell = std::array::from_fn(|i| u32::from(slots[i] == slot && values[i] != 0));
            if cell != [0; 4] && select(slot) {
                then(slot, &mut || self.put(cell))?;
            }
        }
        ControlFlow::Continue(())
    }

    fn members(&self, p: Function) -> Box<dyn sra::backend::Cursor + '_> {
        struct Cursor<'a> {
            table: &'a Table,
            root: Function,
            next: usize,
            value: u64,
        }
        impl sra::backend::Cursor for Cursor<'_> {
            fn next(&mut self) -> Option<&[u64]> {
                while self.next < 4 {
                    let i = self.next;
                    self.next += 1;
                    if self.table.get(self.root)[i] != 0 {
                        self.value = i as u64;
                        return Some(std::slice::from_ref(&self.value));
                    }
                }
                None
            }
        }
        impl Drop for Cursor<'_> {
            fn drop(&mut self) {
                self.table.dec_ref(self.root);
            }
        }
        self.inc_ref(p);
        Box::new(Cursor {
            table: self,
            root: p,
            next: 0,
            value: 0,
        })
    }
}

// --- root release ---------------------------------------------------------

#[test]
fn nested_narrowings_release_roots_on_return_break_and_panic() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use sra::run::{Binding, Mode, Run, Slot, Visit};

    let table = Table::new();
    let mut cx = Context::default();
    cx.register_backend("table", table.clone());
    let algebra = cx.bind("table", table.sort());
    let slots = [Slot {
        algebra,
        read: false,
        fixed: true,
    }];
    for exit in 0..3 {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut eval = Run::new(&cx, &slots);
            eval.set(0, &cx.set(algebra, &[0, 1, 2]));
            let before = eval.root(0).1;
            let mut then = |_: Binding<'_>, _: Binding<'_>| {
                if exit == 2 {
                    panic!("callback failed");
                }
                if exit == 1 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let mut mode = Visit {
                then: &mut then,
                ends: &[0, 0],
                domain: 1,
            };
            let flow = eval.narrow(0, table.set(&[0, 1]), &mut mode, |a, m| {
                a.narrow(0, table.set(&[0]), m, |b, n| n.done(b))
            });
            assert_eq!(flow.is_break(), exit == 1);
            assert_eq!(eval.root(0).1, before);
            assert_eq!(table.refs(), 1);
        }));
        assert_eq!(result.is_err(), exit == 2);
        assert_eq!(table.refs(), 0);
    }
}

// --- collection -----------------------------------------------------------

#[test]
fn collection_compares_contents_with_noncanonical_backend_tokens() {
    use sra::run::{Collect, Mode};

    let table = Table::new();
    let mut cx = Context::default();
    cx.register_backend("table", table.clone());
    let t = cx.sorts().record("T", &[("value", cx.sorts().bv(2))]);
    cx.bind_field(cx.sorts().project(t, 0), "table");
    let field = cx.sorts().project(t, 0);
    let two = Row::bind(&mut cx, field);
    // Leaving value zero outside both regions puts FALSE in the first slot,
    // the attachment shape used by the k-ary collection path.
    let a = set(&cx, &two, [1]);
    let b = set(&cx, &two, [2]);
    let mut mode = Collect::new(a.root().index().clone(), &[0], 1, None);
    let mut sum = mode.sum();
    mode.push(&mut sum, 0, a.root().clone());
    mode.push(&mut sum, 0, b.root().clone());
    let before = table.0.lock().unwrap().cases;
    let out = mode.finish_sum(sum);
    assert_eq!(table.0.lock().unwrap().cases, before + 1);
    let chain = a.root().apply(BoolOp::Or, b.root());
    assert_ne!(out.residual().key(), chain.residual().key());
    for x in 0..4 {
        assert_eq!(out.contains(&[x]), x == 1 || x == 2);
        assert_eq!(chain.contains(&[x]), x == 1 || x == 2);
    }
    assert_eq!(values(&a.union(&b).collect()), [1, 2].into());
}

#[test]
fn an_empty_input_bound_skips_full_and_constrained_sources() {
    use sra::index::Index;
    use sra::run::{Binding, Exec, Mode, Run, Slot, Visit};
    let mut cx = Context::default();
    let row = bv(&mut cx, 2);
    let algebra = row.scalar_algebra();
    let rel = set(&cx, &row, [1]);
    for root in [Index::new(&cx, &[algebra]).full(), rel.root().clone()] {
        let mut eval = Run::new(
            &cx,
            &[Slot {
                algebra,
                read: true,
                fixed: true,
            }],
        );
        eval.set(0, &cx.f(algebra));
        let mut mode = Visit {
            then: &mut |_: Binding<'_>, _: Binding<'_>| -> ControlFlow<()> {
                panic!("empty input reached the callback")
            },
            ends: &[0],
            domain: 1,
        };
        assert!(
            root.run(0..1, &mut eval, &mut (), &mut mode, &mut |e, m| m.done(e))
                .is_continue()
        );
    }
}
