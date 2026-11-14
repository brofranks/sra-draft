//! Checks for the outer index: container interning, node construction, the
//! native Boolean operation, growth, and the traversals that read a root.
//!
//! Everything goes through `IndexRoot` directly: a relation only ever builds
//! the node shapes its loader needs, while the index itself accepts any
//! level, partition, and continuation.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::{IndexRoot, Labels, Level, SCRATCH, intern, set_memo_bound};
use crate::backend::{BoolOp, Function, Refinement};
use crate::context::{ColumnId, Context};
use crate::scalar::ScalarSort;
use crate::util::dag::{Idx, NeedGrowth, Shared};

/// A `w`-bit column on the named registered backend.
fn column(cx: &mut Context, backend: &str, w: u32) -> ColumnId {
    let backend = cx.backend(backend);
    cx.column(backend, ScalarSort::bit_vec(w))
}

/// A referenced node placing each child at its key on `level`, with FALSE
/// elsewhere.
fn at(index: &IndexRoot, level: usize, keys: &[u64], children: &[&IndexRoot]) -> IndexRoot {
    let conts: Vec<_> = children.iter().map(|c| c.function).collect();
    index.values(Level::new(level), keys, &conts, &mut Labels::default())
}

fn t(index: &IndexRoot) -> IndexRoot {
    index.inc_ref(Function::TRUE)
}

/// A referenced root holding exactly the sorted, distinct `rows` below
/// `level`.
fn build(index: &IndexRoot, level: usize, rows: &[Vec<u64>]) -> IndexRoot {
    if rows.is_empty() {
        return index.inc_ref(Function::FALSE);
    }
    if level == index.arity() {
        return t(index);
    }
    let (mut keys, mut children) = (Vec::new(), Vec::new());
    for group in rows.chunk_by(|x, y| x[level] == y[level]) {
        keys.push(group[0][level]);
        children.push(build(index, level + 1, group));
    }
    at(index, level, &keys, &children.iter().collect::<Vec<_>>())
}

/// Every ground row of a root over 4-bit one-word columns, by membership
/// alone.
fn ground(root: &IndexRoot) -> BTreeSet<Vec<u64>> {
    (0..16_u64.pow(root.arity() as u32))
        .map(|n| {
            (0..root.arity())
                .rev()
                .map(|i| n >> (4 * i) & 15)
                .collect::<Vec<_>>()
        })
        .filter(|row| root.contains(row))
        .collect()
}

/// The completed maintenance passes of a root's container.
fn growths(index: &IndexRoot) -> u64 {
    index.data.shared.epoch()
}

// --- containers -----------------------------------------------------------

#[test]
fn containers_are_interned_per_exact_column_list() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let b = column(&mut cx, "bdd", 4);
    let r = intern(&cx, &[a, b]);
    let s = intern(&cx, &[a, b]);
    let t = intern(&cx, &[b, a]);
    assert!(r.same_index(&s) && !r.same_index(&t));
    assert_eq!((r.arity(), r.column(0), r.column(1)), (2, a, b));
    assert!(r.context() == &cx);
    assert_eq!(cx.layouts.lock().unwrap().len(), 2);
    assert!(r.function.is_f() && r.clone().function.is_f());
}

#[test]
#[should_panic(expected = "column belongs to another context")]
fn containers_hold_only_their_context_columns_is_rejected() {
    let mut other = Context::default();
    let a = column(&mut other, "sparse", 4);
    intern(&Context::default(), &[a]);
}

#[test]
fn roots_outlive_the_callers_context_and_release_it_last() {
    for kind in 0..3 {
        let (roots, owner) = {
            let mut cx = Context::default();
            let a = column(&mut cx, "sparse", 4);
            let roots = [intern(&cx, &[a]), intern(&cx, &[a, a])].map(|index| match kind {
                0 => index,
                1 => t(&index),
                _ => at(&index, index.arity() - 1, &[1], &[&t(&index)]),
            });
            (roots, Arc::downgrade(&cx.data))
        };
        assert!(owner.upgrade().is_some());
        for root in &roots {
            assert_eq!(root.contains(&vec![1; root.arity()]), kind != 0);
        }
        let last = roots[1].clone();
        drop(roots);
        assert!(owner.upgrade().is_some());
        drop(last);
        assert!(owner.upgrade().is_none());
    }
}

// --- references -----------------------------------------------------------

#[test]
fn inc_ref_adopts_constants_without_touching_the_container() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a]);
    let node = at(&idx, 0, &[1], &[&t(&idx)]);
    let held = idx.inc_ref(node.function);
    drop(node);
    // The second owner keeps the node alive and readable.
    assert_eq!(idx.top(held.function), Level::new(0));
    assert!(t(&idx).function.is_t() && idx.inc_ref(Function::FALSE).function.is_f());
}

#[test]
#[should_panic(expected = "cannot acquire an unreferenced native root")]
fn inc_ref_rejects_an_unreferenced_root_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a]);
    let node = at(&idx, 0, &[1], &[&t(&idx)]);
    let f = node.function;
    drop(node);
    let _ = idx.inc_ref(f);
}

// --- reading a root -------------------------------------------------------

#[test]
fn count_folds_cells_universes_and_shared_suffixes() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a, a]);
    assert_eq!(idx.count().to_string(), "0");
    // A true root constrains nothing, so every skipped column contributes
    // its universe.
    assert_eq!(t(&idx).count().to_string(), "4096");
    // `c` is reached both below `x` and directly from the root, so the
    // memoized subcount is reused under a different number of skipped
    // columns.
    let c = at(&idx, 2, &[0, 1], &[&t(&idx), &t(&idx)]);
    let x = at(&idx, 1, &[0], &[&c]);
    let root = at(&idx, 0, &[0, 1], &[&x, &c]);
    assert_eq!(root.count().to_string(), "34");
}

#[test]
fn head_and_top_describe_one_node() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let root = at(&idx, 0, &[1, 2], &[&t(&idx), &t(&idx)]);
    assert_eq!(idx.top(root.function), Level::new(0));
    let (level, partition, hole) = idx.head(root.function);
    // Both keys lead to TRUE, so one cell holds them and the rest of the
    // column is the hole.
    assert_eq!(
        (level, partition.column, partition.slots),
        (Level::new(0), a, 2)
    );
    assert!(hole != u32::MAX);
}

#[test]
fn constrains_reports_every_level_reached() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a, a]);
    assert_eq!(idx.constrains(), [false, false, false]);
    assert_eq!(t(&idx).constrains(), [false, false, false]);
    // `c` is reached twice, so the traversal must not revisit it.
    let c = at(&idx, 2, &[0], &[&t(&idx)]);
    let x = at(&idx, 1, &[0], &[&c]);
    let root = at(&idx, 0, &[0, 1], &[&x, &c]);
    assert_eq!(root.constrains(), [true, true, true]);
}

#[test]
fn refine_visits_nonfalse_slots_and_stops_at_a_break() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let deep = at(&idx, 1, &[3], &[&t(&idx)]);
    let root = at(&idx, 0, &[1, 2], &[&t(&idx), &deep]);
    let mut seen = Vec::new();
    let flow = idx.refine(
        root.function,
        Function::TRUE,
        &mut Refinement::default(),
        |p, child| {
            seen.push(child);
            cx.resolve(a).dec_ref(p);
            ControlFlow::<()>::Continue(())
        },
    );
    assert!(flow.is_continue());
    assert_eq!(
        seen.len(),
        2,
        "the FALSE hole is declined without reading the node"
    );
    assert!(seen.contains(&Function::TRUE) && seen.contains(&deep.function));
    // The same traversal from an already read head, stopped at its first
    // cell.
    let (_, partition, hole) = idx.head(root.function);
    let mut n = 0;
    let flow = idx.refine_cells(
        root.function,
        partition,
        hole,
        Function::TRUE,
        &mut Refinement::default(),
        |p, _| {
            n += 1;
            cx.resolve(a).dec_ref(p);
            ControlFlow::Break(7)
        },
    );
    assert_eq!((flow, n), (ControlFlow::Break(7), 1));
}

#[test]
fn contains_walks_skipped_and_multiword_levels() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let wide = column(&mut cx, "sparse", 100);
    let idx = intern(&cx, &[a, wide, a]);
    // Only the middle, two-word level is constrained; the outer levels are
    // free.
    let root = at(&idx, 1, &[7, 1], &[&t(&idx)]);
    assert!(root.contains(&[3, 7, 1, 5]));
    assert!(!root.contains(&[3, 7, 0, 5]));
    assert!(!idx.contains(&[3, 7, 1, 5]));
    assert!(t(&idx).contains(&[3, 7, 1, 5]));
}

#[test]
#[should_panic(expected = "one encoded value is required per index level")]
fn contains_checks_the_encoded_row_length_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    intern(&cx, &[a, a]).contains(&[1]);
}

#[test]
#[should_panic(expected = "encoded row contains a value outside its scalar sort")]
fn contains_checks_every_value_against_its_sort_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    intern(&cx, &[a, a]).contains(&[1, 16]);
}

// --- node construction ----------------------------------------------------

#[test]
fn cases_drop_default_regions_and_share_labels() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (p, q) = (cx.set(a, &[1, 2]), cx.set(a, &[3]));
    let (r, s) = (cx.set(a, &[4]), cx.set(a, &[5]));
    let (x, y) = (
        at(&idx, 1, &[1], &[&t(&idx)]),
        at(&idx, 1, &[2], &[&t(&idx)]),
    );
    // No case leads anywhere but the default, so the whole level
    // disappears.
    let none = idx.cases(Level::new(0), x.function, a, &[(p.function(), x.function)]);
    assert_eq!(none.function, x.function);
    // The first nondefault continuation takes label 1, a repeat keeps it,
    // and a third continuation takes a fresh label that its own repeat
    // reuses.
    let root = idx.cases(
        Level::new(0),
        Function::FALSE,
        a,
        &[
            (p.function(), x.function),
            (q.function(), x.function),
            (r.function(), y.function),
            (s.function(), y.function),
        ],
    );
    assert_eq!(idx.top(root.function), Level::new(0));
    assert!(root.contains(&[1, 1]) && root.contains(&[3, 1]));
    assert!(root.contains(&[4, 2]) && root.contains(&[5, 2]));
    assert!(!root.contains(&[0, 1]) && !root.contains(&[1, 2]));
}

#[test]
#[should_panic(expected = "a base-map root belongs to another column")]
fn cases_checks_the_column_of_its_level_is_rejected() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 4), column(&mut cx, "bdd", 4));
    let idx = intern(&cx, &[a, a]);
    idx.cases(
        Level::new(0),
        Function::FALSE,
        b,
        &[(Function::TRUE, Function::TRUE)],
    );
}

#[test]
fn attachment_recognizes_the_single_cell_shape() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let p = cx.set(a, &[1, 2]);
    let deep = at(&idx, 1, &[3], &[&t(&idx)]);
    // One cell against its complement, which is the hole: the attachment
    // shape.
    let cell = idx.cases(
        Level::new(0),
        Function::FALSE,
        a,
        &[(p.function(), Function::TRUE)],
    );
    let (partition, child) = idx
        .attachment(cell.function, Level::new(0))
        .expect("attachment");
    assert!(child.function.is_t());
    assert_eq!(cx.partition(a, partition, 2).slots, 2);
    // A nonconstant continuation is returned owned.
    let deeper = idx.cases(
        Level::new(0),
        Function::FALSE,
        a,
        &[(p.function(), deep.function)],
    );
    assert_eq!(
        idx.attachment(deeper.function, Level::new(0))
            .expect("attachment")
            .1
            .function,
        deep.function
    );
    // Another level, and a node with three slots, are not attachments.
    assert!(idx.attachment(cell.function, Level::new(1)).is_none());
    let wide = at(&idx, 0, &[1, 2], &[&t(&idx), &deep]);
    assert!(idx.attachment(wide.function, Level::new(0)).is_none());
    // Two slots, but the cell rather than its complement leads nowhere.
    let flipped = idx.cases(
        Level::new(0),
        Function::TRUE,
        a,
        &[(p.function(), Function::FALSE)],
    );
    assert!(idx.attachment(flipped.function, Level::new(0)).is_none());
}

#[test]
fn values_labels_children_by_first_appearance() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (p, q) = (
        at(&idx, 1, &[1], &[&t(&idx)]),
        at(&idx, 1, &[2], &[&t(&idx)]),
    );
    let mut labels = Labels::default();
    // The first continuation takes label 1, a repeat of it the same label,
    // a second continuation a fresh label, and its repeat the label already
    // recorded.
    let conts = [p.function, p.function, q.function, q.function];
    let root = idx.values(Level::new(0), &[1, 2, 3, 4], &conts, &mut labels);
    assert_eq!(labels.conts, [Function::FALSE, p.function, q.function]);
    let (_, partition, _) = idx.head(root.function);
    // Cells: {1,2} -> p, {3,4} -> q, and everything else -> FALSE.
    assert_eq!(partition.slots, 3);
    assert!(root.contains(&[1, 1]) && root.contains(&[2, 1]) && root.contains(&[3, 2]));
    assert!(!root.contains(&[0, 1]) && !root.contains(&[1, 2]));
    // A second call with the same shape finds the interned node and releases
    // its partition.
    let again = idx.values(Level::new(0), &[1, 2, 3, 4], &conts, &mut labels);
    assert_eq!(again.function, root.function);
}

#[test]
#[should_panic(expected = "an explicit case must differ from the default")]
fn values_rejects_a_false_continuation_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    // A false continuation labels its key with the default label, which the
    // store refuses as an explicit case: the elsewhere region already
    // carries FALSE.
    idx.values(
        Level::new(0),
        &[1],
        &[Function::FALSE],
        &mut Labels::default(),
    );
}

#[test]
fn rejected_children_release_their_acquired_partitions() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use crate::backend::{Backend, Column};
    use crate::bdd::BddStore;
    use crate::sparse::SparseTable;

    struct SharedBackend<B>(Arc<B>);

    impl<B: Backend> Backend for SharedBackend<B> {
        fn column(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Column>> {
            Arc::clone(&self.0).column(sort)
        }
    }

    fn leaked<B: Backend + 'static>(store: Arc<B>, cases: bool) -> bool {
        let sort = ScalarSort::bit_vec(1);
        let observer = Arc::clone(&store).column(sort).unwrap();
        let partition = {
            let mut cx = Context::new();
            let backend = cx.register_backend("observed", SharedBackend(store));
            let c = cx.column(backend, sort);
            let idx = intern(&cx, &[c, c]);
            let early = at(&idx, 0, &[1], &[&t(&idx)]);
            let p = cx.set(c, &[1]);
            let rejected = catch_unwind(AssertUnwindSafe(|| {
                if cases {
                    idx.cases(
                        Level::new(1),
                        Function::FALSE,
                        c,
                        &[(p.function(), early.function)],
                    )
                } else {
                    at(&idx, 1, &[1], &[&early])
                }
            }));
            assert!(rejected.is_err());
            idx.head(early.function).1.function
        };
        // All legitimate index and predicate owners, including the context,
        // have dropped. A successful decrement exposes an orphaned reference.
        catch_unwind(AssertUnwindSafe(|| observer.dec_ref(partition))).is_ok()
    }

    let results = [
        leaked(Arc::new(BddStore::with_capacity(64, 128, 1)), false),
        leaked(Arc::new(BddStore::with_capacity(64, 128, 1)), true),
        leaked(Arc::new(SparseTable::with_capacity(64, 128, 1)), false),
        leaked(Arc::new(SparseTable::with_capacity(64, 128, 1)), true),
    ];
    assert_eq!(results, [false; 4], "BDD values/cases, sparse values/cases");
}

#[test]
fn overlapping_default_cases_are_checked_before_normalization() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    for backend in ["bdd", "sparse"] {
        let mut cx = Context::default();
        let c = column(&mut cx, backend, 1);
        let idx = intern(&cx, &[c, c]);
        let p = cx.set(c, &[1]);
        let empty = cx.f(c);
        let deep = at(&idx, 1, &[1], &[&t(&idx)]);
        for cases in [
            [
                (p.function(), Function::FALSE),
                (p.function(), Function::TRUE),
            ],
            [
                (p.function(), Function::TRUE),
                (p.function(), Function::FALSE),
            ],
            [
                (p.function(), deep.function),
                (p.function(), Function::TRUE),
            ],
        ] {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    idx.cases(Level::new(0), Function::FALSE, c, &cases)
                }))
                .is_err(),
                "{backend} accepted conflicting cases"
            );
        }
        for (cases, expected) in [
            (
                [
                    (p.function(), Function::FALSE),
                    (p.function(), Function::FALSE),
                ],
                false,
            ),
            (
                [
                    (p.function(), Function::TRUE),
                    (p.function(), Function::TRUE),
                ],
                true,
            ),
            (
                [
                    (empty.function(), Function::FALSE),
                    (p.function(), Function::TRUE),
                ],
                true,
            ),
        ] {
            let root = idx.cases(Level::new(0), Function::FALSE, c, &cases);
            for a in 0..2 {
                for b in 0..2 {
                    assert_eq!(root.contains(&[a, b]), expected && a == 1);
                }
            }
        }
        // A backward continuation remains valid when normalization removes
        // the requested level and therefore retains no backward edge.
        let early = at(&idx, 0, &[1], &[&t(&idx)]);
        assert_eq!(
            at(&idx, 1, &[0, 1], &[&early, &early]).function,
            early.function
        );
        assert_eq!(
            idx.cases(Level::new(1), early.function, c, &[]).function,
            early.function
        );
        assert_eq!(
            idx.cases(
                Level::new(1),
                early.function,
                c,
                &[(p.function(), early.function)]
            )
            .function,
            early.function
        );
    }
}

#[test]
#[should_panic(expected = "outer-index child must follow its parent level")]
fn values_rejects_a_child_above_its_level_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let low = at(&idx, 0, &[1], &[&t(&idx)]);
    at(&idx, 1, &[2, 3], &[&low, &t(&idx)]);
}

#[test]
fn a_single_slot_map_collapses_to_its_continuation() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let deep = at(&idx, 1, &[3], &[&t(&idx)]);
    // Every value of the level leads to the same continuation, so the level
    // vanishes.
    let keys: Vec<u64> = (0..16).collect();
    let conts = vec![deep.function; 16];
    let root = idx.values(Level::new(0), &keys, &conts, &mut Labels::default());
    assert_eq!(root.function, deep.function);
    assert_eq!(idx.top(root.function), Level::new(1));
}

// --- copying suffixes -----------------------------------------------------

#[test]
fn copy_suffixes_rebuilds_roots_over_trailing_columns() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let source = intern(&cx, &[a, a]);
    let target = intern(&cx, &[a, a, a]);
    assert!(target.copy_suffixes(&[]).is_empty());
    // `c` is shared by both source roots and by two slots, so the memo
    // serves it three times.
    let c = at(&source, 1, &[3], &[&t(&source)]);
    let x = at(&source, 0, &[0, 1], &[&c, &c]);
    let y = at(&source, 0, &[2], &[&c]);
    let copies = target.copy_suffixes(&[&x, &y, &source.inc_ref(Function::TRUE)]);
    assert_eq!(copies.len(), 3);
    assert!(copies[2].function.is_t());
    assert_eq!(target.top(copies[0].function), Level::new(1));
    assert!(copies[0].contains(&[9, 0, 3]) && !copies[0].contains(&[9, 2, 3]));
    assert!(copies[1].contains(&[9, 2, 3]) && !copies[1].contains(&[9, 0, 3]));
}

#[test]
#[should_panic(expected = "source must be no wider than target")]
fn copy_suffixes_rejects_a_wider_source_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let wide = intern(&cx, &[a, a]);
    intern(&cx, &[a]).copy_suffixes(&[&wide]);
}

#[test]
#[should_panic(expected = "native copying requires one context")]
fn copy_suffixes_requires_one_context_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let mut other = Context::default();
    let b = column(&mut other, "sparse", 4);
    let source = intern(&other, &[b]);
    intern(&cx, &[a]).copy_suffixes(&[&source]);
}

#[test]
#[should_panic(expected = "sources require one index")]
fn copy_suffixes_requires_one_source_index_is_rejected() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 4), column(&mut cx, "bdd", 4));
    let (x, y) = (intern(&cx, &[a]), intern(&cx, &[b]));
    intern(&cx, &[a, a]).copy_suffixes(&[&x, &y]);
}

#[test]
#[should_panic(expected = "suffix columns differ")]
fn copy_suffixes_requires_identical_suffix_columns_is_rejected() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 4), column(&mut cx, "bdd", 4));
    let source = intern(&cx, &[b]);
    intern(&cx, &[a, a]).copy_suffixes(&[&source]);
}

// --- per-key minimum ------------------------------------------------------

#[test]
fn min_at_keeps_the_least_value_under_every_key_path() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let index = intern(&cx, &[a, a, a]);
    // Key 1 stores (3, {5, 6}) and (1, {7}); key 2 skips both lower levels.
    let five_six = at(&index, 2, &[5, 6], &[&t(&index), &t(&index)]);
    let seven = at(&index, 2, &[7], &[&t(&index)]);
    let under_one = at(&index, 1, &[1, 3], &[&seven, &five_six]);
    let root = at(&index, 0, &[1, 2], &[&under_one, &t(&index)]);
    let min = root.min_at(Level::new(1));
    let mut expected: BTreeSet<Vec<u64>> = (0..16).map(|x| vec![2, 0, x]).collect();
    expected.insert(vec![1, 1, 7]);
    assert_eq!(ground(&min), expected);
    // The result is canonical: it equals the directly built root, token for
    // token.
    let direct = at(
        &index,
        0,
        &[1, 2],
        &[
            &at(&index, 1, &[1], &[&seven]),
            &at(&index, 1, &[0], &[&t(&index)]),
        ],
    );
    assert_eq!(min.function, direct.function);
}

#[test]
fn min_at_retains_parent_owned_descendants_across_growth() {
    for backend in ["sparse", "bdd"] {
        for skipped in [false, true] {
            let mut cx = Context::default();
            let a = column(&mut cx, backend, 4);
            // The source fills the edge arena exactly. Constructing its minimum
            // must maintain the index while the source alone retains its children.
            cx.layouts.lock().unwrap().push(Arc::new(super::IndexData {
                columns: [a, a, a].into(),
                shared: Shared::new(super::State::with_capacity(
                    8,
                    if skipped { 4 } else { 6 },
                    (0..2).map(|_| super::IndexNode {
                        level: Level::TERMINAL,
                        partition: Function::FALSE,
                        hole: u32::MAX,
                    }),
                )),
            }));
            let index = intern(&cx, &[a, a, a]);
            let child = at(&index, 2, &[7], &[&t(&index)]);
            let middle = if skipped {
                child.clone()
            } else {
                at(&index, 1, &[1, 3], &[&child, &child])
            };
            let root = at(&index, 0, &[2], &[&middle]);
            let token = child.function;
            drop(child);
            drop(middle);
            assert!(!index.read().has_refs(token));
            assert_eq!(growths(&index), 0);
            let result = root.min_at(Level::new(1));
            assert!(growths(&index) > 0);
            assert_eq!(
                ground(&result),
                BTreeSet::from([vec![2, u64::from(!skipped), 7]])
            );
            assert!(root.contains(&[2, 3, 7]));
        }
    }
}

#[test]
fn min_at_of_a_true_root_selects_the_least_value() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let index = intern(&cx, &[a, a, a]);
    let min = t(&index).min_at(Level::new(1));
    assert_eq!(index.top(min.function), Level::new(1));
    assert_eq!(min.count(), 256_u32.into());
    assert!(min.contains(&[5, 0, 9]) && !min.contains(&[5, 1, 9]));
    assert!(
        index
            .inc_ref(Function::FALSE)
            .min_at(Level::new(0))
            .function
            .is_f()
    );
}

#[test]
fn min_at_merges_classes_with_one_minimum() {
    let mut cx = Context::default();
    let (b, a) = (column(&mut cx, "bdd", 4), column(&mut cx, "sparse", 4));
    let index = intern(&cx, &[b, a]);
    let two_seven = at(&index, 1, &[2, 7], &[&t(&index), &t(&index)]);
    let two_nine = at(&index, 1, &[2, 9], &[&t(&index), &t(&index)]);
    // Two symbolic classes with different candidates but the same least
    // value.
    let cells = |keys: &[u64], children: &[&IndexRoot]| at(&index, 0, keys, children);
    let low: Vec<_> = (0..8).collect();
    let all: Vec<_> = (0..16).collect();
    let both: Vec<&IndexRoot> = [&two_seven; 8].into_iter().chain([&two_nine; 8]).collect();
    let covering = cells(&all, &both).min_at(Level::new(1));
    // The classes collapse to one continuation, and their union is the
    // universe, so the symbolic level disappears.
    assert_eq!(index.top(covering.function), Level::new(1));
    assert_eq!(covering.count(), 16_u32.into());
    assert!(covering.contains(&[3, 2]) && covering.contains(&[12, 2]));
    assert!(!covering.contains(&[3, 7]) && !covering.contains(&[12, 9]));
    // A partial class keeps the symbolic level, with one slot for the
    // covered worlds.
    let partial = cells(&low, &[&two_seven; 8]).min_at(Level::new(1));
    assert_eq!(index.top(partial.function), Level::new(0));
    assert_eq!(partial.count(), 8_u32.into());
    assert!(partial.contains(&[3, 2]) && !partial.contains(&[12, 2]));
}

#[test]
#[should_panic(expected = "symbols have no numeric ordering")]
fn min_at_rejects_a_symbol_level_is_rejected() {
    let mut cx = Context::default();
    let sparse = cx.backend("sparse");
    let s = cx.column(sparse, ScalarSort::symbol());
    t(&intern(&cx, &[s])).min_at(Level::new(0));
}

#[test]
#[should_panic(expected = "level exceeds the index arity")]
fn min_at_rejects_a_level_past_the_arity_is_rejected() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    t(&intern(&cx, &[a])).min_at(Level::new(1));
}

// --- native Boolean operations --------------------------------------------

#[test]
fn apply_bool_simplifies_swaps_and_memoizes() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (x, y) = (
        at(&idx, 0, &[1], &[&t(&idx)]),
        at(&idx, 0, &[2], &[&t(&idx)]),
    );
    // Constant and identical operands never reach a level.
    assert!(
        idx.apply_bool(BoolOp::And, x.function, Function::FALSE)
            .function
            .is_f()
    );
    assert_eq!(
        idx.apply_bool(BoolOp::Or, x.function, x.function).function,
        x.function
    );
    // A commutative operation orders its operands, so both spellings share
    // one result.
    let left = idx.apply_bool(BoolOp::Or, x.function, y.function);
    let right = idx.apply_bool(BoolOp::Or, y.function, x.function);
    assert_eq!(left.function, right.function);
    assert!(left.contains(&[1, 0]) && left.contains(&[2, 0]) && !left.contains(&[3, 0]));
    // AndNot is not commutative and keeps its operand order.
    assert_eq!(
        idx.apply_bool(BoolOp::AndNot, left.function, y.function)
            .function,
        x.function
    );
}

#[test]
fn apply_bool_aligns_partitions_and_descends_one_level_at_a_time() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (p, q) = (
        at(&idx, 1, &[1], &[&t(&idx)]),
        at(&idx, 1, &[2], &[&t(&idx)]),
    );
    // Overlapping cells with different continuations: the level's
    // partitions must be aligned.
    let left = at(&idx, 0, &[0, 1], &[&p, &q]);
    let right = at(&idx, 0, &[1, 2], &[&p, &q]);
    let both = idx.apply_bool(BoolOp::Or, left.function, right.function);
    assert!(both.contains(&[0, 1]) && both.contains(&[1, 1]) && both.contains(&[1, 2]));
    assert!(both.contains(&[2, 2]) && !both.contains(&[3, 1]) && !both.contains(&[2, 1]));
    // One operand starts deeper, so it repeats itself under every slot of
    // the other. A commutative operation orders its operands first, so the
    // deeper one appears on either side depending on the operation.
    let deep = idx.apply_bool(BoolOp::Or, left.function, p.function);
    assert!(deep.contains(&[3, 1]) && deep.contains(&[0, 1]) && !deep.contains(&[3, 2]));
    let kept = idx.apply_bool(BoolOp::AndNot, left.function, p.function);
    assert!(kept.contains(&[1, 2]) && !kept.contains(&[0, 1]) && !kept.contains(&[3, 1]));
    // Both operands split the level the same way: one partition serves the
    // whole frame.
    let same = idx.apply_bool(
        BoolOp::Or,
        left.function,
        at(&idx, 0, &[0, 1], &[&q, &p]).function,
    );
    assert!(same.contains(&[0, 1]) && same.contains(&[0, 2]) && !same.contains(&[2, 1]));
}

#[test]
fn apply_bool_quotients_and_collapses_agreeing_levels() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (p, q) = (
        at(&idx, 1, &[1], &[&t(&idx)]),
        at(&idx, 1, &[2], &[&t(&idx)]),
    );
    // Mirrored continuations agree after the operation, so two of the three
    // slots merge.
    let left = at(&idx, 0, &[0, 1], &[&p, &q]);
    let right = at(&idx, 0, &[0, 1], &[&q, &p]);
    let merged = idx.apply_bool(BoolOp::Or, left.function, right.function);
    let (_, partition, _) = idx.head(merged.function);
    assert_eq!(
        partition.slots, 2,
        "the equal slots are quotiented together"
    );
    assert!(merged.contains(&[0, 1]) && merged.contains(&[1, 2]) && !merged.contains(&[2, 1]));
    // Every slot agrees, so the level is unconstrained and vanishes.
    let rest: Vec<u64> = (2..16).collect();
    let all = idx.apply_bool(
        BoolOp::Or,
        merged.function,
        at(&idx, 0, &rest, &[&p; 14]).function,
    );
    let full = idx.apply_bool(
        BoolOp::Or,
        all.function,
        at(&idx, 0, &rest, &[&q; 14]).function,
    );
    assert_eq!(idx.top(full.function), Level::new(1));
}

#[test]
fn apply_bool_classifies_a_wide_frame_through_the_probe_table() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 8), column(&mut cx, "sparse", 16));
    let idx = intern(&cx, &[a, b]);
    // Node tokens are handed out in creation order, so keeping every eighth
    // node spreads the frame's children over many tokens; with two buckets
    // per child, some of them share a bucket and are told apart by the
    // probe.
    let deep: Vec<IndexRoot> = (0..512).map(|k| at(&idx, 1, &[k], &[&t(&idx)])).collect();
    let picked: Vec<&IndexRoot> = deep.iter().step_by(8).collect();
    let keys: Vec<u64> = (0..picked.len() as u64).collect();
    let left = at(&idx, 0, &keys, &picked);
    // Every slot of the wider operand keeps its own continuation, so the
    // frame classifies the existing spread-out children rather than freshly
    // numbered results.
    let right = at(&idx, 0, &[0], &[picked[0]]);
    let both = idx.apply_bool(BoolOp::Or, left.function, right.function);
    assert_eq!(both.function, left.function);
    assert!(both.contains(&[0, 0]) && both.contains(&[3, 24]) && !both.contains(&[3, 25]));
}

#[test]
fn apply_bool_restarts_its_memo_on_another_container() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 4), column(&mut cx, "bdd", 4));
    let (one, two) = (intern(&cx, &[a]), intern(&cx, &[b]));
    let u = at(&two, 0, &[5], &[&t(&two)]);
    let v = at(&two, 0, &[6], &[&t(&two)]);
    // Seed a stale entry under this container's actual normalized operand key.
    // Matching the epoch and staying within the bound isolates container
    // invalidation from the other reasons restart clears the memo.
    SCRATCH.with_borrow_mut(|s| {
        s.memo.clear();
        s.memo.insert(
            (
                BoolOp::Or,
                u.function.min(v.function),
                u.function.max(v.function),
            ),
            Function::FALSE,
        );
        s.container = one.data.shared.id;
        s.epoch = two.data.shared.epoch();
        s.bound = 1;
        assert_ne!(s.container, two.data.shared.id);
        assert!(s.memo.len() <= s.bound);
    });
    let w = two.apply_bool(BoolOp::Or, u.function, v.function);
    assert_eq!(ground(&w), BTreeSet::from([vec![5], vec![6]]));
}

#[test]
fn apply_bool_restarts_its_memo_after_growth() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a]);
    let x = at(&idx, 0, &[1], &[&t(&idx)]);
    let y = at(&idx, 0, &[2], &[&t(&idx)]);
    drop(idx.apply_bool(BoolOp::Or, x.function, y.function));
    let mut first = true;
    idx.data.shared.retry((&cx, &idx.data.columns), |_| {
        if first {
            first = false;
            Err(NeedGrowth::Nodes)
        } else {
            Ok(())
        }
    });
    // Collection reclaimed the result, so a stale memo entry would return a
    // reclaimed token.
    let again = idx.apply_bool(BoolOp::Or, x.function, y.function);
    assert_eq!(ground(&again), BTreeSet::from([vec![1], vec![2]]));
}

#[test]
fn apply_bool_drops_a_memo_past_its_bound() {
    let mut cx = Context::default();
    let a = column(&mut cx, "sparse", 4);
    let idx = intern(&cx, &[a, a]);
    let (p, q) = (
        at(&idx, 1, &[1], &[&t(&idx)]),
        at(&idx, 1, &[2], &[&t(&idx)]),
    );
    let left = at(&idx, 0, &[0, 1], &[&p, &q]);
    // Every entry is one too many, so the next operation on the same
    // container starts over.
    set_memo_bound(0);
    let once = idx.apply_bool(BoolOp::Or, left.function, p.function);
    let twice = idx.apply_bool(BoolOp::Or, left.function, q.function);
    assert!(once.contains(&[3, 1]) && twice.contains(&[3, 2]));
    let (l, r) = (left.function.min(p.function), left.function.max(p.function));
    assert!(SCRATCH.with_borrow(|s| !s.memo.contains_key(&(BoolOp::Or, l, r))));
}

#[test]
fn apply_bool_matches_set_algebra_on_every_backend() {
    let mut seed = 0x1234_5678_u64;
    let mut next = move |n: u64| {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 33) % n
    };
    for kinds in [
        ["bdd", "bdd", "bdd"],
        ["bdd", "sparse", "bdd"],
        ["sparse", "bdd", "sparse"],
    ] {
        let mut cx = Context::default();
        let columns: Vec<_> = kinds.iter().map(|k| column(&mut cx, k, 4)).collect();
        let idx = intern(&cx, &columns);
        for round in 0..20 {
            // Narrow prefixes make operands share whole cells.
            let m = if round % 3 == 0 { 3 } else { 16 };
            let mut sets: [BTreeSet<Vec<u64>>; 2] = Default::default();
            for set in &mut sets {
                for _ in 0..next(40) {
                    set.insert(vec![next(m), next(m), next(16)]);
                }
            }
            let [l, r] = sets
                .each_ref()
                .map(|set| build(&idx, 0, &set.iter().cloned().collect::<Vec<_>>()));
            for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor, BoolOp::AndNot] {
                let want: BTreeSet<_> = match op {
                    BoolOp::And => sets[0].intersection(&sets[1]).cloned().collect(),
                    BoolOp::Or => sets[0].union(&sets[1]).cloned().collect(),
                    BoolOp::Xor => sets[0].symmetric_difference(&sets[1]).cloned().collect(),
                    BoolOp::AndNot => sets[0].difference(&sets[1]).cloned().collect(),
                };
                let got = idx.apply_bool(op, l.function, r.function);
                assert_eq!(ground(&got), want, "{kinds:?} {op:?} round {round}");
                // Canonical: the directly built relation is the same node.
                let direct = build(&idx, 0, &want.into_iter().collect::<Vec<_>>());
                assert_eq!(got.function, direct.function, "{kinds:?} {op:?}");
            }
        }
    }
}

// --- growth ---------------------------------------------------------------

#[test]
fn growth_collects_and_retries_interning() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 32), column(&mut cx, "sparse", 32));
    let idx = intern(&cx, &[a, b]);
    // More distinct nodes than the container was created with: interning
    // fails, the container collects and grows under exclusive access, and
    // the attempt is retried.
    let held: Vec<IndexRoot> = (0..70_000u64)
        .map(|k| at(&idx, 1, &[k], &[&t(&idx)]))
        .collect();
    assert!(growths(&idx) > 0, "the container was maintained");
    assert!(held.iter().all(|n| idx.top(n.function) == Level::new(1)));
    assert!(held[69_999].contains(&[5, 69_999]) && !held[69_999].contains(&[5, 0]));
}

#[test]
fn growth_restarts_an_operation_that_failed_below_its_root() {
    let mut cx = Context::default();
    let (a, b) = (column(&mut cx, "sparse", 4), column(&mut cx, "sparse", 32));
    let idx = intern(&cx, &[a, b, b]);
    // A result that widens by one slot per step, under a level that both
    // operands split the same way: the edge arena runs out while the deeper
    // frame interns, so the shallower frame releases its own partition and
    // reports the growth its caller retries.
    let mut root = idx.inc_ref(Function::FALSE);
    for k in 0..800u64 {
        let deep = at(&idx, 2, &[k], &[&t(&idx)]);
        let leaf = at(&idx, 0, &[0], &[&at(&idx, 1, &[k], &[&deep])]);
        root = idx.apply_bool(BoolOp::Or, root.function, leaf.function);
    }
    assert!(growths(&idx) > 0, "the edge arena was maintained");
    assert!(root.contains(&[0, 0, 0]) && root.contains(&[0, 799, 799]));
    assert!(!root.contains(&[0, 0, 1]) && !root.contains(&[1, 0, 0]));
}
