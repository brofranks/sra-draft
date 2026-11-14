//! One explicit shared context, source snapshots, and independent query
//! invocations.
//!
//! Workers use the ordinary CPS executor, and the caller chooses partitions
//! and unions results. This small fixture demonstrates usage and correctness
//! without measuring speedup.

use std::iter;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use sra::{BoolOp, Context, IntoQuery, Query, Rel};

/// Claims the next chunk index, preserving exhaustion.
fn claim(next: &AtomicUsize, chunks: usize) -> Option<usize> {
    next.try_update(Ordering::Relaxed, Ordering::Relaxed, |i| {
        if i < chunks { Some(i + 1) } else { None }
    })
    .ok()
}

/// Composes stored relations by splitting the declared source universe into
/// `chunks` contiguous ranges across scoped workers.
///
/// Each chunk keeps the original endpoint universes, and native union checks
/// compatible result indexes.
///
/// # Panics
///
/// Panics if `chunks` or `workers` is zero, or if the source domain is an
/// aggregate or anything other than a bitvector of at most 64 bits.
pub fn collect_partitioned(a: &Rel, b: &Rel, chunks: usize, workers: usize) -> Rel {
    assert!(
        chunks > 0 && workers > 0,
        "chunks and workers must be positive"
    );
    let width = a
        .domain
        .sort
        .scalar()
        .bit_width()
        .filter(|&w| w <= 64)
        .expect("source domain must be a bitvector of at most 64 bits");
    let n = 1_u128 << width;
    let chunks = (chunks as u128).min(n) as usize;
    let next = AtomicUsize::new(0);
    let cx = a.root.context();
    thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    iter::from_fn(|| claim(&next, chunks))
                        .map(|i| {
                            // Inclusive bounds cover even the full 64-bit
                            // universe without overflow.
                            let lo = (n * i as u128 / chunks as u128) as u64;
                            let hi = (n * (i as u128 + 1) / chunks as u128 - 1) as u64;
                            // Apply the source restriction before traversal,
                            // leaving input shards uncollected.
                            Query::id(cx, a.domain.clone())
                                .with(lo..=hi)
                                .select(a)
                                .select(b)
                                .collect()
                        })
                        .reduce(|left, right| left.apply_bool(BoolOp::Or, &right))
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .reduce(|left, right| left.apply_bool(BoolOp::Or, &right))
            .unwrap()
    })
}

fn main() {
    let mut cx = Context::default();
    let node = cx.tuple_decl("Node", [("id", cx.bv_sort(3))]);
    let id = node.accessors[0];
    let backend = cx.backend("bdd");
    cx.set_representation(id, backend);
    let mut edges = Rel::new(&cx, id, id);
    edges.insert_rows([(0_u64, 1_u64), (0, 2), (1, 3), (2, 3), (4, 1)]);

    let query = edges.select(&edges);
    let expected = query.collect();
    let mut paths = Rel::new(&cx, id, id);
    paths.insert_rows([(0_u64, 3_u64), (4, 3)]);
    assert!(expected == paths);

    // The query owns source snapshots. All workers borrow the same context and
    // query, while each terminal creates its own state and each worker owns
    // its callback.
    thread::scope(|scope| {
        let workers = [0, 1].map(|_| {
            scope.spawn(|| {
                assert_eq!(cx.bv_sort(3), id.resolve(&cx).range);
                assert!(!query.is_empty());
                let mut visits = 0;
                let _ = query.drive(&mut |_, _| {
                    visits += 1;
                    // A nested invocation has independent state too.
                    assert!(!query.is_empty());
                    ControlFlow::Break(())
                });
                assert_eq!(visits, 1);
                query.collect()
            })
        });
        for worker in workers {
            assert!(worker.join().unwrap() == expected);
        }
    });

    // Workers reuse configured fields and intern logical sorts through the
    // shared context. Partition the source coordinate before the first
    // occurrence, keeping the second whole.
    const WORKERS: usize = 2;
    const CHUNKS_PER_WORKER: usize = 2;
    let result = collect_partitioned(&edges, &edges, WORKERS * CHUNKS_PER_WORKER, WORKERS);
    assert!(result == expected);
    // An uneven count yields multi-value ranges of different lengths.
    assert!(collect_partitioned(&edges, &edges, 3, WORKERS) == expected);

    // Separately restrict just the first occurrence by its range, using
    // symbolic restrictions. Keep the second occurrence whole: applying the
    // same range restriction to both loses the path 4 -> 1 -> 3. These checks
    // collect complete compositions.
    let results = [
        edges.with(..2_u64).select(&edges).collect(),
        edges.with(2_u64..).select(&edges).collect(),
    ];
    // Both disjoint intermediate-key ranges connect 0 to 3. Composition hides
    // their witnesses, so merge by exact native union. This decomposition uses
    // positive operators.
    assert!(results[0].contains((0_u64, 3_u64)) && results[1].contains((0_u64, 3_u64)));
    assert!(results[0].apply_bool(BoolOp::Or, &results[1]) == expected);

    // A later write leaves the built query's snapshot alone and reaches the
    // next query.
    edges.insert((3_u64, 5_u64));
    assert!(query.collect() == expected);
    assert!(edges.select(&edges).collect().contains((1_u64, 5_u64)));
    println!("PASS: shared queries and caller-partitioned collection");
}

#[cfg(test)]
#[path = "../tests/unit/concurrency.rs"]
mod tests;
