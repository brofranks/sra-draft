use std::ops::ControlFlow;
use std::sync::Arc;

use sra::backend::{Alignment, Backend, Column, Function, MapSpec, Refinement, Values};
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;

fn col(width: u32) -> Box<dyn Column> {
    Arc::new(SparseStore::with_capacity(64, 256, 1))
        .column(ScalarSort::bit_vec(width))
        .unwrap()
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn empty_refine_rejects_released_partition() {
    let c = col(4);
    let f = c.set(&[1]);
    c.dec_ref(f);
    let _ = c.refine(
        f,
        Function::FALSE,
        &mut Refinement::default(),
        &mut |_| true,
        &mut |_, _| ControlFlow::Continue(()),
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "cannot use an unreferenced sparse node")]
fn partial_map_rejects_released_filter() {
    let c = col(4);
    let p = c.set(&[1]);
    let q = c.set(&[2]);
    c.dec_ref(q);
    let map = MapSpec::Rust(Arc::new(|_, _| false));
    let _ = c.preimage(&map, p, q);
}

#[test]
fn public_scratch_reuse_after_discarding_lookup_table() {
    let c = col(4);
    let f = c.set(&[1]);
    let mut scratch = Refinement::default();
    let run = |s| {
        c.refine(
            f,
            Function::TRUE,
            s,
            &mut |_| true,
            &mut |_, _| ControlFlow::Continue(()),
        )
    };
    assert!(run(&mut scratch).is_continue());
    scratch.slots.clear();
    assert!(run(&mut scratch).is_continue());
    c.dec_ref(f);
}

#[test]
fn alignment_reports_the_exact_suffix() {
    let c = col(4);
    let f = c.set(&[1]);
    let g = c.set(&[2]);
    let mut labels = vec![(9, 9)];
    let h = c.align(f, g, &mut labels, &mut Alignment::default());
    assert_eq!(labels, [(9, 9), (0, 0), (1, 0), (0, 1)]);
    for f in [f, g, h] {
        c.dec_ref(f);
    }
}

#[test]
#[should_panic(expected = "there must be one case label per key")]
fn sparse_values_rejects_missing_label() {
    let c = col(65);
    let _ = c.values(9, &[1, 0, 0, 1], &[7], &mut Vec::new(), &mut Values::default());
}
