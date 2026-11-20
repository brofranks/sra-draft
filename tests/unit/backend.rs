//! White-box tests for the refinement grouping buffers.

use std::ops::ControlFlow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use super::{Backend, Groups, Refinement, UNREACHED};
use crate::scalar::ScalarSort;
use crate::sparse::SparseStore;

#[test]
fn grouping_preserves_selected_classes_and_forgets_declined_ones() {
    let mut r = Groups::default();
    // Class 2 is declined once. Accepted classes retain their groups.
    let mut select = |class| class != 2;
    r.group(0, 5, &mut select).unwrap().push(8);
    assert!(r.group(2, 5, &mut select).is_none());
    assert!(
        r.group(2, 5, &mut |_| panic!(
            "a declined class is never asked again"
        ))
        .is_none()
    );
    assert!(r.group(3, 6, &mut select).unwrap().is_empty());
    assert_eq!(r.group(0, 9, &mut select).unwrap().as_slice(), [8]);
    assert_eq!(r.reached(), 3);
    assert!(
        r.selected()
            .map(|(class, initial, _)| (class, initial))
            .eq([(0, 5), (3, 6)])
    );

    // The next call clears the lookup and reuses the existing buffers.
    r.restart();
    assert!(r.classes.iter().all(|&i| i == UNREACHED));
    assert!(r.group(3, 0, &mut select).unwrap().is_empty());
    assert_eq!(r.reached(), 1);
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((3, 0)));
}

#[test]
fn refinement_restart_clears_classes_after_a_selector_panic() {
    let mut r = Groups::default();
    r.group(3, 5, &mut |_| true).unwrap().extend([8, 9]);
    assert!(r.group(1, 5, &mut |_| false).is_none());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            r.group(2, 5, &mut |_| panic!("selector failed"));
        }))
        .is_err()
    );

    // Restart clears every class touched before the selector panicked.
    r.restart();
    assert_eq!(r.reached(), 0);
    assert!(r.selected().next().is_none());
    assert!(r.classes.iter().all(|&i| i == UNREACHED));
    let mut asked = Vec::new();
    let mut select = |class| {
        asked.push(class);
        true
    };
    assert!(r.group(1, 7, &mut select).unwrap().is_empty());
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((1, 7)));
    assert_eq!(r.reached(), 1);

    r.restart();
    assert!(r.group(3, 9, &mut select).unwrap().is_empty());
    assert_eq!(asked, [1, 3]);
    assert_eq!(r.selected().next().map(|(c, p, _)| (c, p)), Some((3, 9)));
}

#[test]
#[cfg(target_pointer_width = "32")]
fn an_unrepresentable_class_leaves_grouping_reusable() {
    let mut r = Groups::default();
    r.group(1, 1, &mut |_| true).unwrap().push(8);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            r.group(u32::MAX, 0, &mut |_| true);
        }))
        .is_err()
    );
    r.restart();
    assert_eq!(r.reached(), 0);
    assert!(r.group(1, 0, &mut |_| true).unwrap().is_empty());
}

#[test]
fn sparse_refinement_sizes_scratch_by_reached_classes() {
    let c = Arc::new(SparseStore::with_capacity(2, 0, 1))
        .column(ScalarSort::bit_vec(6))
        .unwrap();
    let f = c.set(&(0..64).step_by(4).collect::<Vec<_>>());
    let p = c.set(&[4, 5]);
    let mut scratch = Refinement::default();
    let mut found = Vec::new();
    assert!(
        c.refine(f, p, &mut scratch, &mut |_| true, &mut |class, build| {
            let cell = build();
            let mut cursor = c.members(cell);
            found.push((class, cursor.next().unwrap()[0]));
            assert!(cursor.next().is_none());
            c.dec_ref(cell);
            ControlFlow::Continue(())
        })
        .is_continue()
    );
    found.sort();
    assert_eq!(found, [(0, 5), (1, 4)]);
    assert_eq!(scratch.groups.classes.len(), 2);
    c.dec_ref(f);
    c.dec_ref(p);
}
