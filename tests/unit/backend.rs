//! White-box tests for the refinement grouping buffers.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{DECLINED, Groups, UNREACHED};

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
fn a_reserved_group_index_is_rejected_before_publication() {
    // Seed the selected count at the first reserved index. The limit must
    // be checked before a lookup entry or a group buffer uses that index.
    let mut r = Groups {
        len: DECLINED as usize,
        ..Groups::default()
    };
    assert!(r.group(0, 0, &mut |_| false).is_none());
    let panic = catch_unwind(AssertUnwindSafe(|| {
        r.group(1, 0, &mut |_| true);
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"selected group index overflow")
    );
    assert_eq!(r.classes, [DECLINED, UNREACHED]);
    assert_eq!(r.len, DECLINED as usize);
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
fn restart_retains_the_lookup_through_the_highest_class() {
    let mut r = Groups::default();
    r.group(3, 0, &mut |_| true).unwrap().push(8);
    r.restart();
    // A lower class reuses the lookup sized by the earlier call.
    assert!(r.group(1, 0, &mut |_| true).unwrap().is_empty());
    assert_eq!(r.classes.len(), 4);
    assert!(
        r.classes
            .iter()
            .enumerate()
            .all(|(class, &i)| (class == 1) == (i != UNREACHED))
    );
}
