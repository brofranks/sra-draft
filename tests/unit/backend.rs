//! White-box tests for the refinement grouping buffers.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::{DECLINED, Groups, UNREACHED};

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
