//! Internal storage checks for logical declarations.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::Sorts;

#[test]
fn conflicting_declarations_do_not_grow_the_store() {
    let sorts = Sorts::default();
    let flag = sorts.bool();
    let members = [("a", flag), ("b", flag)];
    let sort = sorts.record("Pair", &members);
    let capacity = sorts.state.read(()).node_capacity();

    for i in 0..=capacity {
        let label = format!("member{i}");
        let conflict = catch_unwind(AssertUnwindSafe(|| {
            sorts.record("Pair", &[(&label, flag), ("b", flag)])
        }));
        assert!(conflict.is_err(), "conflicting definition was accepted");
    }

    assert_eq!(sorts.state.read(()).node_capacity(), capacity);
    assert_eq!(sorts.record("Pair", &members), sort);
}
