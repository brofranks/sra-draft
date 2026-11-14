//! Checks for membership masks: masks span words, operations stay in one
//! context, universes bound their indices, and rows check the width.

use std::collections::HashSet;

use super::{FiniteSet, Universe};
use crate::context::Context;
use crate::typed::{Rel, Sorted};

/// The members `0..70`, spanning two words.
struct Digits;

impl Universe for Digits {
    type Member = u8;
    const LEN: u32 = 70;
    fn index(_: u64, m: &u8) -> Option<u32> {
        (*m < 70).then_some(u32::from(*m))
    }
}

type DigitSet = FiniteSet<Digits>;

/// A universe whose mapping violates its declared length.
struct Broken;

impl Universe for Broken {
    type Member = u8;
    const LEN: u32 = 2;
    fn index(_: u64, m: &u8) -> Option<u32> {
        Some(u32::from(*m))
    }
}

// --- membership and set operations --------------------------------------

#[test]
fn membership_masks_span_multiple_words() {
    let cx = Context::default();
    let mut set = DigitSet::from_members(&cx, [1_u8, 65]);
    assert!(set.contains(65_u8));
    assert!(!set.contains(64_u8));
    assert_eq!(set.as_ref(), [2, 2]);
    assert_eq!(set, DigitSet::from_words(&cx, [2, 2]));
    assert_eq!(
        format!("{set:?}"),
        format!("FiniteSet {{ context: {}, words: [2, 2] }}", cx.id)
    );
    assert_eq!(
        HashSet::from([set.clone(), set.clone()]).len(),
        1,
        "equal sets hash alike"
    );
    set.remove(65_u8);
    set.insert(3_u8);
    assert_eq!(set.as_ref(), [10, 0]);
    let other = DigitSet::from_members(&cx, [1_u8, 65]);
    assert_eq!(set.union_with(&other).as_ref(), [10, 2]);
    assert_eq!(set.intersect_with(&other).as_ref(), [2, 0]);
    assert_eq!(DigitSet::full(&cx).as_ref(), [u64::MAX, 0x3f]);
    assert_eq!(
        DigitSet::empty(&cx).union_with(&DigitSet::full(&cx)),
        DigitSet::full(&cx)
    );
}

#[test]
#[should_panic(expected = "member must belong to this set universe")]
fn members_must_belong_to_the_universe_is_rejected() {
    DigitSet::empty(&Context::default()).contains(70_u8);
}

#[test]
#[should_panic(expected = "universe index exceeds its declared length")]
fn universe_indices_are_bounded_is_rejected() {
    FiniteSet::<Broken>::empty(&Context::default()).contains(2_u8);
}

#[test]
#[should_panic(expected = "words do not encode this set universe")]
fn words_must_fit_the_universe_is_rejected() {
    DigitSet::from_words(&Context::default(), [0, 0x40]);
}

#[test]
fn set_equality_includes_the_creation_context() {
    let cx = Context::default();
    let other = Context::default();
    let set = DigitSet::from_members(&cx, [1_u8]);
    assert!(set.contains(1_u8));
    assert_ne!(set, DigitSet::from_members(&other, [1_u8]));
    assert_eq!(set.clone(), set);
}

#[test]
#[should_panic(expected = "set belongs to another context")]
fn set_operations_stay_within_one_context_is_rejected() {
    let cx = Context::default();
    let other = Context::default();
    DigitSet::empty(&cx).union_with(&DigitSet::empty(&other));
}

#[test]
#[should_panic(expected = "set belongs to another context")]
fn intersections_stay_within_one_context_is_rejected() {
    let cx = Context::default();
    let other = Context::default();
    DigitSet::empty(&cx).intersect_with(&DigitSet::empty(&other));
}

// --- rows ---------------------------------------------------------------

#[test]
fn sets_pack_into_rows() {
    let mut cx = Context::default();
    DigitSet::bind(&mut cx);
    let mut sets = Rel::<DigitSet, DigitSet>::new_set(&cx);
    sets.insert(DigitSet::from_members(&cx, [1_u8]));
    assert!(sets.contains(DigitSet::from_members(&cx, [1_u8])));
    assert!(!sets.contains(DigitSet::empty(&cx)));
}

#[test]
#[should_panic(expected = "set belongs to another context")]
fn rows_reject_foreign_sets_is_rejected() {
    let mut cx = Context::default();
    DigitSet::bind(&mut cx);
    let foreign = DigitSet::empty(&Context::default());
    Rel::<DigitSet, DigitSet>::new_set(&cx).insert(foreign);
}

#[test]
#[should_panic(expected = "set value requires its universe width")]
fn rows_check_the_universe_width_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let set = FiniteSet::<Broken>::empty(&cx);
    Rel::<u8, u8>::new_set(&cx).insert(set);
}
