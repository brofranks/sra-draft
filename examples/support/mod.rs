//! Verification helpers shared by the examples, included with `mod support;`.
//!
//! Every example states its expected outcome independently of the engine: an
//! ordinary Rust computation, or a table written out by hand. These helpers
//! hold the three idioms that independence keeps producing - a worklist
//! fixpoint, a comparison against rows rebuilt in the engine's own layout, and
//! decoding the members of a predicate - so each example spends its lines on
//! what it actually demonstrates. The directory has no `main.rs`, so Cargo
//! does not treat it as an example target; each example uses a subset, hence
//! `dead_code`.

#![allow(dead_code)]

use std::collections::{BTreeSet, VecDeque};
use std::net::Ipv4Addr;

use sra::typed::{Rel, Sorted};
use sra::{IntoRow, Predicate};

/// The least set containing `seeds` and closed under `step`, by an ordinary
/// worklist.
///
/// Reference computations use this: no engine relation, predicate,
/// or frontier takes part.
pub fn closure<T: Ord + Clone>(
    seeds: impl IntoIterator<Item = T>,
    step: impl Fn(&T) -> Vec<T>,
) -> BTreeSet<T> {
    let mut seen: BTreeSet<T> = seeds.into_iter().collect();
    let mut queue: VecDeque<T> = seen.iter().cloned().collect();
    while let Some(state) = queue.pop_front() {
        for next in step(&state) {
            if seen.insert(next.clone()) {
                queue.push_back(next);
            }
        }
    }
    seen
}

/// Asserts that `actual` holds exactly `rows`.
///
/// The expectation is rebuilt with
/// `empty_like`, so both sides keep the declared field backends and the same
/// set or binary universe.
pub fn assert_rows<D: Sorted, R: Sorted>(
    actual: &Rel<D, R>,
    rows: impl IntoIterator<Item = impl IntoRow>,
) {
    let mut expected = actual.empty_like();
    expected.insert_rows(rows);
    assert!(
        *actual == expected,
        "relation differs from the expected rows"
    );
}

/// Every member of a scalar predicate, as raw words per member.
///
/// Callers
/// interpret the words at their own layer: an integer, a location, or an
/// interned name.
pub fn members(p: &Predicate) -> Vec<Vec<u64>> {
    let mut cursor = p.context().resolve(p.column()).members(p.function());
    let mut out = Vec::new();
    let mut value = Vec::new();
    while cursor.next(&mut value) {
        out.push(value.clone());
    }
    out
}

/// The paper's literal address form.
pub fn ip(s: &str) -> Ipv4Addr {
    s.parse().unwrap()
}
