//! Sparse fixtures shared by the public and white-box suites.

use super::{Arc, Backend, Column, Function, ScalarSort, SparseStore, Values};

/// A four-bit packed sort.
pub const BV4: ScalarSort = ScalarSort::bit_vec(4);
/// A six-bit packed sort with room for the shared partition's keys.
pub const BV6: ScalarSort = ScalarSort::bit_vec(6);

/// The table bound to one sort.
pub fn col(store: &Arc<SparseStore>, d: ScalarSort) -> Box<dyn Column> {
    Arc::clone(store).column(d).unwrap()
}

/// Every member of a one-word sort.
pub fn members(c: &dyn Column, p: Function) -> Vec<u64> {
    let mut cursor = c.members(p);
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.push(v[0]);
    }
    out
}

/// The predicate for the sorted values of a one-word sort.
pub fn set(c: &dyn Column, values: impl IntoIterator<Item = u64>) -> Function {
    c.set(&values.into_iter().collect::<Vec<_>>())
}

/// Whether the value is a member of the predicate.
pub fn holds(c: &dyn Column, f: Function, x: u64) -> bool {
    c.evaluate(f, &[x]) != 0
}

/// A partition of a one-word sort with cells `{10, 30}`, `{20}`, `{40}` and
/// the rest, labeled with the given case labels.
pub fn parts(c: &dyn Column, default: u32, labels: [u32; 4]) -> (Function, Vec<u32>) {
    let mut out = Vec::new();
    let f = c.values(
        default,
        &[10, 20, 30, 40],
        &labels,
        &mut out,
        &mut Values::default(),
    );
    (f, out)
}
