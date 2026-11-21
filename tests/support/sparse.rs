//! Sparse fixtures shared by the public and white-box suites.

use super::{Algebra, Arc, Backend, Function, ScalarSort, SparseStore, Values};

/// A four-bit packed sort.
pub const BV4: ScalarSort = ScalarSort::bv(4);
/// A six-bit packed sort with room for the shared partition's keys.
pub const BV6: ScalarSort = ScalarSort::bv(6);

/// Binds the store to one sort.
pub fn bind(store: &Arc<SparseStore>, d: ScalarSort) -> Box<dyn Algebra> {
    Arc::clone(store).bind(d).unwrap()
}

/// Every member of a one-word sort.
pub fn members(alg: &dyn Algebra, p: Function) -> Vec<u64> {
    let mut cursor = alg.members(p);
    let mut out = Vec::new();
    while let Some(v) = cursor.next() {
        out.push(v[0]);
    }
    out
}

/// The predicate for the sorted values of a one-word sort.
pub fn set(alg: &dyn Algebra, values: impl IntoIterator<Item = u64>) -> Function {
    alg.set(&values.into_iter().collect::<Vec<_>>())
}

/// Whether the value is a member of the predicate.
pub fn holds(alg: &dyn Algebra, f: Function, x: u64) -> bool {
    alg.evaluate(f, &[x]) != 0
}

/// A partition with the given labels at keys 10, 20, 30, and 40 of a one-word
/// sort, and the default label elsewhere.
pub fn parts(alg: &dyn Algebra, default: u32, labels: [u32; 4]) -> (Function, Vec<u32>) {
    let mut out = Vec::new();
    let f = alg.values(
        default,
        &[10, 20, 30, 40],
        &labels,
        &mut out,
        &mut Values::default(),
    );
    (f, out)
}
