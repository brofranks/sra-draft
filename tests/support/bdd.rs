//! BDD fixtures shared by the public and white-box suites.

use super::{Algebra, Arc, Backend, BddStore, Function, ScalarSort};

/// Four value bits fit in one level. A nonconstant diagram has one branch
/// over terminals.
pub const BV4: ScalarSort = ScalarSort::bv(4);
/// Two levels.
pub const BV12: ScalarSort = ScalarSort::bv(12);
/// Three levels, allowing two paths to reach the same pair of branch nodes.
pub const BV18: ScalarSort = ScalarSort::bv(18);

/// A store with room for neither a node nor an edge beyond its terminals.
pub fn tiny() -> Arc<BddStore> {
    Arc::new(BddStore::with_capacity(2, 0, 1))
}

/// A store with room for the shared three-level fixture without maintenance.
pub fn roomy() -> Arc<BddStore> {
    Arc::new(BddStore::with_capacity(128, 512, 1))
}

/// Binds the store to one scalar sort.
pub fn bind(store: &Arc<BddStore>, sort: ScalarSort) -> Box<dyn Algebra> {
    Arc::clone(store).bind(sort).unwrap()
}

/// Every member of a predicate over a one-word sort.
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

/// The predicate testing `mask` bits of a one-word sort against `value`.
pub fn bit(alg: &dyn Algebra, mask: u64, value: u64) -> Function {
    alg.masked_eq(&[mask], &[value])
}

/// A three-level diagram whose two level-one nodes share both of their
/// level-two children.
///
/// A product can reach the same operand pair by two routes and reuse its
/// memoized result. The `low` pair names the two accepted low-bit values.
pub fn shared(alg: &dyn Algebra, low: (u64, u64)) -> Function {
    let (hi, mid) = (bit(alg, 1 << 12, 0), bit(alg, 1 << 6, 0));
    let (a, b) = (bit(alg, 0x3F, low.0), bit(alg, 0x3F, low.1));
    let (yes, no) = (alg.ite(mid, a, b), alg.ite(mid, b, a));
    let root = alg.ite(hi, yes, no);
    for f in [hi, mid, a, b, yes, no] {
        alg.dec_ref(f);
    }
    root
}

/// The concrete meaning of [`shared`], as plain integer arithmetic.
pub fn shared_holds(low: (u64, u64), x: u64) -> bool {
    let (hi, mid) = ((x >> 12) & 1 == 0, (x >> 6) & 1 == 0);
    x & 0x3F == if hi == mid { low.0 } else { low.1 }
}

/// Points spread over a three-level sort, for comparing a predicate with its
/// meaning.
pub fn samples() -> impl Iterator<Item = u64> {
    let spread = (0..1 << 18).step_by(701).chain(0..200);
    // Exercise every high/middle branch and its shared children under Miri.
    let branches = [0, 64, 4096, 4160]
        .into_iter()
        .flat_map(|base| (0..8).map(move |low| base + low));
    branches.chain(spread.take(if cfg!(miri) { 0 } else { usize::MAX }))
}

/// The class number of each four-bit value.
pub fn classes(alg: &dyn Algebra, f: Function) -> Vec<u32> {
    (0..16).map(|v| alg.evaluate(f, &[v])).collect()
}

/// The partition labeling values below 16 by their residue modulo `n`, with
/// label zero elsewhere.
pub fn residues(alg: &dyn Algebra, n: u64) -> Function {
    let cases: Vec<(Function, u32)> = (1..n)
        .map(|r| (set(alg, (0..16).filter(|v| v % n == r)), r as u32))
        .collect();
    let root = alg.cases(0, &cases, &mut Vec::new());
    for (f, _) in cases {
        alg.dec_ref(f);
    }
    root
}
