//! Small scalar stores and index arenas for query fixtures.

use sra::bdd::BddStore;
use sra::context::{AlgebraId, Context};
use sra::index::Index;
use sra::row::Row;
use sra::sparse::SparseStore;

pub fn tiny() -> Context {
    let mut cx = Context::empty();
    cx.register_backend("bdd", BddStore::with_capacity(32, 64, 1));
    cx.register_backend("sparse", SparseStore::with_capacity(32, 64, 1));
    Index::with_capacity(&cx, &[], 32, 64);
    cx
}

/// Registers the homogeneous layouts used by the query fixtures.
pub fn indexes(cx: &Context, algebra: AlgebraId) {
    let columns = [algebra; 4];
    for n in 1..=columns.len() {
        Index::with_capacity(cx, &columns[..n], 32, 64);
    }
}

/// A bound bit-vector row with small indexes for its projections and joins.
pub fn bv(cx: &mut Context, w: u32) -> Row {
    let sort = cx.sorts().bv(w);
    let row = Row::bind(cx, sort);
    indexes(cx, row.scalar_algebra());
    row
}
