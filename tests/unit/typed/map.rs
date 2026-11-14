//! Checks for typed bitwise maps over integer endpoints.

use std::collections::BTreeSet;

use super::{self as map};
use crate::context::Context;
use crate::typed::{IntoQuery, Rel, Sorted};

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &crate::rel::Rel) -> BTreeSet<Vec<u64>> {
    let cx = rel.root.context();
    let widths: Vec<u32> = (0..rel.root.arity())
        .map(|i| cx.resolve(rel.root.column(i)).sort().bit_width().unwrap())
        .collect();
    let mut out = BTreeSet::new();
    let mut row = vec![0; widths.len()];
    loop {
        if rel.root.contains(&row) {
            out.insert(row.clone());
        }
        let Some(i) = (0..row.len()).rev().find(|&i| row[i] + 1 < 1 << widths[i]) else {
            return out;
        };
        row[i] += 1;
        row[i + 1..].fill(0);
    }
}

/// The ground rows of a binary relation over one-word columns, as pairs.
fn pairs(rel: &crate::rel::Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

#[test]
fn bitwise_maps_transform_integer_ranges() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    let mut bytes = Rel::<u8, u8>::new_set(&cx);
    bytes.insert_rows([0b0101_u8, 0b1100]);
    // Selections key the image by the original value.
    assert_eq!(
        pairs(&(&bytes).select(map::bitor(0b0010_u8)).collect().raw),
        [(0b0101, 0b0111), (0b1100, 0b1110)].into()
    );
    assert_eq!(
        pairs(&(&bytes).select(map::bitand(0b0110_u8)).collect().raw),
        [(0b0101, 0b0100), (0b1100, 0b0100)].into()
    );
    assert_eq!(
        pairs(&(&bytes).select(map::bitxor(0b1111_u8)).collect().raw),
        [(0b0101, 0b1010), (0b1100, 0b0011)].into()
    );

    // Signedness leaves the operation on the stored bits unchanged.
    let mut signed = Rel::<i8, i8>::new_set(&cx);
    signed.insert_rows([-1_i8, 1]);
    assert_eq!(
        pairs(&(&signed).select(map::bitand(0b0110_i8)).collect().raw),
        [(1, 0), (0xff, 0b0110)].into()
    );
}
