//! Checks for bit-level maps, Rust functions, and the equation they register
//! while running.
//!
//! The `image` fixture collects the pairs that a set relates through a map
//! operand. Local fixtures bind bit-vector rows, build sets, and list ground
//! pairs.

#[path = "support/query.rs"]
mod fixture;

use fixture::{bv, tiny};

use std::collections::BTreeSet;

use sra::context::Context;
use sra::map;
use sra::query::{IntoQuery, Operand, Query};
use sra::rel::Rel;
use sra::row::Row;

fn image(s: &Rel, m: impl Operand) -> BTreeSet<(u64, u64)> {
    pairs(&s.select(m).collect())
}

/// A 4-bit set holding `values`, over the already bound row `four`.
fn set(cx: &Context, four: &Row, values: impl IntoIterator<Item = u8>) -> Rel {
    let mut s = Rel::new_set(cx, four.clone());
    s.insert_rows(values);
    s
}

/// Every ground row of a relation over one-word columns, by membership
/// alone.
fn rows(rel: &Rel) -> BTreeSet<Vec<u64>> {
    let cx = rel.root().index().context();
    let widths: Vec<u32> = (0..rel.root().index().arity())
        .map(|i| {
            cx.resolve(rel.root().index().algebra(i))
                .sort()
                .bv_width()
                .unwrap()
        })
        .collect();
    let mut out = BTreeSet::new();
    let mut row = vec![0; widths.len()];
    loop {
        if rel.root().contains(&row) {
            out.insert(row.clone());
        }
        let Some(i) = (0..row.len()).rev().find(|&i| row[i] + 1 < 1 << widths[i]) else {
            return out;
        };
        row[i] += 1;
        row[i + 1..].fill(0);
    }
}

/// The ground rows of a binary relation over 4-bit columns, as pairs.
fn pairs(rel: &Rel) -> BTreeSet<(u64, u64)> {
    rows(rel).into_iter().map(|r| (r[0], r[1])).collect()
}

// --- bit-level maps -------------------------------------------------------

#[test]
fn bit_maps_relate_each_input_to_its_image() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let s = set(&cx, &four, [1, 2, 3]);
    assert_eq!(image(&s, map::copy()), [(1, 1), (2, 2), (3, 3)].into());
    assert_eq!(image(&s, map::bvor(4_u8)), [(1, 5), (2, 6), (3, 7)].into());
    assert_eq!(image(&s, map::bvand(2_u8)), [(1, 0), (2, 2), (3, 2)].into());
    assert_eq!(image(&s, map::bvxor(1_u8)), [(1, 0), (2, 3), (3, 2)].into());
    assert_eq!(
        image(
            &set(&cx, &four, [1, 2, 3, 9]),
            map::assign(0b1100_u8, 0b0100_u8)
        ),
        [(1, 5), (2, 6), (3, 7), (9, 5)].into()
    );
    // A partial Rust function drops inputs without an image.
    assert_eq!(
        image(
            &s,
            map::func(|x, y| {
                y[0] = x[0] * 2;
                x[0] != 3
            })
        ),
        [(1, 2), (2, 4)].into()
    );
    // The image side is a fresh cell: composing back through the inverse map
    // recovers the set.
    let back = s.select(map::bvxor(1_u8)).select(map::bvxor(1_u8));
    assert_eq!(pairs(&back.collect()), [(1, 1), (2, 2), (3, 3)].into());
    assert_eq!(
        s.select(map::bvor(4_u8))
            .with(&set(&cx, &four, [6]))
            .count()
            .to_string(),
        "1"
    );
}

#[test]
#[should_panic(expected = "assigned value has bits outside its mask")]
fn assigned_value_outside_its_mask_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    set(&cx, &four, [1]).select(map::assign(0b1100_u8, 0b0001_u8));
}

// --- Rust functions -------------------------------------------------------

#[test]
#[should_panic(expected = "a BDD store cannot apply a Rust map")]
fn rust_map_on_a_symbolic_column_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    let t = cx.sorts().record("T", &[("x", four.sort())]);
    let column = cx.bind_field(cx.sorts().project(t, 0), "bdd");
    fixture::indexes(&cx, column);
    Row::bind(&mut cx, t);
    let mut s = Rel::new_set(&cx, cx.sorts().project(t, 0));
    s.insert(1_u8);
    s.select(map::func(|x, y| {
        y[0] = x[0];
        true
    }))
    .count();
}

// --- equations ------------------------------------------------------------

#[test]
#[should_panic(expected = "equations between cells must not form a cycle")]
fn equating_map_endpoints_is_rejected() {
    let mut cx = tiny();
    let four = bv(&mut cx, 4);
    // x -> (x, x ^ 1), then restricting the pair to the diagonal equates the
    // map's two cells.
    let q = Query::id(&cx, four.clone());
    let diag = q.clone().and(q);
    Query::id(&cx, four)
        .and(map::bvxor(1_u8))
        .with(diag.inv())
        .count();
}
