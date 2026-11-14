//! Stored field correlations, relational replacement, and exact duplicate
//! projections.
//!
//! This example is executable documentation for the source-refinement
//! executor.

use sra::map::{bvand, bvor, bvxor};
use sra::query::project;
use sra::row::Domain;
use sra::{BoolOp, Bv, Context, IntoQuery, Query, Rel, pred};

fn main() {
    // Backend configuration and required storage bindings precede the first
    // owning relation or predicate. Logical declarations and sort interning
    // remain available through the shared context.
    let mut cx = Context::default();
    let pair = cx.tuple_decl("Pair", [("x", cx.bv_sort(3)), ("y", cx.bv_sort(3))]);
    let x = pair.accessors[0];
    let y = pair.accessors[1];
    let bdd = cx.backend("bdd");
    let xc = cx.set_representation(x, bdd);
    cx.set_representation(y, bdd);
    let row = pair.sort.bind(&mut cx);
    let unit = cx.unit_sort().bind(&mut cx);
    let wide = ["bdd", "sparse"].map(|name| {
        let schema = cx.tuple_decl(format!("Wide_{name}"), [("value", cx.bv_sort(70))]);
        let backend = cx.backend(name);
        cx.set_representation(schema.accessors[0], backend);
        schema.accessors[0]
    });
    let diagram = wide[0];

    // The stored set contains exactly (0,1), (1,2), and (2,3). Ordinary field
    // partitions retain this correlation through their continuations; no
    // symbolic graph is synthesized.
    let mut pairs = Rel::new_set(&cx, pair.sort);
    pairs.insert_rows([(0_u64, 1_u64), (1, 2), (2, 3)]);
    assert!(pairs.contains((0_u64, 1_u64)) && !pairs.contains((0_u64, 3_u64)));

    // Projection existentially hides x. The result already has y's canonical
    // field storage.
    let image = pairs.select(y).values().collect();
    let mut expected = Rel::new_set(&cx, y);
    expected.insert_rows([1_u64, 2, 3]);
    assert!(image == expected);

    // The old y is hidden by the range image, preserving the source witness
    // connecting x and new y.
    let mut next = Rel::new(&cx, y, y);
    next.insert_rows((0..8_u64).map(|value| (value, (value + 1) & 7)));
    let composed = pairs
        .select(Query::decl(&cx, x).and(Query::decl(&cx, y).select(&next)))
        .select(pair.constructor)
        .values()
        .collect();
    assert!(composed.contains((2_u64, 4_u64)));

    // Insertion is noninjective: old x=0 and x=1 merge, carrying both y
    // continuations.
    let mut add = Rel::new(&cx, x, x);
    add.insert_rows((0..8_u64).map(|value| (value, value | 1)));
    let merged = pairs
        .select(Query::decl(&cx, x).select(&add).and(y))
        .select(pair.constructor)
        .values()
        .collect();
    assert!(merged.contains((1_u64, 1_u64)) && merged.contains((1_u64, 2_u64)));

    // A map is an equation between two cells rather than a stored relation.
    // Hiding its input transfers the whole symbolic predicate at once, while
    // retaining both endpoints enumerates the output values, one exact branch
    // each, like an aliased output cell.
    let flipped = pairs.select(y).select(bvxor(1_u64)).values().collect();
    let mut expected = Rel::new_set(&cx, y);
    expected.insert_rows([0_u64, 3, 2]);
    assert!(flipped == expected);
    let both = pairs.select(y).values().select(bvor(4_u64)).collect();
    assert!(
        both.contains((1_u64, 5_u64))
            && both.contains((3_u64, 7_u64))
            && !both.contains((1_u64, 7_u64))
    );
    // Emptiness needs no enumeration: the map's preimage of the filtered
    // output empties the input.
    assert!(
        Query::id(&cx, y)
            .select(bvand(0_u64))
            .with(pred::ne(0_u64))
            .is_empty()
    );

    // The source-free identity ranges over the universe. Composition uses the
    // explicitly stored replacement pairs.
    let odd = Query::id(&cx, x).select(&add).values().collect();
    assert!(odd.contains(1_u64) && !odd.contains(0_u64));

    // Each product retains its continuation. Repeating the same predicate
    // means independent choices, while duplicate projections identify one
    // logical variable.
    let mut rectangle = Rel::new_set(&cx, pair.sort);
    let values = cx.set(xc, &[0, 2]);
    rectangle.insert([&values, &values]);
    assert!(rectangle.contains((0_u64, 2_u64)) && rectangle.contains((2_u64, 0_u64)));
    let rebuilt = pair
        .constructor
        .apply(rectangle.select(Query::decl(&cx, x).and(y)))
        .values()
        .collect();
    assert!(rebuilt == rectangle);
    let diagonal = rectangle
        .select(Query::decl(&cx, x).and(x))
        .values()
        .collect();
    assert!(diagonal.contains((0_u64, 0_u64)) && diagonal.contains((2_u64, 2_u64)));
    assert!(!diagonal.contains((0_u64, 2_u64)));
    let outside = diagonal.not();
    assert!(outside.contains((0_u64, 2_u64)) && !outside.contains((0_u64, 0_u64)));
    assert!(diagonal.apply_bool(BoolOp::Or, &outside).not().is_empty());

    // Reusing query syntax still binds two independent source occurrences and
    // hidden witnesses.
    let query = {
        let mut source = Rel::new(&cx, x, y);
        source.insert_rows([(0_u64, 1_u64), (0, 2)]);
        let q = (&source).into_query();
        q.clone().and(q)
    };
    let duplicated = query.collect();
    assert!(duplicated.contains((0_u64, 1_u64, 2_u64)));
    let _ = query.drive(&mut |_, r| {
        assert_eq!(r.cells.len(), 2);
        // Reentrant query construction is valid: no execution guard reaches
        // this callback.
        assert!(!Query::id(&cx, y).is_empty());
        std::ops::ControlFlow::Break(())
    });
    assert!(!query.is_empty());

    // Unit has one possible row, including after existential projection of a
    // nonempty source.
    let mut one = Rel::new_set(&cx, unit.clone());
    assert!(one.is_empty());
    one.insert(());
    assert!(one.contains(()));
    assert!(one.complement().is_empty());
    assert!(Query::id(&cx, unit.clone()).collect() == one);
    assert!(
        rectangle
            .select(&project(&cx, &row, [], unit))
            .values()
            .collect()
            == one
    );

    // Both backends preserve exact multiword values and zero-extend native
    // literals.
    for field in wide {
        let mut rel = Rel::new_set(&cx, field);
        rel.insert([5_u64, 32]);
        rel.insert(7_u64);
        assert!(rel.contains([5_u64, 32]) && rel.contains([7_u64, 0]));
        let id = Query::id(&cx, field);
        let copied = rel.select((&id).and(&id)).values().collect();
        assert!(copied.contains((&[5_u64, 32], &[5_u64, 32])));
        assert!(!copied.contains((&[5_u64, 32], &[7_u64, 0])));
    }

    // Shared predicate handles retain independent roots. Bit 69 exercises the
    // complete logical width.
    let guard = Bv::new(&cx, diagram).extract(69, 69).eq(&cx, 1_u64);
    let kept = guard.clone();
    drop(guard);
    assert!(kept.contains(&[0, 32]) && !kept.contains(&[0, 0]));
    let mut symbolic = Rel::new_set(&cx, diagram);
    symbolic.insert(&kept);
    assert!(symbolic.into_query().collect() == symbolic);
    println!(
        "PASS: correlations, relational updates, diagonals, unit, wide values, and direct \
         predicates"
    );
}

#[test]
fn main_passes() {
    main();
}
