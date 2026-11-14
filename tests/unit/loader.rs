//! Checks for the SMT-LIB guard loader.
//!
//! `INPUT` declares the sole input, `input` binds it to an 8-bit BDD column,
//! `load` renders errors as strings, and `accepted` lists a guard's values.

use std::io;

use crate::context::{Context, Predicate};
use crate::loader;
use crate::rel::Rel;
use crate::row::{Bv, Domain, Row};

const INPUT: &str = "(declare-const x (_ BitVec 8))\n";

/// A bound 8-bit BDD input; the loader binds the file's sole declaration to
/// it.
fn input(cx: &mut Context) -> Row {
    let sort = cx.bv_sort(8);
    let t = cx.tuple_decl("Loaded", [("x", sort)]);
    let backend = cx.backend("bdd");
    cx.set_representation(t.accessors[0], backend);
    t.accessors[0].bind(cx)
}

fn load(cx: &Context, x: &Row, source: &str, roots: &[&str]) -> Result<Vec<Predicate>, String> {
    loader::smt2(
        cx,
        x.clone(),
        &mut source.as_bytes() as &mut dyn io::Read,
        roots,
    )
    .map_err(|e| e.to_string())
}

/// The concrete input values accepted by a loaded guard.
fn accepted(p: &Predicate) -> Vec<u64> {
    (0..256).filter(|&v| p.contains(&[v])).collect()
}

// --- definitions ------------------------------------------------------------

#[test]
fn definitions_evaluate_over_the_declared_input() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    let source = format!(
        "(set-logic QF_BV) ; logic comment\n{INPUT}\
        (define-fun low () Bool (= ((_ extract 3 0) x) #x5))\n\
        (define-fun high () Bool (= #b10 ((_ extract 7 6) x)))\n\
        (define-fun |odd bit| () Bool (= (_ bv257 8) x))\n\
        (define-fun nested () Bool (= ((_ extract 0 0) ((_ extract 3 1) x)) #b1))\n\
        (define-fun all () Bool (and true low high (not false)))\n\
        (define-fun any () Bool (or low (ite high all |odd bit|)))\n\
        (define-fun none () Bool (or false false)) ; trailing comment"
    );
    let guards = load(
        &cx,
        &x,
        &source,
        &["low", "high", "odd bit", "nested", "all", "any", "none"],
    )
    .unwrap();
    assert_eq!(
        accepted(&guards[0]),
        (0..256).filter(|v| v & 15 == 5).collect::<Vec<_>>()
    );
    assert_eq!(accepted(&guards[1]), (128..192).collect::<Vec<_>>());
    assert_eq!(accepted(&guards[2]), [1]);
    assert_eq!(
        accepted(&guards[3]),
        (0..256).filter(|v| v & 2 != 0).collect::<Vec<_>>()
    );
    assert_eq!(
        accepted(&guards[4]),
        (128..192).filter(|v| v & 15 == 5).collect::<Vec<_>>()
    );
    assert_eq!(
        accepted(&guards[5]),
        (0..256)
            .filter(|v| v & 15 == 5 || *v == 1)
            .collect::<Vec<_>>()
    );
    assert!(accepted(&guards[6]).is_empty());
    assert!(
        load(&cx, &x, "(declare-const x (_ BitVec 8))", &[])
            .unwrap()
            .is_empty()
    );
}

// --- located errors ---------------------------------------------------------

#[test]
fn errors_locate_the_offending_token() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    assert_eq!(
        load(&cx, &x, "(set-logic QF_LIA)", &[]).err().unwrap(),
        "1:12: unsupported logic: QF_LIA"
    );
    assert_eq!(
        load(
            &cx,
            &x,
            &format!("{INPUT}(define-fun f () Bool\n  (and true))"),
            &[]
        )
        .err()
        .unwrap(),
        "3:4: invalid arity for And"
    );
    assert_eq!(
        load(&cx, &x, INPUT, &["missing"]).err().unwrap(),
        "2:1: unknown guard root \"missing\""
    );
    for (source, error) in [
        ("(assert true)", "2:2: unsupported command: assert"),
        (
            "(define-fun f () Bool (let ((a true)) a))",
            "2:24: unsupported binder: let",
        ),
        (
            "(define-fun f () Bool (! true :named n))",
            "2:24: unsupported term syntax: !",
        ),
        (
            "(define-fun f () Bool (bvadd x x))",
            "2:24: unsupported operator: bvadd",
        ),
        (
            "(define-fun f () Bool ((_ zero_extend 1) x))",
            "2:27: unsupported operator: zero_extend",
        ),
    ] {
        let source = format!("{INPUT}{source}");
        assert_eq!(load(&cx, &x, &source, &[]).err().unwrap(), error);
    }
    struct Failing;
    impl io::Read for Failing {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("disk gone"))
        }
    }
    let e = loader::smt2(&cx, x.clone(), &mut Failing as &mut dyn io::Read, &["f"])
        .err()
        .unwrap();
    assert_eq!((e.line, e.column, e.message.as_str()), (1, 1, "disk gone"));
}

#[test]
fn invalid_definition_bodies_locate_the_body() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    let source = format!("{INPUT}(define-fun f () Bool\n  x)");
    let error = load(&cx, &x, &source, &["f"]).err().unwrap();
    assert_eq!(error, "3:3: definition must have sort Bool");
}

#[test]
fn rejected_syntax_reports_its_message() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    let def = |body: &str| format!("{INPUT}(define-fun f () Bool {body})");
    for (source, message) in [
        (
            "(define-fun f () Bool true)".to_string(),
            "expected one bit-vector input declaration",
        ),
        (
            "(declare-const x Bool)".into(),
            "input must have a bit-vector sort",
        ),
        (
            "(declare-const x (_ BitVec 4))".into(),
            "declared input width differs from the column",
        ),
        (
            "(declare-const x (_ BitVec 0))".into(),
            "bit-vector width must be positive",
        ),
        (
            "(declare-const x (_ BitVec 4294967296))".into(),
            "bit width or index exceeds u32",
        ),
        (
            "(declare-const x (_ BitVec a))".into(),
            "expected a natural numeral",
        ),
        (
            "(declare-const x (_ FloatingPoint 8))".into(),
            "unsupported sort: FloatingPoint",
        ),
        (
            "(declare-const x (Array))".into(),
            "unsupported sort: Array",
        ),
        ("(declare-const x (5))".into(), "expected a symbol"),
        ("(declare-const 5 (_ BitVec 8))".into(), "expected a symbol"),
        (
            "(declare-const true (_ BitVec 8))".into(),
            "reserved symbol",
        ),
        (
            "(declare-const x (_ BitVec 8)".into(),
            "unexpected end of input",
        ),
        (
            "(declare-const x (_ BitVec 8) x)".into(),
            "expected \")\", found \"x\"",
        ),
        ("((".into(), "expected a command name"),
        ("(:kw".into(), "expected a command name"),
        ("(assert true)".into(), "unsupported command: assert"),
        (
            format!("{INPUT}(declare-const y (_ BitVec 8))"),
            "expected a Boolean definition after the input declaration",
        ),
        (
            format!("{INPUT}(define-fun x () Bool true)"),
            "duplicate symbol",
        ),
        (
            format!("{INPUT}(define-fun f () Bool true)(define-fun f () Bool true)"),
            "duplicate symbol",
        ),
        (
            format!("{INPUT}(define-fun f ((a Bool)) Bool true)"),
            "unsupported definition: function parameters",
        ),
        (
            format!("{INPUT}(define-fun f (a) Bool true)"),
            "expected a parameter list",
        ),
        (
            format!("{INPUT}(define-fun f () Int true)"),
            "unsupported sort: Int",
        ),
        (
            format!("{INPUT}(define-fun f () (_ BitVec 8) x)"),
            "definition must have sort Bool",
        ),
        (def("x"), "definition must have sort Bool"),
        (def("(_ bv5 8)"), "definition must have sort Bool"),
        (def("(_ bv5 0)"), "empty bit-vector literal"),
        (def("(_ bv 8)"), "unsupported indexed literal: bv"),
        (def("(_ bvx 8)"), "unsupported indexed literal: bvx"),
        (def("(_ 5 8)"), "expected a symbol"),
        (
            def("((_ zero_extend 2) x)"),
            "unsupported operator: zero_extend",
        ),
        (
            def("((extract 1 0) x)"),
            "expected \"_\", found \"extract\"",
        ),
        (def("(let ((a true)) a)"), "unsupported binder: let"),
        (def("(! true :named x)"), "unsupported term syntax: !"),
        (def("(bvadd x x)"), "unsupported operator: bvadd"),
        (def("(extract x)"), "unsupported operator: extract"),
        (def(")"), "unexpected closing parenthesis"),
        (def("y"), "unknown symbol: y"),
        (def("1"), "unsupported literal: Numeral"),
        (def("1.5"), "unsupported literal: Decimal"),
        (def("\"a\"\"b\""), "unsupported literal: String"),
        (def(":kw"), "expected a symbol"),
        (def("_"), "expected a symbol"),
        (
            format!("{INPUT}(define-fun f () Bool"),
            "unexpected end of input",
        ),
        (
            format!("{INPUT}(define-fun f () Bool ("),
            "unexpected end of input",
        ),
        (
            format!("{INPUT}(define-fun f () Bool ((_"),
            "unexpected end of input",
        ),
        (def("((_ 5"), "expected a symbol"),
        (def("((_ extract x"), "expected a natural numeral"),
        (def("((_ extract 1 x"), "expected a natural numeral"),
        (def("((_ extract 1 0 x"), "expected \")\", found \"x\""),
        (
            format!("{INPUT}(define-fun f () Bool (_"),
            "unexpected end of input",
        ),
        (def("(_ bv5 x)"), "expected a natural numeral"),
        (def("(_ bv5 8 x)"), "expected \")\", found \"x\""),
        (def("(5)"), "expected a symbol"),
        ("x".into(), "expected \"(\", found \"x\""),
        ("(".into(), "unexpected end of input"),
        ("(set-logic".into(), "unexpected end of input"),
        ("(set-logic 5)".into(), "expected a symbol"),
        ("(set-logic QF_BV x)".into(), "expected \")\", found \"x\""),
        (
            "(set-logic QF_BV) (assert true)".into(),
            "unsupported command: assert",
        ),
        ("(declare-const".into(), "unexpected end of input"),
        ("(declare-const x".into(), "unexpected end of input"),
        ("(declare-const x 5)".into(), "expected a symbol"),
        ("(declare-const x (".into(), "unexpected end of input"),
        ("(declare-const x (_".into(), "unexpected end of input"),
        ("(declare-const x (_ 5 8))".into(), "expected a symbol"),
        (
            "(declare-const x (_ BitVec".into(),
            "unexpected end of input",
        ),
        (
            "(declare-const x (_ BitVec 8 x))".into(),
            "expected \")\", found \"x\"",
        ),
        (format!("{INPUT}("), "unexpected end of input"),
        (
            format!("{INPUT}(assert true)"),
            "unsupported command: assert",
        ),
        (format!("{INPUT}(define-fun"), "unexpected end of input"),
        (
            format!("{INPUT}(define-fun 5 () Bool true)"),
            "expected a symbol",
        ),
        (
            format!("{INPUT}(define-fun f x"),
            "expected \"(\", found \"x\"",
        ),
        (format!("{INPUT}(define-fun f ("), "unexpected end of input"),
        (def("#b"), "invalid token: \"#b\""),
        (def("#b2"), "invalid token: \"#b2\""),
        (def("#x"), "invalid token: \"#x\""),
        (def("#xg"), "invalid token: \"#xg\""),
        (def(":"), "invalid token: \":\""),
        (def("5abc"), "invalid token: \"5abc\""),
        (def("1."), "invalid token: \"1.\""),
        (def("01.5"), "invalid token: \"01.5\""),
        (def("1.5a"), "invalid token: \"1.5a\""),
        (def("a.b"), "unknown symbol: a.b"),
        (def("|abc"), "unterminated quoted token"),
        (def("|a\\b|"), "invalid character in quoted token"),
        (def("|a\x7fb|"), "invalid character in quoted token"),
        (def("|a\x01b|"), "invalid character in quoted token"),
        (def("|a\tb|"), "unknown symbol: a\tb"),
        (def("\"a\\b\""), "unsupported literal: String"),
        (def("|a|b"), "expected a delimiter after quoted token"),
        (def("(not)"), "invalid arity for Not"),
        (def("(= x)"), "invalid arity for Eq"),
        (def("(ite true)"), "invalid arity for Ite"),
        (def("((_ extract 1 0))"), "invalid arity for Extract(1, 0)"),
        (
            def("((_ extract 0 1) x)"),
            "invalid operands or bit bounds for Extract(0, 1)",
        ),
        (
            def("((_ extract 8 0) x)"),
            "invalid operands or bit bounds for Extract(8, 0)",
        ),
        (
            def("((_ extract 1 0) true)"),
            "invalid operands or bit bounds for Extract(1, 0)",
        ),
        (def("(= x #b1)"), "invalid operands or bit bounds for Eq"),
        (def("(= #b1 x)"), "invalid operands or bit bounds for Eq"),
        (def("(not x)"), "invalid operands or bit bounds for Not"),
        (
            def("(ite x true false)"),
            "invalid operands or bit bounds for Ite",
        ),
        (
            def("(and true x)"),
            "Boolean connective requires Bool operands",
        ),
        (
            def("(or x true true)"),
            "Boolean connective requires Bool operands",
        ),
        (
            format!("{INPUT}(define-fun f () Bool true"),
            "unexpected end of input",
        ),
    ] {
        let error = load(&cx, &x, &source, &[]).err().unwrap();
        assert!(
            error.ends_with(&format!(": {message}")),
            "{source:?} gave {error:?}"
        );
    }
}

// --- guard files ------------------------------------------------------------

#[test]
fn guards_file_agrees_with_direct_construction() {
    let mut cx = Context::default();
    let sort = cx.bv_sort(3);
    let t = cx.tuple_decl("Route", [("world", sort)]);
    let backend = cx.backend("bdd");
    cx.set_representation(t.accessors[0], backend);
    let world = t.accessors[0].bind(&mut cx);
    // These views bind one existing column; extraction introduces no
    // independent variables.
    let x = Bv::new(&cx, world.clone());
    let prefix = x.extract(2, 2).eq(&cx, 1_u64);
    let a = x.extract(1, 1).eq(&cx, 1_u64);
    let b = x.extract(0, 0).eq(&cx, 1_u64);
    let permitted = prefix.ite(&b, &a);
    // The file's declaration binds to the same column. Repeated named roots
    // have independent owning handles, so dropping one or the loader's
    // temporaries does not invalidate another.
    let source = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/guards.smt2"));
    let mut guards = loader::smt2(
        &cx,
        world.clone(),
        source.as_slice(),
        &["permitted", "permitted"],
    )
    .unwrap();
    drop(guards.pop());
    for value in 0..8 {
        let expected = if value & 4 != 0 {
            value & 1 != 0
        } else {
            value & 2 != 0
        };
        assert_eq!(permitted.contains(&[value]), expected);
        assert_eq!(guards[0].contains(&[value]), expected);
    }
    let mut routes = Rel::new_set(&cx, world.clone());
    routes.insert(&guards[0]);
    drop(guards);
    assert!(routes.contains(2_u64) && !routes.contains(4_u64));
}

// --- literals ---------------------------------------------------------------

#[test]
fn literal_words_agree_with_numeric_modulo() {
    use num_bigint::BigUint;

    for width in [1, 3, 63, 64, 65, 127, 128, 129, 1025] {
        let modulus = BigUint::from(1_u8) << width;
        for (radix, digits) in [
            (2, "000101011001".repeat(100)),
            (16, "000123456789abcdef".repeat(30)),
            (10, "1234567890".repeat(30)),
        ] {
            let expected = BigUint::parse_bytes(digits.as_bytes(), radix).unwrap() % &modulus;
            let actual = loader::Parser::literal(&digits, radix, width).words;
            assert_eq!(
                BigUint::new(
                    actual
                        .iter()
                        .flat_map(|&v| [v as u32, (v >> 32) as u32])
                        .collect()
                ),
                expected
            );
            assert_eq!(actual.len(), width.div_ceil(64) as usize);
        }
    }
}

#[test]
fn literal_width_preserves_digit_counts_and_rejects_overflow() {
    for bits in [1_u32, 4] {
        for digits in [
            0,
            1,
            2,
            (u32::MAX / 4) as usize,
            (u32::MAX / 4) as usize + 1,
            u32::MAX as usize,
            usize::MAX,
        ] {
            let exact = digits as u128 * u128::from(bits);
            let expected = u32::try_from(exact).ok();
            assert_eq!(loader::literal_width(digits, bits), expected);
        }
    }
    let mut cx = Context::default();
    let x = input(&mut cx);
    let source = format!("{INPUT}(define-fun leading_zero () Bool (= x #x01))");
    let p = load(&cx, &x, &source, &["leading_zero"]).unwrap();
    assert_eq!(accepted(&p[0]), [1]);
    let source = format!("{INPUT}(define-fun wide () Bool (= x #x001))");
    assert!(load(&cx, &x, &source, &["wide"]).is_err());
}

#[test]
fn indexed_literals_wider_than_the_input_are_rejected_at_their_name() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    // The largest width stays cheap because rejection precedes allocation.
    for width in [9, u32::MAX] {
        let source = format!("{INPUT}(define-fun f () Bool (= x (_ bv1 {width})))");
        assert_eq!(
            load(&cx, &x, &source, &[]).err().unwrap(),
            "2:31: bit-vector literal exceeds the input width"
        );
    }
    // A literal within the input but wider than its view stays an operand
    // error.
    let source = format!("{INPUT}(define-fun f () Bool (= ((_ extract 3 0) x) (_ bv1 8)))");
    assert_eq!(
        load(&cx, &x, &source, &[]).err().unwrap(),
        "2:24: invalid operands or bit bounds for Eq"
    );
}

// --- comments ---------------------------------------------------------------

#[test]
fn comments_end_at_a_carriage_return_or_a_line_feed() {
    let mut cx = Context::default();
    let x = input(&mut cx);
    for end in ["\r", "\n", "\r\n"] {
        let source = format!(
            "{INPUT}(define-fun f () Bool (or (= x #x01) ; note{end}(= x #x02)\n(= x #x03)))"
        );
        assert_eq!(
            accepted(&load(&cx, &x, &source, &["f"]).unwrap()[0]),
            [1, 2, 3]
        );
    }
    // Text after a bare carriage return is still checked.
    let source = format!("{INPUT}; note\r(assert true)");
    let error = load(&cx, &x, &source, &[]).err().unwrap();
    assert!(
        error.ends_with(": unsupported command: assert"),
        "{error:?}"
    );
}
