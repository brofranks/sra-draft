//! Expansion of `Schema` against the crate it generates code for.

use sra::typed::{IntoQuery, Record, Rel, Sorted};
use sra::{Context, ScalarSort, Schema};

// A plain record: no annotations, so the derive binds every member to its
// default storage.
#[derive(Schema)]
struct Route {
    src: u32,
    dst: u32,
}

// A record whose member is individually marked, so binding selects the
// symbolic backend for it. One field is a raw identifier, and another is named
// like the generated private input, which the derive renames around.
#[derive(Schema)]
pub struct Packet {
    #[symbolic]
    pub r#type: u8,
    pub __sra_input: u16,
}

#[test]
fn a_record_declares_its_members_and_accessors() {
    let mut cx = Context::default();
    Route::bind(&mut cx);
    let schema = Route::schema(&cx);
    assert_eq!(schema.sort, <Route as Sorted>::sort(&cx));
    assert_eq!(schema.sort.len, 2);
    assert_eq!(<Route as Record>::declaration(&cx).accessors.len(), 2);
    let mut routes = Rel::<Route, Route>::new_set(&cx);
    routes.insert(Route { src: 1, dst: 2 });
    assert!(routes.contains(Route { src: 1, dst: 2 }));
    // The generated view exposes each member as an expression, and its
    // operators.
    assert_eq!(
        (&routes).select(schema.src).filt(|s| s.eq(1_u32)).count(),
        1_u32.into()
    );
    assert_eq!(
        (&routes)
            .filt(|r: RouteView<_, _, _>| r.src.eq(1_u32))
            .count(),
        1_u32.into()
    );
    assert_eq!(
        (&routes)
            .filt(|r: RouteView<_, _, _>| r.dst.ne(2_u32))
            .count(),
        0_u32.into()
    );
    // The schema also projects around one member: two routes sharing `src`
    // share the key.
    let mut other = Rel::<Route, Route>::new_set(&cx);
    other.insert(Route { src: 1, dst: 5 });
    assert!(
        (&routes).select(schema.except(&cx, schema.dst)).values()
            == (&other).select(schema.except(&cx, schema.dst)).values()
    );
    assert!(
        (&routes).select(schema.except(&cx, schema.src)).values()
            != (&other).select(schema.except(&cx, schema.src)).values()
    );
    // A whole view re-assembles the record, so it is a query of its own.
    assert_eq!(
        (&routes).map(|r: RouteView<_, _, _>| r).count(),
        1_u32.into()
    );
}

#[test]
fn marked_members_bind_to_the_symbolic_backend() {
    let mut cx = Context::default();
    let decl = <Packet as Record>::declaration(&cx);
    Packet::bind(&mut cx);
    // Columns are interned per backend and sort, so the bound column
    // identifies the backend.
    let marked = cx.column(cx.backend("bdd"), ScalarSort::bit_vec(8));
    let unmarked = cx.column(cx.backend("sparse"), ScalarSort::bit_vec(16));
    assert_eq!(cx.field(decl.accessors[0]), Some(marked));
    assert_eq!(cx.field(decl.accessors[1]), Some(unmarked));
    // Binding again keeps the columns.
    Packet::bind(&mut cx);
    assert_eq!(cx.field(decl.accessors[0]), Some(marked));
    assert_eq!(cx.field(decl.accessors[1]), Some(unmarked));
    let schema = Packet::schema(&cx);
    assert_eq!(decl.sort, schema.sort);
    let mut packets = Rel::<Packet, Packet>::new_set(&cx);
    packets.insert(Packet {
        r#type: 3,
        __sra_input: 4,
    });
    assert!(packets.contains((3_u8, 4_u16)));
    assert_eq!(
        (&packets)
            .select(schema.r#type)
            .filt(|t| t.eq(3_u8))
            .count(),
        1_u32.into()
    );
    assert_eq!(
        (&packets)
            .select(schema.__sra_input)
            .filt(|u| u.eq(4_u16))
            .count(),
        1_u32.into()
    );
}

#[test]
fn an_explicit_representation_survives_binding() {
    let mut cx = Context::default();
    let decl = <Packet as Record>::declaration(&cx);
    let column = cx.set_representation(decl.accessors[0], cx.backend("sparse"));
    Packet::bind(&mut cx);
    assert_eq!(cx.field(decl.accessors[0]), Some(column));
}

#[test]
fn tuples_bind_nested_record_defaults_and_preserve_overrides() {
    fn check<T: Sorted>() {
        for backend in ["bdd", "sparse"] {
            let mut cx = Context::default();
            let decl = Packet::declaration(&cx);
            if backend == "sparse" {
                cx.set_representation(decl.accessors[0], cx.backend(backend));
            }
            let column = cx.column(cx.backend(backend), ScalarSort::bit_vec(8));
            T::bind(&mut cx);
            assert_eq!(cx.field(decl.accessors[0]), Some(column));
            T::bind(&mut cx);
            assert_eq!(cx.field(decl.accessors[0]), Some(column));
        }
    }
    check::<(Packet, u32)>();
    check::<(u32, Packet)>();
    check::<((Packet, u32), u16)>();
}

#[test]
fn generated_names_do_not_shadow_the_record() {
    macro_rules! check {
        ($name:ident) => {{
            #[derive(Schema)]
            struct $name {
                value: u8,
                other: u8,
            }
            let mut cx = Context::default();
            $name::bind(&mut cx);
            let schema = $name::schema(&cx);
            let mut rel = Rel::<$name, $name>::new_set(&cx);
            rel.insert($name { value: 1, other: 2 });
            assert_eq!(
                (&rel).map(|v| ::std::clone::Clone::clone(&v)).count(),
                1_u32.into()
            );
            assert_eq!(
                (&rel)
                    .filt(|v| v.test(sra::typed::pred::eq($name { value: 1, other: 2 })))
                    .count(),
                1_u32.into()
            );
            assert_eq!(
                (&rel).select(schema.except(&cx, schema.other)).count(),
                1_u32.into()
            );
        }};
    }
    check!(D);
    check!(T);
    check!(P);
    check!(F0);
    check!(Clone);
    check!(r#D);
}

#[test]
#[should_panic(expected = "record value requires its declared sort")]
fn a_record_value_with_the_wrong_sort_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    sra::Rel::new_set(&cx, cx.bv_sort(8)).insert(Route { src: 1, dst: 2 });
}

#[test]
fn raw_record_names_declare_the_plain_name() {
    #[derive(Schema)]
    struct r#Raw {
        r#x: u8,
    }
    let cx = Context::default();
    let sort = Raw::sort(&cx);
    let name = concat!(module_path!(), "::Raw");
    assert_eq!(Raw::schema(&cx).constructor.name(&cx), name);
    assert_eq!(cx.tuple_sort(name, &[("x", u8::sort(&cx))]), Some(sort));
}

#[test]
#[should_panic(expected = "conflicting definition for named sort")]
fn a_raw_record_with_other_labels_than_its_plain_name_is_rejected() {
    fn plain(cx: &Context) {
        #[derive(Schema)]
        struct Shared {
            x: u8,
        }
        Shared::sort(cx);
    }
    fn raw(cx: &Context) {
        #[derive(Schema)]
        struct r#Shared {
            y: u8,
        }
        Shared::sort(cx);
    }
    let cx = Context::default();
    plain(&cx);
    raw(&cx);
}

#[test]
fn a_blanket_clone_method_leaves_views_cloneable() {
    #[allow(dead_code)]
    trait Hostile {
        fn clone(&self) -> u8 {
            0
        }
    }
    impl<T> Hostile for T {}
    #[derive(Schema)]
    struct Pair {
        a: u8,
        b: u8,
    }
    let mut cx = Context::default();
    Pair::bind(&mut cx);
    let mut rel = Rel::<Pair, Pair>::new_set(&cx);
    rel.insert(Pair { a: 1, b: 2 });
    assert_eq!(
        (&rel).map(|v| ::std::clone::Clone::clone(&v)).count(),
        1_u32.into()
    );
}

#[test]
#[should_panic(expected = "conflicting definition for named sort")]
fn a_same_named_record_with_other_member_widths_is_rejected() {
    fn declared(cx: &mut Context) -> sra::Rel {
        #[derive(Schema)]
        struct Local {
            x: u32,
        }
        Local::bind(cx);
        sra::Rel::new_set(cx, Local::sort(cx))
    }
    #[derive(Schema)]
    struct Local {
        x: u8,
    }
    let mut cx = Context::default();
    declared(&mut cx).insert(Local { x: 1 });
}

#[test]
#[should_panic(expected = "conflicting definition for named sort")]
fn a_same_named_record_with_other_labels_is_rejected() {
    fn declared(cx: &mut Context) -> sra::Rel {
        #[derive(Schema)]
        struct Named {
            x: u8,
        }
        Named::bind(cx);
        sra::Rel::new_set(cx, Named::sort(cx))
    }
    #[derive(Schema)]
    struct Named {
        y: u8,
    }
    let mut cx = Context::default();
    declared(&mut cx).insert(Named { y: 1 });
}

#[test]
#[should_panic(expected = "representation requires a declared scalar member")]
fn a_marked_aggregate_field_is_rejected() {
    #[derive(Schema)]
    struct Marked {
        #[symbolic]
        pair: (u8, u8),
    }
    Marked::bind(&mut Context::default());
}

#[test]
fn rejected_forms_report_their_own_errors() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}

#[test]
fn raw_field_names_leave_the_private_input_fresh() {
    #[derive(Schema)]
    struct Raw {
        r#__sra_input: u8,
        r#__sra_input_: u16,
    }
    let mut cx = Context::default();
    Raw::bind(&mut cx);
    let schema = Raw::schema(&cx);
    let mut rel = Rel::<Raw, Raw>::new_set(&cx);
    rel.insert(Raw {
        r#__sra_input: 3,
        r#__sra_input_: 4,
    });
    assert_eq!((&rel).select(schema.r#__sra_input).count(), 1_u32.into());
    assert_eq!((&rel).map(|v| v).count(), 1_u32.into());
    assert!(rel.contains((3_u8, 4_u16)));
}

#[test]
fn field_type_self_names_the_source_record() {
    trait Field {
        type Scalar;
    }
    #[derive(Schema)]
    struct Associated {
        value: <Self as Field>::Scalar,
        pair: (<Self as Field>::Scalar, bool),
    }
    impl Field for Associated {
        type Scalar = sra::typed::UInt<8>;
    }
    let mut cx = Context::default();
    Associated::bind(&mut cx);
    let schema = Associated::schema(&cx);
    let mut rel = Rel::<Associated, Associated>::new_set(&cx);
    rel.insert(Associated {
        value: sra::typed::UInt::new(3),
        pair: (sra::typed::UInt::new(4), true),
    });
    assert_eq!((&rel).select(schema.value).count(), 1_u32.into());
    assert_eq!((&rel).map(|v| v.pair).count(), 1_u32.into());
    assert!(rel.contains((
        sra::typed::UInt::<8>::new(3),
        (sra::typed::UInt::<8>::new(4), true)
    )));
}

#[test]
fn packed_alignment_one_fields_keep_noncopy_ownership() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DROPS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Schema)]
    #[repr(transparent)]
    struct Byte {
        value: u8,
    }
    impl Drop for Byte {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[derive(Schema)]
    struct Empty {}
    #[derive(Schema)]
    #[repr(C, packed)]
    struct Packed {
        tag: u8,
        value: Byte,
        empty: Empty,
    }
    let mut cx = Context::default();
    Packed::bind(&mut cx);
    let mut rel = Rel::<Packed, Packed>::new_set(&cx);
    let value = Packed {
        tag: 3,
        value: Byte { value: 4 },
        empty: Empty {},
    };
    rel.insert(&value);
    assert!(rel.contains(&value));
    assert_eq!(DROPS.load(Ordering::Relaxed), 0);
    drop(value);
    assert_eq!(DROPS.load(Ordering::Relaxed), 1);
}

#[allow(unused_macros)]
mod caller_macros {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use sra::Context;
    use sra::typed::{Rel, Sorted};

    macro_rules! assert_eq {
        ($($tokens:tt)*) => {
            ()
        };
    }
    macro_rules! debug_assert_eq {
        ($($tokens:tt)*) => {
            ()
        };
    }

    macro_rules! concat {
        ($($tokens:tt)*) => {
            "hijacked"
        };
    }
    macro_rules! module_path {
        () => {
            "hijacked"
        };
    }
    macro_rules! stringify {
        ($($tokens:tt)*) => {
            "hijacked"
        };
    }

    #[derive(sra::Schema)]
    struct A {
        x: u8,
    }
    #[derive(sra::Schema)]
    struct B {
        x: u8,
    }

    #[test]
    fn generated_checks_preserve_nominal_record_identity() {
        let mut cx = Context::default();
        A::bind(&mut cx);
        B::bind(&mut cx);
        let schema = A::schema(&cx);
        ::core::assert_eq!(
            schema.constructor.name(&cx),
            ::core::concat!(::core::module_path!(), "::A")
        );
        ::core::assert_eq!(schema.x.name(&cx), "x");
        let mut rows = Rel::<A, A>::new_set(&cx);
        ::core::assert!(catch_unwind(AssertUnwindSafe(|| rows.insert(B { x: 1 }))).is_err());
        rows.insert(A { x: 1 });
        ::core::assert!(rows.contains(A { x: 1 }));
    }
}
