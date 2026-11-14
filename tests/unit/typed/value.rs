//! Checks for custom-width integers and IPv4 addresses: construction checks
//! the width, ordering follows the Rust interpretation, and rows accept only
//! the declared width.

use std::net::Ipv4Addr;

use super::{Int, UInt};
use crate::Schema;
use crate::context::Context;
use crate::typed::{IntoQuery, Rel, Sorted};

// --- construction and order ---------------------------------------------

#[test]
fn unsigned_values_check_their_width() {
    assert_eq!(UInt::<8>::new(255).as_ref(), [255]);
    assert_eq!(UInt::<128>::new(u128::MAX).as_ref(), [u64::MAX, u64::MAX]);
    assert_eq!(
        UInt::<70>::from_words([1, 2]),
        UInt::<70>::new((2 << 64) | 1)
    );
    assert!(UInt::<70>::from_words([0, 1]) > UInt::<70>::new(u64::MAX.into()));
    assert!(UInt::<8>::new(3) < UInt::<8>::new(4));
}

#[test]
#[should_panic(expected = "integer literal exceeds its scalar sort")]
fn unsigned_values_reject_high_bits_is_rejected() {
    UInt::<8>::new(256);
}

#[test]
#[should_panic(expected = "words do not encode this integer width")]
fn unsigned_words_must_be_canonical_is_rejected() {
    UInt::<8>::from_words([256]);
}

#[test]
fn signed_values_order_by_sign_then_magnitude() {
    assert_eq!(Int::<8>::new(-1).as_ref(), [255]);
    assert_eq!(Int::<8>::new(-128), Int::<8>::from_words([128]));
    assert_eq!(Int::<128>::new(-1).as_ref(), [u64::MAX, u64::MAX]);
    assert!(Int::<8>::new(-1) < Int::<8>::new(0));
    assert!(Int::<8>::new(-2) < Int::<8>::new(-1));
    assert!(Int::<8>::new(127) > Int::<8>::new(1));
    assert!(Int::<70>::new(-1) < Int::<70>::new(1 << 65));
    assert_eq!(Int::<8>::new(5).0, UInt::<8>::new(5));
}

#[test]
#[should_panic(expected = "signed integer exceeds its width")]
fn signed_values_reject_overflow_is_rejected() {
    Int::<8>::new(128);
}

#[test]
#[should_panic(expected = "bit-vector sorts must have nonzero width")]
fn signed_values_need_a_sign_bit_is_rejected() {
    Int::<0>::new(0);
}

// --- rows ---------------------------------------------------------------

/// A record with two symbolic members and one sparse member.
#[derive(Clone, Debug, Schema)]
struct Hop {
    #[symbolic]
    address: Ipv4Addr,
    #[symbolic]
    ttl: UInt<8>,
    delta: Int<8>,
}

#[test]
fn custom_integers_pack_into_rows() {
    let mut cx = Context::default();
    Hop::bind(&mut cx);
    let HopSchema {
        address,
        ttl,
        delta,
        ..
    } = Hop::schema(&cx);
    let mut hops = Rel::<Hop, Hop>::new_set(&cx);
    hops.insert(Hop {
        address: Ipv4Addr::new(10, 0, 0, 1),
        ttl: UInt::new(64),
        delta: Int::new(-1),
    });
    hops.insert((
        Ipv4Addr::new(10, 0, 0, 2),
        UInt::<8>::new(63),
        Int::<8>::new(1),
    ));
    assert!(hops.contains((
        Ipv4Addr::new(10, 0, 0, 1),
        UInt::<8>::new(64),
        Int::<8>::new(-1)
    )));
    assert_eq!(
        (&hops).with(address.eq(Ipv4Addr::new(10, 0, 0, 2))).count(),
        1_u32.into()
    );
    assert_eq!(
        (&hops).with(ttl.eq(UInt::<8>::new(64))).count(),
        1_u32.into()
    );
    assert_eq!(
        (&hops).with(delta.eq(Int::<8>::new(1))).count(),
        1_u32.into()
    );
}

#[test]
fn signed_columns_order_negative_values_first() {
    let mut cx = Context::default();
    Hop::bind(&mut cx);
    let HopSchema { delta, .. } = Hop::schema(&cx);
    let mut hops = Rel::<Hop, Hop>::new_set(&cx);
    for value in [-128, -1, 0, 127] {
        hops.insert(Hop {
            address: Ipv4Addr::new(10, 0, 0, 1),
            ttl: UInt::new(64),
            delta: Int::new(value),
        });
    }
    assert_eq!(
        (&hops).with(delta.lt(Int::<8>::new(0))).count(),
        2_u32.into()
    );
    assert_eq!(
        (&hops).with(delta.ge(Int::<8>::new(-1))).count(),
        3_u32.into()
    );
}

#[test]
#[should_panic(expected = "integer value requires its declared width")]
fn integers_reject_other_widths_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    Rel::<u8, u8>::new_set(&cx).insert(UInt::<16>::new(1));
}

#[test]
#[should_panic(expected = "IPv4 value requires 32 bits")]
fn addresses_require_their_width_is_rejected() {
    let mut cx = Context::default();
    u8::bind(&mut cx);
    Rel::<u8, u8>::new_set(&cx).insert(Ipv4Addr::new(10, 0, 0, 1));
}
