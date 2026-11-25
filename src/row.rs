//! Bound logical rows, physical layouts, and concrete or symbolic row
//! arguments.
//!
//! Named records preserve their member grouping. Anonymous products normalize
//! their grouping. Rows of the same sort remain independent occurrences, each
//! with its own layout.

use std::iter;
use std::sync::Arc;

use crate::ast::{FuncDecl, Sort, SortKind, Sorts};
use crate::context::{AlgebraId, Context, Predicate};
use crate::scalar::ScalarSort;

const DEFAULT_BACKEND: &str = "sparse";

/// A logical sort or the range of a declaration to bind as a row.
#[derive(Clone, Copy, Debug)]
pub enum Source {
    /// A sort with its declared member bindings.
    Sort(Sort),
    /// A declaration's range, using the field binding for a scalar record
    /// projection.
    Range(FuncDecl),
}

impl From<Sort> for Source {
    fn from(sort: Sort) -> Self {
        Self::Sort(sort)
    }
}

impl From<FuncDecl> for Source {
    fn from(decl: FuncDecl) -> Self {
        Self::Range(decl)
    }
}

// The walk holds no store guard across a backend call. A scalar root needs
// no traversal allocation.
struct Leaves {
    first: Option<(Sort, Option<FuncDecl>)>,
    stack: Vec<(Sort, Option<FuncDecl>)>,
}

impl Leaves {
    fn new(sorts: &Sorts, source: Source) -> (Sort, Self) {
        let (sort, field) = match source {
            Source::Sort(sort) => {
                sorts.check(sort);
                (sort, None)
            }
            Source::Range(decl) => {
                let info = decl.resolve(sorts);
                (info.range, info.field_sort().is_some().then_some(decl))
            }
        };
        (
            sort,
            Self {
                first: Some((sort, field)),
                stack: Vec::new(),
            },
        )
    }

    fn next(&mut self, sorts: &Sorts) -> Option<(ScalarSort, Option<FuncDecl>)> {
        while let Some((part, field)) = self.first.take().or_else(|| self.stack.pop()) {
            if let SortKind::Scalar(scalar) = part.kind() {
                return Some((scalar, field));
            }
            if part.scalar_len() == 0 {
                continue;
            }
            let members = sorts.members(part);
            self.stack
                .extend(members.into_iter().enumerate().rev().map(|(i, child)| {
                    let accessor = (part.kind() == SortKind::Record
                        && matches!(child.kind(), SortKind::Scalar(_)))
                    .then(|| sorts.project(part, i));
                    (child, accessor)
                }));
        }
        None
    }
}

/// A logical sort bound to an immutable slice of scalar algebras.
///
/// Member rows share the layout. A row retains its sort's identity without
/// retaining the context, so other fields can still be bound during setup.
#[derive(Clone, Debug)]
pub struct Row {
    sort: Sort,
    layout: Arc<[AlgebraId]>,
    start: usize,
}

impl Row {
    /// A row over the slice of `layout` starting at `start`.
    ///
    /// Each algebra must have the exact scalar sort at its position. The
    /// caller selects the backend independently for each position.
    ///
    /// # Panics
    ///
    /// Panics if the slice does not cover the sort, a selected algebra has
    /// the wrong scalar sort, or the sort or a selected algebra is foreign.
    pub fn new(cx: &Context, sort: Sort, layout: Arc<[AlgebraId]>, start: usize) -> Self {
        let (_, mut leaves) = Leaves::new(cx.sorts(), sort.into());
        assert!(
            start <= layout.len() && sort.scalar_len() <= layout.len() - start,
            "row layout must cover the logical sort"
        );
        let row = Self {
            sort,
            layout,
            start,
        };
        for &id in row.algebras() {
            let (scalar, _) = leaves
                .next(cx.sorts())
                .expect("row sort must supply every scalar leaf");
            assert_eq!(
                cx.resolve(id).sort(),
                scalar,
                "row algebra must match its logical scalar sort"
            );
        }
        row
    }

    // Bound leaves match their scalar sorts by construction, so these rows
    // skip the layout validation of `Row::new`.
    fn collect(sort: Sort, ids: impl Iterator<Item = AlgebraId>) -> Self {
        let mut layout = Vec::with_capacity(sort.scalar_len());
        layout.extend(ids);
        Self {
            sort,
            layout: layout.into(),
            start: 0,
        }
    }

    /// Binds the row's scalar leaves and returns their layout.
    ///
    /// Configured record members keep their algebras. Unbound members are
    /// bound to the registered `sparse` backend, which becomes their field
    /// binding. Anonymous scalars use the same backend. A projection binds
    /// only its range. If binding fails partway, earlier bindings remain
    /// available.
    ///
    /// # Panics
    ///
    /// Panics if the source is foreign, a needed backend binding fails, or a
    /// new binding is needed while the context is shared.
    pub fn bind(cx: &mut Context, source: impl Into<Source>) -> Self {
        let (sort, mut leaves) = Leaves::new(cx.sorts(), source.into());
        Self::collect(
            sort,
            iter::from_fn(|| {
                let (scalar, field) = leaves.next(cx.sorts())?;
                Some(match field {
                    Some(accessor) => cx
                        .field(accessor)
                        .unwrap_or_else(|| cx.bind_field(accessor, DEFAULT_BACKEND)),
                    None => cx.bind(DEFAULT_BACKEND, scalar),
                })
            }),
        )
    }

    /// The row over a source's previously bound scalar leaves.
    ///
    /// # Panics
    ///
    /// Panics if the source is foreign, a leaf is unbound, or an anonymous
    /// scalar requires an unregistered `sparse` backend.
    pub fn of(cx: &Context, source: impl Into<Source>) -> Self {
        let (sort, mut leaves) = Leaves::new(cx.sorts(), source.into());
        Self::collect(
            sort,
            iter::from_fn(|| leaves.next(cx.sorts())).map(|(scalar, field)| {
                match field {
                    Some(accessor) => cx.field(accessor),
                    None => cx.algebra(DEFAULT_BACKEND, scalar),
                }
                .expect("row field must be bound")
            }),
        )
    }

    /// The normalized product of the rows, with their layouts concatenated
    /// in order.
    ///
    /// # Panics
    ///
    /// Panics if a row belongs to another context or the scalar-leaf count
    /// overflows.
    pub fn product(cx: &Context, rows: &[&Self]) -> Self {
        let sort = cx.sorts().product(rows.iter().map(|row| row.sort));
        Self::collect(sort, rows.iter().flat_map(|row| row.algebras()).copied())
    }

    /// The logical sort carried by this row.
    #[inline(always)]
    pub fn sort(&self) -> Sort {
        self.sort
    }

    /// The algebras in scalar-leaf order, one per independent position.
    #[inline(always)]
    pub fn algebras(&self) -> &[AlgebraId] {
        &self.layout[self.start..][..self.sort.scalar_len()]
    }

    /// The algebra bound to this scalar row.
    ///
    /// # Panics
    ///
    /// Panics if the row is a record or product.
    #[inline(always)]
    pub fn scalar_algebra(&self) -> AlgebraId {
        self.sort.scalar();
        self.algebras()[0]
    }

    /// The immediate member rows in declaration order, sharing this layout.
    ///
    /// # Panics
    ///
    /// Panics if the row belongs to another context or has a scalar sort.
    pub fn members(&self, cx: &Context) -> impl ExactSizeIterator<Item = Self> + '_ {
        let members = cx.sorts().members(self.sort);
        assert!(
            !matches!(self.sort.kind(), SortKind::Scalar(_)),
            "row members require a record or product"
        );
        let mut start = self.start;
        members.into_iter().map(move |sort| {
            let row = Self {
                sort,
                layout: Arc::clone(&self.layout),
                start,
            };
            start += sort.scalar_len();
            row
        })
    }

    /// Changes the row's record wrappers, preserving its physical layout.
    ///
    /// # Panics
    ///
    /// Panics if the row or sort is foreign or their sorts cannot be
    /// converted as in [`Sorts::coercible`].
    pub fn coerce(&mut self, cx: &Context, sort: Sort) {
        assert!(
            cx.sorts().coercible(self.sort, sort),
            "coercion requires matching sorts or a record and its declared arguments"
        );
        self.sort = sort;
    }
}

/// A stored relation endpoint supplied as a bound row, sort, or declaration.
pub trait Domain {
    /// The endpoint row over its previously bound scalar algebras.
    ///
    /// # Panics
    ///
    /// Panics if the endpoint is foreign or one of its leaves is unbound.
    fn row(self, cx: &Context) -> Row;
}

impl<T: Into<Source>> Domain for T {
    fn row(self, cx: &Context) -> Row {
        Row::of(cx, self)
    }
}

impl Domain for Row {
    fn row(self, cx: &Context) -> Row {
        cx.sorts().check(self.sort);
        self
    }
}

impl Domain for &Row {
    fn row(self, cx: &Context) -> Row {
        self.clone().row(cx)
    }
}

/// A row borrowed for argument conversion without retaining its context or
/// physical layout.
#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a> {
    /// The context interpreting the sort and algebras.
    pub cx: &'a Context,
    /// The logical sort.
    pub sort: Sort,
    /// The bound algebras in scalar-leaf order.
    pub algebras: &'a [AlgebraId],
    /// The immediate member sorts of an aggregate, in declaration order.
    pub members: &'a [Sort],
}

fn member_sorts(cx: &Context, sort: Sort) -> Vec<Sort> {
    match sort.kind() {
        SortKind::Scalar(_) => Vec::new(),
        _ => cx.sorts().members(sort),
    }
}

impl RowRef<'_> {
    /// Passes the member or product at a scalar offset to its adapter.
    fn part(self, sort: Sort, offset: usize, then: impl FnOnce(RowRef<'_>)) {
        then(RowRef {
            cx: self.cx,
            sort,
            algebras: &self.algebras[offset..][..sort.scalar_len()],
            members: &member_sorts(self.cx, sort),
        });
    }
}

/// A value packable as row arguments following the declaration's immediate
/// members.
///
/// Word slices encode one scalar, never an implicit list of columns. Packing
/// borrows values so predicates can reuse literals and collections without
/// cloning. Implementations append canonical little-endian words or scalar
/// predicates without modifying existing output. The consuming sink
/// determines whether symbolic inputs are allowed.
pub trait IntoRow {
    /// Number of components after flattening anonymous products; named
    /// records count as one.
    fn arity(&self) -> usize {
        1
    }

    /// The nominal record sort of this value in the consuming row's context,
    /// even through a normalized singleton product.
    fn record_sort(&self, _cx: &Context) -> Option<Sort> {
        None
    }

    /// The number of normalized destination components this value matches
    /// at the start of `sorts`.
    fn span(&self, cx: &Context, sorts: &[Sort]) -> Option<usize> {
        let n = self.arity();
        let parts = sorts.get(..n)?;
        match self.record_sort(cx) {
            Some(sort) => (parts == [sort]).then_some(n),
            None => parts
                .iter()
                .all(|sort| matches!(sort.kind(), SortKind::Scalar(_)))
                .then_some(n),
        }
    }

    /// Appends this value's encoding for `row` to `out`.
    ///
    /// A nominal record adapter checks that `row.sort` is its declared sort.
    /// The packing boundary validates the emitted words and predicate bindings.
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink);
}

/// Output of row conversion.
///
/// A sink stores packed words and, if it is symbolic, predicates. Packing
/// reserves each predicate's scalar word span and retains the predicate at
/// that span's starting offset. Ground sinks have no predicate list, so
/// packing a predicate into one panics.
pub trait RowSink {
    /// The packed words appended so far.
    fn words(&mut self) -> &mut Vec<u64>;

    /// The predicates retained so far, each with the word offset it starts
    /// at, or `None` for a ground sink.
    fn predicates(&mut self) -> Option<&mut Vec<(usize, Predicate)>> {
        None
    }
}

impl RowSink for Vec<u64> {
    fn words(&mut self) -> &mut Vec<u64> {
        self
    }
}

/// Appends one checked row argument to a ground or symbolic sink.
///
/// # Panics
///
/// Panics as [`append_rows`] does, possibly leaving a partial or invalid row
/// in `out`.
#[inline(always)]
pub fn append(cx: &Context, args: impl IntoRow, row: &Row, out: &mut impl RowSink) {
    append_rows(cx, iter::once(args), row, out);
}

/// Appends the rows and reports whether the input supplied any.
///
/// The arguments share the destination layout and its immediate members.
/// Each adapter's output is checked before the next row is consumed.
///
/// # Panics
///
/// Panics if the row belongs to another context or the appended words do
/// not encode exactly its scalar values. Passing any predicate to a ground
/// sink also panics. In a symbolic sink each predicate must match its
/// destination offset, context, and algebra.
///
/// A panic can leave a partially written or invalid row in `out` after the
/// rows already appended.
pub fn append_rows(
    cx: &Context,
    rows: impl IntoIterator<Item = impl IntoRow>,
    row: &Row,
    out: &mut impl RowSink,
) -> bool {
    cx.sorts().check(row.sort());
    let members = member_sorts(cx, row.sort());
    let view = RowRef {
        cx,
        sort: row.sort(),
        algebras: row.algebras(),
        members: &members,
    };
    let mut any = false;
    for args in rows {
        let start = out.words().len();
        let mut p = out.predicates().map_or(0, |v| v.len());
        args.append(view, out);
        let mut i = start;
        for column in row.algebras() {
            let sort = cx.resolve(*column).sort();
            assert!(
                i + sort.words() <= out.words().len(),
                "row arguments have too few words"
            );
            assert!(
                sort.accepts(&out.words()[i..i + sort.words()]),
                "row value does not encode its scalar sort"
            );
            if let Some((offset, predicate)) = out
                .predicates()
                .and_then(|v| v.get(p))
                .filter(|(offset, _)| *offset < i + sort.words())
            {
                assert_eq!(*offset, i, "predicate must start at a scalar boundary");
                assert!(
                    predicate.context() == cx,
                    "predicate belongs to another context"
                );
                assert_eq!(
                    predicate.algebra(),
                    *column,
                    "predicate uses an incompatible algebra"
                );
                p += 1;
            }
            i += sort.words();
        }
        assert_eq!(i, out.words().len(), "row arguments have too many words");
        assert_eq!(
            p,
            out.predicates().map_or(0, |v| v.len()),
            "row arguments have misplaced predicates"
        );
        any = true;
    }
    any
}

/// Appends the words of a numeric literal checked against the declared
/// width.
///
/// No bits are silently truncated. Negative literals use two's complement and
/// must fit the signed width. The resulting bit pattern has no implicit
/// signedness: the predicate operation selects its interpretation.
///
/// # Panics
///
/// Panics if `sort` is not a bit-vector sort or `value` exceeds its width.
#[inline(always)]
fn integer(value: u128, negative: bool, sort: ScalarSort, out: &mut Vec<u64>) {
    let width = sort
        .bv_width()
        .expect("integer literal sort must be a bit vector");
    if negative {
        assert!(
            (!value).bit_width() < width,
            "negative integer literal exceeds its signed width"
        );
    } else {
        assert!(
            value.bit_width() <= width,
            "integer literal exceeds its scalar sort"
        );
    }
    // Sign-extend to the sort's words, then clear the bits beyond its width.
    let fill = if negative { u64::MAX } else { 0 };
    out.push(value as u64);
    for i in 1..sort.words() {
        out.push(if i == 1 { (value >> 64) as u64 } else { fill });
    }
    *out.last_mut().unwrap() &= sort.top_mask();
}

macro_rules! integers {
    (unsigned: $($u:ty),*; signed: $($i:ty),*) => {
        $(impl IntoRow for $u {
            #[inline(always)]
            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                integer(*self as u128, false, row.sort.scalar(), out.words());
            }
        })*
        $(impl IntoRow for $i {
            #[inline(always)]
            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                integer(*self as i128 as u128, *self < 0, row.sort.scalar(), out.words());
            }
        })*
    };
}

integers!(unsigned: u8, u16, u32, u64, u128, usize; signed: i8, i16, i32, i64, i128, isize);

impl<T: IntoRow + ?Sized> IntoRow for &T {
    fn arity(&self) -> usize {
        (**self).arity()
    }

    fn record_sort(&self, cx: &Context) -> Option<Sort> {
        (**self).record_sort(cx)
    }

    fn span(&self, cx: &Context, sorts: &[Sort]) -> Option<usize> {
        (**self).span(cx, sorts)
    }

    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        (**self).append(row, out);
    }
}

impl IntoRow for bool {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        assert!(
            row.sort.kind() == SortKind::Scalar(ScalarSort::bool()),
            "a Boolean literal requires the Boolean sort"
        );
        out.words().push(u64::from(*self));
    }
}

impl IntoRow for [u64] {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        // Rejects aggregate destinations, including one-member records.
        let sort = row.sort.scalar();
        assert!(sort.accepts(self), "words do not encode the scalar sort");
        out.words().extend_from_slice(self);
    }
}

impl<const N: usize> IntoRow for [u64; N] {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        self.as_slice().append(row, out);
    }
}

/// Appends `p` as the predicate over `column` of `cx`, reserving its scalar
/// span.
fn predicate(p: &Predicate, cx: &Context, column: AlgebraId, out: &mut impl RowSink) {
    assert!(p.context() == cx, "predicate belongs to another context");
    assert_eq!(
        p.algebra(),
        column,
        "predicate uses an incompatible algebra"
    );
    let i = out.words().len();
    out.predicates()
        .expect("row arguments must be ground values")
        .push((i, p.clone()));
    out.words()
        .resize(i + p.context().resolve(p.algebra()).sort().words(), 0);
}

impl IntoRow for Predicate {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        // Rejects aggregate destinations, including one-member records.
        row.sort.scalar();
        predicate(self, row.cx, row.algebras[0], out);
    }
}

// Dynamic predicate rows follow physical scalar order: domain then range for
// a binary relation.
macro_rules! predicate_args {
    ($t:ty) => {
        impl IntoRow for [$t] {
            fn arity(&self) -> usize {
                self.len()
            }

            fn span(&self, _cx: &Context, sorts: &[Sort]) -> Option<usize> {
                let mut n = 0;
                for (i, sort) in sorts.iter().enumerate() {
                    // Like a scalar, a nonempty slice starts at its first
                    // column and leaves any empty record before it to an
                    // explicit unit argument. An empty slice may fill one.
                    if sort.scalar_len() == 0 && n == 0 && !self.is_empty() {
                        break;
                    }
                    n += sort.scalar_len();
                    if n == self.len() {
                        return Some(i + 1);
                    }
                    if n > self.len() {
                        break;
                    }
                }
                self.is_empty().then_some(0)
            }

            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                assert_eq!(
                    self.len(),
                    row.sort.scalar_len(),
                    "predicate row has the wrong column count"
                );
                for (p, &column) in self.iter().zip(row.algebras) {
                    predicate(p, row.cx, column, out);
                }
            }
        }

        impl<const N: usize> IntoRow for [$t; N] {
            fn arity(&self) -> usize {
                N
            }

            fn span(&self, cx: &Context, sorts: &[Sort]) -> Option<usize> {
                self.as_slice().span(cx, sorts)
            }

            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                self.as_slice().append(row, out);
            }
        }
    };
}
predicate_args!(Predicate);
predicate_args!(&Predicate);

impl IntoRow for () {
    fn arity(&self) -> usize {
        0
    }

    fn span(&self, cx: &Context, sorts: &[Sort]) -> Option<usize> {
        Some(usize::from(sorts.first().is_some_and(|sort| {
            sort.kind() == SortKind::Record && cx.sorts().members(*sort).is_empty()
        })))
    }

    fn append(&self, row: RowRef<'_>, _out: &mut impl RowSink) {
        assert!(
            !matches!(row.sort.kind(), SortKind::Scalar(_)) && row.members.is_empty(),
            "unit arguments require an empty aggregate"
        );
    }
}

macro_rules! tuple_args {
    ($($n:literal: $($t:ident $i:tt),+);+ $(;)?) => {$(
        impl<$($t: IntoRow),+> IntoRow for ($($t,)+) {
            fn arity(&self) -> usize { 0 $(+ self.$i.arity())+ }

            fn record_sort(&self, cx: &Context) -> Option<Sort> {
                if self.arity() != 1 { return None; }
                None $(.or_else(|| self.$i.record_sort(cx)))+
            }

            fn span(&self, cx: &Context, sorts: &[Sort]) -> Option<usize> {
                let first = sorts.first().filter(|sort| sort.kind() == SortKind::Record);
                // Whether the tuple supplies exactly the record's fields, each
                // element matching one member's normalized components.
                let record = first.is_some_and(|sort| {
                    let parts = cx.sorts().members(*sort);
                    parts.len() == $n && true $(&& {
                        let members;
                        let fields = if parts[$i].kind() == SortKind::Product {
                            members = cx.sorts().members(parts[$i]);
                            members.as_slice()
                        } else {
                            std::slice::from_ref(&parts[$i])
                        };
                        self.$i.span(cx, fields) == Some(fields.len())
                    })+
                });
                if record {
                    Some(1)
                } else {
                    let mut start = 0;
                    $(start += self.$i.span(cx, &sorts[start..])?;)+
                    // A group covering exactly one record must be that record's
                    // value, because `append` takes the member path otherwise.
                    (start != 1 || first.is_none() || self.record_sort(cx) == first.copied())
                        .then_some(start)
                }
            }

            #[inline(always)]
            fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
                // A record destination takes its members from the tuple unless the
                // tuple wraps a value of that record, which falls through to its group.
                if row.sort.kind() == SortKind::Record
                    && self.record_sort(row.cx) != Some(row.sort)
                {
                    assert_eq!(row.members.len(), $n, "row arguments have the wrong member count");
                    assert_eq!(
                        self.span(row.cx, std::slice::from_ref(&row.sort)),
                        Some(1),
                        "row arguments must match the destination members"
                    );
                    let mut offset = 0;
                    $({
                        let sort = row.members[$i];
                        row.part(sort, offset, |part| self.$i.append(part, out));
                        offset += sort.scalar_len();
                    })+
                    debug_assert_eq!(
                        offset,
                        row.sort.scalar_len(),
                        "record members must cover the row"
                    );
                    return;
                }
                let mut sorts = match row.sort.kind() {
                    SortKind::Product => row.members,
                    _ => std::slice::from_ref(&row.sort),
                };
                let mut offset = 0;
                $({
                    // Match anonymous groups and record fields by their destination span,
                    // including named empty records that occupy no physical columns.
                    let sort = match self.$i.span(row.cx, sorts)
                        .and_then(|n| sorts.split_off(..n))
                        .expect("row arguments must match the destination members")
                    {
                        [sort] => *sort,
                        parts => row.cx.sorts().product(parts.iter().copied()),
                    };
                    row.part(sort, offset, |part| self.$i.append(part, out));
                    offset += sort.scalar_len();
                })+
                assert!(sorts.is_empty(), "row arguments have the wrong member count");
                debug_assert_eq!(offset, row.sort.scalar_len(), "row members must cover the row");
            }
        }
    )+};
}

tuple_args!(
    1: A 0;
    2: A 0, B 1;
    3: A 0, B 1, C 2;
    4: A 0, B 1, C 2, D 3;
    5: A 0, B 1, C 2, D 3, E 4;
    6: A 0, B 1, C 2, D 3, E 4, F 5;
    7: A 0, B 1, C 2, D 3, E 4, F 5, G 6;
    8: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7;
    9: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8;
    10: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9;
    11: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10;
    12: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11;
    13: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12;
    14: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13;
    15: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14;
    16: A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11, M 12, N 13, O 14, P 15;
);
