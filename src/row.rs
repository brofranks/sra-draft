//! Bound logical rows and concrete or symbolic row arguments.
//!
//! Named records preserve their member grouping. Anonymous products
//! normalize their grouping. Physical occurrences always remain independent
//! unless a query explicitly identifies them.

use std::sync::Arc;

use crate::ast::{FuncDecl, Sort, SortKind};
use crate::context::{ColumnId, Context, Field, Predicate};
use crate::scalar::ScalarSort;

fn members(cx: &Context, sort: Sort) -> Vec<Sort> {
    if sort.kind.is_scalar() {
        Vec::new()
    } else {
        cx.members(sort)
    }
}

/// A row borrowed for argument conversion: a [`Row`] or one of its members,
/// without ownership of the context or layout, so converting each member of
/// each row retains nothing.
#[derive(Clone, Copy)]
pub struct RowRef<'a> {
    /// The context the sort and columns belong to.
    pub cx: &'a Context,
    /// The logical sort.
    pub sort: Sort,
    /// The row's physical columns.
    pub columns: &'a [ColumnId],
    /// The sort's immediate members; a scalar has none.
    pub members: &'a [Sort],
}

/// A bound logical sort and its immutable physical columns.
///
/// Occurrences remain independent.
#[derive(Clone)]
pub struct Row {
    /// The logical sort, whose context the columns belong to.
    pub sort: Sort,
    /// The shared physical layout this row occupies a slice of.
    pub layout: Arc<[ColumnId]>,
    /// The offset of the row's first column in `layout`.
    pub start: usize,
    /// The sort's immediate members; a scalar has none.
    pub members: Arc<[Sort]>,
}

impl Row {
    /// Binds `sort` to the columns of `layout` starting at `start`.
    ///
    /// # Panics
    ///
    /// Panics if `sort` or a selected column belongs to another context, the
    /// selected layout does not cover the sort, or its scalar sorts differ.
    pub fn new(cx: &Context, sort: Sort, layout: Arc<[ColumnId]>, start: usize) -> Row {
        assert!(sort.context == cx.id, "sort belongs to another context");
        assert!(
            start <= layout.len() && sort.len <= layout.len() - start,
            "row layout must cover the logical sort"
        );
        // Match scalar leaves in physical order without recursion or a store guard
        // across a backend call. A scalar row needs no traversal allocation.
        let mut stack = Vec::new();
        let mut part = sort;
        for &column in &layout[start..][..sort.len] {
            while !part.kind.is_scalar() {
                stack.extend(cx.members(part).into_iter().rev());
                part = stack.pop().expect("row sort must supply every scalar leaf");
            }
            assert_eq!(
                cx.resolve(column).sort(),
                part.scalar(),
                "row column must match its logical scalar sort"
            );
            if let Some(next) = stack.pop() {
                part = next;
            }
        }
        Row {
            sort,
            layout,
            start,
            members: members(cx, sort).into(),
        }
    }

    /// Borrows the row for argument conversion.
    ///
    /// # Panics
    ///
    /// Panics if the row belongs to another context.
    #[inline(always)]
    pub fn view<'a>(&'a self, cx: &'a Context) -> RowRef<'a> {
        assert!(self.sort.context == cx.id, "row belongs to another context");
        RowRef {
            cx,
            sort: self.sort,
            columns: columns(self),
            members: &self.members,
        }
    }
}

/// Runs `then` on the member of `sort` starting `offset` columns into an
/// aggregate row.
#[inline(always)]
pub fn part<T>(
    row: RowRef<'_>,
    sort: Sort,
    offset: usize,
    then: impl FnOnce(RowRef<'_>) -> T,
) -> T {
    then(RowRef {
        cx: row.cx,
        sort,
        columns: &row.columns[offset..offset + sort.len],
        members: &members(row.cx, sort),
    })
}

/// A logical sort or declaration range whose physical storage is bound in
/// two phases: [`Domain::bind`] fixes every field while the context is
/// unshared, and [`Domain::row`] looks those fields up afterwards.
///
/// A record member uses its configured storage; a scalar sort uses sparse
/// storage.
pub trait Domain {
    /// Binds each unbound field to its default storage and returns the row of
    /// this sort or range.
    ///
    /// # Panics
    ///
    /// Panics if this domain belongs to another context, no `sparse`
    /// backend is registered when a default is needed, or a new binding is
    /// needed while `cx` is shared.
    fn bind(self, cx: &mut Context) -> Row;

    /// The row of this sort or range over its previously bound leaves.
    ///
    /// # Panics
    ///
    /// Panics if this domain belongs to another context or a field is not
    /// bound.
    fn row(self, cx: &Context) -> Row;
}

/// The fields of `sort`'s scalar leaves in column order: the accessor for a
/// record member and the sort for an anonymous scalar, whose own field is
/// `field`.
fn leaves(cx: &Context, sort: Sort, field: Field) -> Vec<Field> {
    assert!(sort.context == cx.id, "sort belongs to another context");
    let mut leaves = Vec::new();
    let mut stack = vec![(sort, field)];
    while let Some((part, leaf)) = stack.pop() {
        if part.kind.is_scalar() {
            leaves.push(leaf);
            continue;
        }
        stack.extend(
            cx.members(part)
                .into_iter()
                .enumerate()
                .rev()
                .map(|(i, child)| {
                    if part.kind == SortKind::Tuple && child.kind.is_scalar() {
                        (child, cx.func_decl(part, child, Some(i)).into())
                    } else {
                        (child, child.into())
                    }
                }),
        );
    }
    leaves
}

// Bound fields match their scalar sorts by construction, so these rows skip
// the layout validation of `Row::new`.
fn bind(cx: &mut Context, sort: Sort, field: Field) -> Row {
    Row {
        sort,
        layout: leaves(cx, sort, field)
            .into_iter()
            .map(|leaf| cx.bind_field(leaf))
            .collect(),
        start: 0,
        members: members(cx, sort).into(),
    }
}

fn row(cx: &Context, sort: Sort, field: Field) -> Row {
    Row {
        sort,
        layout: leaves(cx, sort, field)
            .into_iter()
            .map(|leaf| cx.field(leaf))
            .collect(),
        start: 0,
        members: members(cx, sort).into(),
    }
}

impl Domain for Sort {
    fn bind(self, cx: &mut Context) -> Row {
        bind(cx, self, self.into())
    }

    fn row(self, cx: &Context) -> Row {
        row(cx, self, self.into())
    }
}

/// The range of `decl` and its field: the accessor itself for a record
/// member and the range sort otherwise.
fn range_field(decl: FuncDecl, cx: &Context) -> (Sort, Field) {
    let info = decl.resolve(cx);
    if info.member.is_some() && info.domain.kind == SortKind::Tuple {
        (info.range, decl.into())
    } else {
        (info.range, info.range.into())
    }
}

impl Domain for FuncDecl {
    fn bind(self, cx: &mut Context) -> Row {
        let (sort, field) = range_field(self, cx);
        bind(cx, sort, field)
    }

    fn row(self, cx: &Context) -> Row {
        let (sort, field) = range_field(self, cx);
        row(cx, sort, field)
    }
}

impl Domain for Row {
    fn bind(self, cx: &mut Context) -> Row {
        self.row(cx)
    }

    fn row(self, cx: &Context) -> Row {
        assert!(self.sort.context == cx.id, "row belongs to another context");
        self
    }
}

/// The row's physical columns.
#[inline(always)]
pub fn columns(row: &Row) -> &[ColumnId] {
    &row.layout[row.start..row.start + row.sort.len]
}

/// Changes only a row's record wrappers, preserving declared argument order
/// and storage.
///
/// Nested records remain nominal; conversion never searches through multiple
/// wrappers.
///
/// # Panics
///
/// Panics if the row or `sort` belongs to another context, or `sort` is
/// incompatible with the row's sort.
pub fn coerce(cx: &Context, row: &mut Row, sort: Sort) {
    assert!(
        cx.compatible(row.sort, sort),
        "incompatible logical sorts: coercion requires a record and its declared argument sort"
    );
    row.sort = sort;
    row.members = members(cx, sort).into();
}

/// The immediate components of the row, sharing its immutable column
/// bindings.
///
/// Record boundaries are retained.
///
/// # Panics
///
/// Panics if the row is a scalar, or, during iteration, belongs to another
/// context.
pub fn children<'a>(cx: &'a Context, row: &'a Row) -> impl ExactSizeIterator<Item = Row> + 'a {
    assert!(
        !row.sort.kind.is_scalar(),
        "aggregate arguments require a record or product"
    );
    let mut start = row.start;
    row.members.iter().map(move |&sort| {
        let part = Row::new(cx, sort, row.layout.clone(), start);
        start += sort.len;
        part
    })
}

/// The single column of a scalar row.
///
/// # Panics
///
/// Panics if the row is an aggregate.
pub fn scalar_column(row: &Row) -> ColumnId {
    row.sort.scalar();
    columns(row)[0]
}

/// The row's single bit-vector column, keeping Boolean and aggregate inputs
/// distinct.
///
/// # Panics
///
/// Panics if the row is an aggregate or its sort has no bit width.
#[inline(always)]
pub fn bitvector_column(row: &Row) -> ColumnId {
    assert!(
        row.sort.scalar().bit_width().is_some(),
        "operation requires a bit-vector sort"
    );
    scalar_column(row)
}

/// A value packable as row arguments following the declaration's immediate
/// members.
///
/// Word slices encode one scalar, never an implicit list of columns. Packing
/// borrows values so predicates can reuse literals and collections without
/// cloning.
/// Implementations append canonical little-endian words or scalar
/// predicates without modifying existing output; the consuming sink
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
            None => parts.iter().all(|sort| sort.kind.is_scalar()).then_some(n),
        }
    }

    /// Appends this value's encoding for `row` to `out`.
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink);
}

/// Output of row conversion.
///
/// Symbolic sinks reserve each predicate's scalar
/// word span and retain its owner separately at that span's starting offset.
/// Ground sinks reject predicates. Sinks append words and predicate entries
/// without modifying existing output.
pub trait RowSink {
    /// The packed words appended so far.
    fn words(&mut self) -> &mut Vec<u64>;

    /// The predicates retained so far, each with the word offset it starts
    /// at.
    fn predicates(&self) -> &[(usize, Predicate)] {
        &[]
    }

    /// Reserves `p`'s scalar span and retains it; ground sinks reject it.
    fn predicate(&mut self, _p: &Predicate) {
        panic!("row arguments must be ground values");
    }
}

impl RowSink for Vec<u64> {
    fn words(&mut self) -> &mut Vec<u64> {
        self
    }
}

/// Checks the complete adapter output before it enters relation construction
/// or lookup.
///
/// # Panics
///
/// Panics if the row belongs to another context, the appended words do not
/// encode exactly the row's columns, or a predicate starts inside a scalar,
/// belongs to another context, or addresses another column.
#[inline(always)]
pub fn append(cx: &Context, args: impl IntoRow, row: &Row, out: &mut impl RowSink) {
    let start = out.words().len();
    let mut p = out.predicates().len();
    args.append(row.view(cx), out);
    let mut i = start;
    for column in columns(row) {
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
            .get(p)
            .filter(|(offset, _)| *offset < i + sort.words())
        {
            assert_eq!(*offset, i, "predicate must start at a scalar boundary");
            assert!(
                predicate.context() == cx,
                "predicate belongs to another context"
            );
            assert_eq!(
                predicate.column(),
                *column,
                "predicate uses an incompatible column"
            );
            p += 1;
        }
        i += sort.words();
    }
    assert_eq!(i, out.words().len(), "row arguments have too many words");
    assert_eq!(
        p,
        out.predicates().len(),
        "row arguments have misplaced predicates"
    );
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
pub fn integer(value: u128, negative: bool, sort: ScalarSort, out: &mut Vec<u64>) {
    let width = sort
        .bit_width()
        .expect("integer literal sort must be a bit vector");
    if negative {
        assert!(
            width >= 128 || value >= u128::MAX - ((1_u128 << (width - 1)) - 1),
            "negative integer literal exceeds its signed width"
        );
    } else {
        assert!(
            width >= 128 || value < 1_u128 << width,
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
            row.sort.kind == SortKind::Scalar(ScalarSort::bool()),
            "a Boolean literal requires the Boolean sort"
        );
        out.words().push(u64::from(*self));
    }
}

impl IntoRow for [u64] {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
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

/// Appends `p` as the predicate over `column` of `cx`.
fn predicate(p: &Predicate, cx: &Context, column: ColumnId, out: &mut impl RowSink) {
    assert!(p.context() == cx, "predicate belongs to another context");
    assert_eq!(p.column(), column, "predicate uses an incompatible column");
    out.predicate(p);
}

impl IntoRow for Predicate {
    fn append(&self, row: RowRef<'_>, out: &mut impl RowSink) {
        row.sort.scalar();
        predicate(self, row.cx, row.columns[0], out);
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
                    // A nonempty slice spans from its first column to its last.
                    // Like scalars, it leaves an empty record before that span
                    // to an explicit unit argument; an empty slice may fill one.
                    if sort.len == 0 && n == 0 && !self.is_empty() {
                        break;
                    }
                    n += sort.len;
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
                    row.sort.len,
                    "predicate row has the wrong column count"
                );
                for (p, &column) in self.iter().zip(row.columns) {
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
            sort.kind == SortKind::Tuple && cx.members(*sort).is_empty()
        })))
    }

    fn append(&self, row: RowRef<'_>, _out: &mut impl RowSink) {
        assert!(
            !row.sort.kind.is_scalar() && row.members.is_empty(),
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
                let first = sorts.first().filter(|sort| sort.kind == SortKind::Tuple);
                // Whether the tuple supplies exactly the record's fields, each
                // element matching one member's normalized components.
                let record = first.is_some_and(|sort| {
                    let parts = cx.members(*sort);
                    parts.len() == $n && true $(&& {
                        let members;
                        let fields = if parts[$i].kind == SortKind::Product {
                            members = cx.members(parts[$i]);
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
                if row.sort.kind == SortKind::Tuple && self.record_sort(row.cx) != Some(row.sort) {
                    assert_eq!(row.members.len(), $n, "row arguments have the wrong member count");
                    assert_eq!(
                        self.span(row.cx, std::slice::from_ref(&row.sort)),
                        Some(1),
                        "row arguments must match the destination members"
                    );
                    let mut offset = 0;
                    $({
                        let sort = row.members[$i];
                        part(row, sort, offset, |p| self.$i.append(p, out));
                        offset += sort.len;
                    })+
                    debug_assert_eq!(offset, row.sort.len, "record members must cover the row");
                    return;
                }
                let sorts = match row.sort.kind {
                    SortKind::Product => row.members,
                    _ => std::slice::from_ref(&row.sort),
                };
                let mut start = 0;
                let mut offset = 0;
                $({
                    // Match anonymous groups and record fields by their destination span,
                    // including named empty records that occupy no physical columns.
                    let n = self.$i.span(row.cx, &sorts[start..])
                        .expect("row arguments must match the destination members");
                    let end = start + n;
                    let sort = match &sorts[start..end] {
                        [sort] => *sort,
                        parts => row.cx.product_sort(parts.iter().copied()),
                    };
                    part(row, sort, offset, |p| self.$i.append(p, out));
                    start = end;
                    offset += sort.len;
                })+
                assert_eq!(start, sorts.len(), "row arguments have the wrong member count");
                debug_assert_eq!(offset, row.sort.len, "row members must cover the row");
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

/// The concatenation of two rows' physical positions, normalizing only their
/// anonymous product type.
///
/// # Panics
///
/// Panics if a row belongs to another context.
pub fn product(cx: &Context, rows: [&Row; 2]) -> Row {
    for row in rows {
        assert!(row.sort.context == cx.id, "row belongs to another context");
    }
    let [a, b] = rows;
    Row::new(
        cx,
        cx.product_sort([a.sort, b.sort]),
        columns(a).iter().chain(columns(b)).copied().collect(),
        0,
    )
}

#[cfg(test)]
#[path = "../tests/unit/row.rs"]
mod tests;
