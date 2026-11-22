//! Bound logical rows and their physical layouts.
//!
//! Named records preserve their member grouping. Anonymous products normalize
//! their grouping. Rows of the same sort remain independent occurrences, each
//! with its own layout.

use std::iter;
use std::sync::Arc;

use crate::ast::{FuncDecl, Sort, SortKind, Sorts};
use crate::context::{AlgebraId, Context};
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
