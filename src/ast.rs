//! Logical type declarations of one context.
//!
//! Structural sort dependencies are traced edges. Query operators and source
//! ownership live in ordinary Rust values, outside this declaration store.
//!
//! A sort describes meaning, not a physical store. The scalar leaves of a
//! sort are bound to backends separately, through
//! [`Context::set_representation`] and
//! [`Domain::bind`](crate::row::Domain::bind).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::context::Context;
use crate::scalar::ScalarSort;
use crate::util::dag::{Dag, Exhausted, Grow, NeedGrowth, Shared, structural_hash};

idx! {
    /// One syntax node in a context's AST store.
    ///
    /// Copying its ID neither retains the node nor records its context.
    /// Raw IDs borrow a live handle or structural dependency.
    pub struct AstId;
}

/// Metadata of a named declaration: a symbol namespace or a tuple sort.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NamedInfo {
    /// The declared name, unique within its context.
    pub name: Box<str>,
    /// Field names for a tuple; empty for a symbol namespace.
    pub members: Box<[Box<str>]>,
}

/// The kind of a declared sort.
///
/// Aggregate types stay above the scalar backend boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortKind {
    /// A scalar sort, carrying its complete backend universe.
    Scalar(ScalarSort),
    /// A structural product of member sorts.
    Product,
    /// A named record sort with named members.
    Tuple,
}

impl SortKind {
    /// Whether values of this kind are single scalars stored directly by a
    /// backend.
    pub const fn is_scalar(self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

/// The kind of a declaration node.
///
/// Sorts and generated functions share one declaration store, but remain
/// distinct node kinds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NodeKind {
    /// A sort declaration.
    Sort(SortKind),
    /// A tuple's generated constructor function.
    Constructor,
    /// A tuple's generated member accessor function, with its projected
    /// position.
    Accessor(usize),
}

impl NodeKind {
    /// The sort kind of a sort declaration.
    ///
    /// # Panics
    ///
    /// Panics if `self` is a function declaration.
    pub fn sort(self) -> SortKind {
        match self {
            Self::Sort(kind) => kind,
            _ => panic!("expected a sort declaration"),
        }
    }
}

/// One declaration stored by a context.
///
/// Constructors fix each payload and edge layout; declaration identities and
/// their dependencies live until the context is dropped.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Node {
    /// The declaration kind.
    pub kind: NodeKind,
    /// Flattened scalar-leaf count for sorts; zero for function declarations.
    pub len: usize,
    /// Declaration metadata of a named sort; child and signature sorts remain
    /// DAG edges.
    pub named: Option<Box<NamedInfo>>,
}

/// The declaration DAG: nodes are declarations and edges are their child
/// declarations.
pub type State = Dag<AstId, Node, AstId>;

impl Grow for State {
    type Cx<'a> = ();

    fn grow(&mut self, exhausted: Exhausted, _: ()) {
        // Declarations are few and every canonical node is pinned, so
        // maintenance runs on the calling thread.
        self.maintain(
            exhausted,
            1,
            |&child| Some(child),
            |_| {},
            |node, edges| structural_hash((node, edges)),
        );
    }
}

impl State {
    /// Interns `node` over `children` and pins the canonical result.
    ///
    /// Structural declarations live until their context is dropped.
    /// Unsuccessful publication candidates remain available for collection.
    pub fn intern_decl(&self, node: Node, children: &[AstId]) -> Result<AstId, NeedGrowth> {
        let id = self.intern(
            structural_hash((&node, children)),
            |value, edges| (value, edges) == (&node, children),
            || node.clone(),
            children,
        )?;
        self.pin(id);
        Ok(id)
    }
}

/// The shared syntax pool of one context.
///
/// Reclaimed IDs may be reused, so their numeric order says nothing about DAG
/// order. Traversal order must always be derived from child edges.
///
/// A read guard excludes collection. Guarded traversals use that state
/// directly instead of reacquiring AST access, which could wait for growth
/// while holding its own read guard.
pub struct AstStore {
    /// The declaration DAG, shared between concurrent interning and exclusive
    /// growth.
    pub shared: Shared<State>,
    /// Declared names and their definitions, serialized so each name is
    /// declared once.
    pub named: Mutex<HashMap<Box<str>, AstId>>,
}

impl AstStore {
    /// Declares the named `node` over `children`, retaining one immutable
    /// definition per name for the context's lifetime.
    ///
    /// # Panics
    ///
    /// Panics if `node` is unnamed or its name is already declared with a
    /// different definition.
    pub fn declare(&self, node: Node, children: &[AstId]) -> AstId {
        let name = &node
            .named
            .as_ref()
            .expect("declared node must be named")
            .name;
        let id = *self
            .named
            .lock()
            .unwrap()
            .entry(name.clone())
            .or_insert_with(|| {
                self.shared
                    .retry((), |state| state.intern_decl(node.clone(), children))
            });
        // The name lock is released first, so a conflict panics without
        // poisoning it.
        let same = {
            let state = self.shared.read(());
            let existing = state.node(id);
            *existing == node && existing.edges == children
        };
        assert!(same, "conflicting definition for named sort {name}");
        id
    }
}

impl Default for AstStore {
    fn default() -> Self {
        Self {
            shared: Shared::new(State::with_capacity(256, 512, [])),
            named: Default::default(),
        }
    }
}

/// One context-owned logical type.
///
/// Cached scalar metadata keeps backend access independent of the interner;
/// constructors check the owning context before following aggregate members.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Sort {
    /// The ID of the owning context.
    pub context: u64,
    /// The declaration node in the owning context's syntax store.
    pub id: AstId,
    /// The sort kind, carrying the scalar universe of a scalar sort.
    pub kind: SortKind,
    /// The flattened scalar-leaf count; one for scalars.
    pub len: usize,
}

impl Sort {
    /// The scalar backend domain of this sort.
    ///
    /// # Panics
    ///
    /// Panics if the sort is an aggregate.
    #[inline(always)]
    pub fn scalar(self) -> ScalarSort {
        match self.kind {
            SortKind::Scalar(scalar) => scalar,
            _ => panic!("expected a scalar sort"),
        }
    }
}

/// The combined scalar-leaf count of `sorts`.
///
/// # Panics
///
/// Panics if the count does not fit in `usize`.
fn scalar_len(sorts: &[Sort]) -> usize {
    sorts
        .iter()
        .try_fold(0_usize, |len, sort| len.checked_add(sort.len))
        .expect("scalar leaf count overflow")
}

/// The sort handle of a published sort node.
fn sort_at(context: u64, nodes: &State, id: AstId) -> Sort {
    let node = nodes.node(id);
    Sort {
        context,
        id,
        kind: node.kind.sort(),
        len: node.len,
    }
}

/// A resolved signature copied from one context-owned function declaration.
#[derive(Clone, Copy, Debug)]
pub struct FuncInfo {
    /// The complete relational input type (the argument product for a
    /// constructor).
    pub domain: Sort,
    /// The result sort.
    pub range: Sort,
    /// The projected member position for an accessor; `None` for a
    /// constructor.
    pub member: Option<usize>,
}

impl FuncInfo {
    /// The declaring aggregate: an accessor's domain or a constructor's range.
    pub const fn parent(self) -> Sort {
        if self.member.is_some() {
            self.domain
        } else {
            self.range
        }
    }
}

/// A generated constructor or accessor with a fixed logical signature.
///
/// Like [`Sort`], this is a descriptor: it names a stored declaration in its
/// context and is resolved against that owner when it is used.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FuncDecl {
    /// The ID of the owning context.
    pub context: u64,
    /// The declaration node in the owning context's syntax store.
    pub id: AstId,
}

impl FuncDecl {
    /// Resolves against the supplied owner.
    ///
    /// # Panics
    ///
    /// Panics if the declaration belongs to another context or names a sort.
    pub fn resolve(self, cx: &Context) -> FuncInfo {
        assert_eq!(
            self.context, cx.id,
            "declaration belongs to another context"
        );
        let nodes = cx.ast.shared.read(());
        let node = nodes.node(self.id);
        let member = match node.kind {
            NodeKind::Constructor => None,
            NodeKind::Accessor(i) => Some(i),
            NodeKind::Sort(_) => panic!("expected a function declaration"),
        };
        FuncInfo {
            domain: sort_at(cx.id, &nodes, node.edges[0]),
            range: sort_at(cx.id, &nodes, node.edges[1]),
            member,
        }
    }

    /// The declared name in `cx`: the tuple name for a constructor, the member
    /// name for a tuple accessor, and `project{i}` for a product accessor.
    ///
    /// # Panics
    ///
    /// Panics if the declaration belongs to another context or names a sort.
    pub fn name(self, cx: &Context) -> String {
        let info = self.resolve(cx);
        let nodes = cx.ast.shared.read(());
        let node = nodes.node(info.parent().id);
        match info.member {
            None => node.named.as_ref().unwrap().name.to_string(),
            Some(i) if node.kind == NodeKind::Sort(SortKind::Tuple) => {
                node.named.as_ref().unwrap().members[i].to_string()
            }
            Some(i) => format!("project{i}"),
        }
    }
}

/// A nonrecursive nominal record type and its generated declarations.
///
/// This bundle introduces a logical [`Sort`] together with its constructor and
/// accessor [`FuncDecl`]s. The `Decl` suffix distinguishes the bundle from its
/// `sort` handle; it does not declare a stored relation. Z3 calls similar
/// convenience APIs `TupleSort`; the different name marks that this is a
/// bundle of declarations rather than a distinct sort concept.
///
/// Member representation choices are stored separately, keyed by accessor
/// declaration identity, and do not participate in logical sort identity. An
/// accessor denotes a projection; Z3's predicate representation setting
/// instead targets a Boolean-returning function declaration.
#[derive(Clone)]
pub struct TupleDecl {
    /// The record sort.
    pub sort: Sort,
    /// The constructor from the member product to the record sort.
    pub constructor: FuncDecl,
    /// One accessor per member, in declaration order.
    pub accessors: Vec<FuncDecl>,
}

impl Context {
    /// Appends a product's members or `sort` itself to `parts`.
    fn append_components(&self, sort: Sort, parts: &mut Vec<Sort>) {
        if sort.kind == SortKind::Product {
            parts.extend(self.members(sort));
        } else {
            parts.push(sort);
        }
    }

    /// Equal sorts or direct record/argument conversions at corresponding
    /// product positions.
    ///
    /// Named fields stay intact: conversion never chains through another
    /// record's arguments. A record without arguments may disappear, but only
    /// from one side between corresponding components, so different such
    /// records never match. Storage compatibility is checked when wiring cells.
    ///
    /// # Panics
    ///
    /// Panics if either sort belongs to another context.
    pub fn compatible(&self, a: Sort, b: Sort) -> bool {
        assert_eq!(a.context, self.id, "sort belongs to another context");
        assert_eq!(b.context, self.id, "sort belongs to another context");
        if a == b {
            return true;
        }
        let sides = [a, b].map(|sort| {
            let mut parts = Vec::new();
            self.append_components(sort, &mut parts);
            parts
        });
        let mut args = HashMap::new();
        // Search pairs of positions, with the side that last erased an
        // argumentless record. Such records may disappear from only one side
        // between corresponding components, so different zero-width named
        // members never match by both disappearing.
        let mut seen = HashSet::new();
        let mut stack = vec![([0, 0], None)];
        while let Some(state) = stack.pop() {
            if !seen.insert(state) {
                continue;
            }
            let (at, erased) = state;
            if at[0] == sides[0].len() && at[1] == sides[1].len() {
                return true;
            }
            if at[0] < sides[0].len() && sides[0].get(at[0]) == sides[1].get(at[1]) {
                stack.push(([at[0] + 1, at[1] + 1], None));
            }
            for (i, parts) in sides.iter().enumerate() {
                let Some(&part) = parts.get(at[i]).filter(|p| p.kind == SortKind::Tuple) else {
                    continue;
                };
                let fields = args.entry(part).or_insert_with(|| {
                    let mut fields = Vec::new();
                    for member in self.members(part) {
                        self.append_components(member, &mut fields);
                    }
                    fields
                });
                let mut next = at;
                next[i] += 1;
                if fields.is_empty() {
                    if erased != Some(1 - i) {
                        stack.push((next, Some(i)));
                    }
                } else if sides[1 - i][at[1 - i]..].starts_with(fields) {
                    next[1 - i] += fields.len();
                    stack.push((next, None));
                }
            }
        }
        false
    }

    /// Interns the anonymous product of `sorts`.
    ///
    /// Anonymous products are flat, with unit removed and singletons
    /// collapsed. Named records remain whole components even when they contain
    /// no scalar leaves.
    ///
    /// # Panics
    ///
    /// Panics if a sort belongs to another context or the scalar-leaf count
    /// overflows.
    pub fn product_sort(&self, sorts: impl IntoIterator<Item = Sort>) -> Sort {
        let mut parts = Vec::new();
        for sort in sorts {
            assert_eq!(sort.context, self.id, "sort belongs to another context");
            self.append_components(sort, &mut parts);
        }
        if let [sort] = parts.as_slice() {
            return *sort;
        }
        let children: Vec<_> = parts.iter().map(|sort| sort.id).collect();
        self.intern_sort(SortKind::Product, scalar_len(&parts), &children)
    }

    /// Interns and pins the anonymous sort of `kind` over `children`.
    fn intern_sort(&self, kind: SortKind, len: usize, children: &[AstId]) -> Sort {
        let id = self.ast.shared.retry((), |state| {
            state.intern_decl(
                Node {
                    kind: NodeKind::Sort(kind),
                    len,
                    named: None,
                },
                children,
            )
        });
        Sort {
            context: self.id,
            id,
            kind,
            len,
        }
    }

    /// The empty product in this context.
    pub fn unit_sort(&self) -> Sort {
        self.product_sort([])
    }

    /// Immediate member sorts, without physical binding.
    ///
    /// Scalars have no members.
    ///
    /// # Panics
    ///
    /// Panics if `sort` belongs to another context.
    pub fn members(&self, sort: Sort) -> Vec<Sort> {
        assert_eq!(sort.context, self.id, "sort belongs to another context");
        let nodes = self.ast.shared.read(());
        nodes
            .node(sort.id)
            .edges
            .iter()
            .map(|&id| sort_at(self.id, &nodes, id))
            .collect()
    }

    /// Interns the anonymous scalar sort over `scalar`.
    fn scalar_sort(&self, scalar: ScalarSort) -> Sort {
        self.intern_sort(SortKind::Scalar(scalar), 1, &[])
    }

    /// Interns the exact-width bit-vector sort in this context.
    ///
    /// # Panics
    ///
    /// Panics if `width` is zero.
    pub fn bv_sort(&self, width: u32) -> Sort {
        self.scalar_sort(ScalarSort::bit_vec(width))
    }

    /// Declares a named symbol namespace in this context, with its value
    /// dictionary allocated lazily by [`Context::interner`].
    ///
    /// Every ID is a valid value, so full/complement/count use all 2^64 IDs.
    ///
    /// # Panics
    ///
    /// Panics if `name` already names a different declaration.
    pub fn symbol_sort(&self, name: impl Into<Box<str>>) -> Sort {
        let id = self.ast.declare(
            Node {
                kind: NodeKind::Sort(SortKind::Scalar(ScalarSort::symbol())),
                len: 1,
                named: Some(Box::new(NamedInfo {
                    name: name.into(),
                    members: Box::new([]),
                })),
            },
            &[],
        );
        Sort {
            context: self.id,
            id,
            kind: SortKind::Scalar(ScalarSort::symbol()),
            len: 1,
        }
    }

    /// The Boolean sort in this context, distinct from a one-bit bit-vector.
    pub fn bool_sort(&self) -> Sort {
        self.scalar_sort(ScalarSort::bool())
    }

    /// Interns a generated function without binding any physical storage.
    ///
    /// # Panics
    ///
    /// Panics if a sort belongs to another context, an accessor's `member`
    /// does not project `range` out of `domain`, or a constructor's `range`
    /// is not a tuple whose member product is `domain`.
    pub fn func_decl(&self, domain: Sort, range: Sort, member: Option<usize>) -> FuncDecl {
        assert_eq!(
            domain.context, self.id,
            "declaration belongs to another context"
        );
        assert_eq!(
            range.context, self.id,
            "declaration belongs to another context"
        );
        if let Some(i) = member {
            assert!(
                !domain.kind.is_scalar(),
                "projection requires an aggregate sort"
            );
            let nodes = self.ast.shared.read(());
            assert_eq!(
                nodes.node(domain.id).edges.get(i),
                Some(&range.id),
                "invalid accessor signature"
            );
        } else {
            assert_eq!(
                range.kind,
                SortKind::Tuple,
                "constructor requires a tuple sort"
            );
            let mut parts = Vec::new();
            self.append_components(domain, &mut parts);
            let mut fields = Vec::new();
            for sort in self.members(range) {
                self.append_components(sort, &mut fields);
            }
            assert_eq!(parts, fields, "invalid constructor signature");
        }
        let id = self.ast.shared.retry((), |state| {
            state.intern_decl(
                Node {
                    kind: member.map_or(NodeKind::Constructor, NodeKind::Accessor),
                    len: 0,
                    named: None,
                },
                &[domain.id, range.id],
            )
        });
        FuncDecl {
            context: self.id,
            id,
        }
    }

    /// A positional accessor with an explicit product or record domain.
    ///
    /// # Panics
    ///
    /// Panics if `domain` belongs to another context, is a scalar, or `i` is
    /// out of bounds.
    pub fn project(&self, domain: Sort, i: usize) -> FuncDecl {
        assert!(
            !domain.kind.is_scalar(),
            "projection requires an aggregate sort"
        );
        assert_eq!(domain.context, self.id, "sort belongs to another context");
        let range = {
            let nodes = self.ast.shared.read(());
            let id = *nodes
                .node(domain.id)
                .edges
                .get(i)
                .expect("projection index must name a component of the sort");
            sort_at(self.id, &nodes, id)
        };
        self.func_decl(domain, range, Some(i))
    }

    /// Declares or retrieves an identical nominal record in this context, with
    /// one constructor and ordered, record-scoped accessors.
    ///
    /// Identical declarations reuse their name; conflicting definitions are
    /// caller errors. Declaration neither initializes physical configuration nor
    /// binds storage, and preserves any existing member configuration.
    /// Configure scalar accessors with [`Context::set_representation`] before
    /// binding them; otherwise they bind sparse storage. Nested records use
    /// their own member declarations.
    ///
    /// # Panics
    ///
    /// Panics if a member sort belongs to another context, the scalar-leaf
    /// count overflows, or `name` already names a different declaration.
    pub fn tuple_decl(
        &self,
        name: impl Into<Box<str>>,
        members: impl IntoIterator<Item = (impl Into<Box<str>>, Sort)>,
    ) -> TupleDecl {
        let (names, parts): (Vec<_>, Vec<_>) = members
            .into_iter()
            .map(|(label, sort)| {
                assert_eq!(sort.context, self.id, "sort belongs to another context");
                (label.into(), sort)
            })
            .unzip();
        let len = scalar_len(&parts);
        let id = self.ast.declare(
            Node {
                kind: NodeKind::Sort(SortKind::Tuple),
                len,
                named: Some(Box::new(NamedInfo {
                    name: name.into(),
                    members: names.into(),
                })),
            },
            &parts.iter().map(|sort| sort.id).collect::<Vec<_>>(),
        );
        let sort = Sort {
            context: self.id,
            id,
            kind: SortKind::Tuple,
            len,
        };
        let constructor = self.func_decl(self.product_sort(parts.iter().copied()), sort, None);
        let accessors = parts
            .into_iter()
            .enumerate()
            .map(|(i, range)| self.func_decl(sort, range, Some(i)))
            .collect();
        TupleDecl {
            sort,
            constructor,
            accessors,
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/ast.rs"]
mod tests;
