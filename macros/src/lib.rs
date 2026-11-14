//! Procedural macros for `sra`: the `Schema` derive.
#![warn(missing_docs)]

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::fold::{self, Fold};
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derives a record's runtime sort and its context-bound declaration bundle.
///
/// For `Route`, generates `RouteSchema` and `Route::schema(cx)`, alongside
/// `Record`, `Sorted`, and `IntoRow` implementations. It also generates a
/// symbolic `RouteView<D, F0, ..>`, with one usually inferred parameter per
/// field, so annotations spell `RouteView<_, _, _>` for two fields. Ordinary
/// `map`/`filt` closures receive it, its public fields are member expressions,
/// and `test`, `eq`, and `ne` address the whole record. Those closures need
/// `View` on every field type, while storage and declarations do not.
/// Reassembling a view pairs its fields, so a `map` over a record yields a
/// Binary universe even when it leaves every field untouched. The companion
/// has public accessor fields, `sort`, `constructor`, and `except`, and the
/// same visibility as `Route`. Named, nongeneric structs are supported; `sort`
/// and `constructor` are reserved. Member types must be declared at least as
/// visible as the record, including types of private fields, because
/// `Record::Args` names each one in a public trait implementation. Generated
/// paths use `::sra`; renamed dependencies need a crate-root alias.
///
/// Generated code binds lowercase locals such as `sort`, `offset`, `row`, and
/// `cx`, so a constant, static, or unit struct of such a name in the deriving scope
/// fails to compile at the derive. `Self` inside a macro in a field type names
/// the companion rather than the record. Records nest one tuple level per
/// field, so about 60 fields, counting nested record depth, need a larger
/// crate-level `#![recursion_limit]`.
///
/// Row packing borrows each field, so the layout must provide aligned field
/// references. Packed fields with alignment one remain supported, including
/// non-`Copy` values. Rust rejects references to unaligned packed fields.
/// Runtime declaration names combine the module path and record identifier.
/// Same-named local records share that namespace and must agree on their
/// declaration when used in one context.
///
/// Scalar fields marked `#[symbolic]` default to the `"bdd"` backend
/// (`typed::SYMBOLIC_BACKEND`). The generated `Sorted::bind` binds every
/// member type, selects that backend for each marked member that is still
/// unbound, and then binds the record, so the default applies when the record
/// is bound rather than when it is declared. Other members bind sparse storage.
/// Marked members still unbound at that point require the `"bdd"` backend, and
/// marking an aggregate field panics at bind rather than at compile time.
/// Binding a record also binds its member types so they can serve as standalone
/// relation endpoints. Member types using the default scalar binding therefore
/// require `sparse` for their anonymous columns, even when the record's
/// declared members use another backend.
#[proc_macro_derive(Schema, attributes(symbolic))]
pub fn schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let error = |message| {
        syn::Error::new_spanned(&input.ident, message)
            .to_compile_error()
            .into()
    };
    if !input.generics.params.is_empty() || input.generics.where_clause.is_some() {
        return error("Schema requires a nongeneric struct");
    }
    let Data::Struct(data) = &input.data else {
        return error("Schema requires a struct with named fields");
    };
    let Fields::Named(fields) = &data.fields else {
        return error("Schema requires a struct with named fields");
    };
    let name = &input.ident;
    let vis = &input.vis;
    let schema = format_ident!("{}Schema", name);
    let view = format_ident!("{}View", name);
    let fields = &fields.named;
    if let Some(attr) = input.attrs.iter().find(|a| a.path().is_ident("symbolic")) {
        return syn::Error::new_spanned(attr, "symbolic applies to scalar fields")
            .to_compile_error()
            .into();
    }
    let mut symbolic = Vec::new();
    for (i, field) in fields.iter().enumerate() {
        let mut marked = false;
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("symbolic")) {
            if marked || !matches!(attr.meta, syn::Meta::Path(_)) {
                return syn::Error::new_spanned(attr, "expected one bare #[symbolic] annotation")
                    .to_compile_error()
                    .into();
            }
            marked = true;
        }
        if marked {
            symbolic.push(i);
        }
    }
    let names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    for field in &names {
        if field.unraw() == "sort" || field.unraw() == "constructor" {
            return syn::Error::new_spanned(field, "field name reserved by Schema")
                .to_compile_error()
                .into();
        }
    }
    // Field type paths retain the source record's `Self` in generated companions.
    // Macro tokens in a field type stay opaque, so they keep the companion's.
    struct Receiver<'a>(&'a syn::Ident);
    impl Fold for Receiver<'_> {
        fn fold_path(&mut self, mut path: syn::Path) -> syn::Path {
            if path.leading_colon.is_none() && path.segments[0].ident == "Self" {
                path.segments[0].ident = self.0.clone();
            }
            fold::fold_path(self, path)
        }

        fn fold_item(&mut self, item: syn::Item) -> syn::Item {
            // Nested items have their own `Self` scope.
            item
        }
    }
    let types: Vec<_> = fields
        .iter()
        .map(|f| Receiver(name).fold_type(f.ty.clone()))
        .collect();
    let args = types
        .iter()
        .rev()
        .fold(quote!(()), |rest, ty| quote!((#ty, #rest)));
    let n = fields.len();
    let indices: Vec<_> = (0..n).collect();
    // Generated parameters must not shadow the record they refer to. Keep the
    // context-bearing input private without reserving another user field name.
    let mut used: Vec<_> = names
        .iter()
        .chain([&name])
        .map(|ident| ident.unraw().to_string())
        .collect();
    let mut fresh = |base: &str| {
        let mut ident = format_ident!("{}", base);
        while used.contains(&ident.to_string()) {
            ident = format_ident!("{}_", ident);
        }
        used.push(ident.to_string());
        ident
    };
    let input = fresh("__sra_input");
    let d = fresh("D");
    let t = fresh("T");
    let p = fresh("P");
    // View parameters defer the `View` requirement to `map` and `filt` use and
    // keep user field types out of the scope of the generated `D`.
    let view_types: Vec<_> = (0..n).map(|i| fresh(&format!("F{i}"))).collect();
    let view_params = quote!(#d: ::sra::typed::Sorted, #(#view_types: ::sra::typed::View),*);
    let view_type = quote!(#view<#d, #(#view_types),*>);
    let view_args = view_types
        .iter()
        .rev()
        .fold(quote!(()), |rest, ty| quote!((#ty, #rest)));
    let view_record = quote!(#name: ::sra::typed::Record<Args = #view_args>);
    let view_condition = quote!(::sra::typed::expr::Condition<
        #d,
        ::sra::typed::expr::Test<<Self as ::sra::typed::IntoQuery>::Node>,
    >);
    let mut view_node = quote!(::sra::query::Join<(), ::sra::map::Transform>);
    let mut view_query = quote!(::sra::typed::IntoQuery::into_query(self.#input.unit()).raw);
    for (field, ty) in names.iter().zip(&view_types).rev() {
        view_node = quote!(::sra::query::Join<
            <<#ty as ::sra::typed::View>::Expr<#d> as ::sra::typed::IntoQuery>::Node,
            #view_node
        >);
        view_query = quote!(::sra::query::IntoQuery::and(
            ::sra::typed::IntoQuery::into_query(self.#field).raw,
            #view_query
        ));
    }
    let label = name.unraw().to_string();
    let qualified = quote!(::core::concat!(::core::module_path!(), "::", #label));
    let members: Vec<_> = names.iter().map(|f| f.unraw().to_string()).collect();
    let fields = quote!([#((#members, <#types as ::sra::typed::Sorted>::sort(cx))),*]
        as [(&str, ::sra::Sort); #n]);
    let doc = format!("Context-bound declarations for [`{name}`], created by [`{name}::schema`].");
    let factory_doc = format!("Declares [`{schema}`] in `cx` without binding physical storage.");
    let view_doc = format!(
        "Symbolic member expressions of one [`{name}`] input, for `map` and `filt` closures."
    );
    let accessor_docs: Vec<_> = names
        .iter()
        .map(|f| format!("The accessor for `{name}::{f}`."))
        .collect();
    let member_docs: Vec<_> = names
        .iter()
        .map(|f| format!("The expression for `{name}::{f}`."))
        .collect();
    // Each record gets its own schema handle with the conventional name
    // TypeSchema.
    quote! {
        #[doc = #doc]
        #[derive(::std::clone::Clone, ::std::marker::Copy)]
        #vis struct #schema {
            /// The record sort.
            pub sort: ::sra::Sort,
            /// The constructor from the member product to the record sort.
            pub constructor: ::sra::typed::FuncDecl<<#name as ::sra::typed::Record>::Args, #name>,
            #(#[doc = #accessor_docs] pub #names: ::sra::typed::FuncDecl<#name, #types>,)*
        }
        impl #name {
            #[doc = #factory_doc]
            #[inline(always)]
            #vis fn schema(cx: &::sra::Context) -> #schema {
                let raw = <Self as ::sra::typed::Record>::declaration(cx);
                #schema {
                    sort: raw.sort,
                    constructor: ::sra::typed::FuncDecl::from_raw(cx, raw.constructor),
                    #(#names: ::sra::typed::FuncDecl::from_raw(cx, raw.accessors[#indices]),)*
                }
            }
        }
        impl #schema {
            /// Projects a record onto every field but `decl`, for comparisons
            /// that ignore one attribute.
            ///
            /// # Panics
            ///
            /// Panics if a scalar leaf of the record is not bound in `cx`.
            #[inline(always)]
            pub fn except<#t: ::sra::typed::Sorted>(
                &self,
                cx: &::sra::Context,
                decl: ::sra::typed::FuncDecl<#name, #t>,
            ) -> ::sra::typed::Query<#name, ::sra::typed::Except<#name>> {
                ::sra::typed::Query::except(cx, decl)
            }
        }
        // Every field remains a function of one original closure input,
        // including nested views.
        #[doc = #view_doc]
        #vis struct #view<#d: ::sra::typed::Sorted, #(#view_types: ::sra::typed::View),*> {
            #(#[doc = #member_docs] pub #names: <#view_types as ::sra::typed::View>::Expr<#d>,)*
            #input: ::sra::typed::expr::Expr<#d, #name>,
        }
        impl<#view_params> ::std::clone::Clone for #view_type
        where #(<#view_types as ::sra::typed::View>::Expr<#d>: ::std::clone::Clone,)* {
            fn clone(&self) -> Self {
                Self {
                    #(#names: ::std::clone::Clone::clone(&self.#names),)*
                    #input: ::std::clone::Clone::clone(&self.#input),
                }
            }
        }
        impl<#(#view_types: ::sra::typed::View),*> ::sra::typed::View for #name
        where #view_record {
            type Expr<#d: ::sra::typed::Sorted> = #view_type;
            #[inline(always)]
            fn view<#d: ::sra::typed::Sorted>(
                input: ::sra::typed::expr::Expr<#d, Self>,
            ) -> #view_type {
                #view {
                    #(#names: input.field::<#view_types>(#indices),)*
                    #input: input,
                }
            }
        }
        impl<#view_params> ::sra::typed::IntoQuery for #view_type
        where #view_record {
            type D = #d;
            type R = #name;
            type Node = #view_node;
            #[inline(always)]
            fn into_query(self) -> ::sra::typed::Query<#d, #name, Self::Node> {
                ::sra::typed::Query::coerced(#view_query)
            }
        }
        impl<#view_params> ::sra::typed::expr::Functional for #view_type
        where #view_record {}
        impl<#view_params> #view_type
        where #view_record {
            /// Tests the whole record against a predicate.
            #[inline(always)]
            pub fn test<#p: ::sra::typed::pred::Predicate<#name>>(self, p: #p) -> #view_condition {
                ::sra::typed::expr::Functional::test(self, p)
            }
            /// Holds where the record equals `value`.
            #[inline(always)]
            pub fn eq(self, value: impl ::sra::IntoRow) -> #view_condition {
                ::sra::typed::expr::Functional::eq(self, value)
            }
            /// Holds where the record differs from `value`.
            #[inline(always)]
            pub fn ne(self, value: impl ::sra::IntoRow) -> #view_condition {
                ::sra::typed::expr::Functional::ne(self, value)
            }
        }
        impl ::sra::typed::Record for #name {
            type Args = #args;
            fn declaration(cx: &::sra::Context) -> ::sra::TupleDecl {
                cx.tuple_decl(
                    #qualified,
                    #fields,
                )
            }
        }
        impl ::sra::typed::Sorted for #name {
            type Shape = <<Self as ::sra::typed::Record>::Args as ::sra::typed::Sorted>::Shape;
            // Each call compares the existing declaration, which the context keeps
            // for an equal name and member list, so the sort is stable within one
            // context without a per-type memo. A new or conflicting list declares.
            fn sort(cx: &::sra::Context) -> ::sra::Sort {
                cx.tuple_sort(#qualified, &(#fields))
                    .unwrap_or_else(|| <Self as ::sra::typed::Record>::declaration(cx).sort)
            }
            // Member types bind first so nested records apply their own
            // marks; a marked member bound earlier by another path keeps its
            // column, so the default is a bind-time selection, not a
            // declaration-time one.
            fn bind(cx: &mut ::sra::Context) -> ::sra::Row {
                let decl = <Self as ::sra::typed::Record>::declaration(cx);
                #(<#types as ::sra::typed::Sorted>::bind(cx);)*
                #(if cx.field(decl.accessors[#symbolic]).is_none() {
                    cx.set_representation(
                        decl.accessors[#symbolic],
                        cx.backend(::sra::typed::SYMBOLIC_BACKEND),
                    );
                })*
                <::sra::Sort as ::sra::row::Domain>::bind(decl.sort, cx)
            }
        }
        impl ::sra::IntoRow for #name {
            fn record_sort(&self, cx: &::sra::Context) -> ::std::option::Option<::sra::Sort> {
                ::std::option::Option::Some(<Self as ::sra::typed::Sorted>::sort(cx))
            }
            fn append(&self, row: ::sra::row::RowRef<'_>, out: &mut impl ::sra::row::RowSink) {
                ::core::assert_eq!(
                    row.sort,
                    <Self as ::sra::typed::Sorted>::sort(row.cx),
                    "record value requires its declared sort"
                );
                let mut offset = 0;
                #({
                    let sort = row.members[#indices];
                    ::sra::row::part(row, sort, offset, |part| {
                        <#types as ::sra::IntoRow>::append(&self.#names, part, out)
                    });
                    offset += sort.len;
                })*
                ::core::debug_assert_eq!(offset, row.sort.len, "record members cover the row");
            }
        }
    }
    .into()
}
