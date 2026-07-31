use proc_macro2::{Ident, Literal, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use std::collections::{HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::{
    BinOp, Expr, ExprBinary, ExprCall, ExprGroup, ExprLit, ExprParen, ExprPath, ExprUnary, Lit,
    Path, Result, Token, UnOp, Visibility, braced, bracketed, parenthesized,
};

struct Leaf {
    alias: Ident,
    rust: Path,
    cuda: Ident,
}

enum RawStatement {
    Local {
        name: Ident,
        mutable: bool,
        value: Expr,
    },
    If {
        condition: Expr,
        assignments: Vec<(Ident, Expr)>,
    },
}

struct RawBody {
    statements: Vec<RawStatement>,
    result: Expr,
}

#[derive(Default)]
struct EmissionSurfaces {
    generic: bool,
    runtime: bool,
    order2: bool,
    third: bool,
    fourth: bool,
    witnesses: bool,
    cuda: bool,
}

impl EmissionSurfaces {
    fn insert(&mut self, surface: &Ident) -> Result<()> {
        let selected = match surface.to_string().as_str() {
            "generic" => &mut self.generic,
            "runtime" => &mut self.runtime,
            "order2" => &mut self.order2,
            "third" => &mut self.third,
            "fourth" => &mut self.fourth,
            "witnesses" => &mut self.witnesses,
            "cuda" => &mut self.cuda,
            _ => {
                return Err(syn::Error::new_spanned(
                    surface,
                    "row_program emission surface must be one of `generic`, `runtime`, `order2`, `third`, `fourth`, `witnesses`, or `cuda`",
                ));
            }
        };
        if *selected {
            return Err(syn::Error::new_spanned(
                surface,
                format!("duplicate row_program emission surface `{surface}`"),
            ));
        }
        *selected = true;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        !(self.generic
            || self.runtime
            || self.order2
            || self.third
            || self.fourth
            || self.witnesses
            || self.cuda)
    }
}

pub(crate) struct Input {
    visibility: Visibility,
    name: Ident,
    primaries: Vec<Ident>,
    constants: Vec<Ident>,
    emissions: EmissionSurfaces,
    leaves: Vec<Leaf>,
    witnesses: Vec<Ident>,
    body: RawBody,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let name = input.parse()?;

        let arguments;
        parenthesized!(arguments in input);
        let mut primaries = Vec::new();
        while !arguments.is_empty() && !arguments.peek(Token![;]) {
            primaries.push(arguments.parse::<Ident>()?);
            if arguments.peek(Token![,]) {
                arguments.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        let mut constants = Vec::new();
        if arguments.peek(Token![;]) {
            arguments.parse::<Token![;]>()?;
            while !arguments.is_empty() {
                constants.push(arguments.parse::<Ident>()?);
                if arguments.peek(Token![,]) {
                    arguments.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
        }
        if primaries.is_empty() {
            return Err(input.error("row_program requires at least one primary"));
        }
        if !arguments.is_empty() {
            return Err(arguments.error("invalid row_program argument list"));
        }

        let emit_keyword = input.parse::<Ident>()?;
        if emit_keyword != "emit" {
            return Err(syn::Error::new_spanned(
                emit_keyword,
                "row_program expects mandatory `emit [ ... ];` surfaces",
            ));
        }
        let emission_tokens;
        bracketed!(emission_tokens in input);
        let mut emissions = EmissionSurfaces::default();
        while !emission_tokens.is_empty() {
            let surface = emission_tokens.parse::<Ident>()?;
            emissions.insert(&surface)?;
            if emission_tokens.peek(Token![,]) {
                emission_tokens.parse::<Token![,]>()?;
            } else {
                break;
            }
        }
        if emissions.is_empty() {
            return Err(emission_tokens.error("row_program must emit at least one surface"));
        }
        if !emission_tokens.is_empty() {
            return Err(emission_tokens.error("invalid row_program emission surface list"));
        }
        input.parse::<Token![;]>()?;

        let leaves_keyword = input.parse::<Ident>()?;
        if leaves_keyword != "leaves" {
            return Err(syn::Error::new_spanned(
                leaves_keyword,
                "row_program expects `leaves { ... }`",
            ));
        }
        let leaf_tokens;
        braced!(leaf_tokens in input);
        let mut leaves = Vec::new();
        while !leaf_tokens.is_empty() {
            let alias = leaf_tokens.parse()?;
            leaf_tokens.parse::<Token![=>]>()?;
            let rust = leaf_tokens.parse()?;
            leaf_tokens.parse::<Token![=>]>()?;
            let cuda = leaf_tokens.parse()?;
            leaves.push(Leaf { alias, rust, cuda });
            if leaf_tokens.peek(Token![,]) {
                leaf_tokens.parse::<Token![,]>()?;
            }
        }

        let witnesses_keyword = input.parse::<Ident>()?;
        if witnesses_keyword != "witnesses" {
            return Err(syn::Error::new_spanned(
                witnesses_keyword,
                "row_program expects `witnesses [ ... ]`",
            ));
        }
        let witness_tokens;
        bracketed!(witness_tokens in input);
        let mut witnesses = Vec::new();
        while !witness_tokens.is_empty() {
            witnesses.push(witness_tokens.parse()?);
            if witness_tokens.peek(Token![,]) {
                witness_tokens.parse::<Token![,]>()?;
            }
        }
        input.parse::<Token![;]>()?;

        let body_tokens;
        braced!(body_tokens in input);
        let mut statements = Vec::new();
        let mut result = None;
        while !body_tokens.is_empty() {
            if body_tokens.peek(Token![let]) {
                body_tokens.parse::<Token![let]>()?;
                let mutable = if body_tokens.peek(Token![mut]) {
                    body_tokens.parse::<Token![mut]>()?;
                    true
                } else {
                    false
                };
                let name = body_tokens.parse()?;
                body_tokens.parse::<Token![=]>()?;
                let value = body_tokens.parse()?;
                body_tokens.parse::<Token![;]>()?;
                statements.push(RawStatement::Local {
                    name,
                    mutable,
                    value,
                });
                continue;
            }
            if body_tokens.peek(Token![if]) {
                body_tokens.parse::<Token![if]>()?;
                let condition_tokens;
                parenthesized!(condition_tokens in body_tokens);
                let condition = condition_tokens.parse()?;
                if !condition_tokens.is_empty() {
                    return Err(condition_tokens.error("invalid row_program condition"));
                }
                let assignment_tokens;
                braced!(assignment_tokens in body_tokens);
                let mut assignments = Vec::new();
                while !assignment_tokens.is_empty() {
                    let target = assignment_tokens.parse()?;
                    assignment_tokens.parse::<Token![=]>()?;
                    let value = assignment_tokens.parse()?;
                    assignment_tokens.parse::<Token![;]>()?;
                    assignments.push((target, value));
                }
                statements.push(RawStatement::If {
                    condition,
                    assignments,
                });
                continue;
            }
            if body_tokens.peek(Token![return]) {
                body_tokens.parse::<Token![return]>()?;
                if result.is_some() {
                    return Err(body_tokens.error("row_program has more than one return"));
                }
                result = Some(body_tokens.parse()?);
                body_tokens.parse::<Token![;]>()?;
                if !body_tokens.is_empty() {
                    return Err(body_tokens.error("row_program return must be last"));
                }
                continue;
            }
            return Err(body_tokens.error("row_program supports only let, if, and return"));
        }
        let result = result.ok_or_else(|| input.error("row_program requires a final return"))?;

        Ok(Self {
            visibility,
            name,
            primaries,
            constants,
            emissions,
            leaves,
            witnesses,
            body: RawBody { statements, result },
        })
    }
}

#[derive(Clone)]
enum ProgramExpr {
    Path(Ident),
    Zero,
    Neg(Box<ProgramExpr>),
    Scale(Box<ProgramExpr>, Expr),
    AddConstant(Box<ProgramExpr>, Expr),
    Add(Box<ProgramExpr>, Box<ProgramExpr>),
    Mul(Box<ProgramExpr>, Box<ProgramExpr>),
    Compose {
        leaf: usize,
        value: Ident,
        arguments: Vec<Expr>,
    },
}

enum Statement {
    Local {
        name: Ident,
        mutable: bool,
        value: ProgramExpr,
    },
    If {
        condition: Expr,
        assignments: Vec<(Ident, ProgramExpr)>,
    },
}

fn bare_call_name(call: &ExprCall) -> Result<&Ident> {
    let Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new_spanned(
            &call.func,
            "row_program operations must use bare function names",
        ));
    };
    path.path.get_ident().ok_or_else(|| {
        syn::Error::new_spanned(
            &call.func,
            "row_program operations must use bare function names",
        )
    })
}

fn path_ident(path: &ExprPath) -> Result<&Ident> {
    path.path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(path, "row_program paths must be identifiers"))
}

fn numeric_literal(literal: &ExprLit) -> bool {
    matches!(&literal.lit, Lit::Float(_) | Lit::Int(_))
}

fn validate_scalar(expression: &Expr, constants: &HashSet<String>) -> Result<()> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if constants.contains(&ident.to_string()) {
                Ok(())
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    format!("unknown row_program scalar `{ident}`"),
                ))
            }
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(()),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            validate_scalar(expr, constants)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => validate_scalar(expr, constants),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) if matches!(
            op,
            BinOp::Add(_)
                | BinOp::Sub(_)
                | BinOp::Mul(_)
                | BinOp::Div(_)
                | BinOp::Eq(_)
                | BinOp::Ne(_)
                | BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
        ) =>
        {
            validate_scalar(left, constants)?;
            validate_scalar(right, constants)
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program scalar expression",
        )),
    }
}

fn parse_program_expr(
    expression: &Expr,
    bindings: &HashSet<String>,
    constants: &HashSet<String>,
    leaves: &HashMap<String, usize>,
) -> Result<ProgramExpr> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if bindings.contains(&ident.to_string()) {
                Ok(ProgramExpr::Path(ident.clone()))
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    format!("unknown row_program jet `{ident}`"),
                ))
            }
        }
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            parse_program_expr(expr, bindings, constants, leaves)
        }
        Expr::Call(call) => {
            let operation = bare_call_name(call)?.to_string();
            let arguments = call.args.iter().collect::<Vec<_>>();
            match operation.as_str() {
                "zero" if arguments.is_empty() => Ok(ProgramExpr::Zero),
                "neg" if arguments.len() == 1 => Ok(ProgramExpr::Neg(Box::new(
                    parse_program_expr(arguments[0], bindings, constants, leaves)?,
                ))),
                "scale" | "add_constant" if arguments.len() == 2 => {
                    let value = parse_program_expr(arguments[0], bindings, constants, leaves)?;
                    validate_scalar(arguments[1], constants)?;
                    if operation == "scale" {
                        Ok(ProgramExpr::Scale(Box::new(value), arguments[1].clone()))
                    } else {
                        Ok(ProgramExpr::AddConstant(
                            Box::new(value),
                            arguments[1].clone(),
                        ))
                    }
                }
                "add" | "mul" if arguments.len() == 2 => {
                    let left = parse_program_expr(arguments[0], bindings, constants, leaves)?;
                    let right = parse_program_expr(arguments[1], bindings, constants, leaves)?;
                    if operation == "add" {
                        Ok(ProgramExpr::Add(Box::new(left), Box::new(right)))
                    } else {
                        Ok(ProgramExpr::Mul(Box::new(left), Box::new(right)))
                    }
                }
                "compose" if arguments.len() >= 2 => {
                    let Expr::Path(leaf_path) = arguments[0] else {
                        return Err(syn::Error::new_spanned(
                            arguments[0],
                            "row_program compose leaf must be an identifier",
                        ));
                    };
                    let leaf_ident = path_ident(leaf_path)?;
                    let leaf = leaves
                        .get(&leaf_ident.to_string())
                        .copied()
                        .ok_or_else(|| {
                            syn::Error::new_spanned(
                                leaf_ident,
                                format!("unknown row_program leaf `{leaf_ident}`"),
                            )
                        })?;
                    let Expr::Path(value_path) = arguments[1] else {
                        return Err(syn::Error::new_spanned(
                            arguments[1],
                            "row_program compose value must be a named jet",
                        ));
                    };
                    let value = path_ident(value_path)?.clone();
                    if !bindings.contains(&value.to_string()) {
                        return Err(syn::Error::new_spanned(
                            value,
                            "row_program compose value is not defined",
                        ));
                    }
                    let mut scalar_arguments = Vec::new();
                    for argument in &arguments[2..] {
                        validate_scalar(argument, constants)?;
                        scalar_arguments.push((*argument).clone());
                    }
                    Ok(ProgramExpr::Compose {
                        leaf,
                        value,
                        arguments: scalar_arguments,
                    })
                }
                _ => Err(syn::Error::new_spanned(
                    call,
                    format!(
                        "unsupported row_program operation `{operation}` or wrong argument count"
                    ),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "row_program jet expressions use only named jets and explicit operations",
        )),
    }
}

fn rust_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident),
        ProgramExpr::Zero => quote!(S::constant(0.0)),
        ProgramExpr::Neg(value) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.neg() })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.scale(#scalar) })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_expression(value, leaves);
            quote!({ let value = #value; value.add(&S::constant(#scalar)) })
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_expression(left, leaves);
            let right = rust_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.add(&right) })
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_expression(left, leaves);
            let right = rust_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.mul(&right) })
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let value_ident = value;
            let rust_leaf = &leaves[*leaf].rust;
            quote!({
                let value = #value_ident;
                value.compose_unary(#rust_leaf(value.value(), #(#arguments),*))
            })
        }
    }
}

fn rust_runtime_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident.clone()),
        ProgramExpr::Zero => quote!(S::constant(
            0.0,
            __row_program_dimension,
            __row_program_workspace
        )),
        ProgramExpr::Neg(value) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({ let value = #value; value.neg() })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({ let value = #value; value.scale(#scalar) })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_runtime_expression(value, leaves);
            quote!({
                let value = #value;
                value.add_constant(#scalar)
            })
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_runtime_expression(left, leaves);
            let right = rust_runtime_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.add(&right) })
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_runtime_expression(left, leaves);
            let right = rust_runtime_expression(right, leaves);
            quote!({ let left = #left; let right = #right; left.mul(&right) })
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let value_ident = value;
            let rust_leaf = &leaves[*leaf].rust;
            quote!({
                let value = #value_ident.clone();
                value.compose_unary(#rust_leaf(value.value(), #(#arguments),*))
            })
        }
    }
}

fn rust_scalar_expression(expression: &ProgramExpr, leaves: &[Leaf]) -> TokenStream2 {
    match expression {
        ProgramExpr::Path(ident) => quote!(#ident),
        ProgramExpr::Zero => quote!(0.0),
        ProgramExpr::Neg(value) => {
            let value = rust_scalar_expression(value, leaves);
            quote!(-(#value))
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = rust_scalar_expression(value, leaves);
            quote!((#value) * (#scalar))
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let value = rust_scalar_expression(value, leaves);
            quote!((#value) + (#scalar))
        }
        ProgramExpr::Add(left, right) => {
            let left = rust_scalar_expression(left, leaves);
            let right = rust_scalar_expression(right, leaves);
            quote!((#left) + (#right))
        }
        ProgramExpr::Mul(left, right) => {
            let left = rust_scalar_expression(left, leaves);
            let right = rust_scalar_expression(right, leaves);
            quote!((#left) * (#right))
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let rust_leaf = &leaves[*leaf].rust;
            quote!(#rust_leaf(#value, #(#arguments),*)[0])
        }
    }
}

fn collect_dependencies(expression: &ProgramExpr, dependencies: &mut HashSet<String>) {
    match expression {
        ProgramExpr::Path(ident) => {
            dependencies.insert(ident.to_string());
        }
        ProgramExpr::Zero => {}
        ProgramExpr::Neg(value)
        | ProgramExpr::Scale(value, _)
        | ProgramExpr::AddConstant(value, _) => collect_dependencies(value, dependencies),
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            collect_dependencies(left, dependencies);
            collect_dependencies(right, dependencies);
        }
        ProgramExpr::Compose { value, .. } => {
            dependencies.insert(value.to_string());
        }
    }
}

fn witness_dependencies(statements: &[Statement], witnesses: &[Ident]) -> HashSet<String> {
    let mut dependencies = witnesses
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    loop {
        let previous_len = dependencies.len();
        for statement in statements.iter().rev() {
            match statement {
                Statement::Local { name, value, .. } => {
                    if dependencies.contains(&name.to_string()) {
                        collect_dependencies(value, &mut dependencies);
                    }
                }
                Statement::If { assignments, .. } => {
                    for (target, value) in assignments {
                        if dependencies.contains(&target.to_string()) {
                            collect_dependencies(value, &mut dependencies);
                        }
                    }
                }
            }
        }
        if dependencies.len() == previous_len {
            return dependencies;
        }
    }
}

fn collect_scalar_expression_dependencies(
    expression: &Expr,
    dependencies: &mut HashSet<String>,
) -> Result<()> {
    match expression {
        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                dependencies.insert(ident.to_string());
            }
            Ok(())
        }
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            collect_scalar_expression_dependencies(expr, dependencies)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => collect_scalar_expression_dependencies(expr, dependencies),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) if matches!(
            op,
            BinOp::Add(_)
                | BinOp::Sub(_)
                | BinOp::Mul(_)
                | BinOp::Div(_)
                | BinOp::Eq(_)
                | BinOp::Ne(_)
                | BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
        ) =>
        {
            collect_scalar_expression_dependencies(left, dependencies)?;
            collect_scalar_expression_dependencies(right, dependencies)
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(()),
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program scalar dependency expression",
        )),
    }
}

fn collect_program_scalar_dependencies(
    expression: &ProgramExpr,
    dependencies: &mut HashSet<String>,
) -> Result<()> {
    match expression {
        ProgramExpr::Path(_) | ProgramExpr::Zero => Ok(()),
        ProgramExpr::Neg(value) => collect_program_scalar_dependencies(value, dependencies),
        ProgramExpr::Scale(value, scalar) | ProgramExpr::AddConstant(value, scalar) => {
            collect_program_scalar_dependencies(value, dependencies)?;
            collect_scalar_expression_dependencies(scalar, dependencies)
        }
        ProgramExpr::Add(left, right) | ProgramExpr::Mul(left, right) => {
            collect_program_scalar_dependencies(left, dependencies)?;
            collect_program_scalar_dependencies(right, dependencies)
        }
        ProgramExpr::Compose { arguments, .. } => {
            for argument in arguments {
                collect_scalar_expression_dependencies(argument, dependencies)?;
            }
            Ok(())
        }
    }
}

fn witness_scalar_dependencies(
    statements: &[Statement],
    jet_dependencies: &HashSet<String>,
) -> Result<HashSet<String>> {
    let mut dependencies = HashSet::new();
    for statement in statements {
        match statement {
            Statement::Local { name, value, .. }
                if jet_dependencies.contains(&name.to_string()) =>
            {
                collect_program_scalar_dependencies(value, &mut dependencies)?;
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut condition_is_needed = false;
                for (target, value) in assignments {
                    if jet_dependencies.contains(&target.to_string()) {
                        collect_program_scalar_dependencies(value, &mut dependencies)?;
                        condition_is_needed = true;
                    }
                }
                if condition_is_needed {
                    collect_scalar_expression_dependencies(condition, &mut dependencies)?;
                }
            }
            Statement::Local { .. } => {}
        }
    }
    Ok(dependencies)
}

#[derive(Clone, Copy)]
enum SymbolicTarget {
    Rust,
    Cuda,
}

fn symbolic_scalar(
    expression: &Expr,
    constants: &HashSet<String>,
    target: SymbolicTarget,
) -> Result<String> {
    match expression {
        Expr::Path(path) => {
            let ident = path_ident(path)?;
            if constants.contains(&ident.to_string()) {
                Ok(match target {
                    SymbolicTarget::Rust => ident.to_string(),
                    SymbolicTarget::Cuda => format!("in.{ident}"),
                })
            } else {
                Err(syn::Error::new_spanned(
                    ident,
                    "unknown row_program symbolic scalar",
                ))
            }
        }
        Expr::Lit(literal) if numeric_literal(literal) => Ok(quote!(#literal).to_string()),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            Ok(format!("({})", symbolic_scalar(expr, constants, target)?))
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => Ok(format!("-({})", symbolic_scalar(expr, constants, target)?)),
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let operator = match op {
                BinOp::Add(_) => "+",
                BinOp::Sub(_) => "-",
                BinOp::Mul(_) => "*",
                BinOp::Div(_) => "/",
                BinOp::Eq(_) => "==",
                BinOp::Ne(_) => "!=",
                BinOp::Lt(_) => "<",
                BinOp::Le(_) => "<=",
                BinOp::Gt(_) => ">",
                BinOp::Ge(_) => ">=",
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "unsupported row_program symbolic scalar operator",
                    ));
                }
            };
            Ok(format!(
                "({} {operator} {})",
                symbolic_scalar(left, constants, target)?,
                symbolic_scalar(right, constants, target)?
            ))
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_program symbolic scalar expression",
        )),
    }
}

#[derive(Clone)]
struct SymbolicJet {
    value: String,
    gradient: Vec<Option<String>>,
    // Only entries with a <= b are populated. The generated CUDA computes the
    // packed triangle once and scatters it symmetrically at the output seam.
    hessian: Vec<Option<String>>,
}

#[derive(Clone)]
struct SymbolicSupport {
    gradient: Vec<bool>,
    hessian: Vec<bool>,
}

impl SymbolicSupport {
    fn empty(dimension: usize) -> Self {
        Self {
            gradient: vec![false; dimension],
            hessian: vec![false; dimension * dimension],
        }
    }

    fn include(&mut self, jet: &SymbolicJet) {
        for (present, component) in self.gradient.iter_mut().zip(&jet.gradient) {
            *present |= component.is_some();
        }
        for (present, component) in self.hessian.iter_mut().zip(&jet.hessian) {
            *present |= component.is_some();
        }
    }
}

impl SymbolicJet {
    fn zero(dimension: usize) -> Self {
        Self {
            value: "0.0".to_string(),
            gradient: vec![None; dimension],
            hessian: vec![None; dimension * dimension],
        }
    }

    fn primary(name: &str, axis: usize, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = name.to_string();
        out.gradient[axis] = Some("1.0".to_string());
        out
    }

    fn constant(value: String, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = value;
        out
    }

    fn support(&self) -> SymbolicSupport {
        let mut support = SymbolicSupport::empty(self.gradient.len());
        support.include(self);
        support
    }

    fn reference(name: &str, support: &SymbolicSupport, dimension: usize) -> Self {
        let mut out = Self::zero(dimension);
        out.value = format!("{name}_v");
        for axis in 0..dimension {
            if support.gradient[axis] {
                out.gradient[axis] = Some(format!("{name}_g{axis}"));
            }
            for other in axis..dimension {
                let index = axis * dimension + other;
                if support.hessian[index] {
                    out.hessian[index] = Some(format!("{name}_h{axis}_{other}"));
                }
            }
        }
        out
    }
}

fn symbolic_is_zero(value: &str) -> bool {
    value == "0.0"
}

fn symbolic_is_one(value: &str) -> bool {
    value == "1.0"
}

fn symbolic_is_negative_one(value: &str) -> bool {
    matches!(value, "-1.0" | "-(1.0)" | "(-1.0)")
}

fn symbolic_negate(value: &str) -> String {
    if symbolic_is_zero(value) {
        "0.0".to_string()
    } else if symbolic_is_negative_one(value) {
        "1.0".to_string()
    } else if symbolic_is_one(value) {
        "-1.0".to_string()
    } else {
        format!("-({value})")
    }
}

fn symbolic_add(left: &str, right: &str) -> String {
    if symbolic_is_zero(left) {
        right.to_string()
    } else if symbolic_is_zero(right) {
        left.to_string()
    } else {
        format!("({left} + {right})")
    }
}

fn symbolic_multiply(left: &str, right: &str) -> String {
    if symbolic_is_zero(left) || symbolic_is_zero(right) {
        "0.0".to_string()
    } else if symbolic_is_one(left) {
        right.to_string()
    } else if symbolic_is_one(right) {
        left.to_string()
    } else if symbolic_is_negative_one(left) {
        symbolic_negate(right)
    } else if symbolic_is_negative_one(right) {
        symbolic_negate(left)
    } else {
        format!("({left} * {right})")
    }
}

fn symbolic_add_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(symbolic_add(left, right)),
        (Some(value), None) | (None, Some(value)) => Some(value.clone()),
        (None, None) => None,
    }
}

fn symbolic_multiply_component(left: &Option<String>, right: &Option<String>) -> Option<String> {
    match (left, right) {
        (Some(left), Some(right)) => Some(symbolic_multiply(left, right)),
        _ => None,
    }
}

fn symbolic_scale_component(component: &Option<String>, scalar: &str) -> Option<String> {
    component
        .as_ref()
        .map(|component| symbolic_multiply(component, scalar))
}

fn symbolic_add_jets(left: SymbolicJet, right: SymbolicJet) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_add(&left.value, &right.value),
        gradient: left
            .gradient
            .iter()
            .zip(&right.gradient)
            .map(|(left, right)| symbolic_add_component(left, right))
            .collect(),
        hessian: left
            .hessian
            .iter()
            .zip(&right.hessian)
            .map(|(left, right)| symbolic_add_component(left, right))
            .collect(),
    }
}

fn symbolic_multiply_jets(left: SymbolicJet, right: SymbolicJet) -> SymbolicJet {
    let dimension = left.gradient.len();
    let mut gradient = vec![None; dimension];
    let mut hessian = vec![None; dimension * dimension];
    for axis in 0..dimension {
        gradient[axis] = symbolic_add_component(
            &symbolic_scale_component(&right.gradient[axis], &left.value),
            &symbolic_scale_component(&left.gradient[axis], &right.value),
        );
        for other in axis..dimension {
            let index = axis * dimension + other;
            let inherited_right = symbolic_scale_component(&right.hessian[index], &left.value);
            let cross_forward =
                symbolic_multiply_component(&left.gradient[axis], &right.gradient[other]);
            let cross_reverse =
                symbolic_multiply_component(&left.gradient[other], &right.gradient[axis]);
            let inherited_left = symbolic_scale_component(&left.hessian[index], &right.value);
            hessian[index] = symbolic_add_component(
                &symbolic_add_component(
                    &symbolic_add_component(&inherited_right, &cross_forward),
                    &cross_reverse,
                ),
                &inherited_left,
            );
        }
    }
    SymbolicJet {
        value: symbolic_multiply(&left.value, &right.value),
        gradient,
        hessian,
    }
}

fn symbolic_negate_jet(value: SymbolicJet) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_negate(&value.value),
        gradient: value
            .gradient
            .iter()
            .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
            .collect(),
        hessian: value
            .hessian
            .iter()
            .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
            .collect(),
    }
}

fn symbolic_scale_jet(value: SymbolicJet, scalar: &str) -> SymbolicJet {
    SymbolicJet {
        value: symbolic_multiply(&value.value, scalar),
        gradient: value
            .gradient
            .iter()
            .map(|component| symbolic_scale_component(component, scalar))
            .collect(),
        hessian: value
            .hessian
            .iter()
            .map(|component| symbolic_scale_component(component, scalar))
            .collect(),
    }
}

fn symbolic_compose_jet(input: SymbolicJet, stack: &str, offset: usize) -> SymbolicJet {
    let first = format!("{stack}[{}]", offset + 1);
    let second = format!("{stack}[{}]", offset + 2);
    let dimension = input.gradient.len();
    let mut gradient = vec![None; dimension];
    let mut hessian = vec![None; dimension * dimension];
    for axis in 0..dimension {
        gradient[axis] = symbolic_scale_component(&input.gradient[axis], &first);
        for other in axis..dimension {
            let index = axis * dimension + other;
            let inherited = symbolic_scale_component(&input.hessian[index], &first);
            let curvature =
                symbolic_multiply_component(&input.gradient[axis], &input.gradient[other])
                    .map(|component| symbolic_multiply(&second, &component));
            hessian[index] = symbolic_add_component(&inherited, &curvature);
        }
    }
    SymbolicJet {
        value: format!("{stack}[{offset}]"),
        gradient,
        hessian,
    }
}

fn symbolic_expression(
    expression: &ProgramExpr,
    owner: &str,
    leaves: &[Leaf],
    constants: &HashSet<String>,
    bindings: &HashMap<String, SymbolicJet>,
    target: SymbolicTarget,
    dimension: usize,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<SymbolicJet> {
    let mut child = |expression: &ProgramExpr| {
        symbolic_expression(
            expression,
            owner,
            leaves,
            constants,
            bindings,
            target,
            dimension,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
            syn::Error::new_spanned(ident, "symbolic row_program binding is not defined")
        }),
        ProgramExpr::Zero => Ok(SymbolicJet::zero(dimension)),
        ProgramExpr::Neg(value) => {
            let value = child(value)?;
            Ok(SymbolicJet {
                value: symbolic_negate(&value.value),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| component.as_ref().map(|value| symbolic_negate(value)))
                    .collect(),
            })
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = child(value)?;
            let scalar = symbolic_scalar(scalar, constants, target)?;
            Ok(SymbolicJet {
                value: symbolic_multiply(&value.value, &scalar),
                gradient: value
                    .gradient
                    .iter()
                    .map(|component| symbolic_scale_component(component, &scalar))
                    .collect(),
                hessian: value
                    .hessian
                    .iter()
                    .map(|component| symbolic_scale_component(component, &scalar))
                    .collect(),
            })
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.value = symbolic_add(&value.value, &symbolic_scalar(scalar, constants, target)?);
            Ok(value)
        }
        ProgramExpr::Add(left, right) => Ok(symbolic_add_jets(child(left)?, child(right)?)),
        ProgramExpr::Mul(left, right) => Ok(symbolic_multiply_jets(child(left)?, child(right)?)),
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "symbolic compose input is not defined")
            })?;
            let suffix = *stack_index;
            *stack_index += 1;
            let stack = format!("{owner}_stack{suffix}");
            let mut leaf_arguments = vec![input.value.clone()];
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, target)?);
            }
            match target {
                SymbolicTarget::Rust => {
                    let rust_leaf = &leaves[*leaf].rust;
                    let rust_leaf = quote!(#rust_leaf).to_string();
                    preludes.push(format!(
                        "let {stack} = {rust_leaf}({});",
                        leaf_arguments.join(", ")
                    ));
                }
                SymbolicTarget::Cuda => {
                    let cuda_leaf = &leaves[*leaf].cuda;
                    leaf_arguments.push(stack.clone());
                    preludes.push(format!(
                        "double {stack}[3];\n{cuda_leaf}({});",
                        leaf_arguments.join(", ")
                    ));
                }
            }

            let first = format!("{stack}[1]");
            let second = format!("{stack}[2]");
            let mut gradient = vec![None; dimension];
            let mut hessian = vec![None; dimension * dimension];
            for axis in 0..dimension {
                gradient[axis] = symbolic_scale_component(&input.gradient[axis], &first);
                for other in axis..dimension {
                    let index = axis * dimension + other;
                    let inherited = symbolic_scale_component(&input.hessian[index], &first);
                    let curvature =
                        symbolic_multiply_component(&input.gradient[axis], &input.gradient[other])
                            .map(|component| symbolic_multiply(&second, &component));
                    hessian[index] = symbolic_add_component(&inherited, &curvature);
                }
            }
            Ok(SymbolicJet {
                value: format!("{stack}[0]"),
                gradient,
                hessian,
            })
        }
    }
}

#[derive(Clone)]
struct DirectionalJet {
    base: SymbolicJet,
    u: SymbolicJet,
    v: SymbolicJet,
    uv: SymbolicJet,
}

#[derive(Clone)]
struct DirectionalSupport {
    base: SymbolicSupport,
    u: SymbolicSupport,
    v: SymbolicSupport,
    uv: SymbolicSupport,
}

impl DirectionalSupport {
    fn empty(dimension: usize) -> Self {
        Self {
            base: SymbolicSupport::empty(dimension),
            u: SymbolicSupport::empty(dimension),
            v: SymbolicSupport::empty(dimension),
            uv: SymbolicSupport::empty(dimension),
        }
    }

    fn include(&mut self, jet: &DirectionalJet) {
        self.base.include(&jet.base);
        self.u.include(&jet.u);
        self.v.include(&jet.v);
        self.uv.include(&jet.uv);
    }
}

impl DirectionalJet {
    fn zero(dimension: usize) -> Self {
        Self {
            base: SymbolicJet::zero(dimension),
            u: SymbolicJet::zero(dimension),
            v: SymbolicJet::zero(dimension),
            uv: SymbolicJet::zero(dimension),
        }
    }

    fn primary(name: &str, axis: usize, dimension: usize, fourth: bool) -> Self {
        Self {
            base: SymbolicJet::primary(name, axis, dimension),
            u: SymbolicJet::constant(format!("direction_u[{axis}]"), dimension),
            v: if fourth {
                SymbolicJet::constant(format!("direction_v[{axis}]"), dimension)
            } else {
                SymbolicJet::zero(dimension)
            },
            uv: SymbolicJet::zero(dimension),
        }
    }

    fn support(&self) -> DirectionalSupport {
        let mut support = DirectionalSupport::empty(self.base.gradient.len());
        support.include(self);
        support
    }
}

fn directional_add(left: DirectionalJet, right: DirectionalJet) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_add_jets(left.base, right.base),
        u: symbolic_add_jets(left.u, right.u),
        v: symbolic_add_jets(left.v, right.v),
        uv: symbolic_add_jets(left.uv, right.uv),
    }
}

fn directional_negate(value: DirectionalJet) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_negate_jet(value.base),
        u: symbolic_negate_jet(value.u),
        v: symbolic_negate_jet(value.v),
        uv: symbolic_negate_jet(value.uv),
    }
}

fn directional_scale(value: DirectionalJet, scalar: &str) -> DirectionalJet {
    DirectionalJet {
        base: symbolic_scale_jet(value.base, scalar),
        u: symbolic_scale_jet(value.u, scalar),
        v: symbolic_scale_jet(value.v, scalar),
        uv: symbolic_scale_jet(value.uv, scalar),
    }
}

fn directional_multiply(
    left: DirectionalJet,
    right: DirectionalJet,
    fourth: bool,
) -> DirectionalJet {
    let base = symbolic_multiply_jets(left.base.clone(), right.base.clone());
    let u = symbolic_add_jets(
        symbolic_multiply_jets(left.u.clone(), right.base.clone()),
        symbolic_multiply_jets(left.base.clone(), right.u.clone()),
    );
    if !fourth {
        return DirectionalJet {
            base,
            u,
            v: SymbolicJet::zero(left.base.gradient.len()),
            uv: SymbolicJet::zero(left.base.gradient.len()),
        };
    }
    let v = symbolic_add_jets(
        symbolic_multiply_jets(left.v.clone(), right.base.clone()),
        symbolic_multiply_jets(left.base.clone(), right.v.clone()),
    );
    let uv = symbolic_add_jets(
        symbolic_add_jets(
            symbolic_multiply_jets(left.uv, right.base.clone()),
            symbolic_multiply_jets(left.u, right.v),
        ),
        symbolic_add_jets(
            symbolic_multiply_jets(left.v, right.u),
            symbolic_multiply_jets(left.base, right.uv),
        ),
    );
    DirectionalJet { base, u, v, uv }
}

fn materialize_directional(
    value: DirectionalJet,
    owner: &str,
    fourth: bool,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> DirectionalJet {
    let name = format!("{owner}_directional_tmp{}", *temporary_index);
    *temporary_index += 1;
    let support = value.support();
    let mut source = String::new();
    push_directional_declaration(&mut source, "", &name, "", &value, &support, fourth);
    preludes.push(source);
    directional_reference(&name, &support, value.base.gradient.len(), fourth)
}

struct DirectionalExpressionEnvironment<'a> {
    leaves: &'a [Leaf],
    constants: &'a HashSet<String>,
    dimension: usize,
    fourth: bool,
}

/// Exact normalized multivariate Taylor coefficients through degree four.
///
/// The directional lowering is asymptotically right for wide rows because it
/// never materializes a dense high-order tensor. For one- and two-primary row
/// programs, however, propagating four second-order directional jets performs
/// more arithmetic than propagating the complete tiny Taylor polynomial once.
/// This representation is a compile-time algebra only: emitted production code
/// contains direct scalar formulas, not an automatic-differentiation runtime.
#[derive(Clone)]
struct DenseTaylorJet {
    dimension: usize,
    order: usize,
    coefficients: Vec<Option<String>>,
}

fn dense_taylor_slot_count(dimension: usize) -> usize {
    5usize.pow(dimension as u32)
}

fn dense_taylor_counts(mut index: usize, dimension: usize) -> Vec<usize> {
    let mut counts = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        counts.push(index % 5);
        index /= 5;
    }
    counts
}

fn dense_taylor_index(counts: &[usize]) -> usize {
    counts
        .iter()
        .rev()
        .fold(0usize, |index, count| index * 5 + count)
}

fn dense_taylor_component(value: String, index: usize) -> Option<String> {
    if index != 0 && symbolic_is_zero(&value) {
        None
    } else {
        Some(value)
    }
}

impl DenseTaylorJet {
    fn zero(dimension: usize, order: usize) -> Self {
        let mut coefficients = vec![None; dense_taylor_slot_count(dimension)];
        coefficients[0] = Some("0.0".to_string());
        Self {
            dimension,
            order,
            coefficients,
        }
    }

    fn constant(value: String, dimension: usize, order: usize) -> Self {
        let mut out = Self::zero(dimension, order);
        out.coefficients[0] = Some(value);
        out
    }

    fn primary(name: &str, axis: usize, dimension: usize, order: usize) -> Self {
        let mut out = Self::constant(name.to_string(), dimension, order);
        let mut counts = vec![0usize; dimension];
        counts[axis] = 1;
        out.coefficients[dense_taylor_index(&counts)] = Some("1.0".to_string());
        out
    }

    fn support(&self) -> Vec<bool> {
        self.coefficients.iter().map(Option::is_some).collect()
    }

    fn reference(name: &str, support: &[bool], dimension: usize, order: usize) -> Self {
        let mut out = Self::zero(dimension, order);
        for (index, present) in support.iter().copied().enumerate() {
            if present {
                out.coefficients[index] = Some(format!("{name}_c{index}"));
            }
        }
        out
    }
}

fn dense_taylor_add(left: DenseTaylorJet, right: DenseTaylorJet) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(left.dimension, left.order);
    for index in 0..out.coefficients.len() {
        out.coefficients[index] =
            symbolic_add_component(&left.coefficients[index], &right.coefficients[index])
                .and_then(|value| dense_taylor_component(value, index));
    }
    out
}

fn dense_taylor_negate(value: DenseTaylorJet) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(value.dimension, value.order);
    for (index, component) in value.coefficients.iter().enumerate() {
        out.coefficients[index] = component
            .as_ref()
            .map(|component| symbolic_negate(component))
            .and_then(|component| dense_taylor_component(component, index));
    }
    out
}

fn dense_taylor_scale(value: DenseTaylorJet, scalar: &str) -> DenseTaylorJet {
    let mut out = DenseTaylorJet::zero(value.dimension, value.order);
    for (index, component) in value.coefficients.iter().enumerate() {
        out.coefficients[index] = symbolic_scale_component(component, scalar)
            .and_then(|component| dense_taylor_component(component, index));
    }
    out
}

fn dense_taylor_multiply(left: DenseTaylorJet, right: DenseTaylorJet) -> DenseTaylorJet {
    let dimension = left.dimension;
    let order = left.order;
    let mut out = DenseTaylorJet::zero(dimension, order);
    out.coefficients.fill(None);
    for (left_index, left_component) in left.coefficients.iter().enumerate() {
        let Some(left_component) = left_component else {
            continue;
        };
        let left_counts = dense_taylor_counts(left_index, dimension);
        for (right_index, right_component) in right.coefficients.iter().enumerate() {
            let Some(right_component) = right_component else {
                continue;
            };
            let right_counts = dense_taylor_counts(right_index, dimension);
            let counts = left_counts
                .iter()
                .zip(right_counts)
                .map(|(left, right)| left + right)
                .collect::<Vec<_>>();
            if counts.iter().sum::<usize>() > order {
                continue;
            }
            let index = dense_taylor_index(&counts);
            let product = symbolic_multiply(left_component, right_component);
            out.coefficients[index] =
                symbolic_add_component(&out.coefficients[index], &Some(product))
                    .and_then(|value| dense_taylor_component(value, index));
        }
    }
    if out.coefficients[0].is_none() {
        out.coefficients[0] = Some("0.0".to_string());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn dense_taylor_composition_partitions(
    candidates: &[(usize, String, Vec<usize>)],
    start: usize,
    remaining: usize,
    order: usize,
    counts: &mut [usize],
    selected: &mut Vec<usize>,
    product: Option<String>,
    derivative: &str,
    output: &mut DenseTaylorJet,
) {
    if remaining == 0 {
        let index = dense_taylor_index(counts);
        let mut multiplicity_factorial = 1usize;
        let mut run = 1usize;
        for pair in selected.windows(2) {
            if pair[0] == pair[1] {
                run += 1;
            } else {
                multiplicity_factorial *= match run {
                    1 => 1,
                    2 => 2,
                    3 => 6,
                    4 => 24,
                    _ => unreachable!("dense Taylor order is at most four"),
                };
                run = 1;
            }
        }
        multiplicity_factorial *= match run {
            1 => 1,
            2 => 2,
            3 => 6,
            4 => 24,
            _ => unreachable!("dense Taylor order is at most four"),
        };
        let mut term = symbolic_multiply(
            derivative,
            product
                .as_deref()
                .expect("composition partition has at least one factor"),
        );
        if multiplicity_factorial != 1 {
            term = symbolic_multiply(
                &term,
                &format!("{:.17}", 1.0 / multiplicity_factorial as f64),
            );
        }
        output.coefficients[index] =
            symbolic_add_component(&output.coefficients[index], &Some(term))
                .and_then(|value| dense_taylor_component(value, index));
        return;
    }

    for candidate_index in start..candidates.len() {
        let (_, component, candidate_counts) = &candidates[candidate_index];
        if counts
            .iter()
            .zip(candidate_counts)
            .map(|(left, right)| left + right)
            .sum::<usize>()
            > order
        {
            continue;
        }
        for (count, added) in counts.iter_mut().zip(candidate_counts) {
            *count += added;
        }
        selected.push(candidate_index);
        dense_taylor_composition_partitions(
            candidates,
            candidate_index,
            remaining - 1,
            order,
            counts,
            selected,
            Some(match &product {
                Some(product) => symbolic_multiply(product, component),
                None => component.clone(),
            }),
            derivative,
            output,
        );
        selected.pop();
        for (count, added) in counts.iter_mut().zip(candidate_counts) {
            *count -= added;
        }
    }
}

fn dense_taylor_compose(input: DenseTaylorJet, stack: &str) -> DenseTaylorJet {
    let dimension = input.dimension;
    let order = input.order;
    let candidates = input
        .coefficients
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(index, component)| {
            component.as_ref().map(|component| {
                (
                    index,
                    component.clone(),
                    dense_taylor_counts(index, dimension),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut out = DenseTaylorJet::constant(format!("{stack}[0]"), dimension, order);
    for derivative_order in 1..=order {
        dense_taylor_composition_partitions(
            &candidates,
            0,
            derivative_order,
            order,
            &mut vec![0usize; dimension],
            &mut Vec::with_capacity(derivative_order),
            None,
            &format!("{stack}[{derivative_order}]"),
            &mut out,
        );
    }
    out
}

fn push_dense_taylor_declaration(
    source: &mut String,
    indentation: &str,
    name: &str,
    mutable: &str,
    value: &DenseTaylorJet,
    support: &[bool],
) {
    for (index, present) in support.iter().copied().enumerate() {
        if present {
            source.push_str(&format!(
                "{indentation}let {mutable}{name}_c{index}: f64 = {};\n",
                symbolic_component(&value.coefficients[index]),
            ));
        }
    }
}

fn push_dense_taylor_assignment(
    source: &mut String,
    indentation: &str,
    name: &str,
    value: &DenseTaylorJet,
    support: &[bool],
) {
    for (index, present) in support.iter().copied().enumerate() {
        if present {
            source.push_str(&format!(
                "{indentation}{name}_c{index} = {};\n",
                symbolic_component(&value.coefficients[index]),
            ));
        }
    }
}

fn materialize_dense_taylor(
    value: DenseTaylorJet,
    owner: &str,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> DenseTaylorJet {
    let name = format!("__row_program_{owner}_dense_tmp{}", *temporary_index);
    *temporary_index += 1;
    let support = value.support();
    let mut source = String::new();
    push_dense_taylor_declaration(&mut source, "", &name, "", &value, &support);
    preludes.push(source);
    DenseTaylorJet::reference(&name, &support, value.dimension, value.order)
}

#[allow(clippy::too_many_arguments)]
fn dense_taylor_expression(
    expression: &ProgramExpr,
    owner: &str,
    leaves: &[Leaf],
    constants: &HashSet<String>,
    bindings: &HashMap<String, DenseTaylorJet>,
    dimension: usize,
    order: usize,
    temporary_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<DenseTaylorJet> {
    let mut child = |expression: &ProgramExpr| {
        dense_taylor_expression(
            expression,
            owner,
            leaves,
            constants,
            bindings,
            dimension,
            order,
            temporary_index,
            preludes,
        )
    };
    let value = match expression {
        ProgramExpr::Path(ident) => {
            return bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(ident, "dense row_program binding is not defined")
            });
        }
        ProgramExpr::Zero => DenseTaylorJet::zero(dimension, order),
        ProgramExpr::Neg(value) => dense_taylor_negate(child(value)?),
        ProgramExpr::Scale(value, scalar) => dense_taylor_scale(
            child(value)?,
            &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
        ),
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.coefficients[0] = Some(symbolic_add(
                symbolic_component(&value.coefficients[0]),
                &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
            ));
            value
        }
        ProgramExpr::Add(left, right) => dense_taylor_add(child(left)?, child(right)?),
        ProgramExpr::Mul(left, right) => dense_taylor_multiply(child(left)?, child(right)?),
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "dense compose input is not defined")
            })?;
            let suffix = *temporary_index;
            *temporary_index += 1;
            let stack = format!("__row_program_{owner}_dense_stack{suffix}");
            let mut leaf_arguments = vec![symbolic_component(&input.coefficients[0]).to_string()];
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let rust_leaf = &leaves[*leaf].rust;
            let rust_leaf = quote!(#rust_leaf).to_string();
            preludes.push(format!(
                "let {stack} = {rust_leaf}({});",
                leaf_arguments.join(", ")
            ));
            dense_taylor_compose(input, &stack)
        }
    };
    Ok(materialize_dense_taylor(
        value,
        owner,
        temporary_index,
        preludes,
    ))
}

fn directional_expression(
    expression: &ProgramExpr,
    owner: &str,
    environment: &DirectionalExpressionEnvironment<'_>,
    bindings: &HashMap<String, DirectionalJet>,
    stack_index: &mut usize,
    preludes: &mut Vec<String>,
) -> Result<DirectionalJet> {
    let leaves = environment.leaves;
    let constants = environment.constants;
    let dimension = environment.dimension;
    let fourth = environment.fourth;
    let mut child = |expression: &ProgramExpr| {
        directional_expression(
            expression,
            owner,
            environment,
            bindings,
            stack_index,
            preludes,
        )
    };
    match expression {
        ProgramExpr::Path(ident) => bindings.get(&ident.to_string()).cloned().ok_or_else(|| {
            syn::Error::new_spanned(ident, "directional row_program binding is not defined")
        }),
        ProgramExpr::Zero => Ok(DirectionalJet::zero(dimension)),
        ProgramExpr::Neg(value) => {
            let value = directional_negate(child(value)?);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Scale(value, scalar) => {
            let value = child(value)?;
            let scalar = symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?;
            let value = directional_scale(value, &scalar);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::AddConstant(value, scalar) => {
            let mut value = child(value)?;
            value.base.value = symbolic_add(
                &value.base.value,
                &symbolic_scalar(scalar, constants, SymbolicTarget::Rust)?,
            );
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Add(left, right) => {
            let left = child(left)?;
            let right = child(right)?;
            let value = directional_add(left, right);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Mul(left, right) => {
            let left = child(left)?;
            let right = child(right)?;
            let value = directional_multiply(left, right, fourth);
            Ok(materialize_directional(
                value,
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
        ProgramExpr::Compose {
            leaf,
            value,
            arguments,
        } => {
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "directional compose input is not defined")
            })?;
            let suffix = *stack_index;
            *stack_index += 1;
            let stack = format!("{owner}_directional_stack{suffix}");
            let mut leaf_arguments = vec![input.base.value.clone()];
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let rust_leaf = &leaves[*leaf].rust;
            let rust_leaf = quote!(#rust_leaf).to_string();
            preludes.push(format!(
                "let {stack} = {rust_leaf}({});",
                leaf_arguments.join(", ")
            ));

            let base = symbolic_compose_jet(input.base.clone(), &stack, 0);
            let first = symbolic_compose_jet(input.base.clone(), &stack, 1);
            let u = symbolic_multiply_jets(first.clone(), input.u.clone());
            if !fourth {
                let value = DirectionalJet {
                    base,
                    u,
                    v: SymbolicJet::zero(dimension),
                    uv: SymbolicJet::zero(dimension),
                };
                return Ok(materialize_directional(
                    value,
                    owner,
                    fourth,
                    stack_index,
                    preludes,
                ));
            }
            let v = symbolic_multiply_jets(first.clone(), input.v.clone());
            let second = symbolic_compose_jet(input.base, &stack, 2);
            let uv = symbolic_add_jets(
                symbolic_multiply_jets(symbolic_multiply_jets(second, input.u), input.v),
                symbolic_multiply_jets(first, input.uv),
            );
            Ok(materialize_directional(
                DirectionalJet { base, u, v, uv },
                owner,
                fourth,
                stack_index,
                preludes,
            ))
        }
    }
}

struct SymbolicLocal {
    name: String,
    mutable: bool,
    value: SymbolicJet,
    preludes: Vec<String>,
}

struct SymbolicAssignment {
    target: String,
    value: SymbolicJet,
    preludes: Vec<String>,
}

enum SymbolicStatement {
    Local(SymbolicLocal),
    If {
        condition: String,
        assignments: Vec<SymbolicAssignment>,
    },
}

struct SymbolicSchedule {
    statements: Vec<SymbolicStatement>,
    result: SymbolicJet,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, SymbolicSupport>,
    assigned: HashSet<String>,
    witness_values: Vec<String>,
}

struct DenseTaylorLocal {
    name: String,
    mutable: bool,
    value: DenseTaylorJet,
    preludes: Vec<String>,
}

struct DenseTaylorAssignment {
    target: String,
    value: DenseTaylorJet,
    preludes: Vec<String>,
}

enum DenseTaylorStatement {
    Local(DenseTaylorLocal),
    If {
        condition: String,
        assignments: Vec<DenseTaylorAssignment>,
    },
}

struct DenseTaylorSchedule {
    statements: Vec<DenseTaylorStatement>,
    result: DenseTaylorJet,
    root_compose_stack: Option<String>,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, Vec<bool>>,
    assigned: HashSet<String>,
}

fn include_dense_taylor_support(support: &mut [bool], value: &DenseTaylorJet) {
    for (present, component) in support.iter_mut().zip(&value.coefficients) {
        *present |= component.is_some();
    }
}

fn dense_taylor_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    order: usize,
) -> Result<DenseTaylorSchedule> {
    let dimension = primaries.len();
    let mut bindings = HashMap::<String, DenseTaylorJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            DenseTaylorJet::primary(&primary.to_string(), axis, dimension, order),
        );
    }
    let mut mutable_support = HashMap::<String, Vec<bool>>::new();
    let mut assigned = HashSet::new();
    let mut dense_statements = Vec::new();
    let mut temporary_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let value = dense_taylor_expression(
                    value,
                    &name.to_string(),
                    leaves,
                    constants,
                    &bindings,
                    dimension,
                    order,
                    &mut temporary_index,
                    &mut preludes,
                )?;
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    DenseTaylorJet::reference(&name.to_string(), &support, dimension, order),
                );
                dense_statements.push(DenseTaylorStatement::Local(DenseTaylorLocal {
                    name: name.to_string(),
                    mutable: *mutable,
                    value,
                    preludes,
                }));
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut dense_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let value = dense_taylor_expression(
                        value,
                        &target_name.to_string(),
                        leaves,
                        constants,
                        &bindings,
                        dimension,
                        order,
                        &mut temporary_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable dense Taylor target");
                    include_dense_taylor_support(support, &value);
                    bindings.insert(
                        target_name.to_string(),
                        DenseTaylorJet::reference(
                            &target_name.to_string(),
                            support,
                            dimension,
                            order,
                        ),
                    );
                    dense_assignments.push(DenseTaylorAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                dense_statements.push(DenseTaylorStatement::If {
                    condition: symbolic_scalar(condition, constants, SymbolicTarget::Rust)?,
                    assignments: dense_assignments,
                });
            }
        }
    }
    let mut result_preludes = Vec::new();
    let (result, root_compose_stack) =
        if order == 3 && matches!(result, ProgramExpr::Compose { .. }) {
            let ProgramExpr::Compose {
                leaf,
                value,
                arguments,
            } = result
            else {
                unreachable!("matched dense Taylor root compose")
            };
            let input = bindings.get(&value.to_string()).cloned().ok_or_else(|| {
                syn::Error::new_spanned(value, "dense root compose input is not defined")
            })?;
            let stack = "__row_program_result_dense_root_stack".to_string();
            let mut leaf_arguments = vec![symbolic_component(&input.coefficients[0]).to_string()];
            for argument in arguments {
                leaf_arguments.push(symbolic_scalar(argument, constants, SymbolicTarget::Rust)?);
            }
            let rust_leaf = &leaves[*leaf].rust;
            let rust_leaf = quote!(#rust_leaf).to_string();
            result_preludes.push(format!(
                "let {stack} = {rust_leaf}({});",
                leaf_arguments.join(", ")
            ));
            (input, Some(stack))
        } else {
            (
                dense_taylor_expression(
                    result,
                    "result",
                    leaves,
                    constants,
                    &bindings,
                    dimension,
                    order,
                    &mut temporary_index,
                    &mut result_preludes,
                )?,
                None,
            )
        };
    Ok(DenseTaylorSchedule {
        statements: dense_statements,
        result,
        root_compose_stack,
        result_preludes,
        mutable_support,
        assigned,
    })
}

fn push_preludes(source: &mut String, preludes: &[String], indentation: &str) {
    for prelude in preludes {
        for line in prelude.lines() {
            source.push_str(indentation);
            source.push_str(line);
            source.push('\n');
        }
    }
}

fn symbolic_component(component: &Option<String>) -> &str {
    component.as_deref().unwrap_or("0.0")
}

fn symbolic_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    witnesses: &[Ident],
    target: SymbolicTarget,
) -> Result<SymbolicSchedule> {
    let dimension = primaries.len();
    let mut bindings = HashMap::<String, SymbolicJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            SymbolicJet::primary(&primary.to_string(), axis, dimension),
        );
    }
    let mut mutable_support = HashMap::<String, SymbolicSupport>::new();
    let mut assigned = HashSet::new();
    let mut symbolic_statements = Vec::new();
    // One source-wide namespace makes temporary declarations collision-free,
    // including repeated assignments to the same mutable local in one scope.
    let mut stack_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let value = symbolic_expression(
                    value,
                    &name.to_string(),
                    leaves,
                    constants,
                    &bindings,
                    target,
                    dimension,
                    &mut stack_index,
                    &mut preludes,
                )?;
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    SymbolicJet::reference(&name.to_string(), &support, dimension),
                );
                symbolic_statements.push(SymbolicStatement::Local(SymbolicLocal {
                    name: name.to_string(),
                    mutable: *mutable,
                    value,
                    preludes,
                }));
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut symbolic_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let value = symbolic_expression(
                        value,
                        &target_name.to_string(),
                        leaves,
                        constants,
                        &bindings,
                        target,
                        dimension,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable symbolic target");
                    support.include(&value);
                    bindings.insert(
                        target_name.to_string(),
                        SymbolicJet::reference(&target_name.to_string(), support, dimension),
                    );
                    symbolic_assignments.push(SymbolicAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                symbolic_statements.push(SymbolicStatement::If {
                    condition: symbolic_scalar(condition, constants, target)?,
                    assignments: symbolic_assignments,
                });
            }
        }
    }
    let witness_values = witnesses
        .iter()
        .map(|witness| {
            bindings
                .get(&witness.to_string())
                .map(|jet| jet.value.clone())
                .ok_or_else(|| {
                    syn::Error::new_spanned(witness, "symbolic witness binding is not defined")
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut result_preludes = Vec::new();
    let result = symbolic_expression(
        result,
        "result",
        leaves,
        constants,
        &bindings,
        target,
        dimension,
        &mut stack_index,
        &mut result_preludes,
    )?;
    Ok(SymbolicSchedule {
        statements: symbolic_statements,
        result,
        result_preludes,
        mutable_support,
        assigned,
        witness_values,
    })
}

struct DirectionalLocal {
    name: String,
    mutable: bool,
    value: DirectionalJet,
    preludes: Vec<String>,
}

struct DirectionalAssignment {
    target: String,
    value: DirectionalJet,
    preludes: Vec<String>,
}

enum DirectionalStatement {
    Local(DirectionalLocal),
    If {
        condition: String,
        assignments: Vec<DirectionalAssignment>,
    },
}

struct DirectionalSchedule {
    statements: Vec<DirectionalStatement>,
    result: DirectionalJet,
    result_preludes: Vec<String>,
    mutable_support: HashMap<String, DirectionalSupport>,
    assigned: HashSet<String>,
}

fn directional_prefix(name: &str, component: &str) -> String {
    format!("__row_program_{name}_{component}")
}

fn directional_reference(
    name: &str,
    support: &DirectionalSupport,
    dimension: usize,
    fourth: bool,
) -> DirectionalJet {
    DirectionalJet {
        base: SymbolicJet::reference(&directional_prefix(name, "base"), &support.base, dimension),
        u: SymbolicJet::reference(&directional_prefix(name, "u"), &support.u, dimension),
        v: if fourth {
            SymbolicJet::reference(&directional_prefix(name, "vdir"), &support.v, dimension)
        } else {
            SymbolicJet::zero(dimension)
        },
        uv: if fourth {
            SymbolicJet::reference(&directional_prefix(name, "uv"), &support.uv, dimension)
        } else {
            SymbolicJet::zero(dimension)
        },
    }
}

fn directional_schedule(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<DirectionalSchedule> {
    let dimension = primaries.len();
    let expression_environment = DirectionalExpressionEnvironment {
        leaves,
        constants,
        dimension,
        fourth,
    };
    let mut bindings = HashMap::<String, DirectionalJet>::new();
    for (axis, primary) in primaries.iter().enumerate() {
        bindings.insert(
            primary.to_string(),
            DirectionalJet::primary(&primary.to_string(), axis, dimension, fourth),
        );
    }
    let mut mutable_support = HashMap::<String, DirectionalSupport>::new();
    let mut assigned = HashSet::new();
    let mut directional_statements = Vec::new();
    let mut stack_index = 0;
    for statement in statements {
        match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let mut preludes = Vec::new();
                let value = directional_expression(
                    value,
                    &name.to_string(),
                    &expression_environment,
                    &bindings,
                    &mut stack_index,
                    &mut preludes,
                )?;
                let support = value.support();
                if *mutable {
                    mutable_support.insert(name.to_string(), support.clone());
                }
                bindings.insert(
                    name.to_string(),
                    directional_reference(&name.to_string(), &support, dimension, fourth),
                );
                directional_statements.push(DirectionalStatement::Local(DirectionalLocal {
                    name: name.to_string(),
                    mutable: *mutable,
                    value,
                    preludes,
                }));
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let mut directional_assignments = Vec::new();
                for (target_name, value) in assignments {
                    assigned.insert(target_name.to_string());
                    let mut preludes = Vec::new();
                    let value = directional_expression(
                        value,
                        &target_name.to_string(),
                        &expression_environment,
                        &bindings,
                        &mut stack_index,
                        &mut preludes,
                    )?;
                    let support = mutable_support
                        .get_mut(&target_name.to_string())
                        .expect("validated mutable directional target");
                    support.include(&value);
                    bindings.insert(
                        target_name.to_string(),
                        directional_reference(&target_name.to_string(), support, dimension, fourth),
                    );
                    directional_assignments.push(DirectionalAssignment {
                        target: target_name.to_string(),
                        value,
                        preludes,
                    });
                }
                directional_statements.push(DirectionalStatement::If {
                    condition: symbolic_scalar(condition, constants, SymbolicTarget::Rust)?,
                    assignments: directional_assignments,
                });
            }
        }
    }
    let mut result_preludes = Vec::new();
    let result = directional_expression(
        result,
        "result",
        &expression_environment,
        &bindings,
        &mut stack_index,
        &mut result_preludes,
    )?;
    Ok(DirectionalSchedule {
        statements: directional_statements,
        result,
        result_preludes,
        mutable_support,
        assigned,
    })
}

fn push_symbolic_declaration(
    source: &mut String,
    indentation: &str,
    prefix: &str,
    mutable: &str,
    value: &SymbolicJet,
    support: &SymbolicSupport,
) {
    let dimension = value.gradient.len();
    source.push_str(&format!(
        "{indentation}let {mutable}{prefix}_v: f64 = {};\n",
        value.value
    ));
    for axis in 0..dimension {
        if support.gradient[axis] {
            source.push_str(&format!(
                "{indentation}let {mutable}{prefix}_g{axis}: f64 = {};\n",
                symbolic_component(&value.gradient[axis]),
            ));
        }
        for other in axis..dimension {
            let index = axis * dimension + other;
            if support.hessian[index] {
                source.push_str(&format!(
                    "{indentation}let {mutable}{prefix}_h{axis}_{other}: f64 = {};\n",
                    symbolic_component(&value.hessian[index]),
                ));
            }
        }
    }
}

fn push_symbolic_assignment(
    source: &mut String,
    indentation: &str,
    prefix: &str,
    value: &SymbolicJet,
    support: &SymbolicSupport,
) {
    let dimension = value.gradient.len();
    source.push_str(&format!("{indentation}{prefix}_v = {};\n", value.value));
    for axis in 0..dimension {
        if support.gradient[axis] {
            source.push_str(&format!(
                "{indentation}{prefix}_g{axis} = {};\n",
                symbolic_component(&value.gradient[axis]),
            ));
        }
        for other in axis..dimension {
            let index = axis * dimension + other;
            if support.hessian[index] {
                source.push_str(&format!(
                    "{indentation}{prefix}_h{axis}_{other} = {};\n",
                    symbolic_component(&value.hessian[index]),
                ));
            }
        }
    }
}

fn push_directional_declaration(
    source: &mut String,
    indentation: &str,
    name: &str,
    mutable: &str,
    value: &DirectionalJet,
    support: &DirectionalSupport,
    fourth: bool,
) {
    push_symbolic_declaration(
        source,
        indentation,
        &directional_prefix(name, "base"),
        mutable,
        &value.base,
        &support.base,
    );
    push_symbolic_declaration(
        source,
        indentation,
        &directional_prefix(name, "u"),
        mutable,
        &value.u,
        &support.u,
    );
    if fourth {
        push_symbolic_declaration(
            source,
            indentation,
            &directional_prefix(name, "vdir"),
            mutable,
            &value.v,
            &support.v,
        );
        push_symbolic_declaration(
            source,
            indentation,
            &directional_prefix(name, "uv"),
            mutable,
            &value.uv,
            &support.uv,
        );
    }
}

fn push_directional_assignment(
    source: &mut String,
    indentation: &str,
    name: &str,
    value: &DirectionalJet,
    support: &DirectionalSupport,
    fourth: bool,
) {
    push_symbolic_assignment(
        source,
        indentation,
        &directional_prefix(name, "base"),
        &value.base,
        &support.base,
    );
    push_symbolic_assignment(
        source,
        indentation,
        &directional_prefix(name, "u"),
        &value.u,
        &support.u,
    );
    if fourth {
        push_symbolic_assignment(
            source,
            indentation,
            &directional_prefix(name, "vdir"),
            &value.v,
            &support.v,
        );
        push_symbolic_assignment(
            source,
            indentation,
            &directional_prefix(name, "uv"),
            &value.uv,
            &support.uv,
        );
    }
}

fn dense_taylor_derivative(value: &DenseTaylorJet, axes: &[usize]) -> Option<String> {
    let mut counts = vec![0usize; value.dimension];
    for axis in axes {
        counts[*axis] += 1;
    }
    let component = value.coefficients[dense_taylor_index(&counts)]
        .as_ref()?
        .clone();
    let factorial = counts.iter().fold(1usize, |product, count| {
        product
            * match count {
                0 | 1 => 1,
                2 => 2,
                3 => 6,
                4 => 24,
                _ => unreachable!("dense Taylor order is at most four"),
            }
    });
    if factorial == 1 {
        Some(component)
    } else {
        Some(symbolic_multiply(&component, &format!("{factorial}.0")))
    }
}

fn dense_taylor_contracted_component(
    value: &DenseTaylorJet,
    axis: usize,
    other: usize,
    fourth: bool,
) -> String {
    let mut component = None;
    for direction_axis in 0..value.dimension {
        if fourth {
            for other_direction_axis in 0..value.dimension {
                let derivative = dense_taylor_derivative(
                    value,
                    &[axis, other, direction_axis, other_direction_axis],
                );
                let directed = derivative.map(|derivative| {
                    symbolic_multiply(
                        &symbolic_multiply(&derivative, &format!("direction_u[{direction_axis}]")),
                        &format!("direction_v[{other_direction_axis}]"),
                    )
                });
                component = symbolic_add_component(&component, &directed);
            }
        } else {
            let derivative = dense_taylor_derivative(value, &[axis, other, direction_axis]);
            let directed = derivative.map(|derivative| {
                symbolic_multiply(&derivative, &format!("direction_u[{direction_axis}]"))
            });
            component = symbolic_add_component(&component, &directed);
        }
    }
    symbolic_component(&component).to_string()
}

fn push_dense_taylor_derivative_array(
    source: &mut String,
    name: &str,
    value: &DenseTaylorJet,
    derivative_order: usize,
) {
    source.push_str(&format!("    let {name} = ["));
    for ones in 0..=derivative_order {
        if ones != 0 {
            source.push_str(", ");
        }
        let mut axes = vec![0usize; derivative_order - ones];
        axes.extend(std::iter::repeat_n(1usize, ones));
        source.push_str(symbolic_component(&dense_taylor_derivative(value, &axes)));
    }
    source.push_str("];\n");
}

fn rust_dense_taylor_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let order = if fourth { 4 } else { 3 };
    let schedule = dense_taylor_schedule(primaries, constants, leaves, statements, result, order)?;
    let mut source = "{\n".to_string();
    for statement in &schedule.statements {
        match statement {
            DenseTaylorStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable dense Taylor support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                push_dense_taylor_declaration(
                    &mut source,
                    "    ",
                    &local.name,
                    mutable,
                    &local.value,
                    &support,
                );
            }
            DenseTaylorStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable dense Taylor assignment support exists");
                    push_dense_taylor_assignment(
                        &mut source,
                        "        ",
                        &assignment.target,
                        &assignment.value,
                        support,
                    );
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    if dimension == 2 {
        if let Some(root_stack) = &schedule.root_compose_stack {
            push_dense_taylor_derivative_array(&mut source, "inner_first", &schedule.result, 1);
            push_dense_taylor_derivative_array(&mut source, "inner_second", &schedule.result, 2);
            push_dense_taylor_derivative_array(&mut source, "inner_third", &schedule.result, 3);
            if fourth {
                push_dense_taylor_derivative_array(
                    &mut source,
                    "inner_fourth",
                    &schedule.result,
                    4,
                );
                source.push_str(&format!(
                    "    let inner_u = inner_first[0] * direction_u[0]\n\
                     \x20       + inner_first[1] * direction_u[1];\n\
                     \x20   let inner_v = inner_first[0] * direction_v[0]\n\
                     \x20       + inner_first[1] * direction_v[1];\n\
                     \x20   let inner_uv = inner_second[0] * direction_u[0] * direction_v[0]\n\
                     \x20       + inner_second[1] * (direction_u[0] * direction_v[1]\n\
                     \x20           + direction_u[1] * direction_v[0])\n\
                     \x20       + inner_second[2] * direction_u[1] * direction_v[1];\n\
                     \x20   std::array::from_fn(|axis| std::array::from_fn(|other| {{\n\
                     \x20       let offset = axis + other;\n\
                     \x20       let inner_a = inner_first[axis];\n\
                     \x20       let inner_b = inner_first[other];\n\
                     \x20       let inner_ab = inner_second[offset];\n\
                     \x20       let inner_au = inner_second[axis] * direction_u[0]\n\
                     \x20           + inner_second[axis + 1] * direction_u[1];\n\
                     \x20       let inner_av = inner_second[axis] * direction_v[0]\n\
                     \x20           + inner_second[axis + 1] * direction_v[1];\n\
                     \x20       let inner_bu = inner_second[other] * direction_u[0]\n\
                     \x20           + inner_second[other + 1] * direction_u[1];\n\
                     \x20       let inner_bv = inner_second[other] * direction_v[0]\n\
                     \x20           + inner_second[other + 1] * direction_v[1];\n\
                     \x20       let inner_abu = inner_third[offset] * direction_u[0]\n\
                     \x20           + inner_third[offset + 1] * direction_u[1];\n\
                     \x20       let inner_abv = inner_third[offset] * direction_v[0]\n\
                     \x20           + inner_third[offset + 1] * direction_v[1];\n\
                     \x20       let inner_auv = inner_third[axis] * direction_u[0] * direction_v[0]\n\
                     \x20           + inner_third[axis + 1] * (direction_u[0] * direction_v[1]\n\
                     \x20               + direction_u[1] * direction_v[0])\n\
                     \x20           + inner_third[axis + 2] * direction_u[1] * direction_v[1];\n\
                     \x20       let inner_buv = inner_third[other] * direction_u[0] * direction_v[0]\n\
                     \x20           + inner_third[other + 1] * (direction_u[0] * direction_v[1]\n\
                     \x20               + direction_u[1] * direction_v[0])\n\
                     \x20           + inner_third[other + 2] * direction_u[1] * direction_v[1];\n\
                     \x20       let inner_abuv = inner_fourth[offset] * direction_u[0] * direction_v[0]\n\
                     \x20           + inner_fourth[offset + 1] * (direction_u[0] * direction_v[1]\n\
                     \x20               + direction_u[1] * direction_v[0])\n\
                     \x20           + inner_fourth[offset + 2] * direction_u[1] * direction_v[1];\n\
                     \x20       let second_chain = inner_au * inner_b + inner_a * inner_bu\n\
                     \x20           + inner_u * inner_ab;\n\
                     \x20       let second_chain_v = inner_auv * inner_b + inner_au * inner_bv\n\
                     \x20           + inner_av * inner_bu + inner_a * inner_buv\n\
                     \x20           + inner_uv * inner_ab + inner_u * inner_abv;\n\
                     \x20       {root_stack}[4] * inner_v * inner_u * inner_a * inner_b\n\
                     \x20           + {root_stack}[3] * (inner_uv * inner_a * inner_b\n\
                     \x20               + inner_u * inner_av * inner_b + inner_u * inner_a * inner_bv\n\
                     \x20               + inner_v * second_chain)\n\
                     \x20           + {root_stack}[2] * (second_chain_v + inner_v * inner_abu)\n\
                     \x20           + {root_stack}[1] * inner_abuv\n\
                     \x20   }}))\n"
                ));
            } else {
                source.push_str(&format!(
                    "    let inner_u = inner_first[0] * direction_u[0]\n\
                     \x20       + inner_first[1] * direction_u[1];\n\
                     \x20   std::array::from_fn(|axis| std::array::from_fn(|other| {{\n\
                     \x20       let offset = axis + other;\n\
                     \x20       let inner_a = inner_first[axis];\n\
                     \x20       let inner_b = inner_first[other];\n\
                     \x20       let inner_ab = inner_second[offset];\n\
                     \x20       let inner_au = inner_second[axis] * direction_u[0]\n\
                     \x20           + inner_second[axis + 1] * direction_u[1];\n\
                     \x20       let inner_bu = inner_second[other] * direction_u[0]\n\
                     \x20           + inner_second[other + 1] * direction_u[1];\n\
                     \x20       let inner_abu = inner_third[offset] * direction_u[0]\n\
                     \x20           + inner_third[offset + 1] * direction_u[1];\n\
                     \x20       {root_stack}[3] * inner_u * inner_a * inner_b\n\
                     \x20           + {root_stack}[2] * (inner_au * inner_b + inner_a * inner_bu\n\
                     \x20               + inner_u * inner_ab)\n\
                     \x20           + {root_stack}[1] * inner_abu\n\
                     \x20   }}))\n"
                ));
            }
            source.push_str("}\n");
            let order = if fourth { "fourth" } else { "third" };
            return syn::parse_str(&source).map_err(|error| {
                syn::Error::new(
                    error.span(),
                    format!(
                        "failed to parse generated Rust dense root-compose {order}-order row program: {error}\n{source}"
                    ),
                )
            });
        }
    }
    if dimension == 2 {
        source.push_str("    let dense_derivatives = [");
        for ones in 0..=order {
            if ones != 0 {
                source.push_str(", ");
            }
            let mut axes = vec![0usize; order - ones];
            axes.extend(std::iter::repeat_n(1usize, ones));
            source.push_str(symbolic_component(&dense_taylor_derivative(
                &schedule.result,
                &axes,
            )));
        }
        source.push_str("];\n");
        if fourth {
            source.push_str(
                "    let direction_00 = direction_u[0] * direction_v[0];\n\
                 \x20   let direction_01 = direction_u[0] * direction_v[1]\n\
                 \x20       + direction_u[1] * direction_v[0];\n\
                 \x20   let direction_11 = direction_u[1] * direction_v[1];\n\
                 \x20   std::array::from_fn(|axis| std::array::from_fn(|other| {\n\
                 \x20       let offset = axis + other;\n\
                 \x20       dense_derivatives[offset] * direction_00\n\
                 \x20           + dense_derivatives[offset + 1] * direction_01\n\
                 \x20           + dense_derivatives[offset + 2] * direction_11\n\
                 \x20   }))\n",
            );
        } else {
            source.push_str(
                "    std::array::from_fn(|axis| std::array::from_fn(|other| {\n\
                 \x20       let offset = axis + other;\n\
                 \x20       dense_derivatives[offset] * direction_u[0]\n\
                 \x20           + dense_derivatives[offset + 1] * direction_u[1]\n\
                 \x20   }))\n",
            );
        }
    } else {
        source.push_str("    [\n");
        for axis in 0..dimension {
            source.push_str("        [");
            for other in 0..dimension {
                if other != 0 {
                    source.push_str(", ");
                }
                source.push_str(&dense_taylor_contracted_component(
                    &schedule.result,
                    axis,
                    other,
                    fourth,
                ));
            }
            source.push_str("],\n");
        }
        source.push_str("    ]\n");
    }
    source.push_str("}\n");
    let order = if fourth { "fourth" } else { "third" };
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!(
                "failed to parse generated Rust dense {order}-order row program: {error}\n{source}"
            ),
        )
    })
}

fn rust_directional_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    fourth: bool,
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let schedule = directional_schedule(primaries, constants, leaves, statements, result, fourth)?;
    let mut source = "{\n".to_string();
    for statement in &schedule.statements {
        match statement {
            DirectionalStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable directional support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                push_directional_declaration(
                    &mut source,
                    "    ",
                    &local.name,
                    mutable,
                    &local.value,
                    &support,
                    fourth,
                );
            }
            DirectionalStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable directional assignment support exists");
                    push_directional_assignment(
                        &mut source,
                        "        ",
                        &assignment.target,
                        &assignment.value,
                        support,
                        fourth,
                    );
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    let contracted = if fourth {
        &schedule.result.uv
    } else {
        &schedule.result.u
    };
    source.push_str("    [\n");
    for axis in 0..dimension {
        source.push_str("        [");
        for other in 0..dimension {
            if other != 0 {
                source.push_str(", ");
            }
            let (row, column) = if axis <= other {
                (axis, other)
            } else {
                (other, axis)
            };
            let index = row * dimension + column;
            source.push_str(symbolic_component(&contracted.hessian[index]));
        }
        source.push_str("],\n");
    }
    source.push_str("    ]\n}\n");
    let order = if fourth { "fourth" } else { "third" };
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!(
                "failed to parse generated Rust {order}-order contracted row program: {error}\n{source}"
            ),
        )
    })
}

fn rust_order2_body(
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
    witnesses: &[Ident],
) -> Result<syn::Block> {
    let dimension = primaries.len();
    let schedule = symbolic_schedule(
        primaries,
        constants,
        leaves,
        statements,
        result,
        witnesses,
        SymbolicTarget::Rust,
    )?;
    let mut source = "{\n".to_string();
    for statement in &schedule.statements {
        match statement {
            SymbolicStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let mutable = if schedule.assigned.contains(&local.name) {
                    "mut "
                } else {
                    ""
                };
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable symbolic support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                source.push_str(&format!(
                    "    let {mutable}{}_v: f64 = {};\n",
                    local.name, local.value.value
                ));
                for axis in 0..dimension {
                    if support.gradient[axis] {
                        source.push_str(&format!(
                            "    let {mutable}{}_g{axis}: f64 = {};\n",
                            local.name,
                            symbolic_component(&local.value.gradient[axis]),
                        ));
                    }
                    for other in axis..dimension {
                        let index = axis * dimension + other;
                        if support.hessian[index] {
                            source.push_str(&format!(
                                "    let {mutable}{}_h{axis}_{other}: f64 = {};\n",
                                local.name,
                                symbolic_component(&local.value.hessian[index]),
                            ));
                        }
                    }
                }
            }
            SymbolicStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if {condition} {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable symbolic assignment support exists");
                    source.push_str(&format!(
                        "        {}_v = {};\n",
                        assignment.target, assignment.value.value,
                    ));
                    for axis in 0..dimension {
                        if support.gradient[axis] {
                            source.push_str(&format!(
                                "        {}_g{axis} = {};\n",
                                assignment.target,
                                symbolic_component(&assignment.value.gradient[axis]),
                            ));
                        }
                        for other in axis..dimension {
                            let index = axis * dimension + other;
                            if support.hessian[index] {
                                source.push_str(&format!(
                                    "        {}_h{axis}_{other} = {};\n",
                                    assignment.target,
                                    symbolic_component(&assignment.value.hessian[index]),
                                ));
                            }
                        }
                    }
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    source.push_str(&format!(
        "    let __row_program_value: f64 = {};\n",
        schedule.result.value
    ));
    for axis in 0..dimension {
        source.push_str(&format!(
            "    let __row_program_g{axis}: f64 = {};\n",
            symbolic_component(&schedule.result.gradient[axis]),
        ));
        for other in axis..dimension {
            let index = axis * dimension + other;
            source.push_str(&format!(
                "    let __row_program_h{axis}_{other}: f64 = {};\n",
                symbolic_component(&schedule.result.hessian[index]),
            ));
        }
    }
    source.push_str("    (\n        __row_program_value,\n        [");
    for axis in 0..dimension {
        if axis != 0 {
            source.push_str(", ");
        }
        source.push_str(&format!("__row_program_g{axis}"));
    }
    source.push_str("],\n        [\n");
    for axis in 0..dimension {
        source.push_str("            [");
        for other in 0..dimension {
            if other != 0 {
                source.push_str(", ");
            }
            let (row, column) = if axis <= other {
                (axis, other)
            } else {
                (other, axis)
            };
            source.push_str(&format!("__row_program_h{row}_{column}"));
        }
        source.push_str("],\n");
    }
    source.push_str("        ],\n        [");
    for (index, witness) in schedule.witness_values.iter().enumerate() {
        if index != 0 {
            source.push_str(", ");
        }
        source.push_str(witness);
    }
    source.push_str("]\n    )\n}\n");
    syn::parse_str(&source).map_err(|error| {
        syn::Error::new(
            error.span(),
            format!("failed to parse generated Rust order-2 row program: {error}\n{source}"),
        )
    })
}

fn cuda_source(
    name: &Ident,
    primaries: &[Ident],
    constants: &HashSet<String>,
    leaves: &[Leaf],
    statements: &[Statement],
    result: &ProgramExpr,
) -> Result<String> {
    let dimension = primaries.len();
    let parameters = primaries
        .iter()
        .map(|primary| format!("double {primary}"))
        .chain([
            "const RowIn& in".to_string(),
            "double* row_value".to_string(),
            "double* row_gradient".to_string(),
            "double* row_hessian".to_string(),
        ])
        .collect::<Vec<_>>()
        .join(", ");
    let schedule = symbolic_schedule(
        primaries,
        constants,
        leaves,
        statements,
        result,
        &[],
        SymbolicTarget::Cuda,
    )?;

    let mut source = format!("__device__ __forceinline__ void {name}(\n        {parameters}) {{\n");
    for statement in &schedule.statements {
        match statement {
            SymbolicStatement::Local(local) => {
                push_preludes(&mut source, &local.preludes, "    ");
                let support = if local.mutable {
                    schedule
                        .mutable_support
                        .get(&local.name)
                        .expect("mutable symbolic support exists")
                        .clone()
                } else {
                    local.value.support()
                };
                source.push_str(&format!(
                    "    double {}_v = {};\n",
                    local.name, local.value.value
                ));
                for axis in 0..dimension {
                    if support.gradient[axis] {
                        source.push_str(&format!(
                            "    double {}_g{axis} = {};\n",
                            local.name,
                            symbolic_component(&local.value.gradient[axis]),
                        ));
                    }
                    for other in axis..dimension {
                        let index = axis * dimension + other;
                        if support.hessian[index] {
                            source.push_str(&format!(
                                "    double {}_h{axis}_{other} = {};\n",
                                local.name,
                                symbolic_component(&local.value.hessian[index]),
                            ));
                        }
                    }
                }
            }
            SymbolicStatement::If {
                condition,
                assignments,
            } => {
                source.push_str(&format!("    if ({condition}) {{\n"));
                for assignment in assignments {
                    push_preludes(&mut source, &assignment.preludes, "        ");
                    let support = schedule
                        .mutable_support
                        .get(&assignment.target)
                        .expect("mutable symbolic assignment support exists");
                    source.push_str(&format!(
                        "        {}_v = {};\n",
                        assignment.target, assignment.value.value,
                    ));
                    for axis in 0..dimension {
                        if support.gradient[axis] {
                            source.push_str(&format!(
                                "        {}_g{axis} = {};\n",
                                assignment.target,
                                symbolic_component(&assignment.value.gradient[axis]),
                            ));
                        }
                        for other in axis..dimension {
                            let index = axis * dimension + other;
                            if support.hessian[index] {
                                source.push_str(&format!(
                                    "        {}_h{axis}_{other} = {};\n",
                                    assignment.target,
                                    symbolic_component(&assignment.value.hessian[index]),
                                ));
                            }
                        }
                    }
                }
                source.push_str("    }\n");
            }
        }
    }
    push_preludes(&mut source, &schedule.result_preludes, "    ");
    source.push_str(&format!("    *row_value = {};\n", schedule.result.value));
    for axis in 0..dimension {
        source.push_str(&format!(
            "    row_gradient[{axis}] = {};\n",
            symbolic_component(&schedule.result.gradient[axis]),
        ));
        for other in axis..dimension {
            let index = axis * dimension + other;
            let component = symbolic_component(&schedule.result.hessian[index]);
            source.push_str(&format!(
                "    row_hessian[{}] = {component};\n",
                axis * dimension + other,
            ));
            if axis != other {
                source.push_str(&format!(
                    "    row_hessian[{}] = {component};\n",
                    other * dimension + axis,
                ));
            }
        }
    }
    source.push_str("}\n");
    Ok(source)
}

pub(crate) fn expand(input: Input) -> Result<TokenStream2> {
    let Input {
        visibility,
        name,
        primaries,
        constants,
        emissions,
        leaves,
        witnesses,
        body,
    } = input;

    let mut all_names = HashSet::new();
    for name in primaries.iter().chain(constants.iter()) {
        if !all_names.insert(name.to_string()) {
            return Err(syn::Error::new_spanned(
                name,
                "row_program argument names must be unique",
            ));
        }
    }
    let constant_names = constants
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    let mut leaf_indices = HashMap::new();
    for (index, leaf) in leaves.iter().enumerate() {
        if leaf_indices.insert(leaf.alias.to_string(), index).is_some() {
            return Err(syn::Error::new_spanned(
                &leaf.alias,
                "row_program leaf aliases must be unique",
            ));
        }
    }

    let mut bindings = primaries
        .iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    let mut mutable = HashSet::new();
    let mut statements = Vec::new();
    for raw in body.statements {
        match raw {
            RawStatement::Local {
                name,
                mutable: is_mutable,
                value,
            } => {
                if all_names.contains(&name.to_string()) || bindings.contains(&name.to_string()) {
                    return Err(syn::Error::new_spanned(
                        name,
                        "row_program local name is already defined",
                    ));
                }
                let value = parse_program_expr(&value, &bindings, &constant_names, &leaf_indices)?;
                bindings.insert(name.to_string());
                if is_mutable {
                    mutable.insert(name.to_string());
                }
                statements.push(Statement::Local {
                    name,
                    mutable: is_mutable,
                    value,
                });
            }
            RawStatement::If {
                condition,
                assignments,
            } => {
                validate_scalar(&condition, &constant_names)?;
                let mut parsed_assignments = Vec::new();
                for (target, value) in assignments {
                    if !mutable.contains(&target.to_string()) {
                        return Err(syn::Error::new_spanned(
                            target,
                            "row_program assignment target must be a mutable local",
                        ));
                    }
                    parsed_assignments.push((
                        target,
                        parse_program_expr(&value, &bindings, &constant_names, &leaf_indices)?,
                    ));
                }
                statements.push(Statement::If {
                    condition,
                    assignments: parsed_assignments,
                });
            }
        }
    }
    let result = parse_program_expr(&body.result, &bindings, &constant_names, &leaf_indices)?;
    for witness in &witnesses {
        if !bindings.contains(&witness.to_string()) {
            return Err(syn::Error::new_spanned(
                witness,
                "row_program witness is not a defined jet",
            ));
        }
    }
    let witness_count = witnesses.len();
    if emissions.witnesses && witnesses.is_empty() {
        return Err(syn::Error::new_spanned(
            &name,
            "row_program cannot emit a `witnesses` surface with no declared witnesses",
        ));
    }
    let dimension = primaries.len();

    let generic_function = if emissions.generic {
        let rust_statements = statements.iter().map(|statement| match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let value = rust_expression(value, &leaves);
                if *mutable {
                    quote!(let mut #name = #value;)
                } else {
                    quote!(let #name = #value;)
                }
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let assignments = assignments.iter().map(|(target, value)| {
                    let value = rust_expression(value, &leaves);
                    quote!(#target = #value;)
                });
                quote!(if #condition { #(#assignments)* })
            }
        });
        let rust_result = rust_expression(&result, &leaves);
        let witness_values = witnesses.iter().map(|witness| quote!(#witness.value()));
        quote! {
            #[inline(always)]
            #visibility fn #name<
                const __ROW_PROGRAM_DERIVATIVE_DIMENSION: usize,
                S: ::gam_math::jet_scalar::JetScalar<__ROW_PROGRAM_DERIVATIVE_DIMENSION>,
            >(
                #(#primaries: &S,)*
                #(#constants: f64),*
            ) -> (S, [f64; #witness_count]) {
                #(#rust_statements)*
                let emitted_row_program_value = #rust_result;
                (emitted_row_program_value, [#(#witness_values),*])
            }
        }
    } else {
        quote!()
    };

    let runtime_function = if emissions.runtime {
        let runtime_name = format_ident!("{}_runtime", name);
        let runtime_primary_bindings = primaries
            .iter()
            .map(|primary| quote!(let #primary = (*#primary).clone();));
        let runtime_statements = statements.iter().map(|statement| match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } => {
                let value = rust_runtime_expression(value, &leaves);
                if *mutable {
                    quote!(let mut #name = #value;)
                } else {
                    quote!(let #name = #value;)
                }
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let assignments = assignments.iter().map(|(target, value)| {
                    let value = rust_runtime_expression(value, &leaves);
                    quote!(#target = #value;)
                });
                quote!(if #condition { #(#assignments)* })
            }
        });
        let runtime_result = rust_runtime_expression(&result, &leaves);
        let runtime_witness_values = witnesses.iter().map(|witness| quote!(#witness.value()));
        quote! {
            #[inline(always)]
            #visibility fn #runtime_name<'arena, S: ::gam_math::jet_scalar::RuntimeJetScalar<'arena>>(
                #(#primaries: &S,)*
                #(#constants: f64,)*
                __row_program_dimension: usize,
                __row_program_workspace: &'arena S::Workspace,
            ) -> (S, [f64; #witness_count]) {
                #(#runtime_primary_bindings)*
                #(#runtime_statements)*
                let emitted_row_program_value = #runtime_result;
                (emitted_row_program_value, [#(#runtime_witness_values),*])
            }
        }
    } else {
        quote!()
    };

    let order2_function = if emissions.order2 {
        let order2_name = format_ident!("{}_order2", name);
        let order2_body = rust_order2_body(
            &primaries,
            &constant_names,
            &leaves,
            &statements,
            &result,
            &witnesses,
        )?;
        quote! {
            #[inline(always)]
            #visibility fn #order2_name(
                #(#primaries: f64,)*
                #(#constants: f64),*
            ) -> (
                f64,
                [f64; #dimension],
                [[f64; #dimension]; #dimension],
                [f64; #witness_count],
            ) #order2_body
        }
    } else {
        quote!()
    };

    let third_function = if emissions.third {
        let third_name = format_ident!("{}_third_contracted", name);
        let third_body = if dimension <= 2 {
            rust_dense_taylor_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                false,
            )?
        } else {
            rust_directional_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                false,
            )?
        };
        quote! {
            #[inline(never)]
            #visibility fn #third_name(
                #(#primaries: f64,)*
                #(#constants: f64,)*
                direction_u: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] #third_body
        }
    } else {
        quote!()
    };

    let fourth_function = if emissions.fourth {
        let fourth_name = format_ident!("{}_fourth_contracted", name);
        let fourth_body = if dimension <= 2 {
            rust_dense_taylor_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                true,
            )?
        } else {
            rust_directional_body(
                &primaries,
                &constant_names,
                &leaves,
                &statements,
                &result,
                true,
            )?
        };
        quote! {
            #[inline(never)]
            #visibility fn #fourth_name(
                #(#primaries: f64,)*
                #(#constants: f64,)*
                direction_u: &[f64; #dimension],
                direction_v: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] #fourth_body
        }
    } else {
        quote!()
    };

    let scalar_witness_function = if emissions.witnesses {
        let scalar_witness_dependencies = witness_dependencies(&statements, &witnesses);
        let scalar_witness_scalar_dependencies =
            witness_scalar_dependencies(&statements, &scalar_witness_dependencies)?;
        let scalar_witness_statements = statements.iter().filter_map(|statement| match statement {
            Statement::Local {
                name,
                mutable,
                value,
            } if scalar_witness_dependencies.contains(&name.to_string()) => {
                let value = rust_scalar_expression(value, &leaves);
                Some(if *mutable {
                    quote!(let mut #name = #value;)
                } else {
                    quote!(let #name = #value;)
                })
            }
            Statement::If {
                condition,
                assignments,
            } => {
                let assignments = assignments
                    .iter()
                    .filter(|(target, _)| scalar_witness_dependencies.contains(&target.to_string()))
                    .map(|(target, value)| {
                        let value = rust_scalar_expression(value, &leaves);
                        quote!(#target = #value;)
                    })
                    .collect::<Vec<_>>();
                (!assignments.is_empty()).then(|| quote!(if #condition { #(#assignments)* }))
            }
            Statement::Local { .. } => None,
        });
        let scalar_witness_name = format_ident!("{}_witnesses", name);
        let scalar_witness_primaries = primaries
            .iter()
            .filter(|primary| scalar_witness_dependencies.contains(&primary.to_string()));
        let scalar_witness_constants = constants
            .iter()
            .filter(|constant| scalar_witness_scalar_dependencies.contains(&constant.to_string()));
        let scalar_witness_values = witnesses.iter();
        quote! {
            #[inline(always)]
            #visibility fn #scalar_witness_name(
                #(#scalar_witness_primaries: f64,)*
                #(#scalar_witness_constants: f64),*
            ) -> [f64; #witness_count] {
                #(#scalar_witness_statements)*
                [#(#scalar_witness_values),*]
            }
        }
    } else {
        quote!()
    };

    let cuda_constant = if emissions.cuda {
        let cuda = cuda_source(
            &name,
            &primaries,
            &constant_names,
            &leaves,
            &statements,
            &result,
        )?;
        let cuda_literal = Literal::string(&cuda);
        let cuda_name = format_ident!("{}_CUDA_VGH", name.to_string().to_uppercase());
        quote!(#visibility const #cuda_name: &str = #cuda_literal;)
    } else {
        quote!()
    };

    Ok(quote! {
        #generic_function
        #runtime_function
        #order2_function
        #third_function
        #fourth_function
        #scalar_witness_function
        #cuda_constant
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn emitted_cuda(input: TokenStream2) -> String {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .find_map(|item| {
                let syn::Item::Const(item) = item else {
                    return None;
                };
                let syn::Expr::Lit(expression) = *item.expr else {
                    return None;
                };
                let syn::Lit::Str(source) = expression.lit else {
                    return None;
                };
                Some(source.value())
            })
            .expect("expanded CUDA source constant")
    }

    fn emitted_function(input: TokenStream2, name: &str) -> String {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .find_map(|item| {
                let syn::Item::Fn(item) = item else {
                    return None;
                };
                (item.sig.ident == name).then(|| quote!(#item).to_string())
            })
            .expect("expanded function")
    }

    fn emitted_item_names(input: TokenStream2) -> Vec<String> {
        let input = syn::parse2::<Input>(input).expect("parse row program");
        let expanded = expand(input).expect("expand row program");
        let file = syn::parse2::<syn::File>(expanded).expect("parse macro expansion");
        file.items
            .into_iter()
            .map(|item| match item {
                syn::Item::Fn(item) => item.sig.ident.to_string(),
                syn::Item::Const(item) => item.ident.to_string(),
                _ => panic!("unexpected emitted row-program item"),
            })
            .collect()
    }

    fn parse_error(input: TokenStream2) -> String {
        match syn::parse2::<Input>(input) {
            Ok(_) => panic!("row program unexpectedly parsed"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn emits_generic_and_shared_symbolic_rust_cuda_schedules() {
        let input = syn::parse2::<Input>(quote! {
            pub(crate) fn sample(q, g; weight, event, scale)
            emit [generic, runtime, order2, third, fourth, witnesses, cuda];
            leaves {
                sqrt => sqrt_stack => d_sqrt,
                log => log_stack => d_log,
            }
            witnesses [adjusted];
            {
                let scaled = scale(g, scale);
                let square = add_constant(mul(scaled, scaled), 1.0);
                let correction = compose(sqrt, square);
                let adjusted = mul(q, correction);
                let mut event_term = zero();
                if (event > 0.0) {
                    event_term = scale(compose(log, adjusted), -(weight * event));
                }
                return add(adjusted, event_term);
            }
        })
        .expect("parse row program");
        let expanded = expand(input).expect("expand row program").to_string();
        assert!(expanded.contains("JetScalar < __ROW_PROGRAM_DERIVATIVE_DIMENSION >"));
        assert!(expanded.contains("RuntimeJetScalar"));
        assert!(expanded.contains("fn sample_runtime"));
        assert!(expanded.contains("fn sample_order2"));
        assert!(expanded.contains("fn sample_third_contracted"));
        assert!(expanded.contains("fn sample_fourth_contracted"));
        assert!(expanded.contains("direction_u"));
        assert!(expanded.contains("direction_v"));
        assert!(expanded.contains("sqrt_stack"));
        assert!(expanded.contains("log_stack"));
        assert!(expanded.contains("SAMPLE_CUDA_VGH"));
        assert!(expanded.contains("double event_term_v = 0.0"));
        assert!(expanded.contains("if ((in.event > 0.0))"));
        assert!(expanded.contains("d_log(adjusted_v"));
        assert!(!expanded.contains("J2"));
        assert!(!expanded.contains("j2_"));
    }

    #[test]
    fn emits_exactly_the_mandatory_per_program_surfaces() {
        let names = emitted_item_names(quote! {
            fn selective(x; shift)
            emit [runtime, order2, third, fourth];
            leaves { curve => curve_stack => d_curve }
            witnesses [curved];
            {
                let shifted = add_constant(x, shift);
                let curved = compose(curve, shifted);
                return curved;
            }
        });

        assert_eq!(
            names,
            vec![
                "selective_runtime".to_owned(),
                "selective_order2".to_owned(),
                "selective_third_contracted".to_owned(),
                "selective_fourth_contracted".to_owned(),
            ]
        );
    }

    #[test]
    fn emission_surfaces_are_mandatory_nonempty_known_and_unique() {
        let missing = parse_error(quote! {
            fn missing(x;)
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(missing.contains("mandatory `emit [ ... ];`"));

        let empty = parse_error(quote! {
            fn empty(x;)
            emit [];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(empty.contains("must emit at least one surface"));

        let unknown = parse_error(quote! {
            fn unknown(x;)
            emit [jet];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(
            unknown.contains(
                "must be one of `generic`, `runtime`, `order2`, `third`, `fourth`, `witnesses`, or `cuda`"
            )
        );

        let duplicate = parse_error(quote! {
            fn duplicate(x;)
            emit [runtime, runtime];
            leaves {}
            witnesses [];
            { return x; }
        });
        assert!(duplicate.contains("duplicate row_program emission surface `runtime`"));
    }

    #[test]
    fn rejects_empty_scalar_witness_surface() {
        let input = syn::parse2::<Input>(quote! {
            fn empty_witnesses(x;)
            emit [witnesses];
            leaves {}
            witnesses [];
            { return x; }
        })
        .expect("parse row program");
        let error = expand(input).expect_err("empty witness surface must be rejected");
        assert!(error.to_string().contains("no declared witnesses"));
    }

    #[test]
    fn runtime_rust_schedule_clones_reusable_bindings_and_uses_runtime_workspace() {
        let rust = emitted_function(
            quote! {
                fn runtime_formula(x, y; take, shift)
                emit [runtime];
                leaves { curve => curve_stack => d_curve }
                witnesses [curved];
                {
                    let sum = add(x, y);
                    let shifted = add_constant(sum, shift);
                    let curved = compose(curve, shifted);
                    let mut out = zero();
                    if (take > 0.0) { out = add(curved, x); }
                    return add(out, curved);
                }
            },
            "runtime_formula_runtime",
        )
        .replace(' ', "");

        for formula in [
            "S:::gam_math::jet_scalar::RuntimeJetScalar<'arena>",
            "letx=(*x).clone();",
            "lety=(*y).clone();",
            "value.add_constant(shift)",
            "S::constant(0.0,__row_program_dimension,__row_program_workspace)",
            "letvalue=shifted.clone();",
            "[curved.value()]",
        ] {
            assert!(
                rust.contains(formula),
                "missing generated runtime formula: {formula}\n{rust}"
            );
        }
    }

    #[test]
    fn rust_order2_formulas_pin_sparse_mul_compose_branch_witness_and_symmetry() {
        let rust = emitted_function(
            quote! {
                fn formulas(x, y; take)
                emit [order2];
                leaves { curve => curve_stack => d_curve }
                witnesses [curved];
                {
                    let product = mul(x, y);
                    let curved = compose(curve, product);
                    let mut out = x;
                    if (take > 0.0) { out = add(curved, y); }
                    return out;
                }
            },
            "formulas_order2",
        )
        .replace(' ', "");

        for formula in [
            "fnformulas_order2(x:f64,y:f64,take:f64)",
            "letproduct_g0:f64=y;",
            "letproduct_h0_1:f64=1.0;",
            "letproduct_g1:f64=x;",
            "letcurved_stack0=curve_stack(product_v);",
            "letcurved_g0:f64=(product_g0*curved_stack0[1]);",
            "letcurved_h0_0:f64=(curved_stack0[2]*(product_g0*product_g0));",
            "letcurved_h0_1:f64=((product_h0_1*curved_stack0[1])+(curved_stack0[2]*(product_g0*product_g1)));",
            "letcurved_g1:f64=(product_g1*curved_stack0[1]);",
            "letcurved_h1_1:f64=(curved_stack0[2]*(product_g1*product_g1));",
            "letmutout_g0:f64=1.0;",
            "letmutout_h0_1:f64=0.0;",
            "if(take>0.0){",
            "out_g1=(curved_g1+1.0);",
            "[__row_program_h0_0,__row_program_h0_1],",
            "[__row_program_h0_1,__row_program_h1_1],",
            "[curved_v]",
        ] {
            assert!(
                rust.contains(formula),
                "missing generated formula: {formula}"
            );
        }
        assert!(!rust.contains("JetScalar"));
        assert!(!rust.contains("SparseOrder2"));
        assert!(!rust.contains("*0.0"));
        assert!(!rust.contains("0.0*"));
    }

    #[test]
    fn contracted_formulas_are_direct_sparse_scalar_schedules() {
        let input = quote! {
            fn directional(x, y; take)
            emit [third, fourth];
            leaves { curve => curve_stack => d_curve }
            witnesses [];
            {
                let product = mul(x, y);
                let curved = compose(curve, product);
                let mut out = x;
                if (take > 0.0) { out = add(curved, y); }
                return out;
            }
        };
        let third =
            emitted_function(input.clone(), "directional_third_contracted").replace(' ', "");
        let fourth = emitted_function(input, "directional_fourth_contracted").replace(' ', "");

        for formula in [
            "fndirectional_third_contracted(x:f64,y:f64,take:f64,direction_u:&[f64;2usize],)",
            "let__row_program_product_dense_tmp0_c6:f64=1.0;",
            "let__row_program_curved_dense_stack1=curve_stack(product_c0);",
            "__row_program_curved_dense_stack1[3]",
            "letdense_derivatives=[",
            "dense_derivatives[offset]*direction_u[0]",
            "if(take>0.0){",
        ] {
            assert!(
                third.contains(formula),
                "missing generated third-order formula: {formula}\n{third}"
            );
        }
        for formula in [
            "fndirectional_fourth_contracted(x:f64,y:f64,take:f64,direction_u:&[f64;2usize],direction_v:&[f64;2usize],)",
            "__row_program_curved_dense_stack1[4]",
            "letdirection_01=direction_u[0]*direction_v[1]+direction_u[1]*direction_v[0];",
            "dense_derivatives[offset+1]*direction_01",
        ] {
            assert!(
                fourth.contains(formula),
                "missing generated fourth-order formula: {formula}\n{fourth}"
            );
        }
        for rust in [&third, &fourth] {
            assert!(!rust.contains("JetScalar"));
            assert!(!rust.contains("SparseOrder2"));
            assert!(!rust.contains("*0.0)"));
            assert!(!rust.contains("0.0*"));
        }
    }

    #[test]
    fn rejects_primary_dependent_runtime_branch() {
        let input = syn::parse2::<Input>(quote! {
            fn bad(q; event)
            emit [generic];
            leaves { log => log_stack => d_log }
            witnesses [];
            {
                let mut out = zero();
                if (q > 0.0) { out = compose(log, q); }
                return out;
            }
        })
        .expect("parse row program");
        let error = expand(input).expect_err("primary branch must be rejected");
        assert!(error.to_string().contains("unknown row_program scalar `q`"));
    }

    #[test]
    fn cuda_formulas_pin_sparse_mul_compose_and_mutable_support_union() {
        let cuda = emitted_cuda(quote! {
            fn formulas(x, y; take)
            emit [cuda];
            leaves { curve => curve_stack => d_curve }
            witnesses [];
            {
                let product = mul(x, y);
                let curved = compose(curve, product);
                let mut out = x;
                if (take > 0.0) { out = add(curved, y); }
                return out;
            }
        });

        for formula in [
            "double product_g0 = y;",
            "double product_h0_1 = 1.0;",
            "double product_g1 = x;",
            "double curved_g0 = (product_g0 * curved_stack0[1]);",
            "double curved_h0_0 = (curved_stack0[2] * (product_g0 * product_g0));",
            "double curved_h0_1 = ((product_h0_1 * curved_stack0[1]) + (curved_stack0[2] * (product_g0 * product_g1)));",
            "double curved_g1 = (product_g1 * curved_stack0[1]);",
            "double curved_h1_1 = (curved_stack0[2] * (product_g1 * product_g1));",
            "double out_g0 = 1.0;",
            "double out_h0_0 = 0.0;",
            "double out_h0_1 = 0.0;",
            "double out_g1 = 0.0;",
            "double out_h1_1 = 0.0;",
            "out_g0 = curved_g0;",
            "out_h0_0 = curved_h0_0;",
            "out_h0_1 = curved_h0_1;",
            "out_g1 = (curved_g1 + 1.0);",
            "out_h1_1 = curved_h1_1;",
            "row_hessian[0] = out_h0_0;",
            "row_hessian[1] = out_h0_1;",
            "row_hessian[2] = out_h0_1;",
            "row_hessian[3] = out_h1_1;",
        ] {
            assert!(
                cuda.contains(formula),
                "missing generated formula: {formula}"
            );
        }
        assert!(!cuda.contains("* 0.0"));
        assert!(!cuda.contains("0.0 *"));
    }

    #[test]
    fn cuda_compose_temporaries_are_unique_across_repeated_assignments() {
        let cuda = emitted_cuda(quote! {
            fn repeated(q; event)
            emit [order2, cuda];
            leaves { log => log_stack => d_log }
            witnesses [];
            {
                let mut out = q;
                if (event > 0.0) {
                    out = compose(log, out);
                    out = compose(log, out);
                }
                return out;
            }
        });

        assert_eq!(cuda.matches("double out_stack0[3]").count(), 1);
        assert_eq!(cuda.matches("double out_stack1[3]").count(), 1);
        assert_eq!(cuda.matches("d_log(out_v, out_stack").count(), 2);

        let rust = emitted_function(
            quote! {
                fn repeated(q; event)
                emit [order2, cuda];
                leaves { log => log_stack => d_log }
                witnesses [];
                {
                    let mut out = q;
                    if (event > 0.0) {
                        out = compose(log, out);
                        out = compose(log, out);
                    }
                    return out;
                }
            },
            "repeated_order2",
        );
        assert_eq!(rust.matches("let out_stack0").count(), 1);
        assert_eq!(rust.matches("let out_stack1").count(), 1);
        assert_eq!(rust.matches("log_stack (out_v)").count(), 2);
    }

    #[test]
    fn scalar_witness_schedule_is_dependency_sliced_from_the_same_program() {
        let witness = emitted_function(
            quote! {
                fn sliced(q, g; event)
                emit [witnesses];
                leaves {
                    sqrt => sqrt_stack => d_sqrt,
                    log => log_stack => d_log,
                }
                witnesses [adjusted];
                {
                    let square = add_constant(mul(g, g), 1.0);
                    let correction = compose(sqrt, square);
                    let adjusted = mul(q, correction);
                    let discarded = compose(log, adjusted);
                    return add(adjusted, discarded);
                }
            },
            "sliced_witnesses",
        );

        assert!(witness.contains("sqrt_stack"));
        assert!(witness.contains("adjusted"));
        assert!(!witness.contains("log_stack"));
        assert!(!witness.contains("discarded"));
        assert!(!witness.contains("event : f64"));
    }

    #[test]
    fn scalar_witness_schedule_retains_needed_branch_condition() {
        let witness = emitted_function(
            quote! {
                fn branched(q; event, unused)
                emit [witnesses];
                leaves {}
                witnesses [out];
                {
                    let mut out = zero();
                    if (event > 0.0) { out = q; }
                    return out;
                }
            },
            "branched_witnesses",
        );

        assert!(witness.contains("q : f64"));
        assert!(witness.contains("event : f64"));
        assert!(!witness.contains("unused : f64"));
    }
}
