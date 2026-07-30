//! Restricted build-time differentiation for small row-program atoms.
//!
//! [`row_atom`] accepts one scalar expression and emits two backends from that
//! single source: a generic `JetScalar` evaluator for arbitrary derivative
//! order, and a straight-line scalar value/gradient/packed-Hessian schedule.
//! Symbolic zeros are removed before Rust/LLVM see the generated schedule, so
//! it carries neither runtime dependency masks nor the `0*x` work that ordinary
//! forward jets must preserve for IEEE-754 semantics.
//!
//! Local-coordinate programs can request `order2_at_zero`, `third_at_zero`,
//! and `fourth_at_zero`. Those surfaces differentiate the same expression,
//! substitute zero for every primary, canonicalize the remaining parameter
//! polynomial, and rebuild it as a multivariate Horner schedule. Their emitted
//! functions consequently accept only the runtime constants and directions.

use proc_macro::TokenStream;
use proc_macro2::{Ident, Literal, Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use std::collections::{BTreeMap, HashMap, HashSet};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    BinOp, Expr, ExprBinary, ExprCall, ExprGroup, ExprLit, ExprParen, ExprPath, ExprUnary, Lit,
    Result, Token, UnOp, Visibility, braced, bracketed, parenthesized, parse_macro_input,
};

mod row_program;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Lowering {
    Generic,
    Order2,
    Third,
    Fourth,
    Order2AtZero,
    ThirdAtZero,
    FourthAtZero,
}

struct RowAtomInput {
    visibility: Visibility,
    name: Ident,
    lowerings: HashSet<Lowering>,
    primaries: Vec<Ident>,
    constants: Vec<Ident>,
    expression: Expr,
}

impl Parse for RowAtomInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let name = input.parse()?;
        let lowering_tokens;
        bracketed!(lowering_tokens in input);
        let mut lowerings = HashSet::new();
        for lowering in Punctuated::<Ident, Token![,]>::parse_terminated(&lowering_tokens)? {
            let lowering = match lowering.to_string().as_str() {
                "generic" => Lowering::Generic,
                "order2" => Lowering::Order2,
                "third" => Lowering::Third,
                "fourth" => Lowering::Fourth,
                "order2_at_zero" => Lowering::Order2AtZero,
                "third_at_zero" => Lowering::ThirdAtZero,
                "fourth_at_zero" => Lowering::FourthAtZero,
                _ => {
                    return Err(syn::Error::new_spanned(
                        lowering,
                        "row_atom lowerings are generic, order2, third, fourth, \
                         order2_at_zero, third_at_zero, and fourth_at_zero",
                    ));
                }
            };
            if !lowerings.insert(lowering) {
                return Err(lowering_tokens.error("row_atom lowering listed more than once"));
            }
        }
        if lowerings.is_empty() {
            return Err(lowering_tokens.error("row_atom requires at least one lowering"));
        }
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
        if !arguments.is_empty() {
            return Err(arguments.error("invalid row_atom argument list"));
        }
        if primaries.is_empty() {
            return Err(input.error("row_atom requires at least one primary"));
        }
        let mut bindings = HashSet::new();
        for binding in primaries.iter().chain(constants.iter()) {
            if !bindings.insert(binding.to_string()) {
                return Err(syn::Error::new_spanned(
                    binding,
                    "row_atom argument names must be unique",
                ));
            }
        }
        let body;
        braced!(body in input);
        let expression = body.parse()?;
        if !body.is_empty() {
            return Err(body.error("row_atom body must contain exactly one expression"));
        }
        Ok(Self {
            visibility,
            name,
            lowerings,
            primaries,
            constants,
            expression,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Node {
    Constant(u64),
    Variable(usize),
    Parameter(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Neg(usize),
    Exp(usize),
    Ln(usize),
    Sqrt(usize),
}

struct Graph {
    nodes: Vec<Node>,
    interned: HashMap<Node, usize>,
    derivatives: HashMap<(usize, usize), usize>,
}

type Polynomial = BTreeMap<Vec<usize>, f64>;

impl Graph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            interned: HashMap::new(),
            derivatives: HashMap::new(),
        }
    }

    fn intern(&mut self, node: Node) -> usize {
        if let Some(&id) = self.interned.get(&node) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.interned.insert(node, id);
        id
    }

    fn constant(&mut self, value: f64) -> usize {
        self.intern(Node::Constant(value.to_bits()))
    }

    fn constant_value(&self, id: usize) -> Option<f64> {
        match self.nodes[id] {
            Node::Constant(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    fn is_zero(&self, id: usize) -> bool {
        self.constant_value(id).is_some_and(|value| value == 0.0)
    }

    fn is_one(&self, id: usize) -> bool {
        self.constant_value(id) == Some(1.0)
    }

    fn add(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(left) {
            return right;
        }
        if self.is_zero(right) {
            return left;
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left + right);
        }
        if let Node::Sub(value, removed) = self.nodes[left]
            && removed == right
        {
            return value;
        }
        if let Node::Sub(value, removed) = self.nodes[right]
            && removed == left
        {
            return value;
        }
        self.intern(Node::Add(left, right))
    }

    fn sub(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(right) {
            return left;
        }
        if left == right {
            return self.constant(0.0);
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left - right);
        }
        self.intern(Node::Sub(left, right))
    }

    fn mul(&mut self, left: usize, right: usize) -> usize {
        if self.is_zero(left) || self.is_zero(right) {
            return self.constant(0.0);
        }
        if self.is_one(left) {
            return right;
        }
        if self.is_one(right) {
            return left;
        }
        if let (Some(left), Some(right)) = (self.constant_value(left), self.constant_value(right)) {
            return self.constant(left * right);
        }
        self.intern(Node::Mul(left, right))
    }

    fn div(&mut self, numerator: usize, denominator: usize) -> usize {
        if self.is_zero(numerator) {
            return self.constant(0.0);
        }
        if self.is_one(denominator) {
            return numerator;
        }
        if let (Some(numerator), Some(denominator)) = (
            self.constant_value(numerator),
            self.constant_value(denominator),
        ) {
            return self.constant(numerator / denominator);
        }
        self.intern(Node::Div(numerator, denominator))
    }

    fn neg(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(-value);
        }
        if let Node::Neg(inner) = self.nodes[value] {
            return inner;
        }
        self.intern(Node::Neg(value))
    }

    fn exp(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.exp());
        }
        self.intern(Node::Exp(value))
    }

    fn ln(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.ln());
        }
        self.intern(Node::Ln(value))
    }

    fn sqrt(&mut self, value: usize) -> usize {
        if let Some(value) = self.constant_value(value) {
            return self.constant(value.sqrt());
        }
        self.intern(Node::Sqrt(value))
    }

    fn substitute_zero_primaries(&mut self, id: usize, memo: &mut HashMap<usize, usize>) -> usize {
        if let Some(&specialized) = memo.get(&id) {
            return specialized;
        }
        let node = self.nodes[id].clone();
        let specialized = match node {
            Node::Constant(_) | Node::Parameter(_) => id,
            Node::Variable(_) => self.constant(0.0),
            Node::Add(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.mul(left, right)
            }
            Node::Div(left, right) => {
                let left = self.substitute_zero_primaries(left, memo);
                let right = self.substitute_zero_primaries(right, memo);
                self.div(left, right)
            }
            Node::Neg(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.neg(value)
            }
            Node::Exp(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.exp(value)
            }
            Node::Ln(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.ln(value)
            }
            Node::Sqrt(value) => {
                let value = self.substitute_zero_primaries(value, memo);
                self.sqrt(value)
            }
        };
        memo.insert(id, specialized);
        specialized
    }

    fn polynomial(
        &self,
        id: usize,
        parameter_count: usize,
        memo: &mut HashMap<usize, Option<Polynomial>>,
    ) -> Option<Polynomial> {
        if let Some(polynomial) = memo.get(&id) {
            return polynomial.clone();
        }
        let zero_exponents = || vec![0; parameter_count];
        let polynomial = match self.nodes[id].clone() {
            Node::Constant(bits) => {
                let value = f64::from_bits(bits);
                let mut polynomial = Polynomial::new();
                if value != 0.0 {
                    polynomial.insert(zero_exponents(), value);
                }
                Some(polynomial)
            }
            Node::Parameter(parameter) => {
                let mut exponents = zero_exponents();
                exponents[parameter] = 1;
                Some([(exponents, 1.0)].into_iter().collect())
            }
            Node::Variable(_) | Node::Exp(_) | Node::Ln(_) | Node::Sqrt(_) => None,
            Node::Neg(value) => self
                .polynomial(value, parameter_count, memo)
                .map(|mut value| {
                    for coefficient in value.values_mut() {
                        *coefficient = -*coefficient;
                    }
                    value
                }),
            Node::Add(left, right) | Node::Sub(left, right) => {
                let mut left = self.polynomial(left, parameter_count, memo)?;
                let right = self.polynomial(right, parameter_count, memo)?;
                let sign = if matches!(self.nodes[id], Node::Add(_, _)) {
                    1.0
                } else {
                    -1.0
                };
                for (exponents, coefficient) in right {
                    let total = left.entry(exponents).or_default();
                    *total += sign * coefficient;
                }
                left.retain(|_, coefficient| *coefficient != 0.0);
                Some(left)
            }
            Node::Mul(left, right) => {
                let left = self.polynomial(left, parameter_count, memo)?;
                let right = self.polynomial(right, parameter_count, memo)?;
                let mut product = Polynomial::new();
                for (left_exponents, left_coefficient) in &left {
                    for (right_exponents, right_coefficient) in &right {
                        let exponents = left_exponents
                            .iter()
                            .zip(right_exponents)
                            .map(|(left, right)| left + right)
                            .collect::<Vec<_>>();
                        *product.entry(exponents).or_default() +=
                            left_coefficient * right_coefficient;
                    }
                }
                product.retain(|_, coefficient| *coefficient != 0.0);
                Some(product)
            }
            Node::Div(numerator, denominator) => {
                let mut numerator = self.polynomial(numerator, parameter_count, memo)?;
                let denominator = self.polynomial(denominator, parameter_count, memo)?;
                let coefficient = denominator.get(&zero_exponents()).copied()?;
                if denominator.len() != 1 || coefficient == 0.0 {
                    None
                } else {
                    for value in numerator.values_mut() {
                        *value /= coefficient;
                    }
                    Some(numerator)
                }
            }
        };
        memo.insert(id, polynomial.clone());
        polynomial
    }

    fn polynomial_horner(&mut self, polynomial: &Polynomial, variables: &[usize]) -> usize {
        if polynomial.is_empty() {
            return self.constant(0.0);
        }
        if variables.is_empty() {
            return self.constant(*polynomial.values().next().expect("nonempty polynomial"));
        }
        let variable = variables[0];
        let parameter = self.intern(Node::Parameter(variable));
        let mut coefficients = BTreeMap::<usize, Polynomial>::new();
        for (exponents, coefficient) in polynomial {
            let exponent = exponents[variable];
            let mut coefficient_exponents = exponents.clone();
            coefficient_exponents[variable] = 0;
            coefficients
                .entry(exponent)
                .or_default()
                .insert(coefficient_exponents, *coefficient);
        }
        let mut descending = coefficients.iter().rev();
        let (&highest, leading) = descending.next().expect("nonempty coefficients");
        let mut result = self.polynomial_horner(leading, &variables[1..]);
        let mut previous = highest;
        for (&exponent, coefficient) in descending {
            for _ in exponent..previous {
                result = self.mul(result, parameter);
            }
            let coefficient = self.polynomial_horner(coefficient, &variables[1..]);
            result = self.add(result, coefficient);
            previous = exponent;
        }
        for _ in 0..previous {
            result = self.mul(result, parameter);
        }
        result
    }

    fn normalize_polynomial(&mut self, id: usize, parameter_count: usize) -> usize {
        let Some(polynomial) = self.polynomial(id, parameter_count, &mut HashMap::new()) else {
            return id;
        };
        self.polynomial_horner(&polynomial, &(0..parameter_count).collect::<Vec<_>>())
    }

    fn derivative(&mut self, id: usize, variable: usize) -> usize {
        if let Some(&derivative) = self.derivatives.get(&(id, variable)) {
            return derivative;
        }
        let node = self.nodes[id].clone();
        let derivative = match node {
            Node::Constant(_) | Node::Parameter(_) => self.constant(0.0),
            Node::Variable(axis) => self.constant(f64::from(axis == variable)),
            Node::Add(left, right) => {
                let left = self.derivative(left, variable);
                let right = self.derivative(right, variable);
                self.add(left, right)
            }
            Node::Sub(left, right) => {
                let left = self.derivative(left, variable);
                let right = self.derivative(right, variable);
                self.sub(left, right)
            }
            Node::Mul(left, right) => {
                let left_derivative = self.derivative(left, variable);
                let right_derivative = self.derivative(right, variable);
                let first = self.mul(left, right_derivative);
                let second = self.mul(left_derivative, right);
                self.add(first, second)
            }
            Node::Div(numerator, denominator) => {
                let numerator_derivative = self.derivative(numerator, variable);
                let denominator_derivative = self.derivative(denominator, variable);
                let first = self.mul(numerator_derivative, denominator);
                let second = self.mul(numerator, denominator_derivative);
                let top = self.sub(first, second);
                let bottom = self.mul(denominator, denominator);
                self.div(top, bottom)
            }
            Node::Neg(value) => {
                let derivative = self.derivative(value, variable);
                self.neg(derivative)
            }
            Node::Exp(value) => {
                let exp = self.intern(Node::Exp(value));
                let derivative = self.derivative(value, variable);
                self.mul(exp, derivative)
            }
            Node::Ln(value) => {
                let derivative = self.derivative(value, variable);
                self.div(derivative, value)
            }
            Node::Sqrt(value) => {
                let derivative = self.derivative(value, variable);
                let two = self.constant(2.0);
                let sqrt = self.intern(Node::Sqrt(value));
                let denominator = self.mul(two, sqrt);
                self.div(derivative, denominator)
            }
        };
        self.derivatives.insert((id, variable), derivative);
        derivative
    }
}

enum Binding {
    Primary(usize),
    Constant(usize),
}

fn binding(path: &ExprPath, primaries: &[Ident], constants: &[Ident]) -> Result<Binding> {
    let ident = path
        .path
        .get_ident()
        .ok_or_else(|| syn::Error::new_spanned(path, "row_atom variables must be identifiers"))?;
    if let Some(axis) = primaries.iter().position(|candidate| candidate == ident) {
        return Ok(Binding::Primary(axis));
    }
    constants
        .iter()
        .position(|candidate| candidate == ident)
        .map(Binding::Constant)
        .ok_or_else(|| syn::Error::new_spanned(path, format!("unknown row_atom binding `{ident}`")))
}

fn literal_value(literal: &ExprLit) -> Result<f64> {
    match &literal.lit {
        Lit::Float(value) => value.base10_parse(),
        Lit::Int(value) => value.base10_parse(),
        _ => Err(syn::Error::new_spanned(
            literal,
            "row_atom supports only numeric literals",
        )),
    }
}

fn call_name(call: &ExprCall) -> Result<&Ident> {
    let Expr::Path(path) = call.func.as_ref() else {
        return Err(syn::Error::new_spanned(
            &call.func,
            "row_atom unary calls must use a bare function name",
        ));
    };
    path.path.get_ident().ok_or_else(|| {
        syn::Error::new_spanned(
            &call.func,
            "row_atom unary calls must use a bare function name",
        )
    })
}

fn graph_expression(
    expression: &Expr,
    primaries: &[Ident],
    constants: &[Ident],
    graph: &mut Graph,
) -> Result<usize> {
    match expression {
        Expr::Path(path) => Ok(match binding(path, primaries, constants)? {
            Binding::Primary(axis) => graph.intern(Node::Variable(axis)),
            Binding::Constant(index) => graph.intern(Node::Parameter(index)),
        }),
        Expr::Lit(literal) => Ok(graph.constant(literal_value(literal)?)),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            graph_expression(expr, primaries, constants, graph)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let value = graph_expression(expr, primaries, constants, graph)?;
            Ok(graph.neg(value))
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left = graph_expression(left, primaries, constants, graph)?;
            let right = graph_expression(right, primaries, constants, graph)?;
            let node = match op {
                BinOp::Add(_) => graph.add(left, right),
                BinOp::Sub(_) => graph.sub(left, right),
                BinOp::Mul(_) => graph.mul(left, right),
                BinOp::Div(_) => graph.div(left, right),
                _ => {
                    return Err(syn::Error::new_spanned(
                        op,
                        "row_atom supports +, -, *, and /",
                    ));
                }
            };
            Ok(node)
        }
        Expr::Call(call) => {
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "row_atom unary functions take one argument",
                ));
            }
            let argument = graph_expression(&call.args[0], primaries, constants, graph)?;
            let node = match call_name(call)?.to_string().as_str() {
                "exp" => graph.exp(argument),
                "ln" => graph.ln(argument),
                "sqrt" => graph.sqrt(argument),
                "recip" => {
                    let one = graph.constant(1.0);
                    graph.div(one, argument)
                }
                name => {
                    return Err(syn::Error::new_spanned(
                        call,
                        format!("unsupported row_atom unary function `{name}`"),
                    ));
                }
            };
            Ok(node)
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_atom expression",
        )),
    }
}

fn jet_expression(
    expression: &Expr,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<TokenStream2> {
    match expression {
        Expr::Path(path) => match binding(path, primaries, constants)? {
            Binding::Primary(axis) => {
                let variable = &primaries[axis];
                Ok(quote!(*#variable))
            }
            Binding::Constant(index) => {
                let constant = &constants[index];
                Ok(quote!(S::constant(#constant)))
            }
        },
        Expr::Lit(literal) => Ok(quote!(S::constant((#literal) as f64))),
        Expr::Paren(ExprParen { expr, .. }) | Expr::Group(ExprGroup { expr, .. }) => {
            jet_expression(expr, primaries, constants)
        }
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let value = jet_expression(expr, primaries, constants)?;
            Ok(quote!({ let value = #value; value.neg() }))
        }
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let left = jet_expression(left, primaries, constants)?;
            let right = jet_expression(right, primaries, constants)?;
            match op {
                BinOp::Add(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.add(&right) }))
                }
                BinOp::Sub(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.sub(&right) }))
                }
                BinOp::Mul(_) => {
                    Ok(quote!({ let left = #left; let right = #right; left.mul(&right) }))
                }
                BinOp::Div(_) => Ok(quote!({
                    let left = #left;
                    let right = #right;
                    left.mul(&right.recip())
                })),
                _ => Err(syn::Error::new_spanned(
                    op,
                    "row_atom supports +, -, *, and /",
                )),
            }
        }
        Expr::Call(call) => {
            if call.args.len() != 1 {
                return Err(syn::Error::new_spanned(
                    call,
                    "row_atom unary functions take one argument",
                ));
            }
            let argument = jet_expression(&call.args[0], primaries, constants)?;
            let method = call_name(call)?;
            match method.to_string().as_str() {
                "exp" | "ln" | "sqrt" | "recip" => Ok(quote!({
                    let value = #argument;
                    value.#method()
                })),
                name => Err(syn::Error::new_spanned(
                    call,
                    format!("unsupported row_atom unary function `{name}`"),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            expression,
            "unsupported row_atom expression",
        )),
    }
}

fn topological_order(id: usize, graph: &Graph, seen: &mut HashSet<usize>, order: &mut Vec<usize>) {
    if !seen.insert(id) {
        return;
    }
    match graph.nodes[id] {
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => {}
        Node::Neg(value) | Node::Exp(value) | Node::Ln(value) | Node::Sqrt(value) => {
            topological_order(value, graph, seen, order);
        }
        Node::Add(left, right)
        | Node::Sub(left, right)
        | Node::Mul(left, right)
        | Node::Div(left, right) => {
            topological_order(left, graph, seen, order);
            topological_order(right, graph, seen, order);
        }
    }
    if !matches!(
        graph.nodes[id],
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_)
    ) {
        order.push(id);
    }
}

fn node_reference(
    id: usize,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> TokenStream2 {
    match graph.nodes[id] {
        Node::Constant(bits) => {
            let literal = Literal::f64_unsuffixed(f64::from_bits(bits));
            quote!(#literal)
        }
        Node::Variable(axis) => {
            let variable = &primaries[axis];
            quote!(#variable)
        }
        Node::Parameter(index) => {
            let constant = &constants[index];
            quote!(#constant)
        }
        _ => {
            let temporary = format_ident!("__row_atom_{id}");
            quote!(#temporary)
        }
    }
}

fn node_definition(
    id: usize,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<TokenStream2> {
    let reference = |child| node_reference(child, graph, primaries, constants);
    match graph.nodes[id] {
        Node::Add(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left + #right))
        }
        Node::Sub(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left - #right))
        }
        Node::Mul(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left * #right))
        }
        Node::Div(left, right) => {
            let (left, right) = (reference(left), reference(right));
            Ok(quote!(#left / #right))
        }
        Node::Neg(value) => {
            let value = reference(value);
            Ok(quote!(-#value))
        }
        Node::Exp(value) => {
            let value = reference(value);
            Ok(quote!(#value.exp()))
        }
        Node::Ln(value) => {
            let value = reference(value);
            Ok(quote!(#value.ln()))
        }
        Node::Sqrt(value) => {
            let value = reference(value);
            Ok(quote!(#value.sqrt()))
        }
        Node::Constant(_) | Node::Variable(_) | Node::Parameter(_) => Err(syn::Error::new(
            Span::call_site(),
            "row_atom internal schedule error: a leaf node has no temporary definition",
        )),
    }
}

fn schedule_definitions(
    roots: impl IntoIterator<Item = usize>,
    graph: &Graph,
    primaries: &[Ident],
    constants: &[Ident],
) -> Result<Vec<TokenStream2>> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for root in roots {
        topological_order(root, graph, &mut seen, &mut order);
    }
    order
        .into_iter()
        .map(|id| -> Result<TokenStream2> {
            let temporary = format_ident!("__row_atom_{id}");
            let expression = node_definition(id, graph, primaries, constants)?;
            Ok(quote!(let #temporary: f64 = #expression;))
        })
        .collect()
}

fn expand(input: RowAtomInput) -> Result<TokenStream2> {
    let RowAtomInput {
        visibility,
        name,
        lowerings,
        primaries,
        constants,
        expression,
    } = input;
    let mut graph = Graph::new();
    let value = graph_expression(&expression, &primaries, &constants, &mut graph)?;
    let dimension = primaries.len();
    let mut gradient = Vec::with_capacity(dimension);
    for axis in 0..dimension {
        gradient.push(graph.derivative(value, axis));
    }
    let mut hessian = vec![vec![0usize; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            hessian[row][column] = graph.derivative(gradient[row], column);
        }
    }
    let mut output = Vec::new();

    if lowerings.contains(&Lowering::Generic) {
        let generic_expression = jet_expression(&expression, &primaries, &constants)?;
        output.push(quote! {
            #[inline(always)]
            #visibility fn #name<const K: usize, S: ::gam_math::jet_scalar::JetScalar<K>>(
                #(#primaries: &S,)*
                #(#constants: f64),*
            ) -> S {
                #generic_expression
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Order2, "order2", false),
        (Lowering::Order2AtZero, "order2_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let order2_name = format_ident!("{name}_{suffix}");
        let (value, gradient, hessian) = if at_zero {
            let mut memo = HashMap::new();
            let value = graph.substitute_zero_primaries(value, &mut memo);
            let value = graph.normalize_polynomial(value, constants.len());
            let gradient = gradient
                .iter()
                .map(|&id| {
                    let id = graph.substitute_zero_primaries(id, &mut memo);
                    graph.normalize_polynomial(id, constants.len())
                })
                .collect::<Vec<_>>();
            let hessian = hessian
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&id| {
                            let id = graph.substitute_zero_primaries(id, &mut memo);
                            graph.normalize_polynomial(id, constants.len())
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (value, gradient, hessian)
        } else {
            (value, gradient.clone(), hessian.clone())
        };
        let mut packed_hessian = Vec::with_capacity(dimension * (dimension + 1) / 2);
        for (row, channels) in hessian.iter().enumerate() {
            packed_hessian.extend_from_slice(&channels[row..]);
        }
        let definitions = schedule_definitions(
            std::iter::once(value)
                .chain(gradient.iter().copied())
                .chain(packed_hessian.iter().copied()),
            &graph,
            &primaries,
            &constants,
        )?;
        let value_ref = node_reference(value, &graph, &primaries, &constants);
        let gradient_refs = gradient
            .iter()
            .map(|&id| node_reference(id, &graph, &primaries, &constants));
        let hessian_refs = packed_hessian
            .iter()
            .map(|&id| node_reference(id, &graph, &primaries, &constants));
        let packed = dimension * (dimension + 1) / 2;
        let gradient_bits = gradient
            .iter()
            .enumerate()
            .fold(0u128, |bits, (axis, &id)| {
                bits | (u128::from(!graph.is_zero(id)) << axis)
            });
        let hessian_bits = packed_hessian
            .iter()
            .enumerate()
            .fold(0u128, |bits, (slot, &id)| {
                bits | (u128::from(!graph.is_zero(id)) << slot)
            });
        let gradient_bits = Literal::u128_unsuffixed(gradient_bits);
        let hessian_bits = Literal::u128_unsuffixed(hessian_bits);
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #order2_name(
                #primary_parameters
                #(#constants: f64),*
            ) -> ::gam_math::jet_scalar::StaticOrder2Atom<
                #dimension,
                #packed,
                #gradient_bits,
                #hessian_bits,
            > {
                #(#definitions)*
                ::gam_math::jet_scalar::StaticOrder2Atom::new(
                    #value_ref,
                    [#(#gradient_refs),*],
                    [#(#hessian_refs),*],
                )
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Third, "third_contracted", false),
        (Lowering::ThirdAtZero, "third_contracted_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let third_name = format_ident!("{name}_{suffix}");
        let mut roots = Vec::new();
        let mut assignments = Vec::new();
        let mut memo = HashMap::new();
        for row in 0..dimension {
            for column in row..dimension {
                let mut derivatives = (0..dimension)
                    .map(|axis| graph.derivative(hessian[row][column], axis))
                    .collect::<Vec<_>>();
                if at_zero {
                    for derivative in &mut derivatives {
                        *derivative = graph.substitute_zero_primaries(*derivative, &mut memo);
                        *derivative = graph.normalize_polynomial(*derivative, constants.len());
                    }
                }
                roots.extend(derivatives.iter().copied());
                let terms = derivatives
                    .iter()
                    .enumerate()
                    .filter(|(_, id)| !graph.is_zero(**id))
                    .map(|(axis, &id)| {
                        let derivative = node_reference(id, &graph, &primaries, &constants);
                        quote!(#derivative * direction[#axis])
                    });
                let temporary = format_ident!("__row_atom_third_{row}_{column}");
                assignments.push(if row == column {
                    quote! {
                        let #temporary = 0.0 #(+ #terms)*;
                        out[#row][#column] = #temporary;
                    }
                } else {
                    quote! {
                        let #temporary = 0.0 #(+ #terms)*;
                        out[#row][#column] = #temporary;
                        out[#column][#row] = #temporary;
                    }
                });
            }
        }
        let definitions = schedule_definitions(roots, &graph, &primaries, &constants)?;
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #third_name(
                #primary_parameters
                #(#constants: f64,)*
                direction: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] {
                #(#definitions)*
                let mut out = [[0.0; #dimension]; #dimension];
                #(#assignments)*
                out
            }
        });
    }

    for (lowering, suffix, at_zero) in [
        (Lowering::Fourth, "fourth_contracted", false),
        (Lowering::FourthAtZero, "fourth_contracted_at_zero", true),
    ] {
        if !lowerings.contains(&lowering) {
            continue;
        }
        let fourth_name = format_ident!("{name}_{suffix}");
        let mut roots = Vec::new();
        let mut assignments = Vec::new();
        let mut memo = HashMap::new();
        for row in 0..dimension {
            for column in row..dimension {
                let third = (0..dimension)
                    .map(|axis| graph.derivative(hessian[row][column], axis))
                    .collect::<Vec<_>>();
                let mut fourth = third
                    .iter()
                    .map(|&id| {
                        (0..dimension)
                            .map(|axis| graph.derivative(id, axis))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                if at_zero {
                    for derivative in fourth.iter_mut().flatten() {
                        *derivative = graph.substitute_zero_primaries(*derivative, &mut memo);
                        *derivative = graph.normalize_polynomial(*derivative, constants.len());
                    }
                }
                roots.extend(fourth.iter().flatten().copied());
                let terms = fourth.iter().enumerate().flat_map(|(axis_u, derivatives)| {
                    derivatives
                        .iter()
                        .enumerate()
                        .filter(|(_, id)| !graph.is_zero(**id))
                        .map(|(axis_v, &id)| {
                            let derivative = node_reference(id, &graph, &primaries, &constants);
                            quote!(#derivative * direction_u[#axis_u] * direction_v[#axis_v])
                        })
                        .collect::<Vec<_>>()
                });
                let temporary = format_ident!("__row_atom_fourth_{row}_{column}");
                assignments.push(if row == column {
                    quote! {
                        let #temporary = 0.0 #(+ #terms)*;
                        out[#row][#column] = #temporary;
                    }
                } else {
                    quote! {
                        let #temporary = 0.0 #(+ #terms)*;
                        out[#row][#column] = #temporary;
                        out[#column][#row] = #temporary;
                    }
                });
            }
        }
        let definitions = schedule_definitions(roots, &graph, &primaries, &constants)?;
        let primary_parameters = if at_zero {
            quote!()
        } else {
            quote!(#(#primaries: f64,)*)
        };
        output.push(quote! {
            #[inline(always)]
            #visibility fn #fourth_name(
                #primary_parameters
                #(#constants: f64,)*
                direction_u: &[f64; #dimension],
                direction_v: &[f64; #dimension],
            ) -> [[f64; #dimension]; #dimension] {
                #(#definitions)*
                let mut out = [[0.0; #dimension]; #dimension];
                #(#assignments)*
                out
            }
        });
    }

    Ok(quote!(#(#output)*))
}

/// Define one row atom and emit exactly its requested build-time lowerings.
///
/// ```text
/// row_atom! {
///     pub(crate) fn row [generic, order2, third, fourth](
///         eta, deriv;
///         weight, event
///     ) {
///         weight * (exp(eta) - event * (eta + ln(deriv)))
///     }
/// }
/// ```
///
/// A normalized local-coordinate atom whose production expansion point is
/// identically zero can instead request the `_at_zero` lowerings. This is
/// exact partial evaluation, not a separate derivative expression.
#[proc_macro]
pub fn row_atom(input: TokenStream) -> TokenStream {
    match expand(parse_macro_input!(input as RowAtomInput)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// Define one backend-neutral row program and emit its generic `JetScalar`
/// evaluator plus symbolically sparse order-2 Rust and CUDA functions. The
/// declaration owns the complete algebraic schedule; stable unary primitives
/// are explicit leaves mapped to one Rust derivative-stack builder and one CUDA
/// stack function. Both direct backends consume the same symbolic SSA lowering,
/// compute each nonzero gradient and packed Hessian component once, and scatter
/// Hessian symmetry only at the output seam.
#[proc_macro]
pub fn row_program(input: TokenStream) -> TokenStream {
    match row_program::expand(parse_macro_input!(input as row_program::Input)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
