//! Pratt parser for operator precedence.

use crate::span_error::{ParseError, ParseResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 4. Pratt Parser for Operator Precedence
// ---------------------------------------------------------------------------

/// Associativity for binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assoc {
    Left,
    Right,
}

/// Expression AST node for the Pratt parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Ident(String),
    Prefix {
        op: String,
        operand: Box<Self>,
    },
    Binary {
        op: String,
        left: Box<Self>,
        right: Box<Self>,
    },
    Postfix {
        op: String,
        operand: Box<Self>,
    },
    Group(Box<Self>),
    Call {
        func: Box<Self>,
        args: Vec<Self>,
    },
}

impl Expr {
    /// Evaluate a numeric expression (no variables).
    ///
    /// # Errors
    ///
    /// Returns an error string on unknown identifiers or operators.
    pub fn eval(&self) -> Result<f64, String> {
        match self {
            Self::Number(n) => Ok(*n),
            Self::Ident(name) => Err(format!("unknown variable: {name}")),
            Self::Prefix { op, operand } => {
                let val = operand.eval()?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "+" => Ok(val),
                    "!" => Ok(if val == 0.0 { 1.0 } else { 0.0 }),
                    _ => Err(format!("unknown prefix operator: {op}")),
                }
            }
            Self::Binary { op, left, right } => {
                let l = left.eval()?;
                let r = right.eval()?;
                match op.as_str() {
                    "+" => Ok(l + r),
                    "-" => Ok(l - r),
                    "*" => Ok(l * r),
                    "/" => {
                        if r == 0.0 {
                            Err("division by zero".to_string())
                        } else {
                            Ok(l / r)
                        }
                    }
                    "%" => Ok(l % r),
                    "^" | "**" => Ok(l.powf(r)),
                    _ => Err(format!("unknown binary operator: {op}")),
                }
            }
            Self::Postfix { op, .. } => Err(format!("cannot evaluate postfix: {op}")),
            Self::Group(inner) => inner.eval(),
            Self::Call { func, .. } => {
                if let Self::Ident(name) = func.as_ref() {
                    Err(format!("cannot evaluate function call: {name}"))
                } else {
                    Err("cannot evaluate call expression".to_string())
                }
            }
        }
    }
}

/// Pratt parser for operator-precedence expression parsing.
pub struct PrattParser {
    prefix: HashMap<String, u32>,
    postfix: HashMap<String, u32>,
    infix: HashMap<String, (u32, Assoc)>,
}

impl Default for PrattParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PrattParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefix: HashMap::new(),
            postfix: HashMap::new(),
            infix: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_prefix(mut self, op: &str, bp: u32) -> Self {
        self.prefix.insert(op.to_string(), bp);
        self
    }

    #[must_use]
    pub fn with_postfix(mut self, op: &str, bp: u32) -> Self {
        self.postfix.insert(op.to_string(), bp);
        self
    }

    #[must_use]
    pub fn with_infix(mut self, op: &str, bp: u32, assoc: Assoc) -> Self {
        self.infix.insert(op.to_string(), (bp, assoc));
        self
    }

    /// Create a standard arithmetic Pratt parser.
    #[must_use]
    pub fn arithmetic() -> Self {
        Self::new()
            .with_prefix("-", 90)
            .with_prefix("+", 90)
            .with_infix("+", 10, Assoc::Left)
            .with_infix("-", 10, Assoc::Left)
            .with_infix("*", 20, Assoc::Left)
            .with_infix("/", 20, Assoc::Left)
            .with_infix("%", 20, Assoc::Left)
            .with_infix("^", 30, Assoc::Right)
            .with_infix("**", 30, Assoc::Right)
    }

    /// Parse an expression from a token stream.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` on invalid expressions.
    pub fn parse_expr(&self, input: &str) -> Result<Expr, ParseError> {
        let mut pos = 0;
        let (expr, _) = self.parse_bp(input, &mut pos, 0)?;
        Ok(expr)
    }

    fn skip_ws<'a>(input: &'a str, pos: &mut usize) -> &'a str {
        while *pos < input.len() && input.as_bytes()[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        &input[*pos..]
    }

    fn parse_bp(&self, input: &str, pos: &mut usize, min_bp: u32) -> ParseResult<Expr> {
        let mut lhs = self.parse_atom(input, pos)?;

        loop {
            Self::skip_ws(input, pos);
            if *pos >= input.len() {
                break;
            }

            // Try postfix
            if let Some((op, bp)) = self.try_postfix(input, *pos) {
                if bp < min_bp {
                    break;
                }
                *pos += op.len();
                lhs = (
                    Expr::Postfix {
                        op,
                        operand: Box::new(lhs.0),
                    },
                    *pos,
                );
                continue;
            }

            // Try call syntax: (
            if *pos < input.len() && input.as_bytes()[*pos] == b'(' {
                if min_bp > 0 {
                    // Only parse call at top level or when binding power allows
                }
                *pos += 1;
                let mut args = Vec::new();
                Self::skip_ws(input, pos);
                if *pos < input.len() && input.as_bytes()[*pos] != b')' {
                    let (arg, _) = self.parse_bp(input, pos, 0)?;
                    args.push(arg);
                    Self::skip_ws(input, pos);
                    while *pos < input.len() && input.as_bytes()[*pos] == b',' {
                        *pos += 1;
                        let (arg, _) = self.parse_bp(input, pos, 0)?;
                        args.push(arg);
                        Self::skip_ws(input, pos);
                    }
                }
                if *pos < input.len() && input.as_bytes()[*pos] == b')' {
                    *pos += 1;
                } else {
                    return Err(ParseError::with_expected(*pos, "')'"));
                }
                lhs = (
                    Expr::Call {
                        func: Box::new(lhs.0),
                        args,
                    },
                    *pos,
                );
                continue;
            }

            // Try infix
            let Some((op, bp, assoc)) = self.try_infix(input, *pos) else {
                break;
            };
            if bp < min_bp {
                break;
            }
            *pos += op.len();

            let next_bp = match assoc {
                Assoc::Left => bp + 1,
                Assoc::Right => bp,
            };
            let (rhs, _) = self.parse_bp(input, pos, next_bp)?;
            lhs = (
                Expr::Binary {
                    op,
                    left: Box::new(lhs.0),
                    right: Box::new(rhs),
                },
                *pos,
            );
        }

        Ok(lhs)
    }

    fn parse_atom(&self, input: &str, pos: &mut usize) -> ParseResult<Expr> {
        Self::skip_ws(input, pos);

        if *pos >= input.len() {
            return Err(ParseError::new(*pos, "unexpected end of input"));
        }

        let rest = &input[*pos..];

        // Grouped expression
        if rest.starts_with('(') {
            *pos += 1;
            let (inner, _) = self.parse_bp(input, pos, 0)?;
            Self::skip_ws(input, pos);
            if *pos < input.len() && input.as_bytes()[*pos] == b')' {
                *pos += 1;
                return Ok((Expr::Group(Box::new(inner)), *pos));
            }
            return Err(ParseError::with_expected(*pos, "')'"));
        }

        // Prefix operator
        if let Some((op, bp)) = self.try_prefix(input, *pos) {
            *pos += op.len();
            let (operand, _) = self.parse_bp(input, pos, bp)?;
            return Ok((
                Expr::Prefix {
                    op,
                    operand: Box::new(operand),
                },
                *pos,
            ));
        }

        // Number
        if rest.as_bytes()[0].is_ascii_digit() {
            let start = *pos;
            while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                *pos += 1;
            }
            if *pos < input.len() && input.as_bytes()[*pos] == b'.' {
                *pos += 1;
                while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                    *pos += 1;
                }
            }
            let num: f64 = input[start..*pos]
                .parse()
                .map_err(|_| ParseError::new(start, "invalid number"))?;
            return Ok((Expr::Number(num), *pos));
        }

        // Identifier
        if rest.as_bytes()[0].is_ascii_alphabetic() || rest.as_bytes()[0] == b'_' {
            let start = *pos;
            while *pos < input.len()
                && (input.as_bytes()[*pos].is_ascii_alphanumeric()
                    || input.as_bytes()[*pos] == b'_')
            {
                *pos += 1;
            }
            return Ok((Expr::Ident(input[start..*pos].to_string()), *pos));
        }

        Err(ParseError::new(
            *pos,
            format!(
                "unexpected character: {:?}",
                rest.chars().next().unwrap_or(' ')
            ),
        ))
    }

    fn try_prefix(&self, input: &str, pos: usize) -> Option<(String, u32)> {
        let rest = &input[pos..];
        let mut best: Option<(String, u32)> = None;
        for (op, bp) in &self.prefix {
            if rest.starts_with(op.as_str())
                && best.as_ref().is_none_or(|(b, _)| op.len() > b.len())
            {
                best = Some((op.clone(), *bp));
            }
        }
        best
    }

    fn try_infix(&self, input: &str, pos: usize) -> Option<(String, u32, Assoc)> {
        let rest = &input[pos..];
        let mut best: Option<(String, u32, Assoc)> = None;
        for (op, (bp, assoc)) in &self.infix {
            if rest.starts_with(op.as_str())
                && best.as_ref().is_none_or(|(b, _, _)| op.len() > b.len())
            {
                best = Some((op.clone(), *bp, *assoc));
            }
        }
        best
    }

    fn try_postfix(&self, input: &str, pos: usize) -> Option<(String, u32)> {
        let rest = &input[pos..];
        let mut best: Option<(String, u32)> = None;
        for (op, bp) in &self.postfix {
            if rest.starts_with(op.as_str())
                && best.as_ref().is_none_or(|(b, _)| op.len() > b.len())
            {
                best = Some((op.clone(), *bp));
            }
        }
        best
    }
}
