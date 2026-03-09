#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

//! ALICE-Parser: Parser combinator library with PEG, Pratt parsing,
//! recursive descent, tokenizer/lexer, and error recovery.

use core::fmt;
use std::collections::HashMap;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// 1. Span & ParseError
// ---------------------------------------------------------------------------

/// A byte-offset span in the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    #[must_use]
    pub const fn merge(self, other: Self) -> Self {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Self { start, end }
    }
}

/// A parse error with position and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub position: usize,
    pub message: String,
    pub expected: Vec<String>,
}

impl ParseError {
    #[must_use]
    pub fn new(position: usize, message: impl Into<String>) -> Self {
        Self {
            position,
            message: message.into(),
            expected: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_expected(position: usize, expected: impl Into<String>) -> Self {
        Self {
            position,
            message: String::new(),
            expected: vec![expected.into()],
        }
    }

    #[must_use]
    pub fn merge(mut self, other: Self) -> Self {
        if other.position > self.position {
            return other;
        }
        if other.position == self.position {
            self.expected.extend(other.expected);
            if self.message.is_empty() {
                self.message = other.message;
            }
        }
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at position {}", self.position)?;
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        if !self.expected.is_empty() {
            write!(f, " (expected: {})", self.expected.join(", "))?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// Parse result type.
pub type ParseResult<T> = Result<(T, usize), ParseError>;

// ---------------------------------------------------------------------------
// 2. Tokenizer / Lexer
// ---------------------------------------------------------------------------

/// Token kind produced by the lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Integer(i64),
    Float(String),
    Ident(String),
    StringLit(String),
    Punct(char),
    Operator(String),
    Keyword(String),
    Whitespace,
    Newline,
    Eof,
}

/// A token with its kind and source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Configurable lexer.
pub struct Lexer {
    keywords: Vec<String>,
    operators: Vec<String>,
    skip_whitespace: bool,
}

impl Default for Lexer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lexer {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keywords: Vec::new(),
            operators: Vec::new(),
            skip_whitespace: true,
        }
    }

    #[must_use]
    pub fn with_keywords(mut self, kws: &[&str]) -> Self {
        self.keywords = kws.iter().map(|s| (*s).to_string()).collect();
        self
    }

    #[must_use]
    pub fn with_operators(mut self, ops: &[&str]) -> Self {
        let mut ops: Vec<String> = ops.iter().map(|s| (*s).to_string()).collect();
        ops.sort_by_key(|b| std::cmp::Reverse(b.len()));
        self.operators = ops;
        self
    }

    #[must_use]
    pub const fn with_skip_whitespace(mut self, skip: bool) -> Self {
        self.skip_whitespace = skip;
        self
    }

    /// Tokenize an entire input string.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` on unexpected characters.
    pub fn tokenize(&self, input: &str) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        let mut pos = 0;
        let bytes = input.as_bytes();

        while pos < bytes.len() {
            let b = bytes[pos];

            // Newline
            if b == b'\n' {
                if !self.skip_whitespace {
                    tokens.push(Token {
                        kind: TokenKind::Newline,
                        span: Span::new(pos, pos + 1),
                    });
                }
                pos += 1;
                continue;
            }

            // Whitespace
            if b.is_ascii_whitespace() {
                let start = pos;
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() && bytes[pos] != b'\n' {
                    pos += 1;
                }
                if !self.skip_whitespace {
                    tokens.push(Token {
                        kind: TokenKind::Whitespace,
                        span: Span::new(start, pos),
                    });
                }
                continue;
            }

            // String literal
            if b == b'"' || b == b'\'' {
                let (tok, new_pos) = Self::lex_string(input, pos)?;
                tokens.push(tok);
                pos = new_pos;
                continue;
            }

            // Number
            if b.is_ascii_digit() {
                let (tok, new_pos) = Self::lex_number(input, pos);
                tokens.push(tok);
                pos = new_pos;
                continue;
            }

            // Identifier / keyword
            if b.is_ascii_alphabetic() || b == b'_' {
                let (tok, new_pos) = self.lex_ident(input, pos);
                tokens.push(tok);
                pos = new_pos;
                continue;
            }

            // Multi-char operators
            if let Some((op, new_pos)) = self.try_lex_operator(input, pos) {
                tokens.push(Token {
                    kind: TokenKind::Operator(op),
                    span: Span::new(pos, new_pos),
                });
                pos = new_pos;
                continue;
            }

            // Single punctuation
            if b.is_ascii_punctuation() {
                tokens.push(Token {
                    kind: TokenKind::Punct(b as char),
                    span: Span::new(pos, pos + 1),
                });
                pos += 1;
                continue;
            }

            return Err(ParseError::new(
                pos,
                format!("unexpected character: {:?}", b as char),
            ));
        }

        tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span::new(pos, pos),
        });
        Ok(tokens)
    }

    fn lex_string(input: &str, start: usize) -> Result<(Token, usize), ParseError> {
        let bytes = input.as_bytes();
        let quote = bytes[start];
        let mut pos = start + 1;
        let mut value = String::new();

        while pos < bytes.len() {
            if bytes[pos] == b'\\' && pos + 1 < bytes.len() {
                let escaped = match bytes[pos + 1] {
                    b'n' => '\n',
                    b't' => '\t',
                    b'\\' => '\\',
                    b'"' => '"',
                    b'\'' => '\'',
                    other => other as char,
                };
                value.push(escaped);
                pos += 2;
            } else if bytes[pos] == quote {
                pos += 1;
                return Ok((
                    Token {
                        kind: TokenKind::StringLit(value),
                        span: Span::new(start, pos),
                    },
                    pos,
                ));
            } else {
                value.push(bytes[pos] as char);
                pos += 1;
            }
        }

        Err(ParseError::new(start, "unterminated string literal"))
    }

    fn lex_number(input: &str, start: usize) -> (Token, usize) {
        let bytes = input.as_bytes();
        let mut pos = start;
        let mut is_float = false;

        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }

        if pos < bytes.len() && bytes[pos] == b'.' {
            let next = if pos + 1 < bytes.len() {
                bytes[pos + 1]
            } else {
                0
            };
            if next.is_ascii_digit() {
                is_float = true;
                pos += 1;
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
        }

        let text = &input[start..pos];
        let kind = if is_float {
            TokenKind::Float(text.to_string())
        } else {
            TokenKind::Integer(text.parse::<i64>().unwrap_or(0))
        };

        (
            Token {
                kind,
                span: Span::new(start, pos),
            },
            pos,
        )
    }

    fn lex_ident(&self, input: &str, start: usize) -> (Token, usize) {
        let bytes = input.as_bytes();
        let mut pos = start;
        while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
            pos += 1;
        }
        let word = &input[start..pos];
        let kind = if self.keywords.iter().any(|k| k == word) {
            TokenKind::Keyword(word.to_string())
        } else {
            TokenKind::Ident(word.to_string())
        };
        (
            Token {
                kind,
                span: Span::new(start, pos),
            },
            pos,
        )
    }

    fn try_lex_operator(&self, input: &str, pos: usize) -> Option<(String, usize)> {
        for op in &self.operators {
            if input[pos..].starts_with(op.as_str()) {
                return Some((op.clone(), pos + op.len()));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// 3. PEG Combinator Core
// ---------------------------------------------------------------------------

/// A PEG parser that operates on a string slice.
/// Returns `(value, new_position)` on success.
pub trait Parser<T> {
    /// Parse starting at position `pos` in `input`.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` when the input does not match.
    fn parse(&self, input: &str, pos: usize) -> ParseResult<T>;
}

// Function-based parser wrapper
impl<T, F> Parser<T> for F
where
    F: Fn(&str, usize) -> ParseResult<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<T> {
        self(input, pos)
    }
}

// --- Literal ---

/// Match an exact string literal.
pub struct Literal {
    text: String,
}

impl Literal {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl Parser<String> for Literal {
    fn parse(&self, input: &str, pos: usize) -> ParseResult<String> {
        if input[pos..].starts_with(&self.text) {
            Ok((self.text.clone(), pos + self.text.len()))
        } else {
            Err(ParseError::with_expected(pos, format!("{:?}", self.text)))
        }
    }
}

/// Convenience function for literal matching.
#[must_use]
pub fn literal(text: &str) -> Literal {
    Literal::new(text)
}

// --- Regex-like character class ---

/// Match a single character satisfying a predicate.
pub struct CharPred<F: Fn(char) -> bool> {
    pred: F,
    label: String,
}

impl<F: Fn(char) -> bool> Parser<char> for CharPred<F> {
    fn parse(&self, input: &str, pos: usize) -> ParseResult<char> {
        if let Some(ch) = input[pos..].chars().next() {
            if (self.pred)(ch) {
                return Ok((ch, pos + ch.len_utf8()));
            }
        }
        Err(ParseError::with_expected(pos, &self.label))
    }
}

/// Match a single character by predicate.
#[must_use]
pub fn char_pred(
    pred: impl Fn(char) -> bool + 'static,
    label: &str,
) -> CharPred<impl Fn(char) -> bool> {
    CharPred {
        pred,
        label: label.to_string(),
    }
}

/// Match any single character.
#[must_use]
pub fn any_char() -> CharPred<fn(char) -> bool> {
    CharPred {
        pred: |_| true,
        label: "any character".to_string(),
    }
}

/// Match a specific single character.
#[must_use]
pub fn char_exact(expected: char) -> CharPred<impl Fn(char) -> bool> {
    CharPred {
        pred: move |c| c == expected,
        label: format!("{expected:?}"),
    }
}

// --- Sequence ---

/// Sequence two parsers: (A, B).
pub struct Seq<A, B> {
    first: A,
    second: B,
}

impl<A, B, T1, T2> Parser<(T1, T2)> for Seq<A, B>
where
    A: Parser<T1>,
    B: Parser<T2>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<(T1, T2)> {
        let (v1, pos) = self.first.parse(input, pos)?;
        let (v2, pos) = self.second.parse(input, pos)?;
        Ok(((v1, v2), pos))
    }
}

/// Sequence two parsers.
pub const fn seq<A, B, T1, T2>(first: A, second: B) -> Seq<A, B>
where
    A: Parser<T1>,
    B: Parser<T2>,
{
    Seq { first, second }
}

// --- Ordered Choice ---

/// PEG ordered choice: try first, then second.
pub struct Choice<A, B> {
    first: A,
    second: B,
}

impl<A, B, T> Parser<T> for Choice<A, B>
where
    A: Parser<T>,
    B: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<T> {
        match self.first.parse(input, pos) {
            Ok(r) => Ok(r),
            Err(e1) => match self.second.parse(input, pos) {
                Ok(r) => Ok(r),
                Err(e2) => Err(e1.merge(e2)),
            },
        }
    }
}

/// Ordered choice between two parsers.
pub const fn choice<A, B, T>(first: A, second: B) -> Choice<A, B>
where
    A: Parser<T>,
    B: Parser<T>,
{
    Choice { first, second }
}

// --- Repetition (zero-or-more, one-or-more) ---

/// Zero or more repetitions.
pub struct Many<P> {
    parser: P,
}

impl<P, T> Parser<Vec<T>> for Many<P>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, mut pos: usize) -> ParseResult<Vec<T>> {
        let mut results = Vec::new();
        while let Ok((val, new_pos)) = self.parser.parse(input, pos) {
            if new_pos == pos {
                break; // prevent infinite loop on zero-width match
            }
            results.push(val);
            pos = new_pos;
        }
        Ok((results, pos))
    }
}

/// Zero or more repetitions.
pub const fn many<P, T>(parser: P) -> Many<P>
where
    P: Parser<T>,
{
    Many { parser }
}

/// One or more repetitions.
pub struct Many1<P> {
    parser: P,
}

impl<P, T> Parser<Vec<T>> for Many1<P>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<Vec<T>> {
        let (first, mut pos) = self.parser.parse(input, pos)?;
        let mut results = vec![first];
        while let Ok((val, new_pos)) = self.parser.parse(input, pos) {
            if new_pos == pos {
                break;
            }
            results.push(val);
            pos = new_pos;
        }
        Ok((results, pos))
    }
}

/// One or more repetitions.
pub const fn many1<P, T>(parser: P) -> Many1<P>
where
    P: Parser<T>,
{
    Many1 { parser }
}

// --- Optional ---

/// Optional parser.
pub struct Optional<P> {
    parser: P,
}

impl<P, T> Parser<Option<T>> for Optional<P>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<Option<T>> {
        match self.parser.parse(input, pos) {
            Ok((val, pos)) => Ok((Some(val), pos)),
            Err(_) => Ok((None, pos)),
        }
    }
}

/// Optional parser.
pub const fn optional<P, T>(parser: P) -> Optional<P>
where
    P: Parser<T>,
{
    Optional { parser }
}

// --- Lookahead (PEG & / !) ---

/// Positive lookahead: succeeds without consuming input.
pub struct AndPred<P, T> {
    parser: P,
    _phantom: PhantomData<T>,
}

impl<P, T> Parser<()> for AndPred<P, T>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<()> {
        self.parser.parse(input, pos)?;
        Ok(((), pos))
    }
}

/// Positive lookahead.
pub const fn and_pred<P, T>(parser: P) -> AndPred<P, T>
where
    P: Parser<T>,
{
    AndPred {
        parser,
        _phantom: PhantomData,
    }
}

/// Negative lookahead: succeeds if inner parser fails.
pub struct NotPred<P, T> {
    parser: P,
    _phantom: PhantomData<T>,
}

impl<P, T> Parser<()> for NotPred<P, T>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<()> {
        match self.parser.parse(input, pos) {
            Ok(_) => Err(ParseError::new(pos, "unexpected match")),
            Err(_) => Ok(((), pos)),
        }
    }
}

/// Negative lookahead.
pub const fn not_pred<P, T>(parser: P) -> NotPred<P, T>
where
    P: Parser<T>,
{
    NotPred {
        parser,
        _phantom: PhantomData,
    }
}

// --- Map ---

/// Transform parser output.
pub struct Map<P, F, T> {
    parser: P,
    func: F,
    _phantom: PhantomData<T>,
}

impl<P, F, T, U> Parser<U> for Map<P, F, T>
where
    P: Parser<T>,
    F: Fn(T) -> U,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<U> {
        let (val, pos) = self.parser.parse(input, pos)?;
        Ok(((self.func)(val), pos))
    }
}

/// Map over parser output.
pub const fn map<P, F, T, U>(parser: P, func: F) -> Map<P, F, T>
where
    P: Parser<T>,
    F: Fn(T) -> U,
{
    Map {
        parser,
        func,
        _phantom: PhantomData,
    }
}

// --- Skip whitespace ---

/// Skip ASCII whitespace.
pub struct SkipWs<P> {
    parser: P,
}

impl<P, T> Parser<T> for SkipWs<P>
where
    P: Parser<T>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<T> {
        let mut p = pos;
        while p < input.len() && input.as_bytes()[p].is_ascii_whitespace() {
            p += 1;
        }
        self.parser.parse(input, p)
    }
}

/// Skip leading whitespace then parse.
pub const fn skip_ws<P, T>(parser: P) -> SkipWs<P>
where
    P: Parser<T>,
{
    SkipWs { parser }
}

// --- Separated list ---

/// Parse items separated by a delimiter.
pub struct SepBy<P, S, U> {
    parser: P,
    separator: S,
    _phantom: PhantomData<U>,
}

impl<P, S, T, U> Parser<Vec<T>> for SepBy<P, S, U>
where
    P: Parser<T>,
    S: Parser<U>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<Vec<T>> {
        let mut results = Vec::new();
        let Ok((first, mut pos)) = self.parser.parse(input, pos) else {
            return Ok((results, pos));
        };
        results.push(first);

        loop {
            let Ok((_, sep_pos)) = self.separator.parse(input, pos) else {
                break;
            };
            let Ok((val, new_pos)) = self.parser.parse(input, sep_pos) else {
                break;
            };
            results.push(val);
            pos = new_pos;
        }
        Ok((results, pos))
    }
}

/// Parse items separated by a delimiter.
pub const fn sep_by<P, S, T, U>(parser: P, separator: S) -> SepBy<P, S, U>
where
    P: Parser<T>,
    S: Parser<U>,
{
    SepBy {
        parser,
        separator,
        _phantom: PhantomData,
    }
}

// --- Between ---

/// Parse content between open and close delimiters.
pub struct Between<O, P, C, U, V> {
    open: O,
    parser: P,
    close: C,
    _phantom: PhantomData<(U, V)>,
}

impl<O, P, C, T, U, V> Parser<T> for Between<O, P, C, U, V>
where
    O: Parser<U>,
    P: Parser<T>,
    C: Parser<V>,
{
    fn parse(&self, input: &str, pos: usize) -> ParseResult<T> {
        let (_, pos) = self.open.parse(input, pos)?;
        let (val, pos) = self.parser.parse(input, pos)?;
        let (_, pos) = self.close.parse(input, pos)?;
        Ok((val, pos))
    }
}

/// Parse content between open and close delimiters.
pub const fn between<O, P, C, T, U, V>(open: O, parser: P, close: C) -> Between<O, P, C, U, V>
where
    O: Parser<U>,
    P: Parser<T>,
    C: Parser<V>,
{
    Between {
        open,
        parser,
        close,
        _phantom: PhantomData,
    }
}

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

// ---------------------------------------------------------------------------
// 5. Recursive Descent Parser (JSON as example)
// ---------------------------------------------------------------------------

/// JSON-like value for the recursive descent parser demo.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<Self>),
    Object(Vec<(String, Self)>),
}

/// Recursive descent JSON parser.
pub struct JsonParser;

impl JsonParser {
    /// Parse a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` on invalid JSON.
    pub fn parse(input: &str) -> Result<JsonValue, ParseError> {
        let mut pos = 0;
        let val = Self::parse_value(input, &mut pos)?;
        Self::skip_ws(input, &mut pos);
        if pos < input.len() {
            return Err(ParseError::new(pos, "trailing content after JSON value"));
        }
        Ok(val)
    }

    fn skip_ws(input: &str, pos: &mut usize) {
        while *pos < input.len() && input.as_bytes()[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
    }

    fn parse_value(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        Self::skip_ws(input, pos);
        if *pos >= input.len() {
            return Err(ParseError::new(*pos, "unexpected end of input"));
        }

        match input.as_bytes()[*pos] {
            b'n' => Self::parse_null(input, pos),
            b't' | b'f' => Self::parse_bool(input, pos),
            b'"' => Self::parse_string(input, pos).map(JsonValue::Str),
            b'[' => Self::parse_array(input, pos),
            b'{' => Self::parse_object(input, pos),
            b'-' | b'0'..=b'9' => Self::parse_number(input, pos),
            ch => Err(ParseError::new(
                *pos,
                format!("unexpected character: {:?}", ch as char),
            )),
        }
    }

    fn parse_null(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        if input[*pos..].starts_with("null") {
            *pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(ParseError::with_expected(*pos, "null"))
        }
    }

    fn parse_bool(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        if input[*pos..].starts_with("true") {
            *pos += 4;
            Ok(JsonValue::Bool(true))
        } else if input[*pos..].starts_with("false") {
            *pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(ParseError::with_expected(*pos, "true or false"))
        }
    }

    fn parse_number(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        let start = *pos;
        if *pos < input.len() && input.as_bytes()[*pos] == b'-' {
            *pos += 1;
        }
        if *pos >= input.len() || !input.as_bytes()[*pos].is_ascii_digit() {
            return Err(ParseError::new(start, "invalid number"));
        }
        while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
            *pos += 1;
        }
        if *pos < input.len() && input.as_bytes()[*pos] == b'.' {
            *pos += 1;
            while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                *pos += 1;
            }
        }
        // Exponent
        if *pos < input.len() && (input.as_bytes()[*pos] == b'e' || input.as_bytes()[*pos] == b'E')
        {
            *pos += 1;
            if *pos < input.len()
                && (input.as_bytes()[*pos] == b'+' || input.as_bytes()[*pos] == b'-')
            {
                *pos += 1;
            }
            while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                *pos += 1;
            }
        }
        let num: f64 = input[start..*pos]
            .parse()
            .map_err(|_| ParseError::new(start, "invalid number"))?;
        Ok(JsonValue::Number(num))
    }

    fn parse_string(input: &str, pos: &mut usize) -> Result<String, ParseError> {
        if *pos >= input.len() || input.as_bytes()[*pos] != b'"' {
            return Err(ParseError::with_expected(*pos, "string"));
        }
        *pos += 1;
        let mut result = String::new();
        while *pos < input.len() {
            let ch = input.as_bytes()[*pos];
            if ch == b'"' {
                *pos += 1;
                return Ok(result);
            }
            if ch == b'\\' {
                *pos += 1;
                if *pos >= input.len() {
                    return Err(ParseError::new(*pos, "unexpected end in escape"));
                }
                let escaped = match input.as_bytes()[*pos] {
                    b'"' => '"',
                    b'\\' => '\\',
                    b'/' => '/',
                    b'n' => '\n',
                    b't' => '\t',
                    b'r' => '\r',
                    other => other as char,
                };
                result.push(escaped);
            } else {
                result.push(ch as char);
            }
            *pos += 1;
        }
        Err(ParseError::new(*pos, "unterminated string"))
    }

    fn parse_array(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        *pos += 1; // skip [
        let mut items = Vec::new();
        Self::skip_ws(input, pos);
        if *pos < input.len() && input.as_bytes()[*pos] == b']' {
            *pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = Self::parse_value(input, pos)?;
            items.push(val);
            Self::skip_ws(input, pos);
            if *pos < input.len() && input.as_bytes()[*pos] == b',' {
                *pos += 1;
            } else {
                break;
            }
        }
        Self::skip_ws(input, pos);
        if *pos < input.len() && input.as_bytes()[*pos] == b']' {
            *pos += 1;
            Ok(JsonValue::Array(items))
        } else {
            Err(ParseError::with_expected(*pos, "']'"))
        }
    }

    fn parse_object(input: &str, pos: &mut usize) -> Result<JsonValue, ParseError> {
        *pos += 1; // skip {
        let mut pairs = Vec::new();
        Self::skip_ws(input, pos);
        if *pos < input.len() && input.as_bytes()[*pos] == b'}' {
            *pos += 1;
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            Self::skip_ws(input, pos);
            let key = Self::parse_string(input, pos)?;
            Self::skip_ws(input, pos);
            if *pos >= input.len() || input.as_bytes()[*pos] != b':' {
                return Err(ParseError::with_expected(*pos, "':'"));
            }
            *pos += 1;
            let val = Self::parse_value(input, pos)?;
            pairs.push((key, val));
            Self::skip_ws(input, pos);
            if *pos < input.len() && input.as_bytes()[*pos] == b',' {
                *pos += 1;
            } else {
                break;
            }
        }
        Self::skip_ws(input, pos);
        if *pos < input.len() && input.as_bytes()[*pos] == b'}' {
            *pos += 1;
            Ok(JsonValue::Object(pairs))
        } else {
            Err(ParseError::with_expected(*pos, "'}'"))
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Error Recovery
// ---------------------------------------------------------------------------

/// Error recovery strategy.
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Skip characters until a synchronization token is found.
    SkipUntil(Vec<char>),
    /// Insert a default value and continue.
    InsertDefault,
}

/// A parser wrapper that performs error recovery.
pub struct Recovering<P> {
    parser: P,
    strategy: RecoveryStrategy,
}

impl<P> Recovering<P> {
    pub const fn new(parser: P, strategy: RecoveryStrategy) -> Self {
        Self { parser, strategy }
    }
}

/// Result of parsing with recovery: either a clean parse or a recovered parse with errors.
#[derive(Debug, Clone)]
pub struct RecoveredParse<T> {
    pub value: T,
    pub errors: Vec<ParseError>,
}

impl Parser<String> for Recovering<Literal> {
    fn parse(&self, input: &str, pos: usize) -> ParseResult<String> {
        match self.parser.parse(input, pos) {
            Ok(result) => Ok(result),
            Err(err) => match &self.strategy {
                RecoveryStrategy::SkipUntil(sync_chars) => {
                    let mut p = pos;
                    while p < input.len() {
                        if let Some(ch) = input[p..].chars().next() {
                            if sync_chars.contains(&ch) {
                                return Ok((String::new(), p));
                            }
                            p += ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    Err(err)
                }
                RecoveryStrategy::InsertDefault => Ok((String::new(), pos)),
            },
        }
    }
}

/// Skip input until one of the sync characters is found.
/// Returns the skipped content and the new position.
#[must_use]
pub fn skip_until(input: &str, pos: usize, sync: &[char]) -> (String, usize) {
    let mut p = pos;
    let mut skipped = String::new();
    while p < input.len() {
        if let Some(ch) = input[p..].chars().next() {
            if sync.contains(&ch) {
                break;
            }
            skipped.push(ch);
            p += ch.len_utf8();
        } else {
            break;
        }
    }
    (skipped, p)
}

/// Parse multiple statements with error recovery.
///
/// # Errors
///
/// Returns `ParseError` if no statements could be parsed at all.
pub fn parse_with_recovery<F>(
    input: &str,
    parse_one: F,
    sync_chars: &[char],
) -> RecoveredParse<Vec<String>>
where
    F: Fn(&str, usize) -> ParseResult<String>,
{
    let mut results = Vec::new();
    let mut errors = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        // Skip whitespace
        while pos < input.len() && input.as_bytes()[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= input.len() {
            break;
        }

        match parse_one(input, pos) {
            Ok((val, new_pos)) => {
                results.push(val);
                pos = new_pos;
            }
            Err(err) => {
                errors.push(err);
                let (_, new_pos) = skip_until(input, pos, sync_chars);
                if new_pos == pos {
                    pos += 1; // avoid infinite loop
                } else {
                    pos = new_pos;
                }
                // Skip the sync char itself
                if pos < input.len() && sync_chars.contains(&(input.as_bytes()[pos] as char)) {
                    pos += 1;
                }
            }
        }
    }

    RecoveredParse {
        value: results,
        errors,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === Span tests ===

    #[test]
    fn span_new() {
        let s = Span::new(3, 7);
        assert_eq!(s.start, 3);
        assert_eq!(s.end, 7);
    }

    #[test]
    fn span_len() {
        assert_eq!(Span::new(0, 5).len(), 5);
    }

    #[test]
    fn span_is_empty() {
        assert!(Span::new(3, 3).is_empty());
        assert!(!Span::new(3, 4).is_empty());
    }

    #[test]
    fn span_merge() {
        let merged = Span::new(2, 5).merge(Span::new(4, 9));
        assert_eq!(merged, Span::new(2, 9));
    }

    // === ParseError tests ===

    #[test]
    fn error_new() {
        let e = ParseError::new(5, "oops");
        assert_eq!(e.position, 5);
        assert_eq!(e.message, "oops");
    }

    #[test]
    fn error_with_expected() {
        let e = ParseError::with_expected(10, "digit");
        assert_eq!(e.expected, vec!["digit".to_string()]);
    }

    #[test]
    fn error_merge_same_pos() {
        let e1 = ParseError::with_expected(5, "a");
        let e2 = ParseError::with_expected(5, "b");
        let merged = e1.merge(e2);
        assert_eq!(merged.expected.len(), 2);
    }

    #[test]
    fn error_merge_further_wins() {
        let e1 = ParseError::new(3, "early");
        let e2 = ParseError::new(7, "late");
        let merged = e1.merge(e2);
        assert_eq!(merged.position, 7);
    }

    #[test]
    fn error_display() {
        let e = ParseError::with_expected(5, "number");
        let s = format!("{e}");
        assert!(s.contains("position 5"));
        assert!(s.contains("number"));
    }

    // === Lexer tests ===

    #[test]
    fn lex_integer() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
    }

    #[test]
    fn lex_float() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("3.14").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Float("3.14".to_string()));
    }

    #[test]
    fn lex_ident() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("foo_bar").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident("foo_bar".to_string()));
    }

    #[test]
    fn lex_keyword() {
        let lexer = Lexer::new().with_keywords(&["if", "else", "while"]);
        let tokens = lexer.tokenize("if x else y").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword("if".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Keyword("else".to_string()));
    }

    #[test]
    fn lex_string_double_quotes() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("\"hello\"").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::StringLit("hello".to_string()));
    }

    #[test]
    fn lex_string_single_quotes() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("'world'").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::StringLit("world".to_string()));
    }

    #[test]
    fn lex_string_escape() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("\"a\\nb\"").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::StringLit("a\nb".to_string()));
    }

    #[test]
    fn lex_unterminated_string() {
        let lexer = Lexer::new();
        assert!(lexer.tokenize("\"hello").is_err());
    }

    #[test]
    fn lex_operators() {
        let lexer = Lexer::new().with_operators(&["==", "!=", "<=", ">=", "&&", "||"]);
        let tokens = lexer.tokenize("x == y && z != w").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::Operator("==".to_string()));
        assert_eq!(tokens[3].kind, TokenKind::Operator("&&".to_string()));
        assert_eq!(tokens[5].kind, TokenKind::Operator("!=".to_string()));
    }

    #[test]
    fn lex_punct() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("(a)").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Punct('('));
        assert_eq!(tokens[2].kind, TokenKind::Punct(')'));
    }

    #[test]
    fn lex_whitespace_kept() {
        let lexer = Lexer::new().with_skip_whitespace(false);
        let tokens = lexer.tokenize("a b").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::Whitespace);
    }

    #[test]
    fn lex_eof() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn lex_mixed() {
        let lexer = Lexer::new().with_keywords(&["let"]).with_operators(&["="]);
        let tokens = lexer.tokenize("let x = 42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Keyword("let".to_string()));
        assert_eq!(tokens[1].kind, TokenKind::Ident("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Operator("=".to_string()));
        assert_eq!(tokens[3].kind, TokenKind::Integer(42));
    }

    #[test]
    fn lex_newline_kept() {
        let lexer = Lexer::new().with_skip_whitespace(false);
        let tokens = lexer.tokenize("a\nb").unwrap();
        assert_eq!(tokens[1].kind, TokenKind::Newline);
    }

    #[test]
    fn lex_span_correctness() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("hello 42").unwrap();
        assert_eq!(tokens[0].span, Span::new(0, 5));
        assert_eq!(tokens[1].span, Span::new(6, 8));
    }

    #[test]
    fn lex_underscore_ident() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("_private").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Ident("_private".to_string()));
    }

    #[test]
    fn lex_multiple_numbers() {
        let lexer = Lexer::new();
        let tokens = lexer.tokenize("1 2 3").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(1));
        assert_eq!(tokens[1].kind, TokenKind::Integer(2));
        assert_eq!(tokens[2].kind, TokenKind::Integer(3));
    }

    // === Literal parser tests ===

    #[test]
    fn literal_match() {
        let p = literal("hello");
        let (val, pos) = p.parse("hello world", 0).unwrap();
        assert_eq!(val, "hello");
        assert_eq!(pos, 5);
    }

    #[test]
    fn literal_no_match() {
        let p = literal("xyz");
        assert!(p.parse("abc", 0).is_err());
    }

    #[test]
    fn literal_at_offset() {
        let p = literal("world");
        let (val, _) = p.parse("hello world", 6).unwrap();
        assert_eq!(val, "world");
    }

    // === Char parser tests ===

    #[test]
    fn char_pred_digit() {
        let p = char_pred(|c| c.is_ascii_digit(), "digit");
        let (ch, pos) = p.parse("9abc", 0).unwrap();
        assert_eq!(ch, '9');
        assert_eq!(pos, 1);
    }

    #[test]
    fn char_pred_fail() {
        let p = char_pred(|c| c.is_ascii_digit(), "digit");
        assert!(p.parse("abc", 0).is_err());
    }

    #[test]
    fn any_char_matches() {
        let p = any_char();
        let (ch, _) = p.parse("x", 0).unwrap();
        assert_eq!(ch, 'x');
    }

    #[test]
    fn any_char_empty() {
        let p = any_char();
        assert!(p.parse("", 0).is_err());
    }

    #[test]
    fn char_exact_match() {
        let p = char_exact('a');
        let (ch, _) = p.parse("abc", 0).unwrap();
        assert_eq!(ch, 'a');
    }

    #[test]
    fn char_exact_fail() {
        let p = char_exact('a');
        assert!(p.parse("xyz", 0).is_err());
    }

    // === Sequence tests ===

    #[test]
    fn seq_both_match() {
        let p = seq(literal("ab"), literal("cd"));
        let ((a, b), pos) = p.parse("abcd", 0).unwrap();
        assert_eq!(a, "ab");
        assert_eq!(b, "cd");
        assert_eq!(pos, 4);
    }

    #[test]
    fn seq_first_fails() {
        let p = seq(literal("xx"), literal("yy"));
        assert!(p.parse("xxyy", 1).is_err());
    }

    // === Choice tests ===

    #[test]
    fn choice_first() {
        let p = choice(literal("abc"), literal("xyz"));
        let (val, _) = p.parse("abc", 0).unwrap();
        assert_eq!(val, "abc");
    }

    #[test]
    fn choice_second() {
        let p = choice(literal("abc"), literal("xyz"));
        let (val, _) = p.parse("xyz", 0).unwrap();
        assert_eq!(val, "xyz");
    }

    #[test]
    fn choice_neither() {
        let p = choice(literal("a"), literal("b"));
        assert!(p.parse("c", 0).is_err());
    }

    // === Many tests ===

    #[test]
    fn many_zero() {
        let p = many(literal("x"));
        let (vals, pos) = p.parse("abc", 0).unwrap();
        assert!(vals.is_empty());
        assert_eq!(pos, 0);
    }

    #[test]
    fn many_several() {
        let p = many(literal("ab"));
        let (vals, pos) = p.parse("ababab!", 0).unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(pos, 6);
    }

    #[test]
    fn many1_zero_fails() {
        let p = many1(literal("x"));
        assert!(p.parse("abc", 0).is_err());
    }

    #[test]
    fn many1_several() {
        let p = many1(char_exact('a'));
        let (vals, _) = p.parse("aaab", 0).unwrap();
        assert_eq!(vals.len(), 3);
    }

    // === Optional tests ===

    #[test]
    fn optional_some() {
        let p = optional(literal("hi"));
        let (val, pos) = p.parse("hi!", 0).unwrap();
        assert_eq!(val, Some("hi".to_string()));
        assert_eq!(pos, 2);
    }

    #[test]
    fn optional_none() {
        let p = optional(literal("hi"));
        let (val, pos) = p.parse("bye", 0).unwrap();
        assert!(val.is_none());
        assert_eq!(pos, 0);
    }

    // === Lookahead tests ===

    #[test]
    fn and_pred_succeeds() {
        let p = and_pred(literal("abc"));
        let ((), pos) = p.parse("abcdef", 0).unwrap();
        assert_eq!(pos, 0); // no consumption
    }

    #[test]
    fn and_pred_fails() {
        let p = and_pred(literal("xyz"));
        assert!(p.parse("abc", 0).is_err());
    }

    #[test]
    fn not_pred_succeeds() {
        let p = not_pred(literal("xyz"));
        let ((), pos) = p.parse("abc", 0).unwrap();
        assert_eq!(pos, 0);
    }

    #[test]
    fn not_pred_fails() {
        let p = not_pred(literal("abc"));
        assert!(p.parse("abc", 0).is_err());
    }

    // === Map tests ===

    #[test]
    fn map_transform() {
        let p = map(literal("42"), |s| s.parse::<i32>().unwrap());
        let (val, _) = p.parse("42", 0).unwrap();
        assert_eq!(val, 42);
    }

    // === Skip whitespace tests ===

    #[test]
    fn skip_ws_basic() {
        let p = skip_ws(literal("hello"));
        let (val, _) = p.parse("   hello", 0).unwrap();
        assert_eq!(val, "hello");
    }

    #[test]
    fn skip_ws_no_ws() {
        let p = skip_ws(literal("hello"));
        let (val, _) = p.parse("hello", 0).unwrap();
        assert_eq!(val, "hello");
    }

    // === SepBy tests ===

    #[test]
    fn sep_by_empty() {
        let p = sep_by(literal("x"), literal(","));
        let (vals, _) = p.parse("abc", 0).unwrap();
        assert!(vals.is_empty());
    }

    #[test]
    fn sep_by_one() {
        let p = sep_by(literal("x"), literal(","));
        let (vals, _) = p.parse("x", 0).unwrap();
        assert_eq!(vals.len(), 1);
    }

    #[test]
    fn sep_by_many() {
        let p = sep_by(char_pred(|c| c.is_ascii_digit(), "digit"), literal(","));
        let (vals, _) = p.parse("1,2,3", 0).unwrap();
        assert_eq!(vals, vec!['1', '2', '3']);
    }

    // === Between tests ===

    #[test]
    fn between_parens() {
        let p = between(literal("("), literal("abc"), literal(")"));
        let (val, _) = p.parse("(abc)", 0).unwrap();
        assert_eq!(val, "abc");
    }

    #[test]
    fn between_missing_close() {
        let p = between(literal("("), literal("abc"), literal(")"));
        assert!(p.parse("(abc", 0).is_err());
    }

    // === Pratt parser tests ===

    #[test]
    fn pratt_simple_add() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("1 + 2").unwrap();
        assert!((expr.eval().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_precedence() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("1 + 2 * 3").unwrap();
        assert!((expr.eval().unwrap() - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_left_assoc() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("10 - 3 - 2").unwrap();
        assert!((expr.eval().unwrap() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_right_assoc() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("2 ^ 3 ^ 2").unwrap();
        // 2^(3^2) = 2^9 = 512
        assert!((expr.eval().unwrap() - 512.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_parens() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("(1 + 2) * 3").unwrap();
        assert!((expr.eval().unwrap() - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_unary_minus() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("-5").unwrap();
        assert!((expr.eval().unwrap() - (-5.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_unary_in_expr() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("3 + -2").unwrap();
        assert!((expr.eval().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_nested_parens() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("((1 + 2))").unwrap();
        assert!((expr.eval().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_multiply_divide() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("10 / 2 * 3").unwrap();
        assert!((expr.eval().unwrap() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_modulo() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("7 % 3").unwrap();
        assert!((expr.eval().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_complex_expr() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("2 + 3 * 4 - 1").unwrap();
        assert!((expr.eval().unwrap() - 13.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_division_by_zero() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("1 / 0").unwrap();
        assert!(expr.eval().is_err());
    }

    #[test]
    fn pratt_identifier() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("x + 1").unwrap();
        assert!(matches!(
            expr,
            Expr::Binary {
                op,
                ..
            } if op == "+"
        ));
    }

    #[test]
    fn pratt_call_expr() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("f(1, 2)").unwrap();
        assert!(matches!(expr, Expr::Call { .. }));
        if let Expr::Call { args, .. } = &expr {
            assert_eq!(args.len(), 2);
        }
    }

    #[test]
    fn pratt_postfix() {
        let parser = PrattParser::arithmetic().with_postfix("!", 100);
        let expr = parser.parse_expr("5!").unwrap();
        assert!(matches!(expr, Expr::Postfix { .. }));
    }

    #[test]
    fn pratt_float_literal() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("3.14").unwrap();
        assert!((expr.eval().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn pratt_power_operator() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("2 ** 10").unwrap();
        assert!((expr.eval().unwrap() - 1024.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_empty_input() {
        let parser = PrattParser::arithmetic();
        assert!(parser.parse_expr("").is_err());
    }

    #[test]
    fn pratt_just_number() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("42").unwrap();
        assert!((expr.eval().unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_unary_plus() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("+7").unwrap();
        assert!((expr.eval().unwrap() - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_call_no_args() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("f()").unwrap();
        if let Expr::Call { args, .. } = &expr {
            assert!(args.is_empty());
        } else {
            panic!("expected Call");
        }
    }

    // === JSON (recursive descent) tests ===

    #[test]
    fn json_null() {
        assert_eq!(JsonParser::parse("null").unwrap(), JsonValue::Null);
    }

    #[test]
    fn json_true() {
        assert_eq!(JsonParser::parse("true").unwrap(), JsonValue::Bool(true));
    }

    #[test]
    fn json_false() {
        assert_eq!(JsonParser::parse("false").unwrap(), JsonValue::Bool(false));
    }

    #[test]
    fn json_integer() {
        assert_eq!(JsonParser::parse("42").unwrap(), JsonValue::Number(42.0));
    }

    #[test]
    fn json_negative() {
        assert_eq!(JsonParser::parse("-3").unwrap(), JsonValue::Number(-3.0));
    }

    #[test]
    fn json_float() {
        if let JsonValue::Number(n) = JsonParser::parse("3.14").unwrap() {
            assert!((n - 3.14).abs() < 0.001);
        } else {
            panic!("expected number");
        }
    }

    #[test]
    fn json_exponent() {
        if let JsonValue::Number(n) = JsonParser::parse("1e3").unwrap() {
            assert!((n - 1000.0).abs() < 0.001);
        } else {
            panic!("expected number");
        }
    }

    #[test]
    fn json_string() {
        assert_eq!(
            JsonParser::parse("\"hello\"").unwrap(),
            JsonValue::Str("hello".to_string())
        );
    }

    #[test]
    fn json_string_escape() {
        assert_eq!(
            JsonParser::parse("\"a\\nb\"").unwrap(),
            JsonValue::Str("a\nb".to_string())
        );
    }

    #[test]
    fn json_empty_array() {
        assert_eq!(JsonParser::parse("[]").unwrap(), JsonValue::Array(vec![]));
    }

    #[test]
    fn json_array() {
        let val = JsonParser::parse("[1, 2, 3]").unwrap();
        if let JsonValue::Array(items) = val {
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn json_nested_array() {
        let val = JsonParser::parse("[[1], [2]]").unwrap();
        if let JsonValue::Array(items) = val {
            assert_eq!(items.len(), 2);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn json_empty_object() {
        assert_eq!(JsonParser::parse("{}").unwrap(), JsonValue::Object(vec![]));
    }

    #[test]
    fn json_object() {
        let val = JsonParser::parse("{\"a\": 1, \"b\": 2}").unwrap();
        if let JsonValue::Object(pairs) = val {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].0, "a");
            assert_eq!(pairs[1].0, "b");
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn json_nested_object() {
        let val = JsonParser::parse("{\"x\": {\"y\": 42}}").unwrap();
        if let JsonValue::Object(pairs) = val {
            assert!(matches!(pairs[0].1, JsonValue::Object(_)));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn json_whitespace() {
        let val = JsonParser::parse("  { \"a\" : 1 }  ").unwrap();
        assert!(matches!(val, JsonValue::Object(_)));
    }

    #[test]
    fn json_invalid() {
        assert!(JsonParser::parse("xyz").is_err());
    }

    #[test]
    fn json_trailing_content() {
        assert!(JsonParser::parse("42 extra").is_err());
    }

    #[test]
    fn json_mixed_array() {
        let val = JsonParser::parse("[1, \"two\", true, null]").unwrap();
        if let JsonValue::Array(items) = val {
            assert_eq!(items.len(), 4);
            assert_eq!(items[0], JsonValue::Number(1.0));
            assert_eq!(items[1], JsonValue::Str("two".to_string()));
            assert_eq!(items[2], JsonValue::Bool(true));
            assert_eq!(items[3], JsonValue::Null);
        } else {
            panic!("expected array");
        }
    }

    // === Error recovery tests ===

    #[test]
    fn skip_until_basic() {
        let (skipped, pos) = skip_until("abc;def", 0, &[';']);
        assert_eq!(skipped, "abc");
        assert_eq!(pos, 3);
    }

    #[test]
    fn skip_until_end() {
        let (skipped, pos) = skip_until("abcdef", 0, &[';']);
        assert_eq!(skipped, "abcdef");
        assert_eq!(pos, 6);
    }

    #[test]
    fn recovery_insert_default() {
        let r = Recovering::new(literal("x"), RecoveryStrategy::InsertDefault);
        let (val, pos) = r.parse("abc", 0).unwrap();
        assert_eq!(val, "");
        assert_eq!(pos, 0);
    }

    #[test]
    fn recovery_skip_until() {
        let r = Recovering::new(literal("x"), RecoveryStrategy::SkipUntil(vec![';']));
        let (val, pos) = r.parse("abc;def", 0).unwrap();
        assert_eq!(val, "");
        assert_eq!(pos, 3);
    }

    #[test]
    fn parse_with_recovery_all_good() {
        let result = parse_with_recovery(
            "ab ab ab",
            |input, pos| {
                let p = skip_ws(literal("ab"));
                p.parse(input, pos)
            },
            &[' '],
        );
        assert_eq!(result.value.len(), 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn parse_with_recovery_some_errors() {
        let result = parse_with_recovery(
            "ab XX ab",
            |input, pos| {
                let p = skip_ws(literal("ab"));
                p.parse(input, pos)
            },
            &[' '],
        );
        assert_eq!(result.value.len(), 2); // "ab" and "ab"
        assert!(!result.errors.is_empty());
    }

    // === Lexer default trait ===

    #[test]
    fn lexer_default() {
        let lexer = Lexer::default();
        let tokens = lexer.tokenize("42").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
    }

    // === Pratt default trait ===

    #[test]
    fn pratt_default() {
        let parser = PrattParser::default();
        let expr = parser.parse_expr("42").unwrap();
        assert!((expr.eval().unwrap() - 42.0).abs() < f64::EPSILON);
    }

    // === Expr eval errors ===

    #[test]
    fn eval_unknown_var() {
        assert!(Expr::Ident("x".to_string()).eval().is_err());
    }

    #[test]
    fn eval_unknown_prefix() {
        let e = Expr::Prefix {
            op: "~".to_string(),
            operand: Box::new(Expr::Number(1.0)),
        };
        assert!(e.eval().is_err());
    }

    #[test]
    fn eval_unknown_binary() {
        let e = Expr::Binary {
            op: "??".to_string(),
            left: Box::new(Expr::Number(1.0)),
            right: Box::new(Expr::Number(2.0)),
        };
        assert!(e.eval().is_err());
    }

    #[test]
    fn eval_postfix_error() {
        let e = Expr::Postfix {
            op: "!".to_string(),
            operand: Box::new(Expr::Number(5.0)),
        };
        assert!(e.eval().is_err());
    }

    #[test]
    fn eval_call_error() {
        let e = Expr::Call {
            func: Box::new(Expr::Ident("f".to_string())),
            args: vec![],
        };
        assert!(e.eval().is_err());
    }

    #[test]
    fn eval_call_non_ident_error() {
        let e = Expr::Call {
            func: Box::new(Expr::Number(1.0)),
            args: vec![],
        };
        assert!(e.eval().is_err());
    }

    #[test]
    fn eval_not_prefix() {
        let parser = PrattParser::arithmetic().with_prefix("!", 90);
        let expr = parser.parse_expr("!0").unwrap();
        assert!((expr.eval().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn eval_not_prefix_nonzero() {
        let parser = PrattParser::arithmetic().with_prefix("!", 90);
        let expr = parser.parse_expr("!5").unwrap();
        assert!((expr.eval().unwrap() - 0.0).abs() < f64::EPSILON);
    }

    // === Additional combinator tests ===

    #[test]
    fn fn_parser() {
        let p = |input: &str, pos: usize| -> ParseResult<String> {
            if input[pos..].starts_with("ok") {
                Ok(("ok".to_string(), pos + 2))
            } else {
                Err(ParseError::new(pos, "expected ok"))
            }
        };
        let (val, _) = p.parse("ok", 0).unwrap();
        assert_eq!(val, "ok");
    }

    #[test]
    fn many_prevents_infinite_loop() {
        // A parser that always succeeds with zero-width match
        let p = many(optional(literal("x")));
        let (vals, _) = p.parse("yyy", 0).unwrap();
        // Should stop immediately since each match is zero-width
        assert!(vals.is_empty());
    }

    #[test]
    fn json_string_tab_escape() {
        assert_eq!(
            JsonParser::parse("\"a\\tb\"").unwrap(),
            JsonValue::Str("a\tb".to_string())
        );
    }

    #[test]
    fn json_string_backslash_escape() {
        assert_eq!(
            JsonParser::parse("\"a\\\\b\"").unwrap(),
            JsonValue::Str("a\\b".to_string())
        );
    }

    #[test]
    fn pratt_deeply_nested_parens() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("(((((1)))))").unwrap();
        assert!((expr.eval().unwrap() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pratt_multiple_unary() {
        let parser = PrattParser::arithmetic();
        let expr = parser.parse_expr("--5").unwrap();
        assert!((expr.eval().unwrap() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn lex_operator_precedence_longest_match() {
        let lexer = Lexer::new().with_operators(&["=", "=="]);
        let tokens = lexer.tokenize("==").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Operator("==".to_string()));
    }
}
