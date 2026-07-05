//! PEG combinator core (`Parser` trait + combinators).

use crate::span_error::{ParseError, ParseResult};
use std::marker::PhantomData;

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
