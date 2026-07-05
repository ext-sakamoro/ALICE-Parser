//! Error recovery combinators.

use crate::peg::{Literal, Parser};
use crate::span_error::{ParseError, ParseResult};

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
