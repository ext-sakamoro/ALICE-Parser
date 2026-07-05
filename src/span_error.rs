//! Span byte offset + `ParseError` types.

use core::fmt;

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
