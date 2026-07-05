//! Tokenizer / Lexer (`TokenKind` / `Token` / `Lexer`).

use crate::span_error::{ParseError, Span};

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
