//! Convenience re-export (= `use alice_parser::prelude::*;`).

pub use crate::json::{JsonParser, JsonValue};
pub use crate::peg::{
    any_char, char_exact, char_pred, literal, AndPred, Between, CharPred, Choice, Literal, Many,
    Many1, Map, NotPred, Optional, Parser, SepBy, Seq, SkipWs,
};
pub use crate::pratt::{Assoc, Expr, PrattParser};
pub use crate::recovery::{
    parse_with_recovery, skip_until, RecoveredParse, Recovering, RecoveryStrategy,
};
pub use crate::span_error::{ParseError, ParseResult, Span};
pub use crate::tokenizer::{Lexer, Token, TokenKind};
