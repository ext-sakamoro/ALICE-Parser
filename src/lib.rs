//! ALICE-Parser: Parser combinator library with PEG, Pratt parsing,
//! recursive descent, tokenizer/lexer, and error recovery.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod json;
pub mod peg;
pub mod pratt;
pub mod prelude;
pub mod recovery;
pub mod span_error;
pub mod tokenizer;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::json::*;
pub use crate::peg::*;
pub use crate::pratt::*;
pub use crate::recovery::*;
pub use crate::span_error::*;
pub use crate::tokenizer::*;
