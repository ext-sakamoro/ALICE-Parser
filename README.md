**English** | [日本語](README_JP.md)

# ALICE-Parser

Parser combinator library for the ALICE ecosystem. Provides PEG-style combinators, Pratt parsing for operator precedence, a built-in tokenizer/lexer, JSON parser, and error recovery -- all in pure Rust.

## Features

- **Parser Combinators** -- `literal`, `char_pred`, `Seq`, `Choice`, `Many`, `Many1`, `Optional`, `AndPred`, `NotPred`, `Map`, `SkipWs`, `SepBy`, `Between`
- **Pratt Parsing** -- Operator precedence parsing with prefix, postfix, and infix operators (left/right associativity)
- **Tokenizer/Lexer** -- Configurable lexer with keywords, operators, identifiers, numbers, strings, and whitespace handling
- **JSON Parser** -- Complete JSON parser built on top of the combinator framework
- **Expression Evaluator** -- AST with `eval()` for arithmetic expressions
- **Error Recovery** -- `skip_until` synchronization, `Recovering` wrapper, `RecoveredParse` with error collection
- **Span Tracking** -- Byte-offset span tracking for all parsed elements

## Architecture

```
Lexer (tokenizer)
    |
    v
Parser trait  -->  Combinators (Seq, Choice, Many, Map, ...)
    |
    v
PrattParser   -->  Expr AST with eval()
    |
    v
JsonParser    -->  JsonValue
    |
    v
Error Recovery (skip_until, RecoveredParse)
```

## Quick Start

```rust
use alice_parser::*;

// Pratt parser for arithmetic
let parser = PrattParser::arithmetic();
let expr = parser.parse_expr("2 + 3 * 4").unwrap();
assert_eq!(expr.eval().unwrap(), 14.0);

// JSON parsing
let value = JsonParser::parse(r#"{"key": [1, 2, 3]}"#).unwrap();
```

## License

MIT OR Apache-2.0
