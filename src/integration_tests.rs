//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant
)]

use crate::json::*;
use crate::peg::*;
use crate::pratt::*;
use crate::recovery::*;
use crate::span_error::*;
use crate::tokenizer::*;

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
