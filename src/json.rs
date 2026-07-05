//! Recursive descent parser (JSON as example).

use crate::span_error::ParseError;

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
