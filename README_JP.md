[English](README.md) | **日本語**

# ALICE-Parser

ALICEエコシステム向けのパーサーコンビネータライブラリ。PEGスタイルのコンビネータ、Pratt構文解析、トークナイザ/レクサー、JSONパーサー、エラー回復を純Rustで提供。

## 機能

- **パーサーコンビネータ** -- `literal`, `char_pred`, `Seq`, `Choice`, `Many`, `Many1`, `Optional`, `AndPred`, `NotPred`, `Map`, `SkipWs`, `SepBy`, `Between`
- **Pratt構文解析** -- 前置・後置・中置演算子の優先順位解析(左結合/右結合)
- **トークナイザ/レクサー** -- キーワード、演算子、識別子、数値、文字列、空白処理を設定可能
- **JSONパーサー** -- コンビネータフレームワーク上に構築された完全なJSONパーサー
- **式評価器** -- `eval()`付きの算術式AST
- **エラー回復** -- `skip_until`同期、`Recovering`ラッパー、エラー収集付き`RecoveredParse`
- **スパン追跡** -- 全パース要素のバイトオフセットスパン追跡

## アーキテクチャ

```
Lexer (トークナイザ)
    |
    v
Parser trait  -->  コンビネータ (Seq, Choice, Many, Map, ...)
    |
    v
PrattParser   -->  Expr AST + eval()
    |
    v
JsonParser    -->  JsonValue
    |
    v
エラー回復 (skip_until, RecoveredParse)
```

## クイックスタート

```rust
use alice_parser::*;

// 算術式のPrattパーサー
let parser = PrattParser::arithmetic();
let expr = parser.parse_expr("2 + 3 * 4").unwrap();
assert_eq!(expr.eval().unwrap(), 14.0);

// JSON解析
let value = JsonParser::parse(r#"{"key": [1, 2, 3]}"#).unwrap();
```

## ライセンス

MIT OR Apache-2.0
