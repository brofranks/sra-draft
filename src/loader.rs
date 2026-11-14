//! SMT-LIB guard definitions evaluated directly into native predicates.
//!
//! A file declares one bit-vector input, then zero-argument Boolean
//! definitions in dependency order. Definitions share owning handles, not
//! expanded syntax. This is a construction format, not an SMT solver:
//! assertions, general functions, and arithmetic are outside this subset.
//! Like `typed`, this frontend sits above the runtime and uses only public
//! construction APIs; the relational evaluator and backends have no
//! dependency on file syntax. The serialized object is a predicate, not a
//! checkpoint of a backend's BDD layout. Loading requires masked bit tests
//! and Boolean construction in the selected backend; the same input need not
//! have the same node count or construction cost under another
//! representation or order. The producer's companion metadata defines the
//! meaning of each input bit and the ordered roots. That semantic mapping is
//! independent of a backend's physical decision ordering.
//!
//! Tokenization checks SMT-LIB spelling; parsing recognizes only the
//! construction subset below. Unsupported constructs are rejected at their
//! heads, without validating the rest of their grammar. There is no complete
//! SMT-LIB parser, solver, or automatic backend fallback here. Bit-vector
//! extraction follows
//! <https://smt-lib.org/theories-FixedSizeBitVectors.shtml>.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Read};

use crate::context::{Context, Predicate};
use crate::row::{Bv, Domain};
use crate::scalar::ScalarSort;

/// A file error with a one-based line and column.
///
/// Input errors never become partial results.
#[derive(Debug)]
pub struct LoadError {
    /// The one-based input line.
    pub line: usize,
    /// The one-based input column.
    pub column: usize,
    /// The reason the input was rejected.
    pub message: String,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for LoadError {}

fn numeral(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && (s == "0" || !s.starts_with('0'))
}

fn whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

fn delimiter(b: u8) -> bool {
    whitespace(b) || b"();".contains(&b)
}

fn simple_symbol(s: &str) -> bool {
    let first = |b: u8| b.is_ascii_alphabetic() || b"~!@$%^&*_-+=<>.?/".contains(&b);
    !s.is_empty() && first(s.as_bytes()[0]) && s.bytes().all(|b| first(b) || b.is_ascii_digit())
}

fn reserved(s: &str) -> bool {
    matches!(
        s,
        "BINARY"
            | "DECIMAL"
            | "HEXADECIMAL"
            | "NUMERAL"
            | "STRING"
            | "_"
            | "!"
            | "as"
            | "lambda"
            | "let"
            | "exists"
            | "forall"
            | "match"
            | "par"
            | "assert"
            | "check-sat"
            | "check-sat-assuming"
            | "declare-const"
            | "declare-datatype"
            | "declare-datatypes"
            | "declare-fun"
            | "declare-sort"
            | "declare-sort-parameter"
            | "define-const"
            | "define-fun"
            | "define-fun-rec"
            | "define-funs-rec"
            | "define-sort"
            | "echo"
            | "exit"
            | "get-assertions"
            | "get-assignment"
            | "get-info"
            | "get-model"
            | "get-option"
            | "get-proof"
            | "get-unsat-assumptions"
            | "get-unsat-core"
            | "get-value"
            | "pop"
            | "push"
            | "reset"
            | "reset-assertions"
            | "set-info"
            | "set-logic"
            | "set-option"
    )
}

/// The bit width of a fixed-radix literal, when representable.
fn literal_width(digits: usize, bits: u32) -> Option<u32> {
    u32::try_from(digits).ok()?.checked_mul(bits)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Open,
    Close,
    Symbol,
    Quoted,
    Reserved,
    Keyword,
    Numeral,
    Decimal,
    String,
    Binary,
    Hex,
}

#[derive(Clone, Copy)]
struct Token<'s> {
    kind: TokenKind,
    text: &'s str,
    at: usize,
}

enum TermKind {
    Bv,
    Bits,
    Bool,
}

// TODO(manual-review): Revisit whether the payload enum or this kind-tagged struct is better for size and simplicity.
struct Term {
    kind: TermKind,
    // Views use bit offsets relative to the parser's sole input.
    low: u32,
    width: u32,
    words: Vec<u64>,
    predicate: Option<Predicate>,
}

impl Term {
    fn bv(low: u32, width: u32) -> Self {
        Self {
            kind: TermKind::Bv,
            low,
            width,
            words: Vec::new(),
            predicate: None,
        }
    }

    fn bits(width: u32, words: Vec<u64>) -> Self {
        Self {
            kind: TermKind::Bits,
            low: 0,
            width,
            words,
            predicate: None,
        }
    }

    fn bool(p: Predicate) -> Self {
        Self {
            kind: TermKind::Bool,
            low: 0,
            width: 1,
            words: Vec::new(),
            predicate: Some(p),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Extract(u32, u32),
    Eq,
    And,
    Or,
    Not,
    Ite,
}

struct Parser<'s> {
    cx: &'s Context,
    source: &'s str,
    pos: usize,
    input: Bv,
    name: &'s str,
    symbols: HashMap<&'s str, Predicate>,
    // An explicit operand stack handles nesting without recursive parsing or
    // a scalar AST.
    frames: Vec<(Op, usize, usize)>,
    terms: Vec<Term>,
}

impl<'s> Parser<'s> {
    fn error(&self, at: usize, message: impl Into<String>) -> LoadError {
        let prefix = &self.source[..at];
        LoadError {
            line: prefix.bytes().filter(|&b| b == b'\n').count() + 1,
            column: prefix.rsplit('\n').next().unwrap().chars().count() + 1,
            message: message.into(),
        }
    }

    fn skip(&mut self) -> bool {
        let bytes = self.source.as_bytes();
        while self.pos < bytes.len() {
            match bytes[self.pos] {
                b';' => {
                    while self.pos < bytes.len() && !matches!(bytes[self.pos], b'\r' | b'\n') {
                        self.pos += 1;
                    }
                }
                b if whitespace(b) => self.pos += 1,
                _ => return true,
            }
        }
        false
    }

    fn token(&mut self) -> Result<Token<'s>, LoadError> {
        if !self.skip() {
            return Err(self.error(self.pos, "unexpected end of input"));
        }
        let at = self.pos;
        let bytes = self.source.as_bytes();
        let kind = match bytes[self.pos] {
            b'(' | b')' => {
                self.pos += 1;
                if bytes[at] == b'(' {
                    TokenKind::Open
                } else {
                    TokenKind::Close
                }
            }
            b'|' | b'"' => {
                let quote = bytes[self.pos];
                self.pos += 1;
                loop {
                    if self.pos == bytes.len() {
                        return Err(self.error(at, "unterminated quoted token"));
                    }
                    let c = bytes[self.pos];
                    self.pos += 1;
                    if c == quote {
                        if quote == b'"' && bytes.get(self.pos) == Some(&b'"') {
                            self.pos += 1;
                            continue;
                        }
                        break;
                    }
                    if c == 127 || (c < 32 && !whitespace(c)) || (quote == b'|' && c == b'\\') {
                        return Err(self.error(at, "invalid character in quoted token"));
                    }
                }
                if self.pos < bytes.len() && !delimiter(bytes[self.pos]) {
                    return Err(self.error(at, "expected a delimiter after quoted token"));
                }
                if quote == b'|' {
                    TokenKind::Quoted
                } else {
                    TokenKind::String
                }
            }
            _ => {
                while self.pos < bytes.len() && !delimiter(bytes[self.pos]) {
                    self.pos += 1;
                }
                let text = &self.source[at..self.pos];
                if numeral(text) {
                    TokenKind::Numeral
                } else if text.split_once('.').is_some_and(|(a, b)| {
                    numeral(a) && !b.is_empty() && b.bytes().all(|c| c.is_ascii_digit())
                }) {
                    TokenKind::Decimal
                } else if text
                    .strip_prefix("#b")
                    .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| matches!(c, b'0' | b'1')))
                {
                    TokenKind::Binary
                } else if text
                    .strip_prefix("#x")
                    .is_some_and(|s| !s.is_empty() && s.bytes().all(|c| c.is_ascii_hexdigit()))
                {
                    TokenKind::Hex
                } else if text.strip_prefix(':').is_some_and(simple_symbol) {
                    TokenKind::Keyword
                } else if reserved(text) {
                    TokenKind::Reserved
                } else if simple_symbol(text) {
                    TokenKind::Symbol
                } else {
                    return Err(self.error(at, format!("invalid token: {text:?}")));
                }
            }
        };
        Ok(Token {
            kind,
            text: &self.source[at..self.pos],
            at,
        })
    }

    fn expect(&mut self, expected: &str) -> Result<(), LoadError> {
        let token = self.token()?;
        if token.text != expected {
            return Err(self.error(
                token.at,
                format!("expected {expected:?}, found {:?}", token.text),
            ));
        }
        Ok(())
    }

    fn command(&mut self) -> Result<Token<'s>, LoadError> {
        self.expect("(")?;
        let token = self.token()?;
        if !matches!(token.kind, TokenKind::Symbol | TokenKind::Reserved) {
            return Err(self.error(token.at, "expected a command name"));
        }
        if !matches!(token.text, "set-logic" | "declare-const" | "define-fun") {
            return Err(self.error(token.at, format!("unsupported command: {}", token.text)));
        }
        Ok(token)
    }

    fn number(&mut self) -> Result<u32, LoadError> {
        let token = self.token()?;
        if token.kind != TokenKind::Numeral {
            return Err(self.error(token.at, "expected a natural numeral"));
        }
        token
            .text
            .parse()
            .map_err(|_| self.error(token.at, "bit width or index exceeds u32"))
    }

    fn symbol(&self, token: Token<'s>) -> Result<&'s str, LoadError> {
        match token.kind {
            TokenKind::Symbol => Ok(token.text),
            TokenKind::Quoted => Ok(&token.text[1..token.text.len() - 1]),
            _ => Err(self.error(token.at, "expected a symbol")),
        }
    }

    fn definition_name(&self, token: Token<'s>) -> Result<&'s str, LoadError> {
        let name = self.symbol(token)?;
        if matches!(
            name,
            "true" | "false" | "and" | "or" | "not" | "ite" | "=" | "extract" | "Bool" | "BitVec"
        ) {
            return Err(self.error(token.at, "reserved symbol"));
        }
        Ok(name)
    }

    /// The next sort: a bit-vector width, or `None` for `Bool`.
    fn sort(&mut self) -> Result<Option<u32>, LoadError> {
        let token = self.token()?;
        if token.kind != TokenKind::Open {
            let name = self.symbol(token)?;
            return if name == "Bool" {
                Ok(None)
            } else {
                Err(self.error(token.at, format!("unsupported sort: {name}")))
            };
        }
        let token = self.token()?;
        if token.text != "_" {
            return Err(self.error(
                token.at,
                format!("unsupported sort: {}", self.symbol(token)?),
            ));
        }
        let name = self.token()?;
        if self.symbol(name)? != "BitVec" {
            return Err(self.error(name.at, format!("unsupported sort: {}", name.text)));
        }
        let width = self.number()?;
        if width == 0 {
            return Err(self.error(name.at, "bit-vector width must be positive"));
        }
        self.expect(")")?;
        Ok(Some(width))
    }

    fn literal(digits: &str, radix: u32, width: u32) -> Term {
        let sort = ScalarSort::bit_vec(width);
        let mut words = vec![0_u64; sort.words()];
        if matches!(radix, 2 | 16) {
            let bits = radix.ilog2();
            for (word, chunk) in words
                .iter_mut()
                .zip(digits.as_bytes().rchunks(64 / bits as usize))
            {
                *word = chunk.iter().fold(0, |value, &digit| {
                    value << bits | u64::from((digit as char).to_digit(radix).unwrap())
                });
            }
        } else {
            for digit in digits.chars() {
                let mut carry = digit.to_digit(radix).unwrap() as u128;
                for word in &mut words {
                    carry += u128::from(*word) * u128::from(radix);
                    *word = carry as u64;
                    carry >>= 64;
                }
            }
        }
        // Indexed decimal literals keep their value modulo 2^width. SMT-LIB
        // itself defines `(_ bvX n)` only for X below 2^n.
        *words.last_mut().unwrap() &= sort.top_mask();
        Term::bits(width, words)
    }

    fn expression(&mut self) -> Result<Term, LoadError> {
        loop {
            let token = self.token()?;
            let value = match token.kind {
                TokenKind::Open => {
                    let head = self.token()?;
                    if head.text == "_" {
                        let name = self.token()?;
                        let digits = self
                            .symbol(name)?
                            .strip_prefix("bv")
                            .filter(|s| numeral(s))
                            .ok_or_else(|| {
                                self.error(
                                    name.at,
                                    format!("unsupported indexed literal: {}", name.text),
                                )
                            })?;
                        let width = self.number()?;
                        if width == 0 {
                            return Err(self.error(name.at, "empty bit-vector literal"));
                        }
                        // A wider literal matches no view, so its declared
                        // size must not reach allocation.
                        if width > self.input.width() {
                            return Err(
                                self.error(name.at, "bit-vector literal exceeds the input width")
                            );
                        }
                        self.expect(")")?;
                        Self::literal(digits, 10, width)
                    } else {
                        let op = match (head.kind, head.text) {
                            (TokenKind::Open, _) => {
                                self.expect("_")?;
                                let name = self.token()?;
                                if self.symbol(name)? != "extract" {
                                    return Err(self.error(
                                        name.at,
                                        format!("unsupported operator: {}", name.text),
                                    ));
                                }
                                let high = self.number()?;
                                let low = self.number()?;
                                self.expect(")")?;
                                Op::Extract(high, low)
                            }
                            (
                                TokenKind::Reserved,
                                "let" | "forall" | "exists" | "lambda" | "match",
                            ) => {
                                return Err(self
                                    .error(head.at, format!("unsupported binder: {}", head.text)));
                            }
                            (TokenKind::Reserved, _) => {
                                return Err(self.error(
                                    head.at,
                                    format!("unsupported term syntax: {}", head.text),
                                ));
                            }
                            _ => match self.symbol(head)? {
                                "=" => Op::Eq,
                                "and" => Op::And,
                                "or" => Op::Or,
                                "not" => Op::Not,
                                "ite" => Op::Ite,
                                name => {
                                    return Err(self
                                        .error(head.at, format!("unsupported operator: {name}")));
                                }
                            },
                        };
                        self.frames.push((op, head.at, self.terms.len()));
                        continue;
                    }
                }
                TokenKind::Close => {
                    let (op, at, start) = self
                        .frames
                        .pop()
                        .ok_or_else(|| self.error(token.at, "unexpected closing parenthesis"))?;
                    let value = self.apply(op, at, &self.terms[start..])?;
                    self.terms.truncate(start);
                    value
                }
                TokenKind::Binary | TokenKind::Hex => {
                    let bits = if token.kind == TokenKind::Binary {
                        1
                    } else {
                        4
                    };
                    Self::literal(
                        &token.text[2..],
                        1 << bits,
                        literal_width(token.text.len() - 2, bits).ok_or_else(|| {
                            self.error(token.at, "fixed-radix literal width exceeds u32")
                        })?,
                    )
                }
                TokenKind::Numeral | TokenKind::Decimal | TokenKind::String => {
                    return Err(
                        self.error(token.at, format!("unsupported literal: {:?}", token.kind))
                    );
                }
                _ => {
                    match self.symbol(token)? {
                        "true" => Term::bool(self.cx.t(self.input.column())),
                        "false" => Term::bool(self.cx.f(self.input.column())),
                        name if name == self.name => Term::bv(0, self.input.width()),
                        name => Term::bool(self.symbols.get(name).cloned().ok_or_else(|| {
                            self.error(token.at, format!("unknown symbol: {name}"))
                        })?),
                    }
                }
            };
            self.terms.push(value);
            if self.frames.is_empty() {
                return Ok(self.terms.pop().unwrap());
            }
        }
    }

    fn apply(&self, op: Op, at: usize, args: &[Term]) -> Result<Term, LoadError> {
        use TermKind::{Bits, Bool, Bv};

        if match op {
            Op::And | Op::Or => args.len() < 2,
            Op::Not | Op::Extract(..) => args.len() != 1,
            Op::Eq => args.len() != 2,
            Op::Ite => args.len() != 3,
        } {
            return Err(self.error(at, format!("invalid arity for {op:?}")));
        }
        let value = match (op, args) {
            (Op::Extract(high, low), [input @ Term { kind: Bv, .. }])
                if low <= high && high < input.width =>
            {
                return Ok(Term::bv(input.low + low, high - low + 1));
            }
            (
                Op::Eq,
                [
                    input @ Term { kind: Bv, .. },
                    bits @ Term { kind: Bits, .. },
                ]
                | [
                    bits @ Term { kind: Bits, .. },
                    input @ Term { kind: Bv, .. },
                ],
            ) if bits.width == input.width => self
                .input
                .extract(input.low + input.width - 1, input.low)
                .eq(self.cx, bits.words.as_slice()),
            (
                Op::Not,
                [
                    Term {
                        kind: Bool,
                        predicate: Some(p),
                        ..
                    },
                ],
            ) => p.not(),
            (
                Op::Ite,
                [
                    Term {
                        kind: Bool,
                        predicate: Some(p),
                        ..
                    },
                    Term {
                        kind: Bool,
                        predicate: Some(yes),
                        ..
                    },
                    Term {
                        kind: Bool,
                        predicate: Some(no),
                        ..
                    },
                ],
            ) => p.ite(yes, no),
            (Op::And | Op::Or, _) => {
                if args.iter().any(|arg| arg.predicate.is_none()) {
                    return Err(self.error(at, "Boolean connective requires Bool operands"));
                }
                fn reduce(op: Op, args: &[Term]) -> std::borrow::Cow<'_, Predicate> {
                    if let [arg] = args {
                        return std::borrow::Cow::Borrowed(arg.predicate.as_ref().unwrap());
                    }
                    let (left, right) = args.split_at(args.len() / 2);
                    let a = reduce(op, left);
                    let b = reduce(op, right);
                    std::borrow::Cow::Owned(if matches!(op, Op::And) {
                        a.and(&b)
                    } else {
                        a.or(&b)
                    })
                }
                reduce(op, args).into_owned()
            }
            _ => return Err(self.error(at, format!("invalid operands or bit bounds for {op:?}"))),
        };
        Ok(Term::bool(value))
    }
}

/// Loads requested guards in root order, retaining duplicates independently.
///
/// The sole declared input is bound to `input`, a bound bit-vector domain of
/// `cx`; its width must match. All definitions are checked, including ones
/// not requested. Dropping the loader's symbol table leaves the returned
/// predicates alive.
///
/// The accepted subset is optional `(set-logic QF_BV)`, one `declare-const`
/// of sort `(_ BitVec n)`, and `define-fun` with no parameters and result
/// sort `Bool`. Expressions support earlier names, `true`, `false`,
/// `#b`/`#x`/`(_ bvN n)` literals, input/view `extract`, view-to-literal `=`,
/// `and`, `or`, `not`, and Boolean `ite`. Equality accepts either operand
/// order. Comments and quoted symbols are supported. `and` and `or` require
/// at least two operands. Bit zero is least significant; indexed decimal
/// literals are reduced modulo 2^n for declared width n. Unsupported commands,
/// sorts, operators, and binders are located errors, as are malformed input
/// and invalid operands. Loading stops at the first error; even unrequested
/// definitions are checked.
///
/// # Errors
///
/// Returns [`LoadError`] when the input is malformed or unsupported, located
/// at the first error, when a root names no definition, located at the end of
/// input, or when `reader` fails, located at line one.
///
/// # Panics
///
/// Panics if `input` belongs to another context, is unbound, or has no
/// bit-vector sort, or a bit test in any definition needs masked bit
/// construction unsupported by the selected backend.
pub fn smt2(
    cx: &Context,
    input: impl Domain,
    reader: impl Read,
    roots: &[impl AsRef<str>],
) -> Result<Vec<Predicate>, LoadError> {
    let source = io::read_to_string(reader).map_err(|e| LoadError {
        line: 1,
        column: 1,
        message: e.to_string(),
    })?;
    let mut p = Parser {
        cx,
        source: &source,
        pos: 0,
        input: Bv::new(cx, input),
        name: "",
        symbols: HashMap::new(),
        frames: Vec::new(),
        terms: Vec::new(),
    };
    let mut command = p.command()?;
    if command.text == "set-logic" {
        let logic = p.token()?;
        if p.symbol(logic)? != "QF_BV" {
            return Err(p.error(logic.at, format!("unsupported logic: {}", logic.text)));
        }
        p.expect(")")?;
        command = p.command()?;
    }
    if command.text != "declare-const" {
        return Err(p.error(command.at, "expected one bit-vector input declaration"));
    }
    let token = p.token()?;
    p.name = p.definition_name(token)?;
    p.skip();
    let at = p.pos;
    let Some(width) = p.sort()? else {
        return Err(p.error(at, "input must have a bit-vector sort"));
    };
    if width != p.input.width() {
        return Err(p.error(at, "declared input width differs from the column"));
    }
    p.expect(")")?;
    while p.skip() {
        let define = p.command()?;
        if define.text != "define-fun" {
            return Err(p.error(
                define.at,
                "expected a Boolean definition after the input declaration",
            ));
        }
        let symbol = p.token()?;
        let name = p.definition_name(symbol)?;
        if name == p.name || p.symbols.contains_key(name) {
            return Err(p.error(symbol.at, "duplicate symbol"));
        }
        p.expect("(")?;
        let params = p.token()?;
        if params.kind == TokenKind::Open {
            return Err(p.error(params.at, "unsupported definition: function parameters"));
        }
        if params.kind != TokenKind::Close {
            return Err(p.error(params.at, "expected a parameter list"));
        }
        p.skip();
        let start = p.pos;
        if p.sort()?.is_some() {
            return Err(p.error(start, "definition must have sort Bool"));
        }
        p.skip();
        let start = p.pos;
        let Term {
            kind: TermKind::Bool,
            predicate: Some(value),
            ..
        } = p.expression()?
        else {
            return Err(p.error(start, "definition must have sort Bool"));
        };
        p.expect(")")?;
        p.symbols.insert(name, value);
    }
    roots
        .iter()
        .map(|root| {
            p.symbols
                .get(root.as_ref())
                .cloned()
                .ok_or_else(|| p.error(p.pos, format!("unknown guard root {:?}", root.as_ref())))
        })
        .collect()
}

#[cfg(test)]
#[path = "../tests/unit/loader.rs"]
mod tests;
