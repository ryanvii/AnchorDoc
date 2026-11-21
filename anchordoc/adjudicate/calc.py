"""Deterministic AST arithmetic, unit-safe numerical comparison, and tax-rate matching.

Never lets an LLM perform mental arithmetic or silently assume mismatched units
(e.g., never treats '48 hours' as '48 days').
"""
from __future__ import annotations

import ast
import math
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_UNIT_HOURS = {
    "hour": 1.0,
    "hours": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "小时": 1.0,
    "工作小时": 1.0,
    "day": 24.0,
    "days": 24.0,
    "日": 24.0,
    "天": 24.0,
    "日历天": 24.0,
    "自然日": 24.0,
    "工作日": 24.0,
    "month": 30.0 * 24.0,
    "months": 30.0 * 24.0,
    "月": 30.0 * 24.0,
    "个月": 30.0 * 24.0,
    "year": 365.0 * 24.0,
    "years": 365.0 * 24.0,
    "年": 365.0 * 24.0,
}

_COUNT_UNITS = frozenset({"count", "unit", "units", "item", "items", "台", "套", "件", "个", "名", "人", "份"})


_MAX_EXPR_CHARS = 600
_MAX_POW_EXPONENT = 100.0
_MAX_POW_BASE = 1e12
_NUM_LIT_RE = re.compile(r"\d+(?:\.\d+)?%?|\.\d+%?")


def normalize_expression(expr: str) -> str:
    """Normalize an LLM-written arithmetic expression into a strict ASCII form.

    - Unicode minus / dashes / multiplication / division signs -> ASCII operators
    - ``^`` -> ``**`` (bounded power, e.g. CAGR = (end/start) ** (1/n) - 1)
    - Thousands separators and currency symbols are removed
    - Percent literals: when EVERY numeric literal is a percent (pure percentage-point
      arithmetic such as ``65% - 53%``) the ``%`` sign is dropped; otherwise ``N%`` becomes
      ``(N/100)`` so mixed expressions like ``12.5% * 400`` keep their true meaning.
    """
    s = (expr or "").strip()
    for a, b in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u00d7", "*"), ("\u00f7", "/"), ("^", "**")):
        s = s.replace(a, b)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace(",", "").replace("$", "").replace("¥", "").replace("€", "").replace("£", "")
    lits = _NUM_LIT_RE.findall(s)
    if lits and "%" in s:
        if all(x.endswith("%") for x in lits):
            s = s.replace("%", "")
        else:
            s = re.sub(r"(\d+(?:\.\d+)?|\.\d+)%", r"(\1/100)", s)
    return s


def safe_calc(expr: str) -> float | None:
    """Safely evaluate a pure arithmetic expression via Python AST (no eval/exec).

    Supports + - * / unary signs and a bounded ``**`` (needed for growth rates / CAGR).
    Returns None for anything else (names, calls, attribute access, huge powers, complex results).
    """
    s = normalize_expression(expr)
    if not s or len(s) > _MAX_EXPR_CHARS or not re.fullmatch(r"[\d.+\-*/()\s]+", s):
        return None
    try:
        node = ast.parse(s, mode="eval").body

        def _eval(n: ast.AST) -> float:
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                return float(n.value)
            if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
                return _OPS[type(n.op)](_eval(n.operand))
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
                base, exp = _eval(n.left), _eval(n.right)
                if abs(exp) > _MAX_POW_EXPONENT or abs(base) > _MAX_POW_BASE:
                    raise ValueError("power out of bounds")
                if base < 0 and not float(exp).is_integer():
                    raise ValueError("complex result")
                if base == 0 and exp < 0:
                    raise ZeroDivisionError("zero to negative power")
                return float(base**exp)
            if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
                rhs = _eval(n.right)
                if isinstance(n.op, ast.Div) and abs(rhs) < 1e-12:
                    raise ZeroDivisionError("division by zero")
                return _OPS[type(n.op)](_eval(n.left), rhs)
            raise ValueError("Unsupported AST node")

        val = _eval(node)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def compare_duration_or_count(
    required_num: float,
    required_unit: str,
    offered_num: float,
    offered_unit: str,
    direction: str = "le",
) -> str:
    """Unit-safe numerical comparison returning 'ok' | 'violation' | 'unknown_unit'.

    - direction='le': upper bound (e.g., delivery lead time: offered <= required)
    - direction='ge': lower bound (e.g., warranty / experience / quantity: offered >= required)
    - Strictly refuses silent unit fallback when either unit is unknown or incompatible.
    """
    ru = (required_unit or "").strip().lower()
    ou = (offered_unit or "").strip().lower()
    if ru in _UNIT_HOURS and ou in _UNIT_HOURS:
        rv = required_num * _UNIT_HOURS[ru]
        ov = offered_num * _UNIT_HOURS[ou]
    elif ru in _COUNT_UNITS and ou in _COUNT_UNITS:
        rv, ov = float(required_num), float(offered_num)
    else:
        return "unknown_unit"

    if direction == "le":
        return "ok" if ov <= rv + 1e-9 else "violation"
    if direction == "ge":
        return "ok" if ov + 1e-9 >= rv else "violation"
    return "unknown_unit"


def same_quote_with_tax(
    amount_a: float,
    amount_b: float,
    vat_rates: tuple[float, ...] = (0.13, 0.09, 0.06, 0.03),
    tol: float = 0.005,
) -> float | None:
    """Return matching VAT rate if two amounts represent the same quote excl/incl tax."""
    if amount_a <= 0 or amount_b <= 0:
        return None
    lo, hi = (amount_a, amount_b) if amount_a < amount_b else (amount_b, amount_a)
    ratio = hi / lo
    for r in vat_rates:
        if abs(ratio - (1.0 + r)) <= tol * (1.0 + r):
            return r
    return None
