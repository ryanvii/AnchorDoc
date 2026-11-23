"""Independent claim verification & adversarial financial/audit recomputation engine (v0.9.1).

Allows ANY caller (users of RAG, PageIndex, raw long-context LLMs, or human analysts) to hand
AnchorDoc a PDF plus an external claim (quotes, operands with row/period/unit bindings, formula,
and claimed result) and get deterministic, scope-explicit verification catching:
- ``QUOTE_SIGN_FLIP`` / ``QUOTE_NOT_FOUND``: sign flips (`-12%` vs `12%`, `$(500)` vs `$500`)
- ``SIGN_FLIP``: negative operand/result misrepresented as positive (or vice versa)
- ``PERIOD_MISMATCH``: value exists on the page/row but belongs to a DIFFERENT fiscal period/year
- ``ROW_LABEL_MISMATCH``: value does not belong to the claimed table row / line item
- ``UNIT_SCALE_MISMATCH``: million vs billion vs thousand scale confusion
- ``PP_VS_PCT_CONFUSION``: percentage points (`pp`) confused with relative percent (`%`)
- ``FORMULA_OPERAND_UNBOUND``: formula uses a number or symbol NOT bound to any verified operand
- ``FORMULA_MISMATCH``: independent Python AST recomputation disagrees with ``claimed_value``
- ``CLAIM_UNSUPPORTED_BY_EVIDENCE``: claim asserts numbers/facts not substantiated by verified quotes/computation

Verdict Contract (Scoped Honesty):
- ``VERIFIED``: Every checked layer passed AND no unverified numeric claim assertions remain.
- ``PARTIALLY_VERIFIED``: Some evidence (e.g. a quote) exists verbatim, but the overall claim
  itself is NOT substantiated (e.g., irrelevant quote attached to an unverified numeric claim).
- ``FAILED``: At least one hard contradiction / fabrication / arithmetic / binding error caught.
- ``UNVERIFIED``: No verifiable evidence was supplied.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .adjudicate import safe_calc
from .adjudicate.answer_contract import (
    _TRIVIAL_CONSTANTS,
    _formula_signed_operands,
    operands_grounded,
)
from .docbase import PdfDocument
from .runtime.text import (
    _norm_ws_punct,
    extract_signed_numbers,
    localize_quote_pages,
    verify_quote,
)

# ---------------------------------------------------------------------------
# Unit normalization & scale helpers
# ---------------------------------------------------------------------------

_UNIT_SCALE: dict[str, tuple[str, float]] = {
    "thousand": ("thousand", 1e3),
    "thousands": ("thousand", 1e3),
    "k": ("thousand", 1e3),
    "千": ("thousand", 1e3),
    "万元": ("ten_thousand", 1e4),
    "万": ("ten_thousand", 1e4),
    "million": ("million", 1e6),
    "millions": ("million", 1e6),
    "m": ("million", 1e6),
    "mn": ("million", 1e6),
    "mm": ("million", 1e6),
    "百万": ("million", 1e6),
    "亿": ("hundred_million", 1e8),
    "亿元": ("hundred_million", 1e8),
    "billion": ("billion", 1e9),
    "billions": ("billion", 1e9),
    "b": ("billion", 1e9),
    "bn": ("billion", 1e9),
    "十亿": ("billion", 1e9),
    "%": ("percent", 1.0),
    "percent": ("percent", 1.0),
    "pct": ("percent", 1.0),
    "百分比": ("percent", 1.0),
    "pp": ("pp", 1.0),
    "percentage point": ("pp", 1.0),
    "percentage points": ("pp", 1.0),
    "百分点": ("pp", 1.0),
    "bps": ("bps", 1.0),
    "bp": ("bps", 1.0),
    "基点": ("bps", 1.0),
    "usd": ("unit", 1.0),
    "$": ("unit", 1.0),
    "元": ("unit", 1.0),
    "rmb": ("unit", 1.0),
    "cny": ("unit", 1.0),
}

_TABLE_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bin\s+billions\b|\(\s*\$?\s*billions?\s*\)|单位\s*[:：]?\s*十亿", re.I), "billion"),
    (re.compile(r"\bin\s+millions\b|\(\s*\$?\s*millions?\s*\)|单位\s*[:：]?\s*百万", re.I), "million"),
    (re.compile(r"\bin\s+thousands\b|\(\s*\$?\s*thousands?\s*\)|单位\s*[:：]?\s*千", re.I), "thousand"),
    (re.compile(r"单位\s*[:：]?\s*亿元|（亿元）|\(亿元\)"), "亿元"),
    (re.compile(r"单位\s*[:：]?\s*万元|（万元）|\(万元\)"), "万元"),
    (re.compile(r"单位\s*[:：]?\s*元|（元）|\(元\)"), "元"),
]

_PERIOD_RE = re.compile(
    r"\b(?:FY\s*)?(20\d{2}|19\d{2})\b|(?:Q[1-4]\s*20\d{2})|(?:20\d{2}\s*年)",
    re.I,
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def normalize_unit(unit: str | None) -> tuple[str, float] | None:
    u = (unit or "").strip().lower()
    if not u:
        return None
    u_clean = re.sub(r"^\$?\s*in\s+", "", u).strip("()（） ")
    if u_clean in _UNIT_SCALE:
        return _UNIT_SCALE[u_clean]
    for k, v in _UNIT_SCALE.items():
        if k and len(k) >= 2 and k in u_clean:
            return v
    return None


def detect_page_unit(page_text: str) -> tuple[str, float] | None:
    for pat, label in _TABLE_UNIT_PATTERNS:
        if pat.search(page_text or ""):
            return normalize_unit(label)
    return None


def _norm_period(p: str | None) -> str:
    s = (p or "").strip().upper()
    m = re.search(r"(Q[1-4])\s*(20\d{2})", s)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m2 = re.search(r"(20\d{2}|19\d{2})", s)
    return m2.group(1) if m2 else s


def _extract_periods_in_line(line: str) -> list[str]:
    found: list[str] = []
    for m in _PERIOD_RE.finditer(line or ""):
        p = _norm_period(m.group(0))
        if p and p not in found:
            found.append(p)
    return found


def _safe_eval_bound_formula(
    formula: str,
    symbol_map: dict[str, float],
) -> tuple[float | None, list[str], list[tuple[str, float]]]:
    """Safely evaluate an arithmetic AST expression that may mix bound symbols and numeric literals.

    Returns ``(value, unbound_symbols, numeric_literals)`` where ``numeric_literals`` captures every
    signed numeric constant in the AST so mixed formulas like ``rev - 700`` cannot smuggle unbound
    numbers past operand verification. Never uses ``eval()``.
    """
    expr = (formula or "").strip()
    if not expr:
        return None, [], []
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None, [], []

    unbound: list[str] = []
    literals: list[tuple[str, float]] = []

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            v = float(node.value)
            literals.append((str(node.value), v))
            return v
        if isinstance(node, ast.Name):
            if node.id in symbol_map:
                return float(symbol_map[node.id])
            if node.id not in unbound:
                unbound.append(node.id)
            return 0.0
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
                v = -float(node.operand.value)
                literals.append((f"-{node.operand.value}", v))
                return v
            return float(_UNARY_OPS[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Exponent too large")
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return float(_BIN_OPS[type(node.op)](left, right))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        val = _eval(tree)
    except Exception:
        return None, unbound, literals
    if unbound:
        return None, unbound, literals
    return val, [], literals


# ---------------------------------------------------------------------------
# Line-level table row + header period extraction
# ---------------------------------------------------------------------------


@dataclass
class ParsedRow:
    page: int
    line_no: int
    raw_line: str
    label: str
    values: list[tuple[float, bool, str]]
    periods: list[str]
    table_unit: tuple[str, float] | None


def parse_page_rows(page: int, page_text: str) -> list[ParsedRow]:
    """Parse text lines on a page into structured row candidates with active column periods."""
    lines = (page_text or "").splitlines()
    page_unit = detect_page_unit(page_text)
    active_periods: list[str] = []
    active_unit = page_unit
    rows: list[ParsedRow] = []

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        lu = detect_page_unit(stripped)
        if lu:
            active_unit = lu

        periods = _extract_periods_in_line(stripped)
        nums = extract_signed_numbers(stripped)
        non_year_nums = [
            v for v, pct, _ in nums
            if not (not pct and 1900 <= v <= 2100 and float(int(v)) == v)
        ]
        if len(periods) >= 2 and not non_year_nums:
            active_periods = periods
            continue

        label_part = re.split(
            r"(?:(?:\s{2,}|\t|\|)\s*(?:[$¥€£(+-]?\s*\d))|(?:[:：]\s*(?:[$¥€£(+-]?\s*\d))",
            stripped,
            maxsplit=1,
        )[0].strip(" :：|\t-")
        if not label_part:
            label_part = re.sub(r"[$¥€£(+-]?\d[\d,]*(?:\.\d+)?%?", " ", stripped).strip()

        row_vals: list[tuple[float, bool, str]] = []
        inline_periods = periods if (len(periods) == 1 and not active_periods) else []
        for v, pct, raw_tok in nums:
            if not pct and 1900 <= v <= 2100 and float(int(v)) == v and str(int(v)) in periods:
                continue
            row_vals.append((v, pct, raw_tok))

        if row_vals:
            rows.append(
                ParsedRow(
                    page=page,
                    line_no=idx,
                    raw_line=stripped,
                    label=label_part,
                    values=row_vals,
                    periods=list(active_periods) if active_periods else inline_periods,
                    table_unit=active_unit,
                )
            )
    return rows


def _label_matches(claimed_label: str, row_label: str, raw_line: str) -> bool:
    cl = _norm_ws_punct(claimed_label)
    if not cl:
        return True
    rl = _norm_ws_punct(row_label)
    fl = _norm_ws_punct(raw_line)
    if cl in rl or cl in fl:
        return True
    q_toks = [t for t in cl.split() if len(t) >= 2]
    if not q_toks:
        return False
    hits = sum(1 for t in q_toks if t in fl)
    return hits / len(q_toks) >= 0.75


def _nums_close(a: float, b: float) -> bool:
    if abs(a - b) <= 1e-6:
        return True
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom <= 0.002


# ---------------------------------------------------------------------------
# Operand & Claim Verification
# ---------------------------------------------------------------------------


def verify_operand(
    operand: dict[str, Any],
    doc: PdfDocument,
    scope_pages: Sequence[int],
    *,
    strict_scale: bool = True,
) -> dict[str, Any]:
    """Verify a single structured operand against PDF pages with row/period/unit/sign checks."""
    name = str(operand.get("name") or operand.get("row_label") or "operand")
    raw_val = operand.get("value")
    try:
        val = float(str(raw_val).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return {
            "name": name,
            "verified": False,
            "error_code": "INVALID_OPERAND_VALUE",
            "detail": f"Cannot parse numeric operand value: {raw_val!r}",
        }

    claimed_unit_raw = str(operand.get("unit") or "").strip()
    claimed_unit = normalize_unit(claimed_unit_raw)
    row_label = str(operand.get("row_label") or "").strip()
    period_raw = str(operand.get("period") or "").strip()
    period = _norm_period(period_raw)
    hint_page = operand.get("page")

    pages_to_check: list[int] = []
    if hint_page is not None:
        try:
            hp = int(hint_page)
            if 1 <= hp <= doc.num_pages:
                pages_to_check.append(hp)
        except (TypeError, ValueError):
            pass
    for p in scope_pages:
        if p not in pages_to_check and 1 <= p <= doc.num_pages:
            pages_to_check.append(p)
    if not pages_to_check:
        pages_to_check = list(range(1, doc.num_pages + 1))

    all_rows: list[ParsedRow] = []
    for p in pages_to_check:
        all_rows.extend(parse_page_rows(p, doc.page_text(p)))

    label_rows = [r for r in all_rows if _label_matches(row_label, r.label, r.raw_line)] if row_label else all_rows

    if row_label and not label_rows:
        return {
            "name": name,
            "value": val,
            "row_label": row_label,
            "period": period_raw,
            "unit": claimed_unit_raw,
            "verified": False,
            "error_code": "ROW_LABEL_MISMATCH",
            "detail": f"Row label {row_label!r} not found in scope pages {pages_to_check}.",
        }

    best_diag: tuple[int, str, str, ParsedRow | None] | None = None
    for row in label_rows:
        unit_mismatch = False
        unit_ratio = 1.0
        if claimed_unit and row.table_unit:
            cu_name, cu_mult = claimed_unit
            tu_name, tu_mult = row.table_unit
            monetary = {"unit", "thousand", "ten_thousand", "million", "hundred_million", "billion"}
            if cu_name in monetary and tu_name in monetary and cu_name != tu_name:
                unit_mismatch = True
                unit_ratio = cu_mult / tu_mult

        target_vals: list[tuple[float, bool, str]] = row.values
        other_period_vals: list[tuple[str, float]] = []
        if period and row.periods:
            if period in row.periods:
                period_idx = row.periods.index(period)
                if period_idx < len(row.values):
                    target_vals = [row.values[period_idx]]
                    for i, (ov, _, _) in enumerate(row.values):
                        if i != period_idx and i < len(row.periods):
                            other_period_vals.append((row.periods[i], ov))
            else:
                other_period_vals = [
                    (row.periods[i], row.values[i][0])
                    for i in range(min(len(row.periods), len(row.values)))
                ]
                target_vals = []

        for sv, spct, sraw in target_vals:
            if claimed_unit and claimed_unit[0] == "percent" and not spct and abs(sv) < 1.0 and _nums_close(sv * 100.0, val):
                sv_eff = sv * 100.0
            else:
                sv_eff = sv

            if _nums_close(sv_eff, val):
                if unit_mismatch and strict_scale:
                    cand = (
                        95,
                        "UNIT_SCALE_MISMATCH",
                        f"Value {val} matched line {row.page}:L{row.line_no}, but table unit is {row.table_unit[0]!r} while claim states {claimed_unit[0]!r}.",
                        row,
                    )
                    if best_diag is None or cand[0] > best_diag[0]:
                        best_diag = cand
                    continue
                return {
                    "name": name,
                    "value": val,
                    "row_label": row_label or row.label,
                    "period": period_raw,
                    "unit": claimed_unit_raw or (row.table_unit[0] if row.table_unit else ""),
                    "verified": True,
                    "error_code": None,
                    "matched_page": row.page,
                    "line_anchor": f"p{row.page}:L{row.line_no:02d}",
                    "source_line": row.raw_line,
                    "source_value": sv,
                    "table_unit": row.table_unit[0] if row.table_unit else None,
                }

            if val != 0 and _nums_close(sv_eff, -val):
                cand = (
                    100,
                    "SIGN_FLIP",
                    f"Sign flip on p{row.page}:L{row.line_no:02d} ({row.label}): source has {sraw} ({sv}), claim states {val}.",
                    row,
                )
                if best_diag is None or cand[0] > best_diag[0]:
                    best_diag = cand

            if sv != 0 and strict_scale:
                for factor in (1e3, 1e-3, 1e6, 1e-6):
                    if _nums_close(sv * factor, val):
                        cand = (
                            90,
                            "UNIT_SCALE_MISMATCH",
                            f"Scale mismatch on p{row.page}:L{row.line_no:02d}: source has {sraw} (unit={row.table_unit[0] if row.table_unit else 'unknown'}), claim states {val} {claimed_unit_raw}.",
                            row,
                        )
                        if best_diag is None or cand[0] > best_diag[0]:
                            best_diag = cand

            if unit_mismatch and _nums_close(sv, val * unit_ratio):
                return {
                    "name": name,
                    "value": val,
                    "row_label": row_label or row.label,
                    "period": period_raw,
                    "unit": claimed_unit_raw,
                    "verified": True,
                    "error_code": None,
                    "matched_page": row.page,
                    "line_anchor": f"p{row.page}:L{row.line_no:02d}",
                    "source_line": row.raw_line,
                    "source_value": sv,
                    "table_unit": row.table_unit[0] if row.table_unit else None,
                    "note": f"Unit conversion verified ({sv} {row.table_unit[0]} == {val} {claimed_unit[0]})",
                }

        for other_p, ov in other_period_vals:
            if _nums_close(ov, val):
                target_desc = target_vals[0][2] if target_vals else "N/A"
                cand = (
                    98,
                    "PERIOD_MISMATCH",
                    f"Period mismatch on p{row.page}:L{row.line_no:02d} ({row.label}): claimed {period}={val}, but {val} is actually {other_p} (while {period} is {target_desc}).",
                    row,
                )
                if best_diag is None or cand[0] > best_diag[0]:
                    best_diag = cand

    if best_diag is not None:
        _, code, detail, row = best_diag
        return {
            "name": name,
            "value": val,
            "row_label": row_label,
            "period": period_raw,
            "unit": claimed_unit_raw,
            "verified": False,
            "error_code": code,
            "detail": detail,
            "matched_page": row.page if row else None,
            "line_anchor": f"p{row.page}:L{row.line_no:02d}" if row else None,
            "source_line": row.raw_line if row else None,
        }

    for row in all_rows:
        for sv, _, _ in row.values:
            if _nums_close(sv, val):
                return {
                    "name": name,
                    "value": val,
                    "row_label": row_label,
                    "period": period_raw,
                    "unit": claimed_unit_raw,
                    "verified": False,
                    "error_code": "ROW_LABEL_MISMATCH",
                    "detail": f"Value {val} was NOT found on row {row_label!r}, but appeared on a different row p{row.page}:L{row.line_no:02d} ({row.label!r}).",
                    "matched_page": row.page,
                    "line_anchor": f"p{row.page}:L{row.line_no:02d}",
                    "source_line": row.raw_line,
                }

    return {
        "name": name,
        "value": val,
        "row_label": row_label,
        "period": period_raw,
        "unit": claimed_unit_raw,
        "verified": False,
        "error_code": "OPERAND_NOT_FOUND",
        "detail": f"Operand {name}={val} ({period_raw} {claimed_unit_raw}) not grounded in pages {pages_to_check}.",
    }


def _detect_pp_vs_pct_confusion(
    formula: str,
    claimed_unit: str,
    claim_text: str,
    operands: Sequence[dict[str, Any]],
) -> str | None:
    """Detect percentage-point (pp) vs relative percent (%) confusion."""
    cu = normalize_unit(claimed_unit)
    f = (formula or "").strip()
    text = (claim_text or "").lower()

    is_rate_diff = bool(
        re.fullmatch(r"\s*(?:-?\d+(?:\.\d+)?|[A-Za-z_]\w*)\s*-\s*(?:-?\d+(?:\.\d+)?|[A-Za-z_]\w*)\s*", f)
    )
    ops_are_pct = bool(operands) and all(
        (normalize_unit(str(o.get("unit") or "")) or ("", 0))[0] == "percent"
        for o in operands
    )
    mentions_rate_metric = any(
        w in text for w in ("margin", "rate", "ratio", "毛利率", "净利率", "资产负债率", "利润率")
    )
    if is_rate_diff and (ops_are_pct or mentions_rate_metric):
        if cu and cu[0] == "percent":
            return (
                f"Formula {f!r} subtracts two percentage rates, which yields percentage points (pp), "
                f"not relative percent ({claimed_unit!r})."
            )
        if not cu and re.search(
            r"(?:increased|decreased|rose|fell|up|down|提高|增长|下降)\s*(?:by|了)?\s*-?\d+(?:\.\d+)?\s*%(?!\s*p)",
            text,
        ):
            if "percentage point" not in text and "百分点" not in text and "pp" not in text:
                return (
                    f"Formula {f!r} computes a percentage-point difference, but claim text states `%` instead of `pp`/percentage points."
                )

    # Use AST root structure to distinguish:
    # - Difference of two ratios: `(a / b * 100) - (c / d * 100)` -> percentage points (`pp`)!
    # - Single relative percentage change: `(a - b) / b * 100` -> relative percent (`%`)!
    is_ratio_diff = False
    is_rel_change = False
    try:
        root = ast.parse(f, mode="eval").body
        if isinstance(root, ast.BinOp) and isinstance(root.op, ast.Sub):
            left_has_div = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(root.left))
            right_has_div = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(root.right))
            if left_has_div and right_has_div:
                is_ratio_diff = True
        if not is_ratio_diff and "/" in f and "-" in f and ("100" in f or "100.0" in f):
            is_rel_change = True
    except SyntaxError:
        is_rel_change = "/" in f and ("-" in f) and ("100" in f or "100.0" in f)

    if is_ratio_diff and cu and cu[0] == "percent":
        return (
            f"Formula {f!r} subtracts two percentage ratios, which yields percentage points (pp), "
            f"not relative percent ({claimed_unit!r})."
        )
    if is_rel_change and cu and cu[0] == "pp":
        return (
            f"Formula {f!r} computes a relative percentage change (%), not percentage points ({claimed_unit!r})."
        )
    return None


def _extract_non_year_claim_numbers(claim_text: str) -> list[float]:
    """Extract numeric assertions from claim_text, ignoring fiscal years (1900-2099)."""
    out: list[float] = []
    for v, pct, _ in extract_signed_numbers(claim_text or ""):
        if not pct and 1900 <= v <= 2099 and float(int(v)) == v:
            continue
        out.append(v)
    return out


_CLAIM_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "net income": ("净利润", "净收益"),
    "gross profit": ("毛利", "毛利润"),
    "gross margin": ("毛利率",),
    "operating income": ("营业利润",),
    "operating margin": ("营业利润率",),
    "total revenue": ("营业总收入", "总营收", "总收入"),
    "cost of goods sold": ("营业成本", "销售成本"),
    "restructuring expense": ("重组费用",),
    "total assets": ("总资产", "资产总计"),
    "total liabilities": ("总负债", "负债总计"),
    "long-term debt": ("长期债务", "长期借款"),
    "debt-to-asset ratio": ("资产负债率",),
}

_ADJUSTED_PREFIX_RE = re.compile(
    r"(?:adjusted|non[-\s]?gaap|pro\s+forma|ex[-\s]?[a-z]+|excluding|normalized|core|underlying|调整后|扣非|剔除)\s*$",
    re.IGNORECASE,
)
_CHANGE_SUFFIX_RE = re.compile(
    r"^\s*(?:growth|grew|change|changed|expansion|expanded|contraction|contracted|decline|declined|"
    r"increase|increased|decrease|decreased|yoy|qoq|delta|diff|difference|spread|vs\.?|versus|"
    r"增长|增速|变动|扩张|收缩|下降|同比|环比|差值)",
    re.IGNORECASE,
)
_RATE_SUFFIX_RE = re.compile(r"^\s*(?:margin|ratio|rate|%|率)", re.IGNORECASE)


def _clean_row_label_for_metric(raw_label: str) -> str:
    """Strip trailing numeric columns and parenthetical note pointers from a parsed table row label."""
    s = re.split(r"\s+(?:[$¥€£(+-]?\d)", raw_label or "", maxsplit=1)[0]
    s = re.sub(r"\s*\((?:see\s+)?note\s+\d+[^)]*\)", "", s, flags=re.IGNORECASE)
    return s.strip(" :：|\t-")


def _check_claim_metric_conflict(
    doc: PdfDocument,
    claim_text: str,
    *,
    target_val: float | None,
    target_unit: str,
    target_periods: Sequence[str],
    evidence_pages: Sequence[int],
) -> str | None:
    """Detect when claim_text labels a computed value with a DIFFERENT table row metric whose
    actual value on the evidence statement contradicts the claimed value (e.g. computing
    rev - cogs = 480 [Gross Profit] but claiming 'FY2024 Net Income is 480 million' when
    Net Income on the same statement is 125).
    """
    if target_val is None or not claim_text.strip() or not evidence_pages:
        return None

    cu = normalize_unit(target_unit)
    is_claim_pct = bool(cu and cu[0] in ("percent", "pp"))

    for p in evidence_pages:
        if p < 1 or p > doc.num_pages:
            continue
        rows = parse_page_rows(p, doc.page_text(p))
        for row in rows:
            clean_lbl = _clean_row_label_for_metric(row.label)
            if len(clean_lbl) < 4 or len(clean_lbl.split()) > 6 or clean_lbl.endswith("."):
                continue
            low_lbl = clean_lbl.lower()
            candidates: list[tuple[str, bool]] = [(clean_lbl, False)]
            for alias in _CLAIM_METRIC_ALIASES.get(low_lbl, ()):
                candidates.append((alias, True))

            match_span: tuple[int, int] | None = None
            for cand, is_cjk in candidates:
                if is_cjk:
                    idx = claim_text.find(cand)
                    if idx >= 0:
                        match_span = (idx, idx + len(cand))
                        break
                else:
                    m = re.search(rf"(?<![A-Za-z0-9]){re.escape(cand)}(?![A-Za-z0-9])", claim_text, re.IGNORECASE)
                    if m:
                        match_span = (m.start(), m.end())
                        break
            if not match_span:
                continue

            start, end = match_span
            left_ctx = claim_text[max(0, start - 28) : start]
            right_ctx = claim_text[end : min(len(claim_text), end + 28)]
            if _ADJUSTED_PREFIX_RE.search(left_ctx):
                continue
            if _CHANGE_SUFFIX_RE.search(right_ctx):
                continue
            if not re.search(r"(?:margin|ratio|rate|率)$", low_lbl) and _RATE_SUFFIX_RE.search(right_ctx):
                continue

            # Extract row's actual values for target_periods (if period columns align)
            actual_entries: list[tuple[float, bool, str]] = []
            matched_period_label = ""
            if target_periods and row.periods and len(row.values) >= len(row.periods):
                for tp in target_periods:
                    for idx_p, rp in enumerate(row.periods):
                        if rp == tp and idx_p < len(row.values):
                            actual_entries.append((row.values[idx_p][0], row.values[idx_p][1], rp))
                            matched_period_label = rp
            if not actual_entries:
                actual_entries = [(v, pct, "") for v, pct, _ in row.values]
            if not actual_entries:
                continue

            row_is_pct = all(pct for _, pct, _ in actual_entries)
            if is_claim_pct != row_is_pct:
                # Percent claim vs non-percent row (or vice versa): skip unless unit is omitted on a pct row
                if target_unit.strip() or not row_is_pct:
                    continue

            # Compare target_val (scaled to row's table unit when both are currency scales)
            cmp_val = target_val
            if (
                not row_is_pct
                and cu
                and cu[0] in ("thousand", "ten_thousand", "million", "billion")
                and row.table_unit
                and row.table_unit[0] in ("thousand", "ten_thousand", "million", "billion")
            ):
                cmp_val = target_val * (cu[1] / row.table_unit[1])

            if any(_nums_close(cmp_val, av) for av, _, _ in actual_entries):
                return None  # Explicitly corroborated by the named table row!

            actual_val = actual_entries[0][0]
            period_desc = f" for {matched_period_label}" if matched_period_label else ""
            return (
                f"Claim labels result {target_val} as {clean_lbl!r}, but {clean_lbl!r} on "
                f"p{row.page}:L{row.line_no:02d} ({row.raw_line}) is {actual_val}{period_desc}."
            )
    return None


def verify_claim(
    doc: PdfDocument,
    *,
    claim_text: str = "",
    quotes: Sequence[str | dict[str, Any]] = (),
    formula: str = "",
    claimed_value: float | str | None = None,
    claimed_unit: str = "",
    operands: Sequence[dict[str, Any]] = (),
    cited_pages: Sequence[int] = (),
    strict_scale: bool = True,
) -> dict[str, Any]:
    """Deterministically verify an external claim against a PDF with scoped honesty guarantees.

    Key v0.9.1 guarantees:
    1. Formula-to-Operand Binding: When structured ``operands`` are supplied alongside a ``formula``,
       every non-trivial numeric literal OR symbolic identifier in ``formula`` MUST bind to a
       verified operand (preventing "evidence laundering" where valid operands mask a wrong formula
       like ``1200 - 700 = 500`` when bound operands are ``1200`` and ``720``).
    2. Scoped Verdict Contract: Explicitly reports per-layer status in ``checks``
       (``quotes``, ``operands``, ``computation``, ``claim_alignment``) and ``verified_scope``.
       Attaching a true-but-irrelevant quote to an unverified numeric claim returns
       ``PARTIALLY_VERIFIED`` (``verified=False``), NEVER ``VERIFIED``.
    """
    error_codes: list[str] = []
    warning_codes: list[str] = []
    diagnostics: list[str] = []

    def _add_err(code: str, msg: str) -> None:
        if code not in error_codes:
            error_codes.append(code)
        diagnostics.append(f"[{code}] {msg}")

    def _add_warn(code: str, msg: str) -> None:
        if code not in warning_codes:
            warning_codes.append(code)
        diagnostics.append(f"[{code}] {msg}")

    # 1. Verbatim quote verification (sign- and percent-preserving!)
    quote_results: list[dict[str, Any]] = []
    localized_pages: list[int] = []
    for q in quotes or ():
        if isinstance(q, dict):
            qtext = str(q.get("text") or q.get("quote") or "").strip()
            hint_p = q.get("page")
        else:
            qtext = str(q or "").strip()
            hint_p = None
        if not qtext:
            continue
        found_pages = localize_quote_pages(qtext, doc.pages)
        ok_hint = False
        ov_hint = 0.0
        hp: int | None = None
        if hint_p is not None:
            try:
                hp = int(hint_p)
                ok_hint, ov_hint = verify_quote(qtext, doc.page_text(hp))
            except Exception:
                hp = None
        ok = bool(found_pages) or ok_hint
        err_code: str | None = None
        if not ok:
            nq = _norm_ws_punct(qtext)
            candidate_pages = [hp] if (hp and 1 <= hp <= doc.num_pages) else list(range(1, doc.num_pages + 1))
            sign_flipped = any(nq and nq in _norm_ws_punct(doc.page_text(p)) for p in candidate_pages)
            err_code = "QUOTE_SIGN_FLIP" if sign_flipped else "QUOTE_NOT_FOUND"
            _add_err(
                err_code,
                f"Quote {qtext[:80]!r} failed verification ({'sign/percent mismatch against source' if sign_flipped else 'not found verbatim'}).",
            )
        elif 0 < len(found_pages) <= 3:
            for fp in found_pages:
                if fp not in localized_pages:
                    localized_pages.append(fp)
        quote_results.append({
            "text": qtext[:200],
            "verified": ok,
            "error_code": err_code,
            "hint_page": hp,
            "matched_pages": found_pages,
            "overlap": 1.0 if ok else ov_hint,
        })

    # Grounding scope pages
    valid_cited = [int(p) for p in (cited_pages or ()) if 1 <= int(p) <= doc.num_pages]
    scope_pages: list[int] = list(localized_pages)
    for cp in valid_cited:
        if cp not in scope_pages:
            scope_pages.append(cp)
    for op in operands or ():
        if isinstance(op, dict) and op.get("page") is not None:
            try:
                op_p = int(op["page"])
                if 1 <= op_p <= doc.num_pages and op_p not in scope_pages:
                    scope_pages.append(op_p)
            except (TypeError, ValueError):
                pass
    if localized_pages and valid_cited:
        scope_kind = "quote_and_cited"
    elif localized_pages:
        scope_kind = "quote_localized"
    elif valid_cited:
        scope_kind = "cited_pages"
    elif scope_pages:
        scope_kind = "operand_pages"
    else:
        scope_kind = "document_wide"

    # 2. Structured operand verification (row + period + unit + sign)
    operand_results: list[dict[str, Any]] = []
    verified_op_values: list[float] = []
    verified_op_symbols: dict[str, float] = {}
    for op in operands or ():
        if not isinstance(op, dict):
            continue
        res = verify_operand(op, doc, scope_pages, strict_scale=strict_scale)
        operand_results.append(res)
        if not res["verified"]:
            _add_err(res["error_code"], res.get("detail", ""))
        else:
            v = float(res["value"])
            verified_op_values.append(v)
            op_name = str(res.get("name") or "").strip()
            if op_name and re.fullmatch(r"[A-Za-z_]\w*", op_name):
                verified_op_symbols[op_name] = v
            if res.get("matched_page") and res["matched_page"] not in scope_pages:
                scope_pages.append(res["matched_page"])

    # 3. Formula AST recomputation & STRICT operand binding (symbols AND numeric literals!)
    calc_val: float | None = None
    formula_ops_ok = True
    ungrounded_ops: list[str] = []
    has_symbolic_vars = bool(formula and re.search(r"[A-Za-z_]\w*", formula))

    if formula:
        if has_symbolic_vars:
            calc_val, unbound_syms, ast_literals = _safe_eval_bound_formula(formula, verified_op_symbols)
            if unbound_syms:
                formula_ops_ok = False
                ungrounded_ops.extend(unbound_syms)
                _add_err(
                    "FORMULA_OPERAND_UNBOUND",
                    f"Formula {formula!r} references unbound symbol(s) {unbound_syms} not present in verified operands {sorted(verified_op_symbols)}.",
                )
            elif calc_val is None:
                _add_err("INVALID_FORMULA", f"Cannot evaluate symbolic formula {formula!r} via Python AST.")

            # ALSO check every non-trivial numeric literal inside a mixed formula (e.g., `rev - 700`)!
            if ast_literals:
                unbound_lits: list[str] = []
                for raw_lit, lv in ast_literals:
                    if abs(lv) in _TRIVIAL_CONSTANTS:
                        continue
                    if not any(_nums_close(lv, bv) for bv in verified_op_values):
                        unbound_lits.append(raw_lit)
                if unbound_lits:
                    formula_ops_ok = False
                    ungrounded_ops.extend(unbound_lits)
                    _add_err(
                        "FORMULA_OPERAND_UNBOUND",
                        f"Mixed formula {formula!r} uses unbound numeric literal(s) {unbound_lits} not bound to any verified operand ({verified_op_values}).",
                    )
        else:
            calc_val = safe_calc(formula)
            if calc_val is None:
                _add_err("INVALID_FORMULA", f"Cannot evaluate arithmetic formula {formula!r} via Python AST.")

            # ALWAYS check every numeric operand in formula!
            if operand_results:
                for raw_op, ov in _formula_signed_operands(formula):
                    if abs(ov) in _TRIVIAL_CONSTANTS:
                        continue
                    if not any(_nums_close(ov, bv) for bv in verified_op_values):
                        formula_ops_ok = False
                        ungrounded_ops.append(raw_op)
                if ungrounded_ops:
                    _add_err(
                        "FORMULA_OPERAND_UNBOUND",
                        f"Formula {formula!r} uses operand(s) {ungrounded_ops} that do NOT match any verified bound operand ({verified_op_values}).",
                    )
            else:
                check_pages = scope_pages or list(range(1, doc.num_pages + 1))
                scope_text = "\n".join(doc.page_text(p) for p in check_pages)
                formula_ops_ok, ungrounded_ops = operands_grounded(
                    formula, scope_text, strict_scale=strict_scale
                )
                if not formula_ops_ok:
                    signed_src = [v for v, _, _ in extract_signed_numbers(scope_text)]
                    for raw_op in ungrounded_ops:
                        try:
                            ov = float(raw_op)
                        except ValueError:
                            continue
                        if ov != 0 and any(_nums_close(sv, -ov) for sv in signed_src):
                            _add_err("SIGN_FLIP", f"Formula operand {raw_op} appears in source ONLY with opposite sign.")
                        else:
                            _add_err("OPERAND_NOT_FOUND", f"Formula operand {raw_op} not grounded in scope pages {check_pages}.")

        if claimed_value is not None and calc_val is not None:
            try:
                cv = float(str(claimed_value).replace(",", "").replace("%", "").strip())
            except (TypeError, ValueError):
                cv = None
            if cv is not None:
                if cv != 0 and _nums_close(calc_val, -cv):
                    _add_err("SIGN_FLIP", f"Formula {formula!r} evaluates to {calc_val}, opposite sign of claimed {cv}.")
                elif not _nums_close(calc_val, cv):
                    cu = normalize_unit(claimed_unit)
                    if not (cu and cu[0] == "percent" and _nums_close(calc_val * 100.0, cv)):
                        _add_err("FORMULA_MISMATCH", f"Formula {formula!r} evaluates to {calc_val}, but claimed value is {cv}.")

    # 4. Percentage-point (pp) vs relative percent (%) confusion check
    pp_err = _detect_pp_vs_pct_confusion(formula, claimed_unit, claim_text, operands)
    if pp_err:
        _add_err("PP_VS_PCT_CONFUSION", pp_err)

    # 5. Claim-to-Evidence Alignment Check (Scoped Honesty Contract):
    # - 5a: Fiscal period consistency between claim_text and bound operands (catches FY2023 claim on FY2024 math!).
    # - 5b: Metric-name consistency against actual table rows on evidence pages (catches "Net Income is 480" when
    #       rev - cogs = 480 is Gross Profit and Net Income on the same statement is actually 125!).
    # - In quote-only mode (no bound operands / formula), NEVER grant VERIFIED to a free-text claim
    #   merely because a number like `65%` appears in both sentences (prevents metric/meaning confusion
    #   such as "Gross Margin is 65%" verified by "Debt-to-Asset Ratio <= 65%").
    claim_alignment_status = "not_checked"
    if claim_text.strip():
        # 5a. Period consistency check against bound operands
        claim_periods = _extract_periods_in_line(claim_text)
        bound_periods = [
            _norm_period(str(op.get("period") or ""))
            for op in operand_results
            if op.get("verified") and str(op.get("period") or "").strip()
        ]
        period_conflict = False
        if claim_periods and bound_periods:
            missing_periods = [cp for cp in claim_periods if cp not in bound_periods]
            if missing_periods:
                period_conflict = True
                claim_alignment_status = "failed"
                _add_err(
                    "CLAIM_PERIOD_MISMATCH",
                    f"Claim references period(s) {missing_periods}, but verified bound operands only cover {sorted(set(bound_periods))}.",
                )

        # 5b. Metric-name conflict check against table rows on evidence pages
        metric_conflict = False
        if not period_conflict and (operand_results or formula):
            evidence_pages = sorted(
                {
                    int(op["page"])
                    for op in operand_results
                    if op.get("verified") and op.get("page")
                }
            ) or list(scope_pages or [])
            target_val: float | None = None
            if claimed_value is not None:
                try:
                    target_val = float(str(claimed_value).replace(",", "").replace("%", "").strip())
                except (TypeError, ValueError):
                    target_val = None
            if target_val is None and calc_val is not None:
                target_val = calc_val
            if target_val is None:
                cnums = _extract_non_year_claim_numbers(claim_text)
                if cnums:
                    target_val = cnums[0]
            target_periods = claim_periods or sorted(set(bound_periods))
            metric_err = _check_claim_metric_conflict(
                doc,
                claim_text,
                target_val=target_val,
                target_unit=claimed_unit,
                target_periods=target_periods,
                evidence_pages=evidence_pages,
            )
            if metric_err:
                metric_conflict = True
                claim_alignment_status = "failed"
                _add_err("CLAIM_METRIC_MISMATCH", metric_err)

        if not period_conflict and not metric_conflict:
            claim_nums = _extract_non_year_claim_numbers(claim_text)
            if not operand_results and not formula:
                # Quote-only mode: quote existence proves the quote is verbatim, NOT that the
                # free-text claim's metric/subject/polarity matches the quote semantically.
                claim_alignment_status = "unverified"
                if claim_nums:
                    quote_nums: list[float] = []
                    for qr in quote_results:
                        if qr["verified"]:
                            for qv, _, _ in extract_signed_numbers(qr["text"]):
                                quote_nums.append(qv)
                    unsubstantiated = [
                        cn for cn in claim_nums
                        if not any(_nums_close(cn, qn) for qn in quote_nums)
                    ]
                    if unsubstantiated:
                        _add_warn(
                            "CLAIM_UNSUPPORTED_BY_EVIDENCE",
                            f"Claim asserts number(s) {unsubstantiated} not present in any verified quote.",
                        )
                    else:
                        _add_warn(
                            "CLAIM_ENTAILMENT_NOT_CHECKED",
                            "Quote(s) verified verbatim, but free-text metric/meaning alignment is not verified without structured cell operands or semantic NLI.",
                        )
                elif quote_results:
                    _add_warn(
                        "CLAIM_ENTAILMENT_NOT_CHECKED",
                        "Quote(s) verified verbatim, but semantic entailment of the free-text claim is not verified deterministically.",
                    )
            else:
                # Structured operand / formula mode: check every numeric assertion in claim_text
                substantiated_nums: list[float] = list(verified_op_values)
                if calc_val is not None and not error_codes:
                    substantiated_nums.append(calc_val)
                    cu = normalize_unit(claimed_unit)
                    if cu and cu[0] == "percent" and abs(calc_val) < 1.0:
                        substantiated_nums.append(calc_val * 100.0)
                for qr in quote_results:
                    if qr["verified"]:
                        for qv, _, _ in extract_signed_numbers(qr["text"]):
                            substantiated_nums.append(qv)

                unsubstantiated = [
                    cn for cn in claim_nums
                    if not any(_nums_close(cn, sn) for sn in substantiated_nums)
                ]
                if unsubstantiated:
                    claim_alignment_status = "unverified"
                    _add_warn(
                        "CLAIM_UNSUPPORTED_BY_EVIDENCE",
                        f"Claim asserts number(s) {unsubstantiated} that are NOT substantiated by any verified quote, bound operand, or recomputed formula.",
                    )
                else:
                    claim_alignment_status = "verified"

    # 6. Per-layer structured status & Scoped Verdict
    quotes_status = (
        "not_checked"
        if not quote_results
        else ("verified" if all(r["verified"] for r in quote_results) else "failed")
    )
    operands_status = (
        "not_checked"
        if not operand_results
        else ("verified" if all(r["verified"] for r in operand_results) else "failed")
    )
    computation_status = (
        "not_checked"
        if not formula
        else (
            "failed"
            if any(
                c in error_codes
                for c in (
                    "INVALID_FORMULA",
                    "FORMULA_OPERAND_UNBOUND",
                    "FORMULA_MISMATCH",
                    "SIGN_FLIP",
                    "OPERAND_NOT_FOUND",
                    "PP_VS_PCT_CONFUSION",
                )
            )
            else "verified"
        )
    )

    checks = {
        "quotes": quotes_status,
        "operands": operands_status,
        "computation": computation_status,
        "claim_alignment": claim_alignment_status,
    }
    verified_scope = [k for k, v in checks.items() if v == "verified"]

    has_any_check = bool(quote_results or operand_results or formula)
    if not has_any_check:
        verdict = "UNVERIFIED"
    elif error_codes:
        verdict = "FAILED"
    elif warning_codes:
        # Some evidence checked out (e.g., verbatim quote exists), but overall claim is NOT verified!
        verdict = "PARTIALLY_VERIFIED"
    else:
        verdict = "VERIFIED"

    return {
        "verdict": verdict,
        "verified": verdict == "VERIFIED",
        "checks": checks,
        "verified_scope": verified_scope,
        "error_codes": error_codes,
        "warning_codes": warning_codes,
        "diagnostics": diagnostics,
        "claim_text": claim_text,
        "quotes": quote_results,
        "all_quotes_verified": all(r["verified"] for r in quote_results) if quote_results else False,
        "operands": operand_results,
        "formula": formula,
        "calc_value": calc_val,
        "claimed_value": claimed_value,
        "claimed_unit": claimed_unit,
        "operands_grounded": (
            (all(r["verified"] for r in operand_results) and formula_ops_ok)
            if operand_results
            else formula_ops_ok
        ),
        "ungrounded_operands": ungrounded_ops,
        "grounding_scope": scope_kind,
        "grounding_pages": scope_pages,
        "citations": [f"p.{p}" for p in scope_pages],
    }


# ---------------------------------------------------------------------------
# Cross-Reference & Evidence Sufficiency Gap Detector (Zero-LLM)
# ---------------------------------------------------------------------------

_XREF_CITE_RE = re.compile(
    r"(?:(?:see|disclosed\s+in|refer\s+to|described\s+in|per|detailed\s+in)\s+)?"
    r"\b((?:Note|Appendix|Schedule|Exhibit)\s+(?:\d+[A-Za-z]?|[A-Z](?=\b|\s|\.|,)))\b",
    re.IGNORECASE,
)
_XREF_CITE_ZH_RE = re.compile(r"(?:见|详见|参见|披露于)\s*(附注\s*[一二三四五六七八九十\d]+|附件\s*[A-Z0-9一二三四五六七八九十]+)")


def detect_unresolved_references(
    pages_text: dict[int, str],
    query: str = "",
) -> list[dict[str, Any]]:
    """Detect explicit document cross-references (e.g. ``Note 7``, ``Appendix A``) cited inside
    already-read pages whose defining section header has NOT been read yet.

    This powers sufficiency-driven follow-up reads: when Round-1 pages mention ``See Note 7``
    next to a query-relevant line item, the agent knows its evidence is incomplete and can
    issue a targeted Round-2 ``doc_locate`` / ``doc_read`` before committing to an answer.
    """
    if not pages_text:
        return []

    # 1. Collect section headings ALREADY present in the read pages (e.g. "NOTE 7. RESTRUCTURING...")
    defined_refs: set[str] = set()
    for text in pages_text.values():
        for raw_line in (text or "").splitlines():
            s = raw_line.strip()
            if not s:
                continue
            m_en = re.match(
                r"^((?:Note|Appendix|Schedule|Exhibit)\s+(?:\d+[A-Za-z]?|[A-Z]))(?:\s*[:.\-—]|$)",
                s,
                re.IGNORECASE,
            )
            if m_en:
                defined_refs.add(re.sub(r"\s+", " ", m_en.group(1)).lower())
            m_zh = re.match(r"^(附注\s*[一二三四五六七八九十\d]+|附件\s*[A-Z0-9一二三四五六七八九十]+)(?:\s*[:：.、\-—]|$)", s)
            if m_zh:
                defined_refs.add(re.sub(r"\s+", "", m_zh.group(1)))

    query_toks = {t.lower() for t in re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", query or "")}
    seen_refs: dict[str, dict[str, Any]] = {}

    # 2. Scan read pages for cross-references not yet defined in defined_refs
    for p, text in sorted(pages_text.items()):
        for line_no, raw_line in enumerate((text or "").splitlines(), start=1):
            s = raw_line.strip()
            if not s:
                continue
            # Ignore lines that themselves start with the heading
            if re.match(r"^(?:Note|Appendix|Schedule|Exhibit)\s+(?:\d+[A-Za-z]?|[A-Z])\s*[:.\-—]", s, re.IGNORECASE):
                continue

            found_refs: list[tuple[str, str]] = []
            # Require either explicit pointer verb ("see Note 7", "in Note 7") or parenthetical "(Note 7)"
            for m in _XREF_CITE_RE.finditer(s):
                full = m.group(0).lower()
                ref_raw = re.sub(r"\s+", " ", m.group(1)).strip()
                ref_key = ref_raw.lower()
                before = s[max(0, m.start() - 12) : m.start()].lower()
                has_cue = (
                    any(k in full for k in ("see ", "disclosed in ", "refer to ", "described in ", "per ", "detailed in "))
                    or before.rstrip().endswith(("(", "in", "see", "per"))
                )
                if has_cue:
                    found_refs.append((ref_key, ref_raw))
            for m in _XREF_CITE_ZH_RE.finditer(s):
                ref_raw = m.group(1).strip()
                found_refs.append((re.sub(r"\s+", "", ref_raw), ref_raw))

            line_toks = {t.lower() for t in re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}", s)}
            overlap = sorted(query_toks & line_toks) if query_toks else []

            for ref_key, ref_raw in found_refs:
                if ref_key in defined_refs:
                    continue
                cand = {
                    "ref": ref_raw,
                    "cited_on_page": p,
                    "line_anchor": f"p{p}:L{line_no:02d}",
                    "context_line": s,
                    "query_relevant": bool(overlap) or not query_toks,
                    "matched_terms": overlap,
                    "followup_query": f"{ref_raw} {' '.join(overlap)}".strip(),
                }
                prev = seen_refs.get(ref_key)
                if prev is None or (cand["query_relevant"] and not prev["query_relevant"]) or len(cand["matched_terms"]) > len(prev["matched_terms"]):
                    seen_refs[ref_key] = cand

    results = list(seen_refs.values())
    results.sort(key=lambda r: (not r["query_relevant"], -len(r["matched_terms"]), r["cited_on_page"]))
    return results
