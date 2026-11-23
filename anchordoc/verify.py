"""Independent claim verification & adversarial financial/audit recomputation engine (v0.9.0).

Allows ANY caller (users of RAG, PageIndex, raw long-context LLMs, or human analysts) to hand
AnchorDoc a PDF plus an external claim (quotes, operands with row/period/unit bindings, formula,
and claimed result) and get deterministic, falsifiable verification catching:
- ``QUOTE_SIGN_FLIP`` / ``QUOTE_NOT_FOUND``: sign flips (`-12%` vs `12%`, `$(500)` vs `$500`)
- ``SIGN_FLIP``: negative operand/result misrepresented as positive (or vice versa)
- ``PERIOD_MISMATCH``: value exists on the page/row but belongs to a DIFFERENT fiscal period/year
- ``ROW_LABEL_MISMATCH``: value does not belong to the claimed table row / line item
- ``UNIT_SCALE_MISMATCH``: million vs billion vs thousand scale confusion
- ``PP_VS_PCT_CONFUSION``: percentage points (`pp`) confused with relative percent (`%`)
- ``FORMULA_MISMATCH``: independent Python AST recomputation disagrees with ``claimed_value``
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .adjudicate import safe_calc
from .adjudicate.answer_contract import operands_grounded
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
        # Header line detection: contains >=2 distinct years/periods and no non-year numeric values
        non_year_nums = [
            v for v, pct, _ in nums
            if not (not pct and 1900 <= v <= 2100 and float(int(v)) == v)
        ]
        if len(periods) >= 2 and not non_year_nums:
            active_periods = periods
            continue

        # Extract label prefix (text before first numeric/currency token)
        label_part = re.split(
            r"(?:(?:\s{2,}|\t|\|)\s*(?:[$¥€£(+-]?\s*\d))|(?:[:：]\s*(?:[$¥€£(+-]?\s*\d))",
            stripped,
            maxsplit=1,
        )[0].strip(" :：|\t-")
        if not label_part:
            label_part = re.sub(r"[$¥€£(+-]?\d[\d,]*(?:\.\d+)?%?", " ", stripped).strip()

        # Filter out inline period years from row numeric values when multiple column periods are active
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

    # Evaluate each candidate matching row
    best_diag: tuple[int, str, str, ParsedRow | None] | None = None
    # Severity priority (higher = closer diagnostic match to report if none passes)
    for row in label_rows:
        # 1. Check unit scale compatibility if both claimed unit and table unit are known monetary scales
        unit_mismatch = False
        unit_ratio = 1.0
        if claimed_unit and row.table_unit:
            cu_name, cu_mult = claimed_unit
            tu_name, tu_mult = row.table_unit
            monetary = {"unit", "thousand", "ten_thousand", "million", "hundred_million", "billion"}
            if cu_name in monetary and tu_name in monetary and cu_name != tu_name:
                unit_mismatch = True
                unit_ratio = cu_mult / tu_mult

        # Determine target value(s) for the requested period
        target_vals: list[tuple[float, bool, str]] = row.values
        other_period_vals: list[tuple[str, float]] = []
        period_idx: int | None = None
        if period and row.periods:
            if period in row.periods:
                period_idx = row.periods.index(period)
                if period_idx < len(row.values):
                    target_vals = [row.values[period_idx]]
                    for i, (ov, _, _) in enumerate(row.values):
                        if i != period_idx and i < len(row.periods):
                            other_period_vals.append((row.periods[i], ov))
            else:
                # Row has explicit periods, none matching requested period
                other_period_vals = [
                    (row.periods[i], row.values[i][0])
                    for i in range(min(len(row.periods), len(row.values)))
                ]
                target_vals = []

        # Exact match check on target_vals
        for sv, spct, sraw in target_vals:
            # Check percentage vs non-percentage mismatch when claimed unit is explicit %
            if claimed_unit and claimed_unit[0] == "percent" and not spct and abs(sv) < 1.0 and _nums_close(sv * 100.0, val):
                # e.g., source has 0.12 and claim says 12% -> valid representation
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

            # Sign flip check on exact target cell
            if val != 0 and _nums_close(sv_eff, -val):
                cand = (
                    100,
                    "SIGN_FLIP",
                    f"Sign flip on p{row.page}:L{row.line_no:02d} ({row.label}): source has {sraw} ({sv}), claim states {val}.",
                    row,
                )
                if best_diag is None or cand[0] > best_diag[0]:
                    best_diag = cand

            # Scale mismatch check (e.g. 1200 million claimed as 1.2 million or 1200 billion)
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

            # Also check converted value if claimed unit differs (e.g. table 1200 million, claim 1.2 billion -> VALID!)
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

        # Period mismatch check: value belongs to a DIFFERENT period column on the same row!
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

    # Check if the value exists on another row (cross-row hallucination)
    for row in all_rows:
        for sv, _, sraw in row.values:
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

    # Case 1: Subtracting two percentage rates (e.g., 42.0 - 38.0 = 4.0) produces percentage points (pp),
    # NOT a relative percentage increase (`%` / `percent`)!
    is_rate_diff = bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*-\s*-?\d+(?:\.\d+)?\s*", f))
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
        if not cu and re.search(r"(?:increased|decreased|rose|fell|up|down|提高|增长|下降)\s*(?:by|了)?\s*-?\d+(?:\.\d+)?\s*%(?!\s*p)", text):
            if "percentage point" not in text and "百分点" not in text and "pp" not in text:
                return (
                    f"Formula {f!r} computes a percentage-point difference, but claim text states `%` instead of `pp`/percentage points."
                )

    # Case 2: Relative percentage change `(a - b) / b * 100` claimed as `pp`
    is_rel_change = "/" in f and ("-" in f) and ("100" in f or "100.0" in f)
    if is_rel_change and cu and cu[0] == "pp":
        return (
            f"Formula {f!r} computes a relative percentage change (%), not percentage points ({claimed_unit!r})."
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
    """Deterministically verify an external claim (from any LLM/RAG/PageIndex) against a PDF.

    Zero LLM calls. Returns a falsifiable verification report with explicit ``verdict``
    (``VERIFIED`` | ``FAILED`` | ``UNVERIFIED``), ``error_codes``, per-quote checks,
    per-operand row/period/unit bindings, and independent Python AST recomputation.
    """
    error_codes: list[str] = []
    diagnostics: list[str] = []

    def _add_err(code: str, msg: str) -> None:
        if code not in error_codes:
            error_codes.append(code)
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
            # Distinguish QUOTE_SIGN_FLIP (text matched after stripping punctuation, but sign/percent flipped!)
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

    # Determine grounding scope pages
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
    for op in operands or ():
        if not isinstance(op, dict):
            continue
        res = verify_operand(op, doc, scope_pages, strict_scale=strict_scale)
        operand_results.append(res)
        if not res["verified"]:
            _add_err(res["error_code"], res.get("detail", ""))
        elif res.get("matched_page") and res["matched_page"] not in scope_pages:
            scope_pages.append(res["matched_page"])

    # 3. Formula AST recomputation & operand grounding
    calc_val = safe_calc(formula) if formula else None
    formula_ops_ok = True
    ungrounded_ops: list[str] = []
    if formula:
        if calc_val is None:
            _add_err("INVALID_FORMULA", f"Cannot evaluate arithmetic formula {formula!r} via Python AST.")
        if not operand_results:
            # Ground raw formula operands against scope pages (or all pages if un-scoped)
            check_pages = scope_pages or list(range(1, doc.num_pages + 1))
            scope_text = "\n".join(doc.page_text(p) for p in check_pages)
            formula_ops_ok, ungrounded_ops = operands_grounded(
                formula, scope_text, strict_scale=strict_scale
            )
            if not formula_ops_ok:
                # Check if any ungrounded operand is specifically a sign flip
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
                    # Also check 100x percent vs fraction representation (e.g. 0.40 vs 40%)
                    cu = normalize_unit(claimed_unit)
                    if not (cu and cu[0] == "percent" and _nums_close(calc_val * 100.0, cv)):
                        _add_err("FORMULA_MISMATCH", f"Formula {formula!r} evaluates to {calc_val}, but claimed value is {cv}.")

    # 4. Percentage-point (pp) vs relative percent (%) confusion check
    pp_err = _detect_pp_vs_pct_confusion(formula, claimed_unit, claim_text, operands)
    if pp_err:
        _add_err("PP_VS_PCT_CONFUSION", pp_err)

    has_any_check = bool(quote_results or operand_results or formula)
    if not has_any_check:
        verdict = "UNVERIFIED"
    elif error_codes:
        verdict = "FAILED"
    else:
        verdict = "VERIFIED"

    return {
        "verdict": verdict,
        "verified": verdict == "VERIFIED",
        "error_codes": error_codes,
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
            all(r["verified"] for r in operand_results) if operand_results else formula_ops_ok
        ),
        "ungrounded_operands": ungrounded_ops,
        "grounding_scope": scope_kind,
        "grounding_pages": scope_pages,
        "citations": [f"p.{p}" for p in scope_pages],
    }
