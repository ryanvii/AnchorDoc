"""Deterministic answer contracts, answer composition, and numeric provenance checks.

Three domain-agnostic ideas live here:

1. ``infer_answer_contract``: read the *question* (never the gold answer) and infer the expected
   answer shape — count / percent / ratio / yes-no / date / list / entity / free — plus a requested
   scale ("in USD billions") and rounding ("round to two decimal places").
2. ``compose_numeric_answer``: when Python computed a value, the computed value *replaces* the
   LLM-written number for purely numeric answers; for stance answers (Yes/No/qualitative) the
   stance is kept and the verified value is appended. LLM arithmetic never overrides Python.
3. ``operands_grounded``: every operand of a formula must be traceable to the cited source text
   (allowing thousand/million scaling and rounding) — a falsifiable provenance check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Answer contract inference
# ---------------------------------------------------------------------------

_YESNO_RE = re.compile(
    r"^\s*(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would|should|shall|may)\b", re.I
)
_COUNT_RE = re.compile(r"\bhow\s+many\b(?!\s+(?:percent|percentage)\b)|\bnumber\s+of\b|\bcount\s+of\b", re.I)
_PERCENT_RE = re.compile(r"\bpercent(?:age)?\b|%|\bhow\s+many\s+percent\b|\bmargin\b|\brate\s+of\b", re.I)
_RATIO_RE = re.compile(r"\bratio\b|\bturnover\b|\bcoverage\b|\bmultiple\b|\btimes\b", re.I)
_DATE_RE = re.compile(r"^\s*when\b|\bwhat\s+(?:date|day|year|month|time)\b|\bwhich\s+(?:date|year|month)\b", re.I)
_LIST_RE = re.compile(r"\blist\b|\benumerate\b|\bwhat\s+are\b|\bwhich\s+\w+s\b|\bname\s+all\b|\bin\s+list\s+format\b", re.I)
_ENTITY_RE = re.compile(r"^\s*(?:who|which|what\s+is\s+the\s+name)\b", re.I)
_SCALE_RE = re.compile(
    r"\bin\s+(?:(?:usd|us\$|\$|eur|gbp|rmb|cny|jpy)\s+)?(thousands?|millions?|billions?|trillions?|k|mn|bn)\b", re.I
)
_DECIMALS_RE = re.compile(
    r"\bround(?:ed|ing)?\b[^.?]{0,40}?\b(?:to\s+)?(\d+|one|two|three|four|zero|no)\s+decimal", re.I
)
_WORD_NUM = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4}
_SCALE_FACTOR = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9,
    "trillion": 1e12, "trillions": 1e12,
}


@dataclass
class AnswerContract:
    """Expected answer shape inferred deterministically from the question text."""

    kind: str = "free"  # count | percent | ratio | yesno | date | list | entity | number | free
    scale_label: str = ""  # e.g. "billion"
    scale_factor: float = 1.0
    decimals: int | None = None
    list_expected: bool = False
    cues: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human/LLM-readable description used inside reader prompts."""
        parts = {
            "count": "an integer COUNT of items/people/things (if the page gives a percentage and a base size n, compute count = percentage x n)",
            "percent": "a PERCENTAGE value",
            "ratio": "a RATIO / multiple (a plain number, not a percentage unless asked)",
            "yesno": "a YES/NO stance stated FIRST, consistent with the computed figures and any threshold in the question",
            "date": "a DATE / TIME in the format requested by the question",
            "list": "a LIST containing every qualifying item",
            "entity": "a NAME / ENTITY",
            "number": "a NUMBER",
            "free": "a concise direct answer",
        }
        s = parts.get(self.kind, parts["free"])
        if self.scale_label:
            s += f"; express monetary/numeric amounts in {self.scale_label}s as requested"
        if self.decimals is not None:
            s += f"; round to {self.decimals} decimal place(s)"
        if self.list_expected and self.kind != "list":
            s += "; include ALL qualifying items"
        return s


def infer_answer_contract(question: str) -> AnswerContract:
    """Infer expected answer type/scale/rounding from the question text only."""
    q = question or ""
    c = AnswerContract()
    if _YESNO_RE.search(q):
        c.kind = "yesno"
    elif _DATE_RE.search(q):
        c.kind = "date"
    elif re.search(r"\bhow\s+many\s+percent\b|\bwhat\s+percent(?:age)?\b|\bpercentage\s+of\b", q, re.I):
        c.kind = "percent"
    elif _COUNT_RE.search(q):
        c.kind = "count"
    elif _PERCENT_RE.search(q) and re.search(r"\bwhat\b|\bhow\s+much\b|\bcalculate\b|\bcompute\b", q, re.I):
        c.kind = "percent"
    elif _RATIO_RE.search(q) and re.search(r"\bwhat\b|\bhow\b|\bcalculate\b|\bcompute\b", q, re.I):
        c.kind = "ratio"
    elif _LIST_RE.search(q):
        c.kind = "list"
    elif _ENTITY_RE.search(q):
        c.kind = "entity"
    elif re.search(r"\bhow\s+much\b|\bwhat\s+(?:is|was)\s+the\s+(?:value|amount|total|difference|gap)\b", q, re.I):
        c.kind = "number"
    c.list_expected = c.kind == "list" or bool(_LIST_RE.search(q))
    m = _SCALE_RE.search(q)
    if m:
        key = m.group(1).lower()
        c.scale_factor = _SCALE_FACTOR.get(key, 1.0)
        base = {"k": "thousand", "mn": "million", "bn": "billion"}.get(key, key.rstrip("s"))
        c.scale_label = base
    m = _DECIMALS_RE.search(q)
    if m:
        tok = m.group(1).lower()
        c.decimals = int(tok) if tok.isdigit() else _WORD_NUM.get(tok)
    return c


# ---------------------------------------------------------------------------
# 2. Contract checking & deterministic answer composition
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")
_LEADS_WITH_NUMBER_RE = re.compile(r"^\s*(?:approximately|about|around|roughly|~)?\s*[-(]?\s*[$€£¥]?\s*-?\d")


def parse_numbers(text: str) -> list[float]:
    """Parse numbers from free text; accounting parentheses ``(123)`` are negative."""
    out: list[float] = []
    for raw in _NUM_RE.findall(text or ""):
        neg = raw.startswith("(") and raw.endswith(")")
        s = raw.strip("()").replace("$", "").replace(",", "")
        if not s or s in {"-", "."}:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        out.append(-abs(v) if neg else v)
    return out


def leading_number(text: str) -> float | None:
    """Return the first number when the answer *starts* with a number, else None."""
    if not _LEADS_WITH_NUMBER_RE.match(text or ""):
        return None
    nums = parse_numbers(text)
    return nums[0] if nums else None


def _close(a: float, b: float, rel: float = 0.02) -> bool:
    if abs(b) < 1e-12:
        return abs(a) < 1e-9
    return abs(a - b) <= rel * abs(b) + 1e-9


def number_matches(
    candidate: float,
    target: float,
    rel: float = 0.02,
    allow_percent_scale: bool = True,
) -> bool:
    """True when ``candidate`` equals ``target`` up to rounding, optional x100 percent form, or 1e3/1e6 scale.

    Magnitudes are compared (signs are often expressed in words: "a decline of 2,544").
    """
    c, t = abs(candidate), abs(target)
    factors = (1.0, 100.0, 0.01, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9) if allow_percent_scale else (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)
    for f in factors:
        if _close(c, t * f, rel):
            return True
    return False


def format_number(val: float, decimals: int | None = None) -> str:
    """Deterministic human formatting for a computed value."""
    if decimals is None:
        if float(val).is_integer() and abs(val) >= 1:
            return f"{int(round(val)):,}"
        decimals = 2 if abs(val) >= 1 else 4
    s = f"{val:,.{decimals}f}"
    return s


_PCT_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*%")
_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def check_contract(contract: AnswerContract, answer: str, unit: str = "") -> list[str]:
    """Return explicit, conservative contract violations (empty list = OK).

    Only unambiguous shape errors are flagged (they trigger at most ONE cheap repair call):
    - count expected but the answer only contains percentages / percentage points
    - a specific scale requested (e.g. billions) but the answer/unit uses another scale
    - yes/no question answered without any stance
    """
    a = (answer or "").strip()
    low = a.lower()
    if not a or low in {"not answerable", "unanswerable", "n/a"}:
        return []
    issues: list[str] = []
    if contract.kind == "count":
        if "percentage point" in low:
            issues.append("COUNT_EXPECTED_GOT_PERCENTAGE_POINTS")
        elif _PCT_TOKEN_RE.search(a):
            rest = _YEAR_TOKEN_RE.sub(" ", _PCT_TOKEN_RE.sub(" ", a))
            rest = re.sub(r"\[Python-verified:[^\]]*\]", " ", rest)
            others = [abs(x) for x in parse_numbers(rest)]
            if not any(x > 12 for x in others):
                issues.append("COUNT_EXPECTED_GOT_PERCENT")
    if contract.scale_label:
        lead = leading_number(a)
        blob = f"{low} {(unit or '').lower()}"
        mentions_scale = contract.scale_label in blob or (contract.scale_label == "billion" and re.search(r"\d\s*bn?\b", low))
        other_scale = any(w in blob for w in ("thousand", "million", "billion", "trillion") if w != contract.scale_label)
        if lead is not None and not mentions_scale and (other_scale or abs(lead) >= 1000):
            issues.append(f"SCALE_EXPECTED_{contract.scale_label.upper()}")
    if contract.kind == "yesno" and not re.match(r"^\s*(?:yes|no)\b", low):
        # Only flag if the answer does not contain a clear stance word at all
        if not re.search(r"\b(?:yes|no|not|did not|does not|has not|increase|decrease|improv|declin|grew|fell|higher|lower)\w*\b", low):
            issues.append("YESNO_STANCE_MISSING")
    return issues


def compose_numeric_answer(
    llm_answer: str,
    calc_val: float | None,
    formula: str,
    contract: AnswerContract,
    unit: str = "",
    operand_grounded: bool = True,
) -> tuple[str, str]:
    """Merge the LLM answer with a Python-computed value.

    Returns (final_answer, composition_tag) where tag in
    {"NO_CALC", "CALC_REPLACED", "CALC_CONSISTENT", "CALC_ANNOTATED_STANCE",
     "CALC_CONFLICT_STANCE", "CALC_UNGROUNDED_KEPT"}.

    Guard-rails:
    - Stance / yes-no answers are NEVER overwritten by a bare number, even if the LLM begins
      its sentence with a number; conflicting stance values are recorded in the audit trace/meta
      rather than injected into the prose body.
    - When neither the question nor the formula requests a percentage and ``calc_val`` is a
      decimal ratio in (-1.5, 1.5), a x100 percentage form is NOT accepted as matching.
    - If ``operand_grounded`` is False and the LLM provided an answer, keep the LLM answer
      rather than replacing it with a value derived from ungrounded operands.
    """
    ans = (llm_answer or "").strip()
    if calc_val is None:
        return ans, "NO_CALC"
    unit_s = (unit or "").strip()
    formula_has_100 = "100" in (formula or "")
    ratio_decimal = (contract.kind != "percent") and (not formula_has_100) and (0.0 < abs(calc_val) < 1.5)
    pct = (not ratio_decimal) and (
        "%" in unit_s or "percent" in unit_s.lower() or (contract.kind == "percent" and "%" in ans)
    )
    shown_val = calc_val
    if pct and abs(calc_val) <= 1.5 and not formula_has_100:
        # formula produced a fraction while a percentage was requested
        lead_tmp = leading_number(ans)
        if lead_tmp is None or not _close(lead_tmp, calc_val):
            shown_val = calc_val * 100.0
    if contract.kind == "count" and contract.decimals is None and not pct:
        num_s = f"{int(round(shown_val)):,}"
    else:
        num_s = format_number(shown_val, contract.decimals)
    if pct and not num_s.endswith("%"):
        num_s = f"{num_s}%"
        unit_label = ""
    else:
        unit_label = f" {unit_s}" if unit_s and unit_s.lower() not in {"%", "percent", "ratio", "x", "times"} else ""
    trace = f"[Python-verified: {formula} = {calc_val:.4f}]"
    if ans and not operand_grounded:
        return f"{ans} {trace}".strip(), "CALC_UNGROUNDED_KEPT"
    lead = leading_number(ans)
    # Yes/No questions with non-empty text ALWAYS take the stance branch (never bare-number replace)
    if (not ans) or (lead is not None and contract.kind != "yesno"):
        allow_pct_scale = not ratio_decimal
        if (
            lead is not None
            and number_matches(lead, calc_val, allow_percent_scale=allow_pct_scale)
            and ((lead >= 0) == (calc_val >= 0) or abs(calc_val) < 1e-12)
        ):
            return f"{ans} {trace}".strip(), "CALC_CONSISTENT"
        return f"{num_s}{unit_label} {trace}".strip(), "CALC_REPLACED"
    # Stance / qualitative answer: keep prose untouched; flag conflict in tag/meta only
    if any(number_matches(n, calc_val) for n in parse_numbers(ans)):
        return f"{ans} {trace}".strip(), "CALC_ANNOTATED_STANCE"
    return f"{ans} {trace}".strip(), "CALC_CONFLICT_STANCE"


# ---------------------------------------------------------------------------
# 3. Operand provenance
# ---------------------------------------------------------------------------

_TRIVIAL_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 100.0, 1000.0, 365.0, 360.0, 52.0, 1e6, 1e9}
_FORMULA_SIGNED_NUM_RE = re.compile(r"(?<![0-9.])(?P<sign>-)?(?P<num>\d+(?:\.\d+)?)")


def _formula_signed_operands(formula: str) -> list[tuple[str, float]]:
    """Extract operands from an arithmetic expression, attaching unary minus signs."""
    clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", formula or "")
    out: list[tuple[str, float]] = []
    for m in _FORMULA_SIGNED_NUM_RE.finditer(clean):
        num = m.group("num")
        sign = m.group("sign") or ""
        if sign == "-":
            # Unary minus only if start of string or preceded by operator / '(' (ignoring spaces)
            prev = clean[: m.start()].rstrip()
            if prev and prev[-1] not in "(+-*/,=":
                sign = ""
        raw = f"{sign}{num}"
        try:
            out.append((raw, float(raw)))
        except ValueError:
            continue
    return out


def _source_signed_numbers(text: str) -> list[float]:
    from ..runtime.text import extract_signed_numbers

    return [val for val, _pct, _raw in extract_signed_numbers(text or "")]


def operands_grounded(
    formula: str,
    source_text: str,
    *,
    strict_scale: bool = False,
) -> tuple[bool, list[str]]:
    """Check every non-trivial operand of ``formula`` appears in ``source_text`` with intact sign.

    - Preserves negative signs (`-12`, `$(500)`, `(12.5%)`): a negative operand will NOT match a
      positive-only number, and a positive operand will NOT match when only a negative value exists
      (preventing silent sign flips).
    - When ``strict_scale=True``, disables silent thousand/million scaling (`1e3`/`1e6`).
    Returns (all_grounded, ungrounded_operands).
    """
    ops = _formula_signed_operands(formula)
    if not ops:
        return True, []
    src = _source_signed_numbers(source_text)
    src_set = set(src)
    factors = (1.0,) if strict_scale else (1.0, 1e3, 1e-3, 1e6, 1e-6, 100.0, 0.01)
    missing: list[str] = []
    for raw, v in ops:
        # Skip trivial constants (0, 1, 2, 12, 100, etc.) UNLESS their exact opposite sign
        # appears in source_text (e.g., source has `-12%` and formula writes `+12`!).
        if abs(v) in _TRIVIAL_CONSTANTS and not (v != 0 and (-v in src_set) and (v not in src_set)):
            continue
        if v in src_set:
            continue
        num_part = raw.lstrip("-")
        dec = len(num_part.split(".")[1]) if "." in num_part else 0
        tol = 0.5 * 10 ** (-dec) if dec else 0.5
        ok = False
        for s in src:
            # Enforce sign agreement: never ground -12 on +12 or +12 on -12
            if (v < 0 < s) or (s < 0 < v):
                continue
            for f in factors:
                sv = s * f
                if abs(sv - v) <= max(tol, 1e-9) or (abs(v) > 0 and abs(sv - v) / abs(v) < 0.001):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            missing.append(raw)
    return (not missing), missing


# ---------------------------------------------------------------------------
# 4. Conservative deterministic entity / year attestation
# ---------------------------------------------------------------------------

_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&'\-]{2,}\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Generic English question / document-structure words that are often capitalized in questions.
_GENERIC_CAPS = frozenset(
    """what which who whom whose when where why how is are was were does do did has have had can could will would
    should shall may might the a an in on at of for from to by with among according please your you our their its
    this that these those there here and or but not if than then also both either neither each every all any some
    page pages slide slides figure figures chart charts table tables section sections appendix chapter part item
    note notes image images picture pictures photo photos diagram diagrams graph graphs map maps report document
    article paper cover title legend total overall list year years month months day days time times percent
    percentage number numbers amount value values answer format yes no true false first second third last top
    bottom left right fy q1 q2 q3 q4""".split()
)
# Ubiquitous generic acronyms (often expanded or implied in documents); never treated as hard entities.
_GENERIC_ACRONYMS = frozenset(
    "gdp url urls www ceo cfo coo cto usd eur gbp rmb cny jpy id pdf faq kpi kpis roi eps ebit ebitda yoy qoq "
    "cagr api app apps sms sns tv pc ai it hr pr r&d".split()
)
_FORMAT_EXAMPLE_RE = re.compile(
    r"(?:\blike\b|e\.g\.|\bfor\s+example\b|\bsuch\s+as\b|\bformat(?:ted)?(?:\s+as)?\b)\s*[:,]?\s*"
    r"(?:\[[^\]]*\]|'[^']*'|\"[^\"]*\")",
    re.I,
)


def _haystack_vocab(texts: list[str]) -> tuple[str, set[str]]:
    blob = " ".join(t for t in texts if t).lower()
    vocab = set(re.findall(r"[a-z0-9][a-z0-9&'\-]+", blob))
    return blob, vocab


def unattested_terms(question: str, haystack_texts: list[str], fuzzy_cutoff: float = 0.8) -> list[str]:
    """Return hard proper nouns / years from the question with ZERO attestation in the haystack.

    Deliberately conservative (used only to block entity-substitution hallucinations):
    - sentence-initial and generic structure words are ignored;
    - matching is case-insensitive substring, plus fuzzy match (difflib ratio >= cutoff) to
      tolerate typos and plural/possessive variants in questions;
    - fiscal-year forms are accepted (``2022`` <-> ``FY22`` / ``FY2022`` / ``'22``).
    """
    import difflib

    blob, vocab = _haystack_vocab(haystack_texts)
    if not blob:
        return []
    missing: list[str] = []
    q = _FORMAT_EXAMPLE_RE.sub(" ", question or "")  # answer-format examples are not claims about the document
    for m in _CAP_TOKEN_RE.finditer(q):
        tok = m.group(0)
        if m.start() == 0 or q[: m.start()].rstrip().endswith((".", "?", "!", ":")):
            continue
        low = tok.lower().strip("'-")
        low = re.sub(r"'s$", "", low)
        if low in _GENERIC_CAPS or low in _GENERIC_ACRONYMS or len(low) < 3:
            continue
        if low in blob:
            continue
        stem = low[:-1] if low.endswith("s") else low
        if len(stem) >= 4 and stem in blob:
            continue
        if difflib.get_close_matches(low, vocab, n=1, cutoff=fuzzy_cutoff):
            continue
        missing.append(tok)
    for y in set(_YEAR_RE.findall(q)):
        short = y[2:]
        if y in blob or re.search(rf"\bfy\s?'?{short}\b|'{short}\b|\b{y[:2]}\d{{2}}\s*[-/]\s*{short}\b", blob):
            continue
        missing.append(y)
    return missing
