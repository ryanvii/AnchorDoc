"""Auditable report exporters: JSON, Markdown, and standalone HTML trace viewer."""
from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

from ..types import AuditVerdict


def export_json(verdicts: list[AuditVerdict], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(v) for v in verdicts]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def export_markdown(verdicts: list[AuditVerdict], title: str = "AnchorDoc Audit Report") -> str:
    lines = [
        f"# {title}",
        "",
        "| ID | Item | Status | Cited Pages | Quote Verified | Rule / Summary |",
        "|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        pages = ", ".join(f"p.{p}" for p in v.cited_pages) if v.cited_pages else "-"
        qv = "Yes" if v.quote_verified else "No"
        summary = (v.answer or v.reason or "").replace("\n", " ").replace("|", "\\|")[:140]
        t = v.title.replace("\n", " ").replace("|", "\\|")[:70]
        lines.append(f"| `{v.item_id}` | {t} | **{v.status}** | {pages} | {qv} | `{v.rule_trace}` {summary} |")
    return "\n".join(lines) + "\n"


def export_html(verdicts: list[AuditVerdict], out_path: str | Path, title: str = "AnchorDoc Audit Report") -> Path:
    """Export a clean, zero-dependency standalone HTML audit trace report."""
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    cards_html = []
    for v in verdicts:
        badge_cls = {
            "pass": "badge-pass",
            "answered": "badge-pass",
            "fail": "badge-fail",
            "needs_review": "badge-warn",
            "abstain": "badge-warn",
        }.get(v.status, "badge-warn")
        pages = ", ".join(f"p.{p}" for p in v.cited_pages) or "N/A"
        q_html = (
            f'<blockquote class="quote">"{html.escape(v.verbatim_quote)}" '
            f'<span class="qv">({"Verified" if v.quote_verified else "Unverified"})</span></blockquote>'
            if v.verbatim_quote
            else ""
        )
        calc_html = (
            f'<div class="calc">Python Calc: <code>{html.escape(v.formula)}</code> = <strong>{v.calc_value:.4f}</strong></div>'
            if v.formula and v.calc_value is not None
            else ""
        )
        cards_html.append(
            f"""<article class="card">
              <div class="card-head">
                <span class="badge {badge_cls}">{html.escape(v.status.upper())}</span>
                <strong>{html.escape(v.title)}</strong>
                <span class="meta">ID: {html.escape(v.item_id)} · Pages: {html.escape(pages)} · Rule: <code>{html.escape(v.rule_trace)}</code></span>
              </div>
              <p class="reason">{html.escape(v.answer or v.reason)}</p>
              {calc_html}
              {q_html}
            </article>"""
        )

    summary_pills = " ".join(
        f'<span class="pill">{html.escape(k)}: <strong>{cnt}</strong></span>' for k, cnt in sorted(counts.items())
    )
    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 32px; line-height: 1.6; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    header {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px 28px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; }}
    .pill {{ display: inline-block; background: #f1f5f9; border-radius: 999px; padding: 4px 12px; font-size: 13px; margin-right: 8px; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 22px; margin-bottom: 12px; }}
    .card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 8px; }}
    .meta {{ color: #64748b; font-size: 12.5px; margin-left: auto; }}
    .badge {{ font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 6px; }}
    .badge-pass {{ background: #dcfce7; color: #166534; }}
    .badge-fail {{ background: #fee2e2; color: #991b1b; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; }}
    .reason {{ margin: 6px 0 8px; color: #334155; }}
    .calc {{ font-size: 13.5px; background: #eff6ff; padding: 6px 10px; border-radius: 6px; margin-bottom: 8px; }}
    .quote {{ margin: 8px 0 0; padding: 8px 12px; border-left: 3px solid #94a3b8; background: #f8fafc; font-size: 13.5px; color: #475569; }}
    .qv {{ font-weight: 600; color: #0f172a; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{html.escape(title)}</h1>
      <div>{summary_pills}</div>
    </header>
    {"".join(cards_html)}
  </div>
</body>
</html>
"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc_html, encoding="utf-8")
    return p
