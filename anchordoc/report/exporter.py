"""Auditable report exporters: JSON, Markdown, and interactive two-pane HTML trace workbench."""
from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from ..types import AuditVerdict

if TYPE_CHECKING:
    from ..docbase.pdf_loader import PdfDocument


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


def export_html(
    verdicts: list[AuditVerdict],
    out_path: str | Path,
    title: str = "AnchorDoc Audit Report",
    doc: "PdfDocument | None" = None,
    render_scale: float = 1.15,
    max_embedded_pages: int = 24,
) -> Path:
    """Export a zero-dependency interactive two-pane HTML audit workbench.

    When `doc` is supplied with an underlying PDF, cited original pages are rendered
    via Apache-2.0 pypdfium2 and embedded directly inside the standalone HTML so users
    can click any page pill on the left and inspect the original PDF page on the right
    with zero flicker and zero backend server.
    """
    counts: dict[str, int] = {}
    referenced_pages: list[int] = []
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
        for p in v.cited_pages or []:
            if p not in referenced_pages:
                referenced_pages.append(p)

    # Render referenced pages to embedded base64 PNGs (if PDF is available)
    page_images: dict[int, str] = {}
    if doc is not None and doc.path is not None and doc.path.is_file():
        for p in referenced_pages[:max_embedded_pages]:
            try:
                page_images[p] = doc.render_page_png_b64(p, scale=render_scale)
            except Exception:
                continue

    cards_html = []
    for idx, v in enumerate(verdicts):
        badge_cls = {
            "pass": "badge-pass",
            "answered": "badge-pass",
            "fail": "badge-fail",
            "needs_review": "badge-warn",
            "abstain": "badge-warn",
        }.get(v.status, "badge-warn")
        page_btns = " ".join(
            f'<button class="pg-btn" type="button" onclick="showPage({p}, {idx})">p.{p}</button>'
            for p in (v.cited_pages or [])
        ) or '<span class="muted">No cited page</span>'
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
        first_p = v.cited_pages[0] if v.cited_pages else 0
        trace_layers = (v.retrieval_trace or {}).get("layers") if isinstance(v.retrieval_trace, dict) else None
        trace_html = ""
        if isinstance(trace_layers, dict) and trace_layers:
            parts = [f"{k}: {vals[:5]}" for k, vals in trace_layers.items() if vals]
            if parts:
                trace_html = f'<div class="trace">Trace: {html.escape(" -> ".join(parts))}</div>'

        cards_html.append(
            f"""<article class="card" data-status="{html.escape(v.status)}" onclick="showPage({first_p}, {idx})">
              <div class="card-head">
                <span class="badge {badge_cls}">{html.escape(v.status.upper())}</span>
                <strong class="card-title">{html.escape(v.title)}</strong>
              </div>
              <div class="meta">ID: <code>{html.escape(v.item_id)}</code> · Rule: <code>{html.escape(v.rule_trace or "N/A")}</code> · Pages: {page_btns}</div>
              <p class="reason">{html.escape(v.answer or v.reason)}</p>
              {calc_html}
              {q_html}
              {trace_html}
            </article>"""
        )

    filter_btns = [f'<button class="flt active" onclick="filterStatus(\'all\', this)">All ({len(verdicts)})</button>']
    for st in ("fail", "needs_review", "abstain", "pass", "answered"):
        if counts.get(st):
            filter_btns.append(
                f'<button class="flt" onclick="filterStatus(\'{st}\', this)">{st.upper()} ({counts[st]})</button>'
            )

    page_map_json = json.dumps(page_images)
    first_embedded = next(iter(page_images.keys()), 0)

    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; margin: 0; line-height: 1.55; }}
    header {{ background: #0f172a; color: #fff; padding: 16px 24px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; justify-content: space-between; }}
    header h1 {{ margin: 0; font-size: 18px; letter-spacing: -0.01em; }}
    .filters {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .flt {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; border-radius: 999px; padding: 5px 12px; font-size: 12px; cursor: pointer; }}
    .flt.active {{ background: #2563eb; color: #fff; border-color: #3b82f6; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(340px, 0.92fr); height: calc(100vh - 64px); }}
    .left {{ overflow-y: auto; padding: 18px 20px; border-right: 1px solid #e2e8f0; }}
    .right {{ overflow-y: auto; padding: 18px 20px; background: #f1f5f9; display: flex; flex-direction: column; align-items: center; }}
    .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; cursor: pointer; transition: border-color .15s; }}
    .card:hover, .card.selected {{ border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.08); }}
    .card-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .card-title {{ font-size: 15px; }}
    .meta {{ color: #64748b; font-size: 12px; margin-bottom: 8px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }}
    .badge {{ font-size: 10.5px; font-weight: 700; padding: 2px 8px; border-radius: 5px; flex-shrink: 0; }}
    .badge-pass {{ background: #dcfce7; color: #166534; }}
    .badge-fail {{ background: #fee2e2; color: #991b1b; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; }}
    .pg-btn {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 5px; padding: 1px 7px; font-size: 11.5px; cursor: pointer; font-weight: 600; }}
    .pg-btn:hover {{ background: #dbeafe; }}
    .reason {{ margin: 4px 0 8px; font-size: 14px; color: #334155; }}
    .calc {{ font-size: 12.5px; background: #eff6ff; color: #1e3a8a; padding: 5px 9px; border-radius: 6px; margin-bottom: 6px; }}
    .quote {{ margin: 6px 0 0; padding: 7px 10px; border-left: 3px solid #94a3b8; background: #f8fafc; font-size: 12.5px; color: #475569; }}
    .trace {{ margin-top: 6px; font-size: 11.5px; color: #64748b; font-family: ui-monospace, Menlo, monospace; }}
    .qv {{ font-weight: 600; color: #0f172a; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }}
    .viewer-bar {{ width: 100%; max-width: 680px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-size: 13px; color: #334155; font-weight: 600; }}
    #pageImg {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 16px rgba(15,23,42,0.12); background: #fff; display: none; }}
    #pageEmpty {{ color: #64748b; font-size: 13.5px; margin-top: 60px; text-align: center; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; height: auto; }} }}
  </style>
</head>
<body>
  <header>
    <h1>⚓ {html.escape(title)}</h1>
    <div class="filters">{"".join(filter_btns)}</div>
  </header>
  <div class="grid">
    <section class="left">{"".join(cards_html)}</section>
    <aside class="right">
      <div class="viewer-bar"><span id="viewerTitle">Original Page Evidence Preview</span></div>
      <img id="pageImg" alt="Original PDF page preview" />
      <div id="pageEmpty">Click any finding or page badge on the left to inspect the original PDF page.</div>
    </aside>
  </div>
  <script>
    const PAGES = {page_map_json};
    function showPage(p, cardIdx) {{
      document.querySelectorAll('.card').forEach((el, i) => el.classList.toggle('selected', i === cardIdx));
      const img = document.getElementById('pageImg');
      const empty = document.getElementById('pageEmpty');
      const title = document.getElementById('viewerTitle');
      if (p && PAGES[String(p)]) {{
        img.src = 'data:image/png;base64,' + PAGES[String(p)];
        img.style.display = 'block';
        empty.style.display = 'none';
        title.textContent = 'Original PDF — Physical Page ' + p;
      }} else {{
        img.style.display = 'none';
        empty.style.display = 'block';
        empty.textContent = p ? ('Page ' + p + ' image not embedded in this report.') : 'No cited page for this item.';
        title.textContent = 'Original Page Evidence Preview';
      }}
    }}
    function filterStatus(st, btn) {{
      document.querySelectorAll('.flt').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.card').forEach(c => {{
        c.style.display = (st === 'all' || c.dataset.status === st) ? 'block' : 'none';
      }});
    }}
    if ({first_embedded}) {{ showPage({first_embedded}, 0); }}
  </script>
</body>
</html>
"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc_html, encoding="utf-8")
    return p
