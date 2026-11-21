"""Zero-dependency OOXML DOCX merged-cell table grid extractor (w:vMerge + w:gridSpan).

Preserves physical cells alongside a logical 2D grid of cell anchor IDs so consumers
never lose vertical-merge inherited values (fixing fragmented rows & false-blank responses).
Uses Python standard library only (xml.etree.ElementTree + zipfile).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _int_prop(element: ET.Element, path: str, default: int = 0) -> int:
    prop = element.find(path)
    return int(prop.get(W + "val", str(default))) if prop is not None else default


def _paragraph_text(p: ET.Element) -> str:
    def walk(node: ET.Element) -> str:
        # Exclude deleted tracked changes
        if node.tag in {W + "del", W + "moveFrom"}:
            return ""
        if node.tag == W + "t":
            return node.text or ""
        if node.tag == W + "tab":
            return "\t"
        if node.tag in {W + "br", W + "cr"}:
            return "\n"
        if node.tag == W + "sym":
            value = node.get(W + "char", "")
            return chr(int(value, 16)) if value else ""
        return "".join(walk(child) for child in node)

    return walk(p)


def _children(element: ET.Element, tag: str):
    for child in element:
        if child.tag == W + tag:
            yield child
        elif child.tag in {W + "sdt", W + "sdtContent", W + "customXml"}:
            yield from _children(child, tag)


def _content_blocks(element: ET.Element):
    for child in element:
        if child.tag in {W + "p", W + "tbl"}:
            yield child
        elif child.tag in {W + "sdt", W + "sdtContent", W + "customXml"}:
            yield from _content_blocks(child)


def _cell_text(tc: ET.Element) -> tuple[str, list[str]]:
    parts: list[str] = []
    for child in _content_blocks(tc):
        if child.tag == W + "p":
            parts.append(_paragraph_text(child))
        else:
            nested = _parse_table(child, -1)
            for row in resolve_table_rows(nested):
                parts.append(" | ".join(row))
    images = sorted(
        {
            n.get(R + "embed") or n.get(R + "id")
            for n in tc.iter()
            if n.get(R + "embed") or n.get(R + "id")
        }
    )
    return "\n".join(parts).strip(), images


def _parse_table(tbl: ET.Element, index: int) -> dict[str, Any]:
    width = len(tbl.findall(W + "tblGrid/" + W + "gridCol"))
    cells: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    grid: list[list[str | None]] = []
    active: dict[tuple[int, int], str] = {}

    for r_idx, tr in enumerate(_children(tbl, "tr")):
        before = _int_prop(tr, W + "trPr/" + W + "gridBefore")
        after = _int_prop(tr, W + "trPr/" + W + "gridAfter")
        col = before
        row_ids: list[str | None] = [None] * before
        next_active: dict[tuple[int, int], str] = {}

        for c_phys, tc in enumerate(_children(tr, "tc")):
            span = _int_prop(tc, W + "tcPr/" + W + "gridSpan", 1)
            if span < 1 or span > 2048:
                raise ValueError(f"Invalid OOXML gridSpan: {span}")
            vm = tc.find(W + "tcPr/" + W + "vMerge")
            mode = vm.get(W + "val", "continue") if vm is not None else ""
            text, images = _cell_text(tc)
            key = (col, span)

            if mode == "continue" and key in active:
                cid = active[key]
                anchor = by_id[cid]
                anchor["rowspan"] += 1
                if text and text not in anchor["text"]:
                    anchor["text"] = f"{anchor['text']}\n{text}".strip()
                next_active[key] = cid
            else:
                cid = f"t{index}.r{r_idx+1}c{col+1}"
                cell = {
                    "id": cid,
                    "row": r_idx + 1,
                    "col": col + 1,
                    "rowspan": 1,
                    "colspan": span,
                    "text": text,
                    "images": images,
                }
                cells.append(cell)
                by_id[cid] = cell
                if mode == "restart":
                    next_active[key] = cid

            row_ids.extend([cid] * span)
            col += span

        row_ids.extend([None] * after)
        width = max(width, len(row_ids))
        grid.append(row_ids)
        active = next_active

    # Pad ragged rows to uniform width
    for row_ids in grid:
        if len(row_ids) < width:
            row_ids.extend([None] * (width - len(row_ids)))

    return {
        "table_index": index,
        "rows": len(grid),
        "cols": width,
        "cells": cells,
        "grid": grid,
    }


def resolve_table_rows(table: dict[str, Any]) -> list[list[str]]:
    """Resolve logical 2D text rows with vertical & horizontal merged values inherited."""
    by_id = {c["id"]: c["text"] for c in table.get("cells") or []}
    return [[by_id.get(cid, "") if cid else "" for cid in row] for row in (table.get("grid") or [])]


def extract_docx_tables(docx_path: str | Path) -> list[dict[str, Any]]:
    """Extract all top-level tables from a DOCX file with full merged-cell topology."""
    p = Path(docx_path)
    with ZipFile(p) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(W + "body")
    if body is None:
        return []
    tables: list[dict[str, Any]] = []
    idx = 0
    for block in _content_blocks(body):
        if block.tag == W + "tbl":
            idx += 1
            tables.append(_parse_table(block, idx))
    return tables
