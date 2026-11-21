"""Unit tests for stdlib OOXML merged-cell table grid resolution (w:vMerge + w:gridSpan)."""
from zipfile import ZipFile

from anchordoc.docbase import extract_docx_tables, resolve_table_rows


def _write_synthetic_docx(path):
    # A 3-row x 3-col table where col 1 merges rows 2..3 vertically (w:vMerge),
    # and row 1 col 2..3 merges horizontally (w:gridSpan=2), plus a deleted w:del run.
    doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:tbl>
          <w:tblGrid><w:gridCol/><w:gridCol/><w:gridCol/></w:tblGrid>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc>
            <w:tc>
              <w:tcPr><w:gridSpan w:val="2"/></w:tcPr>
              <w:p><w:r><w:t>Compliance Details</w:t></w:r><w:del><w:r><w:t>DELETED</w:t></w:r></w:del></w:p>
            </w:tc>
          </w:tr>
          <w:tr>
            <w:tc>
              <w:tcPr><w:vMerge w:val="restart"/></w:tcPr>
              <w:p><w:r><w:t>Server Unit A</w:t></w:r></w:p>
            </w:tc>
            <w:tc><w:p><w:r><w:t>CPU >= 32 cores</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Compliant</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc>
              <w:tcPr><w:vMerge/></w:tcPr>
              <w:p/>
            </w:tc>
            <w:tc><w:p><w:r><w:t>RAM >= 128GB</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Compliant</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>"""
    with ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc_xml)


def test_ooxml_vmerge_and_gridspan_inheritance(tmp_path):
    docx_path = tmp_path / "merged.docx"
    _write_synthetic_docx(docx_path)

    tables = extract_docx_tables(docx_path)
    assert len(tables) == 1
    rows = resolve_table_rows(tables[0])
    assert len(rows) == 3
    assert rows[0] == ["Item", "Compliance Details", "Compliance Details"]
    assert "DELETED" not in rows[0][1]
    assert rows[1] == ["Server Unit A", "CPU >= 32 cores", "Compliant"]
    # Vertical merge inherited cleanly — never false-blank!
    assert rows[2] == ["Server Unit A", "RAM >= 128GB", "Compliant"]
