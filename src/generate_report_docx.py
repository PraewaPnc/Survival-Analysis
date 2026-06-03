"""Generate the full bilingual Word report (.docx) from report_full.md.

This script parses the assembled markdown report (``report_full.md`` at the project
root) and renders it to ``report_full.docx`` with proper Thai-capable fonts,
embedded figures (from outputs/figures/), markdown tables and styled headings.

It is a generic markdown -> docx converter supporting:
  - headings:        ``#`` (title/part), ``##`` (section), ``###`` (subsection)
  - images:          ``![caption](path)``  (path relative to project root)
  - tables:          GitHub-style pipe tables
  - bullet lists:    lines starting with ``- `` or ``* ``
  - inline bold:     ``**text**``
  - fenced code:     ```` ``` ```` blocks rendered as monospace
  - block quotes:    ``> text``

Usage:
    python3 src/generate_report_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

ROOT = Path(__file__).resolve().parent.parent
SRC_MD = ROOT / "report_full.md"
OUT_DOCX = ROOT / "report_full.docx"

# Thai-capable font available on macOS; swap to "TH Sarabun New" if installed.
THAI_FONT = "Thonburi"
MONO_FONT = "Menlo"
BODY_SIZE = 11.5
ACCENT = RGBColor(0x1F, 0x3B, 0x66)
GREY = RGBColor(0x44, 0x44, 0x44)

IMG_RE = re.compile(r"^!\[(?P<cap>.*?)\]\((?P<path>.*?)\)\s*$")
LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)")


# --------------------------------------------------------------------------- #
# Low-level styling helpers (preserved from the original generator)
# --------------------------------------------------------------------------- #
def set_run_font(run, size: float = BODY_SIZE, bold: bool = False,
                 color: RGBColor | None = None, font: str = THAI_FONT) -> None:
    """Apply font name (incl. complex-script for Thai), size, weight, colour."""
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), font)


def _clean_inline(text: str) -> str:
    """Strip markdown link syntax down to its visible text."""
    return LINK_RE.sub(lambda m: m.group("text"), text)


def add_rich_run(paragraph, text: str, size: float = BODY_SIZE,
                 base_bold: bool = False, color: RGBColor | None = None,
                 font: str = THAI_FONT) -> None:
    """Add text to a paragraph, honouring **bold** and `code` inline spans."""
    text = _clean_inline(text)
    # Split on bold markers first; odd segments are bold.
    for i, chunk in enumerate(text.split("**")):
        if chunk == "":
            continue
        bold = base_bold or (i % 2 == 1)
        # Handle inline code spans within the chunk.
        for j, piece in enumerate(chunk.split("`")):
            if piece == "":
                continue
            run = paragraph.add_run(piece)
            if j % 2 == 1:
                set_run_font(run, size=size, bold=bold, color=color, font=MONO_FONT)
            else:
                set_run_font(run, size=size, bold=bold, color=color, font=font)


def add_par(doc, text: str = "", size: float = BODY_SIZE, bold: bool = False,
            align=None, color: RGBColor | None = None, space_after: float = 6,
            font: str = THAI_FONT, rich: bool = True):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.3
    if text:
        if rich:
            add_rich_run(p, text, size=size, base_bold=bold, color=color, font=font)
        else:
            run = p.add_run(text)
            set_run_font(run, size=size, bold=bold, color=color, font=font)
    return p


def add_heading(doc, text: str, level: int = 1):
    sizes = {0: 17, 1: 14.5, 2: 12.5}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level <= 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(_clean_inline(text))
    set_run_font(run, size=sizes.get(level, 12.5), bold=True, color=ACCENT)
    return p


def add_table(doc, headers: list[str], rows: list[list[str]]):
    n_cols = len(headers)
    header_size = 9.0 if n_cols > 6 else 10.0
    body_size = 8.5 if n_cols > 6 else 9.5
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_rich_run(hdr[i].paragraphs[0], h, size=header_size, base_bold=True)
    for row in rows:
        cells = table.add_row().cells
        for i in range(n_cols):
            val = row[i] if i < len(row) else ""
            cells[i].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            )
            add_rich_run(cells[i].paragraphs[0], val, size=body_size)
    return table


def add_figure(doc, rel_path: str, caption: str, width: float = 6.0):
    path = (ROOT / rel_path).resolve()
    if not path.exists():
        print(f"  [warn] figure not found, skipping: {rel_path}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    try:
        p.add_run().add_picture(str(path), width=Inches(width))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not embed {rel_path}: {exc}")
        return
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        add_rich_run(cap, caption, size=9.5, color=GREY)


def add_code_block(doc, lines: list[str]):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run("\n".join(lines))
    set_run_font(run, size=9.0, font=MONO_FONT, color=RGBColor(0x22, 0x22, 0x22))


def bullet(doc, text: str, size: float = BODY_SIZE):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.3
    add_rich_run(p, text, size=size)
    return p


# --------------------------------------------------------------------------- #
# Markdown parsing
# --------------------------------------------------------------------------- #
def _split_table_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def _is_table_sep(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}.*$", line)) and "|" in line \
        and set(line.strip()) <= set("|:- ")


def build() -> None:
    if not SRC_MD.exists():
        raise SystemExit(f"source markdown not found: {SRC_MD}")
    lines = SRC_MD.read_text(encoding="utf-8").splitlines()

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = THAI_FONT
    normal.font.size = Pt(BODY_SIZE)
    normal.element.rPr.rFonts.set(qn("w:cs"), THAI_FONT)

    n = len(lines)
    i = 0
    para_buf: list[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            add_par(doc, " ".join(para_buf), size=BODY_SIZE)
            para_buf = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # blank line -> paragraph break
        if stripped == "":
            flush_para()
            i += 1
            continue

        # fenced code block
        if stripped.startswith("```"):
            flush_para()
            block: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            if block:
                add_code_block(doc, block)
            continue

        # image
        m = IMG_RE.match(stripped)
        if m:
            flush_para()
            add_figure(doc, m.group("path").strip(), m.group("cap").strip())
            i += 1
            continue

        # headings
        if stripped.startswith("### "):
            flush_para()
            add_heading(doc, stripped[4:].strip(), level=2)
            i += 1
            continue
        if stripped.startswith("## "):
            flush_para()
            add_heading(doc, stripped[3:].strip(), level=1)
            i += 1
            continue
        if stripped.startswith("# "):
            flush_para()
            add_heading(doc, stripped[2:].strip(), level=0)
            i += 1
            continue

        # horizontal rule
        if re.match(r"^(\*\s*){3,}$", stripped) or re.match(r"^(-\s*){3,}$", stripped) \
                or re.match(r"^(_\s*){3,}$", stripped):
            flush_para()
            i += 1
            continue

        # table: current line has pipes and next line is a separator
        if "|" in stripped and i + 1 < n and _is_table_sep(lines[i + 1]):
            flush_para()
            headers = _split_table_row(stripped)
            i += 2  # skip header + separator
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip() != "":
                if _is_table_sep(lines[i]):
                    i += 1
                    continue
                rows.append(_split_table_row(lines[i]))
                i += 1
            add_table(doc, headers, rows)
            continue

        # block quote
        if stripped.startswith(">"):
            flush_para()
            add_par(doc, stripped.lstrip("> ").strip(), size=BODY_SIZE,
                    color=GREY, font=THAI_FONT)
            i += 1
            continue

        # bullet list
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            bullet(doc, stripped[2:].strip())
            i += 1
            continue

        # numbered list -> render as a normal indented paragraph
        if re.match(r"^\d+\.\s+", stripped):
            flush_para()
            add_par(doc, stripped, size=BODY_SIZE)
            i += 1
            continue

        # default: accumulate prose
        para_buf.append(stripped)
        i += 1

    flush_para()

    doc.save(str(OUT_DOCX))
    print(f"saved -> {OUT_DOCX}")


if __name__ == "__main__":
    build()
