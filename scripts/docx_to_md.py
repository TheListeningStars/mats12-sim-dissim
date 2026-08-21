"""Render a .docx paper as GitHub-flavored Markdown so it displays in the browser.

The paper is written/edited as a Word document (scripts/make_paper.py builds the
first draft from the verified artifacts). GitHub will not render .docx inline, so
this converts it to docs/PAPER.md, which GitHub does render. The .docx stays in
the repo as the downloadable original.

Usage:  python scripts/docx_to_md.py "docs/<paper>.docx" docs/PAPER.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


def escape(text: str) -> str:
    """Escape the few characters GitHub would otherwise read as markup."""
    return re.sub(r"([*_`\[\]<>|])", r"\\\1", text)


def render_runs(par: Paragraph) -> str:
    """Paragraph text with bold/italic preserved, whitespace runs kept clean."""
    out = []
    for run in par.runs:
        text = escape(run.text)
        if not text.strip():
            out.append(run.text)
            continue
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        core = text.strip()
        if run.bold:
            core = f"**{core}**"
        if run.italic:
            core = f"*{core}*"
        out.append(f"{lead}{core}{trail}")
    return "".join(out).strip()


def is_list(par: Paragraph) -> bool:
    return par._p.find(qn("w:pPr") + "/" + qn("w:numPr")) is not None


def render_table(tbl: Table) -> list[str]:
    rows = [[" ".join(c.text.split()) or " " for c in r.cells] for r in tbl.rows]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [" "] * (width - len(r)) for r in rows]
    head, *body = rows
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return lines


def convert(src: Path) -> str:
    """Markdown body. The first heading becomes the H1 title; every other heading
    is demoted one level so the section list nests under it on GitHub."""
    doc = docx.Document(str(src))
    lines: list[str] = []
    seen_title = False
    in_list = False
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            par = Paragraph(child, doc)
            text = render_runs(par)
            if not text:
                continue
            style = par.style.name or ""
            listed = is_list(par)
            if in_list and not listed:
                lines.append("")
            in_list = listed
            if style.startswith("Heading"):
                level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
                level = 1 if not seen_title else min(level + 1, 6)
                seen_title = True
                lines += ["", "#" * level + " " + text, ""]
            elif listed:
                lines.append(f"- {text}")
            else:
                lines += [text, ""]
        elif child.tag == qn("w:tbl"):
            if in_list:
                in_list = False
            lines += [""] + render_table(Table(child, doc)) + [""]

    lines += [
        "",
        "---",
        "",
        f"*Rendered from [`{src.name}`]({src.name}) by `scripts/docx_to_md.py`. "
        "Every number traces to `docs/NUMBERS.md` and the artifacts under `results/`.*",
    ]
    md = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(convert(src), encoding="utf-8")
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
