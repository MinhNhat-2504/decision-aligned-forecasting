"""Convert the reviewed PAPER_MANUSCRIPT.md into a submission-ready LaTeX file.

The converter is intentionally small and deterministic: it supports the Markdown
constructs used by this manuscript and preserves display math, tables, figures, and
numbered references.  Re-run it after editing the manuscript.
"""
from __future__ import annotations

import re
from pathlib import Path


import os

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PAPER_MANUSCRIPT.md"

# JOURNAL=ijaa  -> mau amsart cua International Journal of Analysis and Applications
# JOURNAL=generic (mac dinh) -> mau article dung chung
JOURNAL = os.environ.get("JOURNAL", "generic").lower()
TARGET = ROOT / ("paper_ijaa.tex" if JOURNAL == "ijaa" else "paper.tex")
CITE = r"\cite" if JOURNAL == "ijaa" else r"\citep"


def tex_escape(text: str) -> str:
    """Escape text while leaving Unicode prose and already-normal math readable."""
    replacements = {
        "−": "-", "–": "--", "—": "---", "→": "->",
        "×": "x", "±": "+/-", "≤": "<=", "≥": ">=",
        "τ": "tau", "θ": "theta", "∈": "in",
        "“": "``", "”": "''", "’": "'", "…": r"\ldots{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("\\", r"\textbackslash{}").replace("&", r"\&")
    text = text.replace("%", r"\%").replace("#", r"\#")
    text = text.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return text


def inline(text: str, citations: bool = True) -> str:
    """Convert inline Markdown, keeping \(...\) math intact."""
    math_parts: list[str] = []

    def hold_math(match: re.Match[str]) -> str:
        token = f"@@MATH{len(math_parts)}@@"
        math_parts.append(match.group(0))
        return token

    text = re.sub(r"\\\(.*?\\\)", hold_math, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: rf"\href{{{m.group(2)}}}{{{tex_escape(m.group(1))}}}", text)
    if citations:
        text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]",
                      lambda m: CITE + "{" + ",".join(
                          f"ref{n.strip()}" for n in m.group(1).split(",")) + "}", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", text)
    text = text.replace("`", "")
    # Protect TeX commands introduced by formatting and hyperlinks while escaping prose.
    protected: list[str] = []

    def hold_command(match: re.Match[str]) -> str:
        token = f"@@CMD{len(protected)}@@"
        protected.append(match.group(0))
        return token

    # citep phai dung truoc cite trong nhanh luan phien, neu khong \citep{} se bi
    # khop nham thanh \cite roi con lai chu 'p' lac ra ngoai.
    text = re.sub(r"\\(?:textbf|emph|href|citep|cite)\{[^{}]*\}(?:\{[^{}]*\})?",
                  hold_command, text)
    text = tex_escape(text)
    for i, value in enumerate(protected):
        if value.startswith((r"\textbf{", r"\emph{")):
            value = (value.replace("%", r"\%")
                           .replace("&", r"\&")
                           .replace("_", r"\_"))
        text = text.replace(f"@@CMD{i}@@", value)
    for i, value in enumerate(math_parts):
        text = text.replace(f"@@MATH{i}@@", value)
    return text


def column_spec(header: list[str]) -> str:
    specs = []
    for cell in header:
        specs.append("r" if cell.strip().endswith(":---") or cell.strip().startswith("—")
                     else "l")
    return "".join(specs) or "l"


def table(lines: list[str], caption: str = "") -> list[str]:
    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-+:?", x.replace(" ", "")) for x in cells):
            continue
        rows.append(cells)
    if not rows:
        return []
    n = max(len(row) for row in rows)
    align = "l" + "r" * (n - 1)
    out = [r"\begin{table}[t]", r"\centering"]
    # Caption dat TRUOC bang: quy uoc cua amsart va cua phan lon tap chi.
    if caption:
        out.append(r"\caption{" + inline(caption, citations=False) + "}")
    out.append(r"\small")
    # Co gian CHI KHI bang rong hon khung chu. amsart co khung hep hon article nen
    # nguong n>=5 cot khong du: bang 4 cot van tran le. Idiom duoi day do be rong
    # thuc te roi moi quyet dinh, nen khong lam chu nho di mot cach khong can thiet.
    out.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%")
    out += [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    for i, row in enumerate(rows):
        row = row + [""] * (n - len(row))
        cells = [inline(x, citations=False) for x in row]
        if i == 1:
            out.append(" \\midrule")
        out.append(" & ".join(cells) + r" \\ ")
    out += [r"\bottomrule", r"\end{tabular}", r"}"]
    out.append(r"\end{table}")
    return out


def reference_entries(lines: list[str]) -> dict[int, str]:
    entries: dict[int, str] = {}
    current = None
    for line in lines:
        m = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if m:
            current = int(m.group(1))
            entries[current] = m.group(2).strip()
        elif current is not None and line.strip():
            entries[current] += " " + line.strip()
    return entries


def render_reference(value: str) -> str:
    urls: list[str] = []

    def _grab(m: "re.Match[str]") -> str:
        # Cau tham khao ket thuc bang dau cham; neu nuot no vao \url{} thi lien
        # ket DOI tro thanh sai. Tach dau cau cuoi ra ngoai.
        url = m.group(0)
        trail = ""
        while url and url[-1] in ".,;:":
            trail = url[-1] + trail
            url = url[:-1]
        urls.append(url)
        return f"@@URL{len(urls) - 1}@@" + trail

    value = re.sub(r"https?://\S+", _grab, value)
    rendered = inline(value, citations=False)
    for i, url in enumerate(urls):
        rendered = rendered.replace(f"@@URL{i}@@", rf"\url{{{url}}}")
    return rendered


PREAMBLE_IJAA = r"""\documentclass{amsart}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{amssymb, amscd, amsmath, url}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\numberwithin{equation}{section}

\begin{document}

%===== Dang ngan trong [] la TIEU DE CHAY. amsart mac dinh dung ca tieu de day du
%===== lam tieu de chay, va tieu de nay qua dai nen tran le 168pt tren moi trang.
\title[Forecast Value Is a Property of the Decision]{Forecast Value Is a Property of
the Decision: Calibrated Quantile Demand Forecasting for Urban Last-Mile Capacity
and Routing}

%===== MSC 2020: 62M20 du bao chuoi thoi gian (chinh); 90B06 van tai/logistics;
%===== 62P30 thong ke ung dung trong ky thuat va cong nghiep
\subjclass[2020]{62M20, 90B06, 62P30}

%===== PHAI khop voi dong **Keywords:** trong PAPER_MANUSCRIPT.md.
%===== Hai ben tung troi lech nhau (bai co 8, preamble co 7); audit nay kiem tra.
\keywords{service level; probabilistic forecasting; conformal prediction; newsvendor;
capacity planning; last-mile decision support; forecast value; spatio-temporal graph
neural networks}

\author{\bfseries Kien Pham Xuan$^{1}$, Nhat Trinh Ngoc Minh$^{2}$ and
Linh Nguyen Thi My$^{2,*}$}

\address{$^1$Faculty of Management Information Systems, Ho Chi Minh University of
Banking, Ho Chi Minh City, Vietnam; \textnormal{kienpx@hub.edu.vn}}
\address{$^2$Faculty of Information Technology, School of Technology, Van Lang
University, Ho Chi Minh City, Vietnam;
\textnormal{nhat.2474802010280@vanlanguni.vn}}
\address{$^*$Corresponding author: \textnormal{linh.ntm@vlu.edu.vn}}
\maketitle
"""

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{url}
\usepackage[numbers]{natbib}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\setlist{nosep}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\title{Forecast Value Is a Property of the Decision:\\
Calibrated Quantile Demand Forecasting for Urban Last-Mile Capacity and Routing}
\author{Kien Pham Xuan\thanks{Faculty of Management Information Systems,
Ho Chi Minh University of Banking, Ho Chi Minh City, Vietnam.
kienpx@hub.edu.vn}
\and Nhat Trinh Ngoc Minh\thanks{Faculty of Information Technology, School of
Technology, Van Lang University, Ho Chi Minh City, Vietnam.
nhat.2474802010280@vanlanguni.vn}
\and Linh Nguyen Thi My\thanks{Faculty of Information Technology, School of
Technology, Van Lang University, Ho Chi Minh City, Vietnam.
Corresponding author: linh.ntm@vlu.edu.vn}}
\date{}
\begin{document}
\maketitle
"""


def strip_html_comments(text: str) -> str:
    """
    Bo chu thich HTML cua Markdown.

    Neu de lot, chuoi '-->' se di thang vao LaTeX va render thanh '--¿' o che do
    van ban, tao ra ky tu la giua bai. Chu thich la ghi chu noi bo, khong thuoc
    ban thao nop di.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def convert() -> None:
    lines = strip_html_comments(
        SOURCE.read_text(encoding="utf-8")).splitlines()
    out = [PREAMBLE_IJAA if JOURNAL == "ijaa" else PREAMBLE]
    i = 0
    in_abstract = False
    in_references = False
    appendix_started = False
    pending_caption = ""
    refs: list[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped == "## References":
            in_references = True
            if in_abstract:
                out.append(r"\end{abstract}")
                in_abstract = False
            out.append(r"\begin{thebibliography}{99}")
            refs = lines[i + 1:]
            break
        if stripped.startswith("# "):
            i += 1
            continue
        if stripped == "## Abstract":
            out.append(r"\begin{abstract}")
            in_abstract = True
            i += 1
            continue
        if stripped.startswith("## "):
            if in_abstract:
                out.append(r"\end{abstract}")
                in_abstract = False
            heading = re.sub(r"^\d+\.\s+", "", stripped[3:])
            # Chuyen sang danh so phu luc (A, B, ...) o muc dau tien bat dau bang
            # "Appendix"; LaTeX doi \appendix duoc phat dung mot lan.
            if heading.startswith("Appendix") and not appendix_started:
                out.append(r"\appendix")
                appendix_started = True
                heading = re.sub(r"^Appendix\s+[A-Z]\.?\s*", "", heading)
            out.append("\\section{" + inline(heading, citations=False) + "}")
            i += 1
            continue
        if stripped.startswith("### "):
            heading = re.sub(r"^\d+\.\d+\s+", "", stripped[4:])
            out.append("\\subsection{" + inline(heading, citations=False) + "}")
            i += 1
            continue
        if stripped == r"\[":
            equation = []
            i += 1
            while i < len(lines) and lines[i].strip() != r"\]":
                equation.append(lines[i])
                i += 1
            out.extend([r"\[", *equation, r"\]"])
            i += 1
            continue
        if stripped.startswith("Table: "):
            # Quy uoc: dong "Table: ..." ngay truoc bang la caption cua bang do.
            pending_caption = stripped[len("Table: "):].strip()
            i += 1
            continue
        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.extend(table(block, pending_caption))
            pending_caption = ""
            continue
        if stripped.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if match:
                caption, path = match.groups()
                out += [r"\begin{figure}[t]", r"\centering",
                        rf"\includegraphics[width=0.88\linewidth]{{{path}}}",
                        rf"\caption{{{inline(caption, citations=False)}}}",
                        r"\end{figure}"]
            i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                i += 1
                continuation = []
                while i < len(lines) and lines[i].strip():
                    nxt = lines[i].strip()
                    if (re.match(r"^\d+\.\s+", nxt)
                            or nxt.startswith(("#", "|", "![", "- "))):
                        break
                    continuation.append(nxt)
                    i += 1
                out.append(r"\item " + inline(" ".join([item] + continuation)))
            out.append(r"\end{enumerate}")
            continue
        if stripped.startswith("- "):
            out.append(r"\begin{itemize}")
            while i < len(lines) and lines[i].strip().startswith("- "):
                out.append(r"\item " + inline(lines[i].strip()[2:]))
                i += 1
            out.append(r"\end{itemize}")
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt == "---" or nxt.startswith(("#", "|", "![", "- "))
                    or nxt == r"\[" or re.match(r"^\d+\.\s+", nxt)):
                break
            paragraph.append(nxt)
            i += 1
        text = " ".join(paragraph)
        if text.startswith("**Keywords:**"):
            if JOURNAL == "ijaa":
                # IJAA khai bao tu khoa qua macro \keywords truoc \maketitle,
                # khong dat trong abstract.
                continue
            text = text.replace("**Keywords:**", r"\textbf{Keywords:}", 1)
        # Dong trong la BAT BUOC: trong LaTeX hai dong lien nhau van thuoc cung mot
        # doan van. Neu thieu, toan bo cac doan cua mot muc se dinh thanh mot khoi.
        out.append(inline(text))
        out.append("")

    entries = reference_entries(refs)
    for number in sorted(entries):
        out.append(rf"\bibitem{{ref{number}}} " + render_reference(entries[number]))
    out += [r"\end{thebibliography}", r"\end{document}"]
    TARGET.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    convert()
