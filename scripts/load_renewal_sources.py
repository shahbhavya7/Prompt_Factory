"""Read-only extraction of the renewal campaign's three source artifacts.

Phase-1 scope only. This script parses `data/renewal/`'s CSV, PDF and HTML
into structured, inspectable form and flags the known gaps in them — it
authors NO rule text, answer text or caller phrasing of its own. Turning this
data into actual KB rules is Phases 2-5's job; those phases read FROM what
this script surfaces, they do not invent replacements for what it flags.

The three files, referenced here by relative path, are the only content
source for the renewal build:
  data/renewal/Maya_knowledge_base_1_10_Knowledge_base_.csv   126 Q&As, tiers
  data/renewal/Script_simplified.pdf                          Paths A/B/C
  data/renewal/Coverage_Renewal_Call_Flow_1.html              the flow graph
                                                               (mermaid is
                                                               embedded in the
                                                               HTML source)

Run:  python scripts/load_renewal_sources.py
"""

from __future__ import annotations

import csv
import html as html_module
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "renewal"
CSV_PATH = DATA_DIR / "Maya_knowledge_base_1_10_Knowledge_base_.csv"
PDF_PATH = DATA_DIR / "Script_simplified.pdf"
HTML_PATH = DATA_DIR / "Coverage_Renewal_Call_Flow_1.html"
MERMAID_OUT = DATA_DIR / "Coverage_Renewal_Call_Flow_1.mmd"

# Confirmed by inspection, not assumed: the em-dashes in the source sheet
# decode as cp1252, and plain utf-8 raises on the very first data row.
CSV_ENCODING = "cp1252"


def load_kb_rows() -> list[dict]:
    """Every Q&A row in the CSV, keyed by its header.

    Section-divider rows are excluded — the sheet mixes them into the ID
    column with no Tier of their own (e.g. "THE LETTER AND WHAT IT IS — MC
    216 page 1"), so "has a Tier" is what actually distinguishes a real Q&A
    row from a heading. 142 raw rows split into 126 Q&A rows + 16 headings
    this way, matching the 126 the task description names.
    """
    with open(CSV_PATH, newline="", encoding=CSV_ENCODING) as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
    header = rows[header_idx]
    data_rows = [r for r in rows[header_idx + 1:] if r and r[0].strip()]
    return [
        dict(zip(header, r))
        for r in data_rows
        if len(r) > 2 and r[2].strip()
    ]


def extract_mermaid() -> str:
    """The flow diagram's mermaid SOURCE, as authored — not the rendered
    image. Unescapes the HTML entities the export left in the <pre> block
    (&gt; &quot; &mdash; ...); nothing else about it is touched."""
    html_text = HTML_PATH.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<pre class="mermaid">(.*?)</pre>', html_text, re.S)
    if not m:
        raise SystemExit(f'{HTML_PATH}: no <pre class="mermaid"> block found')
    return html_module.unescape(m.group(1)).strip()


def extract_script_pages() -> list[str]:
    """Script_simplified.pdf, one string per page, in order."""
    from pypdf import PdfReader

    reader = PdfReader(str(PDF_PATH))
    return [(page.extract_text() or "") for page in reader.pages]


def main() -> int:
    rows = load_kb_rows()
    tiers: dict[str, int] = {}
    for r in rows:
        tiers[r["Tier"]] = tiers.get(r["Tier"], 0) + 1
    print(f"[csv]  {CSV_PATH.relative_to(REPO_ROOT)}: {len(rows)} Q&A rows, tiers={tiers}")

    # KNOWN GAP, found by inspection: KB-LTR-04 appears twice with genuinely
    # different content (never received the form vs. received-then-lost it).
    # Not resolved here — which row is authoritative is a question for the
    # script owner, not something to guess.
    ids = [r["ID"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        print(f"[csv]  KNOWN GAP — duplicate ID(s) with different content, not de-duplicated here: {dupes}")
        print("        ambiguous which row is authoritative — flag to the script owner before Phase 2 uses it")

    # KNOWN GAP (named in the task description): T2 answers are templates —
    # "[Read the due date from the case record.]" — not speech. The bracket
    # is an instruction and must never reach TTS; Phase 4B resolves this.
    bracketed = [r["ID"] for r in rows if "[" in r.get("What Maya says", "")]
    print(f"[csv]  {len(bracketed)} row(s) carry a bracketed [instruction] rather than pure speech: {bracketed}")
    print("        the bracket must never reach TTS — Phase 4B resolves this")

    # KNOWN GAP (named in the task description): KB-DIS-01/02 carry a literal
    # [clinic name] placeholder, to be substituted the same way
    # {patient_first_name} already is at assembly time (assemble.py).
    placeholder_rows = [r["ID"] for r in rows if "[clinic name]" in r.get("What Maya says", "")]
    print(f"[csv]  {len(placeholder_rows)} row(s) carry a literal [clinic name] placeholder: {placeholder_rows}")
    print("        route through DEMO_PLACEHOLDERS-style substitution at assembly time")

    mermaid = extract_mermaid()
    MERMAID_OUT.write_text(mermaid + "\n")
    print(f"[html] {HTML_PATH.relative_to(REPO_ROOT)}: extracted mermaid source -> "
          f"{MERMAID_OUT.relative_to(REPO_ROOT)} ({len(mermaid)} chars)")

    pages = extract_script_pages()
    print(f"[pdf]  {PDF_PATH.relative_to(REPO_ROOT)}: {len(pages)} pages")
    # KNOWN GAP (named in the task description): page 5 truncates mid-
    # sentence, mid-page (not at the page's physical end — a plain tail slice
    # would miss it). It's a Human Agent line, not Maya's, so it doesn't block
    # Phase 2 — but it's the same gap in the Path A test fixture. Ask the
    # script owner; do not complete it.
    TRUNCATION_MARKER = "and we'll"
    if len(pages) >= 5:
        # pypdf wraps "and" and "we'll" onto separate lines, so the search
        # runs against whitespace-normalised text rather than the raw string.
        normalised = " ".join(pages[4].split())
        idx = normalised.lower().find(TRUNCATION_MARKER)
        if idx >= 0:
            end = idx + len(TRUNCATION_MARKER)
            context = normalised[max(0, idx - 80):end]
            print(f"[pdf]  KNOWN GAP — page 5 truncates mid-sentence: ...{context!r}")
            print("        a Human Agent line, not Maya's — ask the script owner, do not complete it")
        else:
            print(f"[pdf]  page 5 no longer contains {TRUNCATION_MARKER!r} — "
                  f"re-check whether the known truncation gap has moved or been fixed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
