"""Parse the renewal KB spreadsheet into sace_chat/kb_renewal.py.

Source: data/renewal/source/Maya_knowledge_base 1(10 Knowledge base).csv —
committed unmodified; this script is the only thing that reads it.

PARSING NOTES (all measured against the actual file, not assumed):

  * cp1252, not utf-8 — the file has smart quotes and en/em dashes exported
    from a spreadsheet tool that used the Windows codepage.
  * The real header is row 4 (index 3): rows 1-2 are a title and a usage
    note, row 3 is blank.
  * Section-banner rows ("THE LETTER AND WHAT IT IS...") separate topics
    visually. They have no ID starting with "KB-" and are skipped; the
    `Topic` column on each real row is what actually drives `intent`.

CLASSIFICATION (measured on the 126 KB- rows):

  * A bracketed `[...]` span in "What Maya says" is either a slot to
    substitute from campaign config (`[clinic name]`) or a case-record
    read instruction (`[Read ... from the case record.]`) — never both in
    the same row, and every case-record row is Tier T2 (the only tier
    where that pattern occurs). Anything with no bracket at all is spoken
    exactly as written: verbatim.
  * A verbatim rule's text must contain no bracket, and no rule's text
    (verbatim or not) may contain `{` or `"` — both would either leak a
    meta-instruction into TTS or break engine._rule_spans downstream.

Regenerate with: python scripts/build_kb_renewal.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "data" / "renewal" / "source" / "Maya_knowledge_base 1(10 Knowledge base).csv"
OUTPUT = ROOT / "sace_chat" / "kb_renewal.py"

EXPECTED_HEADER = [
    "ID", "Topic", "Tier", "Canonical question", "How patients actually say it",
    "What Maya says", "Answer source", "Citation", "Design note",
]

# A tier's behaviour is data, not runtime branching: every consumer of RULES
# looks this up by tier rather than special-casing tier strings inline.
#
#   T3 is deliberately `normal`, not `high` — _fetch_by_intent ranks priority
#   above distance within an intent, and most topics are mixed-tier (Income is
#   11 T1 + 3 T3). Give T3 `high` and every income question routes to the
#   transfer line regardless of what was actually asked.
TIER_BEHAVIOR = {
    "T1": {"priority": "normal", "transfer": False, "cacheable": True},
    "T2": {"priority": "normal", "transfer": False, "cacheable": False},
    "T3": {"priority": "normal", "transfer": True, "cacheable": True},
    "T4": {"priority": "critical", "transfer": True, "cacheable": False},
}

# Placeholder campaign config: this branch has no campaign wiring, so these
# are illustrative defaults, not real deployment values. What matters at
# BUILD time is only that every [bracketed] slot token found in the sheet has
# an entry here — an unfilled slot spoken aloud is worse than a build failure.
SLOT_REGISTRY = {
    "clinic_name": "Example Community Clinic",
    "clinic_main_line": "1-800-555-0100",
    "clinic_line": "1-800-555-0100",
    "clinic_phone": "1-800-555-0100",
}

_CASE_RECORD_RE = re.compile(r"^read\s+(.+?)\s+from the case record\.?$", re.IGNORECASE)


def snake_case(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    return re.sub(r"_+", "_", s).strip("_").lower()


def slot_key(bracket_text: str) -> str:
    return snake_case(bracket_text)


def case_fields_from_instruction(bracket_text: str) -> list[str]:
    """"Read worker name and county phone number from the case record." ->
    ["worker_name", "county_phone_number"]. The bracket is an INSTRUCTION —
    never speech — so only its content, never the brackets, is parsed here."""
    m = _CASE_RECORD_RE.match(bracket_text.strip())
    if not m:
        raise ValueError(f"case-record bracket doesn't match the expected pattern: {bracket_text!r}")
    fields = []
    for phrase in m.group(1).split(" and "):
        phrase = phrase.strip()
        if phrase.lower().startswith("the "):
            phrase = phrase[4:]
        fields.append(snake_case(phrase))
    return fields


def read_rows() -> list[list[str]]:
    with open(SOURCE_CSV, encoding="cp1252", newline="") as f:
        rows = list(csv.reader(f))
    header = [c.strip() for c in rows[3]]
    if header != EXPECTED_HEADER:
        raise ValueError(f"unexpected header at row 4: {header}")
    return [r for r in rows[4:] if r and r[0].strip().startswith("KB-")]


def make_id(raw_id: str, seen: dict) -> str:
    base = raw_id.strip().lower().replace("-", "_")
    if base not in seen:
        seen[base] = 1
        return base
    # A genuine duplicate ID in the source sheet (KB-LTR-04 covers two
    # distinct questions). Deterministic, stable suffixing rather than
    # silently overwriting one row with the other.
    seen[base] += 1
    suffix = chr(ord("a") + seen[base] - 1)  # b, c, ...
    return f"{base}_{suffix}"


def build_rules():
    rows = read_rows()
    seen_ids: dict = {}
    rules = []
    unresolved_slots = set()

    for row in rows:
        raw_id, topic, tier, question, cue_raw, answer, answer_source, citation, design_note = (
            (row + [""] * 9)[:9]
        )
        chunk_id = make_id(raw_id, seen_ids)
        tier = tier.strip()
        text = answer.strip()
        cue_variants = [v.strip() for v in cue_raw.split("·") if v.strip()]
        brackets = re.findall(r"\[([^\]]+)\]", text)

        verbatim = not brackets
        slots: list[str] = []
        requires_case_fields: list[str] = []

        if brackets:
            if tier == "T2":
                for b in brackets:
                    requires_case_fields.extend(case_fields_from_instruction(b))
            else:
                for b in brackets:
                    key = slot_key(b)
                    slots.append(key)
                    if key not in SLOT_REGISTRY:
                        unresolved_slots.add((chunk_id, b, key))

        behavior = TIER_BEHAVIOR[tier]

        rules.append({
            "id": chunk_id,
            "title": question.strip(),
            "text": text,
            "cue": "\n".join(cue_variants),
            "cue_variants": cue_variants,
            "intent": snake_case(topic),
            "priority": behavior["priority"],
            "tier": tier,
            "verbatim": verbatim,
            "slots": slots,
            "requires_case_fields": requires_case_fields,
            "tags": {
                "citation": citation.strip(),
                "design_note": design_note.strip(),
                "answer_source": answer_source.strip(),
                "transfer": behavior["transfer"],
                "cacheable": behavior["cacheable"],
            },
        })

    if unresolved_slots:
        lines = "\n".join(f"  {cid}: [{raw}] -> {key!r} not in SLOT_REGISTRY"
                           for cid, raw, key in sorted(unresolved_slots))
        raise ValueError(f"slot token(s) with no SLOT_REGISTRY entry:\n{lines}")

    return rules


def run_assertions(rules):
    errors = []

    if len(rules) != 126:
        errors.append(f"expected 126 rules, got {len(rules)}")

    ids = [r["id"] for r in rules]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate chunk ids after dedup: {dupes}")

    from collections import Counter
    tier_counts = Counter(r["tier"] for r in rules)
    expected_tiers = {"T1": 92, "T2": 3, "T3": 21, "T4": 10}
    if dict(tier_counts) != expected_tiers:
        errors.append(f"tier counts {dict(tier_counts)} != expected {expected_tiers}")

    n_verbatim = sum(1 for r in rules if r["verbatim"])
    n_slots = sum(1 for r in rules if r["slots"])
    n_case = sum(1 for r in rules if r["requires_case_fields"])
    if n_verbatim != 114:
        errors.append(f"expected 114 verbatim rules, got {n_verbatim}")
    if n_slots != 9:
        errors.append(f"expected 9 rules with slots, got {n_slots}")
    if n_case != 3:
        errors.append(f"expected 3 rules with requires_case_fields, got {n_case}")

    for r in rules:
        if r["verbatim"] and ("[" in r["text"] or "]" in r["text"]):
            errors.append(f"{r['id']}: marked verbatim but text contains a bracket")
        if "{" in r["text"] or '"' in r["text"]:
            errors.append(f"{r['id']}: text contains '{{' or '\"'")
        if r["tier"] not in TIER_BEHAVIOR:
            errors.append(f"{r['id']}: unknown tier {r['tier']!r}")

    if errors:
        raise AssertionError("build_kb_renewal assertions failed:\n" + "\n".join(f"  - {e}" for e in errors))


def render(rules) -> str:
    def lit(v) -> str:
        return repr(v)

    out = []
    out.append('"""GENERATED FILE — do not edit by hand.')
    out.append("")
    out.append("Generated by scripts/build_kb_renewal.py from")
    out.append('data/renewal/source/Maya_knowledge_base 1(10 Knowledge base).csv.')
    out.append("Regenerate with: python scripts/build_kb_renewal.py")
    out.append('"""')
    out.append("")
    out.append("from sace_chat.models import Chunk")
    out.append("")
    out.append("# A tier's behaviour (priority/transfer/cacheable), as data — see")
    out.append("# scripts/build_kb_renewal.py for why T3 stays priority=normal.")
    out.append(f"TIER_BEHAVIOR = {lit(TIER_BEHAVIOR)}")
    out.append("")
    out.append("RULES = [")
    for r in rules:
        out.append("    Chunk(")
        out.append(f"        id={lit(r['id'])},")
        out.append(f"        title={lit(r['title'])},")
        out.append(f"        text={lit(r['text'])},")
        out.append(f"        cue={lit(r['cue'])},")
        out.append(f"        cue_variants={lit(r['cue_variants'])},")
        out.append(f"        intent={lit(r['intent'])},")
        out.append(f"        priority={lit(r['priority'])},")
        out.append(f"        tier={lit(r['tier'])},")
        out.append(f"        verbatim={lit(r['verbatim'])},")
        out.append(f"        slots={lit(r['slots'])},")
        out.append(f"        requires_case_fields={lit(r['requires_case_fields'])},")
        out.append(f"        tags={lit(r['tags'])},")
        out.append("    ),")
    out.append("]")
    out.append("")
    return "\n".join(out)


def main():
    rules = build_rules()
    run_assertions(rules)
    OUTPUT.write_text(render(rules), encoding="utf-8")
    tier_counts = {}
    for r in rules:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(rules)} rules, tiers={tier_counts}, "
          f"verbatim={sum(1 for r in rules if r['verbatim'])}, "
          f"slots={sum(1 for r in rules if r['slots'])}, "
          f"requires_case_fields={sum(1 for r in rules if r['requires_case_fields'])}")


if __name__ == "__main__":
    sys.exit(main())
