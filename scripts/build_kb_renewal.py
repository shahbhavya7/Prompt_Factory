"""Parse the renewal KB spreadsheet into sace_chat/kb_renewal.py.

Primary source: data/renewal/source/Maya_KB_New(10 Knowledge base).csv — the
patient-experience review pass, 165 rows (up from 126). Committed unmodified;
this script is the only thing that reads it.

Recovery source: data/renewal/source/Maya_knowledge_base 1(10 Knowledge base).csv
— the PREVIOUS export, still committed and still load-bearing. See "CLIPPING"
below: the new sheet was transcribed from photographs and several cells were
cut off mid-word or mid-sentence. Where the old sheet holds the same row intact,
its value is preferred over the damaged one. That is why the old file cannot be
deleted.

PARSING NOTES (all measured against the actual files, not assumed):

  * cp1252, not utf-8 — both sheets have smart quotes, en/em dashes and the
    U+00B7 middle dot used as the cue separator, exported from a spreadsheet
    tool that used the Windows codepage.
  * The real header is row 4 (index 3). Rows 1-2 are a title and a usage note;
    row 3 was blank in the old sheet and is now the "NEWLY ADDED CONTENT"
    legend — either way it is not the header, so the index is unchanged.
  * The new sheet has ELEVEN columns: the old nine plus "Orig. row" and
    "Green in original?". Both are provenance about the review pass rather
    than call behaviour, so they are carried into `tags` and nothing reads
    them at runtime.
  * Section-banner rows ("THE LETTER AND WHAT IT IS...") separate topics
    visually. They have no ID starting with "KB-" and are skipped; the
    `Topic` column on each real row is what actually drives `intent`.
  * One ID carries an editorial marker: "KB-LTR-04 [dup ID]". The marker is
    stripped, which restores the genuine duplicate the old sheet also had, and
    make_id's deterministic suffixing then reproduces the SAME `kb_ltr_04_b`
    id this file emitted before. Ids are stable across the re-ingest.

CLIPPING — the reason this script is not a straight re-run of the old one.

The new sheet was transcribed from photographs of a printed spreadsheet, and
the photographs cut off text. Measured on the 165 rows: 7 spoken answers,
54 citations, 18 design notes and 9 answer-source cells carry a clipping
artifact ("[...]", "[cell clipped in photo]", "UNVE[RIFIED]", "Tr[ansfer]").

Three repair strategies, applied in this order, each deterministic:

  1. RECOVER — the same row exists intact in the old sheet. Preferred for any
     clipped field. Recovers 18 design notes, 34 citations and 2 answers
     (KB-ADR-01, KB-DUE-06), whose new-sheet text is truncated mid-sentence.
  2. REJOIN — the bracket only marks where the photo cut a single known word:
     "Tr[ansfer]" -> "Transfer", "UNVE[RIFIED]" -> "UNVERIFIED". Accepted only
     when the rejoined string is in a closed vocabulary (KNOWN_WORDS), so this
     can never invent a value it merely guessed at.
  3. ANSWER_REPAIRS — an explicit, reviewed table for spoken answers that
     neither of the above can fix. Five new T2 rows are cut mid-sentence and
     have no old-sheet counterpart.

The invariant that makes all of this safe: a clipping artifact reaching
`text` — the column Maya SPEAKS — is a BUILD FAILURE. There is no path that
ships "letters go [...]" to TTS. A newly clipped row added tomorrow fails the
build until a human looks at it, rather than being read aloud to a patient.
Metadata columns (citation, design note) are allowed to degrade to "" when
unrecoverable; they are provenance and are never spoken.

CLASSIFICATION (measured on the 165 KB- rows):

  * A bracketed `[...]` span in "What Maya says" is either a slot to
    substitute from campaign config (`[clinic name]`) or an operator
    instruction — a case-record read (`[Read ... from the case record.]`) or a
    scheduling-system read (`[Offer only appointments returned ...]`). Never
    both in the same row, and every instruction row is Tier T2. Anything with
    no bracket at all is spoken exactly as written: verbatim.
  * A verbatim rule's text must contain no bracket, and no rule's text
    (verbatim or not) may contain `{` or `"` — both would either leak a
    meta-instruction into TTS or break engine._rule_spans downstream.

Regenerate with: python scripts/build_kb_renewal.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "data" / "renewal" / "source" / "Maya_KB_New(10 Knowledge base).csv"
RECOVERY_CSV = ROOT / "data" / "renewal" / "source" / "Maya_knowledge_base 1(10 Knowledge base).csv"
OUTPUT = ROOT / "sace_chat" / "kb_renewal.py"

EXPECTED_HEADER = [
    "ID", "Topic", "Tier", "Canonical question", "How patients actually say it",
    "What Maya says", "Answer source", "Citation", "Design note",
    "Orig. row", "Green in original?",
]
# The previous export's header, for the recovery sheet.
RECOVERY_HEADER = EXPECTED_HEADER[:9]

# Column indices, named once so the two sheets' differing widths never turn
# into an off-by-one that silently swaps citation for design note.
C_ID, C_TOPIC, C_TIER, C_QUESTION, C_CUE, C_ANSWER = 0, 1, 2, 3, 4, 5
C_SOURCE, C_CITATION, C_NOTE, C_ORIG_ROW, C_GREEN = 6, 7, 8, 9, 10

# Fields worth trying to recover from the old sheet, and whether a clipping
# artifact surviving in them is fatal. `text` is what Maya speaks: fatal.
RECOVERABLE = (C_ANSWER, C_CITATION, C_NOTE, C_SOURCE)
SPOKEN_COLUMN = C_ANSWER

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

# ─────────────────────────────── clipping repair ─────────────────────────────

# A photo-transcription artifact. Three shapes, all measured in the sheet:
#   "[…]" / "[...]"          the transcriber marking a cut
#   "[cell clipped in photo]" / "[row clipped ...]" / "[cut off in photo]"
#   "UNVE[RIFIED]" / "Tr[ansfer]"   a word split by the cut, bracket mid-word
_CLIP_ELLIPSIS = r"\[\s*(?:…|\.\.\.)\s*\]"
_CLIP_NOTE = r"\[[^\]]*(?:clipped|cut off)[^\]]*\]"
_CLIP_SPLIT_WORD = r"(?<=\w)\[[^\]]*\]"
CLIP_RE = re.compile(f"{_CLIP_ELLIPSIS}|{_CLIP_NOTE}|{_CLIP_SPLIT_WORD}")

# Closed vocabulary for REJOIN. A split-word artifact is only repaired when
# the rejoined string lands in here — so "Tr[ansfer]" becomes "Transfer" and
# anything unrecognised is left damaged (and, for a spoken answer, fails the
# build) rather than being invented.
KNOWN_WORDS = frozenset({"Transfer", "Static", "Case record", "UNVERIFIED"})

_SPLIT_WORD_RE = re.compile(r"^(\w+)\[([^\]]+)\]$")


def rejoin_split_word(value: str) -> str | None:
    """"Tr[ansfer]" -> "Transfer", but only into KNOWN_WORDS. None otherwise."""
    m = _SPLIT_WORD_RE.match(value.strip())
    if not m:
        return None
    joined = m.group(1) + m.group(2)
    return joined if joined in KNOWN_WORDS else None


# Spoken answers that clipping damaged and neither RECOVER nor REJOIN can fix.
#
# All five are NEW T2 rows (no counterpart in the old sheet) whose text is
# "[<operator instruction>] <fallback clause>" with the fallback cut
# mid-sentence. Repaired here, in code, where the change is visible in review —
# NOT by editing the source CSV, which stays byte-identical to what was
# supplied.
#
# The repair rule, applied identically to all five: keep the operator
# instruction verbatim, truncate the fallback to its last COMPLETE sentence,
# and where truncation would leave no speakable fallback at all, close with the
# transfer sentence this KB already uses verbatim elsewhere. Nothing here
# states a fact the sheet did not already state — a T2 fallback's entire job is
# to decline to guess and hand off, and that is all these say.
ANSWER_REPAIRS = {
    # "...I want to check the exact status rather [...]" — truncating to the last
    # complete sentence would delete the whole fallback, leaving a T2 rule with
    # no line to speak when the case record is empty. Closed with KB-DIG-06's
    # own wording.
    "KB-DIG-05": (
        "[Read the case status and any outstanding request from the case record.] "
        "If that is not available: I want to check the exact status rather than "
        "guess. Let me connect you with someone who can look at it with you."
    ),
    # "...before I answer. Let me get help [...]" — the sentence before the cut
    # is complete, so the fragment is simply dropped and the handoff restored.
    "KB-STS-01": (
        "[Read the received date or submission status from the case record.] "
        "If it is not available: I want to confirm that before I answer. "
        "Let me connect you with someone who can check it."
    ),
    "KB-STS-02": (
        "[Read the current case status and effective date from the case record.] "
        "If either is missing: I want to confirm the exact status before I answer. "
        "Let me connect you with someone who can check it."
    ),
    "KB-STS-03": (
        "[Read the current status and outstanding county request from the case record.] "
        "If no reason is held: I do not want to guess. "
        "Let me connect you with someone who can check what they are waiting for."
    ),
    # "...who can work through the renewal with [...]" — cut mid-clause; the
    # clause is closed with its obvious object ("you") rather than truncated,
    # because truncation here would delete the offer the row exists to make.
    "KB-APT-01": (
        "[Offer only appointments returned by the scheduling system.] "
        "I can help arrange a time with a counsellor who can work through the "
        "renewal with you."
    ),
}

_CASE_RECORD_RE = re.compile(r"^read\s+(.+?)\s+from the case record\.?$", re.IGNORECASE)
# The second operator-instruction shape in the new sheet: a scheduling-system
# read rather than a case-record read. Same contract — an instruction, never
# speech — so it is recognised here rather than being mistaken for a slot.
_SYSTEM_READ_RE = re.compile(
    r"^(?:offer only|read the configured)\b.*\b(?:system|details)\.?$", re.IGNORECASE
)


def snake_case(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    return re.sub(r"_+", "_", s).strip("_").lower()


def slot_key(bracket_text: str) -> str:
    return snake_case(bracket_text)


def case_fields_from_instruction(bracket_text: str) -> list[str]:
    """"Read worker name and county phone number from the case record." ->
    ["worker_name", "county_phone_number"]. The bracket is an INSTRUCTION —
    never speech — so only its content, never the brackets, is parsed here.

    A scheduling-system instruction ("Offer only appointments returned by the
    scheduling system.") names no case-record field, so it contributes none:
    it still marks the rule non-verbatim, which is the part that matters.
    """
    body = bracket_text.strip()
    if _SYSTEM_READ_RE.match(body):
        return []
    m = _CASE_RECORD_RE.match(body)
    if not m:
        raise ValueError(f"case-record bracket doesn't match the expected pattern: {bracket_text!r}")
    fields = []
    for phrase in m.group(1).split(" and "):
        phrase = phrase.strip()
        if phrase.lower().startswith("the "):
            phrase = phrase[4:]
        fields.append(snake_case(phrase))
    return fields


def _read(path: Path, expected_header: list[str]) -> list[list[str]]:
    with open(path, encoding="cp1252", newline="") as f:
        rows = list(csv.reader(f))
    header = [c.strip() for c in rows[3]][: len(expected_header)]
    if header != expected_header:
        raise ValueError(f"unexpected header at row 4 of {path.name}: {header}")
    width = len(expected_header)
    return [(r + [""] * width)[:width] for r in rows[4:]
            if r and r[0].strip().startswith("KB-")]


def clean_id(raw_id: str) -> str:
    """Strip the editorial "[dup ID]" marker; "KB-LTR-04 [dup ID]" -> "KB-LTR-04"."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", raw_id.strip()).strip()


def row_key(row: list[str]) -> tuple[str, str]:
    """Identity for matching a new row to its old counterpart.

    The ID alone is not unique — KB-LTR-04 covers two distinct questions in both
    sheets — so the canonical question is part of the key. Lower-cased and
    whitespace-collapsed, because the new sheet fixed a double space in one of
    them ("I  got the form" -> "I got the form") and an exact match would miss it.
    """
    return clean_id(row[C_ID]), re.sub(r"\s+", " ", row[C_QUESTION].strip().lower())


def make_id(raw_id: str, seen: dict) -> str:
    base = clean_id(raw_id).lower().replace("-", "_")
    if base not in seen:
        seen[base] = 1
        return base
    # A genuine duplicate ID in the source sheet (KB-LTR-04 covers two
    # distinct questions). Deterministic, stable suffixing rather than
    # silently overwriting one row with the other.
    seen[base] += 1
    suffix = chr(ord("a") + seen[base] - 1)  # b, c, ...
    return f"{base}_{suffix}"


def repair_row(row: list[str], old: list[str] | None, report: list) -> list[str]:
    """Return `row` with clipping artifacts repaired where possible.

    Strategies in order: RECOVER from the old sheet, REJOIN a split word,
    ANSWER_REPAIRS for a reviewed spoken answer. Anything still damaged is left
    as-is for run_assertions to judge — fatal in the spoken column, tolerated
    (and blanked) in metadata.
    """
    row = list(row)
    rid = clean_id(row[C_ID])

    for col in RECOVERABLE:
        value = row[col]
        if not CLIP_RE.search(value):
            continue

        # 1. RECOVER — the old sheet holds this row's field intact.
        if old is not None and old[col].strip() and not CLIP_RE.search(old[col]):
            report.append((rid, col, "recover", old[col]))
            row[col] = old[col]
            continue

        # 2. REJOIN — the bracket only split a single known word.
        rejoined = rejoin_split_word(value)
        if rejoined is not None:
            report.append((rid, col, "rejoin", rejoined))
            row[col] = rejoined
            continue

        # 3. ANSWER_REPAIRS — reviewed replacement for a spoken answer.
        if col == SPOKEN_COLUMN and rid in ANSWER_REPAIRS:
            report.append((rid, col, "repair", ANSWER_REPAIRS[rid]))
            row[col] = ANSWER_REPAIRS[rid]
            continue

        # 4. Metadata with nothing to recover from. Blanked rather than left
        #    holding "[clipped]", which would otherwise be published as though
        #    it were a real citation. The spoken column never reaches here
        #    intact-looking: it stays damaged and fails the build.
        if col != SPOKEN_COLUMN:
            report.append((rid, col, "blank", ""))
            row[col] = ""
        else:
            report.append((rid, col, "UNREPAIRED", value))

    return row


def build_rules():
    new_rows = _read(SOURCE_CSV, EXPECTED_HEADER)
    old_rows = _read(RECOVERY_CSV, RECOVERY_HEADER)
    old_by_key: dict = {}
    for r in old_rows:
        old_by_key.setdefault(row_key(r), (r + [""] * 11)[:11])

    seen_ids: dict = {}
    rules = []
    unresolved_slots = set()
    repair_report: list = []

    for raw_row in new_rows:
        row = repair_row(raw_row, old_by_key.get(row_key(raw_row)), repair_report)

        chunk_id = make_id(row[C_ID], seen_ids)
        tier = row[C_TIER].strip()
        text = row[C_ANSWER].strip()
        cue_variants = [v.strip() for v in row[C_CUE].split("·") if v.strip()]
        brackets = re.findall(r"\[([^\]]+)\]", text)

        verbatim = not brackets
        slots: list[str] = []
        requires_case_fields: list[str] = []

        reads_system = False
        if brackets:
            if tier == "T2":
                for b in brackets:
                    if _SYSTEM_READ_RE.match(b.strip()):
                        # A scheduling-system read ("Offer only appointments
                        # returned by the scheduling system."). It names no
                        # case-record field, so requires_case_fields stays empty
                        # — but the rule is still an instruction row, not a
                        # verbatim one, and must not read as "reads nothing".
                        reads_system = True
                        continue
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
            "title": row[C_QUESTION].strip(),
            "text": text,
            "cue": "\n".join(cue_variants),
            "cue_variants": cue_variants,
            "intent": snake_case(row[C_TOPIC]),
            "priority": behavior["priority"],
            "tier": tier,
            "verbatim": verbatim,
            "slots": slots,
            "requires_case_fields": requires_case_fields,
            "tags": {
                "citation": row[C_CITATION].strip(),
                "design_note": row[C_NOTE].strip(),
                "answer_source": row[C_SOURCE].strip(),
                "transfer": behavior["transfer"],
                "cacheable": behavior["cacheable"],
                # Provenance from the review pass. Descriptive only.
                "orig_row": row[C_ORIG_ROW].strip(),
                "green_in_original": row[C_GREEN].strip().lower() == "yes",
                # True when the row's instruction reads a scheduling/config
                # system rather than the case record — see build_rules.
                "reads_system": reads_system,
            },
        })

    if unresolved_slots:
        lines = "\n".join(f"  {cid}: [{raw}] -> {key!r} not in SLOT_REGISTRY"
                           for cid, raw, key in sorted(unresolved_slots))
        raise ValueError(f"slot token(s) with no SLOT_REGISTRY entry:\n{lines}")

    return rules, repair_report


def run_assertions(rules):
    errors = []

    # Counts are MEASURED against the committed sheet and pinned, not derived
    # from it — a pinned number is what turns "the CSV changed underneath us"
    # into a build failure instead of a silent re-shape of the KB. Update these
    # deliberately, in the same commit as the sheet that moves them.
    if len(rules) != 165:
        errors.append(f"expected 165 rules, got {len(rules)}")

    ids = [r["id"] for r in rules]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate chunk ids after dedup: {dupes}")

    tier_counts = Counter(r["tier"] for r in rules)
    expected_tiers = {"T1": 98, "T2": 10, "T3": 47, "T4": 10}
    if dict(tier_counts) != expected_tiers:
        errors.append(f"tier counts {dict(tier_counts)} != expected {expected_tiers}")

    n_verbatim = sum(1 for r in rules if r["verbatim"])
    n_slots = sum(1 for r in rules if r["slots"])
    n_case = sum(1 for r in rules if r["requires_case_fields"])
    if n_verbatim != 144:
        errors.append(f"expected 144 verbatim rules, got {n_verbatim}")
    if n_slots != 11:
        errors.append(f"expected 11 rules with slots, got {n_slots}")
    if n_case != 7:
        errors.append(f"expected 7 rules with requires_case_fields, got {n_case}")

    for r in rules:
        # THE clipping invariant: a photo artifact must never reach the column
        # Maya speaks. This is the check that makes an un-repairable new
        # clipped row a build failure rather than a line read to a patient.
        if CLIP_RE.search(r["text"]):
            errors.append(
                f"{r['id']}: spoken text still contains a clipping artifact "
                f"({r['text'][:70]!r}) — add a reviewed entry to ANSWER_REPAIRS"
            )
        if r["verbatim"] and ("[" in r["text"] or "]" in r["text"]):
            errors.append(f"{r['id']}: marked verbatim but text contains a bracket")
        if "{" in r["text"] or '"' in r["text"]:
            errors.append(f"{r['id']}: text contains '{{' or '\"'")
        if r["tier"] not in TIER_BEHAVIOR:
            errors.append(f"{r['id']}: unknown tier {r['tier']!r}")
        if not r["text"].strip():
            errors.append(f"{r['id']}: empty spoken text")
        if not r["cue_variants"]:
            errors.append(f"{r['id']}: no cue variants — nothing for retrieval to match on")
        # Every T2 row is an instruction row by definition — it reads this
        # caller's case record or a scheduling system before it can speak. A
        # VERBATIM T2 would be a contradiction: a fixed line claiming to report
        # per-caller data, which is exactly the row that must never be spoken
        # (or cached) as written.
        if r["tier"] == "T2" and r["verbatim"]:
            errors.append(f"{r['id']}: T2 marked verbatim — a case-record row cannot be a fixed line")
        if r["tier"] == "T2" and not (r["requires_case_fields"] or r["tags"]["reads_system"]):
            errors.append(f"{r['id']}: T2 that reads neither the case record nor a system")

    # answer_source is descriptive, but a clipped one ("Tr[ansfer]") means a
    # REJOIN silently failed, so the vocabulary is closed here too.
    valid_sources = {"Static", "Transfer", "Case record", ""}
    bad_sources = sorted({r["tags"]["answer_source"] for r in rules
                          if r["tags"]["answer_source"] not in valid_sources})
    if bad_sources:
        errors.append(f"unrecognised answer_source value(s): {bad_sources}")

    if errors:
        raise AssertionError("build_kb_renewal assertions failed:\n" + "\n".join(f"  - {e}" for e in errors))


def render(rules) -> str:
    def lit(v) -> str:
        return repr(v)

    out = []
    out.append('"""GENERATED FILE — do not edit by hand.')
    out.append("")
    out.append("Generated by scripts/build_kb_renewal.py from")
    out.append(f"data/renewal/source/{SOURCE_CSV.name},")
    out.append(f"with clipped cells recovered from data/renewal/source/{RECOVERY_CSV.name}.")
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
    rules, repair_report = build_rules()

    if repair_report:
        by_strategy = Counter(s for _, _, s, _ in repair_report)
        print(f"clipping repairs: {dict(by_strategy)}")
        colname = {C_ANSWER: "text", C_CITATION: "citation",
                   C_NOTE: "design_note", C_SOURCE: "answer_source"}
        # Spoken-column repairs are printed individually: they are the only
        # ones that change what a patient hears, so they are never summarised
        # away. Metadata repairs are counted, not listed.
        for rid, col, strategy, value in repair_report:
            if col == SPOKEN_COLUMN:
                print(f"  SPOKEN {rid:12} {strategy:10} -> {value[:88]}")
        print(f"  (metadata: {sum(1 for _, c, _, _ in repair_report if c != SPOKEN_COLUMN)} "
              f"citation/design-note/answer-source cells repaired across "
              f"{', '.join(sorted(colname[c] for c in set(c for _, c, _, _ in repair_report) if c != SPOKEN_COLUMN))})")

    run_assertions(rules)
    OUTPUT.write_text(render(rules), encoding="utf-8")
    tier_counts = Counter(r["tier"] for r in rules)
    intents = Counter(r["intent"] for r in rules)
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(rules)} rules, "
          f"tiers={dict(sorted(tier_counts.items()))}, "
          f"intents={len(intents)}, "
          f"verbatim={sum(1 for r in rules if r['verbatim'])}, "
          f"slots={sum(1 for r in rules if r['slots'])}, "
          f"requires_case_fields={sum(1 for r in rules if r['requires_case_fields'])}")


if __name__ == "__main__":
    sys.exit(main())
