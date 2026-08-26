"""Prompt assembly: stable core plus the rules this turn retrieved."""

from sace_chat.manager import VALID_INTENTS

# Demo values substituted into every {placeholder} the rule text carries. Done
# at assembly time only, so the stored rules stay templated and still read as
# policy prose.
DEMO_PLACEHOLDERS = {
    "{patient_first_name}": "Bhavya",
    "{patient_last_name}": "Shah",
    "{callback_number}": "1-800-555-0100",
    "{business_entity}": "Community Medical Center - Downtown Clinic",
    "{current_month}": "this month",
}

HISTORY_TURNS = 6

_INTENT_LIST = "|".join(sorted(VALID_INTENTS))

_TURN_INSTRUCTION = f"""\
# THIS TURN
Return ONLY a JSON object, no prose around it:

{{"intent": "<one of: {_INTENT_LIST}>",
 "reply": "<what Maya says next, one question maximum>",
 "call_should_end": true|false,
 "extracted_fields": {{"<name>": "<value>"}}}}

- reply comes from the GOVERNING RULE and nothing else. Speak its scripted line,
  matching its wording or lightly rephrasing for a natural read. Take no
  sentence, question or closing from REFERENCE, from RECENT TURNS, or from your
  own sense of what such a call normally asks.
- Never re-ask anything under ALREADY ASKED, in any wording.
- Never re-ask for a value already listed under ALREADY ON FILE, and never
  state a different value for it than what is shown there.
- A hedge is an answer. "I'm not sure", "I don't know", "maybe" all answer a
  yes/no question — follow where the governing rule sends an unsure answer
  instead of asking again.
- intent is one of the labels above; "none" when the caller is simply answering
  the pending question. Pick a label when they raise that situation however they
  word it.
- reply is spoken aloud: no JSON, no key names, no bracketed tokens like
  [CALL_END], no rule ids.
- extracted_fields holds only values the caller actually stated this turn.
  Empty object otherwise. Never invent one.
"""

_REINFORCE = """\
# CORRECTION — YOUR PREVIOUS ATTEMPT WAS REJECTED
{reason}
Write the reply using ONLY the GOVERNING RULE above. Speak its scripted line.
Do not draw on REFERENCE, on the conversation so far, or on anything you know
about calls like this. If the governing rule gives a line in quotes, that line
is your reply.
"""


def _substitute_placeholders(text: str) -> str:
    for placeholder, value in DEMO_PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return text


def _governing_section(retrieval) -> str:
    rule = retrieval.governing
    if rule is None:
        return (
            "GOVERNING RULE:\n(none retrieved — say nothing new. Ask again, in the same "
            "words, whatever question is still pending.)"
        )
    lines = [
        "GOVERNING RULE — the ONLY rule that determines this turn's reply. Follow it "
        "exactly. Never add sentences, questions or closings from anywhere else.",
        f"[{rule.chunk.id}] {rule.chunk.title}: {rule.chunk.text}",
    ]
    if rule.chunk.terminal:
        lines.append(
            "THIS ENDS THE CALL. Deliver it in full, add nothing after it — no question, "
            "no offer of further help — and set call_should_end=true."
        )
    return "\n".join(lines)


def _reference_section(retrieval) -> str:
    if not retrieval.reference:
        return "REFERENCE: (none — the governing rule is the only rule in scope)"
    lines = [
        "REFERENCE — background only. Never take a reply, question or closing from here."
    ]
    lines += [f"[{r.chunk.id}] {r.chunk.title}: {r.chunk.text}" for r in retrieval.reference]
    return "\n".join(lines)


def _asked_section(asked_questions) -> str:
    lines = ["ALREADY ASKED (never ask any of these again, in any wording):"]
    lines += [f"- {q}" for q in asked_questions] or ["(nothing yet)"]
    return "\n".join(lines)


def _collected_fields_section(collected_fields) -> str:
    """Surfaces state.collected_fields into the prompt.

    Without this, a value the caller already gave (and that got saved to
    state) is invisible to the model on the next turn — it only sees the
    raw RECENT TURNS dialogue, not the confirmed values extracted from it.
    That made the model re-ask for things already on file, or invent a
    plausible-sounding value instead of using the real stored one.
    """
    lines = ["ALREADY ON FILE (already confirmed this call; never re-ask for these, "
              "never invent a different value — reuse exactly what is shown):"]
    if not collected_fields:
        return lines[0] + "\n(nothing yet)"
    lines += [f"- {k}: {v}" for k, v in collected_fields.items()]
    return "\n".join(lines)


# retrieve.CallState's renewal-only typed fields, in the order they render
# under ALREADY ON FILE — no new section for these (per the Phase 3 review),
# they merge into the existing one. Each entry is (attribute, default) so a
# field still at its default is left out entirely rather than rendered as
# noise ("willingness: undecided" on turn 1 tells the model nothing it
# doesn't already know from ALREADY ASKED). Deliberately excludes
# consent_recorded/address_updated_by_human/disposition/upload_link_sent —
# those are human-agent bookkeeping (sace_chat/disposition.py), not
# something Maya's own reply should ever reason about mid-call.
_RENEWAL_ON_FILE_FIELDS = (
    ("packet_received", None),
    ("address_confirmed", None),
    ("already_submitted", None),
    ("willingness", "undecided"),
    ("available_now", None),
    ("has_camera_phone", None),
    ("helper_at_home", None),
    ("consent_prebriefed", False),
)


def _on_file_view(state, collected_fields: dict) -> dict:
    """collected_fields merged with any non-default renewal CallState field —
    see _RENEWAL_ON_FILE_FIELDS. A no-op merge for coverage: every renewal
    field sits at its listed default for a coverage CallState (which HAS the
    attributes, since CallState is one shared dataclass, but never anything
    else in them), so this returns collected_fields completely unchanged and
    the golden byte-identical prompt is untouched.
    """
    merged = dict(collected_fields)
    for attr, default in _RENEWAL_ON_FILE_FIELDS:
        value = getattr(state, attr, default)
        if value != default and attr not in merged:
            merged[attr] = value
    return merged


def _case_record_section(retrieval, case_record: dict) -> str | None:
    """CASE RECORD — read-only facts about THIS caller, injected only when
    the governing rule declares which fields it needs (Chunk.tags['case_fields'],
    aka a T2 rule's `requires_case_fields`). Returns None (no section at all,
    not an empty placeholder) for every other turn — the whole point is that
    injecting the full case record on every turn would quietly undo the
    prompt-size saving this architecture exists for; scoping it to exactly
    the 1-2 fields one T2 rule names keeps the cost at a few tokens on the
    handful of turns that actually need it, and literally zero everywhere
    else, coverage included (coverage rules carry no case_fields tag at all).
    """
    rule = retrieval.governing
    if rule is None:
        return None
    case_fields = list(getattr(rule.chunk, "case_fields", None) or [])
    if not case_fields:
        return None
    record = case_record or {}
    lines = [
        "CASE RECORD — read-only facts about this caller; you may read these "
        "out. If a field below is absent, say you will confirm and come back "
        "rather than guessing:"
    ]
    for field in case_fields:
        value = record.get(field)
        if value:
            lines.append(f"- {field}: {value}")
        else:
            lines.append(f"- {field}: (not on file — use the governing rule's own "
                          f"fallback line verbatim; never guess this)")
    return "\n".join(lines)


def _history_section(history) -> str:
    recent = (history or [])[-HISTORY_TURNS:]
    lines = ["RECENT TURNS (context for who said what, not a source of content):"]
    lines += recent or ["(none yet — this is the first turn)"]
    return "\n".join(lines)


def build_turn_prompt(stable_core: str, state, retrieval, history, reinforce_reason: str = "") -> str:
    """System prompt for the turn decision. The caller's message is sent
    separately, as the user message."""
    sections = [
        stable_core.strip(),
        _governing_section(retrieval),
        _reference_section(retrieval),
    ]
    case_record_section = _case_record_section(retrieval, getattr(state, "case_record", {}))
    if case_record_section is not None:
        sections.append(case_record_section)
    sections += [
        _collected_fields_section(
            _on_file_view(state, getattr(state, "collected_fields", {}))
        ),
        _asked_section(getattr(state, "asked_questions", [])),
        _history_section(history),
        _TURN_INSTRUCTION,
    ]
    if reinforce_reason:
        sections.append(_REINFORCE.format(reason=reinforce_reason))
    return _substitute_placeholders("\n\n".join(sections))
