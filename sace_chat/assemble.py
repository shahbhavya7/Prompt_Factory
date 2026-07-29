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
        _asked_section(getattr(state, "asked_questions", [])),
        _history_section(history),
        _TURN_INSTRUCTION,
    ]
    if reinforce_reason:
        sections.append(_REINFORCE.format(reason=reinforce_reason))
    return _substitute_placeholders("\n\n".join(sections))
