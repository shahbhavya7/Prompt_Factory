"""Validation and code-enforced precedence.

Intent is classified semantically (retrieve.IntentRouter) and the reply comes
from one structured LLM call. There is no stage machine, so there is no stage to
validate and no advancement to police.

What stays in code, deliberately:
  - the allowed intent vocabulary, so a model that hallucinates a label cannot
    push a value into state that retrieval can never route on;
  - precedence, which is policy and must not be re-litigated per turn: abuse
    outranks dnc (while still flagging opt_out), and an explicit day or time
    outranks a "busy" reading;
  - stripping control tokens out of anything spoken.
"""

import re

# The routable labels. Must stay in step with kb.INTENT_EXEMPLARS' keys and with
# the `intent` values on the rules themselves.
VALID_INTENTS = {
    "none",
    "dnc",
    "abuse",
    "callback_request",
    "busy",
    "redirect",
    "elsewhere",
    "language",
    "clinic_location",
    "clinical_q",
    "pricing_q",
    "ai_question",
    "recorded_q",
    "frustration",
    "garbled_audio",
    "payment_q",
    "appointment_scheduling",
    "eligibility_renewal",
    "complaint_escalation",
}

# Retained ONLY for the callback-over-busy precedence rule. This is not intent
# detection — the router classifies intent; this just checks whether the caller
# actually named a time, which decides which of two labels policy says wins.
_TIME_PATTERNS = [
    r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b",
    r"\btomorrow\b",
    r"\btonight\b",
    r"\bthis (morning|afternoon|evening|week)\b",
    r"\bnext week\b",
    r"\bafter \d{1,2}\b",
    r"\bbefore \d{1,2}\b",
    r"\bat \d{1,2}(:\d{2})?\s*(am|pm)?\b",
    r"\b\d{1,2}\s*(am|pm)\b",
    r"\b\d{1,2}\s*o'?clock\b",
]

_ABUSE_HINTS = [
    r"\bf+u+c+k+\w*\b",
    r"\bshit\b",
    r"\bbullshit\b",
    r"\basshole\b",
    r"\bidiot\b",
    r"\bstupid\b",
    r"\bshut up\b",
]

_DNC_HINTS = [
    r"\bdon'?t call\b",
    r"\bstop calling\b",
    r"\bremove my number\b",
    r"\btake me off\b",
    r"\bnot interested\b",
    r"\bno longer interested\b",
    r"\bstop contacting\b",
    r"\bstop bothering\b",
    r"\bdone with these calls\b",
    r"\blose my number\b",
    r"\bnever want these calls\b",
]

# [CALL_END], [END:Contacted|Needs Review], [opt-out], [active-coverage] …
_CONTROL_TOKEN_RE = re.compile(
    r"\s*\[(?:CALL_END|END:[^\]]*|opt-out|active-coverage|wrong-number)\]"
)


def _matches_any(patterns, text_lower):
    return any(re.search(p, text_lower) for p in patterns)


def mentions_day_or_time(message: str) -> bool:
    return _matches_any(_TIME_PATTERNS, message.lower())


def resolve_precedence(intent: str, message: str) -> tuple[str, bool]:
    """Policy precedence, in code rather than trusted to the model.

    Returns (effective_intent, opt_out).

    - abuse outranks dnc when both read true in one utterance, but the call is
      still tagged opt_out because a do-not-call request was present;
    - an explicit day or time outranks a "busy" reading ("I'm slammed, try me
      Monday" is a callback, not a brush-off);
    - a DNC/abuse signal in the message text overrides whatever intent the
      semantic router picked, not just the case where it already picked dnc
      or abuse. A caller who blends resistance into another remark ("I've
      told you people to stop calling, this is so frustrating") can score
      closer to a softer label like `frustration` by cosine — the router
      only returns its single best-scoring guess, so without this check
      that phrasing would never reach the DNC/abuse branch above at all.
    """
    lower = (message or "").lower()
    opt_out = False

    looks_abusive = _matches_any(_ABUSE_HINTS, lower)
    looks_dnc = _matches_any(_DNC_HINTS, lower)

    if looks_abusive:
        return "abuse", True if looks_dnc else opt_out
    if looks_dnc:
        return "dnc", True

    if intent == "busy" and mentions_day_or_time(message or ""):
        return "callback_request", opt_out

    return intent, opt_out


def strip_control_tokens(reply: str) -> tuple[str, bool]:
    """Control tokens are metadata and must never be spoken. The prompt forbids
    them, but the model still copies them out of rule text sometimes, so they
    are removed here rather than trusted to an instruction."""
    stripped = _CONTROL_TOKEN_RE.sub("", reply)
    if stripped == reply:
        return reply, False
    return re.sub(r"\s{2,}", " ", stripped), True


def validate_turn(decision: dict, state, valid_intents=None) -> dict:
    """Clamp a model decision to values the rest of the system can act on.

    Anything out of vocabulary is dropped and reported in `warnings`, so a bad
    decision degrades to "no intent" rather than corrupting state.

    `valid_intents` defaults to the coverage campaign's vocabulary; a caller
    on a different campaign (see sace_chat.campaign) passes its own so a
    real, in-vocabulary label is not clamped to "none" just because it isn't
    one of THIS campaign's labels.
    """
    warnings = []
    valid_intents = valid_intents if valid_intents is not None else VALID_INTENTS

    intent = decision.get("intent")
    if intent not in valid_intents:
        if intent is not None:
            warnings.append(f"intent {intent!r} not in VALID_INTENTS; using 'none'")
        intent = "none"

    reply = decision.get("reply") or ""
    if not isinstance(reply, str):
        warnings.append("reply was not a string; coerced")
        reply = str(reply)

    reply, had_tokens = strip_control_tokens(reply)
    if had_tokens:
        warnings.append("stripped control token(s) from reply")

    fields = decision.get("extracted_fields") or {}
    if not isinstance(fields, dict):
        warnings.append("extracted_fields was not an object; ignored")
        fields = {}

    return {
        "intent": intent,
        "reply": reply.strip(),
        "call_should_end": bool(decision.get("call_should_end")),
        "extracted_fields": {str(k): str(v) for k, v in fields.items()},
        "warnings": warnings,
    }
