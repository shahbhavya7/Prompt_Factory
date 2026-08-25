"""Code-enforced content guards — checks a prompt cannot be trusted to hold on
its own, run on the reply itself before it can reach TTS.

`check_never_say` backs the renewal campaign's single most important
property: Maya never states whether a caller qualifies, any income limit, any
dollar threshold, or any deadline, UNLESS it was read verbatim from that
call's own case record. RENEWAL_STABLE_CORE says this in prose, but prose is
not how this codebase enforces anything load-bearing elsewhere either —
`terminal` is authoritative over the model's own `call_should_end`, a
`GROUNDING_THRESHOLD` cosine check runs on every reply, and this guard is the
same pattern applied to a different failure mode: not "did the reply wander
off the governing rule" but "did the reply invent a number that isn't this
caller's own."
"""

from __future__ import annotations

import re

# Any dollar figure: "$21,000", "$1,340", "$225". Bare "dollars"/"USD" amounts
# are deliberately NOT matched — every occurrence actually observed in the
# source material carries a dollar sign, and a bare-number pattern would fire
# on innocuous digits (a phone number, a form field number) far more than it
# would catch a real violation.
_CURRENCY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")

# "limit" and its inflections — an income limit or eligibility limit is
# exactly the kind of number this guard exists to keep off Maya's own lips.
_LIMIT_RE = re.compile(r"\blimit(?:s|ed|ing)?\b", re.I)

# Whether the caller qualifies/is eligible is a determination, not a fact Maya
# is ever allowed to state on her own authority.
_QUALIFY_RE = re.compile(r"\b(?:qualif(?:y|ies|ied|ying)|eligib\w*)\b", re.I)

# A bare date: "August 30, 2026", "March 8th", "4/15/26" — a deadline stated
# without having been read off the case record is exactly the failure mode
# named in the never-say list.
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)
_DATE_RE = re.compile(
    rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b",
    re.I,
)

_PATTERNS = (
    ("currency", _CURRENCY_RE),
    ("limit", _LIMIT_RE),
    ("qualify/eligible", _QUALIFY_RE),
    ("bare date", _DATE_RE),
)


def check_never_say(reply_text: str, case_record: dict | None = None) -> tuple[bool, str]:
    """(ok, reason) — ok=False means this reply must not be spoken as-is.

    `case_record` is the current call's own injected data (e.g. its real due
    date, read from the system of record). A match is EXEMPT when the exact
    matched span appears verbatim somewhere in the case record's own values —
    that is the one case where naming a number or a date is correct rather
    than invented: Maya is allowed to say THIS caller's real due date, just
    never a generic one she was not handed.

    Mirrors the (ok, reason) shape `answer_cache.is_cacheable` already uses —
    a boolean plus why, so a refusal is always explainable rather than a bare
    False the caller of this function has to re-derive a reason for.
    """
    if not reply_text:
        return True, ""

    exempt_blob = " ".join(str(v) for v in (case_record or {}).values() if v)

    for label, pattern in _PATTERNS:
        for match in pattern.finditer(reply_text):
            hit = match.group(0)
            if hit in exempt_blob:
                continue
            return False, (
                f"never-say violation ({label}): {hit!r} is not present "
                f"verbatim in this call's case record"
            )
    return True, ""


# Deterministic T4 short-circuit (Phase 2E, Part D). Immigration and distress
# disclosures must not depend on embedding recall — a caller who says "I
# don't want to be here anymore" cannot be allowed to fall through because a
# retrieval query happened to land a few hundredths short of
# GENERAL_MIN_SIMILARITY. Modelled on manager.resolve_precedence: raw text,
# checked BEFORE embedding, not the router's single best label — see
# retrieve.retrieve(), which checks this ahead of everything else.
#
# Ordered most-severe first and the first match wins, since a message like
# "I'm not safe at home and can't afford my insulin" should route to whichever
# is checked first rather than whichever regex happens to match last.
#
# English AND Spanish-language fragments are included for the same reason
# regex beats embedding recall here at all: a caller code-switching mid
# sentence ("no tengo papeles, I'm scared") must not depend on which language
# the embedding space happens to weight more heavily.
_SELF_HARM_RE = re.compile(
    r"\b(?:don'?t want to be here (?:any ?more)?|kill myself|end (?:my|it) all|"
    r"suicid\w*|hurt myself|no quiero estar aqu[ií])\b",
    re.I,
)
_UNSAFE_AT_HOME_RE = re.compile(
    r"\b(?:not safe at home|won'?t let me (?:leave|go)|hurting me|hits me|"
    r"afraid of (?:him|her|them)|no es seguro en mi casa)\b",
    re.I,
)
_CANT_AFFORD_MEDS_RE = re.compile(
    r"\b(?:can'?t afford my (?:medication|meds|pills|insulin)|"
    r"ran out of (?:my )?insulin|can'?t pay for my pills|"
    r"no puedo pagar mis (?:pastillas|medicinas))\b",
    re.I,
)
_ACUTE_DISTRESS_RE = re.compile(
    r"\b(?:sick right now|i'?m in (?:a lot of )?pain|think i need a doctor|"
    r"need an ambulance|me duele mucho|necesito un doctor)\b",
    re.I,
)
_GREEN_CARD_RE = re.compile(r"\bgreen card\b", re.I)
_IMMIGRATION_RE = re.compile(
    r"\b(?:\bICE\b|immigration|deport\w*|undocumented|no status|"
    r"don'?t have (?:papers|status)|my papers|no tengo papeles|"
    r"indocumentad[oa]|sin estatus)\b",
    re.I,
)

# rule_id -> the seed rule that already carries the exact, transfer=True line
# for this situation (see sace_chat/kb_renewal.py). Checked in this order;
# the first pattern to match wins. kb_imm_03 is the generic immigration line
# — the same one campaign.py already uses as the renewal never-say fallback
# — so a phrasing that mentions immigration without a green card specifically
# lands there.
_T4_ROUTES = (
    ("kb_dis_04", _SELF_HARM_RE),
    ("kb_dis_03", _UNSAFE_AT_HOME_RE),
    ("kb_dis_01", _CANT_AFFORD_MEDS_RE),
    ("kb_dis_02", _ACUTE_DISTRESS_RE),
    ("kb_imm_05", _GREEN_CARD_RE),
    ("kb_imm_03", _IMMIGRATION_RE),
)


def t4_shortcircuit(message: str) -> str | None:
    """The T4 rule id this message must be routed to, or None.

    Checked BEFORE embedding (see retrieve.retrieve()) — a caller disclosing
    self-harm, an unsafe home, an unaffordable medication, acute distress, or
    an immigration-status fear gets the fixed, transfer=True line for that
    situation deterministically, never contingent on a cosine happening to
    clear GENERAL_MIN_SIMILARITY. The semantic T4 rules stay in the pool as
    the second net for a phrasing this does not cover.
    """
    if not message:
        return None
    for rule_id, pattern in _T4_ROUTES:
        if pattern.search(message):
            return rule_id
    return None
