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
