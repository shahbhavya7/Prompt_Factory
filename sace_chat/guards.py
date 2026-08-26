"""check_never_say: a code-enforced backstop on generated replies.

Required, not optional, the moment a branch can place a live call with an LLM
fallback path (see Engine.prepare_reply) — nothing else stops a generated
reply from stating an income limit or a deadline it was never given.

Zero exemptions: there is no case record on this branch to exempt anything
against, so this runs on every LLM-generated reply, regardless of which rule
governed it or which campaign is active. It does not run on a cache hit or on
the fixed no-rule-matched fallback line — those are not generated, they are
spoken verbatim.
"""

from __future__ import annotations

import re

# $21,000 / $21000 / $ 21,000 — a currency amount, the single most concrete
# way a reply can fabricate a fact nobody gave it.
_CURRENCY_RE = re.compile(r"\$\s?\d")

# "limit" / "limits" / "limited" — income limits, benefit limits, etc. A rule
# that legitimately needs to say this word says it VERBATIM (see
# Chunk.verbatim); this guard only ever sees LLM-generated text, so there is
# no legitimate case for the model to introduce the word on its own.
_LIMIT_RE = re.compile(r"\blimit(s|ed|ing)?\b", re.IGNORECASE)

# qualify/qualifies/qualified/qualification, eligible/eligibility — an
# eligibility determination stated as fact.
_QUALIFY_RE = re.compile(r"\bqualif(y|ies|ied|ication)\b|\beligib(le|ility)\b", re.IGNORECASE)

# A bare date: a month name plus a day number ("August 26th", "Aug. 26"), or a
# numeric date (8/26, 8-26-2026). Not a bare day-of-week or "tomorrow" —
# those are relative, not a specific deadline being asserted as fact.
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"\b({_MONTHS})\.?\s+\d{{1,2}}(st|nd|rd|th)?\b"
    rf"|\b\d{{1,2}}[/-]\d{{1,2}}([/-]\d{{2,4}})?\b",
    re.IGNORECASE,
)

_CHECKS = (
    (_CURRENCY_RE, "states a currency amount"),
    (_LIMIT_RE, "uses the word 'limit'"),
    (_QUALIFY_RE, "states a qualify/eligibility determination"),
    (_DATE_RE, "states a bare date"),
)


def check_never_say(reply_text: str) -> str | None:
    """A violation reason, or None if the reply is clean.

    Checks in a fixed order and returns the FIRST match — good enough to
    trigger a regeneration or the fallback line; a reply failing more than
    one check is not meaningfully worse than failing one.
    """
    for pattern, reason in _CHECKS:
        if pattern.search(reply_text or ""):
            return reason
    return None
