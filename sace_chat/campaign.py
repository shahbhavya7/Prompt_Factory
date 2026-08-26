"""Which campaign a call belongs to, resolved once via SACE_CAMPAIGN.

Wiring only: retrieval, the answer cache, and the LLM fallback path already
work — a CampaignConfig just says which tables and prompt material a given
call should point them at, so Engine/voice_agent.py/load_kb.py don't have to
hardcode the coverage campaign's chunks/answer_cache/kb.STABLE_CORE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    chunks_table: str
    cache_table: str
    stable_core: str
    placeholders: dict = field(default_factory=dict)
    intent_exemplars: dict = field(default_factory=dict)
    valid_intents: frozenset = field(default_factory=frozenset)


_REGISTRY: dict[str, CampaignConfig] = {}


def register(config: CampaignConfig) -> CampaignConfig:
    _REGISTRY[config.name] = config
    return config


def get_campaign(name: str | None = None) -> CampaignConfig:
    """Resolves via SACE_CAMPAIGN when `name` is not given explicitly."""
    name = name or os.environ.get("SACE_CAMPAIGN", "coverage")
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown campaign {name!r} (from SACE_CAMPAIGN) — registered: {sorted(_REGISTRY)}"
        ) from None


# ─────────────────────────── coverage (existing, default) ───────────────────
from sace_chat.assemble import DEMO_PLACEHOLDERS  # noqa: E402
from sace_chat.kb import INTENT_EXEMPLARS, STABLE_CORE  # noqa: E402

COVERAGE = register(CampaignConfig(
    name="coverage",
    chunks_table="chunks",
    cache_table="answer_cache",
    stable_core=STABLE_CORE,
    placeholders=DEMO_PLACEHOLDERS,
    intent_exemplars=INTENT_EXEMPLARS,
    valid_intents=frozenset(INTENT_EXEMPLARS.keys()),
))


# ─────────────────────────────────── renewal ─────────────────────────────────
from sace_chat.kb_renewal import RULES as RENEWAL_RULES  # noqa: E402

# Bare frame — there is no flow on this branch yet (no packets, no address
# updates, no consent gate, no case record). Only what a caller needs to trust
# the call and get an answer or a transfer.
RENEWAL_STABLE_CORE = """\
# ROLE
You are Maya, calling on behalf of {business_entity} about a patient's yearly \
Medi-Cal renewal. This call is recorded for training. You can continue this \
call in Spanish if the caller prefers. Help is free. The caller can be \
transferred to a person at any time, just by asking.

# HOW YOU DECIDE WHAT TO SAY
The GOVERNING RULE below is the only thing that determines this turn's reply. \
You follow it. You do not add questions, sentences or closings from your own \
sense of how such calls usually go, and you do not take any line from the \
REFERENCE section.

HARD PROHIBITION — never state a dollar amount, an income limit, an \
eligibility determination, or a specific date unless the governing rule gives \
you that exact wording to speak. If you are unsure, say so and offer to \
transfer the caller to a person. Never guess.

If the governing rule gives a scripted line, that line is your reply. Match \
its wording, or rephrase lightly for a natural read, but never swap in a \
different subject.

# SAFETY
No medical advice or diagnoses. Everything discussed is PHI.
"""

RENEWAL_PLACEHOLDERS = {
    "{business_entity}": "Santa Rosa Community Health",
}

# Spoken once, before the caller has said anything — there is nothing to
# retrieve on yet. Identifies Maya, states the purpose, invites questions.
RENEWAL_OPENING_LINE = (
    "Hi, I'm Maya, calling on behalf of Santa Rosa Community Health about your "
    "Medi-Cal renewal. Do you have a couple of minutes? I'm happy to answer "
    "any questions you have about it."
)

# Anything that matches nothing — above or below the cache/pool bar. Never a
# fabricated answer: acknowledge, offer a transfer, stop.
RENEWAL_FALLBACK_REPLY = (
    "I don't have that answer for you, and I don't want to guess. Let me "
    "connect you with someone at Santa Rosa Community Health who can help — "
    "thanks for your patience."
)

# "16-topic intent exemplars built from the CSV's 'how patients actually say
# it' column" — every rule's cue_variants, grouped by its topic-derived
# intent. Tier-agnostic on purpose: intent routing answers "which topic is
# this", not "is this a cacheable topic" — a T2 or T4 question still has to
# route to its own governing rule.
RENEWAL_INTENT_EXEMPLARS: dict[str, list[str]] = {}
for _rule in RENEWAL_RULES:
    RENEWAL_INTENT_EXEMPLARS.setdefault(_rule.intent, []).extend(_rule.cue_variants)
del _rule

RENEWAL = register(CampaignConfig(
    name="renewal",
    chunks_table="chunks_renewal",
    cache_table="answer_cache_renewal",
    stable_core=RENEWAL_STABLE_CORE,
    placeholders=RENEWAL_PLACEHOLDERS,
    intent_exemplars=RENEWAL_INTENT_EXEMPLARS,
    valid_intents=frozenset(RENEWAL_INTENT_EXEMPLARS.keys()),
))
