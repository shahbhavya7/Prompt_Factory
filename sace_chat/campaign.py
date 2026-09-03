"""Which campaign a call belongs to, resolved once via SACE_CAMPAIGN.

Wiring only: retrieval, the answer cache, and the LLM fallback path already
work — a CampaignConfig just says which tables and prompt material a given
call should point them at, so Engine/voice_agent.py/load_kb.py don't have to
hardcode the coverage campaign's chunks/answer_cache/kb.STABLE_CORE.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from sace_chat.answer_cache import CacheBar


@dataclass(frozen=True)
class CampaignConfig:
    name: str
    chunks_table: str
    cache_table: str
    stable_core: str
    # Per-variant retrieval index for this campaign, or None to use the
    # intent-hop path. See db.init_db / retrieve._fetch_by_cue for the
    # measurement (46.1% -> 97.0% rule accuracy) that motivates it.
    cue_table: str | None = None
    placeholders: dict = field(default_factory=dict)
    intent_exemplars: dict = field(default_factory=dict)
    valid_intents: frozenset = field(default_factory=frozenset)
    # This campaign's measured answer-cache serve bar. None means
    # answer_cache.DEFAULT_BAR (threshold only) — the behaviour every campaign
    # had before a measured margin existed.
    cache_bar: CacheBar | None = None
    # The measured bar for speaking a verbatim KB rule directly, with no LLM
    # call (see engine.verbatim_decision). Same two-number shape as the cache
    # bar — "close enough, and unambiguously closer than the runner-up" — so it
    # reuses CacheBar rather than defining an identical type. None disables the
    # path, which is the right default for a campaign whose rules are facts to
    # paraphrase rather than authored spoken lines.
    verbatim_bar: CacheBar | None = None


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

# Intent exemplars built from the CSV's "how patients actually say it" column:
# every rule's cue_variants, grouped by its topic-derived intent. 26 topics on
# the current sheet, up from 16 — the patient-experience review added whole
# areas (digital support, case status, re-enrollment, documents, outreach,
# language access, accessibility, appointments, coaching, member services).
# Tier-agnostic on purpose: intent routing answers "which topic is this", not
# "is this a cacheable topic" — a T2 or T4 question still has to route to its
# own governing rule.
RENEWAL_INTENT_EXEMPLARS: dict[str, list[str]] = {}
for _rule in RENEWAL_RULES:
    RENEWAL_INTENT_EXEMPLARS.setdefault(_rule.intent, []).extend(_rule.cue_variants)
del _rule

# The renewal cache's serve bar is MEASURED, not chosen here.
# scripts/measure_cache_bar_renewal.py runs held-out paraphrases through the
# real answer_cache.lookup and writes the bar that produces zero wrong serves
# into data/renewal/eval/cache_bar.json. Loading it means the number the
# measurement reported is the number the live call enforces — the two cannot
# drift, because there is only one of them.
#
# Falls back to the module default if the file is missing or malformed rather
# than failing to import: a campaign that has not been measured yet should
# behave exactly as it did before, not refuse to start.
_CACHE_BAR_FILE = Path(__file__).resolve().parent.parent / "data" / "renewal" / "eval" / "cache_bar.json"


def _load_measured_bar(path: Path) -> CacheBar | None:
    try:
        measured = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(measured["cache_threshold_renewal"])
        margin = measured.get("cache_margin")
        if margin is None:
            # The measurement ran but found NO margin that made the serve
            # decision safe (see the script's closing message). Refusing to
            # guess one is the whole point; serve nothing rather than serve at
            # a bar nothing verified.
            print(f"[campaign] {path.name} reports no safe margin — renewal cache "
                  f"left on the default bar; re-measure before relying on it")
            return None
        return CacheBar(threshold=threshold, margin=float(margin),
                        require_tier_agreement=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[campaign] could not read a measured cache bar from {path.name} "
              f"({type(exc).__name__}: {exc}); using the default")
        return None


def _load_verbatim_bar(path: Path) -> CacheBar | None:
    """The measured bar for the verbatim fast path, from the same file.

    Absent means the campaign has not measured one, and the path stays OFF.
    That default is deliberate: speaking a rule as written with no model in the
    loop is only correct when someone has checked that retrieval is accurate
    enough at that bar to do it, and the honest behaviour without that evidence
    is to keep the full pipeline.
    """
    try:
        measured = json.loads(path.read_text(encoding="utf-8"))
        vb = measured.get("verbatim_bar")
        if not vb:
            return None
        return CacheBar(threshold=float(vb["threshold"]), margin=float(vb["margin"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[campaign] could not read a measured verbatim bar from {path.name} "
              f"({type(exc).__name__}: {exc}); the verbatim fast path stays off")
        return None


RENEWAL_CACHE_BAR = _load_measured_bar(_CACHE_BAR_FILE)
RENEWAL_VERBATIM_BAR = _load_verbatim_bar(_CACHE_BAR_FILE)

RENEWAL = register(CampaignConfig(
    name="renewal",
    chunks_table="chunks_renewal",
    cache_table="answer_cache_renewal",
    cue_table="chunks_renewal_cues",
    stable_core=RENEWAL_STABLE_CORE,
    placeholders=RENEWAL_PLACEHOLDERS,
    intent_exemplars=RENEWAL_INTENT_EXEMPLARS,
    valid_intents=frozenset(RENEWAL_INTENT_EXEMPLARS.keys()),
    cache_bar=RENEWAL_CACHE_BAR,
    verbatim_bar=RENEWAL_VERBATIM_BAR,
))
