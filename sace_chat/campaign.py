"""The campaign seam: one deployed script's identity, resolved once per process.

Every front end (`load_kb.py`, `voice_agent.py`, `streamlit_app.py`) and the
`Engine` itself currently reach directly into `sace_chat.kb` for `STABLE_CORE`
and `RULES`, and hardcode `table="chunks"`. That is fine while there is only
one script. It stops being fine the moment a second one exists: two campaigns
cannot share one `chunks` table (a Coverage rule and a Renewal rule must never
compete for the same turn), and a front end that imports `sace_chat.kb` by
name has no way to run the other script at all.

A `CampaignConfig` is the unit that varies between scripts: which module
supplies the rules, which Postgres tables back the rule pool and the answer
cache, and the placeholder/intent vocabulary that goes with that rule set.
Nothing here changes how retrieval, assembly or the engine work — `Engine`
already takes `table=`, and this module exists only to decide what value that
table name (and the rest of the KB) should be, once, from `SACE_CAMPAIGN`.

Deliberately NOT here: any rule text, answer text or caller phrasing. Adding a
second campaign means writing a new `kb`-shaped module and registering it —
this module only wires the seam, never the content.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CampaignConfig:
    """Everything about one deployed script that the shared engine needs to
    know but must never hardcode.

    `kb_module` — not `rules` itself — is what's stored, because the rule list
    is the one piece of KB content big enough that copying it into every
    config would be its own source of drift. `load_rules()` below is the one
    place that turns a module name into the list `Engine` wants.
    """

    name: str
    kb_module: str
    chunks_table: str = "chunks"
    # Wired into every reply-cache call site (retrieve.py's lookup, engine.py's
    # store/record_hit — see Engine.cache_table). A shared table was briefly
    # "safe" on the premise that intent vocabularies never collide between
    # campaigns, but that premise was false: coverage's kb.py and
    # kb_renewal.py both reuse five safety-label intents verbatim (dnc,
    # abuse, medical_emergency, garbled_audio, frustration), so a
    # coverage-learned reply on one of those could be replayed to a renewal
    # caller under the wrong clinic's script. See db.py's answer_cache_renewal.
    cache_table: str = "answer_cache"
    stable_core: str = field(default="", repr=False)
    placeholders: dict = field(default_factory=dict)
    intent_exemplars: dict = field(default_factory=dict)
    valid_intents: frozenset = field(default_factory=frozenset)
    # (reply_text, case_record) -> (ok, reason) — run in Engine.prepare_reply
    # only, and only when this campaign supplies one. None disables it
    # entirely; coverage's registration below leaves this at the default.
    never_say_guard: object = None
    # The safe line spoken when a never-say violation survives the
    # regeneration budget and the governing rule is not itself a transfer
    # rule with its own line to fall back to. "" for a campaign with no guard.
    never_say_fallback: str = ""


_CAMPAIGNS: dict[str, CampaignConfig] = {}


def register(config: CampaignConfig) -> None:
    """Add a campaign to the registry. Re-registering a name overwrites it —
    useful for tests, otherwise expected to be called exactly once per name at
    import time, below."""
    _CAMPAIGNS[config.name] = config


def get_campaign(name: str | None = None) -> CampaignConfig:
    """Resolve a campaign by name, or by `SACE_CAMPAIGN`, defaulting to
    `coverage` — the one campaign that exists today, so an unset environment
    behaves exactly as every front end already did before this module
    existed."""
    resolved = name or os.environ.get("SACE_CAMPAIGN", "coverage")
    try:
        return _CAMPAIGNS[resolved]
    except KeyError:
        raise ValueError(
            f"unknown campaign {resolved!r}; registered campaigns: {sorted(_CAMPAIGNS)}"
        ) from None


def load_rules(config: CampaignConfig) -> list:
    """The campaign's `RULES` list, imported from its `kb_module`.

    A fresh list, not the module's own — callers are free to mutate what they
    get back (tests append a probe rule) without corrupting the module-level
    list a second caller would import next.
    """
    module = importlib.import_module(config.kb_module)
    return list(module.RULES)


def _register_coverage() -> None:
    """The one campaign this codebase has shipped so far, registered from the
    exact values `sace_chat.kb` / `sace_chat.assemble` / `sace_chat.manager`
    already define — so resolving `coverage` (the default) changes nothing
    about what any existing front end sends."""
    from sace_chat import kb, manager
    from sace_chat.assemble import DEMO_PLACEHOLDERS

    register(CampaignConfig(
        name="coverage",
        kb_module="sace_chat.kb",
        chunks_table="chunks",
        cache_table="answer_cache",
        stable_core=kb.STABLE_CORE,
        placeholders=dict(DEMO_PLACEHOLDERS),
        intent_exemplars=dict(kb.INTENT_EXEMPLARS),
        valid_intents=frozenset(manager.VALID_INTENTS),
    ))


def _register_renewal() -> None:
    """The renewal campaign: a distinct rule pool (chunks_renewal — see
    db.init_db, "a Coverage rule and a Renewal rule must never compete for
    the same turn"), its own compliance guard, and its own placeholder
    values. sace_chat.kb_renewal is a GENERATED file — see
    scripts/build_kb_renewal.py; nothing here authors KB content.
    """
    from sace_chat import guards, kb_renewal

    by_id = {r.id: r for r in kb_renewal.RULES}
    # Sourced from the CSV verbatim (KB-IMM-03), not authored here — see
    # build_kb_renewal.py's module docstring on why a universal fallback
    # exists at all: a never-say violation on a NON-transfer (T1) rule has no
    # rule of its own to fall back to.
    fallback = by_id["kb_imm_03"].text

    register(CampaignConfig(
        name="renewal",
        kb_module="sace_chat.kb_renewal",
        chunks_table="chunks_renewal",
        cache_table="answer_cache_renewal",
        stable_core=kb_renewal.RENEWAL_STABLE_CORE,
        placeholders={
            # Reuses coverage's own patient-identity/callback keys — the
            # renewal CSV's "[clinic name]" and the PDF's illustrative
            # "Maria Reyes"/"Santa Rosa Community Health" are the same KIND
            # of value coverage already templates, so the same keys and
            # substitution mechanism apply; only the values differ per
            # campaign. Demo values only — a real deployment supplies the
            # actual caller's own name and this clinic's own number.
            "{patient_first_name}": "Maria",
            "{patient_last_name}": "Reyes",
            "{callback_number}": "(707) 555-0142",
            "{business_entity}": "Santa Rosa Community Health",
            "{current_month}": "this month",
        },
        intent_exemplars=dict(kb_renewal.RENEWAL_INTENT_EXEMPLARS),
        valid_intents=frozenset(kb_renewal.RENEWAL_VALID_INTENTS),
        never_say_guard=guards.check_never_say,
        never_say_fallback=fallback,
    ))


_register_coverage()
try:
    _register_renewal()
except ImportError:
    # sace_chat.kb_renewal does not exist until scripts/build_kb_renewal.py
    # has been run once — harmless for anyone who has only pulled the
    # coverage campaign so far; get_campaign("renewal") will raise its own
    # clear error if something then asks for it.
    pass
