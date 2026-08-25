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
    cache_table: str = "answer_cache"
    stable_core: str = field(default="", repr=False)
    placeholders: dict = field(default_factory=dict)
    intent_exemplars: dict = field(default_factory=dict)
    valid_intents: frozenset = field(default_factory=frozenset)


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


_register_coverage()
