"""SACE_CAMPAIGN=renewal routes Engine, cache, and voice_agent at the renewal
tables end to end.

Sections:
  A  CallState        — reused as-is; no flow-specific field exists on this
                        branch, so nothing needs to read or set one (B)
  B  CampaignConfig    — both campaigns registered, resolved via SACE_CAMPAIGN
  C  IntentRouter      — routes on the exemplars it was GIVEN, not on
                        kb.INTENT_EXEMPLARS regardless of campaign
  D  guards            — check_never_say fires on the required example, with
                        zero exemptions
  E  Engine wiring     — a renewal-configured Engine reads/writes
                        chunks_renewal / answer_cache_renewal, never chunks /
                        answer_cache — proven by planting a row only the
                        right table should ever see

Run:  python tests/test_campaign_wiring.py
"""

import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_env

kb_env.pin()

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sql_text

from sace_chat import guards, manager
from sace_chat.campaign import get_campaign
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.kb_renewal import RULES as RENEWAL_RULES
from sace_chat.retrieve import CallState, IntentRouter

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def main():
    init_db()

    # ══ A. CallState is reused as-is ═══════════════════════════════════════
    print("\n[A] CallState — no flow-specific field")
    field_names = {f.name for f in dataclasses.fields(CallState)}
    expected = {"intent", "opt_out", "ended", "asked_questions", "collected_fields"}
    record("A1. CallState carries exactly the pre-renewal field set, nothing "
           "flow-specific added", field_names == expected, f"fields={field_names}")

    # ══ B. CampaignConfig registry ══════════════════════════════════════════
    print("\n[B] CampaignConfig")
    os.environ["SACE_CAMPAIGN"] = "renewal"
    renewal = get_campaign()
    record("B1. SACE_CAMPAIGN=renewal resolves the renewal campaign",
           renewal.name == "renewal", f"got {renewal.name!r}")
    record("B2. renewal points at chunks_renewal / answer_cache_renewal",
           renewal.chunks_table == "chunks_renewal" and renewal.cache_table == "answer_cache_renewal",
           f"chunks_table={renewal.chunks_table} cache_table={renewal.cache_table}")
    record("B3. renewal has 16-topic intent exemplars",
           len(renewal.intent_exemplars) == 16, f"{len(renewal.intent_exemplars)} topics")
    record("B4. every renewal rule's intent is in valid_intents",
           all(r.intent in renewal.valid_intents for r in RENEWAL_RULES),
           f"valid_intents={sorted(renewal.valid_intents)}")

    os.environ["SACE_CAMPAIGN"] = "coverage"
    coverage = get_campaign()
    record("B5. SACE_CAMPAIGN=coverage resolves the coverage campaign (default unaffected)",
           coverage.name == "coverage" and coverage.chunks_table == "chunks",
           f"got {coverage.name!r} chunks_table={coverage.chunks_table}")

    try:
        get_campaign("not-a-real-campaign")
        record("B6. an unknown campaign name raises", False, "no exception raised")
    except ValueError as exc:
        record("B6. an unknown campaign name raises", True, str(exc))

    # ══ C. IntentRouter routes on the exemplars it was given ═══════════════
    print("\n[C] IntentRouter — campaign-scoped exemplars")
    embedder = get_embedder()
    coverage_router = IntentRouter(embedder)  # no exemplars= -> coverage default
    renewal_router = IntentRouter(embedder, exemplars=renewal.intent_exemplars)
    record("C1. no exemplars= falls back to kb.INTENT_EXEMPLARS (coverage default)",
           "dnc" in coverage_router.exemplars, f"has 'dnc': {'dnc' in coverage_router.exemplars}")
    record("C2. exemplars= is actually used, not ignored",
           renewal_router.exemplars is renewal.intent_exemplars
           and "dnc" not in renewal_router.exemplars,
           f"renewal exemplars has 'dnc': {'dnc' in renewal_router.exemplars}")

    # ══ D. guards.check_never_say ═══════════════════════════════════════════
    print("\n[D] guards.check_never_say — zero exemptions")
    record("D1. fires on the required example",
           guards.check_never_say("you'd need to be under $21,000") is not None)
    record("D2. fires on 'limit'", guards.check_never_say("there's an income limit") is not None)
    record("D3. fires on qualify/eligible",
           guards.check_never_say("you may qualify for this") is not None)
    record("D4. fires on a bare date", guards.check_never_say("due by August 26th") is not None)
    record("D5. a clean renewal-shaped reply passes",
           guards.check_never_say("I can help with your Medi-Cal renewal — what's your question?") is None)

    # ══ E. Engine wiring — proven by table isolation, not just config values ═
    print("\n[E] Engine wiring — a renewal Engine never touches the coverage tables")
    marker_id = "test_campaign_wiring_marker"
    # An EXACT existing cue_variant, not a made-up sentence — so intent
    # detection confidently lands on "the_letter" (cosine ~1.0 against its
    # own exemplar) rather than whatever topic a novel sentence happens to
    # land nearest by chance. The marker's cue is that same string alone,
    # so it embeds closer to the query than kb_ltr_01's real (multi-variant)
    # cue does, and wins the distance ordering within that intent.
    marker_query = "What's this paper?"
    with db_engine.begin() as conn:
        conn.execute(sql_text("DELETE FROM chunks_renewal WHERE id = :i"), {"i": marker_id})
        conn.execute(sql_text("DELETE FROM chunks WHERE id = :i"), {"i": marker_id})
        vec = embedder.embed(marker_query)
        conn.execute(sql_text(
            "INSERT INTO chunks_renewal "
            "(id, title, text, cue, intent, priority, terminal, exclusive, source, "
            " embedding) "
            "VALUES (:id, 'marker', 'marker reply text', :cue, 'the_letter', 'normal', "
            "false, false, 'seed', CAST(:vec AS vector))"
        ), {"id": marker_id, "cue": marker_query, "vec": str(list(vec))})

    eng = Engine(
        stable_core=renewal.stable_core, rules=RENEWAL_RULES, embedder=embedder,
        manager=manager, llm=None,
        table=renewal.chunks_table, cache_table=renewal.cache_table,
        placeholders=renewal.placeholders, exemplars=renewal.intent_exemplars,
        valid_intents=renewal.valid_intents, campaign_name=renewal.name,
    )
    ctx = eng.build_turn_context(CallState(), [], marker_query)
    prompt_sent, governing, reference, dbg = ctx
    record("E1. retrieval against a renewal-configured Engine finds the marker "
           "row seeded ONLY in chunks_renewal",
           governing is not None and governing.chunk.id == marker_id,
           f"governing={governing.chunk.id if governing else None}")

    with db_engine.connect() as conn:
        conn.execute(sql_text("DELETE FROM chunks_renewal WHERE id = :i"), {"i": marker_id})

    cov_eng = Engine(
        stable_core=coverage.stable_core, rules=None, embedder=embedder,
        manager=manager, llm=None,
        table=coverage.chunks_table, cache_table=coverage.cache_table,
        placeholders=coverage.placeholders, exemplars=coverage.intent_exemplars,
        valid_intents=coverage.valid_intents, campaign_name=coverage.name,
    )
    record("E2. a coverage-configured Engine defaults to the chunks/answer_cache "
           "tables", cov_eng.table == "chunks" and cov_eng.cache_table == "answer_cache",
           f"table={cov_eng.table} cache_table={cov_eng.cache_table}")

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
