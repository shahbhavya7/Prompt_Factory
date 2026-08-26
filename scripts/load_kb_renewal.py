"""Load the renewal KB into the isolated sace_kb database.

One parse (sace_chat.kb_renewal.RULES), two destinations, written in the same
run so they cannot drift apart:

  chunks_renewal        126 rows, one per entry — the main-memory / fallback
                         pool, read the same way kb.RULES is.
  answer_cache_renewal  one row per caller phrasing (cue_variants) plus the
                         canonical question (title), all pointing at the SAME
                         verbatim answer text — a deterministic "we already
                         know this" shortcut, source='seed'.

T2 (case-record) and T4 (immigration/self-harm) are HARD BLOCKED from the
cache: a case-record answer is per-caller and cannot be replayed to a
stranger, and a stale cached line on a T4 topic is a safety failure, not a
UX one. Only T1/T3 rules are written to answer_cache_renewal at all, and the
load additionally asserts none leaked through before AND after the insert.

Run:  python scripts/load_kb_renewal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# .env.kb pins DATABASE_URL / SACE_EXPECTED_DB at the isolated sace_kb stack —
# loaded with override=True, and loaded FIRST, so nothing below can clobber it.
# The plain .env is loaded after (never override=True) purely for SACE_LLM_KEY —
# get_embedder() needs it and .env.kb deliberately does not duplicate secrets.
load_dotenv(ROOT / ".env.kb", override=True)
load_dotenv(ROOT / ".env")

from sqlalchemy import text  # noqa: E402

from sace_chat.db import SessionLocal, check_embedding, engine, init_db, insert_chunk  # noqa: E402
from sace_chat.embeddings import embed_many, get_embedder  # noqa: E402
from sace_chat.kb_renewal import RULES  # noqa: E402

NEVER_CACHE_TIERS = {"T2", "T4"}


def load_chunks_renewal(session, embedder):
    cues = [(r.cue or "").strip() or r.text for r in RULES]
    vecs = embed_many(embedder, cues)
    session.execute(text("DELETE FROM chunks_renewal WHERE source != 'learned'"))
    for rule, vec in zip(RULES, vecs):
        insert_chunk(session, rule, embedder, table="chunks_renewal", embedding=vec)
    return len(RULES)


def build_cache_entries():
    """(rule, question) pairs to seed — canonical + every phrasing, T1/T3 only."""
    entries = []
    for rule in RULES:
        if rule.tier in NEVER_CACHE_TIERS:
            continue
        seen = set()
        for q in [rule.title.strip(), *[v.strip() for v in rule.cue_variants]]:
            if q and q not in seen:
                seen.add(q)
                entries.append((rule, q))
    return entries


def load_answer_cache_renewal(session, embedder, entries):
    leaked_tier = [r.id for r, _ in entries if r.tier in NEVER_CACHE_TIERS]
    if leaked_tier:
        raise AssertionError(f"HARD BLOCK violated — T2/T4 rule(s) reached the cache build: {leaked_tier}")
    no_intent = [r.id for r, _ in entries if not r.intent]
    if no_intent:
        raise AssertionError(f"rule(s) with no intent would seed an unscoped cache row: {no_intent}")

    questions = [q for _, q in entries]
    vecs = embed_many(embedder, questions)  # ONE batched call, never one per row
    for (rule, q), vec in zip(entries, vecs):
        check_embedding(vec, chunk_id=f"{rule.id}:{q[:24]}")

    session.execute(text("DELETE FROM answer_cache_renewal WHERE source = 'seed'"))
    for i, ((rule, q), vec) in enumerate(zip(entries, vecs)):
        session.execute(
            text(
                "INSERT INTO answer_cache_renewal "
                "(id, question, embedding, reply, intent, governing_rule_id, "
                " tier, source, pending_fingerprint, hit_count, active) "
                "VALUES "
                "(:id, :question, CAST(:embedding AS vector), :reply, :intent, "
                " :governing_rule_id, :tier, 'seed', '', 0, TRUE)"
            ),
            {
                "id": f"cache_seed_{rule.id}_{i}",
                "question": q,
                "embedding": str(list(vec)),
                "reply": rule.text,
                "intent": rule.intent,
                "governing_rule_id": rule.id,
                "tier": rule.tier,
            },
        )
    return len(entries)


def verify(conn):
    bad_tier = conn.execute(
        text("SELECT count(*) FROM answer_cache_renewal WHERE tier IN ('T2','T4')")
    ).scalar()
    if bad_tier:
        raise AssertionError(f"post-insert check failed: {bad_tier} T2/T4 row(s) in answer_cache_renewal")
    null_intent = conn.execute(
        text("SELECT count(*) FROM answer_cache_renewal WHERE intent IS NULL")
    ).scalar()
    if null_intent:
        raise AssertionError(f"post-insert check failed: {null_intent} intent-NULL row(s) in answer_cache_renewal")
    return {
        "chunks_renewal": conn.execute(text("SELECT count(*) FROM chunks_renewal")).scalar(),
        "chunks_renewal_seed": conn.execute(
            text("SELECT count(*) FROM chunks_renewal WHERE source = 'seed'")).scalar(),
        "answer_cache_renewal": conn.execute(text("SELECT count(*) FROM answer_cache_renewal")).scalar(),
        "answer_cache_renewal_seed": conn.execute(
            text("SELECT count(*) FROM answer_cache_renewal WHERE source = 'seed'")).scalar(),
        "answer_cache_renewal_live": conn.execute(
            text("SELECT count(*) FROM answer_cache_renewal WHERE source = 'live'")).scalar(),
    }


def main():
    init_db()
    embedder = get_embedder()

    with SessionLocal() as session:
        n_chunks = load_chunks_renewal(session, embedder)
        entries = build_cache_entries()
        n_cache = load_answer_cache_renewal(session, embedder, entries)
        session.commit()

    with engine.connect() as conn:
        counts = verify(conn)

    print(f"chunks_renewal: wrote {n_chunks} rows -> {counts}")
    print(f"answer_cache_renewal: wrote {n_cache} seed rows -> {counts}")
    print("OK — zero T2/T4 rows, zero intent-NULL rows, live rows preserved")


if __name__ == "__main__":
    sys.exit(main())
