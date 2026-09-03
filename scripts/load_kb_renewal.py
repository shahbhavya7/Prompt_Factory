"""Load the renewal KB into the isolated sace_kb database.

One parse (sace_chat.kb_renewal.RULES), three destinations, written in the
same run so they cannot drift apart:

  chunks_renewal        165 rows, one per entry — the rule pool itself. Holds
                         the spoken text; joined to by the cue index below.
  chunks_renewal_cues   one row per caller phrasing, ALL tiers — the retrieval
                         index. See db.init_db for the measurement that
                         justifies it (46.1% -> 97.0% rule accuracy): the
                         per-rule vector in chunks_renewal is an average of
                         every phrasing and matches none of them well.
  answer_cache_renewal  one row per caller phrasing plus the canonical
                         question, all pointing at the SAME verbatim answer
                         text — a deterministic "we already know this"
                         shortcut, source='seed'. T1/T3 ONLY.

The cue index and the answer cache are seeded from the same list of phrasings
and embedded in ONE batched call shared between them, so a phrasing can never
be in one and missing from the other, and the run costs one embedding pass
rather than two.

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


def phrasings(rule):
    """Every distinct way this rule's question is asked: the canonical title
    first, then each cue variant. Deduplicated, order-stable, so the row ids
    derived from the index are stable across reloads."""
    seen = set()
    out = []
    for kind, q in [("title", rule.title.strip()),
                    *[("cue", v.strip()) for v in rule.cue_variants]]:
        if q and q not in seen:
            seen.add(q)
            out.append((kind, q))
    return out


def build_index_entries():
    """(rule, kind, phrasing) for the cue index — EVERY tier.

    Unrestricted by tier on purpose: this index decides which rule governs a
    turn, and a T2 or T4 question must still find its own rule. What must never
    happen is REPLAYING one, and that is the answer cache's restriction, below.
    """
    return [(rule, kind, q) for rule in RULES for kind, q in phrasings(rule)]


def build_cache_entries():
    """(rule, question) pairs to seed — canonical + every phrasing, T1/T3 only."""
    return [(rule, q) for rule in RULES if rule.tier not in NEVER_CACHE_TIERS
            for _kind, q in phrasings(rule)]


def load_cue_index(session, vec_by_text, entries):
    """The retrieval index. Rebuilt wholesale from RULES each run — it is pure
    derived data, so there is nothing in it worth preserving across a reload."""
    session.execute(text("DELETE FROM chunks_renewal_cues WHERE source = 'seed'"))
    for i, (rule, kind, q) in enumerate(entries):
        session.execute(
            text(
                "INSERT INTO chunks_renewal_cues "
                "(id, rule_id, variant, kind, intent, tier, source, embedding) "
                "VALUES (:id, :rule_id, :variant, :kind, :intent, :tier, 'seed', "
                "        CAST(:embedding AS vector))"
            ),
            {"id": f"cue_{rule.id}_{i}", "rule_id": rule.id, "variant": q,
             "kind": kind, "intent": rule.intent, "tier": rule.tier,
             "embedding": str(list(vec_by_text[q]))},
        )
    return len(entries)


def load_answer_cache_renewal(session, vec_by_text, entries):
    leaked_tier = [r.id for r, _ in entries if r.tier in NEVER_CACHE_TIERS]
    if leaked_tier:
        raise AssertionError(f"HARD BLOCK violated — T2/T4 rule(s) reached the cache build: {leaked_tier}")
    no_intent = [r.id for r, _ in entries if not r.intent]
    if no_intent:
        raise AssertionError(f"rule(s) with no intent would seed an unscoped cache row: {no_intent}")

    # Vectors come from the single batched pass in main() — the same vector a
    # phrasing has in the cue index, so the two tables cannot disagree about
    # where a question sits in vector space.
    vecs = [vec_by_text[q] for _, q in entries]
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
    # The cue index must cover every rule: a rule with no row here is
    # unreachable by retrieval entirely — invisible rather than merely
    # hard to match, which is the failure mode worth a hard check.
    orphans = conn.execute(text(
        "SELECT count(*) FROM chunks_renewal c "
        "WHERE NOT EXISTS (SELECT 1 FROM chunks_renewal_cues v WHERE v.rule_id = c.id)"
    )).scalar()
    if orphans:
        raise AssertionError(
            f"post-insert check failed: {orphans} rule(s) have no cue-index row "
            f"and can never be retrieved"
        )
    return {
        "chunks_renewal": conn.execute(text("SELECT count(*) FROM chunks_renewal")).scalar(),
        "chunks_renewal_cues": conn.execute(
            text("SELECT count(*) FROM chunks_renewal_cues")).scalar(),
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

    index_entries = build_index_entries()
    cache_entries = build_cache_entries()

    # ONE batched embedding pass over every distinct phrasing, shared by the
    # cue index and the answer cache. Doing it twice would cost twice as much
    # and, worse, allow the two tables to hold different vectors for the same
    # string if the model ever changed between the calls.
    distinct = sorted({q for _, _, q in index_entries})
    vec_by_text = dict(zip(distinct, embed_many(embedder, distinct)))

    with SessionLocal() as session:
        n_chunks = load_chunks_renewal(session, embedder)
        n_index = load_cue_index(session, vec_by_text, index_entries)
        n_cache = load_answer_cache_renewal(session, vec_by_text, cache_entries)
        session.commit()

    with engine.connect() as conn:
        counts = verify(conn)

    print(f"chunks_renewal:       {n_chunks} rules")
    print(f"chunks_renewal_cues:  {n_index} phrasings (all tiers) — the retrieval index")
    print(f"answer_cache_renewal: {n_cache} seed rows (T1/T3 only)")
    print(f"counts: {counts}")
    print("OK — every rule has a cue row; zero T2/T4 cache rows, zero intent-NULL, live rows preserved")


if __name__ == "__main__":
    sys.exit(main())
