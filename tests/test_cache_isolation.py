"""Phase 2E, Part C — the two campaigns' reply caches must not leak into
each other.

Before this fix, `answer_cache` was shared: `CampaignConfig.cache_table`
existed but nothing actually passed it to `answer_cache.lookup`/`store`, so
every campaign read and wrote the same table. That was unsafe on the five
safety-label intents both kb.py and kb_renewal.py reuse verbatim (dnc, abuse,
medical_emergency, garbled_audio, frustration): a reply cached from a
coverage call could be served to a renewal caller under the wrong clinic's
script, on the highest-stakes intents in the system.

This stores one row directly in each table (no LLM call needed — cache
isolation is a property of which table is queried, not of what produced the
row) and asserts a lookup against the OTHER table never sees it.

Run:  python tests/test_cache_isolation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sace_chat import answer_cache
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import get_embedder

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def main():
    init_db()
    embedder = get_embedder()
    answer_cache.clear(table="answer_cache")
    answer_cache.clear(table="answer_cache_renewal")

    with db_engine.connect() as conn:
        # NEVER_CACHE_INTENTS makes lookup() itself refuse "dnc" outright —
        # which is correct behaviour, but it also means it can never prove
        # cross-table isolation (a None result is ambiguous between "refused
        # because dnc" and "refused because wrong table"). A neutral,
        # cacheable intent isolates the property actually under test.
        neutral_vec = embedder.embed("what is this packet about")
        cov_id2 = answer_cache.store(
            question="what is this packet about", question_vec=neutral_vec,
            reply="This is coverage's own explanation of the letter.",
            intent="the_letter", governing_rule_id="coverage_letter_rule",
            table="answer_cache",
        )
        ren_id2 = answer_cache.store(
            question="what is this packet about", question_vec=neutral_vec,
            reply="This is renewal's own explanation of the packet.",
            intent="the_letter", governing_rule_id="renewal_letter_rule",
            table="answer_cache_renewal",
        )
        record("setup: stored a neutral-intent row in each table",
               bool(cov_id2) and bool(ren_id2), f"{cov_id2}, {ren_id2}")

        hit_from_coverage = answer_cache.lookup(conn, neutral_vec, "the_letter", table="answer_cache")
        hit_from_renewal = answer_cache.lookup(conn, neutral_vec, "the_letter", table="answer_cache_renewal")

        record("a lookup against answer_cache returns coverage's own row",
               hit_from_coverage is not None
               and hit_from_coverage["governing_rule_id"] == "coverage_letter_rule",
               f"got {hit_from_coverage}")
        record("a lookup against answer_cache_renewal returns renewal's own row",
               hit_from_renewal is not None
               and hit_from_renewal["governing_rule_id"] == "renewal_letter_rule",
               f"got {hit_from_renewal}")
        record("coverage's lookup never returns the renewal row",
               hit_from_coverage is None or hit_from_coverage["governing_rule_id"] != "renewal_letter_rule")
        record("renewal's lookup never returns the coverage row",
               hit_from_renewal is None or hit_from_renewal["governing_rule_id"] != "coverage_letter_rule")

    # Cleanup — this test's rows must not linger and pollute a later run.
    answer_cache.clear(table="answer_cache")
    answer_cache.clear(table="answer_cache_renewal")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
