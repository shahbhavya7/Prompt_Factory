"""Measure the renewal cache's serve bar instead of guessing it — two
constants in this codebase were already wrong in opposite directions from a
guess (answer_cache.py's CACHE_THRESHOLD history).

Scope: answer_cache_renewal must already be seeded (scripts/load_kb_renewal.py)
before this runs. For each T1/T3 rule, ONE held-out paraphrase is generated
mechanically (never one of the seeded variants themselves — see
_paraphrase) and embedded, then compared by cosine, in a single query against
every seeded row, to:

  (i)   the best-scoring row belonging to its OWN rule       -> "own"
  (ii)  the best-scoring row belonging to a DIFFERENT rule   -> "other-rule"
  (iii) the best-scoring row in a DIFFERENT TIER             -> "other-tier"

Prints the three distributions and the gaps between them, so
CACHE_THRESHOLD_RENEWAL and CACHE_MARGIN can be read off the data rather than
guessed.

Run:  SACE_CAMPAIGN=renewal python scripts/measure_cache_bar_renewal.py
"""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SACE_CAMPAIGN", "renewal")

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sql_text

from sace_chat import campaign
from sace_chat.db import engine as db_engine
from sace_chat.embeddings import embed_many, get_embedder

_CACHEABLE_TIERS = ("T1", "T3")

# Mechanical paraphrase templates — never the canonical question or a
# caller_phrasing verbatim, so this measures GENERALISATION, not recall of a
# stored string. Deliberately simple (no LLM call per paraphrase, so this
# script stays cheap to re-run): a caller-ish wrapper around the rule's own
# canonical question, reworded just enough to differ from every seeded row.
_TEMPLATES = [
    "hey, quick question — {q}",
    "sorry, one thing — {q}",
    "can you tell me, {q}",
    "I was wondering, {q}",
]


def _paraphrase(question: str, idx: int) -> str:
    q = question.strip().rstrip("?").lower()
    template = _TEMPLATES[idx % len(_TEMPLATES)]
    return template.format(q=q) + "?"


def main() -> int:
    cfg = campaign.get_campaign()
    if cfg.name != "renewal":
        print(f"refusing to run: SACE_CAMPAIGN={cfg.name!r}, expected 'renewal'")
        return 1

    rules = [r for r in campaign.load_rules(cfg) if r.tier in _CACHEABLE_TIERS]
    embedder = get_embedder()

    with db_engine.connect() as conn:
        n_seeded = conn.execute(
            sql_text("SELECT count(*) FROM answer_cache_renewal WHERE source = 'seed'")
        ).scalar()
    if not n_seeded:
        print("answer_cache_renewal has no seed rows — run scripts/load_kb_renewal.py first")
        return 1

    paraphrases = [_paraphrase(r.title, i) for i, r in enumerate(rules)]
    vectors = embed_many(embedder, paraphrases)

    own, other_rule, other_tier = [], [], []
    with db_engine.connect() as conn:
        for rule, vec in zip(rules, vectors):
            rows = conn.execute(
                sql_text(
                    "SELECT governing_rule_id, tier, "
                    "       1 - (embedding <=> CAST(:q AS vector)) AS similarity "
                    "FROM answer_cache_renewal WHERE source = 'seed' "
                    "ORDER BY embedding <=> CAST(:q AS vector) LIMIT 50"
                ),
                {"q": str(list(vec))},
            ).fetchall()
            own_sims = [row.similarity for row in rows if row.governing_rule_id == rule.id]
            other_rule_sims = [row.similarity for row in rows if row.governing_rule_id != rule.id]
            other_tier_sims = [row.similarity for row in rows if row.tier != rule.tier]
            if own_sims:
                own.append(max(own_sims))
            if other_rule_sims:
                other_rule.append(max(other_rule_sims))
            if other_tier_sims:
                other_tier.append(max(other_tier_sims))

    def _report(name, xs):
        if not xs:
            print(f"  {name}: no samples")
            return
        xs_sorted = sorted(xs)
        print(
            f"  {name}: n={len(xs)}  min={xs_sorted[0]:.4f}  "
            f"p25={xs_sorted[len(xs) // 4]:.4f}  "
            f"median={statistics.median(xs):.4f}  "
            f"p75={xs_sorted[3 * len(xs) // 4]:.4f}  "
            f"max={xs_sorted[-1]:.4f}"
        )

    print(f"held-out paraphrases: {len(rules)} (one per T1/T3 rule), "
          f"measured against {n_seeded} seeded rows\n")
    print("(i) own rule's best match:")
    _report("own", own)
    print("(ii) a DIFFERENT rule's best match:")
    _report("other-rule", other_rule)
    print("(iii) a DIFFERENT TIER's best match:")
    _report("other-tier", other_tier)

    if own and other_rule:
        gap_lo = min(own)
        gap_hi = max(other_rule)
        print(f"\ngap between weakest 'own' ({gap_lo:.4f}) and strongest "
              f"'other-rule' ({gap_hi:.4f}): {gap_lo - gap_hi:+.4f}")
        suggested_threshold = (statistics.median(own) + statistics.median(other_rule)) / 2
        print(f"suggested CACHE_THRESHOLD_RENEWAL (midpoint of medians): "
              f"{suggested_threshold:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
