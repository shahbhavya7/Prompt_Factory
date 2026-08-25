"""Load the renewal campaign's chunk pool AND seed its answer cache, from one
in-memory parse of `campaign.load_rules(cfg)` — so the two destinations can
never drift against each other:

  chunks_renewal        one row per KB entry (main memory / fallback) —
                         delegated to load_kb.main(), the exact same,
                         already-tested loader coverage uses.
  answer_cache_renewal  one row per caller phrasing + the canonical question,
                         source='seed', for every T1/T3 rule — T2 (this
                         caller's own case data) and T4 (safety-critical,
                         must not depend on embedding recall) are hard-
                         blocked, not merely skipped.

Every seed question is embedded in ONE batched call, never one call per row —
see embeddings.embed_many. Re-running this script is safe: `answer_cache.
clear(table=..., source="seed")` drops only what a previous run of THIS
script wrote, never a row a real call earned (source='live').

Run:  SACE_CAMPAIGN=renewal python scripts/load_kb_renewal.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SACE_CAMPAIGN", "renewal")

from dotenv import load_dotenv

load_dotenv()

from sace_chat import answer_cache, campaign
from sace_chat.db import init_db
from sace_chat.embeddings import embed_many, get_embedder

# T2 rules read this caller's own case record — caching one would replay a
# stranger's due date or worker name. T4 rules are the safety net for
# self-harm/abuse/immigration disclosures — Part D's deterministic
# short-circuit and the semantic seed rules in chunks_renewal are the two
# nets for those; the reply cache must never be a third, cosine-dependent one.
_CACHEABLE_TIERS = ("T1", "T3")
_BLOCKED_TIERS = ("T2", "T4")


def _seed_variants(rule) -> list[str]:
    """Every phrasing that should retrieve this rule's answer: the canonical
    question (rule.title, straight from the CSV's "Canonical question"
    column) plus every "how patients actually say it" variant, deduplicated
    so a phrasing identical to the canonical question is not embedded twice."""
    phrasings = list((rule.tags or {}).get("caller_phrasings") or [])
    seen, out = set(), []
    for text in [rule.title, *phrasings]:
        text = (text or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def main() -> int:
    cfg = campaign.get_campaign()
    if cfg.name != "renewal":
        print(f"refusing to run: SACE_CAMPAIGN={cfg.name!r}, expected 'renewal'")
        return 1

    init_db()

    # A) chunks_renewal — delegated to the existing, already-tested loader.
    # `rules` below is loaded a SECOND time only because load_kb.main() does
    # its own campaign.load_rules() call internally; both calls parse the
    # same already-generated sace_chat.kb_renewal.RULES list (a Python
    # import, not a re-parse of the CSV), so this is not a second source of
    # truth — see the module docstring.
    import load_kb

    load_kb.main()

    # B) answer_cache_renewal — seeded from the SAME rule list load_kb.main()
    # just used, so a rule's chunks_renewal row and its cache rows can never
    # describe two different answers.
    rules = campaign.load_rules(cfg)
    by_tier = {}
    for r in rules:
        by_tier.setdefault(r.tier, []).append(r)

    blocked_present = [r.id for t in _BLOCKED_TIERS for r in by_tier.get(t, [])]

    cacheable = [r for t in _CACHEABLE_TIERS for r in by_tier.get(t, [])]
    seed_pairs: list[tuple[str, object]] = []  # (variant_text, rule)
    for rule in cacheable:
        for variant in _seed_variants(rule):
            seed_pairs.append((variant, rule))

    embedder = get_embedder()
    variant_texts = [text for text, _ in seed_pairs]
    vectors = embed_many(embedder, variant_texts)
    assert len(vectors) == len(seed_pairs), (
        f"embed_many returned {len(vectors)} vectors for {len(seed_pairs)} inputs"
    )

    removed = answer_cache.clear(table="answer_cache_renewal", source="seed")

    stored = 0
    for (variant, rule), vec in zip(seed_pairs, vectors):
        # T2/T4 must never reach store() even as a defensive second check —
        # HARD BLOCK asserted above, and re-checked per-row so a future bug
        # in the tier filter above cannot silently smuggle one through.
        assert rule.tier not in _BLOCKED_TIERS, (
            f"refusing to seed cache from {rule.id} (tier {rule.tier}) — "
            f"T2/T4 rules must never be cached"
        )
        cache_id = answer_cache.store(
            question=variant,
            question_vec=vec,
            reply=rule.text,
            intent=rule.intent,
            governing_rule_id=rule.id,
            table="answer_cache_renewal",
            source="seed",
            tier=rule.tier,
        )
        if cache_id:
            stored += 1

    print(
        f"[renewal] chunks_renewal loaded via load_kb.main(); "
        f"answer_cache_renewal: removed {removed} old seed row(s), "
        f"seeded {stored}/{len(seed_pairs)} row(s) from {len(cacheable)} "
        f"T1/T3 rules (blocked tiers present in pool but excluded from "
        f"cache: {len(blocked_present)} rule(s) — {sorted(blocked_present)})."
    )
    if stored != len(seed_pairs):
        print(f"WARNING: {len(seed_pairs) - stored} row(s) refused by store() — see [cache] logs above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
