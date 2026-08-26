"""Read-only usage report for the renewal answer cache. Prints. Never writes.

Instrument first, decide later: this script has no automatic action — no
eviction, no retirement, nothing. A human reads the three sections below and
acts by hand (scripts/retire_cache_row.py sets active=false; never DELETE —
a deleted row loses the evidence for why it was wrong).

Explicitly NOT doing LRU, LFU, or any size-based eviction: this table is a
couple hundred rows and a few MB — not a capacity problem — and recency is
the wrong axis for it anyway. A rare question answered correctly twice in 500
calls is the cache working; an LRU policy would evict exactly that row.

Sections:
  1. zero-hit rows       — never served since load. Retirement candidates,
                            oldest first — but check WHY before retiring: a
                            genuinely rare question is not automatically a bad
                            row (see the module docstring above).
  2. precision problems   — served often enough to trust the number, but
                            correct_hits / hit_count < 0.9. Retire these
                            FIRST: an unreliable answer being served a lot is
                            worse than a correct one nobody's asked yet.
  3. under-covered rules  — misses that, once the full pipeline ran, still
                            grounded back to the SAME rule's own row. The
                            seeded phrasings don't match how people actually
                            ask. These are ADD candidates, not evictions —
                            printed with real caller wording from `turns`
                            where available.

Run:  python scripts/cache_report.py [--min-hits-for-precision N] [--zero-hit-limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.kb", override=True)
load_dotenv(ROOT / ".env")

from sqlalchemy import text  # noqa: E402

from sace_chat.db import engine  # noqa: E402

TABLE = "answer_cache_renewal"


def section_zero_hit(conn, limit):
    print(f"\n[1] zero-hit rows (never served since load) — retirement candidates, oldest first")
    rows = conn.execute(text(
        f"SELECT id, governing_rule_id, tier, question, created_at "
        f"FROM {TABLE} WHERE active = TRUE AND hit_count = 0 "
        f"ORDER BY created_at ASC LIMIT :n"
    ), {"n": limit}).fetchall()
    if not rows:
        print("  (none)")
        return
    for r in rows:
        print(f"  {r.id}  rule={r.governing_rule_id} tier={r.tier}  {r.created_at}  {r.question!r}")
    total = conn.execute(text(
        f"SELECT count(*) FROM {TABLE} WHERE active = TRUE AND hit_count = 0")).scalar()
    print(f"  ({total} total zero-hit row(s); showing up to {limit})")


def section_precision(conn, min_hits):
    print(f"\n[2] precision problems (hit_count >= {min_hits}, correct_hits/hit_count < 0.9) "
          f"— retire these first")
    rows = conn.execute(text(
        f"SELECT id, governing_rule_id, tier, question, hit_count, correct_hits "
        f"FROM {TABLE} WHERE active = TRUE AND hit_count >= :m "
        f"  AND (correct_hits::float / hit_count) < 0.9 "
        f"ORDER BY (correct_hits::float / hit_count) ASC"
    ), {"m": min_hits}).fetchall()
    if not rows:
        print("  (none)")
        return
    for r in rows:
        ratio = r.correct_hits / r.hit_count
        print(f"  {r.id}  rule={r.governing_rule_id} tier={r.tier}  "
              f"correct={r.correct_hits}/{r.hit_count} ({ratio:.0%})  {r.question!r}")


def section_undercovered(conn):
    print(f"\n[3] under-covered rules (misses that still grounded back to their own row's rule) "
          f"— ADD candidates, not evictions")
    rows = conn.execute(text(
        f"SELECT governing_rule_id, count(*) AS n "
        f"FROM {TABLE} "
        f"WHERE miss_grounded_to IS NOT NULL AND miss_grounded_to = governing_rule_id "
        f"GROUP BY governing_rule_id ORDER BY n DESC"
    )).fetchall()
    if not rows:
        print("  (none)")
        return
    for r in rows:
        print(f"  {r.governing_rule_id}: {r.n} near-miss(es) that still resolved to it")
        examples = conn.execute(text(
            "SELECT user_text FROM turns WHERE governing_rule_id = :rid "
            "ORDER BY created_at DESC LIMIT 5"
        ), {"rid": r.governing_rule_id}).fetchall()
        if examples:
            print("    observed caller wording:")
            for e in examples:
                print(f"      - {e.user_text!r}")
        else:
            print("    (no turns logged for this rule yet — nothing to quote)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hits-for-precision", type=int, default=5,
                     help="don't judge precision on fewer than this many hits (default 5)")
    ap.add_argument("--zero-hit-limit", type=int, default=50,
                     help="cap the zero-hit listing (default 50)")
    args = ap.parse_args()

    with engine.connect() as conn:
        active = conn.execute(text(f"SELECT count(*) FROM {TABLE} WHERE active = TRUE")).scalar()
        retired = conn.execute(text(f"SELECT count(*) FROM {TABLE} WHERE active = FALSE")).scalar()
        print(f"{TABLE}: {active} active row(s), {retired} retired")

        section_zero_hit(conn, args.zero_hit_limit)
        section_precision(conn, args.min_hits_for_precision)
        section_undercovered(conn)

    print("\nRead-only — nothing above was changed. Act by hand: "
          "python scripts/retire_cache_row.py <id> [<id> ...]")


if __name__ == "__main__":
    sys.exit(main())
