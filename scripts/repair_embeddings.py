"""Audit and repair the embeddings in the memory pool.

A rule whose embedding is NULL, a zero vector, or the wrong dimension is not
merely degraded — it is invisible. Cosine against a zero vector is 0 for every
query, so the rule can never be retrieved, and in the consolidator's duplicate
and conflict gates it reads as "nothing similar exists", which silently lets
near-duplicates through.

  python scripts/repair_embeddings.py          audit only
  python scripts/repair_embeddings.py --fix    re-embed the bad rows
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from sace_chat.db import EMBEDDING_DIM, engine, init_db
from sace_chat.embeddings import get_embedder


def audit(conn):
    rows = conn.execute(text("SELECT id, text, embedding FROM chunks ORDER BY id")).fetchall()
    bad = []
    for row in rows:
        if row.embedding is None:
            bad.append((row.id, "null", 0, 0.0))
            continue
        vec = [float(x) for x in str(row.embedding).strip("[]").split(",")]
        norm = sum(v * v for v in vec) ** 0.5
        if len(vec) != EMBEDDING_DIM:
            bad.append((row.id, f"dim {len(vec)} != {EMBEDDING_DIM}", len(vec), norm))
        elif norm < 1e-9:
            bad.append((row.id, "zero norm", len(vec), norm))
    return rows, bad


def main():
    fix = "--fix" in sys.argv
    init_db()

    with engine.connect() as conn:
        rows, bad = audit(conn)

    print(f"{len(rows)} rules in the pool, EMBEDDING_DIM={EMBEDDING_DIM}")
    if not bad:
        print("all embeddings valid — nothing to repair")
        return 0

    for rule_id, reason, dim, norm in bad:
        print(f"  BAD  {rule_id:26s} {reason:22s} dim={dim} norm={norm:.6f}")

    if not fix:
        print(f"\n{len(bad)} bad row(s). Re-run with --fix to re-embed them.")
        return 1

    embedder = get_embedder()
    by_id = {r.id: r.text for r in rows}
    with engine.begin() as conn:
        for rule_id, *_ in bad:
            vec = embedder.embed(by_id[rule_id])
            norm = sum(v * v for v in vec) ** 0.5
            if len(vec) != EMBEDDING_DIM or norm < 1e-9:
                print(f"  SKIP {rule_id}: the embedder itself returned dim={len(vec)} norm={norm:.6f}")
                continue
            conn.execute(
                text("UPDATE chunks SET embedding = :v WHERE id = :i"),
                {"v": str(list(vec)), "i": rule_id},
            )
            print(f"  FIXED {rule_id} (norm={norm:.6f})")

    with engine.connect() as conn:
        _, still_bad = audit(conn)
    print(f"\n{len(bad) - len(still_bad)} repaired, {len(still_bad)} still bad")
    return 0 if not still_bad else 1


if __name__ == "__main__":
    sys.exit(main())
