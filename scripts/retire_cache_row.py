"""Manually retire one or more answer_cache_renewal rows. Never deletes —
a deleted row loses the evidence for why it was wrong; a retired row still
answers "what did we used to serve here, and why did we stop".

This is the ONLY thing that acts on scripts/cache_report.py's findings, and
it only acts because a human ran it: nothing in this codebase retires a row
on its own.

Run:  python scripts/retire_cache_row.py <id> [<id> ...]
      python scripts/retire_cache_row.py --reactivate <id> [<id> ...]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="answer_cache_renewal row id(s)")
    ap.add_argument("--reactivate", action="store_true",
                     help="set active=true instead of false")
    args = ap.parse_args()

    active_value = args.reactivate
    with engine.begin() as conn:
        for row_id in args.ids:
            row = conn.execute(text(
                f"SELECT id, governing_rule_id, question, active FROM {TABLE} WHERE id = :i"
            ), {"i": row_id}).fetchone()
            if row is None:
                print(f"  {row_id}: not found — skipped")
                continue
            conn.execute(text(f"UPDATE {TABLE} SET active = :a WHERE id = :i"),
                         {"a": active_value, "i": row_id})
            verb = "reactivated" if active_value else "retired"
            print(f"  {row_id}: {verb} (rule={row.governing_rule_id}, question={row.question!r})")


if __name__ == "__main__":
    sys.exit(main())
