"""Load a campaign's hand-authored/generated rules into its own flat memory
pool. Which campaign: SACE_CAMPAIGN (see sace_chat.campaign), same as Engine,
voice_agent.py and the dashboard resolve it from.
"""

from sqlalchemy import text

from sace_chat.campaign import get_campaign
from sace_chat.db import ChunkRow, SessionLocal, init_db, insert_chunk
from sace_chat.embeddings import get_embedder
from sace_chat.kb import RULES as COVERAGE_RULES
from sace_chat.kb_renewal import RULES as RENEWAL_RULES

_RULES_BY_CAMPAIGN = {"coverage": COVERAGE_RULES, "renewal": RENEWAL_RULES}


def main():
    campaign = get_campaign()
    rules = _RULES_BY_CAMPAIGN[campaign.name]
    table = campaign.chunks_table

    init_db()
    embedder = get_embedder()
    session = SessionLocal()
    try:
        # Seed rules are replaced wholesale; learned rules are left alone.
        if table == "chunks":
            removed = (
                session.query(ChunkRow)
                .filter(ChunkRow.source != "learned")
                .delete(synchronize_session=False)
            )
        else:
            # Not an ORM-mapped table (see insert_chunk) — same filter, raw SQL.
            removed = session.execute(
                text(f"DELETE FROM {table} WHERE source != 'learned'")
            ).rowcount
        for rule in rules:
            insert_chunk(session, rule, embedder, table=table)
        session.commit()

        general = sum(1 for r in rules if r.intent is None)
        if table == "chunks":
            learned = session.query(ChunkRow).filter(ChunkRow.source == "learned").count()
        else:
            learned = session.execute(
                text(f"SELECT count(*) FROM {table} WHERE source = 'learned'")
            ).scalar()
        print(
            f"[{campaign.name}] loaded {len(rules)} seed rules into {table} "
            f"({general} general, {len(rules) - general} intent-routed), replacing "
            f"{removed}. {learned} learned rule(s) preserved."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
