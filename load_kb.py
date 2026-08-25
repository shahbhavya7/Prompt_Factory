"""Load a campaign's hand-authored rules into its flat memory pool.

Which campaign is resolved once, from `SACE_CAMPAIGN` (default `coverage`) —
see sace_chat/campaign.py. Loading is unchanged for the coverage campaign.
"""

from sqlalchemy import text as sql_text

from sace_chat import campaign
from sace_chat.db import ChunkRow, SessionLocal, init_db, insert_chunk
from sace_chat.embeddings import get_embedder


def main():
    cfg = campaign.get_campaign()
    rules = campaign.load_rules(cfg)
    table = cfg.chunks_table

    init_db()
    embedder = get_embedder()
    session = SessionLocal()
    try:
        # Seed rules are replaced wholesale; learned rules are left alone.
        # ChunkRow is bound to "chunks" specifically, so a campaign on its own
        # table (e.g. renewal's chunks_renewal) goes through raw SQL instead —
        # same split insert_chunk itself already makes.
        if table == "chunks":
            removed = (
                session.query(ChunkRow)
                .filter(ChunkRow.source != "learned")
                .delete(synchronize_session=False)
            )
        else:
            removed = session.execute(
                sql_text(f"DELETE FROM {table} WHERE source != 'learned'")
            ).rowcount
        for rule in rules:
            insert_chunk(session, rule, embedder, table=table)
        session.commit()

        general = sum(1 for r in rules if r.intent is None)
        if table == "chunks":
            learned = session.query(ChunkRow).filter(ChunkRow.source == "learned").count()
        else:
            learned = session.execute(
                sql_text(f"SELECT count(*) FROM {table} WHERE source = 'learned'")
            ).scalar()
        print(
            f"[{cfg.name}] Loaded {len(rules)} seed rules ({general} general, "
            f"{len(rules) - general} intent-routed) into {table!r}, replacing {removed}. "
            f"{learned} learned rule(s) preserved."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
