"""Load a campaign's hand-authored rules into its flat memory pool.

Which campaign is resolved once, from `SACE_CAMPAIGN` (default `coverage`) —
see sace_chat/campaign.py. Loading is unchanged for the coverage campaign.
"""

from sace_chat import campaign
from sace_chat.db import ChunkRow, SessionLocal, init_db, insert_chunk
from sace_chat.embeddings import get_embedder


def main():
    cfg = campaign.get_campaign()
    if cfg.chunks_table != "chunks":
        # insert_chunk writes through the ChunkRow ORM model, which is bound
        # to the single physical table "chunks" — a campaign asking for a
        # different table needs that model generalised first. Flagged loudly
        # rather than silently loading into the wrong table.
        raise NotImplementedError(
            f"campaign {cfg.name!r} declares chunks_table={cfg.chunks_table!r}, "
            f"but insert_chunk only writes to 'chunks' today — multi-table "
            f"chunk loading is not implemented yet"
        )
    rules = campaign.load_rules(cfg)

    init_db()
    embedder = get_embedder()
    session = SessionLocal()
    try:
        # Seed rules are replaced wholesale; learned rules are left alone.
        removed = (
            session.query(ChunkRow)
            .filter(ChunkRow.source != "learned")
            .delete(synchronize_session=False)
        )
        for rule in rules:
            insert_chunk(session, rule, embedder)
        session.commit()

        general = sum(1 for r in rules if r.intent is None)
        learned = session.query(ChunkRow).filter(ChunkRow.source == "learned").count()
        print(
            f"[{cfg.name}] Loaded {len(rules)} seed rules ({general} general, "
            f"{len(rules) - general} intent-routed), replacing {removed}. "
            f"{learned} learned rule(s) preserved."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
