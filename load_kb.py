"""Load the hand-authored rules from kb.py into the flat memory pool."""

from sace_chat.db import ChunkRow, SessionLocal, init_db, insert_chunk
from sace_chat.embeddings import get_embedder
from sace_chat.kb import RULES


def main():
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
        for rule in RULES:
            insert_chunk(session, rule, embedder)
        session.commit()

        general = sum(1 for r in RULES if r.intent is None)
        learned = session.query(ChunkRow).filter(ChunkRow.source == "learned").count()
        print(
            f"Loaded {len(RULES)} seed rules ({general} general, {len(RULES) - general} "
            f"intent-routed), replacing {removed}. {learned} learned rule(s) preserved."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
