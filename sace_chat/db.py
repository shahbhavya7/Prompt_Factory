import os

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://sace:sace@localhost:5433/sace_chat"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

Base = declarative_base()


class ChunkRow(Base):
    """One flat pool of rules. `intent` is the only routing key: a rule with an
    intent is reachable when the caller's message classifies as that intent; a
    rule with intent NULL is reachable by plain semantic similarity.

    The old stage / retry_mode / is_minor / field / special / type columns are
    left in place on existing databases (see init_db) but nothing queries them.
    """

    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    # The text `embedding` is computed from — see models.Chunk.cue.
    cue = Column(Text, nullable=False, default="")

    intent = Column(String, nullable=True)
    priority = Column(String, nullable=False, default="normal")
    terminal = Column(Boolean, nullable=False, default=False)
    exclusive = Column(Boolean, nullable=False, default=False)

    source = Column(String, nullable=False, default="seed")  # seed | learned
    # policy | example | failure — set by the consolidator, provenance only.
    learned_kind = Column(String, nullable=True)

    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)


class NeedsReviewRow(Base):
    __tablename__ = "needs_review"

    id = Column(String, primary_key=True)
    candidate_text = Column(Text, nullable=False)
    existing_chunk_id = Column(String, nullable=True)
    reason = Column(String, nullable=False)  # conflict | ungrounded
    created_at = Column(DateTime(timezone=True), server_default=func.now())


engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)


# Columns the stage-machine design needed and this one does not. Kept rather
# than dropped so an existing database with learned rules in it survives the
# migration; nothing reads them.
_DEAD_COLUMNS = ("stage", "is_minor", "retry_mode", "field", "transitions", "type")


def init_db():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

    # create_all only creates missing tables, never missing columns on an
    # existing one, so the new routing columns are added explicitly and then
    # backfilled from the old ones. Idempotent — safe to run every boot.
    with engine.begin() as conn:
        for ddl in (
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cue TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS intent VARCHAR",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS terminal BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS exclusive BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source VARCHAR NOT NULL DEFAULT 'seed'",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS learned_kind VARCHAR",
        ):
            conn.execute(text(ddl))

        # The dead columns may still be NOT NULL from the old schema, which
        # would reject every insert now that nothing populates them.
        for col in _DEAD_COLUMNS:
            conn.execute(
                text(
                    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns "
                    f"WHERE table_name='chunks' AND column_name='{col}') THEN "
                    f"ALTER TABLE chunks ALTER COLUMN {col} DROP NOT NULL; END IF; END $$"
                )
            )

        # Migrate the old `special` tag into `intent`, and derive `source`.
        if _has_column(conn, "special"):
            conn.execute(
                text("UPDATE chunks SET intent = special WHERE intent IS NULL AND special IS NOT NULL")
            )
        conn.execute(
            text("UPDATE chunks SET source = 'learned' WHERE learned_kind IS NOT NULL AND source <> 'learned'")
        )
        # Rules stored before `cue` existed were embedded from their text, so
        # text IS their cue — recording that keeps the column consistent with
        # what the vector actually represents.
        conn.execute(text("UPDATE chunks SET cue = text WHERE cue = ''"))


def _has_column(conn, column: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = :c"
            ),
            {"c": column},
        ).fetchone()
    )


class EmbeddingError(ValueError):
    """An embedding that would be silently useless if stored.

    A zero vector has cosine 0 against everything, so it can never be
    retrieved and scores 0 in the consolidator's duplicate and conflict gates —
    it would look like "nothing similar exists" for every candidate. A
    wrong-dimension vector fails later, inside a query, as a confusing
    `different vector dimensions` error from Postgres. Both are worth refusing
    at the insert.
    """


def check_embedding(vec, *, chunk_id: str = "?"):
    """Validate an embedding before it is stored. Returns the vector."""
    if vec is None:
        raise EmbeddingError(f"{chunk_id}: embedding is None")
    vec = list(vec)
    if not vec:
        raise EmbeddingError(f"{chunk_id}: embedding is empty")
    if len(vec) != EMBEDDING_DIM:
        raise EmbeddingError(
            f"{chunk_id}: embedding has {len(vec)} dimensions, table expects {EMBEDDING_DIM} "
            f"(EMBEDDING_MODE and EMBEDDING_DIM disagree — reload the KB after fixing)"
        )
    norm = sum(float(x) * float(x) for x in vec) ** 0.5
    if norm < 1e-9:
        raise EmbeddingError(f"{chunk_id}: embedding is a zero vector (norm={norm:g})")
    return vec


def insert_chunk(session, chunk, embedder, learned_kind=None, source=None):
    """The single insert path, shared by load_kb.py and consolidator.py, so the
    two can never disagree about which columns get set.

    The embedding comes from the CUE, not the rule text — see models.Chunk.cue.
    """
    cue = (chunk.cue or "").strip() or chunk.text
    vec = check_embedding(embedder.embed(cue), chunk_id=chunk.id)
    row = ChunkRow(
        id=chunk.id,
        title=chunk.title,
        text=chunk.text,
        cue=cue,
        intent=chunk.intent,
        priority=chunk.priority,
        terminal=bool(chunk.terminal),
        exclusive=bool(chunk.exclusive),
        source=source or ("learned" if learned_kind else chunk.source),
        learned_kind=learned_kind or chunk.learned_kind,
        embedding=vec,
    )
    session.merge(row)
    return row
