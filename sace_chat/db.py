import os
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
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


class TurnRow(Base):
    """One row per turn, written by the voice agent (and available to any other
    front end). This is the audit trail behind the dashboard: `prompt_sent` is
    the EXACT string handed to the LLM for that turn, captured at call time by
    Engine.build_turn_context, so the monitor shows the prompt the speaking agent
    actually used rather than a reconstruction.
    """

    __tablename__ = "turns"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    turn_index = Column(Integer, nullable=False)
    source = Column(String, nullable=False)  # voice | chat

    user_text = Column(Text, nullable=False)
    reply_text = Column(Text, nullable=False)
    prompt_sent = Column(Text, nullable=False)

    governing_rule_id = Column(String, nullable=True)
    # Comma-separated ids; a list is at most one entry today, and keeping it text
    # avoids a JSON round-trip for something only ever displayed.
    reference_rule_ids = Column(Text, nullable=True)

    intent = Column(String, nullable=True)
    intent_cosine = Column(Float, nullable=True)
    grounding_cosine = Column(Float, nullable=True)
    validation_outcome = Column(String, nullable=True)
    assembled_tokens = Column(Integer, nullable=True)

    # Total, then the breakdown. stt_ms is the Deepgram finalisation delay,
    # context_ms is SACE retrieval + assembly, llm_ttft_ms is time to first
    # token, tts_ttfb_ms is time to first audio frame.
    latency_ms = Column(Float, nullable=True)
    stt_ms = Column(Float, nullable=True)
    context_ms = Column(Float, nullable=True)
    llm_ttft_ms = Column(Float, nullable=True)
    tts_ttfb_ms = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class CallTranscriptRow(Base):
    """The full transcript of a finished call, plus what the learning loop made
    of it. Written on session end.

    Note: the Streamlit chat app never persisted transcripts — it runs the
    learning loop straight off session state — so this table is new rather than
    shared. The chat path can be pointed at it later without changing anything
    here.
    """

    __tablename__ = "call_transcripts"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)  # voice | chat
    transcript = Column(Text, nullable=False)
    turn_count = Column(Integer, nullable=False, default=0)
    # [{"outcome": ..., "detail": ..., "text": ..., "intent": ...}, ...]
    learning_results = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


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


def record_turn(**fields) -> str:
    """Persist one turn to `turns`. Returns the row id.

    Deliberately tolerant: a logging failure must never take down a live call, so
    this swallows its own errors after printing them. Everything it needs is
    passed in — it reads no global state.
    """
    row_id = fields.pop("id", None) or str(uuid.uuid4())
    refs = fields.pop("reference_rule_ids", None)
    if isinstance(refs, (list, tuple)):
        refs = ",".join(refs) or None
    try:
        with SessionLocal() as session:
            session.add(TurnRow(id=row_id, reference_rule_ids=refs, **fields))
            session.commit()
    except Exception as exc:  # pragma: no cover - logging path
        print(f"[db] record_turn failed: {type(exc).__name__}: {exc}")
    return row_id


def record_call_transcript(session_id, source, transcript, turn_count, learning_results) -> str:
    """Persist a finished call and the learning loop's verdicts."""
    row_id = str(uuid.uuid4())
    try:
        with SessionLocal() as session:
            session.add(CallTranscriptRow(
                id=row_id, session_id=session_id, source=source, transcript=transcript,
                turn_count=turn_count, learning_results=learning_results or [],
            ))
            session.commit()
    except Exception as exc:  # pragma: no cover - logging path
        print(f"[db] record_call_transcript failed: {type(exc).__name__}: {exc}")
    return row_id


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
