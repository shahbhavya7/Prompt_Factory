import json
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
from sqlalchemy.dialects.postgresql import JSONB
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

    # A campaign's own answer tier (e.g. renewal's T1-T4), carried through
    # from its source KB for reporting and for cache/priority policy that
    # reads it (see answer_cache.is_cacheable). NULL for campaigns with no
    # tier concept — coverage.
    tier = Column(String, nullable=True)
    # Whether this rule's reply is expected to hand the caller to a human
    # (renewal's T3/T4). Read by the never-say guard's fallback: a violation
    # on a transfer rule re-speaks that rule's OWN verbatim text rather than
    # inventing a new line.
    transfer = Column(Boolean, nullable=False, default=False)

    # Prerequisite gating for a FLOW rule (intent IS NULL, source='seed'): the
    # subset of state.collected_fields that must already hold before this
    # step may govern a turn. {} (the default) means "no prerequisite" — every
    # existing coverage rule. Read by retrieve._fetch_general, which excludes
    # a row whose requires are not satisfied BEFORE ranking by distance — see
    # that function's docstring for why this has to happen in SQL, not after
    # the fetch. A value of "__any__" means "this key must be SET, any value"
    # rather than an exact match — e.g. {"has_camera_phone": "__any__"}.
    requires = Column(JSONB, nullable=False, default=dict)
    # Which collected_fields key(s) this rule's own turn is expected to
    # populate, once accepted — documentation/audit only; nothing enforces it
    # mechanically. The rule's own text is what actually instructs the model
    # to extract the field (mirroring kb.py's existing convention, e.g.
    # still_has_benefits_plan_check's "Add a county field to extracted_fields").
    sets = Column(JSONB, nullable=False, default=dict)
    # A flow rule's fixed position in its script, used ONLY as the fallback
    # tie-breaker when nothing else governs (see retrieve.py's "lowest-
    # numbered eligible step" fallback) — never as a ranking signal otherwise.
    # NULL for anything that isn't a flow rule.
    step_order = Column(Integer, nullable=True)
    case_fields = Column(JSONB, nullable=False, default=list)


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


class AnswerCacheRow(Base):
    """A confirmed question->reply pair, replayable without an LLM call.

    Written only after a turn was validated as `grounded`, so every row is a
    reply this system already accepted once. On a later turn, a caller message
    whose embedding is within CACHE_THRESHOLD of `question` serves `reply`
    verbatim and skips both prompt assembly and the LLM.

    Scoped by `intent` for the same reason the rule pool is (see
    retrieve._fetch_by_intent): a cached "caller is busy" answer must never be
    reachable from a "caller wants a callback" turn. NULL intent is the general
    section, exactly as in `chunks`.

    `governing_rule_id` is kept so a rule change can invalidate everything
    derived from it — without it, editing a rule would leave stale cached
    replies quoting the old wording, which is the classic cache-coherence bug
    and the one most likely to bite here.
    """

    __tablename__ = "answer_cache"

    id = Column(String, primary_key=True)
    # The caller's message, and the vector it is matched by.
    question = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    # The reply to replay, verbatim.
    reply = Column(Text, nullable=False)

    intent = Column(String, nullable=True, index=True)
    governing_rule_id = Column(String, nullable=True, index=True)
    grounding_cosine = Column(Float, nullable=True)

    # The question that was PENDING when this reply was produced, fingerprinted
    # (see answer_cache.question_fingerprint). Most diversion replies end by
    # returning to whatever Maya had already asked, so the reply is only correct
    # on a turn where the same question is still pending — this column is what
    # the serve path matches on to guarantee that. Empty string for a reply that
    # ends on no question, which is reusable anywhere.
    pending_fingerprint = Column(String, nullable=False, default="", index=True)

    # Provenance and usefulness, for pruning and for the dashboard.
    source_session_id = Column(String, nullable=True)
    hit_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    last_hit_at = Column(DateTime(timezone=True), nullable=True)

    # seed | live. "seed" is a row pre-loaded from a campaign's own answer
    # bank (e.g. the renewal KB's CSV of pre-written Q&As) rather than earned
    # from a live call; "live" (the default) is everything this module has
    # always stored. The distinction is what lets /cache/clear reload seeded
    # answers without wiping ones a real caller's turn actually confirmed —
    # mirrors load_kb.py's source != 'learned' split for the chunks pool.
    source = Column(String, nullable=False, default="live")
    # A campaign's own answer tier (e.g. the renewal KB's T1-T4), carried
    # through from a seed source for reporting; nullable because a live-
    # learned entry has no tier of its own.
    tier = Column(String, nullable=True)


class NeedsReviewRow(Base):
    """The human review queue: every candidate rule that did NOT go straight
    into the pool, for any reason, plus the context a person needs to judge it.

    Three reasons land here, and they are one queue on purpose — from the
    reviewer's point of view "should this rule exist?" is the same question
    regardless of which gate raised it:

      pending    — cleared grounding and duplicate, but a human must approve
                   before it is embedded and inserted. This is the default
                   path for a NEW rule now; nothing is learned silently.
      conflict   — contradicts an existing rule (`existing_chunk_id`).
      ungrounded — the source line was not actually in the transcript.

    The candidate's full shape is stored, not just its text, because approving
    it has to reconstruct a real Chunk: `cue` is what gets embedded (see
    models.Chunk.cue), so a queue that dropped it would force the human to
    re-invent the single field that decides whether the rule is ever retrieved.

    `trigger_message` / `trigger_reply` are the exchange that produced the
    candidate — the reviewer's evidence for whether the rule is warranted.
    """

    __tablename__ = "needs_review"

    id = Column(String, primary_key=True)
    candidate_text = Column(Text, nullable=False)
    existing_chunk_id = Column(String, nullable=True)
    reason = Column(String, nullable=False)  # pending | conflict | ungrounded
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # The rest of the candidate, so an approval can rebuild the Chunk verbatim.
    candidate_cue = Column(Text, nullable=False, default="")
    intent = Column(String, nullable=True)
    priority = Column(String, nullable=False, default="normal")
    learned_kind = Column(String, nullable=True)
    source_line = Column(Text, nullable=False, default="")

    # Provenance: which call, and which exchange in it, produced this.
    session_id = Column(String, nullable=True, index=True)
    trigger_message = Column(Text, nullable=False, default="")
    trigger_reply = Column(Text, nullable=False, default="")

    # Set when a human acts on the row. Rows are deleted on approve/discard,
    # so these exist for the brief window before deletion and for any future
    # audit table that wants to copy them.
    status = Column(String, nullable=False, default="pending")  # pending | approved | discarded
    reviewed_at = Column(DateTime(timezone=True), nullable=True)


# A stuck or slow query would otherwise hang a live turn indefinitely — nothing
# upstream (voice_agent.py, engine.py) bounds how long a pgvector query can take.
# statement_timeout is enforced by Postgres itself, so it applies no matter which
# code path issues the query.
_DB_STATEMENT_TIMEOUT_MS = int(os.environ.get("SACE_DB_STATEMENT_TIMEOUT_MS", "5000"))
engine = create_engine(
    DATABASE_URL, future=True,
    connect_args={"options": f"-c statement_timeout={_DB_STATEMENT_TIMEOUT_MS}"},
)
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
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tier VARCHAR",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS transfer BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS requires JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sets JSONB NOT NULL DEFAULT '{}'::jsonb",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS step_order INTEGER",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS case_fields JSONB NOT NULL DEFAULT '[]'::jsonb",
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

        # retrieve.py's two lookups (_fetch_by_intent, _fetch_general) both
        # filter on `intent` before doing anything else — without an index,
        # that filter is a full sequential scan of the whole pool on every
        # single turn, seed and learned rules alike. This is the "table of
        # contents" fix: Postgres jumps straight to the matching rows instead
        # of reading every row to check whether it qualifies. Same effect for
        # `intent IS NULL` (the general pool) as for a specific label.
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_intent ON chunks (intent)"))

        # A second campaign's rule pool. LIKE ... INCLUDING ALL clones every
        # column (including the ones just migrated above), default, index and
        # constraint from `chunks` at CREATE time — one physical table per
        # campaign is what keeps two campaigns' general pools (intent IS NULL)
        # from ever competing for the same turn; see campaign.py's module
        # docstring. Placed AFTER the chunks migrations above so a fresh
        # chunks_renewal always matches chunks' current shape.
        # INCLUDING ALL brings the intent index (and the primary key, defaults,
        # etc.) along with it under a freshly generated name — no separate
        # CREATE INDEX needed here.
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS chunks_renewal (LIKE chunks INCLUDING ALL)"
        ))
        # LIKE ... INCLUDING ALL only runs at CREATE time — an
        # already-existing chunks_renewal (every deploy after the first)
        # does not retroactively pick up a column added to chunks later, so
        # every chunks migration above needs its own explicit mirror here.
        conn.execute(text(
            "ALTER TABLE chunks_renewal ADD COLUMN IF NOT EXISTS "
            "case_fields JSONB NOT NULL DEFAULT '[]'::jsonb"
        ))

        # needs_review grew from "a log of rejections" into the human approval
        # queue, so the columns an approval needs to rebuild a Chunk (cue,
        # intent, priority, learned_kind) and the provenance a reviewer needs
        # to judge it (session, trigger exchange) are added here. Same
        # idempotent ADD COLUMN IF NOT EXISTS pattern as chunks above — an
        # existing database keeps its old conflict/ungrounded rows, which
        # simply show up with empty cue/trigger fields.
        for ddl in (
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS candidate_cue TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS intent VARCHAR",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS priority VARCHAR NOT NULL DEFAULT 'normal'",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS learned_kind VARCHAR",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS source_line TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS session_id VARCHAR",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS trigger_message TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS trigger_reply TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS status VARCHAR NOT NULL DEFAULT 'pending'",
            "ALTER TABLE needs_review ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ",
        ):
            conn.execute(text(ddl))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_needs_review_status ON needs_review (status)")
        )

        # The reply cache is read on the hot path of every turn, so its intent
        # filter must not be a sequential scan — same reasoning as
        # ix_chunks_intent above. No index on `embedding` itself, deliberately
        # and for the same reason as chunks: the per-section row count is small,
        # an exact scan over it is fast, and an approximate index combined with
        # a strict WHERE can under-return (see the chunks note).
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_answer_cache_intent ON answer_cache (intent)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_answer_cache_rule "
                 "ON answer_cache (governing_rule_id)")
        )
        # Added after the first rows existed. The default matters: '' means "ends
        # on no question", which is the permissive value, so a pre-existing row
        # would become servable on ANY pending question — the exact mistake the
        # column exists to prevent. Existing rows are dropped instead; the cache
        # is an optimisation and rebuilds itself within a few calls.
        if not _has_answer_cache_column(conn, "pending_fingerprint"):
            conn.execute(text(
                "ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS "
                "pending_fingerprint VARCHAR NOT NULL DEFAULT ''"
            ))
            conn.execute(text("DELETE FROM answer_cache"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_answer_cache_pending "
                 "ON answer_cache (pending_fingerprint)")
        )

        # Additive: existing rows all predate the source/tier split, and
        # 'live' is the correct backfill for every one of them — they were all
        # earned from a real call, since seeding didn't exist before this.
        conn.execute(text(
            "ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS "
            "source VARCHAR NOT NULL DEFAULT 'live'"
        ))
        conn.execute(text(
            "ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS tier VARCHAR"
        ))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_answer_cache_source "
                 "ON answer_cache (source)")
        )

        # A second campaign's reply cache — the chunks_renewal pattern
        # repeated for the exact same reason, and not optional the way the
        # module docstring in campaign.py used to claim: coverage's kb.py and
        # renewal's kb_renewal.py both reuse the five safety-label intents
        # verbatim (dnc, abuse, medical_emergency, garbled_audio,
        # frustration), so a shared answer_cache lets a reply CACHED from a
        # coverage call be SERVED to a renewal caller under the same intent
        # label — wrong clinic name, wrong script, on the highest-stakes
        # intents in the system. Placed AFTER the answer_cache migrations
        # above so a fresh answer_cache_renewal always matches its current
        # shape. INCLUDING ALL brings the intent/rule/pending indexes along
        # with it under freshly generated names.
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS answer_cache_renewal (LIKE answer_cache INCLUDING ALL)"
        ))


def _has_answer_cache_column(conn, column: str) -> bool:
    return bool(
        conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'answer_cache' AND column_name = :c"
            ),
            {"c": column},
        ).scalar()
    )


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


def insert_chunk(session, chunk, embedder, learned_kind=None, source=None, table="chunks"):
    """The single insert path, shared by load_kb.py and consolidator.py, so the
    two can never disagree about which columns get set.

    The embedding comes from the CUE, not the rule text — see models.Chunk.cue.

    `table="chunks"` (the default) goes through the ChunkRow ORM object,
    unchanged from before campaigns existed. Any other table — a second
    campaign's own pool, e.g. "chunks_renewal" — goes through a raw-SQL
    upsert instead, because ChunkRow is bound to the single physical table
    name "chunks" and cannot target another one.
    """
    cue = (chunk.cue or "").strip() or chunk.text
    vec = check_embedding(embedder.embed(cue), chunk_id=chunk.id)
    resolved_source = source or ("learned" if learned_kind else chunk.source)
    resolved_learned_kind = learned_kind or chunk.learned_kind

    if table == "chunks":
        row = ChunkRow(
            id=chunk.id,
            title=chunk.title,
            text=chunk.text,
            cue=cue,
            intent=chunk.intent,
            priority=chunk.priority,
            terminal=bool(chunk.terminal),
            exclusive=bool(chunk.exclusive),
            source=resolved_source,
            learned_kind=resolved_learned_kind,
            embedding=vec,
            tier=chunk.tier,
            transfer=bool(chunk.transfer),
            requires=dict(chunk.requires),
            sets=dict(chunk.sets),
            step_order=chunk.step_order,
            case_fields=list(chunk.case_fields),
        )
        session.merge(row)
        return row

    session.execute(
        text(
            f"INSERT INTO {table} "
            f"(id, title, text, cue, intent, priority, terminal, exclusive, "
            f" source, learned_kind, tier, transfer, requires, sets, step_order, "
            f" case_fields, embedding) "
            f"VALUES (:id, :title, :ctext, :cue, :intent, :priority, :terminal, :exclusive, "
            f" :source, :learned_kind, :tier, :transfer, CAST(:requires AS jsonb), "
            f" CAST(:sets AS jsonb), :step_order, CAST(:case_fields AS jsonb), "
            f" CAST(:vec AS vector)) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"  title=EXCLUDED.title, text=EXCLUDED.text, cue=EXCLUDED.cue, "
            f"  intent=EXCLUDED.intent, priority=EXCLUDED.priority, "
            f"  terminal=EXCLUDED.terminal, exclusive=EXCLUDED.exclusive, "
            f"  source=EXCLUDED.source, learned_kind=EXCLUDED.learned_kind, "
            f"  tier=EXCLUDED.tier, transfer=EXCLUDED.transfer, "
            f"  requires=EXCLUDED.requires, sets=EXCLUDED.sets, "
            f"  step_order=EXCLUDED.step_order, case_fields=EXCLUDED.case_fields, "
            f"  embedding=EXCLUDED.embedding"
        ),
        {
            "id": chunk.id, "title": chunk.title, "ctext": chunk.text, "cue": cue,
            "intent": chunk.intent, "priority": chunk.priority,
            "terminal": bool(chunk.terminal), "exclusive": bool(chunk.exclusive),
            "source": resolved_source, "learned_kind": resolved_learned_kind,
            "tier": chunk.tier, "transfer": bool(chunk.transfer),
            "requires": json.dumps(dict(chunk.requires)),
            "sets": json.dumps(dict(chunk.sets)),
            "step_order": chunk.step_order, "vec": str(list(vec)),
            "case_fields": json.dumps(list(chunk.case_fields)),
        },
    )
    return chunk
