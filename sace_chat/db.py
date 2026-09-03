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

# Opt-in startup guard: entry points that must never touch anything but a
# specific database (the isolated renewal-KB stack, docker-compose.kb.yml) set
# SACE_EXPECTED_DB — see .env.kb — and get refused loudly here instead of
# silently writing into whatever DATABASE_URL happened to be left in the shell.
# Unset (the default, everywhere else in the app) makes this a no-op, so the
# shared sace_chat database is never affected.
_EXPECTED_DB = os.environ.get("SACE_EXPECTED_DB")


def _db_name_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


if _EXPECTED_DB:
    _actual_db = _db_name_from_url(DATABASE_URL)
    if _actual_db != _EXPECTED_DB:
        raise RuntimeError(
            f"refusing to start: DATABASE_URL points at database {_actual_db!r}, "
            f"but SACE_EXPECTED_DB requires {_EXPECTED_DB!r}. This usually means "
            f"a stale DATABASE_URL is set in your shell, overriding .env.kb — "
            f"unset it or fix .env.kb before rerunning."
        )

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

    # A campaign's own answer tier (e.g. the renewal KB's T1-T4). Nullable and
    # unused by the coverage campaign — see models.Chunk.tier.
    tier = Column(String, nullable=True)

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
    # Which CampaignConfig this turn ran under (see sace_chat.campaign).
    # Nullable — a turn recorded before this column existed has no campaign
    # of its own; the dashboard treats that the same as "coverage".
    campaign = Column(String, nullable=True)

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

    # seed (loaded straight from a deterministic KB rule, no call ever produced
    # it) | live (written by store(), from an actual validated turn). Existing
    # rows all predate this column and are live by construction, hence the
    # default — no backfill needed for them to keep behaving identically.
    source = Column(String, nullable=False, default="live")
    # A campaign's own answer tier (e.g. the renewal KB's T1-T4), carried over
    # from the governing rule at store time. Nullable — a seed/live entry has
    # no tier of its own; see models.Chunk.tier.
    tier = Column(String, nullable=True)

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

    # Manual retirement, never deletion — a deleted row loses the evidence for
    # why it was wrong. lookup() filters on this unconditionally (default TRUE,
    # so existing rows keep serving exactly as before); a row is retired by
    # hand (scripts/retire_cache_row.py) after scripts/cache_report.py flags it.
    # Explicitly NOT an LRU/LFU/size eviction — see cache_report.py's docstring.
    active = Column(Boolean, nullable=False, default=True)


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


def verify_connected_database() -> str:
    """Assert the SERVER we actually reached is this stack's own database.

    The URL check at import time is necessary but not sufficient, and the gap
    is not theoretical — it was hit while re-ingesting the KB.
    docker-compose.kb.yml publishes the isolated stack on 5433; on that machine
    5433 was already held by an unrelated project's Postgres. DATABASE_URL
    still ended in "/sace_kb", so _EXPECTED_DB passed cleanly and every entry
    point was willing to start, pointed at a stranger's server. It failed on
    password authentication — luck, not a guard. A server that happened to have
    a `sace` role and a `sace_kb` database would have accepted the whole load.

    Note that `current_database()` cannot detect this on its own: Postgres
    connects to the database named in the URL, so on any successful connection
    it returns exactly the name the URL asked for, whichever host answered.
    Comparing it to SACE_EXPECTED_DB re-checks the URL, not the server.

    What distinguishes OUR database is a marker this stack writes itself. So:

      * marker present and matching   -> the stack we expect. Proceed.
      * marker present and different  -> another stack's database. Refuse.
      * marker absent, no tables      -> a fresh database, mid-initialisation.
                                         Proceed; init_db stamps it below.
      * marker absent, our schema     -> a database created before this marker
                                         existed. Adopt it and stamp it, rather
                                         than locking everyone out of a
                                         database that IS theirs.
      * marker absent, foreign schema -> a populated database that is not ours.
                                         Refuse. This is the port-collision
                                         case, and the one worth catching.

    "Our schema" is the presence of `chunks` — the rule pool every stack in
    this repo has and nothing else plausibly does. It is a weaker signal than
    the marker, which is why it only ever ADOPTS an unmarked database and never
    overrides a marker that disagrees.

    A no-op when SACE_EXPECTED_DB is unset, exactly like the URL check — the
    shared sace_chat stack is unaffected.
    """
    if not _EXPECTED_DB:
        with engine.connect() as conn:
            return conn.execute(text("SELECT current_database()")).scalar()

    with engine.connect() as conn:
        actual = conn.execute(text("SELECT current_database()")).scalar()
        marker = None
        if conn.execute(text("SELECT to_regclass('public.sace_stack')")).scalar():
            marker = conn.execute(text("SELECT name FROM sace_stack LIMIT 1")).scalar()
        n_tables = conn.execute(text(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )).scalar()
        has_our_schema = bool(
            conn.execute(text("SELECT to_regclass('public.chunks')")).scalar()
        )

    where = DATABASE_URL.rsplit("@", 1)[-1]
    if marker is not None and marker != _EXPECTED_DB:
        raise RuntimeError(
            f"refusing to continue: the database at {where} is stamped as stack "
            f"{marker!r}, but SACE_EXPECTED_DB requires {_EXPECTED_DB!r}."
        )
    if marker is None and n_tables and has_our_schema:
        # Ours, from before the marker existed. Adopt it.
        with engine.begin() as conn:
            _stamp_stack(conn)
        return actual
    if marker is None and n_tables:
        raise RuntimeError(
            f"refusing to continue: the database at {where} already holds "
            f"{n_tables} table(s) but carries no sace_stack marker, so it is not "
            f"this stack's database. The URL names {actual!r} and resolves to a "
            f"different server — most often another container already holding "
            f"that port. Check `docker ps` and SACE_KB_PORT in .env.kb."
        )
    return actual


def _stamp_stack(conn) -> None:
    """Write this stack's identity marker. Idempotent; see
    verify_connected_database for what reads it."""
    if not _EXPECTED_DB:
        return
    conn.execute(text(
        "CREATE TABLE IF NOT EXISTS sace_stack ("
        "  name VARCHAR PRIMARY KEY,"
        "  stamped_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    ))
    conn.execute(
        text("INSERT INTO sace_stack (name) VALUES (:n) ON CONFLICT (name) DO NOTHING"),
        {"n": _EXPECTED_DB},
    )


# Columns the stage-machine design needed and this one does not. Kept rather
# than dropped so an existing database with learned rules in it survives the
# migration; nothing reads them.
_DEAD_COLUMNS = ("stage", "is_minor", "retry_mode", "field", "transitions", "type")


def init_db():
    # Before any DDL: confirm the server on the other end of DATABASE_URL is
    # the database we were told to write to. See verify_connected_database —
    # the URL-string check at import time cannot catch a port collision.
    verify_connected_database()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Stamped immediately after the extension and before any table is
        # created, so the "tables but no marker" branch above can never fire
        # against a database this function itself populated.
        _stamp_stack(conn)
    Base.metadata.create_all(engine)

    # create_all only creates missing tables, never missing columns on an
    # existing one, so the new routing columns are added explicitly and then
    # backfilled from the old ones. Idempotent — safe to run every boot.
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE turns ADD COLUMN IF NOT EXISTS campaign VARCHAR")
        )
        for ddl in (
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS cue TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS intent VARCHAR",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS terminal BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS exclusive BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS source VARCHAR NOT NULL DEFAULT 'seed'",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS learned_kind VARCHAR",
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tier VARCHAR",
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

        # A second campaign's own flat pool (e.g. the renewal KB), kept in a
        # separate table rather than mixed into `chunks` by a campaign column —
        # retrieval already scopes by intent per-turn, and a second campaign's
        # intents are not guaranteed disjoint from the coverage campaign's.
        # `LIKE chunks INCLUDING ALL` runs AFTER every ALTER TABLE chunks above,
        # so a fresh chunks_renewal always matches chunks' current shape
        # (columns, indexes, defaults) rather than some earlier version of it.
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS chunks_renewal (LIKE chunks INCLUDING ALL)")
        )

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
        # Additive: existing rows all predate the source/tier split, and the
        # column defaults (source='live') are exactly what those rows already
        # are by construction, so no backfill is needed for them to keep
        # behaving identically.
        conn.execute(
            text("ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS "
                 "source VARCHAR NOT NULL DEFAULT 'live'")
        )
        conn.execute(
            text("ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS tier VARCHAR")
        )
        # Manual retirement flag (see AnswerCacheRow.active) — default TRUE so
        # every existing row keeps serving exactly as before.
        conn.execute(
            text("ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS "
                 "active BOOLEAN NOT NULL DEFAULT TRUE")
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

        # A second campaign's reply cache — the chunks_renewal pattern above,
        # applied to answer_cache. Never sits in the same table as the coverage
        # campaign's cache: NEVER_CACHE_INTENTS/NEVER_CACHE_RULES are global
        # safety refusals (dnc, abuse, medical_emergency, garbled_audio, ...)
        # that must not depend on which campaign's rows happen to be present.
        # Run AFTER every ALTER TABLE answer_cache above so a fresh
        # answer_cache_renewal always matches its current shape.
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS answer_cache_renewal (LIKE answer_cache INCLUDING ALL)")
        )
        # Renewal-only usage instrumentation (scripts/cache_report.py reads
        # these; nothing evicts on them — see that script's docstring for why).
        # Run separately from the LIKE above: that only fires on first CREATE,
        # so a database where answer_cache_renewal already existed needs these
        # added explicitly too.
        # ── the per-variant cue index ────────────────────────────────────
        #
        # WHY THIS TABLE EXISTS, measured on the 165-rule renewal KB against
        # 165 held-out paraphrases:
        #
        #   rule chosen via the intent hop (chunks_renewal)   46.1%
        #   rule chosen directly from chunks_renewal          53.9%
        #   rule chosen from THIS table                       97.0%
        #
        # The cause of the gap is one line in the loader. `chunks_renewal`
        # stores ONE vector per rule, embedded from the cue variants JOINED
        # into a single string — so "Nothing came in the mail" and "I didn't
        # get anything" are averaged into a point that is not really either of
        # them. Embedding each phrasing on its own row and taking the nearest
        # recovers the signal that averaging destroyed. It is the same corpus
        # and the same embedder; only the granularity changed.
        #
        # Rows are (rule, one phrasing). Many rows point at one rule; the rule
        # itself still lives in chunks_renewal and is joined in at query time,
        # so there is exactly one copy of the spoken text and no way for the
        # two tables to disagree about what a rule says.
        #
        # Unlike answer_cache_renewal this holds EVERY tier including T2 and
        # T4. That is safe and necessary: this index decides which rule GOVERNS
        # a turn, not what gets replayed to a caller. A T4 self-harm question
        # must still find its own rule — it just must never be served from a
        # cache. The two concerns are separate tables precisely so one can be
        # complete while the other is restricted.
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS chunks_renewal_cues ("
            "  id VARCHAR PRIMARY KEY,"
            "  rule_id VARCHAR NOT NULL,"
            "  variant TEXT NOT NULL,"
            "  kind VARCHAR NOT NULL DEFAULT 'cue',"   # cue | title
            "  intent VARCHAR,"
            "  tier VARCHAR,"
            "  source VARCHAR NOT NULL DEFAULT 'seed',"
            f"  embedding vector({EMBEDDING_DIM}) NOT NULL)"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_renewal_cues_rule "
            "ON chunks_renewal_cues (rule_id)"
        ))

        for ddl in (
            "ALTER TABLE answer_cache_renewal ADD COLUMN IF NOT EXISTS "
            "correct_hits INTEGER NOT NULL DEFAULT 0",
            # The rule id the full pipeline grounded to on a MISS whose nearest
            # row was this one — a repeated (row's own rule == miss_grounded_to)
            # means real callers keep almost-but-not-quite matching a rule this
            # row is supposed to cover: an under-coverage signal, not a bad row.
            "ALTER TABLE answer_cache_renewal ADD COLUMN IF NOT EXISTS "
            "miss_grounded_to VARCHAR",
            # Present on a fresh answer_cache_renewal via the LIKE above, but a
            # database where the table already existed before `active` was
            # added to answer_cache needs it explicitly.
            "ALTER TABLE answer_cache_renewal ADD COLUMN IF NOT EXISTS "
            "active BOOLEAN NOT NULL DEFAULT TRUE",
        ):
            conn.execute(text(ddl))


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


def insert_chunk(session, chunk, embedder, learned_kind=None, source=None,
                  table="chunks", embedding=None):
    """The single insert path, shared by load_kb.py and consolidator.py, so the
    two can never disagree about which columns get set.

    The embedding comes from the CUE, not the rule text — see models.Chunk.cue.
    Pass a precomputed `embedding` to skip the per-row embed call entirely —
    a bulk loader embedding many chunks at once should batch them itself
    (see embeddings.embed_many) rather than pay one round-trip per row here.
    """
    cue = (chunk.cue or "").strip() or chunk.text
    vec = check_embedding(
        embedding if embedding is not None else embedder.embed(cue),
        chunk_id=chunk.id,
    )
    fields = dict(
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
        tier=chunk.tier,
    )
    if table == "chunks":
        row = ChunkRow(embedding=vec, **fields)
        session.merge(row)
        return row

    # A second campaign's own pool (e.g. chunks_renewal) — same shape (LIKE
    # chunks INCLUDING ALL) but not an ORM-mapped class, so this goes through
    # raw SQL. ON CONFLICT (id) mirrors session.merge()'s upsert-by-primary-key
    # semantics above.
    session.execute(
        text(
            f"INSERT INTO {table} "
            f"(id, title, text, cue, intent, priority, terminal, exclusive, "
            f" source, learned_kind, tier, embedding) "
            f"VALUES "
            f"(:id, :title, :text, :cue, :intent, :priority, :terminal, :exclusive, "
            f" :source, :learned_kind, :tier, CAST(:embedding AS vector)) "
            f"ON CONFLICT (id) DO UPDATE SET "
            f"  title=EXCLUDED.title, text=EXCLUDED.text, cue=EXCLUDED.cue, "
            f"  intent=EXCLUDED.intent, priority=EXCLUDED.priority, "
            f"  terminal=EXCLUDED.terminal, exclusive=EXCLUDED.exclusive, "
            f"  source=EXCLUDED.source, learned_kind=EXCLUDED.learned_kind, "
            f"  tier=EXCLUDED.tier, embedding=EXCLUDED.embedding"
        ),
        {**fields, "embedding": str(list(vec))},
    )
    return fields
