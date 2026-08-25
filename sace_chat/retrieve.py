"""Memory-only retrieval: one flat pool, one query, no stages.

Two decisions per turn, both semantic:

  1. Does the caller's message classify as one of the routable intents? Answered
     by cosine against short caller-phrased exemplars (kb.INTENT_EXEMPLARS), not
     by regex and not by an if-tree. Above INTENT_THRESHOLD the closest label
     wins; otherwise there is no intent.

  2. Which rule governs the reply? A message WITH an intent takes the nearest
     rule carrying that intent — one row, nothing else in scope. A message
     WITHOUT one falls through to the general pool (intent IS NULL) and takes
     the two nearest: the closest governs, the runner-up is background only.

The pool query embeds Maya's previous line together with the caller's message.
Without a stage machine that context is the only thing distinguishing "yeah I
guess so" said after "do you have a couple of minutes?" from the same words
said after "would you like me to repeat the number?" — the message alone is
genuinely ambiguous, and the rule texts are written to match the pair.
"""

import json
from dataclasses import dataclass, field as dc_field

from sqlalchemy import text

from sace_chat.embeddings import embed_many
from sace_chat.kb import INTENT_EXEMPLARS
from sace_chat.models import Chunk

# Sentinel for a `requires` value meaning "this collected_fields key must be
# set, to anything" rather than an exact match — e.g. a step that only cares
# whether the caller has answered a question at all, not what they said.
REQUIRES_ANY_SET = "__any__"

# Sentinel for a `requires` value meaning "this collected_fields key must NOT
# be set at all" — the mirror of REQUIRES_ANY_SET. This is what protects a
# flow rule against firing again after its own job is done: a rule that sets
# no field of its own (e.g. it only ASKS a question, and a LATER rule
# captures the answer) has nothing else to gate on to keep it from remaining
# a candidate forever, including on turns nowhere near it in the script —
# see retrieve.py's general-pool branch and the fallback it feeds, both of
# which would otherwise regress to the lowest-numbered rule with an empty
# `requires` every time similarity alone dips, defeating the whole point of
# "resume from roughly where the call actually is."
REQUIRES_NOT_SET = "__not_set__"

# Cosine above which the closest intent exemplar claims the turn.
INTENT_THRESHOLD = 0.45

# How much of Maya's previous line to fold into the pool query. Taken from the
# END of her turn, not the start: her turns open with acknowledgement and
# preamble and close with the actual question, so the tail is the part that says
# which question is pending.
_CONTEXT_CHARS = 90

# How the two are mixed. The caller's message and the pending question are
# embedded separately and combined at this weight, rather than concatenated —
# concatenation weights them by character count, and Maya's line is an order of
# magnitude longer than "alright, appreciate it", so it drowned the caller out
# and retrieval kept landing on the rule that produced the previous turn.
#
# Swept on 19 flow turns: 0.9 -> 18 correct; 1.0 (message only) -> 17;
# 0.8 and below -> 17 or fewer. A little context is decisive, a lot is harmful.
MESSAGE_WEIGHT = 0.9

# If the general pool's nearest chunk is below this, it is not actually a
# retrieved rule; it is just the least-bad neighbor in vector space. Without
# this gate, "tell me a joke" can land on a phone-number repeat rule.
GENERAL_MIN_SIMILARITY = 0.30

_SELECT_COLS = (
    "id, title, text, intent, priority, terminal, exclusive, source, learned_kind, "
    "tier, transfer, requires, sets, step_order"
)

# Priority ranks strictly ABOVE distance — but only inside an intent's own rule
# set, never across the general pool. The two candidate sets differ in kind:
#
#   Within one intent, every candidate handles the same situation, so priority is
#   the only thing that should decide between them. This matters because learned
#   rules default to normal priority and terminal=False: a learned paraphrase of
#   the DNC rule was observed outranking `special_dnc` on distance, which
#   cancelled the call-ending behaviour of a compliance-critical rule. A soft
#   distance discount does not fix that — the rival was 0.255 closer, far more
#   than any sane discount — so critical genuinely has to mean "always wins".
#
#   Across the general pool the candidates are unrelated to each other, so
#   priority carries no comparative meaning and applying it would be actively
#   harmful: `medical_emergency` is critical, and any thumb on its scale would
#   make it outrank the right rule on every ordinary turn. There, distance is the
#   only signal, and it is used alone.
#
# Postgres cannot reference a SELECT alias in an ORDER BY expression, so the
# scored rows go through a subquery.
#   Only `critical` hard-outranks — deliberately binary, not a full ordering.
#   Ranking every tier above distance also made learned rules unreachable for any
#   intent whose seed rule is merely `high` (callback_request, busy, elsewhere,
#   redirect, language — 5 of 14), since learned rules are inserted at `normal`.
#   That silently disabled the learning loop for most intents. `critical` is
#   reserved for the compliance closes that must never be overridden (dnc,
#   abuse); everywhere else the closer rule wins, so a learned rule can actually
#   take effect.
_PRIORITY_RANK = """
    CASE WHEN priority = 'critical' THEN 0 ELSE 1 END
"""


@dataclass
class CallState:
    """What survives between turns. No stage, no retry_mode, no field."""

    # Last classified intent — carried for the UI and the precedence rules only;
    # retrieval re-classifies from scratch every turn.
    intent: str = "none"
    opt_out: bool = False
    ended: bool = False
    # Questions Maya has already asked, fed back into the prompt so she cannot
    # re-ask one.
    asked_questions: list = dc_field(default_factory=list)
    collected_fields: dict = dc_field(default_factory=dict)
    # This call's own injected system-of-record data (e.g. its real due date),
    # for a campaign whose rules read specific facts instead of only speaking
    # policy prose. Never populated by the model — set once, from wherever the
    # call's case data actually comes from — and used two ways: as the
    # never-say guard's exemption set (guards.check_never_say) and, for a
    # T2-tier rule, as the source of the value the rule's own text says to
    # read. Empty for a campaign with no case-record concept — coverage.
    case_record: dict = dc_field(default_factory=dict)


@dataclass
class RetrievedRule:
    chunk: Chunk
    role: str  # governing | reference
    similarity: float  # 1 - pgvector cosine distance


@dataclass
class Retrieval:
    governing: RetrievedRule | None = None
    reference: list = dc_field(default_factory=list)
    intent: str | None = None
    intent_similarity: float = 0.0
    intent_ranked: list = dc_field(default_factory=list)
    query_text: str = ""
    opt_out: bool = False
    notes: list = dc_field(default_factory=list)

    # A cached answer for this turn, if one was close enough — see
    # answer_cache.lookup. When set, the caller may serve `cache_hit["reply"]`
    # and skip assembly and the LLM entirely. Retrieval still ran (the intent
    # and the vector were needed to find this at all), so `governing` is
    # populated too and the turn is still auditable.
    cache_hit: dict | None = None

    # The caller-message vector this turn was computed from, exposed so a
    # completed turn can be stored in the cache without embedding the same
    # string a second time. This is the whole reason a cache miss is cheap.
    message_vec: list | None = dc_field(default=None, repr=False)

    @property
    def rules(self) -> list:
        """Everything in scope, governing first — for grounding and the UI."""
        return ([self.governing] if self.governing else []) + list(self.reference)


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row.id,
        title=row.title,
        text=row.text,
        intent=row.intent,
        priority=row.priority,
        terminal=bool(row.terminal),
        exclusive=bool(row.exclusive),
        source=row.source or "seed",
        learned_kind=row.learned_kind,
        tier=row.tier,
        transfer=bool(row.transfer),
        # psycopg2 deserialises jsonb to a Python dict/list already; a legacy
        # row from before these columns existed reads back as None, so an
        # explicit fallback keeps every caller's dict/list assumption true.
        requires=row.requires if row.requires is not None else {},
        sets=row.sets if row.sets is not None else {},
        step_order=row.step_order,
    )


class IntentRouter:
    """Semantic intent classifier over kb.INTENT_EXEMPLARS.

    Exemplar vectors are embedded once, on first use, in a single batch — with a
    network embedder, doing it per turn would add dozens of sequential
    round-trips to every reply.
    """

    def __init__(self, embedder, threshold: float = INTENT_THRESHOLD, hotpath_embedder=None):
        # `embedder` is the KB embedder (pgvector-compatible). `hotpath` may be a
        # different, smaller, local model — exemplar matching never touches
        # Postgres, so its dimension only has to agree with itself. See
        # embeddings.get_hotpath_embedder for the full rule on which is used where.
        from sace_chat.embeddings import get_hotpath_embedder

        self.embedder = embedder
        self.hotpath = hotpath_embedder or get_hotpath_embedder(embedder)
        self.threshold = threshold
        self._vectors: list[tuple[str, list[float]]] | None = None

    @property
    def shares_kb_embedder(self) -> bool:
        return self.hotpath is self.embedder

    def warm(self):
        if self._vectors is not None:
            return
        labels, texts = [], []
        for intent, exemplars in INTENT_EXEMPLARS.items():
            for exemplar in exemplars:
                labels.append(intent)
                texts.append(exemplar)
        # Embedded once, at startup, in a single batch. Exemplars are static, so
        # doing this per turn would add a round-trip of dead air to every reply.
        self._vectors = list(zip(labels, embed_many(self.hotpath, texts)))

    def detect(self, message: str, kb_message_vec=None) -> tuple[str | None, float, list]:
        """Returns (intent or None, best similarity, ranked [(intent, sim)...]).

        `kb_message_vec` is the vector retrieval already computed with the KB
        embedder. When the hot path uses that same embedder it is reused, so the
        default configuration still costs exactly one embedding call per turn;
        only a genuinely different hot-path model embeds again, locally.
        """
        self.warm()
        if self.shares_kb_embedder and kb_message_vec is not None:
            message_vec = kb_message_vec
        else:
            message_vec = self.hotpath.embed(message)
        best_per_intent: dict[str, float] = {}
        for intent, vec in self._vectors:
            sim = _cosine(message_vec, vec)
            if sim > best_per_intent.get(intent, -1.0):
                best_per_intent[intent] = sim

        ranked = sorted(best_per_intent.items(), key=lambda kv: kv[1], reverse=True)
        if ranked and ranked[0][1] >= self.threshold:
            return ranked[0][0], ranked[0][1], ranked
        return None, (ranked[0][1] if ranked else 0.0), ranked


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def pending_question(history: list | None) -> str:
    """The tail of Maya's most recent turn — the question still on the table."""
    for line in reversed(history or []):
        if line.startswith("Maya:"):
            return line[len("Maya:"):].strip()[-_CONTEXT_CHARS:]
    return ""


def _blend(message_vec, context_vec, weight: float = MESSAGE_WEIGHT):
    mixed = [weight * m + (1 - weight) * c for m, c in zip(message_vec, context_vec)]
    norm = sum(x * x for x in mixed) ** 0.5
    return [x / norm for x in mixed] if norm else mixed


def _fetch_by_intent(conn, intent: str, qvec, table: str) -> Chunk | None:
    row = conn.execute(
        text(
            f"""
            SELECT * FROM (
                SELECT {_SELECT_COLS}, embedding <=> :qvec AS distance
                FROM {table}
                WHERE intent = :intent
            ) AS scored
            ORDER BY {_PRIORITY_RANK}, distance
            LIMIT 1
            """
        ),
        {"qvec": str(list(qvec)), "intent": intent},
    ).fetchone()
    if row is None:
        return None
    chunk = _row_to_chunk(row)
    chunk.tags["distance"] = float(row.distance)
    return chunk


# The prerequisite check shared by _fetch_general and the lowest-eligible-step
# fallback: a row's `requires` must be fully satisfied by the CALLER'S current
# collected_fields before it may be considered at all. Expressed as
# `NOT EXISTS (any requirement that fails)`, so a row with `requires = {}`
# (every existing coverage rule) always passes — there is nothing to fail.
#
# This runs INSIDE the SQL, not filtered out of a Python list after the
# fetch: filtering post-fetch would let a `LIMIT k` truncate the result set
# to ineligible rows before eligible ones are ever seen, silently leaving a
# turn with nothing in scope even though an eligible flow step exists further
# down the distance ordering.
#
# A `False` requirement is deliberately NOT an exact match against the state.
# The field it names (e.g. packet_received) is set by the very turn whose
# CALLER MESSAGE decides between two branches — "no, we moved" sets it False,
# "yes it arrived" sets it True — and retrieval for that decisive turn runs
# BEFORE that turn's own extraction lands in collected_fields. An exact-match
# "False" would then never be satisfied on the one turn it exists to gate:
# the field is simply absent yet, not yet false. So `False` means "not yet
# confirmed true" (absent OR false both pass; only an explicit True blocks) —
# a guard against relapsing into this step once the flow has moved past it,
# not the mechanism that picks the branch in the first place. Picking the
# branch is still cue-similarity's job, same as every other diversion in this
# codebase; `requires` only ever narrows the candidate set, it never ranks it.
_REQUIRES_SATISFIED_SQL = f"""
    NOT EXISTS (
        SELECT 1 FROM jsonb_each_text(requires) AS req(key, val)
        WHERE
            CASE WHEN req.val = '{REQUIRES_ANY_SET}'
                 THEN NOT (CAST(:state AS jsonb) ? req.key)
                 WHEN req.val = '{REQUIRES_NOT_SET}'
                 THEN (CAST(:state AS jsonb) ? req.key)
                 WHEN req.val = 'false'
                 THEN (CAST(:state AS jsonb) ->> req.key) = 'true'
                 ELSE (CAST(:state AS jsonb) ->> req.key) IS DISTINCT FROM req.val
            END
    )
"""


def _state_json(state) -> str:
    """collected_fields as a JSON blob, for the requires-gate SQL parameter.

    A plain dict of JSON-serialisable values by construction — it is filled
    only from validate_turn's extracted_fields, which is itself already
    constrained to a JSON-decoded model response.
    """
    return json.dumps(dict(getattr(state, "collected_fields", None) or {}))


def _fetch_general(conn, state, qvec, table: str, k: int = 2) -> list[Chunk]:
    rows = conn.execute(
        text(
            f"""
            SELECT {_SELECT_COLS}, embedding <=> :qvec AS distance
            FROM {table}
            WHERE intent IS NULL
              AND {_REQUIRES_SATISFIED_SQL}
            ORDER BY distance
            LIMIT :k
            """
        ),
        {"qvec": str(list(qvec)), "k": k, "state": _state_json(state)},
    ).fetchall()
    out = []
    for row in rows:
        chunk = _row_to_chunk(row)
        chunk.tags["distance"] = float(row.distance)
        out.append(chunk)
    return out


def _fetch_lowest_eligible_step(conn, state, table: str) -> Chunk | None:
    """The flow rule with the smallest `step_order` whose prerequisites are
    currently satisfied — the fallback when nothing governs by similarity.

    Never chosen by distance: this is a deterministic "resume the script from
    wherever it can currently continue" pick, used only when similarity-based
    retrieval found nothing usable (either every flow rule was filtered out,
    or the best match fell under GENERAL_MIN_SIMILARITY). A flow call must
    not close itself by exhaustion the way a diversion-only campaign safely
    can — see retrieve()'s general-pool branch.
    """
    row = conn.execute(
        text(
            f"""
            SELECT {_SELECT_COLS}
            FROM {table}
            WHERE intent IS NULL
              AND step_order IS NOT NULL
              AND {_REQUIRES_SATISFIED_SQL}
            ORDER BY step_order ASC
            LIMIT 1
            """
        ),
        {"state": _state_json(state)},
    ).fetchone()
    return _row_to_chunk(row) if row is not None else None


def retrieve(
    conn,
    state: CallState,
    message: str,
    embedder,
    history: list | None = None,
    router: IntentRouter | None = None,
    table: str = "chunks",
    cache_table: str = "answer_cache",
    precedence=None,
    use_cache: bool = True,
) -> Retrieval:
    """`precedence` is an optional (intent, message) -> (intent, opt_out) hook,
    applied to the detected label before the rule is fetched — policy lives in
    manager.resolve_precedence and is injected here rather than duplicated, so
    a flip like dnc -> abuse changes which rule governs without a second query.
    """
    router = router or IntentRouter(embedder)
    router.warm()

    pending = pending_question(history)
    query_text = f"{message}   [pending: {pending}]" if pending else message
    # The bare message classifies intent (Maya's line would only pollute that),
    # and is then blended with the pending question to search the pool.
    if pending:
        message_vec, context_vec = embed_many(embedder, [message, pending])
        pool_vec = _blend(message_vec, context_vec)
    else:
        message_vec = embedder.embed(message)
        pool_vec = message_vec

    intent, intent_sim, ranked = router.detect(message, kb_message_vec=message_vec)
    result = Retrieval(
        intent=intent,
        intent_similarity=round(intent_sim, 3),
        intent_ranked=[(i, round(s, 3)) for i, s in ranked[:4]],
        query_text=query_text,
        # Handed out so a completed turn can be cached without re-embedding.
        message_vec=list(message_vec),
    )

    if precedence is not None:
        effective, opt_out = precedence(intent or "none", message)
        result.opt_out = bool(opt_out)
        effective = None if effective == "none" else effective
        if effective != intent:
            result.notes.append(f"precedence: intent {intent!r} -> {effective!r}")
            intent = effective
            result.intent = effective

    # The cache probe goes HERE, and the position is deliberate on both sides:
    #
    #   * AFTER precedence, so a message that policy re-routed to dnc/abuse is
    #     seen by the never-cache list under its effective intent, not the
    #     router's softer first guess.
    #   * BEFORE the pool fetch, so a hit skips that query too.
    #
    # The lookup is additionally gated on the question currently pending (see
    # answer_cache's trailing-question gate): most diversion replies end by
    # returning to whatever was already asked, so an entry is only eligible on a
    # turn where that same question is still open.
    #
    # It reuses `message_vec`, already computed above — a miss therefore costs
    # one small extra query, not an embedding round-trip. See answer_cache.
    if use_cache and not result.opt_out:
        try:
            from sace_chat import answer_cache

            # The pending question comes from `state`, which retrieve already
            # has — so the gate costs nothing extra on the hot path.
            hit = answer_cache.lookup(
                conn, message_vec, intent,
                pending=answer_cache.pending_fingerprint(state),
                table=cache_table,
            )
            if hit is not None:
                result.cache_hit = hit
                result.notes.append(
                    f"cache hit {hit['id']} (similarity {hit['similarity']:.3f}, "
                    f"pending {hit['pending_fingerprint'] or 'none'}) — "
                    f"reusing a confirmed reply, skipping the LLM"
                )
        except Exception as exc:
            # An optimisation must never be able to fail a turn.
            result.notes.append(f"cache lookup failed, using the full path: {exc}")

    if intent is not None:
        chunk = _fetch_by_intent(conn, intent, pool_vec, table)
        if chunk is not None:
            result.governing = RetrievedRule(chunk, "governing", 1 - chunk.tags["distance"])
            return result
        # An intent with no rule behind it (e.g. a label whose seed rule was
        # deleted) must not silently swallow the turn.
        result.notes.append(f"intent {intent!r} matched no rule; fell through to the general pool")
        result.intent = None

    general = _fetch_general(conn, state, pool_vec, table, k=2)
    if general:
        best_similarity = 1 - general[0].tags["distance"]
        if best_similarity < GENERAL_MIN_SIMILARITY:
            fallback = _fetch_lowest_eligible_step(conn, state, table)
            if fallback is not None:
                result.governing = RetrievedRule(fallback, "governing", 1.0)
                result.notes.append(
                    f"general-pool best match {general[0].id} below relevance threshold "
                    f"({best_similarity:.3f} < {GENERAL_MIN_SIMILARITY:.3f}); fell back to "
                    f"lowest-numbered eligible step {fallback.id} rather than closing the call"
                )
                return result
            result.notes.append(
                f"general-pool best match {general[0].id} below relevance threshold "
                f"({best_similarity:.3f} < {GENERAL_MIN_SIMILARITY:.3f}); no rule retrieved"
            )
            return result

        result.governing = RetrievedRule(general[0], "governing", best_similarity)
        if not general[0].exclusive and len(general) > 1:
            result.reference = [
                RetrievedRule(general[1], "reference", 1 - general[1].tags["distance"])
            ]
        elif general[0].exclusive:
            result.notes.append(f"{general[0].id} is exclusive; reference suppressed")
    else:
        # Every flow rule was excluded by the requires-gate (or the pool has
        # none at all). A diversion-only campaign genuinely has nothing left
        # to say here; a flow-based one must not end the call by exhaustion —
        # see _fetch_lowest_eligible_step's docstring.
        fallback = _fetch_lowest_eligible_step(conn, state, table)
        if fallback is not None:
            result.governing = RetrievedRule(fallback, "governing", 1.0)
            result.notes.append(
                f"general pool empty after prerequisite filtering; fell back to "
                f"lowest-numbered eligible step {fallback.id} rather than closing the call"
            )
        else:
            result.notes.append("memory is empty — no rule retrieved")

    return result
