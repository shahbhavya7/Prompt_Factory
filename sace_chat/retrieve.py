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

from dataclasses import dataclass, field as dc_field

from sqlalchemy import text

from sace_chat.embeddings import embed_many
from sace_chat.kb import INTENT_EXEMPLARS
from sace_chat.models import Chunk

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

_SELECT_COLS = "id, title, text, intent, priority, terminal, exclusive, source, learned_kind"


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
    )


class IntentRouter:
    """Semantic intent classifier over kb.INTENT_EXEMPLARS.

    Exemplar vectors are embedded once, on first use, in a single batch — with a
    network embedder, doing it per turn would add dozens of sequential
    round-trips to every reply.
    """

    def __init__(self, embedder, threshold: float = INTENT_THRESHOLD):
        self.embedder = embedder
        self.threshold = threshold
        self._vectors: list[tuple[str, list[float]]] | None = None

    def warm(self):
        if self._vectors is not None:
            return
        labels, texts = [], []
        for intent, exemplars in INTENT_EXEMPLARS.items():
            for exemplar in exemplars:
                labels.append(intent)
                texts.append(exemplar)
        self._vectors = list(zip(labels, embed_many(self.embedder, texts)))

    def detect(self, message_vec) -> tuple[str | None, float, list]:
        """Returns (intent or None, best similarity, ranked [(intent, sim)...])."""
        self.warm()
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
            SELECT {_SELECT_COLS}, embedding <=> :qvec AS distance
            FROM {table}
            WHERE intent = :intent
            ORDER BY distance
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


def _fetch_general(conn, qvec, table: str, k: int = 2) -> list[Chunk]:
    rows = conn.execute(
        text(
            f"""
            SELECT {_SELECT_COLS}, embedding <=> :qvec AS distance
            FROM {table}
            WHERE intent IS NULL
            ORDER BY distance
            LIMIT :k
            """
        ),
        {"qvec": str(list(qvec)), "k": k},
    ).fetchall()
    out = []
    for row in rows:
        chunk = _row_to_chunk(row)
        chunk.tags["distance"] = float(row.distance)
        out.append(chunk)
    return out


def retrieve(
    conn,
    state: CallState,
    message: str,
    embedder,
    history: list | None = None,
    router: IntentRouter | None = None,
    table: str = "chunks",
    precedence=None,
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

    intent, intent_sim, ranked = router.detect(message_vec)
    result = Retrieval(
        intent=intent,
        intent_similarity=round(intent_sim, 3),
        intent_ranked=[(i, round(s, 3)) for i, s in ranked[:4]],
        query_text=query_text,
    )

    if precedence is not None:
        effective, opt_out = precedence(intent or "none", message)
        result.opt_out = bool(opt_out)
        effective = None if effective == "none" else effective
        if effective != intent:
            result.notes.append(f"precedence: intent {intent!r} -> {effective!r}")
            intent = effective
            result.intent = effective

    if intent is not None:
        chunk = _fetch_by_intent(conn, intent, pool_vec, table)
        if chunk is not None:
            result.governing = RetrievedRule(chunk, "governing", 1 - chunk.tags["distance"])
            return result
        # An intent with no rule behind it (e.g. a label whose seed rule was
        # deleted) must not silently swallow the turn.
        result.notes.append(f"intent {intent!r} matched no rule; fell through to the general pool")
        result.intent = None

    general = _fetch_general(conn, pool_vec, table, k=2)
    if general:
        result.governing = RetrievedRule(general[0], "governing", 1 - general[0].tags["distance"])
        if not general[0].exclusive and len(general) > 1:
            result.reference = [
                RetrievedRule(general[1], "reference", 1 - general[1].tags["distance"])
            ]
        elif general[0].exclusive:
            result.notes.append(f"{general[0].id} is exclusive; reference suppressed")
    else:
        result.notes.append("memory is empty — no rule retrieved")

    return result
