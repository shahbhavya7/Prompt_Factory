"""The human approval queue for learned rules.

Nothing the consolidator proposes reaches the retrieval pool on its own any
more. A candidate that clears grounding and the duplicate check is *queued*
here, with the exchange that triggered it, and waits for a person. Approving
is the only path from this queue into `chunks`.

Why this is a separate module rather than more of consolidator.py: the
consolidator runs between calls and only ever writes. This runs whenever a
human happens to have time — a different process, usually with no call in
flight at all — and both reads and mutates. Keeping them apart is what lets
the queue be reviewed with the agent shut down.

The queue is a Postgres table (`needs_review`), not an in-process list, for
exactly that reason: it has to survive the agent stopping, which is the normal
case, not the exceptional one.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text as sql_text
from sqlalchemy.sql import func

from sace_chat.db import NeedsReviewRow, SessionLocal, insert_chunk
from sace_chat.kb import INTENT_EXEMPLARS
from sace_chat.models import Chunk

# What `intent` values a reviewer may choose from without inventing one. These
# are the labels IntentRouter can actually produce, so a rule tagged with
# anything else is stored and then never retrieved — see below on why a custom
# intent is nonetheless allowed.
KNOWN_INTENTS = sorted(INTENT_EXEMPLARS)

VALID_PRIORITIES = ("critical", "high", "normal", "low")


class ReviewError(ValueError):
    """A review action that cannot be applied as asked."""


def enqueue(
    *,
    candidate,
    reason: str,
    session_id: str | None = None,
    trigger_message: str = "",
    trigger_reply: str = "",
    existing_chunk_id: str | None = None,
    session=None,
) -> str:
    """Put one candidate in the queue. Returns the new row id.

    `candidate` is a consolidator.Candidate — passed whole rather than
    field-by-field so that the row can rebuild a real Chunk on approval,
    `cue` included. A queue that stored only the text would make the human
    re-invent the one field that decides whether the rule is ever found.
    """
    row_id = str(uuid.uuid4())
    row = NeedsReviewRow(
        id=row_id,
        candidate_text=candidate.text,
        candidate_cue=candidate.retrieval_text,
        intent=candidate.intent,
        priority=getattr(candidate, "priority", "normal") or "normal",
        learned_kind=candidate.learned_kind,
        source_line=candidate.source_line or "",
        existing_chunk_id=existing_chunk_id,
        reason=reason,
        session_id=session_id,
        trigger_message=trigger_message or "",
        trigger_reply=trigger_reply or "",
        status="pending",
    )
    owned = session is None
    session = session or SessionLocal()
    try:
        session.add(row)
        session.commit()
    finally:
        if owned:
            session.close()
    return row_id


def _row_to_dict(row) -> dict:
    return {
        "id": row.id,
        "reason": row.reason,
        "text": row.candidate_text,
        "cue": row.candidate_cue or "",
        "intent": row.intent,
        "priority": row.priority or "normal",
        "learned_kind": row.learned_kind,
        "source_line": row.source_line or "",
        "existing_chunk_id": row.existing_chunk_id,
        "session_id": row.session_id,
        "trigger_message": row.trigger_message or "",
        "trigger_reply": row.trigger_reply or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def list_pending(limit: int = 200) -> list[dict]:
    """Everything awaiting a human, oldest first — review order is arrival
    order, so a rule proposed three calls ago isn't buried under newer ones."""
    with SessionLocal() as session:
        rows = (
            session.query(NeedsReviewRow)
            .filter(NeedsReviewRow.status == "pending")
            .order_by(NeedsReviewRow.created_at.asc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]


def pending_count() -> int:
    with SessionLocal() as session:
        return (
            session.query(NeedsReviewRow)
            .filter(NeedsReviewRow.status == "pending")
            .count()
        )


def known_intents() -> list[str]:
    """The routable labels, plus any intent already in use on a stored rule.

    A custom intent approved earlier is included so the next reviewer sees it
    as an existing choice rather than re-typing it (and risking a near-miss
    spelling that silently splits one section into two).
    """
    labels = set(KNOWN_INTENTS)
    with SessionLocal() as session:
        rows = session.execute(
            sql_text("SELECT DISTINCT intent FROM chunks WHERE intent IS NOT NULL")
        ).all()
    labels.update(r[0] for r in rows if r[0])
    return sorted(labels)


def approve(
    review_id: str,
    embedder,
    *,
    text: str | None = None,
    cue: str | None = None,
    intent: str | None = None,
    priority: str | None = None,
    learned_kind: str | None = None,
    set_intent: bool = False,
) -> dict:
    """Approve one queued rule, with optional human edits, and insert it.

    Every field the human can see is overridable, `cue` included — it is what
    gets embedded, so an unfixable bad cue would mean approving a rule that can
    never be retrieved.

    `set_intent` distinguishes "leave the intent as proposed" from "the human
    deliberately chose no intent (a general rule)", which `intent=None` alone
    cannot express.

    A custom intent is accepted even though IntentRouter has no exemplars for
    it: such a rule is reachable only if the label is also added to
    kb.INTENT_EXEMPLARS, so the caller is warned rather than blocked. Storing it
    is still strictly better than silently rewriting the human's choice.
    """
    session = SessionLocal()
    try:
        row = session.get(NeedsReviewRow, review_id)
        if row is None:
            raise ReviewError(f"no review row {review_id!r}")
        if row.status != "pending":
            raise ReviewError(f"review row {review_id!r} is already {row.status}")

        final_text = (text if text is not None else row.candidate_text or "").strip()
        if not final_text:
            raise ReviewError("rule text cannot be empty")

        # Falling back to the text mirrors insert_chunk/Chunk.cue: a rule with
        # no cue is embedded from its own text.
        final_cue = (cue if cue is not None else row.candidate_cue or "").strip()

        final_intent = (intent if set_intent else row.intent)
        if final_intent is not None:
            final_intent = final_intent.strip().lower() or None

        final_priority = (priority or row.priority or "normal").strip().lower()
        if final_priority not in VALID_PRIORITIES:
            raise ReviewError(
                f"priority must be one of {', '.join(VALID_PRIORITIES)}, got {final_priority!r}"
            )

        final_kind = (learned_kind or row.learned_kind or "policy").strip().lower()
        if final_kind not in {"policy", "example", "failure"}:
            final_kind = "policy"

        chunk = Chunk(
            id=f"learned_{uuid.uuid4().hex[:8]}",
            title=f"Learned rule ({final_intent or 'general'})",
            text=final_text,
            cue=final_cue,
            intent=final_intent,
            priority=final_priority,
            source="learned",
            learned_kind=final_kind,
        )
        # Same single validated insert path load_kb.py and the consolidator use,
        # so an approved rule is indistinguishable in shape from any other and a
        # zero-norm or wrong-dimension vector still cannot reach the pool.
        insert_chunk(session, chunk, embedder, learned_kind=final_kind)

        row.status = "approved"
        row.reviewed_at = func.now()
        session.delete(row)
        session.commit()

        unroutable = bool(final_intent) and final_intent not in set(KNOWN_INTENTS)
        return {
            "chunk_id": chunk.id,
            "intent": final_intent,
            "priority": final_priority,
            "learned_kind": final_kind,
            "cue": final_cue,
            "text": final_text,
            # Surfaced so the UI can say so plainly: the rule is stored, but
            # until this label has exemplars in kb.INTENT_EXEMPLARS the router
            # cannot classify a caller into it, so nothing will retrieve it.
            "warning": (
                f"intent {final_intent!r} has no exemplars in kb.INTENT_EXEMPLARS, "
                "so IntentRouter cannot route to it yet — add exemplars for it, "
                "or this rule will not be retrieved"
                if unroutable else None
            ),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def discard(review_id: str) -> dict:
    """Reject one queued rule. It leaves the queue and never enters the pool."""
    with SessionLocal() as session:
        row = session.get(NeedsReviewRow, review_id)
        if row is None:
            raise ReviewError(f"no review row {review_id!r}")
        if row.status != "pending":
            raise ReviewError(f"review row {review_id!r} is already {row.status}")
        text_snapshot = row.candidate_text
        row.status = "discarded"
        row.reviewed_at = func.now()
        session.delete(row)
        session.commit()
    return {"discarded": review_id, "text": text_snapshot}
