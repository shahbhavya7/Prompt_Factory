"""Learned-rule memory: a between-calls consolidation loop.

Runs on a finished transcript, never during a live turn. Extracts candidate
rules, tags each with an intent the semantic router can actually produce (or
none, making it a general rule), then runs each through a verification gate
(grounding -> duplicate -> conflict).

Clearing those gates does NOT insert the rule. It queues it in `needs_review`
for a human, together with the exchange that triggered it; a person approves,
edits, or discards it, and sace_chat.review.approve is the only path from there
into the pool. The automated gates are therefore a filter on what is worth a
human's attention, not an admission decision — the one exception being
duplicates, which are dropped outright because there is nothing to decide about
a rule we already hold.
"""

import re

from sqlalchemy import text as sql_text

from sace_chat.db import SessionLocal, check_embedding
from sace_chat.kb import INTENT_EXEMPLARS
from sace_chat.manager import _matches_any
from sace_chat.review import enqueue

# The labels the semantic router can actually return. A learned rule tagged
# with anything else would be unreachable, so extraction is constrained to
# these (or to no intent at all, which puts it in the general pool).
ROUTABLE_INTENTS = set(INTENT_EXEMPLARS)

# 0.85 (close to the textbook 0.9 duplicate bar) let real near-duplicates
# through: pairwise-checked against this KB's actual learned rules, pairs
# that are clearly the same rule restated — e.g. "caller doubts the
# legitimacy of the call" vs "caller questions the legitimacy of the call",
# both with near-identical scripted replies — scored 0.75-0.85 and were both
# kept as separate rules. 0.72 catches that whole cluster while staying
# above SAME_TOPIC_THRESHOLD, so a candidate that just barely misses
# "duplicate" still gets the conflict check rather than slipping through
# gate-free.
DUPLICATE_THRESHOLD = 0.72
SAME_TOPIC_THRESHOLD = 0.6

_NUMBER_RE = re.compile(r"\b\d+(?:[:.]\d+)?\b")
_ASSERT_WORDS = {"is", "are", "does", "do", "can", "will", "must", "always"}
_DENY_WORDS = {"not", "never", "no", "can't", "cannot", "doesn't", "don't", "won't"}


class Candidate:
    def __init__(self, text, learned_kind, intent=None, priority="normal", source_line="", cue=""):
        self.text = text
        self.learned_kind = learned_kind  # policy | example | failure
        self.intent = intent  # a ROUTABLE_INTENTS label, or None for a general rule
        self.priority = priority
        self.source_line = source_line  # the transcript line this was grounded in
        # What the rule should be retrieved BY (see models.Chunk.cue). A learned
        # rule without one is embedded from its own text, which makes it stick to
        # the turn that produced it and effectively unreachable.
        self.cue = cue

    @property
    def retrieval_text(self) -> str:
        return (self.cue or "").strip() or self.text

    @property
    def tags(self):  # back-compat for callers reading candidate.tags["special"]
        return {"special": self.intent} if self.intent else {}


_EXTRACTION_PROMPT = """\
You review finished call transcripts for a Medi-Cal coverage outreach agent and
propose reusable rules for situations the current playbook handled poorly or
does not cover.

Return ONLY a JSON object of this shape:
{{"candidates": [
  {{"text": "<the rule, imperative, 1-3 sentences>",
   "cue": "<the caller phrasings that should pull this rule up, comma separated>",
   "intent": "<one of: {intents}>  or  null",
   "learned_kind": "<policy|example|failure>",
   "source_line": "<a line copied VERBATIM from the transcript that this rule is based on>"}}
]}}

Rules:
- Propose at most 3 candidates. Return an empty list when the call was routine
  and taught you nothing new.
- "intent" MUST be one of the listed labels, or null. Those labels are the only
  ones the router can recognise, so never invent one; use null when the rule is
  general guidance rather than a handler for one of those situations.
- "text" is what the agent should DO: "When the caller ..., say ...".
- "cue" is what the rule is FOUND BY, and it is what gets embedded. Write it as
  the caller's own words — six to ten short phrasings of the thing they would
  say, comma separated, ending with a brief note on what was pending. Example:
  "hold on let me grab a pen, can you repeat that, say it again, one sec -- the
  caller wants the number again." Never put the agent's own scripted line in the
  cue: the query contains what the agent last said, so a cue echoing it makes
  the rule match the wrong turn.
- "source_line" MUST be copied character-for-character from a line in the
  transcript. It is checked, and the candidate is discarded if it does not
  appear there, so never paraphrase it or write a line nobody said.
- learned_kind: "failure" when the agent handled it badly, "policy" for a new
  standing rule, "example" for a good exemplar exchange.
"""


def extract_candidates(transcript: str, llm=None) -> list[Candidate]:
    """LLM extraction of candidate rules from a finished transcript.

    `intent` is constrained to ROUTABLE_INTENTS or None — those are the only
    values retrieval can act on, so an out-of-vocabulary label would produce a
    rule that is stored and then never found again.
    """
    if llm is None:
        from sace_chat.llm import get_llm

        llm = get_llm()

    system = _EXTRACTION_PROMPT.format(intents="|".join(sorted(ROUTABLE_INTENTS)))

    from sace_chat.llm import parse_json_object

    try:
        raw = llm.chat_json(system, f"TRANSCRIPT:\n{transcript}")
    except AttributeError:
        raw = llm.chat(system, [{"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}])
    except Exception:
        return []

    obj, _ = parse_json_object(raw)
    if not obj:
        return []

    candidates = []
    for item in (obj.get("candidates") or [])[:3]:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        # Accept the old key name too — the extraction prompt changed, the
        # model sometimes echoes the older shape.
        raw_intent = item.get("intent", item.get("special"))
        intent = (raw_intent or "").strip().lower() or None
        if intent in {"none", "null"}:
            intent = None
        if intent is not None and intent not in ROUTABLE_INTENTS:
            intent = None  # unroutable label -> general rule rather than a dead one
        kind = item.get("learned_kind")
        if kind not in {"policy", "example", "failure"}:
            kind = "policy"
        candidates.append(
            Candidate(
                text=text,
                learned_kind=kind,
                intent=intent,
                priority="normal",
                source_line=(item.get("source_line") or "").strip(),
                cue=(item.get("cue") or "").strip(),
            )
        )
    return candidates


def _parse_vector(raw) -> list[float]:
    """pgvector's `vector` column comes back from a raw SQL fetch as its
    string literal form, e.g. "[0,0.1,-0.2,...]" — not a Python list or
    numpy array. Parse it once here rather than relying on a .tolist()
    that a plain string doesn't have."""
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    if hasattr(raw, "tolist"):
        return raw.tolist()
    return list(raw)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_grounded(candidate: Candidate, transcript: str) -> bool:
    """GROUNDING gate: reject a candidate whose key claim isn't actually
    present in (or a close paraphrase of) a real transcript line — guards
    against extractor hallucination. Since this extractor builds candidate
    text directly out of a transcript line it already located, grounding
    here means confirming that line is still verbatim-present in the
    transcript (catches a candidate being fabricated or edited downstream
    of extraction)."""
    return bool(candidate.source_line) and candidate.source_line in transcript


def _is_conflict(candidate_text: str, existing_text: str) -> bool:
    """Same topic (checked by caller via cosine before this is called), but
    do the two texts actually assert something contradictory? Heuristic:
    if both mention numbers and the number sets differ, that's a conflicting
    value (e.g. different callback-hours). If one asserts and the other
    denies the same predicate, that's an assert/deny flip."""
    cand_lower = candidate_text.lower()
    existing_lower = existing_text.lower()

    cand_numbers = set(_NUMBER_RE.findall(cand_lower))
    existing_numbers = set(_NUMBER_RE.findall(existing_lower))
    if cand_numbers and existing_numbers and cand_numbers != existing_numbers:
        return True

    cand_has_assert = bool(_matches_any([rf"\b{w}\b" for w in _ASSERT_WORDS], cand_lower))
    cand_has_deny = bool(_matches_any([rf"\b{re.escape(w)}\b" for w in _DENY_WORDS], cand_lower))
    existing_has_assert = bool(_matches_any([rf"\b{w}\b" for w in _ASSERT_WORDS], existing_lower))
    existing_has_deny = bool(_matches_any([rf"\b{re.escape(w)}\b" for w in _DENY_WORDS], existing_lower))

    if cand_has_deny and existing_has_assert and not existing_has_deny:
        return True
    if cand_has_assert and not cand_has_deny and existing_has_deny:
        return True

    return False


class GateResult:
    def __init__(self, candidate: Candidate, outcome: str, detail: str = "", review_id: str | None = None):
        self.candidate = candidate
        # queued-for-approval | duplicate-skipped | conflict-needs-review | ungrounded-rejected
        #
        # Note there is no "inserted" outcome any more. Clearing the gates now
        # earns a candidate a place in the human review queue, not a place in
        # the pool — sace_chat.review.approve is the only path into `chunks`.
        self.outcome = outcome
        self.detail = detail
        self.review_id = review_id  # the needs_review row a human will act on


def _fetch_pool(conn, table: str, intent: str | None):
    """Rules in the candidate's own section only: same intent, or the general
    pool (intent IS NULL) for a candidate with no intent.

    A "caller is busy" candidate can never be retrieved against a "caller
    wants a callback" turn — retrieve.py routes by intent first and only ever
    considers one section at a time (see retrieve._fetch_by_intent /
    _fetch_general). Comparing it for duplicates or conflicts against rules
    from every other section wasted a growing number of cosine comparisons on
    rules it could never actually collide with, and let two unrelated rules
    that happen to score close in embedding space register as a false
    "conflict" needing human review.

    Note the stored `embedding` is of each rule's CUE, so the candidate must be
    embedded from its cue too for the comparison to mean anything. Cue-to-cue is
    also the better duplicate test: two rules are duplicates when they fire on
    the same situation, which is exactly what a cue describes.
    """
    if intent is None:
        query = f"SELECT id, text, embedding FROM {table} WHERE intent IS NULL"
        params = {}
    else:
        query = f"SELECT id, text, embedding FROM {table} WHERE intent = :intent"
        params = {"intent": intent}
    return conn.execute(sql_text(query), params).fetchall()


def _find_trigger(transcript: str, source_line: str) -> tuple[str, str]:
    """The caller line the candidate came from, and the agent's reply to it.

    This is the reviewer's evidence: "this rule was proposed because the caller
    said X and Maya answered Y." Derived from the transcript rather than
    threaded down from the live turn, because the consolidator only ever
    receives the finished transcript — and `source_line` is already required to
    be verbatim-present in it by the grounding gate, so it can be located.
    """
    if not source_line:
        return "", ""
    lines = transcript.splitlines()
    for i, line in enumerate(lines):
        if source_line in line:
            reply = ""
            for following in lines[i + 1:]:
                if following.startswith("Maya:"):
                    reply = following.split(":", 1)[1].strip()
                    break
            caller = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            return caller, reply
    return "", ""


def run_learning_loop(
    transcript: str,
    embedder,
    conn,
    table: str = "chunks",
    llm=None,
    session_id: str | None = None,
) -> list[GateResult]:
    """Between-calls consolidation. Must only ever run after a call has ended —
    never from engine.step()'s live path.

    The caller is responsible for only invoking this post-call (the UI's
    "End call & learn" button, or state.ended).

    **This no longer writes to the pool.** A candidate that clears grounding and
    the duplicate check is queued in `needs_review` for a human to approve,
    edit, or discard (see sace_chat.review). Duplicates are still dropped
    silently — there is nothing for a person to decide about a rule we already
    hold — and conflicts and ungrounded candidates land in the same queue,
    tagged with why. Approving is the only way a rule reaches `chunks`.
    """
    if not transcript.strip():
        return []

    candidates = extract_candidates(transcript, llm=llm)
    results = []
    session = SessionLocal()

    try:
        for candidate in candidates:
            trigger_message, trigger_reply = _find_trigger(transcript, candidate.source_line)

            if not _is_grounded(candidate, transcript):
                review_id = enqueue(
                    candidate=candidate,
                    reason="ungrounded",
                    session_id=session_id,
                    trigger_message=trigger_message,
                    trigger_reply=trigger_reply,
                    session=session,
                )
                results.append(GateResult(
                    candidate, "ungrounded-rejected",
                    "source line not found in transcript", review_id=review_id,
                ))
                continue

            candidate_vec = check_embedding(
                embedder.embed(candidate.retrieval_text), chunk_id="candidate"
            )
            existing_rows = _fetch_pool(conn, table, candidate.intent)

            best_sim = 0.0
            best_match_id = None
            best_match_text = None
            for row in existing_rows:
                existing_vec = _parse_vector(row.embedding)
                sim = _cosine(candidate_vec, existing_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_match_id = row.id
                    best_match_text = row.text

            if best_sim > DUPLICATE_THRESHOLD:
                results.append(
                    GateResult(candidate, "duplicate-skipped", f"cosine={best_sim:.3f} vs {best_match_id}")
                )
                continue

            if best_sim > SAME_TOPIC_THRESHOLD and _is_conflict(candidate.text, best_match_text):
                review_id = enqueue(
                    candidate=candidate,
                    reason="conflict",
                    session_id=session_id,
                    trigger_message=trigger_message,
                    trigger_reply=trigger_reply,
                    existing_chunk_id=best_match_id,
                    session=session,
                )
                results.append(
                    GateResult(
                        candidate,
                        "conflict-needs-review",
                        f"cosine={best_sim:.3f} vs {best_match_id}, contradicts existing rule",
                        review_id=review_id,
                    )
                )
                continue

            # Cleared every automated gate — which now earns a place in the
            # review queue, not in the pool. The embedding is deliberately NOT
            # computed here: the human may rewrite the cue, and the cue is what
            # gets embedded, so embedding now would either be thrown away or
            # (worse) quietly kept while the text says something else.
            review_id = enqueue(
                candidate=candidate,
                reason="pending",
                session_id=session_id,
                trigger_message=trigger_message,
                trigger_reply=trigger_reply,
                session=session,
            )
            results.append(GateResult(
                candidate, "queued-for-approval",
                f"review_id={review_id}", review_id=review_id,
            ))

    finally:
        session.close()

    return results
