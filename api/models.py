"""Request/response shapes for the demo API.

`SimpleDebug` is what a stakeholder sees inline on every message. `DetailDebug`
is the same field set streamlit_app.py already shows in its right-hand proof
panel — sent in the same response as `simple` (small text, no benefit to a
second round-trip) but only rendered by the frontend once a message's
"technical detail" toggle is opened.
"""

from __future__ import annotations

from pydantic import BaseModel


class StartCallResponse(BaseModel):
    call_id: str


class TurnRequest(BaseModel):
    message: str


class CallStateModel(BaseModel):
    intent: str
    opt_out: bool
    ended: bool
    asked_questions: list[str]
    collected_fields: dict[str, str]


class RuleRef(BaseModel):
    id: str
    title: str
    role: str
    intent: str | None = None
    priority: str
    terminal: bool
    exclusive: bool
    source: str
    learned_kind: str | None = None
    similarity: float
    cosine: float | None = None
    verbatim: bool = False
    char_len: int
    snippet: str
    text: str


class SimpleDebug(BaseModel):
    outcome: str
    governing_rule_id: str | None
    governing_rule_title: str | None
    elapsed_ms: float
    saved_pct: float
    regenerated: bool
    notes: list[str]


class DetailDebug(BaseModel):
    governing_cosine: float
    grounding_threshold: float
    scores: dict
    intent: str | None
    intent_similarity: float
    intent_ranked: list
    query_text: str
    governing: RuleRef | None
    reference: list[RuleRef]
    prompt_sent: str
    prompt_sent_tokens: int
    assembled_prompt_tokens: int
    monolith_tokens: int
    saved_pct: float
    elapsed_ms: float
    turn_json: dict
    raw_llm_output: str
    notes: list[str]
    llm_calls: int


class TurnResponse(BaseModel):
    reply: str
    call_state: CallStateModel
    simple: SimpleDebug
    detail: DetailDebug


class LearningResult(BaseModel):
    text: str
    intent: str
    kind: str | None
    outcome: str
    detail: str


class EndCallResponse(BaseModel):
    results: list[LearningResult] = []
    turn_count: int = 0
    error: str | None = None


class CallStatusResponse(BaseModel):
    call_state: CallStateModel
    turn_count: int


class ApproveRequest(BaseModel):
    """A human's approval of one queued rule, carrying any edits they made.

    Every field is optional: an unset field means "keep what the extractor
    proposed". `set_intent` exists because `intent=None` is itself a meaningful
    choice (a general rule reachable by similarity alone), so a null cannot
    double as "unchanged" — the UI sends set_intent=true whenever the reviewer
    touched the intent control at all.
    """

    text: str | None = None
    cue: str | None = None
    intent: str | None = None
    priority: str | None = None
    learned_kind: str | None = None
    set_intent: bool = False
