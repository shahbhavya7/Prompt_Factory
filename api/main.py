"""Stakeholder-demo API: a thin REST wrapper around sace_chat.engine.Engine.

Boot mirrors streamlit_app.py's boot() exactly — one shared Engine instance,
constructed once at startup, held on app.state for every request to reuse.

Single-process only: CallSession state lives in api.sessions.CALLS, an
in-memory dict. Do not run this with `uvicorn --workers N>1` or behind
multiple processes — a second worker would not see the first worker's calls,
same constraint Streamlit has per-session. Fine for a demo; would need a real
session store (Redis, or a DB-backed CallState) to go beyond one process.

Run:  uvicorn api.main:app --reload --port ${API_PORT:-8000}
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from sace_chat import manager
from sace_chat.consolidator import run_learning_loop
from sace_chat.db import engine as db_engine
from sace_chat.db import init_db, record_call_transcript
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import get_llm

from api.models import (
    CallStateModel,
    CallStatusResponse,
    DetailDebug,
    EndCallResponse,
    LearningResult,
    RuleRef,
    SimpleDebug,
    StartCallResponse,
    TurnRequest,
    TurnResponse,
)
from api.sessions import create_call, get_call


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    embedder = get_embedder()
    llm = get_llm()
    engine = Engine(stable_core=STABLE_CORE, rules=RULES, embedder=embedder, manager=manager, llm=llm)
    # Exemplars embedded once here, at startup, so the first real turn doesn't
    # pay for it as latency — same reasoning as streamlit_app.py's boot().
    engine.router.warm()
    app.state.engine = engine
    app.state.embedder = embedder
    app.state.llm = llm
    print(f"[api] engine ready · {len(RULES)} seed rules")
    yield


app = FastAPI(title="sace-chat demo API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _call_state_model(state) -> CallStateModel:
    return CallStateModel(
        intent=state.intent,
        opt_out=state.opt_out,
        ended=state.ended,
        asked_questions=list(state.asked_questions),
        collected_fields=dict(state.collected_fields),
    )


def _rule_ref(rule: dict | None) -> RuleRef | None:
    return RuleRef(**rule) if rule is not None else None


@app.post("/calls", response_model=StartCallResponse)
def start_call():
    call_id = create_call()
    return StartCallResponse(call_id=call_id)


@app.get("/calls/{call_id}", response_model=CallStatusResponse)
def call_status(call_id: str):
    session = get_call(call_id)
    if session is None:
        raise HTTPException(404, "unknown call_id")
    return CallStatusResponse(
        call_state=_call_state_model(session.state),
        turn_count=len(session.turns),
    )


@app.post("/calls/{call_id}/turns", response_model=TurnResponse)
def send_turn(call_id: str, body: TurnRequest):
    session = get_call(call_id)
    if session is None:
        raise HTTPException(404, "unknown call_id")

    engine: Engine = app.state.engine
    try:
        reply, _, debug = engine.step(session.state, session.history, body.message)
    except Exception as exc:
        # Never a 500 that would break the chat — same fallback shape
        # streamlit_app.py uses around its own engine.step call.
        reply = f"[engine error: {type(exc).__name__}: {exc}]"
        debug = None

    if debug is None:
        return TurnResponse(
            reply=reply,
            call_state=_call_state_model(session.state),
            simple=SimpleDebug(
                outcome="error", governing_rule_id=None, governing_rule_title=None,
                elapsed_ms=0.0, saved_pct=0.0, regenerated=False, notes=[],
            ),
            detail=DetailDebug(
                governing_cosine=0.0, grounding_threshold=0.0, scores={}, intent=None,
                intent_similarity=0.0, intent_ranked=[], query_text="", governing=None,
                reference=[], prompt_sent="", prompt_sent_tokens=0, assembled_prompt_tokens=0,
                monolith_tokens=0, saved_pct=0.0, elapsed_ms=0.0, turn_json={},
                raw_llm_output="", notes=[], llm_calls=0,
            ),
        )

    session.turns.append(debug)
    gov = debug.get("governing")

    return TurnResponse(
        reply=reply,
        call_state=_call_state_model(session.state),
        simple=SimpleDebug(
            outcome=debug["outcome"],
            governing_rule_id=gov["id"] if gov else None,
            governing_rule_title=gov["title"] if gov else None,
            elapsed_ms=debug["elapsed_ms"],
            saved_pct=debug["saved_pct"],
            regenerated=debug["regenerated"],
            notes=debug["notes"],
        ),
        detail=DetailDebug(
            governing_cosine=debug["governing_cosine"],
            grounding_threshold=debug["grounding_threshold"],
            scores=debug["scores"],
            intent=debug["intent"],
            intent_similarity=debug["intent_similarity"],
            intent_ranked=debug["intent_ranked"],
            query_text=debug["query_text"],
            governing=_rule_ref(gov),
            reference=[_rule_ref(r) for r in debug["reference"]],
            prompt_sent=debug["prompt_sent"],
            prompt_sent_tokens=debug["prompt_sent_tokens"],
            assembled_prompt_tokens=debug["assembled_prompt_tokens"],
            monolith_tokens=debug["monolith_tokens"],
            saved_pct=debug["saved_pct"],
            elapsed_ms=debug["elapsed_ms"],
            turn_json=debug["turn_json"],
            raw_llm_output=debug["raw_llm_output"],
            notes=debug["notes"],
            llm_calls=debug["llm_calls"],
        ),
    )


@app.post("/calls/{call_id}/end", response_model=EndCallResponse)
def end_call(call_id: str):
    session = get_call(call_id)
    if session is None:
        raise HTTPException(404, "unknown call_id")

    transcript = "\n".join(session.history)
    if not transcript.strip():
        session.state.ended = True
        return EndCallResponse(error="Nothing to learn from — the call is empty.")

    embedder = app.state.embedder
    llm = app.state.llm
    try:
        with db_engine.connect() as conn:
            results = run_learning_loop(transcript, embedder, conn, llm=llm)
    except Exception as exc:
        session.state.ended = True
        return EndCallResponse(error=f"{type(exc).__name__}: {exc}")

    session.state.ended = True
    learning_payload = [
        {
            "text": r.candidate.text,
            "intent": r.candidate.intent or "(general)",
            "kind": r.candidate.learned_kind,
            "outcome": r.outcome,
            "detail": r.detail,
        }
        for r in results
    ]
    record_call_transcript(
        session_id=call_id, source="demo", transcript=transcript,
        turn_count=len(session.turns), learning_results=learning_payload,
    )
    return EndCallResponse(
        results=[LearningResult(**r) for r in learning_payload],
        turn_count=len(session.turns),
    )
