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
import shutil
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from sace_chat import answer_cache, manager, review
from sace_chat.consolidator import run_learning_loop
from sace_chat.db import engine as db_engine
from sace_chat.db import init_db, record_call_transcript
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import get_llm

from api.models import (
    ApproveRequest,
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


# ───────────────────────────── review queue ─────────────────────────────────
# Deliberately independent of any call: the queue is reviewed whenever a person
# has time, which is usually with no call in flight and often with the voice
# agent shut down entirely. That is also why these live on this HTTP API rather
# than on voice_agent.py's websocket, which only exists while a call is running.


# ── the voice agent as a child process ──────────────────────────────────────
#
# WHY THIS LIVES HERE. The dashboard's "Start call" button is served by Vite and
# its click travels over voice_agent.py's own websocket — so neither of those
# can start voice_agent.py, because both must already be up for the button to
# exist at all. This API is the one piece that IS already running and can spawn
# it, which is why the button routes through here rather than over the ws.
#
# The child is `./run.sh voice`, not `python voice_agent.py console` directly:
# run.sh activates the conda env, loads .env, brings the database up and checks
# the KB is populated. Reimplementing that here would be a second copy of the
# same boot sequence, guaranteed to drift.
#
# run.sh `exec`s into python, so the shell is REPLACED by the agent process
# rather than becoming its parent. That matters for stopping it: the pid we hold
# is the agent itself, so terminating it actually ends the call rather than
# orphaning a python child under a dead shell.

_VOICE_PROC: subprocess.Popen | None = None
_VOICE_LOG = Path("/tmp/sace-voice-agent.log")
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _voice_running() -> bool:
    return _VOICE_PROC is not None and _VOICE_PROC.poll() is None


@app.get("/voice/status")
def voice_status():
    """Whether the agent this API started is still alive.

    `exit_code` is surfaced on a dead process so the dashboard can say WHY it
    is not running — a crash on boot (bad credentials, database down) otherwise
    looks identical to never having been started.
    """
    running = _voice_running()
    # Also counts agents this API did not spawn, so the dashboard's Stop button
    # is offered whenever there is actually something to stop — see
    # _find_voice_pids.
    external = [p for p in _find_voice_pids()
                if not (running and p == _VOICE_PROC.pid)]
    return {
        "running": running or bool(external),
        "owned": running,
        "pid": _VOICE_PROC.pid if running else (external[0] if external else None),
        "pids": ([_VOICE_PROC.pid] if running else []) + external,
        "exit_code": (None if running or _VOICE_PROC is None
                      else _VOICE_PROC.returncode),
        "log": str(_VOICE_LOG),
    }


@app.post("/voice/start")
def voice_start():
    """Spawn `./run.sh voice`, so a call can be placed without a terminal.

    NOTE ON AUDIO: console mode binds the machine's microphone and speakers —
    the SERVER's, which is only the same machine as the browser's because this
    is a local demo. This endpoint is not safe to expose on a shared host, and
    it is a demo convenience rather than a deployment path.
    """
    global _VOICE_PROC

    if _voice_running():
        # Not an error: the button's intent is "I want a live agent", and there
        # already is one. Reported so the dashboard can say so plainly.
        return {"started": False, "already_running": True,
                "pid": _VOICE_PROC.pid, "log": str(_VOICE_LOG)}

    script = _REPO_ROOT / "run.sh"
    if not script.exists():
        raise HTTPException(500, f"run.sh not found at {script}")
    bash = shutil.which("bash")
    if bash is None:
        raise HTTPException(500, "bash not found on PATH")

    # Output goes to a file, not a pipe: nothing here ever reads it, and a full
    # pipe buffer would block the agent mid-call.
    log = open(_VOICE_LOG, "ab", buffering=0)
    log.write(f"\n===== started by /voice/start at {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"=====\n".encode())
    _VOICE_PROC = subprocess.Popen(
        [bash, str(script), "voice"],
        cwd=str(_REPO_ROOT),
        stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # Its own process group, so stopping it cannot signal this API too, and
        # so a Ctrl-C in the API's terminal does not kill a live call.
        start_new_session=True,
    )

    # A boot failure (missing DEEPGRAM_API_KEY, database down) is immediate, and
    # reporting "started" for a process that is already dead sends the user to
    # the dashboard to wait for a websocket that will never open. Give it a
    # moment and check.
    time.sleep(1.5)
    if _VOICE_PROC.poll() is not None:
        tail = ""
        try:
            tail = _VOICE_LOG.read_text(errors="replace")[-800:]
        except Exception:
            pass
        code = _VOICE_PROC.returncode
        _VOICE_PROC = None
        raise HTTPException(500, f"voice agent exited immediately (code {code}). "
                                 f"Last output:\n{tail}")

    return {"started": True, "already_running": False,
            "pid": _VOICE_PROC.pid, "log": str(_VOICE_LOG)}


def _stop_pid(pid: int) -> bool:
    """SIGINT one agent and wait for it. SIGKILL only if it will not go.

    SIGINT rather than SIGKILL because the agent's shutdown callback runs the
    learning loop and persists the transcript — killing outright throws away
    the call that just happened.
    """
    try:
        # Signal the whole process group: run.sh execs into python, so the group
        # is the agent and any helper it spawned.
        os.killpg(os.getpgid(pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        return False

    # Learning can take a few seconds, so give it real time before escalating.
    deadline = time.time() + 25
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.3)

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return True


def _find_voice_pids() -> list[int]:
    """Every voice_agent.py on this machine, whoever started it.

    `_VOICE_PROC` only knows about agents THIS API spawned. One started from a
    terminal (./run.sh voice) is invisible to it — and that is the common case,
    so a Stop button that only handled its own children would silently do
    nothing exactly when it looked most broken. `pgrep -f` finds them all.
    """
    pids: list[int] = []
    try:
        out = subprocess.run(
            ["pgrep", "-f", "voice_agent.py"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            # Never signal ourselves, whatever the pattern matched.
            if pid != os.getpid():
                pids.append(pid)
    except Exception as exc:
        print(f"[voice] pgrep failed: {type(exc).__name__}: {exc}")
    return pids


@app.post("/voice/stop")
def voice_stop():
    """Stop EVERY voice agent on this machine, not only the one this API
    spawned — see _find_voice_pids for why that distinction matters.

    Different from the dashboard's End-call button, which goes over the
    websocket: that ends the CALL and runs the learning loop, leaving the agent
    up for the next one. This is the blunter one — it ends the process and frees
    the websocket port.
    """
    global _VOICE_PROC

    pids = _find_voice_pids()
    # The tracked child may already be gone; including it anyway is harmless
    # (_stop_pid reports False for a pid that no longer exists) and covers the
    # case where pgrep's pattern misses it.
    if _VOICE_PROC is not None and _VOICE_PROC.poll() is None:
        if _VOICE_PROC.pid not in pids:
            pids.append(_VOICE_PROC.pid)

    if not pids:
        _VOICE_PROC = None
        return {"stopped": False, "killed": [], "reason": "no voice agent running"}

    killed = [pid for pid in pids if _stop_pid(pid)]
    _VOICE_PROC = None
    return {"stopped": bool(killed), "killed": killed, "count": len(killed)}


@app.get("/cache/stats")
def cache_stats():
    """Entries, cumulative hits, and the similarity bar — for the dashboard."""
    return answer_cache.stats()


@app.post("/cache/clear")
def cache_clear(source: str | None = None):
    """Drop cached answers. The pool and the rules are untouched; entries
    rebuild themselves from the next few calls.

    With no `source`, drops everything (unchanged behaviour). With
    `source=seed`, drops only pre-seeded rows so a campaign's answer bank can
    be reloaded without wiping ones a real caller's turn confirmed
    (`source=live`) — mirrors load_kb.py's `delete where source != 'learned'`
    split for the chunks pool.
    """
    return {"cleared": answer_cache.clear(source=source)}


@app.get("/review/pending")
def review_pending():
    """Everything awaiting a human, oldest first, plus the intent vocabulary a
    reviewer can assign from."""
    return {
        "pending": review.list_pending(),
        "intents": review.known_intents(),
        "count": review.pending_count(),
    }


@app.post("/review/{review_id}/approve")
def review_approve(review_id: str, body: ApproveRequest):
    """Approve one queued rule, with the human's edits applied, and insert it.

    `intent` is only applied when the payload actually carries the key — that is
    what distinguishes "leave it as proposed" from "the human chose no intent",
    which a null alone cannot express.
    """
    try:
        result = review.approve(
            review_id,
            app.state.embedder,
            text=body.text,
            cue=body.cue,
            intent=body.intent,
            priority=body.priority,
            learned_kind=body.learned_kind,
            set_intent=body.set_intent,
        )
    except review.ReviewError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    return result


@app.post("/review/{review_id}/discard")
def review_discard(review_id: str):
    try:
        return review.discard(review_id)
    except review.ReviewError as exc:
        raise HTTPException(400, str(exc))


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
