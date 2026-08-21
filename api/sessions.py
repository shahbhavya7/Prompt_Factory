"""In-memory per-call state, keyed by call_id.

CallState and history are plain Python objects with no persistence of their
own (see sace_chat.retrieve.CallState) — streamlit_app.py holds one of these
per browser session via st.session_state. This is the same idea made
explicit for an HTTP API with no server-side session cookie: the frontend
generates nothing, the backend hands back a call_id on POST /calls and the
frontend passes it back on every subsequent request.

This is a single-process, in-memory store — it only works correctly under
one uvicorn worker (no --workers, no multiple processes), same as a single
Streamlit process only serving the sessions it holds in its own memory. That
is the right tradeoff for a demo; anything more is real infrastructure this
does not need yet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sace_chat.retrieve import CallState


@dataclass
class CallSession:
    state: CallState = field(default_factory=CallState)
    history: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    learning: dict | None = None


CALLS: dict[str, CallSession] = {}


def create_call() -> str:
    call_id = uuid.uuid4().hex
    CALLS[call_id] = CallSession()
    return call_id


def get_call(call_id: str) -> CallSession | None:
    return CALLS.get(call_id)
