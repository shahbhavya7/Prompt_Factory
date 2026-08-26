"""LiveKit voice entrypoint for the SACE engine.

    Deepgram STT (streaming) -> SACE context build -> streaming LLM -> Deepgram TTS

Pinned to livekit-agents 1.6.7, which is the `AgentSession` API (1.x). The older
0.12.x `VoicePipelineAgent` class does not exist in this release. Every signature
used here was read off the installed package, not guessed:

    Agent.on_user_turn_completed(turn_ctx: llm.ChatContext, new_message: llm.ChatMessage)
    Agent.llm_node(chat_ctx, tools, model_settings) -> AsyncIterable[ChatChunk | str]
    Agent.tts_node(text: AsyncIterable[str], model_settings) -> AsyncIterable[rtc.AudioFrame]
    Agent.default.llm_node / .tts_node          (delegate to framework behaviour)
    AgentSession.say(text, *, allow_interruptions=...)
    AgentSession.generate_reply(*, allow_interruptions=..., ...)
    deepgram.STT(model=..., interim_results=..., endpointing_ms=..., utterance_end_ms=...)
    deepgram.TTS(model="aura-2-...")
    silero.VAD.load(min_silence_duration=...)

WHAT THIS FILE DOES NOT DO: it never asks the SACE engine to make a completion
call. `Engine.build_turn_context` stops short of the LLM; the streaming LLM is
LiveKit's, so tokens go straight to TTS. Validation afterwards is log-only — by
then the words have been spoken, so there is nothing to regenerate.

Run:  python voice_agent.py dev        (connects a worker to LIVEKIT_URL)
      python voice_agent.py console    (local mic/speaker, no LiveKit room)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    StopResponse,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice.agent_session import InterruptionOptions, TurnHandlingOptions
from livekit.agents.voice.turn import PreemptiveGenerationOptions
from livekit import rtc
from livekit.plugins import deepgram, openai, silero

import numpy as np

from sace_audio import Denoiser, SpeakerGate, UtteranceFilter


def _to_frame(samples: np.ndarray, sample_rate: int, channels: int = 1) -> rtc.AudioFrame:
    """float32 [-1,1] back to the int16 AudioFrame the framework expects."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    return rtc.AudioFrame(
        data=pcm.tobytes(), sample_rate=sample_rate,
        num_channels=channels, samples_per_channel=len(pcm) // channels,
    )

from sace_chat import campaign, manager
from sace_chat.db import init_db, record_call_transcript, record_turn
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine, assert_message_present, question_key
from sace_chat.retrieve import CallState

CAMPAIGN = campaign.get_campaign()
RULES = campaign.load_rules(CAMPAIGN)
STABLE_CORE = CAMPAIGN.stable_core

# ─────────────────────────────── config ───────────────────────────────
STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
LLM_MODEL = os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")
AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "sace-calling-agent")
VOICE_WS_PORT = int(os.environ.get("VOICE_WS_PORT", "8765"))

# Deepgram endpointing: silence after speech before the transcript is finalised.
# ~300ms is a natural turn boundary for this kind of call; lower feels clipped,
# higher adds dead air. Exposed so it can be tuned without a code change.
ENDPOINTING_MS = int(os.environ.get("DEEPGRAM_ENDPOINTING_MS", "300"))
UTTERANCE_END_MS = int(os.environ.get("DEEPGRAM_UTTERANCE_END_MS", "1000"))
VAD_MIN_SILENCE = float(os.environ.get("VAD_MIN_SILENCE", "0.30"))

# How confident Silero must be that a frame is speech before the turn opens.
# Silero's default is 0.5; 0.6 is deliberately stricter.
#
# THIS IS A VOLUME GATE, NOT SPEAKER RECOGNITION — the distinction matters and
# is easy to oversell. It rejects the quiet, distant, or half-overlapping speech
# that a room full of people produces, because a far-off voice is attenuated and
# scores lower. It does NOTHING about a colleague talking clearly next to the
# mic: that is loud, confident speech and it will pass. Rejecting a voice
# because of WHOSE it is needs sace_audio's speaker gate, which runs after this.
#
# Raised further and real callers start getting clipped — a softly-spoken person,
# or the first syllable of a sentence. 0.6 is the point where background chatter
# drops off without the caller having to project.
VAD_ACTIVATION = float(os.environ.get("VAD_ACTIVATION", "0.6"))

# Speech must last this long to open a turn. Silero's default of 0.05s is short
# enough that a cough, a door, or one syllable from across the room starts a
# turn and sends whatever follows to the LLM. 0.2s discards those without being
# long enough to swallow a genuine "yes" or "no", which is what the script most
# often needs to hear.
VAD_MIN_SPEECH = float(os.environ.get("VAD_MIN_SPEECH", "0.20"))

# Whether each audio layer starts enabled. The dashboard toggles both at
# runtime; these only decide the state a call begins in.
AUDIO_DENOISE_DEFAULT = os.environ.get(
    "AUDIO_DENOISE", "on").strip().lower() not in {"off", "0", "false"}
AUDIO_GATE_DEFAULT = os.environ.get(
    "AUDIO_VOICE_GATE", "on").strip().lower() not in {"off", "0", "false"}


def _onoff(available: bool, enabled: bool) -> str:
    """How a layer reads in the boot line: unavailable is not the same as off."""
    if not available:
        return "unavailable"
    return "ON" if enabled else "off"


def _fmt_ms(v) -> str:
    return "   —" if v is None else f"{v:4.0f}"


# ───────────────────────── live spectator broadcast ─────────────────────────
# A set of connected browser tabs watching a call happen, fed from the exact
# same data the terminal already prints and the DB already stores. The one
# thing that flows back is "end_call" — everything the dashboard shows is
# still just what the agent already decided and did.
_WS_CLIENTS: set = set()
# The one call this worker process is currently handling, if any — set at the
# top of entrypoint, cleared when it ends. A dashboard's "End call" button
# needs a live AgentSession to close; this is the only place that exists.
_ACTIVE_SESSION: AgentSession | None = None

# The live agent, for the dashboard's audio toggles. Held alongside the session
# because the audio filter lives on the AGENT, and the toggles have to reach the
# instance that is actually processing frames — not a fresh one.
_ACTIVE_AGENT = None
# Captured once, when the ws server starts, on the loop that owns _WS_CLIENTS.
# run_learning (and therefore every "learned"/"learning_done" broadcast) runs
# via asyncio.to_thread — a real OS thread with no running loop of its own —
# so broadcast() cannot rely on asyncio.get_running_loop() the way the
# in-loop call sites (retrieval, turn) can; it needs this to schedule onto
# the right loop from any thread.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None

# The live job's context, kept so the dashboard can start a SECOND call in this
# same worker process. A LiveKit job calls entrypoint once, so without this the
# only way to place another call is to restart the worker.
_JOB_CTX = None

# Guards against two calls running at once. The dashboard can be open in several
# tabs, and a double-click or two tabs both pressing Start would otherwise build
# two AgentSessions sharing one microphone.
_CALL_STARTING = False


async def _ws_handler(websocket):
    _WS_CLIENTS.add(websocket)
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            kind = msg.get("type")
            if kind == "end_call" and _ACTIVE_SESSION is not None:
                print("[dashboard] end_call requested from the browser")
                asyncio.create_task(_close(_ACTIVE_SESSION))
            elif kind == "start_call":
                asyncio.create_task(_start_call_from_dashboard())
            elif kind == "set_audio":
                _set_audio(msg.get("denoise"), msg.get("gate"))
            elif kind == "get_audio":
                broadcast(audio_state())
    finally:
        _WS_CLIENTS.discard(websocket)


async def _start_call_from_dashboard() -> None:
    """Begin a new call because the dashboard asked for one.

    Every refusal is reported back to the browser rather than only logged: the
    button is remote, so a silent no-op looks exactly like a broken button.
    """
    global _CALL_STARTING

    if _ACTIVE_SESSION is not None:
        broadcast({"type": "start_refused", "reason": "a call is already in progress"})
        return
    if _CALL_STARTING:
        broadcast({"type": "start_refused", "reason": "a call is already starting"})
        return
    if _JOB_CTX is None:
        # No job has run yet, so there is no room to join and no prewarmed
        # engine. Nothing sensible to do but say so.
        broadcast({"type": "start_refused",
                   "reason": "the agent has no active job — restart the worker"})
        return

    _CALL_STARTING = True
    print("[dashboard] start_call requested from the browser")
    try:
        await run_call(_JOB_CTX)
    except Exception as exc:
        print(f"[dashboard] start_call failed: {type(exc).__name__}: {exc}")
        broadcast({"type": "start_refused",
                   "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        _CALL_STARTING = False


def audio_state() -> dict:
    """What the dashboard's toggles should show.

    `*_available` is reported separately from `*_enabled` because they fail
    differently and the UI must not conflate them: a layer with no enrolment or
    no library CANNOT be switched on, and showing that as a plain "off" toggle
    invites the user to flip it and watch nothing happen.
    """
    agent = _ACTIVE_AGENT
    f = getattr(agent, "_audio_filter", None) if agent else None
    return {
        "type": "audio_state",
        "denoise_available": bool(f and f.denoiser is not None),
        "denoise_enabled": bool(f and f.denoiser is not None and f.denoise_enabled),
        "gate_available": bool(f and f.gate is not None),
        "gate_enabled": bool(f and f.gate is not None and f.gate_enabled),
        # Nothing to toggle at all — no call yet, or neither layer is installed.
        "filter_present": f is not None,
    }


def _set_audio(denoise, gate) -> None:
    """Flip either layer mid-call. None means "leave this one alone"."""
    agent = _ACTIVE_AGENT
    f = getattr(agent, "_audio_filter", None) if agent else None
    if f is None:
        broadcast({**audio_state(),
                   "note": "no audio filter on this call — nothing to toggle"})
        return

    if denoise is not None and f.denoiser is not None:
        f.denoise_enabled = bool(denoise)
    if gate is not None and f.gate is not None:
        f.gate_enabled = bool(gate)

    print(f"[audio] dashboard set denoise={f.denoise_enabled} "
          f"voice-gate={f.gate_enabled}")
    # Echoed back rather than assumed by the browser: a request to enable a
    # layer that is unavailable is silently ignored above, and the UI has to
    # end up showing what is ACTUALLY on.
    broadcast(audio_state())


def broadcast(event: dict) -> None:
    """Fire-and-forget push to every connected dashboard tab.

    Called both from the event loop (retrieval, turn — inside
    on_user_turn_completed/finish_turn, which run on the loop) and from a
    plain OS thread (run_learning, called via asyncio.to_thread — it has no
    running loop of its own, so asyncio.get_running_loop() inside it always
    raises). Either way this only ever schedules the send; it never awaits
    it, so a slow or dead browser tab can't delay the call.
    """
    if not _WS_CLIENTS:
        return
    payload = json.dumps(event, default=str)

    async def _send_all():
        dead = []
        for ws in list(_WS_CLIENTS):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _WS_CLIENTS.discard(ws)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_all())
    except RuntimeError:
        # No loop running in THIS thread — we're inside run_learning's
        # asyncio.to_thread call. Schedule onto the loop that actually owns
        # _WS_CLIENTS instead of silently dropping the event.
        if _MAIN_LOOP is not None:
            asyncio.run_coroutine_threadsafe(_send_all(), _MAIN_LOOP)


async def _start_ws_server():
    import websockets

    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_running_loop()
    server = await websockets.serve(_ws_handler, "localhost", VOICE_WS_PORT)
    print(f"[dashboard] spectator websocket listening on ws://localhost:{VOICE_WS_PORT}")
    return server


class SaceVoiceAgent(Agent):
    """Wires SACE retrieval into LiveKit's pipeline at the pre-LLM hook.

    One instance per room. Holds the CallState and history for that call, so two
    concurrent callers never share retrieval context.
    """

    def __init__(self, engine: Engine, session_id: str):
        # `instructions` is replaced per turn with the SACE-assembled prompt (see
        # on_user_turn_completed). The stable core is only the initial value so a
        # reply before the first user turn is not unguided.
        super().__init__(instructions=STABLE_CORE, allow_interruptions=True)
        self.engine = engine
        self.session_id = session_id
        self.state = CallState()
        self.history: list[str] = []
        self.turn_index = 0

        # Per-turn scratch, written by the hook and read by llm_node/tts_node.
        self._pending: dict | None = None
        # Set by the UserStateChanged handler: when the caller stopped speaking.
        self._speech_end_at: float | None = None
        self.turn_log: list[dict] = []

        # Caller-voice isolation (sace_audio). Built once per call rather than
        # per turn: loading the ONNX session costs ~100ms and the enrolled
        # reference never changes mid-call.
        #
        # None when no voice is enrolled, which makes stt_node a pass-through —
        # a checkout with no enrolment behaves exactly as before, and the
        # feature is opt-in via scripts/enroll_voice.py.
        # Built whenever EITHER layer is available, not only when a voice is
        # enrolled. The two toggle independently from the dashboard, and
        # constructing the filter only for the gate would leave the denoise
        # switch dead on a machine with no enrolment — which is the default.
        denoiser = Denoiser()
        gate = SpeakerGate()
        if denoiser.enabled or gate.active:
            self._audio_filter = UtteranceFilter(
                denoiser=denoiser if denoiser.enabled else None,
                gate=gate if gate.active else None,
                denoise_enabled=AUDIO_DENOISE_DEFAULT,
                gate_enabled=AUDIO_GATE_DEFAULT,
            )
        else:
            self._audio_filter = None
        print(f"[audio] denoise={_onoff(denoiser.enabled, AUDIO_DENOISE_DEFAULT)}  "
              f"voice-gate={_onoff(gate.active, AUDIO_GATE_DEFAULT)}"
              + ("" if gate.active else
                 "  (no voice enrolled — python scripts/enroll_voice.py)"))

    # ───────────────────── pre-LLM hook: build SACE context ─────────────────
    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """STT has finalised and the LLM has NOT run yet.

        This is where SACE does its work: semantic intent detection, retrieval of
        the governing + reference rules, and prompt assembly — no completion call.
        """
        user_text = (new_message.text_content or "").strip()
        if not user_text:
            raise StopResponse()  # nothing transcribed; do not bother the LLM

        speech_end = self._speech_end_at or time.perf_counter()
        stt_ms = (time.perf_counter() - speech_end) * 1000

        # build_turn_context does blocking work (a pgvector query and an
        # embedding call), so it must not run on the event loop or it stalls
        # audio for every participant in the room.
        prompt_sent, governing, reference, ctx = await asyncio.to_thread(
            self.engine.build_turn_context, self.state, self.history, user_text
        )

        self.turn_index += 1
        self._pending = {
            "turn_index": self.turn_index,
            "user_text": user_text,
            "system_prompt": ctx["system_prompt"],
            "ctx": ctx,
            # Replaced in llm_node with the true provider payload; this is the
            # engine's own capture, kept as a fallback if llm_node never runs.
            "prompt_sent": prompt_sent,
            "retrieval": ctx["retrieval"],
            "governing": governing,
            "reference": reference,
            "assembled_tokens": ctx["assembled_prompt_tokens"],
            "intent": ctx["intent"],
            "intent_cosine": ctx["intent_similarity"],
            "context_ms": ctx["context_ms"],
            "stt_ms": stt_ms,
            "speech_end_at": speech_end,
            "llm_ttft_ms": None,
            "tts_ttfb_ms": None,
            "terminal": bool(governing and governing.chunk.terminal),
            "notes": list(ctx["notes"]),
            # A cached answer for this turn, if retrieval found one. The reply
            # is served by Engine.prepare_reply (in llm_node) without a
            # completion call; recorded here so the latency line and the
            # dashboard can say so.
            "cache_hit": ctx["retrieval"].cache_hit,
        }

        # Live proof, the moment memory is searched — before the model has even
        # replied. governing/reference here are already _rule_debug-shaped
        # dicts (see engine.build_turn_context), so title/snippet/text need no
        # extra lookup — this is literally what retrieval found.
        gov_debug = ctx["governing"]
        broadcast({
            "type": "retrieval",
            "turn_index": self.turn_index,
            "user_text": user_text,
            "intent": ctx["intent"],
            "intent_cosine": ctx["intent_similarity"],
            "governing_rule_id": gov_debug["id"] if gov_debug else None,
            "governing_rule_title": gov_debug["title"] if gov_debug else None,
            "governing_rule_snippet": gov_debug["snippet"] if gov_debug else None,
            "reference_rule_ids": [r["id"] for r in ctx["reference"]],
            "assembled_tokens": ctx["assembled_prompt_tokens"],
            "monolith_tokens": ctx["monolith_tokens"],
            "context_ms": ctx["context_ms"],
        })

        # The framework builds the system message from `instructions` on each LLM
        # call, so this is the documented injection point. llm_node below also
        # verifies it actually landed, and injects it directly if not.
        #
        # AWAITED: Agent.update_instructions is a coroutine in 1.6.7. Calling it
        # without await raised "coroutine was never awaited" and left the
        # instructions un-updated — silently, because an un-awaited coroutine
        # does not propagate an error. _ensure_sace_system in llm_node was
        # covering for it, which is why the prompt still landed and nothing
        # looked broken.
        await self.update_instructions(ctx["system_prompt"])

        # ── barge-in policy ──
        # Terminal rules (dnc, abuse, elsewhere, busy, the closings) require their
        # line to be delivered in full. Let the caller interrupt it and the
        # opt-out confirmation gets truncated, which is the one thing this script
        # cannot get wrong. So for those turns only, take control of the reply and
        # schedule it uninterruptible.
        if self._pending["terminal"]:
            gov_id = governing.chunk.id
            self._pending["notes"].append(
                f"barge-in SUPPRESSED: {gov_id} is terminal, reply must complete"
            )
            print(f"  [barge-in] suppressed for terminal rule {gov_id}")
            asyncio.create_task(self._speak_uninterruptible())
            raise StopResponse()  # we generate it ourselves, below

    async def _speak_uninterruptible(self):
        session = getattr(self, "session", None)
        if session is None:
            print("  [barge-in] no session handle; falling back to interruptible reply")
            return
        handle = session.generate_reply(allow_interruptions=False)
        try:
            await handle
        except Exception as exc:  # pragma: no cover
            print(f"  [barge-in] terminal reply failed: {type(exc).__name__}: {exc}")

    # ────────────────── STT node: denoise + reject other voices ─────────────
    async def stt_node(self, audio, model_settings):
        """Drop anyone who is not the enrolled caller, before STT ever sees it.

        This is the replacement for LiveKit's BVC, which was removed: BVC ran on
        LiveKit's servers, so it did nothing in console mode and tied the
        feature to a paid service. Everything here is local and open source —
        see the sace_audio package, which knows nothing about LiveKit and is the
        piece that survives replacing it. The glue below is the only
        framework-specific part.

        Silero VAD upstream still answers "is anyone speaking". This answers the
        question VAD cannot: "is it the CALLER?"

        Inert unless a voice has been enrolled (scripts/enroll_voice.py) — with
        no enrolment the filter forwards everything, so a fresh checkout behaves
        exactly as it did before.
        """
        if self._audio_filter is None:
            async for frame in audio:
                yield frame
            return

        async def filtered():
            async for frame in audio:
                # LiveKit hands over int16 PCM; sace_audio works in float32,
                # which is what both noisereduce and the ONNX model expect.
                pcm = np.frombuffer(frame.data, dtype=np.int16)
                mono = (pcm.astype(np.float32) / 32768.0)
                for out in self._audio_filter.feed(mono, frame.sample_rate):
                    yield _to_frame(out, frame.sample_rate, frame.num_channels)
            for out in self._audio_filter.drain():
                yield _to_frame(out, 16000, 1)

        async for ev in Agent.default.stt_node(self, filtered(), model_settings):
            yield ev

    # ───────────────────── LLM node: inject + capture + time ────────────────
    async def llm_node(self, chat_ctx: llm.ChatContext, tools, model_settings):
        pending = self._pending
        if pending is not None:
            chat_ctx = self._ensure_sace_system(chat_ctx, pending["system_prompt"])
            pending["llm_started_at"] = time.perf_counter()
            reply, decision = await asyncio.to_thread(
                self.engine.prepare_reply,
                self.state,
                self.history,
                pending["user_text"],
                pending["ctx"],
            )
            pending["llm_ttft_ms"] = (
                time.perf_counter() - pending["llm_started_at"]
            ) * 1000
            pending["prompt_sent"] = decision["prompt_sent"]
            pending["validation"] = decision
            pending["terminal"] = bool(decision["call_should_end"])
            pending["notes"].extend(decision["notes"])
            pending["assembled_tokens"] = decision["assembled_prompt_tokens"]
            pending["intent"] = decision["intent"]
            pending["intent_cosine"] = decision["intent_similarity"]
            yield reply
            return

        # NO PENDING TURN. The only legitimate reason to be here is the headless
        # harness, which drives _delegate_llm on purpose. In a live call it means
        # something generated a reply without going through
        # on_user_turn_completed — preemptive generation is the known cause (see
        # the PreemptiveGenerationOptions note where the session is built) — and
        # the reply about to be spoken has had NO governing rule, no assembled
        # prompt, no grounding check and no cache involvement.
        #
        # Shouted about rather than logged quietly: this failure is invisible
        # from the transcript (the replies still sound right) and cost several
        # debugging rounds once. The dashboard showed "full pipeline · grounded"
        # for turns that never touched the pipeline.
        print("  [pipeline] WARNING: llm_node reached with no pending turn — this "
              "reply is being generated OFF-PIPELINE (no governing rule, no "
              "grounding check, no answer cache). Check that preemptive "
              "generation is disabled.")
        async for chunk in self._delegate_llm(chat_ctx, tools, model_settings):
            yield chunk

    def _delegate_llm(self, chat_ctx, tools, model_settings):
        """The framework's own LLM step. Isolated as a seam so the headless
        harness (tests/test_voice_path.py) can drive the real injection, capture
        and timing code above without a live AgentSession to resolve the LLM
        from."""
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    async def tts_node(self, text, model_settings):
        pending = self._pending
        first = True
        async for frame in Agent.default.tts_node(self, text, model_settings):
            if first and pending is not None:
                pending["tts_ttfb_ms"] = (
                    time.perf_counter() - pending["speech_end_at"]
                ) * 1000
                first = False
            yield frame

    # ─────────────────────────── helpers ────────────────────────────────────
    @staticmethod
    def _ensure_sace_system(chat_ctx: llm.ChatContext, system_prompt: str) -> llm.ChatContext:
        """Guarantee the SACE prompt is the system message the model receives.

        update_instructions() is the framework's mechanism and normally does this
        already; this makes it explicit and independent of that behaviour, so the
        captured payload is always the SACE prompt rather than whatever the
        session was constructed with.
        """
        out = chat_ctx.copy(exclude_instructions=True)
        head = llm.ChatContext.empty()
        head.add_message(role="system", content=system_prompt)
        out.items.insert(0, head.items[0])
        return out

    @staticmethod
    def _render(chat_ctx: llm.ChatContext) -> str:
        """The provider payload as one verbatim string, mirroring
        llm.render_messages in the chat path so the dashboard renders both the
        same way."""
        try:
            messages, _ = chat_ctx.to_provider_format("openai")
        except Exception:
            messages = [
                {"role": getattr(i, "role", "?"), "content": getattr(i, "text_content", "")}
                for i in chat_ctx.items
            ]
        parts = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            parts.append(f"=== {str(m.get('role', '?')).upper()} ===\n{content}")
        return "\n\n".join(parts)

    # ───────────────────── post-reply: validate + persist ───────────────────
    def finish_turn(self, reply_text: str):
        """Called once the agent's reply is complete. Validation is LOG-ONLY."""
        pending, self._pending = self._pending, None
        if pending is None or not reply_text.strip():
            return None

        verdict = pending.get("validation") or self.engine.validate_reply(
            reply_text, pending["retrieval"]
        )

        # Advance the same state the chat path would, minus the regeneration.
        self.state.intent = pending["intent"] or "none"
        self.state.opt_out = self.state.opt_out or pending["retrieval"].opt_out
        self.state.ended = self.state.ended or bool(verdict.get("call_should_end") or pending["terminal"])
        self.state.collected_fields.update(verdict.get("extracted_fields", {}))
        from sace_chat.engine import _sync_renewal_state

        _sync_renewal_state(self.state, pending["governing"], reply_text)
        asked = verdict.get("asked_question")
        if asked and question_key(asked) not in {question_key(q) for q in self.state.asked_questions}:
            self.state.asked_questions.append(asked)
        self.history.append(f"Caller: {pending['user_text']}")
        self.history.append(f"Maya: {reply_text}")

        total_ms = (time.perf_counter() - pending["speech_end_at"]) * 1000
        gov = pending["governing"]
        refs = [r.chunk.id for r in pending["reference"]]

        record_turn(
            session_id=self.session_id,
            turn_index=pending["turn_index"],
            source="voice",
            user_text=pending["user_text"],
            reply_text=reply_text,
            prompt_sent=pending["prompt_sent"],
            governing_rule_id=gov.chunk.id if gov else None,
            reference_rule_ids=refs,
            intent=pending["intent"],
            intent_cosine=pending["intent_cosine"],
            grounding_cosine=verdict["governing_cosine"],
            validation_outcome=verdict["outcome"],
            assembled_tokens=pending["assembled_tokens"],
            latency_ms=total_ms,
            stt_ms=pending["stt_ms"],
            context_ms=pending["context_ms"],
            llm_ttft_ms=pending["llm_ttft_ms"],
            tts_ttfb_ms=pending["tts_ttfb_ms"],
        )

        print(
            f"[turn {pending['turn_index']:>2}] "
            f"intent={str(pending['intent'] or 'none'):<16} "
            f"gov={(gov.chunk.id if gov else '-'):<26} "
            f"tok={pending['assembled_tokens']:>5} "
            f"cos={verdict['governing_cosine']:.3f} "
            f"{('CACHED' if pending.get('cache_hit') else verdict['outcome']):<11} | "
            f"stt {_fmt_ms(pending['stt_ms'])}ms  "
            f"ctx {_fmt_ms(pending['context_ms'])}ms  "
            f"ttft {_fmt_ms(pending['llm_ttft_ms'])}ms  "
            f"ttfb {_fmt_ms(pending['tts_ttfb_ms'])}ms  "
            f"= {total_ms:5.0f}ms"
            + (f"  ⚡ cache hit {pending['cache_hit']['similarity']:.3f}"
               if pending.get("cache_hit") else "")
            + (f"  💾 saved {verdict['cache_stored']['id']}"
               if (verdict.get("cache_stored") or {}).get("stored") else "")
            # And why NOT, when it was not. The cache filling up slowly looks
            # identical to the cache being broken unless the refusal says which
            # gate turned the turn away.
            + (f"  ⃠ not saved: {verdict['cache_stored']['reason']}"
               if (verdict.get("cache_stored") is not None
                   and not verdict["cache_stored"].get("stored")) else "")
        )
        for note in pending["notes"]:
            print(f"           note: {note}")

        row = {**{k: pending[k] for k in ("turn_index", "user_text", "intent",
                                         "assembled_tokens", "stt_ms", "context_ms",
                                         "llm_ttft_ms", "tts_ttfb_ms")},
               "reply_text": reply_text, "outcome": verdict["outcome"],
               "governing": gov.chunk.id if gov else None,
               "grounding_cosine": verdict["governing_cosine"],
               "latency_ms": total_ms, "prompt_sent": pending["prompt_sent"],
               "cache_hit": bool(pending.get("cache_hit")),
               "cache_similarity": (pending["cache_hit"]["similarity"]
                                    if pending.get("cache_hit") else None),
               # Whether this turn was SAVED for future reuse, and if not, why.
               # Set by Engine._maybe_cache on the verdict prepare_reply
               # returned; absent on a turn that never reached it.
               "cache_stored": verdict.get("cache_stored")}
        # The hand-off packet — extra fields on this SAME turn event, not a
        # new event type (see sace_chat/transfer.py's docstring): the
        # frontend ignores fields it doesn't know, so this renders nothing
        # today and needs no frontend change to surface later.
        if gov is not None and gov.chunk.id == "warm_transfer":
            from sace_chat.transfer import build_transfer_packet

            row["transfer_packet"] = build_transfer_packet(self.state)
        self.turn_log.append(row)

        # Completes the story the "retrieval" event started: the reply that
        # was actually spoken, and whether it stuck to the section retrieval
        # found (outcome/grounding_cosine — same verdict the terminal line
        # above prints, live rather than tailed from a log).
        broadcast({"type": "turn", **row})
        return row

    # ───────────────────────── post-call learning ───────────────────────────
    def run_learning(self):
        """The SAME between-calls loop the chat app runs. Never on the hot path —
        this is called after the session closes."""
        # Surfaced once per call: a gate rejecting everything is otherwise
        # invisible until someone notices the agent stopped responding, and the
        # counters make a mis-set threshold obvious immediately.
        if self._audio_filter is not None and self._audio_filter.gate is not None:
            st = self._audio_filter.gate.stats()
            print(f"[audio] speaker gate: {st['passed']} accepted, "
                  f"{st['rejected']} rejected, {st['skipped_short']} too short "
                  f"(threshold {st['threshold']})")
            if st["passed"] == 0 and st["rejected"] > 0:
                print("[audio] WARNING: every utterance was rejected. The "
                      "enrolment likely does not match this microphone — "
                      "re-run scripts/enroll_voice.py")

        transcript = "\n".join(self.history)
        if not transcript.strip():
            print("[learn] empty transcript; nothing to consolidate")
            broadcast({"type": "learning_done", "results": []})
            return []

        from sace_chat.consolidator import run_learning_loop
        from sace_chat.db import engine as db_engine
        from sace_chat.llm import get_llm

        results = []
        try:
            with db_engine.connect() as conn:
                gate_results = run_learning_loop(
                    transcript, self.engine.embedder, conn, llm=get_llm(),
                    session_id=self.session_id,
                )
            for r in gate_results:
                entry = {
                    "outcome": r.outcome, "detail": r.detail,
                    "text": r.candidate.text, "intent": r.candidate.intent,
                    "learned_kind": r.candidate.learned_kind,
                    # Which queue row a human will act on. Nothing here has
                    # entered the pool — see consolidator.run_learning_loop.
                    "review_id": r.review_id,
                }
                results.append(entry)
                print(f"[learn] {r.outcome:<22} intent={str(r.candidate.intent):<18} {r.detail}")
                print(f"        {r.candidate.text[:110]}")
                # Streamed as each candidate is gated, not batched at the end —
                # "what did it learn, under which intent, stored where" as it
                # actually happens. `detail` already carries the DB proof
                # (e.g. "id=learned_408c8991" on insert; the cosine + rule id
                # it matched/conflicted against otherwise) — see GateResult.
                broadcast({"type": "learned", **entry})
        except Exception as exc:
            print(f"[learn] failed: {type(exc).__name__}: {exc}")

        record_call_transcript(
            session_id=self.session_id, source="voice", transcript=transcript,
            turn_count=self.turn_index, learning_results=results,
        )
        print(f"[learn] transcript stored ({self.turn_index} turns, {len(results)} candidates)")

        # How many rules are now waiting on a person — including any left over
        # from earlier calls, which is why it is read from the queue rather
        # than counted from `results`. Lets the dashboard show a live pending
        # badge without polling the HTTP API while a call is in progress.
        try:
            from sace_chat.review import pending_count

            queued = pending_count()
            print(f"[learn] {queued} rule(s) awaiting human approval")
        except Exception as exc:
            print(f"[learn] pending_count failed: {type(exc).__name__}: {exc}")
            queued = None

        # Signals the stream of "learned" events is complete — without this
        # the dashboard has no way to tell "still reviewing" apart from
        # "reviewed, and there was nothing to learn."
        broadcast({"type": "learning_done", "results": results, "pending_count": queued})
        return results


def build_engine() -> Engine:
    """One Engine per worker process. The KB embedder is the OpenAI 1536-dim one
    because it must match the pgvector column; IntentRouter separately picks up
    the hot-path embedder (EMBED_HOTPATH=local|openai) for exemplar matching
    only. See embeddings.get_hotpath_embedder."""
    from sace_chat.llm import get_llm

    init_db()
    engine = Engine(
        stable_core=STABLE_CORE, rules=RULES, embedder=get_embedder(),
        manager=manager, llm=get_llm(), table=CAMPAIGN.chunks_table,
        never_say_guard=CAMPAIGN.never_say_guard,
        never_say_fallback=CAMPAIGN.never_say_fallback,
        cache_table=CAMPAIGN.cache_table,
        t4_shortcircuit=CAMPAIGN.t4_shortcircuit,
        intent_exemplars=CAMPAIGN.intent_exemplars,
    )
    # Exemplars embedded once here, at startup, so no turn pays for them.
    engine.router.warm()
    print(f"[boot] campaign={CAMPAIGN.name} engine ready · {len(RULES)} seed rules · "
          f"hot-path embedder shares KB model: {engine.router.shares_kb_embedder}")
    return engine


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=VAD_MIN_SILENCE,
        activation_threshold=VAD_ACTIVATION,
        min_speech_duration=VAD_MIN_SPEECH,
    )
    proc.userdata["engine"] = build_engine()


async def entrypoint(ctx: JobContext):
    # Started once per worker process, not per call — a second call in the
    # same process reuses the already-listening server. asyncio.create_task
    # rather than awaiting it: entrypoint must not block on the dashboard
    # ever being watched.
    if not getattr(entrypoint, "_ws_started", False):
        entrypoint._ws_started = True
        asyncio.create_task(_start_ws_server())

    # The job's context is remembered so the dashboard's "Start call" can build
    # a FRESH session in this same process after one ends. A LiveKit job runs
    # entrypoint exactly once, so without this the only way to place a second
    # call is to restart the worker — which is what the button exists to avoid.
    global _JOB_CTX
    _JOB_CTX = ctx

    await run_call(ctx)

    # entrypoint must NOT return here. run_call returns as soon as the opening
    # line has been spoken — the call itself continues on the session's own
    # tasks — and a LiveKit entrypoint returning is what ends the job and tears
    # the process down. Before the dashboard could start calls that was
    # harmless, because the process was meant to end with the one call it was
    # given. Now it is not: returning would take the ws server down with it and
    # "Start call" would have nothing left listening.
    #
    # So park here for the life of the job. ctx.shutdown() / the worker's own
    # signal handling still end the process; this only stops it ending early.
    await asyncio.Event().wait()


async def run_call(ctx: JobContext):
    """One call, start to finish. Separated from `entrypoint` so a second call
    can be started in the same process (see _ws_handler's "start_call")."""
    session_id = f"{ctx.room.name}:{uuid.uuid4().hex[:8]}"
    broadcast({"type": "call_started", "session_id": session_id})
    engine = ctx.proc.userdata.get("engine") or build_engine()
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load(
        min_silence_duration=VAD_MIN_SILENCE,
        activation_threshold=VAD_ACTIVATION,
        min_speech_duration=VAD_MIN_SPEECH,
    )

    agent = SaceVoiceAgent(engine, session_id)
    session = AgentSession(
        stt=deepgram.STT(
            model=STT_MODEL,
            interim_results=True,       # partials stream; nothing waits for the final
            endpointing_ms=ENDPOINTING_MS,
            utterance_end_ms=UTTERANCE_END_MS,
            punctuate=True,
            filler_words=True,
        ),
        llm=openai.LLM(
            model=LLM_MODEL,
            api_key=os.environ.get("SACE_LLM_KEY"),
            base_url=os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"),
        ),
        tts=deepgram.TTS(model=TTS_MODEL),
        vad=vad,
        # Barge-in on by default. The session-level `allow_interruptions=` kwarg
        # is deprecated in 1.6.7 (removed in v2.0) in favour of this; the
        # per-reply `allow_interruptions=False` on generate_reply() is NOT
        # deprecated and is what suppresses interruption on terminal rules.
        turn_handling=TurnHandlingOptions(
            interruption=InterruptionOptions(enabled=True),
            # PREEMPTIVE GENERATION MUST STAY OFF, and this is not a tuning
            # choice — it is load-bearing for correctness.
            #
            # It defaults to enabled=True in livekit-agents 1.6.7. When it
            # fires, AgentActivity.on_preemptive_generation calls _generate_reply
            # directly on the first partial transcript, BYPASSING
            # on_user_turn_completed entirely. That is where this agent does all
            # of its work: SACE retrieval, prompt assembly, and setting
            # self._pending. So llm_node then runs with _pending still None,
            # falls through to _delegate_llm, and the FRAMEWORK's raw LLM answers
            # the caller — with no governing rule, no assembled prompt, no
            # grounding check, no answer-cache lookup or store.
            #
            # It fails quietly, which is what makes it dangerous: replies still
            # sound plausible because the instructions are on the agent, so the
            # only symptoms are `ttft —ms` and a null cache field in the turn
            # row. Observed live: every turn of a call answered off-pipeline
            # while the dashboard showed "full pipeline · grounded".
            #
            # Re-enabling it would need llm_node to be able to rebuild the whole
            # turn context itself, which defeats the point of preempting.
            preemptive_generation=PreemptiveGenerationOptions(enabled=False),
        ),
    )

    global _ACTIVE_SESSION, _ACTIVE_AGENT
    _ACTIVE_SESSION = session
    _ACTIVE_AGENT = agent

    @session.on("user_state_changed")
    def _on_user_state(ev):
        # End of speech is the clock start for the latency budget the caller
        # actually experiences: last word spoken -> first audio heard.
        if str(getattr(ev, "old_state", "")) == "speaking":
            agent._speech_end_at = time.perf_counter()

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = getattr(ev, "item", None)
        if item is None or getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None) or ""
        row = agent.finish_turn(text)
        if row and agent.state.ended:
            print("[call] terminal rule delivered; closing session")
            asyncio.create_task(_close(session))

    @session.on("error")
    def _on_error(ev):
        print(f"[error] {getattr(ev, 'error', ev)}")

    # The learning loop runs exactly once, on shutdown, off the hot path. A guard
    # rather than a plain handler because 'close' and the shutdown callback can
    # both fire and consolidating a transcript twice would insert duplicates.
    learned_once = {"done": False}

    async def _consolidate(*_args):
        if learned_once["done"]:
            return
        learned_once["done"] = True
        global _ACTIVE_SESSION, _ACTIVE_AGENT
        _ACTIVE_SESSION = None
        _ACTIVE_AGENT = None
        broadcast({"type": "call_ended", "session_id": session_id})
        await asyncio.to_thread(agent.run_learning)

    # Registered per call. `learned_once` is per call too, so a second call
    # gets its own guard and its own transcript rather than inheriting the
    # first call's "already done" flag.
    ctx.add_shutdown_callback(_consolidate)

    @session.on("close")
    def _on_close(ev):
        print(f"[call] session closed: {getattr(ev, 'reason', '')}")
        asyncio.create_task(_consolidate())

    await session.start(agent=agent, room=ctx.room)
    print(f"[call] session {session_id} started in room {ctx.room.name}")
    # Push the audio state now the agent exists, so the dashboard's toggles show
    # what is actually running rather than staying blank until someone clicks.
    broadcast(audio_state())

    # Maya opens the call — retrieval has nothing to route on until the caller
    # speaks, so the opening line comes from the KB's own opening rule.
    opening = next((r for r in RULES if r.id == "open_greeting"), None)
    if opening is not None:
        from sace_chat.assemble import _substitute_placeholders
        import re as _re
        quoted = _re.findall(r'"([^"]{40,})"', _substitute_placeholders(opening.text))
        if quoted:
            opening_text = quoted[0]
            await session.say(opening_text, allow_interruptions=True)
            agent.history.append(f"Maya: {opening_text}")
            agent.state.asked_questions.append(opening_text)


async def _close(session: AgentSession):
    await asyncio.sleep(0.2)
    try:
        await session.aclose()
    except Exception as exc:
        print(f"[call] close failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=AGENT_NAME,
    ))
