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
from livekit.plugins import deepgram, openai, silero

from sace_chat import manager
from sace_chat.db import init_db, record_call_transcript, record_turn
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine, assert_message_present, question_key
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.retrieve import CallState

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
# Captured once, when the ws server starts, on the loop that owns _WS_CLIENTS.
# run_learning (and therefore every "learned"/"learning_done" broadcast) runs
# via asyncio.to_thread — a real OS thread with no running loop of its own —
# so broadcast() cannot rely on asyncio.get_running_loop() the way the
# in-loop call sites (retrieval, turn) can; it needs this to schedule onto
# the right loop from any thread.
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


async def _ws_handler(websocket):
    _WS_CLIENTS.add(websocket)
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "end_call" and _ACTIVE_SESSION is not None:
                print("[dashboard] end_call requested from the browser")
                asyncio.create_task(_close(_ACTIVE_SESSION))
    finally:
        _WS_CLIENTS.discard(websocket)


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
        self.update_instructions(ctx["system_prompt"])

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
            f"{verdict['outcome']:<11} | "
            f"stt {_fmt_ms(pending['stt_ms'])}ms  "
            f"ctx {_fmt_ms(pending['context_ms'])}ms  "
            f"ttft {_fmt_ms(pending['llm_ttft_ms'])}ms  "
            f"ttfb {_fmt_ms(pending['tts_ttfb_ms'])}ms  "
            f"= {total_ms:5.0f}ms"
        )
        for note in pending["notes"]:
            print(f"           note: {note}")

        row = {**{k: pending[k] for k in ("turn_index", "user_text", "intent",
                                         "assembled_tokens", "stt_ms", "context_ms",
                                         "llm_ttft_ms", "tts_ttfb_ms")},
               "reply_text": reply_text, "outcome": verdict["outcome"],
               "governing": gov.chunk.id if gov else None,
               "grounding_cosine": verdict["governing_cosine"],
               "latency_ms": total_ms, "prompt_sent": pending["prompt_sent"]}
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
        manager=manager, llm=get_llm(),
    )
    # Exemplars embedded once here, at startup, so no turn pays for them.
    engine.router.warm()
    print(f"[boot] engine ready · {len(RULES)} seed rules · "
          f"hot-path embedder shares KB model: {engine.router.shares_kb_embedder}")
    return engine


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE)
    proc.userdata["engine"] = build_engine()


async def entrypoint(ctx: JobContext):
    # Started once per worker process, not per call — a second call in the
    # same process reuses the already-listening server. asyncio.create_task
    # rather than awaiting it: entrypoint must not block on the dashboard
    # ever being watched.
    if not getattr(entrypoint, "_ws_started", False):
        entrypoint._ws_started = True
        asyncio.create_task(_start_ws_server())

    session_id = f"{ctx.room.name}:{uuid.uuid4().hex[:8]}"
    broadcast({"type": "call_started", "session_id": session_id})
    engine = ctx.proc.userdata.get("engine") or build_engine()
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=VAD_MIN_SILENCE)

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
        turn_handling=TurnHandlingOptions(interruption=InterruptionOptions(enabled=True)),
    )

    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session

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
        global _ACTIVE_SESSION
        _ACTIVE_SESSION = None
        broadcast({"type": "call_ended", "session_id": session_id})
        await asyncio.to_thread(agent.run_learning)

    ctx.add_shutdown_callback(_consolidate)

    @session.on("close")
    def _on_close(ev):
        print(f"[call] session closed: {getattr(ev, 'reason', '')}")
        asyncio.create_task(_consolidate())

    await session.start(agent=agent, room=ctx.room)
    print(f"[call] session {session_id} started in room {ctx.room.name}")

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
