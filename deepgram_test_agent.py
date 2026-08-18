"""Bare-bones LiveKit voice agent for testing Indian regional languages —
no SACE, no retrieval, no memory, no DB.

Pipeline: Deepgram STT -> prompt.txt (system prompt, sent verbatim) -> LLM -> ElevenLabs TTS

Deepgram (STT) transcribes fine for hi/ta, but its own TTS (Aura) is English-voice
only — no Hindi/Tamil Aura voice exists. So STT stays on Deepgram and TTS moves to
ElevenLabs, which does support Hindi/Tamil output voices.

Deepgram STT language support (livekit-plugins-deepgram 1.6.7):
  "language=" understands: hi (Hindi), ta (Tamil), en-IN, hi-Latn (romanized
  Hindi), or "multi" for code-switching (nova-3 / nova-2 only). No other Indian
  regional language (Telugu, Kannada, Bengali, Marathi, Malayalam, ...) is
  supported by Deepgram at all as of this writing.

ElevenLabs TTS needs a multilingual model to speak non-English text — use
"eleven_multilingual_v2" or "eleven_turbo_v2_5" (both multilingual) and pick a
voice_id that sounds good in the target language (see elevenlabs.io/app for IDs).

Edit DEEPGRAM_STT_LANGUAGE below (or set env DEEPGRAM_STT_LANGUAGE) to try:
  "hi"      Hindi
  "ta"      Tamil
  "en-IN"   Indian English
  "multi"   code-switched, requires model nova-3 or nova-2

Run:  python deepgram_test_agent.py console                  (local mic/speaker, no LiveKit room)
      python deepgram_test_agent.py console --record         (same, plus save the call)
      python deepgram_test_agent.py dev                       (connects a worker to LIVEKIT_URL)

"--record" is LiveKit's OWN built-in flag (livekit.agents.cli.AgentsConsole /
RecorderIO) — not something this script implements. It mixes the caller's mic and
the agent's TTS into a single properly-muxed audio.ogg via ffmpeg, one full
conversation, no manual PCM offset math involved. Output goes to
console-recordings/session-<timestamp>/audio.ogg by default; this script writes
transcript.json/transcript.txt into that same directory.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.cli import AgentsConsole
from livekit.plugins import deepgram, elevenlabs, openai, silero

# ─────────────────────────────── config ───────────────────────────────
STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-2")
STT_LANGUAGE = os.environ.get("DEEPGRAM_STT_LANGUAGE", "hi")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_LANGUAGE = os.environ.get("ELEVENLABS_LANGUAGE") or None
LLM_MODEL = os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")
AGENT_NAME = os.environ.get("LIVEKIT_AGENT_NAME", "deepgram-lang-test")

PROMPT_FILE = os.environ.get("PROMPT_FILE", os.path.join(os.path.dirname(__file__), "prompt.txt"))


def load_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    # Drop the instructional first line so it never leaks into the model's context.
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("Paste your system prompt"):
        lines = lines[1:]
    prompt = "\n".join(lines).strip()
    if not prompt:
        raise RuntimeError(f"{PROMPT_FILE} is empty — paste a system prompt into it first.")
    return prompt


def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    system_prompt = load_prompt()
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    # AgentsConsole.record is set from the --record CLI flag; session_directory is
    # wherever the built-in RecorderIO is already writing audio.ogg (and
    # session_report.json), so the transcript lands right next to it.
    console = AgentsConsole.get_instance()
    recording = console.enabled and console.record
    transcript: list[dict] = []

    agent = Agent(instructions=system_prompt, allow_interruptions=True)
    session = AgentSession(
        stt=deepgram.STT(
            model=STT_MODEL,
            language=STT_LANGUAGE,
            interim_results=True,
            punctuate=True,
        ),
        llm=openai.LLM(
            model=LLM_MODEL,
            api_key=os.environ.get("SACE_LLM_KEY"),
            base_url=os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"),
        ),
        tts=elevenlabs.TTS(
            voice_id=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
            **({"language": ELEVENLABS_LANGUAGE} if ELEVENLABS_LANGUAGE else {}),
        ),
        vad=vad,
    )

    @session.on("conversation_item_added")
    def _on_item(ev):
        item = getattr(ev, "item", None)
        if item is None:
            return
        role = getattr(item, "role", "?")
        text = getattr(item, "text_content", None) or ""
        print(f"[{role}] {text}")
        if recording:
            transcript.append({"role": role, "text": text})

    async def _save_transcript(*_args):
        if not recording:
            return
        call_dir = ctx.session_directory
        with open(call_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        with open(call_dir / "transcript.txt", "w", encoding="utf-8") as f:
            for turn in transcript:
                f.write(f"{turn['role']}: {turn['text']}\n")
        print(f"[record] transcript saved to {call_dir}/transcript.txt "
              f"(audio: {call_dir}/audio.ogg)")

    if recording:
        ctx.add_shutdown_callback(_save_transcript)

    await session.start(agent=agent, room=ctx.room)
    print(f"[call] session started · stt=deepgram/{STT_MODEL}/{STT_LANGUAGE} "
          f"tts=elevenlabs/{ELEVENLABS_MODEL}/{ELEVENLABS_VOICE_ID}")
    print(f"[call] system prompt loaded from {PROMPT_FILE} ({len(system_prompt)} chars)")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=AGENT_NAME,
    ))
