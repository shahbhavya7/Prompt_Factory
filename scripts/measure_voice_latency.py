"""Measure the real per-stage latency of the voice path.

Splits the budget the caller actually experiences — last word spoken to first
audio heard — into its four stages, measuring each against the real services:

    stt_ms   Deepgram finalising a transcript from audio
    ctx_ms   SACE: embedding + pgvector + prompt assembly   (broken down further)
    ttft_ms  first token out of the LLM
    ttfb_ms  first audio frame out of Deepgram TTS

There is no microphone here, so the STT figure is measured by feeding Deepgram
TTS output back into Deepgram STT rather than live speech. That measures the
service round-trip honestly but excludes real capture/network jitter from a
caller's device.

  python scripts/measure_voice_latency.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import aiohttp

from sace_chat import manager
from sace_chat.assemble import build_turn_prompt
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import embed_many, get_embedder
from sace_chat.engine import Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.retrieve import CallState, IntentRouter, pending_question, retrieve

DG_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
TTS_MODEL = os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
STT_MODEL = os.environ.get("DEEPGRAM_STT_MODEL", "nova-3")

PHRASES = [
    "hello, who is this",
    "yeah sure, I've got a couple of minutes",
    "yep that's me",
    "stop bothering me, I never want these calls again",
]


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


# ── stage 2: SACE context, broken into embedding vs pgvector vs assembly ────
def measure_context(engine, rounds=4):
    state, history = CallState(), []
    embed_ms, query_ms, asm_ms, total_ms = [], [], [], []
    emb = engine.embedder
    router = engine.router

    for phrase in PHRASES:
        pending = pending_question(history)
        texts = [phrase, pending] if pending else [phrase]

        t0 = time.perf_counter()
        embed_many(emb, texts)
        t1 = time.perf_counter()

        with db_engine.connect() as conn:
            t2 = time.perf_counter()
            r = retrieve(conn, state, phrase, emb, history=history, router=router,
                         precedence=manager.resolve_precedence)
            t3 = time.perf_counter()

        t4 = time.perf_counter()
        build_turn_prompt(STABLE_CORE, state, r, history)
        t5 = time.perf_counter()

        embed_ms.append((t1 - t0) * 1000)
        # retrieve() re-embeds internally, so subtract the measured embedding cost
        # to isolate the pgvector round-trip.
        query_ms.append(max(0.0, (t3 - t2) * 1000 - (t1 - t0) * 1000))
        asm_ms.append((t5 - t4) * 1000)
        total_ms.append((t3 - t2) * 1000 + (t5 - t4) * 1000)
        history.append(f"Caller: {phrase}")
        history.append("Maya: Do you still have your Medi-Cal benefits?")

    return embed_ms, query_ms, asm_ms, total_ms


# ── stage 3: LLM time to first token ───────────────────────────────────────
async def measure_ttft(engine, n=4):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["SACE_LLM_KEY"],
        base_url=os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"),
    )
    model = os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")
    state, history = CallState(), []
    out = []
    for phrase in PHRASES[:n]:
        _, _, _, ctx = await asyncio.to_thread(
            engine.build_turn_context, state, history, phrase
        )
        messages, _ = [
            {"role": "system", "content": ctx["system_prompt"]},
            {"role": "user", "content": phrase},
        ], None
        t0 = time.perf_counter()
        stream = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, stream=True
        )
        async for part in stream:
            if part.choices and part.choices[0].delta.content:
                out.append((time.perf_counter() - t0) * 1000)
                break
        await stream.close()
    return out


# ── stage 4: Deepgram TTS time to first audio byte ─────────────────────────
async def measure_tts_ttfb(session, texts):
    url = f"https://api.deepgram.com/v1/speak?model={TTS_MODEL}&encoding=linear16&sample_rate=24000"
    headers = {"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"}
    ttfb, audio = [], {}
    for t in texts:
        t0 = time.perf_counter()
        async with session.post(url, headers=headers, json={"text": t}) as resp:
            resp.raise_for_status()
            first = await resp.content.readany()
            ttfb.append((time.perf_counter() - t0) * 1000)
            rest = await resp.read()
            audio[t] = first + rest
    return ttfb, audio


# ── stage 1: Deepgram STT finalisation ─────────────────────────────────────
async def measure_stt(session, audio: dict):
    url = (f"https://api.deepgram.com/v1/listen?model={STT_MODEL}"
           f"&punctuate=true&encoding=linear16&sample_rate=24000")
    headers = {"Authorization": f"Token {DG_KEY}", "Content-Type": "audio/raw"}
    lat, transcripts = [], []
    for text, pcm in audio.items():
        t0 = time.perf_counter()
        async with session.post(url, headers=headers, data=pcm) as resp:
            resp.raise_for_status()
            body = await resp.json()
        lat.append((time.perf_counter() - t0) * 1000)
        alt = body["results"]["channels"][0]["alternatives"][0]
        transcripts.append((text, alt.get("transcript", "")))
    return lat, transcripts


async def main():
    init_db()
    from sace_chat.llm import get_llm

    engine = Engine(stable_core=STABLE_CORE, rules=RULES, embedder=get_embedder(),
                    manager=manager, llm=get_llm())
    engine.router.warm()

    print("=" * 78)
    print("VOICE LATENCY — measured per stage against the real services")
    print("=" * 78)

    print("\n[2] SACE context build")
    emb, qry, asm, ctx_total = measure_context(engine)
    print(f"  embedding (OpenAI, 1 batched call)  median {pct(emb, .5):6.0f} ms   max {max(emb):6.0f} ms")
    print(f"  pgvector query (exact scan)         median {pct(qry, .5):6.0f} ms   max {max(qry):6.0f} ms")
    print(f"  prompt assembly (pure python)       median {pct(asm, .5):6.0f} ms   max {max(asm):6.0f} ms")
    print(f"  -> ctx_ms                           median {pct(ctx_total, .5):6.0f} ms   max {max(ctx_total):6.0f} ms")

    print("\n[3] LLM time to first token")
    ttft = await measure_ttft(engine)
    print(f"  ttft_ms                             median {pct(ttft, .5):6.0f} ms   max {max(ttft):6.0f} ms")

    if not DG_KEY:
        print("\n[1][4] Deepgram: DEEPGRAM_API_KEY not set — skipped")
        return 1

    async with aiohttp.ClientSession() as session:
        print("\n[4] Deepgram TTS time to first audio byte")
        try:
            ttfb, audio = await measure_tts_ttfb(session, PHRASES)
        except Exception as exc:
            print(f"  TTS failed: {type(exc).__name__}: {exc}")
            return 1
        print(f"  ttfb_ms                             median {pct(ttfb, .5):6.0f} ms   max {max(ttfb):6.0f} ms")

        print("\n[1] Deepgram STT (TTS audio fed back in — not a live mic)")
        try:
            stt, transcripts = await measure_stt(session, audio)
        except Exception as exc:
            print(f"  STT failed: {type(exc).__name__}: {exc}")
            return 1
        print(f"  stt_ms                              median {pct(stt, .5):6.0f} ms   max {max(stt):6.0f} ms")
        for spoken, heard in transcripts:
            mark = "ok " if heard.lower().strip(".,") in spoken.lower() or spoken.lower() in heard.lower() else "~~ "
            print(f"    {mark} spoken {spoken!r}\n        heard  {heard!r}")

    print("\n" + "=" * 78)
    print("END-TO-END: last word spoken -> first audio heard")
    print("=" * 78)
    budget = pct(stt, .5) + pct(ctx_total, .5) + pct(ttft, .5) + pct(ttfb, .5)
    worst = max(stt) + max(ctx_total) + max(ttft) + max(ttfb)
    print(f"  median  stt {pct(stt,.5):.0f} + ctx {pct(ctx_total,.5):.0f} + "
          f"ttft {pct(ttft,.5):.0f} + ttfb {pct(ttfb,.5):.0f}  =  {budget:.0f} ms")
    print(f"  worst   stt {max(stt):.0f} + ctx {max(ctx_total):.0f} + "
          f"ttft {max(ttft):.0f} + ttfb {max(ttfb):.0f}  =  {worst:.0f} ms")
    print(f"  budget  1500 ms  ->  median {'PASS' if budget < 1500 else 'FAIL'}, "
          f"worst {'PASS' if worst < 1500 else 'FAIL'}")
    print("\n  NOTE: stages overlap in production (TTS begins on the first sentence\n"
          "  while the LLM is still streaming), so the true figure is lower than\n"
          "  this sum. Summing them is the pessimistic bound.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
