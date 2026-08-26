"""Headless verification of the voice path.

WHAT THIS DOES AND DOES NOT COVER — read this before trusting the output.

Covered for real, against the live OpenAI LLM and the real Postgres pool:
  * SaceVoiceAgent.on_user_turn_completed  (SACE retrieval + assembly, no LLM)
  * SaceVoiceAgent.llm_node                (system-prompt injection, exact
                                            payload capture, TTFT timing)
  * SaceVoiceAgent.finish_turn             (log-only validation, turns row)
  * SaceVoiceAgent.run_learning            (extraction + three gates + insert)
  * terminal barge-in suppression decision and its logging

NOT covered — needs LiveKit + Deepgram credentials and a microphone, neither of
which exists in this environment:
  * Deepgram STT/TTS themselves, so stt_ms and tts_ttfb_ms are always None here
  * real end-of-speech-to-first-audio latency
  * WebRTC transport and actual barge-in behaviour

The one substitution is Agent._delegate_llm: without an AgentSession there is no
activity for Agent.default.llm_node to resolve an LLM from, so the harness calls
the same OpenAI model directly with the SAME chat_ctx the production path built.
Everything the SACE work depends on is production code.

Run:  python tests/test_voice_path.py
"""

import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_env

kb_env.pin()

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import StopResponse, llm
from sqlalchemy import text as sql_text

import voice_agent as VA
from sace_chat.assemble import DEMO_PLACEHOLDERS
from sace_chat.db import engine as db_engine

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


class HarnessAgent(VA.SaceVoiceAgent):
    """Production agent with only the framework's LLM plumbing replaced."""

    def __init__(self, engine, session_id, model=None):
        super().__init__(engine, session_id)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=os.environ["SACE_LLM_KEY"],
            base_url=os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"),
        )
        self._model = model or os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")

    async def _delegate_llm(self, chat_ctx, tools, model_settings):
        messages, _ = chat_ctx.to_provider_format("openai")
        stream = await self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.2, stream=True
        )
        async for part in stream:
            delta = part.choices[0].delta.content if part.choices else None
            if delta:
                yield delta


async def drive_turn(agent, user_text):
    """One turn, exercising the same hooks the framework would call in order."""
    ctx = llm.ChatContext.empty()
    for line in agent.history[-6:]:
        role = "user" if line.startswith("Caller:") else "assistant"
        ctx.add_message(role=role, content=line.split(":", 1)[1].strip())
    msg = ctx.add_message(role="user", content=user_text)

    suppressed = False
    try:
        await agent.on_user_turn_completed(ctx, msg)
    except StopResponse:
        # Terminal rule: the agent took control to deliver it uninterruptible.
        suppressed = True

    pending = agent._pending
    started = time.perf_counter()
    parts = []
    async for chunk in agent.llm_node(ctx, [], None):
        parts.append(chunk if isinstance(chunk, str) else getattr(chunk, "delta", "") or "")
    reply = "".join(parts).strip()
    # No TTS in the harness, so stand in for the audio clock with reply completion.
    if pending is not None and pending.get("tts_ttfb_ms") is None:
        pending["tts_ttfb_ms"] = None
    row = agent.finish_turn(reply)
    if row:
        row["_suppressed"] = suppressed
        row["_wall_ms"] = (time.perf_counter() - started) * 1000
    return row


def rule_text(engine, rule_id):
    return next((r.text for r in engine.rules if r.id == rule_id), None)


async def main():
    print("=" * 74)
    print("VOICE PATH — headless verification")
    print("=" * 74)
    engine = VA.build_engine()
    session_id = f"harness:{uuid.uuid4().hex[:8]}"

    # ── scenario A: normal flow, three turns ────────────────────────────────
    print("\n[A] three spoken turns (transcripts simulated, everything else real)")
    agent = HarnessAgent(engine, session_id)
    rows = []
    for text in ["hello? who's this", "yeah sure, I've got a minute", "yep that's me"]:
        rows.append(await drive_turn(agent, text))

    record("A1. every turn produced a turns row", all(rows), f"{len(rows)} rows")
    record(
        "A2. context build stays well under the latency budget",
        all(r["context_ms"] < 900 for r in rows),
        "context_ms per turn: " + ", ".join(f"{r['context_ms']:.0f}" for r in rows),
    )

    # ── VERIFY 3: prompt_sent contains the transcript AND the governing rule ─
    print("\n[3] prompt_sent is the exact payload — asserted programmatically")
    for r in rows:
        prompt = r["prompt_sent"]
        gov_text = rule_text(engine, r["governing"]) or ""
        # The prompt has demo placeholders substituted, so compare on a span of
        # the rule that carries none.
        spans = [s for s in gov_text.split(". ") if "{" not in s and len(s) > 40]
        span_ok = (spans[0] in prompt) if spans else True
        record(
            f"3.{r['turn_index']} turn {r['turn_index']} prompt contains the transcript verbatim",
            r["user_text"] in prompt,
            f"user_text={r['user_text']!r}",
        )
        record(
            f"3.{r['turn_index']}b turn {r['turn_index']} prompt contains the governing rule text",
            span_ok and r["governing"] in prompt,
            f"governing={r['governing']}",
        )

    # ── VERIFY 2: spoken DNC ────────────────────────────────────────────────
    print("\n[2] 'stop bothering me, I never want these calls again'")
    dnc_agent = HarnessAgent(engine, f"{session_id}-dnc")
    await drive_turn(dnc_agent, "hi there")
    d = await drive_turn(dnc_agent, "stop bothering me, I never want these calls again")

    record("2a. intent classified dnc", d["intent"] == "dnc", f"intent={d['intent']}")
    record("2b. governing rule is special_dnc", d["governing"] == "special_dnc",
           f"governing={d['governing']}")
    record("2c. nothing else in scope", not d["prompt_sent"].count("[benefits_check]"),
           "REFERENCE suppressed (special_dnc is exclusive)")
    handoff = [w for w in ("KEEP", "counselor", "1-800-555-0100") if w in d["reply_text"]]
    record("2d. no counselor/KEEP content spliced into the close", not handoff,
           f"reply={d['reply_text']}")
    record("2e. barge-in suppressed for the terminal rule", d["_suppressed"],
           "on_user_turn_completed raised StopResponse to deliver it uninterruptible")
    record("2f. call ended", dnc_agent.state.ended, f"ended={dnc_agent.state.ended}")
    record("2g. validation grounded", d["outcome"] == "grounded",
           f"outcome={d['outcome']} cos={d['grounding_cosine']:.3f}")

    # ── VERIFY 4: learning loop on call end ─────────────────────────────────
    print("\n[4] ending the call runs the learning loop")
    before = _pool_size()
    learn_agent = HarnessAgent(engine, f"{session_id}-learn")
    for text in [
        "hi",
        "yeah go ahead",
        "that's me",
        "actually my daughter handles all my insurance paperwork, call her instead",
    ]:
        await drive_turn(learn_agent, text)
    learned = learn_agent.run_learning()
    after = _pool_size()

    record("4a. learning loop ran and reported gate outcomes", learned is not None,
           f"{len(learned)} candidate(s): " + ", ".join(x["outcome"] for x in learned))
    # Nothing enters the pool from a call any more — clearing the gates queues
    # the candidate for a human instead (see consolidator.run_learning_loop and
    # sace_chat.review). So the check is that the pool did NOT grow, and that
    # anything proposed landed in the review queue with a row id.
    queued = [x for x in learned if x.get("review_id")]
    record("4b. nothing was inserted into the pool without a human",
           after == before, f"pool {before} -> {after}")
    if learned:
        record("4b2. every proposed candidate got a review row",
               len(queued) == len([x for x in learned if x["outcome"] != "duplicate-skipped"]),
               f"outcomes={[x['outcome'] for x in learned]}")
    else:
        record("4b2. every proposed candidate got a review row", True,
               "nothing proposed this run — nothing to queue")
    record("4c. transcript persisted to call_transcripts", _transcript_rows(f"{session_id}-learn") == 1,
           f"pool {before} -> {after} rules")

    # ── VERIFY 5: an APPROVED rule is retrieved in a LATER call ─────────────
    # The human approval queue changed what this can assert. A rule proposed by
    # the call above is NOT live — it is queued, and nothing retrieves it until
    # a person approves it. So the loop is closed here explicitly: queue a
    # candidate, approve it the way the dashboard would, and only then check a
    # later call retrieves it. That is the real end-to-end claim now.
    print("\n[5] a rule a human APPROVED is retrieved in a later voice call")
    from sace_chat import review
    from sace_chat.consolidator import Candidate

    approved_id = None
    try:
        review_id = review.enqueue(
            candidate=Candidate(
                text=(
                    "When the caller asks whether the clinic has a night pharmacy, say: "
                    '"Our pharmacy counter is open until 9pm on weeknights."'
                ),
                learned_kind="policy",
                intent=None,
                source_line="Caller: is the pharmacy open late",
                cue="is the pharmacy open late, night pharmacy hours, "
                    "can I pick up a prescription in the evening",
            ),
            reason="pending",
            session_id=f"{session_id}-approve",
            trigger_message="is the pharmacy open late",
            trigger_reply="I'm not sure about that.",
        )
        outcome = review.approve(review_id, engine.embedder)
        approved_id = outcome["chunk_id"]

        probe = HarnessAgent(engine, f"{session_id}-probe")
        await drive_turn(probe, "hi")
        p = await drive_turn(probe, "quick one — can I collect a prescription in the evening")
        record(
            "5. an approved rule governs a later turn",
            p["governing"] == approved_id,
            f"governing={p['governing']} · approved={approved_id}",
        )
    finally:
        # This is a test fixture, not something a demo should inherit.
        if approved_id:
            with db_engine.begin() as c:
                c.execute(sql_text("DELETE FROM chunks WHERE id = :i"), {"i": approved_id})

    # ── VERIFY 1: latency, to the extent it is measurable here ─────────────
    print("\n[1] latency breakdown per turn (stt/tts absent — no Deepgram in harness)")
    print(f"  {'turn':<5}{'intent':<14}{'governing':<26}{'ctx ms':>8}{'ttft ms':>9}{'total ms':>10}")
    allrows = rows + [d]
    for r in allrows:
        print(f"  {r['turn_index']:<5}{str(r['intent'] or 'none'):<14}"
              f"{str(r['governing']):<26}{r['context_ms']:>8.0f}"
              f"{(r['llm_ttft_ms'] or 0):>9.0f}{r['latency_ms']:>10.0f}")
    ctx_ok = all(r["context_ms"] + (r["llm_ttft_ms"] or 0) < 1500 for r in allrows)
    record("1. context + LLM TTFT within the 1.5s budget (excludes STT/TTS)", ctx_ok,
           "these are the two stages SACE owns; STT and TTS are unmeasured here")

    print("\n" + "=" * 74)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def _pool_size():
    with db_engine.connect() as c:
        return c.execute(sql_text("SELECT count(*) FROM chunks")).scalar()


def _learned_ids():
    with db_engine.connect() as c:
        return {r[0] for r in c.execute(
            sql_text("SELECT id FROM chunks WHERE source='learned'")).all()}


def _last_learned_id():
    with db_engine.connect() as c:
        return c.execute(sql_text(
            "SELECT id FROM chunks WHERE source='learned' ORDER BY id DESC LIMIT 1")).scalar()


def _norms(ids):
    out = []
    with db_engine.connect() as c:
        for i in ids:
            if not i:
                continue
            raw = c.execute(sql_text("SELECT embedding FROM chunks WHERE id=:i"), {"i": i}).scalar()
            if raw is None:
                out.append(0.0)
                continue
            v = [float(x) for x in str(raw).strip("[]").split(",")]
            out.append(round(sum(x * x for x in v) ** 0.5, 6))
    return out


def _transcript_rows(session_id):
    with db_engine.connect() as c:
        return c.execute(sql_text(
            "SELECT count(*) FROM call_transcripts WHERE session_id=:s"), {"s": session_id}).scalar()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
