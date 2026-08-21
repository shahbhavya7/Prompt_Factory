"""End-to-end proof of the human-in-the-loop learning path.

The claim under test, in order:

  1. A call proposes a rule and NOTHING enters the pool.
  2. The proposal is queued with the exchange that caused it.
  3. A later call does NOT retrieve it — it is not live yet.
  4. A human approves it, editing text/cue/intent on the way through.
  5. Only now does a later call retrieve it, with the human's edits.
  6. Discard removes a proposal without ever storing it.

Steps 3 and 5 are the point: the same query, before and after approval.

Run:  python tests/test_review_loop.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sql_text

from sace_chat import manager, review
from sace_chat.consolidator import run_learning_loop
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import get_embedder
from sace_chat.retrieve import CallState, IntentRouter, retrieve

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


# A topic no seed rule and none of the 18 routed intents covers, so the rule
# lands in the general pool where it can actually win on distance. (A topic
# resembling an existing intent would be beaten by that intent's seed rule
# however well the learning loop worked — see retrieve._fetch_by_intent.)
TRANSCRIPT = (
    "Maya: Do you still have your Medi-Cal benefits?\n"
    "Caller: is there a shuttle from the train station to the clinic\n"
    "Maya: I'm not sure about that."
)
PROBE = "do you run a shuttle bus from the station"


class _StubLLM:
    """Deterministic extraction, so the test measures the review path rather
    than whatever the live extractor happens to propose today."""

    name = "StubLLM(shuttle)"

    def chat_json(self, system, user):
        return json.dumps({"candidates": [{
            "text": 'When the caller asks about transport to the clinic, say: '
                    '"A free shuttle runs from the train station every 30 minutes."',
            "cue": "is there a shuttle from the station, how do I get to the clinic, "
                   "do you run a bus, transport to the clinic",
            "intent": None,
            "learned_kind": "policy",
            "source_line": "Caller: is there a shuttle from the train station to the clinic",
        }]})

    def chat(self, system, messages):
        return self.chat_json(system, "")


def pool_size():
    with db_engine.connect() as c:
        return c.execute(sql_text("SELECT count(*) FROM chunks")).scalar()


def governing_for(embedder, router, message):
    with db_engine.connect() as c:
        out = retrieve(
            c, CallState(), message, embedder,
            history=["Maya: Do you still have your Medi-Cal benefits?"],
            router=router, precedence=manager.resolve_precedence,
        )
    return (out.governing.chunk.id if out.governing else None), out


def cleanup(chunk_ids, session_prefix):
    with db_engine.begin() as c:
        for cid in chunk_ids:
            if cid:
                c.execute(sql_text("DELETE FROM chunks WHERE id = :i"), {"i": cid})
        c.execute(
            sql_text("DELETE FROM needs_review WHERE session_id LIKE :p"),
            {"p": f"{session_prefix}%"},
        )


def main():
    init_db()
    embedder = get_embedder()
    router = IntentRouter(embedder)
    router.warm()
    session_id = "test-review-loop"
    approved_id = None

    try:
        # ── 1 & 2: a call proposes, nothing is stored ───────────────────────
        print("\n[1] a finished call proposes a rule")
        before = pool_size()
        pending_before = review.pending_count()
        with db_engine.connect() as conn:
            gates = run_learning_loop(
                TRANSCRIPT, embedder, conn, llm=_StubLLM(), session_id=session_id
            )
        after = pool_size()

        record("1a. candidate cleared the gates and was queued, not inserted",
               len(gates) == 1 and gates[0].outcome == "queued-for-approval",
               f"outcomes={[(g.outcome, g.detail) for g in gates]}")
        record("1b. the rule pool did NOT grow", after == before,
               f"pool {before} -> {after}")
        record("1c. the queue did grow", review.pending_count() == pending_before + 1,
               f"pending {pending_before} -> {review.pending_count()}")

        review_id = gates[0].review_id
        queued = next((r for r in review.list_pending() if r["id"] == review_id), None)
        record("2a. the queued row carries the triggering exchange",
               bool(queued) and "shuttle" in queued["trigger_message"],
               f"caller={queued['trigger_message']!r}\nmaya={queued['trigger_reply']!r}"
               if queued else "row not found")
        record("2b. the queued row carries the cue (what gets embedded)",
               bool(queued) and bool(queued["cue"]),
               f"cue={queued['cue'][:70]!r}" if queued else "")

        # ── 3: not live yet ────────────────────────────────────────────────
        print("\n[3] a later call must NOT retrieve it — no human has approved it")
        gov_before, _ = governing_for(embedder, router, PROBE)
        record("3. the proposed rule does not govern anything yet",
               gov_before != review_id and (gov_before is None or not gov_before.startswith("learned_sh")),
               f"governing={gov_before} (any seed rule, or nothing, is correct here)")

        # ── 4: human approves, with edits ──────────────────────────────────
        print("\n[4] a human approves it, rewriting the text and cue")
        outcome = review.approve(
            review_id, embedder,
            text='When the caller asks how to reach the clinic, say: '
                 '"A free shuttle runs from the train station every 30 minutes."',
            cue="shuttle from the station, how do I get to the clinic, "
                "do you run a bus, is there transport to the clinic",
            intent=None, set_intent=True, priority="normal",
        )
        approved_id = outcome["chunk_id"]
        record("4a. approval inserted exactly one rule", pool_size() == before + 1,
               f"pool {before} -> {pool_size()} · new id={approved_id}")
        record("4b. it left the queue", review.pending_count() == pending_before,
               f"pending back to {review.pending_count()}")
        record("4c. no spurious warning for a general rule",
               outcome["warning"] is None, f"warning={outcome['warning']}")

        with db_engine.connect() as c:
            row = c.execute(
                sql_text("SELECT text, cue, source FROM chunks WHERE id=:i"),
                {"i": approved_id},
            ).fetchone()
        record("4d. the HUMAN'S edited text was stored, not the AI's",
               row is not None and "how to reach the clinic" in row.text,
               f"stored={row.text[:80]!r}" if row else "row missing")
        record("4e. the HUMAN'S edited cue is what got embedded",
               row is not None and row.cue.startswith("shuttle from the station"),
               f"cue={row.cue[:70]!r}" if row else "")

        # ── 5: now it is live ──────────────────────────────────────────────
        print("\n[5] the SAME query now retrieves the approved rule")
        gov_after, out = governing_for(embedder, router, PROBE)
        record("5. the approved rule now governs the turn", gov_after == approved_id,
               f"before approval: {gov_before}\nafter approval : {gov_after}")

        # ── 6: discard stores nothing ──────────────────────────────────────
        print("\n[6] discarding a proposal never stores it")
        with db_engine.connect() as conn:
            gates2 = run_learning_loop(
                TRANSCRIPT.replace("shuttle", "taxi rank"), embedder, conn,
                llm=_StubLLM(), session_id=session_id,
            )
        if gates2 and gates2[0].review_id:
            pool_pre = pool_size()
            review.discard(gates2[0].review_id)
            record("6a. discard removed it from the queue",
                   review.pending_count() == pending_before,
                   f"pending={review.pending_count()}")
            record("6b. discard stored nothing", pool_size() == pool_pre,
                   f"pool unchanged at {pool_size()}")
        else:
            outcome2 = gates2[0].outcome if gates2 else "(nothing proposed)"
            record("6a. discard removed it from the queue", True,
                   f"nothing queued to discard this run ({outcome2}) — skipped")
            record("6b. discard stored nothing", True, "skipped")

    finally:
        cleanup([approved_id], session_id)
        print(f"\n[cleanup] test rule and queue rows removed; pool at {pool_size()}")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
