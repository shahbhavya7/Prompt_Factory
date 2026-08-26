"""Free-form verification of the memory-only turn loop.

Every caller line is deliberately not a scripted keyword phrase: retrieval is
semantic all the way down, so "yeah I guess so" has to work without appearing in
any table.

Run:  python tests/test_freeform_turns.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_env

kb_env.pin()

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sql_text

from sace_chat import manager
from sace_chat.consolidator import DUPLICATE_THRESHOLD, run_learning_loop
from sace_chat.db import EMBEDDING_DIM, engine as db_engine, init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine, _extract_question, question_key
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import get_llm
from sace_chat.retrieve import CallState

# Subjects no rule mentions. A reply containing one was invented.
INVENTED = [
    "contact info", "contact information", "phone or address", "primary care doctor",
    "primary care physician", "member id", "policy number", "date of birth",
    "household size", "income", "social security",
    # Nothing in the pool asks for consent — the old design had a stage merely
    # NAMED step2_consent, and the model used to infer a question from the name.
    "consent to discuss", "do you consent", "your consent",
]

# Content that belongs only to the counselor hand-off. Appearing in a DNC or
# abuse close means a closing was spliced together from two rules.
HANDOFF_MARKERS = ["keep", "counselor", "1-800-555-0100", "covered california"]

CONTROL_TOKENS = ["[CALL_END]", "[END:", "[opt-out]", "[active-coverage]"]

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def make_engine():
    init_db()
    eng = Engine(
        stable_core=STABLE_CORE,
        rules=RULES,
        embedder=get_embedder(),
        manager=manager,
        llm=get_llm(),
    )
    eng.router.warm()
    return eng


def turn(eng, state, history, msg, label=""):
    reply, _, dbg = eng.step(state, history, msg)
    gov = dbg["governing"]["id"] if dbg["governing"] else "(none)"
    refs = ",".join(r["id"] for r in dbg["reference"]) or "-"
    print(f"    caller : {msg}")
    print(f"    maya   : {reply}")
    print(
        f"    turn   : governing={gov}  reference={refs}  intent={dbg['intent']}  "
        f"outcome={dbg['outcome']}  cos={dbg['governing_cosine']:.3f}"
        f"{'  regenerated' if dbg['regenerated'] else ''}  ended={dbg['state_snapshot']['ended']}"
    )
    return reply, dbg


def scenario_1(eng):
    print("\n[1] free-form DNC — governing must be the DNC rule alone, call ends")
    state, history = CallState(), []
    turn(eng, state, history, "sure, go ahead")
    reply, dbg = turn(eng, state, history, "stop bothering me, I never want these calls again")

    gov = dbg["governing"]["id"] if dbg["governing"] else None
    record("1a. governing rule is special_dnc", gov == "special_dnc", f"governing={gov}")
    record("1b. nothing else in scope", not dbg["reference"],
           f"reference={[r['id'] for r in dbg['reference']]}")
    record("1c. reply grounded in it", dbg["outcome"] == "grounded",
           f"outcome={dbg['outcome']} cos={dbg['governing_cosine']:.3f}")
    spliced = [m for m in HANDOFF_MARKERS if m in reply.lower()]
    record("1d. no counselor/KEEP content spliced into the close", not spliced, f"found={spliced}")
    record("1e. call ends", dbg["state_snapshot"]["ended"])
    record("1f. opt-out recorded", dbg["state_snapshot"]["opt_out"])
    leaked = [t for t in CONTROL_TOKENS if t in reply]
    record("1g. no control token spoken", not leaked, f"leaked={leaked}")


def scenario_2(eng):
    print("\n[2] free-form pricing question — counselor redirect, call continues")
    state, history = CallState(), []
    turn(eng, state, history, "alright")
    turn(eng, state, history, "that's me")
    reply, dbg = turn(eng, state, history, "is this even worth it vs my work plan")

    gov = dbg["governing"]["id"] if dbg["governing"] else None
    record("2a. governing rule is special_pricing_q", gov == "special_pricing_q", f"governing={gov}")
    low = reply.lower()
    record("2b. redirects to the counselors", "counselor" in low, f"reply={reply}")
    record("2c. no price answered directly",
           not any(w in low for w in ("$", "per month", "costs about", "free")), f"reply={reply}")
    record("2d. call does NOT end", not dbg["state_snapshot"]["ended"])
    record("2e. reply grounded", dbg["outcome"] == "grounded",
           f"outcome={dbg['outcome']} cos={dbg['governing_cosine']:.3f}")


def scenario_3(eng):
    print("\n[3] no clear intent — a general rule governs, nothing invented")
    state, history = CallState(), []
    turn(eng, state, history, "hello? who is this")
    reply, dbg = turn(eng, state, history, "yeah I guess so")

    record("3a. no intent matched", dbg["intent"] is None,
           f"intent={dbg['intent']} closest={dbg['intent_ranked'][:2]}")
    gov = dbg["governing"]
    record("3b. a general rule governs", bool(gov) and gov["intent"] is None,
           f"governing={gov['id'] if gov else None} intent={gov['intent'] if gov else None}")
    invented = [t for t in INVENTED if t in reply.lower()]
    record("3c. nothing invented", not invented, f"found={invented}")
    record("3d. reply grounded", dbg["outcome"] == "grounded",
           f"outcome={dbg['outcome']} cos={dbg['governing_cosine']:.3f}")


class _StubLLM:
    """Returns one candidate that copies an existing rule verbatim, so the
    duplicate gate is exercised with something it must certainly catch."""

    name = "StubLLM(duplicate)"

    def __init__(self, candidate):
        self._candidate = candidate

    def chat_json(self, system, user):
        return json.dumps({"candidates": [self._candidate]})

    def chat(self, system, messages):
        return self.chat_json(system, "")


def scenario_4(eng):
    print("\n[4] embedding integrity and the duplicate gate")

    with db_engine.connect() as conn:
        rows = conn.execute(
            sql_text("SELECT id, source, embedding FROM chunks ORDER BY id")
        ).fetchall()

    bad = []
    learned = []
    for row in rows:
        if row.embedding is None:
            bad.append((row.id, "null"))
            continue
        vec = [float(x) for x in str(row.embedding).strip("[]").split(",")]
        norm = sum(v * v for v in vec) ** 0.5
        if len(vec) != EMBEDDING_DIM:
            bad.append((row.id, f"dim {len(vec)}"))
        elif norm <= 0:
            bad.append((row.id, f"norm {norm}"))
        if row.source == "learned":
            learned.append((row.id, round(norm, 6)))

    record(f"4a. all {len(rows)} embeddings valid (dim {EMBEDDING_DIM}, norm > 0)", not bad, f"bad={bad}")
    record(f"4b. {len(learned)} learned rule(s), all with norm > 0",
           bool(learned) and all(n > 0 for _, n in learned), f"learned={learned}")

    # A candidate proposing the DNC rule again, cue and all, with a grounded
    # source line — the clearest possible duplicate.
    dnc = next(r for r in RULES if r.id == "special_dnc")
    transcript = "Caller: stop calling me\nMaya: Of course — I'm making a note of that."
    stub = _StubLLM({
        "text": dnc.text,
        "cue": dnc.cue,
        "intent": "dnc",
        "learned_kind": "policy",
        "source_line": "Caller: stop calling me",
    })
    with db_engine.connect() as conn:
        gate = run_learning_loop(transcript, eng.embedder, conn, llm=stub)

    outcomes = [(g.outcome, g.detail) for g in gate]
    record("4c. a duplicate candidate is caught, not inserted",
           bool(gate) and gate[0].outcome == "duplicate-skipped",
           f"threshold={DUPLICATE_THRESHOLD}  outcomes={outcomes}")


def scenario_5(eng):
    print("\n[5] six improvised turns — no repeats, no spliced closings")
    state, history = CallState(), []
    script = [
        "who's this now?",
        "mhm, go ahead I've got a minute",
        "yeah you've got the right person",
        "hmm, I'm honestly not certain anymore",
        "sure, that'd be helpful",
        "alright, appreciate it",
    ]

    questions, invented, tokens, outcomes, govs = [], [], [], [], []
    for msg in script:
        reply, dbg = turn(eng, state, history, msg)
        outcomes.append(dbg["outcome"])
        govs.append(dbg["governing"]["id"] if dbg["governing"] else "(none)")
        low = reply.lower()
        invented += [t for t in INVENTED if t in low]
        tokens += [t for t in CONTROL_TOKENS if t in reply]
        q = _extract_question(reply)
        if q:
            questions.append(question_key(q))
        if dbg["state_snapshot"]["ended"]:
            break

    dupes = [q for q in set(questions) if questions.count(q) > 1]
    record("5a. no question asked twice", not dupes, f"dupes={dupes}")
    record("5b. nothing invented", not invented, f"found={sorted(set(invented))}")
    record("5c. no control token spoken", not tokens, f"found={sorted(set(tokens))}")
    record("5d. every turn grounded, none spliced",
           all(o == "grounded" for o in outcomes), f"outcomes={outcomes}")
    record("5e. the call actually progressed", len(set(govs)) >= 4, f"governing={govs}")


def main():
    eng = make_engine()
    print(f"model: {getattr(eng.llm, 'name', type(eng.llm).__name__)}")
    for fn in (scenario_1, scenario_2, scenario_3, scenario_4, scenario_5):
        fn(eng)

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
