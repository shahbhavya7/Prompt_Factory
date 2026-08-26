"""The prompt viewer must show what was actually sent, per turn.

Sends three turns and checks each turn's captured payload independently: that it
contains that turn's caller message verbatim, that it contains the rules and
sections that were in scope for that turn, and that the three captures differ
from one another (a reconstruction from current state would make them converge).

Run:  python tests/test_prompt_capture.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_env

kb_env.pin()

from dotenv import load_dotenv

load_dotenv()

from sace_chat import manager
from sace_chat.db import init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine, PromptCaptureError, assert_message_present
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import build_messages, get_llm, render_messages
from sace_chat.retrieve import CallState

TURNS = [
    "hello? who's calling please",
    "yeah alright, I've got a minute or two",
    "yep, you've got the right person",
]

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def test_assertion_fires():
    print("\n[0] the guard itself fails loudly")
    sent = render_messages(build_messages("SYSTEM TEXT", "the real message"))
    try:
        assert_message_present(sent, "a message that was never sent")
        record("0a. mismatch raises PromptCaptureError", False, "no exception raised")
    except PromptCaptureError as exc:
        record("0a. mismatch raises PromptCaptureError", True, str(exc).splitlines()[0])
    try:
        assert_message_present(sent, "the real message")
        record("0b. a genuine match passes", True)
    except PromptCaptureError as exc:
        record("0b. a genuine match passes", False, str(exc))


def main():
    init_db()
    eng = Engine(
        stable_core=STABLE_CORE,
        rules=RULES,
        embedder=get_embedder(),
        manager=manager,
        llm=get_llm(),
    )
    eng.router.warm()

    test_assertion_fires()

    print(f"\n[1] three turns, model {getattr(eng.llm, 'name', '?')}")
    state, history = CallState(), []
    turns = []
    for msg in TURNS:
        reply, _, dbg = eng.step(state, history, msg)
        turns.append((msg, reply, dbg))
        gov = dbg["governing"]["id"] if dbg["governing"] else "(none)"
        print(f"    turn {len(turns)}: {msg!r}")
        print(f"      governing={gov}  calls={dbg['llm_calls']}  "
              f"sent={dbg['prompt_sent_tokens']} tok")

    print("\n[2] each capture is that turn's own")
    for i, (msg, reply, dbg) in enumerate(turns, start=1):
        sent = dbg["prompt_sent"]
        record(f"2a.{i} turn {i} capture contains its own caller message verbatim",
               msg in sent, f"message={msg!r}")

        others = [m for j, (m, _, _) in enumerate(turns, start=1) if j != i]
        # A later turn legitimately quotes earlier messages back in RECENT TURNS,
        # so only LATER messages must be absent — those cannot have existed yet.
        future = others[i - 1:] if i - 1 < len(others) else []
        leaked = [m for m in future if m in sent]
        record(f"2b.{i} turn {i} capture contains no message from a later turn",
               not leaked, f"leaked={leaked}")

        record(f"2c.{i} turn {i} capture is the real two-message payload",
               [m["role"] for m in dbg["llm_messages"]] == ["system", "user"]
               and dbg["llm_messages"][1]["content"] == msg,
               f"roles={[m['role'] for m in dbg['llm_messages']]}")

    print("\n[3] every required section is present verbatim")
    # A span of STABLE_CORE carrying no {placeholder} — those are substituted at
    # assembly time, so a line containing one will never appear raw in the sent
    # prompt.
    core_marker = next(
        line for line in STABLE_CORE.splitlines()
        if len(line) > 60 and "{" not in line and not line.startswith("#")
    )
    for i, (msg, reply, dbg) in enumerate(turns, start=1):
        sent = dbg["prompt_sent"]
        gov_text = dbg["governing"]["text"] if dbg["governing"] else None
        # The prompt substitutes demo values into {placeholders}, so compare on a
        # span of the rule that carries none.
        gov_span = None
        if gov_text:
            plain = [s for s in gov_text.split(". ") if "{" not in s and len(s) > 40]
            gov_span = plain[0] if plain else None

        checks = {
            "STABLE_CORE": core_marker[:60] in sent,
            "GOVERNING RULE heading": "GOVERNING RULE" in sent,
            "governing rule text": (gov_span in sent) if gov_span else True,
            "REFERENCE section": "REFERENCE" in sent,
            "ALREADY ASKED section": "ALREADY ASKED" in sent,
            "RECENT TURNS section": "RECENT TURNS" in sent,
            "caller message": msg in sent,
            "JSON output instruction": "call_should_end" in sent and "# THIS TURN" in sent,
            "=== SYSTEM === marker": "=== SYSTEM ===" in sent,
            "=== USER === marker": "=== USER ===" in sent,
        }
        missing = [k for k, ok in checks.items() if not ok]
        record(f"3.{i} turn {i} capture has all sections", not missing, f"missing={missing}")

    print("\n[4] the three captures are distinct")
    sents = [d["prompt_sent"] for _, _, d in turns]
    pairs = [(i + 1, j + 1) for i in range(len(sents)) for j in range(i + 1, len(sents))
             if sents[i] == sents[j]]
    record("4a. no two turns captured an identical prompt", not pairs, f"identical pairs={pairs}")
    record("4b. captures grow as history accumulates",
           len(sents[0]) < len(sents[2]),
           f"lengths={[len(s) for s in sents]}")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
