"""Renewal campaign — Path A flow ordering, address_capture non-termination,
and the never-say guard.

Drives one continuous call through the "help" branch with an address
correction (the more complex of the two main branches), asserting the
governing rule advances in the expected order with no step skipped or
re-asked, and that the CSV tier counts match what build_kb_renewal.py
reported.

Run:  python tests/test_renewal_flow.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sace_chat import campaign, guards, manager
from sace_chat.db import init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.llm import get_llm
from sace_chat.retrieve import CallState

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


PATH_A_TURNS = [
    ("hello?", "open_identify"),
    ("Yes, speaking.", "disclose_ai_and_recording"),
    ("Oh yeah, I got a text about this yesterday, now's fine.", "verify_dob"),
    ("March 8th, 1983.", "packet_check"),
    ("No, we moved. In May I think.", "address_capture"),
    ("1180 Dutton Avenue, apartment 6, Santa Rosa.", "address_capture"),
    ("Yes, that's right.", "already_submitted_check"),
    ("No, I never got it.", "willingness_ask"),
    ("Okay, they can do it, that's easier for me.", "m1_availability"),
    ("Now is okay, I'm home.", "m2_camera_phone"),
    ("Yeah, I'm on my phone.", "m3_helper_at_home"),
    ("My daughter helps sometimes, but she's at school.", "consent_prebrief"),
    ("Okay, that sounds okay.", "warm_transfer"),
]


def main():
    init_db()
    cfg = campaign.get_campaign("renewal")
    embedder = get_embedder()
    engine = Engine(
        stable_core=cfg.stable_core, rules=campaign.load_rules(cfg),
        embedder=embedder, manager=manager, llm=get_llm(), table=cfg.chunks_table,
        never_say_guard=cfg.never_say_guard, never_say_fallback=cfg.never_say_fallback,
    )
    engine.router.warm()

    tiers = {}
    for r in engine.rules:
        if r.tier:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
    record("0. tier counts match T1=92 T2=3 T3=21 T4=10",
           tiers == {"T1": 92, "T2": 3, "T3": 21, "T4": 10}, f"got {tiers}")

    print("\n[1] Path A walkthrough (help branch, with address correction)")
    state, history = CallState(), []
    seen_ids = []
    address_capture_ended_call = False
    reask_found = False
    for i, (message, expected) in enumerate(PATH_A_TURNS, start=1):
        reply, _, debug = engine.step(state, history, message)
        gov = debug["governing"]["id"] if debug["governing"] else None
        seen_ids.append(gov)
        print(f"    turn {i:>2}: {message!r}")
        print(f"      governing={gov}  expected={expected}  reply={reply[:90]!r}")
        for note in debug.get("notes", []):
            if "re-asked" in note:
                reask_found = True
                print(f"      NOTE: {note}")
        if gov == "address_capture" and debug.get("call_should_end"):
            address_capture_ended_call = True
        record(f"1.{i} turn {i} governing rule matches expected ({expected})",
               gov == expected, f"got {gov!r}")

    record("2. no turn re-asked a question already in ALREADY ASKED", not reask_found)
    record("3. address_capture never ended the call", not address_capture_ended_call)
    record("4. the call ended by the time warm_transfer governed", state.ended)

    print("\n[5] the never-say guard")
    fixture_case_record = {"due_date": "August 30, 2026"}
    ok1, why1 = guards.check_never_say(
        "Based on what you told me, you'd need to be under $21,000 to qualify.",
        fixture_case_record,
    )
    record("5a. fires on an invented dollar threshold", ok1 is False, why1)
    ok2, why2 = guards.check_never_say(
        "Your renewal is due by August 30, 2026 — that's straight from your case record.",
        fixture_case_record,
    )
    record("5b. does not fire on a due date present in the case record", ok2 is True, why2)

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
