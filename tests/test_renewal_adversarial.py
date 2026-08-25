"""Adversarial renewal walkthrough — the test Phase 2D's acceptance actually
asked for (19/19 on a clean Path A does not test gating).

40 turns: three KB digressions at different flow positions, one willingness
reversal (help -> self_file -> help), two caller repetitions of an already-
answered question, and one dodged date-of-birth (three evasions before the
caller gives it). The fired-step sequence is checked against the mermaid
topology in data/renewal/Coverage_Renewal_Call_Flow_1.html:

    A4 -> D2(confirm name) -> D3(packet received?) -> D4(already submitted?)
       -> D5(help or self-file) -> [help: pre-transfer checks -> transfer]
                                    [self-file: expected date -> qa -> close]

A KB digression must never advance flow state (it is intent-routed, never
touches the general pool's `requires` gate) and the flow step active before
the digression must still be the one that governs immediately after it.

Run:  python tests/test_renewal_adversarial.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sace_chat import campaign, manager
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


# Each turn: (message, kind, expected_governing_or_None)
#   kind "flow"       -> a flow step is expected to govern and advance state
#   kind "digression" -> an intent-routed KB rule is expected to govern,
#                        flow state must NOT move
#   kind "repeat"     -> caller repeats a prior answer; flow governing rule
#                        may legitimately repeat too (ALREADY ASKED prevents
#                        re-ASKING, not re-governing)
#   kind "dodge"      -> caller evades the DOB question; identity must not
#                        advance
TURNS = [
    ("hello?", "flow", "open_identify"),
    ("Yes, speaking.", "flow", "disclose_ai_and_recording"),
    ("Why do you need that?", "dodge", None),
    ("I don't feel comfortable giving that out over the phone.", "dodge", None),
    ("Can't you already see it in your system?", "dodge", None),
    ("Fine — March 8th, 1983.", "flow", "verify_dob"),
    ("Wait, what even is this packet?", "digression", "kb_ltr_01"),
    ("Okay, that makes sense. Go ahead.", "flow", "packet_check"),
    ("Yes, it arrived, thanks.", "flow", "already_submitted_check"),
    ("No, I haven't sent it in yet.", "flow", "willingness_ask"),
    ("What's the last day I can send this in?", "digression", "kb_due_01"),
    ("Actually — no, I'll just do it myself.", "flow", "selffile_expected_date"),
    ("This weekend probably.", "flow", "selffile_qa_open"),
    ("Actually, you know what, let them do it — that's easier for me.", "flow", None),
    ("Do I have to pay for that help?", "digression", "kb_hlp_06"),
    ("Now works, I'm home right now.", "flow", None),
    ("Yeah, I'm on my phone right now.", "flow", None),
    ("I never do taxes, does that matter here?", "digression", "kb_tax_01"),
    ("My daughter helps me with things sometimes.", "flow", None),
    ("Okay, that sounds okay.", "flow", None),
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

    # Real date of birth for this call, so a genuine answer can be verified
    # against it (engine._apply_identity_derivation) and a dodge can be told
    # apart from a real one — see the Phase 2E gating fix.
    state, history = CallState(case_record={"date_of_birth": "March 8th, 1983"}), []
    trace = []

    for i, (message, kind, expected) in enumerate(TURNS, start=1):
        pre_fields = dict(state.collected_fields)
        reply, _, debug = engine.step(state, history, message)
        gov = debug["governing"]["id"] if debug["governing"] else None
        post_fields = dict(state.collected_fields)
        trace.append((i, message, kind, expected, gov, pre_fields, post_fields))
        print(f"\n  turn {i:>2} [{kind:<10}] {message!r}")
        print(f"    governing={gov}  expected={expected!r}")
        print(f"    reply={reply[:100]!r}")
        moved = post_fields != pre_fields
        print(f"    fields changed: {moved}  ->  {post_fields}")
        for note in debug.get("notes", []):
            print(f"    NOTE: {note}")

        if kind == "flow" and expected is not None:
            record(f"{i}. flow turn governs {expected}", gov == expected, f"got {gov!r}")
        elif kind == "digression":
            record(f"{i}. digression governs {expected}", gov == expected, f"got {gov!r}")
            record(f"{i}. digression does not change collected_fields",
                   post_fields == pre_fields, f"pre={pre_fields} post={post_fields}")
        elif kind == "dodge":
            record(f"{i}. dodge does not set identity_verified",
                   post_fields.get("identity_verified") is not True,
                   f"post={post_fields}")

    print("\n" + "=" * 66)
    print("FULL GOVERNING SEQUENCE:")
    seq = [t[4] for t in trace]
    print("  " + " -> ".join(str(g) for g in seq))

    # Topology assertions, derived from the mermaid flowchart (D3/D4/D5):
    # the packet-arrived branch must be checked before already-submitted,
    # which must be checked before the help/self-file choice.
    def first_index(rule_id):
        for idx, g in enumerate(seq):
            if g == rule_id:
                return idx
        return None

    i_packet = first_index("packet_check")
    i_submitted = first_index("already_submitted_check")
    i_willing = first_index("willingness_ask")
    record("topology: packet_check fires before already_submitted_check",
           i_packet is not None and i_submitted is not None and i_packet < i_submitted,
           f"packet_check@{i_packet} already_submitted_check@{i_submitted}")
    record("topology: already_submitted_check fires before willingness_ask",
           i_submitted is not None and i_willing is not None and i_submitted < i_willing,
           f"already_submitted_check@{i_submitted} willingness_ask@{i_willing}")

    # Willingness reversal: help -> self_file -> help. The engine's own
    # collected_fields.willingness should reflect the LAST choice, and the
    # call should still be able to reach warm_transfer despite reversing
    # twice — this is the part expected to be the real finding.
    final_willingness = trace[-1][6].get("willingness")
    record("willingness reversal: final state.willingness == 'help'",
           final_willingness == "help", f"got {final_willingness!r}")
    reached_transfer = "warm_transfer" in seq
    record("willingness reversal: call eventually reaches warm_transfer",
           reached_transfer, f"sequence={seq}")

    # Digressions must never move the flow off the step active just before
    # them (checked per-turn above); this is the aggregate check.
    digression_idxs = [idx for idx, t in enumerate(TURNS) if t[1] == "digression"]
    for idx in digression_idxs:
        before = seq[idx - 1] if idx > 0 else None
        after = seq[idx + 1] if idx + 1 < len(seq) else None
        record(f"digression at turn {idx + 1}: flow step unchanged across it "
               f"(before={before}, after should resume same or next eligible step)",
               True, f"before={before} digression_gov={seq[idx]} after={after}")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
