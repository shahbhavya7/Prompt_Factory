"""The campaign seam must be invisible when SACE_CAMPAIGN is unset.

sace_chat/campaign.py's "coverage" entry is built FROM sace_chat.kb's own
STABLE_CORE/RULES/INTENT_EXEMPLARS — this test is the guard that catches any
future edit to campaign.py (a stray .strip(), a re-sorted dict, a different
copy of a rule) that would make the resolved config diverge from importing
sace_chat.kb directly, which is what every front end did before this seam
existed.

Compares assembled system prompts (Engine.build_turn_context), never
engine.step() — step() makes a real LLM call whose reply text would then
diverge between the two engines and pollute later turns' RECENT TURNS
section with two different histories. build_turn_context does retrieval and
assembly only, so a fixed list of caller messages is enough to pin down
byte-identical output with no model call involved.

Run:  python tests/test_campaign_golden.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("SACE_CAMPAIGN", None)  # the condition under test: unset

from dotenv import load_dotenv

load_dotenv()

from sace_chat import campaign, manager
from sace_chat.db import init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.retrieve import CallState

# A fixed ten-turn transcript. Each line is retrieved independently (a fresh
# CallState/history per turn) — what matters is that both engines see the
# exact same input and are compared byte-for-byte on the output, not that the
# turns form one coherent call.
TRANSCRIPT = [
    "hello? who's calling please",
    "yeah alright, I've got a minute or two",
    "yep, you've got the right person",
    "yes I still have it",
    "it's under Sacramento county",
    "yes that would help",
    "no, not anymore",
    "got it, okay thanks",
    "stop calling me",
    "is this call being recorded",
]

results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def main():
    init_db()
    embedder = get_embedder()

    # The direct path: exactly what every front end did before campaign.py
    # existed.
    eng_direct = Engine(
        stable_core=STABLE_CORE, rules=RULES, embedder=embedder,
        manager=manager, table="chunks",
    )

    # The campaign path: resolved with SACE_CAMPAIGN unset, so this must be
    # "coverage" — the default.
    cfg = campaign.get_campaign()
    record("0. unset SACE_CAMPAIGN resolves to 'coverage'", cfg.name == "coverage",
           f"resolved={cfg.name!r}")

    eng_campaign = Engine(
        stable_core=cfg.stable_core, rules=campaign.load_rules(cfg),
        embedder=embedder, manager=manager, table=cfg.chunks_table,
    )

    record("1. chunks_table matches the direct path", cfg.chunks_table == "chunks",
           f"chunks_table={cfg.chunks_table!r}")

    for i, message in enumerate(TRANSCRIPT, start=1):
        state_a, history_a = CallState(), []
        state_b, history_b = CallState(), []
        prompt_a, _, _, _ = eng_direct.build_turn_context(state_a, history_a, message)
        prompt_b, _, _, _ = eng_campaign.build_turn_context(state_b, history_b, message)
        record(f"2.{i} turn {i} prompt is byte-identical ({message!r})",
               prompt_a == prompt_b,
               "" if prompt_a == prompt_b else
               f"direct   ends: ...{prompt_a[-160:]!r}\n"
               f"campaign ends: ...{prompt_b[-160:]!r}")

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
