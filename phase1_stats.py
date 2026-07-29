"""Token budget: the monolith against the stable core plus one turn's rules."""

from collections import Counter

from sace_chat.assemble import _TURN_INSTRUCTION
from sace_chat.kb import INTENT_EXEMPLARS, RULES, STABLE_CORE
from sace_chat.tokens import est_tokens

MONOLITH_PATH = "data/base_prompt_coverage.txt"


def main():
    with open(MONOLITH_PATH, encoding="utf-8") as f:
        monolith_text = f.read()

    monolith_tokens = est_tokens(monolith_text)
    core_tokens = est_tokens(STABLE_CORE)
    instruction_tokens = est_tokens(_TURN_INSTRUCTION)
    fixed = core_tokens + instruction_tokens

    rule_tokens = sorted(est_tokens(r.text) for r in RULES)
    median = rule_tokens[len(rule_tokens) // 2]
    general = [r for r in RULES if r.intent is None]

    print(f"Monolith tokens:         {monolith_tokens}")
    print(f"STABLE_CORE tokens:      {core_tokens}")
    print(f"Turn instruction tokens: {instruction_tokens}")
    print(f"Fixed per-turn overhead: {fixed}")
    print()
    print(f"Rules in the pool:  {len(RULES)}  ({len(general)} general, {len(RULES) - len(general)} intent-routed)")
    print(f"Rule tokens min/median/max: {rule_tokens[0]} / {median} / {rule_tokens[-1]}")
    print(f"Sum of ALL rules:   {sum(rule_tokens)}  (never sent — this is what a monolith pays every turn)")
    print()

    # A turn sends the core, the instruction, at most two rules, plus history.
    # Two rules is the worst case (no intent matched); an intent match sends one.
    for label, rules_tokens in (
        ("intent match (1 median rule)", median),
        ("general (2 median rules)", median * 2),
        ("worst case (2 largest)", sum(rule_tokens[-2:])),
    ):
        total = fixed + rules_tokens
        print(f"  {label:30s} {total:>5} tok   saved {100 * (1 - total / monolith_tokens):5.1f}%")

    print()
    print(f"Intent labels with exemplars: {len(INTENT_EXEMPLARS)}")
    print(f"Exemplars total:              {sum(len(v) for v in INTENT_EXEMPLARS.values())}")
    print()
    print("Rules per intent:")
    for k, v in sorted(Counter(r.intent or "(general)" for r in RULES).items()):
        print(f"  {k:20s} {v}")
    print()
    print("Terminal:  " + ", ".join(r.id for r in RULES if r.terminal))
    print("Exclusive: " + ", ".join(r.id for r in RULES if r.exclusive))


if __name__ == "__main__":
    main()
