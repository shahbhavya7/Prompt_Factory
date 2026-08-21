# SACE — how it works, in one picture

One flat pool of rules in Postgres replaces one giant system prompt. Every turn,
retrieval picks the **one rule** that governs this reply, sends only that to the
model, and checks the reply actually came from it.

## Live call

```mermaid
flowchart LR
    A["caller's message"] --> B["retrieve()
    cosine search over rule pool"]
    B --> C["governing rule
    (+ 1 reference rule, background only)"]
    C --> D["assemble prompt
    stable core + governing rule"]
    D --> E["one LLM call
    returns: intent, reply,
    call_should_end, fields"]
    E --> F{"grounding check
    is the reply's cosine
    close to the governing rule?"}
    F -- "no, ungrounded or spliced" --> G["regenerate once
    with correction"]
    G --> F
    F -- "yes" --> H["reply is spoken /
    shown to caller"]
    H --> I["state + turn logged"]

    DB[("chunks table
    Postgres + pgvector")] -.-> B
```

**Why this matters:** the model never sees the whole rulebook — just the one rule
that applies right now. Add rule #100 and every other turn's cost stays the same.

## After the call ends (separate, offline)

```mermaid
flowchart LR
    T["finished transcript"] --> X["LLM extracts
    candidate rules"]
    X --> G1{"grounded in
    transcript?"}
    G1 -- no --> R1["discarded"]
    G1 -- yes --> G2{"duplicate of an
    existing rule?"}
    G2 -- yes --> R2["skipped"]
    G2 -- no --> G3{"conflicts with an
    existing rule?"}
    G3 -- yes --> R3["needs_review
    (a human decides)"]
    G3 -- no --> INS["inserted into
    the SAME chunks table"]
    INS -.->|"reachable on the
    next call"| DB[("chunks table")]
```

**Why this matters:** nothing the agent "learns" is applied silently — it either
passes all three checks or lands in a review queue for a person to look at.

## The two pieces together

```
                 live call path                 offline learning path
                 (every turn)                   (once, after call ends)
┌─────────────────────────────┐        ┌───────────────────────────────┐
│ caller → retrieve → LLM →    │        │ transcript → extract →        │
│ grounding check → reply      │        │ 3 gates → insert or review    │
└──────────────┬───────────────┘        └───────────────┬───────────────┘
               │  reads from                              │ writes to
               ▼                                          ▼
                     ┌───────────────────────┐
                     │   chunks (the pool)   │
                     │  seed rules + learned │
                     └───────────────────────┘
```

## Key numbers

| | |
|---|---|
| Rules in the pool | 33 seed (19 general + 14 intent-routed) + N learned |
| Monolith baseline | 5,782 tokens, sent every turn |
| Retrieved-pool cost | 965–1,296 tokens/turn (78–83% smaller) |
| Grounding floor | cosine ≥ 0.45 against the governing rule |
| Regeneration budget | 1 retry, with an explicit correction |

Full detail (every function, every table column, every edge case) is in
[ARCHITECTURE.md](ARCHITECTURE.md). This page is the one-screen version.
