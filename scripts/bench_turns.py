"""Benchmark harness: replay a fixed corpus of turns through one campaign and
report cache/latency/accuracy numbers. Written and run BEFORE any renewal
content exists — this is the baseline every later change to the renewal
campaign gets compared against.

Every threshold in this system (grounding, splice, cache, dedup) was measured
against real embeddings, and every one of them is meaningless under
MockEmbedder's 384-dim hashed bag-of-words — see ARCHITECTURE.md §4 and §6.
So this script refuses to run at all under the mock embedder, loudly, rather
than silently producing numbers that look like a benchmark and mean nothing.

Corpus format: one JSON object per line —
    {"message": "<what the caller said>",
     "expected_rule": "<governing rule id, or null>",
     "expected_tier": "<the source KB's tier for that rule, or null>"}

Each line is replayed as an independent, single-turn call (fresh CallState,
empty history) through Engine.step() — so what's measured is "a caller opens
with this line", not a multi-turn conversation. That is deliberately the same
shape as the corpus format: one line, one governing rule, one tier.

Reports p50 AND p95 for every latency figure. A p50 that holds steady while
p95 doubles is a real regression a mean would hide — most often the
regeneration path (one rejected attempt doubles that turn's cost) or a
cache/DB tail latency, neither of which moves the median at all.

Run:  python scripts/bench_turns.py
      python scripts/bench_turns.py --campaign coverage --commit-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sace_chat import campaign, manager
from sace_chat.db import init_db
from sace_chat.embeddings import MockEmbedder, get_embedder
from sace_chat.engine import Engine
from sace_chat.llm import get_llm
from sace_chat.retrieve import CallState

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = {
    "coverage": REPO_ROOT / "data" / "bench" / "coverage_turns.jsonl",
}


def _percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile. None for an empty series — a metric
    with no data (e.g. p95 of cache-hit latency when nothing hit) must read as
    absent, not as a misleading 0."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_corpus(path: Path) -> list[dict]:
    turns = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON — {exc}")
    return turns


def git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_bench(cfg, corpus: list[dict], embedder, llm) -> dict:
    engine = Engine(
        stable_core=cfg.stable_core, rules=campaign.load_rules(cfg),
        embedder=embedder, manager=manager, llm=llm, table=cfg.chunks_table,
    )
    engine.router.warm()

    n = 0
    cache_hit = 0
    cache_ms, full_ms = [], []
    full_prompt_tokens = []
    grounded = 0
    rule_checks, rule_correct = 0, 0

    for turn in corpus:
        message = turn["message"]
        expected_rule = turn.get("expected_rule")

        state, history = CallState(), []
        try:
            _, _, debug = engine.step(state, history, message)
        except Exception as exc:
            print(f"  ERROR on turn {message!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        n += 1
        outcome = debug.get("outcome")
        elapsed = debug.get("elapsed_ms", 0.0)

        if outcome == "cached":
            cache_hit += 1
            cache_ms.append(elapsed)
        else:
            full_ms.append(elapsed)
            full_prompt_tokens.append(debug.get("assembled_prompt_tokens", 0))

        if outcome in ("grounded", "cached"):
            grounded += 1

        if expected_rule is not None:
            rule_checks += 1
            gov = debug.get("governing")
            actual_rule = gov["id"] if gov else None
            if actual_rule == expected_rule:
                rule_correct += 1

    return {
        "campaign": cfg.name,
        "n": n,
        "cache_hit_rate": (cache_hit / n) if n else None,
        "p50_cache_hit_ms": _percentile(cache_ms, 0.50),
        "p95_cache_hit_ms": _percentile(cache_ms, 0.95),
        "p50_full_path_ms": _percentile(full_ms, 0.50),
        "p95_full_path_ms": _percentile(full_ms, 0.95),
        "p50_prompt_tokens": _percentile(full_prompt_tokens, 0.50),
        "governing_rule_accuracy": (rule_correct / rule_checks) if rule_checks else None,
        "governing_rule_checks": rule_checks,
        "grounded_rate": (grounded / n) if n else None,
    }


def print_table(rows: list[dict]) -> None:
    cols = [
        ("campaign", "{:<10}"), ("n", "{:>3}"),
        ("cache_hit_rate", "{:>14}"),
        ("p50_cache_ms", "{:>12}"), ("p95_cache_ms", "{:>12}"),
        ("p50_full_ms", "{:>11}"), ("p95_full_ms", "{:>11}"),
        ("p50_tokens", "{:>10}"),
        ("gov_accuracy", "{:>12}"), ("grounded_rate", "{:>13}"),
    ]
    key_map = {
        "cache_hit_rate": "cache_hit_rate", "p50_cache_ms": "p50_cache_hit_ms",
        "p95_cache_ms": "p95_cache_hit_ms", "p50_full_ms": "p50_full_path_ms",
        "p95_full_ms": "p95_full_path_ms", "p50_tokens": "p50_prompt_tokens",
        "gov_accuracy": "governing_rule_accuracy", "grounded_rate": "grounded_rate",
    }

    def fmt(v, is_pct=False, is_ms=False, is_tok=False):
        if v is None:
            return "—"
        if is_pct:
            return f"{v * 100:.1f}%"
        if is_ms:
            return f"{v:.0f}ms"
        if is_tok:
            return f"{v:.0f}"
        return str(v)

    header = " | ".join(fmt_.format(name) for name, fmt_ in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = [
            row["campaign"], row["n"],
            fmt(row["cache_hit_rate"], is_pct=True),
            fmt(row["p50_cache_hit_ms"], is_ms=True), fmt(row["p95_cache_hit_ms"], is_ms=True),
            fmt(row["p50_full_path_ms"], is_ms=True), fmt(row["p95_full_path_ms"], is_ms=True),
            fmt(row["p50_prompt_tokens"], is_tok=True),
            fmt(row["governing_rule_accuracy"], is_pct=True),
            fmt(row["grounded_rate"], is_pct=True),
        ]
        print(" | ".join(fmt_.format(v) for (_, fmt_), v in zip(cols, vals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=None,
                         help="campaign name (default: SACE_CAMPAIGN env, else 'coverage')")
    parser.add_argument("--corpus", type=Path, default=None,
                         help="JSONL corpus path (default: data/bench/<campaign>_turns.jsonl)")
    parser.add_argument("--out", type=Path, default=None,
                         help="write the JSON report here")
    parser.add_argument("--commit-baseline", action="store_true",
                         help="also write bench/<campaign>-<sha>.json")
    args = parser.parse_args()

    embedder = get_embedder()
    if isinstance(embedder, MockEmbedder):
        raise SystemExit(
            "\nREFUSING TO RUN: EMBEDDING_MODE is 'mock'.\n"
            "Every threshold this benchmark reports against (grounding, cache,\n"
            "duplicate) was measured against real embeddings and is meaningless\n"
            "under MockEmbedder's 384-dim hashed bag-of-words — see\n"
            "ARCHITECTURE.md §4 and §6. Set EMBEDDING_MODE=openai (and a real\n"
            "SACE_LLM_KEY) and re-run.\n"
        )

    init_db()
    cfg = campaign.get_campaign(args.campaign)
    corpus_path = args.corpus or DEFAULT_CORPUS.get(cfg.name)
    if corpus_path is None:
        raise SystemExit(
            f"no default corpus registered for campaign {cfg.name!r} — pass --corpus"
        )
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise SystemExit(f"{corpus_path}: corpus is empty")

    llm = get_llm()
    print(f"campaign={cfg.name}  corpus={corpus_path}  n={len(corpus)}  "
          f"embedder={type(embedder).__name__}  llm={getattr(llm, 'name', type(llm).__name__)}\n")

    t0 = time.time()
    row = run_bench(cfg, corpus, embedder, llm)
    wall_s = time.time() - t0

    print_table([row])
    print(f"\n(wall time {wall_s:.1f}s)")

    report = {
        "sha": git_short_sha(),
        "campaign": cfg.name,
        "corpus": str(corpus_path.relative_to(REPO_ROOT)),
        "embedding_mode": os.environ.get("EMBEDDING_MODE"),
        "llm_model": os.environ.get("SACE_LLM_MODEL"),
        "wall_time_s": round(wall_s, 2),
        **row,
    }

    out_path = args.out
    if args.commit_baseline and out_path is None:
        out_path = REPO_ROOT / "bench" / f"{cfg.name}-{report['sha']}.json"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
