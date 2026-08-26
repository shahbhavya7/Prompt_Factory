"""Generate the held-out measurement set for the renewal cache bar.

For each of the 126 KB entries, one NEW caller paraphrase of the canonical
question — distinct from the title and every seeded cue_variant — so
measure_cache_bar_renewal.py can measure real embedding behaviour against
phrasings the cache was never seeded with. Guessing a threshold from the
seeded variants would be measuring the cache against its own training data.

Batched LLM calls (SACE_LLM_MODEL), not one call per rule. Every returned
paraphrase is checked against the rule's own title/cue_variants (case-
insensitive) and regenerated once if it collides — a paraphrase that IS a
seeded variant would make D measure nothing.

Output: data/renewal/eval/paraphrases.jsonl, one JSON object per line:
  {"id": ..., "tier": ..., "intent": ..., "canonical": ..., "paraphrase": ...}

Run:  python scripts/generate_paraphrases_renewal.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.kb", override=True)
load_dotenv(ROOT / ".env")

from sace_chat.kb_renewal import RULES  # noqa: E402

OUTPUT = ROOT / "data" / "renewal" / "eval" / "paraphrases.jsonl"
BATCH_SIZE = 20


def _client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ["SACE_LLM_KEY"]
    return OpenAI(api_key=api_key, base_url=os.environ.get("SACE_LLM_BASE", "https://api.openai.com/v1"))


def _model():
    return os.environ.get("SACE_LLM_MODEL", "gpt-4o-mini")


def _ask_batch(client, model, batch, retry_ids=None):
    items = []
    for r in batch:
        if retry_ids is not None and r.id not in retry_ids:
            continue
        known = "; ".join([r.title, *r.cue_variants])
        items.append(f'- id "{r.id}": canonical question "{r.title}". Already-known phrasings: {known}')
    if not items:
        return {}

    prompt = (
        "For each numbered item below, a Medi-Cal renewal call-center caller's question is given, "
        "along with phrasings we already have on file for it. Write exactly ONE new, natural, "
        "colloquial way a real patient on the phone might ask the SAME question — different in "
        "wording from every already-known phrasing listed, but still asking that same question. "
        "Keep it short (under 15 words), first-person, spoken register, no quotation marks.\n\n"
        + "\n".join(items)
        + "\n\nRespond with a JSON object mapping each id to its paraphrase string."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    return json.loads(resp.choices[0].message.content)


def _collides(paraphrase: str, rule) -> bool:
    p = paraphrase.strip().lower()
    known = {rule.title.strip().lower(), *(v.strip().lower() for v in rule.cue_variants)}
    return p in known


def main():
    client = _client()
    model = _model()
    by_id = {r.id: r for r in RULES}
    results: dict[str, str] = {}

    for i in range(0, len(RULES), BATCH_SIZE):
        batch = RULES[i:i + BATCH_SIZE]
        got = _ask_batch(client, model, batch)
        results.update(got)
        print(f"  batch {i // BATCH_SIZE + 1}: {len(got)} paraphrases")

    # One retry pass for anything missing or that collided with a known variant.
    bad = [rid for rid, p in results.items() if by_id.get(rid) and _collides(p, by_id[rid])]
    missing = [r.id for r in RULES if r.id not in results]
    retry_ids = set(bad) | set(missing)
    if retry_ids:
        print(f"  retrying {len(retry_ids)} collision(s)/missing: {sorted(retry_ids)}")
        retry_batch = [r for r in RULES if r.id in retry_ids]
        got = _ask_batch(client, model, retry_batch, retry_ids=retry_ids)
        results.update(got)

    still_bad = [rid for rid, p in results.items() if by_id.get(rid) and _collides(p, by_id[rid])]
    still_missing = [r.id for r in RULES if r.id not in results]
    if still_bad or still_missing:
        raise AssertionError(
            f"could not produce a held-out paraphrase for: collided={still_bad} missing={still_missing}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in RULES:
            f.write(json.dumps({
                "id": r.id,
                "tier": r.tier,
                "intent": r.intent,
                "canonical": r.title,
                "paraphrase": results[r.id].strip(),
            }, ensure_ascii=False) + "\n")

    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(RULES)} held-out paraphrases")


if __name__ == "__main__":
    sys.exit(main())
