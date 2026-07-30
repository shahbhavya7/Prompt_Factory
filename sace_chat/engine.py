"""The turn loop: retrieve -> one structured LLM call -> validate -> apply.

Retrieval is memory-only (see retrieve.py): no stage, no branch logic, one flat
pool. That puts the whole weight of correctness on the governing rule actually
governing, so the reply is checked against it before it is accepted:

  a. it must be semantically close to the governing rule;
  b. it must not be closer to a reference rule than to the governing one, which
     is what a spliced-in sentence looks like from the outside;
  c. a terminal rule forces the call to end, and nothing may trail the closing.

A failure at (a) or (b) buys exactly one regeneration with a correction block.
Anything still failing after that is reported rather than hidden.
"""

import re
import time

from sace_chat.assemble import build_turn_prompt
from sace_chat.db import engine as db_engine
from sace_chat.llm import build_messages, parse_json_object, render_messages
from sace_chat.retrieve import IntentRouter, retrieve
from sace_chat.tokens import est_tokens

# Pinned monolith baseline for the savings comparison (data/base_prompt_coverage.txt).
MONOLITH_TOKENS = 5782

# Below this, the reply is not traceable to the governing rule and is regenerated.
GROUNDING_THRESHOLD = 0.45

_QUOTED_RE = re.compile(r'"([^"]{12,})"')
_WORD_RE = re.compile(r"[a-z0-9']+")

# Rule text is static, so span embeddings are cached across turns — with a
# network embedder, re-embedding every span every turn is dozens of sequential
# round-trips of dead air.
_SPAN_VEC_CACHE: dict[str, list[float]] = {}
_MAX_SPANS_PER_RULE = 3


def _norm_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _rule_spans(rule_text: str) -> list[str]:
    """What to compare a reply against.

    A rule is prose wrapping one or two scripted lines, and scoring a short
    reply against all that prose compresses every score into a narrow band
    (measured: 0.30 invented vs 0.31 grounded — not separable). Against the
    quoted scripted lines the same cases separate cleanly.

    Learned rules carry no quotes at all, so they fall back to the whole text.
    Without that fallback every learned rule scores exactly 0.000 and would be
    regenerated forever — which is what the 0.000 seen against learned_71a695d9
    actually was.
    """
    spans = [s for s in _QUOTED_RE.findall(rule_text) if len(_norm_words(s)) >= 4]
    spans.sort(key=len, reverse=True)
    return spans[:_MAX_SPANS_PER_RULE] or [rule_text]


def _has_verbatim_overlap(reply_text: str, rule_text: str) -> bool:
    reply_words = _norm_words(reply_text)
    if not reply_words:
        return False
    joined = " " + " ".join(reply_words) + " "
    for quoted in _QUOTED_RE.findall(rule_text):
        words = _norm_words(quoted)
        if len(words) < 6:
            continue
        if any(f" {' '.join(words[i:i + 6])} " in joined for i in range(len(words) - 5)):
            return True
    return False


def score_reply(reply_text: str, rules, embedder) -> dict:
    """Cosine of the reply against each rule in scope, so a paraphrase still
    scores (substring matching alone misses those). Keyed by rule id."""
    if not reply_text or not rules:
        return {}

    wanted = {span for r in rules for span in _rule_spans(r.chunk.text)}
    missing = [s for s in wanted if s not in _SPAN_VEC_CACHE]
    try:
        from sace_chat.embeddings import embed_many

        if missing:
            for span, vec in zip(missing, embed_many(embedder, missing)):
                _SPAN_VEC_CACHE[span] = vec
        reply_vec = embedder.embed(reply_text)
    except Exception:
        return {}  # diagnostic only — never break a turn

    scores = {}
    for r in rules:
        best = 0.0
        for span in _rule_spans(r.chunk.text):
            vec = _SPAN_VEC_CACHE.get(span)
            if vec:
                best = max(best, _cosine(reply_vec, vec))
        scores[r.chunk.id] = {
            "id": r.chunk.id,
            "role": r.role,
            "cosine": round(best, 3),
            "verbatim": _has_verbatim_overlap(reply_text, r.chunk.text),
        }
    return scores


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_after_terminal(reply_text: str) -> tuple[str, bool]:
    """A terminal rule closes the call, so nothing may trail it. In practice
    what trails it is a tacked-on question ("...take care! Is there anything
    else?"), so any interrogative sentence is dropped."""
    parts = [p for p in _SENTENCE_SPLIT.split(reply_text.strip()) if p.strip()]
    kept = [p for p in parts if not p.strip().endswith("?")]
    if not kept or len(kept) == len(parts):
        return reply_text, False
    return " ".join(kept), True


def _extract_question(reply_text: str) -> str | None:
    """The question Maya asked, for ALREADY ASKED. Last sentence ending in '?'."""
    if not reply_text or "?" not in reply_text:
        return None
    for part in reversed(_SENTENCE_SPLIT.split(reply_text.strip())):
        if part.strip().endswith("?"):
            return part.strip()
    return None


class PromptCaptureError(AssertionError):
    """The captured payload does not contain this turn's caller message.

    Raised rather than logged. The whole point of the capture is that what the
    viewer shows IS what was sent; if that invariant breaks, the display is
    lying, and the off-by-one or stale-state bug behind it needs to surface
    immediately rather than be discovered later from a plausible-looking
    transcript.
    """


def assert_message_present(prompt_sent: str, user_message: str):
    """The caller's message for a turn must appear verbatim in that turn's
    captured payload. Guards against capturing the wrong turn's prompt, or
    rebuilding it after state has advanced."""
    if not user_message or not user_message.strip():
        return
    if user_message not in prompt_sent:
        raise PromptCaptureError(
            f"caller message not found verbatim in the captured prompt.\n"
            f"  message: {user_message!r}\n"
            f"  captured {len(prompt_sent)} chars ending: ...{prompt_sent[-200:]!r}"
        )


def question_key(question: str) -> str:
    """Identity of a question for dedup.

    Exact-string matching is too weak: "Just to confirm — do you still have your
    Medi-Cal benefits?" and "...Nothing to worry about. Do you still have your
    Medi-Cal benefits?" are one question in two preambles, and storing both lets
    the model re-ask indefinitely. The last 8 normalised words drop the preamble
    and compare the interrogative core.
    """
    words = _norm_words(question)
    return " ".join(words[-8:]) if words else ""


class Engine:
    def __init__(self, stable_core, rules=None, embedder=None, manager=None, llm=None,
                 monolith_text=None, table="chunks", chunks=None):
        self.stable_core = stable_core
        self.rules = rules if rules is not None else (chunks or [])
        self.embedder = embedder
        self.manager = manager
        self.llm = llm
        self.monolith_tokens = est_tokens(monolith_text) if monolith_text else MONOLITH_TOKENS
        self.table = table
        self.router = IntentRouter(embedder)

    def _retrieve(self, state, message, history):
        with db_engine.connect() as conn:
            return retrieve(
                conn, state, message, self.embedder,
                history=history, router=self.router, table=self.table,
                precedence=self.manager.resolve_precedence,
            )

    def _decide(self, prompt, user_message, state, notes, sent_log):
        # Capture the payload HERE, immediately before the call, from the same
        # builder the client uses — never rebuilt afterwards from state that may
        # since have moved on.
        messages = build_messages(prompt, user_message)
        prompt_sent = render_messages(messages)
        assert_message_present(prompt_sent, user_message)
        sent_log.append({
            "prompt_sent": prompt_sent,
            "messages": messages,
            "tokens": est_tokens(prompt_sent),
        })

        raw = self.llm.chat_json(prompt, user_message)
        decision, parse_error = parse_json_object(raw)
        if decision is None:
            notes.append(f"decision unparseable ({parse_error}); raw text used as the reply")
            decision = {
                "intent": "none",
                "reply": (raw or "").strip(),
                "call_should_end": False,
                "extracted_fields": {},
            }
        valid = self.manager.validate_turn(decision, state)
        notes.extend(valid["warnings"])
        return valid, raw

    def step(self, state, history, user_message):
        start = time.perf_counter()
        notes = []
        # One entry per LLM call this turn, appended at call time. A regeneration
        # adds a second entry, so both payloads stay inspectable.
        sent_log = []

        # 1. One query against the flat pool.
        retrieval = self._retrieve(state, user_message, history)
        notes.extend(retrieval.notes)
        governing = retrieval.governing

        # 2. One structured decision.
        prompt = build_turn_prompt(self.stable_core, state, retrieval, history)
        valid, raw = self._decide(prompt, user_message, state, notes, sent_log)

        # 3. Validate the reply against the rule that was supposed to govern it.
        scores = score_reply(valid["reply"], retrieval.rules, self.embedder)
        outcome, reason = self._judge(scores, governing, retrieval)

        regenerated = False
        if reason:
            notes.append(f"regenerating: {reason}")
            prompt = build_turn_prompt(
                self.stable_core, state, retrieval, history, reinforce_reason=reason
            )
            valid, raw = self._decide(prompt, user_message, state, notes, sent_log)
            scores = score_reply(valid["reply"], retrieval.rules, self.embedder)
            outcome, still_wrong = self._judge(scores, governing, retrieval)
            regenerated = True
            if still_wrong:
                notes.append(f"still failing after one regeneration: {still_wrong}")

        # 4. `terminal` decides whether the call ends — in both directions. The
        # model is not allowed a vote: it hung up on non-terminal rules (ending
        # the call right after handing over the counselors' number, before the
        # caller could reply), and it also missed endings the rule required.
        reply_text = valid["reply"]
        if governing is not None:
            if governing.chunk.terminal:
                if not valid["call_should_end"]:
                    notes.append(f"{governing.chunk.id} is terminal; forced the call to end")
                valid["call_should_end"] = True
                reply_text, trimmed = strip_after_terminal(reply_text)
                if trimmed:
                    notes.append("dropped a trailing question from a terminal reply")
            elif valid["call_should_end"]:
                notes.append(f"{governing.chunk.id} is not terminal; refused to end the call")
                valid["call_should_end"] = False

        # 5. Apply to state.
        state.intent = retrieval.intent or "none"
        state.opt_out = state.opt_out or retrieval.opt_out
        state.ended = state.ended or valid["call_should_end"]
        state.collected_fields.update(valid["extracted_fields"])

        question = _extract_question(reply_text)
        if question:
            key = question_key(question)
            if key and key in {question_key(q) for q in state.asked_questions}:
                notes.append(f"re-asked a question already in ALREADY ASKED: {question!r}")
            else:
                state.asked_questions.append(question)

        history.append(f"Caller: {user_message}")
        history.append(f"Maya: {reply_text}")

        governing_cos = scores.get(governing.chunk.id, {}).get("cosine", 0.0) if governing else 0.0
        prompt_tokens = est_tokens(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000

        final_sent = sent_log[-1]
        debug = {
            # What the model actually received for this turn, captured at call
            # time: system + user, verbatim. `prompt_sent` is the payload behind
            # the reply shown; `sent_log` holds every call made this turn, so a
            # regeneration's first attempt stays visible too.
            "prompt_sent": final_sent["prompt_sent"],
            "prompt_sent_tokens": final_sent["tokens"],
            "llm_messages": final_sent["messages"],
            "sent_log": sent_log,
            "llm_calls": len(sent_log),
            "caller_message": user_message,
            # The system half alone, for the token comparison against the
            # monolith system prompt (like against like).
            "assembled_prompt": prompt,
            "assembled_prompt_tokens": prompt_tokens,
            "monolith_tokens": self.monolith_tokens,
            "saved_pct": (1 - prompt_tokens / self.monolith_tokens) * 100,
            "elapsed_ms": elapsed_ms,
            "turn_json": valid,
            "raw_llm_output": raw,
            "notes": notes,
            # validation outcome
            "outcome": outcome,
            "grounded": outcome == "grounded",
            "spliced": outcome == "spliced",
            "regenerated": regenerated,
            "governing_cosine": governing_cos,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "scores": scores,
            # retrieval proof
            "intent": retrieval.intent,
            "intent_similarity": retrieval.intent_similarity,
            "intent_ranked": retrieval.intent_ranked,
            "query_text": retrieval.query_text,
            "governing": _rule_debug(governing, scores) if governing else None,
            "reference": [_rule_debug(r, scores) for r in retrieval.reference],
            "state_snapshot": {
                "intent": state.intent,
                "opt_out": state.opt_out,
                "ended": state.ended,
                "asked_questions": list(state.asked_questions),
                "collected_fields": dict(state.collected_fields),
            },
        }
        return reply_text, state, debug

    def _judge(self, scores, governing, retrieval) -> tuple[str, str]:
        """(outcome, reason-to-regenerate). An empty reason means accept."""
        if governing is None:
            return "no-rule", ""
        if not scores:
            return "unscored", ""

        governing_cos = scores.get(governing.chunk.id, {}).get("cosine", 0.0)

        # (b) before (a): a reply closer to a reference rule than to the
        # governing one is a splice, and its governing cosine may still be
        # respectable, so checking the threshold first would let it through.
        for ref in retrieval.reference:
            ref_cos = scores.get(ref.chunk.id, {}).get("cosine", 0.0)
            if ref_cos > governing_cos:
                return "spliced", (
                    f"the reply matched the REFERENCE rule {ref.chunk.id} (cosine {ref_cos:.3f}) "
                    f"more closely than the GOVERNING rule {governing.chunk.id} "
                    f"(cosine {governing_cos:.3f}). You took content from REFERENCE."
                )

        if governing_cos < GROUNDING_THRESHOLD:
            return "ungrounded", (
                f"the reply is not traceable to the GOVERNING rule {governing.chunk.id} "
                f"(cosine {governing_cos:.3f}, needs {GROUNDING_THRESHOLD}). You spoke "
                f"something the rule does not contain."
            )

        return "grounded", ""


def _rule_debug(rule, scores) -> dict:
    score = scores.get(rule.chunk.id, {})
    return {
        "id": rule.chunk.id,
        "title": rule.chunk.title,
        "role": rule.role,
        "intent": rule.chunk.intent,
        "priority": rule.chunk.priority,
        "terminal": rule.chunk.terminal,
        "exclusive": rule.chunk.exclusive,
        "source": rule.chunk.source,
        "learned_kind": rule.chunk.learned_kind,
        "similarity": round(rule.similarity, 3),
        "cosine": score.get("cosine"),
        "verbatim": score.get("verbatim", False),
        "char_len": len(rule.chunk.text),
        "snippet": rule.chunk.text[:220],
        "text": rule.chunk.text,
    }
