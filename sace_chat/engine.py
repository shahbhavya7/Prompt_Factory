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
from sace_chat.retrieve import IntentRouter, RulePoolCache, retrieve
from sace_chat.tokens import est_tokens

# Pinned monolith baseline for the savings comparison (data/base_prompt_coverage.txt).
MONOLITH_TOKENS = 5782

# Below this, the reply is not traceable to the governing rule and is regenerated.
GROUNDING_THRESHOLD = 0.45

NO_RULE_REPLY = (
    "I'm sorry, I can only help with Medi-Cal coverage status on this call. "
    "Thanks for your time — take care!"
)

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


def no_rule_decision(user_message: str, retrieval, elapsed_ms: float = 0.0) -> dict:
    """Deterministic fallback when memory has no relevant chunk.

    This is intentionally polite and terminal: if no rule governs the turn,
    the agent should not improvise from an unrelated nearest neighbor. The
    transcript is still kept, so the between-calls learner can add a real rule
    if this was an important new situation.
    """
    messages = build_messages(
        "NO RELEVANT RULE RETRIEVED. Politely close the call without inventing policy.",
        user_message,
    )
    prompt_sent = render_messages(messages)
    return {
        "prompt_sent": prompt_sent,
        "prompt_sent_tokens": est_tokens(prompt_sent),
        "llm_messages": messages,
        "sent_log": [{"prompt_sent": prompt_sent, "messages": messages, "tokens": est_tokens(prompt_sent)}],
        "llm_calls": 0,
        "caller_message": user_message,
        "assembled_prompt": messages[0]["content"],
        "assembled_prompt_tokens": est_tokens(messages[0]["content"]),
        "monolith_tokens": MONOLITH_TOKENS,
        "saved_pct": (1 - est_tokens(messages[0]["content"]) / MONOLITH_TOKENS) * 100,
        "elapsed_ms": elapsed_ms,
        "turn_json": {
            "intent": retrieval.intent or "none",
            "reply": NO_RULE_REPLY,
            "call_should_end": True,
            "extracted_fields": {},
        },
        "raw_llm_output": "",
        "notes": list(retrieval.notes) + ["no relevant rule; used polite terminal fallback"],
        "outcome": "no-rule",
        "grounded": False,
        "spliced": False,
        "regenerated": False,
        "governing_cosine": 0.0,
        "grounding_threshold": GROUNDING_THRESHOLD,
        "scores": {},
        "intent": retrieval.intent,
        "intent_similarity": retrieval.intent_similarity,
        "intent_ranked": retrieval.intent_ranked,
        "query_text": retrieval.query_text,
        "governing": None,
        "reference": [],
        "call_should_end": True,
        "extracted_fields": {},
        "asked_question": None,
        "retrieval": retrieval,
    }


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
        # Loaded lazily by warm_pool(); until then _retrieve falls back to a
        # live Postgres query per turn, so behaviour is identical either way.
        self.pool_cache = RulePoolCache()

    def warm_pool(self) -> int:
        """Load the whole rule pool into memory so live turns search it with
        Python cosine instead of a Postgres round-trip every turn. Call once
        at boot, and again after the learning loop inserts new rules — the
        pool otherwise only changes between calls (see docs/ARCHITECTURE.md),
        so a cache refreshed at those two points is never stale mid-call.
        """
        with db_engine.connect() as conn:
            return self.pool_cache.refresh(conn, self.table)

    def _retrieve(self, state, message, history):
        if self.pool_cache.loaded:
            return retrieve(
                None, state, message, self.embedder,
                history=history, router=self.router, table=self.table,
                precedence=self.manager.resolve_precedence, pool_cache=self.pool_cache,
            )
        with db_engine.connect() as conn:
            return retrieve(
                conn, state, message, self.embedder,
                history=history, router=self.router, table=self.table,
                precedence=self.manager.resolve_precedence,
            )

    def build_turn_context(self, state, history, user_message):
        """Everything step() does UP TO the LLM call — and nothing more.

        Makes NO completion call. This is the shared path: engine.step() calls it
        for the chat app, and voice_agent.py calls it to build the system prompt
        it hands to LiveKit's streaming LLM. Retrieval, precedence and assembly
        therefore exist in exactly one place, so the prompt a voice caller hears
        is built by the same code as the prompt a chat user sees.

        Returns (prompt_sent, governing, reference, debug).

        `debug["retrieval"]` is the live Retrieval object, kept so callers can
        score a reply against the same rules without re-querying. It holds Chunk
        objects, so it is not JSON-serialisable — drop it before persisting.
        """
        t0 = time.perf_counter()
        retrieval = self._retrieve(state, user_message, history)
        system_prompt = build_turn_prompt(self.stable_core, state, retrieval, history)

        # Captured here, at the moment of assembly, from the same builder the
        # client sends — never rebuilt afterwards from state that has moved on.
        messages = build_messages(system_prompt, user_message)
        prompt_sent = render_messages(messages)
        assert_message_present(prompt_sent, user_message)

        prompt_tokens = est_tokens(system_prompt)
        debug = {
            "system_prompt": system_prompt,
            "prompt_sent": prompt_sent,
            "llm_messages": messages,
            "prompt_sent_tokens": est_tokens(prompt_sent),
            "assembled_prompt": system_prompt,
            "assembled_prompt_tokens": prompt_tokens,
            "monolith_tokens": self.monolith_tokens,
            "saved_pct": (1 - prompt_tokens / self.monolith_tokens) * 100,
            "context_ms": (time.perf_counter() - t0) * 1000,
            "intent": retrieval.intent,
            "intent_similarity": retrieval.intent_similarity,
            "intent_ranked": retrieval.intent_ranked,
            "query_text": retrieval.query_text,
            "opt_out": retrieval.opt_out,
            "notes": list(retrieval.notes),
            "governing": _rule_debug(retrieval.governing, {}) if retrieval.governing else None,
            "reference": [_rule_debug(r, {}) for r in retrieval.reference],
            "retrieval": retrieval,
            "sent_entry": {
                "prompt_sent": prompt_sent,
                "messages": messages,
                "tokens": est_tokens(prompt_sent),
            },
        }
        return prompt_sent, retrieval.governing, retrieval.reference, debug

    def validate_reply(self, reply_text, retrieval) -> dict:
        """Log-only validation of a reply that has ALREADY been delivered.

        Same scoring and same verdicts as the chat path's gate, but it cannot ask
        for a regeneration — by the time the voice path calls this, the words
        have been spoken. The result is recorded so a bad turn is visible after
        the fact rather than silently lost.
        """
        scores = score_reply(reply_text, retrieval.rules, self.embedder)
        outcome, reason = self._judge(scores, retrieval.governing, retrieval)
        gov_id = retrieval.governing.chunk.id if retrieval.governing else None
        return {
            "outcome": outcome,
            "governing_cosine": scores.get(gov_id, {}).get("cosine", 0.0) if gov_id else 0.0,
            "grounded": outcome == "grounded",
            "spliced": outcome == "spliced",
            "reason": reason,
            "scores": scores,
        }

    def prepare_reply(self, state, history, user_message, ctx=None, validate_only=False):
        """Generate the next spoken reply without mutating call state.

        This is the pre-speech half of ``step()``. The chat UI can do it all in
        one blocking call, but the voice path must validate and correct the
        model's JSON decision before TTS speaks anything. Returning the pending
        state changes lets voice persist the turn only after the audio is done.
        """
        start = time.perf_counter()
        notes = []
        sent_log = []

        if ctx is None:
            _, governing, _, ctx = self.build_turn_context(state, history, user_message)
        else:
            governing = ctx["retrieval"].governing
        retrieval = ctx["retrieval"]
        prompt = ctx["system_prompt"]
        notes.extend(ctx["notes"])
        if retrieval.governing is None:
            return NO_RULE_REPLY, no_rule_decision(
                user_message, retrieval, (time.perf_counter() - start) * 1000
            )

        valid, raw = self._decide(
            prompt, user_message, state, notes, sent_log, sent_entry=ctx.get("sent_entry")
        )

        scores = score_reply(valid["reply"], retrieval.rules, self.embedder)
        outcome, reason = self._judge(scores, governing, retrieval)

        regenerated = False
        if reason and validate_only:
            notes.append(f"validate-only: {outcome} recorded, not corrected ({reason})")
        elif reason:
            notes.append(f"regenerating before speech: {reason}")
            prompt = build_turn_prompt(
                self.stable_core, state, retrieval, history, reinforce_reason=reason
            )
            valid, raw = self._decide(prompt, user_message, state, notes, sent_log)
            scores = score_reply(valid["reply"], retrieval.rules, self.embedder)
            outcome, still_wrong = self._judge(scores, governing, retrieval)
            regenerated = True
            if still_wrong:
                notes.append(f"still failing after one regeneration: {still_wrong}")

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

        question = _extract_question(reply_text)
        if question:
            key = question_key(question)
            if key and key in {question_key(q) for q in state.asked_questions}:
                notes.append(f"re-asked a question already in ALREADY ASKED: {question!r}")

        governing_cos = scores.get(governing.chunk.id, {}).get("cosine", 0.0) if governing else 0.0
        final_sent = sent_log[-1]
        debug = {
            "prompt_sent": final_sent["prompt_sent"],
            "prompt_sent_tokens": final_sent["tokens"],
            "llm_messages": final_sent["messages"],
            "sent_log": sent_log,
            "llm_calls": len(sent_log),
            "caller_message": user_message,
            "assembled_prompt": prompt,
            "assembled_prompt_tokens": est_tokens(prompt),
            "monolith_tokens": self.monolith_tokens,
            "saved_pct": (1 - est_tokens(prompt) / self.monolith_tokens) * 100,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "turn_json": valid,
            "raw_llm_output": raw,
            "notes": notes,
            "outcome": outcome,
            "grounded": outcome == "grounded",
            "spliced": outcome == "spliced",
            "regenerated": regenerated,
            "governing_cosine": governing_cos,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "scores": scores,
            "intent": retrieval.intent,
            "intent_similarity": retrieval.intent_similarity,
            "intent_ranked": retrieval.intent_ranked,
            "query_text": retrieval.query_text,
            "governing": _rule_debug(governing, scores) if governing else None,
            "reference": [_rule_debug(r, scores) for r in retrieval.reference],
            "call_should_end": bool(valid["call_should_end"]),
            "extracted_fields": dict(valid["extracted_fields"]),
            "asked_question": question,
            "retrieval": retrieval,
        }
        return reply_text, debug

    def _decide(self, prompt, user_message, state, notes, sent_log, sent_entry=None):
        # The first call of a turn reuses the capture build_turn_context already
        # made; a regeneration captures its own, since its prompt differs.
        if sent_entry is None:
            messages = build_messages(prompt, user_message)
            prompt_sent = render_messages(messages)
            assert_message_present(prompt_sent, user_message)
            sent_entry = {
                "prompt_sent": prompt_sent,
                "messages": messages,
                "tokens": est_tokens(prompt_sent),
            }
        sent_log.append(sent_entry)

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

    def step(self, state, history, user_message, validate_only=False):
        """The chat turn: retrieve, decide, validate, apply.

        `validate_only=True` records a failed validation instead of acting on it —
        no regeneration. The default is False, so the Streamlit chat app's
        blocking behaviour is unchanged.
        """
        start = time.perf_counter()
        notes = []
        # One entry per LLM call this turn, appended at call time. A regeneration
        # adds a second entry, so both payloads stay inspectable.
        sent_log = []

        # 1 + 2. Retrieval and assembly, shared with the voice path.
        _, governing, _, ctx = self.build_turn_context(state, history, user_message)
        retrieval = ctx["retrieval"]
        prompt = ctx["system_prompt"]
        notes.extend(ctx["notes"])
        if retrieval.governing is None:
            final = no_rule_decision(
                user_message, retrieval, (time.perf_counter() - start) * 1000
            )
            state.intent = retrieval.intent or "none"
            state.opt_out = state.opt_out or retrieval.opt_out
            state.ended = True
            history.append(f"Caller: {user_message}")
            history.append(f"Maya: {NO_RULE_REPLY}")
            final["state_snapshot"] = {
                "intent": state.intent,
                "opt_out": state.opt_out,
                "ended": state.ended,
                "asked_questions": list(state.asked_questions),
                "collected_fields": dict(state.collected_fields),
            }
            return NO_RULE_REPLY, state, final

        # 3. One structured decision, reusing the capture made during assembly.
        valid, raw = self._decide(
            prompt, user_message, state, notes, sent_log, sent_entry=ctx["sent_entry"]
        )

        # 4. Validate the reply against the rule that was supposed to govern it.
        scores = score_reply(valid["reply"], retrieval.rules, self.embedder)
        outcome, reason = self._judge(scores, governing, retrieval)

        regenerated = False
        if reason and validate_only:
            notes.append(f"validate-only: {outcome} recorded, not corrected ({reason})")
        elif reason:
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

        # 5. `terminal` decides whether the call ends — in both directions. The
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

        # 6. Apply to state.
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
