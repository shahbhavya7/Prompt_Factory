"""The turn loop: retrieve -> one structured LLM call -> validate -> apply.

Retrieval is memory-only (see retrieve.py): no stage, no branch logic, one flat
pool. That puts the whole weight of correctness on the governing rule actually
governing, so the reply is checked against it before it is accepted:

  a. it must be semantically close to the governing rule (GROUNDING_THRESHOLD);
  b. having cleared (a), it must not lean on a reference rule by a decisive
     margin (SPLICE_MARGIN), which is what a spliced-in sentence looks like from
     the outside. The margin matters: reference rules are topically adjacent by
     construction, so a bare "closer to a reference rule" test fires on noise
     and rejects correct replies.
  c. a terminal rule forces the call to end, and nothing may trail the closing.

A failure at (a) or (b) buys exactly one regeneration with a correction block.
Anything still failing after that is reported rather than hidden.
"""

import os
import re
import time

from sace_chat import answer_cache
from sace_chat.assemble import build_turn_prompt
from sace_chat.db import engine as db_engine
from sace_chat.llm import build_messages, parse_json_object, render_messages
from sace_chat.retrieve import IntentRouter, retrieve
from sace_chat.tokens import est_tokens

# Pinned monolith baseline for the savings comparison (data/base_prompt_coverage.txt).
MONOLITH_TOKENS = 5782

# Below this, the reply is not traceable to the governing rule and is regenerated.
GROUNDING_THRESHOLD = 0.45

# How much MORE a reference rule must score than the governing rule before the
# reply is called a splice. Only consulted once the reply has already cleared
# GROUNDING_THRESHOLD against its governing rule, so this is never the thing
# that lets an ungrounded reply through — it only decides whether a *grounded*
# reply is nonetheless leaning on the wrong rule.
#
# A bare `ref_cos > governing_cos` (this was the original test) has no margin at
# all, so it fires on ties and on noise. Reference rules are selected for
# topical adjacency, and on short replies the gap between two adjacent rules is
# routinely a few thousandths in either direction — which is not evidence of
# anything. 0.08 is wide enough to ignore that band and narrow enough to still
# catch a reply that genuinely answered from the reference rule.
SPLICE_MARGIN = float(os.environ.get("SACE_SPLICE_MARGIN", "0.08"))

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


def cached_decision(user_message: str, retrieval, elapsed_ms: float = 0.0) -> dict:
    """The turn, served from a confirmed reply already in the cache.

    Zero LLM calls and zero prompt assembly: the reply was generated and
    grounding-checked on an earlier turn, and `answer_cache.lookup` has already
    confirmed the caller's message is a near-restatement of the question that
    produced it (>= CACHE_THRESHOLD, far stricter than the retrieval bar).

    The debug shape deliberately matches what a normal turn returns, so the
    dashboard, `turns` row and latency line need no special-casing — a cache
    hit is a turn like any other, just a much faster one. `outcome` is
    "cached" rather than "grounded": the reply was grounded when it was
    stored, and re-scoring it here would spend the embedding call the cache
    exists to avoid.
    """
    hit = retrieval.cache_hit
    governing = retrieval.governing
    reply = hit["reply"]

    # A cache hit sends no payload, but it must still be inspectable — the
    # transparency guarantee is that every turn can show what produced its
    # reply, and "nothing, it was replayed" is an answer that still has to be
    # shown rather than left as an empty box. So a real message list is built
    # describing the replay, and it carries the caller's message verbatim so
    # assert_message_present-style checks hold on cached turns too.
    explain = (
        f"CACHE HIT — no prompt was assembled and no model was called.\n"
        f"This reply was generated on an earlier turn, passed the grounding "
        f"check against {hit['governing_rule_id']}, and was stored for reuse.\n"
        f"Stored question: {hit['question']}\n"
        f"Similarity to this turn: {hit['similarity']:.4f} "
        f"(bar: {answer_cache.CACHE_THRESHOLD})\n"
        f"Pending question it is pinned to: "
        f"{hit['pending_fingerprint'] or '(none — reusable on any turn)'}\n"
        f"Replayed reply: {reply}"
    )
    messages = build_messages(explain, user_message)
    prompt_sent = render_messages(messages)
    # Terminal rules are never cached (answer_cache.is_cacheable), so a cached
    # turn never ends the call. Asserted rather than assumed: serving a stored
    # reply that hung up on someone would be the worst failure mode here.
    ends = bool(governing and governing.chunk.terminal)
    return {
        "prompt_sent": prompt_sent,
        # Zero, deliberately: no prompt tokens were spent this turn. The
        # explanation above is not a payload that was sent anywhere.
        "prompt_sent_tokens": 0,
        "llm_messages": messages,
        "sent_log": [{"prompt_sent": prompt_sent, "messages": messages, "tokens": 0}],
        "llm_calls": 0,
        "caller_message": user_message,
        "assembled_prompt": "",
        "assembled_prompt_tokens": 0,
        "monolith_tokens": MONOLITH_TOKENS,
        # The whole prompt was avoided, so the saving against the monolith is total.
        "saved_pct": 100.0,
        "elapsed_ms": elapsed_ms,
        "turn_json": {
            "intent": retrieval.intent or "none",
            "reply": reply,
            "call_should_end": ends,
            "extracted_fields": {},
        },
        "raw_llm_output": "",
        "notes": list(retrieval.notes),
        "outcome": "cached",
        "grounded": True,
        "spliced": False,
        "regenerated": False,
        "governing_cosine": hit["similarity"],
        "grounding_threshold": GROUNDING_THRESHOLD,
        "scores": {},
        "intent": retrieval.intent,
        "intent_similarity": retrieval.intent_similarity,
        "intent_ranked": retrieval.intent_ranked,
        "query_text": retrieval.query_text,
        "governing": _rule_debug(governing, {}) if governing else None,
        "reference": [],
        "call_should_end": ends,
        "extracted_fields": {},
        "asked_question": _extract_question(reply),
        "retrieval": retrieval,
        "cache_hit": hit,
    }


def _apply_field_defaults(governing, valid: dict, state) -> None:
    """Deterministically fill in a flow rule's own `sets` defaults for any
    field the model did not itself extract this turn — see models.Chunk.sets
    for why this cannot be left to the model's own initiative.

    Never overwrites a field already known from an earlier turn: which rule
    governs only answers the question the FIRST time it is reached (e.g.
    already_submitted_check's own default of packet_received=True is only
    right for the direct "yes it arrived" path — on the address-correction
    path packet_received is already False from address_capture, and a
    same-turn default must not clobber that).
    """
    if governing is None:
        return
    sets = getattr(governing.chunk, "sets", None)
    if not sets:
        return
    known = getattr(state, "collected_fields", None) or {}
    for field_name, default in sets.items():
        if field_name in valid["extracted_fields"] or field_name in known:
            continue
        valid["extracted_fields"][field_name] = default


_DIGIT_RE = re.compile(r"\d+")


def _digit_tokens(text: str) -> set[str]:
    """Every run of digits in `text`, e.g. "March 8th, 1983" -> {"8", "1983"}.

    Deliberately loose: callers say a date of birth in wildly different
    formats ("March 8th, 1983", "3/8/1983", "08-03-1983") and a case record
    is unlikely to be stored in exactly the same one. Comparing digit tokens
    rather than the full string sidesteps format-matching entirely — see
    _apply_identity_derivation.
    """
    return set(_DIGIT_RE.findall(text or ""))


def _dob_matches_case_record(extracted: str, on_record: str) -> bool:
    """Loose date-of-birth match: at least two digit tokens in common (e.g.
    day and year), so "March 8th, 1983" matches "1983-03-08" without either
    side needing a canonical format. A single shared token (e.g. just the
    year) is not enough — too many birthdates share a year."""
    extracted_digits = _digit_tokens(extracted)
    record_digits = _digit_tokens(on_record)
    if not extracted_digits or not record_digits:
        return False
    return len(extracted_digits & record_digits) >= 2


def _apply_identity_derivation(valid: dict, state) -> None:
    """Derive identity_verified by comparing an extracted date_of_birth
    against this call's own case record — never taking the model's word for
    it, and never stamped as a rule default (see models.Chunk.sets and
    scripts/build_kb_renewal.py's Phase 2E note on why identity_verified
    does not appear in any flow rule's `sets`).

    A caller who never gives a date of birth, or gives one that does not
    match the record, leaves identity_verified unset — which keeps
    verify_dob/packet_check's own `requires` gate open so the flow can only
    ask again, never silently assume the caller is who they claim.
    """
    if "identity_verified" in valid["extracted_fields"]:
        return
    extracted_dob = valid["extracted_fields"].get("date_of_birth")
    if not extracted_dob:
        return
    on_record = (getattr(state, "case_record", None) or {}).get("date_of_birth")
    if on_record and _dob_matches_case_record(str(extracted_dob), str(on_record)):
        valid["extracted_fields"]["identity_verified"] = True


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
                 monolith_text=None, table="chunks", chunks=None,
                 never_say_guard=None, never_say_fallback="",
                 cache_table="answer_cache"):
        self.stable_core = stable_core
        self.rules = rules if rules is not None else (chunks or [])
        self.embedder = embedder
        self.manager = manager
        self.llm = llm
        self.monolith_tokens = est_tokens(monolith_text) if monolith_text else MONOLITH_TOKENS
        self.table = table
        # Which reply-cache table this campaign reads and writes — see
        # campaign.py's CampaignConfig.cache_table and db.py's
        # answer_cache_renewal (Phase 2E: previously every campaign shared
        # "answer_cache", which let a coverage-learned reply on one of the
        # five safety intents both campaigns reuse verbatim — dnc, abuse,
        # medical_emergency, garbled_audio, frustration — be served on a
        # renewal call under the wrong clinic's script).
        self.cache_table = cache_table
        self.router = IntentRouter(embedder)
        # Optional (reply_text, case_record) -> (ok, reason) predicate, run in
        # prepare_reply only — see that method. None (the default) disables it
        # entirely, which is what every existing construction site gets: this
        # is a renewal-campaign-specific compliance concern, not a change to
        # coverage's shipped behaviour. See sace_chat/guards.py.
        self._never_say_guard = never_say_guard
        self._never_say_fallback = never_say_fallback

    def _retrieve(self, state, message, history):
        with db_engine.connect() as conn:
            return retrieve(
                conn, state, message, self.embedder,
                history=history, router=self.router, table=self.table,
                cache_table=self.cache_table,
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

        # THE FAST PATH. Everything below this point — assembly, the completion
        # call, scoring, the regeneration budget — is what a hit skips, and it
        # is where all of the turn's latency lives.
        if retrieval.cache_hit is not None:
            dbg = cached_decision(
                user_message, retrieval, (time.perf_counter() - start) * 1000
            )
            answer_cache.record_hit(retrieval.cache_hit["id"], table=self.cache_table)
            return dbg["turn_json"]["reply"], dbg

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

        # The never-say guard: code-enforced, because prose in the prompt is
        # not how this codebase trusts anything load-bearing to hold (see
        # guards.check_never_say's docstring). Disabled entirely unless a
        # guard was passed in at construction — every existing coverage
        # engine gets None here and this block is a no-op for it.
        if self._never_say_guard is not None:
            ok, why = self._never_say_guard(reply_text, state.case_record)
            if not ok and not regenerated:
                # Reuses the SAME one-retry budget the grounding check has —
                # not a second regeneration on top of it. If grounding already
                # spent it, this falls straight to the deterministic fallback
                # below instead of a third LLM call.
                notes.append(f"regenerating before speech (never-say): {why}")
                prompt = build_turn_prompt(
                    self.stable_core, state, retrieval, history,
                    reinforce_reason=(
                        f"Your reply violated a hard safety rule: {why}. Never state a "
                        f"dollar amount, income limit, eligibility determination or "
                        f"deadline unless it is copied verbatim from the case record "
                        f"shown to you. Rewrite the reply without it."
                    ),
                )
                valid, raw = self._decide(prompt, user_message, state, notes, sent_log)
                reply_text = valid["reply"]
                regenerated = True
                ok, why = self._never_say_guard(reply_text, state.case_record)
            if not ok:
                notes.append(f"never-say violation survived the regeneration budget; "
                             f"forced a safe fallback instead of speaking it: {why}")
                if governing is not None and governing.chunk.transfer:
                    from sace_chat.assemble import _substitute_placeholders

                    reply_text = _substitute_placeholders(governing.chunk.text)
                else:
                    reply_text = self._never_say_fallback
                if governing is not None and governing.chunk.terminal:
                    reply_text, _ = strip_after_terminal(reply_text)
                valid["reply"] = reply_text
                # A forced fallback is not the scored reply any more — scores
                # and outcome above describe text that is no longer what gets
                # spoken. Relabelled so _maybe_cache's is_cacheable check
                # (outcome != "grounded" -> refuse) never stores it: whatever
                # this fallback is, it is a safety escape hatch, not a
                # confirmed-good answer worth replaying to the next caller.
                outcome = "never-say-fallback"

        # prepare_reply does not mutate state itself (see the voice-path note
        # below), but the voice path applies extracted_fields from THIS dict
        # once the audio finishes — see voice_agent.py's finish_turn — so the
        # same deterministic defaults have to land here too, not only in
        # step()'s own chat-path copy of this logic.
        _apply_identity_derivation(valid, state)
        _apply_field_defaults(governing, valid, state)

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
        # history has NOT yet been appended for this turn (finish_turn does it
        # on the voice path), so it holds only completed turns.
        # prepare_reply does not mutate state, so the pending question is still
        # exactly what it was when the reply was produced.
        self._maybe_cache(user_message, reply_text, debug,
                          turn_index=len(history) // 2 + 1,
                          pending_before=answer_cache.pending_fingerprint(state))
        return reply_text, debug

    def _maybe_cache(self, user_message, reply_text, debug, turn_index=2,
                     pending_before=""):
        """Store a confirmed turn so the next caller asking this skips the LLM.

        Reuses `retrieval.message_vec` — the vector retrieval already computed
        for this turn — so storing costs one small insert and no embedding call.
        Every exclusion is in answer_cache.is_cacheable; this only plumbs it.
        """
        retrieval = debug.get("retrieval")
        if retrieval is None:
            # A caller built a debug dict without the Retrieval. Say so rather
            # than silently never caching — that failure is invisible from the
            # outside (everything works, just slowly) and cost real debugging
            # time once already.
            debug.setdefault("notes", []).append(
                "not cached: debug carries no retrieval object"
            )
            return
        if retrieval.cache_hit is not None:
            return  # a hit is already stored, by definition
        if retrieval.message_vec is None:
            debug.setdefault("notes", []).append("not cached: no message vector")
            return
        governing = retrieval.governing
        # Passed in explicitly rather than derived from history, because the two
        # callers differ: prepare_reply has not yet appended this turn's lines,
        # step() already has. Counting here would need a different rule per
        # caller and silently break if either moved.
        ok, why = answer_cache.is_cacheable(
            governing=governing,
            outcome=debug.get("outcome", ""),
            regenerated=bool(debug.get("regenerated")),
            extracted_fields=debug.get("extracted_fields"),
            reply=reply_text,
            intent=retrieval.intent,
            turn_index=turn_index,
            question=user_message,
        )
        if not ok:
            debug.setdefault("notes", []).append(f"not cached: {why}")
            # Structured too, so a UI can say WHY a turn was not saved rather
            # than leaving the reader to guess from a notes string.
            debug["cache_stored"] = {"stored": False, "reason": why}
            return
        # The trailing-question gate, store side. A diversion reply typically
        # ends by returning to the question already pending; that half is call
        # state, so the entry is pinned to it and only served when the same
        # question is pending again (see answer_cache).
        #
        # The reply's trailing question is required to BE the pending one. If it
        # trails some other question, the model did not merely return to the
        # script — it advanced or invented, and the reply is not a reusable
        # answer-plus-re-ask. Refusing there is what keeps a stored entry
        # meaning exactly "this fixed fact, then this specific pending question".
        trailing = answer_cache.trailing_question(reply_text)
        pending = pending_before
        if trailing is not None:
            trailing_fp = answer_cache.question_fingerprint(trailing)
            if trailing_fp != pending:
                why = (
                    f"reply ends on a question that was not the pending one "
                    f"({trailing!r}) — it advanced the script rather than "
                    f"returning to it, so it is not replayable"
                )
                debug.setdefault("notes", []).append(f"not cached: {why}")
                debug["cache_stored"] = {"stored": False, "reason": why}
                return
        else:
            # Ends on no question: a bare statement of fact, reusable on any
            # turn. Stored with an empty fingerprint, which matches anything.
            pending = ""

        cache_id = answer_cache.store(
            question=user_message,
            question_vec=retrieval.message_vec,
            reply=reply_text,
            intent=retrieval.intent,
            governing_rule_id=governing.chunk.id if governing else None,
            grounding_cosine=debug.get("governing_cosine"),
            pending=pending,
            table=self.cache_table,
        )
        if cache_id:
            debug.setdefault("notes", []).append(f"cached as {cache_id} for reuse")
            debug["cache_stored"] = {
                "stored": True,
                "id": cache_id,
                "question": user_message,
                "intent": retrieval.intent,
                # Empty means "reusable on any turn"; a value means the entry is
                # pinned to that pending question.
                "pending": pending,
            }
        else:
            debug["cache_stored"] = {"stored": False, "reason": "store failed"}

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

        # THE FAST PATH — steps 3, 4 and the regeneration budget are all skipped.
        # Only a turn whose reply does not depend on call state is ever cached
        # (answer_cache.is_cacheable), so applying state here is deliberately
        # minimal: nothing was extracted and the call cannot have ended.
        if retrieval.cache_hit is not None:
            final = cached_decision(
                user_message, retrieval, (time.perf_counter() - start) * 1000
            )
            answer_cache.record_hit(retrieval.cache_hit["id"], table=self.cache_table)
            reply_text = final["turn_json"]["reply"]
            state.intent = retrieval.intent or "none"
            state.opt_out = state.opt_out or retrieval.opt_out
            asked = final.get("asked_question")
            if asked:
                key = question_key(asked)
                if key and key not in {question_key(q) for q in state.asked_questions}:
                    state.asked_questions.append(asked)
            history.append(f"Caller: {user_message}")
            history.append(f"Maya: {reply_text}")
            final["state_snapshot"] = {
                "intent": state.intent,
                "opt_out": state.opt_out,
                "ended": state.ended,
                "asked_questions": list(state.asked_questions),
                "collected_fields": dict(state.collected_fields),
            }
            return reply_text, state, final

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
        # Snapshot BEFORE the mutation below. `state.asked_questions` gets this
        # turn's question appended a few lines down, and _maybe_cache needs the
        # question that was pending when the reply was PRODUCED — which is the
        # last entry as of now. Reading it afterwards would sometimes return the
        # reply's own question, and the store-side gate would then compare that
        # question against itself and pass everything.
        pending_before = answer_cache.pending_fingerprint(state)

        state.intent = retrieval.intent or "none"
        state.opt_out = state.opt_out or retrieval.opt_out
        state.ended = state.ended or valid["call_should_end"]
        _apply_identity_derivation(valid, state)
        _apply_field_defaults(governing, valid, state)
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
            # The live Retrieval, same as build_turn_context and prepare_reply
            # expose. _maybe_cache needs it for the message vector it already
            # computed; holds Chunk objects, so drop it before persisting.
            "retrieval": retrieval,
        }
        # history HAS already been appended for this turn (above), so the two
        # lines for it are included — hence //2 rather than //2 + 1.
        self._maybe_cache(user_message, reply_text, debug,
                          turn_index=len(history) // 2,
                          pending_before=pending_before)
        return reply_text, state, debug

    def _judge(self, scores, governing, retrieval) -> tuple[str, str]:
        """(outcome, reason-to-regenerate). An empty reason means accept."""
        if governing is None:
            return "no-rule", ""
        if not scores:
            return "unscored", ""

        governing_cos = scores.get(governing.chunk.id, {}).get("cosine", 0.0)

        if governing_cos < GROUNDING_THRESHOLD:
            # (a) The reply is not traceable to its governing rule at all. Check
            # this FIRST: it is the absolute failure, and a reply that fails it
            # is wrong whatever the reference rules scored. If a reference rule
            # outscored the governing one here, name it — that is the likely
            # source of the drift and it makes the correction block concrete.
            culprit = max(
                retrieval.reference,
                key=lambda r: scores.get(r.chunk.id, {}).get("cosine", 0.0),
                default=None,
            )
            culprit_cos = (
                scores.get(culprit.chunk.id, {}).get("cosine", 0.0) if culprit else 0.0
            )
            if culprit is not None and culprit_cos > governing_cos:
                return "spliced", (
                    f"the reply is not traceable to the GOVERNING rule "
                    f"{governing.chunk.id} (cosine {governing_cos:.3f}, needs "
                    f"{GROUNDING_THRESHOLD}), and matched the REFERENCE rule "
                    f"{culprit.chunk.id} (cosine {culprit_cos:.3f}) more closely. "
                    f"You took content from REFERENCE instead of answering from "
                    f"GOVERNING."
                )
            return "ungrounded", (
                f"the reply is not traceable to the GOVERNING rule {governing.chunk.id} "
                f"(cosine {governing_cos:.3f}, needs {GROUNDING_THRESHOLD}). You spoke "
                f"something the rule does not contain."
            )

        # (b) The reply IS traceable to the governing rule. A reference rule
        # scoring marginally higher is not by itself a defect — reference rules
        # are in the prompt precisely because they are topically adjacent, and
        # on a short reply ("Yes, that's right.") the cosine ordering between
        # two adjacent rules is close to noise.
        #
        # This check used to run BEFORE (a) and with no absolute floor, so ANY
        # ref_cos > governing_cos was called "spliced" — which meant a reply at
        # 0.85 to its governing rule was rejected because a reference rule hit
        # 0.86. Measured on live calls that fired on roughly a third of turns,
        # every one of them paying for a regeneration and none of them reaching
        # the answer cache. The reply was almost always correct.
        #
        # What IS a real splice, once the floor is cleared: the reply leans on a
        # reference rule by a decisive margin, meaning it answered from the
        # wrong rule and merely brushed past the right one.
        for ref in retrieval.reference:
            ref_cos = scores.get(ref.chunk.id, {}).get("cosine", 0.0)
            # 1e-9 absorbs binary-float error: 0.58 - 0.50 evaluates to
            # 0.07999999999999996, which would miss a >= 0.08 test on a case
            # that is exactly at the margin by intent.
            if ref_cos - governing_cos >= SPLICE_MARGIN - 1e-9:
                return "spliced", (
                    f"the reply matched the REFERENCE rule {ref.chunk.id} (cosine {ref_cos:.3f}) "
                    f"decisively more closely than the GOVERNING rule {governing.chunk.id} "
                    f"(cosine {governing_cos:.3f}, margin {ref_cos - governing_cos:.3f} "
                    f">= {SPLICE_MARGIN}). You took content from REFERENCE."
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
