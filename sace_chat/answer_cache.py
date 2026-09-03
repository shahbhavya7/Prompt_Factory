"""A reply cache in front of the turn loop: skip the LLM when we already know.

The idea (Karpathy's "LLM wiki"): once a reply has been generated AND validated
as grounded, the pair (what the caller said -> what we correctly answered) is
worth keeping. A later caller saying almost the same thing gets the stored reply
directly, with no prompt assembly and no completion call.

WHY A MISS IS NEARLY FREE — this is the design's load-bearing property.

The obvious implementation makes the cache a standalone pre-check that embeds
the caller's message itself. That is the one thing to avoid: it adds a full
embedding round-trip (a network call, ~100-300ms) to EVERY turn, and on a miss
that cost is pure waste on top of the normal path. It would make the common
case slower to make the lucky case faster.

Instead `lookup` takes an ALREADY-COMPUTED vector. retrieve.py embeds the
caller's message on every turn regardless — it needs that vector to classify
intent — so the cache reuses it. A miss therefore costs exactly one extra
pgvector query against a small, intent-scoped table: single-digit milliseconds
against a ~250ms context build and a ~2000ms LLM call. There is no meaningful
worst case, only a best case that is dramatically better.

The corollary is an ordering constraint: the cache CANNOT be checked before
embedding, so it cannot skip the embed. It skips assembly, the LLM call, the
grounding check, and (on the voice path) the regeneration budget — which is
where essentially all of the latency actually lives.

WHAT IS DELIBERATELY NEVER CACHED

A cached reply is context-free: it cannot know what this call already asked or
already collected. So a turn is only cacheable when its reply does not depend on
call state, and never when getting it wrong is unrecoverable:

  * seed flow rules     — a SEED rule with no intent is part of the call SCRIPT,
                          and its reply is a function of where in the call it is
                          rather than of what the caller asked. This is the
                          structural rule, and the one that makes the default
                          "deny" for rules nobody has written yet. The `seed`
                          qualifier matters: on a LEARNED rule intent=None means
                          "the consolidator could not classify it", not
                          "positional" — see NEVER_CACHE_RULES.
  * terminal rules      — a closing ends the call; replaying one by cosine
                          accident hangs up on someone mid-conversation.
  * critical priority   — dnc, abuse, medical_emergency. Compliance and safety
                          turns get the full pipeline every single time. This is
                          the explicit safety rule, enforced in code, not by
                          convention.
  * field extraction    — a reply that captured a county/name/number is about
                          THAT caller; replaying it would assert a stranger's
                          data back at someone.
  * name leakage        — a reply naming THIS caller is personal to one call.
                          Handled by folding the name back to its placeholder
                          on store and re-substituting on serve, so the entry
                          is reusable and the row holds no name to leak; only a
                          name normalise could not fold is a refusal. Note the
                          other placeholders (callback number, clinic, current
                          month) are campaign constants, NOT per-caller, and a
                          reply quoting them is exactly what should be cached.
  * regenerated turns   — the first attempt was rejected; the second is not
                          evidence of a reliably good answer.
  * ungrounded/spliced  — only `grounded` outcomes are ever stored.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import text as sql_text

from sace_chat.assemble import DEMO_PLACEHOLDERS
from sace_chat.db import SessionLocal, check_embedding

# How close a caller's message must be to a stored question to reuse its reply.
#
# Far stricter than the 0.45 retrieval threshold, and for a structural reason:
# retrieval picks the best of several rules and an LLM then adapts the wording
# to the actual utterance, whereas a cache hit replays fixed words with no
# adaptation step. "Close enough to route" is nowhere near "close enough to say
# verbatim".
#
# MEASURED on text-embedding-3-small, against a realistically-phrased stored
# question ("quick thing, is this call being recorded" — real callers prefix
# their questions, and the prefix dilutes the vector, so measuring against a
# bare question overstates the scores):
#
#   the SAME question, re-asked        0.764 - 0.875
#     "is this call being recorded" 0.875 · "wait, is this call recorded" 0.856
#     "so this call is being recorded?" 0.832 · "sorry, is the call being
#     recorded" 0.800 · "is this being recorded" 0.764
#
#   a DIFFERENT question               0.120 - 0.538
#     "who is this calling" 0.538 · "is my daughter allowed on the call" 0.480
#     "can you call me back tomorrow" 0.419 · "stop calling me" 0.332
#
# Those bands do not overlap, and 0.68 sits in the empty gap between them:
# 0.08 below the weakest true restatement, 0.14 above the strongest impostor.
#
# Two earlier values were wrong in opposite directions, both because they were
# guessed rather than measured: 0.93 essentially never fired (the cache existed
# but did nothing), and 0.82 sat INSIDE the same-question band, rejecting about
# half of genuine restatements.
#
# The asymmetry still governs the choice of margin: a false hit speaks the wrong
# words to a caller, while a false miss merely costs the latency we were paying
# anyway. So when in doubt this errs high.
CACHE_THRESHOLD = float(os.environ.get("SACE_CACHE_THRESHOLD", "0.68"))

# How close two STORED questions must be before the second overwrites the first
# instead of becoming its own row. Strictly higher than CACHE_THRESHOLD, and the
# gap between them is the point.
#
# Both bars were CACHE_THRESHOLD, which quietly capped how much the cache could
# ever learn. Anything similar enough to be SERVED from an existing row is, by
# definition, also similar enough to REPLACE it — so the table could never hold
# two neighbouring phrasings of one question. Each new phrasing overwrote the
# last, and the row's stored question wandered around the cluster instead of the
# cluster accumulating coverage of it. Worse, a phrasing sitting just above 0.68
# from the stored row but below it from the NEXT caller's wording would keep
# taking that row over and losing the wording that was actually getting hits.
#
# At 0.93 only a near-duplicate collapses (a re-transcription, an added "um"),
# which is genuinely one entry. Distinct phrasings each get a row, so the
# intent's section fills in around the question and coverage rises with use —
# which is the behaviour that makes the cache worth having. The cost is a few
# more rows per intent; sections are small and intent-scoped, so the lookup is
# unaffected.
DEDUP_THRESHOLD = float(os.environ.get("SACE_CACHE_DEDUP_THRESHOLD", "0.93"))

# Intents that must always run the full pipeline, whatever their rule flags say.
# Kept as an explicit list rather than derived only from rule metadata so the
# safety rule survives someone flipping `terminal` off on a rule by mistake.
NEVER_CACHE_INTENTS = frozenset({"dnc", "abuse", "complaint_escalation"})

# Tiers that must never be cached, in ANY campaign that uses a `tier` column
# (renewal's T2/T4 today). T2 is this caller's own case record — replaying it
# asserts a stranger's data back at someone; T4 is immigration/enforcement and
# self-harm/distress, where a stale cached line is a safety failure, not a
# UX one. Checked in both store() and lookup() — a row that reaches the table
# by some future path (a migration, a bulk load bug) must still not be
# servable, not just un-writable.
NEVER_CACHE_TIERS = frozenset({"T2", "T4"})

# THE STRUCTURAL RULE: a cacheable turn is an intent-routed diversion.
#
# This replaced a hand-maintained blocklist of rule ids, and the reason is
# dependability rather than tidiness. The blocklist was a list of the flow rules
# somebody had thought of, which made it wrong in both directions at once: it
# named 16 rules and still missed several real flow rules (`still_has_benefits_
# close`, `counselor_ack_close`, `send_details_by_text`, `retry_line`,
# `wrong_person_close`, `counselor_declined_close`), and any rule added to kb.py
# later defaulted to CACHEABLE — a new flow rule would silently become
# replayable, which is exactly the failure this module exists to prevent. A
# safety filter whose default is "allow" is not a safety filter.
#
# The rule set already encodes the distinction properly, so the check reads it
# off the data instead. Every rule in kb.py is one of two kinds:
#
#   intent=None      the CALL SCRIPT. Its reply is a function of position in the
#                    flow, not of what the caller asked. "hello? who is this"
#                    and "hello? who's calling please" are the same question,
#                    but the right reply is the greeting on turn 1 and something
#                    else entirely later. Replaying one restarts the script.
#                    All 21 of these, without exception.
#
#   intent=<label>   a DIVERSION — the caller asked something off-script and the
#                    reply is a fixed fact ("Yes, this call is recorded", "that's
#                    one for the coverage counselors, text KEEP"). Position-
#                    independent by construction: the whole point of a diversion
#                    rule is that it answers the same way wherever it fires, then
#                    hands control back to the pending question.
#
# So: a SEED rule with intent=None is never cacheable, and that holds for seed
# rules nobody has written yet. A new flow rule is excluded the moment it exists,
# with no list to update.
#
# THE SEED QUALIFIER IS LOAD-BEARING, and leaving it out is a bug this had. The
# rule above reads "intent=None means call script", which is exactly true of the
# 21 hand-written rules in kb.py — and false of learned rules. The consolidator
# leaves `intent` NULL whenever the situation it extracted does not map onto the
# fixed router vocabulary (manager.VALID_INTENTS), so for a learned rule
# intent=None means UNCLASSIFIED, not POSITIONAL. Measured on this pool: 20 of 38
# learned rules had intent=None, and they were plainly FAQ-shaped — "asks about
# services or amenities at the clinic", "asks about specific treatment coverage",
# "expresses concern about privacy", "doubt about the legitimacy of the call".
# Treating those as flow rules silently excluded the single largest group of
# genuinely cacheable turns, which is how a live call could run end to end with
# every FAQ answered and nothing stored.
#
# A learned rule is a diversion by construction: the consolidator writes one
# BECAUSE a caller went off-script, so it was never part of the flow whose
# position could matter. `source` is therefore the right discriminator, and the
# two halves of the check are:
#
#   source == "seed"    and intent is None  ->  call script, never cacheable
#   source == "learned"                     ->  a diversion; cacheable, subject
#                                               to every other gate below
#
# Note this is a property of the RULE, checked against `governing.chunk`, not of
# the router's label for the turn. They normally agree — retrieval fetches by
# intent — but the rule's own fields are what say whether its text is
# position-dependent, so that is what is trusted.
#
# A handful of intent-labelled rules are still unsafe for reasons of their own,
# and those are named below. That list is allowed to be a list: it is a set of
# specific exceptions to a rule that already defaults to "deny", not the rule
# itself.
NEVER_CACHE_RULES = frozenset({
    # Safety and compliance. Belt and braces — these are also critical priority
    # and/or on NEVER_CACHE_INTENTS, and the point is that all three checks have
    # to be removed before one of these can ever be replayed.
    "medical_emergency", "special_dnc", "special_abuse",
    "special_complaint_escalation",

    # Intent-routed, but the reply is about THIS caller's data, not a fixed
    # fact. Each collects or commits to something specific — a day and time, a
    # third party's number and name, a language preference recorded for the
    # callback. `extracted_fields` catches most of these turns anyway; naming
    # the rules too means a turn that happened to extract nothing still cannot
    # be stored.
    "special_callback_request", "special_redirect", "special_language",

    # Intent-routed, but the reply carries no fixed fact of its own — it is
    # ONLY a re-ask of whatever was pending ("Sorry, trouble hearing you — once
    # more?", "I hear you — let me keep this really short."). There is no
    # position-independent half to cache, so the trailing-question gate below
    # cannot rescue them the way it does the FAQ rules.
    "special_garbled_audio", "special_frustration",
})

# ───────────────────────────── the serve decision ────────────────────────────
#
# WHY THIS IS A DATACLASS AND NOT THREE MODULE CONSTANTS.
#
# scripts/measure_cache_bar_renewal.py measures this campaign's real cosine
# behaviour against held-out paraphrases and reports the bar that produces zero
# wrong serves. Until this existed, it reported it into
# data/renewal/eval/cache_bar.json and NOTHING READ THE FILE: lookup() served on
# a single global threshold with no margin and no tier check, so the measurement
# was decorative and the safety property it demonstrated was not in force.
#
# Measured on the 126-paraphrase run that produced that file: at margin 0.0 the
# decision made 1 false serve and 1 CROSS-TIER serve; at margin 0.02, zero of
# both. A cross-tier serve is the one that matters — it answers a T3 question
# ("will I qualify?", whose only correct reply is a handoff) with a confident T1
# fixed answer, or the reverse. That is not a stale-cache annoyance; it is Maya
# stating something she is explicitly forbidden to state.
#
# So the bar travels WITH the campaign (see campaign.CampaignConfig.cache_bar),
# is loaded from the measured file rather than retyped, and the default below
# reproduces the coverage campaign's existing behaviour exactly — margin 0, no
# tier agreement — so adding this changes nothing for a campaign that has not
# measured its own.
@dataclass(frozen=True)
class CacheBar:
    """How close, and how much closer than the runner-up, a serve requires.

    threshold   minimum cosine to the nearest stored question. On its own this
                is a bar on "is anything here close?".
    margin      how far the nearest must beat the SECOND nearest. This is a bar
                on "is the answer unambiguous?", which is a different question
                and the one a single threshold cannot ask. Two stored questions
                nearly tied means the corpus does not actually distinguish them
                at this phrasing, and serving either is a coin flip.
    require_tier_agreement
                refuse when the top two candidates come from different tiers.
                Disagreement means the neighbourhood straddles a routing
                boundary — answer-directly vs hand-off — and the margin alone
                may not separate them. Only meaningful for a campaign whose
                rows carry a tier.
    """

    threshold: float
    margin: float = 0.0
    require_tier_agreement: bool = False


_ENABLED = os.environ.get("SACE_CACHE", "on").strip().lower() not in {"off", "0", "false"}


# The coverage campaign's historical behaviour, unchanged: one threshold, no
# margin, no tier check. Used whenever a caller passes no bar of its own.
DEFAULT_BAR = CacheBar(threshold=CACHE_THRESHOLD)


def enabled() -> bool:
    return _ENABLED


# Which substituted values make a reply personal to ONE call, and which are
# constants of the campaign.
#
# Getting this split wrong is what made the cache store nothing at all. The
# check used to refuse any reply containing ANY value from DEMO_PLACEHOLDERS,
# which sounds safe and is actually fatal: the prompt tells Maya to address the
# patient by first name, so essentially every reply on this script contains it,
# and every turn was refused as "caller-specific". Meanwhile three of the five
# placeholders are not caller-specific in the slightest — the callback number,
# the clinic name and the current month are the same for every caller in the
# campaign, and a reply quoting them is exactly what should be cached.
#
# So only the patient's own name is per-caller. And even that does not have to
# disqualify the reply: it is folded back to its placeholder on the way in and
# re-substituted on the way out (see normalise/personalise below), so one
# stored entry serves every caller with their own name in it.
_PERSONAL_KEYS = ("{patient_first_name}", "{patient_last_name}")

# Any {placeholder} token, for stripping out of a fingerprint.
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def _personal_values() -> list[str]:
    """The substituted values that identify ONE caller. Longest first, so a
    replacement pass cannot leave a fragment of a longer value behind."""
    vals = [str(DEMO_PLACEHOLDERS[k]) for k in _PERSONAL_KEYS
            if DEMO_PLACEHOLDERS.get(k) and len(str(DEMO_PLACEHOLDERS[k])) > 2]
    return sorted(vals, key=len, reverse=True)


def normalise(reply: str) -> str:
    """Fold this caller's name back to its placeholder, for storage.

    A cached reply is served to a DIFFERENT caller, so a literal name in it
    would assert a stranger's identity back at someone — the failure the old
    blanket refusal was built to prevent. Storing the placeholder instead keeps
    the entry reusable and makes that failure impossible rather than merely
    unlikely: there is no name in the row to leak.
    """
    for key in _PERSONAL_KEYS:
        value = str(DEMO_PLACEHOLDERS.get(key) or "")
        if len(value) > 2:
            reply = reply.replace(value, key)
    return reply


def personalise(reply: str) -> str:
    """Put the current caller's name back, for serving. Inverse of normalise.

    Prompt assembly substitutes placeholders on the way into the LLM
    (assemble._substitute_placeholders), but a cache hit never reaches assembly
    — so without this the caller would hear the literal "{patient_first_name}".
    """
    for placeholder, value in DEMO_PLACEHOLDERS.items():
        reply = reply.replace(placeholder, str(value))
    return reply


def _leaked_personal_values(reply: str) -> list[str]:
    """Any caller-identifying value still literal in a reply about to be stored.
    Should always be empty after normalise; checked anyway, because this is the
    one invariant whose failure speaks a stranger's name to someone."""
    return [v for v in _personal_values() if v in reply]


# Openers and fillers that carry no question. A turn-1 utterance made only of
# these has nothing for retrieval to match on, so whatever rule it landed on was
# close to arbitrary — see the turn-1 note in is_cacheable.
_FILLER = frozenset({
    "hello", "hi", "hey", "yes", "yeah", "yep", "yup", "no", "nope", "ok", "okay",
    "sure", "hmm", "mhm", "uh", "um", "er", "oh", "so", "well", "right", "alright",
    "who", "this", "is", "it", "that", "there", "a", "an", "the", "please",
    "speaking", "what", "sorry", "pardon", "again", "and", "you", "i", "me", "my",
})

# How many non-filler words an OPENING utterance needs before its reply may be
# cached. Two is enough to separate "hello? who is this" (0 significant words)
# from "is this call being recorded" (recorded, call -> 2).
MIN_OPENING_WORDS = 2


def _significant_words(text: str) -> list[str]:
    """Words that actually carry the question, filler stripped."""
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in _FILLER and len(w) > 2]


def trailing_question(reply: str) -> str | None:
    """The question a reply ends on, or None. Mirrors engine._extract_question.

    Kept here rather than imported so this module has no dependency on the
    engine (the engine imports this one).
    """
    if not reply or "?" not in reply:
        return None
    for part in reversed(re.split(r"(?<=[.!?])\s+", reply.strip())):
        if part.strip().endswith("?"):
            return part.strip()
    return None


# Function words dropped when fingerprinting a pending question. Separate from
# _FILLER above on purpose: that set answers "did the caller say anything with
# content?" and deliberately keeps possessives and determiners, because a bare
# "who is this" needs to score zero. This set answers a different question —
# "are these two phrasings the same question?" — where a stray "your" or "any"
# must not split one question into two entries.
_STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "been", "am", "do", "does", "did", "have", "has", "had", "can", "could",
    "will", "would", "shall", "should", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "by", "from", "as", "that", "this", "these",
    "those", "it", "its", "you", "your", "yours", "i", "me", "my", "mine", "we",
    "us", "our", "ours", "they", "them", "their", "he", "him", "his", "she",
    "her", "hers", "there", "here", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "so", "just", "still", "any", "some", "all",
    "now", "then", "again", "very", "really", "quite", "okay", "ok", "please",
    "sorry", "thanks", "like", "want", "need", "get", "got", "take", "make",
    "say", "said", "tell", "know", "think", "one", "up", "out", "about",
})


def question_fingerprint(question: str | None) -> str:
    """A phrasing-insensitive identity for a pending question.

    Compared rather than embedded: this runs on the store and the serve path of
    every diversion turn, and the whole design rests on those staying free of
    extra network calls. Content words, sorted and deduplicated — so
    "do you still have your Medi-Cal benefits?" and
    "do you still have Medi-Cal benefits?" are the same pending question, while
    neither matches "would you like me to repeat that information?".

    Sorted, so word order does not matter: a re-ask is routinely reordered
    ("do you still have Medi-Cal benefits" / "your Medi-Cal benefits — do you
    still have them"), and treating those as different questions would split one
    entry into two and halve the hit rate for no gain in safety.

    Deliberately coarse. It gates WHICH cached reply may be served, and every
    candidate has already had to clear CACHE_THRESHOLD on the caller's message
    within a single intent section — so this only has to separate the handful of
    questions the script can actually have pending, not act as a general
    paraphrase detector.
    """
    if not question:
        return ""
    # Normalised first, so the caller's name does not enter the fingerprint. A
    # pending question is "do you still have Medi-Cal benefits" whoever it is
    # addressed to; leaving the name in would give every caller their own
    # fingerprint and pin every entry to the one call that produced it —
    # silently reducing the cache to a per-caller store that never hits.
    # Placeholders are dropped entirely rather than normalised into words: the
    # SAME pending question is phrased both with the patient's name and without
    # it ("do you still have Medi-Cal benefits" / "does {patient_first_name}
    # still have Medi-Cal benefits"), and those must fingerprint alike. Folding
    # the name to a placeholder and then tokenising it would leave "patient
    # first name" in the fingerprint and split the two apart again.
    question = _PLACEHOLDER_RE.sub(" ", normalise(question))
    words = [w for w in re.findall(r"[a-z0-9']+", question.lower())
             if w not in _STOP and len(w) > 1]
    return " ".join(sorted(set(words)))


# The trailing-question gate.
#
# THE PROBLEM THIS SOLVES, and it is the one that decides whether this cache can
# exist at all. Almost every diversion rule in kb.py ends with "then return to
# the pending question" — so a correct diversion reply is two halves:
#
#     "That's something our coverage counselors can help with — you can text
#      KEEP, or call 1-800-555-1234."          <- the fixed fact. Position-free.
#     "For now, do you still have Medi-Cal benefits?"   <- the PENDING question.
#
# The first half is exactly what a cache should serve. The second half is call
# state, and replaying the wrong one is a real failure: it re-asks a question
# already answered, or jumps to one not yet reached, and the caller hears the
# script lurch. Blocking every rule that re-asks would have blocked 12 of the 18
# diversion rules — effectively the whole cache.
#
# So the gate: an entry records the pending question its reply trails, and it is
# only served on a turn where the SAME question is pending. The fixed half is
# reused; the conversational half can never land in the wrong place. A reply
# with no trailing question (a bare statement of fact) is unconditionally
# reusable and stores an empty fingerprint that matches any turn.
#
# The effect on hit rate is the opposite of what blocking would have been: one
# question asked at two different points in a call gets two entries, each
# correct where it fires, rather than one entry that is wrong half the time.
def pending_fingerprint(state) -> str:
    """The question currently awaiting an answer, as a fingerprint.

    `state.asked_questions` is append-ordered, so the last entry is the question
    Maya most recently put to the caller — which is what a diversion reply must
    return to.
    """
    asked = getattr(state, "asked_questions", None) or []
    return question_fingerprint(asked[-1] if asked else None)


def is_cacheable(*, governing, outcome: str, regenerated: bool,
                 extracted_fields: dict | None, reply: str, intent: str | None,
                 turn_index: int = 2, question: str | None = None) -> tuple[bool, str]:
    """Whether this completed turn may be stored. Returns (ok, reason_if_not).

    Read the module docstring for why each exclusion exists — every one of them
    is a way a replayed reply could be actively wrong rather than merely stale.
    """
    if not _ENABLED:
        return False, "cache disabled"
    if outcome != "grounded":
        return False, f"outcome={outcome}"
    if regenerated:
        return False, "reply was regenerated"
    if governing is None:
        return False, "no governing rule"

    chunk = getattr(governing, "chunk", governing)
    if getattr(chunk, "terminal", False):
        return False, f"{chunk.id} is terminal"
    if getattr(chunk, "priority", "") == "critical":
        return False, f"{chunk.id} is critical priority"

    # The structural rule — see NEVER_CACHE_RULES above. A SEED rule with no
    # intent is part of the call script, so its reply depends on position in the
    # flow and must never be replayed. This is the check that makes the default
    # "deny": a flow rule added to kb.py tomorrow is excluded without anyone
    # remembering to list it.
    #
    # Scoped to seed rules deliberately. For a LEARNED rule intent=None means
    # the consolidator could not map the situation onto the router vocabulary,
    # not that the reply is positional — a learned rule exists because a caller
    # went off-script, so it is a diversion by construction. See the long note
    # on NEVER_CACHE_RULES; conflating the two silently blocked most of the
    # cacheable turns on this pool.
    is_learned = (getattr(chunk, "source", "seed") == "learned"
                  or getattr(chunk, "learned_kind", None))
    if getattr(chunk, "intent", None) is None and not is_learned:
        return False, (
            f"{chunk.id} is a seed flow rule (intent=None) — its reply depends "
            f"on where in the call it is, so it can never be replayed"
        )

    if chunk.id in NEVER_CACHE_RULES:
        return False, f"{chunk.id} is on the never-cache list"
    if (intent or "") in NEVER_CACHE_INTENTS:
        return False, f"intent {intent!r} is on the never-cache list"
    if extracted_fields:
        return False, f"turn extracted {sorted(extracted_fields)}"

    # The turn-1 hazard, narrowly. This was originally "never cache turn 1",
    # which is safe but too blunt to be usable: a caller who asks a real
    # question straight away ("hi, quick thing — is this call recorded?") is
    # asking on turn 1, so blanket-blocking the position meant a natural short
    # call could never populate the cache at all.
    #
    # The actual danger is a SHORT, CONTENTLESS opening. A bare "hello?" or
    # "yes?" carries no question to match on, so it routes on almost nothing —
    # observed landing on `ai_question`, which would have stored a greeting as
    # the answer to "are you a robot?". An opening utterance with real substance
    # does not have that problem: whatever it matched, it matched on words the
    # caller actually said.
    #
    # The flow-rule blocklist above already covers the greeting/identity rules
    # themselves, so this only has to catch a contentless opener that routed to
    # some other rule by accident. Judged on the CALLER's message, because that
    # is what retrieval matched and what a later turn is compared against.
    if turn_index <= 1 and len(_significant_words(question or "")) < MIN_OPENING_WORDS:
        return False, (
            f"opening turn with too little to match on "
            f"({len(_significant_words(question or ''))} significant words, "
            f"need {MIN_OPENING_WORDS})"
        )

    # Checked against the NORMALISED reply (the caller's name already folded
    # back to its placeholder), so this only fires if normalise missed
    # something — a name the LLM inflected or spelled differently, say. Then it
    # is a genuine refusal rather than the blanket one that used to reject
    # every turn on the script.
    leaked = _leaked_personal_values(normalise(reply))
    if leaked:
        return False, f"reply still contains caller-specific detail {leaked!r}"

    if not reply.strip():
        return False, "empty reply"
    return True, ""


def lookup(conn, message_vec, intent: str | None, *, pending: str = "",
           table: str = "answer_cache", bar: CacheBar | None = None):
    """Nearest cached answer in this intent's section, or None.

    `message_vec` MUST be a vector the caller already computed (retrieve.py's
    per-turn message embedding). Never embed here — see the module docstring:
    embedding inside the cache is what would turn a miss into a real latency
    cost instead of a rounding error.

    `pending` is the fingerprint of the question currently awaiting an answer
    (answer_cache.pending_fingerprint). Only entries whose reply trails that
    same question — or trails no question at all, stored as "" — are eligible.
    See the trailing-question gate above; this is the filter that lets a
    diversion reply be replayed without the script lurching.

    `bar` is this campaign's measured serve decision (see CacheBar). Defaults
    to DEFAULT_BAR, which is threshold-only — the behaviour this function had
    before the margin and tier-agreement checks existed.
    """
    bar = bar or DEFAULT_BAR
    if not _ENABLED:
        return None
    if (intent or "") in NEVER_CACHE_INTENTS:
        return None
    # NOTE: intent=None is NOT short-circuited here. A learned rule the
    # consolidator could not classify stores under the NULL section and is
    # legitimately cacheable (see is_cacheable), so the general section has to
    # be queryable. An earlier version returned None here and made every such
    # entry permanently unreachable — stored but never served, which reads
    # exactly like the cache not working.

    # Same intent-scoping rule as retrieval: one section per turn, never across.
    # The pending filter is applied IN SQL rather than by discarding a row after
    # the fact: the nearest entry overall may be one pinned to a different
    # pending question, and filtering afterwards would report a miss while a
    # perfectly good entry for THIS turn sat second in the ordering.
    where_intent = "intent IS NULL" if intent is None else "intent = :intent"
    params = {"q": str(list(message_vec)), "pending": pending}
    if intent is not None:
        params["intent"] = intent

    # LIMIT 2, not 1. The runner-up is not a candidate — it is the evidence
    # for whether the winner is actually distinguishable. Fetching it is free
    # (the same index scan, one more row) and it is the only way the margin and
    # tier-agreement checks below can be asked at all.
    rows = conn.execute(
        sql_text(
            f"SELECT id, question, reply, governing_rule_id, hit_count, tier, "
            f"       pending_fingerprint, "
            f"       embedding <=> CAST(:q AS vector) AS distance "
            f"FROM {table} WHERE {where_intent} "
            f"  AND active = TRUE "
            f"  AND (pending_fingerprint = '' OR pending_fingerprint = :pending) "
            f"ORDER BY distance LIMIT 2"
        ),
        params,
    ).fetchall()

    if not rows:
        return None
    row = rows[0]
    similarity = 1.0 - float(row.distance)
    if similarity < bar.threshold:
        return None
    if row.governing_rule_id in NEVER_CACHE_RULES:
        return None
    if row.tier in NEVER_CACHE_TIERS:
        return None

    # The runner-up checks. Both are skipped when there IS no runner-up: a
    # section holding one eligible row has nothing to be ambiguous with, and
    # refusing there would disable the cache for every thinly-covered intent.
    runner_up = rows[1] if len(rows) > 1 else None
    if runner_up is not None:
        # Same rule, different phrasing, is not ambiguity — it is the coverage
        # the cache is supposed to accumulate (see DEDUP_THRESHOLD). Two rows
        # for one rule sitting close together should serve, not block; only a
        # near-tie between DIFFERENT rules is a coin flip.
        if runner_up.governing_rule_id != row.governing_rule_id:
            if bar.margin > 0.0:
                second = 1.0 - float(runner_up.distance)
                if (similarity - second) < bar.margin:
                    return None
            if bar.require_tier_agreement and runner_up.tier != row.tier:
                return None

    return {
        "id": row.id,
        "question": row.question,
        # Re-substituted here, because a hit never reaches prompt assembly and
        # the row deliberately stores "{patient_first_name}" rather than a name.
        "reply": personalise(row.reply),
        "governing_rule_id": row.governing_rule_id,
        "similarity": round(similarity, 4),
        # Surfaced so the dashboard can show WHY this entry was eligible: ""
        # means the reply ends on no question and is reusable anywhere, a value
        # means it was pinned to this turn's pending question and matched.
        "pending_fingerprint": row.pending_fingerprint,
    }


def record_hit(cache_id: str, *, table: str = "answer_cache") -> None:
    """Bookkeeping for a served hit. Tolerant by design — a stats update must
    never be able to break a live call. Fire-and-forget: call this, don't
    await it — it must never add to turn latency."""
    try:
        with SessionLocal() as session:
            session.execute(
                sql_text(
                    f"UPDATE {table} SET hit_count = hit_count + 1, "
                    f"last_hit_at = now() WHERE id = :i"
                ),
                {"i": cache_id},
            )
            session.commit()
    except Exception as exc:  # pragma: no cover - stats path
        print(f"[cache] record_hit failed: {type(exc).__name__}: {exc}")


def record_correct_hit(cache_id: str, *, table: str = "answer_cache") -> None:
    """A served hit that a later signal (no correction, regeneration, or
    transfer followed it) confirmed was right. Call this from wherever that
    signal is detected — NOT from record_hit, which fires the instant a reply
    is served and cannot yet know whether it held up.

    Deliberately only ever increments: a hit that WAS corrected just never
    gets this call, so hit_count - correct_hits IS the wrong-hit count. No
    separate "mark wrong" path to keep in sync with it.
    """
    try:
        with SessionLocal() as session:
            session.execute(
                sql_text(f"UPDATE {table} SET correct_hits = correct_hits + 1 WHERE id = :i"),
                {"i": cache_id},
            )
            session.commit()
    except Exception as exc:  # pragma: no cover - stats path
        print(f"[cache] record_correct_hit failed: {type(exc).__name__}: {exc}")


def nearest_row(conn, message_vec, intent: str | None, *, table: str = "answer_cache"):
    """The single nearest row in this intent's section, ignoring
    CACHE_THRESHOLD/active/tier — for miss instrumentation only (see
    record_miss), never for serving. A MISS still has a nearest row; this is
    how record_miss knows which row to attribute it to.
    """
    where_intent = "intent IS NULL" if intent is None else "intent = :intent"
    params = {"q": str(list(message_vec))}
    if intent is not None:
        params["intent"] = intent
    row = conn.execute(
        sql_text(
            f"SELECT id, governing_rule_id, "
            f"       embedding <=> CAST(:q AS vector) AS distance "
            f"FROM {table} WHERE {where_intent} "
            f"ORDER BY distance LIMIT 1"
        ),
        params,
    ).fetchone()
    if row is None:
        return None
    return {"id": row.id, "governing_rule_id": row.governing_rule_id,
            "similarity": round(1.0 - float(row.distance), 4)}


def record_miss(nearest_id: str, grounded_rule_id: str, *, table: str = "answer_cache") -> None:
    """On a cache MISS, once the full pipeline has determined which rule
    actually governed the turn: record that on the nearest row (from
    nearest_row(), computed BEFORE the miss was known to be a miss — this
    never triggers a second embedding call). cache_report.py looks for rows
    where this repeatedly equals the row's own governing_rule_id: real callers
    keep almost matching a rule this row is supposed to cover, which means the
    seeded phrasings for it don't match how people actually talk — an ADD
    candidate, not a reason to retire anything.
    """
    if not nearest_id:
        return
    try:
        with SessionLocal() as session:
            session.execute(
                sql_text(f"UPDATE {table} SET miss_grounded_to = :r WHERE id = :i"),
                {"r": grounded_rule_id, "i": nearest_id},
            )
            session.commit()
    except Exception as exc:  # pragma: no cover - stats path
        print(f"[cache] record_miss failed: {type(exc).__name__}: {exc}")


def store(*, question: str, question_vec, reply: str, intent: str | None,
          governing_rule_id: str | None, grounding_cosine: float | None = None,
          session_id: str | None = None, pending: str = "", tier: str | None = None,
          table: str = "answer_cache") -> str | None:
    """Persist one confirmed question->reply pair. Returns the row id, or None.

    Like record_hit, deliberately tolerant: caching is an optimisation, and a
    failure to store must never surface as a failed call.
    """
    if not _ENABLED:
        return None
    # intent=None is allowed: a learned rule the consolidator could not classify
    # stores in the general section and is served from it (see the note in
    # lookup). Only the never-cache guards are enforced here.
    if (intent or "") in NEVER_CACHE_INTENTS or governing_rule_id in NEVER_CACHE_RULES:
        print(f"[cache] refusing to store {governing_rule_id} (never-cache)")
        return None
    if tier in NEVER_CACHE_TIERS:
        print(f"[cache] refusing to store {governing_rule_id} (never-cache tier {tier})")
        return None
    try:
        vec = check_embedding(question_vec, chunk_id="cache-entry")
    except Exception as exc:
        print(f"[cache] refusing to store an invalid vector: {exc}")
        return None

    # Fold this caller's name back to its placeholder before it is persisted,
    # so the row holds no name at all and lookup() can safely serve it to
    # anyone. personalise() puts the current caller's name back on the way out.
    reply = normalise(reply)

    row_id = f"cache_{uuid.uuid4().hex[:10]}"
    try:
        with SessionLocal() as session:
            # A near-identical question already stored under this intent is an
            # update, not a second row: otherwise every repetition of a common
            # question grows the table and slows the very lookup it should
            # speed up.
            #
            # Deduped at DEDUP_THRESHOLD, deliberately NOT at CACHE_THRESHOLD —
            # see that constant for why collapsing at the serve bar keeps the
            # cache permanently thin.
            # Scoped to the same pending question as well as the same intent.
            # Without that, the two entries the gate exists to keep apart — one
            # question asked at two points in the call, each trailing a different
            # re-ask — would collapse into one row and the gate would have
            # nothing to distinguish.
            where_intent = "intent IS NULL" if intent is None else "intent = :intent"
            dedup_params = {"q": str(list(vec)), "pending": pending}
            if intent is not None:
                dedup_params["intent"] = intent
            existing = session.execute(
                sql_text(
                    f"SELECT id, embedding <=> CAST(:q AS vector) AS distance "
                    f"FROM {table} WHERE {where_intent} "
                    f"  AND pending_fingerprint = :pending "
                    f"ORDER BY distance LIMIT 1"
                ),
                dedup_params,
            ).fetchone()

            if existing is not None and (1.0 - float(existing.distance)) >= DEDUP_THRESHOLD:
                session.execute(
                    sql_text(
                        f"UPDATE {table} SET reply = :r, question = :q, "
                        f"governing_rule_id = :g, grounding_cosine = :c, tier = :t "
                        f"WHERE id = :i"
                    ),
                    {"r": reply, "q": question, "g": governing_rule_id,
                     "c": grounding_cosine, "t": tier, "i": existing.id},
                )
                session.commit()
                return existing.id

            session.execute(
                sql_text(
                    f"INSERT INTO {table} "
                    f"(id, question, embedding, reply, intent, governing_rule_id, "
                    f" grounding_cosine, source_session_id, pending_fingerprint, "
                    f" source, tier, hit_count, active) "
                    f"VALUES "
                    f"(:id, :question, CAST(:embedding AS vector), :reply, :intent, "
                    f" :governing_rule_id, :grounding_cosine, :source_session_id, "
                    f" :pending_fingerprint, :source, :tier, 0, TRUE)"
                ),
                {
                    "id": row_id, "question": question, "embedding": str(list(vec)),
                    "reply": reply, "intent": intent,
                    "governing_rule_id": governing_rule_id,
                    "grounding_cosine": grounding_cosine,
                    "source_session_id": session_id, "pending_fingerprint": pending,
                    "source": "live", "tier": tier,
                },
            )
            session.commit()
        return row_id
    except Exception as exc:  # pragma: no cover - optimisation path
        print(f"[cache] store failed: {type(exc).__name__}: {exc}")
        return None


def invalidate_for_rule(rule_id: str, *, table: str = "answer_cache") -> int:
    """Drop every cached answer derived from one rule. Called when that rule
    changes, so an edited rule cannot keep serving its old wording forever."""
    try:
        with SessionLocal() as session:
            n = session.execute(
                sql_text(f"DELETE FROM {table} WHERE governing_rule_id = :i"),
                {"i": rule_id},
            ).rowcount
            session.commit()
        return int(n or 0)
    except Exception as exc:
        print(f"[cache] invalidate_for_rule failed: {type(exc).__name__}: {exc}")
        return 0


def invalidate_for_intent(intent: str | None) -> int:
    """Drop every cached answer in one section.

    Called when a rule is approved into that section: the cached replies there
    were confirmed against the old rule set, so a new rule could be silently
    shadowed by an entry that predates it.
    """
    where = "intent IS NULL" if intent is None else "intent = :i"
    params = {} if intent is None else {"i": intent}
    try:
        with SessionLocal() as session:
            n = session.execute(
                sql_text(f"DELETE FROM answer_cache WHERE {where}"), params
            ).rowcount
            session.commit()
        return int(n or 0)
    except Exception as exc:
        print(f"[cache] invalidate_for_intent failed: {type(exc).__name__}: {exc}")
        return 0


def stats(*, table: str = "answer_cache") -> dict:
    with SessionLocal() as session:
        row = session.execute(sql_text(
            f"SELECT count(*) AS entries, COALESCE(sum(hit_count),0) AS hits "
            f"FROM {table}"
        )).fetchone()
        return {"entries": int(row.entries), "hits": int(row.hits),
                "threshold": CACHE_THRESHOLD,
                "dedup_threshold": DEDUP_THRESHOLD, "enabled": _ENABLED}


def clear(*, table: str = "answer_cache") -> int:
    with SessionLocal() as session:
        n = session.execute(sql_text(f"DELETE FROM {table}")).rowcount
        session.commit()
    return int(n or 0)
