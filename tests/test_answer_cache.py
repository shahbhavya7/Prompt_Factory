"""The reply cache — correctness, safety, and the latency claim.

Written to try to BREAK the cache, not to confirm it works. Where a check is a
weak form of the thing it claims to test, or could not be verified in this
environment, it says so in its own output rather than passing quietly.

Turns are driven through realistic multi-turn calls (not a fresh call per
message) because cacheability is partly positional: turn 1 is never cached, so
a suite that only ever drove first turns would test almost nothing while
appearing green. An earlier version of this file did exactly that.

Sections:
  A  storage      — what gets stored, and what is refused
  B  serving      — a hit replays the confirmed reply, with no model call
  C  false hits   — an unrelated question must NEVER be served a stored reply
  D  safety       — dnc/abuse/emergency/terminal/flow turns always run in full
  E  latency      — a hit is much faster; a MISS costs ~nothing (the key claim)
  F  invalidation — approving a rule clears the section it belongs to
  G  state        — no cached reply can leak one caller's data to another
  H  structural   — a SEED flow rule can NEVER be cached, exhaustively over
                    kb.RULES, including a rule invented here that is on no
                    list; and the converse, that a LEARNED rule with intent=None
                    IS cacheable and its section is reachable
  I  gate         — a diversion reply is only replayed where its trailing
                    re-ask still fits the pending question
  J  thresholds   — storing and serving use different bars, and why
  K  the name      — folded out on store, back in on serve; campaign constants
                     are not caller-specific (a blanket check once disabled the
                     whole cache while every individual check still passed)

Run:  python tests/test_answer_cache.py
"""

import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text as sql_text

from sace_chat import answer_cache, manager
from sace_chat.db import engine as db_engine, init_db
from sace_chat.embeddings import get_embedder
from sace_chat.engine import Engine
from sace_chat.kb import RULES, STABLE_CORE
from sace_chat.llm import get_llm
from sace_chat.retrieve import CallState

results = []
caveats = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    for line in str(detail).splitlines():
        if line.strip():
            print(f"          {line}")


def caveat(text):
    """Something this suite does NOT actually prove. Printed at the end so a
    green run cannot be mistaken for more coverage than it has."""
    caveats.append(text)


def make_engine():
    init_db()
    eng = Engine(stable_core=STABLE_CORE, rules=RULES, embedder=get_embedder(),
                 manager=manager, llm=get_llm())
    eng.router.warm()
    return eng


class Call:
    """A realistic call: state and history persist across turns, so turn N is
    genuinely turn N. Cacheability depends on this."""

    def __init__(self, eng):
        self.eng = eng
        self.state = CallState()
        self.history = []

    def say(self, msg, quiet=False):
        t0 = time.perf_counter()
        reply, _, dbg = self.eng.step(self.state, self.history, msg)
        ms = (time.perf_counter() - t0) * 1000
        if not quiet:
            gov = dbg["governing"]["id"] if dbg["governing"] else "-"
            cache_notes = [n for n in dbg["notes"] if "cach" in n.lower()]
            print(f"    caller : {msg}")
            print(f"    maya   : {reply}")
            print(f"    turn   : gov={gov} outcome={dbg['outcome']} {ms:.0f}ms"
                  + (f"  {cache_notes}" if cache_notes else ""))
        return reply, dbg, ms


def warm_up(call):
    """Get past the opening turns so the next thing said is a mid-call turn.
    Returns the number of turns consumed."""
    call.say("yeah go ahead, I've got a minute", quiet=True)
    call.say("yes, that's me speaking", quiet=True)
    return 2


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return d / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# A question with a fixed factual answer: no field extraction, non-terminal,
# not a flow rule. Q2 is a light rewording; measured cosines against Q1 on
# text-embedding-3-small are printed by the suite itself rather than asserted
# from memory.
Q1 = "quick thing, is this call being recorded"
Q2 = "wait, is this call recorded"          # measured ~0.86 vs Q1
Q_FAR = "do you have my home address on file"


def main():
    eng = make_engine()
    print(f"model     : {getattr(eng.llm, 'name', type(eng.llm).__name__)}")
    print(f"embedder  : {type(eng.embedder).__name__}")
    print(f"cache     : enabled={answer_cache.enabled()} threshold={answer_cache.CACHE_THRESHOLD}")
    if type(eng.embedder).__name__ == "MockEmbedder":
        caveat("Ran on MockEmbedder: cosine values are meaningless, so every "
               "threshold result in this run is uninformative. Re-run with "
               "EMBEDDING_MODE=openai before trusting section C.")
    answer_cache.clear()

    emb = eng.embedder
    v1, v2, vfar = emb.embed(Q1), emb.embed(Q2), emb.embed(Q_FAR)
    print(f"\nmeasured  : Q1~Q2 = {cos(v1, v2):.4f}   Q1~Q_FAR = {cos(v1, vfar):.4f}"
          f"   (bar {answer_cache.CACHE_THRESHOLD})")

    # ══ A. storage ═════════════════════════════════════════════════════════
    print("\n[A] what gets stored — a mid-call FAQ turn, and what is refused")
    call = Call(eng)
    call.say("hello? who's this", quiet=True)      # turn 1 — never cacheable
    warm_up(call)
    reply1, dbg1, ms_full = call.say(Q1)

    record("A1. the FAQ turn ran the full path", dbg1["outcome"] != "cached",
           f"outcome={dbg1['outcome']} llm_calls={dbg1['llm_calls']}")
    stored = [n for n in dbg1["notes"] if "cached as" in n]
    record("A2. it was stored for reuse", bool(stored),
           "; ".join(stored) or f"notes={dbg1['notes']}")

    with db_engine.connect() as c:
        rows = c.execute(sql_text(
            "SELECT intent, question, reply FROM answer_cache")).fetchall()
    record("A3. exactly one entry exists, under the right intent",
           len(rows) == 1 and rows[0].intent == "recorded_q",
           f"rows={[(r.intent, r.question) for r in rows]}")

    # The frontend renders this field directly, so its shape is part of the
    # contract, not an implementation detail.
    cs = dbg1.get("cache_stored")
    record("A3b. the save is reported in a structured field the UI can render",
           isinstance(cs, dict) and cs.get("stored") is True and cs.get("id"),
           f"cache_stored={cs}")

    print("\n    turn 1 and flow turns must be refused:")
    # Asserted against is_cacheable directly rather than by driving a turn: a
    # first turn that happens to be SERVED from an existing entry returns before
    # _maybe_cache runs, so an empty notes list would be indistinguishable from
    # a refusal. This tests the rule itself, which is what matters.
    faq = next(r for r in RULES if r.id == "special_recorded_q")

    class _Wrap:
        chunk = faq

    def cacheable(question, turn_index):
        return answer_cache.is_cacheable(
            governing=_Wrap(), outcome="grounded", regenerated=False,
            extracted_fields=None, reply="Yes, this call is recorded.",
            intent="recorded_q", turn_index=turn_index, question=question)

    # A contentless opener has nothing for retrieval to match on, so whatever it
    # landed on was close to arbitrary — refuse it. A turn-1 utterance with real
    # substance matched on words the caller actually said, so allow it: an
    # earlier blanket "never cache turn 1" was safe but unusable, because a
    # caller asking a real question immediately never populated the cache.
    ok_bare, why_bare = cacheable("hello? who is this", 1)
    ok_real, _ = cacheable("hi quick thing, is this call recorded", 1)
    ok_later, _ = cacheable("is this call being recorded", 3)
    record("A4. a contentless opener is refused; a substantive question is not",
           (not ok_bare) and ok_real and ok_later,
           f"turn1 'hello? who is this'  -> refused: {why_bare}\n"
           f"turn1 real question         -> cacheable: {ok_real}\n"
           f"turn3 same question         -> cacheable: {ok_later}")

    c2 = Call(eng)
    c2.say("hello?", quiet=True)
    _, d_flow, _ = c2.say("yeah go ahead", quiet=True)
    why_flow = [n for n in d_flow["notes"] if "not cached" in n]
    record("A5. a conversational-flow turn is never cached",
           bool(why_flow),
           f"gov={d_flow['governing']['id'] if d_flow['governing'] else None} · {why_flow}")
    cs_no = d_flow.get("cache_stored")
    record("A5b. a refusal reports a readable reason for the UI",
           isinstance(cs_no, dict) and cs_no.get("stored") is False and cs_no.get("reason"),
           f"cache_stored={cs_no}")

    # ══ B. serving ═════════════════════════════════════════════════════════
    print("\n[B] the same question, reworded, on a later call")
    call2 = Call(eng)
    call2.say("hi, yes?", quiet=True)
    warm_up(call2)
    reply2, dbg2, ms_hit = call2.say(Q2)

    served = dbg2["outcome"] == "cached"
    record("B1. served from the cache", served,
           f"outcome={dbg2['outcome']} similarity={dbg2.get('governing_cosine')}")
    record("B2. no model call was made", dbg2["llm_calls"] == 0,
           f"llm_calls={dbg2['llm_calls']}")
    record("B3. no prompt was assembled", dbg2["assembled_prompt_tokens"] == 0,
           f"assembled_tokens={dbg2['assembled_prompt_tokens']}")
    record("B4. the reply is the confirmed one, verbatim", reply2 == reply1,
           f"served  ={reply2!r}\nconfirmed={reply1!r}")
    # The transparency guarantee: a cached turn must still be inspectable.
    record("B5. the cached turn still has an auditable payload",
           bool(dbg2["prompt_sent"]) and Q2 in dbg2["prompt_sent"]
           and len(dbg2["llm_messages"]) == 2,
           f"prompt_sent carries the caller message and the replay reason; "
           f"messages={[m['role'] for m in dbg2['llm_messages']]}")
    with db_engine.connect() as c:
        hits = c.execute(sql_text("SELECT hit_count FROM answer_cache LIMIT 1")).scalar()
    record("B6. the hit was counted", (hits or 0) >= 1, f"hit_count={hits}")

    # ══ C. false hits — the dangerous failure ══════════════════════════════
    print("\n[C] an unrelated question must NEVER be served a stored reply")
    call3 = Call(eng)
    call3.say("hello", quiet=True)
    warm_up(call3)
    reply_far, dbg_far, _ = call3.say(Q_FAR)
    record("C1. an unrelated question was not served the cached reply",
           reply_far != reply1,
           f"cached reply  ={reply1[:60]!r}\nunrelated reply={reply_far[:60]!r}")

    # Probe the lookup directly across a spread of phrasings — this is the real
    # test of where the boundary sits, independent of what the LLM happened to
    # say. Anything that would be served is listed with its cosine.
    print("\n    lookup probe (what WOULD be served for each phrasing):")
    # True  = a genuine re-ask of the stored question; a miss here is merely
    #         conservative (costs latency, says nothing wrong).
    # False = a DIFFERENT question; a hit here is a real defect — it would speak
    #         the wrong words to a caller. "who is this calling" is the hardest
    #         impostor measured (0.538), so it is the one that matters most.
    probes = [
        ("is this call being recorded", True),
        ("wait, is this call recorded", True),
        ("so this call is being recorded?", True),
        ("sorry, is the call being recorded", True),
        ("is this being recorded", True),
        ("who is this calling", False),
        ("is my daughter allowed on the call", False),
        ("can you call me back tomorrow", False),
        ("stop calling me", False),
        ("how much does the plan cost", False),
        ("what county am I in", False),
        ("can I bring my service dog", False),
        ("do you have my home address on file", False),
    ]
    # The probe must use the SAME pending fingerprint the entry was stored
    # under, or every lookup misses for a reason that has nothing to do with the
    # similarity boundary this section is measuring. Read off the row rather
    # than reconstructed, so the probe cannot drift from what was actually
    # stored.
    with db_engine.connect() as c:
        stored_pending = c.execute(sql_text(
            "SELECT pending_fingerprint FROM answer_cache "
            "WHERE intent = 'recorded_q' LIMIT 1")).scalar() or ""
    print(f"      (entries pinned to pending={stored_pending!r})")

    wrong, missed = [], []
    with db_engine.connect() as c:
        for text_, should_hit in probes:
            hit = answer_cache.lookup(c, emb.embed(text_), "recorded_q",
                                      pending=stored_pending)
            sim = cos(v1, emb.embed(text_))
            got = hit is not None
            verdict = "served" if got else "  —   "
            flag = ""
            if should_hit is True and not got:
                flag = "  <-- missed (conservative: costs latency only)"
                missed.append((text_, sim))
            if should_hit is False and got:
                flag = "  <-- FALSE HIT (would speak the wrong words)"
                wrong.append((text_, sim))
            print(f"      {verdict}  cos={sim:.4f}  {text_!r}{flag}")
    record("C2. no DIFFERENT question is ever served this reply", not wrong,
           f"false hits={wrong}" if wrong
           else f"all {sum(1 for _, s in probes if s is False)} impostors declined")
    # Reported, not asserted: a conservative miss is a cost, not a defect. But
    # it is printed so a threshold that quietly stops firing is visible rather
    # than showing up as a green run with a useless cache.
    hit_rate = 1 - len(missed) / max(1, sum(1 for _, s in probes if s is True))
    record("C3. genuine re-asks are actually served (else the cache is useless)",
           hit_rate >= 0.6,
           f"served {hit_rate*100:.0f}% of genuine re-asks"
           + (f"; missed={[(t, round(s,3)) for t, s in missed]}" if missed else ""))

    # ══ D. safety ══════════════════════════════════════════════════════════
    print("\n[D] compliance and safety turns always run the full pipeline")
    for label, msg in [("dnc", "stop calling me, take me off your list"),
                       ("dnc reworded", "remove my number, don't call me again"),
                       ("abuse", "you people are idiots, stop wasting my time"),
                       ("emergency", "she's collapsed and can't breathe")]:
        cc = Call(eng)
        cc.say("hello", quiet=True)
        warm_up(cc)
        _, dsafe, _ = cc.say(msg, quiet=True)
        record(f"D1.{label}: never served from cache", dsafe["outcome"] != "cached",
               f"outcome={dsafe['outcome']} gov={dsafe['governing']['id'] if dsafe['governing'] else None}")

    with db_engine.connect() as c:
        unsafe = c.execute(sql_text(
            "SELECT count(*) FROM answer_cache WHERE intent IN "
            "('dnc','abuse','complaint_escalation') OR governing_rule_id IN "
            "('special_dnc','special_abuse','medical_emergency')")).scalar()
        terminal = c.execute(sql_text(
            "SELECT count(*) FROM answer_cache ac JOIN chunks ch "
            "ON ac.governing_rule_id = ch.id WHERE ch.terminal")).scalar()
        critical = c.execute(sql_text(
            "SELECT count(*) FROM answer_cache ac JOIN chunks ch "
            "ON ac.governing_rule_id = ch.id WHERE ch.priority = 'critical'")).scalar()
    record("D2. nothing unsafe ever reached the table", unsafe == 0, f"rows={unsafe}")
    record("D3. no terminal rule's reply was stored", terminal == 0, f"rows={terminal}")
    record("D4. no critical rule's reply was stored", critical == 0, f"rows={critical}")

    # ══ H. the structural rule ═════════════════════════════════════════════
    # Exhaustive over kb.RULES rather than a sample. The whole value of deriving
    # cacheability from `intent` instead of a name list is that it holds for
    # rules nobody has written yet — a spot-check of a few rules would not
    # demonstrate that, and would pass just as happily against the old list.
    print("\n[H] a flow rule can never be cached, and that must hold for ALL of them")

    class _Gov:
        def __init__(self, chunk):
            self.chunk = chunk

    # kb.RULES is the SEED set, which is what the structural rule governs.
    # Learned rules are checked separately in H4 below — for them intent=None
    # means "unclassified", not "positional", and they must NOT be refused.
    flow = [r for r in RULES if r.intent is None]
    leaked_flow = []
    for r in flow:
        ok, _ = answer_cache.is_cacheable(
            governing=_Gov(r), outcome="grounded", regenerated=False,
            extracted_fields=None, reply="Some reply.", intent=None,
            turn_index=3, question="a substantive caller question here")
        if ok:
            leaked_flow.append(r.id)
    record(f"H1. all {len(flow)} flow rules (intent=None) are refused",
           not leaked_flow,
           f"WRONGLY CACHEABLE: {leaked_flow}" if leaked_flow
           else f"checked every intent=None rule in kb.RULES")

    # The property that makes the default safe: a NEW flow rule, of a kind that
    # does not exist yet and is on no list anywhere, is still refused.
    from sace_chat.models import Chunk as _Chunk
    invented = _Chunk(
        id="a_flow_rule_invented_by_this_test", title="t",
        text="AFTER something, WHEN something else, say a thing.",
        cue="c", intent=None, priority="normal")
    ok_new, why_new = answer_cache.is_cacheable(
        governing=_Gov(invented), outcome="grounded", regenerated=False,
        extracted_fields=None, reply="Some reply.", intent=None,
        turn_index=3, question="a substantive caller question here")
    record("H2. a flow rule that is on no list is refused anyway "
           "(the default is deny)",
           not ok_new, f"refused: {why_new}" if not ok_new
           else "A RULE NOBODY LISTED WAS CACHEABLE — the default is allow")

    # And the converse: the FAQ rules are actually reachable. A safety filter
    # that refuses everything is safe and worthless, so this is asserted too.
    faqs = []
    for r in RULES:
        if r.intent is None:
            continue
        ok, _ = answer_cache.is_cacheable(
            governing=_Gov(r), outcome="grounded", regenerated=False,
            extracted_fields=None, reply="A fixed fact.", intent=r.intent,
            turn_index=3, question="a substantive caller question here")
        if ok:
            faqs.append(r.id)
    record("H3. the FAQ diversion rules ARE cacheable (the filter is not "
           "vacuous)", len(faqs) >= 5, f"{len(faqs)} cacheable: {faqs}")

    # H4 is the counterpart to H1, and it exists because conflating these two
    # cases silently disabled most of the cache. The consolidator leaves
    # `intent` NULL whenever the situation it extracted does not map onto
    # manager.VALID_INTENTS, so on a LEARNED rule intent=None means
    # "unclassified", not "positional" — a learned rule is a diversion by
    # construction, written because a caller went off-script. Measured on the
    # live pool: 20 of 38 learned rules had intent=None and were plainly
    # FAQ-shaped ("asks about services at the clinic", "concern about privacy"),
    # and treating them as flow rules meant a whole call could answer every FAQ
    # and store nothing.
    learned_null = _Chunk(
        id="learned_null_intent_probe", title="t",
        text="When the caller asks how we got their information, reassure them.",
        cue="how did you get my information", intent=None, priority="normal",
        source="learned", learned_kind="policy")
    ok_ln, why_ln = answer_cache.is_cacheable(
        governing=_Gov(learned_null), outcome="grounded", regenerated=False,
        extracted_fields=None, reply="We only use what your provider shared.",
        intent=None, turn_index=3, question="how did you get my information")
    record("H4. a LEARNED rule with intent=None is cacheable "
           "(NULL means unclassified there, not positional)",
           ok_ln, f"refused: {why_ln}" if not ok_ln
           else "learned rules are diversions by construction")

    # And the general (NULL) section must be reachable on both paths, or such an
    # entry is stored and then never served — indistinguishable from a broken
    # cache. An earlier version short-circuited intent=None in lookup().
    #
    # Does NOT clear the table: section F still needs the row section A stored,
    # and this probe writes into the NULL section which nothing else uses.
    nq = "how did you get hold of my information"
    nid = answer_cache.store(
        question=nq, question_vec=emb.embed(nq),
        reply="We only use what your provider shared with us.",
        intent=None, governing_rule_id="learned_null_intent_probe", pending="")
    with db_engine.connect() as c:
        # A genuine restatement (measured 0.786 against the stored question).
        # An earlier probe used "where did you get my details from", which
        # measures 0.6715 — just under the 0.68 bar — so the check failed on the
        # threshold rather than on the NULL-section plumbing it is testing.
        n_hit = answer_cache.lookup(
            c, emb.embed("where did you get my information"), None, pending="")
    record("H5. the NULL-intent section is both storable and servable",
           bool(nid) and n_hit is not None and n_hit["id"] == nid,
           f"stored={nid} served={n_hit['id'] if n_hit else None} "
           f"sim={n_hit['similarity'] if n_hit else None}")
    if nid:
        with db_engine.begin() as c:
            c.execute(sql_text("DELETE FROM answer_cache WHERE id = :i"), {"i": nid})

    # ══ I. the trailing-question gate ══════════════════════════════════════
    print("\n[I] a diversion reply is only replayed where its re-ask still fits")

    fp = answer_cache.question_fingerprint
    record("I1. the same question, reworded, fingerprints alike",
           fp("do you still have your Medi-Cal benefits?")
           == fp("do you still have Medi-Cal benefits?")
           == fp("Do you still have Medi-Cal benefits"),
           f"fp={fp('do you still have your Medi-Cal benefits?')!r}")
    record("I2. two DIFFERENT pending questions fingerprint differently",
           fp("do you still have Medi-Cal benefits?")
           != fp("would you like me to repeat that information?"),
           f"{fp('do you still have Medi-Cal benefits?')!r} vs "
           f"{fp('would you like me to repeat that information?')!r}")
    record("I3. a reply ending on no question is unpinned (reusable anywhere)",
           answer_cache.trailing_question("Yes, recorded for training.") is None
           and fp(None) == "",
           "trailing=None -> fingerprint '' -> matches any pending question")
    record("I4. the trailing question is the LAST one, not the first",
           answer_cache.trailing_question(
               "Is that clear? For now, do you still have Medi-Cal benefits?")
           == "For now, do you still have Medi-Cal benefits?")

    # The gate as the DB sees it: an entry pinned to one pending question must
    # not be served on a turn where a different question is pending. Driven
    # against the real table, because the SQL filter is where this actually
    # lives and a pure-function test would not exercise it.
    with db_engine.connect() as c:
        pinned = c.execute(sql_text(
            "SELECT id, question, pending_fingerprint FROM answer_cache "
            "WHERE pending_fingerprint <> '' LIMIT 1")).fetchone()
    if pinned is None:
        caveat("[I5] no PINNED entry existed after this run (every stored reply "
               "ended on no question), so the gate's SQL filter was not "
               "exercised against real rows. The pure-function checks above "
               "still ran.")
    else:
        with db_engine.connect() as c:
            v = emb.embed(pinned.question)
            same = answer_cache.lookup(c, v, None, pending=pinned.pending_fingerprint)
            # Look it up under its own intent, with a DIFFERENT pending question.
            intent_of = c.execute(sql_text(
                "SELECT intent FROM answer_cache WHERE id = :i"),
                {"i": pinned.id}).scalar()
            served_right = answer_cache.lookup(
                c, v, intent_of, pending=pinned.pending_fingerprint)
            served_wrong = answer_cache.lookup(
                c, v, intent_of, pending=fp("what is your date of birth"))
        record("I5. a pinned entry IS served when its question is pending",
               served_right is not None,
               f"entry={pinned.id} pending={pinned.pending_fingerprint!r}")
        record("I6. the same entry is NOT served when a DIFFERENT question "
               "is pending",
               served_wrong is None or served_wrong["id"] != pinned.id,
               f"served={served_wrong['id'] if served_wrong else None} "
               f"(must not be {pinned.id})")

    # ══ K. the caller's name ═══════════════════════════════════════════════
    # This section exists because the placeholder check silently disabled the
    # whole cache once. The prompt tells Maya to address the patient by first
    # name, so nearly every reply contains it, and a blanket "refuse any reply
    # containing a placeholder value" refused every turn — a cache that stored
    # nothing while every individual check still passed.
    print("\n[K] the caller's name is folded out on store, back in on serve")

    named = ("Yes, this call is being recorded for training purposes. I'm reaching "
             "out regarding Bhavya's Medi-Cal coverage status. Can you tell me if "
             "Bhavya still has Medi-Cal benefits?")
    normed = answer_cache.normalise(named)
    record("K1. a reply naming the caller is NOT refused outright",
           "{patient_first_name}" in normed
           and not answer_cache._leaked_personal_values(normed),
           f"normalised: {normed[:100]}…")
    record("K2. normalise/personalise round-trips exactly",
           answer_cache.personalise(normed) == named)
    record("K3. campaign constants are NOT treated as caller-specific",
           not answer_cache._leaked_personal_values(
               "Text KEEP or call 1-800-555-0100 — that's Community Medical "
               "Center - Downtown Clinic, as of this month."),
           "callback number, clinic name and current month are the same for "
           "every caller in the campaign, so a reply quoting them is cacheable")
    # The name must not enter the pending fingerprint either, or every caller
    # gets their own and no entry is ever reachable from another call.
    record("K4. the name does not enter the pending fingerprint",
           answer_cache.question_fingerprint(
               "Can you tell me if Bhavya still has Medi-Cal benefits?")
           == answer_cache.question_fingerprint(
               "Do you still have Medi-Cal benefits?"),
           f"both -> {answer_cache.question_fingerprint('Do you still have Medi-Cal benefits?')!r}")

    # End to end against the table: stored without a name, served with one.
    # Deliberately does NOT clear the table — section F still needs the entry
    # section A stored, and an earlier version of this block wiped it and made
    # F1 fail for a reason that had nothing to do with invalidation.
    pend_k = answer_cache.question_fingerprint(
        "Can you tell me if Bhavya still has Medi-Cal benefits?")
    kid = answer_cache.store(
        question="Is the call being recorded?", question_vec=emb.embed("Is the call being recorded?"),
        reply=named, intent="recorded_q", governing_rule_id="special_recorded_q",
        pending=pend_k)
    with db_engine.connect() as c:
        on_disk = c.execute(sql_text("SELECT reply FROM answer_cache WHERE id = :i"),
                            {"i": kid}).scalar() or ""
        # Read the row back directly rather than via lookup(): section A has
        # already stored its own recorded_q entry, so the nearest match to this
        # probe is not necessarily K's row, and the assertion below is about
        # personalise() on the way out — not about which row wins a cosine race.
        k_row = c.execute(sql_text("SELECT reply FROM answer_cache WHERE id = :i"),
                          {"i": kid}).scalar() or ""
        k_hit = {"reply": answer_cache.personalise(k_row)}
    record("K5. the stored row holds NO caller name", "Bhavya" not in on_disk,
           f"on disk: {on_disk[:90]}…")
    record("K6. the served reply has the name back, no placeholder leaked",
           k_hit is not None and "Bhavya" in k_hit["reply"] and "{" not in k_hit["reply"],
           f"served: {k_hit['reply'][:90] if k_hit else None}…")
    # Remove only THIS section's row, leaving everything else intact.
    if kid:
        with db_engine.begin() as c:
            c.execute(sql_text("DELETE FROM answer_cache WHERE id = :i"), {"i": kid})

    # ══ J. the two thresholds ══════════════════════════════════════════════
    print("\n[J] storing and serving use different bars, on purpose")
    record("J1. the dedup bar is strictly above the serve bar",
           answer_cache.DEDUP_THRESHOLD > answer_cache.CACHE_THRESHOLD,
           f"serve={answer_cache.CACHE_THRESHOLD} "
           f"dedup={answer_cache.DEDUP_THRESHOLD}\n"
           f"If these were equal, anything similar enough to be SERVED from a "
           f"row would also OVERWRITE it, so the table could never hold two "
           f"phrasings of one question and coverage could not grow.")

    # ══ G. state leakage ═══════════════════════════════════════════════════
    print("\n[G] no cached reply may carry one caller's data to another")
    with db_engine.connect() as c:
        stored_replies = [r[0] for r in c.execute(
            sql_text("SELECT reply FROM answer_cache")).fetchall()]
    from sace_chat.assemble import DEMO_PLACEHOLDERS
    personal = [str(v) for v in DEMO_PLACEHOLDERS.values() if v and len(str(v)) > 2]
    leaked = [(r[:50], p) for r in stored_replies for p in personal if p in r]
    record("G1. no stored reply contains caller-specific detail", not leaked,
           f"leaks={leaked}" if leaked
           else f"checked {len(stored_replies)} entries against {len(personal)} placeholders")

    # ══ E. latency — the claim that motivated the feature ══════════════════
    print("\n[E] latency")
    if served:
        record("E1. a hit is much faster than the full path", ms_hit < ms_full * 0.5,
               f"full path {ms_full:.0f}ms -> cache hit {ms_hit:.0f}ms "
               f"({(1 - ms_hit / ms_full) * 100:.0f}% faster)")
    else:
        record("E1. a hit is much faster than the full path", False,
               "no hit was served, so this was not measured")

    print("\n    a MISS must not be slower than having no cache at all —")
    print("    the lookup reuses retrieval's own embedding, so a miss is one extra query")

    def sample(use_cache):
        out = []
        for q in ["what happens if I move to another county",
                  "who else can see my file",
                  "can you tell me which plan I had before",
                  "do I need to bring anything with me"]:
            answer_cache._ENABLED = use_cache
            cc = Call(eng)
            cc.say("hello", quiet=True)
            warm_up(cc)
            _, _, ms = cc.say(q, quiet=True)
            out.append(ms)
        return out

    try:
        off = sample(False)
        on = sample(True)
    finally:
        answer_cache._ENABLED = True

    med_off, med_on = statistics.median(off), statistics.median(on)
    overhead = med_on - med_off
    # Bar: the miss penalty must be a small fraction of a turn, not a second
    # embedding round-trip (~100-300ms). Generous, because these are live LLM
    # calls whose variance dwarfs the thing being measured.
    record("E2. a cache miss adds negligible latency", overhead < 150,
           f"cache OFF median {med_off:7.0f}ms   {[f'{x:.0f}' for x in off]}\n"
           f"cache ON  median {med_on:7.0f}ms   {[f'{x:.0f}' for x in on]}\n"
           f"miss overhead {overhead:+.0f}ms")
    caveat("E2 measures a miss against live LLM latency, whose run-to-run "
           "variance (hundreds of ms) is larger than the overhead being "
           "measured — a negative result means 'lost in the noise', not "
           "'faster'. It bounds the overhead; it does not measure it precisely.")

    # ══ F. invalidation ════════════════════════════════════════════════════
    print("\n[F] approving a rule clears the cached answers in its section")
    before = answer_cache.stats()["entries"]
    dropped = answer_cache.invalidate_for_intent("recorded_q")
    after = answer_cache.stats()["entries"]
    record("F1. the section's entries were removed",
           dropped >= 1 and after == before - dropped,
           f"entries {before} -> {after}, dropped {dropped}")
    with db_engine.connect() as c:
        left = c.execute(sql_text(
            "SELECT count(*) FROM answer_cache WHERE intent='recorded_q'")).scalar()
    record("F2. that section is now empty", left == 0, f"remaining={left}")
    caveat("F only tests answer_cache.invalidate_for_intent directly. The "
           "review.approve -> invalidate wiring is exercised by "
           "tests/test_review_loop.py, not here.")

    print(f"\n[stats] {answer_cache.stats()}")

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} checks passed")
    if caveats:
        print("\nWHAT THIS RUN DOES NOT PROVE:")
        for c_ in caveats:
            print(f"  · {c_}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
