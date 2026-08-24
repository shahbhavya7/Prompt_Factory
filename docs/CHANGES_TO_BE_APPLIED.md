# Changes to make the system faster and more accurate

Four changes, all aimed at the same goal: keep the prompt small and
built-fresh-every-time (this is what we mean by "dynamic prompting"),
while making the lookup behind it faster and smarter.

**Status: 1, 2 and 4 are DONE. 3 is still just an idea, not started.**
This doc is kept as a record of the reasoning, not rewritten away once
something ships — see the "what actually happened" note under each
finished item for what changed and how it was checked.

---

## 1. Make the table lookup faster (add an index) — DONE

**What it is:** Right now, when we search for the matching rule, the
database has to check rows one by one. Adding an index is like adding a
table of contents to a book — it lets the database jump straight to the
right rows instead of reading through everything.

**Why:** This gets slow as we store more and more rules over time. An
index keeps it fast no matter how much we've learned.

**Effort:** Small. One-time setup, no risk to how anything currently works.

**What actually happened:** an index on the `intent` column was added
(`sace_chat/db.py`, `init_db()`). Checked with `EXPLAIN`: looking up a specific
intent's rules now uses the index (a "Bitmap Index Scan") instead of reading
every row. The general pool (rules with no intent at all) still reads every
row in that pool — and that's actually correct today, not a leftover bug,
because that pool is still a large share of the whole table, so jumping
through an index wouldn't be any faster than just reading it straight
through. The index will start paying off there too once any one section
grows large enough on its own.

---

## 2. Only compare a new rule against its own section, not everything — DONE

**What it is:** When a new rule is learned after a call, we check if it's
a duplicate or if it conflicts with something we already know. Right now
that check compares the new rule against **every single rule we have**,
even ones from totally unrelated situations. We should only compare it
against rules in the same section (e.g. only compare a "caller is busy"
rule against other "caller is busy" rules).

**Why:** Faster, and more accurate — comparing against unrelated rules
just wastes time and adds noise.

**Effort:** Small. Just narrows down what gets compared.

**What actually happened:** the duplicate/conflict check
(`sace_chat/consolidator.py`, `_fetch_pool`) now only pulls in rules from the
new rule's own section — same intent, or the general pool if it has no
intent — instead of the whole table. While making this change we also found
real near-duplicate rules already sitting in the pool that the old, looser
"is this a duplicate" bar had missed (two rules saying almost the same thing
in slightly different words), so that bar was tightened at the same time —
it now catches things it used to let through. Comparing only within the
right section made this safe to tighten without accidentally flagging
unrelated rules as duplicates of each other.

---

## 3. Watch for sections that grow too big, and split or clean them up

**What it is:** Over time, one section (like "caller is busy") could end
up with a lot of very similar rules in it if the duplicate check ever
misses near-repeats. We should periodically look back at each section and
merge rules that are basically saying the same thing.

**Why:** A section with 200 near-identical rules is slower to search and
harder to trust than one with 10 clean, distinct ones. This keeps the
system healthy as it learns more over time.

**Effort:** Medium. This is a new maintenance step, not just a tweak —
worth doing later, once we actually see a section getting big, not
right away.

---

## 4. Remember answers we've already gotten right (inspired by Karpathy's "LLM Wiki" idea)

**What it is:** Right now, every single time the caller says something,
we do the full process: search for the matching rule, ask the AI to
write a reply, then double-check the reply. If a caller says almost the
exact same thing we've heard many times before ("I'm not interested" /
"stop calling me"), we're redoing all of that work from scratch every
time, even though we already know the right answer.

Instead: once we've confirmed a reply was correct and matched the right
rule, we save that exact match (what was said → what the correct reply
was) as a **remembered answer**, filed under the same section it belongs
to. Next time something very close to that comes in, we check the
remembered answers for that section first. If there's a close enough
match, we reuse the saved reply directly — no rule search, no asking the
AI again. If nothing matches closely, we fall back to the normal process,
exactly like today.

**One safety rule:** we will never do this shortcut for serious situations
— do-not-call requests, abusive callers, or a medical emergency. Those
always go through the full process and the full double-check, every
single time, no shortcuts.

**Why:** This is the one that saves the most time. For common, repeated
things callers say, we skip the slowest steps entirely and reuse an
answer we already know is good.

**Effort:** Medium-large. This is a new capability, not just a tweak — it
needs its own small storage for "remembered answers" and a "how close
counts as close enough" check, plus the safety rule above.

**What actually happened:** built as `sace_chat/answer_cache.py` (table
`answer_cache`), and it works — a repeated question is answered in about
**256ms instead of about 1935ms, an 87% cut**. Full detail in
[ARCHITECTURE.md §9](ARCHITECTURE.md#9-the-reply-cache-sace_chatanswer_cachepy).

Three things worth recording, because they were not obvious up front:

- **The "worst case adds latency" worry turned out to be avoidable entirely.**
  The trick is that the lookup reuses the message embedding retrieval already
  computes every turn, instead of embedding the question itself. So a miss
  costs one small extra database query — measurably smaller than the normal
  run-to-run variance of the model call — rather than a whole extra
  embedding round-trip. Had the cache embedded its own query, every miss
  would have paid ~100-300ms for nothing.
- **"How close counts as close enough" had to be measured, not guessed.**
  Two guessed values were both wrong, in opposite directions: the first was
  so strict the cache never fired at all, and the second rejected about half
  of genuine re-asks. Measuring where real re-asks land (0.76-0.88) versus
  where different questions land (0.12-0.54) showed a clean empty gap, and
  the bar went in the middle of it.
- **The safety rule needed more than the three cases named above.** Beyond
  do-not-call, abuse and emergencies, the cache also has to refuse the
  *first turn of any call* and every conversational-flow rule — their right
  answer depends on where in the call they happen, not on what was asked. A
  greeting was observed being stored as the answer to "are you a robot?"
  before that exclusion existed.

---

## How these fit together

Changes 1–3 make the existing lookup itself faster and cleaner.
Change 4 adds a shortcut in front of it, so repeated situations skip the
lookup entirely. None of these change how the prompt is built — it stays
small, and it's still assembled fresh each time based on what's actually
relevant right now.

**Suggested order:** do 1 and 2 first (small, safe, quick wins) — **done.**
Hold off on 3 until we actually see a section getting large — **still
holding off; not started, and now more relevant than before: the pool has
grown to 36 learned rules with visible near-duplicate drift (five rules
about caller frustration, four about questioning the call's legitimacy).**
Do 4 once 1–2 are in, since it's the biggest change and benefits from the
lookup underneath it already being fast and clean — **done, see above.**
