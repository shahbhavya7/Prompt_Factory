from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One rule in the flat memory pool.

    There is no stage machine any more, so a chunk carries only what retrieval
    and the prompt actually need:

      intent     the routable label this rule handles, or None for a general
                 rule reachable by plain semantic similarity. The only routing
                 key there is.
      terminal   speaking this rule ends the call.
      exclusive  nothing else may be in scope alongside it — when this rule
                 governs a turn the REFERENCE list is emptied, so there is no
                 second rule for the model to splice a sentence out of.
      source     seed (hand-authored) | learned (written by the consolidator).
    """

    id: str
    title: str
    text: str
    # What this rule is RETRIEVED BY, as opposed to what it says: the caller-side
    # situation that should pull it in. This is the text that gets embedded.
    #
    # It has to be separate from `text`, because `text` contains the line Maya
    # SPEAKS — and the turn query contains Maya's previous line, so embedding
    # `text` makes retrieval sticky: the rule that produced the last thing Maya
    # said always scores highest, and the conversation cannot move on. Measured
    # on 19 flow turns: embedding `text` picked the right rule 1 time; embedding
    # `cue` picks it 18 times.
    #
    # Empty means fall back to `text`.
    cue: str = ""
    intent: str | None = None
    priority: str = "normal"  # critical, high, normal, low
    terminal: bool = False
    exclusive: bool = False
    source: str = "seed"
    learned_kind: str | None = None
    # Provenance extras only; never queried.
    tags: dict = field(default_factory=dict)

    # A campaign's own answer tier (e.g. renewal's T1-T4). None for campaigns
    # with no tier concept — coverage.
    tier: str | None = None
    # Whether this rule's reply is expected to hand the caller to a human.
    # Read by the never-say guard's fallback (see sace_chat/guards.py).
    transfer: bool = False
    # Prerequisite gating for a FLOW rule (intent=None): the subset of
    # state.collected_fields that must already hold before this step may
    # govern a turn. {} means "no prerequisite" — every existing coverage
    # rule. A value of "__any__" means "this key must be SET, any value"
    # rather than an exact match. See retrieve._fetch_general.
    requires: dict = field(default_factory=dict)
    # {field: default_value} this rule's own turn is expected to populate in
    # collected_fields once accepted. Not documentation-only: the model does
    # not reliably emit a synthetic gating field on its own initiative even
    # when the rule's text asks for it (measured: it extracts a different,
    # more "natural" field instead, or nothing) — so Engine deterministically
    # fills in any field here the model's own extracted_fields is missing,
    # the same way `terminal` already overrides the model's own
    # call_should_end rather than trusting it. Which rule governed the turn
    # already answers the question for a field named here (the rule was
    # chosen by cue similarity to what the caller actually said), so its
    # default is authoritative for a field with no better answer.
    sets: dict = field(default_factory=dict)
    # A flow rule's fixed position in its script — the fallback tie-breaker
    # when nothing else governs (retrieve.py's "lowest-numbered eligible
    # step"), never a ranking signal otherwise. None for anything that isn't
    # a flow rule.
    step_order: int | None = None
