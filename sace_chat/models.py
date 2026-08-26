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
    tier       a campaign's own answer tier (e.g. the renewal KB's T1-T4), or
               None for campaigns with no tier concept. Purely descriptive
               here — a tier's actual behaviour (priority/transfer/cacheable)
               is a lookup against that campaign's own behaviour table, not
               anything this dataclass interprets.
    verbatim   True if `text` is spoken exactly as written, with no slot
               substitution and no case-record fill. A verbatim rule's text
               must contain no bracket, placeholder, or meta-instruction — if
               it needs any of those to be spoken correctly, it is not
               verbatim.
    slots      [bracketed] tokens in `text` that get substituted from campaign
               config at serve time (e.g. "clinic name"). Empty for a
               verbatim rule.
    requires_case_fields   case-record fields `text`'s leading [bracketed]
               instruction reads before falling back to the spoken clause
               after it. That bracket is an instruction, never speech, and
               must never reach TTS as written.
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
    # The raw phrasings `cue` was built FROM (source column split on "·"), kept
    # verbatim alongside the joined cue so nothing is lost in the join.
    cue_variants: list[str] = field(default_factory=list)
    intent: str | None = None
    priority: str = "normal"  # critical, high, normal, low
    terminal: bool = False
    exclusive: bool = False
    source: str = "seed"
    learned_kind: str | None = None
    tier: str | None = None
    verbatim: bool = False
    slots: list[str] = field(default_factory=list)
    requires_case_fields: list[str] = field(default_factory=list)
    # Provenance extras only; never queried.
    tags: dict = field(default_factory=dict)
