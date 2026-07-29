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
