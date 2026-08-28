"""Small v0 vocabulary. These are documented strings, not a closed taxonomy.

Fields that use these values remain `str` so later work can introduce new
labels without a schema break.
"""


class Visibility:
    """How an information item relates to an observer."""

    PUBLIC = "public"
    PRIVATE = "private"
    OTHER_PRIVATE = "other_private"
    HIDDEN = "hidden"
    INFERRED = "inferred"
    COMMUNICATED = "communicated"
    UNKNOWN_TO_SELF = "unknown_to_self"


class Interaction:
    """How players' goals relate, at the rules level."""

    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    MIXED = "mixed"


class DecisionTiming:
    """Whether players choose in order or at the same time."""

    SEQUENTIAL = "sequential"
    SIMULTANEOUS = "simultaneous"


class TransitionKind:
    """What caused a state change."""

    PLAYER = "player"
    CHANCE = "chance"
    SYSTEM = "system"
