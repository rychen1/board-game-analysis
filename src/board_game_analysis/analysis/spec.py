"""Typed catalog of proposed measurements. No calculations live here."""

from pydantic import BaseModel, ConfigDict, Field


class MeasurementRole:
    """Where a quantity sits in later statistical work."""

    DESCRIPTIVE = "descriptive"
    PREDICTIVE = "predictive"
    OUTCOME = "outcome"


class MeasurementStatus:
    """How seriously to treat the metric today."""

    PROPOSED = "proposed"
    HYPOTHESIS = "hypothesis"
    FUTURE = "future"


class MeasurementScope:
    """Default grain at which a metric would be recorded."""

    GAME = "game"
    PLAYER = "player"
    STATE = "state"
    OBSERVATION = "observation"
    TRAJECTORY = "trajectory"


class EvidenceLevel:
    """Levels of evidence. Do not treat them as interchangeable."""

    OBSERVED_FACT = "observed_fact"
    EXTRACTED_INTERPRETATION = "extracted_interpretation"
    STRUCTURAL_MEASUREMENT = "structural_measurement"
    SIMULATION_RESULT = "simulation_result"
    HUMAN_OUTCOME = "human_outcome"


class MeasurementSpec(BaseModel):
    """A proposed metric, not a computed result.

    `computable_from_ontology_v0` means a later engine *could* attempt a
    naive operationalization from a complete `PlaySituation`. It does not
    mean the operationalization is valid.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1)
    role: str
    status: str
    scope: str
    description: str
    operationalizations: list[str] = Field(default_factory=list)
    computable_from_ontology_v0: bool = False
    requires: list[str] = Field(default_factory=list)
    distinct_from: list[str] = Field(default_factory=list)
    notes: str | None = None
    unit: str | None = None


def measurement_catalog() -> tuple[MeasurementSpec, ...]:
    """Analysis specification v0. Values are not computed here."""
    return _CATALOG


def spec_by_id(spec_id: str) -> MeasurementSpec:
    for spec in _CATALOG:
        if spec.id == spec_id:
            return spec
    msg = f"unknown measurement spec {spec_id}"
    raise KeyError(msg)


_CATALOG: tuple[MeasurementSpec, ...] = (
    MeasurementSpec(
        id="information_volume",
        name="information_volume",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.OBSERVATION,
        description=(
            "How much information an observer has at a point. Item count is "
            "a poor proxy: one payload may encode a whole board."
        ),
        operationalizations=[
            "count of InformationItems (naive, not recommended as truth)",
            "weighted count by payload size or component coverage",
            "bits relative to a later formal state encoding",
        ],
        computable_from_ontology_v0=True,
        requires=["observation"],
        notes="v0 can count items; it cannot measure bits.",
        unit="unspecified",
    ),
    MeasurementSpec(
        id="information_visibility",
        name="information_visibility",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.OBSERVATION,
        description=(
            "Share of relevant underlying state that is directly observable "
            "(content_known), as opposed to hidden, inferred, or unknown."
        ),
        operationalizations=[
            "fraction of items with content_known True",
            "fraction of GameState.data keys mirrored as known items",
            "exclude inferred items from the numerator",
        ],
        computable_from_ontology_v0=True,
        requires=["observation", "game_state"],
        notes="Requires a definition of 'relevant' state, which v0 lacks.",
    ),
    MeasurementSpec(
        id="information_asymmetry",
        name="information_asymmetry",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.STATE,
        description="How different two or more players' information spaces are.",
        operationalizations=[
            "set difference of content_known items about the same holders",
            "distance between payload maps (needs a similarity function)",
            "self-knowledge vs other-knowledge split (see self_knowledge_asymmetry)",
        ],
        computable_from_ontology_v0=True,
        requires=["multiple_observations_of_same_state"],
        distinct_from=["information_uncertainty", "hidden_information"],
        notes="Structure is present; no canonical distance is defined.",
    ),
    MeasurementSpec(
        id="information_uncertainty",
        name="information_uncertainty",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.OBSERVATION,
        description=(
            "How uncertain an observer is about relevant hidden state. "
            "Not the same as 'a hidden item exists'."
        ),
        operationalizations=[
            "entropy over a later belief distribution",
            "count of inferred items and their payload confidence if present",
            "count of content_known=False items the observer is aware of",
        ],
        computable_from_ontology_v0=False,
        requires=["belief_model"],
        distinct_from=["hidden_information", "randomness"],
        notes="Poker range payloads are hints, not calibrated probabilities.",
    ),
    MeasurementSpec(
        id="information_ownership",
        name="information_ownership",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.STATE,
        description="Who knows which information (holder, known_to, observer).",
        operationalizations=[
            "partition items by holder_id and known_to",
            "per-component observer sets",
        ],
        computable_from_ontology_v0=True,
        requires=["information_items"],
    ),
    MeasurementSpec(
        id="information_flow",
        name="information_flow",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description="How information moves between players over time.",
        operationalizations=[
            "trace communicated_by and predecessor_id across states",
            "count clue/communication actions per turn",
        ],
        computable_from_ontology_v0=False,
        requires=["multi_state_trace"],
        notes="Short fixtures can show a single hop, not a flow regime.",
    ),
    MeasurementSpec(
        id="information_gain",
        name="information_gain",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.TRAJECTORY,
        description="How much an action or event changes what a player knows.",
        operationalizations=[
            "new content_known items after a transition",
            "reduction in listed unknown slots",
            "later: KL divergence between belief states",
        ],
        computable_from_ontology_v0=True,
        requires=["paired_observations_across_transition"],
        notes="Naive item-count gain is available; information-theoretic gain is not.",
    ),
    MeasurementSpec(
        id="information_loss",
        name="information_loss",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.TRAJECTORY,
        description=(
            "How much relevant information becomes unavailable, obsolete, "
            "or more uncertain."
        ),
        operationalizations=[
            "items that drop content_known after a transition",
            "forgotten or transformed predecessor chains",
        ],
        computable_from_ontology_v0=True,
        requires=["paired_observations_across_transition"],
        distinct_from=["information_gain"],
        notes="Telestrations transformation is lossy; v0 does not quantify bits lost.",
    ),
    MeasurementSpec(
        id="self_knowledge_asymmetry",
        name="self_knowledge_asymmetry",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.OBSERVATION,
        description=(
            "Whether a player can observe information about others that they "
            "cannot observe about themselves (Hanabi-type inversion)."
        ),
        operationalizations=[
            "own holder items unknown_to_self vs other_private content_known",
            "binary flag for inverted visibility",
        ],
        computable_from_ontology_v0=True,
        requires=["observation"],
        distinct_from=["information_asymmetry"],
    ),
    MeasurementSpec(
        id="legal_action_type_count",
        name="legal_action_type_count",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.GAME,
        description="How many action types the rules permit in principle.",
        operationalizations=["len(GameDefinition.legal_action_types)"],
        computable_from_ontology_v0=True,
        requires=["game_definition"],
        unit="action_types",
        notes="Types, not instantiated moves. Chess 'move' is one type.",
    ),
    MeasurementSpec(
        id="legal_decision_count",
        name="legal_decision_count",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.STATE,
        description=(
            "Number of legal actions at a state, including those not currently "
            "available if the spec treats legality at the instance level."
        ),
        operationalizations=[
            "count DecisionOption with legal True (only if the space lists them)",
            "full legal-move generator (not in v0)",
        ],
        computable_from_ontology_v0=False,
        requires=["complete_legal_move_set"],
        distinct_from=["available_decision_count", "decision_branching_factor"],
        notes="Ontology lists options; it does not prove the list is all legal moves.",
    ),
    MeasurementSpec(
        id="available_decision_count",
        name="available_decision_count",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.PLAYER,
        description="Number of currently executable actions in a decision space.",
        operationalizations=["count options where legal and available"],
        computable_from_ontology_v0=True,
        requires=["decision_space"],
        unit="actions",
        notes="Trust only when DecisionSpace.complete is True.",
    ),
    MeasurementSpec(
        id="decision_branching_factor",
        name="decision_branching_factor",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.STATE,
        description="Number of possible actions at a state (often available count).",
        operationalizations=[
            "available_decision_count at the active player",
            "mean available count over a trace",
        ],
        computable_from_ontology_v0=True,
        requires=["decision_space"],
        distinct_from=["decision_diversity", "meaningful_action_count"],
        notes="A sample space with complete=False is not a branching factor.",
    ),
    MeasurementSpec(
        id="decision_diversity",
        name="decision_diversity",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.PLAYER,
        description=(
            "How structurally different available actions are. Ten near-duplicate "
            "moves should not equal ten different action types."
        ),
        operationalizations=[
            "unique action_type count (very crude)",
            "cluster parameters (needs a similarity function)",
            "outcome-divergent clusters from simulation",
        ],
        computable_from_ontology_v0=False,
        requires=["action_similarity_or_simulation"],
        distinct_from=["available_decision_count", "meaningful_action_count"],
    ),
    MeasurementSpec(
        id="decision_constraint",
        name="decision_constraint",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.PLAYER,
        description="How strongly the current state restricts possible actions.",
        operationalizations=[
            "available / legal if both counts exist",
            "count of legal-but-unavailable options",
        ],
        computable_from_ontology_v0=True,
        requires=["decision_space"],
        notes="Pandemic charter vs treat is the v0 illustration, not a scale.",
    ),
    MeasurementSpec(
        id="reversibility",
        name="reversibility",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.STATE,
        description="How easily the consequences of a decision can be undone.",
        operationalizations=[
            "existence of an inverse action in later states",
            "repair cost in a search model",
        ],
        computable_from_ontology_v0=False,
        requires=["forward_model"],
    ),
    MeasurementSpec(
        id="consequence",
        name="consequence",
        role=MeasurementRole.PREDICTIVE,
        status=MeasurementStatus.HYPOTHESIS,
        scope=MeasurementScope.STATE,
        description="How much an action alters future state or outcome prospects.",
        operationalizations=[
            "edit distance between GameState.data before/after (crude)",
            "value-function delta under a later model",
        ],
        computable_from_ontology_v0=False,
        requires=["forward_model_or_value_fn"],
        distinct_from=["agency"],
    ),
    MeasurementSpec(
        id="agency",
        name="agency",
        role=MeasurementRole.PREDICTIVE,
        status=MeasurementStatus.HYPOTHESIS,
        scope=MeasurementScope.PLAYER,
        description="How much a player's decision can affect outcomes.",
        operationalizations=[
            "range of reachable outcome values from current options",
            "counterfactual outcome spread under simulation",
        ],
        computable_from_ontology_v0=False,
        requires=["simulation_or_value_fn"],
        distinct_from=["available_decision_count", "consequence"],
        notes="Many options can coincide with low agency if they do not matter.",
    ),
    MeasurementSpec(
        id="strategic_dominance",
        name="strategic_dominance",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.PLAYER,
        description="Whether some available actions are dominated by others.",
        operationalizations=[
            "strict dominance in a later utility model",
            "agent agreement that an option is never chosen",
        ],
        computable_from_ontology_v0=False,
        requires=["utility_or_agents"],
        distinct_from=["meaningful_action_count"],
        notes="Not implemented. Do not label options good/bad on the ontology.",
    ),
    MeasurementSpec(
        id="meaningful_action_count",
        name="meaningful_action_count",
        role=MeasurementRole.PREDICTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.PLAYER,
        description=(
            "Count of decisions that matter. Not legal count and not available "
            "count. Several identification methods remain open."
        ),
        operationalizations=[
            "counterfactual outcome differences",
            "simulation / agent disagreement",
            "utility or value differences",
            "sensitivity of future state",
            "human empirical choice data",
        ],
        computable_from_ontology_v0=False,
        requires=["analysis_engine_not_built"],
        distinct_from=[
            "legal_decision_count",
            "available_decision_count",
            "decision_diversity",
            "strategic_dominance",
        ],
        notes="No operationalization is privileged in v0.",
    ),
    MeasurementSpec(
        id="state_complexity_trajectory",
        name="state_complexity_trajectory",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description="How complex the underlying state becomes over a game.",
        operationalizations=[
            "size of GameState.data over turns",
            "component counts if a later encoding exists",
        ],
        computable_from_ontology_v0=False,
        requires=["full_play_trace"],
    ),
    MeasurementSpec(
        id="decision_space_trajectory",
        name="decision_space_trajectory",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description="How size and diversity of decision spaces change over a game.",
        operationalizations=["time series of available_decision_count and diversity"],
        computable_from_ontology_v0=False,
        requires=["full_play_trace"],
    ),
    MeasurementSpec(
        id="information_trajectory",
        name="information_trajectory",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description="How each player's information changes throughout a game.",
        operationalizations=["per-player series of visibility and volume proxies"],
        computable_from_ontology_v0=False,
        requires=["full_play_trace"],
    ),
    MeasurementSpec(
        id="information_convergence",
        name="information_convergence",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description=(
            "Whether players become more similarly informed or more asymmetric."
        ),
        operationalizations=["time series of information_asymmetry"],
        computable_from_ontology_v0=False,
        requires=["full_play_trace"],
        distinct_from=["information_asymmetry"],
    ),
    MeasurementSpec(
        id="tension_trajectory",
        name="tension_trajectory",
        role=MeasurementRole.PREDICTIVE,
        status=MeasurementStatus.HYPOTHESIS,
        scope=MeasurementScope.TRAJECTORY,
        description=(
            "Hypothesized path of experienced tension. Not an established metric."
        ),
        operationalizations=[
            "later composite of uncertainty, consequence, scarcity, proximity",
            "human-rated tension curves as the validation target",
        ],
        computable_from_ontology_v0=False,
        requires=["human_ratings_or_validated_model"],
        distinct_from=["agency", "information_uncertainty"],
        notes="No product formula is adopted. See docs/analysis.md.",
    ),
    MeasurementSpec(
        id="structural_interaction",
        name="structural_interaction",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.GAME,
        description=(
            "The rules require players to affect each other (shared board, "
            "mandatory communication, blocking)."
        ),
        operationalizations=[
            "GameDefinition.interaction plus action types that target others",
            "extracted labels such as constrained_communication",
        ],
        computable_from_ontology_v0=True,
        requires=["game_definition"],
        distinct_from=["behavioral_interaction"],
        notes="Cooperative vs competitive is rules structure, not a score.",
    ),
    MeasurementSpec(
        id="behavioral_interaction",
        name="behavioral_interaction",
        role=MeasurementRole.PREDICTIVE,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.TRAJECTORY,
        description="Players happen to interact strategically in actual play.",
        operationalizations=[
            "mutual best-response rates in simulation",
            "empirical table talk or blocking rates",
        ],
        computable_from_ontology_v0=False,
        requires=["simulation_or_empirical_play"],
        distinct_from=["structural_interaction"],
    ),
    MeasurementSpec(
        id="hidden_information",
        name="hidden_information",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.OBSERVATION,
        description=(
            "Information that exists in the underlying state but is unavailable."
        ),
        operationalizations=[
            "items with content_known False in an observation",
            "GameState.data keys absent from the observer's known payloads",
        ],
        computable_from_ontology_v0=True,
        requires=["observation", "game_state"],
        distinct_from=["information_uncertainty", "randomness"],
    ),
    MeasurementSpec(
        id="randomness",
        name="randomness",
        role=MeasurementRole.DESCRIPTIVE,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.GAME,
        description="Stochastic events in the rules (dealing, dice, infection draws).",
        operationalizations=[
            "presence of chance-kind transitions in a trace",
            "rules-level flag extracted later from the rulebook",
        ],
        computable_from_ontology_v0=False,
        requires=["chance_model_or_full_trace"],
        distinct_from=["hidden_information", "information_uncertainty"],
        notes="Chess has no chance events; poker has both hidden cards and chance.",
    ),
    MeasurementSpec(
        id="outcome_rating",
        name="rating",
        role=MeasurementRole.OUTCOME,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.GAME,
        description=(
            "External quality/enjoyment proxy such as a published rating. "
            "This is an observed fact when taken from a source, not a structural "
            "measurement, and not 'fun' stored on Game as a derived field."
        ),
        operationalizations=["use Game.rating as an outcome variable in later stats"],
        computable_from_ontology_v0=False,
        requires=["external_source"],
        notes="Correlation with structure is not causation.",
    ),
    MeasurementSpec(
        id="outcome_rating_count",
        name="rating_count",
        role=MeasurementRole.OUTCOME,
        status=MeasurementStatus.PROPOSED,
        scope=MeasurementScope.GAME,
        description="How many ratings support an acclaim signal.",
        operationalizations=["Game.rating_count from a source"],
        computable_from_ontology_v0=False,
        requires=["external_source"],
    ),
    MeasurementSpec(
        id="outcome_playtest_rating",
        name="playtest_rating",
        role=MeasurementRole.OUTCOME,
        status=MeasurementStatus.FUTURE,
        scope=MeasurementScope.GAME,
        description="Human playtest judgments of quality, tension, or replayability.",
        operationalizations=["separate playtest protocol, not inferred from rules"],
        computable_from_ontology_v0=False,
        requires=["human_playtest"],
        distinct_from=["outcome_rating"],
    ),
)
