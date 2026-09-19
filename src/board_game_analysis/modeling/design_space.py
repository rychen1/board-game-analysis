"""Game and situation structural representations. Not metadata embeddings."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ds_platform.modeling.representations import (
    RepresentationTable,
    vector_distance,
)
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import PlaySituation
from board_game_analysis.modeling._util import (
    composite_source_payload_id,
    jaccard_distance,
    known_item_ids,
    mean_floats,
    mean_vector,
)
from board_game_analysis.modeling.counterfactuals import COUNTERFACTUAL_REPR_LOGICAL_KEY
from board_game_analysis.modeling.encoders.action import ActionBagEmbedder
from board_game_analysis.modeling.encoders.observation import ObservationBagEmbedder
from board_game_analysis.modeling.encoders.state import StateBagEmbedder
from board_game_analysis.modeling.examples import (
    ExampleBundle,
    ObservationExample,
    examples_from_situation,
    situation_id_for,
)
from board_game_analysis.modeling.higher_order import (
    HIGHER_ORDER_REPR_LOGICAL_KEY,
    HigherOrderPerspective,
    count_asymmetric_pairs,
    higher_order_pairs,
)
from board_game_analysis.modeling.interventions import (
    InterventionSummary,
    intervention_summaries_for_observations,
    observation_scalars,
)
from board_game_analysis.modeling.pairs import (
    TransitionPairBundle,
    transition_pairs_from_situation,
)
from board_game_analysis.modeling.sequences import (
    SequenceBundle,
    sequence_from_situation,
)

DESIGN_SPACE_FAMILY = "design-space-v0"
SCHEMA_VERSION = "design-space-v0"

FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "title",
        "publisher",
        "publishers",
        "designer",
        "designers",
        "release_year",
        "year",
        "rating",
        "rating_count",
        "popularity",
        "category",
        "categories",
        "mechanic",
        "mechanics",
        "source",
        "sources",
        "url",
        "corpus",
        "rank",
        "complexity",
        "min_players",
        "max_players",
        "min_play_time_minutes",
        "max_play_time_minutes",
    }
)

_STATE_COLUMNS = (
    "n_states",
    "n_phases",
    "n_active_players",
    "centroid_norm",
    "dispersion",
)
_PERSPECTIVE_COLUMNS = (
    "n_observations",
    "n_observers",
    "mean_volume",
    "mean_visibility",
    "mean_hidden",
    "n_asymmetry_pairs",
    "mean_asymmetry",
    "dispersion",
)
_ACTION_COLUMNS = (
    "n_actions",
    "n_action_types",
    "n_transitions",
    "n_actors",
    "n_decision_spaces",
    "n_decision_options",
    "dispersion",
)
_SEQUENCE_COLUMNS = ("n_trajectories", "mean_steps", "max_steps")
_INTERVENTION_COLUMNS = (
    "n_items",
    "mean_hide_visibility_delta",
    "mean_reveal_visibility_delta",
)
_HIGHER_ORDER_COLUMNS = (
    "n_pairs",
    "mean_distance",
    "mean_only_focal",
    "mean_only_target",
    "mean_shared",
    "n_asymmetric",
)
_REGION_COLUMNS = ("n_units", "dispersion", "range")

_CORPUS_NOTE = (
    "structural position in a learned representation space; not game "
    "quality, fun, commercial potential, or objective originality"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureFamilySpec(_FrozenModel):
    name: str
    columns: tuple[str, ...]


class FeatureSchema(_FrozenModel):
    version: str = SCHEMA_VERSION
    families: tuple[FeatureFamilySpec, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for family in self.families:
            names.append(f"{family.name}__present")
            names.extend(f"{family.name}__{column}" for column in family.columns)
        return tuple(names)

    @property
    def dim(self) -> int:
        return len(self.column_names)


class FamilyBlock(_FrozenModel):
    name: str
    present: bool
    values: tuple[tuple[str, float], ...] = ()
    source_entity_ids: tuple[str, ...] = ()

    def value(self, column: str) -> float:
        for key, item in self.values:
            if key == column:
                return item
        raise KeyError(f"no column {column!r} in family {self.name!r}")


class SituationRepresentation(_FrozenModel):
    entity_id: str
    game_id: str
    situation_id: str
    families: tuple[FamilyBlock, ...]
    vector: tuple[float, ...]
    feature_schema: FeatureSchema
    source_artifacts: tuple[str, ...]
    source_payload_id: str
    notes: tuple[str, ...] = ()

    def family(self, name: str) -> FamilyBlock:
        for block in self.families:
            if block.name == name:
                return block
        raise KeyError(f"no feature family {name!r}")

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class GameRepresentation(_FrozenModel):
    game_id: str
    families: tuple[FamilyBlock, ...]
    vector: tuple[float, ...]
    n_situations: int
    situation_dispersion: float
    situation_range: float
    situation_ids: tuple[str, ...]
    situation_centroid: tuple[float, ...]
    feature_schema: FeatureSchema
    source_artifacts: tuple[str, ...]
    source_payload_id: str
    notes: tuple[str, ...] = ()

    def family(self, name: str) -> FamilyBlock:
        for block in self.families:
            if block.name == name:
                return block
        raise KeyError(f"no feature family {name!r}")

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class PlayLayer(_FrozenModel):
    """Encoded play fragments used to build structural summaries."""

    situations: tuple[PlaySituation, ...]
    examples: tuple[ExampleBundle, ...]
    pairs: tuple[TransitionPairBundle, ...]
    sequences: tuple[SequenceBundle, ...]
    z_states: RepresentationTable
    z_observations: RepresentationTable
    z_actions: RepresentationTable
    encoder_dim: int
    encoder_family: str


class Phase6Context(_FrozenModel):
    """Phase 6 tables consumed by Phase 7 intervention and hop families."""

    higher_order_pairs: tuple[HigherOrderPerspective, ...] = ()
    intervention_summaries: tuple[InterventionSummary, ...] = ()
    source_logical_keys: tuple[str, ...] = ()


def build_phase6_context(layer: PlayLayer) -> Phase6Context:
    """Build the Phase 6 tables that Phase 7 feature families compose."""
    pairs: list[HigherOrderPerspective] = []
    summaries: list[InterventionSummary] = []
    for bundle in layer.examples:
        summaries.extend(intervention_summaries_for_observations(bundle.observations))
        if len(bundle.observations) >= 2:
            pairs.extend(higher_order_pairs(bundle.observations, layer.z_observations))
    return Phase6Context(
        higher_order_pairs=tuple(pairs),
        intervention_summaries=tuple(summaries),
        source_logical_keys=(
            HIGHER_ORDER_REPR_LOGICAL_KEY,
            COUNTERFACTUAL_REPR_LOGICAL_KEY,
        ),
    )


def phase6_context_from_runs(
    *,
    higher_order_pairs: Sequence[HigherOrderPerspective] = (),
    intervention_summaries: Sequence[InterventionSummary] = (),
) -> Phase6Context:
    """Compose Phase 7 features from persisted Phase 6 experiment outputs."""
    keys: list[str] = []
    if higher_order_pairs:
        keys.append(HIGHER_ORDER_REPR_LOGICAL_KEY)
    if intervention_summaries:
        keys.append(COUNTERFACTUAL_REPR_LOGICAL_KEY)
    return Phase6Context(
        higher_order_pairs=tuple(higher_order_pairs),
        intervention_summaries=tuple(intervention_summaries),
        source_logical_keys=tuple(keys),
    )


def default_feature_schema() -> FeatureSchema:
    schema = FeatureSchema(
        families=(
            FeatureFamilySpec(name="state", columns=_STATE_COLUMNS),
            FeatureFamilySpec(name="perspective", columns=_PERSPECTIVE_COLUMNS),
            FeatureFamilySpec(name="action", columns=_ACTION_COLUMNS),
            FeatureFamilySpec(name="sequence", columns=_SEQUENCE_COLUMNS),
            FeatureFamilySpec(name="intervention", columns=_INTERVENTION_COLUMNS),
            FeatureFamilySpec(name="higher_order", columns=_HIGHER_ORDER_COLUMNS),
            FeatureFamilySpec(name="region", columns=_REGION_COLUMNS),
        )
    )
    assert_schema_excludes_metadata(schema)
    return schema


def assert_schema_excludes_metadata(schema: FeatureSchema) -> None:
    """Raise if a schema column looks like catalog metadata."""
    for name in schema.column_names:
        token = name.split("__")[-1]
        if token in FORBIDDEN_METADATA_FIELDS or name in FORBIDDEN_METADATA_FIELDS:
            raise ValueError(f"metadata field is not a structural feature: {name}")


def build_play_layer(situations: Sequence[PlaySituation], *, dim: int) -> PlayLayer:
    if dim <= 0:
        raise ValueError("dim must be positive")
    example_bundles = tuple(examples_from_situation(item) for item in situations)
    pair_bundles = tuple(transition_pairs_from_situation(item) for item in situations)
    sequence_bundles = tuple(sequence_from_situation(item) for item in situations)

    state_examples = [
        example for bundle in example_bundles for example in bundle.states
    ]
    obs_examples = [
        example for bundle in example_bundles for example in bundle.observations
    ]
    action_examples = [example for bundle in pair_bundles for example in bundle.actions]

    state_encoder = StateBagEmbedder(dim=dim)
    z_states = _encode_or_empty(
        state_encoder,
        [example.entity_id for example in state_examples],
        [example.to_encoder_record() for example in state_examples],
        dim,
    )
    obs_encoder = ObservationBagEmbedder(dim=dim)
    z_obs = _encode_or_empty(
        obs_encoder,
        [example.entity_id for example in obs_examples],
        [example.to_encoder_record() for example in obs_examples],
        dim,
    )
    action_encoder = ActionBagEmbedder(dim=dim)
    z_actions = _encode_or_empty(
        action_encoder,
        [example.entity_id for example in action_examples],
        [example.to_encoder_record() for example in action_examples],
        dim,
    )
    return PlayLayer(
        situations=tuple(situations),
        examples=example_bundles,
        pairs=pair_bundles,
        sequences=sequence_bundles,
        z_states=z_states,
        z_observations=z_obs,
        z_actions=z_actions,
        encoder_dim=dim,
        encoder_family=state_encoder.family,
    )


def represent_situation(
    situation: PlaySituation,
    layer: PlayLayer,
    schema: FeatureSchema | None = None,
    *,
    phase6: Phase6Context | None = None,
) -> SituationRepresentation:
    chosen = schema or default_feature_schema()
    resolved = phase6 or build_phase6_context(layer)
    index = _situation_index(layer, situation)
    examples = layer.examples[index]
    pairs = layer.pairs[index]
    sequences = layer.sequences[index]
    situation_id = situation_id_for(situation)
    hop = _higher_order_block_from_pairs(resolved.higher_order_pairs, examples, chosen)
    blocks = (
        _state_block(examples, layer.z_states, chosen),
        _perspective_block(examples, layer.z_observations, chosen),
        _action_block(situation, pairs, layer.z_actions, chosen),
        _sequence_block(sequences, chosen),
        _intervention_block_from_summaries(
            resolved.intervention_summaries, situation_id, chosen
        ),
        hop,
        _region_block(chosen, n_units=1, dispersion=0.0, spread=0.0),
    )
    vector = flatten_families(blocks, chosen)
    sources = _unique(
        list(_source_ids(examples, pairs, sequences))
        + list(hop.source_entity_ids)
        + list(resolved.source_logical_keys)
    )
    lineage = _situation_encoder_payload_ids(examples, pairs, layer)
    return SituationRepresentation(
        entity_id=situation_id,
        game_id=situation.game.id,
        situation_id=situation_id,
        families=blocks,
        vector=vector,
        feature_schema=chosen,
        source_artifacts=sources,
        source_payload_id=composite_source_payload_id(lineage),
        notes=(_CORPUS_NOTE,),
    )


def represent_situations(
    layer: PlayLayer,
    schema: FeatureSchema | None = None,
    *,
    phase6: Phase6Context | None = None,
) -> tuple[SituationRepresentation, ...]:
    chosen = schema or default_feature_schema()
    resolved = phase6 or build_phase6_context(layer)
    return tuple(
        represent_situation(situation, layer, chosen, phase6=resolved)
        for situation in layer.situations
    )


def represent_game(
    situation_reps: Sequence[SituationRepresentation],
) -> GameRepresentation:
    if not situation_reps:
        raise ValueError("game representation requires at least one situation")
    game_ids = {item.game_id for item in situation_reps}
    if len(game_ids) != 1:
        raise ValueError("game representation requires a single game_id")
    schema = situation_reps[0].feature_schema
    if any(item.feature_schema != schema for item in situation_reps):
        raise ValueError("situation feature schemas must match")
    ordered = tuple(sorted(situation_reps, key=lambda item: item.situation_id))
    vectors = [item.vector for item in ordered]
    centroid = mean_vector(vectors)
    dispersion = _dispersion(vectors)
    spread = _pairwise_range(vectors)
    blocks = tuple(
        _region_block(
            schema,
            n_units=len(ordered),
            dispersion=dispersion,
            spread=spread,
            source_ids=tuple(item.entity_id for item in ordered),
        )
        if schema_family.name == "region"
        else _aggregate_family(schema_family, ordered)
        for schema_family in schema.families
    )
    sources = _unique(
        [source for item in ordered for source in item.source_artifacts]
        + [item.entity_id for item in ordered]
    )
    return GameRepresentation(
        game_id=ordered[0].game_id,
        families=blocks,
        vector=flatten_families(blocks, schema),
        n_situations=len(ordered),
        situation_dispersion=dispersion,
        situation_range=spread,
        situation_ids=tuple(item.situation_id for item in ordered),
        situation_centroid=centroid,
        feature_schema=schema,
        source_artifacts=sources,
        source_payload_id=composite_source_payload_id(
            [item.source_payload_id for item in ordered]
        ),
        notes=(_CORPUS_NOTE,),
    )


def represent_games(
    situation_reps: Sequence[SituationRepresentation],
) -> tuple[GameRepresentation, ...]:
    by_game: dict[str, list[SituationRepresentation]] = {}
    order: list[str] = []
    for item in situation_reps:
        if item.game_id not in by_game:
            order.append(item.game_id)
            by_game[item.game_id] = []
        by_game[item.game_id].append(item)
    order = sorted(order)
    return tuple(represent_game(by_game[game_id]) for game_id in order)


def flatten_families(
    blocks: Sequence[FamilyBlock], schema: FeatureSchema
) -> tuple[float, ...]:
    """Concatenate presence flags and family values. Missing families stay flagged."""
    by_name = {block.name: block for block in blocks}
    values: list[float] = []
    for family in schema.families:
        block = by_name.get(family.name)
        if block is None or not block.present:
            values.append(0.0)
            values.extend(0.0 for _ in family.columns)
            continue
        values.append(1.0)
        values.extend(block.value(column) for column in family.columns)
    return tuple(values)


def situation_table(
    items: Sequence[SituationRepresentation],
) -> RepresentationTable:
    return _table(
        [item.entity_id for item in items],
        [item.vector for item in items],
        [item.source_payload_id for item in items],
    )


def game_table(items: Sequence[GameRepresentation]) -> RepresentationTable:
    return _table(
        [item.game_id for item in items],
        [item.vector for item in items],
        [item.source_payload_id for item in items],
    )


def external_metadata(situation: PlaySituation) -> dict[str, Any]:
    """Catalog fields for post-hoc description. Never a representation input."""
    game = situation.game
    return {
        "title": game.title,
        "release_year": game.release_year,
        "rating": game.rating,
        "popularity": game.popularity,
        "publishers": list(game.publishers),
        "designers": list(game.designers),
        "categories": list(game.categories),
        "mechanics": [mechanic.name for mechanic in game.mechanics],
    }


def _encode_or_empty(
    encoder: StateBagEmbedder | ObservationBagEmbedder | ActionBagEmbedder,
    entity_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    dim: int,
) -> RepresentationTable:
    if not entity_ids:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=dim,
            source_payload_ids=(),
        )
    return encoder.encode(entity_ids, records)


def _situation_index(layer: PlayLayer, situation: PlaySituation) -> int:
    for index, item in enumerate(layer.situations):
        if item is situation or (
            item.game.id == situation.game.id
            and situation_id_for(item) == situation_id_for(situation)
        ):
            return index
    raise KeyError(f"situation {situation.game.id!r} is not in the play layer")


def _state_block(
    examples: ExampleBundle,
    z_states: RepresentationTable,
    schema: FeatureSchema,
) -> FamilyBlock:
    columns = _columns(schema, "state")
    states = examples.states
    if not states:
        return FamilyBlock(name="state", present=False)
    vectors = _lookup_vectors(z_states, [item.entity_id for item in states])
    phases = {item.phase for item in states if item.phase}
    players = {item.active_player_id for item in states if item.active_player_id}
    values = {
        "n_states": float(len(states)),
        "n_phases": float(len(phases)),
        "n_active_players": float(len(players)),
        "centroid_norm": _centroid_norm(vectors),
        "dispersion": _dispersion(vectors),
    }
    return FamilyBlock(
        name="state",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(item.entity_id for item in states),
    )


def _perspective_block(
    examples: ExampleBundle,
    z_obs: RepresentationTable,
    schema: FeatureSchema,
) -> FamilyBlock:
    columns = _columns(schema, "perspective")
    observations = examples.observations
    if not observations:
        return FamilyBlock(name="perspective", present=False)
    vectors = _lookup_vectors(z_obs, [item.entity_id for item in observations])
    scalars = [observation_scalars(item) for item in observations]
    pairs = _asymmetry_pairs(observations)
    values = {
        "n_observations": float(len(observations)),
        "n_observers": float(len({item.observer_id for item in observations})),
        "mean_volume": mean_floats([item.volume for item in scalars]),
        "mean_visibility": mean_floats([item.visibility for item in scalars]),
        "mean_hidden": mean_floats([item.hidden for item in scalars]),
        "n_asymmetry_pairs": float(len(pairs)),
        "mean_asymmetry": mean_floats(pairs) if pairs else 0.0,
        "dispersion": _dispersion(vectors),
    }
    return FamilyBlock(
        name="perspective",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(item.entity_id for item in observations),
    )


def _action_block(
    situation: PlaySituation,
    pairs: TransitionPairBundle,
    z_actions: RepresentationTable,
    schema: FeatureSchema,
) -> FamilyBlock:
    columns = _columns(schema, "action")
    actions = situation.actions
    transitions = situation.transitions
    decisions = situation.decision_spaces
    if not actions and not transitions and not decisions:
        return FamilyBlock(name="action", present=False)
    types = {action.action_type for action in actions}
    actors = {action.player_id for action in actions if action.player_id}
    n_options = sum(len(space.options) for space in decisions)
    action_ids = [example.entity_id for example in pairs.actions]
    vectors = _lookup_vectors(z_actions, action_ids) if action_ids else ()
    values = {
        "n_actions": float(len(actions)),
        "n_action_types": float(len(types)),
        "n_transitions": float(len(transitions)),
        "n_actors": float(len(actors)),
        "n_decision_spaces": float(len(decisions)),
        "n_decision_options": float(n_options),
        "dispersion": _dispersion(vectors) if vectors else 0.0,
    }
    return FamilyBlock(
        name="action",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(action_ids or [action.id for action in actions]),
    )


def _sequence_block(bundle: SequenceBundle, schema: FeatureSchema) -> FamilyBlock:
    columns = _columns(schema, "sequence")
    sequences = bundle.sequences
    if not sequences:
        return FamilyBlock(name="sequence", present=False)
    lengths = [float(len(item.steps)) for item in sequences]
    values = {
        "n_trajectories": float(len(sequences)),
        "mean_steps": mean_floats(lengths),
        "max_steps": max(lengths),
    }
    return FamilyBlock(
        name="sequence",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(item.entity_id for item in sequences),
    )


def _intervention_block_from_summaries(
    summaries: Sequence[InterventionSummary],
    situation_id: str,
    schema: FeatureSchema,
) -> FamilyBlock:
    columns = _columns(schema, "intervention")
    scoped = [item for item in summaries if item.situation_id == situation_id]
    if not scoped:
        return FamilyBlock(name="intervention", present=False)
    values = {
        "n_items": float(len(scoped)),
        "mean_hide_visibility_delta": mean_floats(
            [item.hide_visibility_delta for item in scoped]
        ),
        "mean_reveal_visibility_delta": mean_floats(
            [item.reveal_visibility_delta for item in scoped]
        ),
    }
    return FamilyBlock(
        name="intervention",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(item.source_entity_id for item in scoped),
    )


def _higher_order_block_from_pairs(
    pairs: Sequence[HigherOrderPerspective],
    examples: ExampleBundle,
    schema: FeatureSchema,
) -> FamilyBlock:
    columns = _columns(schema, "higher_order")
    if len(examples.observations) < 2:
        return FamilyBlock(name="higher_order", present=False)
    situation_id = examples.states[0].situation_id
    scoped = [pair for pair in pairs if pair.situation_id == situation_id]
    if not scoped:
        return FamilyBlock(name="higher_order", present=False)
    n_asymmetric = count_asymmetric_pairs(scoped)
    values = {
        "n_pairs": float(len(scoped)),
        "mean_distance": mean_floats([pair.distance for pair in scoped]),
        "mean_only_focal": mean_floats([float(pair.n_only_focal) for pair in scoped]),
        "mean_only_target": mean_floats([float(pair.n_only_target) for pair in scoped]),
        "mean_shared": mean_floats([float(pair.n_shared_known) for pair in scoped]),
        "n_asymmetric": float(n_asymmetric),
    }
    return FamilyBlock(
        name="higher_order",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(pair.entity_id for pair in scoped),
    )


def _region_block(
    schema: FeatureSchema,
    *,
    n_units: int,
    dispersion: float,
    spread: float,
    source_ids: Sequence[str] = (),
) -> FamilyBlock:
    columns = _columns(schema, "region")
    values = {
        "n_units": float(n_units),
        "dispersion": float(dispersion),
        "range": float(spread),
    }
    return FamilyBlock(
        name="region",
        present=True,
        values=_ordered(values, columns),
        source_entity_ids=tuple(source_ids),
    )


def _aggregate_family(
    spec: FeatureFamilySpec,
    situations: Sequence[SituationRepresentation],
) -> FamilyBlock:
    present_blocks = [
        item.family(spec.name) for item in situations if item.family(spec.name).present
    ]
    if not present_blocks:
        return FamilyBlock(name=spec.name, present=False)
    values = {
        column: mean_floats([block.value(column) for block in present_blocks])
        for column in spec.columns
    }
    sources = _unique(
        [entity_id for block in present_blocks for entity_id in block.source_entity_ids]
    )
    return FamilyBlock(
        name=spec.name,
        present=True,
        values=_ordered(values, spec.columns),
        source_entity_ids=sources,
    )


def _asymmetry_pairs(observations: Sequence[ObservationExample]) -> list[float]:
    by_state: dict[str, list[ObservationExample]] = {}
    for example in observations:
        by_state.setdefault(example.state_id, []).append(example)
    scores: list[float] = []
    for group in by_state.values():
        known = [known_item_ids(example) for example in group]
        for left_index, left in enumerate(known):
            for right in known[left_index + 1 :]:
                scores.append(jaccard_distance(left, right))
    return scores


def _lookup_vectors(
    table: RepresentationTable, entity_ids: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    index = {entity_id: i for i, entity_id in enumerate(table.entity_ids)}
    vectors: list[tuple[float, ...]] = []
    for entity_id in entity_ids:
        try:
            vectors.append(table.vectors[index[entity_id]])
        except KeyError as exc:
            raise KeyError(f"missing representation {entity_id!r}") from exc
    return tuple(vectors)


def _source_ids(
    examples: ExampleBundle,
    pairs: TransitionPairBundle,
    sequences: SequenceBundle,
) -> tuple[str, ...]:
    return _unique(
        [item.entity_id for item in examples.states]
        + [item.entity_id for item in examples.observations]
        + [item.entity_id for item in pairs.pairs]
        + [item.entity_id for item in sequences.sequences]
    )


def _situation_encoder_payload_ids(
    examples: ExampleBundle,
    pairs: TransitionPairBundle,
    layer: PlayLayer,
) -> tuple[str, ...]:
    """Upstream encoder payload ids that contributed to one situation row."""
    payload_ids: list[str] = []
    for table, entity_ids in (
        (layer.z_states, [item.entity_id for item in examples.states]),
        (
            layer.z_observations,
            [item.entity_id for item in examples.observations],
        ),
        (layer.z_actions, [item.entity_id for item in pairs.actions]),
    ):
        index = {entity_id: idx for idx, entity_id in enumerate(table.entity_ids)}
        for entity_id in entity_ids:
            payload_ids.append(table.source_payload_ids[index[entity_id]])
    return tuple(payload_ids)


def _table_payload_ids(
    table: RepresentationTable, entity_ids: Sequence[str]
) -> tuple[str, ...]:
    index = {entity_id: idx for idx, entity_id in enumerate(table.entity_ids)}
    return tuple(table.source_payload_ids[index[entity_id]] for entity_id in entity_ids)


def _columns(schema: FeatureSchema, name: str) -> tuple[str, ...]:
    for family in schema.families:
        if family.name == name:
            return family.columns
    raise KeyError(f"no feature family {name!r}")


def _ordered(
    values: Mapping[str, float], columns: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    return tuple((column, float(values[column])) for column in columns)


def _dispersion(vectors: Sequence[tuple[float, ...]]) -> float:
    if len(vectors) < 2:
        return 0.0
    centroid = mean_vector(vectors)
    total = sum(vector_distance(vector, centroid, metric="l2") for vector in vectors)
    return total / len(vectors)


def _pairwise_range(vectors: Sequence[tuple[float, ...]]) -> float:
    if len(vectors) < 2:
        return 0.0
    return max(
        vector_distance(left, right, metric="l2")
        for index, left in enumerate(vectors)
        for right in vectors[index + 1 :]
    )


def _centroid_norm(vectors: Sequence[tuple[float, ...]]) -> float:
    if not vectors:
        return 0.0
    centroid = mean_vector(vectors)
    return math.sqrt(sum(value * value for value in centroid))


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _table(
    entity_ids: Sequence[str],
    vectors: Sequence[tuple[float, ...]],
    source_payload_ids: Sequence[str],
) -> RepresentationTable:
    if not entity_ids:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=0,
            source_payload_ids=(),
        )
    if len(source_payload_ids) != len(entity_ids):
        raise ValueError("source_payload_ids length must match entity_ids length")
    return RepresentationTable(
        entity_ids=tuple(entity_ids),
        vectors=tuple(vectors),
        dim=len(vectors[0]),
        source_payload_ids=tuple(source_payload_ids),
    )


ViewName = Literal["canonical", "normalized", "projected"]
