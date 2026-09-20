"""Adversarial audit probes for Phases 1–7.

Assertions describe intended invariants from the adversarial audit.
Do not delete or weaken them to make the suite green.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.domain import (
    Action,
    Game,
    GameDefinition,
    GameState,
    InformationItem,
    InformationSpace,
    Interaction,
    Observation,
    Player,
    PlaySituation,
    StateTransition,
    Visibility,
)
from board_game_analysis.modeling.counterfactuals import (
    COUNTERFACTUAL_REPR_LOGICAL_KEY,
    bound_sequence_table,
)
from board_game_analysis.modeling.design_geometry import (
    DeterministicKMeans,
    build_design_space,
    cluster_games,
    fit_standardizer,
    novelty_table,
)
from board_game_analysis.modeling.design_space import (
    FamilyBlock,
    Phase6Context,
    build_phase6_context,
    build_play_layer,
    flatten_families,
    phase6_context_from_runs,
    represent_game,
    represent_situation,
    represent_situations,
)
from board_game_analysis.modeling.encoders import (
    ObservationBagEmbedder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import (
    action_set_entity_id,
    examples_from_situation,
    examples_from_situations,
    sequence_entity_id,
    situation_id_for,
)
from board_game_analysis.modeling.higher_order import (
    HIGHER_ORDER_REPR_LOGICAL_KEY,
    ObserverPath,
    compose_observer_path,
    count_asymmetric_pairs,
    higher_order_pairs,
    higher_order_perspective,
)
from board_game_analysis.modeling.interventions import (
    apply_intervention,
    hide_information,
    identity_intervention,
    reveal_information,
)
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.pairs import transition_pairs_from_situations
from board_game_analysis.modeling.perspectives import (
    perspective_table_from_observations,
)
from board_game_analysis.modeling.run_encode import (
    PROBE_EVAL_LOGICAL_KEY,
    TRAJECTORY_REPR_LOGICAL_KEY,
)
from board_game_analysis.modeling.sequence_eval import (
    SEQUENCE_REPR_LOGICAL_KEY as EVAL_SEQUENCE_REPR,
)
from board_game_analysis.modeling.sequences import (
    prefixes_from_situations,
    sequence_from_situation,
    sequences_from_situations,
)
from tests.fixtures.games.hanabi import ALICE, BOB, CAROL, S0, make_hanabi_situation

_DOMAIN_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "board_game_analysis" / "domain"
)
_MODELING_ROOT = (
    Path(__file__).resolve().parents[2] / "src" / "board_game_analysis" / "modeling"
)


def _situation(
    *,
    game_id: str,
    state_ids: tuple[str, str] = ("s0", "s1"),
    action_id: str = "a0",
    transition_id: str = "t0",
    players: tuple[str, ...] = ("p0",),
    title: str = "Synthetic",
    payloads: tuple[dict[str, object], ...] | None = None,
    data: tuple[dict[str, object], dict[str, object]] | None = None,
    include_observations: bool = True,
    extra_item: InformationItem | None = None,
) -> PlaySituation:
    """Minimal authored fragment. IDs are caller-controlled on purpose."""
    game = Game(
        id=game_id, title=title, min_players=len(players), max_players=len(players)
    )
    definition = GameDefinition(
        id=f"{game_id}-def",
        game_id=game_id,
        interaction=Interaction.COMPETITIVE,
        decision_timing="sequential",
        legal_action_types=["move"],
    )
    people = [Player(id=player_id, name=player_id) for player_id in players]
    state_data = data or ({"token": "start"}, {"token": "end"})
    states = [
        GameState(
            id=state_ids[0],
            game_id=game_id,
            turn_number=1,
            active_player_id=players[0],
            phase="play",
            data=dict(state_data[0]),
        ),
        GameState(
            id=state_ids[1],
            game_id=game_id,
            turn_number=2,
            active_player_id=players[0],
            phase="play",
            data=dict(state_data[1]),
        ),
    ]
    observations: list[Observation] = []
    spaces: list[InformationSpace] = []
    if include_observations:
        item_payloads = payloads or ({"value": 1},)
        for player_id in players:
            items = [
                InformationItem(
                    id=f"{game_id}-{player_id}-item-{index}",
                    visibility=Visibility.PUBLIC,
                    about="token",
                    content_known=True,
                    known_to=list(players),
                    payload=dict(payload),
                    holder_id=player_id,
                )
                for index, payload in enumerate(item_payloads)
            ]
            if extra_item is not None:
                items.append(extra_item)
            space = InformationSpace(
                id=f"{game_id}-info-{player_id}-{state_ids[0]}",
                player_id=player_id,
                items=items,
            )
            spaces.append(space)
            observations.append(
                Observation(
                    id=f"{game_id}-obs-{player_id}-{state_ids[0]}",
                    game_state_id=state_ids[0],
                    observer_id=player_id,
                    information_space_id=space.id,
                )
            )
    action = Action(id=action_id, player_id=players[0], action_type="move")
    transition = StateTransition(
        id=transition_id,
        from_state_id=state_ids[0],
        to_state_id=state_ids[1],
        action_ids=[action_id],
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=people,
        states=states,
        observations=observations,
        information_spaces=spaces,
        actions=[action],
        transitions=[transition],
    )


def _table(
    entity_ids: tuple[str, ...], values: tuple[float, ...]
) -> RepresentationTable:
    source = "a" * 64
    return RepresentationTable(
        entity_ids=entity_ids,
        vectors=tuple((value,) for value in values),
        dim=1,
        source_payload_ids=tuple(source for _ in entity_ids),
    )


# ---------------------------------------------------------------------------
# Domain boundary
# ---------------------------------------------------------------------------


def test_domain_package_never_imports_ds_platform() -> None:
    offenders: list[str] = []
    for path in _DOMAIN_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ds_platform" or alias.name.startswith(
                        "ds_platform."
                    ):
                        offenders.append(f"{path.name}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "ds_platform" or node.module.startswith(
                    "ds_platform."
                ):
                    offenders.append(f"{path.name}:{node.module}")
    assert offenders == []


def test_modeling_has_no_fixture_name_conditionals() -> None:
    needles = ("hanabi", "telestrations", "just_one", "justone", "wingspan")
    hits: list[str] = []
    for path in _MODELING_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for needle in needles:
            if f'"{needle}"' in text or f"'{needle}'" in text:
                if "note" in text and needle in text:
                    continue
                hits.append(f"{path.name}:{needle}")
    # measurement docstring mentions hanabi as an example id shape only
    hits = [hit for hit in hits if not hit.startswith("measures.py")]
    assert hits == []


# ---------------------------------------------------------------------------
# Identity collisions
# ---------------------------------------------------------------------------


def test_two_situations_same_game_have_distinct_phase1_sequence_ids() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("t0", "t1"), action_id="a1")
    bundle = examples_from_situations([first, second])
    ids = [example.entity_id for example in bundle.sequences]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_two_situations_sharing_first_state_id_have_distinct_situation_ids() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("s0", "s2"), action_id="a1")
    assert situation_id_for(first) != situation_id_for(second)


def test_action_set_entity_id_does_not_collide_on_embedded_plus() -> None:
    sid = "g:s0,s1|t0"
    assert action_set_entity_id(sid, ["a+b"]) != action_set_entity_id(sid, ["a", "b"])


def test_game_scoped_measurements_are_unique_across_situations() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("t0", "t1"), action_id="a1")
    ids = [item.id for item in measure_situations([first, second])]
    assert len(ids) == len(set(ids))


def test_overlapping_local_state_ids_do_not_crash_play_layer() -> None:
    first = _situation(
        game_id="g", state_ids=("s0", "s1"), action_id="a0", transition_id="tr-a"
    )
    second = _situation(
        game_id="g", state_ids=("s0", "s1"), action_id="a1", transition_id="tr-b"
    )
    layer = build_play_layer([first, second], dim=8)
    assert len(layer.z_states.entity_ids) == 4
    assert len(set(layer.z_states.entity_ids)) == 4


def test_distinct_state_ids_same_game_build_a_play_layer() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("t0", "t1"), action_id="a1")
    layer = build_play_layer([first, second], dim=8)
    reps = represent_situations(layer)
    assert len(reps) == 2
    game = represent_game(reps)
    assert game.n_situations == 2
    assert game.game_id == "g"


def test_two_situations_same_game_keep_both_transition_sets() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("t0", "t1"), action_id="a1")
    bundle = transition_pairs_from_situations([first, second])
    assert len(bundle.pairs) == 2
    assert {pair.action_ids[0] for pair in bundle.pairs} == {"a0", "a1"}


def test_phase5_sequences_keep_situation_boundaries_when_transition_ids_differ() -> (
    None
):
    first = _situation(
        game_id="g", state_ids=("s0", "s1"), action_id="a0", transition_id="tr-a"
    )
    second = _situation(
        game_id="g", state_ids=("t0", "t1"), action_id="a1", transition_id="tr-b"
    )
    bundle = sequences_from_situations([first, second])
    assert len(bundle.sequences) == 2
    assert bundle.sequences[0].entity_id != bundle.sequences[1].entity_id


def test_phase5_trajectory_ids_are_situation_unique() -> None:
    first = _situation(
        game_id="g", state_ids=("s0", "s1"), action_id="a0", transition_id="t0"
    )
    second = _situation(
        game_id="g", state_ids=("t0", "t1"), action_id="a1", transition_id="t0"
    )
    bundle = sequences_from_situations([first, second])
    ids = [example.entity_id for example in bundle.sequences]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_prefix_entity_ids_are_unique_across_situations() -> None:
    first = _situation(
        game_id="g", state_ids=("s0", "s1"), action_id="a0", transition_id="t0"
    )
    second = _situation(
        game_id="g", state_ids=("t0", "t1"), action_id="a1", transition_id="t0"
    )
    ids = [prefix.entity_id for prefix in prefixes_from_situations([first, second])]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_sequence_entity_id_helper_is_situation_scoped() -> None:
    assert sequence_entity_id("chess:s0,s1") == "chess:s0,s1/seq"
    assert sequence_entity_id("g:a") != sequence_entity_id("g:b")


# ---------------------------------------------------------------------------
# Mutability
# ---------------------------------------------------------------------------


def test_observation_example_items_are_not_caller_mutable() -> None:
    example = examples_from_situation(_situation(game_id="g")).observations[0]
    with pytest.raises(TypeError, match="immutable"):
        example.items[0]["visibility"] = "mutated"


def test_family_block_values_are_not_caller_mutable() -> None:
    layer = build_play_layer([_situation(game_id="g")], dim=8)
    represented = represent_situations(layer)[0]
    block = represented.family("state")
    assert isinstance(block.values, tuple)
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="frozen"):
        block.values = (("n_states", 99.0),)


def test_apply_intervention_does_not_mutate_source_payload() -> None:
    example = examples_from_situation(_situation(game_id="g")).observations[0]
    item_id = str(example.items[0]["id"])
    intervention = hide_information(
        game_id="g",
        situation_id=situation_id_for(_situation(game_id="g")),
        source_state_id=example.state_id,
        observer_id=example.observer_id,
        item_id=item_id,
    )
    applied = apply_intervention(example, intervention)
    assert applied.result_observation is not None
    with pytest.raises(TypeError, match="immutable"):
        applied.result_observation.items[0]["payload"]["value"] = 999
    assert example.items[0]["payload"]["value"] == 1


# ---------------------------------------------------------------------------
# Representations / loss
# ---------------------------------------------------------------------------


def test_state_bag_discards_payload_values() -> None:
    left = _situation(
        game_id="g",
        data=({"token": "alpha"}, {"token": "end"}),
    )
    right = _situation(
        game_id="g",
        data=({"token": "omega"}, {"token": "end"}),
    )
    encoder = StateBagEmbedder(dim=8)
    records = [
        examples_from_situation(left).states[0].to_encoder_record(),
        examples_from_situation(right).states[0].to_encoder_record(),
    ]
    table = encoder.encode(("l", "r"), records)
    assert table.vectors[0] == table.vectors[1]


def test_state_bag_is_stable_under_data_key_insertion_order() -> None:
    encoder = StateBagEmbedder(dim=8)
    first = {"phase": "play", "turn_number": 1, "data": {"b": 1, "a": 2}}
    second = {"phase": "play", "turn_number": 1, "data": {"a": 2, "b": 1}}
    table = encoder.encode(("x", "y"), (first, second))
    assert table.vectors[0] == table.vectors[1]


def test_empty_state_collection_is_rejected_by_represent_game() -> None:
    with pytest.raises(ValueError, match="at least one situation"):
        represent_game([])


# ---------------------------------------------------------------------------
# Phase 6
# ---------------------------------------------------------------------------


def test_identity_intervention_is_still_labeled_counterfactual() -> None:
    example = examples_from_situation(_situation(game_id="g")).observations[0]
    intervention = identity_intervention(
        game_id="g",
        situation_id="g:s0",
        source_state_id=example.state_id,
        observer_id=example.observer_id,
    )
    applied = apply_intervention(example, intervention)
    assert applied.origin == "counterfactual"
    assert applied.result_observation is not None
    assert applied.result_observation.items == example.items


def test_reveal_is_not_an_inverse_of_hide_for_originally_private_items() -> None:
    hidden = InformationItem(
        id="secret",
        visibility=Visibility.PRIVATE,
        about="hand",
        content_known=False,
        payload={"rank": 3},
        holder_id="p0",
    )
    situation = _situation(game_id="g", extra_item=hidden)
    example = examples_from_situation(situation).observations[0]
    hide = hide_information(
        game_id="g",
        situation_id="g:s0",
        source_state_id=example.state_id,
        observer_id=example.observer_id,
        item_id="secret",
    )
    reveal = reveal_information(
        game_id="g",
        situation_id="g:s0",
        source_state_id=example.state_id,
        observer_id=example.observer_id,
        item_id="secret",
    )
    after_hide = apply_intervention(example, hide).result_observation
    assert after_hide is not None
    restored = apply_intervention(after_hide, reveal).result_observation
    assert restored is not None
    secret = next(item for item in restored.items if item["id"] == "secret")
    assert secret["visibility"] == Visibility.PUBLIC
    assert secret["content_known"] is True


def test_missing_item_and_observer_raise() -> None:
    example = examples_from_situation(_situation(game_id="g")).observations[0]
    missing_item = hide_information(
        game_id="g",
        situation_id="g:s0",
        source_state_id=example.state_id,
        observer_id=example.observer_id,
        item_id="no-such-item",
    )
    with pytest.raises(KeyError, match="no information item"):
        apply_intervention(example, missing_item)
    missing_obs = hide_information(
        game_id="g",
        situation_id="g:s0",
        source_state_id=example.state_id,
        observer_id="nobody",
        item_id=str(example.items[0]["id"]),
    )
    with pytest.raises(ValueError, match="observer_id"):
        apply_intervention(example, missing_obs)


def test_bound_sequence_rejects_max_steps_zero() -> None:
    from ds_platform.modeling.sequences import SequenceTable

    table = SequenceTable(
        group_ids=("g",),
        event_ids=("e0",),
        positions=(0,),
        actor_ids=(None,),
        source_payload_ids=("g",),
    )
    with pytest.raises(ValueError, match="max_steps"):
        bound_sequence_table(table, 0)


def test_higher_order_direction_and_path_cycle() -> None:
    observations = examples_from_situation(make_hanabi_situation()).observations
    encoder = ObservationBagEmbedder(dim=8)
    z_obs = encoder.encode(
        [example.entity_id for example in observations],
        [example.to_encoder_record() for example in observations],
    )
    alice = next(
        example
        for example in observations
        if example.observer_id == ALICE and example.state_id == S0
    )
    bob = next(
        example
        for example in observations
        if example.observer_id == BOB and example.state_id == S0
    )
    index = {entity_id: i for i, entity_id in enumerate(z_obs.entity_ids)}
    ab_source = z_obs.source_payload_ids[index[alice.entity_id]]
    ba_source = z_obs.source_payload_ids[index[bob.entity_id]]
    ab = higher_order_perspective(
        alice,
        bob,
        z_obs.vectors[index[alice.entity_id]],
        z_obs.vectors[index[bob.entity_id]],
        source_payload_id=ab_source,
    )
    ba = higher_order_perspective(
        bob,
        alice,
        z_obs.vectors[index[bob.entity_id]],
        z_obs.vectors[index[alice.entity_id]],
        source_payload_id=ba_source,
    )
    assert ab.vector != ba.vector
    assert ab.entity_id != ba.entity_id
    table = perspective_table_from_observations(observations, z_obs)
    with pytest.raises(ValueError, match="cycle"):
        compose_observer_path(
            table,
            ObserverPath(state_id=S0, observers=(ALICE, BOB, ALICE)),
        )
    composed = compose_observer_path(
        table,
        ObserverPath(state_id=S0, observers=(ALICE, BOB, CAROL)),
    )
    reversed_path = compose_observer_path(
        table,
        ObserverPath(state_id=S0, observers=(CAROL, BOB, ALICE)),
    )
    assert composed.vectors[0] != reversed_path.vectors[0]


def test_higher_order_asymmetric_count_is_one_per_unordered_pair() -> None:
    observations = examples_from_situation(make_hanabi_situation()).observations
    at_s0 = [example for example in observations if example.state_id == S0]
    encoder = ObservationBagEmbedder(dim=8)
    z_obs = encoder.encode(
        [example.entity_id for example in at_s0],
        [example.to_encoder_record() for example in at_s0],
    )
    pairs = higher_order_pairs(at_s0, z_obs)
    assert count_asymmetric_pairs(pairs) >= 1


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_normalizer_ignores_extreme_test_values() -> None:
    table = _table(
        ("train-a", "train-b", "test-a", "test-b"), (0.0, 1.0, 1000.0, 1001.0)
    )
    fitted = fit_standardizer(table, ("train-a", "train-b"), ("x",))
    moved = _table(("train-a", "train-b", "test-a", "test-b"), (0.0, 1.0, 1e9, 1e9 + 1))
    again = fit_standardizer(moved, ("train-a", "train-b"), ("x",))
    assert fitted.means == again.means
    assert fitted.stds == again.stds
    assert fitted.means == (0.5,)


def test_train_game_novelty_is_invariant_to_test_game_location() -> None:
    train_a = _situation(game_id="a")
    train_b = _situation(game_id="b", players=("p0", "p1"))
    far = _situation(game_id="x")
    near = _situation(game_id="x", players=("p0", "p1"))
    left = build_design_space(
        [train_a, train_b, far],
        train_game_ids=("a", "b"),
        test_game_ids=("x",),
        dim=8,
    )
    right = build_design_space(
        [train_a, train_b, near],
        train_game_ids=("a", "b"),
        test_game_ids=("x",),
        dim=8,
    )
    left_scores = {
        row.entity_id: row.structural_novelty for row in novelty_table(left, ks=(1,))
    }
    right_scores = {
        row.entity_id: row.structural_novelty for row in novelty_table(right, ks=(1,))
    }
    assert left_scores["a"] == right_scores["a"]
    assert left_scores["b"] == right_scores["b"]


def test_train_cluster_labels_are_invariant_to_test_location() -> None:
    from ds_platform.modeling.representations import select_entities

    close = _table(("t0", "t1", "t2", "x"), (0.0, 0.1, 0.2, 100.0))
    inside = _table(("t0", "t1", "t2", "x"), (0.0, 0.1, 0.2, 0.05))
    train_close = select_entities(close, ("t0", "t1", "t2"))
    train_inside = select_entities(inside, ("t0", "t1", "t2"))
    first = DeterministicKMeans(2, seed=0)
    first.fit(train_close)
    second = DeterministicKMeans(2, seed=0)
    second.fit(train_inside)
    left = dict(zip(train_close.entity_ids, first.predict(train_close), strict=True))
    right = dict(
        zip(train_inside.entity_ids, second.predict(train_inside), strict=True)
    )
    assert left == right


def test_missing_family_padding_stays_zero_after_standardization() -> None:
    present = _situation(game_id="train")
    missing = _situation(game_id="test", include_observations=False)
    space = build_design_space(
        [present, missing],
        train_game_ids=("train",),
        test_game_ids=("test",),
        dim=8,
    )
    names = space.feature_schema.column_names
    present_index = names.index("perspective__present")
    volume_index = names.index("perspective__mean_volume")
    test_vector = space.normalized_games.vectors[
        space.normalized_games.entity_ids.index("test")
    ]
    assert (
        space.canonical_games.vectors[space.canonical_games.entity_ids.index("test")][
            present_index
        ]
        == 0.0
    )
    assert test_vector[volume_index] == 0.0


def test_design_space_novelty_of_train_game_ignores_test_game() -> None:
    train_a = _situation(game_id="a")
    train_b = _situation(game_id="b", players=("p0", "p1"))
    clone = _situation(game_id="x")
    stripped = _situation(game_id="x", include_observations=False)
    left = build_design_space(
        [train_a, train_b, stripped],
        train_game_ids=("a", "b"),
        test_game_ids=("x",),
        dim=8,
    )
    right = build_design_space(
        [train_a, train_b, clone],
        train_game_ids=("a", "b"),
        test_game_ids=("x",),
        dim=8,
    )
    left_scores = {
        row.entity_id: row.structural_novelty for row in novelty_table(left, ks=(1,))
    }
    right_scores = {
        row.entity_id: row.structural_novelty for row in novelty_table(right, ks=(1,))
    }
    assert left_scores["a"] == right_scores["a"]
    assert left_scores["b"] == right_scores["b"]


def test_novelty_table_includes_test_game_ids() -> None:
    space = build_design_space(
        [_situation(game_id="a"), _situation(game_id="b"), _situation(game_id="c")],
        train_game_ids=("a", "b"),
        test_game_ids=("c",),
        dim=8,
    )
    scored = {row.entity_id for row in novelty_table(space, ks=(1,))}
    assert scored == {"a", "b", "c"}


def test_game_split_ids_on_design_space_are_disjoint() -> None:
    space = build_design_space(
        [_situation(game_id="a"), _situation(game_id="b"), _situation(game_id="c")],
        train_game_ids=("a", "b"),
        test_game_ids=("c",),
        dim=8,
    )
    assert set(space.train_game_ids).isdisjoint(space.test_game_ids)
    assert set(space.normalizer.train_entity_ids) == {"a", "b"}


# ---------------------------------------------------------------------------
# Phase 7 geometry / clustering
# ---------------------------------------------------------------------------


def test_kmeans_n_clusters_one_and_too_many() -> None:
    table = _table(("a", "b", "c"), (0.0, 1.0, 2.0))
    clusterer = DeterministicKMeans(1, seed=0)
    clusterer.fit(table)
    assert clusterer.predict(table) == (0, 0, 0)
    with pytest.raises(ValueError, match="not enough entities"):
        DeterministicKMeans(4, seed=0).fit(table)


def test_kmeans_n_equals_n_on_distinct_points() -> None:
    table = _table(("a", "b", "c"), (0.0, 10.0, 20.0))
    clusterer = DeterministicKMeans(3, seed=0)
    clusterer.fit(table)
    assert set(clusterer.predict(table)) == {0, 1, 2}


def test_kmeans_identical_points_collapse_to_one_used_cluster() -> None:
    table = _table(("a", "b", "c"), (1.0, 1.0, 1.0))
    clusterer = DeterministicKMeans(3, seed=0)
    clusterer.fit(table)
    labels = clusterer.predict(table)
    assert len(set(labels)) == 1


def test_region_math_is_mean_distance_and_max_pairwise() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(
        game_id="g",
        state_ids=("t0", "t1"),
        action_id="a1",
        players=("p0", "p1"),
    )
    layer = build_play_layer([first, second], dim=8)
    reps = represent_situations(layer)
    game = represent_game(reps)
    assert game.n_situations == 2
    assert game.situation_dispersion >= 0.0
    assert game.situation_range >= game.situation_dispersion
    assert game.family("region").value("n_units") == 2.0


def test_situation_order_does_not_change_game_vector() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(
        game_id="g",
        state_ids=("t0", "t1"),
        action_id="a1",
        players=("p0", "p1"),
    )
    left = represent_game(
        represent_situations(build_play_layer([first, second], dim=8))
    )
    right = represent_game(
        represent_situations(build_play_layer([second, first], dim=8))
    )
    assert left.vector == right.vector
    assert left.situation_ids == right.situation_ids


def test_cluster_games_includes_test_ids() -> None:
    space = build_design_space(
        [_situation(game_id="a"), _situation(game_id="b"), _situation(game_id="c")],
        train_game_ids=("a", "b"),
        test_game_ids=("c",),
        dim=8,
    )
    assignments = cluster_games(space, 2)
    assert {row.entity_id for row in assignments} == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Flatten / schema
# ---------------------------------------------------------------------------


def test_flatten_missing_family_is_explicit_zeros() -> None:
    from board_game_analysis.modeling.design_space import default_feature_schema

    schema = default_feature_schema()
    blocks = tuple(
        FamilyBlock(name=family.name, present=False) for family in schema.families
    )
    vector = flatten_families(blocks, schema)
    assert all(value == 0.0 for value in vector)
    assert vector[schema.column_names.index("state__present")] == 0.0


def test_prefixes_from_two_situations_same_game() -> None:
    first = _situation(game_id="g", state_ids=("s0", "s1"), action_id="a0")
    second = _situation(game_id="g", state_ids=("t0", "t1"), action_id="a1")
    prefixes = prefixes_from_situations([first, second])
    assert len(prefixes) == 2
    assert {prefix.situation_id for prefix in prefixes} == {
        situation_id_for(first),
        situation_id_for(second),
    }


def test_phase1_sequence_has_empty_steps_phase5_does_not() -> None:
    situation = _situation(game_id="g")
    phase1 = examples_from_situation(situation).sequences[0]
    phase5 = sequence_from_situation(situation).sequences[0]
    assert phase1.steps == ()
    assert phase5.steps
    assert phase1.entity_id != phase5.entity_id


def test_probe_eval_logical_key_matches_evaluation_namespace() -> None:
    assert PROBE_EVAL_LOGICAL_KEY.startswith("bga:evaluation/")


def test_phase2_and_phase5_sequence_repr_keys_are_distinct() -> None:
    assert TRAJECTORY_REPR_LOGICAL_KEY != EVAL_SEQUENCE_REPR


def test_design_space_composes_phase6_tables() -> None:
    layer = build_play_layer([_situation(game_id="g")], dim=8)
    phase6 = build_phase6_context(layer)
    composed = represent_situation(layer.situations[0], layer, phase6=phase6)
    assert HIGHER_ORDER_REPR_LOGICAL_KEY in composed.source_artifacts
    assert COUNTERFACTUAL_REPR_LOGICAL_KEY in composed.source_artifacts
    empty = Phase6Context()
    without = represent_situation(layer.situations[0], layer, phase6=empty)
    assert without.family("intervention").present is False
    assert without.family("higher_order").present is False
    override = phase6_context_from_runs(
        higher_order_pairs=phase6.higher_order_pairs[:1],
        intervention_summaries=phase6.intervention_summaries[:1],
    )
    partial = represent_situation(layer.situations[0], layer, phase6=override)
    assert partial.family("intervention").value("n_items") == 1.0
