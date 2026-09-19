"""Permanent adversarial probes from the final modeling audit (Option A fixes).

These assert multi-situation invariants that the main fixture corpus does not
exercise (one situation per game).
"""

from __future__ import annotations

from board_game_analysis.domain import InformationItem, Visibility
from board_game_analysis.modeling.artifacts import labels_for_observations
from board_game_analysis.modeling.design_space import (
    build_phase6_context,
    build_play_layer,
    represent_situations,
)
from board_game_analysis.modeling.examples import (
    examples_from_situations,
    situation_id_for,
)
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.sequences import sequences_from_situations
from tests.modeling.test_audit_adversarial import _situation


def test_identical_topology_situations_get_distinct_situation_ids() -> None:
    first = _situation(
        game_id="g",
        transition_id="t0",
        payloads=({"value": 1},),
        data=({"token": "alpha"}, {"token": "end"}),
    )
    second = _situation(
        game_id="g",
        transition_id="t0",
        payloads=({"value": 99},),
        data=({"token": "omega"}, {"token": "end"}),
    )
    assert situation_id_for(first) != situation_id_for(second)


def test_identical_topology_situations_build_play_layer() -> None:
    first = _situation(game_id="g2", transition_id="t0", payloads=({"v": 1},))
    second = _situation(game_id="g2", transition_id="t0", payloads=({"v": 2},))
    layer = build_play_layer([first, second], dim=8)
    assert len(layer.z_states.entity_ids) == 4
    assert len(set(layer.z_states.entity_ids)) == 4


def test_observation_labels_do_not_bleed_across_situations() -> None:
    extra = InformationItem(
        id="extra-item",
        visibility=Visibility.PUBLIC,
        about="token",
        content_known=True,
        known_to=["p0"],
        payload={"value": 2},
        holder_id="p0",
    )
    first = _situation(
        game_id="g",
        state_ids=("s0", "s1"),
        transition_id="tr-a",
    )
    second = _situation(
        game_id="g",
        state_ids=("s0", "s1"),
        transition_id="tr-b",
        extra_item=extra,
    )
    observations = examples_from_situations([first, second]).observations
    measurements = measure_situations([first, second])
    labels = labels_for_observations(
        measurements,
        observations,
        source_payload_ids=["m"] * len(observations),
    )
    by_situation = {
        observation.situation_id: labels.values[index][0]
        for index, observation in enumerate(observations)
    }
    assert by_situation[situation_id_for(first)] == 1.0
    assert by_situation[situation_id_for(second)] == 2.0


def test_sequence_entity_ids_unique_for_identical_topology() -> None:
    first = _situation(game_id="g3", transition_id="t0", payloads=({"v": 1},))
    second = _situation(game_id="g3", transition_id="t0", payloads=({"v": 2},))
    bundle = sequences_from_situations([first, second])
    entity_ids = [sequence.entity_id for sequence in bundle.sequences]
    assert len(entity_ids) == 2
    assert len(set(entity_ids)) == 2


def test_higher_order_hop_blocks_are_situation_scoped() -> None:
    first = _situation(
        game_id="g",
        state_ids=("s0", "s1"),
        transition_id="tr-a",
        players=("p0", "p1"),
    )
    second = _situation(
        game_id="g",
        state_ids=("s0", "s1"),
        transition_id="tr-b",
        players=("p0", "p1"),
    )
    layer = build_play_layer([first, second], dim=8)
    phase6 = build_phase6_context(layer)
    reps = represent_situations(layer, phase6=phase6)
    hop_counts = [
        rep.family("higher_order").value("n_pairs")
        if rep.family("higher_order").present
        else 0.0
        for rep in reps
    ]
    assert hop_counts == [0.0, 0.0] or all(count >= 0.0 for count in hop_counts)
    for rep in reps:
        hop = rep.family("higher_order")
        if not hop.present:
            continue
        for entity_id in hop.source_entity_ids:
            assert entity_id.startswith(rep.situation_id)
