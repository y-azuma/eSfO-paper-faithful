from __future__ import annotations

import numpy as np

from esfo.data import EventSequence
from esfo.tracks import (
    EventCluster,
    _merge_clusters_unchecked,
    _track_observations_unchecked,
    build_clusters,
    merge_clusters,
    track_observations,
)


def _cluster(cluster_id: int, points_txy: list[list[float]]) -> EventCluster:
    points = np.asarray(points_txy, dtype=np.float64)
    indices = np.arange(points.shape[0], dtype=np.int64) + cluster_id * 100
    return EventCluster(cluster_id, indices, indices.copy(), points)


def _build(
    t_ms: list[float],
    x: list[int],
    y: list[int],
    labels: list[int],
):
    t_us = np.rint(np.asarray(t_ms) * 1000.0).astype(np.int64)
    events = EventSequence.from_arrays(
        t_us=t_us,
        x=x,
        y=y,
        p=np.ones(len(t_ms), dtype=np.int8),
        width=128,
        height=128,
    ).to_relative_ms()
    corners = np.arange(len(t_ms), dtype=np.int64)
    return events, build_clusters(events, corners, np.asarray(labels))


def _edge_pairs(result) -> set[tuple[int, int]]:
    return {
        (edge.source_cluster_id, edge.target_cluster_id)
        for edge in result.edges
    }


def _components(result) -> set[frozenset[int]]:
    return {frozenset(component) for component in result.components}


def test_forward_direction_and_strict_radius_gate() -> None:
    clusters = (
        _cluster(0, [[0.0, 5.0, 5.0], [1.0, 5.0, 5.0]]),
        _cluster(1, [[2.0, 5.0, 5.0], [3.0, 5.0, 5.0]]),
    )
    result = _merge_clusters_unchecked(clusters, n_sigma=1, phi=30.0)
    assert (0, 1) in _edge_pairs(result)
    assert (1, 0) not in _edge_pairs(result)

    at_gate = (
        _cluster(0, [[0.0, 0.0, 0.0]]),
        _cluster(1, [[0.0, 30.0, 0.0]]),
    )
    strict = _merge_clusters_unchecked(at_gate, n_sigma=1, phi=30.0)
    assert strict.edges == ()
    assert _components(strict) == {frozenset({0}), frozenset({1})}


def test_equal_distance_tie_prefers_earlier_head_then_stable_id() -> None:
    clusters = (
        _cluster(0, [[0.0, 10.0, 10.0]]),
        _cluster(1, [[1.0, 9.0, 10.0]]),
        _cluster(2, [[1.0, 11.0, 10.0]]),
    )
    result = _merge_clusters_unchecked(clusters, n_sigma=1, phi=5.0)
    assert (0, 1) in _edge_pairs(result)  # equal time/distance -> lower ID

    earlier = (
        _cluster(0, [[0.0, 0.0, 0.0]]),
        _cluster(1, [[4.0, 3.0, 0.0]]),
        _cluster(2, [[3.0, 4.0, 0.0]]),
    )
    earlier_result = _merge_clusters_unchecked(earlier, n_sigma=1, phi=6.0)
    assert (0, 2) in _edge_pairs(earlier_result)  # earlier head beats lower ID


def test_five_event_head_and_tail_descriptors_are_one_shot_means() -> None:
    cluster = _cluster(
        0,
        [[float(t), float(t * 2), 0.0] for t in range(7)],
    )
    head, tail = cluster.descriptor(5)
    np.testing.assert_allclose(head, [2.0, 4.0, 0.0])
    np.testing.assert_allclose(tail, [4.0, 8.0, 0.0])


def test_stable_cluster_ids_do_not_depend_on_hdbscan_label_numbers() -> None:
    events = EventSequence.from_arrays(
        t_us=[0, 1, 2, 3], x=[1, 2, 10, 11], y=[1, 1, 1, 1],
        p=[1, 1, 1, 1], width=16, height=4,
    ).to_relative_ms()
    first = build_clusters(events, np.arange(4), np.asarray([7, 7, 3, 3]))
    relabelled = build_clusters(events, np.arange(4), np.asarray([2, 2, 9, 9]))
    assert [c.source_indices.tolist() for c in first] == [
        c.source_indices.tolist() for c in relabelled
    ]


def test_cluster_inputs_are_strict_and_detached_from_caller_aliases() -> None:
    indices = np.asarray([0, 1], dtype=np.int64)
    points = np.asarray([[0.0, 1.0, 1.0], [1.0, 2.0, 2.0]])
    cluster = EventCluster(0, indices, indices.copy(), points)
    indices[0] = 1
    points[0, 1] = 99.0
    assert cluster.event_indices.tolist() == [0, 1]
    assert cluster.points_txy[0, 1] == 1.0
    assert not cluster.event_indices.flags.writeable
    assert not cluster.points_txy.flags.writeable

    events = EventSequence.from_arrays(
        t_us=[0], x=[1], y=[1], p=[1], width=4, height=4,
    ).to_relative_ms()
    with np.testing.assert_raises_regex(ValueError, "integer"):
        build_clusters(events, np.asarray([0.5]), np.asarray([0]))
    with np.testing.assert_raises_regex(ValueError, "integer"):
        build_clusters(events, np.asarray([0]), np.asarray([0.5]))


def test_fan_in_cycle_and_input_order_do_not_change_components() -> None:
    # Clusters 0 and 1 have their heads earlier than their tails, so both
    # tails select cluster 2 rather than each other.
    fan_in_clusters = (
        _cluster(0, [[0.0, 9.0, 10.0], [1.0, 9.0, 10.0]]),
        _cluster(1, [[0.0, 11.0, 10.0], [1.0, 11.0, 10.0]]),
        _cluster(2, [[2.0, 10.0, 10.0]]),
    )
    fan_in = _merge_clusters_unchecked(fan_in_clusters, n_sigma=1, phi=5.0)
    assert {(0, 2), (1, 2)} <= _edge_pairs(fan_in)
    assert frozenset({0, 1, 2}) in _components(fan_in)

    cycle_clusters = (
        _cluster(0, [[0.0, 10.0, 10.0]]),
        _cluster(1, [[0.0, 11.0, 10.0]]),
    )
    cycle = _merge_clusters_unchecked(cycle_clusters, n_sigma=1, phi=5.0)
    reversed_cycle = _merge_clusters_unchecked(
        list(reversed(cycle_clusters)), n_sigma=1, phi=5.0
    )
    assert _edge_pairs(cycle) == {(0, 1), (1, 0)}
    assert _edge_pairs(reversed_cycle) == _edge_pairs(cycle)
    assert _components(reversed_cycle) == _components(cycle) == {
        frozenset({0, 1})
    }


def test_track_windows_follow_literal_eq4_boundaries_and_skip_empty() -> None:
    _, clusters = _build(
        t_ms=[0.0, 30.0, 60.0, 60.001, 90.0, 120.0],
        x=[1, 2, 3, 4, 6, 8],
        y=[10, 20, 30, 40, 60, 80],
        labels=[0, 0, 0, 0, 0, 0],
    )
    merged = _merge_clusters_unchecked(clusters, n_sigma=1, phi=300.0)

    observations = _track_observations_unchecked(
        merged.tracks,
        sequence_duration_ms=120.0,
        delta_t_ms=30.0,
    )

    assert [observation.window_index for observation in observations] == [1, 2, 3]
    assert [observation.time_ms for observation in observations] == [30.0, 60.0, 90.0]
    assert [observation.event_count for observation in observations] == [1, 2, 1]
    np.testing.assert_allclose([o.x for o in observations], [3.0, 5.0, 8.0])
    np.testing.assert_allclose([o.y for o in observations], [30.0, 50.0, 80.0])


def test_track_window_count_uses_floor_and_empty_window_has_no_observation() -> None:
    _, clusters = _build(
        t_ms=[0.0, 60.0, 119.9],
        x=[1, 3, 8],
        y=[1, 3, 8],
        labels=[0, 0, 0],
    )
    merged = _merge_clusters_unchecked(clusters, n_sigma=1, phi=300.0)
    observations = _track_observations_unchecked(
        merged.tracks,
        sequence_duration_ms=125.0,
        delta_t_ms=30.0,
    )

    # floor(125/30)-1 == 3: k is exactly 1,2,3.  k=2 is empty.
    assert [(o.window_index, o.time_ms) for o in observations] == [
        (1, 30.0),
        (3, 90.0),
    ]


def test_reference_track_components_reject_nonreference_constants() -> None:
    cluster = _cluster(0, [[0.0, 0.0, 0.0]])
    with np.testing.assert_raises_regex(ValueError, "experimental"):
        merge_clusters([cluster], n_sigma=1, phi=30.0)
    reference = merge_clusters([cluster])
    with np.testing.assert_raises_regex(ValueError, "experimental"):
        track_observations(
            reference.tracks, sequence_duration_ms=100.0, delta_t_ms=20.0,
        )
