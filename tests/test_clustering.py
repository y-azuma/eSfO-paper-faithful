from __future__ import annotations

import numpy as np

from esfo.config import FrontendConfig
from esfo.data import EventSequence
from esfo.frontend import cluster_corners


def _canonical_partition(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            tuple(np.flatnonzero(labels == label).tolist())
            for label in sorted(set(labels.tolist()))
            if label != -1
        )
    )


def _two_clusters_and_noise():
    # Each dense group has 15 samples, comfortably above min_cluster_size=10.
    t_us = list(range(0, 1_500, 100)) + list(range(100_000, 101_500, 100))
    x = [5] * 15 + [40] * 15
    y = [5] * 15 + [40] * 15
    p = [1, 0] * 15  # clustering must ignore polarity
    # A single distant point should remain HDBSCAN noise.
    t_us.append(250_000)
    x.append(20)
    y.append(20)
    p.append(1)
    return EventSequence.from_arrays(
        t_us=t_us,
        x=x,
        y=y,
        p=p,
        width=64,
        height=64,
    ).to_relative_ms()


def test_hdbscan_reference_parameters_and_noise_exclusion() -> None:
    events = _two_clusters_and_noise()
    corners = np.arange(len(events.t_ms), dtype=np.int64)

    result = cluster_corners(events, corners, FrontendConfig())

    assert _canonical_partition(result.labels) == (
        tuple(range(15)),
        tuple(range(15, 30)),
    )
    assert result.labels[-1] == -1
    assert len(result.clusters) == 2
    assert all(cluster.cluster_id >= 0 for cluster in result.clusters)
    assert result.metadata["implementation"] == "hdbscan.HDBSCAN"
    assert result.metadata["version"] == "0.8.33"
    parameters = result.metadata["parameters"]
    assert parameters["min_cluster_size"] == 10
    assert parameters["min_samples"] is None
    assert result.metadata["effective_min_samples"] == 10
    assert parameters["cluster_selection_epsilon"] == 5.0
    assert parameters["metric"] == "euclidean"
    assert parameters["cluster_selection_method"] == "eom"
    assert parameters["alpha"] == 1.0
    assert parameters["approx_min_span_tree"] is True
    assert parameters["allow_single_cluster"] is False
    assert parameters["algorithm"] == "best"
    assert parameters["core_dist_n_jobs"] == 4
    with np.testing.assert_raises(TypeError):
        result.metadata["cluster_count"] = 0
    with np.testing.assert_raises(TypeError):
        result.metadata["parameters"]["alpha"] = 2.0


def test_hdbscan_partition_is_deterministic_for_identical_input() -> None:
    events = _two_clusters_and_noise()
    corners = np.arange(len(events.t_ms), dtype=np.int64)

    first = cluster_corners(events, corners, FrontendConfig())
    second = cluster_corners(events, corners, FrontendConfig())

    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.metadata == second.metadata


def test_nonempty_small_input_propagates_hdbscan_failure() -> None:
    events = EventSequence.from_arrays(
        t_us=[0], x=[1], y=[1], p=[1], width=4, height=4,
    ).to_relative_ms()
    with np.testing.assert_raises(ValueError):
        cluster_corners(events, np.asarray([0]), FrontendConfig())
