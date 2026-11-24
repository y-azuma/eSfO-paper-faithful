from __future__ import annotations

import numpy as np

from esfo.config import FrontendConfig
from esfo.data import EventSequence
from esfo.frontend import density_filter


def _relative_events(
    *,
    t_us: list[int],
    x: list[int],
    y: list[int],
    p: list[int],
):
    return EventSequence.from_arrays(
        t_us=t_us,
        x=x,
        y=y,
        p=p,
        width=64,
        height=64,
    ).to_relative_ms()


def test_density_uses_raw_events_inclusive_radius_and_global_mean() -> None:
    # Event 2 is deliberately not a corner.  It nevertheless contributes to
    # the density of corners 0 and 1.  Events 0 and 1 are exactly lambda=7
    # apart, so this also fixes the inclusive boundary.
    events = _relative_events(
        t_us=[0, 0, 0, 0],
        x=[0, 7, 3, 20],
        y=[0, 0, 0, 0],
        p=[1, 1, 1, 1],
    )

    result = density_filter(
        events,
        np.asarray([0, 1, 3], dtype=np.int64),
        FrontendConfig(),
    )

    np.testing.assert_allclose(result.scores, [3 / 7, 3 / 7, 1 / 7])
    assert result.thresholds[1] == np.mean(result.scores)
    np.testing.assert_array_equal(result.keep, [True, True, False])
    np.testing.assert_array_equal(result.kept_indices, [0, 1])


def test_density_is_spatiotemporal_and_same_polarity() -> None:
    # The opposite-polarity event is spatially adjacent but must not count.
    # The last same-polarity event is 8 ms away and is outside the 3-D radius.
    events = _relative_events(
        t_us=[0, 0, 8_000],
        x=[0, 1, 0],
        y=[0, 0, 0],
        p=[1, 0, 1],
    )

    result = density_filter(
        events,
        np.asarray([0, 1], dtype=np.int64),
        FrontendConfig(),
    )

    np.testing.assert_allclose(result.scores, [1 / 7, 1 / 7])
    np.testing.assert_allclose(result.thresholds, [1 / 7, 1 / 7])
    np.testing.assert_array_equal(result.keep, [True, True])


def test_density_includes_time_boundary_at_seven_milliseconds() -> None:
    events = _relative_events(
        t_us=[0, 7_000],
        x=[0, 0],
        y=[0, 0],
        p=[1, 1],
    )

    result = density_filter(
        events,
        np.asarray([0, 1], dtype=np.int64),
        FrontendConfig(),
    )

    np.testing.assert_allclose(result.scores, [2 / 7, 2 / 7])


def test_density_threshold_is_computed_separately_per_polarity() -> None:
    events = _relative_events(
        t_us=[0, 0, 0, 0],
        x=[0, 1, 20, 40],
        y=[0, 0, 0, 0],
        p=[1, 1, 1, 0],
    )
    result = density_filter(events, np.arange(4), FrontendConfig())

    np.testing.assert_allclose(result.scores, [2 / 7, 2 / 7, 1 / 7, 1 / 7])
    np.testing.assert_allclose(result.thresholds[:3], np.full(3, 5 / 21))
    assert result.thresholds[3] == 1 / 7
    np.testing.assert_array_equal(result.keep, [True, True, False, True])


def test_empty_python_corner_list_is_supported_and_results_are_immutable() -> None:
    events = _relative_events(t_us=[], x=[], y=[], p=[])
    result = density_filter(events, [], FrontendConfig())
    assert result.corner_indices.size == 0
    assert not result.corner_indices.flags.writeable
    assert not result.scores.flags.writeable
