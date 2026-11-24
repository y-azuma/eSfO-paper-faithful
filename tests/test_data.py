from __future__ import annotations

import numpy as np
import pytest

from esfo.data import EventSequence


def test_event_sequence_converts_microseconds_to_relative_milliseconds() -> None:
    events = EventSequence.from_arrays(
        t_us=[1_000_000, 1_001_000, 1_007_500],
        x=[2, 3, 4],
        y=[5, 6, 7],
        p=[1, 0, 1],
        width=16,
        height=12,
    )

    relative = events.to_relative_ms()

    np.testing.assert_allclose(relative.t_ms, [0.0, 1.0, 7.5])
    np.testing.assert_array_equal(relative.source_index, [0, 1, 2])
    np.testing.assert_array_equal(relative.p, [1, 0, 1])


def test_equal_timestamp_input_order_is_preserved() -> None:
    events = EventSequence.from_arrays(
        t_us=[10, 10, 10],
        x=[3, 1, 2],
        y=[0, 0, 0],
        p=[1, 0, 1],
        width=4,
        height=1,
    )

    relative = events.to_relative_ms()

    np.testing.assert_array_equal(relative.x, [3, 1, 2])
    np.testing.assert_array_equal(relative.source_index, [0, 1, 2])


@pytest.mark.parametrize("polarity", [-1, 2, -2, 0.5])
def test_invalid_polarity_is_rejected(polarity: float) -> None:
    with pytest.raises(ValueError, match="polarity|p"):
        EventSequence.from_arrays(
            t_us=[0],
            x=[0],
            y=[0],
            p=[polarity],
            width=2,
            height=2,
        )


def test_unsorted_timestamps_fail_closed_by_default() -> None:
    with pytest.raises(ValueError, match="sort|time|monotonic"):
        EventSequence.from_arrays(
            t_us=[2, 1],
            x=[0, 0],
            y=[0, 0],
            p=[1, 1],
            width=2,
            height=2,
        )


def test_spatial_bounds_and_lengths_are_validated() -> None:
    with pytest.raises(ValueError, match="length|shape"):
        EventSequence.from_arrays(
            t_us=[0, 1],
            x=[0],
            y=[0, 0],
            p=[1, 1],
            width=2,
            height=2,
        )

    with pytest.raises(ValueError, match="bounds|x|coordinate"):
        EventSequence.from_arrays(
            t_us=[0],
            x=[2],
            y=[0],
            p=[1],
            width=2,
            height=2,
        )


def test_direct_constructor_cannot_bypass_event_invariants() -> None:
    with pytest.raises(ValueError, match="polarity"):
        EventSequence(
            np.asarray([0]), np.asarray([0]), np.asarray([0]),
            np.asarray([-1]), 2, 2,
        )
    with pytest.raises(ValueError, match="coordinate|sensor"):
        EventSequence(
            np.asarray([0]), np.asarray([2]), np.asarray([0]),
            np.asarray([1]), 2, 2,
        )
    with pytest.raises(ValueError, match="represented|coordinate|integer"):
        EventSequence(
            np.asarray([0]), np.asarray([0.9]), np.asarray([0]),
            np.asarray([1]), 2, 2,
        )


def test_input_aliases_cannot_mutate_frozen_event_sequence() -> None:
    x = np.asarray([0, 1], dtype=np.int64)
    events = EventSequence.from_arrays(
        t_us=[0, 1], x=x, y=[0, 0], p=[0, 1], width=2, height=1,
    )
    x[0] = 1
    assert events.x.tolist() == [0, 1]
    assert not events.x.flags.writeable
