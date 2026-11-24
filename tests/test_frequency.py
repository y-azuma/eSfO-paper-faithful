from __future__ import annotations

import numpy as np
import pytest

from esfo.data import EventSequence, RelativeEventSequence
from esfo.frequency import (
    FrequencyInitializationError,
    _fft_peak_unchecked,
    _select_positive_peak,
    initialize_frequency_fft_raw,
)
from esfo.provenance import canonical_hash


def _events(t_us, x, *, y=None, p=None) -> EventSequence:
    count = len(t_us)
    return EventSequence.from_arrays(
        t_us=t_us,
        x=x,
        y=np.zeros(count, dtype=np.int64) if y is None else y,
        p=np.zeros(count, dtype=np.int64) if p is None else p,
        width=64,
        height=48,
    )


def test_integer_windows_use_open_left_closed_right_and_floor() -> None:
    events = _events(
        [0, 20_000, 20_001, 40_000, 40_001, 60_000, 100_001, 120_000, 125_000],
        [50, 51, 0, 2, 7, 9, 3, 5, 63],
    )
    result = initialize_frequency_fft_raw(events)
    signal = result.windowed_signal

    assert signal.duration_us == 125_000
    np.testing.assert_array_equal(signal.window_indices, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(signal.window_left_us, [20_000, 40_000, 60_000, 80_000, 100_000])
    np.testing.assert_array_equal(signal.window_right_us, [40_000, 60_000, 80_000, 100_000, 120_000])
    np.testing.assert_array_equal(signal.event_counts, [2, 2, 0, 0, 2])
    np.testing.assert_allclose(
        signal.mean_x, [1.0, 8.0, 13.0 / 3.0, 13.0 / 3.0, 4.0],
    )
    assert signal.empty_fill_x == 13.0 / 3.0
    assert signal.used_event_count == 6
    assert signal.excluded_prefix_count == 2
    assert signal.excluded_suffix_count == 1


def test_empty_fill_is_unweighted_mean_of_window_means() -> None:
    events = _events(
        [0, 20_001, 30_000, 60_001, 190_000],
        [50, 0, 2, 7, 63],
    )
    signal = initialize_frequency_fft_raw(events).windowed_signal

    np.testing.assert_array_equal(signal.event_counts, [2, 0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(signal.mean_x, [1, 4, 7, 4, 4, 4, 4, 4])
    assert signal.empty_fill_x == 4.0  # not the event-weighted value 3.0


def test_exact_bin_dc_exclusion_no_padding_and_raw_power() -> None:
    result = _fft_peak_unchecked(
        np.array([102, 101, 100, 101, 102, 101, 100, 101], dtype=np.float64),
        sample_period_s=0.020,
    )
    frequency, peak_bin, peak_power, resolution, frequencies, power, zero = result

    np.testing.assert_allclose(frequencies, [0, 6.25, 12.5, 18.75, 25])
    np.testing.assert_allclose(power, [652_864, 0, 16, 0, 0], atol=1e-25)
    assert (frequency, peak_bin, peak_power, resolution, zero) == (
        12.5, 2, 16.0, 6.25, False,
    )


def test_exact_tie_uses_lowest_positive_frequency_and_zero_flag() -> None:
    impulse = _fft_peak_unchecked(
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        sample_period_s=0.020,
    )
    assert impulse[0] == 6.25
    assert impulse[1] == 1
    np.testing.assert_array_equal(impulse[5][1:], np.ones(4))
    assert not impulse[6]

    flat = _fft_peak_unchecked(
        np.ones(8, dtype=np.float64), sample_period_s=0.020,
    )
    assert flat[0] == 6.25
    assert flat[1] == 1
    assert flat[6]


def test_near_tie_is_not_collapsed_by_a_tolerance() -> None:
    power = np.array([100.0, 1.0, np.nextafter(1.0, np.inf)])
    assert _select_positive_peak(power) == (2, False)


def test_no_harmonic_correction_or_frequency_prior() -> None:
    signal = np.array([16, 9, 7, 12, 7, 9, 16, 9, 7, 12, 7, 9], dtype=float)
    result = _fft_peak_unchecked(signal, sample_period_s=0.020)
    assert result[1] == 4
    assert result[0] == pytest.approx(16.666666666666668)
    assert result[5][4] > result[5][2]


def test_nyquist_is_a_candidate() -> None:
    result = _fft_peak_unchecked(
        np.array([2, 0, 2, 0, 2, 0, 2, 0], dtype=float),
        sample_period_s=0.020,
    )
    assert result[1] == 4
    assert result[0] == 25.0


def test_all_raw_events_are_used_independent_of_y_and_polarity() -> None:
    t = [0, 20_001, 30_000, 40_001, 50_000, 60_001]
    x = [5, 0, 8, 1, 3, 2]
    left = initialize_frequency_fft_raw(_events(t, x))
    right = initialize_frequency_fft_raw(_events(
        t, x, y=[47, 0, 46, 1, 45, 2], p=[1, 0, 1, 0, 1, 0],
    ))
    np.testing.assert_array_equal(left.windowed_signal.mean_x, [4.0, 2.0])
    np.testing.assert_array_equal(
        left.windowed_signal.mean_x, right.windowed_signal.mean_x,
    )
    assert left.frequency_hz == right.frequency_hz


def test_raw_timestamp_shift_input_detachment_and_result_immutability() -> None:
    t = np.array([0, 20_001, 40_000, 40_001, 60_000, 60_001, 80_000])
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    events = _events(t, x)
    shifted = _events(t + 987_654_321, x)
    result = initialize_frequency_fft_raw(events)
    shifted_result = initialize_frequency_fft_raw(shifted)
    x[:] = 63

    np.testing.assert_array_equal(result.windowed_signal.mean_x, [2.5, 4.5, 6.5])
    np.testing.assert_array_equal(
        result.windowed_signal.mean_x, shifted_result.windowed_signal.mean_x,
    )
    assert canonical_hash(result) == canonical_hash(shifted_result)
    for array in (
        result.frequencies_hz, result.power,
        result.windowed_signal.mean_x, result.windowed_signal.event_counts,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    with pytest.raises(TypeError):
        result.metadata["range_prior"] = True


@pytest.mark.parametrize(
    ("events", "reason"),
    [
        (_events([], []), "empty_event_sequence"),
        (_events([0, 59_999], [1, 2]), "insufficient_fft_windows"),
        (_events([0, 20_000, 60_001], [1, 2, 3]), "all_fft_windows_empty"),
    ],
)
def test_reference_failures_are_reason_coded(events, reason) -> None:
    with pytest.raises(FrequencyInitializationError) as captured:
        initialize_frequency_fft_raw(events)
    assert captured.value.reason == reason


def test_reference_api_requires_raw_event_sequence() -> None:
    relative = RelativeEventSequence(
        t_ms=[0, 20, 40, 60], x=[1, 2, 3, 4], y=[0, 0, 0, 0],
        p=[0, 0, 0, 0], source_index=[0, 1, 2, 3], width=8, height=8,
    )
    with pytest.raises(TypeError, match="raw timestamps"):
        initialize_frequency_fft_raw(relative)  # type: ignore[arg-type]


def test_metadata_records_every_result_independent_choice() -> None:
    result = initialize_frequency_fft_raw(
        _events([0, 20_001, 40_001, 60_001], [1, 2, 3, 4]),
    )
    assert result.metadata == {
        "paper_equations": (9, 10),
        "source_events": "all_raw_events",
        "window_us": 20_000,
        "window_indices": "1..floor(T/delta_tf)-1",
        "window_bounds": "open_left_closed_right",
        "empty_fill": "unweighted_mean_of_nonempty_window_means",
        "fft": "numpy.fft.rfft",
        "fft_norm": "backward",
        "window_function": "rectangular",
        "zero_padding": False,
        "detrend": False,
        "power": "abs(rfft)^2",
        "dc_candidate": False,
        "nyquist_candidate": True,
        "tie_break": "exact_lowest_frequency",
        "zero_positive_power_policy": "lowest_positive_bin_with_diagnostic",
        "interpolation": False,
        "range_prior": False,
        "harmonic_correction": False,
    }
