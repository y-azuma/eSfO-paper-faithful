"""Paper Sec. 5.1 FFT frequency initialization under frozen conventions.

The paper specifies a 20 ms sequence of mean image ``x`` coordinates followed
by a Fourier transform.  Gate 0 freezes the otherwise unspecified discrete
details: raw events, integer-microsecond window boundaries, an unwindowed
``rfft`` without padding, DC exclusion, and no physical-frequency prior or
harmonic correction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from .data import EventSequence


REFERENCE_DELTA_TF_US = 20_000


class FrequencyInitializationError(ValueError):
    """A fail-closed, reason-coded reference initialization failure."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(f"{reason}: {message}")


def _readonly(value: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        result = np.array(array, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has an invalid dtype") from error
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _deep_freeze(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class WindowedMeanX:
    """Frozen mean-``x`` signal used by paper Eq. (10)."""

    duration_us: int
    delta_tf_us: int
    window_indices: np.ndarray
    window_left_us: np.ndarray
    window_right_us: np.ndarray
    event_counts: np.ndarray
    empty_mask: np.ndarray
    mean_x: np.ndarray
    empty_fill_x: float
    source_event_count: int
    used_event_count: int
    excluded_prefix_count: int
    excluded_suffix_count: int

    def __post_init__(self) -> None:
        for name in (
            "duration_us", "delta_tf_us", "source_event_count",
            "used_event_count", "excluded_prefix_count",
            "excluded_suffix_count",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.delta_tf_us <= 0:
            raise ValueError("delta_tf_us must be positive")
        indices = _readonly(
            self.window_indices, np.dtype(np.int64), "window_indices",
        )
        left = _readonly(self.window_left_us, np.dtype(np.int64), "window_left_us")
        right = _readonly(
            self.window_right_us, np.dtype(np.int64), "window_right_us",
        )
        counts = _readonly(self.event_counts, np.dtype(np.int64), "event_counts")
        empty = _readonly(self.empty_mask, np.dtype(np.bool_), "empty_mask")
        mean_x = _readonly(self.mean_x, np.dtype(np.float64), "mean_x")
        lengths = {item.size for item in (indices, left, right, counts, empty, mean_x)}
        if len(lengths) != 1:
            raise ValueError("windowed signal arrays have inconsistent lengths")
        if indices.size and not np.array_equal(
            indices, np.arange(1, indices.size + 1, dtype=np.int64),
        ):
            raise ValueError("window indices must be consecutive and start at one")
        if np.any(counts < 0) or not np.array_equal(empty, counts == 0):
            raise ValueError("empty mask and event counts are inconsistent")
        if not np.array_equal(left, indices * self.delta_tf_us):
            raise ValueError("window left boundaries are inconsistent")
        if not np.array_equal(right, (indices + 1) * self.delta_tf_us):
            raise ValueError("window right boundaries are inconsistent")
        if not np.all(np.isfinite(mean_x)):
            raise ValueError("mean_x contains non-finite values")
        if not math.isfinite(float(self.empty_fill_x)):
            raise ValueError("empty_fill_x must be finite")
        if int(np.sum(counts, dtype=np.int64)) != self.used_event_count:
            raise ValueError("used event count is inconsistent")
        if (self.used_event_count + self.excluded_prefix_count
                + self.excluded_suffix_count != self.source_event_count):
            raise ValueError("source event accounting is inconsistent")
        object.__setattr__(self, "window_indices", indices)
        object.__setattr__(self, "window_left_us", left)
        object.__setattr__(self, "window_right_us", right)
        object.__setattr__(self, "event_counts", counts)
        object.__setattr__(self, "empty_mask", empty)
        object.__setattr__(self, "mean_x", mean_x)
        object.__setattr__(self, "empty_fill_x", float(self.empty_fill_x))

    @property
    def duration_ms(self) -> float:
        return self.duration_us / 1000.0

    @property
    def delta_tf_ms(self) -> float:
        return self.delta_tf_us / 1000.0


@dataclass(frozen=True)
class FFTFrequencyInitialization:
    """Reference FFT peak and the full auditable discrete spectrum."""

    frequency_hz: float
    peak_bin: int
    peak_power: float
    frequency_resolution_hz: float
    frequencies_hz: np.ndarray
    power: np.ndarray
    zero_positive_power: bool
    windowed_signal: WindowedMeanX
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("frequency_hz", "peak_power", "frequency_resolution_hz"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.peak_power < 0.0 or self.frequency_resolution_hz <= 0.0:
            raise ValueError("power and frequency resolution are invalid")
        if (isinstance(self.peak_bin, bool)
                or not isinstance(self.peak_bin, (int, np.integer))):
            raise ValueError("peak_bin must be an integer")
        peak_bin = int(self.peak_bin)
        frequencies = _readonly(
            self.frequencies_hz, np.dtype(np.float64), "frequencies_hz",
        )
        power = _readonly(self.power, np.dtype(np.float64), "power")
        if frequencies.size != power.size or frequencies.size < 2:
            raise ValueError("spectrum arrays have inconsistent lengths")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(power)):
            raise ValueError("spectrum contains non-finite values")
        if np.any(frequencies < 0.0) or np.any(power < 0.0):
            raise ValueError("spectrum values must be non-negative")
        if peak_bin <= 0 or peak_bin >= frequencies.size:
            raise ValueError("peak_bin must select a positive rFFT bin")
        if frequencies[peak_bin] != self.frequency_hz:
            raise ValueError("frequency_hz does not match peak_bin")
        if power[peak_bin] != self.peak_power:
            raise ValueError("peak_power does not match peak_bin")
        if not isinstance(self.zero_positive_power, (bool, np.bool_)):
            raise ValueError("zero_positive_power must be boolean")
        if bool(self.zero_positive_power) != bool(np.all(power[1:] == 0.0)):
            raise ValueError("zero_positive_power is inconsistent")
        object.__setattr__(self, "peak_bin", peak_bin)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "zero_positive_power", bool(self.zero_positive_power))
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def _window_mean_x_unchecked(
    events: EventSequence,
    *,
    delta_tf_us: int,
) -> WindowedMeanX:
    """Construct Eq. (10)'s signal using exact integer-us boundaries."""

    if not len(events):
        raise FrequencyInitializationError(
            "empty_event_sequence", "FFT initialization requires raw events",
        )
    relative_us = events.t_us - events.t_us[0]
    duration_us = int(relative_us[-1])
    window_count = duration_us // int(delta_tf_us) - 1
    if window_count < 2:
        raise FrequencyInitializationError(
            "insufficient_fft_windows",
            "at least two displayed 20 ms windows are required",
        )
    indices = np.arange(1, window_count + 1, dtype=np.int64)
    left = indices * int(delta_tf_us)
    right = (indices + 1) * int(delta_tf_us)

    # Windows are contiguous after the excluded first 20 ms interval.  The
    # right-inclusive bin of an event is ceil(t/dt)-1, equivalently the
    # insertion position immediately to the left of an exact right boundary.
    starts = np.searchsorted(relative_us, left, side="right")
    stops = np.searchsorted(relative_us, right, side="right")
    counts = (stops - starts).astype(np.int64, copy=False)
    means = np.empty(window_count, dtype=np.float64)
    for row, (start, stop) in enumerate(zip(starts, stops)):
        means[row] = (
            float(np.mean(events.x[start:stop], dtype=np.float64))
            if stop > start else np.nan
        )
    empty = counts == 0
    if np.all(empty):
        raise FrequencyInitializationError(
            "all_fft_windows_empty",
            "the displayed FFT windows contain no events",
        )
    empty_fill = float(np.mean(means[~empty], dtype=np.float64))
    means[empty] = empty_fill
    if not np.all(np.isfinite(means)):
        raise FrequencyInitializationError(
            "nonfinite_window_signal", "window means are not finite",
        )
    prefix_count = int(np.searchsorted(
        relative_us, int(delta_tf_us), side="right",
    ))
    last_right = int((window_count + 1) * int(delta_tf_us))
    suffix_count = len(events) - int(np.searchsorted(
        relative_us, last_right, side="right",
    ))
    return WindowedMeanX(
        duration_us=duration_us,
        delta_tf_us=int(delta_tf_us),
        window_indices=indices,
        window_left_us=left,
        window_right_us=right,
        event_counts=counts,
        empty_mask=empty,
        mean_x=means,
        empty_fill_x=empty_fill,
        source_event_count=len(events),
        used_event_count=int(np.sum(counts, dtype=np.int64)),
        excluded_prefix_count=prefix_count,
        excluded_suffix_count=suffix_count,
    )


def _fft_peak_unchecked(
    signal: np.ndarray,
    *,
    sample_period_s: float,
) -> tuple[float, int, float, float, np.ndarray, np.ndarray, bool]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise FrequencyInitializationError(
            "insufficient_fft_windows", "FFT signal must contain at least two values",
        )
    if not np.all(np.isfinite(values)):
        raise FrequencyInitializationError(
            "nonfinite_window_signal", "FFT signal contains non-finite values",
        )
    if not math.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.fft.rfft(values, norm="backward")
        power = transformed.real ** 2 + transformed.imag ** 2
    frequencies = np.fft.rfftfreq(values.size, d=sample_period_s)
    if frequencies.size < 2:
        raise FrequencyInitializationError(
            "insufficient_fft_windows", "FFT has no positive-frequency bin",
        )
    if not np.all(np.isfinite(power)):
        raise FrequencyInitializationError(
            "nonfinite_fft_power", "FFT power overflowed or is non-finite",
        )
    peak_bin, zero_positive = _select_positive_peak(power)
    if zero_positive:
        # There is no observable positive-frequency maximum when every
        # positive bin has exactly zero power.  Returning the first bin would
        # turn an information-free tie into an arbitrary physical speed.
        # This is an exact fail-closed test, not a confidence threshold.
        raise FrequencyInitializationError(
            "fft_frequency_unobservable",
            "all positive-frequency FFT bins have exactly zero power",
        )
    frequency = float(frequencies[peak_bin])
    peak_power = float(power[peak_bin])
    resolution = float(1.0 / (values.size * sample_period_s))
    return (
        frequency, peak_bin, peak_power, resolution,
        frequencies, power, zero_positive,
    )


def _select_positive_peak(power: np.ndarray) -> tuple[int, bool]:
    """Select the exact positive-bin maximum without tolerance grouping."""

    values = np.asarray(power, dtype=np.float64)
    if (values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values))
            or np.any(values < 0.0)):
        raise ValueError("power must be a finite non-negative 1-D spectrum")
    # np.argmax returns the first exact maximum; bins are frequency-ordered,
    # hence an exact tie resolves to the lowest positive frequency. A one-ulp
    # larger later bin remains larger and must win.
    return int(np.argmax(values[1:])) + 1, bool(np.all(values[1:] == 0.0))


def initialize_frequency_fft_raw(
    events: EventSequence,
) -> FFTFrequencyInitialization:
    """Initialize rotation frequency from all raw events (paper Eqs. 9--10).

    This reference entry point intentionally has no tunable parameters.  It
    does not accept corner masks, physical speed ranges, or a confidence
    threshold, because none is part of the frozen paper interpretation.
    """

    if not isinstance(events, EventSequence):
        raise TypeError("events must be an EventSequence with raw timestamps")
    windowed = _window_mean_x_unchecked(
        events, delta_tf_us=REFERENCE_DELTA_TF_US,
    )
    (
        frequency, peak_bin, peak_power, resolution,
        frequencies, power, zero_positive,
    ) = _fft_peak_unchecked(
        windowed.mean_x,
        sample_period_s=REFERENCE_DELTA_TF_US / 1_000_000.0,
    )
    metadata = {
        "paper_equations": (9, 10),
        "source_events": "all_raw_events",
        "window_us": REFERENCE_DELTA_TF_US,
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
        "zero_positive_power_policy": "fail_fft_frequency_unobservable",
        "interpolation": False,
        "range_prior": False,
        "harmonic_correction": False,
    }
    return FFTFrequencyInitialization(
        frequency_hz=frequency,
        peak_bin=peak_bin,
        peak_power=peak_power,
        frequency_resolution_hz=resolution,
        frequencies_hz=frequencies,
        power=power,
        zero_positive_power=zero_positive,
        windowed_signal=windowed,
        metadata=metadata,
    )


__all__ = [
    "FFTFrequencyInitialization", "FrequencyInitializationError",
    "REFERENCE_DELTA_TF_US", "WindowedMeanX", "initialize_frequency_fft_raw",
]
