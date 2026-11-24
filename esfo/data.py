"""Strict in-memory event schema used by the reference front-end."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def _readonly(array: np.ndarray) -> np.ndarray:
    # Always detach from caller-owned storage.  Merely clearing WRITEABLE on a
    # view would let a retained alias mutate an otherwise frozen dataclass.
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _one_dimensional(name: str, values: Iterable, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if array.size == 0:
        return _readonly(np.empty(0, dtype=dtype))
    try:
        converted = array.astype(dtype, casting="safe", copy=False)
    except TypeError as error:
        raise ValueError(f"{name} cannot be represented as {np.dtype(dtype)}") from error
    return _readonly(converted)


@dataclass(frozen=True)
class RelativeEventSequence:
    """Events with the first raw timestamp shifted to zero milliseconds."""

    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    source_index: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        t_input = np.asarray(self.t_ms)
        if t_input.ndim != 1 or t_input.dtype.kind not in "iuf":
            raise ValueError("t_ms must be a one-dimensional numeric array")
        t_ms = _readonly(t_input.astype(np.float64, copy=False))
        x = _one_dimensional("x", self.x, np.dtype(np.int64))
        y = _one_dimensional("y", self.y, np.dtype(np.int64))
        source_index = _one_dimensional(
            "source_index", self.source_index, np.dtype(np.int64),
        )
        p_wide = _one_dimensional("p", self.p, np.dtype(np.int64))
        if p_wide.size and not np.all((p_wide == 0) | (p_wide == 1)):
            raise ValueError("event polarity must be encoded as 0 or 1")
        p = _readonly(p_wide.astype(np.uint8))
        arrays = (t_ms, x, y, p, source_index)
        if len({array.size for array in arrays}) != 1:
            raise ValueError("relative event arrays have inconsistent lengths")
        if not np.all(np.isfinite(t_ms)):
            raise ValueError("relative timestamps contain non-finite values")
        if t_ms.size and (
            t_ms[0] != 0.0 or np.any(np.diff(t_ms) < 0.0)
        ):
            raise ValueError("relative timestamps must start at zero and be sorted")
        if source_index.size and np.any(np.diff(source_index) <= 0):
            raise ValueError("source indices must be strictly increasing")
        if (
            isinstance(self.width, bool) or not isinstance(self.width, (int, np.integer))
            or isinstance(self.height, bool)
            or not isinstance(self.height, (int, np.integer))
            or int(self.width) <= 0 or int(self.height) <= 0
        ):
            raise ValueError("sensor dimensions must be positive")
        width, height = int(self.width), int(self.height)
        if x.size and (
            np.any(x < 0) or np.any(x >= width)
            or np.any(y < 0) or np.any(y >= height)
        ):
            raise ValueError("event coordinate lies outside the sensor")
        object.__setattr__(self, "t_ms", t_ms)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "source_index", source_index)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def __len__(self) -> int:
        return int(self.t_ms.size)

    @property
    def duration_ms(self) -> float:
        return float(self.t_ms[-1]) if len(self) else 0.0

    def points_txy(self) -> np.ndarray:
        """Return the paper's mixed coordinate as ``[t_ms,x,y]`` rows."""

        if not len(self):
            return np.empty((0, 3), dtype=np.float64)
        return np.column_stack((self.t_ms, self.x, self.y)).astype(
            np.float64, copy=False,
        )

@dataclass(frozen=True)
class EventSequence:
    """Raw timestamped events in the TOPSPIN microsecond convention."""

    t_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    width: int
    height: int

    @classmethod
    def from_arrays(
        cls,
        t_us: Iterable,
        x: Iterable,
        y: Iterable,
        p: Iterable,
        *,
        width: int,
        height: int,
        require_sorted: bool = True,
    ) -> "EventSequence":
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, (int, np.integer)):
            raise ValueError("height must be a positive integer")
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError("sensor dimensions must be positive")
        t_array = _one_dimensional("t_us", t_us, np.dtype(np.int64))
        x_array = _one_dimensional("x", x, np.dtype(np.int64))
        y_array = _one_dimensional("y", y, np.dtype(np.int64))
        # Validate in a wide signed type before narrowing. ``casting='safe'``
        # correctly rejects float polarities but would also reject the common
        # int64 -> uint8 conversion before values can be checked.
        p_wide = _one_dimensional("p", p, np.dtype(np.int64))
        if p_wide.size and not np.all((p_wide == 0) | (p_wide == 1)):
            raise ValueError("event polarity must be encoded as 0 or 1")
        p_array = _readonly(p_wide.astype(np.uint8))
        if len({array.size for array in (t_array, x_array, y_array, p_array)}) != 1:
            raise ValueError("event arrays have inconsistent lengths")
        if require_sorted and t_array.size and np.any(np.diff(t_array) < 0):
            raise ValueError("events must be in nondecreasing timestamp order")
        if not require_sorted and t_array.size and np.any(np.diff(t_array) < 0):
            # Sorting silently changes equal-timestamp SAE order and is not
            # permitted by the reference path.
            raise ValueError("unsorted events are not reordered implicitly")
        if x_array.size and (
            np.any(x_array < 0) or np.any(x_array >= width)
            or np.any(y_array < 0) or np.any(y_array >= height)
        ):
            raise ValueError("event coordinate lies outside the sensor")
        return cls(t_array, x_array, y_array, p_array, width, height)

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool) or not isinstance(self.width, (int, np.integer))
            or isinstance(self.height, bool)
            or not isinstance(self.height, (int, np.integer))
            or int(self.width) <= 0 or int(self.height) <= 0
        ):
            raise ValueError("sensor dimensions must be positive")
        width, height = int(self.width), int(self.height)
        t_us = _one_dimensional("t_us", self.t_us, np.dtype(np.int64))
        x = _one_dimensional("x", self.x, np.dtype(np.int64))
        y = _one_dimensional("y", self.y, np.dtype(np.int64))
        p_wide = _one_dimensional("p", self.p, np.dtype(np.int64))
        if p_wide.size and not np.all((p_wide == 0) | (p_wide == 1)):
            raise ValueError("event polarity must be encoded as 0 or 1")
        p = _readonly(p_wide.astype(np.uint8))
        if len({array.size for array in (t_us, x, y, p)}) != 1:
            raise ValueError("event arrays have inconsistent lengths")
        if t_us.size and np.any(np.diff(t_us) < 0):
            raise ValueError("events must be sorted")
        if x.size and (
            np.any(x < 0) or np.any(x >= width)
            or np.any(y < 0) or np.any(y >= height)
        ):
            raise ValueError("event coordinate lies outside the sensor")
        object.__setattr__(self, "t_us", t_us)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)

    def __len__(self) -> int:
        return int(self.t_us.size)

    def to_relative_ms(self, scale_us_to_ms: float = 1.0 / 1000.0) -> RelativeEventSequence:
        """Apply the frozen ``(t_us-t_first)/1000`` front-end mapping."""

        if isinstance(scale_us_to_ms, bool) or not math.isfinite(scale_us_to_ms):
            raise ValueError("timestamp scale must be finite")
        if scale_us_to_ms <= 0.0:
            raise ValueError("timestamp scale must be positive")
        if len(self):
            # Subtract as int64 first. TOPSPIN durations are far below int64
            # overflow and this keeps the first event exactly zero.
            t_ms = (self.t_us - self.t_us[0]).astype(np.float64)
            if float(scale_us_to_ms) == 1.0 / 1000.0:
                # Keep the reference path literally identical to the frozen
                # equation.  Multiplication by 0.001 differs by one ulp for
                # some integer-microsecond timestamps.
                t_ms /= 1000.0
            else:
                # Non-reference scales are available only to the explicitly
                # labelled sensitivity path.
                t_ms *= float(scale_us_to_ms)
        else:
            t_ms = np.empty(0, dtype=np.float64)
        return RelativeEventSequence(
            _readonly(t_ms), _readonly(self.x.astype(np.int64, copy=False)),
            _readonly(self.y.astype(np.int64, copy=False)), _readonly(self.p),
            _readonly(np.arange(len(self), dtype=np.int64)),
            self.width, self.height,
        )


__all__ = ["EventSequence", "RelativeEventSequence"]
