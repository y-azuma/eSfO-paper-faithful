"""Independent eFAST-criterion implementation for Gate A.

This module follows the published/reference arc test but does not copy the
GPL implementation cited by eSfO.  Bitwise identity with an unknown author
build is therefore not claimed.
"""

from __future__ import annotations

import numpy as np

from .data import RelativeEventSequence


# Ordered clockwise as in the public rpg_corner_events criterion.  Coordinates
# are (dx, dy); an arc may wrap from the final element to the first.
CIRCLE3 = np.asarray((
    (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2),
    (-1, 3),
), dtype=np.int64)
CIRCLE4 = np.asarray((
    (0, 4), (1, 4), (2, 3), (3, 2), (4, 1), (4, 0), (4, -1), (3, -2),
    (2, -3), (1, -4), (0, -4), (-1, -4), (-2, -3), (-3, -2), (-4, -1),
    (-4, 0), (-4, 1), (-3, 2), (-2, 3), (-1, 4),
), dtype=np.int64)
CIRCLE3.setflags(write=False)
CIRCLE4.setflags(write=False)


def has_newest_arc(ring_timestamps: np.ndarray, min_arc: int, max_arc: int) -> bool:
    """Whether a circular SAE ring contains a valid newest contiguous arc.

    For one length in ``[min_arc,max_arc]``, the oldest timestamp inside the
    arc must be strictly newer than the newest timestamp outside it.  Strict
    separation makes equal timestamps on opposite sides fail deterministically.
    """

    values = np.asarray(ring_timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("ring_timestamps must be a one-dimensional ring")
    if isinstance(min_arc, bool) or isinstance(max_arc, bool):
        raise ValueError("arc lengths must be integers")
    if not isinstance(min_arc, (int, np.integer)) or not isinstance(
        max_arc, (int, np.integer)
    ):
        raise ValueError("arc lengths must be integers")
    min_arc, max_arc = int(min_arc), int(max_arc)
    if min_arc <= 0 or max_arc < min_arc or max_arc >= values.size:
        raise ValueError("invalid arc-length range")
    size = int(values.size)
    offsets = np.arange(size)
    for start in range(size):
        relative = (offsets - start) % size
        for length in range(min_arc, max_arc + 1):
            inside = values[relative < length]
            outside = values[relative >= length]
            if float(np.min(inside)) > float(np.max(outside)):
                return True
    return False


class _EFastDetectorUnchecked:
    """Stateful, polarity-separated Surface-of-Active-Events detector."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        inner_min_arc: int = 3,
        inner_max_arc: int = 6,
        outer_min_arc: int = 4,
        outer_max_arc: int = 8,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, (int, np.integer)):
            raise ValueError("height must be a positive integer")
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sensor dimensions must be positive")
        # Validate ranges even before a ring exists.
        has_newest_arc(np.full(16, -np.inf), inner_min_arc, inner_max_arc)
        has_newest_arc(np.full(20, -np.inf), outer_min_arc, outer_max_arc)
        self.inner_min_arc = int(inner_min_arc)
        self.inner_max_arc = int(inner_max_arc)
        self.outer_min_arc = int(outer_min_arc)
        self.outer_max_arc = int(outer_max_arc)
        self.sae = np.full((2, self.height, self.width), -np.inf, dtype=np.float64)

    def reset(self) -> None:
        self.sae.fill(-np.inf)

    @staticmethod
    def _ring(surface: np.ndarray, x: int, y: int,
              circle: np.ndarray) -> np.ndarray:
        return surface[y + circle[:, 1], x + circle[:, 0]]

    def detect(
        self,
        t_ms: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Update the SAE in input order and return one corner flag per event."""

        t_input = np.asarray(t_ms)
        x_input = np.asarray(x)
        y_input = np.asarray(y)
        p_input = np.asarray(p)
        if any(array.ndim != 1 for array in (t_input, x_input, y_input, p_input)):
            raise ValueError("event arrays must be one-dimensional")
        if t_input.dtype.kind not in "iuf":
            raise ValueError("timestamps must be numeric")
        for name, array in (("x", x_input), ("y", y_input)):
            if array.dtype.kind not in "iu":
                raise ValueError(f"{name} coordinates must be integers")
        if p_input.dtype.kind not in "biu":
            raise ValueError("polarity must be encoded as integer 0 or 1")
        t_ms = t_input.astype(np.float64, copy=False)
        x = x_input.astype(np.int64, casting="safe", copy=False)
        y = y_input.astype(np.int64, casting="safe", copy=False)
        p = p_input.astype(np.int64, casting="safe", copy=False)
        if len({array.size for array in (t_ms, x, y, p)}) != 1:
            raise ValueError("event arrays have inconsistent lengths")
        if t_ms.size and (
            not np.all(np.isfinite(t_ms)) or np.any(np.diff(t_ms) < 0.0)
        ):
            raise ValueError("timestamps must be finite and nondecreasing")
        if x.size and (
            np.any(x < 0) or np.any(x >= self.width)
            or np.any(y < 0) or np.any(y >= self.height)
        ):
            raise ValueError("event coordinate lies outside the sensor")
        if p.size and not np.all((p == 0) | (p == 1)):
            raise ValueError("polarity must be 0 or 1")
        result = np.zeros(t_ms.size, dtype=np.bool_)
        for index in range(t_ms.size):
            xi, yi, polarity = int(x[index]), int(y[index]), int(p[index])
            surface = self.sae[polarity]
            # The public eFAST implementation updates the SAE before testing.
            surface[yi, xi] = float(t_ms[index])
            if xi < 4 or xi >= self.width - 4 or yi < 4 or yi >= self.height - 4:
                continue
            inner = self._ring(surface, xi, yi, CIRCLE3)
            outer = self._ring(surface, xi, yi, CIRCLE4)
            result[index] = (
                has_newest_arc(inner, self.inner_min_arc, self.inner_max_arc)
                and has_newest_arc(outer, self.outer_min_arc, self.outer_max_arc)
            )
        return result

    def detect_events(self, events: RelativeEventSequence) -> np.ndarray:
        if events.width != self.width or events.height != self.height:
            raise ValueError("event and detector sensor dimensions differ")
        return self.detect(events.t_ms, events.x, events.y, events.p)


class EFastDetector(_EFastDetectorUnchecked):
    """Reference eFAST detector with the reproduction arcs frozen."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        inner_min_arc: int = 3,
        inner_max_arc: int = 6,
        outer_min_arc: int = 4,
        outer_max_arc: int = 8,
    ) -> None:
        requested = (
            inner_min_arc, inner_max_arc, outer_min_arc, outer_max_arc,
        )
        if requested != (3, 6, 4, 8):
            raise ValueError(
                "non-reference eFAST arcs belong to esfo.experimental"
            )
        super().__init__(
            width, height,
            inner_min_arc=inner_min_arc,
            inner_max_arc=inner_max_arc,
            outer_min_arc=outer_min_arc,
            outer_max_arc=outer_max_arc,
        )


__all__ = ["CIRCLE3", "CIRCLE4", "EFastDetector", "has_newest_arc"]
