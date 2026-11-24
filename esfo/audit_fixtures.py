"""Deterministic synthetic inputs used only for result-independent audits."""

from __future__ import annotations

import numpy as np

from .efast import CIRCLE3, CIRCLE4


EFAST_ORACLE_RANDOM_SEED = 20_240_510
EFAST_ORACLE_RANDOM_EVENT_COUNT = 5_000
EFAST_ORACLE_PULSE_COUNT = 1_000


def build_efast_oracle_events(
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positive-time random events followed by known eFAST corner pulses."""

    rng = np.random.default_rng(EFAST_ORACLE_RANDOM_SEED)
    random_dt = rng.integers(
        1, 80, size=EFAST_ORACLE_RANDOM_EVENT_COUNT, dtype=np.int64,
    )
    random_t_us = 1_000_000 + np.cumsum(random_dt)
    random_x = rng.integers(
        0, 240, size=EFAST_ORACLE_RANDOM_EVENT_COUNT, dtype=np.int64,
    )
    random_y = rng.integers(
        0, 180, size=EFAST_ORACLE_RANDOM_EVENT_COUNT, dtype=np.int64,
    )
    random_p = rng.integers(
        0, 2, size=EFAST_ORACLE_RANDOM_EVENT_COUNT, dtype=np.int64,
    )
    t_values: list[int] = list(random_t_us)
    x_values: list[int] = list(random_x)
    y_values: list[int] = list(random_y)
    p_values: list[int] = list(random_p)
    base = int(random_t_us[-1] + 1_000)
    for index in range(EFAST_ORACLE_PULSE_COUNT):
        cx = 20 + (index % 20) * 10
        cy = 20 + ((index // 20) % 12) * 11
        polarity = index % 2
        start = base + index * 100
        for offset, (dx, dy) in enumerate(CIRCLE3[:3]):
            t_values.append(start + offset)
            x_values.append(cx + int(dx))
            y_values.append(cy + int(dy))
            p_values.append(polarity)
        for offset, (dx, dy) in enumerate(CIRCLE4[:4]):
            t_values.append(start + 10 + offset)
            x_values.append(cx + int(dx))
            y_values.append(cy + int(dy))
            p_values.append(polarity)
        t_values.append(start + 20)
        x_values.append(cx)
        y_values.append(cy)
        p_values.append(polarity)
    arrays = tuple(np.asarray(value, dtype=np.int64) for value in (
        t_values, x_values, y_values, p_values,
    ))
    return arrays  # type: ignore[return-value]


__all__ = [
    "EFAST_ORACLE_PULSE_COUNT", "EFAST_ORACLE_RANDOM_EVENT_COUNT",
    "EFAST_ORACLE_RANDOM_SEED", "build_efast_oracle_events",
]
