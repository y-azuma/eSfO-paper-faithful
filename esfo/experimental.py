"""Mechanically separate, explicitly non-reference sensitivity execution."""

from __future__ import annotations

from dataclasses import dataclass

from .config import FrontendConfig, REFERENCE_FRONTEND, reference_mismatches
from .data import EventSequence, RelativeEventSequence
from .frontend import FrontendResult, _run_frontend_unchecked


@dataclass(frozen=True)
class ExperimentalFrontendResult:
    evidence_class: str
    reference_mismatches: tuple[str, ...]
    result: FrontendResult


def _require_sensitivity(config: FrontendConfig) -> tuple[str, ...]:
    mismatches = tuple(reference_mismatches(config))
    if not mismatches:
        raise ValueError("experimental runner requires a non-reference setting")
    return mismatches


def run_frontend_sensitivity(
    events: RelativeEventSequence,
    config: FrontendConfig,
) -> ExperimentalFrontendResult:
    if (config.timestamp_scale_us_to_ms
            != REFERENCE_FRONTEND.timestamp_scale_us_to_ms):
        raise ValueError(
            "timestamp-scale sensitivity requires raw EventSequence input"
        )
    mismatches = _require_sensitivity(config)
    return ExperimentalFrontendResult(
        "Experimental", mismatches, _run_frontend_unchecked(events, config),
    )


def run_frontend_raw_sensitivity(
    events: EventSequence,
    config: FrontendConfig,
) -> ExperimentalFrontendResult:
    mismatches = _require_sensitivity(config)
    relative = events.to_relative_ms(config.timestamp_scale_us_to_ms)
    return ExperimentalFrontendResult(
        "Experimental", mismatches, _run_frontend_unchecked(relative, config),
    )


__all__ = [
    "ExperimentalFrontendResult", "run_frontend_raw_sensitivity",
    "run_frontend_sensitivity",
]
