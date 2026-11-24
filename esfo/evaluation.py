"""Fail-closed Gate-D scene outcomes and orbit reprojection summaries.

This module reports the frozen reference path; it does not add recovery,
filtering, or an alternate success criterion.  In particular, GTSAM's
constant cheirality penalty is never interpreted as a predicted pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import gtsam
import numpy as np

from .colmap import ColmapPipelineError
from .frequency import FrequencyInitializationError
from .gtsam_backend import OrbitFactorBinding, _factor_evaluation
from .orbit_factor import OrbitFactorError


SCENE_FAILURE_CODES = frozenset({
    "input_validation_failure",
    "frontend_failure",
    "frequency_initialization_failure",
    "fft_frequency_unobservable",
    "colmap_execution_failure",
    "colmap_validator_failure",
    "colmap_method_failure",
    "geometry_initialization_failure",
    "optimizer_indeterminate_linear_system",
    "orbit_factor_normal_degenerate",
    "orbit_factor_u_degenerate",
    "orbit_factor_projection_nonfinite",
    "orbit_factor_failure",
    "final_nonpositive_depth",
    "resource_timeout",
    "resource_oom",
    "metric_unavailable",
})

# COLMAP can finish normally yet produce no reconstruction usable by the
# frozen eSfO stages.  Those outcomes are method failures, not corrupt-output
# validator failures.  Keep this small allow-list tied to stable
# ``ColmapPipelineError.code`` values; every other error raised while auditing
# a database/model remains a validator failure.
COLMAP_METHOD_FAILURE_DETAIL_CODES = frozenset({
    "no_raw_pairs",
    "no_verified_pairs",
    "no_model",
    "insufficient_registered_views",
})


class FinalNonpositiveDepthError(RuntimeError):
    """Fail-closed final-state cheirality outcome for the reference path.

    GTSAM is allowed to traverse non-positive-depth trial states with the
    pinned constant residual, but an accepted final state may not retain any
    such factor.  Keeping this outcome as a dedicated exception prevents the
    integration runner from collapsing it into a generic optimizer failure.
    """

    code = "final_nonpositive_depth"

    def __init__(self, count: int, factor_count: int):
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or int(count) <= 0):
            raise ValueError("count must be a positive integer")
        if (isinstance(factor_count, bool)
                or not isinstance(factor_count, (int, np.integer))
                or int(factor_count) < int(count)):
            raise ValueError("factor_count must be an integer >= count")
        self.count = int(count)
        self.factor_count = int(factor_count)
        super().__init__(
            "final_nonpositive_depth: "
            f"{self.count}/{self.factor_count} final factors have "
            "non-positive depth",
        )


@dataclass(frozen=True)
class SceneFailureRecord:
    """Stable first-failure identity for one reference scene."""

    stage: str
    code: str
    detail_code: str | None
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if self.code not in SCENE_FAILURE_CODES:
            raise ValueError(f"unknown scene failure code: {self.code}")
        if self.detail_code is not None and (
            not isinstance(self.detail_code, str) or not self.detail_code.strip()
        ):
            raise ValueError("detail_code must be None or a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")


def classify_scene_failure(error: BaseException, *, stage: str) -> SceneFailureRecord:
    """Map an exception to the frozen reporting taxonomy without recovery.

    The caller supplies the stage because a COLMAP validation failure and a
    COLMAP execution failure can carry the same low-level exception type.
    The three reportable direct-factor failures are always kept separate.
    """

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")

    detail = getattr(error, "code", None)
    if isinstance(error, FinalNonpositiveDepthError):
        code = "final_nonpositive_depth"
        detail = error.code
    elif isinstance(error, OrbitFactorError):
        code = {
            "normal_degenerate": "orbit_factor_normal_degenerate",
            "u_degenerate": "orbit_factor_u_degenerate",
            "projection_nonfinite": "orbit_factor_projection_nonfinite",
        }.get(error.code, "orbit_factor_failure")
    elif isinstance(error, FrequencyInitializationError):
        detail = error.reason
        code = (
            "fft_frequency_unobservable"
            if error.reason == "fft_frequency_unobservable"
            else "frequency_initialization_failure"
        )
    elif isinstance(error, ColmapPipelineError):
        if error.code in COLMAP_METHOD_FAILURE_DETAIL_CODES:
            code = "colmap_method_failure"
        elif stage in {"colmap_validation", "colmap_method"}:
            code = "colmap_validator_failure"
        else:
            code = "colmap_execution_failure"
    elif error.__class__.__name__ == "IndeterminantLinearSystemException":
        code = "optimizer_indeterminate_linear_system"
        detail = error.__class__.__name__
    elif isinstance(error, MemoryError):
        code = "resource_oom"
        detail = error.__class__.__name__
    elif isinstance(error, TimeoutError):
        code = "resource_timeout"
        detail = error.__class__.__name__
    elif isinstance(error, (TypeError, ValueError)) and stage == "input":
        code = "input_validation_failure"
        detail = error.__class__.__name__
    else:
        code = {
            "frontend": "frontend_failure",
            "frequency": "frequency_initialization_failure",
            "colmap": "colmap_execution_failure",
            "colmap_validation": "colmap_validator_failure",
            "geometry": "geometry_initialization_failure",
            "optimization": "orbit_factor_failure",
            "metrics": "metric_unavailable",
        }.get(stage, "metric_unavailable")
        detail = detail or error.__class__.__name__
    return SceneFailureRecord(stage, code, detail, str(error) or repr(error))


def _finite_nonnegative(name: str, value: Any) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))
            or float(value) < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class ResidualSummary:
    """Summary of per-factor L2 pixel residuals."""

    count: int
    mean_px: float
    median_px: float
    rmse_px: float
    p90_px: float
    p95_px: float
    max_px: float

    def __post_init__(self) -> None:
        if (isinstance(self.count, bool) or not isinstance(self.count, (int, np.integer))
                or int(self.count) <= 0):
            raise ValueError("count must be a positive integer")
        object.__setattr__(self, "count", int(self.count))
        for name in (
            "mean_px", "median_px", "rmse_px", "p90_px", "p95_px", "max_px",
        ):
            object.__setattr__(self, name, _finite_nonnegative(name, getattr(self, name)))


def summarize_residuals(residual_xy: Any) -> ResidualSummary:
    """Summarize finite two-pixel residual vectors without hidden filtering."""

    values = np.asarray(residual_xy)
    if values.dtype.kind not in "iuf" or values.ndim != 2 or values.shape[1:] != (2,):
        raise ValueError("residual_xy must have finite numeric shape [N,2]")
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        raise ValueError("at least one residual is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("residual_xy contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    return ResidualSummary(
        count=int(norms.size),
        mean_px=float(np.mean(norms)),
        median_px=float(np.median(norms)),
        rmse_px=float(np.sqrt(np.mean(norms * norms))),
        p90_px=float(np.percentile(norms, 90.0)),
        p95_px=float(np.percentile(norms, 95.0)),
        max_px=float(np.max(norms)),
    )


@dataclass(frozen=True)
class MetricAvailability:
    """JSON-safe availability wrapper; unavailable values are exactly null."""

    status: str
    reason: str | None
    value: ResidualSummary | None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("metric status must be available or unavailable")
        if self.status == "available":
            if self.reason is not None or not isinstance(self.value, ResidualSummary):
                raise ValueError("available metric requires a value and no reason")
        elif (not isinstance(self.reason, str) or not self.reason.strip()
              or self.value is not None):
            raise ValueError("unavailable metric requires a reason and null value")


@dataclass(frozen=True)
class InFrontResidualDiagnostic:
    """Secondary diagnostic over positive-depth factors only."""

    metric: MetricAvailability
    in_front_count: int
    excluded_nonpositive_depth_count: int
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.metric, MetricAvailability):
            raise TypeError("metric must be MetricAvailability")
        for name in ("in_front_count", "excluded_nonpositive_depth_count"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.metric.status == "available":
            assert self.metric.value is not None
            if self.metric.value.count != self.in_front_count:
                raise ValueError("diagnostic count and summary disagree")
        elif self.in_front_count != 0:
            raise ValueError("a nonempty in-front diagnostic must be available")
        if self.diagnostic_only is not True:
            raise ValueError("in-front metric must remain diagnostic-only")


@dataclass(frozen=True)
class OrbitReprojectionReport:
    """Before/after metrics for the exact same fixed factor set."""

    factor_count: int
    initial_nonpositive_depth_count: int
    final_nonpositive_depth_count: int
    before_primary: MetricAvailability
    after_primary: MetricAvailability
    before_in_front: InFrontResidualDiagnostic
    after_in_front: InFrontResidualDiagnostic

    def __post_init__(self) -> None:
        for name in (
            "factor_count", "initial_nonpositive_depth_count",
            "final_nonpositive_depth_count",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if (self.initial_nonpositive_depth_count > self.factor_count
                or self.final_nonpositive_depth_count > self.factor_count):
            raise ValueError("non-positive-depth count exceeds factor count")
        for item in (
            self.before_primary, self.after_primary,
            self.before_in_front, self.after_in_front,
        ):
            if not isinstance(item, (MetricAvailability, InFrontResidualDiagnostic)):
                raise TypeError("invalid reprojection report component")


def _diagnostic_at_values(
    values: gtsam.Values,
    bindings: Sequence[OrbitFactorBinding],
) -> InFrontResidualDiagnostic:
    if not isinstance(values, gtsam.Values):
        raise TypeError("values must be gtsam.Values")
    residuals: list[np.ndarray] = []
    excluded = 0
    for binding in bindings:
        if not isinstance(binding, OrbitFactorBinding):
            raise TypeError("bindings must contain OrbitFactorBinding values")
        evaluation = _factor_evaluation(values, binding.keys, binding.measurement)
        if evaluation.cheirality_penalty:
            excluded += 1
        else:
            residuals.append(np.asarray(evaluation.residual_xy, dtype=np.float64))
    if residuals:
        metric = MetricAvailability(
            "available", None, summarize_residuals(np.stack(residuals)),
        )
    else:
        metric = MetricAvailability("unavailable", "no_in_front_factors", None)
    return InFrontResidualDiagnostic(metric, len(residuals), excluded)


def build_orbit_reprojection_report(
    initial_values: gtsam.Values,
    final_values: gtsam.Values,
    bindings: Sequence[OrbitFactorBinding],
) -> OrbitReprojectionReport:
    """Evaluate frozen factors before and after optimization.

    A nonzero final cheirality count invalidates both primary distributions.
    When only the initial state has invalid depths, the before distribution is
    unavailable rather than silently dropping factors.  Positive-depth-only
    values remain separately labelled diagnostics and never affect success.
    """

    bindings_tuple = tuple(bindings)
    before = _diagnostic_at_values(initial_values, bindings_tuple)
    after = _diagnostic_at_values(final_values, bindings_tuple)
    factor_count = len(bindings_tuple)
    initial_bad = before.excluded_nonpositive_depth_count
    final_bad = after.excluded_nonpositive_depth_count

    if factor_count == 0:
        unavailable = MetricAvailability("unavailable", "no_factors", None)
        before_primary = after_primary = unavailable
    elif final_bad:
        unavailable = MetricAvailability(
            "unavailable", "final_nonpositive_depth", None,
        )
        before_primary = after_primary = unavailable
    else:
        assert after.metric.status == "available"
        after_primary = MetricAvailability("available", None, after.metric.value)
        before_primary = (
            MetricAvailability("unavailable", "initial_nonpositive_depth", None)
            if initial_bad else
            MetricAvailability("available", None, before.metric.value)
        )
    return OrbitReprojectionReport(
        factor_count, initial_bad, final_bad,
        before_primary, after_primary, before, after,
    )


__all__ = [
    "FinalNonpositiveDepthError",
    "InFrontResidualDiagnostic", "MetricAvailability",
    "OrbitReprojectionReport", "ResidualSummary",
    "COLMAP_METHOD_FAILURE_DETAIL_CODES", "SCENE_FAILURE_CODES",
    "SceneFailureRecord", "build_orbit_reprojection_report",
    "classify_scene_failure", "summarize_residuals",
]
