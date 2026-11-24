"""Lossless Gate-D aggregation of frozen reference-stage diagnostics.

The values collected here are observational only.  They never filter tracks,
factors, views, or scenes and therefore cannot alter the reference result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .colmap import (
    ExtractedColmapModel,
    IndexedView,
    expected_raw_correspondence_count,
    iter_pair_matches,
)
from .data import EventSequence
from .frequency import FFTFrequencyInitialization
from .frontend import FrontendResult
from .geometry import OrbitGeometryInitialization
from .gtsam_backend import (
    FactorObservationSet,
    OrbitOptimizationDiagnostics,
)


def _freeze(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType({str(key): item for key, item in value.items()})


def _count(name: str, value: Any) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or int(value) < 0):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class SceneDiagnostics:
    """Required per-scene diagnostic sections for Gate E reporting."""

    frontend: Mapping[str, Any]
    colmap: Mapping[str, Any]
    geometry: Mapping[str, Any]
    frequency: Mapping[str, Any]
    factors: Mapping[str, Any]
    optimization: Mapping[str, Any]

    def __post_init__(self) -> None:
        required = {
            "frontend": {
                "event_count", "corner_count", "density_kept_corner_count",
                "hdbscan_noise_count", "cluster_count", "track_count",
                "observation_count", "merged_track_count",
                "temporally_overlapping_cluster_pair_count",
                "temporally_overlapping_track_count",
                "hdbscan_repeat_labels_equal",
            },
            "colmap": {
                "logical_window_count", "registered_window_count",
                "unregistered_window_count", "registered_window_ratio",
                "all_pairs_image_pair_count", "raw_correspondence_row_count",
                "verified_inlier_row_count", "mapped_track_count",
                "split_track_count", "merged_track_count",
                "uninitialized_track_count", "excluded_track_union_count",
            },
            "geometry": {
                "temporal_total_pair_count", "temporal_eligible_pair_count",
                "temporal_excluded_pair_count", "temporal_max_phase_gap_rad",
            },
            "frequency": {
                "frequency_hz", "peak_bin", "peak_power",
                "frequency_resolution_hz", "zero_positive_power",
            },
            "factors": {
                "total_factor_count", "registered_window_factor_count",
                "unregistered_window_factor_count", "registered_factor_ratio",
                "retained_colmap_track_factor_count",
                "absent_colmap_track_factor_count",
                "missing_track_window_pair_count",
            },
            "optimization": {
                "factor_count", "initial_nonpositive_depth_count",
                "final_nonpositive_depth_count", "signed_radius",
                "signed_frequency_hz",
            },
        }
        for name, names in required.items():
            section = _freeze(getattr(self, name), name)
            missing = names - set(section)
            if missing:
                raise ValueError(f"{name} diagnostics lack {sorted(missing)}")
            object.__setattr__(self, name, section)
        logical = _count(
            "logical_window_count", self.colmap["logical_window_count"],
        )
        registered = _count(
            "registered_window_count", self.colmap["registered_window_count"],
        )
        unregistered = _count(
            "unregistered_window_count", self.colmap["unregistered_window_count"],
        )
        if registered + unregistered != logical:
            raise ValueError("registered window accounting is inconsistent")
        ratio = self.colmap["registered_window_ratio"]
        if logical == 0:
            if ratio is not None:
                raise ValueError("zero-window registered ratio must be null")
        elif (not isinstance(ratio, (int, float, np.number))
              or not math.isfinite(float(ratio))
              or float(ratio) != registered / logical):
            raise ValueError("registered window ratio is inconsistent")
        total_factors = _count(
            "total_factor_count", self.factors["total_factor_count"],
        )
        registered_factors = _count(
            "registered_window_factor_count",
            self.factors["registered_window_factor_count"],
        )
        unregistered_factors = _count(
            "unregistered_window_factor_count",
            self.factors["unregistered_window_factor_count"],
        )
        if registered_factors + unregistered_factors != total_factors:
            raise ValueError("registered factor accounting is inconsistent")
        factor_ratio = self.factors["registered_factor_ratio"]
        if total_factors == 0:
            if factor_ratio is not None:
                raise ValueError("zero-factor registered ratio must be null")
        elif float(factor_ratio) != registered_factors / total_factors:
            raise ValueError("registered factor ratio is inconsistent")
        if _count("optimizer factor_count", self.optimization["factor_count"]) != total_factors:
            raise ValueError("factor and optimizer diagnostics disagree")


def aggregate_scene_diagnostics(
    *,
    events: EventSequence,
    frontend: FrontendResult,
    views: Sequence[IndexedView],
    model: ExtractedColmapModel,
    geometry: OrbitGeometryInitialization,
    frequency: FFTFrequencyInitialization,
    factors: FactorObservationSet,
    optimization: OrbitOptimizationDiagnostics,
    verified_inlier_row_count: int,
    hdbscan_repeat_labels_equal: bool,
    signed_radius: float,
    signed_frequency_hz: float,
) -> SceneDiagnostics:
    """Copy every N5 diagnostic into one immutable scene record."""

    if not isinstance(events, EventSequence):
        raise TypeError("events must be EventSequence")
    if not isinstance(frontend, FrontendResult):
        raise TypeError("frontend must be FrontendResult")
    views_tuple = tuple(views)
    if any(not isinstance(view, IndexedView) for view in views_tuple):
        raise TypeError("views must contain IndexedView values")
    if not isinstance(model, ExtractedColmapModel):
        raise TypeError("model must be ExtractedColmapModel")
    if not isinstance(geometry, OrbitGeometryInitialization):
        raise TypeError("geometry must be OrbitGeometryInitialization")
    if not isinstance(frequency, FFTFrequencyInitialization):
        raise TypeError("frequency must be FFTFrequencyInitialization")
    if not isinstance(factors, FactorObservationSet):
        raise TypeError("factors must be FactorObservationSet")
    if not isinstance(optimization, OrbitOptimizationDiagnostics):
        raise TypeError("optimization must be OrbitOptimizationDiagnostics")
    if not isinstance(hdbscan_repeat_labels_equal, bool):
        raise TypeError("hdbscan_repeat_labels_equal must be bool")
    verified_rows = _count(
        "verified_inlier_row_count", verified_inlier_row_count,
    )
    for name, value in (
        ("signed_radius", signed_radius),
        ("signed_frequency_hz", signed_frequency_hz),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))):
            raise ValueError(f"{name} must be finite")

    association = frontend.association.diagnostics
    registered_images = set(int(value) for value in model.trajectory.view_ids)
    registered_windows = sum(
        view.image_id in registered_images for view in views_tuple
    )
    logical_windows = len(views_tuple)
    mapping = model.mapping
    factor_values = factors.diagnostics
    plane = geometry.plane_diagnostics
    factor_total = factor_values.total_factor_count
    return SceneDiagnostics(
        frontend={
            "event_count": len(events),
            "corner_count": int(np.count_nonzero(frontend.corner_mask)),
            "density_kept_corner_count": int(frontend.density.kept_indices.size),
            "hdbscan_noise_count": int(frontend.clustering.metadata["noise_count"]),
            "cluster_count": len(frontend.clustering.clusters),
            "track_count": len(frontend.association.tracks),
            "observation_count": len(frontend.observations),
            "merged_track_count": association.merged_track_count,
            "temporally_overlapping_cluster_pair_count": (
                association.temporally_overlapping_cluster_pair_count
            ),
            "temporally_overlapping_track_count": (
                association.temporally_overlapping_track_count
            ),
            "hdbscan_repeat_labels_equal": hdbscan_repeat_labels_equal,
            "hdbscan": frontend.clustering.metadata,
        },
        colmap={
            "logical_window_count": logical_windows,
            "registered_window_count": registered_windows,
            "unregistered_window_count": logical_windows - registered_windows,
            "registered_window_ratio": (
                None if logical_windows == 0 else registered_windows / logical_windows
            ),
            "all_pairs_image_pair_count": sum(1 for _ in iter_pair_matches(views_tuple)),
            "raw_correspondence_row_count": expected_raw_correspondence_count(
                views_tuple,
            ),
            "verified_inlier_row_count": verified_rows,
            "mapped_track_count": len(mapping.unique_pairs),
            "split_track_count": len(mapping.split_track_ids),
            "merged_track_count": len(mapping.merged_track_ids),
            "uninitialized_track_count": len(mapping.uninitialized_track_ids),
            "excluded_track_union_count": len(mapping.excluded_track_ids),
        },
        geometry={
            "temporal_total_pair_count": plane.temporal_total_pair_count,
            "temporal_eligible_pair_count": plane.temporal_eligible_pair_count,
            "temporal_excluded_pair_count": plane.temporal_excluded_pair_count,
            "temporal_max_phase_gap_rad": plane.temporal_max_phase_gap_rad,
        },
        frequency={
            "frequency_hz": frequency.frequency_hz,
            "peak_bin": frequency.peak_bin,
            "peak_power": frequency.peak_power,
            "frequency_resolution_hz": frequency.frequency_resolution_hz,
            "zero_positive_power": frequency.zero_positive_power,
        },
        factors={
            "total_factor_count": factor_total,
            "registered_window_factor_count": (
                factor_values.registered_window_factor_count
            ),
            "unregistered_window_factor_count": (
                factor_values.unregistered_window_factor_count
            ),
            "registered_factor_ratio": (
                None if factor_total == 0
                else factor_values.registered_window_factor_count / factor_total
            ),
            "retained_colmap_track_factor_count": (
                factor_values.retained_colmap_track_factor_count
            ),
            "absent_colmap_track_factor_count": (
                factor_values.absent_colmap_track_factor_count
            ),
            "missing_track_window_pair_count": (
                factor_values.missing_track_window_pair_count
            ),
            "split_track_count": factor_values.split_track_count,
            "merged_track_count": factor_values.merged_track_count,
            "uninitialized_track_count": factor_values.uninitialized_track_count,
        },
        optimization={
            "factor_count": optimization.factor_count,
            "initial_nonpositive_depth_count": (
                optimization.initial_nonpositive_depth_count
            ),
            "final_nonpositive_depth_count": (
                optimization.final_nonpositive_depth_count
            ),
            "signed_radius": float(signed_radius),
            "signed_frequency_hz": float(signed_frequency_hz),
        },
    )


__all__ = ["SceneDiagnostics", "aggregate_scene_diagnostics"]
