"""Paper Sec. 4 front-end stages under frozen Gate-0 conventions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.metadata
from types import MappingProxyType
import warnings
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .config import FrontendConfig, assert_reference_config
from .data import EventSequence, RelativeEventSequence
from .efast import _EFastDetectorUnchecked
from .tracks import (
    AssociationResult, EventCluster, TrackObservation,
    _merge_clusters_unchecked, _track_observations_unchecked, build_clusters,
)


def _indices(value: np.ndarray | list[int], size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        if array.ndim != 1 or array.size != size:
            raise ValueError(f"{name} boolean mask has the wrong shape")
        return np.flatnonzero(array).astype(np.int64, copy=False)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.size:
        return np.empty(0, dtype=np.int64)
    try:
        array = array.astype(np.int64, casting="safe", copy=False)
    except TypeError as error:
        raise ValueError(f"{name} must contain integer indices") from error
    if array.size and (
        np.any(array < 0) or np.any(array >= size) or np.any(np.diff(array) <= 0)
    ):
        raise ValueError(f"{name} must be unique, increasing, and in range")
    return array


def _frozen_array(value: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
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
        return MappingProxyType({str(key): _deep_freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class DensityFilterResult:
    """Eq. (2) values in the same order as the supplied corner indices."""

    corner_indices: np.ndarray
    scores: np.ndarray
    thresholds: np.ndarray
    keep: np.ndarray
    kept_indices: np.ndarray

    def __post_init__(self) -> None:
        corner_indices = _frozen_array(
            self.corner_indices, np.dtype(np.int64), "corner_indices",
        )
        scores = _frozen_array(self.scores, np.dtype(np.float64), "scores")
        thresholds = _frozen_array(
            self.thresholds, np.dtype(np.float64), "thresholds",
        )
        keep = _frozen_array(self.keep, np.dtype(np.bool_), "keep")
        kept_indices = _frozen_array(
            self.kept_indices, np.dtype(np.int64), "kept_indices",
        )
        if not (corner_indices.size == scores.size == thresholds.size == keep.size):
            raise ValueError("density result arrays have inconsistent lengths")
        object.__setattr__(self, "corner_indices", corner_indices)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "keep", keep)
        object.__setattr__(self, "kept_indices", kept_indices)


@dataclass(frozen=True)
class ClusterResult:
    """HDBSCAN labels and stable clusters; label ``-1`` remains noise."""

    corner_indices: np.ndarray
    labels: np.ndarray
    clusters: tuple[EventCluster, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        corners = _frozen_array(
            self.corner_indices, np.dtype(np.int64), "corner_indices",
        )
        labels = _frozen_array(self.labels, np.dtype(np.int64), "labels")
        if corners.size != labels.size:
            raise ValueError("cluster result arrays have inconsistent lengths")
        object.__setattr__(self, "corner_indices", corners)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


@dataclass(frozen=True)
class FrontendResult:
    """Complete Sec. 4 output for one offline event sequence."""

    corner_mask: np.ndarray
    density: DensityFilterResult
    clustering: ClusterResult
    association: AssociationResult
    observations: tuple[TrackObservation, ...]

    def __post_init__(self) -> None:
        corner_mask = _frozen_array(
            self.corner_mask, np.dtype(np.bool_), "corner_mask",
        )
        object.__setattr__(self, "corner_mask", corner_mask)
        object.__setattr__(self, "observations", tuple(self.observations))


def _density_filter_unchecked(
    events: RelativeEventSequence,
    corner_indices: np.ndarray | list[int],
    config: FrontendConfig | None = None,
) -> DensityFilterResult:
    """Evaluate and threshold paper Eq. (2).

    Reference mode queries raw, same-polarity events in a Euclidean
    ``(t_ms,x,y)`` ball of inclusive radius seven.  The mean threshold is
    computed over all corner queries of each polarity for the entire sequence.
    """

    config = FrontendConfig() if config is None else config
    corners = _indices(corner_indices, len(events), "corner_indices")
    scores = np.zeros(corners.size, dtype=np.float64)
    thresholds = np.zeros(corners.size, dtype=np.float64)
    if not corners.size:
        keep = np.zeros(0, dtype=np.bool_)
        kept = corners.copy()
        for array in (corners, scores, thresholds, keep, kept):
            array.setflags(write=False)
        return DensityFilterResult(
            corners, scores, thresholds, keep, kept,
        )
    if config.density_neighbors == "raw_events":
        neighbor_indices = np.arange(len(events), dtype=np.int64)
    else:
        neighbor_indices = corners
    all_points = events.points_txy()
    if config.density_distance_domain == "x_y":
        all_points = all_points[:, 1:]
    for polarity in (0, 1):
        query_rows = np.flatnonzero(events.p[corners] == polarity)
        if not query_rows.size:
            continue
        pool = neighbor_indices[events.p[neighbor_indices] == polarity]
        if not pool.size:
            continue
        tree = cKDTree(all_points[pool])
        counts = tree.query_ball_point(
            all_points[corners[query_rows]],
            r=config.density_radius,
            return_length=True,
        ).astype(np.float64)
        if not config.density_include_query:
            # Every corner query is part of both allowed neighbor pools.
            counts -= 1.0
        scores[query_rows] = counts / config.density_radius
        thresholds[query_rows] = float(np.mean(scores[query_rows]))
    if config.density_threshold_scope == "global_whole_sequence":
        thresholds[:] = float(np.mean(scores))
    keep = scores >= thresholds  # The paper removes only D < mu_D.
    kept = corners[keep]
    for array in (corners, scores, thresholds, keep, kept):
        array.setflags(write=False)
    return DensityFilterResult(corners, scores, thresholds, keep, kept)


def density_filter(
    events: RelativeEventSequence,
    corner_indices: np.ndarray | list[int],
    config: FrontendConfig | None = None,
) -> DensityFilterResult:
    """Evaluate Eq. (2) only under the frozen reference configuration."""

    config = FrontendConfig() if config is None else config
    assert_reference_config(config)
    return _density_filter_unchecked(events, corner_indices, config)


def _hdbscan_parameters_unchecked(config: FrontendConfig) -> dict[str, Any]:
    """Return every effective HDBSCAN option recorded in provenance."""

    return {
        "min_cluster_size": config.hdbscan_min_cluster_size,
        "min_samples": config.hdbscan_min_samples,
        "cluster_selection_epsilon": config.hdbscan_cluster_selection_epsilon,
        "metric": config.hdbscan_metric,
        "alpha": config.hdbscan_alpha,
        "algorithm": config.hdbscan_algorithm,
        "leaf_size": config.hdbscan_leaf_size,
        "approx_min_span_tree": config.hdbscan_approx_min_span_tree,
        "gen_min_span_tree": config.hdbscan_gen_min_span_tree,
        "core_dist_n_jobs": config.hdbscan_core_dist_n_jobs,
        "cluster_selection_method": config.hdbscan_cluster_selection_method,
        "allow_single_cluster": config.hdbscan_allow_single_cluster,
        "prediction_data": config.hdbscan_prediction_data,
        "match_reference_implementation": (
            config.hdbscan_match_reference_implementation
        ),
        # Explicitly record otherwise inherited hdbscan==0.8.33 defaults.
        "max_cluster_size": 0,
        "p": None,
    }


def hdbscan_parameters(config: FrontendConfig | None = None) -> dict[str, Any]:
    """Return the reference HDBSCAN options recorded in provenance."""

    config = FrontendConfig() if config is None else config
    assert_reference_config(config)
    return _hdbscan_parameters_unchecked(config)


def _cluster_corners_unchecked(
    events: RelativeEventSequence,
    corner_indices: np.ndarray | list[int],
    config: FrontendConfig | None = None,
) -> ClusterResult:
    """Cluster the complete filtered-corner stream in ``(t_ms,x,y)``."""

    config = FrontendConfig() if config is None else config
    corners = _indices(corner_indices, len(events), "corner_indices")
    parameters = _hdbscan_parameters_unchecked(config)
    try:
        version = importlib.metadata.version("hdbscan")
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("reference clustering requires hdbscan==0.8.33") from error
    metadata: dict[str, Any] = {
        "implementation": "hdbscan.HDBSCAN",
        "version": version,
        "coordinate_order": ["t_ms", "x", "y"],
        "polarity_used": False,
        "whole_sequence": True,
        "parameters": parameters.copy(),
    }
    if version != "0.8.33":
        raise RuntimeError(
            f"reference clustering requires hdbscan==0.8.33, found {version}"
        )
    labels = np.full(corners.size, -1, dtype=np.int64)
    effective_min_samples = (
        config.hdbscan_min_cluster_size
        if config.hdbscan_min_samples is None
        else config.hdbscan_min_samples
    )
    metadata["effective_min_samples"] = effective_min_samples
    if corners.size:
        try:
            import hdbscan
        except ImportError as error:  # pragma: no cover - metadata check above catches it.
            raise ImportError("reference clustering requires hdbscan==0.8.33") from error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            labels = np.asarray(
                hdbscan.HDBSCAN(**parameters).fit_predict(
                    events.points_txy()[corners]
                ),
                dtype=np.int64,
            )
    labels.setflags(write=False)
    corners.setflags(write=False)
    clusters = build_clusters(events, corners, labels)
    metadata["noise_count"] = int(np.count_nonzero(labels == -1))
    metadata["cluster_count"] = len(clusters)
    return ClusterResult(corners, labels, clusters, metadata)


def cluster_corners(
    events: RelativeEventSequence,
    corner_indices: np.ndarray | list[int],
    config: FrontendConfig | None = None,
) -> ClusterResult:
    """Cluster corners only under the frozen reference configuration."""

    config = FrontendConfig() if config is None else config
    assert_reference_config(config)
    return _cluster_corners_unchecked(events, corner_indices, config)


def _run_frontend_unchecked(
    events: RelativeEventSequence,
    config: FrontendConfig | None = None,
) -> FrontendResult:
    """Execute Gate A without chunking, subsampling, or public-data tuning."""

    config = FrontendConfig() if config is None else config
    detector = _EFastDetectorUnchecked(
        events.width, events.height,
        inner_min_arc=config.efast_inner_min_arc,
        inner_max_arc=config.efast_inner_max_arc,
        outer_min_arc=config.efast_outer_min_arc,
        outer_max_arc=config.efast_outer_max_arc,
    )
    corner_mask = detector.detect_events(events)
    corner_mask.setflags(write=False)
    density = _density_filter_unchecked(
        events, np.flatnonzero(corner_mask), config,
    )
    clustering = _cluster_corners_unchecked(
        events, density.kept_indices, config,
    )
    association = _merge_clusters_unchecked(
        clustering.clusters,
        n_sigma=config.association_descriptor_events,
        phi=config.association_radius,
    )
    observations = _track_observations_unchecked(
        association.tracks,
        sequence_duration_ms=events.duration_ms,
        delta_t_ms=config.track_window_ms,
    )
    return FrontendResult(
        corner_mask, density, clustering, association, observations,
    )


def run_frontend(
    events: RelativeEventSequence,
    config: FrontendConfig | None = None,
) -> FrontendResult:
    """Execute only the frozen reference profile.

    Sensitivity configurations are deliberately rejected here and belong to
    :mod:`esfo.experimental`.
    """
    config = FrontendConfig() if config is None else config
    assert_reference_config(config)
    return _run_frontend_unchecked(events, config)


def run_frontend_raw(
    events: EventSequence,
    config: FrontendConfig | None = None,
) -> FrontendResult:
    """Convert raw microseconds with the configured scale, then run Gate A."""
    config = FrontendConfig() if config is None else config
    assert_reference_config(config)
    relative = events.to_relative_ms(config.timestamp_scale_us_to_ms)
    return _run_frontend_unchecked(relative, config)


__all__ = [
    "ClusterResult", "DensityFilterResult", "FrontendResult",
    "cluster_corners", "density_filter", "hdbscan_parameters",
    "run_frontend", "run_frontend_raw",
]
