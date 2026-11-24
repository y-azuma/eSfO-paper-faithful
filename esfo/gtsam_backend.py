"""Pinned GTSAM adapter for the paper-faithful eSfO orbit factor.

The numerical geometry lives in :mod:`esfo.orbit_factor`.  This module only
binds its seven variable blocks to a direct ``gtsam.CustomFactor`` and freezes
the Eq. (4) observation set used by Gate C.  No prior, robust loss, positivity
transform, or optimizer retry is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import gtsam
import numpy as np

from .colmap import ExtractedColmapModel, IndexedView
from .orbit_factor import evaluate_orbit_factor


GTSAM_DISTRIBUTION = "gtsam"
GTSAM_VERSION = "4.2"


def assert_gtsam_version() -> None:
    """Fail closed unless the frozen direct backend is exactly GTSAM 4.2."""

    try:
        installed = version(GTSAM_DISTRIBUTION)
    except PackageNotFoundError as error:  # pragma: no cover - import implies it
        raise RuntimeError("gtsam distribution is not installed") from error
    if installed != GTSAM_VERSION:
        raise RuntimeError(
            f"expected {GTSAM_DISTRIBUTION}=={GTSAM_VERSION}, got {installed}",
        )


def _readonly(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape:
        raise ValueError(f"expected a finite numeric array with shape {shape}")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError("array contains non-finite values")
    result.setflags(write=False)
    return result


def _key(name: str, value: Any) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or int(value) < 0):
        raise ValueError(f"{name} must be a non-negative GTSAM key")
    return int(value)


@dataclass(frozen=True)
class OrbitFactorKeys:
    """Seven GTSAM keys in the frozen local-coordinate order."""

    point_w: int
    center_w: int
    radius: int
    frequency_hz: int
    R0: int
    normal_raw: int
    u_raw: int

    def __post_init__(self) -> None:
        names = (
            "point_w", "center_w", "radius", "frequency_hz", "R0",
            "normal_raw", "u_raw",
        )
        values = tuple(_key(name, getattr(self, name)) for name in names)
        if len(set(values)) != len(values):
            raise ValueError("orbit-factor keys must be unique")
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)

    def ordered(self) -> tuple[int, ...]:
        return (
            self.point_w, self.center_w, self.radius, self.frequency_hz,
            self.R0, self.normal_raw, self.u_raw,
        )

    @classmethod
    def conventional(cls, point_index: int) -> "OrbitFactorKeys":
        """Return deterministic symbols with globals shared across points."""

        if (isinstance(point_index, bool) or not isinstance(point_index, int)
                or point_index < 0):
            raise ValueError("point_index must be a non-negative integer")
        return cls(
            gtsam.symbol("p", point_index), gtsam.symbol("c", 0),
            gtsam.symbol("r", 0), gtsam.symbol("f", 0),
            gtsam.symbol("R", 0), gtsam.symbol("n", 0),
            gtsam.symbol("u", 0),
        )


@dataclass(frozen=True)
class OrbitFactorMeasurement:
    """One fixed-undistorted Eq. (4) pixel observation."""

    time_s: float
    t_ref_s: float
    intrinsic: np.ndarray
    observed_xy: np.ndarray

    def __post_init__(self) -> None:
        for name in ("time_s", "t_ref_s"):
            value = getattr(self, name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be a finite numeric scalar")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "intrinsic", _readonly(
            self.intrinsic, (3, 3),
        ))
        object.__setattr__(self, "observed_xy", _readonly(
            self.observed_xy, (2,),
        ))


def _factor_evaluation(
    values: gtsam.Values,
    keys: OrbitFactorKeys,
    measurement: OrbitFactorMeasurement,
):
    return evaluate_orbit_factor(
        point_w=values.atPoint3(keys.point_w),
        center_w=values.atPoint3(keys.center_w),
        radius=values.atDouble(keys.radius),
        frequency_hz=values.atDouble(keys.frequency_hz),
        R0=values.atRot3(keys.R0).matrix(),
        normal_raw=values.atPoint3(keys.normal_raw),
        u_raw=values.atPoint3(keys.u_raw),
        time_s=measurement.time_s,
        t_ref_s=measurement.t_ref_s,
        intrinsic=measurement.intrinsic,
        observed_xy=measurement.observed_xy,
    )


def make_orbit_custom_factor(
    keys: OrbitFactorKeys,
    measurement: OrbitFactorMeasurement,
) -> gtsam.CustomFactor:
    """Create one direct two-residual factor with isotropic ``sigma=1 px``."""

    assert_gtsam_version()
    if not isinstance(keys, OrbitFactorKeys):
        raise TypeError("keys must be OrbitFactorKeys")
    if not isinstance(measurement, OrbitFactorMeasurement):
        raise TypeError("measurement must be OrbitFactorMeasurement")

    def error_function(
        _factor: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: list[np.ndarray] | None,
    ) -> np.ndarray:
        evaluation = _factor_evaluation(values, keys, measurement)
        if jacobians is not None:
            ordered = evaluation.jacobians.ordered()
            if len(jacobians) != len(ordered):
                raise RuntimeError("GTSAM supplied an unexpected Jacobian block count")
            for index, block in enumerate(ordered):
                jacobians[index] = np.asarray(block, dtype=np.float64)
        return np.asarray(evaluation.residual_xy, dtype=np.float64)

    return gtsam.CustomFactor(
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
        list(keys.ordered()),
        error_function,
    )


def optimize_orbit_graph(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
) -> gtsam.Values:
    """Run the frozen default LM optimizer once and surface every failure.

    In particular, ``IndeterminantLinearSystemException`` is deliberately not
    caught.  The caller records it as a reference-path failure; this function
    never adds a prior, retries, or changes the solver.
    """

    assert_gtsam_version()
    return gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()


@dataclass(frozen=True)
class FactorObservation:
    """One deterministic orbit-factor input selected by frozen rule D-009."""

    track_id: int
    point_id: int
    point_index: int
    image_id: int
    window_index: int
    time_s: float
    observed_xy: np.ndarray
    initial_point_w: np.ndarray
    registered_window: bool
    retained_in_colmap_point_track: bool | None

    def __post_init__(self) -> None:
        for name in (
            "track_id", "point_id", "point_index", "image_id", "window_index",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.image_id == 0 or self.window_index == 0:
            raise ValueError("image_id and window_index must be positive")
        if (isinstance(self.time_s, bool)
                or not isinstance(self.time_s, (int, float, np.number))
                or not math.isfinite(float(self.time_s))):
            raise ValueError("time_s must be finite")
        if not isinstance(self.registered_window, bool):
            raise ValueError("registered_window must be bool")
        if (self.retained_in_colmap_point_track is not None
                and not isinstance(self.retained_in_colmap_point_track, bool)):
            raise ValueError("retained_in_colmap_point_track must be bool or None")
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "observed_xy", _readonly(
            self.observed_xy, (2,),
        ))
        object.__setattr__(self, "initial_point_w", _readonly(
            self.initial_point_w, (3,),
        ))


@dataclass(frozen=True)
class FactorObservationDiagnostics:
    total_factor_count: int
    registered_window_factor_count: int
    unregistered_window_factor_count: int
    retained_colmap_track_factor_count: int | None
    absent_colmap_track_factor_count: int | None
    mapped_track_count: int
    logical_window_count: int
    missing_track_window_pair_count: int
    split_track_count: int
    merged_track_count: int
    uninitialized_track_count: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                    or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
            object.__setattr__(self, name, int(value))
        if (self.registered_window_factor_count
                + self.unregistered_window_factor_count
                != self.total_factor_count):
            raise ValueError("registered/unregistered factor counts are inconsistent")
        if self.retained_colmap_track_factor_count is None:
            if self.absent_colmap_track_factor_count is not None:
                raise ValueError("COLMAP point-track counts must both be unknown")
        elif (self.absent_colmap_track_factor_count is None
              or self.retained_colmap_track_factor_count
              + self.absent_colmap_track_factor_count != self.total_factor_count):
            raise ValueError("COLMAP point-track factor counts are inconsistent")


@dataclass(frozen=True)
class FactorObservationSet:
    observations: tuple[FactorObservation, ...]
    diagnostics: FactorObservationDiagnostics
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if any(not isinstance(item, FactorObservation) for item in observations):
            raise ValueError("observations must contain FactorObservation values")
        order = tuple((item.track_id, item.window_index) for item in observations)
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ValueError("factor observations must be unique and deterministically sorted")
        if len(observations) != self.diagnostics.total_factor_count:
            raise ValueError("factor observations and diagnostics disagree")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _retained_feature_set(
    views: Sequence[IndexedView],
    model: ExtractedColmapModel,
    selected_point_observations: Mapping[int, Iterable[tuple[int, int]]] | None,
) -> set[tuple[int, int]] | None:
    if selected_point_observations is None:
        return None
    view_by_image = {view.image_id: view for view in views}
    point_to_track = {
        int(point_id): int(track_id)
        for track_id, point_id in model.mapping.unique_pairs
    }
    mapped_point_ids = set(point_to_track)
    supplied_mapped_ids: set[int] = set()
    retained: set[tuple[int, int]] = set()
    for point_id, features_input in selected_point_observations.items():
        if (isinstance(point_id, bool) or not isinstance(point_id, (int, np.integer))
                or int(point_id) < 0):
            raise ValueError("point-track observations contain an invalid point ID")
        point_id = int(point_id)
        expected_track = point_to_track.get(point_id)
        if expected_track is not None:
            supplied_mapped_ids.add(point_id)
        feature_count = 0
        for feature in features_input:
            feature_count += 1
            if (not isinstance(feature, tuple) or len(feature) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           for item in feature)):
                raise ValueError("point-track feature must be (image_id, feature_index)")
            image_id, feature_index = feature
            view = view_by_image.get(image_id)
            if (view is None or feature_index < 0
                    or feature_index >= view.track_ids.size):
                raise ValueError("point-track feature references an unknown image feature")
            if (expected_track is not None
                    and int(view.track_ids[feature_index]) != expected_track):
                raise ValueError("point-track feature does not match its mapped source track")
            if feature in retained:
                raise ValueError("point-track feature is duplicated")
            retained.add(feature)
        if feature_count == 0:
            raise ValueError("point-track observations contain an empty point track")
    missing = mapped_point_ids - supplied_mapped_ids
    if missing:
        raise ValueError(
            "point-track observations are missing mapped point IDs: "
            + ",".join(str(value) for value in sorted(missing)),
        )
    return retained


def build_factor_observations(
    views: Sequence[IndexedView],
    model: ExtractedColmapModel,
    *,
    selected_point_observations: Mapping[
        int, Iterable[tuple[int, int]]
    ] | None = None,
) -> FactorObservationSet:
    """Select all original Eq. (4) observations for bijectively mapped tracks.

    Unregistered logical windows and features absent from the selected COLMAP
    point track remain factors.  Empty pairs are not invented, and tracks
    without a unique 3-D initialization are excluded as whole tracks.

    The extracted model preserves every selected-model ``points3D.txt`` track,
    so the retained/absent diagnostic is available without reparsing private
    files. ``selected_point_observations`` exists only as an explicit audit
    override; it never changes factor membership.
    """

    if not isinstance(model, ExtractedColmapModel):
        raise TypeError("model must be ExtractedColmapModel")
    views_tuple = tuple(views)
    if any(not isinstance(view, IndexedView) for view in views_tuple):
        raise TypeError("views must contain IndexedView values")
    if not views_tuple:
        raise ValueError("at least one logical view is required")
    if (len({view.image_id for view in views_tuple}) != len(views_tuple)
            or len({view.window_index for view in views_tuple}) != len(views_tuple)):
        raise ValueError("view image and window IDs must be unique")
    if sorted(view.window_index for view in views_tuple) != list(
        range(1, len(views_tuple) + 1)
    ):
        raise ValueError("views must contain the complete contiguous window sequence 1..K")

    expected_unregistered = tuple(sorted(
        view.window_index for view in views_tuple
        if view.image_id not in set(int(value) for value in model.trajectory.view_ids)
    ))
    if expected_unregistered != model.unregistered_window_indices:
        raise ValueError("views and model disagree on unregistered windows")

    retained_features = _retained_feature_set(
        views_tuple, model,
        (model.point_observations if selected_point_observations is None
         else selected_point_observations),
    )
    mapped: dict[int, tuple[int, int, np.ndarray]] = {}
    for point_index, (track_id, point_id, point_w) in enumerate(zip(
        model.mapped_track_ids, model.mapped_point_ids, model.mapped_points_w,
    )):
        mapped[int(track_id)] = (int(point_id), point_index, point_w)
    registered_images = set(int(value) for value in model.trajectory.view_ids)
    records: list[FactorObservation] = []
    retained_count = 0
    for view in views_tuple:
        for feature_index, (track_id_raw, observed) in enumerate(zip(
            view.track_ids, view.keypoints_xy,
        )):
            track_id = int(track_id_raw)
            if track_id not in mapped:
                continue
            point_id, point_index, point_w = mapped[track_id]
            retained = (
                None if retained_features is None
                else (view.image_id, feature_index) in retained_features
            )
            if retained:
                retained_count += 1
            records.append(FactorObservation(
                track_id=track_id,
                point_id=point_id,
                point_index=point_index,
                image_id=view.image_id,
                window_index=view.window_index,
                time_s=view.time_ms / 1000.0,
                observed_xy=observed,
                initial_point_w=point_w,
                registered_window=view.image_id in registered_images,
                retained_in_colmap_point_track=retained,
            ))
    records.sort(key=lambda item: (item.track_id, item.window_index))
    registered_count = sum(item.registered_window for item in records)
    total = len(records)
    split = set(model.mapping.split_track_ids)
    uninitialized = set(model.mapping.uninitialized_track_ids)
    merged_tracks = set(model.mapping.merged_track_ids)
    diagnostics = FactorObservationDiagnostics(
        total_factor_count=total,
        registered_window_factor_count=registered_count,
        unregistered_window_factor_count=total - registered_count,
        retained_colmap_track_factor_count=(
            None if retained_features is None else retained_count
        ),
        absent_colmap_track_factor_count=(
            None if retained_features is None else total - retained_count
        ),
        mapped_track_count=len(mapped),
        logical_window_count=len(views_tuple),
        missing_track_window_pair_count=len(mapped) * len(views_tuple) - total,
        split_track_count=len(split),
        merged_track_count=len(merged_tracks),
        uninitialized_track_count=len(uninitialized),
    )
    return FactorObservationSet(
        tuple(records), diagnostics,
        {
            "membership_rule": "all_original_Eq4_observations_for_bijective_tracks",
            "factor_order": "track_id_then_window_index",
            "residual_or_depth_pruning": False,
            "colmap_point_track_diagnostic_available": retained_features is not None,
        },
    )


__all__ = [
    "FactorObservation", "FactorObservationDiagnostics", "FactorObservationSet",
    "GTSAM_DISTRIBUTION", "GTSAM_VERSION", "OrbitFactorKeys",
    "OrbitFactorMeasurement", "assert_gtsam_version",
    "build_factor_observations", "make_orbit_custom_factor",
    "optimize_orbit_graph",
]
