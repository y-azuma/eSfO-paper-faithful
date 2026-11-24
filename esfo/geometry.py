"""Paper-faithful Gate-B orbit-geometry initialization.

The paper reports the ordinary ``z = ax + by + c`` plane regression and the
implicit circle fit, but leaves several executable conventions unstated.  This
module implements only the result-independent conventions frozen in
``docs/geometry_conventions.md``.  It deliberately fails on degeneracies
instead of switching to a geometrically nicer alternative fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np


_EPS = np.finfo(np.float64).eps
_PARALLEL_TOLERANCE = 32.0 * _EPS
_ROTATION_TOLERANCE = 256.0 * _EPS


def _readonly(value: Any, *, dtype: np.dtype = np.dtype(np.float64)) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
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


def _finite_scalar(name: str, value: Any) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(value)


def _finite_array(name: str, value: Any, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != len(shape):
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    if any(expected is not None and actual != expected
           for actual, expected in zip(array.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}")
    result = _readonly(array)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _unit_vector(name: str, value: Any) -> np.ndarray:
    vector = _finite_array(name, value, (3,))
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return _readonly(vector / norm)


def _validate_rotation(name: str, value: Any) -> np.ndarray:
    rotation = _finite_array(name, value, (3, 3))
    orthogonality_error = float(np.linalg.norm(
        rotation.T @ rotation - np.eye(3), ord=np.inf,
    ))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if (orthogonality_error > _ROTATION_TOLERANCE
            or determinant_error > _ROTATION_TOLERANCE):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return rotation


def _condition_number(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[-1] == 0.0:
        return math.inf
    return float(singular[0] / singular[-1])


class GeometryInitializationError(ValueError):
    """Explicit reference-path failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class RegisteredTrajectory:
    """Temporally ordered COLMAP world-to-camera poses."""

    view_ids: np.ndarray
    time_s: np.ndarray
    R_cw: np.ndarray
    t_cw: np.ndarray

    def __post_init__(self) -> None:
        view_input = np.asarray(self.view_ids)
        if view_input.ndim != 1 or view_input.dtype.kind not in "iu":
            raise ValueError("view_ids must be a one-dimensional integer array")
        view_ids = _readonly(view_input, dtype=np.dtype(np.int64))
        time_s = _finite_array("time_s", self.time_s, (None,))
        rotations = _finite_array("R_cw", self.R_cw, (None, 3, 3))
        translations = _finite_array("t_cw", self.t_cw, (None, 3))
        count = time_s.size
        if count < 3:
            raise ValueError("at least three registered views are required")
        if not (view_ids.size == count == rotations.shape[0] == translations.shape[0]):
            raise ValueError("trajectory arrays have inconsistent lengths")
        if np.any(view_ids < 0) or np.unique(view_ids).size != count:
            raise ValueError("view_ids must be unique and non-negative")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("registered views must have strictly increasing times")
        validated = np.empty_like(rotations)
        for index, rotation in enumerate(rotations):
            validated[index] = _validate_rotation(f"R_cw[{index}]", rotation)
        object.__setattr__(self, "view_ids", view_ids)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "R_cw", _readonly(validated))
        object.__setattr__(self, "t_cw", translations)

    @property
    def centers_w(self) -> np.ndarray:
        centers = -np.einsum("kji,kj->ki", self.R_cw, self.t_cw)
        return _readonly(centers)


@dataclass(frozen=True)
class PlaneFit:
    centroid_w: np.ndarray
    coefficients_abc: np.ndarray
    normal_w: np.ndarray
    design_rank: int
    condition_number: float
    temporal_score: float
    temporal_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centroid_w", _finite_array(
            "centroid_w", self.centroid_w, (3,),
        ))
        object.__setattr__(self, "coefficients_abc", _finite_array(
            "coefficients_abc", self.coefficients_abc, (3,),
        ))
        object.__setattr__(self, "normal_w", _unit_vector(
            "normal_w", self.normal_w,
        ))
        if (isinstance(self.design_rank, bool)
                or not isinstance(self.design_rank, (int, np.integer))
                or int(self.design_rank) != 3):
            raise ValueError("plane design rank must be exactly three")
        condition = _finite_scalar("condition_number", self.condition_number)
        score = _finite_scalar("temporal_score", self.temporal_score)
        tolerance = _finite_scalar("temporal_tolerance", self.temporal_tolerance)
        if condition < 1.0 or tolerance < 0.0 or score <= tolerance:
            raise ValueError("plane diagnostics are inconsistent")
        object.__setattr__(self, "design_rank", int(self.design_rank))
        object.__setattr__(self, "condition_number", condition)
        object.__setattr__(self, "temporal_score", score)
        object.__setattr__(self, "temporal_tolerance", tolerance)


@dataclass(frozen=True)
class AlignmentResult:
    rotation: np.ndarray
    axis: np.ndarray
    angle_rad: float
    branch: Literal["general", "parallel", "antiparallel"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _validate_rotation(
            "alignment rotation", self.rotation,
        ))
        axis = _finite_array("alignment axis", self.axis, (3,))
        if self.branch != "parallel":
            axis = _unit_vector("alignment axis", axis)
        object.__setattr__(self, "axis", axis)
        if self.branch not in {"general", "parallel", "antiparallel"}:
            raise ValueError("invalid alignment branch")
        if not math.isfinite(float(self.angle_rad)):
            raise ValueError("alignment angle must be finite")


@dataclass(frozen=True)
class OrbitGeometryInitialization:
    center_w: np.ndarray
    radius: float
    normal_w: np.ndarray
    u_w: np.ndarray
    v_w: np.ndarray
    basis_w: np.ndarray
    plane_to_xy: np.ndarray
    first_view_id: int
    first_view_index: int
    t_ref_s: float
    plane_diagnostics: PlaneFit
    circle_rank: int
    circle_condition_number: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        center = _finite_array("center_w", self.center_w, (3,))
        normal = _unit_vector("normal_w", self.normal_w)
        u_w = _unit_vector("u_w", self.u_w)
        v_w = _unit_vector("v_w", self.v_w)
        basis = _validate_rotation("basis_w", self.basis_w)
        alignment = _validate_rotation("plane_to_xy", self.plane_to_xy)
        radius = float(self.radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if abs(float(normal @ u_w)) > _ROTATION_TOLERANCE:
            raise ValueError("u_w must lie in the orbit plane")
        if not np.allclose(v_w, np.cross(normal, u_w), rtol=0.0,
                           atol=_ROTATION_TOLERANCE):
            raise ValueError("v_w must equal normal_w cross u_w")
        if not np.allclose(basis, np.column_stack((u_w, v_w, normal)),
                           rtol=0.0, atol=_ROTATION_TOLERANCE):
            raise ValueError("basis_w columns must be [u_w,v_w,normal_w]")
        if not math.isfinite(float(self.t_ref_s)):
            raise ValueError("t_ref_s must be finite")
        if not np.allclose(alignment @ normal, [0.0, 0.0, 1.0], rtol=0.0,
                           atol=_ROTATION_TOLERANCE):
            raise ValueError("plane_to_xy must map normal_w to z")
        if not np.allclose(self.plane_diagnostics.normal_w, normal, rtol=0.0,
                           atol=_ROTATION_TOLERANCE):
            raise ValueError("plane diagnostics and normal_w disagree")
        for name, value in (("first_view_id", self.first_view_id),
                            ("first_view_index", self.first_view_index),
                            ("circle_rank", self.circle_rank)):
            if (isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))):
                raise ValueError(f"{name} must be an integer")
        if int(self.first_view_id) < 0 or int(self.first_view_index) != 0:
            raise ValueError("reference phase origin must be the first registered view")
        if int(self.circle_rank) != 3:
            raise ValueError("circle design rank must be exactly three")
        circle_condition = _finite_scalar(
            "circle_condition_number", self.circle_condition_number,
        )
        if circle_condition < 1.0:
            raise ValueError("circle condition number must be at least one")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "center_w", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "normal_w", normal)
        object.__setattr__(self, "u_w", u_w)
        object.__setattr__(self, "v_w", v_w)
        object.__setattr__(self, "basis_w", basis)
        object.__setattr__(self, "plane_to_xy", alignment)
        object.__setattr__(self, "first_view_id", int(self.first_view_id))
        object.__setattr__(self, "first_view_index", int(self.first_view_index))
        object.__setattr__(self, "circle_rank", int(self.circle_rank))
        object.__setattr__(self, "circle_condition_number", circle_condition)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


@dataclass(frozen=True)
class R0Initialization:
    rotation: np.ndarray
    candidate_rotations: np.ndarray
    mean_matrix: np.ndarray
    singular_values: np.ndarray
    determinant_correction: float
    rank_tolerance: float
    reflection_tie_tolerance: float

    def __post_init__(self) -> None:
        rotation = _validate_rotation("R0", self.rotation)
        candidates = _finite_array(
            "candidate_rotations", self.candidate_rotations, (None, 3, 3),
        )
        if candidates.shape[0] == 0:
            raise ValueError("at least one R0 candidate is required")
        for index, candidate in enumerate(candidates):
            _validate_rotation(f"candidate_rotations[{index}]", candidate)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "candidate_rotations", candidates)
        object.__setattr__(self, "mean_matrix", _finite_array(
            "mean_matrix", self.mean_matrix, (3, 3),
        ))
        singular = _finite_array(
            "singular_values", self.singular_values, (3,),
        )
        if np.any(singular < 0.0) or np.any(np.diff(singular) > 0.0):
            raise ValueError("singular_values must be non-negative and descending")
        correction = _finite_scalar(
            "determinant_correction", self.determinant_correction,
        )
        if abs(abs(correction) - 1.0) > _ROTATION_TOLERANCE:
            raise ValueError("determinant_correction must be plus or minus one")
        rank_tolerance = _finite_scalar("rank_tolerance", self.rank_tolerance)
        tie_tolerance = _finite_scalar(
            "reflection_tie_tolerance", self.reflection_tie_tolerance,
        )
        if rank_tolerance < 0.0 or tie_tolerance < 0.0:
            raise ValueError("R0 tolerances must be non-negative")
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "determinant_correction",
                           -1.0 if correction < 0.0 else 1.0)
        object.__setattr__(self, "rank_tolerance", rank_tolerance)
        object.__setattr__(self, "reflection_tie_tolerance", tie_tolerance)


@dataclass(frozen=True)
class AnalyticOrbitPoses:
    time_s: np.ndarray
    phase_rad: np.ndarray
    centers_w: np.ndarray
    R_cw: np.ndarray
    t_cw: np.ndarray

    def __post_init__(self) -> None:
        time_s = _finite_array("time_s", self.time_s, (None,))
        count = time_s.size
        phase = _finite_array("phase_rad", self.phase_rad, (count,))
        centers = _finite_array("centers_w", self.centers_w, (count, 3))
        rotations = _finite_array("R_cw", self.R_cw, (count, 3, 3))
        translations = _finite_array("t_cw", self.t_cw, (count, 3))
        for index, rotation in enumerate(rotations):
            _validate_rotation(f"R_cw[{index}]", rotation)
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(self, "centers_w", centers)
        object.__setattr__(self, "R_cw", rotations)
        object.__setattr__(self, "t_cw", translations)


def fit_orbit_plane(centers_w: Any, time_s: Any) -> PlaneFit:
    """Fit the paper's ordinary regression and fix its temporal sign."""

    centers = _finite_array("centers_w", centers_w, (None, 3))
    times = _finite_array("time_s", time_s, (centers.shape[0],))
    if centers.shape[0] < 3 or np.any(np.diff(times) <= 0.0):
        raise GeometryInitializationError(
            "plane_rank_deficient", "three time-ordered centres are required",
        )
    centroid = np.mean(centers, axis=0, dtype=np.float64)
    centered = centers - centroid
    design = np.column_stack((centered[:, 0], centered[:, 1],
                              np.ones(centers.shape[0], dtype=np.float64)))
    rcond = _EPS * max(design.shape)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, centered[:, 2], rcond=rcond,
    )
    if rank != 3 or not np.all(np.isfinite(coefficients)):
        raise GeometryInitializationError(
            "plane_rank_deficient", "Eq. (6) ordinary regression has rank below 3",
        )
    normal = np.asarray([coefficients[0], coefficients[1], -1.0])
    normal /= np.linalg.norm(normal)
    projected = centered - np.outer(centered @ normal, normal)
    cross = np.cross(projected[:-1], projected[1:])
    score = float(np.sum(cross @ normal, dtype=np.float64))
    scale = float(np.sum(
        np.linalg.norm(projected[:-1], axis=1)
        * np.linalg.norm(projected[1:], axis=1),
        dtype=np.float64,
    ))
    tolerance = 64.0 * _EPS * scale
    if abs(score) <= tolerance:
        raise GeometryInitializationError(
            "temporal_sign_unobservable",
            "temporal cross-product score is numerically zero",
        )
    if score < 0.0:
        normal = -normal
        score = -score
    return PlaneFit(
        _readonly(centroid), _readonly(coefficients), _readonly(normal),
        int(rank), _condition_number(design), score, tolerance,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    generator = _skew(axis)
    return (np.eye(3) + math.sin(angle) * generator
            + (1.0 - math.cos(angle)) * (generator @ generator))


def active_align_to_z(normal_w: Any) -> AlignmentResult:
    """Return the frozen active Rodrigues rotation mapping ``normal_w`` to z."""

    normal = _unit_vector("normal_w", normal_w)
    z_axis = np.asarray([0.0, 0.0, 1.0])
    cross = np.cross(normal, z_axis)
    magnitude = float(np.linalg.norm(cross))
    dot = float(normal @ z_axis)
    if magnitude <= _PARALLEL_TOLERANCE:
        if dot >= 0.0:
            result = AlignmentResult(
                _readonly(np.eye(3)), _readonly(np.zeros(3)), 0.0, "parallel",
            )
        else:
            axes = np.eye(3)
            selected = axes[int(np.argmin(np.abs(axes @ normal)))]
            axis = selected - float(selected @ normal) * normal
            axis /= np.linalg.norm(axis)
            result = AlignmentResult(
                _readonly(_axis_angle(axis, math.pi)), _readonly(axis),
                math.pi, "antiparallel",
            )
    else:
        axis = cross / magnitude
        angle = math.atan2(magnitude, dot)
        result = AlignmentResult(
            _readonly(_axis_angle(axis, angle)), _readonly(axis),
            angle, "general",
        )
    if not np.allclose(result.rotation @ normal, z_axis, rtol=0.0,
                       atol=_ROTATION_TOLERANCE):
        raise GeometryInitializationError(
            "invalid_rotation", "Rodrigues alignment does not map n to z",
        )
    return result


def _checked_circle_radius_squared(
    coefficients: np.ndarray, points_2d: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Convert implicit-circle coefficients with the frozen roundoff rule."""

    center_2d = 0.5 * coefficients[:2]
    radius_squared = float(coefficients[2] + center_2d @ center_2d)
    scale = max(
        float(np.max(np.sum(points_2d * points_2d, axis=1))),
        float(center_2d @ center_2d), abs(float(coefficients[2])),
    )
    tolerance = 64.0 * _EPS * scale
    if radius_squared < -tolerance:
        raise GeometryInitializationError(
            "negative_radius_squared", "implicit circle fit produced negative r^2",
        )
    return center_2d, max(radius_squared, 0.0), tolerance


def fit_orbit_geometry(trajectory: RegisteredTrajectory) -> OrbitGeometryInitialization:
    """Apply the frozen Eq. (6)--(8) initialization to registered poses."""

    centers = trajectory.centers_w
    plane = fit_orbit_plane(centers, trajectory.time_s)
    alignment = active_align_to_z(plane.normal_w)
    centered = centers - plane.centroid_w
    points_2d = (alignment.rotation @ centered.T).T[:, :2]
    design = np.column_stack((
        points_2d[:, 0], points_2d[:, 1],
        np.ones(points_2d.shape[0], dtype=np.float64),
    ))
    rhs = np.sum(points_2d * points_2d, axis=1)
    rcond = _EPS * max(design.shape)
    coefficients, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=rcond)
    if rank != 3 or not np.all(np.isfinite(coefficients)):
        raise GeometryInitializationError(
            "circle_rank_deficient", "implicit circle fit has rank below 3",
        )
    center_2d, radius_squared, tolerance = _checked_circle_radius_squared(
        coefficients, points_2d,
    )
    radius = math.sqrt(radius_squared)
    if radius == 0.0:
        raise GeometryInitializationError(
            "radial_direction_unobservable", "fitted circle has zero radius",
        )
    center_aligned = np.asarray([center_2d[0], center_2d[1], 0.0])
    center_w = alignment.rotation.T @ center_aligned + plane.centroid_w
    radial = centers[0] - center_w
    radial -= float(radial @ plane.normal_w) * plane.normal_w
    radial_norm = float(np.linalg.norm(radial))
    radial_tolerance = 64.0 * _EPS * max(
        radius, float(np.max(np.linalg.norm(centers - center_w, axis=1))),
    )
    if radial_norm <= radial_tolerance:
        raise GeometryInitializationError(
            "radial_direction_unobservable",
            "earliest registered centre has no stable in-plane direction",
        )
    u_w = radial / radial_norm
    v_w = np.cross(plane.normal_w, u_w)
    v_w /= np.linalg.norm(v_w)
    basis = np.column_stack((u_w, v_w, plane.normal_w))
    metadata = {
        "plane_fit": "ordinary_z_on_x_y_intercept",
        "plane_lstsq_rcond": rcond,
        "temporal_sign_tolerance": "64*eps*sum(norm(q_k)*norm(q_k+1))",
        "alignment_parallel_tolerance": _PARALLEL_TOLERANCE,
        "alignment_branch": alignment.branch,
        "rotation_validation_tolerance": _ROTATION_TOLERANCE,
        "circle_fit": "centered_implicit_least_squares",
        "circle_lstsq_rcond": rcond,
        "radius_squared_tolerance": (
            "64*eps*max(max(norm(y)^2),norm(o)^2,abs(theta_2))"
        ),
        "radius_squared_tolerance_value": tolerance,
        "radial_tolerance": "64*eps*max(radius,max(norm(C-c)))",
        "radial_tolerance_value": radial_tolerance,
        "phase_origin": "earliest_registered_view",
    }
    return OrbitGeometryInitialization(
        _readonly(center_w), radius, _readonly(plane.normal_w),
        _readonly(u_w), _readonly(v_w), _readonly(basis),
        _readonly(alignment.rotation), int(trajectory.view_ids[0]), 0,
        float(trajectory.time_s[0]), plane, int(rank),
        _condition_number(design), metadata,
    )


def orbit_phase(time_s: Any, frequency_hz: float, t_ref_s: float) -> np.ndarray:
    times = _finite_array("time_s", time_s, (None,))
    frequency = _finite_scalar("frequency_hz", frequency_hz)
    reference = _finite_scalar("t_ref_s", t_ref_s)
    return _readonly(2.0 * math.pi * frequency * (times - reference))


def _rotations_z(angles: np.ndarray) -> np.ndarray:
    result = np.zeros((angles.size, 3, 3), dtype=np.float64)
    cosine, sine = np.cos(angles), np.sin(angles)
    result[:, 0, 0] = cosine
    result[:, 0, 1] = -sine
    result[:, 1, 0] = sine
    result[:, 1, 1] = cosine
    result[:, 2, 2] = 1.0
    return result


def orbit_centers(
    geometry: OrbitGeometryInitialization,
    frequency_hz: float,
    time_s: Any,
) -> np.ndarray:
    phase = orbit_phase(time_s, frequency_hz, geometry.t_ref_s)
    orbit_coordinates = np.column_stack((
        geometry.radius * np.cos(phase),
        geometry.radius * np.sin(phase),
        np.zeros(phase.size, dtype=np.float64),
    ))
    return _readonly(geometry.center_w + orbit_coordinates @ geometry.basis_w.T)


def orbit_poses(
    geometry: OrbitGeometryInitialization,
    frequency_hz: float,
    R0: Any,
    time_s: Any,
) -> AnalyticOrbitPoses:
    times = _finite_array("time_s", time_s, (None,))
    r0 = _validate_rotation("R0", R0)
    phase = orbit_phase(times, frequency_hz, geometry.t_ref_s)
    centers = orbit_centers(geometry, frequency_hz, times)
    negative_phase = _rotations_z(-phase)
    rotations = np.einsum(
        "ij,kjl,lm->kim", r0, negative_phase, geometry.basis_w.T,
    )
    translations = -np.einsum("kij,kj->ki", rotations, centers)
    return AnalyticOrbitPoses(
        times, phase, centers, _readonly(rotations), _readonly(translations),
    )


def initialize_r0(
    trajectory: RegisteredTrajectory,
    geometry: OrbitGeometryInitialization,
    frequency_hz: float,
) -> R0Initialization:
    """Remove the orbit term and project the uniform chordal mean to SO(3)."""

    frequency = _finite_scalar("frequency_hz", frequency_hz)
    if frequency <= 0.0:
        raise ValueError("reference R0 initialization requires positive frequency")
    phase = orbit_phase(trajectory.time_s, frequency, geometry.t_ref_s)
    # The temporal-normal sign in ``fit_orbit_plane`` is inferred from
    # consecutive sampled chords. Once a positive frequency estimate is
    # available, a phase gap of pi or more makes that direction Nyquist-
    # ambiguous: the same sampled centres admit the opposite rotation sense.
    # Fail closed instead of letting the chordal R0 mean hide an aliased sign.
    if np.any(np.diff(phase) >= math.pi):
        raise GeometryInitializationError(
            "temporal_sign_aliased",
            "a registered-view phase gap is at least pi radians",
        )
    positive_phase = _rotations_z(phase)
    candidates = np.einsum(
        "kij,jl,klm->kim", trajectory.R_cw, geometry.basis_w, positive_phase,
    )
    mean = np.mean(candidates, axis=0, dtype=np.float64)
    left, singular, right_t = np.linalg.svd(mean)
    rank_tolerance = _EPS * max(mean.shape) * float(singular[0])
    if singular[-1] <= rank_tolerance:
        raise GeometryInitializationError(
            "r0_mean_degenerate", "chordal rotation mean is rank deficient",
        )
    raw_correction = float(np.linalg.det(left @ right_t))
    correction = -1.0 if raw_correction < 0.0 else 1.0
    # When the unconstrained polar factor is a reflection, SO(3) projection
    # flips the singular direction with the smallest value.  A tie between
    # the two smallest values makes that direction (and therefore R0)
    # non-unique even though the mean matrix is full rank.
    if (correction < 0.0
            and singular[-2] - singular[-1] <= rank_tolerance):
        raise GeometryInitializationError(
            "r0_mean_degenerate",
            "reflection correction has no unique smallest singular direction",
        )
    diagonal = np.diag([1.0, 1.0, correction])
    rotation = left @ diagonal @ right_t
    return R0Initialization(
        _readonly(rotation), _readonly(candidates), _readonly(mean),
        _readonly(singular), correction, rank_tolerance, rank_tolerance,
    )


__all__ = [
    "AlignmentResult", "AnalyticOrbitPoses", "GeometryInitializationError",
    "OrbitGeometryInitialization", "PlaneFit", "R0Initialization",
    "RegisteredTrajectory", "active_align_to_z", "fit_orbit_geometry",
    "fit_orbit_plane", "initialize_r0", "orbit_centers", "orbit_phase",
    "orbit_poses",
]
