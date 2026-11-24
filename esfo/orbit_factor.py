"""Pure eSfO orbit reprojection factor and analytic Jacobians.

The paper's orbit factor is evaluated without depending on GTSAM so that its
geometry, residual sign, and every local derivative can be audited directly.
The GTSAM adapter is intentionally a thin wrapper around this module.

The seven variable blocks are ordered as ``point, c, r, f, R0, n, u``.
Vectors use additive coordinates and ``R0`` uses a right-local perturbation,
``R0 * Exp(delta)``.  The residual is predicted minus observed pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_EPS = np.finfo(np.float64).eps
_ROTATION_TOLERANCE = 256.0 * _EPS
_UNIT_DEGENERACY_SCALE = 64.0 * _EPS
_CHEIRALITY_PENALTY_FOCAL_MULTIPLIER = 2.0


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape:
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    result = _readonly(array)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _finite_scalar(name: str, value: Any) -> float:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not math.isfinite(float(value))):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(value)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _validate_rotation(value: Any) -> np.ndarray:
    rotation = _finite_array("R0", value, (3, 3))
    orthogonality_error = float(np.linalg.norm(
        rotation.T @ rotation - np.eye(3), ord=np.inf,
    ))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if (orthogonality_error > _ROTATION_TOLERANCE
            or determinant_error > _ROTATION_TOLERANCE):
        raise ValueError("R0 must be a proper rotation matrix")
    return rotation


def _validate_intrinsic(value: Any) -> np.ndarray:
    intrinsic = _finite_array("intrinsic", value, (3, 3))
    scale = max(1.0, float(np.max(np.abs(intrinsic))))
    tolerance = _ROTATION_TOLERANCE * scale
    expected_zero = np.asarray([
        intrinsic[0, 1], intrinsic[1, 0], intrinsic[2, 0], intrinsic[2, 1],
    ])
    if (intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0
            or np.any(np.abs(expected_zero) > tolerance)
            or abs(float(intrinsic[2, 2]) - 1.0) > tolerance):
        raise ValueError("intrinsic must be a fixed zero-skew PINHOLE matrix")
    return intrinsic


class OrbitFactorError(ValueError):
    """Explicit factor-evaluation failure with a stable diagnostic code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class OrbitFactorJacobians:
    """Residual Jacobians in the frozen seven-block key order."""

    point_w: np.ndarray
    center_w: np.ndarray
    radius: np.ndarray
    frequency_hz: np.ndarray
    R0: np.ndarray
    normal_raw: np.ndarray
    u_raw: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("point_w", (2, 3)), ("center_w", (2, 3)),
            ("radius", (2, 1)), ("frequency_hz", (2, 1)),
            ("R0", (2, 3)), ("normal_raw", (2, 3)),
            ("u_raw", (2, 3)),
        ):
            object.__setattr__(self, name, _finite_array(
                name, getattr(self, name), shape,
            ))

    def ordered(self) -> tuple[np.ndarray, ...]:
        return (
            self.point_w, self.center_w, self.radius, self.frequency_hz,
            self.R0, self.normal_raw, self.u_raw,
        )


@dataclass(frozen=True)
class OrbitFactorEvaluation:
    residual_xy: np.ndarray
    predicted_xy: np.ndarray
    camera_point: np.ndarray
    normal_w: np.ndarray
    u_w: np.ndarray
    v_w: np.ndarray
    basis_w: np.ndarray
    phase_rad: float
    cheirality_penalty: bool
    jacobians: OrbitFactorJacobians

    def __post_init__(self) -> None:
        for name, shape in (
            ("residual_xy", (2,)), ("predicted_xy", (2,)),
            ("camera_point", (3,)), ("normal_w", (3,)),
            ("u_w", (3,)), ("v_w", (3,)), ("basis_w", (3, 3)),
        ):
            object.__setattr__(self, name, _finite_array(
                name, getattr(self, name), shape,
            ))
        phase = _finite_scalar("phase_rad", self.phase_rad)
        object.__setattr__(self, "phase_rad", phase)
        if not isinstance(self.cheirality_penalty, bool):
            raise ValueError("cheirality_penalty must be bool")
        if not isinstance(self.jacobians, OrbitFactorJacobians):
            raise ValueError("jacobians must be OrbitFactorJacobians")


def _zero_jacobians() -> OrbitFactorJacobians:
    """Return frozen zero blocks for the GTSAM cheirality penalty."""

    return OrbitFactorJacobians(
        np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)),
        np.zeros((2, 1)), np.zeros((2, 3)), np.zeros((2, 3)),
        np.zeros((2, 3)),
    )


def _normalized_orbit_basis(
    normal_raw: np.ndarray,
    u_raw: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Return ``n,u,v,B`` and their raw-vector differential blocks."""

    normal_norm = float(np.linalg.norm(normal_raw))
    if normal_norm == 0.0:
        raise OrbitFactorError("normal_degenerate", "normal vector is zero")
    normal = normal_raw / normal_norm
    dnormal_dn = (
        np.eye(3) - np.outer(normal, normal)
    ) / normal_norm

    projection = np.eye(3) - np.outer(normal, normal)
    u_projected = projection @ u_raw
    u_projected_norm = float(np.linalg.norm(u_projected))
    u_scale = max(1.0, float(np.linalg.norm(u_raw)))
    if u_projected_norm <= _UNIT_DEGENERACY_SCALE * u_scale:
        raise OrbitFactorError(
            "u_degenerate", "u is zero or numerically collinear with n",
        )
    u_w = u_projected / u_projected_norm
    du_dprojected = (
        np.eye(3) - np.outer(u_w, u_w)
    ) / u_projected_norm
    normal_dot_u = float(normal @ u_raw)
    dprojected_dn = -(
        normal_dot_u * np.eye(3) + np.outer(normal, u_raw)
    ) @ dnormal_dn
    du_dn = du_dprojected @ dprojected_dn
    du_du = du_dprojected @ projection

    v_w = np.cross(normal, u_w)
    dv_dn = -_skew(u_w) @ dnormal_dn + _skew(normal) @ du_dn
    dv_du = _skew(normal) @ du_du
    basis = np.column_stack((u_w, v_w, normal))
    return (
        normal, u_w, v_w, basis,
        dnormal_dn, du_dn, du_du, dv_dn, dv_du,
    )


def evaluate_orbit_factor(
    *,
    point_w: Any,
    center_w: Any,
    radius: float,
    frequency_hz: float,
    R0: Any,
    normal_raw: Any,
    u_raw: Any,
    time_s: float,
    t_ref_s: float,
    intrinsic: Any,
    observed_xy: Any,
) -> OrbitFactorEvaluation:
    """Evaluate one paper-faithful two-pixel orbit reprojection factor.

    ``radius`` and ``frequency_hz`` are deliberately unconstrained real
    scalars.  In particular, negative optimized values are evaluated and
    reported rather than silently transformed to positive values.
    """

    point = _finite_array("point_w", point_w, (3,))
    center = _finite_array("center_w", center_w, (3,))
    radius_value = _finite_scalar("radius", radius)
    frequency = _finite_scalar("frequency_hz", frequency_hz)
    rotation0 = _validate_rotation(R0)
    normal_input = _finite_array("normal_raw", normal_raw, (3,))
    u_input = _finite_array("u_raw", u_raw, (3,))
    time = _finite_scalar("time_s", time_s)
    reference = _finite_scalar("t_ref_s", t_ref_s)
    K = _validate_intrinsic(intrinsic)
    observed = _finite_array("observed_xy", observed_xy, (2,))

    (
        normal, u_w, v_w, basis,
        dnormal_dn, du_dn, du_du, dv_dn, dv_du,
    ) = _normalized_orbit_basis(normal_input, u_input)

    tau = time - reference
    phase = 2.0 * math.pi * frequency * tau
    cosine, sine = math.cos(phase), math.sin(phase)
    phase_coordinates = np.asarray([cosine, sine, 0.0])
    phase_derivative = np.asarray([-sine, cosine, 0.0])
    negative_phase_rotation = np.asarray([
        [cosine, sine, 0.0],
        [-sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])

    relative = point - center
    orbit_coordinates = basis.T @ relative - radius_value * phase_coordinates
    rotated_orbit_coordinates = negative_phase_rotation @ orbit_coordinates
    camera_point = rotation0 @ rotated_orbit_coordinates
    if not np.all(np.isfinite(camera_point)):
        raise OrbitFactorError(
            "projection_nonfinite", "camera-space point is non-finite",
        )
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    depth = float(camera_point[2])
    if depth <= 0.0:
        # GTSAM's GenericProjectionFactor with throwCheirality=false uses a
        # constant 2*fx residual and zero Jacobians.  Returning the same
        # result lets LM reject an invalid trial step instead of aborting the
        # complete optimization.  The factor remains in the graph.
        residual = np.full(
            2, _CHEIRALITY_PENALTY_FOCAL_MULTIPLIER * fx,
            dtype=np.float64,
        )
        predicted = observed + residual
        return OrbitFactorEvaluation(
            _readonly(residual), _readonly(predicted), _readonly(camera_point),
            _readonly(normal), _readonly(u_w), _readonly(v_w), _readonly(basis),
            phase, True, _zero_jacobians(),
        )

    predicted = np.asarray([
        fx * float(camera_point[0]) / depth + cx,
        fy * float(camera_point[1]) / depth + cy,
    ])
    residual = predicted - observed
    projection_jacobian = np.asarray([
        [fx / depth, 0.0, -fx * camera_point[0] / (depth * depth)],
        [0.0, fy / depth, -fy * camera_point[1] / (depth * depth)],
    ])

    rotation_orbit_to_camera = rotation0 @ negative_phase_rotation
    dq_dpoint = rotation_orbit_to_camera @ basis.T
    dq_dcenter = -dq_dpoint
    dq_dradius = -rotation_orbit_to_camera @ phase_coordinates[:, None]

    z_axis_skew = _skew(np.asarray([0.0, 0.0, 1.0]))
    dnegative_rotation_dphase = (
        -z_axis_skew @ negative_phase_rotation
    )
    dq_dphase = rotation0 @ (
        dnegative_rotation_dphase @ orbit_coordinates
        - radius_value * negative_phase_rotation @ phase_derivative
    )
    dq_dfrequency = dq_dphase[:, None] * (2.0 * math.pi * tau)

    # GTSAM Rot3 uses right-local coordinates: R0' = R0 Exp(delta).
    dq_drotation0 = -rotation0 @ _skew(rotated_orbit_coordinates)

    dz_dnormal = np.vstack((
        relative @ du_dn,
        relative @ dv_dn,
        relative @ dnormal_dn,
    ))
    dz_du = np.vstack((
        relative @ du_du,
        relative @ dv_du,
        np.zeros(3),
    ))
    dq_dnormal = rotation_orbit_to_camera @ dz_dnormal
    dq_du = rotation_orbit_to_camera @ dz_du

    jacobians = OrbitFactorJacobians(
        projection_jacobian @ dq_dpoint,
        projection_jacobian @ dq_dcenter,
        projection_jacobian @ dq_dradius,
        projection_jacobian @ dq_dfrequency,
        projection_jacobian @ dq_drotation0,
        projection_jacobian @ dq_dnormal,
        projection_jacobian @ dq_du,
    )
    return OrbitFactorEvaluation(
        _readonly(residual), _readonly(predicted), _readonly(camera_point),
        _readonly(normal), _readonly(u_w), _readonly(v_w), _readonly(basis),
        phase, False, jacobians,
    )


__all__ = [
    "OrbitFactorError", "OrbitFactorEvaluation", "OrbitFactorJacobians",
    "evaluate_orbit_factor",
]
