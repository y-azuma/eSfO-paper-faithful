from __future__ import annotations

import math

import numpy as np
import pytest

from esfo.orbit_factor import OrbitFactorError, evaluate_orbit_factor


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    generator = _skew(axis)
    return (np.eye(3) + math.sin(angle) * generator
            + (1.0 - math.cos(angle)) * (generator @ generator))


def _rotz(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([
        [cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0],
    ])


def _fixture() -> dict[str, object]:
    center = np.asarray([1.3, -0.7, 2.1])
    radius = 2.3
    frequency = 0.37
    time_s, t_ref_s = 1.7, 0.4
    normal_raw = np.asarray([0.3, -0.4, 1.2])
    normal = normal_raw / np.linalg.norm(normal_raw)
    u_raw = np.asarray([1.0, 0.3, 0.2])
    u_w = u_raw - float(normal @ u_raw) * normal
    u_w /= np.linalg.norm(u_w)
    v_w = np.cross(normal, u_w)
    basis = np.column_stack((u_w, v_w, normal))
    R0 = _axis_rotation(np.asarray([0.4, -0.8, 1.1]), 0.57)
    phase = 2.0 * math.pi * frequency * (time_s - t_ref_s)
    phase_coordinates = np.asarray([math.cos(phase), math.sin(phase), 0.0])
    desired_camera_point = np.asarray([0.4, -0.3, 5.0])
    orbit_coordinates = _rotz(-phase).T @ R0.T @ desired_camera_point
    point = center + basis @ (orbit_coordinates + radius * phase_coordinates)
    intrinsic = np.asarray([
        [320.0, 0.0, 160.0], [0.0, 321.0, 120.0], [0.0, 0.0, 1.0],
    ])
    return {
        "point_w": point,
        "center_w": center,
        "radius": radius,
        "frequency_hz": frequency,
        "R0": R0,
        "normal_raw": normal_raw,
        "u_raw": u_raw,
        "time_s": time_s,
        "t_ref_s": t_ref_s,
        "intrinsic": intrinsic,
        "observed_xy": np.asarray([181.0, 96.0]),
    }


def _residual(arguments: dict[str, object]) -> np.ndarray:
    return evaluate_orbit_factor(**arguments).residual_xy


def _numeric_additive(
    arguments: dict[str, object], name: str,
) -> np.ndarray:
    original = np.atleast_1d(np.asarray(arguments[name], dtype=np.float64))
    result = np.empty((2, original.size), dtype=np.float64)
    relative_step = np.finfo(np.float64).eps ** (1.0 / 3.0)
    for index, value in enumerate(original):
        step = relative_step * max(1.0, abs(float(value)))
        plus, minus = arguments.copy(), arguments.copy()
        plus_value, minus_value = original.copy(), original.copy()
        plus_value[index] += step
        minus_value[index] -= step
        plus[name] = (float(plus_value[0]) if np.isscalar(arguments[name])
                      else plus_value)
        minus[name] = (float(minus_value[0]) if np.isscalar(arguments[name])
                       else minus_value)
        result[:, index] = (_residual(plus) - _residual(minus)) / (2.0 * step)
    return result


def _numeric_rotation(arguments: dict[str, object]) -> np.ndarray:
    rotation = np.asarray(arguments["R0"], dtype=np.float64)
    step = np.finfo(np.float64).eps ** (1.0 / 3.0)
    result = np.empty((2, 3), dtype=np.float64)
    for index in range(3):
        axis = np.eye(3)[index]
        plus, minus = arguments.copy(), arguments.copy()
        plus["R0"] = rotation @ _axis_rotation(axis, step)
        minus["R0"] = rotation @ _axis_rotation(axis, -step)
        result[:, index] = (_residual(plus) - _residual(minus)) / (2.0 * step)
    return result


def test_exact_residual_and_frozen_residual_sign() -> None:
    arguments = _fixture()
    initial = evaluate_orbit_factor(**arguments)
    arguments["observed_xy"] = initial.predicted_xy
    exact = evaluate_orbit_factor(**arguments)

    np.testing.assert_allclose(exact.residual_xy, np.zeros(2), rtol=0.0,
                               atol=0.0)
    arguments["observed_xy"] = initial.predicted_xy + np.asarray([2.0, -3.0])
    signed = evaluate_orbit_factor(**arguments)
    np.testing.assert_allclose(signed.residual_xy, [-2.0, 3.0], rtol=0.0,
                               atol=3e-14)


@pytest.mark.parametrize(
    ("name", "jacobian_name"),
    [
        ("point_w", "point_w"), ("center_w", "center_w"),
        ("radius", "radius"), ("frequency_hz", "frequency_hz"),
        ("normal_raw", "normal_raw"), ("u_raw", "u_raw"),
    ],
)
def test_analytic_additive_jacobians_match_central_difference(
    name: str, jacobian_name: str,
) -> None:
    arguments = _fixture()
    analytic = getattr(
        evaluate_orbit_factor(**arguments).jacobians, jacobian_name,
    )
    numeric = _numeric_additive(arguments, name)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-7)


def test_right_local_rotation_jacobian_matches_central_difference() -> None:
    arguments = _fixture()
    analytic = evaluate_orbit_factor(**arguments).jacobians.R0
    numeric = _numeric_rotation(arguments)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-7)


def test_raw_normal_and_u_gauge_directions_leave_factor_unchanged() -> None:
    arguments = _fixture()
    reference = evaluate_orbit_factor(**arguments)
    normal = np.asarray(arguments["normal_raw"])
    u_raw = np.asarray(arguments["u_raw"])

    changed = arguments.copy()
    changed["normal_raw"] = 3.7 * normal
    changed["u_raw"] = 2.4 * u_raw + 0.8 * normal
    evaluated = evaluate_orbit_factor(**changed)

    np.testing.assert_allclose(evaluated.predicted_xy, reference.predicted_xy,
                               rtol=0.0, atol=2e-13)
    np.testing.assert_allclose(
        reference.jacobians.normal_raw @ normal, np.zeros(2),
        rtol=0.0, atol=2e-13,
    )
    np.testing.assert_allclose(
        reference.jacobians.u_raw @ u_raw, np.zeros(2),
        rtol=0.0, atol=1e-13,
    )
    np.testing.assert_allclose(
        reference.jacobians.u_raw @ normal, np.zeros(2),
        rtol=0.0, atol=2e-13,
    )


def test_common_time_shift_and_global_world_transform_are_invariant() -> None:
    arguments = _fixture()
    reference = evaluate_orbit_factor(**arguments)

    shifted = arguments.copy()
    shifted["time_s"] = float(arguments["time_s"]) + 123.4
    shifted["t_ref_s"] = float(arguments["t_ref_s"]) + 123.4
    np.testing.assert_allclose(
        evaluate_orbit_factor(**shifted).predicted_xy, reference.predicted_xy,
        rtol=0.0, atol=2e-12,
    )

    world_rotation = _axis_rotation(np.asarray([1.0, -0.3, 0.7]), 0.43)
    world_translation = np.asarray([-2.0, 0.8, 1.4])
    transformed = arguments.copy()
    transformed["point_w"] = (
        world_rotation @ np.asarray(arguments["point_w"]) + world_translation
    )
    transformed["center_w"] = (
        world_rotation @ np.asarray(arguments["center_w"]) + world_translation
    )
    transformed["normal_raw"] = world_rotation @ np.asarray(arguments["normal_raw"])
    transformed["u_raw"] = world_rotation @ np.asarray(arguments["u_raw"])
    np.testing.assert_allclose(
        evaluate_orbit_factor(**transformed).predicted_xy,
        reference.predicted_xy, rtol=0.0, atol=8e-14,
    )


def test_sign_equivalent_representation_and_negative_scalars_are_evaluated() -> None:
    arguments = _fixture()
    reference = evaluate_orbit_factor(**arguments)
    sign_transform = np.diag([1.0, -1.0, -1.0])
    equivalent = arguments.copy()
    equivalent["frequency_hz"] = -float(arguments["frequency_hz"])
    equivalent["normal_raw"] = -np.asarray(arguments["normal_raw"])
    equivalent["R0"] = np.asarray(arguments["R0"]) @ sign_transform
    np.testing.assert_allclose(
        evaluate_orbit_factor(**equivalent).predicted_xy,
        reference.predicted_xy, rtol=0.0, atol=8e-14,
    )

    negative_radius = arguments.copy()
    negative_radius["radius"] = -float(arguments["radius"])
    evaluated = evaluate_orbit_factor(**negative_radius)
    assert np.all(np.isfinite(evaluated.residual_xy))
    assert np.all(np.isfinite(evaluated.jacobians.radius))


def test_degenerate_basis_cheirality_and_intrinsic_fail_explicitly() -> None:
    arguments = _fixture()
    zero_normal = arguments.copy()
    zero_normal["normal_raw"] = np.zeros(3)
    with pytest.raises(OrbitFactorError) as captured:
        evaluate_orbit_factor(**zero_normal)
    assert captured.value.code == "normal_degenerate"

    collinear_u = arguments.copy()
    collinear_u["u_raw"] = 2.0 * np.asarray(arguments["normal_raw"])
    with pytest.raises(OrbitFactorError) as captured:
        evaluate_orbit_factor(**collinear_u)
    assert captured.value.code == "u_degenerate"

    behind = arguments.copy()
    normal = np.asarray(arguments["normal_raw"])
    normal /= np.linalg.norm(normal)
    u_w = np.asarray(arguments["u_raw"])
    u_w -= float(u_w @ normal) * normal
    u_w /= np.linalg.norm(u_w)
    basis = np.column_stack((u_w, np.cross(normal, u_w), normal))
    phase = 2.0 * math.pi * float(arguments["frequency_hz"]) * (
        float(arguments["time_s"]) - float(arguments["t_ref_s"])
    )
    camera_point = np.asarray([0.0, 0.0, -1.0])
    orbit_coordinates = (
        _rotz(-phase).T @ np.asarray(arguments["R0"]).T @ camera_point
    )
    phase_coordinates = np.asarray([math.cos(phase), math.sin(phase), 0.0])
    behind["point_w"] = (
        np.asarray(arguments["center_w"])
        + basis @ (orbit_coordinates + float(arguments["radius"])
                   * phase_coordinates)
    )
    with pytest.raises(OrbitFactorError) as captured:
        evaluate_orbit_factor(**behind)
    assert captured.value.code == "point_behind_camera"

    skewed = arguments.copy()
    invalid_intrinsic = np.asarray(arguments["intrinsic"]).copy()
    invalid_intrinsic[0, 1] = 1.0
    skewed["intrinsic"] = invalid_intrinsic
    with pytest.raises(ValueError, match="zero-skew PINHOLE"):
        evaluate_orbit_factor(**skewed)


def test_results_and_jacobians_are_detached_and_read_only() -> None:
    arguments = _fixture()
    point = np.asarray(arguments["point_w"])
    result = evaluate_orbit_factor(**arguments)
    point[:] = 0.0
    assert not result.predicted_xy.flags.writeable
    assert not result.jacobians.point_w.flags.writeable
    assert np.linalg.norm(result.camera_point) > 1.0
    with pytest.raises(ValueError):
        result.residual_xy[0] = 0.0
