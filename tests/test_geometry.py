from __future__ import annotations

import math

import numpy as np
import pytest

from esfo.geometry import (
    GeometryInitializationError,
    OrbitGeometryInitialization,
    PlaneFit,
    RegisteredTrajectory,
    _checked_circle_radius_squared,
    active_align_to_z,
    fit_orbit_geometry,
    fit_orbit_plane,
    initialize_r0,
    orbit_centers,
    orbit_poses,
)


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _rotz(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _exact_orbit() -> tuple[
    RegisteredTrajectory, np.ndarray, float, np.ndarray, np.ndarray, float,
]:
    normal = np.asarray([2.0, -3.0, 4.0])
    normal /= np.linalg.norm(normal)
    seed = np.asarray([1.0, 2.0, -0.5])
    u_w = seed - float(seed @ normal) * normal
    u_w /= np.linalg.norm(u_w)
    v_w = np.cross(normal, u_w)
    basis = np.column_stack((u_w, v_w, normal))
    center = np.asarray([1.25, -2.5, 4.0])
    radius = 3.0
    frequency = 0.4
    t_ref = 4.7
    phase = np.asarray([0.0, 0.37, 0.91, 1.58, 2.41, 3.55, 5.12])
    time_s = t_ref + phase / (2.0 * math.pi * frequency)
    coordinates = np.column_stack((
        radius * np.cos(phase), radius * np.sin(phase), np.zeros(phase.size),
    ))
    centers = center + coordinates @ basis.T
    r0 = _axis_rotation(np.asarray([1.0, -2.0, 0.5]), 0.73)
    rotations = np.stack([r0 @ _rotz(-angle) @ basis.T for angle in phase])
    translations = -np.einsum("kij,kj->ki", rotations, centers)
    trajectory = RegisteredTrajectory(
        np.asarray([11, 14, 18, 23, 31, 42, 57]), time_s,
        rotations, translations,
    )
    return trajectory, center, radius, normal, r0, frequency


def test_exact_non_axis_orbit_recovers_geometry_r0_and_poses() -> None:
    trajectory, center, radius, normal, expected_r0, frequency = _exact_orbit()

    geometry = fit_orbit_geometry(trajectory, frequency)
    initialized_r0 = initialize_r0(trajectory, geometry, frequency)
    predicted = orbit_poses(
        geometry, frequency, initialized_r0.rotation, trajectory.time_s,
    )

    np.testing.assert_allclose(geometry.center_w, center, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        geometry.plane_diagnostics.temporal_center_w,
        geometry.center_w, rtol=0.0, atol=2e-14,
    )
    assert geometry.radius == pytest.approx(radius, abs=2e-14)
    np.testing.assert_allclose(geometry.normal_w, normal, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(geometry.u_w,
                               (trajectory.centers_w[0] - center) / radius,
                               rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(initialized_r0.rotation, expected_r0,
                               rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(predicted.centers_w, trajectory.centers_w,
                               rtol=0.0, atol=4e-14)
    np.testing.assert_allclose(predicted.R_cw, trajectory.R_cw,
                               rtol=0.0, atol=4e-14)
    np.testing.assert_allclose(predicted.t_cw, trajectory.t_cw,
                               rtol=0.0, atol=2e-13)
    assert geometry.first_view_id == 11
    assert geometry.t_ref_s == trajectory.time_s[0]


def test_orbit_centres_are_correct_at_zero_quarter_and_half_period() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    times = geometry.t_ref_s + np.asarray([
        0.0, 1.0 / (4.0 * frequency), 1.0 / (2.0 * frequency),
    ])

    centers = orbit_centers(geometry, frequency, times)

    expected = np.stack((
        geometry.center_w + geometry.radius * geometry.u_w,
        geometry.center_w + geometry.radius * geometry.v_w,
        geometry.center_w - geometry.radius * geometry.u_w,
    ))
    np.testing.assert_allclose(centers, expected, rtol=0.0, atol=3e-14)


def test_common_absolute_time_shift_leaves_initialization_and_poses_unchanged() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    r0 = initialize_r0(trajectory, geometry, frequency).rotation
    original = orbit_poses(geometry, frequency, r0, trajectory.time_s)
    shift = 1234.5
    shifted_trajectory = RegisteredTrajectory(
        trajectory.view_ids, trajectory.time_s + shift,
        trajectory.R_cw, trajectory.t_cw,
    )
    shifted_geometry = fit_orbit_geometry(shifted_trajectory, frequency)
    shifted_r0 = initialize_r0(
        shifted_trajectory, shifted_geometry, frequency,
    ).rotation
    shifted = orbit_poses(
        shifted_geometry, frequency, shifted_r0,
        shifted_trajectory.time_s,
    )

    np.testing.assert_allclose(shifted.centers_w, original.centers_w,
                               rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(shifted.R_cw, original.R_cw,
                               rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(shifted.t_cw, original.t_cw,
                               rtol=0.0, atol=2e-11)


def test_exact_nondegenerate_global_frame_change_is_equivariant() -> None:
    trajectory, center, radius, normal, _, frequency = _exact_orbit()
    original_geometry = fit_orbit_geometry(trajectory, frequency)
    original_r0 = initialize_r0(
        trajectory, original_geometry, frequency,
    ).rotation
    world_rotation = _axis_rotation(np.asarray([0.4, 1.0, -0.2]), 0.51)
    world_translation = np.asarray([-3.0, 0.7, 2.2])
    transformed_centers = (
        trajectory.centers_w @ world_rotation.T + world_translation
    )
    transformed_rotations = np.einsum(
        "kij,jl->kil", trajectory.R_cw, world_rotation.T,
    )
    transformed_translations = -np.einsum(
        "kij,kj->ki", transformed_rotations, transformed_centers,
    )
    transformed_trajectory = RegisteredTrajectory(
        trajectory.view_ids, trajectory.time_s,
        transformed_rotations, transformed_translations,
    )

    transformed_geometry = fit_orbit_geometry(
        transformed_trajectory, frequency,
    )
    transformed_r0 = initialize_r0(
        transformed_trajectory, transformed_geometry, frequency,
    ).rotation

    np.testing.assert_allclose(
        transformed_geometry.center_w,
        world_rotation @ center + world_translation,
        rtol=0.0, atol=4e-14,
    )
    assert transformed_geometry.radius == pytest.approx(radius, abs=3e-14)
    np.testing.assert_allclose(
        transformed_geometry.normal_w, world_rotation @ normal,
        rtol=0.0, atol=3e-14,
    )
    np.testing.assert_allclose(
        transformed_geometry.u_w, world_rotation @ original_geometry.u_w,
        rtol=0.0, atol=3e-14,
    )
    # With camera coordinates held fixed, the global world-frame transform is
    # absorbed by B and leaves the phase-zero camera-frame rotation unchanged.
    np.testing.assert_allclose(transformed_r0, original_r0,
                               rtol=0.0, atol=5e-14)


@pytest.mark.parametrize(
    ("normal", "branch"),
    [
        (np.asarray([0.0, 0.0, 1.0]), "parallel"),
        (np.asarray([0.0, 0.0, -1.0]), "antiparallel"),
        (np.asarray([1.0, 2.0, 3.0]), "general"),
    ],
)
def test_rodrigues_alignment_covers_all_frozen_branches(
    normal: np.ndarray, branch: str,
) -> None:
    normal = normal / np.linalg.norm(normal)
    alignment = active_align_to_z(normal)
    assert alignment.branch == branch
    np.testing.assert_allclose(
        alignment.rotation @ normal, [0.0, 0.0, 1.0],
        rtol=0.0, atol=6e-14,
    )
    assert np.linalg.det(alignment.rotation) == pytest.approx(1.0, abs=6e-14)
    if branch == "antiparallel":
        np.testing.assert_allclose(alignment.axis, [1.0, 0.0, 0.0],
                                   rtol=0.0, atol=1e-15)


def test_rodrigues_parallel_threshold_is_exercised_on_both_sides() -> None:
    tolerance = 32.0 * np.finfo(np.float64).eps
    for sign, expected_close_branch in ((1.0, "parallel"),
                                        (-1.0, "antiparallel")):
        close_angle = 0.5 * tolerance
        outside_angle = 2.0 * tolerance
        close = np.asarray([
            math.sin(close_angle), 0.0, sign * math.cos(close_angle),
        ])
        outside = np.asarray([
            math.sin(outside_angle), 0.0, sign * math.cos(outside_angle),
        ])
        assert active_align_to_z(close).branch == expected_close_branch
        assert active_align_to_z(outside).branch == "general"


def test_temporal_sign_is_canonical_and_unobservable_motion_fails() -> None:
    trajectory, _, _, normal, _, frequency = _exact_orbit()
    fitted = fit_orbit_plane(
        trajectory.centers_w, trajectory.time_s, frequency,
    )
    assert fitted.temporal_score > fitted.temporal_tolerance
    np.testing.assert_allclose(fitted.normal_w, normal, rtol=0.0, atol=2e-14)

    angles = np.asarray([0.0, math.pi / 2.0, math.pi,
                         math.pi / 2.0, 0.0])
    centers = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(5)))
    with pytest.raises(GeometryInitializationError) as captured:
        fit_orbit_plane(centers, np.arange(5, dtype=np.float64), 0.1)
    assert captured.value.code == "temporal_sign_unobservable"


def test_temporal_sign_uses_only_nonaliased_registered_view_pairs() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    sign_frequency = 2.25 * frequency

    geometry = fit_orbit_geometry(trajectory, sign_frequency)

    diagnostics = geometry.plane_diagnostics
    assert diagnostics.temporal_total_pair_count == 6
    assert diagnostics.temporal_eligible_pair_count == 5
    assert diagnostics.temporal_excluded_pair_count == 1
    assert diagnostics.temporal_max_phase_gap_rad == pytest.approx(3.5325)


def test_sparse_temporal_sign_is_referenced_to_fitted_circle_center() -> None:
    # Only the middle chord is eligible.  Referencing it to the sample
    # centroid reverses its sign, whereas referencing it to the fitted circle
    # centre gives the exact counter-clockwise orientation.
    phase = np.asarray([0.0, 5.80070104, 7.53891608, 12.68894639])
    centers = np.column_stack((
        np.cos(phase), np.sin(phase), np.zeros(phase.size),
    ))
    frequency = 1.0
    time_s = phase / (2.0 * math.pi * frequency)
    centroid = np.mean(centers, axis=0)
    centroid_relative = centers - centroid
    old_cross_z = float(np.cross(
        centroid_relative[1], centroid_relative[2],
    )[2])
    assert old_cross_z < 0.0

    fitted = fit_orbit_plane(centers, time_s, frequency)

    np.testing.assert_allclose(
        fitted.temporal_center_w, np.zeros(3), rtol=0.0, atol=2e-14,
    )
    np.testing.assert_allclose(
        fitted.normal_w, [0.0, 0.0, 1.0], rtol=0.0, atol=2e-14,
    )
    assert fitted.temporal_total_pair_count == 3
    assert fitted.temporal_eligible_pair_count == 1
    assert fitted.temporal_excluded_pair_count == 2


def test_sparse_circle_center_sign_is_world_frame_equivariant() -> None:
    phase = np.asarray([0.0, 5.80070104, 7.53891608, 12.68894639])
    base = np.column_stack((
        np.cos(phase), np.sin(phase), np.zeros(phase.size),
    ))
    angle = 0.41
    rotation = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ])
    translation = np.asarray([2.3, -1.4, 0.7])
    centers = (rotation @ base.T).T + translation
    fitted = fit_orbit_plane(centers, phase / (2.0 * math.pi), 1.0)

    np.testing.assert_allclose(
        fitted.temporal_center_w, translation, rtol=0.0, atol=3e-14,
    )
    np.testing.assert_allclose(
        fitted.normal_w, rotation @ np.asarray([0.0, 0.0, 1.0]),
        rtol=0.0, atol=3e-14,
    )


def test_temporal_sign_fails_only_when_no_nonaliased_pair_exists() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()

    with pytest.raises(GeometryInitializationError) as captured:
        fit_orbit_geometry(trajectory, 10.0 * frequency)
    assert captured.value.code == "temporal_sign_aliased"


def test_r0_requires_the_frequency_used_for_temporal_sign() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)

    with pytest.raises(ValueError, match="temporal-sign"):
        initialize_r0(
            trajectory, geometry, np.nextafter(frequency, math.inf),
        )


def test_r0_accepts_sparse_views_after_sign_was_identified() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    indices = np.asarray([0, 5, 6])
    sparse = RegisteredTrajectory(
        trajectory.view_ids[indices], trajectory.time_s[indices],
        trajectory.R_cw[indices], trajectory.t_cw[indices],
    )

    initialized = initialize_r0(sparse, geometry, frequency)

    np.testing.assert_allclose(
        initialized.rotation,
        initialize_r0(trajectory, geometry, frequency).rotation,
        rtol=0.0, atol=3e-14,
    )


def test_vertical_or_collinear_plane_fails_instead_of_switching_to_tls() -> None:
    centers = np.asarray([
        [0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 2.0, 0.0],
        [0.0, 3.0, -1.0],
    ])
    with pytest.raises(GeometryInitializationError) as captured:
        fit_orbit_plane(centers, np.arange(4, dtype=np.float64), 0.1)
    assert captured.value.code == "plane_rank_deficient"


def test_earliest_view_at_fitted_circle_center_fails_radial_initialization() -> None:
    centers = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
    ])
    rotations = np.repeat(np.eye(3)[None, :, :], centers.shape[0], axis=0)
    trajectory = RegisteredTrajectory(
        np.arange(centers.shape[0]), np.arange(centers.shape[0], dtype=float),
        rotations, -centers,
    )
    with pytest.raises(GeometryInitializationError) as captured:
        fit_orbit_geometry(trajectory, 0.1)
    assert captured.value.code == "radial_direction_unobservable"


def test_circle_radius_squared_roundoff_clamps_or_fails_by_frozen_tolerance() -> None:
    points = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    _, radius_squared, tolerance = _checked_circle_radius_squared(
        np.asarray([0.0, 0.0, -1e-14]), points,
    )
    assert tolerance > 1e-14
    assert radius_squared == 0.0
    with pytest.raises(GeometryInitializationError) as captured:
        _checked_circle_radius_squared(np.asarray([0.0, 0.0, -1.0]), points)
    assert captured.value.code == "negative_radius_squared"


def test_pose_center_relative_rotation_and_projection_invariants() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    r0 = initialize_r0(trajectory, geometry, frequency).rotation
    times = geometry.t_ref_s + np.asarray([0.17, 0.83])
    poses = orbit_poses(geometry, frequency, r0, times)

    recovered_centers = -np.einsum("kji,kj->ki", poses.R_cw, poses.t_cw)
    np.testing.assert_allclose(recovered_centers, poses.centers_w,
                               rtol=0.0, atol=4e-14)
    relative_camera_to_world = poses.R_cw[1].T @ poses.R_cw[0]
    expected_relative = (
        geometry.basis_w
        @ _rotz(float(poses.phase_rad[1] - poses.phase_rad[0]))
        @ geometry.basis_w.T
    )
    np.testing.assert_allclose(relative_camera_to_world, expected_relative,
                               rtol=0.0, atol=4e-14)

    world_point = geometry.center_w + np.asarray([0.2, -0.4, 5.0])
    camera_point = poses.R_cw[0] @ world_point + poses.t_cw[0]
    round_trip = poses.R_cw[0].T @ (camera_point - poses.t_cw[0])
    np.testing.assert_allclose(round_trip, world_point, rtol=0.0, atol=2e-14)
    intrinsic = np.asarray([[320.0, 0.0, 160.0],
                            [0.0, 321.0, 120.0],
                            [0.0, 0.0, 1.0]])
    homogeneous_pose = np.column_stack((poses.R_cw[0], poses.t_cw[0]))
    point_h = np.append(world_point, 1.0)
    np.testing.assert_allclose(
        intrinsic @ camera_point,
        intrinsic @ homogeneous_pose @ point_h,
        rtol=0.0, atol=2e-12,
    )


def test_sign_equivalent_orbit_representation_is_algebraically_identical() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    r0 = initialize_r0(trajectory, geometry, frequency).rotation
    times = geometry.t_ref_s + np.asarray([0.0, 0.31, 0.92])
    canonical = orbit_poses(geometry, frequency, r0, times)
    theta_prime = -canonical.phase_rad
    transform = np.diag([1.0, -1.0, -1.0])
    basis_prime = geometry.basis_w @ transform
    r0_prime = r0 @ transform
    coordinates_prime = np.column_stack((
        geometry.radius * np.cos(theta_prime),
        geometry.radius * np.sin(theta_prime),
        np.zeros(theta_prime.size),
    ))
    centers_prime = geometry.center_w + coordinates_prime @ basis_prime.T
    rotations_prime = np.stack([
        r0_prime @ _rotz(-angle) @ basis_prime.T
        for angle in theta_prime
    ])
    translations_prime = -np.einsum(
        "kij,kj->ki", rotations_prime, centers_prime,
    )

    np.testing.assert_allclose(centers_prime, canonical.centers_w,
                               rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(rotations_prime, canonical.R_cw,
                               rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(translations_prime, canonical.t_cw,
                               rtol=0.0, atol=2e-13)


def _geometry_with_identity_basis() -> OrbitGeometryInitialization:
    plane = PlaneFit(
        np.zeros(3), np.zeros(3), np.asarray([0.0, 0.0, -1.0]),
        np.asarray([0.0, 0.0, 1.0]), 3, 1.0, 1.0, 0.0,
        0.125, 2, 2, 0, math.pi / 4.0,
    )
    return OrbitGeometryInitialization(
        np.zeros(3), 1.0, np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]),
        np.eye(3), np.eye(3), 0, 0, 0.0, plane, 3, 1.0, {},
    )


def _trajectory_with_r0_candidates(
    candidates: np.ndarray, frequency: float = 0.125,
) -> RegisteredTrajectory:
    times = np.arange(candidates.shape[0], dtype=np.float64)
    phase = 2.0 * math.pi * frequency * times
    rotations = np.stack([
        candidate @ _rotz(-angle)
        for candidate, angle in zip(candidates, phase)
    ])
    return RegisteredTrajectory(
        np.arange(candidates.shape[0]), times, rotations,
        np.zeros((candidates.shape[0], 3)),
    )


def test_chordal_mean_applies_unique_determinant_correction() -> None:
    rx = np.diag([1.0, -1.0, -1.0])
    ry = np.diag([-1.0, 1.0, -1.0])
    rz = np.diag([-1.0, -1.0, 1.0])
    candidates = np.stack([rx] * 5 + [ry] * 4 + [rz] * 3)
    trajectory = _trajectory_with_r0_candidates(candidates)

    initialized = initialize_r0(
        trajectory, _geometry_with_identity_basis(), frequency_hz=0.125,
    )

    assert initialized.determinant_correction == -1.0
    np.testing.assert_allclose(initialized.rotation, rx, rtol=0.0, atol=1e-15)


def test_nonunique_chordal_mean_fails_closed() -> None:
    identity = np.eye(3)
    rx = np.diag([1.0, -1.0, -1.0])
    ry = np.diag([-1.0, 1.0, -1.0])
    rz = np.diag([-1.0, -1.0, 1.0])
    trajectory = _trajectory_with_r0_candidates(
        np.stack((identity, rx, ry, rz)),
    )
    with pytest.raises(GeometryInitializationError) as captured:
        initialize_r0(
            trajectory, _geometry_with_identity_basis(), frequency_hz=0.125,
        )
    assert captured.value.code == "r0_mean_degenerate"


def test_full_rank_reflection_mean_with_tied_smallest_values_fails_closed() -> None:
    rx = np.diag([1.0, -1.0, -1.0])
    ry = np.diag([-1.0, 1.0, -1.0])
    rz = np.diag([-1.0, -1.0, 1.0])
    # mean(Rx,Ry,Rz) = -I/3: full rank, but every pi rotation is an equally
    # close SO(3) projection, so the SVD basis must not decide the answer.
    trajectory = _trajectory_with_r0_candidates(np.stack((rx, ry, rz)))
    with pytest.raises(GeometryInitializationError) as captured:
        initialize_r0(
            trajectory, _geometry_with_identity_basis(), frequency_hz=0.125,
        )
    assert captured.value.code == "r0_mean_degenerate"


def test_reference_r0_initialization_requires_positive_numeric_frequency() -> None:
    trajectory, _, _, _, _, frequency = _exact_orbit()
    geometry = fit_orbit_geometry(trajectory, frequency)
    for value in (0.0, -0.5, "0.4"):
        with pytest.raises(ValueError, match="frequency"):
            initialize_r0(trajectory, geometry, value)  # type: ignore[arg-type]


def test_geometry_result_rejects_inconsistent_alignment_and_deep_freezes_metadata() -> None:
    geometry = _geometry_with_identity_basis()
    nested = {"record": {"values": [1, 2]}}
    copied = OrbitGeometryInitialization(
        geometry.center_w, geometry.radius, geometry.normal_w,
        geometry.u_w, geometry.v_w, geometry.basis_w, geometry.plane_to_xy,
        geometry.first_view_id, geometry.first_view_index, geometry.t_ref_s,
        geometry.plane_diagnostics, geometry.circle_rank,
        geometry.circle_condition_number, nested,
    )
    nested["record"]["values"].append(3)
    assert copied.metadata["record"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        copied.metadata["new"] = 1  # type: ignore[index]

    tilted = np.asarray([1.0, 0.0, 0.0])
    tilted_plane = PlaneFit(
        np.zeros(3), np.zeros(3), np.asarray([0.0, 0.0, -1.0]),
        tilted, 3, 1.0, 1.0, 0.0,
        0.125, 2, 2, 0, math.pi / 4.0,
    )
    with pytest.raises(ValueError, match="plane_to_xy"):
        OrbitGeometryInitialization(
            np.zeros(3), 1.0, tilted,
            np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0]),
            np.column_stack((
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]), tilted,
            )),
            np.eye(3), 0, 0, 0.0, tilted_plane, 3, 1.0, {},
        )


def test_trajectory_inputs_are_detached_and_invalid_rotations_rejected() -> None:
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    trajectory = RegisteredTrajectory(
        np.arange(3), np.arange(3, dtype=np.float64),
        rotations, np.zeros((3, 3)),
    )
    rotations[0, 0, 0] = 2.0
    assert trajectory.R_cw[0, 0, 0] == 1.0
    assert not trajectory.R_cw.flags.writeable

    invalid = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    invalid[1, 0, 0] = 2.0
    with pytest.raises(ValueError, match="proper rotation"):
        RegisteredTrajectory(
            np.arange(3), np.arange(3, dtype=np.float64),
            invalid, np.zeros((3, 3)),
        )
