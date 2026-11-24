from __future__ import annotations

import math

import gtsam
import numpy as np
import pytest

from esfo.colmap import (
    ExtractedColmapModel, IndexedView, ModelSummary, TrackPointMapping,
)
from esfo.geometry import RegisteredTrajectory
from esfo.gtsam_backend import (
    GTSAM_VERSION, OrbitFactorBinding, OrbitFactorKeys, OrbitFactorMeasurement,
    assert_gtsam_version, build_factor_observations,
    count_nonpositive_depth_factors, make_orbit_custom_factor,
    optimize_orbit_graph, optimize_orbit_graph_with_diagnostics,
)
from esfo.orbit_factor import evaluate_orbit_factor


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _factor_fixture():
    keys = OrbitFactorKeys.conventional(0)
    point = np.asarray([0.3, -0.4, 6.0])
    center = np.asarray([0.1, -0.2, 0.3])
    radius = 1.7
    frequency = 0.23
    r0 = _rotation(np.asarray([1.0, 2.0, -0.5]), 0.31)
    normal = np.asarray([0.2, -0.3, 1.1])
    u_raw = np.asarray([1.2, 0.4, -0.1])
    time_s, t_ref_s = 0.71, 0.13
    intrinsic = np.asarray([
        [320.0, 0.0, 173.0],
        [0.0, 318.0, 131.0],
        [0.0, 0.0, 1.0],
    ])
    first = evaluate_orbit_factor(
        point_w=point, center_w=center, radius=radius,
        frequency_hz=frequency, R0=r0, normal_raw=normal, u_raw=u_raw,
        time_s=time_s, t_ref_s=t_ref_s, intrinsic=intrinsic,
        observed_xy=np.zeros(2),
    )
    measurement = OrbitFactorMeasurement(
        time_s, t_ref_s, intrinsic, first.predicted_xy,
    )
    values = gtsam.Values()
    values.insert(keys.point_w, point)
    values.insert(keys.center_w, center)
    values.insert(keys.radius, radius)
    values.insert(keys.frequency_hz, frequency)
    values.insert(keys.R0, gtsam.Rot3(r0))
    values.insert(keys.normal_raw, normal)
    values.insert(keys.u_raw, u_raw)
    return keys, measurement, values


def _point_for_camera_point(
    keys: OrbitFactorKeys,
    measurement: OrbitFactorMeasurement,
    values: gtsam.Values,
    desired_camera_point: np.ndarray,
) -> np.ndarray:
    normal = values.atPoint3(keys.normal_raw)
    normal = normal / np.linalg.norm(normal)
    u_w = values.atPoint3(keys.u_raw)
    u_w = u_w - normal * float(normal @ u_w)
    u_w = u_w / np.linalg.norm(u_w)
    basis = np.column_stack((u_w, np.cross(normal, u_w), normal))
    phase = 2.0 * math.pi * values.atDouble(keys.frequency_hz) * (
        measurement.time_s - measurement.t_ref_s
    )
    cosine, sine = math.cos(phase), math.sin(phase)
    negative_phase_rotation = np.asarray([
        [cosine, sine, 0.0],
        [-sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    phase_coordinates = np.asarray([cosine, sine, 0.0])
    orbit_coordinates = (
        negative_phase_rotation.T
        @ values.atRot3(keys.R0).matrix().T
        @ np.asarray(desired_camera_point)
    )
    return (
        values.atPoint3(keys.center_w)
        + basis @ (
            orbit_coordinates
            + values.atDouble(keys.radius) * phase_coordinates
        )
    )


def test_exact_gtsam_version_and_right_local_rot3_coordinates() -> None:
    assert GTSAM_VERSION == "4.2"
    assert_gtsam_version()
    rotation = gtsam.Rot3(_rotation(np.asarray([0.4, -0.2, 0.9]), 0.57))
    delta = np.asarray([2e-4, -3e-4, 5e-4])

    retracted = rotation.retract(delta).matrix()
    expected = rotation.matrix() @ gtsam.Rot3.Expmap(delta).matrix()

    np.testing.assert_allclose(retracted, expected, rtol=0.0, atol=2e-16)


def test_version_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("esfo.gtsam_backend.version", lambda _name: "4.2.0")
    with pytest.raises(RuntimeError, match="expected gtsam==4.2"):
        assert_gtsam_version()


def test_custom_factor_has_frozen_keys_noise_residual_and_jacobians() -> None:
    keys, measurement, values = _factor_fixture()
    factor = make_orbit_custom_factor(keys, measurement)

    assert tuple(factor.keys()) == keys.ordered()
    np.testing.assert_array_equal(factor.noiseModel().sigmas(), [1.0, 1.0])
    np.testing.assert_allclose(
        factor.unwhitenedError(values), np.zeros(2), rtol=0.0, atol=1e-12,
    )
    expected = evaluate_orbit_factor(
        point_w=values.atPoint3(keys.point_w),
        center_w=values.atPoint3(keys.center_w),
        radius=values.atDouble(keys.radius),
        frequency_hz=values.atDouble(keys.frequency_hz),
        R0=values.atRot3(keys.R0).matrix(),
        normal_raw=values.atPoint3(keys.normal_raw),
        u_raw=values.atPoint3(keys.u_raw),
        time_s=measurement.time_s, t_ref_s=measurement.t_ref_s,
        intrinsic=measurement.intrinsic, observed_xy=measurement.observed_xy,
    )
    linear = factor.linearize(values)
    expected_a = np.concatenate(expected.jacobians.ordered(), axis=1)
    np.testing.assert_allclose(
        linear.getA(), expected_a, rtol=0.0, atol=2e-13,
    )
    np.testing.assert_allclose(
        linear.getb(), -expected.residual_xy, rtol=0.0, atol=2e-13,
    )


def test_default_lm_runs_direct_custom_factor_without_prior_or_retry() -> None:
    keys, measurement, values = _factor_fixture()
    graph = gtsam.NonlinearFactorGraph()
    graph.add(make_orbit_custom_factor(keys, measurement))

    result = optimize_orbit_graph(graph, values)

    assert graph.size() == 1
    assert isinstance(graph.at(0), gtsam.CustomFactor)
    assert graph.error(result) == pytest.approx(0.0, abs=1e-20)
    # A zero-residual underdetermined graph is accepted without adding an
    # anchoring prior; gauge handling belongs to result comparison.
    assert result.size() == values.size() == 7


def test_cheirality_penalty_has_zero_jacobian_and_lm_does_not_abort() -> None:
    keys, measurement, values = _factor_fixture()
    behind_point = _point_for_camera_point(
        keys, measurement, values, np.asarray([0.2, -0.1, -1.0]),
    )
    behind = gtsam.Values(values)
    behind.erase(keys.point_w)
    behind.insert(keys.point_w, behind_point)

    factor = make_orbit_custom_factor(keys, measurement)
    expected_penalty = 2.0 * measurement.intrinsic[0, 0]
    np.testing.assert_array_equal(
        factor.unwhitenedError(behind),
        [expected_penalty, expected_penalty],
    )
    linear = factor.linearize(behind)
    np.testing.assert_array_equal(linear.getA(), np.zeros((2, 17)))
    np.testing.assert_array_equal(
        linear.getb(), [-expected_penalty, -expected_penalty],
    )

    binding = OrbitFactorBinding(keys, measurement)
    assert count_nonpositive_depth_factors(behind, (binding,)) == 1
    graph = gtsam.NonlinearFactorGraph()
    graph.add(factor)
    optimized = optimize_orbit_graph_with_diagnostics(
        graph, behind, (binding,),
    )
    assert optimized.diagnostics.factor_count == 1
    assert optimized.diagnostics.initial_nonpositive_depth_count == 1
    assert optimized.diagnostics.final_nonpositive_depth_count == 1
    # The frozen zero-Jacobian penalty makes the invalid state stationary,
    # but crucially the CustomFactor no longer aborts the complete LM call.
    np.testing.assert_array_equal(
        factor.unwhitenedError(optimized.values),
        [expected_penalty, expected_penalty],
    )

    wrong_binding = OrbitFactorBinding(
        OrbitFactorKeys.conventional(1), measurement,
    )
    with pytest.raises(ValueError, match="binding key order"):
        optimize_orbit_graph_with_diagnostics(
            graph, behind, (wrong_binding,),
        )


def test_default_lm_evaluates_a_negative_depth_trial_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys, measurement, initial = _factor_fixture()
    behind_point = _point_for_camera_point(
        keys, measurement, initial, np.asarray([0.2, -0.1, -2.0]),
    )
    graph = gtsam.NonlinearFactorGraph()
    graph.add(make_orbit_custom_factor(keys, measurement))

    # Test-only anchors isolate the CustomFactor callback behavior; the
    # reference optimizer never adds these priors.  A strong point target
    # deliberately drives an LM trial through non-positive depth.
    fixed3 = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
    fixed1 = gtsam.noiseModel.Isotropic.Sigma(1, 1e-6)
    graph.add(gtsam.PriorFactorPoint3(
        keys.point_w, behind_point,
        gtsam.noiseModel.Isotropic.Sigma(3, 1e-3),
    ))
    graph.add(gtsam.PriorFactorPoint3(
        keys.center_w, initial.atPoint3(keys.center_w), fixed3,
    ))
    graph.add(gtsam.PriorFactorDouble(
        keys.radius, initial.atDouble(keys.radius), fixed1,
    ))
    graph.add(gtsam.PriorFactorDouble(
        keys.frequency_hz, initial.atDouble(keys.frequency_hz), fixed1,
    ))
    graph.add(gtsam.PriorFactorRot3(
        keys.R0, initial.atRot3(keys.R0),
        gtsam.noiseModel.Isotropic.Sigma(3, 1e-6),
    ))
    graph.add(gtsam.PriorFactorPoint3(
        keys.normal_raw, initial.atPoint3(keys.normal_raw), fixed3,
    ))
    graph.add(gtsam.PriorFactorPoint3(
        keys.u_raw, initial.atPoint3(keys.u_raw), fixed3,
    ))

    from esfo import gtsam_backend
    original_evaluation = gtsam_backend._factor_evaluation
    penalty_visits: list[bool] = []

    def tracked_evaluation(*args, **kwargs):
        evaluation = original_evaluation(*args, **kwargs)
        penalty_visits.append(evaluation.cheirality_penalty)
        return evaluation

    monkeypatch.setattr(gtsam_backend, "_factor_evaluation", tracked_evaluation)
    result = optimize_orbit_graph(graph, initial)

    assert any(penalty_visits)
    assert result.size() == initial.size()


def _view(
    image_id: int,
    window_index: int,
    track_ids: list[int],
) -> IndexedView:
    points = np.asarray([
        [10.0 * track_id + window_index, 20.0 + window_index]
        for track_id in track_ids
    ], dtype=np.float64).reshape((-1, 2))
    return IndexedView(
        image_id, window_index, 30.0 * window_index,
        f"window_{window_index:06d}.png",
        np.asarray(track_ids, dtype=np.int64), points,
    )


def _scope_fixture():
    views = (
        _view(1, 1, [1, 2]),
        _view(2, 2, [1]),
        _view(3, 3, [1]),
        _view(4, 4, [3]),
    )
    rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    trajectory = RegisteredTrajectory(
        np.asarray([1, 3, 4]), np.asarray([0.03, 0.09, 0.12]),
        rotations, np.zeros((3, 3)),
    )
    mapping = TrackPointMapping(
        unique_pairs=((1, 10),),
        # Track 3 deliberately belongs to both exclusion categories so the
        # provenance counts cannot be reconstructed by set subtraction.
        split_track_ids=(3,),
        merged_point_ids=(99,),
        merged_track_ids=(3,),
        uninitialized_track_ids=(2,),
        excluded_track_ids=(2, 3),
    )
    model = ExtractedColmapModel(
        ModelSummary(0, 3, 2), trajectory, mapping,
        np.asarray([1]), np.asarray([10]), np.asarray([[1.0, 2.0, 3.0]]),
        {10: ((1, 0), (3, 0)), 99: ((4, 0),)}, (2,), {},
    )
    return views, model


def test_observation_scope_keeps_unregistered_and_colmap_absent_observations() -> None:
    views, model = _scope_fixture()

    selected = build_factor_observations(
        views, model,
        selected_point_observations={10: ((1, 0), (3, 0))},
    )

    assert tuple(
        (item.track_id, item.window_index) for item in selected.observations
    ) == ((1, 1), (1, 2), (1, 3))
    assert selected.observations[1].registered_window is False
    assert selected.observations[1].retained_in_colmap_point_track is False
    np.testing.assert_array_equal(
        selected.observations[1].observed_xy,
        views[1].keypoints_xy[0],
    )
    diagnostics = selected.diagnostics
    assert diagnostics.total_factor_count == 3
    assert diagnostics.registered_window_factor_count == 2
    assert diagnostics.unregistered_window_factor_count == 1
    assert diagnostics.retained_colmap_track_factor_count == 2
    assert diagnostics.absent_colmap_track_factor_count == 1
    assert diagnostics.mapped_track_count == 1
    assert diagnostics.logical_window_count == 4
    assert diagnostics.missing_track_window_pair_count == 1
    assert diagnostics.split_track_count == 1
    assert diagnostics.merged_track_count == 1
    assert diagnostics.uninitialized_track_count == 1
    assert selected.metadata["residual_or_depth_pruning"] is False


def test_observation_scope_does_not_need_colmap_track_membership_to_select() -> None:
    views, model = _scope_fixture()
    selected = build_factor_observations(views, model)
    assert len(selected.observations) == 3
    assert selected.diagnostics.retained_colmap_track_factor_count == 2
    assert selected.diagnostics.absent_colmap_track_factor_count == 1
    assert [item.retained_in_colmap_point_track
            for item in selected.observations] == [True, False, True]


def test_observation_scope_rejects_a_mismatched_colmap_feature() -> None:
    views, model = _scope_fixture()
    with pytest.raises(ValueError, match="mapped source track"):
        build_factor_observations(
            views, model, selected_point_observations={10: ((1, 1),)},
        )


def test_observation_scope_accepts_full_points3d_tracks_and_requires_mapped_ids() -> None:
    views, model = _scope_fixture()
    selected = build_factor_observations(
        views, model,
        # Point 99 is deliberately excluded/merged. A natural points3D map
        # contains it, but it must not change factors for uniquely mapped 10.
        selected_point_observations={
            10: ((1, 0), (3, 0)),
            99: ((4, 0),),
        },
    )
    assert len(selected.observations) == 3
    assert selected.diagnostics.retained_colmap_track_factor_count == 2

    with pytest.raises(ValueError, match="missing mapped point IDs: 10"):
        build_factor_observations(
            views, model,
            selected_point_observations={99: ((4, 0),)},
        )


def test_observation_scope_requires_complete_contiguous_logical_windows() -> None:
    views, model = _scope_fixture()
    with pytest.raises(ValueError, match="contiguous window sequence"):
        build_factor_observations((views[0], views[2], views[3]), model)


def test_factor_keys_are_unique() -> None:
    keys = OrbitFactorKeys.conventional(3)
    with pytest.raises(ValueError, match="unique"):
        OrbitFactorKeys(
            keys.point_w, keys.point_w, keys.radius, keys.frequency_hz,
            keys.R0, keys.normal_raw, keys.u_raw,
        )


def _gate_c_problem():
    rng = np.random.default_rng(4)
    intrinsic = np.asarray([
        [300.0, 0.0, 160.0],
        [0.0, 300.0, 120.0],
        [0.0, 0.0, 1.0],
    ])
    center = np.asarray([0.2, -0.1, 0.0])
    radius, frequency = 1.6, 0.21
    R0 = np.eye(3)
    normal = np.asarray([0.0, 0.0, 1.0])
    u_raw = np.asarray([1.0, 0.0, 0.0])
    times = np.linspace(0.0, 1.4, 9)
    points = np.column_stack((
        rng.uniform(-0.5, 0.5, 12),
        rng.uniform(-0.5, 0.5, 12),
        rng.uniform(5.0, 6.0, 12),
    ))
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    global_keys = OrbitFactorKeys.conventional(0)
    initial.insert(global_keys.center_w, center + np.asarray([0.02, -0.01, 0.01]))
    initial.insert(global_keys.radius, radius * 0.98)
    initial.insert(global_keys.frequency_hz, frequency * 1.03)
    initial.insert(
        global_keys.R0,
        gtsam.Rot3.Expmap(np.asarray([0.006, -0.005, 0.004])),
    )
    initial.insert(global_keys.normal_raw, normal + np.asarray([0.004, -0.003, 0.002]))
    initial.insert(global_keys.u_raw, u_raw + np.asarray([0.002, 0.003, -0.004]))
    for point_index, point in enumerate(points):
        keys = OrbitFactorKeys.conventional(point_index)
        initial.insert(keys.point_w, point + rng.normal(0.0, 0.01, 3))
        for time_s in times:
            generated = evaluate_orbit_factor(
                point_w=point, center_w=center, radius=radius,
                frequency_hz=frequency, R0=R0, normal_raw=normal,
                u_raw=u_raw, time_s=float(time_s), t_ref_s=0.0,
                intrinsic=intrinsic, observed_xy=np.zeros(2),
            )
            graph.add(make_orbit_custom_factor(
                keys,
                OrbitFactorMeasurement(
                    float(time_s), 0.0, intrinsic, generated.predicted_xy,
                ),
            ))
    truth = {
        "intrinsic": intrinsic, "center": center, "radius": radius,
        "frequency": frequency, "R0": R0, "normal": normal,
        "u_raw": u_raw, "times": times, "points": points,
    }
    return graph, initial, global_keys, truth


def _orbit_centers(
    center: np.ndarray,
    radius: float,
    frequency: float,
    normal_raw: np.ndarray,
    u_raw: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = normal_raw / np.linalg.norm(normal_raw)
    u_w = u_raw - normal * float(normal @ u_raw)
    u_w /= np.linalg.norm(u_w)
    v_w = np.cross(normal, u_w)
    phase = 2.0 * math.pi * frequency * times
    centers = center + radius * (
        np.cos(phase)[:, None] * u_w + np.sin(phase)[:, None] * v_w
    )
    return centers, normal, u_w


def _similarity_source_to_target(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean, target_mean = np.mean(source, axis=0), np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    diagonal = np.diag([1.0, 1.0, np.linalg.det(left @ right_t)])
    rotation = left @ diagonal @ right_t
    variance = float(np.sum(source_centered * source_centered) / source.shape[0])
    scale = float(np.trace(np.diag(singular) @ diagonal) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def test_default_lm_recovers_noise_free_problem_up_to_similarity_gauge() -> None:
    graph, initial, global_keys, truth = _gate_c_problem()
    initial_error = graph.error(initial)

    result = optimize_orbit_graph(graph, initial)

    recovered_points = np.stack([
        result.atPoint3(OrbitFactorKeys.conventional(index).point_w)
        for index in range(truth["points"].shape[0])
    ])
    recovered_centers, recovered_normal, recovered_u = _orbit_centers(
        result.atPoint3(global_keys.center_w),
        result.atDouble(global_keys.radius),
        result.atDouble(global_keys.frequency_hz),
        result.atPoint3(global_keys.normal_raw),
        result.atPoint3(global_keys.u_raw),
        truth["times"],
    )
    true_centers, _, _ = _orbit_centers(
        truth["center"], truth["radius"], truth["frequency"],
        truth["normal"], truth["u_raw"], truth["times"],
    )
    recovered_geometry = np.vstack((recovered_points, recovered_centers))
    true_geometry = np.vstack((truth["points"], true_centers))
    scale, rotation, translation = _similarity_source_to_target(
        recovered_geometry, true_geometry,
    )
    aligned = scale * (recovered_geometry @ rotation.T) + translation

    assert initial_error > 100.0
    assert graph.error(result) < 1e-20
    np.testing.assert_allclose(aligned, true_geometry, rtol=0.0, atol=8e-12)
    assert scale * result.atDouble(global_keys.radius) == pytest.approx(
        truth["radius"], abs=8e-12,
    )
    assert result.atDouble(global_keys.frequency_hz) == pytest.approx(
        truth["frequency"], abs=8e-13,
    )
    np.testing.assert_allclose(
        rotation @ recovered_normal, truth["normal"], rtol=0.0, atol=8e-12,
    )
    np.testing.assert_allclose(
        rotation @ recovered_u, truth["u_raw"], rtol=0.0, atol=8e-12,
    )
    # R0 maps the phase-zero orbit frame to the camera frame and is unchanged
    # by a global world-frame Sim(3) gauge transformation.
    np.testing.assert_allclose(
        result.atRot3(global_keys.R0).matrix(), truth["R0"],
        rtol=0.0, atol=8e-12,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def test_stacked_factor_jacobian_has_exactly_the_ten_known_gauges() -> None:
    _, _, _, truth = _gate_c_problem()
    points = truth["points"]
    point_count = points.shape[0]
    dimension = 3 * point_count + 14
    rows: list[np.ndarray] = []
    for point_index, point in enumerate(points):
        for time_s in truth["times"]:
            evaluation = evaluate_orbit_factor(
                point_w=point, center_w=truth["center"],
                radius=truth["radius"], frequency_hz=truth["frequency"],
                R0=truth["R0"], normal_raw=truth["normal"],
                u_raw=truth["u_raw"], time_s=float(time_s), t_ref_s=0.0,
                intrinsic=truth["intrinsic"], observed_xy=np.zeros(2),
            )
            blocks = evaluation.jacobians
            row = np.zeros((2, dimension), dtype=np.float64)
            row[:, 3 * point_index:3 * point_index + 3] = blocks.point_w
            offset = 3 * point_count
            for block in (
                blocks.center_w, blocks.radius, blocks.frequency_hz,
                blocks.R0, blocks.normal_raw, blocks.u_raw,
            ):
                row[:, offset:offset + block.shape[1]] = block
                offset += block.shape[1]
            rows.append(row)
    jacobian = np.vstack(rows)
    gauge = np.zeros((dimension, 10), dtype=np.float64)
    orbit_offset = 3 * point_count

    # Sim(3): three translations, three active world rotations, one scale.
    for axis in range(3):
        for point_index in range(point_count):
            gauge[3 * point_index + axis, axis] = 1.0
        gauge[orbit_offset + axis, axis] = 1.0
    for point_index, point in enumerate(points):
        gauge[3 * point_index:3 * point_index + 3, 3:6] = -_skew(point)
    gauge[orbit_offset:orbit_offset + 3, 3:6] = -_skew(truth["center"])
    gauge[orbit_offset + 8:orbit_offset + 11, 3:6] = -_skew(truth["normal"])
    gauge[orbit_offset + 11:orbit_offset + 14, 3:6] = -_skew(truth["u_raw"])
    for point_index, point in enumerate(points):
        gauge[3 * point_index:3 * point_index + 3, 6] = point
    gauge[orbit_offset:orbit_offset + 3, 6] = truth["center"]
    gauge[orbit_offset + 3, 6] = truth["radius"]
    # Raw-vector representation gauges: |n|, |u|, and u's component along n.
    gauge[orbit_offset + 8:orbit_offset + 11, 7] = truth["normal"]
    gauge[orbit_offset + 11:orbit_offset + 14, 8] = truth["u_raw"]
    gauge[orbit_offset + 11:orbit_offset + 14, 9] = truth["normal"]

    assert np.linalg.matrix_rank(gauge) == 10
    np.testing.assert_allclose(
        jacobian @ gauge, np.zeros((jacobian.shape[0], 10)),
        rtol=0.0, atol=2e-12,
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank_tolerance = (
        np.finfo(np.float64).eps * max(jacobian.shape) * singular[0]
    )
    rank = int(np.count_nonzero(singular > rank_tolerance))
    assert dimension - rank == 10
    # The first identifiable singular direction is cleanly separated from the
    # ten algebraic null directions on this preregistered synthetic fixture.
    assert singular[-11] > 1e10 * singular[-10]
