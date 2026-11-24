from __future__ import annotations

import math

import gtsam
import numpy as np
import pytest

from esfo.colmap import ColmapPipelineError
from esfo.evaluation import (
    FinalNonpositiveDepthError,
    MetricAvailability,
    ResidualSummary,
    SceneFailureRecord,
    build_orbit_reprojection_report,
    classify_scene_failure,
    summarize_residuals,
)
from esfo.frequency import FrequencyInitializationError
from esfo.gtsam_backend import (
    OrbitFactorBinding, OrbitFactorKeys, OrbitFactorMeasurement,
)
from esfo.orbit_factor import OrbitFactorError, evaluate_orbit_factor
from esfo.provenance import canonical_json


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
    orbit_coordinates = (
        negative_phase_rotation.T
        @ values.atRot3(keys.R0).matrix().T
        @ np.asarray(desired_camera_point)
    )
    return (
        values.atPoint3(keys.center_w)
        + basis @ (
            orbit_coordinates
            + values.atDouble(keys.radius) * np.asarray([cosine, sine, 0.0])
        )
    )


def _replace_point(
    values: gtsam.Values, key: int, point: np.ndarray,
) -> gtsam.Values:
    result = gtsam.Values(values)
    result.erase(key)
    result.insert(key, point)
    return result


def test_residual_summary_uses_l2_pixel_norms() -> None:
    summary = summarize_residuals([[3.0, 4.0], [0.0, 0.0], [5.0, 12.0]])

    assert summary.count == 3
    assert summary.mean_px == pytest.approx(6.0)
    assert summary.median_px == pytest.approx(5.0)
    assert summary.rmse_px == pytest.approx(math.sqrt((25.0 + 169.0) / 3.0))
    assert summary.p90_px == pytest.approx(11.4)
    assert summary.p95_px == pytest.approx(12.2)
    assert summary.max_px == 13.0


def test_summary_and_availability_reject_nan_or_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        summarize_residuals([[math.nan, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        ResidualSummary(1, 0, 0, math.inf, 0, 0, 0)
    with pytest.raises(ValueError, match="requires a value"):
        MetricAvailability("available", None, None)
    with pytest.raises(ValueError, match="null value"):
        MetricAvailability(
            "unavailable", "reason", ResidualSummary(1, 0, 0, 0, 0, 0, 0),
        )


def test_valid_factor_report_has_available_before_and_after_metrics() -> None:
    keys, measurement, values = _factor_fixture()
    report = build_orbit_reprojection_report(
        values, values, (OrbitFactorBinding(keys, measurement),),
    )

    assert report.factor_count == 1
    assert report.initial_nonpositive_depth_count == 0
    assert report.final_nonpositive_depth_count == 0
    assert report.before_primary.status == "available"
    assert report.after_primary.status == "available"
    assert report.after_primary.value is not None
    assert report.after_primary.value.max_px == pytest.approx(0.0, abs=1e-12)
    assert report.after_in_front.diagnostic_only is True
    assert report.after_in_front.in_front_count == 1
    assert report.after_in_front.excluded_nonpositive_depth_count == 0


def test_final_behind_camera_invalidates_primary_and_excludes_pseudo_pixel() -> None:
    keys, measurement, initial = _factor_fixture()
    behind_point = _point_for_camera_point(
        keys, measurement, initial, np.asarray([0.2, -0.1, -1.0]),
    )
    final = _replace_point(initial, keys.point_w, behind_point)

    report = build_orbit_reprojection_report(
        initial, final, (OrbitFactorBinding(keys, measurement),),
    )

    assert report.final_nonpositive_depth_count == 1
    assert report.before_primary == MetricAvailability(
        "unavailable", "final_nonpositive_depth", None,
    )
    assert report.after_primary == report.before_primary
    assert report.after_in_front.in_front_count == 0
    assert report.after_in_front.excluded_nonpositive_depth_count == 1
    assert report.after_in_front.metric == MetricAvailability(
        "unavailable", "no_in_front_factors", None,
    )
    # The GTSAM-compatible 2*fx constant never appears as a metric value.
    assert "640" not in canonical_json(report)


def test_initial_behind_camera_is_only_a_separate_in_front_diagnostic() -> None:
    keys, measurement, final = _factor_fixture()
    behind_point = _point_for_camera_point(
        keys, measurement, final, np.asarray([0.2, -0.1, -1.0]),
    )
    initial = _replace_point(final, keys.point_w, behind_point)

    report = build_orbit_reprojection_report(
        initial, final, (OrbitFactorBinding(keys, measurement),),
    )

    assert report.initial_nonpositive_depth_count == 1
    assert report.final_nonpositive_depth_count == 0
    assert report.before_primary == MetricAvailability(
        "unavailable", "initial_nonpositive_depth", None,
    )
    assert report.after_primary.status == "available"
    assert report.before_in_front.metric.reason == "no_in_front_factors"
    assert report.before_in_front.diagnostic_only is True


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("normal_degenerate", "orbit_factor_normal_degenerate"),
        ("u_degenerate", "orbit_factor_u_degenerate"),
        ("projection_nonfinite", "orbit_factor_projection_nonfinite"),
        ("other", "orbit_factor_failure"),
    ],
)
def test_factor_failures_keep_distinct_stable_codes(detail, expected) -> None:
    record = classify_scene_failure(
        OrbitFactorError(detail, "synthetic failure"), stage="optimization",
    )
    assert record == SceneFailureRecord(
        "optimization", expected, detail, f"{detail}: synthetic failure",
    )


def test_final_nonpositive_depth_has_its_own_fail_closed_code() -> None:
    error = FinalNonpositiveDepthError(3, 17)
    record = classify_scene_failure(error, stage="optimization")

    assert record == SceneFailureRecord(
        "optimization",
        "final_nonpositive_depth",
        "final_nonpositive_depth",
        "final_nonpositive_depth: 3/17 final factors have non-positive depth",
    )
    assert error.count == 3
    assert error.factor_count == 17


@pytest.mark.parametrize(("count", "factor_count"), [(0, 1), (2, 1)])
def test_final_nonpositive_depth_rejects_impossible_counts(
    count: int, factor_count: int,
) -> None:
    with pytest.raises(ValueError):
        FinalNonpositiveDepthError(count, factor_count)


def test_frequency_colmap_and_resource_failures_are_reason_coded() -> None:
    frequency = classify_scene_failure(
        FrequencyInitializationError(
            "fft_frequency_unobservable", "zero positive power",
        ),
        stage="frequency",
    )
    assert frequency.code == "fft_frequency_unobservable"
    assert frequency.detail_code == "fft_frequency_unobservable"

    validation = classify_scene_failure(
        ColmapPipelineError("raw_match_mismatch", "wrong rows"),
        stage="colmap_validation",
    )
    assert validation.code == "colmap_validator_failure"
    assert validation.detail_code == "raw_match_mismatch"
    assert classify_scene_failure(MemoryError(), stage="frontend").code == "resource_oom"
    assert classify_scene_failure(TimeoutError(), stage="colmap").code == "resource_timeout"


@pytest.mark.parametrize(
    "detail",
    [
        "no_raw_pairs",
        "no_verified_pairs",
        "no_model",
        "insufficient_registered_views",
    ],
)
def test_colmap_unusable_outcomes_are_method_failures(detail) -> None:
    # The classification is based on the stable low-level outcome, not on
    # whether it was noticed by the execution or validation wrapper.
    for stage in ("colmap", "colmap_validation", "colmap_method"):
        record = classify_scene_failure(
            ColmapPipelineError(detail, "no usable reconstruction"),
            stage=stage,
        )
        assert record.code == "colmap_method_failure"
        assert record.detail_code == detail


@pytest.mark.parametrize(
    "detail",
    [
        "raw_match_mismatch",
        "verification_pair_mismatch",
        "invalid_verified_matches",
        "invalid_model",
        "fixed_camera_changed",
    ],
)
def test_colmap_corrupt_or_inconsistent_outputs_are_validator_failures(detail) -> None:
    record = classify_scene_failure(
        ColmapPipelineError(detail, "synthetic structural mismatch"),
        stage="colmap_validation",
    )
    assert record.code == "colmap_validator_failure"
    assert record.detail_code == detail


def test_colmap_process_error_remains_an_execution_failure() -> None:
    record = classify_scene_failure(
        ColmapPipelineError("colmap_version_mismatch", "wrong executable"),
        stage="colmap",
    )
    assert record.code == "colmap_execution_failure"
    assert record.detail_code == "colmap_version_mismatch"


def test_empty_factor_set_is_json_safe_and_unavailable() -> None:
    values = gtsam.Values()
    report = build_orbit_reprojection_report(values, values, ())

    assert report.factor_count == 0
    assert report.before_primary.reason == "no_factors"
    assert report.after_primary.reason == "no_factors"
    assert report.before_in_front.metric.reason == "no_in_front_factors"
    assert "NaN" not in canonical_json(report)

