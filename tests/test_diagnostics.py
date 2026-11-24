from __future__ import annotations

import math

import numpy as np
import pytest

from esfo.colmap import (
    ExtractedColmapModel,
    IndexedView,
    ModelSummary,
    TrackPointMapping,
)
from esfo.data import EventSequence
from esfo.diagnostics import SceneDiagnostics, aggregate_scene_diagnostics
from esfo.frequency import initialize_frequency_fft_raw
from esfo.frontend import run_frontend_raw
from esfo.geometry import RegisteredTrajectory, fit_orbit_geometry
from esfo.gtsam_backend import (
    OrbitOptimizationDiagnostics,
    build_factor_observations,
)
from esfo.provenance import canonical_json


def _fixture():
    events = EventSequence.from_arrays(
        t_us=[0, 20_001, 40_001, 60_001, 80_001, 100_001, 120_001],
        x=[0, 1, 2, 3, 2, 1, 0],
        y=[0, 0, 0, 0, 0, 0, 0],
        p=[0, 0, 0, 0, 0, 0, 0],
        width=20,
        height=20,
    )
    frontend = run_frontend_raw(events)
    frequency = initialize_frequency_fft_raw(events)
    views = tuple(
        IndexedView(
            index,
            index,
            30.0 * index,
            f"window_{index:06d}.png",
            np.asarray([1], dtype=np.int64),
            np.asarray([[10.0 + index, 12.0]], dtype=np.float64),
        )
        for index in range(1, 5)
    )
    angles = np.asarray([0.0, 0.4, 0.8, 1.2])
    centers = np.column_stack((
        3.0 * np.cos(angles),
        3.0 * np.sin(angles),
        np.zeros(angles.size),
    ))
    trajectory = RegisteredTrajectory(
        np.arange(1, 5, dtype=np.int64),
        np.arange(1, 5, dtype=np.float64) * 0.03,
        np.repeat(np.eye(3)[None, :, :], 4, axis=0),
        -centers,
    )
    mapping = TrackPointMapping(
        unique_pairs=((1, 10),),
        split_track_ids=(),
        merged_point_ids=(),
        merged_track_ids=(),
        uninitialized_track_ids=(),
        excluded_track_ids=(),
    )
    model = ExtractedColmapModel(
        ModelSummary(0, 4, 1),
        trajectory,
        mapping,
        np.asarray([1]),
        np.asarray([10]),
        np.asarray([[0.0, 0.0, 5.0]]),
        {10: tuple((index, 0) for index in range(1, 5))},
        (),
        {},
    )
    geometry = fit_orbit_geometry(trajectory, frequency.frequency_hz)
    factors = build_factor_observations(views, model)
    optimization = OrbitOptimizationDiagnostics(
        len(factors.observations), 0, 0,
    )
    return (
        events, frontend, views, model, geometry, frequency, factors,
        optimization,
    )


def test_aggregate_scene_diagnostics_is_lossless_and_json_safe() -> None:
    (
        events, frontend, views, model, geometry, frequency, factors,
        optimization,
    ) = _fixture()
    result = aggregate_scene_diagnostics(
        events=events,
        frontend=frontend,
        views=views,
        model=model,
        geometry=geometry,
        frequency=frequency,
        factors=factors,
        optimization=optimization,
        verified_inlier_row_count=6,
        hdbscan_repeat_labels_equal=True,
        signed_radius=geometry.radius,
        signed_frequency_hz=frequency.frequency_hz,
    )

    assert result.frontend["event_count"] == len(events)
    assert result.frontend["hdbscan_repeat_labels_equal"] is True
    assert result.colmap["logical_window_count"] == 4
    assert result.colmap["registered_window_count"] == 4
    assert result.colmap["registered_window_ratio"] == 1.0
    assert result.colmap["all_pairs_image_pair_count"] == math.comb(4, 2)
    assert result.colmap["raw_correspondence_row_count"] == math.comb(4, 2)
    assert result.factors["total_factor_count"] == 4
    assert result.factors["registered_factor_ratio"] == 1.0
    assert result.optimization["initial_nonpositive_depth_count"] == 0
    assert "NaN" not in canonical_json(result)


def test_scene_diagnostics_rejects_missing_fields_and_bad_accounting() -> None:
    with pytest.raises(ValueError, match="frontend diagnostics lack"):
        SceneDiagnostics({}, {}, {}, {}, {}, {})

    (
        events, frontend, views, model, geometry, frequency, factors,
        optimization,
    ) = _fixture()
    valid = aggregate_scene_diagnostics(
        events=events,
        frontend=frontend,
        views=views,
        model=model,
        geometry=geometry,
        frequency=frequency,
        factors=factors,
        optimization=optimization,
        verified_inlier_row_count=6,
        hdbscan_repeat_labels_equal=True,
        signed_radius=geometry.radius,
        signed_frequency_hz=frequency.frequency_hz,
    )
    bad_colmap = dict(valid.colmap)
    bad_colmap["registered_window_count"] = 3
    with pytest.raises(ValueError, match="window accounting"):
        SceneDiagnostics(
            valid.frontend,
            bad_colmap,
            valid.geometry,
            valid.frequency,
            valid.factors,
            valid.optimization,
        )
