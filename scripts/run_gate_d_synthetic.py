#!/usr/bin/env python3
"""Gate-D synthetic integration proof for the frozen eSfO pipeline.

The runner never reads TOPSPIN or any other public scene. It starts from a
deterministic raw event stream, executes the real front end, and carries the
result through the frozen geometry, FFT and direct-GTSAM stages. Unit tests
use a truth-derived COLMAP model conspicuously labelled ``fixture_only``.
That mode proves orchestration and diagnostics only; canonical Linux evidence
must use ``--colmap-mode sidecar``.
"""

from __future__ import annotations

import argparse
import copyreg
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import gtsam
import numpy as np

from esfo.colmap import (
    COLMAP_TAG_COMMIT, COLMAP_VERSION, Calibration, ColmapPipelineError,
    ExtractedColmapModel, IndexedView, ModelSummary, assert_colmap_version,
    build_colmap_commands, build_indexed_views,
    build_model_converter_command, discover_sparse_models,
    expected_raw_correspondence_count, extract_text_model,
    populate_known_camera_database, select_largest_model,
    validate_fixed_camera_database, validate_geometric_verification,
    validate_imported_raw_matches, validate_track_point_mapping,
    write_matches_importer_file,
)
from esfo.data import EventSequence
from esfo.diagnostics import aggregate_scene_diagnostics
from esfo.efast import CIRCLE3, CIRCLE4
from esfo.evaluation import (
    FinalNonpositiveDepthError, build_orbit_reprojection_report,
)
from esfo.frequency import FFTFrequencyInitialization, initialize_frequency_fft_raw
from esfo.frontend import FrontendResult, cluster_corners, run_frontend_raw
from esfo.geometry import (
    OrbitGeometryInitialization, RegisteredTrajectory, fit_orbit_geometry,
    initialize_r0,
)
from esfo.gtsam_backend import (
    FactorObservationSet, OrbitFactorBinding, OrbitFactorKeys,
    OrbitFactorMeasurement, OrbitOptimizationResult,
    build_factor_observations, make_orbit_custom_factor,
    optimize_orbit_graph_with_diagnostics,
)
from esfo.pipeline import (
    PipelineRunResult, PipelineStage, StageExecution, run_reference_pipeline,
)
from esfo.provenance import (
    atomic_write_json, canonical_hash, code_identity, dependency_manifest,
    file_identity,
)


WIDTH = 640
HEIGHT = 480
PHASE_COUNT = 64
POINT_COUNT = 18
PULSES_PER_PHASE = 20
PHASE_PERIOD_US = 30_000
PULSE_PERIOD_US = 1_400
ORBIT_RADIUS = 10.0
CAMERA_Z = 2.5
FIXTURE_SEED = 20_241_124
SIDECAR_IMAGE = "esfo-colmap:3.9.1-cpu"
FIXTURE_IDENTITY_QUANTUM = 1e-9
FIXTURE_IDENTITY_SCHEME = (
    "recipe_plus_integer_events_plus_round_to_nearest_1e-9_v1"
)
CANONICAL_PHASE_COUNT = 128
CANONICAL_POINT_COUNT = 120
CANONICAL_CENTER_CADENCE_US = 800
CANONICAL_EVENT_COUNT = 414_721
CANONICAL_FIXTURE_HASH = (
    "499c760fa35a0a1ec00f824065337115eec421f54c8cf144809d47e7fb05566f"
)

# Frozen before canonical execution.  These are indices into the first 60,000
# PCG64 points drawn uniformly from [-3.8, 3.8]^3 with FIXTURE_SEED.  The
# result-independent selection used the full 128-view orbit and greedily
# maximised the minimum same-view image separation, after requiring an eFAST
# margin and <14 px adjacent-view displacement.  Freezing indices avoids
# repeating the expensive selection or making future evidence depend on it.
# A fixed +0.05 world-x translation breaks the selected cloud's near-perfect
# xy centroid symmetry, so the frozen raw mean-x FFT selects the full-orbit
# fundamental (bin 1) rather than its second harmonic.  This was fixed using
# synthetic truth before any public scene and does not alter method settings.
CANONICAL_POINT_INDICES = np.asarray([
    8, 12869, 29820, 10816, 26367, 131, 45764, 429, 57145, 21818,
    9700, 491, 50370, 44811, 50982, 19840, 25340, 13404, 32503, 56377,
    52053, 58423, 33033, 19926, 1067, 2017, 58808, 37104, 42384, 55435,
    16855, 16930, 11364, 46522, 44914, 54331, 36637, 5250, 14723, 28858,
    2578, 7326, 9964, 47558, 14106, 32986, 435, 57825, 23101, 24896,
    55908, 31575, 38457, 48502, 41151, 38091, 3764, 32724, 59245, 3302,
    17392, 17317, 18065, 28393, 1, 34902, 34082, 53146, 23730, 11976,
    46342, 50183, 52483, 26728, 8996, 1433, 5292, 13763, 44007, 23641,
    37084, 28390, 1868, 32579, 50229, 52341, 51167, 41673, 34954, 29120,
    45666, 46024, 5608, 30896, 13672, 32401, 16239, 28730, 55072, 53221,
    23506, 57591, 25931, 58392, 33969, 24134, 45173, 38888, 7721, 7022,
    56447, 37374, 31098, 8426, 157, 5550, 46930, 56135, 38698, 52901,
], dtype=np.int64)
CANONICAL_POINT_INDICES.setflags(write=False)

# Result-independent array frozen before this integrated runner existed.
REFERENCE_POINTS_W = np.asarray([
    [1.5822143071973098, .25991522310467285, 1.9998784175315802],
    [.2539183600939099, .16876512041710034, -1.9899290428063647],
    [.442474269586989, .4136695543548581, .10120078990306869],
    [-.8022319160181741, 2.487865876550776, -1.767576619461245],
    [1.901154234008561, 2.1393560882895635, -1.724384200768331],
    [-2.3600405244107607, .2741889700782596, -1.8609745717773256],
    [2.1088853095892732, -1.990905929040598, -1.896796072702648],
    [-1.0003796688940458, -1.9836258664751938, -1.8486697898649753],
    [.7510629335679941, .6156296636661969, 1.0658962462370996],
    [2.3320418648662162, .21518401059154302, -1.823156403300822],
    [-2.4290494131014078, 2.00943405237986, -1.6372357240673154],
    [-2.4374360113612434, -1.4418127304042039, -1.7248183462580413],
    [.8047851978536613, -2.4671459432779477, -1.9881665075827621],
    [.49197883546919385, 1.882919177295161, -1.8911323433654386],
    [-1.1239400197283596, 1.0482064947686478, -1.9581007374408537],
    [-1.0768963038821955, -.4087566074953064, -1.965118227504424],
    [1.3651280274397863, -.6965088510147526, -1.9827142760338798],
    [-2.08403065073424, -1.1055467448757446, 1.1527480326538742],
], dtype=np.float64)
REFERENCE_POINTS_W.setflags(write=False)


def _semantic_float_grid(value: Any) -> np.ndarray:
    """Cross-platform fixture identity grid, never a method input."""

    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("semantic identity input contains non-finite values")
    scaled = np.rint(array / FIXTURE_IDENTITY_QUANTUM)
    limit = np.iinfo(np.int64).max
    if np.any(np.abs(scaled) > limit):
        raise ValueError("semantic identity input exceeds int64 range")
    return scaled.astype(np.int64)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_docker_image_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and _is_sha256(value.removeprefix("sha256:"))
    )


@dataclass(frozen=True)
class RawFixture:
    profile: str
    events: EventSequence
    calibration: Calibration
    points_w: np.ndarray
    centers_w: np.ndarray
    rotations_cw: np.ndarray
    projected_xy: np.ndarray
    fixture_hash: str
    phase_count: int
    pulses_per_phase: int
    event_pattern: str

    @property
    def point_count(self) -> int:
        return int(self.points_w.shape[0])


def _look_at_rotation(center_w: np.ndarray) -> np.ndarray:
    forward = -np.asarray(center_w, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-14, rtol=0.0)
            or np.linalg.det(rotation) <= 0.0):
        raise RuntimeError("synthetic look-at rotation is not proper")
    return rotation


def _project(
    calibration: Calibration,
    rotation_cw: np.ndarray,
    center_w: np.ndarray,
    point_w: np.ndarray,
) -> np.ndarray:
    point_c = rotation_cw @ (point_w - center_w)
    if point_c[2] <= 0.0:
        raise RuntimeError("fixture point lies behind a synthetic camera")
    return np.asarray([
        calibration.fx * point_c[0] / point_c[2] + calibration.cx,
        calibration.fy * point_c[1] / point_c[2] + calibration.cy,
    ])


def _canonical_points() -> np.ndarray:
    generated = np.random.default_rng(FIXTURE_SEED).uniform(
        -3.8, 3.8, size=(60_000, 3),
    )
    points = np.asarray(
        generated[CANONICAL_POINT_INDICES], dtype=np.float64,
    ) + np.asarray([0.05, 0.0, 0.0])
    points.setflags(write=False)
    return points


def build_raw_fixture(profile: str = "unit") -> RawFixture:
    """Build a frozen raw fixture without public data or tuned thresholds.

    ``unit`` is the original fast 18-point orchestration fixture.
    ``canonical-sidecar`` is a separate 120-point full-orbit fixture whose
    real front end preserves 106 tracks in every logical view and pair.  The
    latter exists because the unmodified COLMAP 3.9.1 mapper requires 100
    initial-pair inliers; duplicating unit tracks or lowering that frozen
    default would not be valid reference evidence.
    """

    if profile == "unit":
        points = REFERENCE_POINTS_W
        phase_count = PHASE_COUNT
        event_pattern = "repeated_ring_then_center"
        expected_event_count = 184_321
    elif profile == "canonical-sidecar":
        points = _canonical_points()
        phase_count = CANONICAL_PHASE_COUNT
        event_pattern = "single_ring_then_20_centers_800us"
        expected_event_count = CANONICAL_EVENT_COUNT
    else:
        raise ValueError("fixture profile must be unit or canonical-sidecar")
    calibration = Calibration(
        400.0, 400.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    angles = 2.0 * math.pi * np.arange(phase_count) / phase_count
    centers = np.column_stack((
        ORBIT_RADIUS * np.cos(angles),
        ORBIT_RADIUS * np.sin(angles),
        np.full(phase_count, CAMERA_Z),
    ))
    rotations = np.stack([_look_at_rotation(center) for center in centers])
    projected = np.empty((phase_count, len(points), 2), dtype=np.float64)
    for phase in range(phase_count):
        for point_index, point in enumerate(points):
            projected[phase, point_index] = _project(
                calibration, rotations[phase], centers[phase], point,
            )
    pixel_centers = np.rint(projected).astype(np.int64)
    if (np.any(pixel_centers[:, :, 0] < 5)
            or np.any(pixel_centers[:, :, 0] > WIDTH - 5)
            or np.any(pixel_centers[:, :, 1] < 5)
            or np.any(pixel_centers[:, :, 1] > HEIGHT - 5)):
        raise RuntimeError("fixture projection lacks an eFAST-ring margin")

    t_values: list[int] = []
    x_values: list[int] = []
    y_values: list[int] = []
    p_values: list[int] = []
    if profile == "unit":
        for phase in range(phase_count):
            for pulse in range(PULSES_PER_PHASE):
                common = phase * PHASE_PERIOD_US + 1_000 + pulse * PULSE_PERIOD_US
                for point_index, center in enumerate(pixel_centers[phase]):
                    base = common + point_index * 40
                    polarity = point_index & 1
                    for offset, (dx, dy) in enumerate(CIRCLE3[:3]):
                        t_values.append(base + offset)
                        x_values.append(int(center[0] + dx))
                        y_values.append(int(center[1] + dy))
                        p_values.append(polarity)
                    for offset, (dx, dy) in enumerate(CIRCLE4[:4]):
                        t_values.append(base + 10 + offset)
                        x_values.append(int(center[0] + dx))
                        y_values.append(int(center[1] + dy))
                        p_values.append(polarity)
                    t_values.append(base + 20)
                    x_values.append(int(center[0]))
                    y_values.append(int(center[1]))
                    p_values.append(polarity)
    else:
        # Repeating only the centre preserves the eFAST ring time surface.
        # The 800 us cadence was frozen by synthetic probing before canonical
        # execution: all 20 centres are corners, clusters remain per phase,
        # and the 14.8 ms phase gap remains inside the unmodified phi=30 gate.
        for phase in range(phase_count):
            for point_index, center in enumerate(pixel_centers[phase]):
                base = phase * PHASE_PERIOD_US + 1_000
                polarity = point_index & 1
                for offset, (dx, dy) in enumerate(CIRCLE3[:3]):
                    t_values.append(base + offset)
                    x_values.append(int(center[0] + dx))
                    y_values.append(int(center[1] + dy))
                    p_values.append(polarity)
                for offset, (dx, dy) in enumerate(CIRCLE4[:4]):
                    t_values.append(base + 10 + offset)
                    x_values.append(int(center[0] + dx))
                    y_values.append(int(center[1] + dy))
                    p_values.append(polarity)
                for pulse in range(PULSES_PER_PHASE):
                    t_values.append(
                        base + 20 + pulse * CANONICAL_CENTER_CADENCE_US,
                    )
                    x_values.append(int(center[0]))
                    y_values.append(int(center[1]))
                    p_values.append(polarity)
    t_values.append(phase_count * PHASE_PERIOD_US + 1)
    x_values.append(5)
    y_values.append(5)
    p_values.append(0)
    # Canonical points intentionally share pulse timestamps.  Stable temporal
    # sorting gives EventSequence its required nondecreasing order without
    # creating a result-dependent tie break.
    order = np.argsort(np.asarray(t_values, dtype=np.int64), kind="stable")
    events = EventSequence.from_arrays(
        np.asarray(t_values, dtype=np.int64)[order],
        np.asarray(x_values, dtype=np.int64)[order],
        np.asarray(y_values, dtype=np.int64)[order],
        np.asarray(p_values, dtype=np.int64)[order],
        width=WIDTH, height=HEIGHT,
    )
    if len(events) != expected_event_count:
        raise RuntimeError("raw integration fixture event count changed")
    fixture_hash = canonical_hash({
        "schema": "esfo-gate-d-raw-fixture-identity-v3",
        "identity_scheme": FIXTURE_IDENTITY_SCHEME,
        "identity_quantum_decimal": "0.000000001",
        "profile": profile,
        "seed": FIXTURE_SEED,
        "recipe": {
            "sensor": [WIDTH, HEIGHT],
            "phase_count": phase_count,
            "phase_period_us": PHASE_PERIOD_US,
            "pulses_per_phase": PULSES_PER_PHASE,
            "pulse_period_us": PULSE_PERIOD_US,
            "canonical_center_cadence_us": CANONICAL_CENTER_CADENCE_US,
            "orbit_radius": ORBIT_RADIUS,
            "camera_z": CAMERA_Z,
            "canonical_point_indices": (
                CANONICAL_POINT_INDICES if profile == "canonical-sidecar" else []
            ),
            "canonical_world_translation_quantized": (
                _semantic_float_grid([0.05, 0.0, 0.0])
                if profile == "canonical-sidecar" else []
            ),
        },
        "event_pattern": event_pattern,
        "events": {
            "t_us": events.t_us, "x": events.x, "y": events.y,
            "p": events.p, "width": WIDTH, "height": HEIGHT,
        },
        "calibration_quantized": _semantic_float_grid(
            calibration.pinhole_parameters,
        ),
        "points_w_quantized": _semantic_float_grid(points),
        "centers_w_quantized": _semantic_float_grid(centers),
        "rotations_cw_quantized": _semantic_float_grid(rotations),
        "projected_xy_quantized": _semantic_float_grid(projected),
    })
    return RawFixture(
        profile=profile,
        events=events,
        calibration=calibration,
        points_w=np.asarray(points, dtype=np.float64).copy(),
        centers_w=centers,
        rotations_cw=rotations,
        projected_xy=projected,
        fixture_hash=fixture_hash,
        phase_count=phase_count,
        pulses_per_phase=PULSES_PER_PHASE,
        event_pattern=event_pattern,
    )


def _frontend_and_views(
    fixture: RawFixture, *, repeat_check: bool,
) -> tuple[FrontendResult, tuple[IndexedView, ...], bool]:
    frontend = run_frontend_raw(fixture.events)
    repeated_equal = True
    if repeat_check:
        # N6 asks whether *identical HDBSCAN input* gives identical labels.
        # Re-run that operation directly on the exact frozen kept-corner
        # indices instead of repeating deterministic eFAST and density work.
        # This preserves the required check and keeps a canonical large
        # fixture from paying twice for unrelated front-end stages.
        repeated = cluster_corners(
            fixture.events.to_relative_ms(),
            frontend.density.kept_indices,
        )
        repeated_equal = np.array_equal(
            frontend.clustering.labels, repeated.labels,
        )
    relative = fixture.events.to_relative_ms()
    max_window = math.floor(relative.duration_ms / 30.0) - 1
    views = build_indexed_views(
        frontend.observations, max_window_index=max_window,
    )
    counts = {
        "raw": len(fixture.events),
        "corners": int(np.count_nonzero(frontend.corner_mask)),
        "kept": int(frontend.density.kept_indices.size),
        "clusters": len(frontend.clustering.clusters),
        "noise": int(frontend.clustering.metadata["noise_count"]),
        "tracks": len(frontend.association.tracks),
        "observations": len(frontend.observations),
        "views": len(views),
    }
    expected = (
        {
            "raw": 184_321, "corners": 27_317, "kept": 16_308,
            "clusters": 1_076, "noise": 0, "tracks": 18,
            "observations": 1_116, "views": 62,
        }
        if fixture.profile == "unit"
        else {
            "raw": 414_721, "corners": 312_312, "kept": 168_960,
            "clusters": 15_343, "noise": 0, "tracks": 120,
            "observations": 13_928, "views": 126,
        }
    )
    if counts != expected:
        raise RuntimeError(f"integrated frontend fixture changed: {counts}")
    per_track = {
        track.track_id: sum(
            observation.track_id == track.track_id
            for observation in frontend.observations
        )
        for track in frontend.association.tracks
    }
    if fixture.profile == "unit":
        if set(per_track.values()) != {62}:
            raise RuntimeError(
                "unit frontend fixture no longer yields 62 observations/track",
            )
    else:
        full_track_count = sum(value == len(views) for value in per_track.values())
        minimum_view_tracks = min(len(view.track_ids) for view in views)
        minimum_pair_tracks = min(
            np.intersect1d(left.track_ids, right.track_ids, assume_unique=True).size
            for left_index, left in enumerate(views)
            for right in views[left_index + 1:]
        )
        if (full_track_count != 106 or minimum_view_tracks != 106
                or minimum_pair_tracks != 106):
            raise RuntimeError(
                "canonical sidecar fixture lost its >=100-track margin: "
                f"full={full_track_count}, min_view={minimum_view_tracks}, "
                f"min_pair={minimum_pair_tracks}",
            )
    if not repeated_equal:
        raise RuntimeError("identical HDBSCAN input produced different labels")
    return frontend, views, repeated_equal


def _track_to_reference_point(
    fixture: RawFixture, views: Sequence[IndexedView],
) -> tuple[dict[int, int], float]:
    """Bind real front-end track IDs to fixture points without ID assumptions."""

    from scipy.optimize import linear_sum_assignment

    views_tuple = tuple(views)
    if not views_tuple:
        raise RuntimeError("fixture front end produced no logical views")
    point_count = fixture.point_count
    track_ids = np.asarray(views_tuple[0].track_ids, dtype=np.int64)
    if track_ids.size != point_count:
        raise RuntimeError("fixture view does not contain every reference point")
    if any(not np.array_equal(view.track_ids, track_ids) for view in views_tuple):
        raise RuntimeError("fixture track membership changed across logical views")
    cost = np.zeros((point_count, point_count), dtype=np.float64)
    for view in views_tuple:
        expected = np.rint(
            fixture.projected_xy[view.window_index],
        ).astype(np.float64)
        delta = view.keypoints_xy[:, None, :] - expected[None, :, :]
        cost += np.sum(delta * delta, axis=2)
    cost /= len(views_tuple)
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(point_count)):
        raise RuntimeError("fixture assignment omitted a source track")
    maximum_rmse = float(math.sqrt(float(np.max(cost[rows, columns]))))
    # eFAST centres and HDBSCAN cluster membership are discrete; the Eq. (4)
    # mean is therefore not exactly the rounded projection.  The frozen
    # fixture was checked before public data and keeps a conservative 2 px
    # identity gate, far below the separation between reference tracks.
    if maximum_rmse > 2.0:
        raise RuntimeError(
            f"front-end observations drifted from fixture pixels: {maximum_rmse} px",
        )
    return {
        int(track_ids[row]): int(point_index)
        for row, point_index in zip(rows, columns)
    }, maximum_rmse


def build_fixture_colmap_model(
    fixture: RawFixture,
    views: Sequence[IndexedView],
) -> ExtractedColmapModel:
    """Build a truth-derived model for unit plumbing, never reference evidence."""

    views_tuple = tuple(views)
    track_to_point, assignment_rmse = _track_to_reference_point(
        fixture, views_tuple,
    )
    point_observations: dict[int, list[tuple[int, int]]] = {}
    for view in views_tuple:
        for feature_index, track_id_raw in enumerate(view.track_ids):
            point_index = track_to_point[int(track_id_raw)]
            point_id = 10_000 + point_index
            point_observations.setdefault(point_id, []).append(
                (view.image_id, feature_index),
            )
    frozen_observations = {
        point_id: tuple(features)
        for point_id, features in sorted(point_observations.items())
    }
    mapping = validate_track_point_mapping(views_tuple, frozen_observations)
    pairs = mapping.unique_pairs
    mapped_track_ids = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    mapped_point_ids = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    point_id_to_index = {
        10_000 + index: index for index in range(fixture.point_count)
    }
    mapped_points = np.stack([
        fixture.points_w[point_id_to_index[int(point_id)]]
        for point_id in mapped_point_ids
    ])
    phase_indices = np.asarray(
        [view.window_index for view in views_tuple], dtype=np.int64,
    )
    rotations = fixture.rotations_cw[phase_indices]
    centers = fixture.centers_w[phase_indices]
    translations = -np.einsum("kij,kj->ki", rotations, centers)
    trajectory = RegisteredTrajectory(
        np.asarray([view.image_id for view in views_tuple], dtype=np.int64),
        np.asarray([view.time_ms / 1000.0 for view in views_tuple]),
        rotations,
        translations,
    )
    return ExtractedColmapModel(
        ModelSummary(0, len(views_tuple), fixture.point_count),
        trajectory,
        mapping,
        mapped_track_ids,
        mapped_point_ids,
        mapped_points,
        frozen_observations,
        (),
        {
            "fixture_only": True,
            "truth_derived": True,
            "reference_evidence_eligible": False,
            "requires_sidecar_evidence": True,
            "track_assignment_method": "minimum_global_pixel_rmse",
            "maximum_assignment_rmse_px": assignment_rmse,
        },
    )


def _container_path(path: Path, root: Path) -> str:
    return "/work/" + str(path.resolve().relative_to(root.resolve()))


def _run_sidecar_command(
    *, image: str, root: Path, logical_argv: Sequence[str], log: Path,
) -> dict[str, Any]:
    if not logical_argv or logical_argv[0] != "colmap":
        raise ValueError("logical COLMAP argv must begin with colmap")
    command = [
        "docker", "run", "--rm", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp",
        "--volume", f"{root.resolve()}:/work", image, *logical_argv[1:],
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    stdout_path = log.with_suffix(".stdout.txt")
    stderr_path = log.with_suffix(".stderr.txt")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "argv": command,
        "returncode": int(completed.returncode),
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed ({completed.returncode}); see {stderr_path}",
        )
    return record


def _run_bound_sidecar_command(
    context: RunContext,
    *,
    root: Path,
    logical_argv: Sequence[str],
    log: Path,
) -> dict[str, Any]:
    identity = context.sidecar_identity
    if not isinstance(identity, dict):
        raise RuntimeError("COLMAP sidecar identity is not bound")
    _sidecar_checkpoint_identity(identity)
    return _run_sidecar_command(
        image=str(identity["image_id"]), root=root,
        logical_argv=logical_argv, log=log,
    )


def _collect_sidecar_identity(image: str) -> dict[str, Any]:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, check=False,
    )
    if inspected.returncode != 0:
        raise RuntimeError(f"cannot inspect COLMAP sidecar image: {image}")
    records = json.loads(inspected.stdout)
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("Docker returned an invalid image inspection record")
    record = records[0]
    image_id = record.get("Id")
    if not _is_docker_image_id(image_id):
        raise RuntimeError("COLMAP sidecar image ID is malformed")
    help_run = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image_id, "--help"],
        capture_output=True, text=True, check=False,
    )
    help_output = help_run.stdout + help_run.stderr
    if help_run.returncode != 0:
        raise RuntimeError("COLMAP sidecar --help failed")
    assert_colmap_version(help_output)
    labels = record.get("Config", {}).get("Labels", {}) or {}
    if labels.get("org.opencontainers.image.revision") != COLMAP_TAG_COMMIT:
        raise RuntimeError("COLMAP sidecar revision label is not the pinned commit")
    if labels.get("org.opencontainers.image.version") != COLMAP_VERSION:
        raise RuntimeError("COLMAP sidecar version label is not pinned")
    diagnostics_run = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--entrypoint", "/bin/bash", image_id, "-lc",
            "set -eu; sha256sum /usr/local/bin/colmap; "
            "ldd /usr/local/bin/colmap; "
            "git -C /opt/colmap-src rev-parse HEAD; "
            "dpkg-query -W -f='${Package}=${Version}\\n' | sort",
        ],
        capture_output=True, text=True, check=False,
    )
    if diagnostics_run.returncode != 0:
        raise RuntimeError("COLMAP sidecar identity diagnostics failed")
    diagnostics = diagnostics_run.stdout + diagnostics_run.stderr
    normalized_diagnostics = re.sub(
        r"0x[0-9a-fA-F]+", "<address>", diagnostics,
    )
    binary_match = re.search(
        r"(?m)^([0-9a-f]{64})\s+/usr/local/bin/colmap$", diagnostics,
    )
    if binary_match is None or COLMAP_TAG_COMMIT not in diagnostics.splitlines():
        raise RuntimeError("COLMAP binary/source identity is malformed")
    return {
        "logical_name": image,
        "image_id": image_id,
        "repo_digests": record.get("RepoDigests") or [],
        "created": record.get("Created"),
        "labels": labels,
        "colmap_version": COLMAP_VERSION,
        "colmap_tag_commit": COLMAP_TAG_COMMIT,
        "colmap_binary_sha256": binary_match.group(1),
        "help_sha256": canonical_hash(help_output),
        "diagnostics_sha256": canonical_hash(diagnostics),
        "diagnostics_normalized_sha256": canonical_hash(
            normalized_diagnostics,
        ),
        "diagnostics": diagnostics,
    }


def _sidecar_checkpoint_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Project full audit diagnostics onto stable immutable image identity."""

    labels = identity.get("labels")
    projection = {
        "logical_name": identity.get("logical_name"),
        "image_id": identity.get("image_id"),
        "repo_digests": sorted(identity.get("repo_digests") or []),
        "revision_label": (
            labels.get("org.opencontainers.image.revision")
            if isinstance(labels, dict) else None
        ),
        "version_label": (
            labels.get("org.opencontainers.image.version")
            if isinstance(labels, dict) else None
        ),
        "colmap_version": identity.get("colmap_version"),
        "colmap_tag_commit": identity.get("colmap_tag_commit"),
        "colmap_binary_sha256": identity.get("colmap_binary_sha256"),
        "help_sha256": identity.get("help_sha256"),
        "diagnostics_normalized_sha256": identity.get(
            "diagnostics_normalized_sha256",
        ),
    }
    required = (
        projection["logical_name"], projection["image_id"],
        projection["revision_label"], projection["version_label"],
        projection["colmap_version"],
        projection["colmap_tag_commit"],
        projection["colmap_binary_sha256"], projection["help_sha256"],
        projection["diagnostics_normalized_sha256"],
    )
    if any(not isinstance(value, str) or not value for value in required):
        raise RuntimeError("COLMAP sidecar checkpoint identity is incomplete")
    return projection


def _verify_bound_sidecar_identity(context: RunContext) -> dict[str, Any]:
    """Re-read the sidecar identity and reject tag-content replacement."""

    expected = context.sidecar_identity
    if not isinstance(expected, dict):
        raise RuntimeError("COLMAP sidecar identity was not bound before execution")
    observed = _collect_sidecar_identity(context.sidecar_image)
    if canonical_hash(_sidecar_checkpoint_identity(observed)) != canonical_hash(
        _sidecar_checkpoint_identity(expected),
    ):
        raise RuntimeError(
            "COLMAP sidecar identity changed after checkpoint fingerprinting",
        )
    return observed


@dataclass
class RunContext:
    fixture: RawFixture
    colmap_mode: str
    sidecar_image: str
    repeat_check: bool
    evidence_code_identity: dict[str, Any] | None = None
    injected_failure_stage: str | None = None
    callback_calls: list[str] | None = None
    frontend: FrontendResult | None = None
    views: tuple[IndexedView, ...] | None = None
    hdbscan_repeat_labels_equal: bool | None = None
    model: ExtractedColmapModel | None = None
    verified_inlier_row_count: int | None = None
    colmap_execution_root: Path | None = None
    selected_model_summary: ModelSummary | None = None
    sidecar_commands: list[dict[str, Any]] | None = None
    sidecar_identity: dict[str, Any] | None = None
    frequency: FFTFrequencyInitialization | None = None
    geometry: OrbitGeometryInitialization | None = None
    r0: Any = None
    factors: FactorObservationSet | None = None
    initial_values: gtsam.Values | None = None
    bindings: tuple[OrbitFactorBinding, ...] | None = None
    optimization: OrbitOptimizationResult | None = None
    optimization_diagnostics: Any = None
    signed_radius: float | None = None
    signed_frequency_hz: float | None = None
    reprojection: Any = None

    def __post_init__(self) -> None:
        if self.callback_calls is None:
            self.callback_calls = []
        if self.sidecar_commands is None:
            self.sidecar_commands = []


def _synthetic_reference_eligibility(
    context: RunContext,
) -> tuple[bool, list[str]]:
    """Return canonical eligibility and explicit fail-closed reasons."""

    reasons: list[str] = []
    code = context.evidence_code_identity or {}
    if code.get("git_dirty") is not False:
        reasons.append("dirty_worktree")
    if not isinstance(code.get("git_revision"), str) or not code["git_revision"]:
        reasons.append("missing_git_revision")
    if not _is_sha256(code.get("source_tree_sha256")):
        reasons.append("missing_source_tree_sha256")
    if platform.system() != "Linux":
        reasons.append("non_linux_host")
    if context.colmap_mode != "sidecar":
        reasons.append("sidecar_not_executed")
    fixture = context.fixture
    if not (
        fixture.profile == "canonical-sidecar"
        and fixture.fixture_hash == CANONICAL_FIXTURE_HASH
        and len(fixture.events) == CANONICAL_EVENT_COUNT
        and fixture.phase_count == CANONICAL_PHASE_COUNT
        and fixture.point_count == CANONICAL_POINT_COUNT
    ):
        reasons.append("noncanonical_fixture_profile")
    if not context.repeat_check:
        reasons.append("repeat_check_not_requested")
    if context.hdbscan_repeat_labels_equal is not True:
        reasons.append("repeat_check_not_equal")
    identity = context.sidecar_identity
    if not (
        isinstance(identity, dict)
        and identity.get("logical_name") == context.sidecar_image
        and _is_docker_image_id(identity.get("image_id"))
        and identity.get("colmap_version") == COLMAP_VERSION
        and identity.get("colmap_tag_commit") == COLMAP_TAG_COMMIT
        and _is_sha256(identity.get("colmap_binary_sha256"))
        and _is_sha256(identity.get("help_sha256"))
        and isinstance(identity.get("labels"), dict)
        and identity["labels"].get("org.opencontainers.image.revision")
        == COLMAP_TAG_COMMIT
        and identity["labels"].get("org.opencontainers.image.version")
        == COLMAP_VERSION
    ):
        reasons.append("invalid_pinned_sidecar_identity")
    return not reasons, reasons


def _restore_mapping_proxy(value: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(value)


def _reduce_mapping_proxy(
    value: Mapping[str, Any],
) -> tuple[Any, tuple[dict[str, Any]]]:
    return _restore_mapping_proxy, (dict(value),)


copyreg.pickle(type(MappingProxyType({})), _reduce_mapping_proxy)


def _state_dict(context: RunContext) -> dict[str, Any]:
    names = (
        # ``fixture`` is deterministically rebuilt from its hash-bound profile
        # in every process.  Excluding this script-defined class also keeps
        # trusted checkpoints importable after a real interpreter restart
        # (where the CLI module name changes from ``__main__``).
        "frontend", "views", "hdbscan_repeat_labels_equal",
        "model", "verified_inlier_row_count", "colmap_execution_root",
        "selected_model_summary", "sidecar_commands", "frequency",
        "sidecar_identity",
        "geometry", "r0", "factors", "optimization_diagnostics",
        "signed_radius", "signed_frequency_hz", "reprojection",
    )
    return {
        name: getattr(context, name)
        for name in names
        if getattr(context, name) is not None
    }


def _write_state(work: Path, context: RunContext) -> Path:
    path = work / "state.pkl"
    temporary = work / ".state.pkl.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(_state_dict(context), stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _hydrate_from_upstream(
    context: RunContext,
    upstream: Any,
) -> None:
    if upstream is None:
        return
    identity = upstream.output_files.get("state")
    if not isinstance(identity, Mapping):
        raise RuntimeError("upstream checkpoint lacks rehydratable state")
    path = Path(str(identity["path"]))
    if file_identity(path)["sha256"] != identity["sha256"]:
        raise RuntimeError("upstream state identity changed")
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict):
        raise RuntimeError("upstream checkpoint state is malformed")
    for name, value in state.items():
        if not hasattr(context, name):
            raise RuntimeError(f"unknown checkpoint state field: {name}")
        setattr(context, name, value)


def _stage_start(context: RunContext, name: str) -> None:
    assert context.callback_calls is not None
    context.callback_calls.append(name)
    if context.injected_failure_stage == name:
        raise RuntimeError(f"injected_gate_d_failure:{name}")


def _json_output(work: Path, name: str, payload: Any) -> Path:
    path = work / name
    atomic_write_json(path, payload)
    return path


def _input_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "input")
    fixture = context.fixture
    arrays = work / "raw_events.npz"
    np.savez_compressed(
        arrays,
        t_us=fixture.events.t_us,
        x=fixture.events.x,
        y=fixture.events.y,
        p=fixture.events.p,
    )
    identity = _json_output(work, "fixture.json", {
        "schema": "esfo-gate-d-raw-fixture-record-v1",
        "fixture_hash": fixture.fixture_hash,
        "fixture_identity_scheme": FIXTURE_IDENTITY_SCHEME,
        "fixture_identity_quantum_decimal": "0.000000001",
        "profile": fixture.profile,
        "seed": FIXTURE_SEED,
        "event_count": len(fixture.events),
        "sensor": [WIDTH, HEIGHT],
        "phase_count": fixture.phase_count,
        "point_count": fixture.point_count,
        "pulses_per_phase": fixture.pulses_per_phase,
        "event_pattern": fixture.event_pattern,
        "public_data_used": False,
    })
    state = _write_state(work, context)
    return StageExecution(
        {"events": arrays, "fixture": identity, "state": state},
        {"fixture_hash": fixture.fixture_hash, "event_count": len(fixture.events)},
    )


def _frontend_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "frontend")
    frontend, views, repeated_equal = _frontend_and_views(
        context.fixture, repeat_check=context.repeat_check,
    )
    context.frontend = frontend
    context.views = views
    context.hdbscan_repeat_labels_equal = repeated_equal
    full_track_count = sum(
        sum(
            observation.track_id == track.track_id
            for observation in frontend.observations
        ) == len(views)
        for track in frontend.association.tracks
    )
    minimum_view_track_count = min(len(view.track_ids) for view in views)
    minimum_pair_track_count = min(
        np.intersect1d(left.track_ids, right.track_ids, assume_unique=True).size
        for left_index, left in enumerate(views)
        for right in views[left_index + 1:]
    )
    result = _json_output(work, "frontend.json", {
        "schema": "esfo-gate-d-frontend-v1",
        "corner_count": int(np.count_nonzero(frontend.corner_mask)),
        "density_kept_corner_count": int(frontend.density.kept_indices.size),
        "cluster_count": len(frontend.clustering.clusters),
        "hdbscan_noise_count": int(frontend.clustering.metadata["noise_count"]),
        "track_count": len(frontend.association.tracks),
        "observation_count": len(frontend.observations),
        "logical_window_count": len(views),
        "full_length_track_count": full_track_count,
        "minimum_view_track_count": minimum_view_track_count,
        "minimum_pair_track_count": minimum_pair_track_count,
        "hdbscan_repeat_check_executed": context.repeat_check,
        "hdbscan_repeat_labels_equal": repeated_equal,
        "whole_sequence": True,
        "chunking": False,
        "subsampling": False,
        "external_approximation": False,
        "labels_sha256": canonical_hash(frontend.clustering.labels),
    })
    state = _write_state(work, context)
    return StageExecution(
        {"frontend": result, "state": state},
        {
            "track_count": len(frontend.association.tracks),
            "observation_count": len(frontend.observations),
            "full_length_track_count": full_track_count,
            "minimum_view_track_count": minimum_view_track_count,
            "minimum_pair_track_count": minimum_pair_track_count,
            "hdbscan_repeat_labels_equal": repeated_equal,
        },
    )


def _colmap_execute_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "colmap_execute")
    if context.views is None:
        raise RuntimeError("frontend views are unavailable")
    views = context.views
    if context.colmap_mode == "fixture-only":
        marker = _json_output(work, "fixture_only.json", {
            "schema": "esfo-gate-d-colmap-execution-v1",
            "mode": "fixture_only",
            "sidecar_executed": False,
            "reference_evidence_eligible": False,
            "requires_sidecar_evidence": True,
        })
        state = _write_state(work, context)
        return StageExecution(
            {"execution": marker, "state": state},
            {"mode": "fixture_only", "sidecar_executed": False},
        )
    if platform.system() != "Linux":
        raise RuntimeError("COLMAP sidecar evidence requires Linux")

    context.sidecar_identity = _verify_bound_sidecar_identity(context)
    resolved_sidecar_image = str(context.sidecar_identity["image_id"])
    context.colmap_execution_root = work
    database = work / "database.db"
    matches = work / "all_pairs.txt"
    empty_images = work / "empty_images"
    models = work / "models"
    text_models = work / "models_txt"
    logs = work / "logs"
    for directory in (empty_images, models, text_models, logs):
        directory.mkdir()
    plan = build_colmap_commands(
        database_path=_container_path(database, work),
        match_list_path=_container_path(matches, work),
        image_path=_container_path(empty_images, work),
        output_path=_container_path(models, work),
    )
    commands = context.sidecar_commands
    assert commands is not None
    commands.append(_run_bound_sidecar_command(
        context, root=work,
        logical_argv=plan.database_creator, log=logs / "01_database_creator",
    ))
    populate_known_camera_database(
        database, calibration=context.fixture.calibration,
        width=WIDTH, height=HEIGHT, views=views,
    )
    pair_count, raw_rows = write_matches_importer_file(matches, views)
    commands.append(_run_bound_sidecar_command(
        context, root=work,
        logical_argv=plan.matches_importer, log=logs / "02_matches_importer",
    ))
    commands.append(_run_bound_sidecar_command(
        context, root=work,
        logical_argv=plan.mapper, log=logs / "03_mapper",
    ))
    summaries = discover_sparse_models(models)
    selected = select_largest_model(summaries)
    context.selected_model_summary = selected
    converted = text_models / str(selected.directory_id)
    converted.mkdir()
    converter = build_model_converter_command(
        input_path=_container_path(models / str(selected.directory_id), work),
        output_path=_container_path(converted, work),
    )
    commands.append(_run_bound_sidecar_command(
        context, root=work,
        logical_argv=converter, log=logs / "04_model_converter",
    ))
    _verify_bound_sidecar_identity(context)
    marker = _json_output(work, "execution.json", {
        "schema": "esfo-gate-d-colmap-execution-v1",
        "mode": "sidecar",
        "sidecar_image": context.sidecar_image,
        "resolved_sidecar_image_id": resolved_sidecar_image,
        "sidecar_identity": context.sidecar_identity,
        "sidecar_executed": True,
        "pair_count": pair_count,
        "raw_correspondence_row_count": raw_rows,
        "selected_model": selected,
        "commands": commands,
    })
    outputs: dict[str, Path] = {
        "execution": marker,
        "database": database,
        "matches": matches,
        "cameras_bin": models / str(selected.directory_id) / "cameras.bin",
        "images_bin": models / str(selected.directory_id) / "images.bin",
        "points3D_bin": models / str(selected.directory_id) / "points3D.bin",
        "cameras_txt": converted / "cameras.txt",
        "images_txt": converted / "images.txt",
        "points3D_txt": converted / "points3D.txt",
    }
    outputs["state"] = _write_state(work, context)
    return StageExecution(
        outputs,
        {
            "mode": "sidecar", "sidecar_executed": True,
            "pair_count": pair_count,
            "raw_correspondence_row_count": raw_rows,
        },
    )


def _colmap_validate_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "colmap_validate")
    if context.views is None:
        raise RuntimeError("frontend views are unavailable")
    views = context.views
    if context.colmap_mode == "fixture-only":
        model = build_fixture_colmap_model(context.fixture, views)
        verified_rows = expected_raw_correspondence_count(views)
        source = "fixture_assumed_all_raw_rows"
    else:
        root = context.colmap_execution_root
        summary = context.selected_model_summary
        if root is None or summary is None:
            raise RuntimeError("COLMAP execution artifacts are unavailable")
        database = root / "database.db"
        validate_fixed_camera_database(
            database, calibration=context.fixture.calibration,
            width=WIDTH, height=HEIGHT,
        )
        validate_imported_raw_matches(database, views)
        verified = validate_geometric_verification(database, views)
        verified_rows = sum(int(value) for value in verified.values())
        model = extract_text_model(
            root / "models_txt" / str(summary.directory_id),
            summary=summary,
            views=views,
            calibration=context.fixture.calibration,
            width=WIDTH,
            height=HEIGHT,
        )
        source = "validated_colmap_two_view_geometries"
    context.model = model
    context.verified_inlier_row_count = verified_rows
    eligible, noncanonical_reasons = _synthetic_reference_eligibility(context)
    marker = _json_output(work, "model.json", {
        "schema": "esfo-gate-d-colmap-model-v1",
        "mode": context.colmap_mode,
        "summary": model.summary,
        "metadata": model.metadata,
        "mapped_track_ids": model.mapped_track_ids,
        "mapped_point_ids": model.mapped_point_ids,
        "verified_inlier_row_count": verified_rows,
        "verified_inlier_count_source": source,
        "fixture_only": context.colmap_mode == "fixture-only",
        "reference_evidence_eligible": eligible,
        "noncanonical_reasons": noncanonical_reasons,
    })
    state = _write_state(work, context)
    return StageExecution(
        {"model": marker, "state": state},
        {
            "mode": context.colmap_mode,
            "registered_image_count": model.summary.registered_image_count,
            "mapped_track_count": len(model.mapping.unique_pairs),
            "verified_inlier_row_count": verified_rows,
            "verified_inlier_count_source": source,
        },
    )


def _frequency_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "frequency")
    frequency = initialize_frequency_fft_raw(context.fixture.events)
    context.frequency = frequency
    output = _json_output(work, "frequency.json", frequency)
    state = _write_state(work, context)
    return StageExecution(
        {"frequency": output, "state": state},
        {
            "frequency_hz": frequency.frequency_hz,
            "peak_bin": frequency.peak_bin,
            "peak_power": frequency.peak_power,
            "zero_positive_power": frequency.zero_positive_power,
        },
    )


def _geometry_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "geometry")
    if context.model is None or context.frequency is None:
        raise RuntimeError("COLMAP model or FFT initialization is unavailable")
    geometry = fit_orbit_geometry(
        context.model.trajectory, context.frequency.frequency_hz,
    )
    r0 = initialize_r0(
        context.model.trajectory, geometry, context.frequency.frequency_hz,
    )
    context.geometry = geometry
    context.r0 = r0
    output = _json_output(work, "geometry.json", {
        "geometry": geometry,
        "r0": r0,
    })
    state = _write_state(work, context)
    return StageExecution(
        {"geometry": output, "state": state},
        {
            "radius": geometry.radius,
            "center_w": geometry.center_w,
            "normal_w": geometry.normal_w,
            "r0_correction": r0.determinant_correction,
        },
    )


def _build_graph(
    context: RunContext,
) -> tuple[
    FactorObservationSet, gtsam.NonlinearFactorGraph, gtsam.Values,
    tuple[OrbitFactorBinding, ...],
]:
    if (context.views is None or context.model is None
            or context.geometry is None or context.frequency is None
            or context.r0 is None):
        raise RuntimeError("optimization prerequisites are unavailable")
    factors = build_factor_observations(context.views, context.model)
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    global_keys = OrbitFactorKeys.conventional(0)
    values.insert(global_keys.center_w, context.geometry.center_w)
    values.insert(global_keys.radius, context.geometry.radius)
    values.insert(global_keys.frequency_hz, context.frequency.frequency_hz)
    values.insert(global_keys.R0, gtsam.Rot3(context.r0.rotation))
    values.insert(global_keys.normal_raw, context.geometry.normal_w)
    values.insert(global_keys.u_raw, context.geometry.u_w)
    for point_index, point_w in enumerate(context.model.mapped_points_w):
        values.insert(
            OrbitFactorKeys.conventional(point_index).point_w,
            point_w,
        )
    bindings: list[OrbitFactorBinding] = []
    intrinsic = context.fixture.calibration.camera_matrix
    for observation in factors.observations:
        keys = OrbitFactorKeys.conventional(observation.point_index)
        measurement = OrbitFactorMeasurement(
            observation.time_s,
            context.geometry.t_ref_s,
            intrinsic,
            observation.observed_xy,
        )
        graph.add(make_orbit_custom_factor(keys, measurement))
        bindings.append(OrbitFactorBinding(keys, measurement))
    return factors, graph, values, tuple(bindings)


def _optimization_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "optimization")
    factors, graph, initial, bindings = _build_graph(context)
    optimization = optimize_orbit_graph_with_diagnostics(
        graph, initial, bindings,
    )
    reprojection = build_orbit_reprojection_report(
        initial, optimization.values, bindings,
    )
    context.factors = factors
    context.initial_values = initial
    context.bindings = bindings
    context.optimization = optimization
    context.optimization_diagnostics = optimization.diagnostics
    context.reprojection = reprojection
    global_keys = OrbitFactorKeys.conventional(0)
    signed_radius = optimization.values.atDouble(global_keys.radius)
    signed_frequency = optimization.values.atDouble(global_keys.frequency_hz)
    context.signed_radius = signed_radius
    context.signed_frequency_hz = signed_frequency
    output = _json_output(work, "optimization.json", {
        "diagnostics": optimization.diagnostics,
        "reprojection": reprojection,
        "initial_graph_error": graph.error(initial),
        "final_graph_error": graph.error(optimization.values),
        "signed_radius": signed_radius,
        "signed_frequency_hz": signed_frequency,
    })
    state = _write_state(work, context)
    return StageExecution(
        {"optimization": output, "state": state},
        {
            "factor_count": optimization.diagnostics.factor_count,
            "initial_nonpositive_depth_count": (
                optimization.diagnostics.initial_nonpositive_depth_count
            ),
            "final_nonpositive_depth_count": (
                optimization.diagnostics.final_nonpositive_depth_count
            ),
            "signed_radius": signed_radius,
            "signed_frequency_hz": signed_frequency,
        },
    )


def _scene_report_callback(
    context: RunContext, work: Path, upstream: Any,
) -> StageExecution:
    _hydrate_from_upstream(context, upstream)
    _stage_start(context, "scene_report")
    if (context.optimization_diagnostics is not None
            and context.optimization_diagnostics.final_nonpositive_depth_count > 0):
        raise FinalNonpositiveDepthError(
            context.optimization_diagnostics.final_nonpositive_depth_count,
            context.optimization_diagnostics.factor_count,
        )
    required = (
        context.frontend, context.views, context.model, context.geometry,
        context.frequency, context.factors, context.optimization_diagnostics,
        context.reprojection,
    )
    if any(value is None for value in required):
        raise RuntimeError("scene report prerequisites are unavailable")
    assert context.frontend is not None
    assert context.views is not None
    assert context.model is not None
    assert context.geometry is not None
    assert context.frequency is not None
    assert context.factors is not None
    assert context.optimization_diagnostics is not None
    assert context.hdbscan_repeat_labels_equal is not None
    assert context.verified_inlier_row_count is not None
    assert context.signed_radius is not None
    assert context.signed_frequency_hz is not None
    signed_radius = context.signed_radius
    signed_frequency = context.signed_frequency_hz
    diagnostics = aggregate_scene_diagnostics(
        events=context.fixture.events,
        frontend=context.frontend,
        views=context.views,
        model=context.model,
        geometry=context.geometry,
        frequency=context.frequency,
        factors=context.factors,
        optimization=context.optimization_diagnostics,
        verified_inlier_row_count=context.verified_inlier_row_count,
        hdbscan_repeat_labels_equal=context.hdbscan_repeat_labels_equal,
        signed_radius=signed_radius,
        signed_frequency_hz=signed_frequency,
    )
    eligible, noncanonical_reasons = _synthetic_reference_eligibility(context)
    evidence = {
        "schema": "esfo-gate-d-scene-evidence-v1",
        "status": (
            "passed" if eligible else
            "plumbing_passed_fixture_only"
            if context.colmap_mode == "fixture-only" else
            "completed_noncanonical"
        ),
        "public_data_used": False,
        "fixture_hash": context.fixture.fixture_hash,
        "colmap_mode": context.colmap_mode,
        "fixture_only": context.colmap_mode == "fixture-only",
        "sidecar_executed": context.colmap_mode == "sidecar",
        "reference_evidence_eligible": eligible,
        "noncanonical_reasons": noncanonical_reasons,
        "requires_sidecar_evidence": context.colmap_mode != "sidecar",
        "first_failure": None,
        "scene_diagnostics": diagnostics,
        "reprojection_report": context.reprojection,
    }
    output = _json_output(work, "scene_evidence.json", evidence)
    state = _write_state(work, context)
    return StageExecution(
        {"scene_evidence": output, "state": state},
        {
            "reference_evidence_eligible": eligible,
            "factor_count": context.optimization_diagnostics.factor_count,
            "final_nonpositive_depth_count": (
                context.optimization_diagnostics.final_nonpositive_depth_count
            ),
        },
    )


def build_pipeline_stages(context: RunContext) -> tuple[PipelineStage, ...]:
    common = {
        "fixture_hash": context.fixture.fixture_hash,
        "public_data_used": False,
        "colmap_mode": context.colmap_mode,
        "sidecar_image": context.sidecar_image,
        "sidecar_identity_sha256": (
            canonical_hash(_sidecar_checkpoint_identity(
                context.sidecar_identity,
            ))
            if context.sidecar_identity is not None else None
        ),
    }

    def stage(
        name: str,
        callback: Any,
        *,
        failure_stage: str,
        extra: dict[str, Any] | None = None,
    ) -> PipelineStage:
        config = dict(common)
        config.update({
            "stage_role": name,
            "failure_injected": context.injected_failure_stage == name,
        })
        if extra:
            config.update(extra)
        return PipelineStage(
            name,
            config,
            lambda work, upstream, function=callback: function(
                context, work, upstream,
            ),
            failure_stage=failure_stage,
        )

    return (
        stage("input", _input_callback, failure_stage="input"),
        stage(
            "frontend", _frontend_callback, failure_stage="frontend",
            extra={"hdbscan_repeat_check": context.repeat_check},
        ),
        stage(
            "colmap_execute", _colmap_execute_callback,
            failure_stage="colmap",
            extra={"sidecar_image": context.sidecar_image},
        ),
        stage(
            "colmap_validate", _colmap_validate_callback,
            failure_stage="colmap_validation",
        ),
        stage("frequency", _frequency_callback, failure_stage="frequency"),
        stage("geometry", _geometry_callback, failure_stage="geometry"),
        stage(
            "optimization", _optimization_callback,
            failure_stage="optimization",
        ),
        stage("scene_report", _scene_report_callback, failure_stage="metrics"),
    )


def _checkpoint_identities(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: file_identity(path)
        for path in sorted((root / "stages").glob("*.stage.json"))
    }


def _record_hashes(result: PipelineRunResult) -> list[str]:
    return [canonical_hash(record) for record in result.records]


def _output_hashes(result: PipelineRunResult) -> dict[str, dict[str, str]]:
    return {
        record.stage: {
            label: str(identity["sha256"])
            for label, identity in sorted(record.output_files.items())
        }
        for record in result.records
    }


def _load_verified_scene_evidence(
    identity: Mapping[str, Any], context: RunContext,
) -> tuple[dict[str, Any], bool, list[str]]:
    """Verify the checkpointed file and recompute canonical eligibility."""

    path = Path(str(identity.get("path", "")))
    observed = file_identity(path)
    for field in ("path", "size_bytes", "sha256"):
        if observed[field] != identity.get(field):
            raise RuntimeError("scene evidence output identity changed after checkpoint")
    payload = json.loads(path.read_text(encoding="utf-8"))
    eligible, reasons = _synthetic_reference_eligibility(context)
    if (
        payload.get("reference_evidence_eligible") is not eligible
        or payload.get("noncanonical_reasons") != reasons
    ):
        raise RuntimeError(
            "scene evidence eligibility disagrees with recomputed run identity",
        )
    return payload, eligible, reasons


def _run_pipeline(
    *,
    root: Path,
    fixture: RawFixture,
    colmap_mode: str,
    sidecar_image: str,
    repeat_check: bool,
    injected_failure_stage: str | None,
    code: dict[str, Any],
    dependencies: dict[str, Any],
    stage_limit: int | None = None,
) -> tuple[PipelineRunResult, RunContext]:
    bound_sidecar_identity = (
        _collect_sidecar_identity(sidecar_image)
        if colmap_mode == "sidecar" else None
    )
    context = RunContext(
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        evidence_code_identity=code,
        injected_failure_stage=injected_failure_stage,
        sidecar_identity=bound_sidecar_identity,
    )
    stages = build_pipeline_stages(context)
    selected = stages if stage_limit is None else stages[:stage_limit]
    bound_dependencies = dict(dependencies)
    if bound_sidecar_identity is not None:
        bound_dependencies["colmap_sidecar_identity"] = (
            _sidecar_checkpoint_identity(bound_sidecar_identity)
        )
    result = run_reference_pipeline(
        selected,
        work_root=root,
        initial_input_identity={
            "fixture_hash": fixture.fixture_hash,
            "fixture_profile": fixture.profile,
            "fixture_identity_scheme": FIXTURE_IDENTITY_SCHEME,
            "schema": "esfo-gate-d-raw-fixture-identity-v3",
        },
        code_identity=code,
        dependencies=bound_dependencies,
        resume=True,
    )
    return result, context


def _runner_code_identity(repository: Path) -> dict[str, Any]:
    identity = code_identity(repository)
    identity["gate_d_synthetic_runner"] = file_identity(Path(__file__))
    dockerfile = repository / "containers" / "colmap-3.9.1" / "Dockerfile"
    if dockerfile.is_file():
        identity["colmap_sidecar_dockerfile"] = file_identity(dockerfile)
    return identity


def run_proof(
    *,
    work_root: str | Path,
    colmap_mode: str = "fixture-only",
    sidecar_image: str = SIDECAR_IMAGE,
    failure_stage: str = "frontend",
    repeat_check: bool = True,
    fixture_profile: str | None = None,
) -> dict[str, Any]:
    """Run prefix-fresh, partial/full resume, and failure proofs."""

    if colmap_mode not in {"fixture-only", "sidecar"}:
        raise ValueError("colmap_mode must be fixture-only or sidecar")
    if fixture_profile is None:
        fixture_profile = (
            "canonical-sidecar" if colmap_mode == "sidecar" else "unit"
        )
    if fixture_profile not in {"unit", "canonical-sidecar"}:
        raise ValueError("invalid fixture profile")
    if colmap_mode == "sidecar" and fixture_profile != "canonical-sidecar":
        raise ValueError(
            "sidecar reference evidence requires the canonical-sidecar fixture",
        )
    if colmap_mode == "sidecar" and platform.system() != "Linux":
        raise RuntimeError("canonical COLMAP sidecar evidence requires Linux")
    allowed_failure_stages = {
        "input", "frontend", "colmap_execute", "colmap_validate",
        "frequency", "geometry", "optimization", "scene_report",
    }
    if failure_stage not in allowed_failure_stages:
        raise ValueError("failure_stage is not a pipeline stage")
    root = Path(work_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("work root must be absent or empty; stale reuse is forbidden")
    root.mkdir(parents=True, exist_ok=True)
    fixture = build_raw_fixture(fixture_profile)
    repository = Path(__file__).resolve().parents[1]
    code = _runner_code_identity(repository)
    dependencies = dependency_manifest(names=(
        "numpy", "scipy", "hdbscan", "scikit-learn", "joblib",
        "threadpoolctl", "Cython", "gtsam", "opencv-python-headless",
    ))

    success_root = root / "fresh_resume"
    prefix, prefix_context = _run_pipeline(
        root=success_root,
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        injected_failure_stage=None,
        code=code,
        dependencies=dependencies,
        stage_limit=4,
    )
    if not prefix.completed or len(prefix.records) != 4:
        raise RuntimeError(f"fresh prefix failed: {prefix.first_failure}")
    partial, partial_context = _run_pipeline(
        root=success_root,
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        injected_failure_stage=None,
        code=code,
        dependencies=dependencies,
    )
    expected_names = tuple(
        stage.name for stage in build_pipeline_stages(partial_context)
    )
    partial_resume_proved = (
        partial.completed
        and partial.resumed_stages == expected_names[:4]
        and partial_context.callback_calls == list(expected_names[4:])
    )
    if not partial_resume_proved:
        raise RuntimeError("partial resume/rehydration proof failed")

    before_checkpoints = _checkpoint_identities(success_root)
    before_records = _record_hashes(partial)
    before_outputs = _output_hashes(partial)
    resumed, resumed_context = _run_pipeline(
        root=success_root,
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        injected_failure_stage=None,
        code=code,
        dependencies=dependencies,
    )
    after_checkpoints = _checkpoint_identities(success_root)
    record_hashes_equal = before_records == _record_hashes(resumed)
    output_hashes_equal = before_outputs == _output_hashes(resumed)
    checkpoint_hashes_equal = {
        name: value["sha256"] for name, value in before_checkpoints.items()
    } == {
        name: value["sha256"] for name, value in after_checkpoints.items()
    }
    full_resume_proved = (
        resumed.completed
        and resumed.resumed_stages == expected_names
        and resumed_context.callback_calls == []
        and record_hashes_equal
        and output_hashes_equal
        and checkpoint_hashes_equal
    )
    if not full_resume_proved:
        raise RuntimeError("exact full-resume proof failed")

    final_identity = partial.records[-1].output_files["scene_evidence"]
    scene_evidence, eligible, noncanonical_reasons = (
        _load_verified_scene_evidence(final_identity, partial_context)
    )
    failure_root = root / "failure"
    failed, failure_context = _run_pipeline(
        root=failure_root,
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        injected_failure_stage=failure_stage,
        code=code,
        dependencies=dependencies,
    )
    reported_stage = {
        "colmap_execute": "colmap",
        "colmap_validate": "colmap_validation",
        "scene_report": "metrics",
    }.get(failure_stage, failure_stage)
    failure_proved = (
        not failed.completed
        and failed.first_failure is not None
        and failed.first_failure.stage == reported_stage
        and failed.records[-1].status == "failed"
        and failure_context.callback_calls[-1] == failure_stage
    )
    if not failure_proved:
        raise RuntimeError("first-failure stop proof failed")

    manifest = {
        "schema": "esfo-gate-d-synthetic-proof-v1",
        "status": (
            "passed" if eligible else
            "plumbing_passed_fixture_only"
            if colmap_mode == "fixture-only" else
            "completed_noncanonical"
        ),
        "public_data_used": False,
        "fixture": {
            "hash": fixture.fixture_hash,
            "profile": fixture.profile,
            "identity_scheme": FIXTURE_IDENTITY_SCHEME,
            "identity_quantum_decimal": "0.000000001",
            "seed": FIXTURE_SEED,
            "event_count": len(fixture.events),
            "phase_count": fixture.phase_count,
            "point_count": fixture.point_count,
            "pulses_per_phase": fixture.pulses_per_phase,
            "event_pattern": fixture.event_pattern,
        },
        "colmap": {
            "mode": colmap_mode,
            "sidecar_image": sidecar_image,
            "sidecar_identity": partial_context.sidecar_identity,
            "fixture_only": colmap_mode == "fixture-only",
            "sidecar_executed": colmap_mode == "sidecar",
            "reference_evidence_eligible": eligible,
            "noncanonical_reasons": noncanonical_reasons,
            "requires_linux_sidecar_evidence": colmap_mode != "sidecar",
        },
        "checkpoint_state": {
            "format": "trusted_local_pickle_protocol_5",
            "untrusted_pickle_permitted": False,
            "identity_verified_before_load": True,
            "code_and_dependency_identity_bound": True,
        },
        "code_identity": code,
        "dependencies": dependencies,
        "runner": file_identity(Path(__file__)),
        "fresh_prefix": {
            "completed": prefix.completed,
            "callback_calls": prefix_context.callback_calls,
            "record_hashes": _record_hashes(prefix),
            "first_failure": prefix.first_failure,
        },
        "partial_resume": {
            "completed": partial.completed,
            "resumed_stages": partial.resumed_stages,
            "callback_calls": partial_context.callback_calls,
            "proved": partial_resume_proved,
        },
        "full_resume": {
            "completed": resumed.completed,
            "resumed_stages": resumed.resumed_stages,
            "callback_calls": resumed_context.callback_calls,
            "record_hashes_equal": record_hashes_equal,
            "output_hashes_equal": output_hashes_equal,
            "checkpoint_hashes_equal": checkpoint_hashes_equal,
            "proved": full_resume_proved,
        },
        "stage_records": partial.records,
        "checkpoints": before_checkpoints,
        "failure_proof": {
            "injected_stage": failure_stage,
            "callback_calls": failure_context.callback_calls,
            "first_failure": failed.first_failure,
            "records": failed.records,
            "scene_diagnostics": None,
            "proved": failure_proved,
        },
        "scene": scene_evidence,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def run_checkpoint_phase(
    *,
    work_root: str | Path,
    action: str,
    colmap_mode: str = "fixture-only",
    sidecar_image: str = SIDECAR_IMAGE,
    repeat_check: bool = True,
    fixture_profile: str | None = None,
    stage_limit: int = 4,
) -> dict[str, Any]:
    """Run a restartable prefix or continue it in a later process.

    This is the public crash/restart entry point.  ``prefix`` requires an
    empty root and commits the requested number of ordinary stage
    checkpoints.  ``continue`` requires those checkpoints and executes the
    complete stage list with normal fingerprint/hash validation.  It never
    weakens stale-work protection for a fresh proof.
    """

    if action not in {"prefix", "continue"}:
        raise ValueError("checkpoint action must be prefix or continue")
    if colmap_mode not in {"fixture-only", "sidecar"}:
        raise ValueError("colmap_mode must be fixture-only or sidecar")
    if fixture_profile is None:
        fixture_profile = (
            "canonical-sidecar" if colmap_mode == "sidecar" else "unit"
        )
    if fixture_profile not in {"unit", "canonical-sidecar"}:
        raise ValueError("invalid fixture profile")
    if colmap_mode == "sidecar" and fixture_profile != "canonical-sidecar":
        raise ValueError(
            "sidecar reference evidence requires the canonical-sidecar fixture",
        )
    if colmap_mode == "sidecar" and platform.system() != "Linux":
        raise RuntimeError("canonical COLMAP sidecar evidence requires Linux")
    root = Path(work_root).resolve()
    if action == "prefix":
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                "prefix work root must be absent or empty; stale reuse is forbidden",
            )
        root.mkdir(parents=True, exist_ok=True)
    elif (not root.is_dir()
          or not any((root / "stages").glob("*.stage.json"))):
        raise RuntimeError("continue requires an existing checkpoint prefix")
    fixture = build_raw_fixture(fixture_profile)
    repository = Path(__file__).resolve().parents[1]
    code = _runner_code_identity(repository)
    dependencies = dependency_manifest(names=(
        "numpy", "scipy", "hdbscan", "scikit-learn", "joblib",
        "threadpoolctl", "Cython", "gtsam", "opencv-python-headless",
    ))
    stage_count = len(build_pipeline_stages(RunContext(
        fixture, colmap_mode, sidecar_image, repeat_check,
    )))
    if (isinstance(stage_limit, bool) or not isinstance(stage_limit, int)
            or stage_limit < 1 or stage_limit > stage_count):
        raise ValueError(f"stage_limit must lie in [1,{stage_count}]")
    result, context = _run_pipeline(
        root=root,
        fixture=fixture,
        colmap_mode=colmap_mode,
        sidecar_image=sidecar_image,
        repeat_check=repeat_check,
        injected_failure_stage=None,
        code=code,
        dependencies=dependencies,
        stage_limit=stage_limit if action == "prefix" else None,
    )
    evidence = {
        "schema": "esfo-gate-d-checkpoint-phase-v1",
        "action": action,
        "completed": result.completed,
        "fixture_hash": fixture.fixture_hash,
        "fixture_profile": fixture.profile,
        "resumed_stages": result.resumed_stages,
        "callback_calls": context.callback_calls,
        "records": result.records,
        "first_failure": result.first_failure,
        "checkpoint_hashes": _checkpoint_identities(root),
    }
    atomic_write_json(root / f"{action}_phase.json", evidence)
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--colmap-mode", choices=("fixture-only", "sidecar"),
        default="fixture-only",
    )
    parser.add_argument("--sidecar-image", default=SIDECAR_IMAGE)
    parser.add_argument(
        "--fixture-profile", choices=("unit", "canonical-sidecar"),
        default=None,
    )
    parser.add_argument(
        "--checkpoint-action", choices=("proof", "prefix", "continue"),
        default="proof",
    )
    parser.add_argument("--stage-limit", type=int, default=4)
    parser.add_argument(
        "--failure-stage",
        choices=(
            "input", "frontend", "colmap_execute", "colmap_validate",
            "frequency", "geometry", "optimization", "scene_report",
        ),
        default="frontend",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    if arguments.checkpoint_action != "proof":
        run_checkpoint_phase(
            work_root=arguments.work_root,
            action=arguments.checkpoint_action,
            colmap_mode=arguments.colmap_mode,
            sidecar_image=arguments.sidecar_image,
            repeat_check=True,
            fixture_profile=arguments.fixture_profile,
            stage_limit=arguments.stage_limit,
        )
        return
    run_proof(
        work_root=arguments.work_root,
        colmap_mode=arguments.colmap_mode,
        sidecar_image=arguments.sidecar_image,
        failure_stage=arguments.failure_stage,
        repeat_check=True,
        fixture_profile=arguments.fixture_profile,
    )


if __name__ == "__main__":
    main()





