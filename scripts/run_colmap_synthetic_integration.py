#!/usr/bin/env python3
"""Run the preregistered Gate-B COLMAP 3.9.1 synthetic integration.

This script uses only deterministic generated geometry.  It deliberately does
not read TOPSPIN or any other public performance data.  The fixed tolerances
below are executable preregistration, not values selected after a run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np

from esfo.colmap import (
    COLMAP_TAG_COMMIT,
    COLMAP_VERSION,
    OPENCV_DISTRIBUTION,
    OPENCV_DISTRIBUTION_VERSION,
    OPENCV_VERSION,
    Calibration,
    ExtractedColmapModel,
    IndexedView,
    assert_colmap_version,
    build_colmap_commands,
    build_indexed_views,
    build_model_converter_command,
    discover_sparse_models,
    expected_raw_correspondence_count,
    extract_text_model,
    populate_known_camera_database,
    select_largest_model,
    undistort_points,
    validate_fixed_camera_database,
    validate_geometric_verification,
    validate_imported_raw_matches,
    validate_text_model_camera,
    write_matches_importer_file,
)
from esfo.provenance import (
    atomic_write_json,
    canonical_hash,
    file_identity,
    source_tree_hash,
)
from esfo.tracks import TrackObservation


SYNTHETIC_SEED = 20_241_124
WIDTH = 640
HEIGHT = 480
CORE_VIEW_COUNT = 12
SPARSE_VIEW_COUNT = 2
EMPTY_VIEW_INDEX = 15
CORE_POINT_COUNT = 200
SPARSE_SHARED_TRACK_COUNT = 14
SINGLETON_TRACK_ID = 10_000

# Fixed before the sidecar is run.  The input keypoints are float32 in the
# COLMAP database, so the reprojection tolerance is intentionally looser than
# the gauge-aligned double-precision geometry tolerance.
MAX_CAMERA_CENTER_RMSE = 5e-3
MAX_POINT_RMSE = 5e-3
MAX_ORIENTATION_ERROR_RAD = 2e-3
MAX_REPROJECTION_RMSE_PX = 5e-2
MAX_REPEAT_ALIGNED_DIFFERENCE = 1e-7
MAX_OPENCV_ROUNDTRIP_ERROR_PX = 1e-6


@dataclass(frozen=True)
class SyntheticFixture:
    calibration: Calibration
    views: tuple[IndexedView, ...]
    true_centers_w: dict[int, np.ndarray]
    true_rotations_cw: dict[int, np.ndarray]
    true_points_w: dict[int, np.ndarray]
    fixture_hash: str


@dataclass(frozen=True)
class CommandRecord:
    argv: tuple[str, ...]
    elapsed_s: float
    returncode: int
    stdout_path: str
    stderr_path: str


def _look_at_rotation(center_w: np.ndarray) -> np.ndarray:
    forward = -np.asarray(center_w, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-14, rtol=0.0):
        raise RuntimeError("synthetic look-at rotation is not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise RuntimeError("synthetic look-at rotation is improper")
    return rotation


def _project(
    calibration: Calibration,
    rotation_cw: np.ndarray,
    center_w: np.ndarray,
    point_w: np.ndarray,
) -> np.ndarray:
    camera = rotation_cw @ (point_w - center_w)
    if camera[2] <= 0.0:
        raise RuntimeError("synthetic point lies behind a camera")
    return np.asarray([
        calibration.fx * camera[0] / camera[2] + calibration.cx,
        calibration.fy * camera[1] / camera[2] + calibration.cy,
    ])


def build_fixture() -> SyntheticFixture:
    calibration = Calibration(
        400.0, 405.0, WIDTH / 2.0, HEIGHT / 2.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )
    rng = np.random.default_rng(SYNTHETIC_SEED)
    points = rng.uniform(-1.0, 1.0, size=(CORE_POINT_COUNT, 3))
    # A fixed anisotropic perturbation avoids accidental symmetries while
    # remaining well within every camera's field of view.
    points[:, 0] += 0.11 * points[:, 1] * points[:, 2]
    points[:, 1] += 0.07 * points[:, 0] ** 2
    true_points = {index: points[index].copy()
                   for index in range(CORE_POINT_COUNT)}
    true_centers: dict[int, np.ndarray] = {}
    true_rotations: dict[int, np.ndarray] = {}
    observations: list[TrackObservation] = []
    total_nonempty = CORE_VIEW_COUNT + SPARSE_VIEW_COUNT
    for zero_index in range(total_nonempty):
        view_id = zero_index + 1
        angle = 2.0 * math.pi * zero_index / total_nonempty
        center = np.asarray([
            7.0 * math.cos(angle), 7.0 * math.sin(angle),
            2.2 + 0.25 * math.sin(2.0 * angle),
        ])
        rotation = _look_at_rotation(center)
        true_centers[view_id] = center
        true_rotations[view_id] = rotation
        visible_count = (
            CORE_POINT_COUNT if view_id <= CORE_VIEW_COUNT
            else SPARSE_SHARED_TRACK_COUNT
        )
        for track_id in range(visible_count):
            pixel = _project(calibration, rotation, center, points[track_id])
            if not (0.0 < pixel[0] < WIDTH - 1.0
                    and 0.0 < pixel[1] < HEIGHT - 1.0):
                raise RuntimeError("synthetic point lies outside the sensor")
            observations.append(TrackObservation(
                track_id, view_id, float(30 * view_id),
                float(pixel[0]), float(pixel[1]), 1,
            ))
    singleton_pixel = _project(
        calibration, true_rotations[1], true_centers[1],
        np.asarray([0.15, -0.12, 0.21]),
    )
    observations.append(TrackObservation(
        SINGLETON_TRACK_ID, 1, 30.0,
        float(singleton_pixel[0]), float(singleton_pixel[1]), 1,
    ))
    views = build_indexed_views(
        observations, max_window_index=EMPTY_VIEW_INDEX,
    )
    payload = {
        "seed": SYNTHETIC_SEED,
        "K": calibration.camera_matrix,
        "views": [
            {
                "image_id": view.image_id,
                "tracks": view.track_ids,
                "keypoints": view.keypoints_xy,
            }
            for view in views
        ],
        "centers": true_centers,
        "rotations": true_rotations,
        "points": true_points,
    }
    expected_matches = (
        SPARSE_SHARED_TRACK_COUNT * math.comb(14, 2)
        + (CORE_POINT_COUNT - SPARSE_SHARED_TRACK_COUNT) * math.comb(12, 2)
    )
    actual_matches = expected_raw_correspondence_count(views)
    if actual_matches != expected_matches or actual_matches != 13_550:
        raise RuntimeError(
            f"synthetic correspondence count changed: {actual_matches}"
        )
    return SyntheticFixture(
        calibration, views, true_centers, true_rotations, true_points,
        canonical_hash(payload),
    )


def _docker_base(image: str, work_root: Path) -> list[str]:
    return [
        "docker", "run", "--rm", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp",
        "--volume", f"{work_root.resolve()}:/work", image,
    ]


def _containerize_path(path: Path, work_root: Path) -> str:
    return "/work/" + str(path.resolve().relative_to(work_root.resolve()))


def _run_colmap(
    *,
    image: str,
    work_root: Path,
    colmap_argv: Sequence[str],
    log_prefix: Path,
) -> CommandRecord:
    if not colmap_argv or colmap_argv[0] != "colmap":
        raise ValueError("COLMAP argv must start with the logical executable name")
    command = _docker_base(image, work_root) + list(colmap_argv[1:])
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    stdout_path = log_prefix.with_suffix(".stdout.txt")
    stderr_path = log_prefix.with_suffix(".stderr.txt")
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = CommandRecord(
        tuple(command), elapsed, int(completed.returncode),
        str(stdout_path), str(stderr_path),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed ({completed.returncode}); see {stderr_path}"
        )
    return record


def _docker_capture(
    image: str,
    args: Sequence[str],
    *,
    entrypoint: str | None = None,
) -> str:
    command = ["docker", "run", "--rm", "--network", "none"]
    if entrypoint is not None:
        command += ["--entrypoint", entrypoint]
    command += [image, *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return completed.stdout + completed.stderr


def collect_sidecar_identity(image: str, repository_root: Path) -> dict[str, Any]:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, check=True,
    )
    image_record = json.loads(inspected.stdout)[0]
    help_output = _docker_capture(image, ["--help"])
    assert_colmap_version(help_output)
    diagnostics = _docker_capture(
        image,
        ["-lc", (
            "set -eu; sha256sum /usr/local/bin/colmap; "
            "ldd /usr/local/bin/colmap; "
            "c++ --version | head -n 1; cmake --version | head -n 1; "
            "git -C /opt/colmap-src rev-parse HEAD; "
            "dpkg-query -W -f='${Package}=${Version}\\n' | sort"
        )],
        entrypoint="/bin/bash",
    )
    if COLMAP_TAG_COMMIT not in diagnostics:
        raise RuntimeError("sidecar source commit does not match the frozen tag")
    labels = image_record.get("Config", {}).get("Labels", {}) or {}
    if labels.get("org.opencontainers.image.revision") != COLMAP_TAG_COMMIT:
        raise RuntimeError("sidecar OCI revision label is missing or incorrect")
    return {
        "logical_name": image,
        "image_id": image_record.get("Id"),
        "repo_digests": image_record.get("RepoDigests") or [],
        "created": image_record.get("Created"),
        "labels": labels,
        "colmap_version": COLMAP_VERSION,
        "colmap_commit": COLMAP_TAG_COMMIT,
        "help_sha256": canonical_hash(help_output),
        "diagnostics": diagnostics,
        "dockerfile": file_identity(
            repository_root / "containers" / "colmap-3.9.1" / "Dockerfile"
        ),
    }


def validate_real_opencv() -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "install the exact colmap extra: opencv-python-headless==4.9.0.80"
        ) from error
    if cv2.__version__ != OPENCV_VERSION:
        raise RuntimeError(
            f"expected OpenCV {OPENCV_VERSION}, got {cv2.__version__}"
        )
    try:
        distribution_version = importlib.metadata.version(OPENCV_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"missing frozen distribution {OPENCV_DISTRIBUTION}"
        ) from error
    if distribution_version != OPENCV_DISTRIBUTION_VERSION:
        raise RuntimeError(
            "expected "
            f"{OPENCV_DISTRIBUTION}=={OPENCV_DISTRIBUTION_VERSION}, "
            f"got {distribution_version}"
        )
    zero = Calibration(400, 405, 320, 240, 0, 0, 0, 0, 0)
    pixels = np.asarray([[120.25, 80.75], [320.0, 240.0], [510.5, 370.25]])
    identity = undistort_points(pixels, zero, cv2_module=cv2)
    zero_error = float(np.max(np.abs(identity - pixels)))

    distorted_calibration = Calibration(
        400, 405, 320, 240, 0.08, -0.03, 0.002, -0.0015, 0.006,
    )
    normalized = np.asarray([
        [-0.20, 0.15], [0.25, -0.10], [0.0, 0.0], [0.12, 0.21],
    ])
    x, y = normalized[:, 0], normalized[:, 1]
    radius2 = x * x + y * y
    radial = (1.0 + distorted_calibration.k1 * radius2
              + distorted_calibration.k2 * radius2 ** 2
              + distorted_calibration.k3 * radius2 ** 3)
    distorted_x = (x * radial + 2.0 * distorted_calibration.p1 * x * y
                   + distorted_calibration.p2 * (radius2 + 2.0 * x * x))
    distorted_y = (y * radial + distorted_calibration.p1 * (radius2 + 2.0 * y * y)
                   + 2.0 * distorted_calibration.p2 * x * y)
    distorted_pixels = np.column_stack((
        distorted_calibration.fx * distorted_x + distorted_calibration.cx,
        distorted_calibration.fy * distorted_y + distorted_calibration.cy,
    ))
    expected = np.column_stack((
        distorted_calibration.fx * x + distorted_calibration.cx,
        distorted_calibration.fy * y + distorted_calibration.cy,
    ))
    recovered = undistort_points(
        distorted_pixels, distorted_calibration, cv2_module=cv2,
    )
    distortion_error = float(np.max(np.abs(recovered - expected)))
    if (zero_error > MAX_OPENCV_ROUNDTRIP_ERROR_PX
            or distortion_error > MAX_OPENCV_ROUNDTRIP_ERROR_PX):
        raise RuntimeError(
            "OpenCV distortion round trip exceeded the preregistered tolerance"
        )
    return {
        "distribution": OPENCV_DISTRIBUTION,
        "distribution_version": distribution_version,
        "version": cv2.__version__,
        "zero_distortion_max_error_px": zero_error,
        "known_distortion_max_error_px": distortion_error,
        "tolerance_px": MAX_OPENCV_ROUNDTRIP_ERROR_PX,
    }


def _umeyama_similarity(
    source: np.ndarray, target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("similarity inputs must have matching [N,3] shape")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular, right_t = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_t) < 0.0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_t
    variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    if variance <= 0.0:
        raise RuntimeError("similarity source has zero variance")
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _apply_similarity(
    points: np.ndarray, scale: float, rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def evaluate_reconstruction(
    extracted: ExtractedColmapModel,
    fixture: SyntheticFixture,
) -> dict[str, Any]:
    registered = [int(value) for value in extracted.trajectory.view_ids]
    expected_registered = list(range(1, CORE_VIEW_COUNT + 1))
    if registered != expected_registered:
        raise RuntimeError(
            f"registered views changed: expected {expected_registered}, got {registered}"
        )
    expected_unregistered = tuple(range(CORE_VIEW_COUNT + 1, EMPTY_VIEW_INDEX + 1))
    if extracted.unregistered_window_indices != expected_unregistered:
        raise RuntimeError(
            "sparse/empty views did not remain unregistered as preregistered"
        )
    mapped_tracks = [int(value) for value in extracted.mapped_track_ids]
    if mapped_tracks != list(range(CORE_POINT_COUNT)):
        raise RuntimeError("not every core track was reconstructed bijectively")
    if SINGLETON_TRACK_ID not in extracted.mapping.uninitialized_track_ids:
        raise RuntimeError("singleton track unexpectedly initialized")

    reconstructed_centers = extracted.trajectory.centers_w
    true_centers = np.stack([
        fixture.true_centers_w[image_id] for image_id in registered
    ])
    scale, frame_rotation, translation = _umeyama_similarity(
        reconstructed_centers, true_centers,
    )
    aligned_centers = _apply_similarity(
        reconstructed_centers, scale, frame_rotation, translation,
    )
    center_error = aligned_centers - true_centers
    center_rmse = float(np.sqrt(np.mean(np.sum(center_error ** 2, axis=1))))

    true_points = np.stack([
        fixture.true_points_w[int(track_id)] for track_id in mapped_tracks
    ])
    aligned_points = _apply_similarity(
        extracted.mapped_points_w, scale, frame_rotation, translation,
    )
    point_error = aligned_points - true_points
    point_rmse = float(np.sqrt(np.mean(np.sum(point_error ** 2, axis=1))))

    orientation_errors: list[float] = []
    for row, image_id in enumerate(registered):
        aligned_rotation_cw = extracted.trajectory.R_cw[row] @ frame_rotation.T
        delta = fixture.true_rotations_cw[image_id] @ aligned_rotation_cw.T
        orientation_errors.append(_rotation_angle(delta))
    max_orientation_error = max(orientation_errors)

    point_by_track = {
        int(track_id): extracted.mapped_points_w[row]
        for row, track_id in enumerate(extracted.mapped_track_ids)
    }
    pose_row = {image_id: row for row, image_id in enumerate(registered)}
    squared_pixel_errors: list[float] = []
    for view in fixture.views[:CORE_VIEW_COUNT]:
        row = pose_row[view.image_id]
        for feature_index, track_id in enumerate(view.track_ids):
            if int(track_id) not in point_by_track:
                continue
            point = point_by_track[int(track_id)]
            camera = (extracted.trajectory.R_cw[row] @ point
                      + extracted.trajectory.t_cw[row])
            if camera[2] <= 0.0:
                raise RuntimeError("reconstructed point has negative depth")
            pixel = np.asarray([
                fixture.calibration.fx * camera[0] / camera[2]
                + fixture.calibration.cx,
                fixture.calibration.fy * camera[1] / camera[2]
                + fixture.calibration.cy,
            ])
            reference = np.asarray(view.keypoints_xy[feature_index],
                                   dtype=np.float32).astype(np.float64)
            squared_pixel_errors.append(float(np.sum((pixel - reference) ** 2)))
    reprojection_rmse = float(math.sqrt(np.mean(squared_pixel_errors)))

    if center_rmse > MAX_CAMERA_CENTER_RMSE:
        raise RuntimeError("camera-center RMSE exceeded preregistered tolerance")
    if point_rmse > MAX_POINT_RMSE:
        raise RuntimeError("point RMSE exceeded preregistered tolerance")
    if max_orientation_error > MAX_ORIENTATION_ERROR_RAD:
        raise RuntimeError("orientation error exceeded preregistered tolerance")
    if reprojection_rmse > MAX_REPROJECTION_RMSE_PX:
        raise RuntimeError("reprojection RMSE exceeded preregistered tolerance")

    semantic_arrays = {
        "aligned_centers": np.round(aligned_centers, decimals=9),
        "aligned_points_by_track": np.round(aligned_points, decimals=9),
        "registered": registered,
        "mapped_tracks": mapped_tracks,
    }
    return {
        "registered_image_ids": registered,
        "unregistered_window_indices": list(extracted.unregistered_window_indices),
        "mapped_track_count": len(mapped_tracks),
        "singleton_uninitialized": True,
        "similarity_scale": scale,
        "camera_center_rmse": center_rmse,
        "point_rmse": point_rmse,
        "max_orientation_error_rad": max_orientation_error,
        "reprojection_rmse_px": reprojection_rmse,
        "semantic_hash": canonical_hash(semantic_arrays),
        "aligned_centers": aligned_centers.tolist(),
        "aligned_points_by_track": aligned_points.tolist(),
    }


def _record_to_dict(record: CommandRecord) -> dict[str, Any]:
    return {
        "argv": list(record.argv),
        "elapsed_s": record.elapsed_s,
        "returncode": record.returncode,
        "stdout_path": record.stdout_path,
        "stderr_path": record.stderr_path,
    }


def run_once(
    *,
    image: str,
    work_root: Path,
    run_index: int,
    fixture: SyntheticFixture,
) -> dict[str, Any]:
    run_root = work_root / f"run_{run_index:02d}"
    run_root.mkdir()
    database = run_root / "database.db"
    matches = run_root / "all_pairs.txt"
    empty_images = run_root / "empty_images"
    models = run_root / "models"
    text_models = run_root / "models_txt"
    logs = run_root / "logs"
    for directory in (empty_images, models, text_models, logs):
        directory.mkdir()
    plan = build_colmap_commands(
        database_path=_containerize_path(database, work_root),
        match_list_path=_containerize_path(matches, work_root),
        image_path=_containerize_path(empty_images, work_root),
        output_path=_containerize_path(models, work_root),
    )
    commands: list[CommandRecord] = []
    commands.append(_run_colmap(
        image=image, work_root=work_root,
        colmap_argv=plan.database_creator,
        log_prefix=logs / "01_database_creator",
    ))
    populate_known_camera_database(
        database, calibration=fixture.calibration,
        width=WIDTH, height=HEIGHT, views=fixture.views,
    )
    validate_fixed_camera_database(
        database, calibration=fixture.calibration, width=WIDTH, height=HEIGHT,
    )
    with sqlite3.connect(database) as connection:
        schema_user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        descriptor_rows = int(connection.execute(
            "SELECT COUNT(*) FROM descriptors",
        ).fetchone()[0])
    if descriptor_rows != 0:
        raise RuntimeError("descriptor rows unexpectedly exist before raw import")
    pair_count, raw_correspondence_count = write_matches_importer_file(
        matches, fixture.views,
    )
    commands.append(_run_colmap(
        image=image, work_root=work_root,
        colmap_argv=plan.matches_importer,
        log_prefix=logs / "02_matches_importer",
    ))
    validate_imported_raw_matches(database, fixture.views)
    verified_counts = validate_geometric_verification(database, fixture.views)
    with sqlite3.connect(database) as connection:
        descriptor_rows_after = int(connection.execute(
            "SELECT COUNT(*) FROM descriptors",
        ).fetchone()[0])
    if descriptor_rows_after != 0:
        raise RuntimeError("raw importer unexpectedly created descriptor rows")
    positive_pairs = {
        int(pair_id): int(count) for pair_id, count in verified_counts.items()
        if count > 0
    }
    if len(positive_pairs) != math.comb(CORE_VIEW_COUNT, 2):
        raise RuntimeError("unexpected number of positively verified core pairs")
    commands.append(_run_colmap(
        image=image, work_root=work_root,
        colmap_argv=plan.mapper,
        log_prefix=logs / "03_mapper",
    ))
    summaries = discover_sparse_models(models)
    selected = select_largest_model(summaries)
    extracted_by_id: dict[int, ExtractedColmapModel] = {}
    for summary in summaries:
        output = text_models / str(summary.directory_id)
        output.mkdir()
        converter = build_model_converter_command(
            input_path=_containerize_path(models / str(summary.directory_id), work_root),
            output_path=_containerize_path(output, work_root),
        )
        commands.append(_run_colmap(
            image=image, work_root=work_root, colmap_argv=converter,
            log_prefix=logs / f"04_model_converter_{summary.directory_id}",
        ))
        validate_text_model_camera(
            output, calibration=fixture.calibration, width=WIDTH, height=HEIGHT,
        )
        extracted_by_id[summary.directory_id] = extract_text_model(
            output, summary=summary, views=fixture.views,
            calibration=fixture.calibration, width=WIDTH, height=HEIGHT,
        )
    evaluation = evaluate_reconstruction(
        extracted_by_id[selected.directory_id], fixture,
    )
    return {
        "run_index": run_index,
        "schema_user_version": schema_user_version,
        "descriptor_rows_before_import": descriptor_rows,
        "descriptor_rows_after_import": descriptor_rows_after,
        "pair_count": pair_count,
        "raw_correspondence_count": raw_correspondence_count,
        "verified_pair_counts": {
            str(pair_id): int(count)
            for pair_id, count in verified_counts.items()
        },
        "positive_verified_pair_count": len(positive_pairs),
        "models": [
            {
                "directory_id": item.directory_id,
                "registered_image_count": item.registered_image_count,
                "point3d_count": item.point3d_count,
            }
            for item in summaries
        ],
        "selected_model_id": selected.directory_id,
        "colmap_plan_metadata": plan.metadata,
        "commands": [_record_to_dict(record) for record in commands],
        "evaluation": evaluation,
    }


def collect_output_file_identities(work_root: Path) -> dict[str, dict[str, Any]]:
    """Bind every non-manifest synthetic output to the final manifest."""

    identities: dict[str, dict[str, Any]] = {}
    for path in sorted(work_root.rglob("*")):
        if (not path.is_file()
                or path.name in {"manifest.json", "manifest.partial.json"}):
            continue
        identity = file_identity(path)
        identities[str(path.relative_to(work_root))] = {
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
        }
    return identities


def compare_repeats(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 2:
        return {"repeat_count": len(results), "compared": False}
    baseline = results[0]
    baseline_evaluation = baseline["evaluation"]
    max_center_difference = 0.0
    max_point_difference = 0.0
    for candidate in results[1:]:
        for key in (
            "pair_count", "raw_correspondence_count",
            "verified_pair_counts", "models", "selected_model_id",
        ):
            if candidate[key] != baseline[key]:
                raise RuntimeError(f"repeat semantic mismatch in {key}")
        evaluation = candidate["evaluation"]
        for key in (
            "registered_image_ids", "unregistered_window_indices",
            "mapped_track_count",
        ):
            if evaluation[key] != baseline_evaluation[key]:
                raise RuntimeError(f"repeat evaluation mismatch in {key}")
        center_difference = float(np.max(np.abs(
            np.asarray(evaluation["aligned_centers"])
            - np.asarray(baseline_evaluation["aligned_centers"])
        )))
        point_difference = float(np.max(np.abs(
            np.asarray(evaluation["aligned_points_by_track"])
            - np.asarray(baseline_evaluation["aligned_points_by_track"])
        )))
        max_center_difference = max(max_center_difference, center_difference)
        max_point_difference = max(max_point_difference, point_difference)
    if (max_center_difference > MAX_REPEAT_ALIGNED_DIFFERENCE
            or max_point_difference > MAX_REPEAT_ALIGNED_DIFFERENCE):
        raise RuntimeError("repeat reconstruction exceeded semantic tolerance")
    return {
        "repeat_count": len(results),
        "compared": True,
        "max_aligned_center_difference": max_center_difference,
        "max_aligned_point_difference": max_point_difference,
        "tolerance": MAX_REPEAT_ALIGNED_DIFFERENCE,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--image", default="esfo-colmap:3.9.1-cpu")
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.repeat <= 0:
        raise ValueError("--repeat must be positive")
    work_root = arguments.work_root.resolve()
    if work_root.exists() and any(work_root.iterdir()):
        raise RuntimeError("work root must be absent or empty; stale reuse is forbidden")
    work_root.mkdir(parents=True, exist_ok=True)
    repository_root = Path(__file__).resolve().parents[1]
    fixture = build_fixture()
    manifest: dict[str, Any] = {
        "schema": "esfo-colmap-gate-b-synthetic-v1",
        "public_data_used": False,
        "fixture": {
            "seed": SYNTHETIC_SEED,
            "hash": fixture.fixture_hash,
            "core_views": CORE_VIEW_COUNT,
            "sparse_views": SPARSE_VIEW_COUNT,
            "empty_view_index": EMPTY_VIEW_INDEX,
            "core_points": CORE_POINT_COUNT,
            "sparse_shared_tracks": SPARSE_SHARED_TRACK_COUNT,
            "singleton_track_id": SINGLETON_TRACK_ID,
        },
        "preregistered_tolerances": {
            "camera_center_rmse": MAX_CAMERA_CENTER_RMSE,
            "point_rmse": MAX_POINT_RMSE,
            "orientation_error_rad": MAX_ORIENTATION_ERROR_RAD,
            "reprojection_rmse_px": MAX_REPROJECTION_RMSE_PX,
            "repeat_aligned_difference": MAX_REPEAT_ALIGNED_DIFFERENCE,
            "opencv_roundtrip_error_px": MAX_OPENCV_ROUNDTRIP_ERROR_PX,
        },
        "source_tree_sha256": source_tree_hash(
            repository_root,
            roots=("esfo", "tests", "scripts", "containers"),
        ),
        "integration_script": file_identity(Path(__file__)),
        "sidecar": collect_sidecar_identity(arguments.image, repository_root),
        "opencv": validate_real_opencv(),
    }
    atomic_write_json(work_root / "manifest.partial.json", manifest)
    results = [
        run_once(
            image=arguments.image, work_root=work_root,
            run_index=index, fixture=fixture,
        )
        for index in range(1, arguments.repeat + 1)
    ]
    manifest["runs"] = results
    manifest["repeat_comparison"] = compare_repeats(results)
    manifest["output_files"] = collect_output_file_identities(work_root)
    manifest["status"] = "passed"
    atomic_write_json(work_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
