#!/usr/bin/env python3
"""Run the deterministic Gate-D N6 scale/resource benchmark.

This script uses generated data only.  It measures three explicitly scoped
operations:

* whole-stream reference HDBSCAN with no external chunking/subsampling; and
* complete all-pairs raw-correspondence materialization and auxiliary SQLite
  injection using the pinned COLMAP database representation; and
* on canonical Linux runs, the actual pinned COLMAP 3.9.1
  ``matches_importer`` including geometric verification, measured from its
  in-container ``/proc`` process tree.

The auxiliary injection deliberately stops before geometric verification and
the mapper.  Its manifest labels that boundary explicitly; it must not be
reported as total COLMAP runtime or memory.  The separately named sidecar
measurement includes verification but still excludes the mapper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sqlite3
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from esfo.colmap import (
    COLMAP_TAG_COMMIT, COLMAP_VERSION, Calibration,
    IndexedView,
    assert_colmap_version,
    build_colmap_commands,
    expected_raw_correspondence_count,
    iter_pair_matches,
    populate_known_camera_database,
    validate_geometric_verification,
    validate_imported_raw_matches,
    write_matches_importer_file,
)
from esfo.data import RelativeEventSequence
from esfo.frontend import cluster_corners, hdbscan_parameters
from esfo.provenance import (
    atomic_write_json,
    canonical_hash,
    code_identity,
    dependency_manifest,
    file_identity,
)
from esfo.resources import ResourceMeasurement, measure_subprocess


SCALE_SEED = 20_241_124
DEFAULT_EVENT_COUNT = 168_960
DEFAULT_CLUSTER_COUNT = 32
DEFAULT_VIEW_COUNT = 128
DEFAULT_TRACK_COUNT = 512
SENSOR_WIDTH = 640
SENSOR_HEIGHT = 480
SIDECAR_IMAGE = "esfo-colmap:3.9.1-cpu"
SCALE_IDENTITY_QUANTUM = 1e-9
SCALE_IDENTITY_SCHEME = (
    "recipe_plus_integer_fields_plus_round_to_nearest_1e-9_v1"
)
CANONICAL_HDBSCAN_FIXTURE_HASH = (
    "6e9bc65b0375506f96e69eb766c4b1384fca6f4e417e837f38445dfc85ee9263"
)
CANONICAL_COLMAP_FIXTURE_HASH = (
    "f9dc9f77434273f8470a42988e815cfd032238d9d2cedb3ee35eb15bffdd1df6"
)
SCALE_DEPENDENCIES = (
    "numpy", "scipy", "hdbscan", "scikit-learn", "joblib",
    "threadpoolctl", "Cython",
)


def _semantic_float_grid(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("scale identity input contains non-finite values")
    scaled = np.rint(array / SCALE_IDENTITY_QUANTUM)
    if np.any(np.abs(scaled) > np.iinfo(np.int64).max):
        raise ValueError("scale identity input exceeds int64 range")
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


def _scale_reference_eligibility(
    *,
    arguments: argparse.Namespace,
    code: dict[str, Any],
    host_measurements: Sequence[ResourceMeasurement],
    hdbscan_worker: dict[str, Any],
    colmap_worker: dict[str, Any],
    sidecar_importer: dict[str, Any] | None,
    hdbscan_fixture_hash: str,
    colmap_fixture_hash: str,
) -> tuple[bool, list[str]]:
    """Fail closed unless this is the preregistered canonical N6 run."""

    reasons: list[str] = []
    if code.get("git_dirty") is not False:
        reasons.append("dirty_worktree")
    if not isinstance(code.get("git_revision"), str) or not code["git_revision"]:
        reasons.append("missing_git_revision")
    if not _is_sha256(code.get("source_tree_sha256")):
        reasons.append("missing_source_tree_sha256")
    if platform.system() != "Linux":
        reasons.append("non_linux_host")
    dimensions = (
        int(arguments.event_count), int(arguments.cluster_count),
        int(arguments.view_count), int(arguments.track_count),
    )
    if dimensions != (
        DEFAULT_EVENT_COUNT, DEFAULT_CLUSTER_COUNT,
        DEFAULT_VIEW_COUNT, DEFAULT_TRACK_COUNT,
    ):
        reasons.append("noncanonical_dimensions")
    if float(arguments.sample_interval_ms) != 10.0:
        reasons.append("noncanonical_sample_interval_ms")
    if hdbscan_fixture_hash != CANONICAL_HDBSCAN_FIXTURE_HASH:
        reasons.append("noncanonical_hdbscan_fixture_hash")
    if colmap_fixture_hash != CANONICAL_COLMAP_FIXTURE_HASH:
        reasons.append("noncanonical_colmap_fixture_hash")
    if not (
        len(host_measurements) == 2
        and all(
            measurement.platform == "Linux"
            and measurement.measurement_method
            == "linux_proc_process_tree_sampled"
            and measurement.sample_interval_ms == 10.0
            and measurement.returncode == 0
            and measurement.timed_out is False
            and measurement.wall_time_s > 0.0
            and measurement.peak_rss_bytes > 0
            for measurement in host_measurements
        )
    ):
        reasons.append("invalid_host_process_tree_measurements")
    expected_pairs = DEFAULT_VIEW_COUNT * (DEFAULT_VIEW_COUNT - 1) // 2
    expected_rows = expected_pairs * DEFAULT_TRACK_COUNT
    if not (
        hdbscan_worker.get("fixture_hash") == CANONICAL_HDBSCAN_FIXTURE_HASH
        and hdbscan_worker.get("event_count") == DEFAULT_EVENT_COUNT
        and hdbscan_worker.get("requested_cluster_count")
        == DEFAULT_CLUSTER_COUNT
        and hdbscan_worker.get("result_cluster_count")
        == DEFAULT_CLUSTER_COUNT
        and hdbscan_worker.get("noise_count") == 0
        and hdbscan_worker.get("whole_sequence") is True
        and hdbscan_worker.get("external_chunking") is False
        and hdbscan_worker.get("external_subsampling") is False
        and hdbscan_worker.get("external_approximation") is False
    ):
        reasons.append("invalid_canonical_hdbscan_outcome")
    if not (
        colmap_worker.get("fixture_hash") == CANONICAL_COLMAP_FIXTURE_HASH
        and colmap_worker.get("view_count") == DEFAULT_VIEW_COUNT
        and colmap_worker.get("track_count") == DEFAULT_TRACK_COUNT
        and colmap_worker.get("all_pairs_image_pair_count") == expected_pairs
        and colmap_worker.get("raw_correspondence_row_count") == expected_rows
        and colmap_worker.get("theoretical_raw_correspondence_row_count")
        == expected_rows
        and colmap_worker.get("whole_sequence") is True
        and colmap_worker.get("external_chunking") is False
        and colmap_worker.get("external_subsampling") is False
        and colmap_worker.get("external_approximation") is False
        and colmap_worker.get("includes_geometric_verification") is False
        and colmap_worker.get("includes_mapper") is False
        and colmap_worker.get("measurement_scope")
        == "all_pairs_materialization_and_raw_sqlite_injection_only"
    ):
        reasons.append("invalid_canonical_colmap_auxiliary_outcome")
    importer_measurement = (
        sidecar_importer.get("measurement")
        if isinstance(sidecar_importer, dict) else None
    )
    if not (
        isinstance(importer_measurement, dict)
        and importer_measurement.get("platform") == "Linux"
        and importer_measurement.get("measurement_method")
        == "linux_container_proc_process_tree_sampled"
        and importer_measurement.get("sample_interval_ms") == 10.0
        and importer_measurement.get("returncode") == 0
        and importer_measurement.get("timed_out") is False
        and importer_measurement.get("wall_time_s", 0.0) > 0.0
        and importer_measurement.get("peak_rss_bytes", 0) > 0
    ):
        reasons.append("invalid_actual_importer_process_tree_measurement")
    if not (
        isinstance(sidecar_importer, dict)
        and sidecar_importer.get("fixture_hash")
        == CANONICAL_COLMAP_FIXTURE_HASH
        and sidecar_importer.get("view_count") == DEFAULT_VIEW_COUNT
        and sidecar_importer.get("track_count") == DEFAULT_TRACK_COUNT
        and sidecar_importer.get("all_pairs_image_pair_count") == expected_pairs
        and sidecar_importer.get("raw_correspondence_row_count") == expected_rows
        and sidecar_importer.get("verified_pair_count") == expected_pairs
        and sidecar_importer.get("verified_inlier_row_count") == expected_rows
        and sidecar_importer.get("whole_sequence") is True
        and sidecar_importer.get("external_chunking") is False
        and sidecar_importer.get("external_subsampling") is False
        and sidecar_importer.get("external_approximation") is False
        and sidecar_importer.get("includes_geometric_verification") is True
        and sidecar_importer.get("includes_mapper") is False
        and sidecar_importer.get("measurement_scope")
        == "pinned_colmap_3_9_1_matches_importer_raw"
    ):
        reasons.append("invalid_canonical_actual_importer_outcome")
    identity = (
        sidecar_importer.get("sidecar_identity")
        if isinstance(sidecar_importer, dict) else None
    )
    if not (
        isinstance(identity, dict)
        and identity.get("logical_name") == arguments.sidecar_image
        and _is_docker_image_id(identity.get("image_id"))
        and identity.get("colmap_version") == COLMAP_VERSION
        and identity.get("revision") == COLMAP_TAG_COMMIT
        and _is_sha256(identity.get("colmap_binary_sha256"))
        and _is_sha256(identity.get("help_sha256"))
    ):
        reasons.append("invalid_pinned_sidecar_identity")
    return not reasons, reasons


def _hardware_metadata(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Collect non-identifying hardware context for interpreting N6 RSS/time."""

    system = platform.system()
    result: dict[str, Any] = {
        "platform": system,
        "logical_cpu_count": os.cpu_count(),
        "physical_core_count": None,
        "physical_core_count_reason": "unavailable_non_linux",
        "cpu_model": None,
        "mem_total_bytes": None,
        "source": "unavailable_non_linux",
    }
    if system != "Linux":
        return result
    cpuinfo = (proc_root / "cpuinfo").read_text(
        encoding="utf-8", errors="replace",
    )
    meminfo = (proc_root / "meminfo").read_text(
        encoding="utf-8", errors="replace",
    )
    model = next((
        line.split(":", 1)[1].strip()
        for line in cpuinfo.splitlines()
        if line.lower().startswith("model name") and ":" in line
    ), None)
    memory_kib = next((
        int(line.split()[1])
        for line in meminfo.splitlines()
        if line.startswith("MemTotal:")
    ), None)
    if not model or memory_kib is None:
        raise RuntimeError("Linux /proc hardware metadata is incomplete")
    core_pairs: set[tuple[str, str]] = set()
    for block in cpuinfo.split("\n\n"):
        fields = {
            line.split(":", 1)[0].strip().lower(): line.split(":", 1)[1].strip()
            for line in block.splitlines()
            if ":" in line
        }
        if "physical id" in fields and "core id" in fields:
            core_pairs.add((fields["physical id"], fields["core id"]))
    result.update({
        "cpu_model": model,
        "physical_core_count": len(core_pairs) if core_pairs else None,
        "physical_core_count_reason": (
            None if core_pairs else "physical_core_topology_unavailable"
        ),
        "mem_total_bytes": memory_kib * 1024,
        "source": "linux_proc_cpuinfo_meminfo",
    })
    return result


def _execution_identity_snapshot(script: Path) -> dict[str, Any]:
    return {
        "code": code_identity(script.parents[1]),
        "script": file_identity(script),
        "dependencies": dependency_manifest(names=SCALE_DEPENDENCIES),
    }


def _assert_execution_identity_matches(
    expected: dict[str, Any], observed: dict[str, Any],
) -> None:
    if canonical_hash(expected) != canonical_hash(observed):
        raise RuntimeError("scale runner identity changed during measurement")


def _recheck_execution_identity(
    script: Path, expected: dict[str, Any],
) -> None:
    _assert_execution_identity_matches(
        expected, _execution_identity_snapshot(script),
    )


def build_hdbscan_fixture(
    *, event_count: int, cluster_count: int,
) -> tuple[RelativeEventSequence, str]:
    if event_count < cluster_count * 10:
        raise ValueError("event_count must provide at least 10 events per cluster")
    if cluster_count <= 1:
        raise ValueError("cluster_count must exceed one")
    rng = np.random.default_rng(SCALE_SEED)
    assignments = np.arange(event_count, dtype=np.int64) % cluster_count
    # Separated centres prevent the scale fixture itself from being an
    # accidental single-cluster/noise edge case.  The input is sorted only by
    # timestamp, exactly as a real RelativeEventSequence requires.
    grid_columns = int(math.ceil(math.sqrt(cluster_count)))
    center_x = 20 + 45 * (assignments % grid_columns)
    center_y = 20 + 45 * (assignments // grid_columns)
    center_x = np.minimum(center_x, SENSOR_WIDTH - 20)
    center_y = np.minimum(center_y, SENSOR_HEIGHT - 20)
    center_t = 50.0 * assignments
    t_ms = center_t + rng.normal(0.0, 0.65, event_count)
    x = center_x + rng.integers(-2, 3, event_count)
    y = center_y + rng.integers(-2, 3, event_count)
    order = np.lexsort((np.arange(event_count, dtype=np.int64), t_ms))
    t_ms = t_ms[order]
    # Shift to the same non-negative relative-time convention as the method.
    t_ms -= t_ms[0]
    x = x[order].astype(np.int64, copy=False)
    y = y[order].astype(np.int64, copy=False)
    # ``source_index`` is the monotonically increasing position in the
    # generated source stream (rather than the pre-sort RNG array index).
    # This mirrors the invariant enforced by ``RelativeEventSequence``.
    source = np.arange(event_count, dtype=np.int64)
    events = RelativeEventSequence(
        t_ms=t_ms,
        x=x,
        y=y,
        p=np.zeros(event_count, dtype=np.int64),
        source_index=source,
        width=SENSOR_WIDTH,
        height=SENSOR_HEIGHT,
    )
    fixture_hash = canonical_hash({
        "schema": "esfo-gate-d-hdbscan-scale-identity-v2",
        "identity_scheme": SCALE_IDENTITY_SCHEME,
        "identity_quantum_decimal": "0.000000001",
        "seed": SCALE_SEED,
        "recipe": {
            "event_count": event_count,
            "cluster_count": cluster_count,
            "sensor": [SENSOR_WIDTH, SENSOR_HEIGHT],
        },
        "t_ms_quantized": _semantic_float_grid(events.t_ms),
        "x": events.x,
        "y": events.y,
        "p": events.p,
        "source_index": events.source_index,
        "width": events.width,
        "height": events.height,
    })
    return events, fixture_hash


def build_colmap_fixture(
    *, view_count: int, track_count: int,
) -> tuple[tuple[IndexedView, ...], str]:
    if view_count < 2 or track_count <= 0:
        raise ValueError("view_count must be >=2 and track_count positive")
    track_ids = np.arange(track_count, dtype=np.int64)
    # A real non-coplanar, noise-free SfM problem avoids turning the actual
    # matches_importer benchmark into a planar-homography/watermark edge case.
    # All points lie well inside every inward-looking full-orbit camera.
    points_w = np.random.default_rng(SCALE_SEED + 1).uniform(
        -2.5, 2.5, size=(track_count, 3),
    )
    calibration = Calibration(
        400.0, 400.0, SENSOR_WIDTH / 2.0, SENSOR_HEIGHT / 2.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )
    views: list[IndexedView] = []
    for view_index in range(1, view_count + 1):
        angle = 2.0 * math.pi * (view_index - 1) / view_count
        center_w = np.asarray([
            10.0 * math.cos(angle), 10.0 * math.sin(angle), 2.5,
        ])
        forward = -center_w / np.linalg.norm(center_w)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation_cw = np.stack((right, down, forward))
        points_c = (rotation_cw @ (points_w - center_w).T).T
        if np.any(points_c[:, 2] <= 0.0):
            raise RuntimeError("COLMAP scale point lies behind a camera")
        points = np.column_stack((
            calibration.fx * points_c[:, 0] / points_c[:, 2] + calibration.cx,
            calibration.fy * points_c[:, 1] / points_c[:, 2] + calibration.cy,
        ))
        if (np.any(points[:, 0] <= 1.0)
                or np.any(points[:, 0] >= SENSOR_WIDTH - 1.0)
                or np.any(points[:, 1] <= 1.0)
                or np.any(points[:, 1] >= SENSOR_HEIGHT - 1.0)):
            raise RuntimeError("COLMAP scale projection lies outside the sensor")
        views.append(IndexedView(
            image_id=view_index,
            window_index=view_index,
            time_ms=30.0 * view_index,
            image_name=f"window_{view_index:06d}.png",
            track_ids=track_ids,
            keypoints_xy=points,
        ))
    result = tuple(views)
    fixture_hash = canonical_hash({
        "schema": "esfo-gate-d-colmap-injection-scale-identity-v2",
        "identity_scheme": SCALE_IDENTITY_SCHEME,
        "identity_quantum_decimal": "0.000000001",
        "seed": SCALE_SEED,
        "recipe": {
            "view_count": view_count,
            "track_count": track_count,
            "sensor": [SENSOR_WIDTH, SENSOR_HEIGHT],
            "orbit_radius": 10.0,
            "camera_z": 2.5,
        },
        "geometry": {
            "type": "noncoplanar_points_full_orbit_cameras",
            "points_w_quantized": _semantic_float_grid(points_w),
            "intrinsics_quantized": _semantic_float_grid(
                calibration.pinhole_parameters,
            ),
            "orbit_radius": 10.0,
            "camera_z": 2.5,
        },
        "views": [
            {
                "image_id": view.image_id,
                "track_ids": view.track_ids,
                "keypoints_xy_quantized": _semantic_float_grid(
                    view.keypoints_xy,
                ),
            }
            for view in result
        ],
    })
    return result, fixture_hash


def _create_colmap_schema(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE cameras(
                camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                height INTEGER, params BLOB, prior_focal_length INTEGER
            );
            CREATE TABLE images(
                image_id INTEGER PRIMARY KEY, name TEXT UNIQUE, camera_id INTEGER,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            );
            CREATE TABLE keypoints(
                image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE matches(
                pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE descriptors(
                image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE two_view_geometries(
                pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB,
                config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB,
                qvec BLOB, tvec BLOB
            );
            """
        )


def run_hdbscan_worker(
    *, event_count: int, cluster_count: int, output: Path,
) -> None:
    events, fixture_hash = build_hdbscan_fixture(
        event_count=event_count, cluster_count=cluster_count,
    )
    corners = np.arange(len(events), dtype=np.int64)
    result = cluster_corners(events, corners)
    atomic_write_json(output, {
        "schema": "esfo-gate-d-hdbscan-worker-v1",
        "fixture_hash": fixture_hash,
        "fixture_identity_scheme": SCALE_IDENTITY_SCHEME,
        "fixture_identity_quantum_decimal": "0.000000001",
        "event_count": len(events),
        "requested_cluster_count": cluster_count,
        "result_cluster_count": len(result.clusters),
        "noise_count": int(np.count_nonzero(result.labels == -1)),
        "labels_sha256": canonical_hash(result.labels),
        "effective_parameters": hdbscan_parameters(),
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
    })


def run_colmap_injection_worker(
    *, view_count: int, track_count: int, output: Path, work_root: Path,
) -> None:
    views, fixture_hash = build_colmap_fixture(
        view_count=view_count, track_count=track_count,
    )
    database = work_root / "scale.db"
    match_list = work_root / "all_pairs.txt"
    _create_colmap_schema(database)
    calibration = Calibration(
        400.0, 400.0, SENSOR_WIDTH / 2.0, SENSOR_HEIGHT / 2.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )
    populate_known_camera_database(
        database,
        calibration=calibration,
        width=SENSOR_WIDTH,
        height=SENSOR_HEIGHT,
        views=views,
    )
    pair_count, correspondence_count = write_matches_importer_file(
        match_list, views,
    )
    # This is the raw-match representation consumed by COLMAP 3.9.1's
    # matches_importer.  We inject every generated row to measure the
    # quadratic materialization/database burden without pretending that this
    # worker includes geometric verification or mapping.
    with sqlite3.connect(database) as connection:
        with connection:
            for pair in iter_pair_matches(views):
                connection.execute(
                    "INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,?,?)",
                    (
                        pair.pair_id,
                        pair.matches.shape[0],
                        2,
                        pair.matches.tobytes(order="C"),
                    ),
                )
    validate_imported_raw_matches(database, views)
    theoretical = track_count * math.comb(view_count, 2)
    if correspondence_count != theoretical:
        raise RuntimeError("scale correspondence formula changed")
    atomic_write_json(output, {
        "schema": "esfo-gate-d-colmap-injection-worker-v1",
        "fixture_hash": fixture_hash,
        "fixture_identity_scheme": SCALE_IDENTITY_SCHEME,
        "fixture_identity_quantum_decimal": "0.000000001",
        "fixture_geometry": "seeded_noncoplanar_3d_points_full_orbit_cameras",
        "view_count": view_count,
        "track_count": track_count,
        "all_pairs_image_pair_count": pair_count,
        "raw_correspondence_row_count": correspondence_count,
        "theoretical_raw_correspondence_row_count": theoretical,
        "match_list": file_identity(match_list),
        "database": file_identity(database),
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
        "includes_geometric_verification": False,
        "includes_mapper": False,
        "measurement_scope": (
            "all_pairs_materialization_and_raw_sqlite_injection_only"
        ),
    })


def _docker_base(*, image: str, root: Path) -> list[str]:
    return [
        "docker", "run", "--rm", "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "HOME=/tmp", "--volume", f"{root.resolve()}:/work",
        image,
    ]


def _sidecar_identity(image: str) -> dict[str, Any]:
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True, check=False,
    )
    if inspected.returncode != 0:
        raise RuntimeError(f"cannot inspect pinned COLMAP sidecar: {image}")
    records = json.loads(inspected.stdout)
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("invalid Docker image inspection response")
    record = records[0]
    image_id = record.get("Id")
    if not _is_docker_image_id(image_id):
        raise RuntimeError("pinned COLMAP sidecar image ID is malformed")
    labels = record.get("Config", {}).get("Labels", {}) or {}
    if labels.get("org.opencontainers.image.revision") != COLMAP_TAG_COMMIT:
        raise RuntimeError("COLMAP sidecar revision label is not pinned")
    binary = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--entrypoint", "/usr/bin/sha256sum", image_id,
            "/usr/local/bin/colmap",
        ],
        capture_output=True, text=True, check=False,
    )
    fields = binary.stdout.strip().split()
    if (binary.returncode != 0 or len(fields) != 2
            or len(fields[0]) != 64
            or any(character not in "0123456789abcdef" for character in fields[0])):
        raise RuntimeError("cannot hash pinned COLMAP binary")
    help_run = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", image_id, "--help"],
        capture_output=True, text=True, check=False,
    )
    help_output = help_run.stdout + help_run.stderr
    if help_run.returncode != 0:
        raise RuntimeError("cannot query pinned COLMAP version")
    assert_colmap_version(help_output)
    return {
        "logical_name": image,
        "image_id": image_id,
        "repo_digests": record.get("RepoDigests") or [],
        "revision": labels["org.opencontainers.image.revision"],
        "colmap_version": COLMAP_VERSION,
        "colmap_binary_sha256": fields[0],
        "help_sha256": canonical_hash(help_output),
    }


def _sidecar_immutable_projection(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "logical_name": identity.get("logical_name"),
        "image_id": identity.get("image_id"),
        "repo_digests": sorted(identity.get("repo_digests") or []),
        "revision": identity.get("revision"),
        "colmap_version": identity.get("colmap_version"),
        "colmap_binary_sha256": identity.get("colmap_binary_sha256"),
        "help_sha256": identity.get("help_sha256"),
    }


def _assert_sidecar_tag_unchanged(
    before: dict[str, Any], after: dict[str, Any],
) -> None:
    if canonical_hash(_sidecar_immutable_projection(before)) != canonical_hash(
        _sidecar_immutable_projection(after),
    ):
        raise RuntimeError("COLMAP sidecar tag changed during measurement")


def _matches_importer_wrapper(
    *, logical_command: str, sample_interval_ms: float,
) -> str:
    """Build the auditable in-container Linux process-tree sampler."""

    if not isinstance(logical_command, str) or not logical_command.strip():
        raise ValueError("logical_command must be non-empty")
    if (isinstance(sample_interval_ms, bool)
            or not isinstance(sample_interval_ms, (int, float))
            or not math.isfinite(float(sample_interval_ms))
            or float(sample_interval_ms) <= 0.0):
        raise ValueError("sample_interval_ms must be positive and finite")
    interval_s = float(sample_interval_ms) / 1000.0
    return f"""
set -u
tree_rss_kib() {{
  local process_id="$1" total=0 own=0 child child_total
  if [ -r "/proc/$process_id/status" ]; then
    own=$(awk '/^VmRSS:/ {{print $2; exit}}' "/proc/$process_id/status")
    own=${{own:-0}}
    total=$own
  fi
  if [ -r "/proc/$process_id/task/$process_id/children" ]; then
    for child in $(cat "/proc/$process_id/task/$process_id/children"); do
      child_total=$(tree_rss_kib "$child")
      total=$((total + child_total))
    done
  fi
  printf '%s\\n' "$total"
}}
started=$(date +%s%N)
{logical_command} > /work/actual_matches_importer/matches_importer.stdout.txt 2> /work/actual_matches_importer/matches_importer.stderr.txt &
worker_pid=$!
peak_kib=0
while kill -0 "$worker_pid" 2>/dev/null; do
  current_kib=$(tree_rss_kib "$worker_pid")
  if [ "$current_kib" -gt "$peak_kib" ]; then peak_kib=$current_kib; fi
  sleep {interval_s:.9f}
done
set +e
wait "$worker_pid"
worker_rc=$?
set -e
ended=$(date +%s%N)
printf '%s\\t%s\\t%s\\n' "$((ended-started))" "$peak_kib" "$worker_rc" > /work/actual_matches_importer/resource.tsv
exit "$worker_rc"
""".strip()


def run_actual_matches_importer(
    *,
    root: Path,
    image: str,
    view_count: int,
    track_count: int,
    timeout_s: float,
    sample_interval_ms: float,
) -> dict[str, Any]:
    """Measure the actual pinned ``matches_importer`` inside its container.

    Docker daemon processes are not descendants of the client CLI.  The
    wrapper therefore samples the complete ``/proc`` tree *inside* the pinned
    sidecar, where COLMAP and any children are visible, and writes only its
    numeric outcome to the mounted evidence directory.
    """

    if platform.system() != "Linux":
        raise RuntimeError("actual matches_importer resource evidence requires Linux")
    sidecar_identity = _sidecar_identity(image)
    resolved_image = str(sidecar_identity["image_id"])
    work = root / "actual_matches_importer"
    work.mkdir(parents=True, exist_ok=False)
    database = work / "database.db"
    matches = work / "all_pairs.txt"
    stdout_path = work / "matches_importer.stdout.txt"
    stderr_path = work / "matches_importer.stderr.txt"
    report_path = work / "resource.tsv"
    views, fixture_hash = build_colmap_fixture(
        view_count=view_count, track_count=track_count,
    )
    calibration = Calibration(
        400.0, 400.0, SENSOR_WIDTH / 2.0, SENSOR_HEIGHT / 2.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    )
    creator = _docker_base(image=resolved_image, root=root) + [
        "database_creator", "--database_path",
        "/work/actual_matches_importer/database.db",
    ]
    created = subprocess.run(
        creator, capture_output=True, text=True, check=False,
    )
    if created.returncode != 0:
        raise RuntimeError(
            "pinned COLMAP database_creator failed: " + created.stderr[-1000:],
        )
    populate_known_camera_database(
        database, calibration=calibration, width=SENSOR_WIDTH,
        height=SENSOR_HEIGHT, views=views,
    )
    pair_count, raw_rows = write_matches_importer_file(matches, views)
    plan = build_colmap_commands(
        database_path="/work/actual_matches_importer/database.db",
        match_list_path="/work/actual_matches_importer/all_pairs.txt",
        image_path="/work/unused_images",
        output_path="/work/unused_models",
        executable="/usr/local/bin/colmap",
    )
    logical_command = shlex.join(plan.matches_importer)
    wrapper = _matches_importer_wrapper(
        logical_command=logical_command,
        sample_interval_ms=sample_interval_ms,
    )
    container_name = f"esfo_gate_d_matches_{os.getpid()}"
    command = [
        "docker", "run", "--rm", "--name", container_name,
        "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
        "--env", "HOME=/tmp", "--volume", f"{root.resolve()}:/work",
        "--entrypoint", "/bin/bash", resolved_image, "-lc", wrapper,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired as error:
        subprocess.run(
            ["docker", "rm", "--force", container_name],
            capture_output=True, text=True, check=False,
        )
        raise RuntimeError("actual matches_importer timed out") from error
    if not report_path.is_file():
        raise RuntimeError(
            "matches_importer wrapper produced no resource report: "
            + completed.stderr[-1000:],
        )
    fields = report_path.read_text(encoding="ascii").strip().split("\t")
    if len(fields) != 3:
        raise RuntimeError("malformed matches_importer resource report")
    wall_ns, peak_kib, worker_rc = (int(field) for field in fields)
    if worker_rc != completed.returncode or worker_rc != 0:
        raise RuntimeError(
            f"actual matches_importer failed with return code {worker_rc}",
        )
    validate_imported_raw_matches(database, views)
    verified = validate_geometric_verification(database, views)
    _assert_sidecar_tag_unchanged(
        sidecar_identity, _sidecar_identity(image),
    )
    measurement = ResourceMeasurement(
        argv=tuple(command),
        wall_time_s=wall_ns / 1_000_000_000.0,
        peak_rss_bytes=peak_kib * 1024,
        platform="Linux",
        measurement_method="linux_container_proc_process_tree_sampled",
        sample_interval_ms=float(sample_interval_ms),
        returncode=worker_rc,
        timed_out=False,
        stdout_path=str(stdout_path.resolve()),
        stderr_path=str(stderr_path.resolve()),
    )
    return {
        "schema": "esfo-gate-d-actual-matches-importer-v1",
        "fixture_hash": fixture_hash,
        "fixture_identity_scheme": SCALE_IDENTITY_SCHEME,
        "fixture_identity_quantum_decimal": "0.000000001",
        "fixture_geometry": "seeded_noncoplanar_3d_points_full_orbit_cameras",
        "view_count": view_count,
        "track_count": track_count,
        "all_pairs_image_pair_count": pair_count,
        "raw_correspondence_row_count": raw_rows,
        "verified_pair_count": len(verified),
        "verified_inlier_row_count": sum(int(value) for value in verified.values()),
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
        "includes_geometric_verification": True,
        "includes_mapper": False,
        "measurement_scope": "pinned_colmap_3_9_1_matches_importer_raw",
        "sidecar_identity": sidecar_identity,
        "database": file_identity(database),
        "match_list": file_identity(matches),
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
        "measurement": _measurement_dict(measurement),
    }


def _load_worker(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("fixture_hash"):
        raise RuntimeError(f"invalid worker output: {path}")
    return payload


def _measurement_dict(measurement: ResourceMeasurement) -> dict[str, Any]:
    return asdict(measurement)


def run_benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    root = arguments.work_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("work root must be absent or empty; stale reuse is forbidden")
    root.mkdir(parents=True, exist_ok=True)
    hdbscan_output = root / "hdbscan_worker.json"
    colmap_output = root / "colmap_injection_worker.json"
    script = Path(__file__).resolve()
    execution_identity = _execution_identity_snapshot(script)
    base = [sys.executable, str(script)]
    hdbscan_measurement = measure_subprocess(
        base + [
            "--worker", "hdbscan",
            "--worker-output", str(hdbscan_output),
            "--event-count", str(arguments.event_count),
            "--cluster-count", str(arguments.cluster_count),
        ],
        stdout_path=root / "hdbscan.stdout.txt",
        stderr_path=root / "hdbscan.stderr.txt",
        timeout_s=arguments.timeout_s,
        sample_interval_ms=arguments.sample_interval_ms,
        check=True,
    )
    _recheck_execution_identity(script, execution_identity)
    colmap_measurement = measure_subprocess(
        base + [
            "--worker", "colmap_injection",
            "--worker-output", str(colmap_output),
            "--worker-root", str(root / "colmap_worker"),
            "--view-count", str(arguments.view_count),
            "--track-count", str(arguments.track_count),
        ],
        stdout_path=root / "colmap_injection.stdout.txt",
        stderr_path=root / "colmap_injection.stderr.txt",
        timeout_s=arguments.timeout_s,
        sample_interval_ms=arguments.sample_interval_ms,
        check=True,
    )
    _recheck_execution_identity(script, execution_identity)
    hdbscan = _load_worker(hdbscan_output)
    colmap = _load_worker(colmap_output)
    _, expected_hdbscan_hash = build_hdbscan_fixture(
        event_count=arguments.event_count,
        cluster_count=arguments.cluster_count,
    )
    _, expected_colmap_hash = build_colmap_fixture(
        view_count=arguments.view_count,
        track_count=arguments.track_count,
    )
    if hdbscan["fixture_hash"] != expected_hdbscan_hash:
        raise RuntimeError("HDBSCAN worker fixture identity mismatch")
    if colmap["fixture_hash"] != expected_colmap_hash:
        raise RuntimeError("COLMAP worker fixture identity mismatch")
    run_sidecar = (
        arguments.sidecar_importer == "required"
        or (
            arguments.sidecar_importer == "auto"
            and platform.system() == "Linux"
        )
    )
    if arguments.sidecar_importer == "required" and platform.system() != "Linux":
        raise RuntimeError("required actual matches_importer evidence needs Linux")
    if run_sidecar:
        sidecar_importer: dict[str, Any] | None = run_actual_matches_importer(
            root=root,
            image=arguments.sidecar_image,
            view_count=arguments.view_count,
            track_count=arguments.track_count,
            timeout_s=arguments.timeout_s,
            sample_interval_ms=arguments.sample_interval_ms,
        )
        if sidecar_importer["fixture_hash"] != expected_colmap_hash:
            raise RuntimeError("actual matches_importer fixture identity mismatch")
    else:
        sidecar_importer = None
    _recheck_execution_identity(script, execution_identity)
    measurements = (hdbscan_measurement, colmap_measurement)
    code = execution_identity["code"]
    reference_evidence_eligible, noncanonical_reasons = (
        _scale_reference_eligibility(
            arguments=arguments,
            code=code,
            host_measurements=measurements,
            hdbscan_worker=hdbscan,
            colmap_worker=colmap,
            sidecar_importer=sidecar_importer,
            hdbscan_fixture_hash=expected_hdbscan_hash,
            colmap_fixture_hash=expected_colmap_hash,
        )
    )
    manifest = {
        "schema": "esfo-gate-d-scale-manifest-v1",
        "status": (
            "passed"
            if reference_evidence_eligible
            else "measurement_completed_noncanonical"
        ),
        "reference_evidence_eligible": reference_evidence_eligible,
        "reference_measurement_requirement": {
            "platform": "Linux",
            "measurement_method": "linux_proc_process_tree_sampled",
            "actual_matches_importer_required": True,
            "actual_matches_importer_method": (
                "linux_container_proc_process_tree_sampled"
            ),
            "observed_host_platform": platform.system(),
            "canonical_dimensions": {
                "event_count": DEFAULT_EVENT_COUNT,
                "cluster_count": DEFAULT_CLUSTER_COUNT,
                "view_count": DEFAULT_VIEW_COUNT,
                "track_count": DEFAULT_TRACK_COUNT,
            },
            "noncanonical_reasons": noncanonical_reasons,
            "noncanonical_reason": (
                None
                if reference_evidence_eligible
                else "; ".join(noncanonical_reasons)
            ),
        },
        "public_data_used": False,
        "seed": SCALE_SEED,
        "fixture_identity_scheme": SCALE_IDENTITY_SCHEME,
        "fixture_identity_quantum_decimal": "0.000000001",
        "code_identity": code,
        "hardware": _hardware_metadata(),
        "dependencies": execution_identity["dependencies"],
        "script": execution_identity["script"],
        "policy": {
            "whole_sequence": True,
            "external_chunking": False,
            "external_subsampling": False,
            "external_approximation": False,
            "hdbscan_internal_approx_min_span_tree": True,
            "note": (
                "external approximation is prohibited; the separately named "
                "HDBSCAN 0.8.33 approx_min_span_tree=True is frozen reference behavior"
            ),
        },
        "hdbscan": {
            "worker": hdbscan,
            "measurement": _measurement_dict(hdbscan_measurement),
        },
        "colmap_injection": {
            "worker": colmap,
            "measurement": _measurement_dict(colmap_measurement),
        },
        "actual_matches_importer": (
            sidecar_importer
            if sidecar_importer is not None
            else {
                "status": "skipped_noncanonical",
                "requested_mode": arguments.sidecar_importer,
                "reference_evidence_eligible": False,
            }
        ),
    }
    _recheck_execution_identity(script, execution_identity)
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--worker", choices=("hdbscan", "colmap_injection"), default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--event-count", type=int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument("--cluster-count", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--view-count", type=int, default=DEFAULT_VIEW_COUNT)
    parser.add_argument("--track-count", type=int, default=DEFAULT_TRACK_COUNT)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument(
        "--sidecar-importer", choices=("auto", "required", "skip"),
        default="auto",
    )
    parser.add_argument("--sidecar-image", default=SIDECAR_IMAGE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    if arguments.worker is not None:
        if arguments.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        arguments.worker_output.parent.mkdir(parents=True, exist_ok=True)
        if arguments.worker == "hdbscan":
            run_hdbscan_worker(
                event_count=arguments.event_count,
                cluster_count=arguments.cluster_count,
                output=arguments.worker_output,
            )
        else:
            if arguments.worker_root is None:
                raise ValueError("--worker-root is required for COLMAP injection")
            arguments.worker_root.mkdir(parents=True, exist_ok=False)
            run_colmap_injection_worker(
                view_count=arguments.view_count,
                track_count=arguments.track_count,
                output=arguments.worker_output,
                work_root=arguments.worker_root,
            )
        return
    if arguments.work_root is None:
        raise ValueError("--work-root is required")
    run_benchmark(arguments)


if __name__ == "__main__":
    main()
