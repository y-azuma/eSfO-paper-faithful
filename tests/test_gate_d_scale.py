from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gate_d_scale.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_gate_d_scale", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_hashes_are_deterministic_and_content_sensitive() -> None:
    module = _module()
    assert module.DEFAULT_EVENT_COUNT == 168_960
    events1, hash1 = module.build_hdbscan_fixture(
        event_count=320, cluster_count=8,
    )
    events2, hash2 = module.build_hdbscan_fixture(
        event_count=320, cluster_count=8,
    )
    _, changed = module.build_hdbscan_fixture(
        event_count=328, cluster_count=8,
    )
    assert hash1 == hash2
    assert hash1 != changed
    assert len(events1) == len(events2) == 320

    views1, view_hash1 = module.build_colmap_fixture(
        view_count=6, track_count=8,
    )
    views2, view_hash2 = module.build_colmap_fixture(
        view_count=6, track_count=8,
    )
    assert view_hash1 == view_hash2
    assert len(views1) == len(views2) == 6
    assert module.expected_raw_correspondence_count(views1) == 8 * 15

    _, canonical_hdbscan = module.build_hdbscan_fixture(
        event_count=module.DEFAULT_EVENT_COUNT,
        cluster_count=module.DEFAULT_CLUSTER_COUNT,
    )
    _, canonical_colmap = module.build_colmap_fixture(
        view_count=module.DEFAULT_VIEW_COUNT,
        track_count=module.DEFAULT_TRACK_COUNT,
    )
    assert canonical_hdbscan == module.CANONICAL_HDBSCAN_FIXTURE_HASH
    assert canonical_colmap == module.CANONICAL_COLMAP_FIXTURE_HASH


def test_scale_identity_grid_ignores_only_subquantum_libm_noise() -> None:
    module = _module()
    source = module.np.asarray([0.1234567891, -4.0000000001])
    assert module.np.array_equal(
        module._semantic_float_grid(source),
        module._semantic_float_grid(source + 1e-12),
    )
    assert not module.np.array_equal(
        module._semantic_float_grid(source),
        module._semantic_float_grid(source + 2e-9),
    )


def test_scale_execution_identity_drift_is_rejected() -> None:
    module = _module()
    expected = {
        "code": {"source_tree_sha256": "a" * 64},
        "script": {"sha256": "b" * 64},
        "dependencies": {"python": "3.10"},
    }
    module._assert_execution_identity_matches(expected, dict(expected))
    changed = dict(expected)
    changed["script"] = {"sha256": "c" * 64}
    with pytest.raises(RuntimeError, match="components=script"):
        module._assert_execution_identity_matches(expected, changed)


def test_scale_reference_eligibility_requires_clean_canonical_run(
    monkeypatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    arguments = SimpleNamespace(
        event_count=module.DEFAULT_EVENT_COUNT,
        cluster_count=module.DEFAULT_CLUSTER_COUNT,
        view_count=module.DEFAULT_VIEW_COUNT,
        track_count=module.DEFAULT_TRACK_COUNT,
        sample_interval_ms=10.0,
        sidecar_image=module.SIDECAR_IMAGE,
    )
    code = {
        "git_dirty": False,
        "git_revision": "revision",
        "source_tree_sha256": "a" * 64,
    }
    expected_pairs = 128 * 127 // 2
    expected_rows = expected_pairs * 512
    measurement = module.ResourceMeasurement(
        argv=("worker",), wall_time_s=1.0, peak_rss_bytes=1024,
        platform="Linux",
        measurement_method="linux_proc_process_tree_sampled",
        sample_interval_ms=10.0, returncode=0, timed_out=False,
    )
    hdbscan = {
        "fixture_hash": module.CANONICAL_HDBSCAN_FIXTURE_HASH,
        "event_count": 168_960,
        "requested_cluster_count": 32,
        "result_cluster_count": 32,
        "noise_count": 0,
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
    }
    colmap = {
        "fixture_hash": module.CANONICAL_COLMAP_FIXTURE_HASH,
        "view_count": 128,
        "track_count": 512,
        "all_pairs_image_pair_count": expected_pairs,
        "raw_correspondence_row_count": expected_rows,
        "theoretical_raw_correspondence_row_count": expected_rows,
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
        "includes_geometric_verification": False,
        "includes_mapper": False,
        "measurement_scope": (
            "all_pairs_materialization_and_raw_sqlite_injection_only"
        ),
    }
    sidecar_measurement = {
        "platform": "Linux",
        "measurement_method": "linux_container_proc_process_tree_sampled",
        "sample_interval_ms": 10.0,
        "returncode": 0,
        "timed_out": False,
        "wall_time_s": 1.0,
        "peak_rss_bytes": 1024,
    }
    sidecar = {
        "fixture_hash": module.CANONICAL_COLMAP_FIXTURE_HASH,
        "view_count": 128,
        "track_count": 512,
        "all_pairs_image_pair_count": expected_pairs,
        "raw_correspondence_row_count": expected_rows,
        "verified_pair_count": expected_pairs,
        "verified_inlier_row_count": expected_rows,
        "whole_sequence": True,
        "external_chunking": False,
        "external_subsampling": False,
        "external_approximation": False,
        "includes_geometric_verification": True,
        "includes_mapper": False,
        "measurement_scope": "pinned_colmap_3_9_1_matches_importer_raw",
        "measurement": sidecar_measurement,
        "sidecar_identity": {
            "logical_name": module.SIDECAR_IMAGE,
            "image_id": "sha256:" + "c" * 64,
            "colmap_version": module.COLMAP_VERSION,
            "version_label": module.COLMAP_VERSION,
            "revision": module.COLMAP_TAG_COMMIT,
            "colmap_binary_sha256": "b" * 64,
            "help_sha256": "d" * 64,
        },
    }
    common = dict(
        arguments=arguments,
        host_measurements=(measurement, measurement),
        hdbscan_worker=hdbscan,
        colmap_worker=colmap,
        sidecar_importer=sidecar,
        hdbscan_fixture_hash=module.CANONICAL_HDBSCAN_FIXTURE_HASH,
        colmap_fixture_hash=module.CANONICAL_COLMAP_FIXTURE_HASH,
    )
    assert module._scale_reference_eligibility(
        code=code, **common,
    ) == (True, [])

    dirty = dict(code, git_dirty=True)
    eligible, reasons = module._scale_reference_eligibility(
        code=dirty, **common,
    )
    assert eligible is False
    assert reasons == ["dirty_worktree"]

    arguments.track_count -= 1
    eligible, reasons = module._scale_reference_eligibility(
        code=code, **common,
    )
    assert eligible is False
    assert "noncanonical_dimensions" in reasons

    arguments.track_count = module.DEFAULT_TRACK_COUNT
    arguments.sample_interval_ms = 10_000.0
    eligible, reasons = module._scale_reference_eligibility(
        code=code, **common,
    )
    assert eligible is False
    assert reasons == ["noncanonical_sample_interval_ms"]

    arguments.sample_interval_ms = 10.0
    sidecar["verified_inlier_row_count"] -= 1
    eligible, reasons = module._scale_reference_eligibility(
        code=code, **common,
    )
    assert eligible is False
    assert reasons == ["invalid_canonical_actual_importer_outcome"]


def test_linux_hardware_metadata_comes_from_proc(tmp_path, monkeypatch) -> None:
    module = _module()
    (tmp_path / "cpuinfo").write_text(
        "processor : 0\nmodel name : Test CPU 123\n"
        "physical id : 0\ncore id : 0\n\n"
        "processor : 1\nmodel name : Test CPU 123\n"
        "physical id : 0\ncore id : 1\n",
        encoding="utf-8",
    )
    (tmp_path / "meminfo").write_text(
        "MemTotal:       32768 kB\n", encoding="utf-8",
    )
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.os, "cpu_count", lambda: 16)

    assert module._hardware_metadata(tmp_path) == {
        "platform": "Linux",
        "logical_cpu_count": 16,
        "physical_core_count": 2,
        "physical_core_count_reason": None,
        "cpu_model": "Test CPU 123",
        "mem_total_bytes": 32768 * 1024,
        "source": "linux_proc_cpuinfo_meminfo",
    }


def test_linux_hardware_metadata_marks_missing_physical_topology(
    tmp_path, monkeypatch,
) -> None:
    module = _module()
    (tmp_path / "cpuinfo").write_text(
        "processor : 0\nmodel name : Topology Free CPU\n", encoding="utf-8",
    )
    (tmp_path / "meminfo").write_text(
        "MemTotal: 1024 kB\n", encoding="utf-8",
    )
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    metadata = module._hardware_metadata(tmp_path)
    assert metadata["physical_core_count"] is None
    assert metadata["physical_core_count_reason"] == (
        "physical_core_topology_unavailable"
    )


def test_tiny_scale_runner_writes_complete_evidence(tmp_path) -> None:
    work = tmp_path / "scale"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--work-root", str(work),
            "--event-count", "640",
            "--cluster-count", "8",
            "--view-count", "6",
            "--track-count", "8",
            "--timeout-s", "30",
            "--sample-interval-ms", "5",
            "--sidecar-importer", "skip",
        ],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "measurement_completed_noncanonical"
    assert manifest["reference_evidence_eligible"] is False
    reasons = manifest["reference_measurement_requirement"][
        "noncanonical_reasons"
    ]
    assert "noncanonical_dimensions" in reasons
    assert "invalid_actual_importer_process_tree_measurement" in reasons
    assert "invalid_pinned_sidecar_identity" in reasons
    assert manifest["public_data_used"] is False
    assert manifest["policy"]["whole_sequence"] is True
    assert manifest["policy"]["external_chunking"] is False
    assert manifest["policy"]["external_subsampling"] is False
    assert manifest["policy"]["external_approximation"] is False
    hdbscan = manifest["hdbscan"]
    assert hdbscan["worker"]["event_count"] == 640
    assert hdbscan["worker"]["whole_sequence"] is True
    assert hdbscan["measurement"]["wall_time_s"] > 0.0
    assert hdbscan["measurement"]["peak_rss_bytes"] >= 0
    colmap = manifest["colmap_injection"]
    assert colmap["worker"]["fixture_geometry"] == (
        "seeded_noncoplanar_3d_points_full_orbit_cameras"
    )
    assert colmap["worker"]["raw_correspondence_row_count"] == 8 * 15
    assert colmap["worker"]["theoretical_raw_correspondence_row_count"] == 8 * 15
    assert colmap["worker"]["includes_geometric_verification"] is False
    assert colmap["worker"]["includes_mapper"] is False
    assert colmap["measurement"]["wall_time_s"] > 0.0
    assert manifest["actual_matches_importer"] == {
        "status": "skipped_noncanonical",
        "requested_mode": "skip",
        "reference_evidence_eligible": False,
    }
    assert "NaN" not in (work / "manifest.json").read_text(encoding="utf-8")


def test_scale_runner_refuses_stale_work_root(tmp_path) -> None:
    work = tmp_path / "not_empty"
    work.mkdir()
    (work / "stale").write_text("x", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--work-root", str(work)],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "stale reuse is forbidden" in completed.stderr


def test_hdbscan_fixture_rejects_too_few_points() -> None:
    module = _module()
    with pytest.raises(ValueError, match="at least 10"):
        module.build_hdbscan_fixture(event_count=79, cluster_count=8)


def test_sidecar_identity_requires_revision_and_binary_hash(monkeypatch) -> None:
    module = _module()
    binary_hash = "a" * 64
    responses = iter((
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{
                "Id": "sha256:" + "1" * 64,
                "RepoDigests": ["example@sha256:digest"],
                "Config": {"Labels": {
                    "org.opencontainers.image.revision": module.COLMAP_TAG_COMMIT,
                    "org.opencontainers.image.version": module.COLMAP_VERSION,
                }},
            }]),
            stderr="",
        ),
        SimpleNamespace(
            returncode=0,
            stdout=f"{binary_hash}  /usr/local/bin/colmap\n",
            stderr="",
        ),
        SimpleNamespace(
            returncode=0,
            stdout="COLMAP 3.9.1 -- help\n",
            stderr="",
        ),
    ))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return next(responses)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    identity = module._sidecar_identity("pinned")

    assert identity["image_id"] == "sha256:" + "1" * 64
    assert identity["revision"] == module.COLMAP_TAG_COMMIT
    assert identity["colmap_binary_sha256"] == binary_hash
    assert identity["help_sha256"]
    assert calls[1][-2] == identity["image_id"]
    assert calls[2][-2] == identity["image_id"]


def test_matches_importer_sampler_is_valid_bash_and_records_process_tree() -> None:
    module = _module()
    wrapper = module._matches_importer_wrapper(
        logical_command="/usr/local/bin/colmap matches_importer --help",
        sample_interval_ms=10.0,
    )
    checked = subprocess.run(
        ["bash", "-n"], input=wrapper, text=True,
        capture_output=True, check=False,
    )
    assert checked.returncode == 0, checked.stderr
    assert "/proc/$process_id/status" in wrapper
    assert "/proc/$process_id/task/$process_id/children" in wrapper
    assert "VmRSS" in wrapper
    assert "sleep 0.010000000" in wrapper
    assert "resource.tsv" in wrapper
    assert wrapper.count(
        "/usr/local/bin/colmap matches_importer --help"
    ) == 1


def test_scale_sidecar_tag_replacement_is_rejected() -> None:
    module = _module()
    before = {
        "logical_name": "pinned",
        "image_id": "sha256:" + "1" * 64,
        "repo_digests": [],
        "revision": module.COLMAP_TAG_COMMIT,
        "version_label": module.COLMAP_VERSION,
        "colmap_version": module.COLMAP_VERSION,
        "colmap_binary_sha256": "a" * 64,
        "help_sha256": "b" * 64,
    }
    module._assert_sidecar_tag_unchanged(before, dict(before))
    with pytest.raises(RuntimeError, match="tag changed"):
        module._assert_sidecar_tag_unchanged(
            before, dict(before, image_id="sha256:" + "2" * 64),
        )
