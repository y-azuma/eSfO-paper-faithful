from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from esfo.colmap import IndexedView
from esfo.gtsam_backend import OrbitOptimizationDiagnostics
from esfo.pipeline import PipelineStage, StageExecution, run_reference_pipeline

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_gate_d_synthetic.py"
SPEC = importlib.util.spec_from_file_location("run_gate_d_synthetic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate_d = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate_d
SPEC.loader.exec_module(gate_d)


IDENTITIES = {
    "initial_input_identity": {"fixture": "gate-d-test"},
    "code_identity": {"source_tree_sha256": "test-source"},
    "dependencies": {"python": "test"},
}


def test_raw_fixture_is_deterministic_and_has_frozen_shape() -> None:
    first = gate_d.build_raw_fixture()
    second = gate_d.build_raw_fixture()

    assert first.fixture_hash == second.fixture_hash
    assert len(first.events) == 184_321
    assert first.points_w.shape == (18, 3)
    assert first.centers_w.shape == (64, 3)
    assert first.rotations_cw.shape == (64, 3, 3)
    assert first.projected_xy.shape == (64, 18, 2)
    assert np.all(np.diff(first.events.t_us) >= 0)


def test_canonical_sidecar_fixture_has_frozen_scale_and_separation() -> None:
    fixture = gate_d.build_raw_fixture("canonical-sidecar")

    assert fixture.fixture_hash == gate_d.CANONICAL_FIXTURE_HASH
    assert fixture.profile == "canonical-sidecar"
    assert fixture.event_pattern == "single_ring_then_20_centers_800us"
    assert len(fixture.events) == 414_721
    assert fixture.points_w.shape == (120, 3)
    assert fixture.centers_w.shape == (128, 3)
    assert fixture.projected_xy.shape == (128, 120, 2)
    projected_by_track = fixture.projected_xy.transpose(1, 0, 2)
    minimum_separation = min(
        np.min(np.linalg.norm(
            projected_by_track[index, None] - projected_by_track[:index],
            axis=2,
        ))
        for index in range(1, fixture.point_count)
    )
    maximum_step = np.max(np.linalg.norm(
        np.roll(projected_by_track, -1, axis=1) - projected_by_track,
        axis=2,
    ))
    assert minimum_separation == pytest.approx(11.468028162187279, abs=1e-12)
    assert maximum_step == pytest.approx(14.096279846417193, abs=1e-12)
    frequency = gate_d.initialize_frequency_fft_raw(fixture.events)
    assert frequency.peak_bin == 1
    assert frequency.frequency_hz == pytest.approx(0.2631578947368421)
    assert np.all(np.diff(fixture.events.t_us) >= 0)


def test_fixture_identity_grid_ignores_only_subquantum_libm_noise() -> None:
    source = np.asarray([0.1234567891, -4.0000000001])
    assert np.array_equal(
        gate_d._semantic_float_grid(source),
        gate_d._semantic_float_grid(source + 1e-12),
    )
    assert not np.array_equal(
        gate_d._semantic_float_grid(source),
        gate_d._semantic_float_grid(source + 2e-9),
    )


def test_reference_eligibility_requires_clean_canonical_sidecar(
    monkeypatch,
) -> None:
    fixture = gate_d.build_raw_fixture("canonical-sidecar")
    context = gate_d.RunContext(
        fixture=fixture,
        colmap_mode="sidecar",
        sidecar_image=gate_d.SIDECAR_IMAGE,
        repeat_check=True,
        evidence_code_identity={
            "git_dirty": False,
            "git_revision": "revision",
            "source_tree_sha256": "a" * 64,
        },
    )
    context.hdbscan_repeat_labels_equal = True
    context.sidecar_identity = {
        "logical_name": gate_d.SIDECAR_IMAGE,
        "image_id": "sha256:" + "c" * 64,
        "colmap_version": gate_d.COLMAP_VERSION,
        "colmap_tag_commit": gate_d.COLMAP_TAG_COMMIT,
        "colmap_binary_sha256": "b" * 64,
        "help_sha256": "d" * 64,
        "labels": {
            "org.opencontainers.image.revision": gate_d.COLMAP_TAG_COMMIT,
            "org.opencontainers.image.version": gate_d.COLMAP_VERSION,
        },
    }
    monkeypatch.setattr(gate_d.platform, "system", lambda: "Linux")

    assert gate_d._synthetic_reference_eligibility(context) == (True, [])

    context.evidence_code_identity["git_dirty"] = True
    eligible, reasons = gate_d._synthetic_reference_eligibility(context)
    assert eligible is False
    assert reasons == ["dirty_worktree"]

    context.evidence_code_identity["git_dirty"] = False
    context.sidecar_identity["colmap_binary_sha256"] = "not-a-hash"
    eligible, reasons = gate_d._synthetic_reference_eligibility(context)
    assert eligible is False
    assert reasons == ["invalid_pinned_sidecar_identity"]


def test_fixture_only_colmap_model_uses_observations_not_track_id_assumption() -> None:
    fixture = gate_d.build_raw_fixture()
    track_ids = np.arange(100, 136, 2, dtype=np.int64)
    permutation = np.asarray([
        7, 1, 15, 4, 11, 0, 16, 9, 3, 17, 6, 13, 2, 10, 5, 14, 8, 12,
    ])
    views = tuple(
        IndexedView(
            window, window, 30.0 * window,
            f"window_{window:06d}.png",
            track_ids,
            np.rint(fixture.projected_xy[window, permutation]),
        )
        for window in range(1, 63)
    )

    model = gate_d.build_fixture_colmap_model(fixture, views)

    assert len(model.mapping.unique_pairs) == 18
    assert model.metadata["fixture_only"] is True
    assert model.metadata["reference_evidence_eligible"] is False
    assert model.metadata["maximum_assignment_rmse_px"] == 0.0
    for track_id, point_id in model.mapping.unique_pairs:
        row = int(np.flatnonzero(track_ids == track_id)[0])
        assert point_id == 10_000 + int(permutation[row])


def test_new_context_partially_resumes_from_hashed_upstream_state(tmp_path) -> None:
    fixture = gate_d.build_raw_fixture()
    first_context = gate_d.RunContext(
        fixture, "fixture-only", gate_d.SIDECAR_IMAGE, False,
    )

    def first_callback(work, upstream):
        assert upstream is None
        first_context.callback_calls.append("first")
        first_context.signed_radius = 12.5
        state = gate_d._write_state(work, first_context)
        marker = work / "first.json"
        marker.write_text('{"first":true}\n', encoding="utf-8")
        return StageExecution({"state": state, "marker": marker}, {})

    def should_not_repeat(_work, _upstream):
        raise AssertionError("completed prefix callback was repeated")

    first_stage = PipelineStage("first", {"fixed": True}, first_callback)
    prefix = run_reference_pipeline(
        (first_stage,), work_root=tmp_path, **IDENTITIES,
    )
    assert prefix.completed

    second_context = gate_d.RunContext(
        fixture, "fixture-only", gate_d.SIDECAR_IMAGE, False,
    )

    def second_callback(work, upstream):
        gate_d._hydrate_from_upstream(second_context, upstream)
        second_context.callback_calls.append("second")
        assert second_context.fixture.fixture_hash == fixture.fixture_hash
        assert second_context.signed_radius == 12.5
        marker = work / "second.json"
        marker.write_text('{"second":true}\n', encoding="utf-8")
        return StageExecution({"marker": marker}, {})

    continued = run_reference_pipeline(
        (
            PipelineStage("first", {"fixed": True}, should_not_repeat),
            PipelineStage("second", {"fixed": True}, second_callback),
        ),
        work_root=tmp_path,
        **IDENTITIES,
    )

    assert continued.completed
    assert continued.resumed_stages == ("first",)
    assert second_context.callback_calls == ["second"]


def test_checkpoint_rehydrates_after_real_interpreter_restart(tmp_path) -> None:
    prefix_source = textwrap.dedent("""
        import importlib.util, pathlib, sys
        from esfo.pipeline import PipelineStage, StageExecution, run_reference_pipeline
        script = pathlib.Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location("run_gate_d_synthetic", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        root = pathlib.Path(sys.argv[1])
        fixture = module.build_raw_fixture()
        context = module.RunContext(fixture, "fixture-only", module.SIDECAR_IMAGE, False)
        def first(work, upstream):
            assert upstream is None
            context.callback_calls.append("first")
            context.signed_radius = 12.5
            marker = work / "first.json"
            marker.write_text('{"first":true}\\n', encoding="utf-8")
            return StageExecution({"state": module._write_state(work, context), "marker": marker}, {})
        result = run_reference_pipeline(
            (PipelineStage("first", {"fixed": True}, first),),
            work_root=root,
            initial_input_identity={"fixture": "restart-test"},
            code_identity={"source_tree_sha256": "restart-source"},
            dependencies={"python": "restart-test"},
        )
        assert result.completed
    """)
    continue_source = textwrap.dedent("""
        import importlib.util, json, pathlib, sys
        from esfo.pipeline import PipelineStage, StageExecution, run_reference_pipeline
        script = pathlib.Path(sys.argv[2])
        spec = importlib.util.spec_from_file_location("run_gate_d_synthetic", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        root = pathlib.Path(sys.argv[1])
        fixture = module.build_raw_fixture()
        context = module.RunContext(fixture, "fixture-only", module.SIDECAR_IMAGE, False)
        def first(_work, _upstream):
            raise AssertionError("completed callback repeated after restart")
        def second(work, upstream):
            module._hydrate_from_upstream(context, upstream)
            context.callback_calls.append("second")
            assert context.signed_radius == 12.5
            marker = work / "second.json"
            marker.write_text('{"second":true}\\n', encoding="utf-8")
            return StageExecution({"marker": marker}, {})
        result = run_reference_pipeline(
            (PipelineStage("first", {"fixed": True}, first),
             PipelineStage("second", {"fixed": True}, second)),
            work_root=root,
            initial_input_identity={"fixture": "restart-test"},
            code_identity={"source_tree_sha256": "restart-source"},
            dependencies={"python": "restart-test"},
        )
        print(json.dumps({
            "completed": result.completed,
            "resumed": result.resumed_stages,
            "calls": context.callback_calls,
        }))
    """)
    first = subprocess.run(
        [sys.executable, "-c", prefix_source, str(tmp_path), str(SCRIPT)],
        cwd=SCRIPT.parents[1], capture_output=True, text=True, check=False,
    )
    assert first.returncode == 0, first.stderr
    checkpoint = next((tmp_path / "stages").glob("*first*.stage.json"))
    before_hash = gate_d.file_identity(checkpoint)["sha256"]

    continued = subprocess.run(
        [sys.executable, "-c", continue_source, str(tmp_path), str(SCRIPT)],
        cwd=SCRIPT.parents[1], capture_output=True, text=True, check=False,
    )
    assert continued.returncode == 0, continued.stderr
    payload = json.loads(continued.stdout)
    assert payload == {
        "completed": True,
        "resumed": ["first"],
        "calls": ["second"],
    }
    assert gate_d.file_identity(checkpoint)["sha256"] == before_hash


def test_final_nonpositive_depth_preserves_unavailable_report_then_fails(
    tmp_path,
) -> None:
    fixture = gate_d.build_raw_fixture()
    context = gate_d.RunContext(
        fixture, "fixture-only", gate_d.SIDECAR_IMAGE, False,
    )

    def optimization_callback(work, _upstream):
        context.optimization_diagnostics = OrbitOptimizationDiagnostics(4, 0, 1)
        context.reprojection = {
            "after_primary": {
                "status": "unavailable",
                "reason": "final_nonpositive_depth",
                "value": None,
            },
        }
        report = work / "optimization.json"
        report.write_text(json.dumps(context.reprojection) + "\n", encoding="utf-8")
        state = gate_d._write_state(work, context)
        return StageExecution({"optimization": report, "state": state}, {})

    result = run_reference_pipeline(
        (
            PipelineStage(
                "optimization", {"fixed": True}, optimization_callback,
                failure_stage="optimization",
            ),
            PipelineStage(
                "scene_report", {"fixed": True},
                lambda work, upstream: gate_d._scene_report_callback(
                    context, work, upstream,
                ),
                failure_stage="metrics",
            ),
        ),
        work_root=tmp_path,
        **IDENTITIES,
    )

    assert not result.completed
    assert [record.status for record in result.records] == ["completed", "failed"]
    assert result.first_failure is not None
    assert result.first_failure.code == "final_nonpositive_depth"
    assert result.first_failure.detail_code == "final_nonpositive_depth"
    report_path = result.records[0].output_files["optimization"]["path"]
    preserved = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert preserved["after_primary"] == {
        "status": "unavailable",
        "reason": "final_nonpositive_depth",
        "value": None,
    }


def test_runner_identity_binds_script_and_sidecar_definition() -> None:
    repository = Path(gate_d.__file__).resolve().parents[1]
    identity = gate_d._runner_code_identity(repository)
    assert identity["gate_d_synthetic_runner"]["sha256"]
    assert identity["colmap_sidecar_dockerfile"]["sha256"]


def test_sidecar_identity_records_image_binary_version_and_linkage(monkeypatch) -> None:
    binary_hash = "a" * 64
    inspect_payload = [{
        "Id": "sha256:" + "1" * 64,
        "RepoDigests": ["example@sha256:digest"],
        "Created": "2026-11-24T00:00:00Z",
        "Config": {"Labels": {
            "org.opencontainers.image.revision": gate_d.COLMAP_TAG_COMMIT,
            "org.opencontainers.image.version": gate_d.COLMAP_VERSION,
        }},
    }]
    diagnostics = (
        f"{binary_hash}  /usr/local/bin/colmap\n"
        "libfoo.so => /usr/lib/libfoo.so (0x1)\n"
        f"{gate_d.COLMAP_TAG_COMMIT}\n"
        "libfoo=1.0\n"
    )
    responses = iter((
        SimpleNamespace(returncode=0, stdout=json.dumps(inspect_payload), stderr=""),
        SimpleNamespace(returncode=0, stdout="COLMAP 3.9.1 -- help\n", stderr=""),
        SimpleNamespace(returncode=0, stdout=diagnostics, stderr=""),
    ))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return next(responses)

    monkeypatch.setattr(gate_d.subprocess, "run", fake_run)

    identity = gate_d._collect_sidecar_identity("pinned-image")

    assert identity["image_id"] == "sha256:" + "1" * 64
    assert identity["repo_digests"] == ["example@sha256:digest"]
    assert identity["colmap_version"] == "3.9.1"
    assert identity["colmap_tag_commit"] == gate_d.COLMAP_TAG_COMMIT
    assert identity["colmap_binary_sha256"] == binary_hash
    assert "libfoo.so" in identity["diagnostics"]
    assert identity["diagnostics_sha256"]
    assert identity["diagnostics_normalized_sha256"]
    assert calls[1][-2] == identity["image_id"]
    assert calls[2][-3] == identity["image_id"]


def test_sidecar_identity_is_bound_to_checkpoint_fingerprint(
    tmp_path, monkeypatch,
) -> None:
    fixture = gate_d.build_raw_fixture()
    identity = {
        "logical_name": "pinned-image",
        "image_id": "sha256:" + "1" * 64,
        "repo_digests": ["example@sha256:" + "2" * 64],
        "labels": {
            "org.opencontainers.image.revision": gate_d.COLMAP_TAG_COMMIT,
            "org.opencontainers.image.version": gate_d.COLMAP_VERSION,
        },
        "colmap_version": gate_d.COLMAP_VERSION,
        "colmap_tag_commit": gate_d.COLMAP_TAG_COMMIT,
        "colmap_binary_sha256": "a" * 64,
        "help_sha256": "b" * 64,
        "diagnostics_normalized_sha256": "c" * 64,
        "diagnostics_sha256": "d" * 64,
        "diagnostics": "lib.so (0x1111)\n",
    }
    monkeypatch.setattr(
        gate_d, "_collect_sidecar_identity", lambda _image: dict(identity),
    )
    arguments = dict(
        root=tmp_path,
        fixture=fixture,
        colmap_mode="sidecar",
        sidecar_image="pinned-image",
        repeat_check=True,
        injected_failure_stage=None,
        code={"source_tree_sha256": "source"},
        dependencies={"python": "test"},
        stage_limit=1,
    )
    first, _ = gate_d._run_pipeline(**arguments)
    assert first.completed

    identity["image_id"] = "sha256:" + "9" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        gate_d._run_pipeline(**arguments)


def test_sidecar_identity_is_rechecked_immediately_before_execution(
    monkeypatch,
) -> None:
    expected = {
        "logical_name": "pinned-image",
        "image_id": "sha256:" + "1" * 64,
        "repo_digests": ["example@sha256:" + "2" * 64],
        "labels": {
            "org.opencontainers.image.revision": gate_d.COLMAP_TAG_COMMIT,
            "org.opencontainers.image.version": gate_d.COLMAP_VERSION,
        },
        "colmap_version": gate_d.COLMAP_VERSION,
        "colmap_tag_commit": gate_d.COLMAP_TAG_COMMIT,
        "colmap_binary_sha256": "a" * 64,
        "help_sha256": "b" * 64,
        "diagnostics_normalized_sha256": "c" * 64,
        "diagnostics_sha256": "d" * 64,
        "diagnostics": "lib.so (0x1111)\n",
    }
    context = SimpleNamespace(
        sidecar_image="pinned-image", sidecar_identity=expected,
    )
    monkeypatch.setattr(
        gate_d, "_collect_sidecar_identity", lambda _image: dict(expected),
    )
    assert gate_d._verify_bound_sidecar_identity(context) == expected

    observed_with_new_aslr = dict(
        expected,
        diagnostics_sha256="e" * 64,
        diagnostics="lib.so (0x9999)\n",
    )
    monkeypatch.setattr(
        gate_d, "_collect_sidecar_identity",
        lambda _image: observed_with_new_aslr,
    )
    assert gate_d._verify_bound_sidecar_identity(context) == (
        observed_with_new_aslr
    )

    monkeypatch.setattr(
        gate_d, "_collect_sidecar_identity",
        lambda _image: dict(expected, image_id="sha256:" + "9" * 64),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        gate_d._verify_bound_sidecar_identity(context)


def test_sidecar_commands_use_bound_immutable_image_id(tmp_path, monkeypatch) -> None:
    image_id = "sha256:" + "1" * 64
    identity = {
        "logical_name": "mutable-tag",
        "image_id": image_id,
        "repo_digests": [],
        "labels": {
            "org.opencontainers.image.revision": gate_d.COLMAP_TAG_COMMIT,
            "org.opencontainers.image.version": gate_d.COLMAP_VERSION,
        },
        "colmap_version": gate_d.COLMAP_VERSION,
        "colmap_tag_commit": gate_d.COLMAP_TAG_COMMIT,
        "colmap_binary_sha256": "a" * 64,
        "help_sha256": "b" * 64,
        "diagnostics_normalized_sha256": "c" * 64,
    }
    context = SimpleNamespace(sidecar_identity=identity)
    observed = {}

    def fake_run_sidecar_command(**kwargs):
        observed.update(kwargs)
        return {"returncode": 0}

    monkeypatch.setattr(
        gate_d, "_run_sidecar_command", fake_run_sidecar_command,
    )
    result = gate_d._run_bound_sidecar_command(
        context, root=tmp_path, logical_argv=("colmap", "--help"),
        log=tmp_path / "log",
    )
    assert result == {"returncode": 0}
    assert observed["image"] == image_id


def test_scene_evidence_is_hash_verified_and_eligibility_recomputed(
    tmp_path,
) -> None:
    context = gate_d.RunContext(
        gate_d.build_raw_fixture(), "fixture-only", gate_d.SIDECAR_IMAGE, True,
    )
    context.hdbscan_repeat_labels_equal = True
    eligible, reasons = gate_d._synthetic_reference_eligibility(context)
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "reference_evidence_eligible": eligible,
        "noncanonical_reasons": reasons,
    }) + "\n", encoding="utf-8")
    identity = gate_d.file_identity(path)
    payload, observed_eligible, observed_reasons = (
        gate_d._load_verified_scene_evidence(identity, context)
    )
    assert payload["reference_evidence_eligible"] is False
    assert observed_eligible is False
    assert observed_reasons == reasons

    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output identity changed"):
        gate_d._load_verified_scene_evidence(identity, context)

    path.write_text(json.dumps({
        "reference_evidence_eligible": True,
        "noncanonical_reasons": [],
    }) + "\n", encoding="utf-8")
    stale_but_hash_valid = gate_d.file_identity(path)
    with pytest.raises(RuntimeError, match="eligibility disagrees"):
        gate_d._load_verified_scene_evidence(stale_but_hash_valid, context)


@pytest.mark.skipif(platform.system() == "Linux", reason="non-Linux guard only")
def test_sidecar_mode_fails_closed_off_linux(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="requires Linux"):
        gate_d.run_proof(work_root=tmp_path, colmap_mode="sidecar")
    assert not any(tmp_path.iterdir())


def test_sidecar_mode_refuses_small_unit_fixture_before_execution(tmp_path) -> None:
    with pytest.raises(ValueError, match="canonical-sidecar fixture"):
        gate_d.run_proof(
            work_root=tmp_path,
            colmap_mode="sidecar",
            fixture_profile="unit",
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(
    os.environ.get("RUN_ESFO_GATE_D_CANONICAL_FRONTEND") != "1",
    reason="explicit slow canonical raw-front-end opt-in",
)
def test_canonical_frontend_preserves_mapper_initial_pair_margin() -> None:
    fixture = gate_d.build_raw_fixture("canonical-sidecar")
    frontend, views, repeated_equal = gate_d._frontend_and_views(
        fixture, repeat_check=True,
    )
    full_tracks = sum(
        sum(
            observation.track_id == track.track_id
            for observation in frontend.observations
        ) == len(views)
        for track in frontend.association.tracks
    )
    minimum_pair = min(
        np.intersect1d(left.track_ids, right.track_ids, assume_unique=True).size
        for left_index, left in enumerate(views)
        for right in views[left_index + 1:]
    )
    assert repeated_equal is True
    assert full_tracks == 106
    assert minimum_pair == 106


@pytest.mark.skipif(
    os.environ.get("RUN_ESFO_GATE_D_SYNTHETIC") != "1",
    reason="explicit slow deterministic integration opt-in",
)
def test_complete_fixture_only_fresh_partial_resume_and_failure(tmp_path) -> None:
    manifest = gate_d.run_proof(
        work_root=tmp_path,
        colmap_mode="fixture-only",
        failure_stage="frontend",
        repeat_check=True,
    )
    assert manifest["status"] == "plumbing_passed_fixture_only"
    assert manifest["partial_resume"]["proved"] is True
    assert manifest["full_resume"]["proved"] is True
    assert manifest["failure_proof"]["proved"] is True
    assert manifest["scene"]["scene_diagnostics"]["frontend"][
        "hdbscan_repeat_labels_equal"
    ] is True
    assert manifest["scene"]["reprojection_report"]["after_primary"][
        "status"
    ] == "available"
