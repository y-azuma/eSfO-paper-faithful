from __future__ import annotations

import json

import pytest

from esfo.pipeline import (
    PipelineStage, StageExecution, run_reference_pipeline,
)


IDENTITIES = {
    "initial_input_identity": {"fixture": {"sha256": "input"}},
    "code_identity": {"source_tree_sha256": "source"},
    "dependencies": {"python": "3.10.14"},
}


def _stage(name: str, calls: list[str], *, fail: bool = False) -> PipelineStage:
    def callback(work, upstream):
        calls.append(name)
        if fail:
            raise RuntimeError(f"{name} failed")
        output = work / "result.json"
        output.write_text(json.dumps({
            "stage": name,
            "upstream": None if upstream is None else upstream.stage,
        }) + "\n", encoding="utf-8")
        return StageExecution(
            {"result": output},
            {"upstream_stage": None if upstream is None else upstream.stage},
        )
    return PipelineStage(name, {"reference": True}, callback)


def test_fresh_then_exact_resume_skips_every_callback(tmp_path) -> None:
    calls: list[str] = []
    stages = tuple(_stage(name, calls) for name in ("frontend", "colmap", "orbit"))

    fresh = run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    assert fresh.completed
    assert calls == ["frontend", "colmap", "orbit"]
    assert fresh.resumed_stages == ()

    calls.clear()
    resumed = run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    assert resumed.completed
    assert calls == []
    assert resumed.resumed_stages == ("frontend", "colmap", "orbit")
    assert [record.fingerprint for record in resumed.records] == [
        record.fingerprint for record in fresh.records
    ]


def test_upstream_output_tamper_blocks_resume_and_downstream(tmp_path) -> None:
    calls: list[str] = []
    stages = (_stage("frontend", calls), _stage("colmap", calls))
    result = run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    assert result.completed
    first_output = next((tmp_path / "work" / "01_frontend").glob("*.json"))
    first_output.write_text('{"tampered":true}\n', encoding="utf-8")
    calls.clear()

    with pytest.raises(ValueError, match="output.*identity"):
        run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    assert calls == []


def test_identity_change_rejects_existing_checkpoint(tmp_path) -> None:
    calls: list[str] = []
    stages = (_stage("frontend", calls),)
    run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    changed = dict(IDENTITIES)
    changed["dependencies"] = {"python": "3.10.13"}

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_reference_pipeline(stages, work_root=tmp_path, **changed)


def test_first_failure_stops_later_stages_and_is_not_resumable(tmp_path) -> None:
    calls: list[str] = []
    stages = (
        _stage("frontend", calls),
        _stage("geometry", calls, fail=True),
        _stage("orbit", calls),
    )
    failed = run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)

    assert not failed.completed
    assert calls == ["frontend", "geometry"]
    assert [record.status for record in failed.records] == ["completed", "failed"]
    assert failed.first_failure is not None
    assert failed.first_failure.stage == "geometry"
    assert failed.first_failure.code == "geometry_initialization_failure"
    calls.clear()
    with pytest.raises(ValueError, match="completed"):
        run_reference_pipeline(stages, work_root=tmp_path, **IDENTITIES)
    assert calls == []


def test_partial_checkpoint_is_never_overwritten(tmp_path) -> None:
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir(parents=True)
    (stages_dir / "01_frontend.stage.json").write_text(
        '{"schema":', encoding="utf-8",
    )
    calls: list[str] = []
    with pytest.raises(ValueError, match="invalid stage checkpoint"):
        run_reference_pipeline(
            (_stage("frontend", calls),), work_root=tmp_path, **IDENTITIES,
        )
    assert calls == []
