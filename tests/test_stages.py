from __future__ import annotations

import json

import pytest

from esfo.provenance import canonical_hash
from esfo.stages import (
    StageFailure,
    load_completed_stage,
    make_stage_fingerprint,
    make_stage_record,
    output_file_identities,
    write_stage_checkpoint,
)


def _identity_arguments() -> dict:
    return {
        "stage": "frontend",
        "config": {"radius": 7.0},
        "input_identity": {"events": {"sha256": "input"}},
        "code_identity": {"source_tree_sha256": "source"},
        "dependencies": {"python": "3.10.14", "hdbscan": "0.8.33"},
    }


def test_completed_stage_round_trip_verifies_outputs(tmp_path) -> None:
    output = tmp_path / "frontend.json"
    output.write_text('{"tracks":3}\n', encoding="utf-8")
    arguments = _identity_arguments()
    record = make_stage_record(
        **arguments,
        status="completed",
        output_files=output_file_identities({"frontend": output}),
        diagnostics={"track_count": 3},
    )
    checkpoint = tmp_path / "frontend.stage.json"
    write_stage_checkpoint(checkpoint, record)

    loaded = load_completed_stage(
        checkpoint,
        expected_stage="frontend",
        expected_fingerprint=record.fingerprint,
    )
    assert loaded.status == "completed"
    assert loaded.diagnostics["track_count"] == 3


def test_fingerprint_rejects_each_identity_change() -> None:
    arguments = _identity_arguments()
    baseline = make_stage_fingerprint(**arguments)
    for section, changed in (
        ("config", {"radius": 8.0}),
        ("input_identity", {"events": {"sha256": "other"}}),
        ("code_identity", {"source_tree_sha256": "other"}),
        ("dependencies", {"python": "3.10.13"}),
    ):
        candidate = dict(arguments)
        candidate[section] = changed
        assert make_stage_fingerprint(**candidate) != baseline


def test_failed_stage_is_never_resumable(tmp_path) -> None:
    arguments = _identity_arguments()
    record = make_stage_record(
        **arguments,
        status="failed",
        failure=StageFailure(
            "frontend_failure", "no_tracks", "no feature tracks survived",
        ),
    )
    checkpoint = tmp_path / "failed.stage.json"
    write_stage_checkpoint(checkpoint, record)
    with pytest.raises(ValueError, match="completed"):
        load_completed_stage(
            checkpoint,
            expected_stage="frontend",
            expected_fingerprint=record.fingerprint,
        )


def test_record_and_output_tampering_are_rejected(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text("original\n", encoding="utf-8")
    arguments = _identity_arguments()
    record = make_stage_record(
        **arguments,
        status="completed",
        output_files=output_file_identities({"result": output}),
    )
    checkpoint = tmp_path / "stage.json"
    write_stage_checkpoint(checkpoint, record)

    output.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output.*identity"):
        load_completed_stage(
            checkpoint,
            expected_stage="frontend",
            expected_fingerprint=record.fingerprint,
        )

    output.write_text("original\n", encoding="utf-8")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["record"]["diagnostics"]["track_count"] = 99
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record hash"):
        load_completed_stage(
            checkpoint,
            expected_stage="frontend",
            expected_fingerprint=record.fingerprint,
        )


def test_upstream_checkpoint_hash_chains_downstream_fingerprint(tmp_path) -> None:
    upstream = tmp_path / "upstream.json"
    upstream.write_text('{"record":1}\n', encoding="utf-8")
    args = _identity_arguments()
    first = make_stage_fingerprint(
        stage="colmap",
        config=args["config"],
        input_identity={"upstream": canonical_hash(upstream.read_text())},
        code_identity=args["code_identity"],
        dependencies=args["dependencies"],
    )
    upstream.write_text('{"record":2}\n', encoding="utf-8")
    second = make_stage_fingerprint(
        stage="colmap",
        config=args["config"],
        input_identity={"upstream": canonical_hash(upstream.read_text())},
        code_identity=args["code_identity"],
        dependencies=args["dependencies"],
    )
    assert first != second


def test_partial_json_and_wrong_stage_are_rejected(tmp_path) -> None:
    partial = tmp_path / "partial.json"
    partial.write_text('{"schema":', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid stage checkpoint"):
        load_completed_stage(
            partial, expected_stage="frontend", expected_fingerprint="x",
        )

    arguments = _identity_arguments()
    record = make_stage_record(**arguments, status="completed")
    checkpoint = tmp_path / "stage.json"
    write_stage_checkpoint(checkpoint, record)
    with pytest.raises(ValueError, match="different stage"):
        load_completed_stage(
            checkpoint,
            expected_stage="colmap",
            expected_fingerprint=record.fingerprint,
        )
