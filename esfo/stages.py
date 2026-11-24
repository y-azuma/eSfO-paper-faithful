"""Verified stage boundaries for the Gate-D reference pipeline.

The paper does not define checkpointing.  This module therefore adds only an
execution/provenance layer around the frozen method: a completed stage can be
reused when (and only when) its input, configuration, source, dependency, and
output identities still match exactly.  It never changes numerical method
outputs or retries a failed method stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .provenance import (
    atomic_write_json,
    canonical_hash,
    file_identity,
    run_fingerprint,
)


STAGE_RECORD_SCHEMA = "esfo-stage-record-v1"
STAGE_CHECKPOINT_SCHEMA = "esfo-stage-checkpoint-v1"
_STATUSES = {"completed", "failed"}


def _freeze_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType({str(key): item for key, item in value.items()})


@dataclass(frozen=True)
class StageFailure:
    """Stable first-failure record; traceback text is deliberately excluded."""

    category: str
    code: str
    message: str

    def __post_init__(self) -> None:
        for name in ("category", "code", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class StageRecord:
    """One complete or failed numerical stage with exact identities."""

    stage: str
    fingerprint: str
    status: str
    input_identity: Mapping[str, Any]
    config: Mapping[str, Any]
    code_identity: Mapping[str, Any]
    dependencies: Mapping[str, Any]
    output_files: Mapping[str, Mapping[str, Any]]
    diagnostics: Mapping[str, Any]
    failure: StageFailure | None = None
    upstream_record_sha256: str | None = None
    schema: str = STAGE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_RECORD_SCHEMA:
            raise ValueError("unsupported stage record schema")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ValueError("fingerprint must be a non-empty string")
        if self.status not in _STATUSES:
            raise ValueError("stage status must be completed or failed")
        if self.status == "completed" and self.failure is not None:
            raise ValueError("completed stage cannot carry a failure")
        if self.status == "failed" and not isinstance(self.failure, StageFailure):
            raise ValueError("failed stage requires StageFailure")
        if (self.upstream_record_sha256 is not None
                and (not isinstance(self.upstream_record_sha256, str)
                     or not self.upstream_record_sha256.strip())):
            raise ValueError("upstream record hash must be non-empty or None")
        for name in (
            "input_identity", "config", "code_identity", "dependencies",
            "output_files", "diagnostics",
        ):
            object.__setattr__(self, name, _freeze_mapping(
                getattr(self, name), name,
            ))
        for label, identity in self.output_files.items():
            if not isinstance(identity, Mapping):
                raise TypeError(f"output identity {label!r} must be a mapping")
            required = {"path", "size_bytes", "sha256"}
            if not required.issubset(identity):
                raise ValueError(
                    f"output identity {label!r} lacks {sorted(required - set(identity))}",
                )


def make_stage_fingerprint(
    *,
    stage: str,
    config: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> str:
    """Bind every identity which is allowed to authorize stage reuse."""

    return run_fingerprint(
        stage=stage,
        config=config,
        dependencies=dependencies,
        input_identity=input_identity,
        code=code_identity,
    )


def output_file_identities(
    outputs: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """Hash named stage outputs after their writers have closed them."""

    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping")
    return {
        str(label): file_identity(path)
        for label, path in sorted(outputs.items(), key=lambda item: str(item[0]))
    }


def make_stage_record(
    *,
    stage: str,
    status: str,
    config: Mapping[str, Any],
    input_identity: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    output_files: Mapping[str, Mapping[str, Any]] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    failure: StageFailure | None = None,
    upstream_record_sha256: str | None = None,
) -> StageRecord:
    fingerprint = make_stage_fingerprint(
        stage=stage,
        config=config,
        input_identity=input_identity,
        code_identity=code_identity,
        dependencies=dependencies,
    )
    return StageRecord(
        stage=stage,
        fingerprint=fingerprint,
        status=status,
        input_identity=input_identity,
        config=config,
        code_identity=code_identity,
        dependencies=dependencies,
        output_files={} if output_files is None else output_files,
        diagnostics={} if diagnostics is None else diagnostics,
        failure=failure,
        upstream_record_sha256=upstream_record_sha256,
    )


def _record_payload(record: StageRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": record.schema,
        "stage": record.stage,
        "fingerprint": record.fingerprint,
        "status": record.status,
        "input_identity": record.input_identity,
        "config": record.config,
        "code_identity": record.code_identity,
        "dependencies": record.dependencies,
        "output_files": record.output_files,
        "diagnostics": record.diagnostics,
        "upstream_record_sha256": record.upstream_record_sha256,
        "failure": None,
    }
    if record.failure is not None:
        payload["failure"] = {
            "category": record.failure.category,
            "code": record.failure.code,
            "message": record.failure.message,
        }
    return payload


def write_stage_checkpoint(path: str | Path, record: StageRecord) -> None:
    """Atomically persist one immutable stage-boundary record."""

    if not isinstance(record, StageRecord):
        raise TypeError("record must be StageRecord")
    payload = _record_payload(record)
    atomic_write_json(path, {
        "schema": STAGE_CHECKPOINT_SCHEMA,
        "fingerprint": record.fingerprint,
        "record": payload,
        "record_sha256": canonical_hash(payload),
    })


def _parse_record(payload: Mapping[str, Any]) -> StageRecord:
    failure_value = payload.get("failure")
    failure = None
    if failure_value is not None:
        if not isinstance(failure_value, Mapping):
            raise ValueError("stage failure is malformed")
        try:
            failure = StageFailure(
                str(failure_value["category"]),
                str(failure_value["code"]),
                str(failure_value["message"]),
            )
        except KeyError as error:
            raise ValueError("stage failure is malformed") from error
    try:
        return StageRecord(
            schema=str(payload["schema"]),
            stage=str(payload["stage"]),
            fingerprint=str(payload["fingerprint"]),
            status=str(payload["status"]),
            input_identity=payload["input_identity"],
            config=payload["config"],
            code_identity=payload["code_identity"],
            dependencies=payload["dependencies"],
            output_files=payload["output_files"],
            diagnostics=payload["diagnostics"],
            failure=failure,
            upstream_record_sha256=payload.get("upstream_record_sha256"),
        )
    except KeyError as error:
        raise ValueError("stage record is missing required fields") from error


def load_completed_stage(
    path: str | Path,
    *,
    expected_stage: str,
    expected_fingerprint: str,
    verify_outputs: bool = True,
) -> StageRecord:
    """Resume only a complete, untampered, exact-identity stage.

    Failed stages, partial JSON, mismatched fingerprints, and missing or
    modified outputs are all rejected.  There is no permissive recovery path.
    """

    import json

    path = Path(path)
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid stage checkpoint: {path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("stage checkpoint must be a JSON object")
    if checkpoint.get("schema") != STAGE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported stage checkpoint schema")
    record_payload = checkpoint.get("record")
    if not isinstance(record_payload, Mapping):
        raise ValueError("stage checkpoint record is missing")
    if checkpoint.get("record_sha256") != canonical_hash(record_payload):
        raise ValueError("stage checkpoint record hash mismatch")
    if checkpoint.get("fingerprint") != expected_fingerprint:
        raise ValueError("stage checkpoint fingerprint mismatch")
    record = _parse_record(record_payload)
    if record.fingerprint != expected_fingerprint:
        raise ValueError("stage record fingerprint mismatch")
    if record.stage != expected_stage:
        raise ValueError("stage checkpoint belongs to a different stage")
    if record.status != "completed":
        raise ValueError("only completed stages may be resumed")
    if verify_outputs:
        for label, expected in record.output_files.items():
            try:
                actual = file_identity(expected["path"])
            except (ValueError, OSError) as error:
                raise ValueError(f"stage output {label!r} is missing") from error
            if (actual["size_bytes"] != expected["size_bytes"]
                    or actual["sha256"] != expected["sha256"]):
                raise ValueError(f"stage output {label!r} identity mismatch")
    return record


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Identity used as the next stage's immutable upstream input."""

    return file_identity(path)


__all__ = [
    "STAGE_CHECKPOINT_SCHEMA", "STAGE_RECORD_SCHEMA", "StageFailure",
    "StageRecord", "checkpoint_identity", "load_completed_stage",
    "make_stage_fingerprint", "make_stage_record", "output_file_identities",
    "write_stage_checkpoint",
]
