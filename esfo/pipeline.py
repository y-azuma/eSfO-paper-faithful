"""Stage-boundary orchestration for the frozen eSfO reference path.

This is execution plumbing, not an alternative method.  A stage runs once,
persists an exact-identity checkpoint, and downstream stages consume that
checkpoint identity.  Failed or mismatched checkpoints are never retried or
silently replaced inside the same run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .evaluation import SceneFailureRecord, classify_scene_failure
from .stages import (
    StageFailure,
    StageRecord,
    checkpoint_identity,
    load_completed_stage,
    make_stage_fingerprint,
    make_stage_record,
    output_file_identities,
    write_stage_checkpoint,
)


@dataclass(frozen=True)
class StageExecution:
    """Files and diagnostics produced by one successful stage callback."""

    outputs: Mapping[str, str | Path]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics)),
        )


StageCallback = Callable[[Path, StageRecord | None], StageExecution]


@dataclass(frozen=True)
class PipelineStage:
    """One ordered reference stage and its frozen configuration."""

    name: str
    config: Mapping[str, Any]
    callback: StageCallback
    failure_stage: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must be a non-empty string")
        if not isinstance(self.config, Mapping):
            raise TypeError("stage config must be a mapping")
        if not callable(self.callback):
            raise TypeError("stage callback must be callable")
        if (self.failure_stage is not None
                and (not isinstance(self.failure_stage, str)
                     or not self.failure_stage.strip())):
            raise ValueError("failure_stage must be non-empty or None")
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True)
class PipelineRunResult:
    records: tuple[StageRecord, ...]
    resumed_stages: tuple[str, ...]
    first_failure: SceneFailureRecord | None

    @property
    def completed(self) -> bool:
        return self.first_failure is None and all(
            record.status == "completed" for record in self.records
        )


def _checkpoint_path(work_root: Path, index: int, stage: PipelineStage) -> Path:
    safe = stage.name.replace("/", "_").replace(" ", "_")
    return work_root / "stages" / f"{index:02d}_{safe}.stage.json"


def _failure_to_stage_failure(failure: SceneFailureRecord) -> StageFailure:
    return StageFailure(
        category=failure.code,
        code=failure.detail_code or failure.code,
        message=failure.message,
    )


def run_reference_pipeline(
    stages: Sequence[PipelineStage],
    *,
    work_root: str | Path,
    initial_input_identity: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    resume: bool = True,
) -> PipelineRunResult:
    """Run or exactly resume an ordered reference pipeline.

    A checkpoint already present at a stage boundary must verify exactly.  A
    mismatch, failed checkpoint, modified output, or malformed partial file is
    surfaced rather than overwritten.  The caller may start a new run root;
    this function never performs adaptive recovery.
    """

    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    stage_root = work_root / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    stages_tuple = tuple(stages)
    if not stages_tuple:
        raise ValueError("at least one pipeline stage is required")
    if len({stage.name for stage in stages_tuple}) != len(stages_tuple):
        raise ValueError("pipeline stage names must be unique")
    for name, value in (
        ("initial_input_identity", initial_input_identity),
        ("code_identity", code_identity),
        ("dependencies", dependencies),
    ):
        if not isinstance(value, Mapping) or not value:
            raise ValueError(f"{name} must be a non-empty mapping")

    records: list[StageRecord] = []
    resumed: list[str] = []
    upstream_checkpoint: Path | None = None
    upstream_record: StageRecord | None = None
    for index, stage in enumerate(stages_tuple, start=1):
        input_identity: Mapping[str, Any]
        upstream_hash: str | None = None
        if upstream_checkpoint is None:
            input_identity = dict(initial_input_identity)
        else:
            upstream = checkpoint_identity(upstream_checkpoint)
            input_identity = {"upstream_checkpoint": upstream}
            upstream_hash = upstream["sha256"]
        fingerprint = make_stage_fingerprint(
            stage=stage.name,
            config=stage.config,
            input_identity=input_identity,
            code_identity=code_identity,
            dependencies=dependencies,
        )
        checkpoint = _checkpoint_path(work_root, index, stage)
        if checkpoint.exists():
            if not resume:
                raise ValueError(
                    f"stage checkpoint already exists and resume is disabled: {checkpoint}",
                )
            record = load_completed_stage(
                checkpoint,
                expected_stage=stage.name,
                expected_fingerprint=fingerprint,
            )
            records.append(record)
            resumed.append(stage.name)
            upstream_checkpoint, upstream_record = checkpoint, record
            continue

        stage_work = work_root / "work" / f"{index:02d}_{stage.name}"
        stage_work.mkdir(parents=True, exist_ok=False)
        try:
            execution = stage.callback(stage_work, upstream_record)
            if not isinstance(execution, StageExecution):
                raise TypeError("stage callback must return StageExecution")
            outputs = output_file_identities(execution.outputs)
            record = make_stage_record(
                stage=stage.name,
                status="completed",
                config=stage.config,
                input_identity=input_identity,
                code_identity=code_identity,
                dependencies=dependencies,
                output_files=outputs,
                diagnostics=execution.diagnostics,
                upstream_record_sha256=upstream_hash,
            )
        except Exception as error:
            failure = classify_scene_failure(
                error, stage=stage.failure_stage or stage.name,
            )
            record = make_stage_record(
                stage=stage.name,
                status="failed",
                config=stage.config,
                input_identity=input_identity,
                code_identity=code_identity,
                dependencies=dependencies,
                failure=_failure_to_stage_failure(failure),
                diagnostics={"first_failure": failure},
                upstream_record_sha256=upstream_hash,
            )
            write_stage_checkpoint(checkpoint, record)
            records.append(record)
            return PipelineRunResult(tuple(records), tuple(resumed), failure)

        write_stage_checkpoint(checkpoint, record)
        records.append(record)
        upstream_checkpoint, upstream_record = checkpoint, record

    return PipelineRunResult(tuple(records), tuple(resumed), None)


__all__ = [
    "PipelineRunResult", "PipelineStage", "StageExecution",
    "run_reference_pipeline",
]
