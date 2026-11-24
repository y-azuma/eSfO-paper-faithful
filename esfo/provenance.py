"""Deterministic identities and atomic stage-boundary persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        # ``dataclasses.asdict`` deep-copies leaves and cannot copy immutable
        # MappingProxyType metadata.  Walk fields directly so frozen runtime
        # results remain serializable without weakening their immutability.
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_json(value: Any) -> str:
    """Serialize without NaN/Infinity or ordering-dependent whitespace."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing input file: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def source_tree_hash(
    repository: str | Path,
    *,
    roots: Iterable[str] = ("esfo", "tests"),
    top_level_files: Iterable[str] = ("pyproject.toml",),
) -> str:
    """Hash executable sources including uncommitted working-tree content."""
    repository = Path(repository).resolve()
    files: list[Path] = []
    allowed = {".py", ".toml", ".json", ".jsonc", ".yml", ".yaml", ".txt"}
    for root_name in roots:
        root = repository / root_name
        if root.is_dir():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
                and path.suffix.lower() in allowed
            )
    for name in top_level_files:
        path = repository / name
        if path.is_file():
            files.append(path)
    # Small synthetic fixtures often pass a directory that is itself the
    # source root rather than a repository containing ``esfo/``.  Falling
    # back only when none of the declared roots exist keeps real repository
    # hashes from accidentally traversing virtual environments.
    if not files:
        files.extend(
            path for path in repository.rglob("*")
            if path.is_file()
            and not ({".git", ".venv", "venv", "__pycache__"} & set(path.parts))
            and path.suffix.lower() in allowed
        )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda item: str(item.relative_to(repository))):
        relative = str(path.relative_to(repository)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def code_identity(repository: str | Path) -> dict[str, Any]:
    repository = Path(repository).resolve()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    return {
        "git_revision": revision.stdout.strip() or None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_tree_sha256": source_tree_hash(repository),
    }


def dependency_manifest(
    names: Iterable[str] = (
        "numpy", "scipy", "hdbscan", "scikit-learn", "joblib",
        "threadpoolctl", "Cython",
    ),
) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }


def run_fingerprint(
    *,
    stage: str,
    config: Any,
    dependencies: Any,
    source_hash: str | None = None,
    input_hashes: Any = None,
    input_identity: Any = None,
    code: Any = None,
) -> str:
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")
    if source_hash is not None:
        if code is not None:
            raise ValueError("provide source_hash or code, not both")
        if not isinstance(source_hash, str) or not source_hash.strip():
            raise ValueError("source hash must be non-empty")
        code = {"source_tree_sha256": source_hash}
    if input_hashes is not None:
        if input_identity is not None:
            raise ValueError("provide input_hashes or input_identity, not both")
        input_identity = input_hashes
    if code is None or input_identity is None or dependencies is None:
        raise ValueError(
            "source/code identity, input identity, and dependencies are required"
        )
    if not code or not input_identity or not dependencies:
        raise ValueError(
            "source/code identity, input identity, and dependencies must be non-empty"
        )
    return canonical_hash({
        "schema": "esfo-stage-v1",
        "stage": stage,
        "config": config,
        "input_identity": input_identity,
        "code_identity": code,
        "dependencies": dependencies,
    })


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Replace a JSON file only after a same-directory write and fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        _jsonable(value), sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Persist the directory entry where the platform supports directory fsync.
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_verified_checkpoint(path: str | Path, expected_fingerprint: str) -> Any:
    """Load a stage checkpoint only when its exact identity matches."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint fingerprint mismatch")
    if "record" not in payload or payload.get("record_sha256") != canonical_hash(
        payload["record"]
    ):
        raise ValueError("checkpoint record hash mismatch")
    return payload["record"]


__all__ = [
    "atomic_write_json", "canonical_hash", "canonical_json", "code_identity",
    "dependency_manifest", "file_identity", "file_sha256",
    "load_verified_checkpoint", "run_fingerprint", "source_tree_hash",
]
