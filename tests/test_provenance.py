from __future__ import annotations

import hashlib
import json

import numpy as np

from esfo.data import EventSequence
from esfo.frontend import run_frontend_raw
from esfo.provenance import (
    atomic_write_json,
    canonical_hash,
    dependency_manifest,
    file_sha256,
    load_verified_checkpoint,
    run_fingerprint,
    source_tree_hash,
)


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}

    assert canonical_hash(left) == canonical_hash(right)
    assert canonical_hash(left) != canonical_hash({"a": {"x": 2}, "b": [2, 3]})


def test_file_and_source_tree_hashes_are_content_sensitive(tmp_path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    first = root / "a.py"
    second = root / "b.py"
    first.write_text("x = 1\n", encoding="utf-8")
    second.write_text("y = 2\n", encoding="utf-8")

    assert file_sha256(first) == hashlib.sha256(b"x = 1\n").hexdigest()
    before = source_tree_hash(root)
    second.write_text("y = 3\n", encoding="utf-8")
    after = source_tree_hash(root)

    assert before != after


def test_run_fingerprint_binds_config_source_and_inputs() -> None:
    dependencies = {"python": "3.10.6", "packages": {"hdbscan": "0.8.33"}}
    base = run_fingerprint(
        stage="gate-a",
        config={"lambda_radius": 7.0},
        dependencies=dependencies,
        source_hash="source-a",
        input_hashes={"events": "input-a"},
    )
    assert base == run_fingerprint(
        stage="gate-a",
        config={"lambda_radius": 7.0},
        dependencies=dependencies,
        source_hash="source-a",
        input_hashes={"events": "input-a"},
    )
    assert base != run_fingerprint(
        stage="gate-a",
        config={"lambda_radius": 7.0},
        dependencies=dependencies,
        source_hash="source-b",
        input_hashes={"events": "input-a"},
    )
    assert base != run_fingerprint(
        stage="gate-a",
        config={"lambda_radius": 7.0},
        dependencies=dependencies,
        source_hash="source-a",
        input_hashes={"events": "input-b"},
    )
    assert base != run_fingerprint(
        stage="gate-a", config={"lambda_radius": 7.0},
        dependencies={"python": "3.10.7"}, source_hash="source-a",
        input_hashes={"events": "input-a"},
    )


def test_dependency_manifest_records_runtime_and_transitive_stack() -> None:
    manifest = dependency_manifest()
    assert manifest["python"]["version"].startswith("3.10.")
    assert manifest["platform"]["system"]
    assert manifest["packages"]["hdbscan"] == "0.8.33"
    for name in ("scikit-learn", "joblib", "threadpoolctl", "Cython"):
        assert name in manifest["packages"]


def test_run_fingerprint_rejects_empty_identity_sections() -> None:
    for field in ("dependencies", "source_hash", "input_hashes"):
        arguments = {
            "stage": "gate-a",
            "config": {},
            "dependencies": {"python": "3.10.6"},
            "source_hash": "source",
            "input_hashes": {"events": "input"},
        }
        arguments[field] = {} if field != "source_hash" else ""
        with np.testing.assert_raises_regex(ValueError, "non-empty"):
            run_fingerprint(**arguments)


def test_atomic_write_json_replaces_complete_document(tmp_path) -> None:
    target = tmp_path / "nested" / "manifest.json"
    atomic_write_json(target, {"step": 1, "value": "old"})
    atomic_write_json(target, {"step": 2, "value": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "step": 2,
        "value": "new",
    }
    assert not list(target.parent.glob(f".{target.name}.*"))


def test_frontend_result_is_hashable_and_atomically_serializable(tmp_path) -> None:
    events = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=8, height=8,
    )
    result = run_frontend_raw(events)
    first = canonical_hash(result)
    assert first == canonical_hash(result)

    target = tmp_path / "frontend.json"
    atomic_write_json(target, result)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["clustering"]["metadata"]["parameters"][
        "min_cluster_size"
    ] == 10
    assert payload["association"]["diagnostics"] == {
        "merged_track_count": 0,
        "temporal_overlap_pairs": [],
        "temporal_overlap_rule": "closed_cluster_time_span_intersection",
        "temporally_overlapping_cluster_pair_count": 0,
        "temporally_overlapping_track_count": 0,
        "temporally_overlapping_track_ids": [],
    }
    assert payload["corner_mask"] == []


def test_checkpoint_loader_rejects_record_tampering(tmp_path) -> None:
    target = tmp_path / "checkpoint.json"
    record = {"clusters": 3}
    payload = {
        "fingerprint": "expected",
        "record": record,
        "record_sha256": canonical_hash(record),
    }
    atomic_write_json(target, payload)
    assert load_verified_checkpoint(target, "expected") == record

    payload["record"]["clusters"] = 4
    atomic_write_json(target, payload)
    with np.testing.assert_raises_regex(ValueError, "hash"):
        load_verified_checkpoint(target, "expected")
