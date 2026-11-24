from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from esfo.audit_fixtures import (
    EFAST_ORACLE_PULSE_COUNT,
    EFAST_ORACLE_RANDOM_EVENT_COUNT,
    EFAST_ORACLE_RANDOM_SEED,
    build_efast_oracle_events,
)
from esfo.data import EventSequence
from esfo.efast import EFastDetector


UPSTREAM_COMMIT = "e6c88625479a62e649ff391e2489ef6470955411"
UPSTREAM_FILES = {
    "corner_event_detector/src/fast_detector.cpp": (
        "39dfb374dad435a2a143e0b2f32e9f2e6023139baa433e192a20b0fed48dae81"
    ),
    "corner_event_detector/include/corner_event_detector/fast_detector.h": (
        "511b4423d534c75202d039fbcd02863a3d6d2fe9eb3f6737d33357fa6d6f32bf"
    ),
    "corner_event_detector/include/corner_event_detector/detector.h": (
        "0fa7f2cee3a78d5d7c8c8d65d4b69d56e5a7fac0d734ef65372eebc7e93a8756"
    ),
}


def test_independent_detector_matches_pinned_external_gpl_oracle() -> None:
    manifest = json.loads(
        (Path(__file__).parent / "fixtures" / "efast_upstream_oracle.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "esfo-efast-upstream-oracle-v1"
    assert manifest["license_boundary"] == (
        "external_GPL_checkout_test_only_not_in_wheel"
    )
    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert manifest["upstream_file_sha256"] == UPSTREAM_FILES
    assert manifest["input_generator"] == {
        "random_seed": EFAST_ORACLE_RANDOM_SEED,
        "random_event_count": EFAST_ORACLE_RANDOM_EVENT_COUNT,
        "pulse_count": EFAST_ORACLE_PULSE_COUNT,
        "timestamps": "positive_integer_microseconds_converted_to_seconds",
    }

    t_us, x, y, p = build_efast_oracle_events()
    inputs = np.column_stack((t_us, x, y, p)).astype(np.int64, copy=False)
    assert hashlib.sha256(inputs.tobytes()).hexdigest() == manifest["input_sha256"]
    events = EventSequence.from_arrays(
        t_us=t_us, x=x, y=y, p=p, width=240, height=180,
    )
    mask = EFastDetector(240, 180).detect_events(events.to_relative_ms())
    packed = np.packbits(mask, bitorder="little").tobytes()

    assert mask.size == manifest["event_count"]
    assert int(np.count_nonzero(mask)) == manifest["corner_count"]
    assert hashlib.sha256(packed).hexdigest() == manifest["mask_sha256"]
