from __future__ import annotations

from dataclasses import asdict
import json
import platform
import sys

import pytest

from esfo.resources import (
    ResourceExecutionError,
    ResourceMeasurement,
    ResourceTimeoutError,
    measure_subprocess,
)


def test_measure_worker_records_output_wall_time_and_json_safe_result(tmp_path) -> None:
    stdout = tmp_path / "logs" / "stdout.txt"
    stderr = tmp_path / "logs" / "stderr.txt"
    result = measure_subprocess(
        (
            sys.executable,
            "-c",
            (
                "import sys,time; a=bytearray(4000000); "
                "print('ready'); print('note', file=sys.stderr); time.sleep(0.08)"
            ),
        ),
        stdout_path=stdout,
        stderr_path=stderr,
        sample_interval_ms=5.0,
        timeout_s=2.0,
        check=True,
    )

    assert result.returncode == 0
    assert result.timed_out is False
    assert result.wall_time_s >= 0.05
    assert result.peak_rss_bytes >= 0
    assert stdout.read_text(encoding="utf-8").strip() == "ready"
    assert stderr.read_text(encoding="utf-8").strip() == "note"
    assert result.measurement_method == (
        "linux_proc_process_tree_sampled"
        if platform.system() == "Linux"
        else "rusage_children_cumulative_fallback"
    )
    json.dumps(asdict(result), allow_nan=False)


def test_timeout_raises_with_complete_measurement() -> None:
    with pytest.raises(ResourceTimeoutError) as captured:
        measure_subprocess(
            (sys.executable, "-c", "import time; time.sleep(5)"),
            timeout_s=0.05,
            sample_interval_ms=5.0,
        )
    result = captured.value.measurement
    assert result.timed_out is True
    assert result.returncode != 0
    assert result.wall_time_s < 2.0


def test_timeout_can_be_returned_instead_of_raised() -> None:
    result = measure_subprocess(
        (sys.executable, "-c", "import time; time.sleep(5)"),
        timeout_s=0.05,
        sample_interval_ms=5.0,
        raise_on_timeout=False,
    )
    assert result.timed_out is True
    assert result.returncode != 0


def test_nonzero_return_code_can_fail_closed() -> None:
    with pytest.raises(ResourceExecutionError) as captured:
        measure_subprocess(
            (sys.executable, "-c", "raise SystemExit(7)"),
            timeout_s=2.0,
            check=True,
        )
    assert captured.value.measurement.returncode == 7
    assert captured.value.measurement.timed_out is False


def test_measurement_rejects_non_json_safe_or_inconsistent_fields() -> None:
    base = {
        "argv": ("worker",),
        "wall_time_s": 0.1,
        "peak_rss_bytes": 1,
        "platform": "Linux",
        "measurement_method": "linux_proc_process_tree_sampled",
        "sample_interval_ms": 10.0,
        "returncode": 0,
        "timed_out": False,
    }
    with pytest.raises(ValueError, match="wall_time_s"):
        ResourceMeasurement(**{**base, "wall_time_s": float("nan")})
    with pytest.raises(ValueError, match="peak_rss_bytes"):
        ResourceMeasurement(**{**base, "peak_rss_bytes": -1})
    with pytest.raises(ValueError, match="argv"):
        ResourceMeasurement(**{**base, "argv": ()})


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux /proc only")
def test_linux_measurement_sees_worker_memory() -> None:
    result = measure_subprocess(
        (
            sys.executable,
            "-c",
            "import time; a=bytearray(8000000); time.sleep(0.1)",
        ),
        timeout_s=2.0,
        sample_interval_ms=2.0,
        check=True,
    )
    assert result.peak_rss_bytes >= 4_000_000
