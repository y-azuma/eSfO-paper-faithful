"""Isolated subprocess resource measurements for Gate-D evidence.

On Linux, the canonical benchmark path samples the resident-set size of the
complete ``/proc`` process tree rooted at the launched worker.  This includes
workers spawned by HDBSCAN or COLMAP and avoids attributing earlier pipeline
stages to the measured peak.  Other platforms use an explicitly labelled
fallback and are not accepted as the canonical N6 memory measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import platform as platform_module
import resource
import signal
import subprocess
import time
from typing import Sequence


@dataclass(frozen=True)
class ResourceMeasurement:
    """JSON-safe outcome for one isolated worker process."""

    argv: tuple[str, ...]
    wall_time_s: float
    peak_rss_bytes: int
    platform: str
    measurement_method: str
    sample_interval_ms: float
    returncode: int
    timed_out: bool
    stdout_path: str | None = None
    stderr_path: str | None = None

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        object.__setattr__(self, "argv", argv)
        for name in ("wall_time_s", "sample_interval_ms"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if self.sample_interval_ms <= 0.0:
            raise ValueError("sample_interval_ms must be positive")
        if (isinstance(self.peak_rss_bytes, bool)
                or not isinstance(self.peak_rss_bytes, int)
                or self.peak_rss_bytes < 0):
            raise ValueError("peak_rss_bytes must be a non-negative integer")
        if (isinstance(self.returncode, bool)
                or not isinstance(self.returncode, int)):
            raise ValueError("returncode must be an integer")
        if not isinstance(self.timed_out, bool):
            raise ValueError("timed_out must be boolean")
        for name in ("platform", "measurement_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("stdout_path", "stderr_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")


class ResourceTimeoutError(TimeoutError):
    """Worker timeout carrying the complete measured outcome."""

    def __init__(self, measurement: ResourceMeasurement):
        self.measurement = measurement
        super().__init__(
            "resource_timeout: worker exceeded its wall-clock limit "
            f"after {measurement.wall_time_s:.6f} s",
        )


class ResourceExecutionError(RuntimeError):
    """Nonzero worker result carrying the complete measured outcome."""

    def __init__(self, measurement: ResourceMeasurement):
        self.measurement = measurement
        super().__init__(
            "resource_worker_failed: worker exited with return code "
            f"{measurement.returncode}",
        )


def _linux_children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        source = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ()
    if not source:
        return ()
    children: list[int] = []
    for token in source.split():
        try:
            child = int(token)
        except ValueError:
            continue
        if child > 0:
            children.append(child)
    return tuple(children)


def _linux_process_tree(root_pid: int) -> tuple[int, ...]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen or pid <= 0:
            continue
        seen.add(pid)
        pending.extend(_linux_children(pid))
    return tuple(sorted(seen))


def _linux_rss_bytes(pid: int) -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text(
            encoding="ascii", errors="strict",
        ).splitlines()
    except (OSError, UnicodeError):
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return max(0, int(fields[1])) * 1024
                except ValueError:
                    return 0
    return 0


def _linux_tree_rss_bytes(root_pid: int) -> int:
    return sum(_linux_rss_bytes(pid) for pid in _linux_process_tree(root_pid))


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait()


def measure_subprocess(
    argv: Sequence[str],
    *,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    timeout_s: float | None = None,
    sample_interval_ms: float = 10.0,
    check: bool = False,
    raise_on_timeout: bool = True,
) -> ResourceMeasurement:
    """Run one isolated worker and measure wall time and peak memory.

    When output paths are omitted, output is discarded rather than buffered;
    this prevents a verbose worker from blocking on a full pipe.  A timeout
    terminates only the newly created process group.  With the default
    ``raise_on_timeout=True`` it raises :class:`ResourceTimeoutError`, whose
    ``measurement`` field preserves the return code and resource evidence.
    """

    command = tuple(argv)
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("argv must contain non-empty strings")
    if (isinstance(sample_interval_ms, bool)
            or not isinstance(sample_interval_ms, (int, float))
            or not math.isfinite(float(sample_interval_ms))
            or float(sample_interval_ms) <= 0.0):
        raise ValueError("sample_interval_ms must be positive and finite")
    interval_s = float(sample_interval_ms) / 1000.0
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise ValueError("timeout_s must be positive and finite or None")
    stdout_file = None
    stderr_file = None
    stdout_name = None if stdout_path is None else str(Path(stdout_path).resolve())
    stderr_name = None if stderr_path is None else str(Path(stderr_path).resolve())
    try:
        if stdout_name is not None:
            Path(stdout_name).parent.mkdir(parents=True, exist_ok=True)
            stdout_file = Path(stdout_name).open("wb")
        if stderr_name is not None:
            Path(stderr_name).parent.mkdir(parents=True, exist_ok=True)
            stderr_file = Path(stderr_name).open("wb")
        system = platform_module.system()
        fallback_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL if stdout_file is None else stdout_file,
            stderr=subprocess.DEVNULL if stderr_file is None else stderr_file,
            start_new_session=True,
        )
        peak = 0
        timed_out = False
        while process.poll() is None:
            if system == "Linux":
                peak = max(peak, _linux_tree_rss_bytes(process.pid))
            elapsed = time.perf_counter() - started
            if timeout_s is not None and elapsed >= float(timeout_s):
                timed_out = True
                _terminate_process_group(process)
                break
            time.sleep(interval_s)
        if system == "Linux":
            peak = max(peak, _linux_tree_rss_bytes(process.pid))
            method = "linux_proc_process_tree_sampled"
        else:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            # Darwin reports bytes; the other Unix platforms using this
            # fallback conventionally report KiB.  The method name makes
            # clear that this is cumulative and non-canonical for N6.
            peak = max(0, int(max(usage, fallback_before)))
            if system != "Darwin":
                peak *= 1024
            method = "rusage_children_cumulative_fallback"
        wall = time.perf_counter() - started
        returncode = process.wait()
    finally:
        if stdout_file is not None:
            stdout_file.close()
        if stderr_file is not None:
            stderr_file.close()

    measurement = ResourceMeasurement(
        argv=command,
        wall_time_s=wall,
        peak_rss_bytes=int(peak),
        platform=system,
        measurement_method=method,
        sample_interval_ms=float(sample_interval_ms),
        returncode=int(returncode),
        timed_out=timed_out,
        stdout_path=stdout_name,
        stderr_path=stderr_name,
    )
    if timed_out and raise_on_timeout:
        raise ResourceTimeoutError(measurement)
    if check and returncode != 0:
        raise ResourceExecutionError(measurement)
    return measurement


__all__ = [
    "ResourceExecutionError", "ResourceMeasurement", "ResourceTimeoutError",
    "measure_subprocess",
]
