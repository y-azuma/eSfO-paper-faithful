#!/usr/bin/env python3
"""Regenerate the isolated pinned-upstream eFAST decision oracle.

No upstream source is copied into this repository.  The script compiles the
external GPL file in a temporary directory with clean-room interface stubs,
runs deterministic positive-timestamp events, and emits only a boolean mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from esfo.audit_fixtures import (
    EFAST_ORACLE_PULSE_COUNT,
    EFAST_ORACLE_RANDOM_EVENT_COUNT,
    EFAST_ORACLE_RANDOM_SEED,
    build_efast_oracle_events,
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_stubs(root: Path) -> None:
    (root / "include" / "ros").mkdir(parents=True)
    (root / "include" / "dvs_msgs").mkdir(parents=True)
    (root / "include" / "Eigen").mkdir(parents=True)
    (root / "include" / "ros" / "ros.h").write_text(r"""
#pragma once
#include <cstdint>
#include <string>
namespace ros {
struct Time { double value = 0.0; double toSec() const { return value; } };
class Publisher { public: template<class T> void publish(const T&) {} };
class Subscriber {};
class NodeHandle {
 public:
  template<class T> Publisher advertise(const std::string&, int) { return {}; }
  template<class C, class M> Subscriber subscribe(
      const std::string&, int, void (C::*)(const M&), C*) { return {}; }
};
}
#define ROS_INFO(...) do {} while (0)
""", encoding="utf-8")
    (root / "include" / "dvs_msgs" / "EventArray.h").write_text(r"""
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <ros/ros.h>
namespace dvs_msgs {
struct Event { std::uint16_t x=0, y=0; bool polarity=false; ros::Time ts; };
struct EventArray {
  using ConstPtr = std::shared_ptr<const EventArray>;
  int header=0; std::uint16_t width=240, height=180; std::vector<Event> events;
};
}
""", encoding="utf-8")
    # The upstream uses MatrixXd only as a coordinate-addressed SAE.  This
    # stub intentionally stores the (x,y) pair directly and does no algebra.
    (root / "include" / "Eigen" / "Dense").write_text(r"""
#pragma once
#include <array>
namespace Eigen {
class MatrixXd {
 public:
  static MatrixXd Zero(int, int) { return MatrixXd(); }
  double& operator()(int x, int y) { return values_[x * 180 + y]; }
  const double& operator()(int x, int y) const { return values_[x * 180 + y]; }
 private:
  std::array<double, 240 * 180> values_{};
};
}
""", encoding="utf-8")
    (root / "detector_stub.cpp").write_text(r"""
#include "corner_event_detector/detector.h"
namespace corner_event_detector {
Detector::Detector(bool) : total_time_(0.), total_events_(0), total_corners_(0) {}
Detector::~Detector() = default;
void Detector::eventCallback(const dvs_msgs::EventArray::ConstPtr&) {}
}
""", encoding="utf-8")
    (root / "main.cpp").write_text(r"""
#include <iostream>
#include "corner_event_detector/fast_detector.h"
int main() {
  corner_event_detector::FastDetector detector(false);
  double time_s; int x, y, polarity;
  while (std::cin >> time_s >> x >> y >> polarity) {
    dvs_msgs::Event event;
    event.ts.value = time_s;
    event.x = static_cast<std::uint16_t>(x);
    event.y = static_cast<std::uint16_t>(y);
    event.polarity = polarity != 0;
    std::cout << (detector.isFeature(event) ? '1' : '0');
  }
  std::cout << '\n';
}
""", encoding="utf-8")


def regenerate(upstream: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f"expected upstream {UPSTREAM_COMMIT}, got {revision}")
    for relative, expected in UPSTREAM_FILES.items():
        actual = _sha256(upstream / relative)
        if actual != expected:
            raise RuntimeError(f"upstream identity mismatch: {relative}")
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required")
    with tempfile.TemporaryDirectory(prefix="esfo-gpl-efast-") as temporary:
        root = Path(temporary)
        _write_stubs(root)
        binary = root / "efast_oracle"
        subprocess.run([
            compiler, "-std=c++17", "-O2", "-DNDEBUG", "-DUTILS_TIMER_H_",
            "-I", str(root / "include"),
            "-I", str(upstream / "corner_event_detector" / "include"),
            str(upstream / "corner_event_detector" / "src" / "fast_detector.cpp"),
            str(root / "detector_stub.cpp"), str(root / "main.cpp"),
            "-o", str(binary),
        ], check=True)
        t_us, x, y, p = build_efast_oracle_events()
        input_text = "".join(
            f"{time / 1_000_000.0:.9f} {xi} {yi} {pi}\n"
            for time, xi, yi, pi in zip(t_us, x, y, p)
        )
        output = subprocess.run(
            [str(binary)], input=input_text,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    if len(output) != t_us.size or set(output) - {"0", "1"}:
        raise RuntimeError("upstream oracle returned an invalid mask")
    mask = np.fromiter((char == "1" for char in output), dtype=np.bool_)
    packed = np.packbits(mask, bitorder="little").tobytes()
    inputs = np.column_stack((t_us, x, y, p)).astype(np.int64, copy=False)
    return {
        "schema": "esfo-efast-upstream-oracle-v1",
        "license_boundary": "external_GPL_checkout_test_only_not_in_wheel",
        "upstream_repository": "https://github.com/uzh-rpg/rpg_corner_events",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_file_sha256": UPSTREAM_FILES,
        "input_generator": {
            "random_seed": EFAST_ORACLE_RANDOM_SEED,
            "random_event_count": EFAST_ORACLE_RANDOM_EVENT_COUNT,
            "pulse_count": EFAST_ORACLE_PULSE_COUNT,
            "timestamps": "positive_integer_microseconds_converted_to_seconds",
        },
        "input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
        "event_count": int(mask.size),
        "corner_count": int(np.count_nonzero(mask)),
        "mask_encoding": "numpy.packbits(bitorder=little)",
        "mask_sha256": hashlib.sha256(packed).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    record = regenerate(arguments.upstream_dir.resolve())
    arguments.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
