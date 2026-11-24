"""Validated configuration for the frozen eSfO front-end.

The defaults in :class:`FrontendConfig` are the Gate-0 reference
interpretation. Constructing a different configuration is allowed for an
explicit sensitivity experiment, but :func:`assert_reference_config` prevents
such an output from being labelled as the paper-reference path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


def _finite(name: str, value: float, *, positive: bool = False,
            non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class FrontendConfig:
    """Gate-A constants and result-independent interpretations."""

    timestamp_scale_us_to_ms: float = 1.0 / 1000.0

    # eFAST, Mueggler et al. The two simultaneous arc tests are fixed.
    efast_inner_min_arc: int = 3
    efast_inner_max_arc: int = 6
    efast_outer_min_arc: int = 4
    efast_outer_max_arc: int = 8

    # eSfO Eq. (2).
    density_radius: float = 7.0
    density_include_query: bool = True
    density_neighbors: str = "raw_events"
    density_distance_domain: str = "t_ms_x_y"
    density_threshold_scope: str = "per_polarity_whole_sequence"

    # eSfO Sec. 4.2 and Table 2. Arguments which otherwise inherit hdbscan
    # defaults are explicit so provenance contains the effective algorithm.
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_epsilon: float = 5.0
    hdbscan_metric: str = "euclidean"
    hdbscan_alpha: float = 1.0
    hdbscan_algorithm: str = "best"
    hdbscan_leaf_size: int = 40
    hdbscan_approx_min_span_tree: bool = True
    hdbscan_gen_min_span_tree: bool = False
    hdbscan_core_dist_n_jobs: int = 4
    hdbscan_cluster_selection_method: str = "eom"
    hdbscan_allow_single_cluster: bool = False
    hdbscan_prediction_data: bool = False
    hdbscan_match_reference_implementation: bool = False

    # eSfO Eqs. (3)-(4).
    association_descriptor_events: int = 5
    association_radius: float = 30.0
    track_window_ms: float = 30.0

    def __post_init__(self) -> None:
        _finite(
            "timestamp_scale_us_to_ms", self.timestamp_scale_us_to_ms,
            positive=True,
        )
        for name in (
            "efast_inner_min_arc", "efast_inner_max_arc",
            "efast_outer_min_arc", "efast_outer_max_arc",
            "hdbscan_min_cluster_size", "hdbscan_leaf_size",
            "hdbscan_core_dist_n_jobs", "association_descriptor_events",
        ):
            _positive_integer(name, getattr(self, name))
        if self.hdbscan_min_samples is not None:
            _positive_integer("hdbscan_min_samples", self.hdbscan_min_samples)
        if self.efast_inner_min_arc > self.efast_inner_max_arc:
            raise ValueError("inner eFAST minimum arc exceeds its maximum")
        if self.efast_outer_min_arc > self.efast_outer_max_arc:
            raise ValueError("outer eFAST minimum arc exceeds its maximum")
        for name in ("density_radius", "hdbscan_alpha",
                     "association_radius", "track_window_ms"):
            _finite(name, getattr(self, name), positive=True)
        _finite(
            "hdbscan_cluster_selection_epsilon",
            self.hdbscan_cluster_selection_epsilon,
            non_negative=True,
        )
        if self.density_neighbors not in {"raw_events", "corner_events"}:
            raise ValueError("density_neighbors is invalid")
        if self.density_distance_domain not in {"t_ms_x_y", "x_y"}:
            raise ValueError("density_distance_domain is invalid")
        if self.density_threshold_scope not in {
            "per_polarity_whole_sequence", "global_whole_sequence",
        }:
            raise ValueError("density_threshold_scope is invalid")
        if self.hdbscan_metric != "euclidean":
            raise ValueError("Gate A supports only Euclidean HDBSCAN")
        if self.hdbscan_cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("invalid HDBSCAN cluster selection method")
        if self.hdbscan_algorithm not in {
            "best", "generic", "prims_kdtree", "prims_balltree",
            "boruvka_kdtree", "boruvka_balltree",
        }:
            raise ValueError("invalid HDBSCAN algorithm")
        for name in (
            "density_include_query", "hdbscan_approx_min_span_tree",
            "hdbscan_gen_min_span_tree", "hdbscan_allow_single_cluster",
            "hdbscan_prediction_data", "hdbscan_match_reference_implementation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


REFERENCE_FRONTEND = FrontendConfig()


def reference_mismatches(config: FrontendConfig) -> list[str]:
    """Return every departure from the frozen Gate-0 reference profile."""

    expected = REFERENCE_FRONTEND.as_dict()
    actual = config.as_dict()
    return [
        f"{name}: expected {expected[name]!r}, got {actual[name]!r}"
        for name in expected
        if actual[name] != expected[name]
    ]


def assert_reference_config(config: FrontendConfig) -> None:
    """Refuse to label a sensitivity setting as the reference method."""

    mismatches = reference_mismatches(config)
    if mismatches:
        raise ValueError(
            "configuration is experimental, not the frozen reference: "
            + "; ".join(mismatches)
        )


def config_from_dict(value: dict[str, Any]) -> FrontendConfig:
    """Build a front-end config while rejecting misspelled/unknown keys.

    Both ``{"frontend": {...}}`` and a bare front-end mapping are accepted;
    no other top-level section is accepted at Gate A.
    """

    if not isinstance(value, dict):
        raise ValueError("configuration must be a JSON object")
    if "frontend" in value:
        if set(value) != {"frontend"} or not isinstance(value["frontend"], dict):
            raise ValueError("Gate A config may contain only a frontend object")
        value = value["frontend"]
    known = set(FrontendConfig.__dataclass_fields__)
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"unknown frontend keys: {sorted(unknown)}")
    return FrontendConfig(**value)


def _strip_jsonc_comments(source: str) -> str:
    """Remove ``//`` comments without treating text inside strings as comments."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(source) and source[index + 1] == "/":
            while index < len(source) and source[index] not in "\r\n":
                output.append(" ")
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def load_config(path: str | Path) -> FrontendConfig:
    path = Path(path)
    try:
        payload = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid eSfO config: {path}") from error
    return config_from_dict(payload)


__all__ = [
    "FrontendConfig", "REFERENCE_FRONTEND", "assert_reference_config",
    "config_from_dict", "load_config", "reference_mismatches",
]
