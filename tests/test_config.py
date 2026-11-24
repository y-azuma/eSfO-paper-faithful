from __future__ import annotations

from dataclasses import replace

import pytest

from esfo.config import (
    FrontendConfig,
    assert_reference_config,
    config_from_dict,
    reference_mismatches,
)
from esfo.data import EventSequence
from esfo.experimental import (
    run_frontend_raw_sensitivity,
    run_frontend_sensitivity,
)
from esfo.frontend import cluster_corners, density_filter, run_frontend_raw


def test_default_frontend_config_is_the_only_reference_configuration() -> None:
    reference = FrontendConfig()
    assert reference_mismatches(reference) == []
    assert_reference_config(reference)

    sensitivity = replace(reference, timestamp_scale_us_to_ms=1e-6)
    mismatches = reference_mismatches(sensitivity)
    assert any("timestamp_scale_us_to_ms" in item for item in mismatches)
    with pytest.raises(ValueError, match="reference|timestamp_scale_us_to_ms"):
        assert_reference_config(sensitivity)


def test_config_loader_accepts_only_bare_or_frontend_namespace() -> None:
    bare = config_from_dict({"density_radius": 8.0})
    nested = config_from_dict({"frontend": {"density_radius": 8.0}})
    assert bare == nested
    assert bare.density_radius == 8.0

    with pytest.raises(ValueError, match="frontend|unknown|experimental"):
        config_from_dict(
            {
                "frontend": {},
                "experimental": {"harmonic_correction": True},
            }
        )


def test_unknown_or_misspelled_options_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown|unexpected|field"):
        config_from_dict({"density_raduis": 7.0})


def test_frozen_hdbscan_and_window_defaults_match_gate_zero() -> None:
    config = FrontendConfig()
    assert config.timestamp_scale_us_to_ms == 1e-3
    assert (config.efast_inner_min_arc, config.efast_inner_max_arc) == (3, 6)
    assert (config.efast_outer_min_arc, config.efast_outer_max_arc) == (4, 8)
    assert config.density_radius == 7.0
    assert config.density_neighbors == "raw_events"
    assert config.density_distance_domain == "t_ms_x_y"
    assert config.density_threshold_scope == "per_polarity_whole_sequence"
    assert config.hdbscan_min_cluster_size == 10
    assert config.hdbscan_min_samples is None
    assert config.hdbscan_cluster_selection_epsilon == 5.0
    assert config.hdbscan_metric == "euclidean"
    assert config.hdbscan_cluster_selection_method == "eom"
    assert config.hdbscan_approx_min_span_tree is True
    assert config.hdbscan_allow_single_cluster is False
    assert config.association_descriptor_events == 5
    assert config.association_radius == 30.0
    assert config.track_window_ms == 30.0


def test_reference_runner_rejects_sensitivity_configuration() -> None:
    events = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=8, height=8,
    )
    sensitivity = replace(FrontendConfig(), timestamp_scale_us_to_ms=1e-6)
    with pytest.raises(ValueError, match="experimental|reference"):
        run_frontend_raw(events, sensitivity)


def test_reference_component_apis_reject_sensitivity_configuration() -> None:
    events = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=8, height=8,
    ).to_relative_ms()
    density_sensitivity = replace(
        FrontendConfig(), density_neighbors="corner_events",
    )
    clustering_sensitivity = replace(
        FrontendConfig(), hdbscan_cluster_selection_epsilon=4.0,
    )
    with pytest.raises(ValueError, match="experimental|reference"):
        density_filter(events, [], density_sensitivity)
    with pytest.raises(ValueError, match="experimental|reference"):
        cluster_corners(events, [], clustering_sensitivity)


def test_experimental_runner_is_labelled_and_rejects_reference_profile() -> None:
    events = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=8, height=8,
    )
    sensitivity = replace(FrontendConfig(), timestamp_scale_us_to_ms=1e-6)
    output = run_frontend_raw_sensitivity(events, sensitivity)
    assert output.evidence_class == "Experimental"
    assert any("timestamp_scale_us_to_ms" in item
               for item in output.reference_mismatches)
    with pytest.raises(ValueError, match="non-reference"):
        run_frontend_raw_sensitivity(events, FrontendConfig())


def test_relative_experimental_runner_cannot_mislabel_timestamp_scale() -> None:
    raw = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=8, height=8,
    )
    scale_sensitivity = replace(
        FrontendConfig(), timestamp_scale_us_to_ms=1e-6,
    )
    with pytest.raises(ValueError, match="raw EventSequence"):
        run_frontend_sensitivity(raw.to_relative_ms(), scale_sensitivity)

    density_sensitivity = replace(
        FrontendConfig(), density_neighbors="corner_events",
    )
    output = run_frontend_sensitivity(
        raw.to_relative_ms(), density_sensitivity,
    )
    assert output.evidence_class == "Experimental"


def test_boolean_options_do_not_accept_strings() -> None:
    with pytest.raises(ValueError, match="boolean"):
        config_from_dict({"density_include_query": "false"})
