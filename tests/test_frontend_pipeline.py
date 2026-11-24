from __future__ import annotations

from esfo.data import EventSequence
from esfo.frontend import run_frontend_raw


def test_empty_synthetic_frontend_pipeline_is_well_defined() -> None:
    events = EventSequence.from_arrays(
        t_us=[], x=[], y=[], p=[], width=16, height=16,
    )

    result = run_frontend_raw(events)

    assert result.corner_mask.size == 0
    assert result.density.corner_indices.size == 0
    assert result.clustering.labels.size == 0
    assert result.association.tracks == ()
    assert result.observations == ()
