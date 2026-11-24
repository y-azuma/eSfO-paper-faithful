from __future__ import annotations

import numpy as np

from esfo.efast import CIRCLE3, CIRCLE4, EFastDetector, has_newest_arc


def test_newest_arc_accepts_contiguous_and_wraparound_segments() -> None:
    ring = np.zeros(16, dtype=np.float64)
    ring[4:7] = 2.0
    assert has_newest_arc(ring, min_arc=3, max_arc=6)

    wraparound = np.zeros(16, dtype=np.float64)
    wraparound[[-1, 0, 1]] = 2.0
    assert has_newest_arc(wraparound, min_arc=3, max_arc=6)


def test_newest_arc_uses_strict_order_and_maximum_arc_length() -> None:
    tied = np.zeros(16, dtype=np.float64)
    tied[3:6] = 2.0
    tied[10] = 2.0
    assert not has_newest_arc(tied, min_arc=3, max_arc=6)

    too_long = np.zeros(16, dtype=np.float64)
    too_long[2:9] = 2.0
    assert not has_newest_arc(too_long, min_arc=3, max_arc=6)


def test_both_ring_arc_boundaries_are_discriminated() -> None:
    for ring_size, minimum, maximum in ((16, 3, 6), (20, 4, 8)):
        for newest_count, expected in (
            (minimum - 1, False), (minimum, True),
            (maximum, True), (maximum + 1, False),
        ):
            ring = np.zeros(ring_size, dtype=np.float64)
            ring[:newest_count] = 1.0
            assert has_newest_arc(ring, minimum, maximum) is expected


def test_detector_sae_initialization_and_polarity_are_explicit() -> None:
    detector = EFastDetector(width=15, height=15)
    # A first event cannot be a corner because both polarity SAEs start empty.
    assert not detector.detect(
        t_ms=np.asarray([0.0]),
        x=np.asarray([7]),
        y=np.asarray([7]),
        p=np.asarray([1]),
    )[0]

    center = np.asarray([7, 7])
    inner_xy = center + CIRCLE3
    outer_xy = center + CIRCLE4
    # Populate both rings' non-arc points first. Then populate a three-point
    # newest inner arc and a four-point newest outer arc, as required by the
    # conjunction in the cited eFAST criterion.
    xy = np.vstack(
        (
            inner_xy[3:],
            outer_xy[4:],
            inner_xy[:3],
            outer_xy[:4],
        )
    )
    t_ms = np.r_[np.full(29, 1.0), np.full(7, 2.0), 3.0]
    x = np.r_[xy[:, 0], center[0]]
    y = np.r_[xy[:, 1], center[1]]
    p = np.ones(37, dtype=np.int8)

    mask = EFastDetector(width=15, height=15).detect(t_ms, x, y, p)
    assert mask.shape == (37,)
    assert mask[-1]

    # The same ring on the opposite-polarity SAE must not trigger a positive
    # query at the center.
    opposite = np.r_[np.zeros(36, dtype=np.int8), 1]
    opposite_mask = EFastDetector(width=15, height=15).detect(
        t_ms, x, y, opposite
    )
    assert not opposite_mask[-1]


def _query_after_rings(*, inner_new: bool, outer_new: bool, offset: float = 0.0):
    center = np.asarray([7, 7])
    inner_xy = center + CIRCLE3
    outer_xy = center + CIRCLE4
    parts = [inner_xy, outer_xy]
    time_parts = [np.full(36, offset + 1.0)]
    if inner_new:
        parts.append(inner_xy[:3])
        time_parts.append(np.full(3, offset + 2.0))
    if outer_new:
        parts.append(outer_xy[:4])
        time_parts.append(np.full(4, offset + 2.0))
    xy = np.vstack(parts)
    times = np.concatenate((*time_parts, np.asarray([offset + 3.0])))
    return EFastDetector(width=15, height=15).detect(
        times, np.r_[xy[:, 0], center[0]], np.r_[xy[:, 1], center[1]],
        np.ones(times.size, dtype=np.int8),
    )[-1]


def test_detector_requires_inner_and_outer_arcs() -> None:
    assert _query_after_rings(inner_new=True, outer_new=True)
    assert not _query_after_rings(inner_new=True, outer_new=False)
    assert not _query_after_rings(inner_new=False, outer_new=True)


def test_detector_is_invariant_to_common_timestamp_shift() -> None:
    assert _query_after_rings(inner_new=True, outer_new=True, offset=0.0)
    assert _query_after_rings(inner_new=True, outer_new=True, offset=10_000.0)


def test_detector_processes_equal_timestamps_in_input_order() -> None:
    # Equal timestamps are legal.  Repeated calls must preserve input order
    # and produce the same mask rather than sorting spatially.
    t_ms = np.zeros(4, dtype=np.float64)
    x = np.asarray([3, 4, 5, 6])
    y = np.asarray([3, 3, 3, 3])
    p = np.asarray([1, 1, 1, 1], dtype=np.int8)

    first = EFastDetector(width=10, height=10).detect(t_ms, x, y, p)
    second = EFastDetector(width=10, height=10).detect(t_ms, x, y, p)
    np.testing.assert_array_equal(first, second)


def test_detector_rejects_queries_inside_four_pixel_border() -> None:
    detector = EFastDetector(width=12, height=12)
    mask = detector.detect(
        np.asarray([1.0]), np.asarray([3]), np.asarray([6]), np.asarray([1]),
    )
    assert not mask[0]


def test_detector_rejects_fractional_coordinates_instead_of_truncating() -> None:
    with np.testing.assert_raises_regex(ValueError, "integer"):
        EFastDetector(width=12, height=12).detect(
            np.asarray([1.0]), np.asarray([3.9]), np.asarray([6]),
            np.asarray([1]),
        )


def test_reference_detector_rejects_nonreference_arc_settings() -> None:
    with np.testing.assert_raises_regex(ValueError, "experimental"):
        EFastDetector(width=12, height=12, outer_max_arc=7)
