"""Auditable COLMAP 3.9.1 input preparation for Gate B.

This module does not implement a substitute SfM algorithm.  It prepares the
known-camera database and raw all-pairs track matches consumed by the pinned
COLMAP CLI, records the exact commands, and validates the resulting database.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .tracks import TrackObservation
from .geometry import RegisteredTrajectory


COLMAP_VERSION = "3.9.1"
COLMAP_TAG_COMMIT = "e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68"
OPENCV_VERSION = "4.9.0"
OPENCV_DISTRIBUTION = "opencv-python-headless"
OPENCV_DISTRIBUTION_VERSION = "4.9.0.80"
MAX_IMAGE_ID = 2_147_483_647
PINHOLE_MODEL_ID = 1
REFERENCE_TRACK_WINDOW_MS = 30.0


def _readonly(value: Any, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _deep_freeze(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


class ColmapPipelineError(ValueError):
    """Reference-path failure carrying a stable diagnostic code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class Calibration:
    """TOPSPIN ``calib.txt`` pinhole and five-coefficient distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2,
            self.p1, self.p2, self.k3,
        ))
        if not all(math.isfinite(value) for value in values):
            raise ColmapPipelineError(
                "invalid_calibration", "calibration values must be finite",
            )
        if values[0] <= 0.0 or values[1] <= 0.0:
            raise ColmapPipelineError(
                "invalid_calibration", "fx and fy must be positive",
            )
        for name, value in zip(self.__dataclass_fields__, values):
            object.__setattr__(self, name, value)

    @property
    def camera_matrix(self) -> np.ndarray:
        return _readonly([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], np.dtype(np.float64))

    @property
    def distortion(self) -> np.ndarray:
        return _readonly(
            [self.k1, self.k2, self.p1, self.p2, self.k3],
            np.dtype(np.float64),
        )

    @property
    def pinhole_parameters(self) -> np.ndarray:
        return _readonly(
            [self.fx, self.fy, self.cx, self.cy], np.dtype(np.float64),
        )


def parse_calibration_text(source: str) -> Calibration:
    """Parse exactly ``fx fy cx cy k1 k2 p1 p2 k3`` and nothing else."""

    if not isinstance(source, str):
        raise ColmapPipelineError(
            "invalid_calibration", "calibration source must be text",
        )
    tokens = source.split()
    if len(tokens) != 9:
        raise ColmapPipelineError(
            "invalid_calibration", "calib.txt must contain exactly nine values",
        )
    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ColmapPipelineError(
            "invalid_calibration", "calib.txt contains a non-numeric token",
        ) from error
    return Calibration(*values)


def load_calibration(path: str | Path) -> Calibration:
    try:
        source = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ColmapPipelineError(
            "invalid_calibration", f"cannot read calibration: {path}",
        ) from error
    return parse_calibration_text(source)


def _point_rows(points_xy: Any, *, name: str = "points_xy") -> np.ndarray:
    points = np.asarray(points_xy)
    if points.ndim != 2 or points.shape[1:] != (2,) or points.dtype.kind not in "iuf":
        raise ColmapPipelineError(
            "invalid_points", f"{name} must have finite numeric shape [N,2]",
        )
    result = np.array(points, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ColmapPipelineError(
            "invalid_points", f"{name} contains non-finite values",
        )
    return result


def undistort_points(
    points_xy: Any,
    calibration: Calibration,
    *,
    cv2_module: Any | None = None,
) -> np.ndarray:
    """Apply OpenCV 4.9 ``undistortPoints(..., R=None, P=K)``.

    The original camera matrix is deliberately reused as ``P``.  This helper
    performs no crop, resize, clipping, or half-pixel correction.
    """

    points = _point_rows(points_xy)
    if cv2_module is None:
        try:
            import cv2 as cv2_module  # type: ignore[no-redef]
        except ImportError as error:
            raise ColmapPipelineError(
                "opencv_unavailable",
                "reference undistortion requires opencv-python-headless==4.9.0.80",
            ) from error
    module_version = str(getattr(cv2_module, "__version__", ""))
    if module_version != OPENCV_VERSION:
        raise ColmapPipelineError(
            "opencv_version_mismatch",
            f"expected OpenCV {OPENCV_VERSION}, got {module_version or 'unknown'}",
        )
    if not points.size:
        return _readonly(np.empty((0, 2)), np.dtype(np.float64))
    matrix = calibration.camera_matrix
    distortion = calibration.distortion
    try:
        output = cv2_module.undistortPoints(
            points.reshape((-1, 1, 2)), matrix, distortion,
            R=None, P=matrix,
        )
    except Exception as error:
        raise ColmapPipelineError(
            "undistortion_failed", "OpenCV undistortPoints failed",
        ) from error
    undistorted = np.asarray(output, dtype=np.float64).reshape((-1, 2))
    if undistorted.shape != points.shape or not np.all(np.isfinite(undistorted)):
        raise ColmapPipelineError(
            "undistortion_failed", "OpenCV returned invalid point coordinates",
        )
    return _readonly(undistorted, np.dtype(np.float64))


@dataclass(frozen=True)
class IndexedView:
    """One logical Eq. (4) window mapped to one COLMAP image."""

    image_id: int
    window_index: int
    time_ms: float
    image_name: str
    track_ids: np.ndarray
    keypoints_xy: np.ndarray

    def __post_init__(self) -> None:
        if (isinstance(self.image_id, bool) or not isinstance(self.image_id, int)
                or not 0 < self.image_id < MAX_IMAGE_ID):
            raise ColmapPipelineError("invalid_view", "invalid COLMAP image ID")
        if (isinstance(self.window_index, bool)
                or not isinstance(self.window_index, int)
                or self.window_index <= 0):
            raise ColmapPipelineError("invalid_view", "invalid logical window index")
        if not math.isfinite(float(self.time_ms)):
            raise ColmapPipelineError("invalid_view", "view time must be finite")
        if not isinstance(self.image_name, str) or not self.image_name:
            raise ColmapPipelineError("invalid_view", "image name must be non-empty")
        track_input = np.asarray(self.track_ids)
        if track_input.ndim != 1 or track_input.dtype.kind not in "iu":
            raise ColmapPipelineError(
                "invalid_view", "track IDs must be a one-dimensional integer array",
            )
        track_ids = _readonly(track_input, np.dtype(np.int64))
        keypoints = _readonly(_point_rows(self.keypoints_xy), np.dtype(np.float64))
        if track_ids.size != keypoints.shape[0]:
            raise ColmapPipelineError("invalid_view", "track/keypoint count mismatch")
        if (np.any(track_ids < 0) or np.unique(track_ids).size != track_ids.size
                or np.any(np.diff(track_ids) <= 0)):
            raise ColmapPipelineError(
                "invalid_view", "track IDs must be unique, non-negative, and sorted",
            )
        object.__setattr__(self, "time_ms", float(self.time_ms))
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "keypoints_xy", keypoints)


PointTransform = Callable[[np.ndarray], np.ndarray]


def build_indexed_views(
    observations: Iterable[TrackObservation],
    *,
    max_window_index: int,
    point_transform: PointTransform | None = None,
) -> tuple[IndexedView, ...]:
    """Create every logical window, including windows with zero keypoints."""

    if (isinstance(max_window_index, bool)
            or not isinstance(max_window_index, int)
            or max_window_index <= 0
            or max_window_index >= MAX_IMAGE_ID):
        raise ColmapPipelineError(
            "invalid_view", "max_window_index must be a positive COLMAP-safe integer",
        )
    grouped: dict[int, list[TrackObservation]] = {
        index: [] for index in range(1, max_window_index + 1)
    }
    seen: set[tuple[int, int]] = set()
    for observation in observations:
        if not isinstance(observation, TrackObservation):
            raise ColmapPipelineError(
                "invalid_observation", "all observations must be TrackObservation",
            )
        if (isinstance(observation.track_id, bool)
                or not isinstance(observation.track_id, int)
                or observation.track_id < 0):
            raise ColmapPipelineError("invalid_observation", "invalid track ID")
        if observation.window_index not in grouped:
            raise ColmapPipelineError(
                "invalid_observation", "observation is outside logical windows",
            )
        expected_time = observation.window_index * REFERENCE_TRACK_WINDOW_MS
        if float(observation.time_ms) != expected_time:
            raise ColmapPipelineError(
                "invalid_observation", "observation time violates Eq. (4) indexing",
            )
        if not all(math.isfinite(float(value)) for value in (
            observation.x, observation.y,
        )):
            raise ColmapPipelineError(
                "invalid_observation", "observation contains non-finite pixels",
            )
        key = (observation.window_index, observation.track_id)
        if key in seen:
            raise ColmapPipelineError(
                "duplicate_observation", "track is repeated within one window",
            )
        seen.add(key)
        grouped[observation.window_index].append(observation)
    views: list[IndexedView] = []
    for image_id, window_index in enumerate(range(1, max_window_index + 1), 1):
        window = sorted(grouped[window_index], key=lambda item: item.track_id)
        track_ids = np.asarray([item.track_id for item in window], dtype=np.int64)
        points = np.asarray(
            [[item.x, item.y] for item in window], dtype=np.float64,
        ).reshape((-1, 2))
        if point_transform is not None:
            transformed = np.asarray(point_transform(points), dtype=np.float64)
            if transformed.shape != points.shape or not np.all(np.isfinite(transformed)):
                raise ColmapPipelineError(
                    "undistortion_failed", "point transform returned invalid coordinates",
                )
            points = transformed
        views.append(IndexedView(
            image_id=image_id,
            window_index=window_index,
            time_ms=window_index * REFERENCE_TRACK_WINDOW_MS,
            image_name=f"window_{window_index:06d}.png",
            track_ids=track_ids,
            keypoints_xy=points,
        ))
    return tuple(views)


def build_undistorted_views(
    observations: Iterable[TrackObservation],
    *,
    max_window_index: int,
    calibration: Calibration,
    cv2_module: Any | None = None,
) -> tuple[IndexedView, ...]:
    return build_indexed_views(
        observations,
        max_window_index=max_window_index,
        point_transform=lambda points: undistort_points(
            points, calibration, cv2_module=cv2_module,
        ),
    )


@dataclass(frozen=True)
class PairMatches:
    image_id1: int
    image_id2: int
    image_name1: str
    image_name2: str
    matches: np.ndarray

    def __post_init__(self) -> None:
        if not (0 < self.image_id1 < self.image_id2 < MAX_IMAGE_ID):
            raise ColmapPipelineError("invalid_matches", "image pair IDs are invalid")
        match_input = np.asarray(self.matches)
        if match_input.ndim != 2 or match_input.shape[1:] != (2,):
            raise ColmapPipelineError("invalid_matches", "matches must have shape [N,2]")
        if match_input.dtype.kind not in "iu":
            raise ColmapPipelineError("invalid_matches", "matches must be integer indices")
        matches = _readonly(match_input, np.dtype(np.uint32))
        if matches.size and (
            np.unique(matches[:, 0]).size != matches.shape[0]
            or np.unique(matches[:, 1]).size != matches.shape[0]
        ):
            raise ColmapPipelineError(
                "invalid_matches", "a feature may occur at most once in a pair",
            )
        object.__setattr__(self, "matches", matches)

    @property
    def pair_id(self) -> int:
        return image_ids_to_pair_id(self.image_id1, self.image_id2)


def iter_pair_matches(views: Sequence[IndexedView]) -> Iterator[PairMatches]:
    """Yield nonempty all-pairs shared-track matches in deterministic order."""

    ordered = tuple(sorted(views, key=lambda item: item.image_id))
    if len({view.image_id for view in ordered}) != len(ordered):
        raise ColmapPipelineError("invalid_view", "duplicate image ID")
    for left_index, left in enumerate(ordered):
        left_features = {
            int(track_id): feature_index
            for feature_index, track_id in enumerate(left.track_ids)
        }
        for right in ordered[left_index + 1:]:
            right_features = {
                int(track_id): feature_index
                for feature_index, track_id in enumerate(right.track_ids)
            }
            shared = sorted(set(left_features) & set(right_features))
            if not shared:
                continue
            matches = np.asarray([
                (left_features[track_id], right_features[track_id])
                for track_id in shared
            ], dtype=np.uint32)
            yield PairMatches(
                left.image_id, right.image_id,
                left.image_name, right.image_name, matches,
            )


def expected_raw_correspondence_count(views: Sequence[IndexedView]) -> int:
    counts: dict[int, int] = {}
    for view in views:
        for track_id in view.track_ids:
            counts[int(track_id)] = counts.get(int(track_id), 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def iter_matches_importer_lines(views: Sequence[IndexedView]) -> Iterator[str]:
    for pair in iter_pair_matches(views):
        yield f"{pair.image_name1} {pair.image_name2}\n"
        for first, second in pair.matches:
            yield f"{int(first)} {int(second)}\n"
        yield "\n"


def write_matches_importer_file(
    path: str | Path, views: Sequence[IndexedView],
) -> tuple[int, int]:
    """Stream COLMAP raw-match blocks without retaining all pairs in RAM."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_count = 0
    correspondence_count = 0
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            for pair in iter_pair_matches(views):
                pair_count += 1
                correspondence_count += pair.matches.shape[0]
                stream.write(f"{pair.image_name1} {pair.image_name2}\n")
                for first, second in pair.matches:
                    stream.write(f"{int(first)} {int(second)}\n")
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        expected = expected_raw_correspondence_count(views)
        if correspondence_count != expected:
            raise ColmapPipelineError(
                "match_count_mismatch",
                f"generated {correspondence_count} matches, expected {expected}",
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return pair_count, correspondence_count


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    if (isinstance(image_id1, bool) or isinstance(image_id2, bool)
            or not isinstance(image_id1, int) or not isinstance(image_id2, int)
            or image_id1 == image_id2
            or min(image_id1, image_id2) <= 0
            or max(image_id1, image_id2) >= MAX_IMAGE_ID):
        raise ColmapPipelineError("invalid_matches", "invalid COLMAP image pair")
    first, second = sorted((image_id1, image_id2))
    return first * MAX_IMAGE_ID + second


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id <= 0:
        raise ColmapPipelineError("invalid_matches", "invalid COLMAP pair ID")
    second = pair_id % MAX_IMAGE_ID
    first = (pair_id - second) // MAX_IMAGE_ID
    if not 0 < first < second < MAX_IMAGE_ID:
        raise ColmapPipelineError("invalid_matches", "invalid COLMAP pair ID")
    return first, second


_REQUIRED_DATABASE_COLUMNS: Mapping[str, frozenset[str]] = MappingProxyType({
    "cameras": frozenset({
        "camera_id", "model", "width", "height", "params",
        "prior_focal_length",
    }),
    "images": frozenset({
        "image_id", "name", "camera_id", "prior_qw", "prior_qx",
        "prior_qy", "prior_qz", "prior_tx", "prior_ty", "prior_tz",
    }),
    "keypoints": frozenset({"image_id", "rows", "cols", "data"}),
    "matches": frozenset({"pair_id", "rows", "cols", "data"}),
    "descriptors": frozenset({"image_id", "rows", "cols", "data"}),
    "two_view_geometries": frozenset({
        "pair_id", "rows", "cols", "data", "config", "F", "E", "H",
        "qvec", "tvec",
    }),
})


# COLMAP 3.9.1 ``TwoViewGeometry::ConfigurationType`` values that can be
# emitted by the command plan in :func:`build_colmap_commands`.  With
# ``compute_relative_pose=false`` and ``multiple_models=false``, PLANAR (4),
# PANORAMIC (5), and MULTIPLE (8) cannot be direct verifier outputs.
_DEGENERATE_TWO_VIEW_CONFIG = 1
_WATERMARK_TWO_VIEW_CONFIG = 7
_POSITIVE_TWO_VIEW_CONFIGS = frozenset((2, 3, 6, 7))
_TWO_VIEW_MIN_NUM_INLIERS = 15


def _assert_colmap_schema(connection: sqlite3.Connection) -> None:
    for table, expected in _REQUIRED_DATABASE_COLUMNS.items():
        actual = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not expected <= actual:
            raise ColmapPipelineError(
                "invalid_database",
                f"COLMAP database table {table} is missing columns",
            )


def populate_known_camera_database(
    database_path: str | Path,
    *,
    calibration: Calibration,
    width: int,
    height: int,
    views: Sequence[IndexedView],
) -> None:
    """Populate a fresh database produced by ``colmap database_creator``."""

    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0):
        raise ColmapPipelineError(
            "invalid_camera", "sensor dimensions must be positive integers",
        )
    ordered = tuple(sorted(views, key=lambda item: item.image_id))
    if len({view.image_id for view in ordered}) != len(ordered):
        raise ColmapPipelineError("invalid_view", "duplicate image IDs")
    if len({view.image_name for view in ordered}) != len(ordered):
        raise ColmapPipelineError("invalid_view", "duplicate image names")
    try:
        connection = sqlite3.connect(str(database_path))
    except sqlite3.Error as error:
        raise ColmapPipelineError(
            "invalid_database", "cannot open COLMAP database",
        ) from error
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        _assert_colmap_schema(connection)
        existing = sum(int(connection.execute(
            f"SELECT COUNT(*) FROM {table}",
        ).fetchone()[0]) for table in (
            "cameras", "images", "keypoints", "descriptors", "matches",
            "two_view_geometries",
        ))
        if existing:
            raise ColmapPipelineError(
                "nonempty_database", "every COLMAP input/output table must be empty",
            )
        with connection:
            connection.execute(
                "INSERT INTO cameras(camera_id,model,width,height,params,"
                "prior_focal_length) VALUES(?,?,?,?,?,?)",
                (1, PINHOLE_MODEL_ID, width, height,
                 calibration.pinhole_parameters.tobytes(order="C"), 1),
            )
            for view in ordered:
                connection.execute(
                    "INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)",
                    (view.image_id, view.image_name, 1),
                )
                keypoints = np.asarray(view.keypoints_xy, dtype=np.float32, order="C")
                connection.execute(
                    "INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)",
                    (view.image_id, keypoints.shape[0], 2,
                     keypoints.tobytes(order="C")),
                )
    except sqlite3.Error as error:
        raise ColmapPipelineError(
            "invalid_database", "failed to populate COLMAP database",
        ) from error
    finally:
        connection.close()


def validate_fixed_camera_database(
    database_path: str | Path,
    *,
    calibration: Calibration,
    width: int,
    height: int,
) -> None:
    with sqlite3.connect(str(database_path)) as connection:
        _assert_colmap_schema(connection)
        rows = connection.execute(
            "SELECT camera_id,model,width,height,params,prior_focal_length "
            "FROM cameras ORDER BY camera_id",
        ).fetchall()
        image_camera_ids = connection.execute(
            "SELECT DISTINCT camera_id FROM images ORDER BY camera_id",
        ).fetchall()
    if len(rows) != 1:
        raise ColmapPipelineError("fixed_camera_changed", "expected one camera")
    camera_id, model, stored_width, stored_height, blob, prior = rows[0]
    params = np.frombuffer(blob, dtype=np.float64)
    if (camera_id != 1 or model != PINHOLE_MODEL_ID
            or stored_width != width or stored_height != height or prior != 1
            or not np.array_equal(params, calibration.pinhole_parameters)):
        raise ColmapPipelineError(
            "fixed_camera_changed", "COLMAP camera differs from fixed input K",
        )
    if image_camera_ids and image_camera_ids != [(1,)]:
        raise ColmapPipelineError(
            "fixed_camera_changed", "an image no longer references fixed camera 1",
        )


def validate_imported_raw_matches(
    database_path: str | Path,
    views: Sequence[IndexedView],
) -> None:
    expected = {pair.pair_id: pair.matches for pair in iter_pair_matches(views)}
    with sqlite3.connect(str(database_path)) as connection:
        _assert_colmap_schema(connection)
        rows = connection.execute(
            "SELECT pair_id,rows,cols,data FROM matches ORDER BY pair_id",
        ).fetchall()
    actual_ids = {int(row[0]) for row in rows}
    if actual_ids != set(expected):
        raise ColmapPipelineError(
            "match_count_mismatch", "database pair IDs differ from generated pairs",
        )
    total = 0
    for pair_id, row_count, columns, blob in rows:
        if columns != 2:
            raise ColmapPipelineError(
                "match_count_mismatch", "raw match table must have two columns",
            )
        try:
            actual = np.frombuffer(blob, dtype=np.uint32).reshape(
                (row_count, columns),
            )
        except (TypeError, ValueError) as error:
            raise ColmapPipelineError(
                "match_count_mismatch", "raw match blob has an invalid size",
            ) from error
        if not np.array_equal(actual, expected[int(pair_id)]):
            raise ColmapPipelineError(
                "match_count_mismatch", "raw matches differ from generated matches",
            )
        total += int(row_count)
    if total != expected_raw_correspondence_count(views):
        raise ColmapPipelineError(
            "match_count_mismatch", "database raw correspondence total is wrong",
        )


def validate_geometric_verification(
    database_path: str | Path,
    views: Sequence[IndexedView],
) -> Mapping[int, int]:
    """Verify every generated pair, including zero-inlier geometry rows.

    Returned counts include WATERMARK geometry because it was genuinely
    verified, but at least one positive non-WATERMARK pair is required: the
    pinned COLMAP 3.9.1 mapper excludes WATERMARK pairs from its correspondence
    graph.
    """

    expected = {pair.pair_id: pair.matches for pair in iter_pair_matches(views)}
    if not expected:
        raise ColmapPipelineError(
            "no_raw_pairs", "no shared-track pair was generated for verification",
        )
    with sqlite3.connect(str(database_path)) as connection:
        _assert_colmap_schema(connection)
        rows = connection.execute(
            "SELECT pair_id,rows,cols,data,config FROM two_view_geometries "
            "ORDER BY pair_id",
        ).fetchall()
    actual_ids = {int(row[0]) for row in rows}
    if actual_ids != set(expected):
        raise ColmapPipelineError(
            "verification_pair_mismatch",
            "verified geometry rows do not match every generated raw pair",
        )
    counts: dict[int, int] = {}
    has_mapper_usable_pair = False
    for pair_id, row_count, columns, blob, config in rows:
        if (not isinstance(row_count, int) or row_count < 0 or columns != 2):
            raise ColmapPipelineError(
                "invalid_verified_matches", "verified match dimensions are invalid",
            )
        expected_configs = (
            {_DEGENERATE_TWO_VIEW_CONFIG}
            if row_count == 0 else _POSITIVE_TWO_VIEW_CONFIGS
        )
        if (isinstance(config, bool) or not isinstance(config, int)
                or config not in expected_configs):
            raise ColmapPipelineError(
                "invalid_verified_matches",
                "verified match count and COLMAP configuration disagree",
            )
        if 0 < row_count < _TWO_VIEW_MIN_NUM_INLIERS:
            raise ColmapPipelineError(
                "invalid_verified_matches",
                "non-degenerate geometry has fewer than 15 inliers",
            )
        if row_count == 0:
            if blob not in (None, b""):
                raise ColmapPipelineError(
                    "invalid_verified_matches", "zero-row geometry has match data",
                )
            counts[int(pair_id)] = 0
            continue
        try:
            inliers = np.frombuffer(blob, dtype=np.uint32).reshape(
                (row_count, columns),
            )
        except (TypeError, ValueError) as error:
            raise ColmapPipelineError(
                "invalid_verified_matches", "verified match blob has invalid size",
            ) from error
        inlier_rows = [tuple(int(value) for value in match) for match in inliers]
        if len(set(inlier_rows)) != len(inlier_rows):
            raise ColmapPipelineError(
                "invalid_verified_matches",
                "geometric-verification inlier rows must be unique",
            )
        raw = {tuple(int(value) for value in match)
               for match in expected[int(pair_id)]}
        if any(match not in raw for match in inlier_rows):
            raise ColmapPipelineError(
                "invalid_verified_matches",
                "a geometric-verification inlier is absent from raw matches",
            )
        counts[int(pair_id)] = int(row_count)
        if config != _WATERMARK_TWO_VIEW_CONFIG:
            has_mapper_usable_pair = True
    if not has_mapper_usable_pair:
        raise ColmapPipelineError(
            "no_verified_pairs",
            "geometric verification produced no mapper-usable inlier pair",
        )
    return MappingProxyType(counts)


def verified_pair_counts(
    database_path: str | Path,
    views: Sequence[IndexedView],
) -> Mapping[int, int]:
    """Compatibility name for the strict verification postcondition."""

    return validate_geometric_verification(database_path, views)


@dataclass(frozen=True)
class ColmapCommandPlan:
    database_creator: tuple[str, ...]
    matches_importer: tuple[str, ...]
    mapper: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("database_creator", "matches_importer", "mapper"):
            command = tuple(getattr(self, name))
            if not command or any(not isinstance(item, str) or not item for item in command):
                raise ValueError(f"{name} must be a nonempty argv tuple")
            object.__setattr__(self, name, command)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def build_colmap_commands(
    *,
    database_path: str | Path,
    match_list_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    executable: str = "colmap",
) -> ColmapCommandPlan:
    if not isinstance(executable, str) or not executable:
        raise ValueError("COLMAP executable must be nonempty")
    database = str(Path(database_path))
    matches = str(Path(match_list_path))
    images = str(Path(image_path))
    output = str(Path(output_path))
    return ColmapCommandPlan(
        (executable, "database_creator", "--database_path", database),
        (
            executable, "matches_importer",
            "--database_path", database,
            "--match_list_path", matches,
            "--match_type", "raw",
            "--SiftMatching.use_gpu", "0",
            "--random_seed", "0",
        ),
        (
            executable, "mapper",
            "--database_path", database,
            "--image_path", images,
            "--output_path", output,
            "--random_seed", "0",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.extract_colors", "0",
        ),
        {
            "colmap_version": COLMAP_VERSION,
            "colmap_tag_commit": COLMAP_TAG_COMMIT,
            "verification": "matches_importer_match_type_raw",
            "verification_bypassed": False,
            "camera_model": "PINHOLE",
            "camera_intrinsics_fixed": True,
            "match_pairs": "all_windows_sharing_track",
            "descriptor_table_required": False,
            "mapper_extract_colors": False,
            "two_view_geometry_defaults": {
                "min_num_inliers": _TWO_VIEW_MIN_NUM_INLIERS,
                "min_E_F_inlier_ratio": 0.95,
                "max_H_inlier_ratio": 0.8,
                "watermark_min_inlier_ratio": 0.7,
                "watermark_border_size": 0.1,
                "detect_watermark": True,
                "multiple_ignore_watermark": True,
                "force_H_use": False,
                "compute_relative_pose": False,
                "multiple_models": False,
                "ransac_options": {
                    "max_error": 4.0,
                    "min_inlier_ratio": 0.25,
                    "confidence": 0.999,
                    "dyn_num_trials_multiplier": 3.0,
                    "min_num_trials": 100,
                    "max_num_trials": 10_000,
                },
            },
        },
    )


def assert_colmap_version(version_output: str) -> None:
    if not isinstance(version_output, str):
        raise ColmapPipelineError(
            "colmap_version_mismatch", "version output must be text",
        )
    # The pinned release must be a complete whitespace-delimited version
    # token.  A word boundary alone would incorrectly accept, for example,
    # ``3.9.1-dev`` and ``3.9.1.99``.
    if re.search(r"\bCOLMAP[ \t]+3\.9\.1(?=$|\s)", version_output) is None:
        raise ColmapPipelineError(
            "colmap_version_mismatch",
            f"expected COLMAP {COLMAP_VERSION}",
        )


@dataclass(frozen=True)
class ModelSummary:
    directory_id: int
    registered_image_count: int
    point3d_count: int

    def __post_init__(self) -> None:
        for name in (
            "directory_id", "registered_image_count", "point3d_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def select_largest_model(models: Iterable[ModelSummary]) -> ModelSummary:
    candidates = tuple(models)
    if not candidates:
        raise ColmapPipelineError("no_model", "COLMAP produced no sparse model")
    if len({item.directory_id for item in candidates}) != len(candidates):
        raise ColmapPipelineError("invalid_model", "duplicate model directory ID")
    return max(candidates, key=lambda item: (
        item.registered_image_count, item.point3d_count, -item.directory_id,
    ))


@dataclass(frozen=True)
class TrackPointMapping:
    unique_pairs: tuple[tuple[int, int], ...]
    split_track_ids: tuple[int, ...]
    merged_point_ids: tuple[int, ...]
    merged_track_ids: tuple[int, ...]
    uninitialized_track_ids: tuple[int, ...]
    excluded_track_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.unique_pairs)
        if any(
            not isinstance(pair, tuple) or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in pair)
            for pair in pairs
        ):
            raise ValueError("unique_pairs must contain non-negative integer pairs")
        if (pairs != tuple(sorted(set(pairs)))
                or len({pair[0] for pair in pairs}) != len(pairs)
                or len({pair[1] for pair in pairs}) != len(pairs)):
            raise ValueError("unique_pairs must be a sorted one-to-one mapping")
        categories: dict[str, tuple[int, ...]] = {}
        for name in (
            "split_track_ids", "merged_point_ids", "merged_track_ids",
            "uninitialized_track_ids", "excluded_track_ids",
        ):
            values = tuple(getattr(self, name))
            if (any(isinstance(value, bool) or not isinstance(value, int)
                    or value < 0 for value in values)
                    or values != tuple(sorted(set(values)))):
                raise ValueError(f"{name} must be sorted unique non-negative integers")
            categories[name] = values
            object.__setattr__(self, name, values)
        excluded_expected = (
            set(categories["split_track_ids"])
            | set(categories["merged_track_ids"])
            | set(categories["uninitialized_track_ids"])
        )
        if set(categories["excluded_track_ids"]) != excluded_expected:
            raise ValueError("excluded tracks must equal the union of all categories")
        mapped_track_ids = {pair[0] for pair in pairs}
        mapped_point_ids = {pair[1] for pair in pairs}
        if mapped_track_ids & excluded_expected:
            raise ValueError("a uniquely mapped track cannot also be excluded")
        if mapped_point_ids & set(categories["merged_point_ids"]):
            raise ValueError("a uniquely mapped point cannot also be merged")
        object.__setattr__(self, "unique_pairs", pairs)


def validate_track_point_mapping(
    views: Sequence[IndexedView],
    point_observations: Mapping[int, Iterable[tuple[int, int]]],
) -> TrackPointMapping:
    """Keep only one-original-track to one-COLMAP-point correspondences."""

    feature_to_track: dict[tuple[int, int], int] = {}
    all_tracks: set[int] = set()
    if len({view.image_id for view in views}) != len(views):
        raise ColmapPipelineError("invalid_model", "duplicate view ID")
    for view in views:
        for feature_index, track_id in enumerate(view.track_ids):
            feature_to_track[(view.image_id, feature_index)] = int(track_id)
            all_tracks.add(int(track_id))
    point_to_tracks: dict[int, set[int]] = {}
    track_to_points: dict[int, set[int]] = {}
    seen_features: set[tuple[int, int]] = set()
    for point_id in sorted(point_observations):
        if isinstance(point_id, bool) or not isinstance(point_id, int) or point_id < 0:
            raise ColmapPipelineError("invalid_model", "invalid point3D ID")
        observations = tuple(point_observations[point_id])
        if not observations:
            raise ColmapPipelineError(
                "invalid_model", "a reconstructed point has an empty track",
            )
        tracks: set[int] = set()
        for observation in observations:
            if (not isinstance(observation, tuple) or len(observation) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in observation)):
                raise ColmapPipelineError(
                    "invalid_model", "point observation must be (image,feature)",
                )
            if observation in seen_features:
                raise ColmapPipelineError(
                    "invalid_model", "one feature belongs to multiple 3D points",
                )
            seen_features.add(observation)
            if observation not in feature_to_track:
                raise ColmapPipelineError(
                    "invalid_model", "3D point references unknown feature",
                )
            tracks.add(feature_to_track[observation])
        point_to_tracks[point_id] = tracks
        for track_id in tracks:
            track_to_points.setdefault(track_id, set()).add(point_id)
    merged_points = tuple(sorted(
        point_id for point_id, tracks in point_to_tracks.items() if len(tracks) != 1
    ))
    merged_tracks = {
        track_id for point_id in merged_points for track_id in point_to_tracks[point_id]
    }
    split_tracks = tuple(sorted(
        track_id for track_id, points in track_to_points.items() if len(points) != 1
    ))
    excluded = set(split_tracks) | merged_tracks
    pairs: list[tuple[int, int]] = []
    for track_id, points in sorted(track_to_points.items()):
        if track_id in excluded or len(points) != 1:
            continue
        point_id = next(iter(points))
        if len(point_to_tracks[point_id]) == 1:
            pairs.append((track_id, point_id))
    initialized = set(track_to_points)
    uninitialized = tuple(sorted(all_tracks - initialized))
    excluded.update(uninitialized)
    return TrackPointMapping(
        tuple(pairs), split_tracks, merged_points,
        tuple(sorted(merged_tracks)), uninitialized,
        tuple(sorted(excluded)),
    )


def _binary_model_count(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            header = stream.read(8)
    except OSError as error:
        raise ColmapPipelineError(
            "invalid_model", f"cannot read model file: {path}",
        ) from error
    if len(header) != 8:
        raise ColmapPipelineError(
            "invalid_model", f"model file has a truncated count: {path}",
        )
    return int(struct.unpack("<Q", header)[0])


def discover_sparse_models(output_path: str | Path) -> tuple[ModelSummary, ...]:
    """Discover binary model directories without choosing by filesystem order."""

    root = Path(output_path)
    if not root.is_dir():
        raise ColmapPipelineError("no_model", "sparse output directory is missing")
    summaries: list[ModelSummary] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            directory_id = int(directory.name)
        except ValueError:
            continue
        if directory_id < 0 or str(directory_id) != directory.name:
            raise ColmapPipelineError(
                "invalid_model", "model directory ID must be canonical non-negative int",
            )
        for required in ("cameras.bin", "images.bin", "points3D.bin"):
            if not (directory / required).is_file():
                raise ColmapPipelineError(
                    "invalid_model", f"model {directory_id} lacks {required}",
                )
        summaries.append(ModelSummary(
            directory_id,
            _binary_model_count(directory / "images.bin"),
            _binary_model_count(directory / "points3D.bin"),
        ))
    return tuple(sorted(summaries, key=lambda item: item.directory_id))


def build_model_converter_command(
    *,
    input_path: str | Path,
    output_path: str | Path,
    executable: str = "colmap",
) -> tuple[str, ...]:
    if not isinstance(executable, str) or not executable:
        raise ValueError("COLMAP executable must be nonempty")
    return (
        executable, "model_converter",
        "--input_path", str(Path(input_path)),
        "--output_path", str(Path(output_path)),
        "--output_type", "TXT",
    )


def _data_lines(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ColmapPipelineError(
            "invalid_model", f"cannot read text model file: {path}",
        ) from error
    return [line.strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")]


def validate_text_model_camera(
    model_path: str | Path,
    *,
    calibration: Calibration,
    width: int,
    height: int,
) -> None:
    """Fail if a converted sparse model changed the fixed PINHOLE camera."""

    lines = _data_lines(Path(model_path) / "cameras.txt")
    if len(lines) != 1:
        raise ColmapPipelineError(
            "fixed_camera_changed", "sparse model must contain exactly one camera",
        )
    tokens = lines[0].split()
    if len(tokens) != 8:
        raise ColmapPipelineError(
            "fixed_camera_changed", "PINHOLE camera line must have eight fields",
        )
    try:
        camera_id = int(tokens[0])
        model = tokens[1]
        stored_width, stored_height = int(tokens[2]), int(tokens[3])
        parameters = np.asarray([float(value) for value in tokens[4:]],
                                dtype=np.float64)
    except ValueError as error:
        raise ColmapPipelineError(
            "fixed_camera_changed", "sparse model camera fields are invalid",
        ) from error
    if (camera_id != 1 or model != "PINHOLE" or stored_width != width
            or stored_height != height
            or not np.array_equal(parameters, calibration.pinhole_parameters)):
        raise ColmapPipelineError(
            "fixed_camera_changed", "sparse model camera differs from fixed input K",
        )


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    if qvec.shape != (4,) or not np.all(np.isfinite(qvec)):
        raise ColmapPipelineError(
            "invalid_model", "COLMAP quaternion must contain four finite values",
        )
    norm = float(np.linalg.norm(qvec))
    if abs(norm - 1.0) > 256.0 * np.finfo(np.float64).eps:
        raise ColmapPipelineError(
            "invalid_model", "COLMAP quaternion is not unit length",
        )
    w, x, y, z = qvec
    return np.asarray([
        [1.0 - 2.0 * y * y - 2.0 * z * z,
         2.0 * x * y - 2.0 * w * z,
         2.0 * x * z + 2.0 * w * y],
        [2.0 * x * y + 2.0 * w * z,
         1.0 - 2.0 * x * x - 2.0 * z * z,
         2.0 * y * z - 2.0 * w * x],
        [2.0 * x * z - 2.0 * w * y,
         2.0 * y * z + 2.0 * w * x,
         1.0 - 2.0 * x * x - 2.0 * y * y],
    ], dtype=np.float64)


def _parse_images_text(
    path: Path,
    views: Sequence[IndexedView],
) -> tuple[RegisteredTrajectory, set[int], dict[tuple[int, int], int]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ColmapPipelineError(
            "invalid_model", f"cannot read text model file: {path}",
        ) from error
    by_name = {view.image_name: view for view in views}
    if len(by_name) != len(views):
        raise ColmapPipelineError("invalid_view", "duplicate logical image name")
    records: list[tuple[IndexedView, int, np.ndarray, np.ndarray]] = []
    registered_image_ids: set[int] = set()
    image_feature_points: dict[tuple[int, int], int] = {}
    index = 0
    while index < len(lines):
        pose_line = lines[index].strip()
        index += 1
        if not pose_line or pose_line.startswith("#"):
            continue
        if index >= len(lines):
            raise ColmapPipelineError(
                "invalid_model", "images.txt lacks a points2D line",
            )
        point_line = lines[index].strip()
        index += 1
        pose_tokens = pose_line.split()
        if len(pose_tokens) != 10:
            raise ColmapPipelineError(
                "invalid_model", "images.txt pose line must have ten fields",
            )
        try:
            image_id = int(pose_tokens[0])
            qvec = np.asarray([float(value) for value in pose_tokens[1:5]])
            translation = np.asarray([float(value) for value in pose_tokens[5:8]])
            camera_id = int(pose_tokens[8])
        except ValueError as error:
            raise ColmapPipelineError(
                "invalid_model", "images.txt pose fields are invalid",
            ) from error
        name = pose_tokens[9]
        if (image_id in registered_image_ids or name not in by_name
                or camera_id != 1 or not np.all(np.isfinite(translation))):
            raise ColmapPipelineError(
                "invalid_model", "registered image is duplicate or unknown",
            )
        registered_image_ids.add(image_id)
        view = by_name[name]
        if image_id != view.image_id:
            raise ColmapPipelineError(
                "invalid_model", "registered image ID/name mapping changed",
            )
        point_tokens = point_line.split()
        if len(point_tokens) % 3 != 0:
            raise ColmapPipelineError(
                "invalid_model", "images.txt points2D line is malformed",
            )
        feature_count = len(point_tokens) // 3
        if feature_count != view.track_ids.size:
            raise ColmapPipelineError(
                "invalid_model", "registered image feature count changed",
            )
        if feature_count:
            try:
                parsed_xy = np.asarray([
                    [float(point_tokens[offset]), float(point_tokens[offset + 1])]
                    for offset in range(0, len(point_tokens), 3)
                ])
                point_ids = [int(point_tokens[offset + 2])
                             for offset in range(0, len(point_tokens), 3)]
            except ValueError as error:
                raise ColmapPipelineError(
                    "invalid_model", "images.txt points2D fields are invalid",
                ) from error
            expected_xy = np.asarray(view.keypoints_xy, dtype=np.float32).astype(
                np.float64,
            )
            if not np.array_equal(parsed_xy, expected_xy):
                raise ColmapPipelineError(
                    "invalid_model", "registered image keypoints changed",
                )
            if any(point_id < -1 for point_id in point_ids):
                raise ColmapPipelineError(
                    "invalid_model", "images.txt has an invalid point3D ID",
                )
            for feature_index, point_id in enumerate(point_ids):
                image_feature_points[(image_id, feature_index)] = point_id
        records.append((view, image_id, _qvec_to_rotation(qvec), translation))
    if len(records) < 3:
        raise ColmapPipelineError(
            "insufficient_registered_views", "largest model registered fewer than 3 views",
        )
    records.sort(key=lambda item: (item[0].time_ms, item[1]))
    trajectory = RegisteredTrajectory(
        np.asarray([item[1] for item in records], dtype=np.int64),
        np.asarray([item[0].time_ms / 1000.0 for item in records]),
        np.stack([item[2] for item in records]),
        np.stack([item[3] for item in records]),
    )
    return trajectory, registered_image_ids, image_feature_points


def _parse_points3d_text(
    path: Path,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[tuple[int, int], ...]]]:
    points: dict[int, np.ndarray] = {}
    observations: dict[int, tuple[tuple[int, int], ...]] = {}
    for line in _data_lines(path):
        tokens = line.split()
        if len(tokens) < 8 or (len(tokens) - 8) % 2 != 0:
            raise ColmapPipelineError(
                "invalid_model", "points3D.txt line is malformed",
            )
        try:
            point_id = int(tokens[0])
            xyz = np.asarray([float(value) for value in tokens[1:4]])
            rgb = tuple(int(value) for value in tokens[4:7])
            error_value = float(tokens[7])
            track = tuple(
                (int(tokens[offset]), int(tokens[offset + 1]))
                for offset in range(8, len(tokens), 2)
            )
        except ValueError as error:
            raise ColmapPipelineError(
                "invalid_model", "points3D.txt fields are invalid",
            ) from error
        if (point_id < 0 or point_id in points or not np.all(np.isfinite(xyz))
                or any(value < 0 or value > 255 for value in rgb)
                or not math.isfinite(error_value) or error_value < 0.0
                or not track):
            raise ColmapPipelineError(
                "invalid_model", "point3D ID, coordinates, or track are invalid",
            )
        points[point_id] = xyz
        observations[point_id] = track
    return points, observations


def _validate_bidirectional_point_tracks(
    image_feature_points: Mapping[tuple[int, int], int],
    point_observations: Mapping[int, tuple[tuple[int, int], ...]],
) -> None:
    track_feature_points: dict[tuple[int, int], int] = {}
    for point_id, track in point_observations.items():
        for feature in track:
            if feature in track_feature_points:
                raise ColmapPipelineError(
                    "invalid_model", "one image feature occurs in multiple point tracks",
                )
            track_feature_points[feature] = int(point_id)
    image_positive = {
        feature: point_id for feature, point_id in image_feature_points.items()
        if point_id >= 0
    }
    if image_positive != track_feature_points:
        raise ColmapPipelineError(
            "invalid_model",
            "images.txt point IDs and points3D.txt tracks are inconsistent",
        )


@dataclass(frozen=True)
class ExtractedColmapModel:
    summary: ModelSummary
    trajectory: RegisteredTrajectory
    mapping: TrackPointMapping
    mapped_track_ids: np.ndarray
    mapped_point_ids: np.ndarray
    mapped_points_w: np.ndarray
    point_observations: Mapping[int, tuple[tuple[int, int], ...]]
    unregistered_window_indices: tuple[int, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        track_ids = _readonly(self.mapped_track_ids, np.dtype(np.int64))
        point_ids = _readonly(self.mapped_point_ids, np.dtype(np.int64))
        points = np.asarray(self.mapped_points_w)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("mapped_points_w must have shape [N,3]")
        points = _readonly(points, np.dtype(np.float64))
        if (track_ids.size != point_ids.size or track_ids.size != points.shape[0]
                or not np.all(np.isfinite(points))):
            raise ValueError("mapped point arrays are inconsistent")
        if not isinstance(self.point_observations, Mapping):
            raise ValueError("point_observations must be a mapping")
        point_observations: dict[int, tuple[tuple[int, int], ...]] = {}
        seen_features: set[tuple[int, int]] = set()
        for point_id_raw, track_input in sorted(self.point_observations.items()):
            if (isinstance(point_id_raw, bool)
                    or not isinstance(point_id_raw, (int, np.integer))
                    or int(point_id_raw) < 0):
                raise ValueError("point_observations has an invalid point ID")
            point_id = int(point_id_raw)
            track = tuple(track_input)
            if not track:
                raise ValueError("point_observations has an empty point track")
            normalized: list[tuple[int, int]] = []
            for feature in track:
                if (not isinstance(feature, tuple) or len(feature) != 2
                        or any(isinstance(value, bool)
                               or not isinstance(value, (int, np.integer))
                               or int(value) < 0 for value in feature)):
                    raise ValueError(
                        "point_observations features must be non-negative pairs"
                    )
                pair = (int(feature[0]), int(feature[1]))
                if pair in seen_features:
                    raise ValueError(
                        "one image feature belongs to multiple point tracks"
                    )
                seen_features.add(pair)
                normalized.append(pair)
            point_observations[point_id] = tuple(normalized)
        if len(point_observations) != self.summary.point3d_count:
            raise ValueError("point_observations count disagrees with model summary")
        expected_pairs = tuple(
            (int(track_id), int(point_id))
            for track_id, point_id in zip(track_ids, point_ids)
        )
        if (expected_pairs != self.mapping.unique_pairs
                or len(set(expected_pairs)) != len(expected_pairs)
                or expected_pairs != tuple(sorted(expected_pairs))):
            raise ValueError("mapped arrays disagree with the unique mapping")
        if any(int(point_id) not in point_observations for point_id in point_ids):
            raise ValueError("a mapped point lacks its selected-model point track")
        unregistered = tuple(self.unregistered_window_indices)
        if (any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                or int(value) <= 0 for value in unregistered)
                or tuple(sorted(set(int(value) for value in unregistered)))
                != tuple(int(value) for value in unregistered)):
            raise ValueError("unregistered windows must be sorted unique positive IDs")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "mapped_track_ids", track_ids)
        object.__setattr__(self, "mapped_point_ids", point_ids)
        object.__setattr__(self, "mapped_points_w", points)
        object.__setattr__(self, "point_observations", MappingProxyType(
            point_observations,
        ))
        object.__setattr__(self, "unregistered_window_indices",
                           tuple(int(value) for value in unregistered))
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))


def extract_text_model(
    model_path: str | Path,
    *,
    summary: ModelSummary,
    views: Sequence[IndexedView],
    calibration: Calibration,
    width: int,
    height: int,
) -> ExtractedColmapModel:
    """Ingest a 3.9.1 model converted to text and bind it to source tracks.

    ``model_path`` must be a real (non-symlink) directory whose final path
    component is the canonical decimal ``summary.directory_id``.  The
    converter workflow therefore preserves the binary model directory ID and
    cannot silently label a same-sized converted model with another summary.
    """

    model = Path(model_path)
    expected_directory_name = str(summary.directory_id)
    if (model.name != expected_directory_name or model.is_symlink()
            or not model.is_dir()):
        raise ColmapPipelineError(
            "invalid_model",
            "text model path is not bound to the summary directory ID",
        )
    validate_text_model_camera(
        model, calibration=calibration, width=width, height=height,
    )
    trajectory, registered_ids, image_feature_points = _parse_images_text(
        model / "images.txt", views,
    )
    if trajectory.view_ids.size != summary.registered_image_count:
        raise ColmapPipelineError(
            "invalid_model", "registered image count differs from model summary",
        )
    points, observations = _parse_points3d_text(model / "points3D.txt")
    if len(points) != summary.point3d_count:
        raise ColmapPipelineError(
            "invalid_model", "point3D count differs from model summary",
        )
    _validate_bidirectional_point_tracks(image_feature_points, observations)
    mapping = validate_track_point_mapping(views, observations)
    mapped_track_ids = np.asarray(
        [track_id for track_id, _ in mapping.unique_pairs], dtype=np.int64,
    )
    mapped_point_ids = np.asarray(
        [point_id for _, point_id in mapping.unique_pairs], dtype=np.int64,
    )
    mapped_points = np.asarray(
        [points[int(point_id)] for point_id in mapped_point_ids],
        dtype=np.float64,
    ).reshape((-1, 3))
    unregistered = tuple(
        view.window_index for view in views if view.image_id not in registered_ids
    )
    return ExtractedColmapModel(
        summary, trajectory, mapping, mapped_track_ids, mapped_point_ids,
        mapped_points, observations, unregistered,
        {
            "camera_checked": "fixed_PINHOLE_exact_roundtrip",
            "pose_convention": "world_to_camera_qvec_tvec",
            "model_directory_id": summary.directory_id,
            "track_point_policy": "bijective_only",
            "mapped_track_count": len(mapping.unique_pairs),
            "split_track_count": len(mapping.split_track_ids),
            "merged_point_count": len(mapping.merged_point_ids),
            "merged_track_count": len(mapping.merged_track_ids),
            "uninitialized_track_count": len(mapping.uninitialized_track_ids),
        },
    )


__all__ = [
    "COLMAP_TAG_COMMIT", "COLMAP_VERSION", "Calibration",
    "ColmapCommandPlan", "ColmapPipelineError", "IndexedView", "MAX_IMAGE_ID",
    "ModelSummary", "OPENCV_DISTRIBUTION", "OPENCV_DISTRIBUTION_VERSION",
    "OPENCV_VERSION", "PINHOLE_MODEL_ID", "PairMatches",
    "TrackPointMapping", "assert_colmap_version", "build_colmap_commands",
    "ExtractedColmapModel", "build_model_converter_command",
    "build_indexed_views", "build_undistorted_views",
    "discover_sparse_models", "extract_text_model",
    "expected_raw_correspondence_count", "image_ids_to_pair_id",
    "iter_matches_importer_lines", "iter_pair_matches", "load_calibration",
    "pair_id_to_image_ids", "parse_calibration_text",
    "populate_known_camera_database", "select_largest_model",
    "undistort_points", "validate_fixed_camera_database",
    "validate_geometric_verification", "validate_imported_raw_matches",
    "validate_text_model_camera", "validate_track_point_mapping",
    "verified_pair_counts", "write_matches_importer_file",
]
