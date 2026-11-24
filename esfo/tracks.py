"""Offline cluster association and Eq. (4) track-observation extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .data import RelativeEventSequence


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_vector(name: str, value: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not array.size:
        return _readonly(np.empty(0, dtype=np.int64))
    try:
        converted = array.astype(np.int64, casting="safe", copy=False)
    except TypeError as error:
        raise ValueError(f"{name} must contain integers") from error
    return _readonly(converted)


def _point_rows(name: str, value: np.ndarray | list[list[float]]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must have numeric shape [N,3]")
    result = _readonly(array.astype(np.float64, copy=False))
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class EventCluster:
    """One original HDBSCAN cluster with a stable, input-derived ID."""

    cluster_id: int
    event_indices: np.ndarray
    source_indices: np.ndarray
    points_txy: np.ndarray

    def __post_init__(self) -> None:
        if (isinstance(self.cluster_id, bool)
                or not isinstance(self.cluster_id, (int, np.integer))
                or int(self.cluster_id) < 0):
            raise ValueError("cluster ID must be non-negative")
        event_indices = _integer_vector("event_indices", self.event_indices)
        source_indices = _integer_vector("source_indices", self.source_indices)
        points_txy = _point_rows("points_txy", self.points_txy)
        if not (event_indices.size == source_indices.size == points_txy.shape[0]):
            raise ValueError("cluster arrays have inconsistent lengths")
        if not event_indices.size:
            raise ValueError("an event cluster cannot be empty")
        if (np.any(event_indices < 0) or np.any(source_indices < 0)
                or np.unique(event_indices).size != event_indices.size
                or np.unique(source_indices).size != source_indices.size):
            raise ValueError("cluster indices must be unique and non-negative")
        keys = np.lexsort((source_indices, points_txy[:, 0]))
        if not np.array_equal(keys, np.arange(keys.size)):
            raise ValueError("cluster events must be sorted by time and source order")
        object.__setattr__(self, "cluster_id", int(self.cluster_id))
        object.__setattr__(self, "event_indices", event_indices)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "points_txy", points_txy)

    def descriptor(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one-shot ``(head,tail)`` means of first/last ``N_sigma``."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise ValueError("descriptor count must be a positive integer")
        count = int(count)
        if count <= 0:
            raise ValueError("descriptor count must be positive")
        used = min(count, self.points_txy.shape[0])
        return (
            np.mean(self.points_txy[:used], axis=0, dtype=np.float64),
            np.mean(self.points_txy[-used:], axis=0, dtype=np.float64),
        )


@dataclass(frozen=True)
class AssociationEdge:
    source_cluster_id: int
    target_cluster_id: int
    distance: float


@dataclass(frozen=True)
class FeatureTrack:
    track_id: int
    cluster_ids: tuple[int, ...]
    event_indices: np.ndarray
    source_indices: np.ndarray
    points_txy: np.ndarray

    def __post_init__(self) -> None:
        if (isinstance(self.track_id, bool)
                or not isinstance(self.track_id, (int, np.integer))):
            raise ValueError("track ID must be an integer")
        cluster_ids = tuple(self.cluster_ids)
        if (not cluster_ids
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, np.integer))
                       or int(value) < 0 for value in cluster_ids)):
            raise ValueError("cluster IDs must be non-negative integers")
        cluster_ids = tuple(int(value) for value in cluster_ids)
        if cluster_ids != tuple(sorted(set(cluster_ids))):
            raise ValueError("cluster IDs must be unique and sorted")
        if int(self.track_id) != min(cluster_ids):
            raise ValueError("track ID must be the smallest component cluster ID")
        event_indices = _integer_vector("event_indices", self.event_indices)
        source_indices = _integer_vector("source_indices", self.source_indices)
        points_txy = _point_rows("points_txy", self.points_txy)
        if not (event_indices.size == source_indices.size == points_txy.shape[0]):
            raise ValueError("track arrays have inconsistent lengths")
        if (not event_indices.size or np.any(event_indices < 0)
                or np.any(source_indices < 0)
                or np.unique(event_indices).size != event_indices.size
                or np.unique(source_indices).size != source_indices.size):
            raise ValueError("track indices must be unique and non-negative")
        keys = np.lexsort((source_indices, points_txy[:, 0]))
        if not np.array_equal(keys, np.arange(keys.size)):
            raise ValueError("track events must be sorted by time and source order")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "cluster_ids", cluster_ids)
        object.__setattr__(self, "event_indices", event_indices)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "points_txy", points_txy)


@dataclass(frozen=True)
class AssociationResult:
    """Directed choices plus the one-shot weak-component merge result."""
    edges: tuple[AssociationEdge, ...]
    components: tuple[tuple[int, ...], ...]
    tracks: tuple[FeatureTrack, ...]


@dataclass(frozen=True)
class TrackObservation:
    """One Eq. (4) mean-pixel observation."""
    track_id: int
    window_index: int
    time_ms: float
    x: float
    y: float
    event_count: int


def build_clusters(
    events: RelativeEventSequence,
    corner_indices: np.ndarray | list[int],
    labels: np.ndarray | list[int],
) -> tuple[EventCluster, ...]:
    """Convert arbitrary HDBSCAN labels into stable original clusters."""
    corner_indices = _integer_vector("corner_indices", corner_indices)
    labels = _integer_vector("labels", labels)
    if corner_indices.size != labels.size:
        raise ValueError("corner indices and labels have inconsistent lengths")
    if labels.size and np.any(labels < -1):
        raise ValueError("only HDBSCAN noise label -1 may be negative")
    if corner_indices.size and (
        np.any(corner_indices < 0) or np.any(corner_indices >= len(events))
        or np.any(np.diff(corner_indices) <= 0)
    ):
        raise ValueError("corner indices must be unique, increasing, and in range")
    groups: list[tuple[int, np.ndarray]] = []
    for label in np.unique(labels):
        if label < 0:
            continue
        members = np.flatnonzero(labels == label)
        minimum_source = int(np.min(events.source_index[corner_indices[members]]))
        groups.append((minimum_source, members))
    groups.sort(key=lambda item: item[0])
    points = events.points_txy()
    clusters: list[EventCluster] = []
    for stable_id, (_, members) in enumerate(groups):
        event_indices = corner_indices[members]
        source_indices = events.source_index[event_indices]
        order = np.lexsort((source_indices, events.t_ms[event_indices]))
        clusters.append(EventCluster(
            stable_id,
            _readonly(event_indices[order]),
            _readonly(source_indices[order]),
            _readonly(points[event_indices[order]]),
        ))
    return tuple(clusters)


class _DisjointSet:
    def __init__(self, values: list[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            keep, remove = sorted((left_root, right_root))
            self.parent[remove] = keep


def _merge_clusters_unchecked(
    clusters: tuple[EventCluster, ...] | list[EventCluster],
    *,
    n_sigma: int = 5,
    phi: float = 30.0,
) -> AssociationResult:
    """Apply the frozen order-independent interpretation of paper Eq. (3)."""
    clusters = tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id))
    ids = [cluster.cluster_id for cluster in clusters]
    if len(ids) != len(set(ids)):
        raise ValueError("cluster IDs must be unique")
    if isinstance(n_sigma, bool) or not isinstance(n_sigma, (int, np.integer)):
        raise ValueError("n_sigma must be a positive integer")
    n_sigma = int(n_sigma)
    if n_sigma <= 0:
        raise ValueError("n_sigma must be positive")
    if isinstance(phi, bool) or not isinstance(phi, (int, float)):
        raise ValueError("phi must be positive and finite")
    phi = float(phi)
    if not math.isfinite(phi) or phi <= 0.0:
        raise ValueError("phi must be positive and finite")

    # Descriptors are frozen before any edge or component is constructed.
    descriptors = {
        cluster.cluster_id: cluster.descriptor(n_sigma) for cluster in clusters
    }
    edges: list[AssociationEdge] = []
    for source in clusters:
        _, tail = descriptors[source.cluster_id]
        candidates: list[tuple[float, float, int]] = []
        for target in clusters:
            if target.cluster_id == source.cluster_id:
                continue
            head, _ = descriptors[target.cluster_id]
            if head[0] < tail[0]:
                continue
            distance = float(np.linalg.norm(head - tail))
            if distance < phi:  # Strict gate in the frozen interpretation.
                candidates.append((distance, float(head[0]), target.cluster_id))
        if candidates:
            distance, _, target_id = min(candidates)
            edges.append(AssociationEdge(source.cluster_id, target_id, distance))

    disjoint = _DisjointSet(ids)
    for edge in edges:
        disjoint.union(edge.source_cluster_id, edge.target_cluster_id)
    component_map: dict[int, list[int]] = {}
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    for cluster_id in ids:
        component_map.setdefault(disjoint.find(cluster_id), []).append(cluster_id)
    components = tuple(
        tuple(sorted(members))
        for _, members in sorted(component_map.items(), key=lambda item: min(item[1]))
    )
    tracks: list[FeatureTrack] = []
    for members in components:
        component_clusters = [by_id[cluster_id] for cluster_id in members]
        event_indices = np.concatenate(
            [cluster.event_indices for cluster in component_clusters]
        )
        source_indices = np.concatenate(
            [cluster.source_indices for cluster in component_clusters]
        )
        points = np.concatenate(
            [cluster.points_txy for cluster in component_clusters], axis=0,
        )
        order = np.lexsort((source_indices, points[:, 0]))
        tracks.append(FeatureTrack(
            min(members), members, _readonly(event_indices[order]),
            _readonly(source_indices[order]), _readonly(points[order]),
        ))
    return AssociationResult(
        tuple(sorted(edges, key=lambda edge: edge.source_cluster_id)),
        components,
        tuple(tracks),
    )


def merge_clusters(
    clusters: tuple[EventCluster, ...] | list[EventCluster],
    *,
    n_sigma: int = 5,
    phi: float = 30.0,
) -> AssociationResult:
    """Apply Eq. (3) only with the frozen reference constants."""

    if n_sigma != 5 or phi != 30.0:
        raise ValueError(
            "non-reference association settings belong to esfo.experimental"
        )
    return _merge_clusters_unchecked(clusters, n_sigma=n_sigma, phi=phi)


def _track_observations_unchecked(
    tracks: tuple[FeatureTrack, ...] | list[FeatureTrack],
    *,
    sequence_duration_ms: float,
    delta_t_ms: float = 30.0,
) -> tuple[TrackObservation, ...]:
    """Extract Eq. (4) observations with its literal displayed indexing.

    ``k=1..floor(T/delta_t)-1`` and each window is
    ``(k*delta_t,(k+1)*delta_t]``. Empty track/window pairs emit no synthetic
    observation. The first ``delta_t`` interval is therefore omitted.
    """
    for name, value in (
        ("sequence_duration_ms", sequence_duration_ms),
        ("delta_t_ms", delta_t_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    sequence_duration_ms = float(sequence_duration_ms)
    delta_t_ms = float(delta_t_ms)
    if sequence_duration_ms < 0.0 or delta_t_ms <= 0.0:
        raise ValueError("duration must be non-negative and delta_t positive")
    final_k = math.floor(sequence_duration_ms / delta_t_ms) - 1
    if final_k < 1:
        return ()
    observations: list[TrackObservation] = []
    for k in range(1, final_k + 1):
        left, right = k * delta_t_ms, (k + 1) * delta_t_ms
        for track in sorted(tracks, key=lambda item: item.track_id):
            times = track.points_txy[:, 0]
            selected = (times > left) & (times <= right)
            count = int(np.count_nonzero(selected))
            if not count:
                continue
            observations.append(TrackObservation(
                track.track_id, k, left,
                float(np.mean(track.points_txy[selected, 1], dtype=np.float64)),
                float(np.mean(track.points_txy[selected, 2], dtype=np.float64)),
                count,
            ))
    return tuple(observations)


def track_observations(
    tracks: tuple[FeatureTrack, ...] | list[FeatureTrack],
    *,
    sequence_duration_ms: float,
    delta_t_ms: float = 30.0,
) -> tuple[TrackObservation, ...]:
    """Apply Eq. (4) only with the frozen reference window size."""

    if delta_t_ms != 30.0:
        raise ValueError(
            "non-reference track windows belong to esfo.experimental"
        )
    return _track_observations_unchecked(
        tracks,
        sequence_duration_ms=sequence_duration_ms,
        delta_t_ms=delta_t_ms,
    )


__all__ = [
    "AssociationEdge", "AssociationResult", "EventCluster", "FeatureTrack",
    "TrackObservation", "build_clusters", "merge_clusters",
    "track_observations",
]
