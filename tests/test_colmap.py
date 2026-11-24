from __future__ import annotations

from pathlib import Path
import sqlite3
import struct

import numpy as np
import pytest

from esfo.colmap import (
    COLMAP_TAG_COMMIT,
    OPENCV_DISTRIBUTION,
    OPENCV_DISTRIBUTION_VERSION,
    Calibration,
    ColmapPipelineError,
    ModelSummary,
    assert_colmap_version,
    build_colmap_commands,
    build_indexed_views,
    build_model_converter_command,
    discover_sparse_models,
    expected_raw_correspondence_count,
    extract_text_model,
    image_ids_to_pair_id,
    iter_pair_matches,
    pair_id_to_image_ids,
    parse_calibration_text,
    populate_known_camera_database,
    select_largest_model,
    undistort_points,
    validate_fixed_camera_database,
    validate_imported_raw_matches,
    validate_track_point_mapping,
    verified_pair_counts,
    write_matches_importer_file,
)
from esfo.tracks import TrackObservation


def _calibration() -> Calibration:
    return Calibration(400.0, 410.0, 320.0, 240.0, 0.1, -0.02, 0.003, -0.004, 0.005)


def _observation(track: int, window: int, x: float, y: float) -> TrackObservation:
    return TrackObservation(track, window, float(window * 30), x, y, 3)


def _views():
    observations = (
        _observation(20, 1, 20.0, 1.0),
        _observation(10, 1, 10.0, 1.0),
        _observation(30, 2, 30.0, 2.0),
        _observation(10, 2, 10.0, 2.0),
        _observation(30, 3, 30.0, 3.0),
        _observation(20, 3, 20.0, 3.0),
        _observation(10, 3, 10.0, 3.0),
    )
    return build_indexed_views(observations, max_window_index=4)


def _dense_verification_views():
    observations = tuple(
        _observation(track, window, float(track), float(window))
        for window in (1, 2, 3)
        for track in range(100, 115)
    )
    return build_indexed_views(observations, max_window_index=3)


def _create_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE cameras(
                camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER,
                height INTEGER, params BLOB, prior_focal_length INTEGER
            );
            CREATE TABLE images(
                image_id INTEGER PRIMARY KEY, name TEXT UNIQUE, camera_id INTEGER,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            );
            CREATE TABLE keypoints(
                image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE matches(
                pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE descriptors(
                image_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB
            );
            CREATE TABLE two_view_geometries(
                pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB,
                config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB,
                qvec BLOB, tvec BLOB
            );
            """
        )


def test_calibration_parser_is_strict_and_preserves_opencv_order() -> None:
    calibration = parse_calibration_text(
        "400 410 320 240 0.1 -0.02 0.003 -0.004 0.005\n"
    )

    np.testing.assert_array_equal(
        calibration.camera_matrix,
        [[400.0, 0.0, 320.0], [0.0, 410.0, 240.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(
        calibration.distortion, [0.1, -0.02, 0.003, -0.004, 0.005],
    )
    assert not calibration.camera_matrix.flags.writeable
    with pytest.raises(ColmapPipelineError) as count_error:
        parse_calibration_text("400 410 320 240 0 0 0 0")
    assert count_error.value.code == "invalid_calibration"
    with pytest.raises(ColmapPipelineError):
        parse_calibration_text("0 410 320 240 0 0 0 0 0")
    with pytest.raises(ColmapPipelineError):
        parse_calibration_text("400 410 320 240 0 0 nan 0 0")


def test_undistortion_uses_original_k_as_p_without_coordinate_postprocessing() -> None:
    class FakeCv2:
        __version__ = "4.9.0"

        def __init__(self) -> None:
            self.call = None

        def undistortPoints(self, points, matrix, distortion, *, R, P):
            self.call = (points.copy(), matrix.copy(), distortion.copy(), R, P.copy())
            return points + np.asarray([[[0.25, -0.5]]])

    fake = FakeCv2()
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    output = undistort_points(source, _calibration(), cv2_module=fake)

    np.testing.assert_allclose(output, [[1.25, 1.5], [3.25, 3.5]])
    assert fake.call[3] is None
    np.testing.assert_array_equal(fake.call[1], _calibration().camera_matrix)
    np.testing.assert_array_equal(fake.call[4], _calibration().camera_matrix)
    assert not output.flags.writeable
    source[0, 0] = 99.0
    assert output[0, 0] == 1.25

    fake.__version__ = "4.10.0"
    with pytest.raises(ColmapPipelineError) as captured:
        undistort_points([[1.0, 2.0]], _calibration(), cv2_module=fake)
    assert captured.value.code == "opencv_version_mismatch"


def test_logical_views_include_empty_windows_and_sort_features_by_track() -> None:
    views = _views()

    assert [view.image_id for view in views] == [1, 2, 3, 4]
    assert [view.window_index for view in views] == [1, 2, 3, 4]
    assert [view.image_name for view in views] == [
        "window_000001.png", "window_000002.png",
        "window_000003.png", "window_000004.png",
    ]
    assert views[0].track_ids.tolist() == [10, 20]
    np.testing.assert_array_equal(views[0].keypoints_xy, [[10.0, 1.0], [20.0, 1.0]])
    assert views[3].keypoints_xy.shape == (0, 2)
    assert not views[0].track_ids.flags.writeable
    assert not views[0].keypoints_xy.flags.writeable


def test_observation_schema_fails_closed() -> None:
    duplicate = (
        _observation(10, 1, 1.0, 2.0),
        _observation(10, 1, 3.0, 4.0),
    )
    with pytest.raises(ColmapPipelineError) as captured:
        build_indexed_views(duplicate, max_window_index=2)
    assert captured.value.code == "duplicate_observation"

    wrong_time = TrackObservation(10, 1, 31.0, 1.0, 2.0, 1)
    with pytest.raises(ColmapPipelineError) as captured:
        build_indexed_views([wrong_time], max_window_index=2)
    assert captured.value.code == "invalid_observation"


def test_all_pairs_matches_are_sorted_and_counted_from_track_lifetimes() -> None:
    views = _views()
    pairs = tuple(iter_pair_matches(views))

    assert [(pair.image_id1, pair.image_id2) for pair in pairs] == [
        (1, 2), (1, 3), (2, 3),
    ]
    np.testing.assert_array_equal(pairs[0].matches, [[0, 0]])
    np.testing.assert_array_equal(pairs[1].matches, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(pairs[2].matches, [[0, 0], [1, 2]])
    assert expected_raw_correspondence_count(views) == 5
    assert sum(pair.matches.shape[0] for pair in pairs) == 5


def test_matches_importer_file_is_deterministic_and_streamed(tmp_path: Path) -> None:
    target = tmp_path / "all_pairs.txt"

    counts = write_matches_importer_file(target, _views())

    assert counts == (3, 5)
    assert target.read_text(encoding="utf-8") == (
        "window_000001.png window_000002.png\n"
        "0 0\n\n"
        "window_000001.png window_000003.png\n"
        "0 0\n1 1\n\n"
        "window_000002.png window_000003.png\n"
        "0 0\n1 2\n\n"
    )


def test_colmap_pair_id_round_trips_and_rejects_invalid_ids() -> None:
    pair_id = image_ids_to_pair_id(7, 3)
    assert pair_id_to_image_ids(pair_id) == (3, 7)
    with pytest.raises(ColmapPipelineError):
        image_ids_to_pair_id(3, 3)
    with pytest.raises(ColmapPipelineError):
        pair_id_to_image_ids(0)


def test_database_population_and_postconditions_use_fixed_pinhole(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _create_database(database)
    views = _views()

    populate_known_camera_database(
        database, calibration=_calibration(), width=640, height=480, views=views,
    )
    validate_fixed_camera_database(
        database, calibration=_calibration(), width=640, height=480,
    )

    with sqlite3.connect(database) as connection:
        camera = connection.execute(
            "SELECT model,prior_focal_length FROM cameras",
        ).fetchone()
        images = connection.execute(
            "SELECT image_id,name,camera_id FROM images ORDER BY image_id",
        ).fetchall()
        keypoints = connection.execute(
            "SELECT rows,cols,data FROM keypoints WHERE image_id=1",
        ).fetchone()
    assert camera == (1, 1)
    assert images[-1] == (4, "window_000004.png", 1)
    assert keypoints[:2] == (2, 2)
    np.testing.assert_array_equal(
        np.frombuffer(keypoints[2], dtype=np.float32).reshape((2, 2)),
        views[0].keypoints_xy,
    )

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE cameras SET params=? WHERE camera_id=1",
            (np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64).tobytes(),),
        )
    with pytest.raises(ColmapPipelineError) as captured:
        validate_fixed_camera_database(
            database, calibration=_calibration(), width=640, height=480,
        )
    assert captured.value.code == "fixed_camera_changed"


def test_database_population_rejects_stale_output_tables(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,?,?)",
            (image_ids_to_pair_id(1, 2), 1, 2,
             np.asarray([[0, 0]], dtype=np.uint32).tobytes()),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        populate_known_camera_database(
            database, calibration=_calibration(), width=640, height=480,
            views=_views(),
        )
    assert captured.value.code == "nonempty_database"


def test_raw_match_and_geometric_verification_postconditions(tmp_path: Path) -> None:
    database = tmp_path / "database.db"
    _create_database(database)
    views = _dense_verification_views()
    populate_known_camera_database(
        database, calibration=_calibration(), width=640, height=480, views=views,
    )
    with sqlite3.connect(database) as connection:
        for pair in iter_pair_matches(views):
            connection.execute(
                "INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,?,?)",
                (pair.pair_id, pair.matches.shape[0], 2,
                 pair.matches.tobytes(order="C")),
            )
        connection.commit()

    validate_imported_raw_matches(database, views)
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "verification_pair_mismatch"

    pairs = tuple(iter_pair_matches(views))
    with sqlite3.connect(database) as connection:
        for index, pair in enumerate(pairs):
            inliers = pair.matches if index == 0 else pair.matches[:0]
            connection.execute(
                "INSERT INTO two_view_geometries"
                "(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (pair.pair_id, inliers.shape[0], 2,
                 inliers.tobytes() if inliers.size else None,
                 2 if inliers.size else 1,
                 None, None, None, None, None),
            )
        connection.commit()
    assert dict(verified_pair_counts(database, views)) == {
        pair.pair_id: (15 if index == 0 else 0)
        for index, pair in enumerate(pairs)
    }

    # The pinned COLMAP 3.9.1 verifier cannot emit a non-degenerate geometry
    # below its default min_num_inliers=15 threshold.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=? WHERE pair_id=?",
            (14, pairs[0].matches[:14].tobytes(), pairs[0].pair_id),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "invalid_verified_matches"

    # A subset check alone is insufficient: duplicated verified rows falsely
    # inflate the inlier count even when each row exists in the raw matches.
    duplicated = np.concatenate((pairs[0].matches, pairs[0].matches[:1]))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=? WHERE pair_id=?",
            (16, duplicated.tobytes(), pairs[0].pair_id),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "invalid_verified_matches"

    # Configuration 0 means verification was not completed.  Configurations
    # 4, 5, and 8 are valid COLMAP enum values but cannot be emitted by the
    # pinned compute_relative_pose=false / multiple_models=false verifier.
    for impossible_config in (0, 4, 5, 8, 9):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE two_view_geometries SET rows=?,data=?,config=? "
                "WHERE pair_id=?",
                (15, pairs[0].matches.tobytes(), impossible_config,
                 pairs[0].pair_id),
            )
            connection.commit()
        with pytest.raises(ColmapPipelineError) as captured:
            verified_pair_counts(database, views)
        assert captured.value.code == "invalid_verified_matches"

    # Conversely, an empty result must be explicitly marked DEGENERATE (1),
    # not carry a positive-geometry configuration.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=?,config=? "
            "WHERE pair_id=?",
            (15, pairs[0].matches.tobytes(), 2, pairs[0].pair_id),
        )
        connection.execute(
            "UPDATE two_view_geometries SET config=? WHERE pair_id=?",
            (2, pairs[1].pair_id),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "invalid_verified_matches"

    # WATERMARK rows are verified and included in counts when another
    # mapper-usable pair exists.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=?,config=? "
            "WHERE pair_id=?",
            (15, pairs[0].matches.tobytes(), 7, pairs[0].pair_id),
        )
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=?,config=? "
            "WHERE pair_id=?",
            (15, pairs[1].matches.tobytes(), 2, pairs[1].pair_id),
        )
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=?,config=? "
            "WHERE pair_id=?",
            (0, None, 1, pairs[2].pair_id),
        )
        connection.commit()
    assert dict(verified_pair_counts(database, views)) == {
        pairs[0].pair_id: 15,
        pairs[1].pair_id: 15,
        pairs[2].pair_id: 0,
    }

    # But COLMAP 3.9.1's mapper excludes WATERMARK geometry, so watermark
    # matches cannot be the only positive rows.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE two_view_geometries SET rows=?,data=?,config=? "
            "WHERE pair_id!=?",
            (0, None, 1, pairs[0].pair_id),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "no_verified_pairs"

    # An unrelated/stale row must never satisfy the verification postcondition.
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM two_view_geometries")
        stale_pair = image_ids_to_pair_id(1, 4)
        connection.execute(
            "INSERT INTO two_view_geometries"
            "(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (stale_pair, 1, 2, np.asarray([[0, 0]], dtype=np.uint32).tobytes(),
             2, None, None, None, None, None),
        )
        connection.commit()
    with pytest.raises(ColmapPipelineError) as captured:
        verified_pair_counts(database, views)
    assert captured.value.code == "verification_pair_mismatch"


def test_commands_require_raw_verification_and_fixed_intrinsics() -> None:
    plan = build_colmap_commands(
        database_path="work/database.db",
        match_list_path="work/all_pairs.txt",
        image_path="work/empty_images",
        output_path="work/models",
    )

    assert plan.database_creator[:2] == ("colmap", "database_creator")
    assert plan.matches_importer[:2] == ("colmap", "matches_importer")
    assert plan.matches_importer[plan.matches_importer.index("--match_type") + 1] == "raw"
    assert "--SiftMatching.use_gpu" in plan.matches_importer
    assert plan.mapper[plan.mapper.index("--Mapper.ba_refine_focal_length") + 1] == "0"
    assert plan.mapper[plan.mapper.index("--Mapper.ba_refine_principal_point") + 1] == "0"
    assert plan.mapper[plan.mapper.index("--Mapper.ba_refine_extra_params") + 1] == "0"
    assert plan.mapper[plan.mapper.index("--Mapper.extract_colors") + 1] == "0"
    assert plan.metadata["colmap_tag_commit"] == COLMAP_TAG_COMMIT
    assert OPENCV_DISTRIBUTION == "opencv-python-headless"
    assert OPENCV_DISTRIBUTION_VERSION == "4.9.0.80"
    assert plan.metadata["verification_bypassed"] is False
    assert plan.metadata["two_view_geometry_defaults"] == {
        "min_num_inliers": 15,
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
    }

    converter = build_model_converter_command(
        input_path="work/models/0", output_path="work/models_txt/0",
    )
    assert converter[:2] == ("colmap", "model_converter")
    assert converter[-2:] == ("--output_type", "TXT")

    assert_colmap_version("COLMAP 3.9.1 -- Structure-from-Motion")
    for rejected_version in (
        "COLMAP 3.10",
        "COLMAP 3.9.1-dev",
        "COLMAP 3.9.1.99",
    ):
        with pytest.raises(ColmapPipelineError) as captured:
            assert_colmap_version(rejected_version)
        assert captured.value.code == "colmap_version_mismatch"


def test_largest_model_rule_is_images_then_points_then_lowest_directory() -> None:
    models = (
        ModelSummary(2, 12, 300),
        ModelSummary(1, 12, 300),
        ModelSummary(0, 12, 200),
        ModelSummary(3, 11, 9000),
    )
    assert select_largest_model(models).directory_id == 1
    with pytest.raises(ColmapPipelineError) as captured:
        select_largest_model([])
    assert captured.value.code == "no_model"


def test_track_point_mapping_excludes_merges_splits_and_uninitialized_tracks() -> None:
    views = _views()
    mapping = validate_track_point_mapping(
        views,
        {
            100: [(1, 0), (2, 0), (3, 0)],  # track 10, unique
            200: [(1, 1), (2, 1)],          # tracks 20 and 30, merged
            201: [(3, 1)],                  # track 20 split across points
        },
    )

    assert mapping.unique_pairs == ((10, 100),)
    assert mapping.merged_point_ids == (200,)
    assert mapping.split_track_ids == (20,)
    assert mapping.uninitialized_track_ids == ()
    assert mapping.excluded_track_ids == (20, 30)

    with pytest.raises(ColmapPipelineError) as captured:
        validate_track_point_mapping(views, {1: [(99, 0)]})
    assert captured.value.code == "invalid_model"


def test_binary_model_discovery_and_text_model_extraction(tmp_path: Path) -> None:
    binary_root = tmp_path / "binary"
    for directory_id, image_count, point_count in ((0, 3, 3), (1, 2, 99)):
        directory = binary_root / str(directory_id)
        directory.mkdir(parents=True)
        (directory / "cameras.bin").write_bytes(struct.pack("<Q", 1))
        (directory / "images.bin").write_bytes(struct.pack("<Q", image_count))
        (directory / "points3D.bin").write_bytes(struct.pack("<Q", point_count))
    summaries = discover_sparse_models(binary_root)
    assert [(item.directory_id, item.registered_image_count, item.point3d_count)
            for item in summaries] == [(0, 3, 3), (1, 2, 99)]
    selected = select_largest_model(summaries)
    assert selected.directory_id == 0

    text_model = tmp_path / "text" / "0"
    text_model.mkdir(parents=True)
    text_model.joinpath("cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 400 410 320 240\n",
        encoding="utf-8",
    )
    text_model.joinpath("images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 0 1 window_000001.png\n"
        "10 1 100 20 1 200\n"
        "2 1 0 0 0 -1 0 0 1 window_000002.png\n"
        "10 2 100 30 2 300\n"
        "3 1 0 0 0 0 -1 0 1 window_000003.png\n"
        "10 3 100 20 3 200 30 3 300\n",
        encoding="utf-8",
    )
    text_model.joinpath("points3D.txt").write_text(
        "# Point list\n"
        "100 0 0 5 255 255 255 0 1 0 2 0 3 0\n"
        "200 1 0 5 255 255 255 0 1 1 3 1\n"
        "300 0 1 5 255 255 255 0 2 1 3 2\n",
        encoding="utf-8",
    )

    extracted = extract_text_model(
        text_model, summary=selected, views=_views(),
        calibration=_calibration(), width=640, height=480,
    )

    np.testing.assert_array_equal(extracted.trajectory.view_ids, [1, 2, 3])
    np.testing.assert_allclose(extracted.trajectory.centers_w,
                               [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert extracted.mapping.unique_pairs == ((10, 100), (20, 200), (30, 300))
    np.testing.assert_array_equal(extracted.mapped_track_ids, [10, 20, 30])
    np.testing.assert_array_equal(extracted.mapped_point_ids, [100, 200, 300])
    np.testing.assert_allclose(extracted.mapped_points_w,
                               [[0, 0, 5], [1, 0, 5], [0, 1, 5]])
    assert extracted.unregistered_window_indices == (4,)
    assert not extracted.mapped_points_w.flags.writeable

    # Counts alone cannot bind a converted model to its binary-model summary.
    # The text directory leaf is the canonical source model ID.
    with pytest.raises(ColmapPipelineError) as captured:
        extract_text_model(
            text_model, summary=ModelSummary(1, 3, 3), views=_views(),
            calibration=_calibration(), width=640, height=480,
        )
    assert captured.value.code == "invalid_model"

    symlink_model = text_model.parent / "1"
    symlink_model.symlink_to(text_model, target_is_directory=True)
    with pytest.raises(ColmapPipelineError) as captured:
        extract_text_model(
            symlink_model, summary=ModelSummary(1, 3, 3), views=_views(),
            calibration=_calibration(), width=640, height=480,
        )
    assert captured.value.code == "invalid_model"

    corrupted = text_model.joinpath("images.txt").read_text(encoding="utf-8")
    text_model.joinpath("images.txt").write_text(
        corrupted.replace("10 1 100", "10 1 -1", 1), encoding="utf-8",
    )
    with pytest.raises(ColmapPipelineError) as captured:
        extract_text_model(
            text_model, summary=selected, views=_views(),
            calibration=_calibration(), width=640, height=480,
        )
    assert captured.value.code == "invalid_model"


def test_text_model_camera_change_is_detected(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    model.joinpath("cameras.txt").write_text(
        "1 PINHOLE 640 480 401 410 320 240\n", encoding="utf-8",
    )
    with pytest.raises(ColmapPipelineError) as captured:
        from esfo.colmap import validate_text_model_camera
        validate_text_model_camera(
            model, calibration=_calibration(), width=640, height=480,
        )
    assert captured.value.code == "fixed_camera_changed"
