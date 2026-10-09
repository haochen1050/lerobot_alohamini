# ruff: noqa: N803, N806  (T_a_b / K / R follow standard transform notation)
import math

import cv2
import numpy as np
import pytest

from lerobot.robots.alohamini.perception import (
    AprilTagTableEstimator,
    CameraIntrinsics,
    CharucoSpec,
    TableTarget,
    TagPlacement,
    calibrate_intrinsics,
    measure_table_pose,
    pick_charuco_variant,
    solve_base_from_camera,
    table_pose_error,
)
from lerobot.robots.alohamini.perception.geometry import (
    average_transforms,
    invert,
    make_transform,
    pose_xyz_yaw,
)

W, H = 640, 480
K = np.array([[520.0, 0, 322.0], [0, 518.0, 236.0], [0, 0, 1]])
INTRINSICS = CameraIntrinsics(K, np.zeros(5), (W, H))


def camera_on_mast(height=1.10, forward=0.10, pitch_down_deg=35.0) -> np.ndarray:
    """T_base_cam for a forward-looking camera pitched down. Camera axes: x right, y down, z optical."""
    p = math.radians(pitch_down_deg)
    forward_dir = np.array([math.cos(p), 0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(forward_dir, right)
    return make_transform(np.column_stack([right, down, forward_dir]), (forward, 0, height))


T_BASE_CAM = camera_on_mast()


def render_plane(
    texture: np.ndarray, plane_size_m: tuple[float, float], T_cam_plane: np.ndarray
) -> np.ndarray:
    """Pinhole render of a texture lying in the z=0 plane of T_cam_plane, centred on the origin.

    Texture pixel (u, v) maps to plane (x, y) = (u*sx - w/2, h/2 - v*sy): image-up is plane +y.
    """
    th, tw = texture.shape[:2]
    pw, ph = plane_size_m
    tex_to_plane = np.array([[pw / tw, 0, -pw / 2], [0, -ph / th, ph / 2], [0, 0, 1]])
    plane_to_img = K @ T_cam_plane[:3][:, [0, 1, 3]]
    Hmat = plane_to_img @ tex_to_plane
    background = np.full((H, W), 90, np.uint8)
    warped = cv2.warpPerspective(texture, Hmat, (W, H), flags=cv2.INTER_LINEAR, borderValue=1)
    mask = cv2.warpPerspective(np.full_like(texture, 255), Hmat, (W, H), borderValue=0)
    out = np.where(mask > 0, warped, background)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def tag_texture(tag_id=3, size_px=400):
    """Tag with a white quiet zone of one cell; returned with its physical scale factor."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, tag_id, size_px)
    margin = size_px // 8
    return cv2.copyMakeBorder(tag, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255), (
        size_px + 2 * margin
    ) / size_px


def render_tag(T_base_tag: np.ndarray, size_m=0.10, tag_id=3) -> np.ndarray:
    texture, scale = tag_texture(tag_id)
    T_cam_tag = invert(T_BASE_CAM) @ T_base_tag
    return render_plane(texture, (size_m * scale, size_m * scale), T_cam_tag)


# ---------------------------------------------------------------- pure geometry / signs


def test_error_signs_match_base_commands():
    target = TableTarget(distance_m=0.30)
    aligned = table_pose_error(pose_xyz_yaw(0.30, 0, 0.75, 0), target, 1.0)
    assert aligned.valid
    assert aligned.distance_error_m == pytest.approx(0)
    assert aligned.lateral_error_m == pytest.approx(0)
    assert aligned.heading_error_deg == pytest.approx(0)

    # Table too far -> drive forward (+vx).
    assert table_pose_error(pose_xyz_yaw(0.45, 0, 0, 0), target, 0).distance_error_m == pytest.approx(0.15)
    # Work region to the robot's left -> strafe left (+vy).
    assert table_pose_error(pose_xyz_yaw(0.30, 0.07, 0, 0), target, 0).lateral_error_m == pytest.approx(0.07)
    # Table normal turned to the left -> rotate CCW (+omega).
    assert table_pose_error(pose_xyz_yaw(0.30, 0, 0, 5), target, 0).heading_error_deg == pytest.approx(5)


def test_moving_by_the_error_aligns_the_robot():
    target = TableTarget(distance_m=0.30, lateral_m=0.02, reference_x_m=0.15)
    T_base_table = pose_xyz_yaw(0.62, -0.11, 0, 8)
    err = table_pose_error(T_base_table, target, 0)
    # Rotate by the heading error, then translate along the (now aligned) axes by the other errors.
    T_new_old = invert(pose_xyz_yaw(0, 0, 0, err.heading_error_deg))
    T_new_table = T_new_old @ T_base_table
    err2 = table_pose_error(T_new_table, target, 0)
    T_moved = invert(pose_xyz_yaw(err2.distance_error_m, err2.lateral_error_m, 0, 0)) @ T_new_table
    final = table_pose_error(T_moved, target, 0)
    assert (final.distance_error_m, final.lateral_error_m, final.heading_error_deg) == pytest.approx(
        (0, 0, 0), abs=1e-9
    )


def test_average_transforms_recovers_mean():
    base = pose_xyz_yaw(0.5, 0.1, 0.2, 30)
    jitter = [base @ pose_xyz_yaw(d, -d, 0, 10 * d) for d in (-0.01, 0.0, 0.01)]
    assert average_transforms(jitter) == pytest.approx(base, abs=1e-6)


# ---------------------------------------------------------------- AprilTag estimator on rendered images


@pytest.mark.parametrize(
    ("x", "y", "yaw_deg"),
    [(0.80, 0.0, 0.0), (0.75, 0.08, 6.0), (0.90, -0.10, -10.0)],
)
def test_estimator_recovers_table_pose_from_rendered_tag(x, y, yaw_deg):
    placement = TagPlacement(x_m=0.12, tag_id=3)
    target = TableTarget(distance_m=0.30)
    # Ground-truth table frame on a 0.75 m high table; tag placed on it as described by `placement`.
    T_base_table = pose_xyz_yaw(x, y, 0.75, yaw_deg)
    image = render_tag(T_base_table @ placement.T_table_tag())

    estimator = AprilTagTableEstimator(INTRINSICS, T_BASE_CAM, placement, target, clock=lambda: 10.0)
    measured = estimator.estimate(image, capture_time_s=10.0)
    expected = table_pose_error(T_base_table, target, 10.0)

    assert measured.valid, measured.reason
    assert measured.distance_error_m == pytest.approx(expected.distance_error_m, abs=0.01)
    # Lateral is measured along the table edge, so heading noise is amplified by the ~0.9 m lever arm
    # (0.6 deg -> ~1 cm). It shrinks as the heading is aligned first.
    assert measured.lateral_error_m == pytest.approx(expected.lateral_error_m, abs=0.015)
    assert measured.heading_error_deg == pytest.approx(expected.heading_error_deg, abs=1.0)


def test_estimator_rejects_stale_missing_wrong_and_mismatched_frames():
    placement = TagPlacement(x_m=0.12, tag_id=3)
    est = AprilTagTableEstimator(INTRINSICS, T_BASE_CAM, placement, TableTarget(), clock=lambda: 10.0)
    image = render_tag(pose_xyz_yaw(0.8, 0, 0.75, 0) @ placement.T_table_tag())

    assert est.estimate(image, 10.0).valid
    stale = est.estimate(image, 9.0)
    assert not stale.valid and "stale" in stale.reason
    assert math.isnan(stale.distance_error_m)

    blank = np.full((H, W, 3), 90, np.uint8)
    assert "saw []" in est.estimate(blank, 10.0).reason

    other_id = render_tag(pose_xyz_yaw(0.8, 0, 0.75, 0) @ placement.T_table_tag(), tag_id=7)
    assert "saw [7]" in est.estimate(other_id, 10.0).reason

    assert "does not match calibration" in est.estimate(cv2.resize(image, (320, 240)), 10.0).reason


@pytest.mark.parametrize("rotation_deg", [90, 180, -90])
def test_estimator_rejects_tag_rotated_relative_to_configured_yaw(rotation_deg):
    placement = TagPlacement(x_m=0.12, tag_id=3)
    T_base_table = pose_xyz_yaw(0.8, 0, 0.75, -6)
    # The physical tag is turned relative to what `placement` says.
    image = render_tag(T_base_table @ placement.T_table_tag() @ pose_xyz_yaw(0, 0, 0, rotation_deg))
    est = AprilTagTableEstimator(INTRINSICS, T_BASE_CAM, placement, TableTarget(), clock=lambda: 0.0)
    err = est.estimate(image, 0.0)
    assert not err.valid
    assert "tag probably rotated" in err.reason
    assert est.last_T_base_table is None


def test_target_load_ignores_metadata(tmp_path):
    target = TableTarget(0.25, -0.006, -6.0)
    target.save(tmp_path / "t.json", tag_placement={"edge_offset_m": 0.15})
    assert TableTarget.load(tmp_path / "t.json") == target


def test_estimator_without_frame_source_is_invalid():
    est = AprilTagTableEstimator(INTRINSICS, T_BASE_CAM, TagPlacement(x_m=0.1), TableTarget())
    assert not est.get_table_pose_error().valid


def test_extrinsics_from_tag_at_known_floor_pose():
    from lerobot.robots.alohamini.perception import AprilTagDetector

    T_base_tag = pose_xyz_yaw(1.30, 0.0, 0.0, -90)
    image = render_tag(T_base_tag)
    detector = AprilTagDetector(INTRINSICS, 0.10)
    ((tag_id, corners),) = detector.detect(image)
    obs = detector.solve(tag_id, corners)
    T_base_cam = solve_base_from_camera(obs.T_cam_tag, T_base_tag)

    # A single 10 cm tag at ~1.7 m gives a few cm / ~1 deg; the resulting constant bias is cancelled by
    # teaching the target pose with the same calibration.
    assert T_base_cam[:3, 3] == pytest.approx(T_BASE_CAM[:3, 3], abs=0.04)
    angle = math.degrees(math.acos(min(1.0, (np.trace(T_base_cam[:3, :3].T @ T_BASE_CAM[:3, :3]) - 1) / 2)))
    assert angle < 1.5


def test_taught_target_cancels_extrinsic_bias():
    placement = TagPlacement(x_m=0.12, tag_id=3)
    # Calibration that is off by 3 cm and 1.5 deg.
    T_base_cam_biased = pose_xyz_yaw(0.03, -0.01, 0.0, 1.5) @ T_BASE_CAM
    T_taught = pose_xyz_yaw(0.70, 0.02, 0.75, 0)
    est = AprilTagTableEstimator(INTRINSICS, T_base_cam_biased, placement, TableTarget(), clock=lambda: 0.0)

    assert est.estimate(render_tag(T_taught @ placement.T_table_tag()), 0.0).valid
    est.target = TableTarget.taught(measure_table_pose(est.last_T_base_table))

    # Back at the taught pose every error reads ~0 even though the calibration is biased...
    again = est.estimate(render_tag(T_taught @ placement.T_table_tag()), 0.0)
    assert (again.distance_error_m, again.lateral_error_m, again.heading_error_deg) == pytest.approx(
        (0, 0, 0), abs=1e-3
    )
    # ...and a displaced robot still reads errors with the right sign and roughly the right size.
    moved = est.estimate(render_tag(pose_xyz_yaw(0.80, 0.07, 0.75, 0) @ placement.T_table_tag()), 0.0)
    assert moved.distance_error_m == pytest.approx(0.10, abs=0.015)
    assert moved.lateral_error_m == pytest.approx(0.05, abs=0.015)


def test_target_round_trip(tmp_path):
    target = TableTarget(0.31, -0.02, 1.5, 0.1, 0.0)
    target.save(tmp_path / "t.json")
    assert TableTarget.load(tmp_path / "t.json") == target


# ---------------------------------------------------------------- ChArUco intrinsics


def render_board(spec: CharucoSpec, T_cam_board: np.ndarray) -> np.ndarray:
    w_m, h_m = spec.squares_x * spec.square_m, spec.squares_y * spec.square_m
    texture = spec.board().generateImage((spec.squares_x * 60, spec.squares_y * 60))
    # Board object points have their origin at a corner with y pointing down the image; shift the
    # rendering so that render_plane's centred, y-up texture lands on the same board coordinates.
    T_center = make_transform(np.diag([1.0, -1.0, -1.0]), (w_m / 2, h_m / 2, 0))
    return render_plane(texture, (w_m, h_m), T_cam_board @ T_center)


def board_views(spec: CharucoSpec) -> list[np.ndarray]:
    w_m, h_m = spec.squares_x * spec.square_m, spec.squares_y * spec.square_m
    views = []
    for rx, ry, rz, dist in [
        (0, 0, 0, 0.55),
        (20, 0, 5, 0.55),
        (-20, 5, -5, 0.6),
        (0, 25, 0, 0.6),
        (0, -25, 10, 0.55),
        (15, 15, -10, 0.65),
        (-15, -15, 0, 0.6),
        (25, -10, 15, 0.7),
        (-10, 20, -15, 0.65),
        (10, -20, 20, 0.6),
        (-25, 0, 0, 0.7),
        (5, 30, 5, 0.7),
    ]:
        R, _ = cv2.Rodrigues(np.radians([rx, ry, rz]))
        center = R @ np.array([w_m / 2, h_m / 2, 0])
        views.append(render_board(spec, make_transform(R, np.array([0, 0, dist]) - center)))
    return views


def test_charuco_calibration_recovers_intrinsics_and_board_variant():
    printed = CharucoSpec(legacy=True)
    views = board_views(printed)

    # Start from the default (non-legacy) spec, as a user would; the right variant must be found.
    variant = pick_charuco_variant(views[:3], CharucoSpec())
    assert (variant.squares_x, variant.squares_y, variant.legacy) == (9, 12, True)

    result = calibrate_intrinsics(views, variant)
    assert result.image_size == (W, H)
    assert result.rms_px < 1.0
    assert result.camera_matrix[0, 0] == pytest.approx(K[0, 0], rel=0.03)
    assert result.camera_matrix[1, 1] == pytest.approx(K[1, 1], rel=0.03)
    assert result.camera_matrix[:2, 2] == pytest.approx(K[:2, 2], abs=10)


def test_intrinsics_round_trip(tmp_path):
    path = tmp_path / "cam.json"
    INTRINSICS.save(path)
    loaded = CameraIntrinsics.load(path)
    assert loaded.camera_matrix == pytest.approx(K)
    assert loaded.image_size == (W, H)
