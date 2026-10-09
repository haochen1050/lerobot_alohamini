#!/usr/bin/env python3
# ruff: noqa: N803, N806  (T_a_b / K / R follow standard transform notation)
"""
Table perception tools for base alignment (milestone 3: perception only, the base never moves).

Run on the robot host with the host process stopped (it holds the cameras). `snapshot` saves a frame
with detected tags drawn, for checking what the camera sees. Steps, in order:

  1. capture-charuco       Save views of the ChArUco board held in front of the camera.
  2. calibrate-intrinsics  Camera matrix + distortion from those views.
  3. calibrate-extrinsics  Camera pose on the robot, from the AprilTag lying flat at a measured floor spot.
  4. watch                 Print distance / lateral / heading errors to the table tag.
  5. teach                 Park the robot at the desired pose by hand and record it as the target.

Frames: base x forward, y left, z up, origin at the base centre on the floor. A tag "yaw" of -90 means
the printed top of the tag points away from the robot with its edges parallel to the base axes.

Example:
  python examples/debug/table_perception.py capture-charuco
  python examples/debug/table_perception.py calibrate-intrinsics
  python examples/debug/table_perception.py calibrate-extrinsics --tag-x 1.00 --tag-y 0.0
  python examples/debug/table_perception.py watch --tag-edge-offset 0.10
  python examples/debug/table_perception.py teach --tag-edge-offset 0.10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.alohamini.perception import (
    AprilTagDetector,
    AprilTagTableEstimator,
    CameraIntrinsics,
    CharucoSpec,
    TableMeasurement,
    TableTarget,
    TagPlacement,
    calibrate_intrinsics,
    load_transform,
    measure_table_pose,
    pick_charuco_variant,
    save_transform,
    solve_base_from_camera,
)
from lerobot.robots.alohamini.perception.geometry import average_transforms, pose_xyz_yaw

OUT = Path("outputs/table_alignment")
DEFAULTS = {
    "camera": "/dev/am_camera_forward",
    "charuco_dir": OUT / "charuco_forward",
    "intrinsics": OUT / "forward_intrinsics.json",
    "extrinsics": OUT / "forward_extrinsics.json",
    "target": OUT / "table_target.json",
}

logger = logging.getLogger("table_perception")


class LatestFrame:
    """Background reader that keeps only the newest frame, so callers never see a buffered stale one."""

    def __init__(self, device: str, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device} (is the host process still running?)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, float] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, image = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = (image, time.monotonic())
            else:
                time.sleep(0.01)

    def __call__(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            return self._frame

    def wait_first(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("No frames from camera")
            time.sleep(0.05)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


def charuco_spec(args) -> CharucoSpec:
    return CharucoSpec(args.squares_x, args.squares_y, args.square_m, args.marker_m)


# ------------------------------------------------------------------ commands


def cmd_capture_charuco(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    cam = LatestFrame(args.camera)
    cam.wait_first()
    print(
        f"Hold the board in view and move it between saves: near/far, all image corners, tilted ~30 deg.\n"
        f"A view is saved when >= {args.min_markers} markers are seen and {args.interval_s}s have passed."
    )
    saved, last = len(list(out.glob("*.png"))), 0.0
    try:
        while saved < args.count:
            image, _ = cam()
            _, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            n = 0 if ids is None else len(ids)
            now = time.monotonic()
            if n >= args.min_markers and now - last >= args.interval_s:
                cv2.imwrite(str(out / f"view_{saved:03d}.png"), image)
                saved, last = saved + 1, now
                print(f"saved {saved}/{args.count} ({n} markers)")
            else:
                print(f"\rmarkers visible: {n:3d}   ", end="", flush=True)
            time.sleep(0.1)
    finally:
        cam.close()
    print(f"\nViews in {out}")


def cmd_calibrate_intrinsics(args) -> None:
    paths = sorted(Path(args.images).glob("*.png"))
    images = [cv2.imread(str(p)) for p in paths]
    if not images:
        raise SystemExit(f"No .png views in {args.images}")
    spec = pick_charuco_variant(images[:8], charuco_spec(args))
    print(f"Board layout: {spec.squares_x}x{spec.squares_y} squares, legacy={spec.legacy}")
    intr = calibrate_intrinsics(images, spec)
    intr.save(args.out)
    K = intr.camera_matrix
    print(f"RMS reprojection {intr.rms_px:.3f} px from {len(images)} views (aim for < 0.5 px)")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}  size={intr.image_size}")
    print(f"dist={np.round(intr.dist_coeffs, 4).tolist()}")
    print(f"Saved {args.out}")


def describe_camera(T_base_cam: np.ndarray) -> str:
    x, y, z = T_base_cam[:3, 3]
    optical = T_base_cam[:3, 2]
    pitch_down = math.degrees(math.asin(-optical[2]))
    yaw = math.degrees(math.atan2(optical[1], optical[0]))
    cam_right = T_base_cam[:3, 0]
    roll = math.degrees(math.asin(-cam_right[2]))
    return (
        f"camera at x={x * 100:.1f} cm (fwd), y={y * 100:.1f} cm (left), z={z * 100:.1f} cm (up); "
        f"pitched down {pitch_down:.1f} deg, yaw {yaw:+.1f} deg, roll {roll:+.1f} deg"
    )


def collect_tag_poses(cam: LatestFrame, detector: AprilTagDetector, tag_id: int | None, frames: int) -> list:
    poses, last_t = [], None
    deadline = time.monotonic() + 10 + frames / 5
    while len(poses) < frames and time.monotonic() < deadline:
        image, t = cam()
        if t == last_t:
            time.sleep(0.01)
            continue
        last_t = t
        found = [d for d in detector.detect(image) if tag_id is None or d[0] == tag_id]
        if len(found) == 1:
            # Rough prior (camera looking forward, pitched ~45 deg down) to pick the right one of the two
            # planar PnP solutions; it only needs to be on the correct side.
            poses.append(detector.solve(*found[0], up_in_cam=np.array([0.0, -0.7071, -0.7071])))
    return poses


def cmd_calibrate_extrinsics(args) -> None:
    intr = CameraIntrinsics.load(args.intrinsics)
    detector = AprilTagDetector(intr, args.tag_size)
    cam = LatestFrame(args.camera, *intr.image_size)
    try:
        cam.wait_first()
        observations = collect_tag_poses(cam, detector, args.tag_id, args.frames)
    finally:
        cam.close()
    if len(observations) < args.frames // 2:
        raise SystemExit(f"Tag seen in only {len(observations)} frames; check it is in view and well lit")

    reproj = np.median([o.reprojection_px for o in observations])
    T_base_tag = pose_xyz_yaw(args.tag_x, args.tag_y, args.tag_z, args.tag_yaw_deg)
    T_base_cam = solve_base_from_camera(average_transforms([o.T_cam_tag for o in observations]), T_base_tag)
    save_transform(
        args.extrinsics,
        T_base_cam,
        tag_pose_in_base=[args.tag_x, args.tag_y, args.tag_z, args.tag_yaw_deg],
        frames=len(observations),
        median_reprojection_px=float(reproj),
    )
    print(f"{len(observations)} frames, median reprojection {reproj:.2f} px")
    print(describe_camera(T_base_cam))
    print("Sanity check against your tape measure (expected roughly x=10 cm, z=110 cm).")
    print(f"Saved {args.extrinsics}")

    # The forward camera looks roughly along +x, so a large yaw means the tag is rotated by a multiple
    # of 90 deg relative to --tag-yaw-deg (the printed tag has no obvious "top").
    cam_yaw = math.degrees(math.atan2(T_base_cam[1, 2], T_base_cam[0, 2]))
    if abs(cam_yaw) > 45:
        suggested = (args.tag_yaw_deg - 90 * round(cam_yaw / 90) + 180) % 360 - 180
        print(
            f"WARNING: camera yaw {cam_yaw:+.0f} deg - the tag is probably rotated relative to "
            f"--tag-yaw-deg {args.tag_yaw_deg:g}. Re-run with --tag-yaw-deg {suggested:g}, and use the same "
            f"value for watch/teach while the tag stays in this orientation."
        )


def table_tag_yaw(args) -> float:
    """--tag-yaw-deg, else the yaw used at extrinsic calibration (robot faced the table then), else -90."""
    if args.tag_yaw_deg is not None:
        return args.tag_yaw_deg
    try:
        yaw = float(json.loads(Path(args.extrinsics).read_text())["tag_pose_in_base"][3])
        print(f"Tag yaw {yaw:g} deg (from {args.extrinsics}; override with --tag-yaw-deg)")
        return yaw
    except (OSError, KeyError, IndexError, ValueError):
        return -90.0


def make_estimator(args, cam: LatestFrame | None, target: TableTarget) -> AprilTagTableEstimator:
    placement = TagPlacement(
        x_m=args.tag_edge_offset,
        y_m=args.tag_lateral,
        yaw_deg=table_tag_yaw(args),
        size_m=args.tag_size,
        tag_id=args.tag_id,
    )
    return AprilTagTableEstimator(
        CameraIntrinsics.load(args.intrinsics), load_transform(args.extrinsics), placement, target, cam
    )


def tag_placement_args(args) -> dict:
    return {
        "edge_offset_m": args.tag_edge_offset,
        "lateral_m": args.tag_lateral,
        "yaw_deg": table_tag_yaw(args),
        "size_m": args.tag_size,
        "tag_id": args.tag_id,
    }


def target_placement_mismatch(args) -> str | None:
    """Description of how the current tag options differ from those used to teach the target, if they do."""
    path = Path(args.target)
    taught_with = json.loads(path.read_text()).get("tag_placement") if path.exists() else None
    now = tag_placement_args(args)
    if taught_with is not None and taught_with != now:
        return f"target was taught with tag placement {taught_with}, now using {now}"
    return None


def load_target(args) -> TableTarget:
    path = Path(args.target)
    if path.exists():
        target = TableTarget.load(path)
        print(f"Target from {path}: {target}")
        if (mismatch := target_placement_mismatch(args)) is not None:
            print(f"WARNING: {mismatch}. Errors will be offset; use the same tag options or re-teach.")
        return target
    print(f"No taught target at {path}; errors are relative to TableTarget() defaults")
    return TableTarget(reference_x_m=args.reference_x)


def cmd_watch(args) -> None:
    target = load_target(args)
    intr = CameraIntrinsics.load(args.intrinsics)
    cam = LatestFrame(args.camera, *intr.image_size)
    estimator = make_estimator(args, cam, target)
    print("Errors: + distance -> drive forward, + lateral -> strafe left, + heading -> rotate CCW")
    try:
        cam.wait_first()
        while True:
            err = estimator.get_table_pose_error()
            if err.valid:
                m = measure_table_pose(
                    estimator.last_T_base_table, target.reference_x_m, target.reference_y_m
                )
                print(
                    f"dist_err {err.distance_error_m * 100:+6.1f} cm  lat_err {err.lateral_error_m * 100:+6.1f} cm  "
                    f"head_err {err.heading_error_deg:+6.1f} deg   | measured dist {m.distance_m * 100:5.1f} cm "
                    f"lat {m.lateral_m * 100:+5.1f} cm head {m.heading_deg:+5.1f} deg"
                )
            else:
                print(f"INVALID: {err.reason}")
            time.sleep(1.0 / args.rate_hz)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


def cmd_teach(args) -> None:
    intr = CameraIntrinsics.load(args.intrinsics)
    cam = LatestFrame(args.camera, *intr.image_size)
    estimator = make_estimator(args, cam, TableTarget(reference_x_m=args.reference_x))
    samples: list[TableMeasurement] = []
    try:
        cam.wait_first()
        last_t = None
        deadline = time.monotonic() + 10 + args.frames / 5
        while len(samples) < args.frames and time.monotonic() < deadline:
            frame = cam()
            if frame[1] == last_t:
                time.sleep(0.01)
                continue
            last_t = frame[1]
            if estimator.estimate(*frame).valid:
                samples.append(measure_table_pose(estimator.last_T_base_table, args.reference_x))
    finally:
        cam.close()
    if len(samples) < args.frames // 2:
        raise SystemExit(f"Only {len(samples)} valid measurements; is the tag in view?")

    arr = np.array([[s.distance_m, s.lateral_m, s.heading_deg] for s in samples])
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    target = TableTarget.taught(TableMeasurement(*mean), reference_x_m=args.reference_x)
    target.save(args.target, tag_placement=tag_placement_args(args))
    print(
        f"Taught from {len(samples)} frames: dist {mean[0] * 100:.1f} cm (+-{std[0] * 100:.2f}), "
        f"lat {mean[1] * 100:+.1f} cm (+-{std[1] * 100:.2f}), head {mean[2]:+.2f} deg (+-{std[2]:.2f})"
    )
    print(f"Saved {args.target}")


def cmd_snapshot(args) -> None:
    cam = LatestFrame(args.camera)
    try:
        cam.wait_first()
        time.sleep(0.5)  # let auto-exposure settle
        image, _ = cam()
    finally:
        cam.close()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, image)
    print(f"AprilTags seen: {[] if ids is None else ids.ravel().tolist()}. Saved {args.out}")


# ------------------------------------------------------------------ CLI


def add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--camera", default=DEFAULTS["camera"])
    p.add_argument("--intrinsics", default=str(DEFAULTS["intrinsics"]))
    p.add_argument("--extrinsics", default=str(DEFAULTS["extrinsics"]))


def board_args(sp) -> None:
    sp.add_argument("--squares-x", type=int, default=9)
    sp.add_argument("--squares-y", type=int, default=12)
    sp.add_argument("--square-m", type=float, default=0.030)
    sp.add_argument("--marker-m", type=float, default=0.0225)


def tag_args(sp) -> None:
    sp.add_argument("--tag-size", type=float, default=0.10, help="Black square edge (m)")
    sp.add_argument(
        "--tag-id", type=int, default=None, help="Only use this tag ID (default: the one tag seen)"
    )


def table_args(sp) -> None:
    """Tag placement on the table and target options, shared by watch/teach and the alignment script."""
    tag_args(sp)
    sp.add_argument(
        "--tag-yaw-deg",
        type=float,
        default=None,
        help="Tag yaw in the table frame (default: the value used for calibrate-extrinsics)",
    )
    sp.add_argument(
        "--tag-edge-offset", type=float, required=True, help="Tag centre distance from the front edge (m)"
    )
    sp.add_argument(
        "--tag-lateral", type=float, default=0.0, help="Tag centre left of the work-region centre (m)"
    )
    sp.add_argument(
        "--reference-x", type=float, default=0.0, help="Robot reference point, fwd of base centre"
    )
    sp.add_argument("--target", default=str(DEFAULTS["target"]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="Save one frame with detected AprilTags drawn (headless preview)")
    sp.add_argument("--out", default=str(OUT / "snapshot.png"))
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("capture-charuco")
    sp.add_argument("--out", default=str(DEFAULTS["charuco_dir"]))
    sp.add_argument("--count", type=int, default=25)
    sp.add_argument("--min-markers", type=int, default=12)
    sp.add_argument("--interval-s", type=float, default=1.5)
    sp.set_defaults(func=cmd_capture_charuco)

    sp = sub.add_parser("calibrate-intrinsics")
    board_args(sp)
    sp.add_argument("--images", default=str(DEFAULTS["charuco_dir"]))
    sp.add_argument("--out", default=str(DEFAULTS["intrinsics"]))
    sp.set_defaults(func=cmd_calibrate_intrinsics)

    sp = sub.add_parser("calibrate-extrinsics")
    tag_args(sp)
    sp.add_argument("--tag-x", type=float, required=True, help="Tag centre forward of the base centre (m)")
    sp.add_argument("--tag-y", type=float, default=0.0, help="Tag centre left of the base centre (m)")
    sp.add_argument("--tag-z", type=float, default=0.0, help="Tag height above the floor (m)")
    sp.add_argument(
        "--tag-yaw-deg",
        type=float,
        default=-90.0,
        help="-90: printed top points away from the robot; 0: printed top points to the robot's left",
    )
    sp.add_argument("--frames", type=int, default=30)
    sp.set_defaults(func=cmd_calibrate_extrinsics)

    sp = sub.add_parser("watch")
    table_args(sp)
    sp.add_argument("--rate-hz", type=float, default=5.0)
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("teach")
    table_args(sp)
    sp.add_argument("--frames", type=int, default=30)
    sp.set_defaults(func=cmd_teach)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
