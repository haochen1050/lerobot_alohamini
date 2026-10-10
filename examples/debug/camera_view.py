#!/usr/bin/env python
"""Live view of one AlohaMini camera on the robot's own screen.

Run on the robot host with the host process stopped (it holds the cameras). The window is drawn by
ffplay because the environment ships opencv-python-headless, so cv2.imshow is not available. Over
SSH the window still opens on the Pi's desktop. Press q or Esc in the window to quit.

With --tags, AprilTags (36h11) are outlined and a status bar says whether table alignment could use
the frame. That check is the one align_to_table.py runs, built from the calibration and the taught
target in outputs/table_alignment (so it assumes the tag placement used when teaching):

  green   TABLE OK      the alignment estimator accepts the frame; errors to the taught target shown
  orange  TAG REJECTED  a tag is detected but the estimator refuses it, with the reason
  red     NO TAG        nothing detected

  python examples/debug/camera_view.py                      # forward camera, 640x480
  python examples/debug/camera_view.py wrist_right
  python examples/debug/camera_view.py forward --size 1280x720
  python examples/debug/camera_view.py /dev/video4
  python examples/debug/camera_view.py --tags               # forward camera with tag / table status
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections import deque
from pathlib import Path

CAMERAS = ("forward", "backward", "chest", "wrist_left", "wrist_right")
OUT = Path("outputs/table_alignment")

GREEN, ORANGE, RED = (0, 200, 0), (0, 150, 255), (0, 0, 230)


def load_estimator(args, size: tuple[int, int]):
    """The alignment estimator as taught, or None (with the reason printed) to only outline tags."""
    from lerobot.robots.alohamini.perception import (
        AprilTagTableEstimator,
        CameraIntrinsics,
        TableTarget,
        TagPlacement,
        load_transform,
    )

    missing = [str(p) for p in (args.intrinsics, args.extrinsics, args.target) if not Path(p).exists()]
    if missing:
        print(f"Missing {', '.join(missing)}: outlining tags only, no table alignment check")
        return None
    intrinsics = CameraIntrinsics.load(args.intrinsics)
    if tuple(intrinsics.image_size) != size:
        print(f"Calibration is for {intrinsics.image_size}, not {size}: outlining tags only")
        return None
    taught_with = json.loads(Path(args.target).read_text()).get("tag_placement")
    if taught_with is None:
        print(f"{args.target} does not record its tag placement: outlining tags only")
        return None
    placement = TagPlacement(
        x_m=taught_with["edge_offset_m"],
        y_m=taught_with["lateral_m"],
        yaw_deg=taught_with["yaw_deg"],
        size_m=taught_with["size_m"],
        tag_id=taught_with["tag_id"],
        mount=taught_with.get("mount", "flat"),
        z_m=taught_with.get("height_m", 0.0),
    )
    print(f"Table check uses the taught tag placement: {placement}")
    return AprilTagTableEstimator(
        intrinsics, load_transform(args.extrinsics), placement, TableTarget.load(args.target)
    )


def view_tags(args, device: Path, ffplay: str) -> None:
    import cv2

    width, height = (int(v) for v in args.size.split("x"))
    # Before opening the camera: importing lerobot takes several seconds on the Pi.
    estimator = load_estimator(args, (width, height))
    cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {device} (is the host process still running?)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    player = subprocess.Popen(  # noqa: S603
        [
            ffplay,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-window_title", f"{args.camera} tags",
            "-",
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip

    last_printed, pending, pending_frames = None, None, 0
    recent = deque(maxlen=30)  # tag detected or not, per frame
    try:
        while player.poll() is None:
            ok, image = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if image.shape[:2] != (height, width):
                sys.exit(f"Camera delivered {image.shape[1]}x{image.shape[0]}, not {args.size}")
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            seen = [] if ids is None else ids.ravel().tolist()
            recent.append(bool(seen))

            # The estimator sees the clean frame; drawing comes after.
            err = estimator.estimate(image, time.monotonic()) if estimator is not None else None
            if seen:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
            if not seen:
                colour, summary, lines = RED, "NO TAG", ["NO TAG"]
            elif err is None:
                colour, summary = GREEN, f"TAG {seen}"
                lines = [f"TAG {seen} detected"]
            elif err.valid:
                colour, summary = GREEN, "TABLE OK"
                lines = [
                    f"TABLE OK  tag {seen}",
                    f"dist {err.distance_error_m * 100:+.1f} cm  lat {err.lateral_error_m * 100:+.1f} cm  "
                    f"head {err.heading_error_deg:+.1f} deg",
                ]
            else:
                colour, summary = ORANGE, f"TAG REJECTED: {err.reason}"
                lines = [f"TAG REJECTED  tag {seen}", *textwrap.wrap(err.reason, 80)]

            # A tag at the limit of detection flickers; the rate says how reliably it is seen.
            lines[0] += f"   (tag in {sum(recent)}/{len(recent)} frames)"
            cv2.rectangle(image, (0, 0), (width, 8 + 20 * len(lines)), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                scale = 0.6 if i == 0 else 0.42
                cv2.putText(
                    image, line, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA
                )
            # Print a status once it has held for 10 frames, so single-frame dropouts stay off the terminal.
            pending_frames = pending_frames + 1 if summary == pending else 1
            pending = summary
            if pending_frames == 10 and summary != last_printed:
                print(f"{summary}   (tag in {sum(recent)}/{len(recent)} frames)")
                last_printed = summary
            player.stdin.write(image.tobytes())
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        cap.release()
        player.terminate()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "camera", nargs="?", default="forward", help=f"one of {', '.join(CAMERAS)}, or a device path"
    )
    p.add_argument("--size", default="640x480", help="WIDTHxHEIGHT (default: 640x480)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tags", action="store_true", help="outline AprilTags and show the table alignment check")
    p.add_argument("--intrinsics", default=str(OUT / "forward_intrinsics.json"))
    p.add_argument("--extrinsics", default=str(OUT / "forward_extrinsics.json"))
    p.add_argument("--target", default=str(OUT / "table_target.json"))
    args = p.parse_args()

    device = Path(f"/dev/am_camera_{args.camera}" if args.camera in CAMERAS else args.camera)
    if not device.exists():
        present = sorted(d.name.removeprefix("am_camera_") for d in Path("/dev").glob("am_camera_*"))
        sys.exit(f"{device} not found. Connected cameras: {', '.join(present) or 'none'}")

    ffplay = shutil.which("ffplay")
    if ffplay is None:
        sys.exit("ffplay not found (it ships with ffmpeg)")

    # No display in the environment (e.g. an SSH shell): use the desktop session on the Pi's screen.
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        os.environ["DISPLAY"] = ":0"

    if args.tags:
        view_tags(args, device, ffplay)
        return

    os.execv(  # noqa: S606
        ffplay,
        [
            ffplay,
            "-hide_banner",
            "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", args.size,
            "-framerate", str(args.fps),
            "-fflags", "nobuffer",
            "-window_title", f"{args.camera} {args.size}",
            str(device),
        ],
    )  # fmt: skip


if __name__ == "__main__":
    main()
