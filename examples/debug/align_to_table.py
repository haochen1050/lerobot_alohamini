#!/usr/bin/env python3
"""
Closed-loop base alignment to the taught table pose (milestones 4-7).

Bring the axes up one at a time:
  --axes heading                      milestone 4
  --axes lateral                      milestone 5
  --axes distance                     milestone 6
  --axes heading,lateral,distance     milestone 7 (full alignment -> BASE_READY)

Prerequisites: camera intrinsics/extrinsics and a taught target (examples/debug/table_perception.py), the
host process stopped, base power on, and someone ready to cut base power. Every pulse is <= 0.3 s and ends
in a verified stop; any lost/stale tag, timeout, or error that grows after a correction stops the base.

  python examples/debug/align_to_table.py --tag-edge-offset 0.15 --axes heading --dry-run
  python examples/debug/align_to_table.py --tag-edge-offset 0.15 --axes heading

If the tag is not usable at the start, the robot first rotates in place to find it (--search-direction), and
if it stands off to the side of the table's approach line it circles the approach point at constant distance
until it is roughly in front (--no-search / --no-orbit disable these).

A JSONL log of every measurement, pulse, stop and state transition is written per trial.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from table_perception import (
    OUT,
    LatestFrame,
    add_camera_args,
    load_target,
    make_estimator,
    table_args,
    target_placement_mismatch,
)

from lerobot.robots.alohamini.base import BaseController, BaseLimits, open_wheel_bus
from lerobot.robots.alohamini.behaviors import AXES, ERROR_FIELD, AlignConfig, AlignToTable
from lerobot.robots.alohamini.model_specs import ROBOT_SPECS
from lerobot.robots.alohamini.perception import CameraIntrinsics

DEFAULT_PORT = "/dev/am_arm_follower_left"

logger = logging.getLogger("align_to_table")


def _raise_interrupt(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(p)
    table_args(p)
    p.add_argument("--port", default=DEFAULT_PORT, help="Wheel serial port")
    p.add_argument("--robot-model", default="alohamini2pro", choices=sorted(ROBOT_SPECS))
    p.add_argument("--axes", default="heading", help=f"Comma-separated subset of {','.join(AXES)}")
    p.add_argument(
        "--dry-run", action="store_true", help="Measure and print the first planned pulse; no motion"
    )
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.add_argument("--log-dir", default=str(OUT / "logs"))
    p.add_argument("--timeout-s", type=float, default=AlignConfig.timeout_s, help="Overall alignment timeout")
    p.add_argument("--no-search", action="store_true", help="Fault instead of rotating to find the tag")
    p.add_argument(
        "--no-orbit",
        action="store_true",
        help="Do not circle the table to correct a robot standing off to the side of the approach line",
    )
    p.add_argument(
        "--search-direction",
        choices=("ccw", "cw"),
        default="ccw",
        help="Rotation direction while the tag is not in view (viewed from above)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    config = AlignConfig(
        axes=axes,
        timeout_s=args.timeout_s,
        search=not args.no_search,
        orbit=not args.no_orbit,
        search_direction=1 if args.search_direction == "ccw" else -1,
    )

    if not Path(args.target).exists():
        raise SystemExit(f"No taught target at {args.target}; run table_perception.py teach first")
    if (mismatch := target_placement_mismatch(args)) is not None:
        raise SystemExit(f"Refusing to align: {mismatch}. Use the same tag options or re-teach.")
    target = load_target(args)

    intr = CameraIntrinsics.load(args.intrinsics)
    cam = LatestFrame(args.camera, *intr.image_size)
    estimator = make_estimator(args, cam, target)
    try:
        cam.wait_first()
        time.sleep(0.5)
        err = estimator.get_table_pose_error()
        if err.valid:
            print(
                f"Current error: distance {err.distance_error_m * 100:+.1f} cm, "
                f"lateral {err.lateral_error_m * 100:+.1f} cm, heading {err.heading_error_deg:+.1f} deg. "
                f"Correcting: {', '.join(axes)}"
            )
        elif config.search:
            print(
                f"Tag not usable yet ({err.reason}). Will rotate {args.search_direction.upper()} in "
                f"{config.search_step_deg:g} deg steps at {config.search_speed_degps:g} deg/s to find it "
                f"(up to {config.max_search_rotation_deg:g} deg), then align: {', '.join(axes)}"
            )
        else:
            raise SystemExit(f"Table pose invalid before starting: {err.reason}")

        if args.dry_run:
            if err.valid:
                for axis in axes:
                    ax, e = config.axis(axis), getattr(err, ERROR_FIELD[axis])
                    if abs(e) <= ax.tolerance:
                        plan = "within tolerance"
                    else:
                        plan = f"first pulse speed {max(-ax.max_speed, min(ax.max_speed, ax.gain * e)):+.3f}"
                    print(f"  {axis}: error {e:+.3f}, tolerance {ax.tolerance}, {plan}")
            print("Dry run: no motion.")
            return 0

        if not args.yes:
            input(
                "Base will move in short pulses. Hand on the power cutoff? Press Enter to start (Ctrl+C aborts). "
            )

        specs = ROBOT_SPECS[args.robot_model]
        bus = open_wheel_bus(args.port, specs["base_motor"])
        base = BaseController(
            bus,
            wheel_radius=specs["wheel_radius"],
            base_radius=specs["base_radius"],
            limits=BaseLimits(
                max_vx_mps=config.distance.max_speed,
                max_vy_mps=config.lateral.max_speed,
                max_omega_degps=config.heading.max_speed,
                max_pulse_s=config.max_pulse_s,
            ),
        )

        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"align_{time.strftime('%Y%m%d_%H%M%S')}_{'-'.join(axes)}.jsonl"
        exit_code = 1
        with log_path.open("w") as log_file:

            def log(event: dict) -> None:
                log_file.write(json.dumps(event, default=str) + "\n")
                log_file.flush()

            log({"event": "trial", "axes": axes, "target": asdict(target), "tag": vars(args)})
            try:
                result = AlignToTable(base, estimator, config, log=log).run()
                e = result.final_error
                print(f"\n{result.state.value} after {result.steps} pulses. {result.reason}")
                if e is not None:
                    print(
                        f"Final error: distance {e.distance_error_m * 100:+.1f} cm, "
                        f"lateral {e.lateral_error_m * 100:+.1f} cm, heading {e.heading_error_deg:+.1f} deg"
                    )
                exit_code = 0 if result.ready else 1
            except KeyboardInterrupt:
                logger.info("Interrupted")
            finally:
                stop = base.stop_base()
                if not stop.confirmed:
                    logger.critical("Final stop not confirmed - CUT BASE POWER")
                    exit_code = 2
                try:
                    bus.disconnect(disable_torque=True)
                except Exception:
                    logger.exception("Disconnect / torque disable failed")
        print(f"Log: {log_path}")
        return exit_code
    finally:
        cam.close()


if __name__ == "__main__":
    sys.exit(main())
