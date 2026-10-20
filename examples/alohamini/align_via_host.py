#!/usr/bin/env python3
"""
Align the base to the table from the PC, through the running AlohaMini host (the path rsi_evolve.py uses).

Unlike examples/debug/align_to_table.py, which runs on the robot with the host stopped, this leaves the host
running: base commands go through it with the arms and lift held still, and the forward camera image comes
from its observations. Try it on its own before an RSI session.

--align_dir holds the robot's calibration and taught target, copied from the Pi:
    scp pi5@<robot>:~/lerobot_alohamini/outputs/table_alignment/{forward_intrinsics,forward_extrinsics,table_target}.json \\
        outputs/table_alignment/

    python examples/alohamini/align_via_host.py --robot.remote_ip 192.168.1.50 --robot.robot_model alohamini2pro \\
        --align_dir outputs/table_alignment --table_radius 0.3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client robot type
from lerobot.robots.alohamini import AlohaMiniClient, AlohaMiniClientConfig
from lerobot.robots.alohamini.behaviors import (
    AlignConfig,
    TableAlignmentSetup,
    align_base_through_host,
    summarize_alignment,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot.remote_ip", dest="remote_ip", default="127.0.0.1")
    p.add_argument("--robot.id", dest="robot_id", default="my_alohamini")
    p.add_argument(
        "--robot.robot_model",
        dest="robot_model",
        default="alohamini2pro",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
    )
    p.add_argument("--align_dir", type=Path, default=Path("outputs/table_alignment"))
    p.add_argument("--table_radius", type=float, default=None, help="Round table radius (m)")
    p.add_argument("--camera", default="forward")
    p.add_argument("--note", default="", help="Free text saved with the trial")
    p.add_argument("--log_dir", type=Path, default=Path("outputs/table_alignment/host_logs"))
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    setup = TableAlignmentSetup.load(args.align_dir)
    print(f"Target {setup.target}, tag {setup.placement}")
    robot = AlohaMiniClient(
        AlohaMiniClientConfig(remote_ip=args.remote_ip, id=args.robot_id, robot_model=args.robot_model)
    )
    robot.connect()
    try:
        if not args.yes:
            input(
                "The base will move in short pulses; arms and lift hold still. Press Enter (Ctrl+C aborts). "
            )
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"align_{time.strftime('%Y%m%d_%H%M%S')}_host.jsonl"
        with log_path.open("w") as file:

            def log(event: dict) -> None:
                file.write(json.dumps(event, default=str) + "\n")
                file.flush()

            log(
                {
                    "event": "trial",
                    "axes": ["heading", "lateral", "distance"],
                    "note": args.note,
                    "via": "host",
                }
            )
            result = align_base_through_host(
                robot, setup, AlignConfig(table_radius_m=args.table_radius), camera=args.camera, log=log
            )
        print(json.dumps(summarize_alignment(result), indent=2))
        print(f"Log: {log_path} (examples/debug/align_report.py {args.log_dir} tabulates these)")
        return 0 if result.ready else 1
    finally:
        robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
