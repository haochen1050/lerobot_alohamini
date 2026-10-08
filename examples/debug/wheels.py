#!/usr/bin/env python3
"""
3-omniwheel base pulse teleop / stop test, built on lerobot.robots.alohamini.base.

Each key press sends ONE short velocity pulse followed by a verified stop. Holding a key does not queue
extra pulses. Keys are read from the terminal, so this works over SSH (no X display needed).

  W / S   forward / backward        (+vx / -vx)
  A / D   strafe left / right       (+vy / -vy)
  Q / E   rotate CCW / CW           (+omega / -omega)
  X       exit                      Ctrl+C also stops and exits

SAFETY: software stop is not an emergency stop. Run on clear floor, away from the table, with someone
ready to cut base power.

Usage:
  python examples/debug/wheels.py --port /dev/am_arm_follower_left             # pulse teleop
  python examples/debug/wheels.py --port /dev/am_arm_follower_left --pulse-test  # milestone 1 check
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import select
import signal
import sys
import termios
import time
import tty

from lerobot.robots.alohamini.base import BaseController, BaseLimits, BaseStopError, open_wheel_bus
from lerobot.robots.alohamini.model_specs import ROBOT_SPECS

# Same as AlohaMiniConfig.left_port: the wheels share the left arm bus.
DEFAULT_PORT = "/dev/am_arm_follower_left"
LIN_SPEED = 0.05  # m/s
ANG_SPEED = 15.0  # deg/s
PULSE_S = 0.3

KEY_TO_CMD = {
    "w": (LIN_SPEED, 0.0, 0.0),
    "s": (-LIN_SPEED, 0.0, 0.0),
    "a": (0.0, LIN_SPEED, 0.0),
    "d": (0.0, -LIN_SPEED, 0.0),
    "q": (0.0, 0.0, ANG_SPEED),
    "e": (0.0, 0.0, -ANG_SPEED),
}

logger = logging.getLogger("wheels")


def _raise_interrupt(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


@contextlib.contextmanager
def cbreak_terminal():
    """Unbuffered single-key input; Ctrl+C still raises KeyboardInterrupt."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(fd: int, timeout_s: float) -> str | None:
    ready, _, _ = select.select([fd], [], [], timeout_s)
    return sys.stdin.read(1).lower() if ready else None


def run_teleop(base: BaseController) -> None:
    print(__doc__.split("SAFETY")[0])
    with cbreak_terminal() as fd:
        while True:
            key = read_key(fd, 0.1)
            if key is None:
                continue
            if key == "x":
                return
            if key not in KEY_TO_CMD:
                continue
            base.move_base_for(*KEY_TO_CMD[key], PULSE_S)
            # Drop key-repeat presses that arrived during the pulse.
            termios.tcflush(fd, termios.TCIFLUSH)


def run_pulse_test(base: BaseController) -> bool:
    """Milestone 1: a 0.3 s forward pulse must end with every wheel at zero."""
    result = base.move_base_for(LIN_SPEED, 0.0, 0.0, PULSE_S)
    print(f"Goal_Velocity after stop: {result.goal_velocity} (attempts={result.attempts})")

    settled = False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        present = {n: base.bus.read("Present_Velocity", n, normalize=False) for n in base.wheel_names}
        print(f"Present_Velocity: {present}")
        if all(v == 0 for v in present.values()):
            settled = True
            break
        time.sleep(0.1)

    passed = result.confirmed and settled
    print("PULSE TEST PASSED" if passed else "PULSE TEST FAILED - wheels did not settle to zero")
    return passed


def parse_args():
    p = argparse.ArgumentParser(description="AlohaMini base pulse teleop / stop test")
    p.add_argument("--port", default=DEFAULT_PORT, help=f"Wheel serial port (default: {DEFAULT_PORT})")
    p.add_argument("--robot-model", default="alohamini2pro", choices=sorted(ROBOT_SPECS))
    p.add_argument("--pulse-test", action="store_true", help="Run one forward pulse and verify the stop")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # SSH hangup / kill should unwind through the same stop path as Ctrl+C.
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    specs = ROBOT_SPECS[args.robot_model]
    bus = open_wheel_bus(args.port, specs["base_motor"])
    base = BaseController(
        bus,
        wheel_radius=specs["wheel_radius"],
        base_radius=specs["base_radius"],
        limits=BaseLimits(
            max_vx_mps=LIN_SPEED, max_vy_mps=LIN_SPEED, max_omega_degps=ANG_SPEED, max_pulse_s=PULSE_S
        ),
    )
    exit_code = 0
    try:
        if args.pulse_test:
            exit_code = 0 if run_pulse_test(base) else 1
        else:
            run_teleop(base)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except BaseStopError:
        logger.critical("Stop could not be confirmed - CUT BASE POWER")
        exit_code = 2
    finally:
        result = base.stop_base()
        if not result.confirmed:
            logger.critical("Final stop not confirmed - CUT BASE POWER")
            exit_code = 2
        try:
            # Torque off is a second layer: the wheels can no longer be driven.
            bus.disconnect(disable_torque=True)
        except Exception:
            logger.exception("Disconnect / torque disable failed")
            exit_code = exit_code or 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
