#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safety-first velocity interface for the three-omniwheel base.

Software stopping is NOT an emergency stop: if the host crashes, the SSH session drops, or the motor bus
fails, the wheels keep their last Goal_Velocity. Keep a physical power cutoff within reach whenever the
base is powered.
"""

import contextlib
import logging
import math
import signal
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .omni_kinematics import DEFAULT_MAX_RAW, body_to_wheel_raw

logger = logging.getLogger(__name__)

WHEEL_IDS: tuple[int, int, int] = (8, 9, 10)
WHEEL_NAMES: tuple[str, str, str] = ("base_left_wheel", "base_back_wheel", "base_right_wheel")

_DEFERRED_SIGNALS = tuple(
    sig for sig in (getattr(signal, n, None) for n in ("SIGINT", "SIGTERM", "SIGHUP")) if sig is not None
)


class BaseStopError(RuntimeError):
    """Zero velocity could not be confirmed on every wheel. Cut power to the base."""


class WheelBusMismatchError(RuntimeError):
    """The serial device does not look like the wheel bus."""


@dataclass(frozen=True)
class BaseLimits:
    """Hard caps applied to every command, independent of any controller above this layer."""

    max_vx_mps: float = 0.10
    max_vy_mps: float = 0.10
    max_omega_degps: float = 30.0
    max_pulse_s: float = 1.0


@dataclass(frozen=True)
class StopResult:
    confirmed: bool
    attempts: int
    # Last error seen for each wheel that never confirmed zero.
    errors: dict[str, str] = field(default_factory=dict)
    # Goal_Velocity read back from each wheel (None when the read failed).
    goal_velocity: dict[str, int | None] = field(default_factory=dict)


@contextlib.contextmanager
def _defer_interrupts() -> Iterator[None]:
    """Hold SIGINT/SIGTERM/SIGHUP until the block finishes, then re-deliver them.

    A Ctrl+C landing between two wheel writes would otherwise abort the stop half-way.
    Signal handlers can only be swapped from the main thread; elsewhere this is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    received: list[int] = []
    previous = {
        sig: signal.signal(sig, lambda signum, _frame: received.append(signum)) for sig in _DEFERRED_SIGNALS
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            # None means the handler was installed outside Python; the default is the closest match.
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        for sig in dict.fromkeys(received):
            signal.raise_signal(sig)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class BaseController:
    """Drives the base through per-wheel ``Goal_Velocity`` writes with verified stops.

    ``bus`` must already be connected with the wheels in velocity mode (see :func:`open_wheel_bus`).
    ``wheel_names`` is ordered ``(left, back, right)``.
    """

    def __init__(
        self,
        bus: Any,
        *,
        wheel_radius: float,
        base_radius: float,
        wheel_names: Sequence[str] = WHEEL_NAMES,
        max_raw: int = DEFAULT_MAX_RAW,
        limits: BaseLimits | None = None,
        stop_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if len(wheel_names) != 3:
            raise ValueError(f"Expected 3 wheel names (left, back, right), got {wheel_names!r}")
        if stop_attempts < 1:
            raise ValueError("stop_attempts must be >= 1")
        self.bus = bus
        self.wheel_names = tuple(wheel_names)
        self.wheel_radius = wheel_radius
        self.base_radius = base_radius
        self.max_raw = max_raw
        self.limits = limits or BaseLimits()
        self.stop_attempts = stop_attempts
        self._sleep = sleep

    def set_base_velocity(self, vx: float, vy: float, omega_degps: float) -> dict[str, int]:
        """Clamp and send a body velocity. Returns the per-wheel raw commands that were written.

        If any wheel write fails the base is stopped and the error is re-raised.
        """
        if not all(math.isfinite(v) for v in (vx, vy, omega_degps)):
            self.stop_base()
            raise ValueError(f"Non-finite base command vx={vx} vy={vy} omega={omega_degps}")

        cmd = (
            _clamp(vx, self.limits.max_vx_mps),
            _clamp(vy, self.limits.max_vy_mps),
            _clamp(omega_degps, self.limits.max_omega_degps),
        )
        if cmd != (vx, vy, omega_degps):
            logger.warning(
                "Base command clamped: (%.3f, %.3f, %.1f) -> (%.3f, %.3f, %.1f)", vx, vy, omega_degps, *cmd
            )
        raw = body_to_wheel_raw(
            *cmd, wheel_radius=self.wheel_radius, base_radius=self.base_radius, max_raw=self.max_raw
        )
        wheel_cmds = dict(zip(self.wheel_names, raw, strict=True))
        logger.info("Base command vx=%.3f vy=%.3f omega=%.1f -> wheels %s", *cmd, wheel_cmds)

        try:
            for name, value in wheel_cmds.items():
                self.bus.write("Goal_Velocity", name, value, normalize=False)
        except Exception:
            logger.exception("Wheel velocity write failed; stopping base")
            self.stop_base()
            raise
        return wheel_cmds

    def stop_base(self) -> StopResult:
        """Write zero to every wheel individually and read it back, retrying unconfirmed wheels.

        A failure on one wheel never prevents the others from being stopped. Never raises; check
        ``StopResult.confirmed``.
        """
        pending = list(self.wheel_names)
        errors: dict[str, str] = {}
        readback: dict[str, int | None] = {}
        attempt = 0
        with _defer_interrupts():
            while pending and attempt < self.stop_attempts:
                attempt += 1
                for name in pending:
                    try:
                        self.bus.write("Goal_Velocity", name, 0, normalize=False)
                    except Exception as e:
                        errors[name] = f"write: {e!r}"
                        logger.error("Stop attempt %d: zero write to %s failed: %r", attempt, name, e)

                for name in pending:
                    try:
                        readback[name] = int(self.bus.read("Goal_Velocity", name, normalize=False))
                    except Exception as e:
                        readback[name] = None
                        errors[name] = f"read: {e!r}"
                        logger.error(
                            "Stop attempt %d: Goal_Velocity readback on %s failed: %r", attempt, name, e
                        )
                        continue
                    if readback[name] != 0:
                        errors[name] = f"readback {readback[name]} != 0"
                        logger.error(
                            "Stop attempt %d: %s still has Goal_Velocity=%s", attempt, name, readback[name]
                        )

                pending = [n for n in pending if readback.get(n) != 0]

        unconfirmed = {n: errors[n] for n in pending}
        result = StopResult(not pending, attempt, unconfirmed, readback)
        if result.confirmed:
            logger.info("Base stop confirmed on all wheels (attempts=%d)", attempt)
        else:
            logger.critical(
                "BASE STOP NOT CONFIRMED for %s - cut power to the base. %s", pending, unconfirmed
            )
        return result

    def move_base_for(self, vx: float, vy: float, omega_degps: float, duration_s: float) -> StopResult:
        """Run a bounded velocity pulse, then stop. The stop runs even on exceptions and Ctrl+C.

        Raises:
            ValueError: duration is not in (0, limits.max_pulse_s]. No motion is commanded.
            BaseStopError: the closing stop could not be confirmed.
        """
        if not (math.isfinite(duration_s) and 0 < duration_s <= self.limits.max_pulse_s):
            raise ValueError(f"Pulse duration {duration_s}s outside (0, {self.limits.max_pulse_s}]s")

        try:
            self.set_base_velocity(vx, vy, omega_degps)
            self._sleep(duration_s)
        finally:
            result = self.stop_base()
            if not result.confirmed:
                raise BaseStopError(f"Base stop not confirmed after pulse: {result.errors}")
        return result

    def __enter__(self) -> "BaseController":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        result = self.stop_base()
        if not result.confirmed and exc is None:
            raise BaseStopError(f"Base stop not confirmed on exit: {result.errors}")


def open_wheel_bus(
    port: str,
    motor_model: str,
    *,
    wheel_ids: Sequence[int] = WHEEL_IDS,
    wheel_names: Sequence[str] = WHEEL_NAMES,
) -> Any:
    """Connect to ``port``, verify it hosts the expected wheel motors, and put them in velocity mode.

    Only the wheel IDs are touched. The bus is left with torque on and Goal_Velocity=0 on every wheel.

    Raises:
        WheelBusMismatchError: a wheel ID does not answer, or answers with the wrong model number.
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
    from lerobot.motors.feetech.tables import MODEL_NUMBER_TABLE

    motors = {
        name: Motor(id=id_, model=motor_model, norm_mode=MotorNormMode.RANGE_M100_100)
        for name, id_ in zip(wheel_names, wheel_ids, strict=True)
    }
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    try:
        expected = MODEL_NUMBER_TABLE[motor_model]
        found = {name: bus.ping(name, num_retry=2) for name in motors}
        bad = {name: model for name, model in found.items() if model != expected}
        if bad:
            hint = (
                " No wheel answered: check that base motor power is switched on."
                if all(model is None for model in found.values())
                else ""
            )
            raise WheelBusMismatchError(
                f"{port} is not the wheel bus: expected model {expected} ({motor_model}) on IDs "
                f"{list(wheel_ids)}, got {found}.{hint}"
            )

        for name in motors:
            try:
                bus.write("Lock", name, 0, normalize=False)
            except Exception as e:
                logger.warning("Unlock of %s failed (continuing): %r", name, e)
            bus.disable_torque(name)
            bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value, normalize=False)
            # Clear any stale goal before torque comes back on.
            bus.write("Goal_Velocity", name, 0, normalize=False)
            bus.enable_torque(name)
    except BaseException:
        bus.disconnect(disable_torque=False)
        raise
    logger.info("Wheel bus %s verified (%s x3, IDs %s), velocity mode", port, motor_model, list(wheel_ids))
    return bus
