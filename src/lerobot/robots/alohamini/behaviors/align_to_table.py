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

"""Closed-loop base alignment to a table: observe, short bounded pulse, stop, observe again.

States: ESTIMATE -> ALIGN_HEADING -> ALIGN_LATERAL -> ALIGN_DISTANCE -> VERIFY -> BASE_READY, with FAULT on
lost/stale perception, timeout, step/workspace limits, or an error that grows after a correction (which is
what a wrong sign looks like). Only the axes in ``AlignConfig.axes`` are corrected and verified, so the
stages can be brought up one at a time.

Safety limits here are independent of the gains: total commanded travel and rotation are capped, and the
robot is never commanded closer to the table than the taught distance minus ``max_approach_overshoot_m``.
"""

import enum
import logging
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..base import BaseController, BaseStopError
from ..perception import TablePoseError, TablePoseEstimator

logger = logging.getLogger(__name__)

AXES = ("heading", "lateral", "distance")
ERROR_FIELD = {"heading": "heading_error_deg", "lateral": "lateral_error_m", "distance": "distance_error_m"}


class AlignState(enum.Enum):
    ESTIMATE = "ESTIMATE"
    ALIGN_HEADING = "ALIGN_HEADING"
    ALIGN_LATERAL = "ALIGN_LATERAL"
    ALIGN_DISTANCE = "ALIGN_DISTANCE"
    VERIFY = "VERIFY"
    BASE_READY = "BASE_READY"
    FAULT = "FAULT"


_ALIGN_STATE = {
    "heading": AlignState.ALIGN_HEADING,
    "lateral": AlignState.ALIGN_LATERAL,
    "distance": AlignState.ALIGN_DISTANCE,
}


@dataclass(frozen=True)
class AxisConfig:
    tolerance: float
    # Commanded speed per unit of error (1/s); combined with pulse_s this sets the fraction of the error
    # corrected per step (gain * pulse_s ~= 0.6).
    gain: float
    max_speed: float
    # Smallest speed that reliably moves the wheels; avoids pulses too small to overcome friction.
    min_speed: float
    # Error growth below this is treated as measurement noise by the divergence check.
    noise: float


@dataclass(frozen=True)
class AlignConfig:
    axes: tuple[str, ...] = AXES
    heading: AxisConfig = AxisConfig(tolerance=2.0, gain=2.0, max_speed=15.0, min_speed=3.0, noise=0.5)
    lateral: AxisConfig = AxisConfig(tolerance=0.02, gain=2.0, max_speed=0.05, min_speed=0.01, noise=0.005)
    distance: AxisConfig = AxisConfig(tolerance=0.02, gain=2.0, max_speed=0.05, min_speed=0.01, noise=0.005)
    pulse_s: float = 0.3
    # Wait after each stop before trusting a frame (wheels settle, camera exposure catches up).
    settle_s: float = 0.4
    samples_per_measurement: int = 3
    measurement_timeout_s: float = 2.0
    verify_consecutive: int = 3
    max_steps: int = 40
    timeout_s: float = 120.0
    max_total_translation_m: float = 0.40
    max_total_rotation_deg: float = 60.0
    # Never command the reference point closer to the edge than (taught distance - this).
    max_approach_overshoot_m: float = 0.03
    # Each stage corrects down to this fraction of its tolerance; VERIFY accepts the full tolerance. The
    # margin keeps small disturbances from later stages from pushing a finished axis back out.
    align_fraction: float = 0.5

    def __post_init__(self) -> None:
        unknown = set(self.axes) - set(AXES)
        if unknown or not self.axes:
            raise ValueError(f"axes must be a non-empty subset of {AXES}, got {self.axes}")

    def axis(self, name: str) -> AxisConfig:
        return getattr(self, name)


@dataclass
class AlignResult:
    state: AlignState
    reason: str = ""
    steps: int = 0
    final_error: TablePoseError | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.state is AlignState.BASE_READY


class _FaultError(Exception):
    pass


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class AlignToTable:
    def __init__(
        self,
        base: BaseController,
        estimator: TablePoseEstimator,
        config: AlignConfig | None = None,
        *,
        log: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base = base
        self.estimator = estimator
        self.config = config or AlignConfig()
        self._log = log
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------ main loop

    def run(self) -> AlignResult:
        cfg = self.config
        result = AlignResult(AlignState.ESTIMATE)
        self._events = result.events
        self._travel = {"x": 0.0, "y": 0.0, "theta": 0.0}
        start = self._clock()
        # Frames captured before this time may show the robot moving.
        self._motion_end = start
        verified = 0
        state = AlignState.ESTIMATE
        self._event("start", config=asdict(cfg))

        try:
            err = self._measure()
            while True:
                if self._clock() - start > cfg.timeout_s:
                    raise _FaultError(f"timeout after {cfg.timeout_s:.0f}s")
                result.final_error = err
                out = self._out_of_tolerance(err)

                if state is AlignState.VERIFY:
                    if not out:
                        verified += 1
                        self._event("verify", count=verified, error=asdict(err))
                        if verified >= cfg.verify_consecutive:
                            break
                        err = self._measure()
                        continue
                    verified = 0
                elif state is not AlignState.ESTIMATE and self._needs_correction(self._axis_of(state), err):
                    if result.steps >= cfg.max_steps:
                        raise _FaultError(f"no convergence after {cfg.max_steps} steps")
                    result.steps += 1
                    # The post-step measurement is the next observation.
                    err = self._step(self._axis_of(state), err)
                    continue

                new_state = _ALIGN_STATE[out[0]] if out else AlignState.VERIFY
                if new_state is not state:
                    self._event("transition", src=state.value, dst=new_state.value, error=asdict(err))
                    state = new_state
                    verified = 0

            stop = self.base.stop_base()
            if not stop.confirmed:
                raise _FaultError(f"final stop not confirmed: {stop.errors}")
            result.state = AlignState.BASE_READY
            self._event("ready", steps=result.steps, error=asdict(result.final_error))
        except _FaultError as fault:
            self.base.stop_base()
            result.state, result.reason = AlignState.FAULT, str(fault)
            self._event("fault", reason=str(fault), steps=result.steps)
            logger.error("Alignment FAULT: %s", fault)
        except BaseStopError as e:
            result.state, result.reason = AlignState.FAULT, f"stop not confirmed: {e}"
            self._event("fault", reason=result.reason, steps=result.steps)
            raise
        except BaseException:
            self.base.stop_base()
            self._event("aborted", steps=result.steps)
            raise
        return result

    # ------------------------------------------------------------------ helpers

    def _axis_of(self, state: AlignState) -> str:
        return next(axis for axis, s in _ALIGN_STATE.items() if s is state)

    def _needs_correction(self, axis: str, err: TablePoseError) -> bool:
        band = self.config.axis(axis).tolerance * self.config.align_fraction
        return abs(getattr(err, ERROR_FIELD[axis])) > band

    def _out_of_tolerance(self, err: TablePoseError) -> list[str]:
        """Enabled axes outside tolerance, in correction order (heading, lateral, distance)."""
        return [
            axis
            for axis in AXES
            if axis in self.config.axes
            and abs(getattr(err, ERROR_FIELD[axis])) > self.config.axis(axis).tolerance
        ]

    def _measure(self) -> TablePoseError:
        """Median of fresh valid samples captured after the robot settled."""
        cfg = self.config
        not_before = self._motion_end + cfg.settle_s
        wait = not_before - self._clock()
        if wait > 0:
            self._sleep(wait)

        deadline = self._clock() + cfg.measurement_timeout_s
        samples: list[TablePoseError] = []
        last_reason, last_ts = "no frame", None
        while len(samples) < cfg.samples_per_measurement:
            if self._clock() > deadline:
                raise _FaultError(f"perception lost: {last_reason}")
            err = self.estimator.get_table_pose_error()
            if err.timestamp_s < not_before or err.timestamp_s == last_ts:
                last_reason = "no fresh frame since last motion"
                self._sleep(0.02)
                continue
            last_ts = err.timestamp_s
            if not err.valid:
                last_reason = err.reason
                self._sleep(0.02)
                continue
            samples.append(err)

        merged = TablePoseError(
            distance_error_m=statistics.median(s.distance_error_m for s in samples),
            lateral_error_m=statistics.median(s.lateral_error_m for s in samples),
            heading_error_deg=statistics.median(s.heading_error_deg for s in samples),
            timestamp_s=samples[-1].timestamp_s,
            valid=True,
        )
        self._event("measure", error=asdict(merged))
        return merged

    def _step(self, axis: str, err: TablePoseError) -> TablePoseError:
        cfg, ax = self.config, self.config.axis(axis)
        error = getattr(err, ERROR_FIELD[axis])
        speed = _clamp(ax.gain * error, ax.max_speed)
        if abs(speed) < ax.min_speed:
            speed = math.copysign(ax.min_speed, error)
        displacement = speed * cfg.pulse_s

        if axis == "distance" and displacement > 0:
            # Hard approach limit: forward travel may not overshoot the taught distance by more than the margin.
            allowed = err.distance_error_m + cfg.max_approach_overshoot_m
            if allowed <= 0:
                raise _FaultError(
                    f"approach limit: already {-err.distance_error_m * 100:.1f} cm inside the target"
                )
            displacement = min(displacement, allowed)
            speed = displacement / cfg.pulse_s

        key = {"distance": "x", "lateral": "y", "heading": "theta"}[axis]
        self._travel[key] += displacement
        translation = math.hypot(self._travel["x"], self._travel["y"])
        if (
            translation > cfg.max_total_translation_m
            or abs(self._travel["theta"]) > cfg.max_total_rotation_deg
        ):
            raise _FaultError(f"workspace limit: commanded travel {self._travel}")

        command = {
            "distance": (speed, 0.0, 0.0),
            "lateral": (0.0, speed, 0.0),
            "heading": (0.0, 0.0, speed),
        }[axis]
        self._event("pulse", axis=axis, error=error, command=command, duration_s=cfg.pulse_s)
        stop = self.base.move_base_for(*command, cfg.pulse_s)
        self._motion_end = self._clock()
        self._event("stopped", confirmed=stop.confirmed, goal_velocity=stop.goal_velocity)

        new_err = self._measure()
        after = getattr(new_err, ERROR_FIELD[axis])
        # A correct step shrinks the error; growth beyond noise means the sign or calibration is wrong.
        if abs(after) > abs(error) + max(ax.noise, 0.5 * abs(displacement)):
            raise _FaultError(
                f"{axis} error grew from {error:+.3f} to {after:+.3f} after a {displacement:+.3f} step "
                "(wrong sign or slipping?)"
            )
        return new_err

    def _event(self, kind: str, **data: Any) -> None:
        event = {"t": self._clock(), "event": kind, **data}
        self._events.append(event)
        if self._log is not None:
            self._log(event)
        logger.info("%s %s", kind, {k: v for k, v in data.items() if k != "config"})
