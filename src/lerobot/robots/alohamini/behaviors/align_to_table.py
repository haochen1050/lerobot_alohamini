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

States: [SEARCH ->] ESTIMATE -> ALIGN_HEADING -> ALIGN_LATERAL -> ALIGN_DISTANCE -> VERIFY -> BASE_READY, with FAULT on
lost/stale perception, timeout, step/workspace limits, or an error that grows after a correction (which is
what a wrong sign looks like). Only the axes in ``AlignConfig.axes`` are corrected and verified, so the
stages can be brought up one at a time.

If the tag is not usable at the start, SEARCH rotates in place in steps smaller than the camera's field of
view (turning towards the tag once it is seen) until a valid measurement arrives, or faults after one turn.

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
    SEARCH = "SEARCH"
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
    # SEARCH: when the tag is not usable at the start, rotate in place in steps (smaller than the camera's
    # field of view) until a valid measurement is obtained, or fault after max_search_rotation_deg.
    search: bool = True
    # +1 = counter-clockwise (left), -1 = clockwise, used while the tag is not in view at all.
    search_direction: int = 1
    search_speed_degps: float = 15.0
    search_step_deg: float = 15.0
    max_search_rotation_deg: float = 360.0
    # A tag seen within this bearing of straight ahead but still unusable is a fault, not something to turn to.
    search_center_deg: float = 10.0
    search_frames_per_look: int = 5

    @property
    def max_pulse_s(self) -> float:
        """Longest pulse this config commands (alignment pulses or search steps)."""
        return max(self.pulse_s, self.search_step_deg / self.search_speed_degps if self.search else 0.0)

    def __post_init__(self) -> None:
        unknown = set(self.axes) - set(AXES)
        if unknown or not self.axes:
            raise ValueError(f"axes must be a non-empty subset of {AXES}, got {self.axes}")
        if self.search_direction not in (1, -1):
            raise ValueError("search_direction must be +1 (CCW) or -1 (CW)")

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
        # Frames captured before this time may show the robot moving.
        self._motion_end = self._clock()
        verified = 0
        state = AlignState.ESTIMATE
        self._event("start", config=asdict(cfg))

        try:
            if cfg.search:
                self._search()
            # Alignment timeout and travel limits apply from here; the search has its own rotation budget.
            start = self._clock()
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

    def _fresh_frames(self):
        """Yield estimator results for frames captured after the robot settled, until the caller stops.

        Raises a fault if no fresh frame arrives within ``measurement_timeout_s``; ``self._last_reason``
        holds why the most recent frame was unusable.
        """
        cfg = self.config
        not_before = self._motion_end + cfg.settle_s
        wait = not_before - self._clock()
        if wait > 0:
            self._sleep(wait)

        deadline = self._clock() + cfg.measurement_timeout_s
        self._last_reason, last_ts = "no frame", None
        while True:
            if self._clock() > deadline:
                raise _FaultError(f"perception lost: {self._last_reason}")
            err = self.estimator.get_table_pose_error()
            if err.timestamp_s < not_before or err.timestamp_s == last_ts:
                self._last_reason = "no fresh frame since last motion"
                self._sleep(0.02)
                continue
            last_ts = err.timestamp_s
            if not err.valid:
                self._last_reason = err.reason
            yield err
            if not err.valid:
                self._sleep(0.02)

    def _measure(self) -> TablePoseError:
        """Median of fresh valid samples captured after the robot settled."""
        samples: list[TablePoseError] = []
        for err in self._fresh_frames():
            if err.valid:
                samples.append(err)
                if len(samples) >= self.config.samples_per_measurement:
                    break

        merged = TablePoseError(
            distance_error_m=statistics.median(s.distance_error_m for s in samples),
            lateral_error_m=statistics.median(s.lateral_error_m for s in samples),
            heading_error_deg=statistics.median(s.heading_error_deg for s in samples),
            timestamp_s=samples[-1].timestamp_s,
            valid=True,
        )
        self._event("measure", error=asdict(merged))
        return merged

    def _look(self) -> tuple[str, float | None]:
        """('valid' | 'seen' | 'none', tag bearing) over a few fresh frames at the current heading."""
        bearings: list[float] = []
        for looked, err in enumerate(self._fresh_frames(), start=1):
            bearing = getattr(self.estimator, "last_tag_bearing_deg", None)
            if err.valid:
                return "valid", bearing
            if bearing is not None:
                bearings.append(bearing)
            if looked >= self.config.search_frames_per_look:
                break
        if bearings:
            return "seen", statistics.median(bearings)
        return "none", None

    def _search(self) -> None:
        cfg = self.config
        if not hasattr(self.estimator, "last_tag_bearing_deg"):
            self._event("search_skipped", reason="estimator does not report tag bearing")
            return
        rotated = 0.0
        while True:
            status, bearing = self._look()
            if status == "valid":
                self._event("search_done", rotated_deg=rotated, bearing_deg=bearing)
                return
            if status == "seen":
                if abs(bearing) <= cfg.search_center_deg:
                    raise _FaultError(
                        f"tag found ahead (bearing {bearing:+.0f} deg) but unusable: {self._last_reason}"
                    )
                # Turn towards the tag, at most one step.
                angle = max(-cfg.search_step_deg, min(cfg.search_step_deg, bearing))
            else:
                angle = cfg.search_direction * cfg.search_step_deg
            if rotated + abs(angle) > cfg.max_search_rotation_deg:
                raise _FaultError(f"tag not found after rotating {rotated:.0f} deg: {self._last_reason}")

            speed = math.copysign(cfg.search_speed_degps, angle)
            duration = abs(angle) / cfg.search_speed_degps
            self._event(
                "search_pulse", status=status, bearing_deg=bearing, omega_degps=speed, duration_s=duration
            )
            self.base.move_base_for(0.0, 0.0, speed, duration)
            self._motion_end = self._clock()
            rotated += abs(angle)

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
