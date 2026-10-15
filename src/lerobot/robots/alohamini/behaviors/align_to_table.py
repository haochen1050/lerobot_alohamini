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

States: [SEARCH ->] [ORBIT ->] ESTIMATE -> ALIGN_HEADING -> ALIGN_LATERAL -> ALIGN_DISTANCE -> VERIFY -> BASE_READY, with FAULT on
lost/stale perception, timeout, step/workspace limits, or an error that grows after a correction (which is
what a wrong sign looks like). Only the axes in ``AlignConfig.axes`` are corrected and verified, so the
stages can be brought up one at a time.

If the tag is not usable at the start, SEARCH rotates in place in steps smaller than the camera's field of
view (turning towards the tag once it is seen) until a valid measurement arrives, or faults after one turn.
If the robot is then far off the table's approach direction (displaced, not just turned), ORBIT circles the
approach point at constant distance, facing the table, until the remaining offset is small.

Once the tag has been fitted, the tag is kept in view (``keep_tag_in_view``): each pulse is checked against
the predicted tag position in the image and shortened, turned to keep looking at the tag, or replaced by a
correction on another axis if it would bring the tag near the image edge; the tag is tracked during the pulse,
which ends early if the tag nears the edge; and if the tag is lost after a pulse anyway, the pulse is reversed.

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

import numpy as np

from ..base import BaseController, BaseStopError
from ..perception import TablePoseError, TablePoseEstimator, measure_table_pose

logger = logging.getLogger(__name__)

AXES = ("heading", "lateral", "distance")
ERROR_FIELD = {"heading": "heading_error_deg", "lateral": "lateral_error_m", "distance": "distance_error_m"}


class AlignState(enum.Enum):
    SEARCH = "SEARCH"
    ORBIT = "ORBIT"
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
    # ORBIT: when the robot stands more than orbit_trigger_deg around the table from the taught approach line
    # (displaced, not just turned), circle the approach point (table frame origin) at constant distance until
    # within orbit_exit_deg; the per-axis stages then finish. Each step must reduce the angle.
    orbit: bool = True
    orbit_trigger_deg: float = 10.0
    orbit_exit_deg: float = 3.0
    orbit_step_deg: float = 10.0
    orbit_speed_degps: float = 10.0
    orbit_max_linear_mps: float = 0.05
    orbit_min_distance_m: float = 0.25
    # Angles beyond this look like a turned tag rather than a displaced robot: fault instead.
    max_orbit_angle_deg: float = 75.0
    max_orbit_total_deg: float = 90.0
    orbit_max_pulse_s: float = 1.0
    # Keeping the tag in view once it has been fitted (needs an estimator with predict_tag_margin and
    # last_tag_margin_px, e.g. AprilTagTableEstimator):
    # 1. Plan: a pulse is only commanded if the tag's predicted corners stay view_margin_px inside the image
    #    along the whole path, including view_overshoot x the commanded travel. Otherwise the pulse is
    #    shortened, a translation also turns to keep looking at the tag, or another axis is corrected first.
    # 2. Watch: during a pulse the tag is tracked; the pulse ends early when the tag comes within
    #    watch_margin_px of the image border or is missed in watch_misses consecutive frames.
    # 3. Recover: if the tag is lost after a pulse anyway, that pulse is reversed and the axis continues with
    #    half-length pulses, at most max_recoveries times per run. If the tag is still not visible (reversing
    #    is not exact), the base turns up to +-recover_scan_deg in recover_scan_step_deg steps to find it,
    #    starting on the side the tag was last seen.
    keep_tag_in_view: bool = True
    view_margin_px: float = 30.0
    view_overshoot: float = 1.3
    watch_margin_px: float = 10.0
    watch_misses: int = 3
    max_recoveries: int = 3
    recover_scan_deg: float = 20.0
    recover_scan_step_deg: float = 10.0

    @property
    def max_pulse_s(self) -> float:
        """Longest pulse this config commands (alignment, search, orbit or recovery scan)."""
        search = self.search_step_deg / self.search_speed_degps if self.search else 0.0
        orbit = self.orbit_max_pulse_s if self.orbit else 0.0
        scan = self.recover_scan_step_deg / self.search_speed_degps if self.keep_tag_in_view else 0.0
        return max(self.pulse_s, search, orbit, scan)

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


class _PerceptionLostError(_FaultError):
    pass


class _ViewBlockedError(Exception):
    """No variant of a planned pulse keeps the tag in view."""


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def body_displacement(vx: float, vy: float, omega_degps: float, t: float) -> tuple[float, float, float]:
    """(dx, dy, dyaw_deg) of the base after a constant body twist for t seconds, in its starting frame."""
    w = math.radians(omega_degps)
    phi = w * t
    if abs(phi) < 1e-9:
        return vx * t, vy * t, 0.0
    dx = (vx * math.sin(phi) + vy * (math.cos(phi) - 1.0)) / w
    dy = (vx * (1.0 - math.cos(phi)) + vy * math.sin(phi)) / w
    return dx, dy, math.degrees(phi)


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
        # Keeping the tag in view: recoveries used, per-axis pulse scale (halved after each recovery), and the
        # last pulse actually executed (command, seconds) so it can be reversed.
        self._recoveries = 0
        self._scale = dict.fromkeys((*AXES, "orbit"), 1.0)
        self._last_pulse: tuple[tuple[float, float, float], float] | None = None
        verified = 0
        state = AlignState.ESTIMATE
        self._event("start", config=asdict(cfg))

        try:
            if cfg.search:
                self._search()
            if cfg.orbit:
                self._orbit()
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
                    err = self._step_keeping_view(self._axis_of(state), err)
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
                raise _PerceptionLostError(f"perception lost: {self._last_reason}")
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
        self._raw_seen = False
        for looked, err in enumerate(self._fresh_frames(), start=1):
            bearing = getattr(self.estimator, "last_tag_bearing_deg", None)
            if err.valid:
                return "valid", bearing
            if bearing is not None:
                bearings.append(bearing)
            if getattr(self.estimator, "last_raw_T_base_table", None) is not None:
                self._raw_seen = True
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
                if cfg.orbit and self._raw_seen:
                    # Tag fitted fine but the robot is off to the side: the orbit approach takes over.
                    self._event("search_done", rotated_deg=rotated, bearing_deg=bearing, needs_orbit=True)
                    return
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

    def _orbit_state(self) -> tuple[float, float, np.ndarray]:
        """(angle around the approach point minus the taught one [deg], distance to the edge [m], approach
        point xy in base) from fresh frames, ignoring the heading limit."""
        target = self.estimator.target
        taught = math.degrees(math.atan2(target.lateral_m, target.distance_m))
        samples: list[tuple[float, float, np.ndarray]] = []
        for looked, _err in enumerate(self._fresh_frames(), start=1):
            raw = getattr(self.estimator, "last_raw_T_base_table", None)
            if raw is not None:
                m = measure_table_pose(raw, target.reference_x_m, target.reference_y_m)
                if m is not None:
                    angle = math.degrees(math.atan2(m.lateral_m, m.distance_m)) - taught
                    samples.append((angle, m.distance_m, raw[:2, 3].copy()))
                    if len(samples) >= self.config.samples_per_measurement:
                        break
            if looked >= 4 * self.config.samples_per_measurement and not samples:
                raise _PerceptionLostError(f"perception lost during orbit: {self._last_reason}")
        samples.sort(key=lambda sample: sample[0])
        return samples[len(samples) // 2]

    def _orbit(self) -> None:
        cfg = self.config
        if not hasattr(self.estimator, "last_raw_T_base_table") or not hasattr(self.estimator, "target"):
            self._event("orbit_skipped", reason="estimator does not expose the raw table pose")
            return
        orbited = 0.0
        angle_err, distance, pivot = self._orbit_state()
        if abs(angle_err) <= cfg.orbit_trigger_deg:
            return
        while abs(angle_err) > cfg.orbit_exit_deg:
            if abs(angle_err) > cfg.max_orbit_angle_deg:
                raise _FaultError(
                    f"robot {angle_err:+.0f} deg around the table from the approach line (limit "
                    f"{cfg.max_orbit_angle_deg:g}); if the robot is roughly in front, the tag was turned"
                )
            if distance < cfg.orbit_min_distance_m:
                raise _FaultError(f"too close to the table to orbit safely ({distance * 100:.0f} cm)")
            radius = float(np.hypot(*pivot))
            # Slow down on large radii so the base's linear speed stays bounded, and keep pulses short.
            omega_deg = min(cfg.orbit_speed_degps, math.degrees(cfg.orbit_max_linear_mps / radius))
            step = min(cfg.orbit_step_deg, omega_deg * cfg.orbit_max_pulse_s) * self._scale["orbit"]
            # Orbiting by +a (CCW) about the approach point decreases the angle error by a.
            orbit = max(-step, min(step, angle_err))
            if orbited + abs(orbit) > cfg.max_orbit_total_deg:
                raise _FaultError(f"orbit limit: {orbited:.0f} deg orbited, still {angle_err:+.0f} deg off")
            omega_deg = math.copysign(omega_deg, orbit)
            omega = math.radians(omega_deg)
            # Body twist whose instantaneous centre of rotation is the approach point (x0, y0):
            # ICR = (-vy / w, vx / w)  =>  vx = y0 * w, vy = -x0 * w.
            command = (pivot[1] * omega, -pivot[0] * omega, omega_deg)
            try:
                command, duration = self._plan_in_view(command, abs(orbit) / abs(omega_deg), look_at=False)
            except _ViewBlockedError as e:
                raise _FaultError(f"orbit would take the tag out of view: {e}") from e
            self._event(
                "orbit_pulse",
                angle_error_deg=angle_err,
                radius_m=radius,
                command=command,
                duration_s=duration,
            )
            _stop, elapsed = self._move(command, duration)
            orbit = omega_deg * elapsed
            orbited += abs(orbit)

            try:
                new_err, distance, pivot = self._orbit_state()
            except _PerceptionLostError as lost:
                angle_err, distance, pivot = self._recover(lost, self._orbit_state)
                orbited -= abs(orbit)
                self._scale["orbit"] *= 0.5
                continue
            if abs(new_err) > abs(angle_err) - 0.3 * abs(orbit):
                raise _FaultError(
                    f"orbit did not reduce the angle ({angle_err:+.1f} -> {new_err:+.1f} deg after a "
                    f"{orbit:+.1f} deg step): slipping, or the tag moved"
                )
            angle_err = new_err
        self._event("orbit_done", orbited_deg=orbited, angle_error_deg=angle_err)

    def _step_keeping_view(self, axis: str, err: TablePoseError) -> TablePoseError:
        """One correction of ``axis``; if no pulse of it keeps the tag in view, correct another axis that needs
        it instead (the current axis is retried on the next iteration). Full-length pulses on any axis are
        preferred over shortened ones, which make little progress and may not overcome wheel friction."""
        others = [a for a in AXES if a != axis and a in self.config.axes and self._needs_correction(a, err)]
        blocked = []
        for full_length_only in (True, False):
            for candidate in (axis, *others):
                try:
                    new_err = self._step(candidate, err, full_length_only=full_length_only)
                except _ViewBlockedError as e:
                    if not full_length_only:
                        blocked.append(f"{candidate}: {e}")
                        self._event("view_blocked", axis=candidate, reason=str(e))
                    continue
                if candidate != axis:
                    self._event("view_switch", axis=axis, corrected=candidate, full_length=full_length_only)
                return new_err
        raise _FaultError(
            "every correction would take the tag out of view (" + "; ".join(blocked) + "). Re-teach with the "
            "tag nearer the image centre, or start closer to the target"
        )

    def _step(self, axis: str, err: TablePoseError, *, full_length_only: bool = False) -> TablePoseError:
        cfg, ax = self.config, self.config.axis(axis)
        error = getattr(err, ERROR_FIELD[axis])
        speed = _clamp(ax.gain * error, ax.max_speed)
        if abs(speed) < ax.min_speed:
            speed = math.copysign(ax.min_speed, error)
        duration = cfg.pulse_s * self._scale[axis]

        if axis == "distance" and speed > 0:
            # Hard approach limit: forward travel may not overshoot the taught distance by more than the margin.
            allowed = err.distance_error_m + cfg.max_approach_overshoot_m
            if allowed <= 0:
                raise _FaultError(
                    f"approach limit: already {-err.distance_error_m * 100:.1f} cm inside the target"
                )
            speed = min(speed * duration, allowed) / duration

        command = {
            "distance": (speed, 0.0, 0.0),
            "lateral": (0.0, speed, 0.0),
            "heading": (0.0, 0.0, speed),
        }[axis]
        command, duration = self._plan_in_view(
            command, duration, look_at=axis != "heading", full_length_only=full_length_only
        )
        # Workspace limits on what may be commanded, before moving.
        self._check_travel(command, duration)

        self._event("pulse", axis=axis, error=error, command=command, duration_s=duration)
        stop, elapsed = self._move(command, duration)
        self._add_travel(command, elapsed)
        self._event("stopped", confirmed=stop.confirmed, goal_velocity=stop.goal_velocity, elapsed_s=elapsed)

        try:
            new_err = self._measure()
        except _PerceptionLostError as lost:
            new_err = self._recover(lost)
            self._scale[axis] *= 0.5
            return new_err
        after = getattr(new_err, ERROR_FIELD[axis])
        displacement = {"distance": command[0], "lateral": command[1], "heading": command[2]}[axis] * elapsed
        # A correct step shrinks the error; growth beyond noise means the sign or calibration is wrong.
        if abs(after) > abs(error) + max(ax.noise, 0.5 * abs(displacement)):
            raise _FaultError(
                f"{axis} error grew from {error:+.3f} to {after:+.3f} after a {displacement:+.3f} step "
                "(wrong sign or slipping?)"
            )
        return new_err

    # ------------------------------------------------------------------ keeping the tag in view

    def _keeps_view(self) -> bool:
        return self.config.keep_tag_in_view and hasattr(self.estimator, "predict_tag_margin")

    def _predicted_margin(self, command: tuple[float, float, float], duration: float) -> float | None:
        """Worst predicted tag margin (px) along the pulse's path, or None if it cannot be predicted."""
        if not self._keeps_view():
            return None
        margins = []
        for fraction in (0.25, 0.5, 0.75, 1.0, self.config.view_overshoot):
            margin = self.estimator.predict_tag_margin(*body_displacement(*command, fraction * duration))
            if margin is None:
                return None
            margins.append(margin)
        return min(margins)

    def _look_at(self, command: tuple[float, float, float], duration: float) -> tuple[float, float, float]:
        """The translation ``command`` plus the rotation that keeps the tag at the same bearing."""
        vx, vy, _ = command
        tag_x, tag_y = self.estimator.last_fit_T_base_tag[:2, 3]
        before = math.degrees(math.atan2(tag_y, tag_x))
        after = math.degrees(math.atan2(tag_y - vy * duration, tag_x - vx * duration))
        omega = _clamp(_wrap_deg(after - before) / duration, self.config.heading.max_speed)
        return vx, vy, omega

    def _plan_in_view(
        self,
        command: tuple[float, float, float],
        duration: float,
        *,
        look_at: bool,
        full_length_only: bool = False,
    ) -> tuple[tuple[float, float, float], float]:
        """``command`` / ``duration``, or a variant of it, whose path keeps the tag view_margin_px inside the
        image - or, if the tag is already closer to the edge than that, never brings it closer than it is now.
        Raises _ViewBlockedError if none does; unchanged when the view cannot be predicted."""
        margin = self._predicted_margin(command, duration)
        if margin is None:
            return command, duration
        now = self.estimator.predict_tag_margin(0.0, 0.0, 0.0)
        required = min(self.config.view_margin_px, now) - 1e-6
        if margin >= required:
            return command, duration
        commands = [command]
        if look_at:
            commands.append(self._look_at(command, duration))
        for fraction in (1.0,) if full_length_only else (1.0, 0.5, 0.25):
            for candidate in commands:
                m = self._predicted_margin(candidate, fraction * duration)
                if m is not None and m >= required:
                    self._event(
                        "view_plan",
                        original=command,
                        command=candidate,
                        duration_s=fraction * duration,
                        margin_px=m,
                        original_margin_px=margin,
                    )
                    return candidate, fraction * duration
        raise _ViewBlockedError(f"tag would come within {margin:.0f}px of the image edge")

    def _watch(self) -> Callable[[], bool]:
        """Mid-pulse check: False (end the pulse now) when the tag nears the border or keeps going missing."""
        cfg = self.config
        state = {"misses": 0, "last_ts": None}

        def watch() -> bool:
            err = self.estimator.get_table_pose_error()
            if err.timestamp_s == state["last_ts"]:
                return True
            state["last_ts"] = err.timestamp_s
            margin = getattr(self.estimator, "last_tag_margin_px", None)
            if margin is None:
                state["misses"] += 1
                if state["misses"] >= cfg.watch_misses:
                    self._event("watch_stop", reason=f"tag missed in {state['misses']} frames: {err.reason}")
                    return False
                return True
            state["misses"] = 0
            if margin < cfg.watch_margin_px:
                self._event("watch_stop", reason=f"tag {margin:.0f}px from the image edge")
                return False
            return True

        return watch

    def _move(self, command: tuple[float, float, float], duration: float, *, watch: bool = True):
        """One pulse (tracking the tag while moving when enabled). Returns (StopResult, seconds moved)."""
        if watch and self._keeps_view() and hasattr(self.estimator, "last_tag_margin_px"):
            stop = self.base.move_base_for(*command, duration, watch=self._watch())
        else:
            stop = self.base.move_base_for(*command, duration)
        self._motion_end = self._clock()
        elapsed = duration if stop.elapsed_s is None else stop.elapsed_s
        self._last_pulse = (command, elapsed)
        return stop, elapsed

    def _recover(self, lost: _PerceptionLostError, remeasure: Callable[[], Any] | None = None) -> Any:
        """Reverse the last pulse (the tag was in view before it) and measure again with ``remeasure``
        (default: _measure)."""
        if (
            not self._keeps_view()
            or self._last_pulse is None
            or self._recoveries >= self.config.max_recoveries
        ):
            raise lost
        self._recoveries += 1
        # Which side the tag was last seen on (+ = left), to scan that way first if reversing is not enough.
        tag = getattr(self.estimator, "last_fit_T_base_tag", None)
        side = 1.0 if tag is None or tag[1, 3] >= 0 else -1.0
        command, elapsed = self._last_pulse
        back = (-command[0], -command[1], -command[2])
        self._event("recover", reason=str(lost), command=back, duration_s=elapsed, count=self._recoveries)
        self._move(back, elapsed, watch=False)
        self._add_travel(back, elapsed)
        self._last_pulse = None
        remeasure = remeasure or self._measure
        try:
            return remeasure()
        except _PerceptionLostError as e:
            found = self._scan_for_tag(remeasure, side)
            if found is None:
                raise _FaultError(
                    f"tag still not visible after reversing the last pulse and turning "
                    f"+-{self.config.recover_scan_deg:g} deg: {e}"
                ) from e
            return found

    def _scan_for_tag(self, remeasure: Callable[[], Any], side: float) -> Any:
        """Turn to +-recover_scan_deg around the current heading in steps, ``side`` first, until ``remeasure``
        succeeds. Returns its result, or None if the tag was not found."""
        cfg = self.config
        step = cfg.recover_scan_step_deg
        if cfg.recover_scan_deg <= 0 or step <= 0:
            return None
        n = max(1, round(cfg.recover_scan_deg / step))
        offsets = [side * step * k for k in range(1, n + 1)] + [-side * step * k for k in range(1, n + 1)]
        current = 0.0
        for offset in offsets:
            while abs(offset - current) > 1e-9:
                turn = _clamp(offset - current, step)
                command = (0.0, 0.0, math.copysign(cfg.search_speed_degps, turn))
                duration = abs(turn) / cfg.search_speed_degps
                self._check_travel(command, duration)
                self._event("recover_turn", offset_deg=current + turn, command=command, duration_s=duration)
                _stop, elapsed = self._move(command, duration, watch=False)
                self._add_travel(command, elapsed)
                current += turn
            self._last_pulse = None  # a scan turn is not a correction to reverse
            try:
                result = remeasure()
            except _PerceptionLostError:
                continue
            self._event("recover_found", offset_deg=current)
            return result
        return None

    # ------------------------------------------------------------------ travel limits

    def _check_travel(self, command: tuple[float, float, float], duration: float) -> None:
        cfg = self.config
        x = self._travel["x"] + command[0] * duration
        y = self._travel["y"] + command[1] * duration
        theta = self._travel["theta"] + command[2] * duration
        if math.hypot(x, y) > cfg.max_total_translation_m or abs(theta) > cfg.max_total_rotation_deg:
            raise _FaultError(
                f"workspace limit: commanded travel would reach x={x:.3f} y={y:.3f} theta={theta:.1f}"
            )

    def _add_travel(self, command: tuple[float, float, float], seconds: float) -> None:
        self._travel["x"] += command[0] * seconds
        self._travel["y"] += command[1] * seconds
        self._travel["theta"] += command[2] * seconds

    def _event(self, kind: str, **data: Any) -> None:
        event = {"t": self._clock(), "event": kind, **data}
        self._events.append(event)
        if self._log is not None:
            self._log(event)
        logger.info("%s %s", kind, {k: v for k, v in data.items() if k != "config"})
