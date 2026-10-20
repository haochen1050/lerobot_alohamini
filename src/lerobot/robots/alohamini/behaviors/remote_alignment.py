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

"""Table alignment run from the PC through the AlohaMini host, so it can share a session with a policy.

``examples/debug/align_to_table.py`` drives the wheels and the camera directly on the robot, with the host
stopped. Here the same ``AlignToTable`` controller runs against the host instead: base commands go out as
``x.vel / y.vel / theta.vel`` actions with the arms and lift held where they were, and the forward camera image
comes from the host's observations. The host's command watchdog stops the base if the PC goes quiet, and every
pulse still ends in a stop confirmed from the host's wheel feedback.

The calibration and taught target come from the files the robot-side tools write
(``outputs/table_alignment/forward_intrinsics.json``, ``forward_extrinsics.json``, ``table_target.json``),
copied to the PC. The tag placement is read from the target file, so it always matches what was taught.
"""

import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..base import BaseLimits, BaseStopError, StopResult
from ..perception import (
    AprilTagTableEstimator,
    CameraIntrinsics,
    TableTarget,
    TagPlacement,
    load_transform,
)
from .align_to_table import AlignConfig, AlignResult, AlignToTable

logger = logging.getLogger(__name__)

BASE_KEYS = ("x.vel", "y.vel", "theta.vel")


class RemoteBase:
    """The base-controller interface ``AlignToTable`` needs, implemented through the host.

    ``robot`` needs ``get_observation()`` and ``send_action(dict)`` (an ``AlohaMiniClient``, or a
    ``ThreadSafeRobot`` around one when a policy engine shares it). Every command holds the arm joints and the
    lift at the pose captured by :meth:`capture_hold`, so only the base moves.
    """

    def __init__(
        self,
        robot: Any,
        *,
        limits: BaseLimits | None = None,
        rate_hz: float = 20.0,
        stop_timeout_s: float = 1.5,
        stop_tolerance: tuple[float, float] = (0.005, 0.5),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.robot = robot
        self.limits = limits or BaseLimits()
        self.period_s = 1.0 / rate_hz
        self.stop_timeout_s = stop_timeout_s
        # |x.vel|, |y.vel| (m/s) and |theta.vel| (deg/s) the host must report for a stop to count.
        self.stop_tolerance = stop_tolerance
        self._clock = clock
        self._sleep = sleep
        self._hold: dict[str, float] | None = None

    def capture_hold(self) -> dict[str, float]:
        """Hold the arms and lift where they are now for all following commands."""
        observation = self.robot.get_observation()
        self._hold = {key: float(value) for key, value in observation.items() if key.endswith(".pos")}
        if "lift_axis.height_mm" in observation:
            self._hold["lift_axis.height_mm"] = float(observation["lift_axis.height_mm"])
        return dict(self._hold)

    def _send(self, vx: float, vy: float, omega_degps: float) -> None:
        if self._hold is None:
            self.capture_hold()
        self.robot.send_action({**self._hold, "x.vel": vx, "y.vel": vy, "theta.vel": omega_degps})

    def _clamped(self, vx: float, vy: float, omega_degps: float) -> tuple[float, float, float]:
        if not all(math.isfinite(v) for v in (vx, vy, omega_degps)):
            raise ValueError(f"Non-finite base command {(vx, vy, omega_degps)}")
        lim = self.limits
        return (
            max(-lim.max_vx_mps, min(lim.max_vx_mps, vx)),
            max(-lim.max_vy_mps, min(lim.max_vy_mps, vy)),
            max(-lim.max_omega_degps, min(lim.max_omega_degps, omega_degps)),
        )

    def stop_base(self) -> StopResult:
        """Command zero until the host reports the base stopped twice in a row. Never raises."""
        deadline = self._clock() + self.stop_timeout_s
        attempts, settled = 0, 0
        feedback: dict[str, float | None] = dict.fromkeys(BASE_KEYS)
        errors: dict[str, str] = {}
        lin_tol, ang_tol = self.stop_tolerance
        while True:
            attempts += 1
            try:
                self._send(0.0, 0.0, 0.0)
                observation = self.robot.get_observation()
                feedback = {key: float(observation[key]) for key in BASE_KEYS}
                still = (
                    abs(feedback["x.vel"]) <= lin_tol
                    and abs(feedback["y.vel"]) <= lin_tol
                    and abs(feedback["theta.vel"]) <= ang_tol
                )
                settled = settled + 1 if still else 0
                errors = {} if still else {"base": f"host reports {feedback}"}
            except Exception as e:  # keep commanding zero whatever one round trip did
                settled = 0
                errors = {"base": repr(e)}
                logger.error("Stop attempt %d through the host failed: %r", attempts, e)
            if settled >= 2:
                return StopResult(True, attempts, {}, feedback)
            if self._clock() >= deadline:
                logger.critical(
                    "BASE STOP NOT CONFIRMED through the host - cut power to the base. %s", errors
                )
                return StopResult(False, attempts, errors, feedback)
            self._sleep(self.period_s)

    def move_base_for(
        self,
        vx: float,
        vy: float,
        omega_degps: float,
        duration_s: float,
        *,
        watch: Callable[[], bool] | None = None,
        poll_s: float = 0.05,
    ) -> StopResult:
        """Command a velocity for ``duration_s`` (re-sent at ``rate_hz``), then stop; see BaseController."""
        if not (math.isfinite(duration_s) and 0 < duration_s <= self.limits.max_pulse_s):
            raise ValueError(f"Pulse duration {duration_s}s outside (0, {self.limits.max_pulse_s}]s")
        command = self._clamped(vx, vy, omega_degps)
        start = self._clock()
        deadline = start + duration_s
        next_watch = start + poll_s
        ended = start
        try:
            while (now := self._clock()) < deadline:
                self._send(*command)
                if watch is not None and now >= next_watch:
                    next_watch = now + poll_s
                    if not watch():
                        break
                self._sleep(min(self.period_s, max(0.0, deadline - self._clock())))
        finally:
            ended = self._clock()
            result = self.stop_base()
            if not result.confirmed:
                raise BaseStopError(f"Base stop not confirmed after pulse: {result.errors}")
        return dataclasses.replace(result, elapsed_s=min(ended, deadline) - start)


class RemoteFrameSource:
    """Forward-camera frames from the host's observations, as ``(BGR image, capture time)``.

    A frame counts as new only when the host sent a freshly captured image with this observation. Its capture
    time is the PC receive time minus ``latency_s``, so frames that merely arrive after a stop are not taken
    for frames captured after it.
    """

    def __init__(
        self,
        robot: Any,
        camera: str = "forward",
        *,
        latency_s: float = 0.15,
        rgb: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.robot = robot
        self.camera = camera
        self.latency_s = latency_s
        self.rgb = rgb
        self._clock = clock
        self._last: tuple[np.ndarray, float] | None = None

    def _fresh(self) -> bool:
        client = getattr(self.robot, "inner", self.robot)
        timing = getattr(client, "latest_host_timing", None)
        if not isinstance(timing, dict):
            return True  # cannot tell; treat every observation as a new frame
        return self.camera in timing.get("camera_capture_monotonic_s", {})

    def __call__(self) -> tuple[np.ndarray, float] | None:
        observation = self.robot.get_observation()
        image = observation.get(self.camera)
        if isinstance(image, np.ndarray) and image.ndim == 3 and self._fresh():
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if self.rgb else image
            self._last = (frame, self._clock() - self.latency_s)
        return self._last


@dataclass
class TableAlignmentSetup:
    """Calibration, tag placement and taught target for one table, loaded from the robot-side tool files."""

    intrinsics: CameraIntrinsics
    T_base_cam: np.ndarray  # noqa: N815
    placement: TagPlacement
    target: TableTarget

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        intrinsics: str = "forward_intrinsics.json",
        extrinsics: str = "forward_extrinsics.json",
        target: str = "table_target.json",
    ) -> "TableAlignmentSetup":
        directory = Path(directory)
        target_path = directory / target
        record = json.loads(target_path.read_text()).get("tag_placement")
        if not record:
            raise ValueError(
                f"{target_path} has no tag_placement record; re-teach it with examples/debug/table_perception.py"
            )
        placement = TagPlacement(
            x_m=record["edge_offset_m"],
            y_m=record.get("lateral_m", 0.0),
            yaw_deg=record["yaw_deg"],
            size_m=record.get("size_m", 0.10),
            tag_id=record.get("tag_id"),
            mount=record.get("mount", "flat"),
            z_m=record.get("height_m", 0.0),
        )
        return cls(
            CameraIntrinsics.load(directory / intrinsics),
            load_transform(directory / extrinsics),
            placement,
            TableTarget.load(target_path),
        )

    def estimator(
        self, frame_source: Callable[[], tuple[np.ndarray, float] | None]
    ) -> AprilTagTableEstimator:
        return AprilTagTableEstimator(
            self.intrinsics, self.T_base_cam, self.placement, self.target, frame_source
        )


def align_base_through_host(
    robot: Any,
    setup: TableAlignmentSetup,
    config: AlignConfig | None = None,
    *,
    camera: str = "forward",
    log: Callable[[dict[str, Any]], None] | None = None,
    base: RemoteBase | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> AlignResult:
    """Align the base to the table through the host, holding the arms and lift where they are.

    Returns the AlignResult (BASE_READY or FAULT); raises BaseStopError only if a stop cannot be confirmed.
    """
    config = config or AlignConfig()
    if base is None:
        base = RemoteBase(
            robot,
            limits=BaseLimits(
                max_vx_mps=config.distance.max_speed,
                max_vy_mps=config.lateral.max_speed,
                max_omega_degps=config.heading.max_speed,
                max_pulse_s=config.max_pulse_s,
            ),
            clock=clock,
            sleep=sleep,
        )
    base.capture_hold()
    estimator = setup.estimator(RemoteFrameSource(robot, camera, clock=clock))
    return AlignToTable(base, estimator, config, log=log, clock=clock, sleep=sleep).run()


def summarize_alignment(result: AlignResult) -> dict[str, Any]:
    """The few numbers worth keeping per alignment, e.g. in an episode record."""
    summary = next((e for e in reversed(result.events) if e.get("event") == "summary"), {})

    def pose(error):
        if error is None:
            return None
        return {
            "distance_cm": round(error.distance_error_m * 100, 1),
            "lateral_cm": round(error.lateral_error_m * 100, 1),
            "heading_deg": round(error.heading_error_deg, 1),
        }

    return {
        "state": result.state.value,
        "reason": result.reason,
        "pulses": result.steps,
        "duration_s": round(result.duration_s, 1),
        "start": pose(result.start_error),
        "start_source": result.start_source,
        "final": pose(result.final_error),
        "search_deg": round(summary.get("search_rotation_deg", 0.0)),
        "radial_cm": round(summary.get("radial_m", 0.0) * 100),
        "orbit_deg": round(summary.get("orbit_deg", 0.0)),
    }
