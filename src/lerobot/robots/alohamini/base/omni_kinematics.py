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

"""Three-omniwheel (120 degree) base kinematics.

Pure-function form of ``AlohaMini._body_to_wheel_raw`` / ``AlohaMini._wheel_raw_to_body`` so the base can be
driven without instantiating the full robot. The math is kept identical (a parity test guards this).

Body frame: ``x`` forward (m/s), ``y`` left (m/s), ``theta`` counter-clockwise yaw rate (deg/s). Wheel order
is always ``(left, back, right)``. These sign conventions must be verified physically (milestone 2).
"""

import numpy as np

STEPS_PER_DEG = 4096.0 / 360.0
DEFAULT_MAX_RAW = 3000

# Wheel mounting angles with the -90 degree offset used by AlohaMini.
_WHEEL_ANGLES_RAD = np.radians(np.array([240, 0, 120]) - 90)


def _kinematic_matrix(base_radius: float) -> np.ndarray:
    return np.array([[np.cos(a), np.sin(a), base_radius] for a in _WHEEL_ANGLES_RAD])


def degps_to_raw(degps: float) -> int:
    """Wheel angular speed (deg/s) -> signed raw Goal_Velocity, clamped to int16."""
    speed_int = int(round(degps * STEPS_PER_DEG))
    return max(-0x8000, min(0x7FFF, speed_int))


def raw_to_degps(raw_speed: int) -> float:
    return raw_speed / STEPS_PER_DEG


def body_to_wheel_raw(
    x_cmd: float,
    y_cmd: float,
    theta_cmd_degps: float,
    *,
    wheel_radius: float,
    base_radius: float,
    max_raw: int = DEFAULT_MAX_RAW,
) -> tuple[int, int, int]:
    """Body velocity -> raw wheel commands ``(left, back, right)``.

    If any wheel would exceed ``max_raw``, all wheels are scaled down proportionally so the direction of
    motion is preserved.
    """
    velocity = np.array([-x_cmd, -y_cmd, np.radians(theta_cmd_degps)])
    wheel_degps = np.degrees(_kinematic_matrix(base_radius).dot(velocity) / wheel_radius)

    peak_raw = float(np.max(np.abs(wheel_degps))) * STEPS_PER_DEG
    if peak_raw > max_raw:
        wheel_degps = wheel_degps * (max_raw / peak_raw)

    left, back, right = (degps_to_raw(v) for v in wheel_degps)
    return left, back, right


def wheel_raw_to_body(
    left: int,
    back: int,
    right: int,
    *,
    wheel_radius: float,
    base_radius: float,
) -> tuple[float, float, float]:
    """Raw wheel speeds ``(left, back, right)`` -> body velocity ``(x m/s, y m/s, theta deg/s)``."""
    wheel_linear = np.radians([raw_to_degps(r) for r in (left, back, right)]) * wheel_radius
    x, y, theta_rad = np.linalg.inv(_kinematic_matrix(base_radius)).dot(wheel_linear)
    return float(-x), float(-y), float(np.degrees(theta_rad))
