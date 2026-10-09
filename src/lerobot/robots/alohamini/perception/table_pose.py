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

# ruff: noqa: N802, N803, N806  (T_a_b / K / R follow standard transform notation)

"""Table-relative pose error, independent of how the table is perceived.

Frames
------
Base ``B``: origin at the base center on the floor, x forward, y left, z up (verified on hardware: W=+x,
A=+y, Q=+yaw).

Table ``T``: origin on the table's front edge (the edge facing the robot) at the centre of the intended
cup-working region; x points from the edge into the table, y along the edge to the left as seen by the
robot, z up.

Sign convention: every error is defined so that a POSITIVE error is corrected by a POSITIVE body command
on the matching axis.

- ``distance_error_m > 0``: the reference point is further from the edge than ``distance_m`` -> +vx.
- ``lateral_error_m > 0``: the work region is to the robot's left of where it should be -> +vy.
- ``heading_error_deg > 0``: the table normal points to the robot's left -> +omega (CCW).
"""

import abc
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TablePoseError:
    distance_error_m: float
    lateral_error_m: float
    heading_error_deg: float
    timestamp_s: float
    valid: bool
    reason: str = ""

    @classmethod
    def invalid(cls, timestamp_s: float, reason: str) -> "TablePoseError":
        return cls(math.nan, math.nan, math.nan, timestamp_s, False, reason)


@dataclass(frozen=True)
class TableTarget:
    """Desired robot pose relative to the table frame.

    Best obtained by teaching (:func:`TableTarget.taught`): park the robot by hand at the desired pose and
    record what the estimator measures. Constant calibration biases then cancel out.
    """

    # Clearance from the reference point to the front edge, measured along the table normal.
    distance_m: float = 0.30
    # Desired offset of the reference point along the edge (+ = left of the work-region centre).
    lateral_m: float = 0.0
    # Desired angle of the table normal in the base frame (+ = to the robot's left).
    heading_deg: float = 0.0
    # Robot reference point in the base frame (x forward, y left). Base centre by default.
    reference_x_m: float = 0.0
    reference_y_m: float = 0.0

    @classmethod
    def taught(cls, measured: "TableMeasurement", reference_x_m: float = 0.0, reference_y_m: float = 0.0):
        return cls(
            measured.distance_m, measured.lateral_m, measured.heading_deg, reference_x_m, reference_y_m
        )

    def save(self, path: str | Path, **metadata) -> None:
        """Extra ``metadata`` (e.g. the tag placement used while teaching) is stored alongside."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({**asdict(self), **metadata}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TableTarget":
        data = json.loads(Path(path).read_text())
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


@dataclass(frozen=True)
class TableMeasurement:
    """Robot pose relative to the table, in the same terms as :class:`TableTarget`."""

    distance_m: float
    lateral_m: float
    heading_deg: float


class TablePoseEstimator(abc.ABC):
    @abc.abstractmethod
    def get_table_pose_error(self) -> TablePoseError: ...


def measure_table_pose(
    T_base_table: np.ndarray, reference_x_m: float = 0.0, reference_y_m: float = 0.0
) -> TableMeasurement | None:
    """Floor-plane (x, y, yaw) pose of the robot reference point relative to the table frame."""
    into_table = T_base_table[:2, 0]
    norm = float(np.linalg.norm(into_table))
    if norm < 1e-6:
        return None
    into_table = into_table / norm
    along_edge_left = np.array([-into_table[1], into_table[0]])
    edge_from_ref = T_base_table[:2, 3] - np.array([reference_x_m, reference_y_m])
    return TableMeasurement(
        distance_m=float(edge_from_ref @ into_table),
        # Position of the reference point along the edge, relative to the work-region centre.
        lateral_m=float(-edge_from_ref @ along_edge_left),
        heading_deg=math.degrees(math.atan2(into_table[1], into_table[0])),
    )


def table_pose_error(T_base_table: np.ndarray, target: TableTarget, timestamp_s: float) -> TablePoseError:
    m = measure_table_pose(T_base_table, target.reference_x_m, target.reference_y_m)
    if m is None:
        return TablePoseError.invalid(timestamp_s, "table x-axis is vertical")
    return TablePoseError(
        distance_error_m=m.distance_m - target.distance_m,
        lateral_error_m=target.lateral_m - m.lateral_m,
        heading_error_deg=m.heading_deg - target.heading_deg,
        timestamp_s=timestamp_s,
        valid=True,
    )
