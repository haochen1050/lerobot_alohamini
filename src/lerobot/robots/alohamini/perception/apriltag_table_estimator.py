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

"""Table pose from a single AprilTag (36h11) lying flat on the table.

Tag frame (OpenCV ArUco convention): origin at the tag centre, x towards the printed right edge, y towards
the printed top edge, z out of the printed face (up, for a tag lying on the table).
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .camera_calibration import CameraIntrinsics
from .geometry import from_rvec_tvec, invert, pose_xyz_yaw
from .table_pose import TablePoseError, TablePoseEstimator, TableTarget, table_pose_error

# Returns (BGR image, capture time on time.monotonic()) or None when no frame is available.
FrameSource = Callable[[], tuple[np.ndarray, float] | None]


@dataclass(frozen=True)
class TagPlacement:
    """Where the tag sits in the table frame (see table_pose.py)."""

    # Distance of the tag centre from the front edge, into the table.
    x_m: float
    # Tag centre along the edge relative to the work-region centre (+ = left as seen by the robot).
    y_m: float = 0.0
    # -90: printed top of the tag points into the table (away from the robot), edges parallel to the edge.
    yaw_deg: float = -90.0
    size_m: float = 0.10
    # None accepts exactly one visible tag.
    tag_id: int | None = None

    def T_table_tag(self) -> np.ndarray:
        return pose_xyz_yaw(self.x_m, self.y_m, 0.0, self.yaw_deg)


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    T_cam_tag: np.ndarray
    reprojection_px: float


def tag_object_points(size_m: float) -> np.ndarray:
    h = size_m / 2
    # Same order as ArUco corners: top-left, top-right, bottom-right, bottom-left.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


class AprilTagDetector:
    def __init__(self, intrinsics: CameraIntrinsics, tag_size_m: float):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params
        )
        self.intrinsics = intrinsics
        self._object_points = tag_object_points(tag_size_m)

    def detect(self, image: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """[(tag_id, 4x2 corners)]"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners, strict=True)]

    def solve(self, tag_id: int, corners: np.ndarray, up_in_cam: np.ndarray | None = None) -> TagObservation:
        """Tag pose in the camera frame.

        A planar square has two near-equivalent PnP solutions. With ``up_in_cam`` (the world up direction in
        camera coordinates) the one whose normal is closest to up is chosen; otherwise the lowest reprojection.
        """
        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            self._object_points,
            corners.astype(np.float32),
            self.intrinsics.camera_matrix,
            self.intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates = [
            TagObservation(tag_id, from_rvec_tvec(r, t), float(np.ravel(e)[0]))
            for r, t, e in zip(rvecs, tvecs, errors, strict=True)
        ]
        if up_in_cam is not None:
            return max(candidates, key=lambda o: float(o.T_cam_tag[:3, 2] @ up_in_cam))
        return min(candidates, key=lambda o: o.reprojection_px)


class AprilTagTableEstimator(TablePoseEstimator):
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        T_base_cam: np.ndarray,
        placement: TagPlacement,
        target: TableTarget,
        frame_source: FrameSource | None = None,
        *,
        max_age_s: float = 0.3,
        max_reprojection_px: float = 2.0,
        max_tilt_deg: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.detector = AprilTagDetector(intrinsics, placement.size_m)
        self.intrinsics = intrinsics
        self.T_base_cam = T_base_cam
        self.placement = placement
        self.target = target
        self.frame_source = frame_source
        self.max_age_s = max_age_s
        self.max_reprojection_px = max_reprojection_px
        self.min_up_cos = math.cos(math.radians(max_tilt_deg))
        self.clock = clock
        self._T_tag_table = invert(placement.T_table_tag())
        self._up_in_cam = T_base_cam[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        # Last accepted tag / table poses in the base frame, for logging and target teaching.
        self.last_T_base_tag: np.ndarray | None = None
        self.last_T_base_table: np.ndarray | None = None

    def get_table_pose_error(self) -> TablePoseError:
        frame = self.frame_source() if self.frame_source is not None else None
        if frame is None:
            return TablePoseError.invalid(self.clock(), "no camera frame")
        return self.estimate(*frame)

    def estimate(self, image: np.ndarray, capture_time_s: float) -> TablePoseError:
        age = self.clock() - capture_time_s
        if age > self.max_age_s:
            return TablePoseError.invalid(capture_time_s, f"stale frame ({age:.2f}s old)")
        h, w = image.shape[:2]
        if (w, h) != tuple(self.intrinsics.image_size):
            return TablePoseError.invalid(
                capture_time_s, f"image {w}x{h} does not match calibration {self.intrinsics.image_size}"
            )

        detections = self.detector.detect(image)
        wanted = self.placement.tag_id
        matches = [d for d in detections if wanted is None or d[0] == wanted]
        if len(matches) != 1:
            seen = [d[0] for d in detections]
            expected = "one tag" if wanted is None else f"tag {wanted}"
            return TablePoseError.invalid(capture_time_s, f"expected {expected}, saw {seen}")

        obs = self.detector.solve(*matches[0], up_in_cam=self._up_in_cam)
        if obs.reprojection_px > self.max_reprojection_px:
            return TablePoseError.invalid(
                capture_time_s, f"reprojection {obs.reprojection_px:.2f}px too high"
            )

        T_base_tag = self.T_base_cam @ obs.T_cam_tag
        if T_base_tag[2, 2] < self.min_up_cos:
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, T_base_tag[2, 2]))))
            return TablePoseError.invalid(capture_time_s, f"tag tilted {tilt:.0f} deg from horizontal")

        self.last_T_base_tag = T_base_tag
        self.last_T_base_table = T_base_tag @ self._T_tag_table
        return table_pose_error(self.last_T_base_table, self.target, capture_time_s)
