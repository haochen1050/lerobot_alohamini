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

from .apriltag_table_estimator import AprilTagDetector, AprilTagTableEstimator, TagPlacement
from .camera_calibration import (
    CameraIntrinsics,
    CharucoSpec,
    calibrate_intrinsics,
    load_transform,
    pick_charuco_variant,
    save_transform,
    solve_base_from_camera,
)
from .table_pose import (
    TableMeasurement,
    TablePoseError,
    TablePoseEstimator,
    TableTarget,
    measure_table_pose,
    table_pose_error,
)

__all__ = [
    "AprilTagDetector",
    "AprilTagTableEstimator",
    "CameraIntrinsics",
    "CharucoSpec",
    "TableMeasurement",
    "TablePoseError",
    "TablePoseEstimator",
    "TableTarget",
    "TagPlacement",
    "calibrate_intrinsics",
    "load_transform",
    "measure_table_pose",
    "pick_charuco_variant",
    "save_transform",
    "solve_base_from_camera",
    "table_pose_error",
]
