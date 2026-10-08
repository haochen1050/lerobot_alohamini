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

"""4x4 homogeneous transform helpers.

Naming: ``T_a_b`` maps points expressed in frame ``b`` into frame ``a`` (``p_a = T_a_b @ p_b``).
"""

import cv2
import numpy as np


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def rot_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_xyz_yaw(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    return make_transform(rot_z(np.radians(yaw_deg)), (x, y, z))


def from_rvec_tvec(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_transform(R, tvec)


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Mean translation and chordal-mean rotation (projected back onto SO(3))."""
    if not transforms:
        raise ValueError("No transforms to average")
    t = np.mean([T[:3, 3] for T in transforms], axis=0)
    U, _, Vt = np.linalg.svd(np.sum([T[:3, :3] for T in transforms], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return make_transform(R, t)
