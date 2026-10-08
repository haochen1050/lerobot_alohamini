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

"""Camera intrinsics (ChArUco) and camera-to-base extrinsics, with JSON persistence."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import invert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray  # (N,)
    image_size: tuple[int, int]  # (width, height)
    rms_px: float = float("nan")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "image_size": list(self.image_size),
            "rms_px": self.rms_px,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text())
        return cls(
            np.array(data["camera_matrix"], dtype=float),
            np.array(data["dist_coeffs"], dtype=float),
            (int(data["image_size"][0]), int(data["image_size"][1])),
            float(data.get("rms_px", float("nan"))),
        )


def save_transform(path: str | Path, T: np.ndarray, **metadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"T": T.tolist(), **metadata}, indent=2))


def load_transform(path: str | Path) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text())["T"], dtype=float)


def solve_base_from_camera(T_cam_tag: np.ndarray, T_base_tag: np.ndarray) -> np.ndarray:
    """Camera pose in the base frame, from one tag seen by the camera at a known base-frame pose."""
    return T_base_tag @ invert(T_cam_tag)


# ------------------------------------------------------------------ ChArUco


@dataclass(frozen=True)
class CharucoSpec:
    squares_x: int = 9
    squares_y: int = 12
    square_m: float = 0.030
    marker_m: float = 0.0225
    dictionary: int = cv2.aruco.DICT_5X5_100
    legacy: bool = False

    def board(self) -> "cv2.aruco.CharucoBoard":
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_m,
            self.marker_m,
            cv2.aruco.getPredefinedDictionary(self.dictionary),
        )
        board.setLegacyPattern(self.legacy)
        return board

    def variants(self) -> list["CharucoSpec"]:
        """Both orientations x both marker layouts; printed boards are ambiguous about these."""
        out = []
        for sx, sy in ((self.squares_x, self.squares_y), (self.squares_y, self.squares_x)):
            for legacy in (False, True):
                out.append(CharucoSpec(sx, sy, self.square_m, self.marker_m, self.dictionary, legacy))
        return out


def detect_charuco(image: np.ndarray, spec: CharucoSpec) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (object_points Nx3, image_points Nx2) for the detected chessboard corners, or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    board = spec.board()
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 6:
        return None
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def pick_charuco_variant(images: list[np.ndarray], spec: CharucoSpec) -> CharucoSpec:
    """The board variant that detects the most corners across ``images``."""

    def score(variant: CharucoSpec) -> int:
        return sum(len(d[0]) for img in images if (d := detect_charuco(img, variant)) is not None)

    scores = {variant: score(variant) for variant in spec.variants()}
    best = max(scores, key=scores.get)
    logger.info(
        "ChArUco variant scores: %s",
        {f"{v.squares_x}x{v.squares_y} legacy={v.legacy}": s for v, s in scores.items()},
    )
    if scores[best] == 0:
        raise ValueError("ChArUco board not detected in any image; check the board spec and lighting")
    return best


def calibrate_intrinsics(
    images: list[np.ndarray], spec: CharucoSpec, min_views: int = 10
) -> CameraIntrinsics:
    detections = [d for img in images if (d := detect_charuco(img, spec)) is not None]
    if len(detections) < min_views:
        raise ValueError(f"Only {len(detections)} usable board views, need >= {min_views}")
    h, w = images[0].shape[:2]
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [d[0] for d in detections], [d[1] for d in detections], (w, h), None, None
    )
    logger.info("Intrinsics from %d views: rms=%.3f px", len(detections), rms)
    return CameraIntrinsics(K, dist.ravel(), (w, h), float(rms))
