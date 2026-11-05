import sys
from pathlib import Path

import cv2
import pytest

from lerobot.robots.alohamini.config_alohamini import alohamini_cameras_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "debug"))
import table_perception  # noqa: E402


class FakeCapture:
    """A camera whose modes are the given sizes; V4L2-like, it settles on the nearest instead of failing."""

    def __init__(self, modes):
        self.modes = modes
        self.calls = []
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: modes[0][0], cv2.CAP_PROP_FRAME_HEIGHT: modes[0][1]}
        self.released = False

    def isOpened(self):  # noqa: N802
        return True

    def set(self, prop, value):
        self.calls.append(prop)
        self.props[prop] = value
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            want = self.props[cv2.CAP_PROP_FRAME_WIDTH]
            mode = min(self.modes, key=lambda m: abs(m[0] - want))
            self.props[cv2.CAP_PROP_FRAME_WIDTH], self.props[cv2.CAP_PROP_FRAME_HEIGHT] = mode
        return True

    def get(self, prop):
        return self.props[prop]

    def read(self):
        return False, None

    def release(self):
        self.released = True


def open_camera(monkeypatch, modes, *args, **kwargs):
    capture = FakeCapture(modes)
    monkeypatch.setattr(table_perception.cv2, "VideoCapture", lambda *a: capture)
    return capture, table_perception.LatestFrame("/dev/fake", *args, **kwargs)


def test_requests_the_pixel_format_before_the_size(monkeypatch):
    capture, cam = open_camera(monkeypatch, [(1280, 720)])
    cam.close()
    assert capture.calls[:2] == [cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH]
    assert capture.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")


def test_refuses_a_size_the_camera_does_not_have(monkeypatch):
    # The Femto Bolt has no 640x480: an old calibration size must not silently run at another one.
    capture = FakeCapture([(1280, 720), (1920, 1080)])
    monkeypatch.setattr(table_perception.cv2, "VideoCapture", lambda *a: capture)
    with pytest.raises(RuntimeError, match="gave 1280x720 when asked for 640x480"):
        table_perception.LatestFrame("/dev/fake", 640, 480)
    assert capture.released


def test_empty_fourcc_leaves_the_format_alone(monkeypatch):
    capture, cam = open_camera(monkeypatch, [(640, 480)], 640, 480, fourcc="")
    cam.close()
    assert cv2.CAP_PROP_FOURCC not in capture.calls


def test_host_opens_the_forward_camera_like_the_calibration_tools():
    forward = alohamini_cameras_config()["forward"]
    assert (forward.width, forward.height) == table_perception.DEFAULT_SIZE
    assert forward.fourcc == table_perception.DEFAULT_FOURCC
