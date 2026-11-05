import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from lerobot.robots.alohamini.config_alohamini import alohamini_cameras_config
from lerobot.robots.alohamini.perception import AprilTagDetector, CameraIntrinsics

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


# ---------------------------------------------------------------- orbbec-intrinsics, with the SDK faked


class Models:
    NONE, MODIFIED_BROWN_CONRADY, BROWN_CONRADY, BROWN_CONRADY_K6 = "none", "modified", "brown", "k6"


class FakeOrbbec:
    """The parts of pyorbbecsdk the command uses, with names checked against its type stubs. The projection is
    OpenCV's, so it agrees with the coefficients unless ``projection_error_px`` says otherwise."""

    OBCameraDistortionModel = Models
    OBFormat = SimpleNamespace(MJPG="mjpg")
    OBSensorType = SimpleNamespace(COLOR_SENSOR="color")

    def __init__(self, intrinsic=None, distortion=None, projection_error_px=0.0, devices=1, frames_after=2):
        self.intrinsic = intrinsic or SimpleNamespace(
            fx=763.0, fy=755.0, cx=640.0, cy=360.0, width=1280, height=720
        )
        self.distortion = distortion or SimpleNamespace(
            model=Models.BROWN_CONRADY_K6,
            k1=0.08,
            k2=-0.05,
            k3=0.01,
            k4=0.02,
            k5=-0.01,
            k6=0.005,
            p1=0.001,
            p2=-0.002,
        )
        self.projection_error_px = projection_error_px
        self.devices = devices
        self.asked = None
        self.enabled = []
        self.running = False
        self.waits = 0
        self.frames_after = frames_after

    def Context(self):  # noqa: N802
        info = SimpleNamespace(
            get_name=lambda: "Femto Bolt",
            get_serial_number=lambda: "CL1",
            get_firmware_version=lambda: "1.1.3",
        )
        device = SimpleNamespace(get_device_info=lambda: info)

        class Context:
            pass

        context = Context()
        alive = weakref.ref(context)

        def get_device(_):
            # Like the real SDK: the device list is unusable once its Context has been freed.
            if alive() is None:
                raise RuntimeError('NULL pointer passed for argument "deviceMgr"')
            return device

        context.query_devices = lambda: SimpleNamespace(
            get_count=lambda: self.devices,
            get_device_by_index=get_device,
            get_device_by_serial_number=get_device,
        )
        return context

    def Config(self):  # noqa: N802
        return SimpleNamespace(enable_stream=lambda profile: self.enabled.append(profile))

    def Pipeline(self, device):  # noqa: N802
        def get_video_stream_profile(*asked):
            self.asked = asked
            return "profile"

        profile = SimpleNamespace(
            get_intrinsic=lambda: self.intrinsic, get_distortion=lambda: self.distortion
        )
        frame = SimpleNamespace(
            get_stream_profile=lambda: SimpleNamespace(as_video_stream_profile=lambda: profile)
        )

        def wait_for_frames(timeout_ms):
            assert self.running  # calibration is only read from a running stream
            self.waits += 1
            if self.frames_after is None or self.waits <= self.frames_after:
                return None
            return SimpleNamespace(get_color_frame=lambda: frame)

        return SimpleNamespace(
            get_stream_profile_list=lambda sensor: SimpleNamespace(
                get_video_stream_profile=get_video_stream_profile
            ),
            start=lambda config: setattr(self, "running", True),
            stop=lambda: setattr(self, "running", False),
            wait_for_frames=wait_for_frames,
        )

    def OBExtrinsic(self):  # noqa: N802
        class Extrinsic:
            """Like the real SDK, which rejected a flat rotation on the Pi."""

            def __setattr__(self, name, value):
                if name == "rot" and np.shape(value) != (3, 3):
                    raise RuntimeError("rot must be a 3x3 array")
                if name == "transform" and np.shape(value) != (3,):
                    raise RuntimeError("transform must be a 3-element array")
                object.__setattr__(self, name, value)

        return Extrinsic()

    def OBPoint3f(self, x, y, z):  # noqa: N802
        return (x, y, z)

    def transformation3dto2d(self, point, intrinsic, distortion, extrinsic):
        d = distortion
        coeffs = (
            np.zeros(5)
            if d.model == Models.NONE
            else np.array([d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6], dtype=float)
        )
        matrix = np.array([[intrinsic.fx, 0, intrinsic.cx], [0, intrinsic.fy, intrinsic.cy], [0, 0, 1.0]])
        px = cv2.projectPoints(np.array([point], dtype=float), np.zeros(3), np.zeros(3), matrix, coeffs)[
            0
        ].ravel()
        return SimpleNamespace(x=px[0] + self.projection_error_px, y=px[1])


def run_orbbec(monkeypatch, tmp_path, sdk, *extra):
    out = tmp_path / "intrinsics.json"
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(sys, "argv", ["table_perception.py", "orbbec-intrinsics", f"--out={out}", *extra])
    table_perception.main()
    return CameraIntrinsics.load(out)


def test_orbbec_intrinsics_saved_in_opencv_order_at_the_capture_size(monkeypatch, tmp_path, capsys):
    sdk = FakeOrbbec()
    saved = run_orbbec(monkeypatch, tmp_path, sdk)
    assert sdk.asked == (1280, 720, "mjpg", 30)
    assert sdk.enabled == ["profile"] and not sdk.running
    assert saved.image_size == (1280, 720)
    assert saved.camera_matrix[0, 0] == 763.0 and saved.camera_matrix[1, 2] == 360.0
    # OpenCV's order: k1 k2 p1 p2 k3 k4 k5 k6.
    assert saved.dist_coeffs.tolist() == [0.08, -0.05, 0.001, -0.002, 0.01, 0.02, -0.01, 0.005]
    assert "80 x 51 deg" in capsys.readouterr().out


def test_orbbec_five_coefficient_model_and_no_distortion(monkeypatch, tmp_path):
    five = FakeOrbbec(
        distortion=SimpleNamespace(model=Models.NONE, k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0)
    )
    assert run_orbbec(monkeypatch, tmp_path, five).dist_coeffs.tolist() == [0.0] * 5

    brown = SimpleNamespace(
        model=Models.BROWN_CONRADY, k1=0.1, k2=-0.2, k3=0.3, k4=0, k5=0, k6=0, p1=0.01, p2=0.02
    )
    assert orbbec_coeffs(brown) == [0.1, -0.2, 0.01, 0.02, 0.3]


def orbbec_coeffs(distortion):
    return table_perception.orbbec_distortion_to_opencv(distortion, Models).tolist()


def test_orbbec_refuses_an_unknown_model(monkeypatch, tmp_path):
    sdk = FakeOrbbec(
        distortion=SimpleNamespace(
            model=Models.MODIFIED_BROWN_CONRADY, k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0
        )
    )
    with pytest.raises(SystemExit, match="no OpenCV equivalent"):
        run_orbbec(monkeypatch, tmp_path, sdk)
    assert not (tmp_path / "intrinsics.json").exists()


def test_orbbec_refuses_coefficients_that_do_not_match_its_own_projection(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="differs from OpenCV"):
        run_orbbec(monkeypatch, tmp_path, FakeOrbbec(projection_error_px=1.0))
    assert not (tmp_path / "intrinsics.json").exists()


@pytest.mark.parametrize(
    "bad",
    [
        {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "width": 0, "height": 0},  # reported on older firmware
        {"fx": 763.0, "fy": 755.0, "cx": 640.0, "cy": 360.0, "width": 1920, "height": 1080},
    ],
)
def test_orbbec_refuses_invalid_intrinsics(monkeypatch, tmp_path, bad):
    with pytest.raises(SystemExit, match="firmware 1.1.3"):
        run_orbbec(monkeypatch, tmp_path, FakeOrbbec(intrinsic=SimpleNamespace(**bad)))
    assert not (tmp_path / "intrinsics.json").exists()


def test_orbbec_stops_the_stream_even_with_no_frames(monkeypatch, tmp_path):
    sdk = FakeOrbbec(frames_after=None)
    with pytest.raises(SystemExit, match="No colour frame"):
        run_orbbec(monkeypatch, tmp_path, sdk)
    assert sdk.waits == 10 and not sdk.running
    assert not (tmp_path / "intrinsics.json").exists()


def test_orbbec_needs_exactly_one_device_unless_a_serial_is_given(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="found 0"):
        run_orbbec(monkeypatch, tmp_path, FakeOrbbec(devices=0))
    run_orbbec(monkeypatch, tmp_path, FakeOrbbec(devices=2), "--serial=CL1")


def test_estimator_accepts_eight_distortion_coefficients():
    # A factory Femto Bolt calibration has 8 where ChArUco gives 5.
    matrix = np.array([[763.0, 0, 640.0], [0, 755.0, 360.0], [0, 0, 1]])
    coeffs = np.array([0.08, -0.05, 0.001, -0.002, 0.01, 0.02, -0.01, 0.005])
    intrinsics = CameraIntrinsics(matrix, coeffs, (1280, 720))
    detector = AprilTagDetector(intrinsics, 0.10)
    half = 0.05
    tag = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=float)
    corners = cv2.projectPoints(tag, np.zeros(3), np.array([0.05, -0.02, 0.8]), matrix, coeffs)[0].reshape(
        4, 2
    )
    observation = detector.solve(3, corners, up_in_cam=np.array([0.0, -1.0, 0.0]))
    assert observation.T_cam_tag[:3, 3] == pytest.approx([0.05, -0.02, 0.8], abs=1e-3)
    # A frontal planar tag has two near-equal poses, so this is small rather than zero.
    assert observation.reprojection_px < 0.5
