import json
import math

import numpy as np
import pytest

from lerobot.robots.alohamini.base import BaseLimits, BaseStopError
from lerobot.robots.alohamini.behaviors import (
    AlignConfig,
    RemoteBase,
    RemoteFrameSource,
    TableAlignmentSetup,
    align_base_through_host,
    summarize_alignment,
)
from lerobot.robots.alohamini.perception import CameraIntrinsics, TableTarget, table_pose_error
from lerobot.robots.alohamini.perception.geometry import invert, pose_xyz_yaw
from lerobot.robots.alohamini.rsi import (
    ALIGN,
    ALIGN_BASE,
    ExperienceRecord,
    ExperienceStore,
    Harness,
    HarnessError,
    HarnessLimits,
    Skill,
    Step,
)
from lerobot.rollout.robot_wrapper import ThreadSafeRobot

ARMS = {"arm_left_shoulder_pan.pos": 12.0, "arm_right_gripper.pos": 40.0}


class FakeHost:
    """An AlohaMiniClient stand-in: integrates the commanded base twist between calls on a fake clock."""

    def __init__(self, x=-0.40, y=0.04, yaw_deg=6.0, *, efficiency=0.85, moving_after_stop=False):
        self.x, self.y, self.yaw = x, y, math.radians(yaw_deg)
        self.t = 50.0
        self.efficiency = efficiency
        self.moving_after_stop = moving_after_stop
        self.command = (0.0, 0.0, 0.0)
        self.actions: list[dict] = []
        self.latest_host_timing: dict = {"camera_capture_monotonic_s": {"forward": 0.0}}
        self.image = np.zeros((4, 6, 3), np.uint8)
        self.image[..., 0] = 200  # red in RGB

    # time
    def clock(self):
        return self.t

    def sleep(self, s):
        self._advance(s)

    def _advance(self, dt):
        vx, vy, w = (
            self.efficiency * v for v in (self.command[0], self.command[1], math.radians(self.command[2]))
        )
        steps = max(1, int(dt / 0.005))
        for _ in range(steps):
            h = dt / steps
            self.yaw += w * h
            self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * h
            self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * h
        self.t += dt

    # client API
    def send_action(self, action):
        self.actions.append(dict(action))
        self.command = (action["x.vel"], action["y.vel"], action["theta.vel"])
        self._advance(0.002)
        return action

    def get_observation(self):
        self._advance(0.01)
        moving = self.command != (0.0, 0.0, 0.0) or self.moving_after_stop
        vel = (
            self.command
            if self.command != (0.0, 0.0, 0.0)
            else ((0.05, 0.0, 0.0) if moving else (0.0, 0.0, 0.0))
        )
        return {
            **ARMS,
            "lift_axis.height_mm": 310.0,
            "x.vel": vel[0],
            "y.vel": vel[1],
            "theta.vel": vel[2],
            "forward": self.image,
        }

    def true_error(self, target):
        return table_pose_error(
            invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw))), target, self.t
        )


def remote_base(host, **kwargs):
    return RemoteBase(host, clock=host.clock, sleep=host.sleep, **kwargs)


# ---------------------------------------------------------------- RemoteBase


def test_pulse_holds_arms_and_lift_and_ends_in_a_confirmed_stop():
    host = FakeHost()
    base = remote_base(host)
    base.capture_hold()
    result = base.move_base_for(0.03, 0.0, 0.0, 0.3)

    assert result.confirmed
    assert result.elapsed_s == pytest.approx(0.3, abs=0.02)
    moving = [a for a in host.actions if a["x.vel"] != 0]
    assert moving and all(a["x.vel"] == pytest.approx(0.03) for a in moving)
    assert all(
        a["arm_left_shoulder_pan.pos"] == 12.0 and a["lift_axis.height_mm"] == 310.0 for a in host.actions
    )
    assert host.actions[-1]["x.vel"] == 0 and host.actions[-1]["theta.vel"] == 0
    # Commands are re-sent through the pulse, well inside the host's 1 s watchdog.
    assert len(moving) >= 5


def test_pulse_is_clamped_and_bounded():
    host = FakeHost()
    base = remote_base(host, limits=BaseLimits(max_vx_mps=0.05, max_omega_degps=10.0, max_pulse_s=1.0))
    base.move_base_for(0.5, 0.0, -90.0, 0.2)
    assert max(a["x.vel"] for a in host.actions) == pytest.approx(0.05)
    assert min(a["theta.vel"] for a in host.actions) == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        base.move_base_for(0.01, 0.0, 0.0, 2.0)
    with pytest.raises(ValueError):
        base.move_base_for(float("nan"), 0.0, 0.0, 0.2)


def test_watch_ends_the_pulse_early():
    host = FakeHost()
    base = remote_base(host)
    calls = []

    def watch():
        calls.append(host.t)
        return len(calls) < 2

    result = base.move_base_for(0.03, 0.0, 0.0, 0.6, watch=watch, poll_s=0.1)
    assert len(calls) == 2
    assert result.elapsed_s < 0.4


def test_unconfirmed_stop_raises():
    host = FakeHost(moving_after_stop=True)
    base = remote_base(host, stop_timeout_s=0.5)
    assert not base.stop_base().confirmed
    with pytest.raises(BaseStopError):
        base.move_base_for(0.03, 0.0, 0.0, 0.2)
    assert host.actions[-1]["x.vel"] == 0  # still commanding zero to the end


# ---------------------------------------------------------------- RemoteFrameSource


def test_frame_source_converts_to_bgr_and_stamps_only_fresh_frames():
    host = FakeHost()
    source = RemoteFrameSource(ThreadSafeRobot(host), latency_s=0.1, clock=host.clock)
    image, stamp = source()
    assert image[0, 0, 2] == 200 and image[0, 0, 0] == 0  # RGB -> BGR
    assert stamp == pytest.approx(host.t - 0.1)

    host.latest_host_timing = {"camera_capture_monotonic_s": {}}  # this observation carried no new image
    assert source()[1] == pytest.approx(stamp)


# ---------------------------------------------------------------- setup files


def write_setup(directory, tag_placement=True):
    CameraIntrinsics(np.array([[390.0, 0, 320], [0, 390, 240], [0, 0, 1]]), np.zeros(5), (640, 480)).save(
        directory / "forward_intrinsics.json"
    )
    (directory / "forward_extrinsics.json").write_text(json.dumps({"T": np.eye(4).tolist()}))
    target = {
        "distance_m": 0.16,
        "lateral_m": 0.04,
        "heading_deg": 3.1,
        "reference_x_m": 0.0,
        "reference_y_m": 0.0,
    }
    if tag_placement:
        target["tag_placement"] = {
            "edge_offset_m": 0.3,
            "lateral_m": 0.0,
            "yaw_deg": -90.0,
            "size_m": 0.1,
            "tag_id": None,
            "mount": "flat",
            "height_m": 0.0,
        }
    (directory / "table_target.json").write_text(json.dumps(target))


def test_setup_loads_the_tag_placement_from_the_taught_target(tmp_path):
    write_setup(tmp_path)
    setup = TableAlignmentSetup.load(tmp_path)
    assert setup.placement.x_m == 0.3 and setup.placement.yaw_deg == -90.0 and setup.placement.mount == "flat"
    assert setup.target.distance_m == 0.16
    assert setup.intrinsics.image_size == (640, 480)

    other = tmp_path / "old"
    other.mkdir()
    write_setup(other, tag_placement=False)
    with pytest.raises(ValueError, match="tag_placement"):
        TableAlignmentSetup.load(other)


# ---------------------------------------------------------------- whole alignment through the host

TARGET = TableTarget(distance_m=0.30)


class HostEstimator:
    """Table pose read from the fake host's true pose, stamped with fresh frame times."""

    def __init__(self, host):
        self.host = host

    def get_table_pose_error(self):
        self.host.get_observation()
        return self.host.true_error(TARGET)


class FakeSetup:
    def __init__(self, host):
        self.host = host

    def estimator(self, frame_source):
        return HostEstimator(self.host)


def test_alignment_through_the_host_reaches_base_ready():
    host = FakeHost(x=-0.40, y=0.04, yaw_deg=6.0)
    base = remote_base(host)
    events = []
    result = align_base_through_host(
        host,
        FakeSetup(host),
        AlignConfig(orbit=False),
        log=events.append,
        base=base,
        clock=host.clock,
        sleep=host.sleep,
    )
    assert result.ready, result.reason
    err = host.true_error(TARGET)
    assert (
        abs(err.distance_error_m) <= 0.02
        and abs(err.lateral_error_m) <= 0.02
        and abs(err.heading_error_deg) <= 2
    )
    # Arms and lift never moved.
    assert all(a["arm_right_gripper.pos"] == 40.0 for a in host.actions)
    summary = summarize_alignment(result)
    assert summary["state"] == "BASE_READY" and summary["pulses"] == result.steps
    assert summary["start"]["distance_cm"] == pytest.approx(10.0, abs=0.5)
    assert set(summary) >= {"final", "start_source", "search_deg", "radial_cm", "orbit_deg", "duration_s"}


# ---------------------------------------------------------------- harness and records


def test_align_base_is_a_built_in_harness_step():
    harness = Harness(
        skills={"pick": Skill("pick up the cup", 30.0)},
        plan=(Step(ALIGN_BASE), Step("pick"), Step(ALIGN_BASE, 40.0)),
    )
    steps = harness.resolve(HarnessLimits(max_align_steps=2, align_time_s=90.0))
    assert [s.kind for s in steps] == [ALIGN, "policy", ALIGN]
    assert steps[0].time_s == 90.0 and steps[2].time_s == 40.0
    assert steps[1].instruction == "pick up the cup"

    with pytest.raises(HarnessError, match="allows 0"):
        harness.resolve(HarnessLimits())  # runner without alignment
    with pytest.raises(HarnessError, match="allows 1"):
        harness.resolve(HarnessLimits(max_align_steps=1))
    with pytest.raises(HarnessError, match=r"0\.\.90|2\.\.90"):
        Harness(harness.skills, (Step(ALIGN_BASE, 200.0),)).resolve(HarnessLimits(max_align_steps=1))
    with pytest.raises(HarnessError, match="built-in"):
        Harness({**harness.skills, ALIGN_BASE: Skill("drive", 10.0)}, (Step("pick"),)).resolve(
            HarnessLimits(max_align_steps=1)
        )


def test_records_keep_base_alignments_and_old_records_still_load(tmp_path):
    store = ExperienceStore(tmp_path / "records.jsonl")
    old = {
        "episode": 0,
        "variant": [],
        "baseline_version": 0,
        "init_state": "a",
        "repeat": 0,
        "steps": [],
        "success": True,
        "duration_s": 10.0,
    }
    store.path.write_text(json.dumps(old) + "\n")
    store.append(
        ExperienceRecord(1, [], 0, "a", 0, [], False, 12.0, base=[{"phase": "start", "state": "BASE_READY"}])
    )
    first, second = store.load()
    assert first.base == [] and second.base[0]["state"] == "BASE_READY"
