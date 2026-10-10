import math
import random

import pytest

from lerobot.robots.alohamini.base import BaseStopError, StopResult
from lerobot.robots.alohamini.behaviors import AlignConfig, AlignState, AlignToTable
from lerobot.robots.alohamini.perception import TablePoseError, TableTarget, table_pose_error
from lerobot.robots.alohamini.perception.geometry import invert, pose_xyz_yaw

TARGET = TableTarget(distance_m=0.30)


class SimWorld:
    """Base pose in the table frame, a clock, and imperfect wheels."""

    def __init__(self, x=-0.30, y=0.0, yaw_deg=0.0, *, efficiency=0.8, signs=(1, 1, 1), noise=0.0, seed=0):
        self.x, self.y, self.yaw = x, y, math.radians(yaw_deg)
        self.t = 100.0
        self.efficiency = efficiency
        self.signs = signs
        self.noise = noise
        self.rng = random.Random(seed)
        self.pulses: list[tuple[float, float, float, float]] = []
        self.stops = 0
        self.stop_ok = True
        self.lost_after: int | None = None
        self.frozen_frames = False
        self.calls = 0

    # -- base
    def move_base_for(self, vx, vy, omega_degps, duration_s):
        self.pulses.append((vx, vy, omega_degps, duration_s))
        vx, vy, w = (
            s * self.efficiency * v
            for s, v in zip(self.signs, (vx, vy, math.radians(omega_degps)), strict=True)
        )
        n = 20
        dt = duration_s / n
        for _ in range(n):
            self.yaw += w * dt
            self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
            self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
        self.t += duration_s
        return self.stop_base()

    def stop_base(self):
        self.stops += 1
        if not self.stop_ok:
            return StopResult(False, 3, {"base_left_wheel": "readback 300 != 0"})
        return StopResult(True, 1)

    # -- perception
    def get_table_pose_error(self) -> TablePoseError:
        self.calls += 1
        self.t += 0.033
        stamp = 0.0 if self.frozen_frames else self.t
        if self.lost_after is not None and self.calls > self.lost_after:
            return TablePoseError.invalid(stamp, "expected one tag, saw []")
        err = table_pose_error(invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw))), TARGET, stamp)
        if not self.noise:
            return err
        g = self.rng.gauss
        return TablePoseError(
            err.distance_error_m + g(0, self.noise),
            err.lateral_error_m + g(0, self.noise),
            err.heading_error_deg + g(0, 20 * self.noise),
            stamp,
            True,
        )

    # -- time
    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def true_error(self):
        return table_pose_error(invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw))), TARGET, 0)


def run(world, **config):
    align = AlignToTable(world, world, AlignConfig(**config), clock=world.clock, sleep=world.sleep)
    return align.run()


@pytest.mark.parametrize("yaw", [-12.0, -6.0, 3.0, 10.0])
def test_heading_only_converges_and_touches_no_other_axis(yaw):
    world = SimWorld(x=-0.36, y=0.04, yaw_deg=yaw)
    result = run(world, axes=("heading",))

    assert result.ready, result.reason
    # Stages correct to half the tolerance, leaving margin for VERIFY.
    assert abs(world.true_error().heading_error_deg) <= 1.0
    assert all(vx == 0 and vy == 0 for vx, vy, _w, _d in world.pulses)
    assert all(abs(w) <= 15.0 and d <= 0.3 for _vx, _vy, w, d in world.pulses)
    assert [e["event"] for e in result.events].count("verify") == 3


@pytest.mark.parametrize(("x", "y", "yaw"), [(-0.45, 0.06, 8.0), (-0.38, -0.08, -10.0), (-0.31, 0.0, 0.5)])
def test_full_alignment_converges_from_nearby_poses(x, y, yaw):
    world = SimWorld(x=x, y=y, yaw_deg=yaw, noise=0.002, seed=1)
    result = run(world)

    assert result.ready, result.reason
    err = world.true_error()
    assert abs(err.distance_error_m) <= 0.025
    assert abs(err.lateral_error_m) <= 0.025
    assert abs(err.heading_error_deg) <= 2.5
    # Last thing that happened to the base was a confirmed stop.
    assert result.events[-1]["event"] == "ready"


def test_already_aligned_does_not_move():
    world = SimWorld()
    result = run(world)
    assert result.ready
    assert world.pulses == []
    assert result.steps == 0


def test_wrong_sign_is_caught_as_divergence():
    world = SimWorld(yaw_deg=8.0, signs=(1, 1, -1))
    result = run(world, axes=("heading",))
    assert result.state is AlignState.FAULT
    assert "grew" in result.reason
    assert len(world.pulses) == 1


def test_tag_loss_faults_and_stops():
    world = SimWorld(yaw_deg=10.0)
    world.lost_after = 5
    result = run(world, axes=("heading",))
    assert result.state is AlignState.FAULT
    assert "perception lost" in result.reason and "saw []" in result.reason
    assert world.stops >= 1


def test_stale_frames_fault_without_moving():
    world = SimWorld(yaw_deg=10.0)
    world.frozen_frames = True
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "no fresh frame" in result.reason
    assert world.pulses == []


def test_approach_never_overshoots_taught_distance():
    # An aggressive distance gain would jump straight past the target; the approach limit must cap it.
    from lerobot.robots.alohamini.behaviors import AxisConfig

    world = SimWorld(x=-0.40, efficiency=1.0)
    closest = []
    original = world.move_base_for

    def move(*args):
        result = original(*args)
        closest.append(-world.x)
        return result

    world.move_base_for = move
    aggressive = AxisConfig(tolerance=0.02, gain=50.0, max_speed=1.0, min_speed=0.01, noise=0.005)
    result = run(world, axes=("distance",), distance=aggressive)

    # Forward pulse capped at the remaining error + 3 cm margin, so the table edge is never crowded.
    assert world.pulses[0][0] * world.pulses[0][3] == pytest.approx(0.10 + 0.03)
    assert min(closest) >= 0.30 - 0.03 - 1e-9
    # Such a gain then overshoots backwards; the divergence check stops it instead of oscillating.
    assert result.state is AlignState.FAULT and "grew" in result.reason


def test_too_close_backs_away():
    world = SimWorld(x=-0.25)  # 5 cm closer than the target
    result = run(world, axes=("distance",))
    # Moving backward is allowed and should fix it.
    assert result.ready, result.reason
    assert all(vx < 0 for vx, *_ in world.pulses)


def test_workspace_limit():
    world = SimWorld(yaw_deg=10.0, efficiency=0.02)  # wheels barely move: keeps commanding rotation
    result = run(world, axes=("heading",), max_total_rotation_deg=10.0)
    assert result.state is AlignState.FAULT
    assert "workspace" in result.reason or "grew" in result.reason


def test_timeout():
    world = SimWorld(yaw_deg=10.0, efficiency=0.05)
    result = run(world, axes=("heading",), timeout_s=3.0, max_total_rotation_deg=1000)
    assert result.state is AlignState.FAULT
    assert "timeout" in result.reason or "no convergence" in result.reason


def test_unconfirmed_stop_propagates():
    world = SimWorld(yaw_deg=10.0)

    def move(*args):
        world.pulses.append(args)
        raise BaseStopError("left wheel still 300")

    world.move_base_for = move
    with pytest.raises(BaseStopError):
        run(world, axes=("heading",))


def test_keyboard_interrupt_stops_base():
    world = SimWorld(yaw_deg=10.0)

    def move(*args):
        raise KeyboardInterrupt

    world.move_base_for = move
    with pytest.raises(KeyboardInterrupt):
        run(world, axes=("heading",))
    assert world.stops >= 1


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        AlignConfig(axes=("yaw",))


# ---------------------------------------------------------------- SEARCH


class FovWorld(SimWorld):
    """SimWorld whose camera only sees the tag within +-fov_deg of straight ahead, and reports its bearing."""

    def __init__(self, *args, fov_deg=31.0, max_heading_deg=45.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fov_deg = fov_deg
        self.max_heading_deg = max_heading_deg
        self.last_tag_bearing_deg = None

    def get_table_pose_error(self):
        self.calls += 1
        self.t += 0.033
        base_table = invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw)))
        tag_x, tag_y = base_table[0, 3], base_table[1, 3]
        bearing = math.degrees(math.atan2(tag_y, tag_x))
        if abs(bearing) > self.fov_deg or tag_x <= 0:
            self.last_tag_bearing_deg = None
            return TablePoseError.invalid(self.t, "expected one tag, saw []")
        self.last_tag_bearing_deg = bearing
        err = table_pose_error(base_table, TARGET, self.t)
        if abs(err.heading_error_deg) > self.max_heading_deg:
            return TablePoseError.invalid(self.t, "table heading exceeds +-45")
        return err


@pytest.mark.parametrize(("yaw", "direction"), [(90.0, 1), (90.0, -1), (-150.0, 1), (60.0, -1)])
def test_search_finds_tag_then_aligns(yaw, direction):
    world = FovWorld(x=-0.36, y=0.03, yaw_deg=yaw)
    result = run(world, search_direction=direction)

    assert result.ready, result.reason
    err = world.true_error()
    assert abs(err.heading_error_deg) <= 2.0
    search = [e for e in result.events if e["event"] == "search_pulse"]
    assert search, "should have searched"
    # Search pulses are pure rotation at <= 15 deg/s and never longer than a 15 deg step.
    assert all(abs(e["omega_degps"]) <= 15.0 and e["duration_s"] <= 1.0 + 1e-9 for e in search)
    assert sum(abs(e["omega_degps"]) * e["duration_s"] for e in search) <= 360.0 + 1e-6
    # While the tag was not visible, it turned in the requested direction.
    blind = [e for e in search if e["status"] == "none"]
    assert all(math.copysign(1, e["omega_degps"]) == direction for e in blind)


def test_search_turns_towards_a_tag_it_can_see_but_not_use():
    # Turned 60 deg CCW with a wide camera: the tag is seen to the right but the heading check rejects it,
    # so the search must turn towards it (clockwise) rather than in the default search direction.
    world = FovWorld(x=-0.36, y=0.0, yaw_deg=60.0, fov_deg=80.0)
    result = run(world, search_direction=1)  # default direction would turn the wrong way
    assert result.ready, result.reason
    seen = [e for e in result.events if e["event"] == "search_pulse" and e["status"] == "seen"]
    assert seen and all(e["omega_degps"] < 0 for e in seen)  # turned clockwise, towards the tag


def test_search_gives_up_after_one_turn():
    world = FovWorld(yaw_deg=180.0, fov_deg=31.0)
    world.lost_after = 0  # tag never detected anywhere

    def never(*_args):
        world.calls += 1
        world.t += 0.033
        world.last_tag_bearing_deg = None
        return TablePoseError.invalid(world.t, "expected one tag, saw []")

    world.get_table_pose_error = never
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "not found after rotating" in result.reason
    total = sum(abs(w) * d for _vx, _vy, w, d in world.pulses)
    assert total <= 360.0 + 1e-6
    assert world.stops >= 1


def test_tag_ahead_but_unusable_faults_instead_of_spinning():
    world = FovWorld(yaw_deg=0.0)

    def unusable(*_args):
        world.calls += 1
        world.t += 0.033
        world.last_tag_bearing_deg = 2.0
        return TablePoseError.invalid(world.t, "tag tilted 90 deg")

    world.get_table_pose_error = unusable
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "unusable" in result.reason and "tilted" in result.reason
    assert world.pulses == []


def test_search_disabled_faults_immediately():
    world = FovWorld(yaw_deg=90.0)
    result = run(world, search=False)
    assert result.state is AlignState.FAULT
    assert "perception lost" in result.reason
    assert world.pulses == []


def test_search_rotation_does_not_count_against_alignment_limits():
    world = FovWorld(x=-0.33, yaw_deg=120.0)
    result = run(world, max_total_rotation_deg=30.0)
    assert result.ready, result.reason


# ---------------------------------------------------------------- ORBIT


class OrbitWorld(FovWorld):
    """FovWorld that also exposes the raw table pose (even beyond the heading limit) and the target."""

    target = TARGET

    def get_table_pose_error(self):
        err = super().get_table_pose_error()
        visible = self.last_tag_bearing_deg is not None
        self.last_raw_T_base_table = (
            invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw))) if visible else None
        )
        return err


def displaced(theta_deg, radius=0.6, extra_turn_deg=0.0, **kwargs):
    """Robot `theta_deg` around the approach point (CCW seen from above), facing it, then turned extra."""
    t = math.radians(theta_deg)
    return OrbitWorld(
        x=-radius * math.cos(t), y=-radius * math.sin(t), yaw_deg=theta_deg + extra_turn_deg, **kwargs
    )


@pytest.mark.parametrize("theta", [54.0, -54.0, 30.0, -60.0, 15.0])
def test_orbit_brings_displaced_robot_in_front_then_aligns(theta):
    world = displaced(theta)
    start_distance_to_edge = -world.x
    edge_distances = []
    original = world.move_base_for

    def move(*args):
        result = original(*args)
        edge_distances.append(-world.x)
        return result

    world.move_base_for = move
    result = run(world)

    assert result.ready, result.reason
    err = world.true_error()
    assert abs(err.heading_error_deg) <= 2.0
    assert abs(err.lateral_error_m) <= 0.02
    assert abs(err.distance_error_m) <= 0.02

    orbit = [e for e in result.events if e["event"] == "orbit_pulse"]
    assert orbit
    assert sum(abs(e["command"][2]) * e["duration_s"] for e in orbit) <= 90.0 + 1e-6
    for e in orbit:
        vx, vy, w = e["command"]
        assert math.hypot(vx, vy) <= 0.05 + 1e-9
        assert abs(w) <= 10.0 + 1e-9
        assert e["duration_s"] <= 1.0 + 1e-9
    # Orbiting towards the approach axis only moves the robot away from the edge line.
    n_orbit = len(orbit)
    assert min(edge_distances[:n_orbit]) >= start_distance_to_edge - 1e-6


def test_search_then_orbit_when_displaced_and_turned_away():
    world = displaced(50.0, extra_turn_deg=70.0)  # tag out of view and robot off to the side
    result = run(world, search_direction=-1)
    assert result.ready, result.reason
    kinds = [e["event"] for e in result.events]
    assert "search_pulse" in kinds and "orbit_pulse" in kinds
    assert kinds.index("search_pulse") < kinds.index("orbit_pulse")


def test_orbit_refuses_turned_tag_signature():
    world = displaced(85.0)
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "tag was turned" in result.reason
    assert world.pulses == []


def test_orbit_faults_when_robot_does_not_follow():
    world = displaced(50.0, efficiency=0.05)  # wheels slipping: off-angle barely changes
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "did not reduce" in result.reason
    assert len([e for e in result.events if e["event"] == "orbit_pulse"]) == 1


def test_orbit_refuses_when_too_close():
    world = displaced(40.0, radius=0.22)
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "too close" in result.reason


def test_orbit_disabled_leaves_large_offsets_to_fault():
    world = displaced(54.0)
    result = run(world, orbit=False)
    assert result.state is AlignState.FAULT


def test_merely_rotated_robot_is_not_orbited():
    # Straight in front of the table, just turned 25 deg: rotation is the heading stage's job, not the orbit's.
    world = displaced(0.0, radius=0.5, extra_turn_deg=25.0)
    result = run(world)
    assert result.ready, result.reason
    assert not [e for e in result.events if e["event"] == "orbit_pulse"]
