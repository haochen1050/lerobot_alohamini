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
    def move_base_for(self, vx, vy, omega_degps, duration_s, *, watch=None, poll_s=0.05):
        self.pulses.append((vx, vy, omega_degps, duration_s))
        vx, vy, w = (
            s * self.efficiency * v
            for s, v in zip(self.signs, (vx, vy, math.radians(omega_degps)), strict=True)
        )
        n = 20
        dt = duration_s / n
        elapsed, next_watch = 0.0, poll_s
        for _ in range(n):
            self.yaw += w * dt
            self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
            self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
            self.t += dt
            elapsed += dt
            self.on_substep()
            if watch is not None and elapsed >= next_watch - 1e-9 and elapsed < duration_s - 1e-9:
                next_watch += poll_s
                if not watch():
                    break
        stop = self.stop_base()
        return StopResult(stop.confirmed, stop.attempts, stop.errors, elapsed_s=elapsed)

    def on_substep(self):
        pass

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


# ---------------------------------------------------------------- keeping the tag in view


class ViewWorld(OrbitWorld):
    """OrbitWorld with an image: the tag's margin to the image border (px) shrinks linearly with its bearing,
    reaching 0 at the edge of the field of view. Tracks the smallest true margin seen during motion."""

    px_per_deg = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tag_margin_px = None
        self.last_fit_T_base_tag = None
        self.min_true_margin = math.inf
        self.lost_until_moved = False

    def _margin(self, tag_x, tag_y):
        if tag_x <= 0:
            return -math.inf
        return (self.fov_deg - abs(math.degrees(math.atan2(tag_y, tag_x)))) * self.px_per_deg

    def true_margin(self):
        tag = invert(pose_xyz_yaw(self.x, self.y, 0, math.degrees(self.yaw)))
        return self._margin(tag[0, 3], tag[1, 3])

    def on_substep(self):
        self.min_true_margin = min(self.min_true_margin, self.true_margin())

    def get_table_pose_error(self):
        if self.lost_until_moved:
            self.calls += 1
            self.t += 0.033
            self.last_tag_bearing_deg = self.last_raw_T_base_table = self.last_tag_margin_px = None
            return TablePoseError.invalid(self.t, "expected one tag, saw []")
        err = super().get_table_pose_error()
        if self.last_tag_bearing_deg is None:
            self.last_tag_margin_px = None
        else:
            self.last_fit_T_base_tag = self.last_raw_T_base_table  # the tag is at the table origin
            self.last_tag_margin_px = self.true_margin()
        return err

    def predict_tag_margin(self, dx, dy, dyaw_deg):
        if self.last_fit_T_base_tag is None:
            return None
        tag = invert(pose_xyz_yaw(dx, dy, 0, dyaw_deg)) @ self.last_fit_T_base_tag
        return self._margin(tag[0, 3], tag[1, 3])


def facing_tag(x, y, **kwargs):
    """Robot at (x, y) in the table frame, turned to look straight at the tag."""
    return ViewWorld(x=x, y=y, yaw_deg=math.degrees(math.atan2(-y, -x)), **kwargs)


def test_tag_stays_in_view_when_aligning_heading_would_lose_it():
    # 15 cm to the side and looking at the tag: turning square to the table first would put the tag ~23 deg
    # off-centre, outside the +-20 deg view. The plan must keep it in view the whole time.
    world = facing_tag(-0.36, 0.15, fov_deg=20.0)
    result = run(world, orbit=False)  # orbit alone would fix this start; test the per-axis stages

    assert result.ready, result.reason
    assert world.min_true_margin > 0, "tag left the image during a pulse"
    err = world.true_error()
    assert abs(err.lateral_error_m) <= 0.02 and abs(err.heading_error_deg) <= 2.0
    kinds = {e["event"] for e in result.events}
    assert "view_switch" in kinds  # corrected another axis while turning square would lose the tag
    assert "recover" not in kinds


def test_without_view_keeping_the_same_start_loses_the_tag():
    world = facing_tag(-0.36, 0.15, fov_deg=20.0)
    result = run(world, keep_tag_in_view=False, orbit=False)
    assert result.state is AlignState.FAULT
    assert "perception lost" in result.reason
    assert world.min_true_margin < 0


def test_watch_ends_a_pulse_before_the_tag_leaves_the_image():
    # Planning disabled (any margin accepted), so only the mid-pulse watch protects the view.
    world = facing_tag(-0.36, 0.15, fov_deg=20.0)
    result = run(world, view_margin_px=-1e9, max_steps=6, orbit=False)

    stops = [e for e in result.events if e["event"] == "watch_stop"]
    assert stops and "image edge" in stops[0]["reason"]
    assert world.min_true_margin > 0
    cut = [e for e in result.events if e["event"] == "stopped" and e["elapsed_s"] < 0.3 - 1e-6]
    assert cut, "a pulse should have been cut short"


class FlakyWorld(ViewWorld):
    """The tag cannot be detected after the n-th move until the base moves again (or ever, if sticky)."""

    def __init__(self, *args, lose_after_move=1, sticky=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lose_after_move = lose_after_move
        self.sticky = sticky
        self.moves = 0

    def move_base_for(self, *args, **kwargs):
        if not self.sticky:
            self.lost_until_moved = False
        result = super().move_base_for(*args, **kwargs)
        self.moves += 1
        if self.moves == self.lose_after_move:
            self.lost_until_moved = True
        return result


def test_tag_lost_after_a_pulse_is_recovered_by_reversing_it():
    world = FlakyWorld(x=-0.40, y=0.03, yaw_deg=6.0, fov_deg=40.0)
    result = run(world)

    assert result.ready, result.reason
    recover = [e for e in result.events if e["event"] == "recover"]
    assert len(recover) == 1
    first, back = world.pulses[0], world.pulses[1]
    assert back[:3] == pytest.approx(tuple(-c for c in first[:3]))
    assert back[3] == pytest.approx(first[3])
    # The same axis continues with half-length pulses.
    heading = [e for e in result.events if e["event"] == "pulse" and e["axis"] == "heading"]
    assert heading[0]["duration_s"] == pytest.approx(0.3)
    assert heading[1]["duration_s"] == pytest.approx(0.15)


def test_tag_still_lost_after_reversing_faults():
    world = FlakyWorld(x=-0.40, y=0.03, yaw_deg=6.0, fov_deg=40.0, sticky=True)
    result = run(world)
    assert result.state is AlignState.FAULT
    assert "still not visible after reversing" in result.reason
    # The pulse, its reversal, then a pure-rotation scan of +-20 deg in 10 deg turns - nothing more.
    first, back, *scan = world.pulses
    assert back[:3] == pytest.approx(tuple(-c for c in first[:3]))
    assert scan and all(vx == 0 and vy == 0 and abs(w) * d <= 10.0 + 1e-6 for vx, vy, w, d in scan)
    offsets = [e["offset_deg"] for e in result.events if e["event"] == "recover_turn"]
    assert max(offsets) == pytest.approx(20.0) and min(offsets) == pytest.approx(-20.0)
    assert abs(sum(w * d for _vx, _vy, w, d in scan)) == pytest.approx(20.0)  # ends on the far side


def test_recoveries_are_limited():
    class AlwaysFlaky(ViewWorld):
        def move_base_for(self, *args, **kwargs):
            self.lost_until_moved = False
            result = super().move_base_for(*args, **kwargs)
            # Lose the tag after every forward pulse, never after a reversal.
            self.lost_until_moved = not getattr(self, "_reversing", False)
            self._reversing = not getattr(self, "_reversing", False) and self.lost_until_moved
            return result

    world = AlwaysFlaky(x=-0.40, y=0.03, yaw_deg=6.0, fov_deg=40.0)
    result = run(world, max_recoveries=2)
    assert result.state is AlignState.FAULT
    assert "perception lost" in result.reason
    assert len([e for e in result.events if e["event"] == "recover"]) == 2


@pytest.mark.parametrize("theta", [-30.0, 25.0])
def test_orbit_keeps_the_tag_in_view(theta):
    t = math.radians(theta)
    world = ViewWorld(x=-0.6 * math.cos(t), y=-0.6 * math.sin(t), yaw_deg=theta, fov_deg=31.0)
    result = run(world)
    assert result.ready, result.reason
    assert any(e["event"] == "orbit_pulse" for e in result.events)
    assert world.min_true_margin > 0


@pytest.mark.parametrize("slip_deg", [20.0, -20.0])
def test_scan_finds_the_tag_when_reversing_leaves_the_robot_turned_away(slip_deg):
    class SlippyReverse(FlakyWorld):
        def move_base_for(self, *args, **kwargs):
            result = super().move_base_for(*args, **kwargs)
            if self.moves == 2:  # the reversal ends turned away from the tag
                self.yaw += math.radians(slip_deg)
            return result

    world = SlippyReverse(x=-0.40, y=0.0, yaw_deg=0.0, fov_deg=15.0)
    world.yaw = math.radians(3.0)  # start slightly turned so the first pulse is a heading correction
    result = run(world, orbit=False)

    assert result.ready, result.reason
    found = [e for e in result.events if e["event"] == "recover_found"]
    assert len(found) == 1
    assert math.copysign(1, found[0]["offset_deg"]) == -math.copysign(1, slip_deg)  # turned back towards it


@pytest.mark.parametrize("turn", [49.0, -49.0, 55.0])
def test_turned_beyond_heading_limit_in_front_of_the_table_squares_up(turn):
    # In front of the table, tag visible and fitted, but turned past +-45 deg: the search turns towards the
    # tag until the heading is within the limit (no orbit - the robot is not displaced).
    world = displaced(0.0, radius=0.5, extra_turn_deg=turn, fov_deg=60.0)
    result = run(world)
    assert result.ready, result.reason
    turns = [e for e in result.events if e["event"] == "search_pulse"]
    assert turns and all(e["status"] == "seen" for e in turns)
    assert all(math.copysign(1, e["omega_degps"]) == -math.copysign(1, turn) for e in turns)
    assert not [e for e in result.events if e["event"] == "orbit_pulse"]
