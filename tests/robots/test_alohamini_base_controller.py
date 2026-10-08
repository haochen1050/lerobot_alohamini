import logging
import os
import signal

import pytest

from lerobot.robots.alohamini.alohamini import AlohaMini
from lerobot.robots.alohamini.base import (
    WHEEL_NAMES,
    BaseController,
    BaseLimits,
    BaseStopError,
    body_to_wheel_raw,
    wheel_raw_to_body,
)

LEFT, BACK, RIGHT = WHEEL_NAMES


class FakeWheelBus:
    """Records per-wheel Goal_Velocity writes; reads return the last value written."""

    def __init__(self, *, fail_writes=None, fail_reads=None, stuck=None):
        self.goal = dict.fromkeys(WHEEL_NAMES, 0)
        self.log: list[tuple[str, str, int | None]] = []
        # name -> number of consecutive failures still to inject (-1 = forever)
        self.fail_writes = dict(fail_writes or {})
        self.fail_reads = dict(fail_reads or {})
        # name -> value the motor reports regardless of what was written
        self.stuck = dict(stuck or {})

    @staticmethod
    def _consume(failures, name):
        remaining = failures.get(name, 0)
        if remaining:
            failures[name] = remaining - 1 if remaining > 0 else -1
            return True
        return False

    def write(self, data_name, motor, value, *, normalize=True, num_retry=3):
        assert data_name == "Goal_Velocity"
        assert isinstance(motor, str), "motor must be a single name (the old stop() passed a list)"
        assert normalize is False
        self.log.append(("write", motor, value))
        if self._consume(self.fail_writes, motor):
            raise ConnectionError(f"no status packet from {motor}")
        self.goal[motor] = value

    def read(self, data_name, motor, *, normalize=True, num_retry=3):
        assert data_name == "Goal_Velocity"
        self.log.append(("read", motor, None))
        if self._consume(self.fail_reads, motor):
            raise ConnectionError(f"no status packet from {motor}")
        return self.stuck.get(motor, self.goal[motor])


def make_controller(bus, sleep=lambda _s: None, **kwargs):
    return BaseController(bus, wheel_radius=0.05, base_radius=0.125, sleep=sleep, **kwargs)


# ---------------------------------------------------------------- kinematics


def test_forward_matches_field_measurement():
    # SSH test on the old 0.05 m wheel constants: vx=0.05 m/s -> left 565, back 0, right -565.
    assert body_to_wheel_raw(0.05, 0, 0, wheel_radius=0.05, base_radius=0.125) == (565, 0, -565)


@pytest.mark.parametrize("cmd", [(0.05, 0, 0), (0, -0.07, 0), (0, 0, 15), (0.08, 0.03, -20), (2.0, 1.0, 300)])
@pytest.mark.parametrize("radii", [(0.05, 0.125), (0.063, 0.195)])
def test_kinematics_match_alohamini(cmd, radii):
    wheel_radius, base_radius = radii
    robot = object.__new__(AlohaMini)
    robot.wheel_radius, robot.base_radius = radii
    expected = robot._body_to_wheel_raw(*cmd)
    raw = body_to_wheel_raw(*cmd, wheel_radius=wheel_radius, base_radius=base_radius)
    assert raw == tuple(expected[n] for n in WHEEL_NAMES)

    expected_body = robot._wheel_raw_to_body(*raw)
    body = wheel_raw_to_body(*raw, wheel_radius=wheel_radius, base_radius=base_radius)
    assert body == pytest.approx(tuple(expected_body[k] for k in ("x.vel", "y.vel", "theta.vel")))


# ---------------------------------------------------------------- stopping


def test_forward_pulse_ends_with_all_wheels_zero():
    bus = FakeWheelBus()
    slept = []
    result = make_controller(bus, sleep=slept.append).move_base_for(0.05, 0, 0, 0.3)

    assert slept == [0.3]
    assert result.confirmed
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)
    writes = [(m, v) for op, m, v in bus.log if op == "write"]
    assert writes == [(LEFT, 565), (BACK, 0), (RIGHT, -565), (LEFT, 0), (BACK, 0), (RIGHT, 0)]


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("perception died")])
def test_pulse_stops_on_interrupt_or_exception(exc):
    bus = FakeWheelBus()

    def sleep(_s):
        assert bus.goal[LEFT] != 0, "base should be moving mid-pulse"
        raise exc

    with pytest.raises(type(exc)):
        make_controller(bus, sleep=sleep).move_base_for(0.05, 0, 0, 0.3)
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)


def test_failed_wheel_does_not_block_others_and_is_retried(caplog):
    bus = FakeWheelBus(fail_writes={BACK: 1})
    bus.goal = {LEFT: 500, BACK: 500, RIGHT: 500}

    result = make_controller(bus).stop_base()

    assert result.confirmed
    assert result.attempts == 2
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)
    assert "zero write to base_back_wheel failed" in caplog.text
    # Second attempt only retries the wheel that was not confirmed.
    second = bus.log[6:]
    assert [m for _op, m, _v in second] == [BACK, BACK]


def test_unconfirmed_stop_is_reported_loudly(caplog):
    bus = FakeWheelBus(stuck={RIGHT: 300})
    result = make_controller(bus, stop_attempts=3).stop_base()

    assert not result.confirmed
    assert result.attempts == 3
    assert set(result.errors) == {RIGHT}
    assert result.goal_velocity[RIGHT] == 300
    assert bus.goal[LEFT] == 0 and bus.goal[BACK] == 0
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_readback_failure_counts_as_unconfirmed():
    bus = FakeWheelBus(fail_reads={LEFT: -1})
    result = make_controller(bus).stop_base()
    assert not result.confirmed
    assert result.goal_velocity[LEFT] is None
    assert "read" in result.errors[LEFT]


def test_pulse_raises_when_stop_not_confirmed():
    bus = FakeWheelBus(stuck={RIGHT: -565})
    with pytest.raises(BaseStopError):
        make_controller(bus).move_base_for(0.05, 0, 0, 0.3)
    assert bus.goal[LEFT] == 0 and bus.goal[BACK] == 0


def test_command_write_failure_stops_base_and_reraises():
    bus = FakeWheelBus(fail_writes={RIGHT: 1})
    with pytest.raises(ConnectionError):
        make_controller(bus).set_base_velocity(0.05, 0, 0)
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)


def test_context_manager_stops_on_exit():
    bus = FakeWheelBus()
    with pytest.raises(RuntimeError), make_controller(bus) as base:
        base.set_base_velocity(0.05, 0, 0)
        raise RuntimeError("boom")
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)


def test_sigint_during_stop_is_deferred_until_all_wheels_zeroed():
    bus = FakeWheelBus()
    bus.goal = {LEFT: 500, BACK: 500, RIGHT: 500}
    original_write = bus.write

    def write_then_ctrl_c(data_name, motor, value, **kwargs):
        original_write(data_name, motor, value, **kwargs)
        if motor == LEFT:
            os.kill(os.getpid(), signal.SIGINT)

    bus.write = write_then_ctrl_c
    with pytest.raises(KeyboardInterrupt):
        make_controller(bus).stop_base()
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ---------------------------------------------------------------- limits


def test_commands_are_clamped_to_limits():
    bus = FakeWheelBus()
    base = make_controller(bus, limits=BaseLimits(max_vx_mps=0.05))
    assert base.set_base_velocity(5.0, 0, 0) == base.set_base_velocity(0.05, 0, 0)


@pytest.mark.parametrize("duration", [0.0, -1.0, 1.5, float("nan")])
def test_invalid_pulse_duration_commands_no_motion(duration):
    bus = FakeWheelBus()
    with pytest.raises(ValueError):
        make_controller(bus).move_base_for(0.05, 0, 0, duration)
    assert bus.log == []


def test_non_finite_command_stops_base():
    bus = FakeWheelBus()
    bus.goal = {LEFT: 500, BACK: 500, RIGHT: 500}
    with pytest.raises(ValueError):
        make_controller(bus).set_base_velocity(float("nan"), 0, 0)
    assert bus.goal == dict.fromkeys(WHEEL_NAMES, 0)


@pytest.mark.parametrize(
    ("answering", "power_hint"),
    [({LEFT: 2825, BACK: 2825}, False), ({}, True)],
)
def test_open_wheel_bus_rejects_wrong_device(monkeypatch, answering, power_hint):
    from lerobot.motors.feetech import feetech
    from lerobot.robots.alohamini.base import WheelBusMismatchError, open_wheel_bus

    disconnected = []

    class FakeFeetech:
        def __init__(self, port, motors):
            self.motors = motors

        def connect(self, handshake):
            pass

        def ping(self, name, num_retry=0):
            return answering.get(name)

        def disconnect(self, disable_torque):
            disconnected.append(disable_torque)

    monkeypatch.setattr(feetech, "FeetechMotorsBus", FakeFeetech)
    monkeypatch.setattr("lerobot.motors.feetech.FeetechMotorsBus", FakeFeetech)
    with pytest.raises(WheelBusMismatchError, match="not the wheel bus") as excinfo:
        open_wheel_bus("/dev/null", "sts3250")
    assert ("base motor power" in str(excinfo.value)) == power_hint
    assert disconnected == [False]
