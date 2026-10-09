# AlohaMini2 Pro — Table Base Alignment Specification

## Objective

Implement a repeatable, safe **base-alignment** behavior before the robot adjusts its lift and arms to clear cups from a table.

```text
Navigate near table
  → Estimate table-relative pose
  → Align heading
  → Align lateral position
  → Align distance
  → Verify and stop
  → BASE_READY
  → (future) Lift / arm TABLE_READY → cup manipulation
```

**Current scope:** mobile base only. Do not implement lift, arm control, grasping, or end-to-end learning yet.

## Hardware and existing code

- Three-omniwheel holonomic base with commands `vx` (forward/backward), `vy` (left/right), and `omega` (yaw rate).
- Reuse the repository's existing `body_to_wheel_raw(x_cmd, y_cmd, theta_cmd_degps)` rather than replacing the kinematics.
- Confirm the actual wheel-bus device before operating; arm serial buses must not be treated as wheel buses.
- The working SSH test has used wheel commands such as `{'left_wheel': 565, 'back_wheel': 0, 'right_wheel': -565}` for `vx=0.05 m/s`.
- Existing failure: `stop()` used a list-valued `bus.write()` call and raised `unhashable type: 'list'`; the robot continued moving after Ctrl+C. **Fix and test stopping before any alignment motion.**

### Manual controls

| Key | Action |
|---|---|
| W | Forward |
| S | Backward |
| A | Strafe left |
| D | Strafe right |
| Q | Rotate counterclockwise |
| E | Rotate clockwise |
| X | Exit |

Initial conservative commands: `0.05 m/s`, `15 deg/s`, `0.3 s` pulses. These are requested values, not guarantees of displacement.

## Safety-critical base interface

Provide:

```python
set_base_velocity(vx: float, vy: float, omega_degps: float)
stop_base()
move_base_for(vx: float, vy: float, omega_degps: float, duration_s: float)
```

The observed working API writes each wheel individually:

```python
for name, value in wheel_commands.items():
    bus.write("Goal_Velocity", name, value, normalize=False)
```

Stop each wheel individually:

```python
for name in motors:
    bus.write("Goal_Velocity", name, 0, normalize=False)
```

Requirements:

- Send zero velocity in `finally` after each timed movement, on normal exit, Ctrl+C, exceptions, and stale/lost perception.
- Log every failed stop; do not silently swallow exceptions.
- Verify all three zero-velocity writes, and keep an independent physical emergency stop/power cutoff available.
- Software cleanup alone is **not** a reliable emergency stop if the Pi crashes, SSH disconnects, or the motor bus fails. Investigate a motor/controller watchdog or hardware-level fail-safe before autonomous operation.
- Cap speed, pulse duration, and workspace; abort on sensor timeout or invalid pose.
- Test in clear floor space, away from the table, with a person ready to cut power.

## Table-relative target pose

Define a coordinate frame fixed to the table's front edge and a calibrated robot-base reference point. Express the desired pose with:

- `D_target`: desired clearance from robot reference to front table edge, measured experimentally (do not assume 0.30 m).
- `L_target`: desired lateral alignment to the intended cup-working region.
- `theta_target`: heading aligned to the table.

Estimator output:

```python
@dataclass
class TablePoseError:
    distance_error_m: float
    lateral_error_m: float
    heading_error_deg: float
    timestamp_s: float
    valid: bool
```

Specify signs explicitly and verify them physically: positive distance error should produce motion toward the table; positive lateral error should produce the appropriate strafe; positive heading error should produce the appropriate turn. These sign conventions depend on the camera/base transforms.

Example initial, adjustable tolerances:

```python
DISTANCE_TOLERANCE_M = 0.02
LATERAL_TOLERANCE_M = 0.02
HEADING_TOLERANCE_DEG = 2.0
```

`BASE_READY` requires **all three errors within tolerance**, fresh valid observations, and a confirmed stop command. Ideally verify stable alignment across several consecutive measurements.

## Alignment state machine

```text
IDLE
  ↓
ESTIMATE_TABLE_POSE
  ↓
ALIGN_HEADING      (omega)
  ↓
ALIGN_LATERAL      (vy)
  ↓
ALIGN_DISTANCE     (vx)
  ↓
VERIFY_ALIGNMENT
  ├── outside tolerance → return to needed alignment stage
  ├── invalid/stale pose or timeout → STOP / FAULT
  └── all within tolerance → STOP → BASE_READY
```

Use closed-loop corrections: observe, command a bounded short movement, stop, observe again. Avoid long open-loop motions near the table. Each correction may change other pose errors, so re-evaluate all three during verification.

A first proportional controller can use:

```python
vx = clamp(K_distance * distance_error_m, -MAX_VX, MAX_VX)
vy = clamp(K_lateral * lateral_error_m, -MAX_VY, MAX_VY)
omega = clamp(K_heading * heading_error_deg, -MAX_OMEGA, MAX_OMEGA)
```

Only enable the relevant axis in each stage initially. Tune gain signs experimentally at low speed. Slow down near target; enforce a minimum safe clearance and collision limits independently of the controller.

## Perception abstraction

Keep table pose estimation separate from motor control:

```python
class TablePoseEstimator:
    def get_table_pose_error(self) -> TablePoseError:
        ...
```

First implementation: an AprilTag fixed to the table, with calibrated camera intrinsics, camera-to-base extrinsics, and tag-to-table transform. Convert observed tag pose into table pose in the base frame. Do not treat raw image pixel offset as metric displacement.

Later replace the estimator with RGB-D/table-plane or learned table detection while preserving the same base controller interface.

## Suggested project organization

```text
base/
  base_controller.py
  omni_kinematics.py
perception/
  table_pose_estimator.py
  apriltag_table_estimator.py
behaviors/
  align_to_table.py
config/
  table_alignment.py
```

## Implementation milestones and acceptance tests

1. **Motor stop fix:** Confirm a 0.3 s forward pulse ends with all wheel velocities set to zero. Ctrl+C and exceptions must also stop the base. Do not continue if the base keeps moving.
2. **Direction verification:** Confirm W/S/A/D/Q/E physically match the documented coordinate system.
3. **Perception only:** Print distance, lateral, and heading errors at several stationary base poses; verify signs and units.
4. **Heading only:** Correct heading at low speed, stop, remeasure.
5. **Lateral only:** Correct side-to-side offset without unnecessary turning.
6. **Distance only:** Correct front-edge clearance, with hard approach limit.
7. **Full alignment:** Start from several nearby poses, converge to `BASE_READY`, and remain stationary.
8. **Fault injection:** Test tag loss, stale frames, serial errors, Ctrl+C, and timeout; ensure the system fails safely.

Log target pose, measured errors, commanded body velocities, per-wheel commands, stop outcomes, and state transitions for each trial.

## Final behavior (future scope)

```text
Navigate → Align base → BASE_READY → Set lift → TABLE_READY arms
         → Detect cups → Grasp/place → Repeat until table clear
```

**Immediate deliverable:** a safe, testable, modular base-alignment subsystem. Do not proceed to arm or lift automation until base stopping and repeatable alignment are validated.
