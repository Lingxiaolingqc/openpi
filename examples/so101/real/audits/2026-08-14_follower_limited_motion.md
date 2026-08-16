# Follower limited-motion audit

- Date: 2026-08-14
- Hardware role: SO-101 Follower
- Port: `COM8`
- Calibration freeze ID: `so101-real-2026-08-13-v1`
- Follower active calibration SHA-256:
  `24EAB28DA65F3C628CA7B9E8A76C8FE02154B164310B4D85D18079F7E5835CFB`
- Result source: operator-reported terminal output and Codex-executed local test output

## Summary

- Six joints tested in both directions: 12 successful bounded motions
- Requested displacement per motion: 3 degrees
- Maximum speed: 15 degrees per second
- Control rate: 30 Hz
- Maximum allowed command step: 0.5 degrees
- One posture-dependent safe abort occurred before the successful wrist-flex pair
- Final read-only audit: torque 0 on IDs 1-6, 26-28 degrees Celsius, 5.2 V, status 0, no communication errors
- Overall bounded-motion gate: PASS at the reviewed compact test pose

## Test 1: shoulder pan, positive direction

```text
initial_deg: shoulder_pan=-2.33 shoulder_lift=15.30 elbow_flex=19.82 wrist_flex=67.69 wrist_roll=-11.34 gripper=-29.27
target_deg: shoulder_pan=0.67 shoulder_lift=15.30 elbow_flex=19.82 wrist_flex=67.69 wrist_roll=-11.34 gripper=-29.27
LIMITED_MOTION_TEST_PASS joint=shoulder_pan delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.56 max_non_target_drift_deg=0.18 max_final_error_deg=0.44 max_temperature_c=31
TORQUE_DISABLED_AND_COM8_CLOSED
```

| Metric | Observed | Limit | Result |
|---|---:|---:|---|
| Command step | 0.50 degrees | 0.50 degrees | PASS |
| Tracking error | 1.56 degrees | 5 degrees | PASS |
| Non-target drift | 0.18 degrees | 3 degrees | PASS |
| Return-to-start error | 0.44 degrees | 3 degrees | PASS |
| Temperature | 31 degrees Celsius | below 65 degrees Celsius | PASS |
| Final state | Torque disabled; COM8 closed | Required | PASS |

Automated outcome: PASS. Physical direction, sound, cable behavior, and collision-free motion still require the
operator's visual confirmation before testing the opposite direction or another joint.

## Test 2: shoulder pan, negative direction

```text
initial_deg: shoulder_pan=-1.89 shoulder_lift=15.30 elbow_flex=19.82 wrist_flex=67.69 wrist_roll=-11.52 gripper=-29.27
target_deg: shoulder_pan=-4.89 shoulder_lift=15.30 elbow_flex=19.82 wrist_flex=67.69 wrist_roll=-11.52 gripper=-29.27
LIMITED_MOTION_TEST_PASS joint=shoulder_pan delta_deg=-3.00 max_command_step_deg=0.43 max_tracking_error_deg=1.51 max_non_target_drift_deg=0.18 max_final_error_deg=0.35 max_temperature_c=34
TORQUE_DISABLED_AND_COM8_CLOSED
```

| Metric | Observed | Limit | Result |
|---|---:|---:|---|
| Command step | 0.43 degrees | 0.50 degrees | PASS |
| Tracking error | 1.51 degrees | 5 degrees | PASS |
| Non-target drift | 0.18 degrees | 3 degrees | PASS |
| Return-to-start error | 0.35 degrees | 3 degrees | PASS |
| Temperature | 34 degrees Celsius | below 65 degrees Celsius | PASS |
| Final state | Torque disabled; COM8 closed | Required | PASS |

Automated outcome: PASS. With operator confirmation of the observed physical direction, sound, cable clearance, and
collision-free motion, shoulder pan has passed the bounded 3-degree test in both directions.

## Test 3: wrist roll, positive direction

```text
LIMITED_MOTION_TEST_PASS joint=wrist_roll delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.06 max_non_target_drift_deg=0.09 max_final_error_deg=0.26 max_temperature_c=32
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 4: wrist roll, negative direction

```text
LIMITED_MOTION_TEST_PASS joint=wrist_roll delta_deg=-3.00 max_command_step_deg=0.50 max_tracking_error_deg=0.98 max_non_target_drift_deg=0.18 max_final_error_deg=0.18 max_temperature_c=42
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 5: gripper, positive direction

```text
LIMITED_MOTION_TEST_PASS joint=gripper delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.32 max_non_target_drift_deg=0.00 max_final_error_deg=0.26 max_temperature_c=31
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 6: gripper, negative direction

```text
LIMITED_MOTION_TEST_PASS joint=gripper delta_deg=-3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.31 max_non_target_drift_deg=0.00 max_final_error_deg=0.18 max_temperature_c=29
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 7: wrist flex, positive direction

```text
initial_deg: shoulder_pan=-2.07 shoulder_lift=69.01 elbow_flex=10.42 wrist_flex=-88.88 wrist_roll=-11.08 gripper=-29.27
target_deg: shoulder_pan=-2.07 shoulder_lift=69.01 elbow_flex=10.42 wrist_flex=-85.88 wrist_roll=-11.08 gripper=-29.27
TORQUE_ENABLED_AT_MEASURED_POSE
MOVING_OUTBOUND joint=wrist_flex delta_deg=3.00
TORQUE_DISABLED_AND_COM8_CLOSED
LIMITED_MOTION_TEST_FAILED: RuntimeError: Non-target drift: shoulder_lift=3.16 deg (limit 3.00 deg)
```

Automated outcome: FAIL. The program stopped before the return segment, disabled all Follower torque, and closed COM8.
All later loaded-joint tests were stopped. A separate read-only register check found torque disabled on IDs 1-6,
temperatures of 26-28 degrees Celsius, 5.2 V on all motors, status 0, and no communication errors. The 3-degree drift
limit must not be relaxed without diagnosing the loaded posture, gravity sag, mechanical play, and power capability.

## Test 8: wrist flex, positive-direction retest

The operator moved the arm from the loaded pose to a more compact pose before this retest.

```text
initial_deg: shoulder_pan=-2.59 shoulder_lift=52.57 elbow_flex=27.21 wrist_flex=-66.55 wrist_roll=-10.99 gripper=-29.19
target_deg: shoulder_pan=-2.59 shoulder_lift=52.57 elbow_flex=27.21 wrist_flex=-63.55 wrist_roll=-10.99 gripper=-29.19
LIMITED_MOTION_TEST_PASS joint=wrist_flex delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.42 max_non_target_drift_deg=0.53 max_final_error_deg=0.26 max_temperature_c=30
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS. The non-target drift decreased from 3.16 degrees in the loaded pose to 0.53 degrees.

## Test 9: wrist flex, negative direction

```text
LIMITED_MOTION_TEST_PASS joint=wrist_flex delta_deg=-3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.23 max_non_target_drift_deg=0.62 max_final_error_deg=0.53 max_temperature_c=34
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 10: elbow flex, positive direction

```text
LIMITED_MOTION_TEST_PASS joint=elbow_flex delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.68 max_non_target_drift_deg=0.62 max_final_error_deg=0.18 max_temperature_c=36
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 11: elbow flex, negative direction

```text
LIMITED_MOTION_TEST_PASS joint=elbow_flex delta_deg=-3.00 max_command_step_deg=0.50 max_tracking_error_deg=1.15 max_non_target_drift_deg=0.44 max_final_error_deg=0.44 max_temperature_c=30
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 12: shoulder lift, positive direction

```text
LIMITED_MOTION_TEST_PASS joint=shoulder_lift delta_deg=3.00 max_command_step_deg=0.50 max_tracking_error_deg=2.12 max_non_target_drift_deg=0.26 max_final_error_deg=0.35 max_temperature_c=47
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Test 13: shoulder lift, negative direction

```text
LIMITED_MOTION_TEST_PASS joint=shoulder_lift delta_deg=-3.00 max_command_step_deg=0.50 max_tracking_error_deg=2.03 max_non_target_drift_deg=0.62 max_final_error_deg=0.09 max_temperature_c=41
TORQUE_DISABLED_AND_COM8_CLOSED
```

Automated outcome: PASS.

## Final read-only state

After all bounded motions, a separate low-level register audit reported for every motor ID 1-6:

- `Torque_Enable=0`
- `Present_Voltage=5.2 V`
- `Present_Temperature=26-28 C`
- `Status=0`
- no communication errors

No further motion was performed after this audit.
