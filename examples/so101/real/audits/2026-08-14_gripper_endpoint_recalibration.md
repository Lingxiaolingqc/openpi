# Gripper endpoint audit and targeted recalibration

- Date: 2026-08-14
- Scope: Leader COM7 and Follower COM8 gripper range endpoints only
- Motor writes: none
- Homing-offset changes: none
- Previous freeze: `so101-real-2026-08-13-v1`
- New freeze: `so101-real-2026-08-14-gripper-v2`

## Read-only endpoint measurements

Both arms were torque-disabled. Each endpoint was held still while 30 raw `Present_Position` samples were collected.
Every 30-sample set had a repeatability span of zero raw counts.

| Arm | Endpoint | Observed raw | Frozen v1 raw | Difference |
|---|---|---:|---:|---:|
| Leader | Gentle closed | 2043 | 2026 | +17 raw / +1.49 degrees |
| Leader | Fully open | 3268 | 3354 | -86 raw / -7.56 degrees |
| Follower | Gentle closed | 2143 | 2042 | +101 raw / +8.88 degrees |
| Follower | Fully open | 3506 | 3528 | -22 raw / -1.93 degrees |

The v1 ranges extended beyond the repeatable gentle mechanical endpoints, especially at the Follower closed end and
Leader open end. Direct centered-degree comparison also hid a 7.21-degree mapped Follower gripper mismatch because the
two arms have different gripper spans.

## v2 change

Only these software calibration fields changed:

| Arm | v1 gripper range | v2 gripper range |
|---|---:|---:|
| Leader | 2026-3354 | 2043-3268 |
| Follower | 2042-3528 | 2143-3506 |

All IDs, drive modes, homing offsets, and the five rotary-joint ranges remain byte-for-byte equivalent as parsed JSON
values. The real-hardware mapping uses the Leader's calibrated range fraction to produce an absolute Follower target;
equal endpoint percentages do not require identical physical jaw gaps.

## Communication observation

The first fully-open audit handshake temporarily missed Follower ID6. A subsequent low-level check read model 777 and
raw position 3506 successfully five times, and the repeated 30-sample audit then completed. Inspect and strain-relieve
the wrist/gripper cable before camera mounting and continuous teleoperation.
