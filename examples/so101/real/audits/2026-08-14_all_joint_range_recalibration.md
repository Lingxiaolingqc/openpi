# All-joint range-only recalibration

- Date: 2026-08-14
- Scope: Leader COM7 and Follower COM8, all six software ranges
- New freeze: `so101-real-2026-08-14-all-joints-v3`
- Previous freeze: `so101-real-2026-08-14-gripper-v2`
- Motor, torque, goal-position, and calibration-register writes during capture: none
- IDs, drive modes, and Homing Offsets changed: none

## Accepted ranges

Both arms were powered, torque-disabled, and moved by hand through gentle usable
endpoints. The first five ranges come from each full-arm capture. Gripper
endpoints were then repeated for three cycles in an isolated capture.

| Joint | Leader v2 | Leader v3 | Follower v2 | Follower v3 |
|---|---:|---:|---:|---:|
| shoulder_pan | 688-3443 | 708-3428 | 697-3454 | 719-3407 |
| shoulder_lift | 823-3245 | 832-3221 | 838-3240 | 842-3218 |
| elbow_flex | 883-3086 | 880-3087 | 849-3060 | 854-3055 |
| wrist_flex | 852-3190 | 861-3184 | 863-3227 | 880-3209 |
| wrist_roll | 120-3965 | 140-3961 | 230-4088 | 246-4064 |
| gripper | 2043-3268 | 2040-3296 | 2143-3506 | 2046-3511 |

The accepted Leader gripper range is the conservative intersection of the new
full-arm result `2037-3296` and isolated result `2040-3312`. The accepted
Follower gripper range is the conservative intersection of `2039-3517` and
`2046-3511`. The Follower repeat confirmed that the v2 closed endpoint `2143`
did not represent the repeatable natural endpoint.

## Capture evidence

| Capture | Candidate SHA-256 | Audit SHA-256 | Samples | Max temperature |
|---|---|---|---:|---:|
| Leader all joints | `B86F03AB9A2A7703D4B97AEF821DB8BB21C025DD87E8B6B38F0E35D20A0BC171` | `1692F5725392532A1121CE54EDD6ADD88FADF4C505F6459F43E67B3D442DE0F1` | 913 | 39 C |
| Leader gripper repeat | `EF37624970E002594BB5FD63977B2C70007CE9C32D35A05B6BF4DADD6A64A247` | `D5F339D86DC5F0037F4570786B37FAF95B799E85098B4AC3EB7C3DD100A294E8` | 281 | 32 C |
| Follower all joints | `306D4FF97E2F2E8AD4C702B833F4362820027F6FA4A9D57ED7F01A97E6865425` | `1A2A7E6B5E65015E8CE867247CEE6912168549AE08AD17E7E57E5384967EA48D` | 754 | 35 C |
| Follower gripper repeat | `3DB7CAB0BB005B67D220974EACC124BC03D3142237B2414F916FB1156747646F` | `779932AC88DD8FC767FB95BE7CBE8262EE7583AD355E190A211C85793EA492D6` | 301 | 34 C |

The candidate files remain evidence only. The accepted merged calibrations are
stored under `calibration/frozen/2026-08-14-all-joints-v3/` and protected by
`calibration_lock.json`. Before further motion, v3 must pass the no-jump test,
bounded motion tests, and the mapped dual-arm alignment gate again.

## Post-freeze validation

The v3 Follower no-jump test passed from the measured pose before any bounded
motion retest:

- duration: 5.00 s
- maximum displacement: 0.35 degrees
- maximum temperature: 29 C
- final state: torque disabled and COM8 closed
- terminal result: `NO_JUMP_TEST_PASS`

All twelve positive and negative three-degree bounded motion tests subsequently
passed at 15 degrees/second and 30 Hz:

| Joint | Delta | Max step | Tracking error | Non-target drift | Final error | Max temperature |
|---|---:|---:|---:|---:|---:|---:|
| shoulder_pan | +3 deg | 0.50 deg | 1.47 deg | 0.00 deg | 0.44 deg | 34 C |
| shoulder_pan | -3 deg | 0.50 deg | 1.61 deg | 0.00 deg | 0.44 deg | 29 C |
| shoulder_lift | +3 deg | 0.50 deg | 1.86 deg | 0.53 deg | 0.09 deg | 42 C |
| shoulder_lift | -3 deg | 0.50 deg | 1.42 deg | 0.53 deg | 0.26 deg | 35 C |
| elbow_flex | +3 deg | 0.50 deg | 1.86 deg | 0.26 deg | 0.35 deg | 37 C |
| elbow_flex | -3 deg | 0.50 deg | 1.77 deg | 0.62 deg | 0.26 deg | 30 C |
| wrist_flex | +3 deg | 0.50 deg | 1.42 deg | 0.44 deg | 0.09 deg | 35 C |
| wrist_flex | -3 deg | 0.50 deg | 1.20 deg | 0.62 deg | 0.44 deg | 42 C |
| wrist_roll | +3 deg | 0.50 deg | 1.06 deg | 0.09 deg | 0.18 deg | 37 C |
| wrist_roll | -3 deg | 0.50 deg | 0.96 deg | 0.18 deg | 0.18 deg | 30 C |
| gripper | +3 deg | 0.50 deg | 1.32 deg | 0.00 deg | 0.26 deg | 32 C |
| gripper | -3 deg | 0.50 deg | 1.23 deg | 0.09 deg | 0.18 deg | 39 C |

The global maxima were 1.86 degrees tracking error, 0.62 degrees
non-target drift, 0.44 degrees final error, and 42 C. Every run ended with
torque disabled and COM8 closed. The remaining v3 gate is the mapped
Leader/Follower alignment test.

The final torque-disabled mapped alignment gate also passed for 10 consecutive
samples over 2.00 seconds. Both ports closed normally without register writes.

| Joint | Mapped target - Follower error |
|---|---:|
| shoulder_pan | -2.96 deg |
| shoulder_lift | +0.17 deg |
| elbow_flex | -2.19 deg |
| wrist_flex | -0.37 deg |
| wrist_roll | -0.66 deg |
| gripper | +0.89 deg |

The maximum absolute error was 2.96 degrees at `shoulder_pan`, just inside the
3.00-degree threshold, and the maximum observed temperature was 30 C. The test
pose was the folded endpoint pose, so this result validates static mapping but
does not authorize torque enable or teleoperation from that pose. A future
torque-enabled start must first place the Follower and mapped Leader pose inside
the configured endpoint margins.

A second mapped alignment gate was then completed in an approximately centered
pose. The operator's live monitor and an independent 30-sample invocation both
reached 10 consecutive passing samples over 2.00 seconds. The independent run
observed the following final errors:

| Joint | Mapped target - Follower error |
|---|---:|
| shoulder_pan | +1.74 deg |
| shoulder_lift | -0.85 deg |
| elbow_flex | -2.28 deg |
| wrist_flex | -0.09 deg |
| wrist_roll | -2.15 deg |
| gripper | -1.50 deg |

The maximum absolute error was 2.28 degrees at `elbow_flex`; the maximum
observed temperature was 30 C. Calibration hashes matched freeze
`so101-real-2026-08-14-all-joints-v3`, both arms remained torque-disabled, and
COM7/COM8 closed normally. This clears the centered static alignment gate only;
bounded Leader-to-Follower dynamic validation remains required before ordinary
teleoperation.

## First bounded Leader-to-Follower validation

The first torque-enabled Leader-to-Follower test passed for `shoulder_pan` in
an approximately centered pose. The Follower was enabled only after its
measured pose had been written as the initial goal. During the eight-second
window, only `shoulder_pan` followed the Leader relative to the validated start
pose; the other five Follower joints held their measured start positions.

| Metric | Result | Test limit |
|---|---:|---:|
| Initial mapped alignment error | 1.99 deg | 3.00 deg |
| Peak Leader excursion | 1.82 deg | 3.00 deg |
| Peak Follower excursion | 1.32 deg | bounded by Leader input |
| Maximum command step | 0.50 deg | 0.50 deg/tick |
| Maximum tracking error | 1.74 deg | 5.00 deg |
| Maximum non-target drift | 0.00 deg | 3.00 deg |
| Maximum temperature | 41 C | below 65 C |

The sampled movement included both negative and positive Leader deltas, with
the Follower response changing in the corresponding direction. The Follower
then returned to its measured start pose. The terminal ended with both
`BOUNDED_LEADER_FOLLOW_PASS` and
`FOLLOWER_TORQUE_DISABLED_AND_PORTS_CLOSED`.

This clears the first single-joint dynamic gate. It does not yet authorize
unbounded or all-joint teleoperation; the remaining joints must be introduced
progressively under the same shared executor and safety limits.
