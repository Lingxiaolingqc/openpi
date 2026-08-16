# Follower no-jump torque-enable audit

- Date: 2026-08-14
- Hardware role: SO-101 Follower
- Port: `COM8`
- Calibration freeze ID: `so101-real-2026-08-13-v1`
- Follower active calibration SHA-256:
  `24EAB28DA65F3C628CA7B9E8A76C8FE02154B164310B4D85D18079F7E5835CFB`
- Command: `examples/so101/real/no_jump_enable_test.py --execute --confirm ENABLE_FOLLOWER_HOLD_CURRENT --duration-s 5`
- Result source: operator-reported terminal output

## Result

```text
NO_JUMP_TEST_PASS max_displacement_deg=0.79 max_temperature_c=29
TORQUE_DISABLED_AND_COM8_CLOSED
```

- Maximum displacement: 0.79 degrees (acceptance limit: 3 degrees)
- Maximum temperature: 29 degrees Celsius (hard limit: 65 degrees Celsius)
- Final torque/port state: torque disabled and COM8 closed
- Outcome: PASS

The commanded goal was the Follower pose measured immediately before torque enable. This gate demonstrates bounded
no-jump enable at one safe interior pose. It does not approve arbitrary targets, Leader/Follower mirroring, or policy
deployment; those remain blocked until their separate low-speed and correspondence tests pass.

## Preceding safe aborts

1. A run with no servo responses stopped during COM8 handshake before any goal or torque write.
2. A run from the fully folded storage pose stopped because shoulder lift, elbow flex, and gripper were within the
   two-degree endpoint margin. It disabled torque and closed COM8 without enabling motion.

These aborts behaved as designed and are not counted as no-jump test failures.
