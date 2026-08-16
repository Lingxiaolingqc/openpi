# Frozen SO-101 real-hardware calibrations

`calibration_lock.json` identifies the calibration that must be used for the
real-only SO-101 experiment. The active LeIsaac cache remains the runtime
source, while `frozen/<version>/` is an independent, version-controlled snapshot.

The current lock is `so101-real-2026-08-14-all-joints-v3`. It preserves all v2
IDs, drive modes, and homing offsets while replacing the six software range
fields for both arms with reviewed, torque-disabled measurements. The original
`2026-08-13` v1 and `2026-08-14-gripper-v2` snapshots remain immutable for
provenance.

Before any motion, recording, training-data conversion, or policy rollout, run:

```powershell
tmp\leisaac-remote-env\python.exe `
    examples\so101\real\calibration\verify_frozen_calibrations.py
```

`CALIBRATION_LOCK_OK` is required. A mismatch is a hard stop and must not be
accepted automatically. If an arm is intentionally recalibrated, create a new
dated snapshot and freeze ID after reviewing the joint ranges; do not replace
an older snapshot.

The active and snapshot SHA-256 values can differ because their JSON whitespace
or line endings differ. The verifier requires each file to match its own locked
hash and also requires their parsed calibration contents to be identical.

This lock protects calibration files and provenance. It does not replace the
runtime checks for the actual servo homing-offset registers, motor IDs, joint
limits, temperature, tracking error, or no-jump torque enable.

To remeasure all six usable joint ranges without changing servo Homing Offsets
or the active calibration, use `../all_joint_range_audit.py` once for the
Leader and once for the Follower. Successful output is stored under
`candidates/` and must be compared and reviewed before creating a new frozen
version. Do not replace the v1 or v2 snapshots.
