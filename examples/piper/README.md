# AgileX PiPER: MuJoCo and pi0.5

This directory is the integration workspace for fine-tuning pi0.5 and evaluating it in MuJoCo with an AgileX PiPER arm.

## Initial scope

- Start from the `agilex_piper` model in MuJoCo Menagerie.
- Define a PiPER-specific observation and action contract before collecting data.
- Keep the MuJoCo environment and the robot control backend replaceable.
- Record demonstrations, convert them to LeRobot, compute normalization statistics, and fine-tune pi0.5.
- Evaluate checkpoints in closed-loop MuJoCo rollouts.

## Status

This is an initialization scaffold only. It does not yet contain a runnable environment, dataset, trained checkpoint, or real-robot controller.

All checkpoints produced from simulated data are simulation-only until separately validated with real PiPER data and hardware safety gates.
