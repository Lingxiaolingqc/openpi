"""Compatibility entrypoint for the camera-warmed RedCubeToBox policy rollout."""

import sys

from red_cube_to_box_policy_rollout import main


if __name__ == "__main__":
    if "--reset_camera_warmup_steps" not in sys.argv:
        sys.argv.extend(("--reset_camera_warmup_steps", "1"))
    raise SystemExit(main())
