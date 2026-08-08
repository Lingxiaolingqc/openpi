"""Headless SO-101 Leader teleoperation with a browser preview and controls.

The physical Leader publishes joint states on Windows. This server-side script
receives them through ``SO101LeaderRemote``, applies the official LeIsaac action
mapping, and exposes the front camera through an MJPEG page bound to localhost.

By default this is a preview-only tool. Passing ``--dataset_file`` enables
LeIsaac's native streaming HDF5 recorder without importing LeRobot. Use an SSH
local forward for the preview port and a reverse forward for Leader states.
"""

from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import threading
import time
import traceback
from urllib.parse import urlparse

from isaaclab.app import AppLauncher

_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 LeIsaac Preview</title>
  <style>
    body { font-family: sans-serif; margin: 1rem; background: #15171a; color: #f4f4f4; }
    main { max-width: 920px; margin: auto; }
    img { display: block; width: 100%; max-width: 800px; background: #000; }
    button { margin: .75rem .5rem .75rem 0; padding: .7rem 1rem; font-size: 1rem; }
    #start { background: #2e7d32; color: white; }
    #success { background: #1565c0; color: white; }
    #discard { background: #c62828; color: white; }
    #stop { background: #555; color: white; }
    pre { padding: .75rem; background: #22262a; white-space: pre-wrap; }
  </style>
</head>
<body>
<main>
  <h1>SO-101 LeIsaac preview</h1>
  <p>HDF5 recording is enabled only when the server is launched with
     <code>--dataset_file</code>. The status block below is authoritative.</p>
  <img src="/stream.mjpg" alt="LeIsaac front camera">
  <div>
    <button id="start" onclick="sendCommand('start')">Start / Resume</button>
    <button id="success" onclick="sendCommand('success')">Success + Reset</button>
    <button id="discard" onclick="sendCommand('discard')">Discard + Reset</button>
    <button id="stop" onclick="sendCommand('stop')">Stop Server</button>
  </div>
  <pre id="status">Connecting...</pre>
</main>
<script>
async function sendCommand(name) {
  await fetch('/command/' + name, {method: 'POST'});
  await refresh();
}
async function refresh() {
  try {
    const response = await fetch('/status.json', {cache: 'no-store'});
    document.getElementById('status').textContent = JSON.stringify(await response.json(), null, 2);
  } catch (error) {
    document.getElementById('status').textContent = String(error);
  }
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""

_COMMANDS = frozenset({"start", "success", "discard", "stop"})


class PreviewState:
    """Thread-safe frames, commands, and status shared with the HTTP server."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._frame_version = 0
        self._commands: deque[str] = deque()
        self._status: dict[str, object] = {
            "phase": "starting",
            "recording_enabled": False,
            "recording": False,
            "dataset_file": None,
            "completed_steps": 0,
            "successful_resets": 0,
            "discarded_resets": 0,
        }

    def update_frame(self, frame: bytes) -> None:
        with self._condition:
            self._frame = frame
            self._frame_version += 1
            self._condition.notify_all()

    def wait_for_frame(self, version: int, timeout: float) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._frame_version > version, timeout=timeout)
            return self._frame_version, self._frame

    def queue_command(self, command: str) -> None:
        if command not in _COMMANDS:
            raise ValueError(f"Unsupported preview command: {command}")
        with self._condition:
            self._commands.append(command)
            self._condition.notify_all()

    def pop_commands(self) -> list[str]:
        with self._condition:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    def update_status(self, **values: object) -> None:
        with self._condition:
            self._status.update(values)

    def status(self) -> dict[str, object]:
        with self._condition:
            return dict(self._status)


class PreviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_handler(state: PreviewState) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_bytes(_HTML, "text/html; charset=utf-8")
            elif path == "/status.json":
                payload = json.dumps(state.status(), sort_keys=True).encode()
                self._send_bytes(payload, "application/json")
            elif path == "/stream.mjpg":
                self._stream_frames()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            prefix = "/command/"
            if not path.startswith(prefix) or path[len(prefix) :] not in _COMMANDS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            state.queue_command(path[len(prefix) :])
            self._send_bytes(b'{"queued":true}', "application/json")

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_frames(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = -1
            try:
                while True:
                    version, frame = state.wait_for_frame(version, timeout=2.0)
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return PreviewHandler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LeIsaac-SO101-LiftCube-v0")
    parser.add_argument("--remote_endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--assets_root", default=os.environ.get("LEISAAC_ASSETS_ROOT"))
    parser.add_argument("--web_host", default="127.0.0.1")
    parser.add_argument("--web_port", type=int, default=5557)
    parser.add_argument("--step_hz", type=float, default=60.0)
    parser.add_argument("--preview_fps", type=float, default=15.0)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--receive_timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset_file",
        help="Enable native LeIsaac HDF5 recording at this path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to an existing --dataset_file.",
    )
    parser.add_argument(
        "--flush_steps",
        type=int,
        default=100,
        help="Number of recorded steps between streaming HDF5 flushes.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _wait_for_leader_state(teleop_interface, timeout: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = teleop_interface.get_device_state()
        if any(abs(value) > 1.0e-6 for value in state.values()):
            return state
        time.sleep(0.05)
    raise TimeoutError(f"No non-zero Leader frame received within {timeout:.1f}s")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.headless:
        parser.error("This remote web teleoperation tool requires --headless")
    if not args.enable_cameras:
        parser.error("The LiftCube environment requires --enable_cameras")
    if not args.assets_root:
        parser.error("Set LEISAAC_ASSETS_ROOT or pass --assets_root")
    if args.step_hz <= 0 or args.preview_fps <= 0:
        parser.error("--step_hz and --preview_fps must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("--jpeg_quality must be between 1 and 95")
    if not 1 <= args.web_port <= 65535:
        parser.error("--web_port must be between 1 and 65535")
    if args.flush_steps <= 0:
        parser.error("--flush_steps must be positive")
    if args.resume and not args.dataset_file:
        parser.error("--resume requires --dataset_file")

    assets_root = Path(args.assets_root).expanduser().resolve()
    if not assets_root.is_dir():
        parser.error(f"Assets root does not exist: {assets_root}")
    os.environ["LEISAAC_ASSETS_ROOT"] = str(assets_root)

    dataset_path = Path(args.dataset_file).expanduser().resolve() if args.dataset_file else None
    if dataset_path is not None and not str(dataset_path).endswith(".hdf5"):
        dataset_path = Path(f"{dataset_path}.hdf5")
    if dataset_path is not None:
        if args.resume and not dataset_path.is_file():
            parser.error(f"Cannot resume because the dataset does not exist: {dataset_path}")
        if not args.resume and dataset_path.exists():
            parser.error(f"Dataset already exists; pass --resume to append: {dataset_path}")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

    print("REMOTE_WEB_TELEOP_PHASE=before_launcher", flush=True)
    print(f"task_id: {args.task}", flush=True)
    print(f"remote_endpoint: {args.remote_endpoint}", flush=True)
    print(f"preview_url: http://{args.web_host}:{args.web_port}", flush=True)
    print(f"requested_device: {args.device}", flush=True)
    print(f"recording_enabled: {dataset_path is not None}", flush=True)
    print(f"dataset_file: {dataset_path}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim must be launched before importing the remaining simulation modules.
    # isort: off
    import gymnasium as gym
    from PIL import Image
    import torch
    from isaaclab.managers import DatasetExportMode, TerminationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg
    import leisaac.tasks  # noqa: F401
    from leisaac.devices import SO101LeaderRemote
    from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
    from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
    # isort: on

    state = PreviewState()
    server = None
    server_thread = None
    teleop_interface = None
    env = None
    status = 1

    state.update_status(
        recording_enabled=dataset_path is not None,
        dataset_file=str(dataset_path) if dataset_path is not None else None,
    )

    def encode_front(observations: dict) -> bytes:
        front = observations["policy"]["front"]
        if front.ndim != 4 or front.shape[0] != 1 or front.shape[-1] != 3:
            raise RuntimeError(f"Unexpected front camera shape: {tuple(front.shape)}")
        array = front[0].detach().cpu().numpy()
        output = io.BytesIO()
        Image.fromarray(array).save(output, format="JPEG", quality=args.jpeg_quality)
        return output.getvalue()

    try:
        print("REMOTE_WEB_TELEOP_PHASE=app_ready", flush=True)
        print(f"app_launcher_device_id: {app_launcher.device_id}", flush=True)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.use_teleop_device("so101leader")
        env_cfg.seed = args.seed
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

        if dataset_path is None:
            env_cfg.recorders = None
            if hasattr(env_cfg.terminations, "success"):
                env_cfg.terminations.success = None
        else:
            env_cfg.recorders.dataset_export_mode = (
                EnhanceDatasetExportMode.EXPORT_ALL_RESUME if args.resume else DatasetExportMode.EXPORT_ALL
            )
            env_cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
            env_cfg.recorders.dataset_filename = dataset_path.stem
            if not hasattr(env_cfg.terminations, "success"):
                env_cfg.terminations.success = None
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        if dataset_path is not None:
            del env.recorder_manager
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = args.flush_steps
            env.recorder_manager.compression = "lzf"
            print("REMOTE_WEB_TELEOP_HDF5_RECORDER_READY", flush=True)

        def set_episode_success(*, success: bool) -> None:
            if dataset_path is None:
                return
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(
                    func=(
                        (lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                        if success
                        else (lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
                    )
                ),
            )
            env.termination_manager.compute()

        observations, _ = env.reset()
        state.update_frame(encode_front(observations))
        print("REMOTE_WEB_TELEOP_ENV_CREATED_OK", flush=True)
        print(f"simulation_device: {env.device}", flush=True)

        teleop_interface = SO101LeaderRemote(env, endpoint=args.remote_endpoint)
        leader_state = _wait_for_leader_state(teleop_interface, args.receive_timeout)
        print(
            "leader_state_first:",
            {name: round(value, 3) for name, value in leader_state.items()},
            flush=True,
        )

        server = PreviewHTTPServer((args.web_host, args.web_port), _make_handler(state))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        running = False
        interrupted = False
        episode_active = False
        completed_steps = 0
        successful_resets = 0
        discarded_resets = 0
        next_step_time = time.monotonic()
        next_preview_time = time.monotonic()
        state.update_status(phase="ready")
        print(f"REMOTE_WEB_TELEOP_READY http://{args.web_host}:{args.web_port}", flush=True)

        while simulation_app.is_running() and not interrupted:
            for command in state.pop_commands():
                if command == "start":
                    running = True
                    state.update_status(phase="running", recording=dataset_path is not None)
                    print("REMOTE_WEB_TELEOP_START", flush=True)
                elif command in {"success", "discard"}:
                    if episode_active:
                        set_episode_success(success=command == "success")
                    observations, _ = env.reset()
                    if episode_active:
                        set_episode_success(success=False)
                    state.update_frame(encode_front(observations))
                    running = False
                    episode_active = False
                    if command == "success":
                        successful_resets += 1
                        print("REMOTE_WEB_TELEOP_SUCCESS_RESET", flush=True)
                    else:
                        discarded_resets += 1
                        print("REMOTE_WEB_TELEOP_DISCARD_RESET", flush=True)
                    state.update_status(
                        phase="ready",
                        recording=False,
                        successful_resets=successful_resets,
                        discarded_resets=discarded_resets,
                    )
                elif command == "stop":
                    if episode_active:
                        set_episode_success(success=False)
                        env.reset()
                        set_episode_success(success=False)
                        episode_active = False
                        discarded_resets += 1
                        print("REMOTE_WEB_TELEOP_STOP_DISCARD_RESET", flush=True)
                    interrupted = True
                    state.update_status(
                        phase="stopping",
                        recording=False,
                        discarded_resets=discarded_resets,
                    )
                    print("REMOTE_WEB_TELEOP_STOP", flush=True)

            if interrupted:
                break

            if not running:
                time.sleep(0.02)
                continue

            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, "so101leader")
            action_request = teleop_interface.input2action()
            action = env.cfg.preprocess_device_action(action_request, teleop_interface)
            if action.shape != (1, 6) or not bool(torch.isfinite(action).all()):
                raise RuntimeError(f"Invalid remote Leader action: shape={tuple(action.shape)}")

            step_result = env.step(action)
            episode_active = True
            observations = step_result[0]
            rewards = step_result[1]
            if not bool(torch.isfinite(rewards).all()):
                raise RuntimeError("A non-finite reward was observed")
            completed_steps += 1

            now = time.monotonic()
            if now >= next_preview_time:
                state.update_frame(encode_front(observations))
                next_preview_time = now + 1.0 / args.preview_fps
            state.update_status(
                phase="running",
                recording=dataset_path is not None,
                completed_steps=completed_steps,
            )

            next_step_time += 1.0 / args.step_hz
            delay = next_step_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step_time = time.monotonic()

        state.update_status(phase="stopped", recording=False)
        print(f"completed_steps: {completed_steps}", flush=True)
        print(f"successful_resets: {successful_resets}", flush=True)
        print(f"discarded_resets: {discarded_resets}", flush=True)
        print("REMOTE_WEB_TELEOP_OK", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
        state.update_status(phase="failed")
        print("REMOTE_WEB_TELEOP_FAILED", flush=True)
    finally:
        if env is not None and dataset_path is not None and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
            print("REMOTE_WEB_TELEOP_HDF5_FINALIZED", flush=True)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2)
        if teleop_interface is not None:
            teleop_interface.disconnect()
        print("REMOTE_WEB_TELEOP_PHASE=immediate_close", flush=True)
        simulation_app.close(skip_cleanup=True)

    return status


if __name__ == "__main__":
    raise SystemExit(main())
