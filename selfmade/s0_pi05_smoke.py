import subprocess
import time

import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as training_config


CHECKPOINT = (
    "/home/data/xiaoqinchuan/models/"
    "openpi-assets/checkpoints/pi05_libero"
)


def gpu_memory_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return max(int(line.strip()) for line in output.splitlines())


config = training_config.get_config("pi05_libero")

load_start = time.perf_counter()
policy = policy_config.create_trained_policy(config, CHECKPOINT)
print(f"模型加载时间: {time.perf_counter() - load_start:.2f}s")

observation = libero_policy.make_libero_example()

start = time.perf_counter()
result = policy.infer(observation)
first_inference = time.perf_counter() - start

actions = np.asarray(result["actions"])
assert actions.shape == (config.model.action_horizon, 7), actions.shape
assert np.isfinite(actions).all()

print(f"首次推理/编译时间: {first_inference:.3f}s")
print(f"输出形状: {actions.shape}")

latencies = []
peak_memory = gpu_memory_mib()

for index in range(20):
    start = time.perf_counter()
    result = policy.infer(observation)
    latency = time.perf_counter() - start

    actions = np.asarray(result["actions"])
    assert actions.shape == (config.model.action_horizon, 7)
    assert np.isfinite(actions).all()

    latencies.append(latency)
    peak_memory = max(peak_memory, gpu_memory_mib())
    print(f"{index + 1:02d}/20: {latency * 1000:.1f}ms")

print(f"平均延迟: {np.mean(latencies) * 1000:.1f}ms")
print(f"p50 延迟: {np.percentile(latencies, 50) * 1000:.1f}ms")
print(f"p95 延迟: {np.percentile(latencies, 95) * 1000:.1f}ms")
print(f"峰值 GPU 显存: {peak_memory} MiB")
print("S0 π0.5 离线推理验收通过")