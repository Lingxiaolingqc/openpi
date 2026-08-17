from __future__ import annotations

from typing import ClassVar

import numpy as np

from examples.so101.act import audit_dataset


class _FakeHFDataset:
    column_names: ClassVar[list[str]] = [
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
    ]

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def select_columns(self, columns):
        return self

    def set_format(self, **kwargs) -> None:
        pass

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            rows = self.rows[item]
            return {key: np.asarray([row[key] for row in rows]) for key in self.column_names}
        return self.rows[item]


class _FakeDataset:
    def __init__(self) -> None:
        self.hf_dataset = _FakeHFDataset(
            [
                {
                    "observation.state": np.zeros(6),
                    "action": np.zeros(6),
                    "episode_index": 0,
                    "frame_index": 0,
                    "timestamp": 0.0,
                },
                {
                    "observation.state": np.ones(6),
                    "action": np.array([101.0, 0, 0, 0, 0, 50]),
                    "episode_index": 0,
                    "frame_index": 1,
                    "timestamp": 1 / 60,
                },
                {
                    "observation.state": np.zeros(6),
                    "action": np.zeros(6),
                    "episode_index": 1,
                    "frame_index": 0,
                    "timestamp": 0.0,
                },
                {
                    "observation.state": np.ones(6),
                    "action": np.ones(6),
                    "episode_index": 1,
                    "frame_index": 1,
                    "timestamp": 1 / 60,
                },
            ]
        )

    def __getitem__(self, index):
        return {"observation.images.front": np.arange(3 * 4 * 5).reshape(3, 4, 5)}


def test_numeric_audit_reports_per_joint_violations_without_clipping() -> None:
    dataset = _FakeDataset()

    report = audit_dataset._scan_numeric_columns(dataset, fps=60, batch_size=3)  # noqa: SLF001

    assert report.frame_count == 4
    assert report.episode_ids == {0, 1}
    assert report.action_violation_count.tolist() == [1, 0, 0, 0, 0, 0]
    assert report.action_max.tolist()[0] == 101.0
    assert report.non_monotonic_timestamp_count == 0
    assert report.off_rate_timestamp_count == 0


def test_frame0_camera_audit_is_dynamic() -> None:
    result = audit_dataset._decode_frame0_samples(  # noqa: SLF001
        _FakeDataset(),
        camera_keys=("observation.images.front",),
        frame0_indices={0: 0, 1: 2},
        maximum_episodes=0,
    )

    assert result["observation.images.front"]["decoded_episode_count"] == 2
    assert result["observation.images.front"]["all_black_episode_count"] == 0
    assert result["observation.images.front"]["observed_shapes"] == [(3, 4, 5)]
