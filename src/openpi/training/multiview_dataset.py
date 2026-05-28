"""LeRobotDataset variant that decodes only ONE camera per __getitem__.

Two modes, selected at construction time:
  - random (default): pick a random camera each call, for multi-view augmentation
  - fixed (set `fixed_camera_key`): always decode the same specified camera

In both modes the chosen frame is exposed under a stable key (default
`observation/image`) so downstream transforms don't need to know which camera
was picked. The unused cameras are never decoded — same I/O cost as a true
single-camera dataset.
"""

import random

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class RandomCameraLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that decodes only one camera per sample (random or fixed)."""

    # Stable key under which the chosen camera frame is exposed in the returned dict.
    target_key: str = "observation/image"

    def __init__(self, *args, fixed_camera_key: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if fixed_camera_key is not None and fixed_camera_key not in self.meta.video_keys:
            raise ValueError(
                f"fixed_camera_key={fixed_camera_key!r} not in dataset video_keys "
                f"{self.meta.video_keys}"
            )
        self._fixed_camera_key = fixed_camera_key

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        all_video_keys = self.meta.video_keys
        if len(all_video_keys) > 0:
            chosen = (
                self._fixed_camera_key
                if self._fixed_camera_key is not None
                else random.choice(all_video_keys)
            )
            current_ts = item["timestamp"].item()
            if query_indices is not None and chosen in query_indices:
                timestamps = self.hf_dataset.select(query_indices[chosen])["timestamp"]
                query_timestamps = {chosen: torch.stack(timestamps).tolist()}
            else:
                query_timestamps = {chosen: [current_ts]}
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item[self.target_key] = video_frames[chosen]

        if self.image_transforms is not None and self.target_key in item:
            item[self.target_key] = self.image_transforms(item[self.target_key])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item
