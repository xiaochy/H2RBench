"""
Dataset for phantom-processed demonstrations stored as Zarr.

Expected zarr layout:
  data/
    image       (T, H, W, 3)  uint8
    agent_pos   (T, D_obs)    float32
    action      (T, D_act)    float32
  meta/
    episode_ends (N_eps,)     int64
"""

import torch
import numpy as np
import zarr
from typing import Dict

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class PhantomImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path: str,
                 horizon: int = 16,
                 pad_before: int = 1,
                 pad_after: int = 7,
                 seed: int = 42,
                 val_ratio: float = 0.02,
                 max_train_episodes: int = None,
                 demo_start_idx: int = None,
                 demo_end_idx: int = None):

        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['image', 'agent_pos', 'action'])

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # restrict to a contiguous range of episodes [demo_start_idx, demo_end_idx)
        if demo_start_idx is not None or demo_end_idx is not None:
            start = demo_start_idx if demo_start_idx is not None else 0
            end = demo_end_idx if demo_end_idx is not None else replay_buffer.n_episodes
            range_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
            range_mask[start:end] = True
            train_mask = train_mask & range_mask

        if max_train_episodes is not None:
            # limit to first N training episodes
            train_indices = np.where(train_mask)[0][:max_train_episodes]
            train_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
            train_mask[train_indices] = True

        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = PhantomImageDataset.__new__(PhantomImageDataset)
        val_set.replay_buffer = self.replay_buffer
        val_set.train_mask = self.train_mask
        val_set.val_mask = self.val_mask
        val_set.horizon = self.horizon
        val_set.pad_before = self.pad_before
        val_set.pad_after = self.pad_after
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask)
        return val_set

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'], mode=mode, **kwargs)
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['agent_pos'], mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # image: (T, H, W, 3) uint8 → (T, 3, H, W) float32 [0,1]
        image = np.moveaxis(data['image'], -1, 1).astype(np.float32) / 255.0

        agent_pos = data['agent_pos'].astype(np.float32)
        action = data['action'].astype(np.float32)

        return {
            'obs': {
                'image': torch.from_numpy(image),
                'agent_pos': torch.from_numpy(agent_pos),
            },
            'action': torch.from_numpy(action),
        }
