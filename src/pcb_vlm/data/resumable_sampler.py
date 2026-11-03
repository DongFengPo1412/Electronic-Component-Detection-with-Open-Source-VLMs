"""Resumable Batch Sampler for deterministic, lossless training continuation."""

import torch
from torch.utils.data import Sampler
from typing import List, Optional, Iterator


class ResumableBatchSampler(Sampler[List[int]]):
    """
    A batch sampler that supports deterministic epoch shuffling and lossless resumption.
    
    Generates deterministic sample permutations per epoch using:
        torch.Generator().manual_seed(seed + epoch)
    
    When training is interrupted and resumed:
    - Restores the exact remaining batch indices of the interrupted epoch.
    - Subsequent epochs continue with their deterministically seeded permutations.
    - Eliminates duplicate sample processing and avoids shuffling discrepancies upon resumption.
    """
    def __init__(
        self,
        dataset_len: int,
        batch_size: int = 1,
        seed: int = 42,
        epoch: int = 0,
        start_batch_idx: int = 0,
        drop_last: bool = False,
        permutation: Optional[List[int]] = None
    ):
        super().__init__(None)
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch_idx = start_batch_idx
        self.drop_last = drop_last
        self._current_permutation = list(permutation) if permutation is not None else None

    def set_epoch(self, epoch: int, start_batch_idx: int = 0, permutation: Optional[List[int]] = None) -> None:
        """Configure epoch number and starting batch offset."""
        self.epoch = epoch
        self.start_batch_idx = start_batch_idx
        if permutation is not None:
            self._current_permutation = list(permutation)
        else:
            g = torch.Generator().manual_seed(self.seed + epoch)
            self._current_permutation = torch.randperm(self.dataset_len, generator=g).tolist()

    def get_permutation(self) -> List[int]:
        """Return the permutation of sample indices for the current epoch."""
        if self._current_permutation is None:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            self._current_permutation = torch.randperm(self.dataset_len, generator=g).tolist()
        return list(self._current_permutation)

    def get_all_batches(self) -> List[List[int]]:
        """Return all batch index lists for the current epoch."""
        perm = self.get_permutation()
        batches = [perm[i:i + self.batch_size] for i in range(0, len(perm), self.batch_size)]
        if self.drop_last and len(batches) > 0 and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self.get_all_batches()
        for b_idx in range(self.start_batch_idx, len(batches)):
            yield batches[b_idx]

    def __len__(self) -> int:
        batches = self.get_all_batches()
        return max(0, len(batches) - self.start_batch_idx)
