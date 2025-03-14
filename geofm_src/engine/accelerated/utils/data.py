# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# adjusted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/


from enum import Enum
from torch.utils.data import DataLoader, DistributedSampler
import logging
import torch
from torch.utils.data.sampler import Sampler
import itertools    
from typing import Any, Optional
from . import distributed
from tqdm import tqdm
from torch.utils.data import Dataset

import numpy as np
import random

logger = logging.getLogger('eval')

class SamplerType(Enum):
    INFINITE = 0 # samples infinitely over training dataset
    EPOCH = 1 # samples the exact dataset (drop_last=False) exactly once



def make_data_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    sampler_type: SamplerType,
    shuffle: bool = True,
    seed: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    sampler_advance: int = 0,
):
    
    sampler = make_sampler(
        dataset, sampler_type, shuffle=shuffle, seed=seed, sampler_advance=sampler_advance
    )

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return data_loader



def make_sampler(
        dataset,
        sampler_type,
        shuffle: bool = True,
        seed: int = 0,
        sampler_advance: int = 0,
    ):

    sample_count = len(dataset)

    if sampler_type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=sampler_advance,)
     
    elif sampler_type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        assert sampler_advance == 0, "sampler_advance is not supported for epoch sampler"
        size = sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,)
    
    else:
        raise ValueError(f"invalid sampler type: {type}")
    


class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


class EpochSampler(Sampler):
    def __init__(
        self,
        *,
        size: int,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._size = size
        self._sample_count = sample_count
        self._shuffle = shuffle
        self._seed = seed
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._epoch = 0

    def _get_iterable(self):
        count = (self._size + self._sample_count - 1) // self._sample_count
        tiled_indices = np.tile(np.arange(self._sample_count), count)
        if self._shuffle:
            seed = self._seed * self._epoch if self._seed != 0 else self._epoch
            rng = np.random.default_rng(seed)
            iterable = rng.choice(tiled_indices, self._size, replace=False)
        else:
            iterable = tiled_indices[: self._size]
        return iterable
        
    def __iter__(self):
        iterable = self._get_iterable()
        yield from itertools.islice(iterable, self._start, None, self._step)

    def __len__(self):
        return (self._size - self._start + self._step - 1) // self._step

    def set_epoch(self, epoch):
        self._epoch = epoch



def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64

def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value


def compute_dataset_stats(dataset: Dataset, num_channels: int = 3, batch_size=256, num_workers=16, subset=1) -> None:
    """ Compute the min, max, mean, and std of a given PyTorch compatible
    datset. Assumes that the data tensor is set as (C, H, W).
    Taken from: https://github.com/stanfordmlgroup/USat/blob/main/usat/utils/helper.py
    """

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if isinstance(subset, int) and subset > 0:
        subset = min(subset, len(dataset))
        indices = indices[:subset]
    elif isinstance(subset, float) and 0 < subset < 1: 
        indices = indices[:int(subset*len(dataset))]
    dataset = torch.utils.data.Subset(dataset, indices)

    num_pixels = 0
    channels_sum = torch.zeros(num_channels)
    channels_squared_sum = torch.zeros(num_channels)

    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    for _, (data, _) in enumerate(tqdm(dl)):
        channels_sum += torch.sum(data, (0,2,3))
        channels_squared_sum += torch.sum(data**2, (0,2,3))
        num_pixels += data.size(0) * data.size(2) * data.size(3)

    mean = channels_sum / num_pixels
    std = torch.sqrt((channels_squared_sum / num_pixels) - (mean ** 2))
    print(f"Dataset len: {len(dataset)}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")