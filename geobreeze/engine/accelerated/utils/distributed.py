# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# adjusted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/distributed/__init__.py

import os
import random
import re
import socket
from typing import Dict, List

import torch
import torch.distributed as dist
import datetime

_LOCAL_RANK = -1
_LOCAL_WORLD_SIZE = -1
_ARTIFICIALLY_BLOCK_DISTRIBUTED = False

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return not _ARTIFICIALLY_BLOCK_DISTRIBUTED and dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    """
    Returns:
        The number of processes in the process group
    """
    return dist.get_world_size() if is_enabled() else 1


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not is_enabled():
        return 0
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_RANK


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not is_enabled():
        return 1
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_WORLD_SIZE


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0


def _restrict_print_to_main_process() -> None:
    """
    This function disables printing when not in the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main_process() or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print