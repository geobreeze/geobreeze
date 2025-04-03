"""Factory utily functions to create datasets and models."""

import hydra
from torch.utils.data import Subset
from geobreeze.engine.model import EvalModelWrapper
from copy import deepcopy

import geobreeze.models as models
import geobreeze.datasets as datasets

import torch
import torch.nn as nn

import kornia.augmentation as K
import kornia
from omegaconf import ListConfig, OmegaConf


def instantiate(cfg, mode='eval', **kwargs):
    if mode == 'globals':
        cfg = deepcopy(cfg)
        _class = cfg.pop('_target_')
        assert _class in globals(), f'Class {_class} not found in globals.'
        return globals()[_class](**cfg, **kwargs)
    elif mode == 'hydra':
        return hydra.utils.instantiate(cfg, **kwargs)
    elif mode == 'eval':
        cfg = deepcopy(cfg)
        return eval(cfg.pop('_target_'))(**cfg, **kwargs)
    else:
        raise ValueError(f'Unsupported mode "{mode}"')

def make_dataset(cfg, seed=21, **kwargs):
    cfg = deepcopy(cfg)

    if isinstance(cfg, list) or isinstance(cfg, ListConfig):
        return [make_dataset(c, seed=seed, **kwargs) for c in cfg]

    trf_cfg = cfg.pop('transform', [])
    transform_list = make_transform_list(trf_cfg)

    subset = cfg.pop('subset', -1)
    ds = datasets.__dict__[cfg.pop('_target_')](**cfg, transform_list=transform_list, **kwargs)
    # ds = instantiate(cfg, mode='hydra', transform_list=transform_list, **kwargs)
    ds = make_subset(ds, subset, seed=seed)

    return ds

def make_transform_list(cfg_list, **kwargs):
    """ kwargs can be called by the configs, e.g. size"""
    transform_list = []
    for cfg in cfg_list: 
        cfg = deepcopy(cfg)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        # for k,v in cfg.items():
        #     print(f'cfg: {k}: {type(v)}, {v}')
        # trf = instantiate(cfg, mode='hydra')
        trf = kornia.augmentation.__dict__[cfg.pop('_target_')](**cfg, **kwargs)
        transform_list.append(trf)
    return transform_list

def make_subset(ds, subset, seed):
    assert not isinstance(ds, torch.utils.data.IterableDataset), 'Dataset must be map-based.'

    if subset > 0:

        def sample_indices(n, k):
            generator = torch.Generator().manual_seed(seed)
            return torch.multinomial(torch.ones(n) / n, k, replacement=False, generator=generator).tolist()
        
        if isinstance(subset, float):
            assert 0.0 < subset <= 1.0, 'Float subset must be in range (0, 1].'
            if subset < 1.0:
                subset_indices = sample_indices(len(ds), int(len(ds)*subset))
                ds = Subset(ds, subset_indices)
        elif isinstance(subset, int):
            assert subset > 0, 'Int subset must be greater than 0.'
            assert subset <= len(ds)
            subset_indices = sample_indices(len(ds), subset)
            ds = Subset(ds, subset_indices)
        else:
            raise ValueError(f'Unsupported subset type "{type(subset)}"')
        print(f'Got subset={subset}, subsampled dataset to #samples {len(ds)} ')

    return ds

def make_model(cfg) -> EvalModelWrapper:
    # cfg = deepcopy(cfg)
    # return models.__dict__[cfg.pop('_target_')](**cfg)
    return instantiate(cfg, mode='hydra')

def make_optimizer(cfg, **kwargs):
    return instantiate(cfg, **kwargs)

def make_criterion(cfg):
    return instantiate(cfg)