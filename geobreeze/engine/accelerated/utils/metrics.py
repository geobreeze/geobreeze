# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from copy import deepcopy
import logging
from typing import Any, Dict, Optional, List

import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MetricCollection, Accuracy, Metric
from torchmetrics.classification import MulticlassAccuracy, MultilabelAveragePrecision, MultilabelF1Score, JaccardIndex, F1Score
from torchmetrics.regression import MeanSquaredError
from torchmetrics.segmentation import MeanIoU

from . import distributed
import torch.nn as nn

logger = logging.getLogger("eval")


def build_metric(metric_cfg: List, num_classes, key_prefix='') -> MetricCollection:
    # TODO I think we can remove this and just target these directyl through the config
    metric_cfg = deepcopy(metric_cfg)
    if isinstance(num_classes, Tensor):
        num_classes = num_classes.item()

    ret = {}
    sync_on_compute = distributed.is_enabled() # important if we plan to do a 
        # custom synchronized backend
    for cfg in metric_cfg:
        id = cfg.pop('id')

        if id == 'MulticlassAccuracy':
            defaults = dict(top_k=1, average='micro')
            defaults.update(cfg)
            key = f'acc_top-{defaults["top_k"]}_{defaults["average"]}'
            val = Accuracy(
                num_classes=num_classes, task='multiclass', sync_on_compute=sync_on_compute, **defaults)

        elif id == 'MultilabelAccuracy':
            defaults = dict(top_k=1, average='micro')
            defaults.update(cfg)
            key = f'MulLabAcc_top-{defaults["top_k"]}_{defaults["average"]}'
            val = Accuracy(
                num_labels=num_classes, task='multilabel', sync_on_compute=sync_on_compute, **defaults)

        elif id == 'MultiLabelAveragePrecision':
            defaults = dict(average='macro')
            defaults.update(cfg)
            key = f'MulLabAvergPrec_{defaults["average"]}'
            val = MultilabelAveragePrecision(
                num_labels=num_classes, sync_on_compute=sync_on_compute, **defaults)

        elif id == 'MultiLabelF1Score': #macro and micro
            defaults = dict(average='macro')
            defaults.update(cfg)
            key = f'MulLabF1ScoreMacro_{defaults["average"]}'
            val = MultilabelF1Score(
                num_labels=num_classes, sync_on_compute=sync_on_compute, **defaults)

        elif id == 'MSE':
            defaults = dict(num_outputs=1) # Average over all outputs
            defaults.update(cfg)
            key = f'MSE'
            val = MeanSquaredError(**defaults, sync_on_compute=sync_on_compute)
        
        elif id == 'RMSE':
            defaults = dict(num_outputs=1, squared=False) # Average over all outputs
            defaults.update(cfg)
            key = f'RMSE'
            val = MeanSquaredError(**defaults, sync_on_compute=sync_on_compute)

        elif id == 'JaccardIndex':
            defaults = dict(average='micro', task='multiclass')
            defaults.update(cfg)
            key = f'jaccard_{defaults["average"]}'
            val = JaccardIndex(num_classes=num_classes, **defaults, sync_on_compute=sync_on_compute)

        elif id == 'mIoU':
            defaults = dict(input_format='index')
            defaults.update(cfg)
            key = 'mIoU' + '_'.join([f'{k}={v}' for k, v in defaults.items() if k != 'input_format'])
            val = MeanIoU(num_classes=num_classes, **defaults, sync_on_compute=sync_on_compute)

        elif id == 'F1Score':
            defaults = dict(average='micro', task='multiclass')
            defaults.update(cfg)
            key = f'F1Score_{defaults["average"]}'
            val = F1Score(num_classes=num_classes, **defaults, sync_on_compute=sync_on_compute)

        elif id == 'MAE':
            defaults = dict(num_outputs=1)
            defaults.update(cfg)
            key = f'MAE'
            val = MeanAbsoluteError(**defaults, sync_on_compute=sync_on_compute)

        elif id == 'CustomJaccard':
            defaults = dict(average='macro', input_format='index')
            defaults.update(cfg)
            key = f'CustomJaccard_{defaults["average"]}'
            val = CustomJaccard(num_classes=num_classes, **defaults, sync_on_compute=sync_on_compute)

        else:
            raise ValueError(f"Unknown metric {id}")
        
        ret[f'{key_prefix}{key}'] = val
    return MetricCollection(ret)



class CustomJaccard(Metric):
    """ Custom implementation of Jaccard Index for multiclass classification to 
        compare against torchmetrics.classification.JaccardIndex and 
        torchmetrics.segmentation.MeanIoU."""

    def __init__(self, num_classes: int, input_format='index', average='macro', **kwargs):
        super().__init__(**kwargs)
        assert input_format == 'index', 'Only index supported yet'
        assert average in ['macro', 'micro'], 'Only macro and micro averaging supported'

        self.num_classes = num_classes
        self.add_state('intersection', 
                       torch.zeros(num_classes, dtype=torch.int), 
                       dist_reduce_fx='sum')
        self.add_state('union', 
                       torch.zeros(num_classes, dtype=torch.int), 
                       dist_reduce_fx='sum')
        self.average = average

    def update(self, preds: Tensor, target: Tensor):

        for c in range(self.num_classes):
            self.intersection[c] += torch.sum((preds == c) & (target == c))
            self.union[c] += torch.sum((preds == c) | (target == c))

    def compute(self) -> Tensor:
        if self.average == 'macro':
            iou = self.intersection / (self.union + 1e-6)
            return iou.mean()
        elif self.average == 'micro':
            intersection = self.intersection.sum()
            union = self.union.sum()
            return (intersection / (union + 1e-6))
        else:
            raise ValueError(f"Unknown average {self.average}")