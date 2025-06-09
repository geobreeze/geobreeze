# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from copy import deepcopy
import logging
from typing import Any, Dict, Optional, List

import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics import Accuracy
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
    sync_on_compute = distributed.is_enabled()
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
            val = MeanSquaredError(**defaults)
        
        elif id == 'RMSE':
            defaults = dict(num_outputs=1, squared=False) # Average over all outputs
            defaults.update(cfg)
            key = f'RMSE'
            val = MeanSquaredError(**defaults)

        elif id == 'JaccardIndex':
            defaults = dict(average='micro', task='multiclass')
            defaults.update(cfg)
            key = 'jaccard'
            val = JaccardIndex(num_classes=num_classes, **defaults)

        elif id == 'MeanIoU':
            defaults = dict(input_format='index')
            defaults.update(cfg)
            key = 'mIoU'
            val = MeanIoU(num_classes=num_classes, )

        elif id == 'F1Score':
            defaults = dict(average='micro', task='multiclass')
            defaults.update(cfg)
            key = 'F1Score'
            val = F1Score(num_classes=num_classes, **defaults)

        elif id == 'MAE':
            defaults = dict(num_outputs=1)
            defaults.update(cfg)
            key = f'MAE'
            val = MeanAbsoluteError(**defaults)
        else:
            raise ValueError(f"Unknown metric {id}")
        
        ret[f'{key_prefix}{key}'] = val
    return MetricCollection(ret)

