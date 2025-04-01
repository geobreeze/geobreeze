import kornia.augmentation as K
import torch
from torchgeo.datasets import RESISC45
from .base import BaseDataset
import logging
import os

import kornia as K

logger = logging.getLogger('eval')


class Resisc45(BaseDataset):

    def __init__(self,
        split: str,
        root: str = None,
        transform_list: list = [],
        normalize: bool = True,
        **kwargs
    ):
        super().__init__('resisc45', **kwargs)

        root = root or os.path.join(os.environ['DATASETS_DIR'], 'resisc45')
        self.dataset = RESISC45(
            root=root,
            split=split,
        )

        MEAN = torch.tensor([0.3682, 0.3808, 0.3434]) # after normalized to [0,1]
        STD = torch.tensor([0.2033, 0.1852, 0.1846]) # after normalized to [0,1]
        self.normalize_trf = K.augmentation.AugmentationSequential(
            K.augmentation.Normalize(mean=0, std=255, keepdim=True),
            K.augmentation.Normalize(mean=MEAN, std=STD, keepdim=True),
            data_keys=['input'])
        self.normalize = normalize

        self.transform = K.augmentation.AugmentationSequential(
            *transform_list, data_keys=['input'])

    def _getitem(self, idx):
        x_dict = self.dataset[idx]
        img = x_dict['image']
        label = x_dict['label']
        
        img = self.transform(img)

        if self.normalize:
            img = self.normalize_trf(img)

        return img, label

    def _len(self):
        return len(self.dataset)