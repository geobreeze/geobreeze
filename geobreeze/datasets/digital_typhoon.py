"""Digital Typhoon Regression dataset."""

import logging
import torch
import kornia.augmentation as K
from torchgeo.datamodules import DigitalTyphoonDataModule
from .base import BaseDataset

logger = logging.getLogger()


class DigitalTyphoon(BaseDataset):
    """Digital Typhoon dataset wrapper.
    
    # if automatic download/extraction fails, extract dataset in the root dir with `cat *.tar.gz.* | tar xvfz -`
    """

    # From 100% of train and val splits
    MEAN = [0.7588]
    STD = [0.2086]

    def __init__(
            self, 
            root: str,
            # sequence_length: int, # should be 1
            split: str,
            transform_list: list = [],
            normalize: bool = True,
            **kwargs
        )-> None:
        super().__init__('digital_typhoon',**kwargs)

        dm = DigitalTyphoonDataModule(root=root, sequence_length=1)
        # use the splits implemented in torchgeo
        dm.setup('fit')
        dm.setup('test')

        self.dataset = {
            'train': dm.train_dataset,
            'val': dm.val_dataset,
            'test': dm.test_dataset,
        }[split]

        self.transform = K.AugmentationSequential(
            *transform_list, data_keys=['image'],
        )

        self.normalize = normalize
        self.normalize_trf = K.Normalize(mean=self.MEAN, std=self.STD, keepdim=True)

    def _getitem(self, idx):
        sample = self.dataset[idx]
        x = sample["image"]

        if self.normalize:
            x = self.normalize_trf(x)

        x = self.transform(x)

        return x, sample["label"].float()
    
    def _len(self):
        return len(self.dataset)