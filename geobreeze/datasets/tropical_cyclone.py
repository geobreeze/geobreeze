"""Tropical Cyclone Regression dataset."""
import torch
import kornia.augmentation as K
from torchgeo.datamodules import TropicalCycloneDataModule
from .base import BaseDataset
from torchgeo.samplers.utils import _to_tuple
import logging
logger = logging.getLogger()


class TropicalCyclone(BaseDataset):

    MEAN = [74.52810919339]
    STD = [60.44378695709062]
    TARGET_MEAN = [50.54925]
    TARGET_STD = [26.836512]


    def __init__(
            self, 
            root: str,
            split: str,
            transform_list: list = [],
            normalize: bool = True,
            **kwargs
        ) -> None:
        super().__init__('tropical_cyclone', **kwargs)

        # use the splits implemented in torchgeo
        dm = TropicalCycloneDataModule(
            root=root, download=False #requires azcopy to download
        )
        dm.setup('fit')
        dm.setup('test')

        self.dataset = {
            "train": dm.train_dataset,
            "val": dm.val_dataset,
            "test": dm.test_dataset
        }[split]

        self.trf = K.AugmentationSequential(*transform_list, data_keys=['image'])
        self.trf_norm = K.Normalize(mean=self.MEAN, std=self.STD, keepdim=True)
        self.normalize = normalize

    def _getitem(self, idx):
        sample = self.dataset[idx]
        sample['image'] = sample['image'][0].unsqueeze(0) # all 3 channels are the same

        if self.normalize:
            sample["image"] = self.trf_norm(sample["image"])
            sample["label"] = (sample["label"] - self.TARGET_MEAN[0]) / self.TARGET_STD[0]

        sample['image'] = self.trf(sample['image'])

        return sample['image'], sample['label'].unsqueeze(0)
    
    def _len(self):
        return len(self.dataset)