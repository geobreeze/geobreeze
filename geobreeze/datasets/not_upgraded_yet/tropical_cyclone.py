"""Tropical Cyclone Regression dataset."""


import torch
import kornia.augmentation as K
from torchgeo.datamodules import TropicalCycloneDataModule
from .base import BaseDataset
from torchgeo.samplers.utils import _to_tuple
from .utils.utils import ChannelSampler
import logging
logger = logging.getLogger()

class RegDataAugmentation(torch.nn.Module):
    def __init__(self, size, source_chn_ids, split="val", 
                 mean=None, std=None, band_ids=None, target_chn_ids=None):
        super().__init__()

        # image statistics for the dataset
        # input_mean = torch.Tensor([0.28154722, 0.28071895, 0.27990073])
        # input_std = torch.Tensor([0.23435517, 0.23392765, 0.23351675])

        # data already comes normalized between 0 and 1
        mean = torch.Tensor([74.52810919339])
        std = torch.Tensor([60.44378695709062])

        self.target_mean = torch.Tensor([50.54925])
        self.target_std = torch.Tensor([26.836512])


        if band_ids is not None:
            chn_sample = ChannelSampler(band_ids)
            logger.info(f'[RegDataAugmentation: train] Sampling channels: {band_ids}')
            self.output_chn_ids = source_chn_ids[band_ids]
            self.transforms = [chn_sample]
        else:
            self.output_chn_ids = source_chn_ids
            self.transforms = []

        if split == "train":
            self.transforms.extend([
                K.Normalize(mean=mean, std=std),
                K.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.transforms.extend([
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            ])

        self.transform = torch.nn.Sequential(*self.transforms)

    def get_chn_ids(self):
        return self.output_chn_ids

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        # normalize the target
        target = (batch["label"].float() - self.target_mean) / self.target_std
        return x_out, target

class TropicalCycloneDataset(BaseDataset):
    """Tropical Cyclone dataset wrapper."""
    def __init__(self, config) -> None:
        """Initialize the dataset wrapper.
        
        Args:
            config: Config object for the dataset, this is the dataset config
        """
        super().__init__(config)

    def create_dataset(self):
        """Create dataset splits for training, validation, and testing."""
        train_transform = RegDataAugmentation(split="train", size=self.img_size, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids)
        eval_transform = RegDataAugmentation(split="test", size=self.img_size, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids)

        # Override the config with the transformed channel ids
        output_chn_ids = train_transform.get_chn_ids() #provides the updated channel ids after augmentation
        if output_chn_ids is not None:
            self.config['wavelengths_mean_nm'] = output_chn_ids[:,0].tolist()
            self.config['wavelengths_sigma_nm'] = output_chn_ids[:,1].tolist()

        dm = TropicalCycloneDataModule(
            root=self.root_dir, download=False #requires azcopy to download
        )
        # use the splits implemented in torchgeo
        dm.setup('fit')
        dm.setup('test')

        dataset_train = dm.train_dataset
        dataset_val = dm.val_dataset
        dataset_test = dm.test_dataset

        
        dataset_val.dataset.transforms = eval_transform
        dataset_test.transforms = eval_transform
        dataset_train.dataset.transforms = train_transform # for some reason this ordering is important

        return dataset_train, dataset_val, dataset_test