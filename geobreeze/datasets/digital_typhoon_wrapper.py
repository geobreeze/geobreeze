"""Digital Typhoon Regression dataset."""

import logging
import torch
import kornia.augmentation as K
from torchgeo.datamodules import DigitalTyphoonDataModule
from .base_dataset import BaseDataset

logger = logging.getLogger()

class RegDataAugmentation(torch.nn.Module):
    def __init__(self, size, source_chn_ids, split, mean=None, std=None, band_ids=None, target_chn_ids=None):
        super().__init__()

        self.output_chn_ids = source_chn_ids

        # From 100% of train and val splits
        mean = torch.Tensor([0.7588])
        std = torch.Tensor([0.2086])

        if band_ids is not None:
            if source_chn_ids is not None:
                self.output_chn_ids = source_chn_ids[band_ids]
            else:
                raise ValueError("[ClsDataAugmentation] source_chn_ids must be provided if band_ids are provided")

        if split == "train":
            self.transform = torch.nn.Sequential(
                K.Normalize(mean=mean, std=std),
                K.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            )
        else:
            self.transform = torch.nn.Sequential(
                
                K.Normalize(mean=mean, std=std),
                K.Resize(size=size, align_corners=True),
            )

        print(self.transform)

    def get_chn_ids(self):
        return self.output_chn_ids

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"].float()

class DigitalTyphoonDataset(BaseDataset):
    """Digital Typhoon dataset wrapper.
    
    # if automatic download/extraction fails, extract dataset in the root dir with `cat *.tar.gz.* | tar xvfz -`
    """
    def __init__(self, config) -> None:
        """Initialize the dataset wrapper.
        
        Args:
            config: Config object for the dataset, this is the dataset config
        """
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 3)
        logger.info(f"[DigitalTyphoon] Temporal sequence length: {self.sequence_length}")

    def create_dataset(self):
        """Create dataset splits for training, validation, and testing."""
        train_transform = RegDataAugmentation(split="train", size=self.img_size, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids)
        eval_transform = RegDataAugmentation(split="test", size=self.img_size, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids)

        # Override the config with the transformed channel ids
        output_chn_ids = train_transform.get_chn_ids() #provides the updated channel ids after augmentation
        if output_chn_ids is not None:
            self.config['wavelengths_mean_nm'] = output_chn_ids[:,0].tolist()
            self.config['wavelengths_sigma_nm'] = output_chn_ids[:,1].tolist()

        # dataset has argument sequence length which actually dictates the number of channels
        # commonly in literature is used 3 channels and pretend it is RGB
        # sequence_length: length of the sequence to return
        dm = DigitalTyphoonDataModule(
            root=self.root_dir, sequence_length=self.sequence_length
        )
        # use the splits implemented in torchgeo
        dm.setup('fit')
        dm.setup('test')

        dataset_train = dm.train_dataset
        dataset_val = dm.val_dataset
        dataset_test = dm.test_dataset

        
        dataset_val.dataset.transforms = eval_transform
        dataset_test.dataset.transforms = eval_transform
        dataset_train.dataset.transforms = train_transform # for some reason this ordering is important

        return dataset_train, dataset_val, dataset_test