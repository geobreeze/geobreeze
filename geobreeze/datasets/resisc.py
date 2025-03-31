import kornia.augmentation as K
import torch
from torchgeo.datasets import RESISC45
from .base import BaseDataset
from .utils.utils import Downsample
import logging
logger = logging.getLogger()

class ClsDataAugmentation(torch.nn.Module):
    def __init__(self, split, size, scale:float=1.0):
        super().__init__()

        mean = torch.tensor([0.3682, 0.3808, 0.3434])
        std = torch.tensor([0.2033, 0.1852, 0.1846])
        
        norm1 = K.Normalize(mean=0, std=255)
        norm2 = K.Normalize(mean=mean, std=std)
        rc = K.RandomResizedCrop(size=size, scale=(0.8, 1.0))
        r = K.Resize(size=size, align_corners=True)
        h = K.RandomHorizontalFlip(p=0.5)
        v = K.RandomVerticalFlip(p=0.5)
        ds = Downsample(scale)


        if split == "train":
            self.transforms = [norm1, norm2, rc, h, v]
        else:
            self.transforms = [norm1, norm2, r]


        if scale != 1.0:
            self.transforms.append(ds)
        self.transform = torch.nn.Sequential(*self.transforms)

    @torch.no_grad()
    def forward(self, batch: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple"""
        x_out = self.transform(batch["image"]).squeeze(0)
        return x_out, batch["label"]


class Resics45Dataset():
    def __init__(self, config):
        self.dataset_config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.scale = config.get("scale", 1.0)
        if self.scale != 1.0:
            print('RESISC45 scale: ', self.scale)

    def create_dataset(self):
        train_transform = ClsDataAugmentation(split="train", size=self.img_size)
        eval_transform = ClsDataAugmentation(split="test", size=self.img_size, scale=self.scale)

        dataset_train = RESISC45(
            root=self.root_dir, split="train", transforms=train_transform,
        )
        dataset_val = RESISC45(
            root=self.root_dir, split="val", transforms=eval_transform
        )
        dataset_test = RESISC45(
            root=self.root_dir, split="test", transforms=eval_transform
        )

        return dataset_train, dataset_val, dataset_test
