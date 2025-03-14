"""SpaceNet1 dataset."""

import kornia as K
import torch
from torchgeo.samplers.utils import _to_tuple
import logging
import torch
from torch import Tensor
from .spacenet import SpaceNet1
from .base_dataset import BaseDataset
from .utils.utils import ChannelSampler

class SegDataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, size, split="valid", band_ids=None, source_chn_ids=None):
        super().__init__()

        self.band_ids = band_ids
            
        self.norm = K.augmentation.Normalize(mean=mean, std=std)

        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.RandomResizedCrop(_to_tuple(size), scale=(0.8, 1.0)),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.Resize(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, x, y):
        x = self.norm(x)
        x_out, y_out = self.transform(x, y)
        if self.band_ids is not None:
            #subset the channels
            x_out = x_out[:, self.band_ids, :, :]
        return x_out, y_out


class SegGeoBenchTransform(object):
    MEAN = torch.tensor([270.4807120500449, 324.15669565737215, 507.8092169819377, 537.8422401853196, 537.7207406977241, 1209.7003277637814, 1866.045165626201, 2004.8784342115375])
    STD = torch.tensor([410.9821842645062, 466.7364072457086, 534.92315090301, 618.9212492576934, 647.7404624683102, 819.0946643361298, 1225.529915856995, 1294.781212164937])


    def __init__(self, split, size, band_ids=None):
        self.transform = SegDataAugmentation(mean=self.MEAN, std=self.STD, size=size, split=split, band_ids=band_ids)

    def __call__(self, sample):
        array, mask = self.transform(sample['image'].unsqueeze(0), sample['mask'].unsqueeze(0).unsqueeze(0))
        array, mask = array.squeeze(0), mask.squeeze(0).squeeze(0)


        return array, mask
    

class SpaceNet1Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.return_rgb = config.get('return_rgb', False)
        self.num_channels = config.get('num_channels', 4)
        if self.num_channels == 3:
            # self.image = '3band'
            self.image = '8band' #hacking for now to subset instead of rgb
            self.band_ids = [4, 2, 1]
        elif self.num_channels == 8:
            self.image = '8band'
            self.band_ids = None

    
    def create_dataset(self):
        train_transform = SegGeoBenchTransform(split="train", size=self.img_size, band_ids=self.band_ids)
        eval_transform = SegGeoBenchTransform(split="test", size=self.img_size, band_ids=self.band_ids)

        print(self.config['wavelengths_mean_nm'])

        #index into the list self.config['wavelengths_mean_nm'] using the band_ids
        if self.band_ids is not None:
            self.config['wavelengths_mean_nm'] = [self.config['wavelengths_mean_nm'][i] for i in self.band_ids]
            self.config['wavelengths_sigma_nm'] = [self.config['wavelengths_sigma_nm'][i] for i in self.band_ids]

        dataset_train = SpaceNet1(
            root=self.root_dir, split="train", transforms=train_transform, image=self.image)
        dataset_val = SpaceNet1(
            root=self.root_dir, split="val", transforms=eval_transform, image=self.image)
        dataset_test = SpaceNet1(
            root=self.root_dir, split="test", transforms=eval_transform, image=self.image)

        return dataset_train, dataset_val, dataset_test