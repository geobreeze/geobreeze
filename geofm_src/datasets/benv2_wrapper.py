"""BigEarthNetv2 dataset."""

import glob
import os
from typing import Callable, Optional

import kornia.augmentation as K
import pandas as pd
import rasterio
import torch
from torch import Generator, Tensor
from torch.utils.data import random_split
from torchgeo.datasets import BigEarthNet
from .base_dataset import BaseDataset
from .utils.utils import ChannelSampler 

class BigEarthNetv2(BigEarthNet):
    """BigEarthNetv2 dataset.

    Automatic download not implemented, get data from below link.
    """

    splits_metadata = { # splits are in a column in metadata.parquet
        "train": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "val": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "test": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
    }
    metadata_locs = {
        "s1": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst",
            "md5": "",  # unknown
            "filename": "BigEarthNet-S1.tar.zst",
            "directory": "BigEarthNet-S1",
        },
        "s2": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst",
            "md5": "",  # unknown
            "filename": "BigEarthNet-S2.tar.zst",
            "directory": "BigEarthNet-S2",
        },
        "maps": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "",  # unknown
            "filename": "Reference_Maps.zst",
            "directory": "Reference_Maps",
        },
    }
    image_size = (120, 120)
    num_channels = 12

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        super().__init__(
            root=root,
            split=split,
            bands=bands,
            num_classes=num_classes,
            transforms=transforms,
            download=download,
            checksum=checksum,
        )
        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])}
        # self._verify()
        # self.folders = self._load_folders() # this is also called in the parent class!

    def get_class2idx(self, label: str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        pass

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))
        split_ = 'validation' if self.split == 'val' else self.split
        self.metadata = self.metadata[self.metadata["split"] == split_]

        folder_file = os.path.join(self.root, f'{self.split}_folders.parquet')
        if os.path.exists(folder_file):
            print(f'[BigEarthNetv2] Loading folders from {folder_file}')
            folders = pd.read_parquet(folder_file).reset_index(drop=True)

        else:
            print(f'[BigEarthNetv2] Constructing folders and saving to {folder_file}')

            def construct_folder_path(root, dir, patch_id, remove_last: int = 2):
                tile_id = "_".join(patch_id.split("_")[:-remove_last])
                return os.path.join(root, dir, tile_id, patch_id)

            folders = [
                {
                    "s1": construct_folder_path(self.root, dir_s1, row["s1_name"], 3),
                    "s2": construct_folder_path(self.root, dir_s2, row["patch_id"], 2),
                    "maps": construct_folder_path(self.root, dir_maps, row["patch_id"], 2),
                }
                for _, row in self.metadata.iterrows()
            ]

            folders = pd.DataFrame(folders)
            folders.to_parquet(folder_file)

        folders = folders.to_dict(orient="records")

        return folders

    def _load_map_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder_maps = self.folders[index]["maps"]
        paths_maps = glob.glob(os.path.join(folder_maps, "*_reference_map.tif"))
        paths_maps = sorted(paths_maps)
        return paths_maps

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_map_paths(index)
        map = None
        for path in paths:
            with rasterio.open(path) as dataset:
                map = dataset.read(
                    # indexes=1,
                    # out_shape=self.image_size,
                    out_dtype="int32",
                    # resampling=Resampling.bilinear,
                )
        return torch.from_numpy(map).float()

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """

        image_labels = self.metadata.iloc[index]["labels"]

        # labels -> indices
        indices = [
            self.get_class2idx(label, level=self.num_classes) for label in image_labels
        ]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target


class ClsDataAugmentation(torch.nn.Module):

    MEAN: dict = { "s2" : torch.tensor([ 360.64678955078125, 438.3720703125, 614.0556640625, 588.4096069335938, 942.7476806640625,
                                        1769.8486328125, 2049.475830078125, 2193.2919921875, 2235.48681640625, 2241.10595703125, 1568.2115478515625, 997.715087890625]) ,
                "s1": torch.tensor([-19.352558135986328, -12.643863677978516,])}
    STD: dict = { "s2" : torch.tensor([563.1734008789062, 607.02685546875,603.2968139648438,684.56884765625,727.5784301757812,
                                        1087.4288330078125,1261.4302978515625,1369.3717041015625,1342.490478515625,1294.35546875, 1063.9197998046875,806.8846435546875]),
                "s1": torch.tensor([5.590505599975586, 5.133493900299072])}

    mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    maxs_raw = torch.tensor(
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    maxs = torch.tensor(
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )


    def __init__(self, split, size, source_chn_ids, band_ids=None, bands="all", num_channels=12, quantile_norm: bool = False):
        """Initialize the data augmentation pipeline.
        
        Args:
            split: train/test split
            size: image size
            bands: which bands to use, one of {s1, s2, rgb, all}
            source_chn_ids: channel ids of the source image
            band_ids: which bands to sample, if None, all bands are used
            num_channels: number of channels to output
            quantile_norm: whether to perform quantile normalization or standard z-score normalization
        """
        super().__init__()

        self.num_channels = num_channels
        self.quantile_norm = quantile_norm

        # if bands == "all":
        #     mins = self.mins
        #     maxs = self.maxs
        # elif bands == "s1":
        #     mins = self.mins[:2]
        #     maxs = self.maxs[:2]
        # elif bands == "s2":
        #     mins = self.mins[2:]
        #     maxs = self.maxs[2:]
        # elif bands == "rgb":
        #     mins = self.mins[2:5].flip(dims=(0,))  # to get RGB order
        #     maxs = self.maxs[2:5].flip(dims=(0,))
        if bands == "s1":
            means = self.MEAN["s1"]
            stds = self.STD["s1"]
        elif bands == "s2":
            means = self.MEAN["s2"]
            stds = self.STD["s2"]
        elif bands == "rgb":
            means = self.MEAN["s2"][1:4].flip(dims=(0,))
            stds = self.STD["s2"][1:4].flip(dims=(0,))
            band_ids = [3,2,1]
        elif bands == "all":
            means = torch.cat([self.MEAN["s1"], self.MEAN["s2"]])
            stds = torch.cat([self.STD["s1"], self.STD["s2"]])
        else:
            raise ValueError(f"[ClsDataAugmentation] Invalid bands: {bands}")
        
        self.bands = bands
        self.mean = means
        self.std = stds
        self.output_chn_ids = source_chn_ids

        self.transforms = []

        if band_ids is not None:
            if bands != "rgb": # no need to sample channels, all bands come in in the RGB order
                chn_sample = ChannelSampler(band_ids)
                if source_chn_ids is not None:
                    self.output_chn_ids = source_chn_ids[band_ids]
                    self.mean = self.mean[band_ids]
                    self.std = self.std[band_ids]

                    print(f'[ClsDataAugmentation] Sampling channels: {band_ids}')
                    self.transforms.append(chn_sample)
                else:
                    raise ValueError("[ClsDataAugmentation] source_chn_ids must be provided if band_ids are provided")
            else:
                self.output_chn_ids = source_chn_ids


        # normalization is handled separately
        if split == "train":
            self.transforms.extend([
                K.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.transforms.extend([
                K.Resize(size=size, align_corners=True),
            ])

        self.transform = torch.nn.Sequential(*self.transforms)

    def get_chn_ids(self):
        return self.output_chn_ids

    @torch.no_grad()
    def forward(self, sample: dict[str,]):
        """Torchgeo returns a dictionary with 'image' and 'label' keys, but engine expects a tuple."""
        if self.bands == "rgb":
            sample["image"] = sample["image"][1:4, ...].flip(dims=(0,))
            # get in rgb order and then normalization can be applied

        # do normalization in a separate function
        x_normed = self.normalize(sample["image"])
        x_out = self.transform(x_normed).squeeze(0)

        if self.bands == "s1":
            # flip the channel ordering for s1 bands, because torchgeo returns them in VH, VV order and models expect VV, VH order
            x_out = x_out.flip(dims=(0,))

        # HACKY: Will only work for models that dont need chn_ids
        # Pad with zeros if num_channels is greater than the actual channels
        if hasattr(self, 'num_channels') and self.num_channels > x_out.shape[0]: 
            padding_channels = self.num_channels - x_out.shape[0]
            padding = torch.zeros((padding_channels, *x_out.shape[1:]), device=x_out.device, dtype=x_out.dtype)
            x_out = torch.cat([x_out, padding], dim=0)
        
        return x_out, sample["label"]

    @torch.no_grad()
    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize the input image."""
        if self.quantile_norm:
            img = self.quantile_normalize(img)
        else:
            img = K.Normalize(mean=self.mean, std=self.std)(img)
        return img

    @torch.no_grad()
    def quantile_normalize(self, sample_img) -> torch.Tensor:
        """Quantile normalize the input image."""

        def channel_norm(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
            """Normalize an image channel."""
            min_value = mean - 2 * std
            max_value = mean + 2 * std
            img = (img - min_value) / (max_value - min_value)
            img = torch.clamp(img, 0, 1)
            return img

        if self.bands == 's1':
            # sample_img = sample["image"]
            ### normalize s1
            self.max_q = torch.quantile(sample_img.reshape(2,-1),0.99,dim=1)      
            self.min_q = torch.quantile(sample_img.reshape(2,-1),0.01,dim=1)
            img_bands = []
            for b in range(2):
                img = sample_img[b,:,:].clone()
                ## outlier
                max_q = self.max_q[b]
                min_q = self.min_q[b]            
                img = torch.clamp(img, min_q, max_q)
                ## normalize
                img = channel_norm(img,self.mean[b],self.std[b])         
                img_bands.append(img)
            sample_img = torch.stack(img_bands,dim=0) # VV,VH (w,h,c)
        elif self.bands == 's2':
            # sample_img = sample["image"]
            img_bands = []
            for b in range(12):
                img = sample_img[b,:,:].clone()
                ## normalize
                img = channel_norm(img,self.mean[b],self.std[b])         
                img_bands.append(img)
                if b==9:
                    # pad zero to B10
                    img_bands.append(torch.zeros_like(img))
            sample_img = torch.stack(img_bands,dim=0)

        return sample_img


class BenV2Dataset(BaseDataset):
    def __init__(self, config):
        """Initialize the BigEarthNetv2 dataset."""
        super().__init__(config)
        self.bands = config.bands
        self.num_classes = config.num_classes
        self.num_channels = config.num_channels

        self.quantile_norm = config.get("quantile_norm", False)
        print('Using quantile norm: ', self.quantile_norm)

        if self.bands == "rgb":
            # start with rgb and extract later
            self.input_bands = "s2"
        else:
            self.input_bands = self.bands

    def create_dataset(self):
        train_transform = ClsDataAugmentation(
            split="train", size=self.img_size, bands=self.bands, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids, num_channels=self.num_channels, quantile_norm=self.quantile_norm
        )
        eval_transform = ClsDataAugmentation(
            split="test", size=self.img_size, bands=self.bands, source_chn_ids=self.source_chn_ids, band_ids=self.band_ids, num_channels=self.num_channels, quantile_norm=self.quantile_norm
        )

        # Override the config with the transformed channel ids
        output_chn_ids = train_transform.get_chn_ids() #provides the updated channel ids after augmentation
        if output_chn_ids is not None:
            self.config['wavelengths_mean_nm'] = output_chn_ids[:,0].tolist()
            self.config['wavelengths_mean_microns'] = [x/1e3 for x in self.config['wavelengths_mean_nm']]
            self.config['wavelengths_sigma_nm'] = output_chn_ids[:,1].tolist()


        dataset_train = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="train",
            bands=self.input_bands,
            transforms=train_transform,
        )

        dataset_val = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="val",
            bands=self.input_bands,
            transforms=eval_transform,
        )
        dataset_test = BigEarthNetv2(
            root=self.root_dir,
            num_classes=self.num_classes,
            split="test",
            bands=self.input_bands,
            transforms=eval_transform,
        )

        return dataset_train, dataset_val, dataset_test
