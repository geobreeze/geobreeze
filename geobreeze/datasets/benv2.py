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
from .base import BaseDataset
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


class BenV2Dataset(BaseDataset):

    MEAN: dict = { "s2" : torch.tensor([ 360.64678955078125, 438.3720703125, 614.0556640625, 588.4096069335938, 942.7476806640625,
                                        1769.8486328125, 2049.475830078125, 2193.2919921875, 2235.48681640625, 2241.10595703125, 1568.2115478515625, 997.715087890625]) ,
                "s1": torch.tensor([-19.352558135986328, -12.643863677978516,])}
    STD: dict = { "s2" : torch.tensor([563.1734008789062, 607.02685546875,603.2968139648438,684.56884765625,727.5784301757812,
                                        1087.4288330078125,1261.4302978515625,1369.3717041015625,1342.490478515625,1294.35546875, 1063.9197998046875,806.8846435546875]),
                "s1": torch.tensor([5.590505599975586, 5.133493900299072])}

    def __init__(
            self,
            root: str,
            bands: str,
            split: str,
            transform_list: list = [],
            norm: str = 'zscore',
            num_classes = 19,
            **kwargs
        ):
        """Initialize the BigEarthNetv2 dataset."""
        super().__init__(f'benv2_{bands}', **kwargs)
        assert bands in ['s1', 's2', 'all'], f"Invalid bands: {bands}. Must be one of ['s1', 's2', 'all']"
        self.bands = bands
        self.num_classes = 19 # overwrite

        if norm == 'zscore':
            if bands == 'all':
                means = torch.cat([self.MEAN["s1"], self.MEAN["s2"]])
                stds = torch.cat([self.STD["s1"], self.STD["s2"]])
            else:
                means = self.MEAN[bands]
                stds = self.STD[bands]
        
            self.norm_trf = K.Normalize(mean=means, std=stds, keepdim=True)
        else:
            raise NotImplementedError(f"Normalization {norm} not implemented")

        self.trf = K.AugmentationSequential(*transform_list, data_keys=["image"])

        self.dataset = BigEarthNetv2(
            root = root,
            split = split,
            num_classes = num_classes,
            bands = bands,
        )

    def _getitem(self, idx):
        sample = self.dataset[idx]
        
        x = self.norm_trf(sample["image"].squeeze(0))
        x = self.trf(x)

        if self.bands == "s1":
            # flip the channel ordering for s1 bands, because torchgeo returns them in VH, VV order and models expect VV, VH order
            x = x.flip(dims=(0,))

        return x, sample["label"]
    
    def _len(self):
        return len(self.dataset)