import glob
import os
from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from typing import Optional, Callable
from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg, MaskTensor
from torchvision import transforms
from .base import BaseDataset
import kornia.augmentation as K
import logging

logger = logging.getLogger('eval')


class HyperviewBenchmark(NonGeoDataset):

    # Calculated over 100% of train+val
    HYPERVIEW_MEAN = [ 357.3309,  356.8241,  352.2739,  359.0434,  367.4655,  384.4400,
         393.9638,  394.4952,  396.3852,  398.7393,  403.4731,  410.9720,
         416.7098,  432.1976,  446.6424,  454.2320,  463.8037,  473.5013,
         483.0237,  483.5212,  491.3178,  506.2690,  528.5842,  553.2881,
         566.9301,  575.3107,  587.1912,  594.3837,  605.8002,  613.8228,
         624.9483,  633.2333,  639.2795,  642.0591,  641.8145,  642.4167,
         648.9135,  653.1238,  660.5588,  669.0240,  678.0240,  683.0078,
         687.1527,  689.9290,  696.5694,  703.5454,  707.0600,  706.4673,
         709.4718,  713.4590,  719.6038,  723.9843,  726.3781,  726.3223,
         730.5421,  735.3784,  740.7471,  754.7546,  761.3821,  758.1700,
         750.7112,  741.2469,  735.4827,  733.5818,  738.0514,  731.8038,
         728.2272,  732.5236,  743.2369,  774.4900,  816.4062,  864.4771,
         911.8911,  952.4366,  984.7453, 1013.8170, 1051.8807, 1087.1163,
        1122.3405, 1157.5697, 1192.7893, 1228.0109, 1262.5994, 1298.7347,
        1333.8929, 1369.2504, 1404.4696, 1437.5973, 1466.4719, 1489.6658,
        1506.9510, 1519.3611, 1529.6740, 1536.5571, 1542.2289, 1553.9817,
        1567.5085, 1581.7065, 1595.7616, 1608.4001, 1618.5583, 1625.8043,
        1630.7007, 1635.0726, 1642.2443, 1648.8663, 1653.5099, 1659.0856,
        1664.8807, 1670.4767, 1676.0770, 1681.6744, 1687.2720, 1692.8679,
        1697.9827, 1705.2157, 1708.4655, 1714.3206, 1719.9102, 1730.0022,
        1736.3169, 1726.1902, 1718.9109, 1723.3282, 1726.9940, 1730.0922,
        1730.0206, 1730.9454, 1724.4452, 1720.3300, 1714.0845, 1721.4249,
        1740.4800, 1748.4839, 1750.7720, 1756.2915, 1758.6555, 1761.3207,
        1763.0050, 1764.2776, 1763.2791, 1765.4841, 1766.7526, 1768.1652,
        1769.6892, 1771.3467, 1772.9883, 1774.1620, 1774.8845, 1775.6278]


    HYPERVIEW_STD = [ 134.5265,  134.4990,  133.3613,  135.3228,  137.9043,  142.7918,
         145.7327,  146.1628,  147.1680,  148.4370,  150.4806,  153.3336,
         155.5741,  160.6224,  165.5196,  168.2650,  171.5105,  174.7307,
         177.9374,  178.5810,  181.5402,  186.8667,  194.4707,  202.7640,
         207.6506,  210.6972,  214.7988,  217.4728,  221.3656,  224.1183,
         227.7696,  230.5495,  232.6815,  233.8724,  234.2247,  234.9455,
         237.6151,  239.5447,  242.4203,  245.5482,  248.7913,  250.6793,
         252.2389,  253.2567,  255.5542,  258.0208,  259.4061,  259.4346,
         260.7409,  262.4182,  264.6888,  266.1444,  266.7072,  266.3409,
         267.3909,  268.7917,  270.7439,  276.0544,  279.1968,  279.0555,
         277.5002,  275.4283,  274.8527,  275.9222,  279.4376,  278.4554,
         279.2577,  282.9919,  287.4024,  295.2889,  302.8371,  310.3907,
         319.6028,  332.3335,  349.7060,  371.7979,  399.6372,  430.9706,
         465.1447,  501.5895,  539.8292,  579.5327,  619.6236,  662.1959,
         705.7219,  749.7064,  792.8994,  832.2605,  864.7523,  888.6902,
         904.2859,  913.5572,  919.9628,  924.5461,  928.2289,  935.7452,
         944.3081,  952.3723,  959.6835,  965.5954,  969.7115,  972.0714,
         973.2435,  974.3464,  976.9781,  979.2787,  980.4351,  982.0870,
         983.8807,  985.5811,  987.2900,  989.0009,  990.7166,  992.4350,
         993.9419,  996.6221,  996.9525,  998.7151, 1000.3409, 1004.9735,
        1008.0629, 1002.0215,  997.7035, 1000.1381, 1002.0745, 1003.5467,
        1003.1268, 1003.3409,  999.4595,  996.8981,  992.7678,  996.0699,
        1006.6043, 1011.1505, 1012.5848, 1015.7120, 1016.4387, 1016.4833,
        1015.0517, 1012.7045, 1008.7697, 1006.6260, 1003.9716, 1001.2909,
         998.6447,  995.7966,  992.4855,  988.7433,  984.6328,  980.2321]
    

    # Label stats
    TARGET_MEAN = torch.tensor([62.3122, 201.5815, 141.3911, 5.9815], dtype=torch.float32)
    TARGET_STD =  torch.tensor([36.6041, 94.0852, 64.3309, 2.2037], dtype=torch.float32)
    # Max per label: tensor([325.0000, 625.0000, 400.0000,   7.8000], dtype=torch.float64)
    # Min per label: tensor([ 20.3000, 109.0000,  26.8000,   5.6000], dtype=torch.float64)

    valid_splits = ["train", "val", "test"]

    split_path = "splits/{}.csv"
    gt_file_path = "gt_{}.csv"

    rgb_indices = [43, 28, 10]
    split_percentages = [0.75, 0.1, 0.15] #train, val, test

    num_channels = 150

    keys = ["P", "K", "Mg", "pH"]

    def __init__(
        self,
        root: str,
        data_dir: str='train_data',
        split: str = "train",
        create_splits: bool = False,
        transforms: Optional[Callable] = None,
        normalize: bool = True,
        normalize_target: bool = True,
        seed: int = 13, #optional, for splitting
        do_mask: bool = True, # to mask or not to mask
        mask_value: float = 0.0, #for masking out the masked values
    ) -> None:
        assert split in self.valid_splits, (
            f"Only supports one of {self.valid_splits}, but found {split}."
        )
        self.split = split
        self.normalize = normalize
        self.root = root
        self.img_path = os.path.join(self.root, data_dir)
        self.splits_path = os.path.join(self.root, self.gt_file_path.format(self.split))
        self.transforms = transforms
        self.do_mask = do_mask
        self.mask_value = mask_value
        #check if the img_path exists
        if create_splits:
            self.seed = seed
            self.split_train_val_test()
        else:
            # Check if split file exists, if not create it
            self.split_file = os.path.join(self.root, self.split_path.format(self.split))
            print(f'[HyperviewBenchmark] Building dataset for split: {self.split}')
            if os.path.exists(self.split_file):
                self.df = pd.read_csv(self.split_file)
            else:
                # self.split_train_val_test()
                raise ValueError(f"Split file does not exist. Please create it first: {self.split_file}")
            
        if self.normalize:
            self.channelwise_transforms = self._build_ch_transforms()

        self.normalize_target = normalize_target

    def split_train_val_test(self) -> list:
        """Split Train/Val/Test at the tile level."""

        from sklearn.model_selection import train_test_split
        from glob import glob

        self.seed = 21 # hardcoded!
        np.random.seed(self.seed)

        if self.gt_path is not None:
            df = pd.read_csv(os.path.join(self.root, 'train_gt.csv'))

        file_paths = sorted(
            glob(os.path.join(self.img_path, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", ""))
        )
        df['file_paths'] = file_paths

        train_val_split = self.split_percentages[0] + self.split_percentages[1]
        train_val_df, test_df = train_test_split(df, test_size=self.split_percentages[2], random_state=self.seed)
        train_df, val_df = train_test_split(train_val_df, test_size=self.split_percentages[1] / train_val_split, random_state=self.seed)

        #save the splits
        #make a new directory under the root directory called splits
        splits_dir = os.path.join(self.root, 'splits')
        if not os.path.exists(splits_dir):
            os.makedirs(splits_dir)

        train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)


    def _build_ch_transforms(self):
        self.MEAN = torch.tensor(self.HYPERVIEW_MEAN)
        self.STD = torch.tensor(self.HYPERVIEW_STD)
        
        return transforms.Compose([
            transforms.Normalize(self.MEAN, self.STD),
        ])

    def read_image(self, img_path: str, return_mask: bool = True) -> np.ndarray:
        """Read image from .npz file."""
        with np.load(img_path) as npz:
            masked_arr = np.ma.MaskedArray(**npz)

        data = masked_arr.data
        if return_mask:
            mask = masked_arr.mask
            return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
        else:
            return torch.tensor(data, dtype=torch.float32)

        
    def load_gt(self, index: int):
        """Load labels for train set from the ground truth file.
        Args:
            file_path (str): Path to the ground truth .csv file.
        Returns:
            [type]: 2D numpy array with soil properties levels
        """
        row = self.df.iloc[index]
        targets = torch.tensor(row[self.keys].values.tolist(), dtype=torch.float32)
        if self.normalize_target:
            targets = (targets - self.TARGET_MEAN) / self.TARGET_STD
        return targets

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        file_path = self.df.iloc[index]["file_paths"]
        image, mask = self.read_image(file_path)
        label = self.load_gt(index)

        if self.normalize:
            image = self.channelwise_transforms(image)

        if self.do_mask:
            bool_mask = mask.bool().expand_as(image)
            image = torch.masked_fill(image, bool_mask, self.mask_value)

        if self.transforms is not None:
            image = self.transforms(image) # the first transform is the mask tensor

        return image, label

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.df)


    def plot(
        self,
        image: torch.Tensor,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        #use the rgb_indices to plot the rgb image
        rgb_image = image[self.rgb_indices].numpy()
        rgb_image = rgb_image.transpose(1, 2, 0)
        plt.imshow(rgb_image)
        plt.show()



class Hyperview(BaseDataset):
    def __init__(self, 
            root,
            split,
            do_mask = False,
            mask_value = 0.0,
            normalize = True,
            normalize_target = True,
            transform_list: list = [],
            **kwargs
        ):
        super().__init__('hyperview', **kwargs)

        transform = K.AugmentationSequential(
            *transform_list, data_keys=['input'])

        self.dataset = HyperviewBenchmark(
            root=root,
            split = split,
            do_mask = do_mask,
            mask_value = mask_value,
            normalize=normalize,
            normalize_target=normalize_target,
            transforms=transform,)


    def _getitem(self, idx):
        return self.dataset[idx]


    def _len(self):
        return len(self.dataset)