import geobench
import kornia as K
import torch
import logging
from geobreeze.datasets.base_dataset import BaseDataset
import torch.nn as nn

logger = logging.getLogger('eval')


class GeoBenchDataset(BaseDataset):

    NAME_TO_BENCHMARK = {
        'm-eurosat': 'classification_v1.0',
        'm-forestnet': 'classification_v1.0',
        'm-pv4ger': 'classification_v1.0',
        'm-brick-kiln': 'classification_v1.0',
        'm-so2sat': 'classification_v1.0',
        'm-bigearthnet': 'classification_v1.0',

        'm-pv4ger-seg': 'segmentation_v1.0',
        'm-chesapeake-landcover': 'segmentation_v1.0',
        'm-cashew-plantation': 'segmentation_v1.0',
        'm-sa-crop-type': 'segmentation_v1.0',
        'm-nz-cattle': 'segmentation_v1.0',
        'm-neontree': 'segmentation_v1.0',
    }

    def __init__(self, 
            ds_name,
            split, 
            transform = nn.Identity(), # possibly problems with segmentation
            **kwargs
        ):
        super().__init__(ds_name, **kwargs)

        split = 'valid' if split == 'val' else split

        benchmark_name = self.NAME_TO_BENCHMARK.get(ds_name)
        self.is_cls = 'classification' in benchmark_name

        task_iter = geobench.task_iterator(benchmark_name=benchmark_name)
        tasks = {task.dataset_name: task for task in task_iter}
        task = tasks.get(ds_name)
        assert task is not None, f'{ds_name} not found in geobench'

        band_names = [b['name'] for b in self.ds_config['bands']]
        MEAN, STD = task.get_dataset(band_names=band_names).normalization_stats()
        self.norm = K.augmentation.Normalize(mean=MEAN, std=STD, keepdim=True)
        self.transform = transform

        self.dataset = task.get_dataset(
            split=split,
        )

    def _getitem(self, idx):
        sample = self.dataset[idx]

        # process label
        if self.is_cls:
            label = torch.tensor(sample.label, dtype=torch.long)
        else:
            label = torch.from_numpy(sample.label.data.astype("float32")).squeeze(-1)

        # process image
        x, band_names = sample.pack_to_3d(
            band_names=None,
            resample=False,
            fill_value=None,
            resample_order=3,
        )  # h,w,c
        x = torch.from_numpy(x.astype("float32")).permute(2, 0, 1)

        x = self.transform(x)
        x = self.norm(x)

        return x, label

    def _len(self):
        return len(self.dataset)
