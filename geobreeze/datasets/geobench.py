import geobench
import kornia as K
import torch
import logging
from geobreeze.datasets.base import BaseDataset
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
        'm-cashew-plant': 'segmentation_v1.0',
        'm-sa-crop-type': 'segmentation_v1.0',
        'm-nz-cattle': 'segmentation_v1.0',
        'm-neontree': 'segmentation_v1.0',
    }

    def __init__(self, 
            ds_name,
            split, 
            transform_list = [],
            normalize = True,
            **kwargs
        ):
        print(kwargs)
        super().__init__(ds_name, **kwargs)

        split = 'valid' if split == 'val' else split

        benchmark_name = self.NAME_TO_BENCHMARK.get(ds_name)
        self.is_cls = 'classification' in benchmark_name

        task_iter = geobench.task_iterator(benchmark_name=benchmark_name)
        tasks = {task.dataset_name: task for task in task_iter}
        task = tasks.get(ds_name)
        assert task is not None, f'{ds_name} not found in geobench'

        band_names = [b['name'] for b in self.ds_config['bands']]
        self.band_names = band_names
        print(band_names)
        MEAN, STD = task.get_dataset(band_names=band_names).normalization_stats()
        self.normalize_trf = K.augmentation.Normalize(mean=MEAN, std=STD, keepdim=True)

        data_keys = ['input'] if self.is_cls else ['input', 'mask']
        self.transform = K.augmentation.AugmentationSequential(
            *transform_list, 
            data_keys=data_keys
        )    

        self.normalize = normalize

        self.dataset = task.get_dataset(
            split=split,
            band_names=band_names
        )

    def _getitem(self, idx):
        sample = self.dataset[idx]

        # process image
        x, band_names = sample.pack_to_3d(
            band_names=self.band_names,
            resample=False,
            fill_value=None,
            resample_order=3,
        )  # h,w,c
        x = torch.from_numpy(x.astype("float32")).permute(2, 0, 1)

        # process label & transform
        if self.is_cls:
            label = torch.tensor(sample.label, dtype=torch.long)
            x = self.transform(x)
        else:
            label = torch.from_numpy(sample.label.data.astype("float32")).squeeze(-1)
            x, label = self.transform(x, label)

        if self.normalize:
            x = self.normalize_trf(x)

        return x, label

    def _len(self):
        return len(self.dataset)
