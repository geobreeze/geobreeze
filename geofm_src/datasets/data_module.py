from lightning import LightningDataModule
import torch
from geofm_src.factory import create_dataset
from torch.utils.data import Subset

class BenchmarkDataModule(LightningDataModule):
    def __init__(self, dataset_config, batch_size, num_workers, pin_memory, seed=42):
        super().__init__()
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def setup(self, stage=None):
        train, val, test = create_dataset(self.dataset_config)
        print('subsetting train (if applicable)')
        self.dataset_train = make_subset(train, self.dataset_config.subset.train, seed=self.seed)
        print('subsetting val (if applicable)')
        self.dataset_val = make_subset(val, self.dataset_config.subset.val, seed=self.seed)
        print('subsetting test (if applicable)')
        self.dataset_test = make_subset(test, self.dataset_config.subset.test, seed=self.seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


def make_subset(ds, subset, seed):
    assert not isinstance(ds, torch.utils.data.IterableDataset), 'Dataset must be map-based.'

    if subset > 0:

        def sample_indices(n, k):
            generator = torch.Generator().manual_seed(seed)
            return torch.multinomial(torch.ones(n) / n, k, replacement=False, generator=generator).tolist()
        
        if isinstance(subset, float):
            assert 0.0 < subset <= 1.0, 'Float subset must be in range (0, 1].'
            if subset < 1.0:
                subset_indices = sample_indices(len(ds), int(len(ds)*subset))
                ds = Subset(ds, subset_indices)
        elif isinstance(subset, int):
            assert subset > 0, 'Int subset must be greater than 0.'
            assert subset <= len(ds)
            subset_indices = sample_indices(len(ds), subset)
            ds = Subset(ds, subset_indices)
        else:
            raise ValueError(f'Unsupported subset type "{type(subset)}"')
        print(f'Got subset={subset}, subsampled dataset to #samples {len(ds)} ')

    return ds