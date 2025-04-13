from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
import torch
import torch
import logging
from omegaconf import OmegaConf, ListConfig
from copy import deepcopy

logger = logging.getLogger('eval')


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            ds_name, 
            band_ids=None, 
            metainfo: dict={'chn_ids': 'gaussian.mu', 'gsd':'GSD', 'srf_filename': 'srf_filename'}):

        self.ds_name = ds_name
        ds_config = OmegaConf.create(load_ds_cfg(ds_name))
        self.ds_config = ds_config
        self.num_classes = ds_config['metainfo']['num_classes']
        self.calibrate(metainfo=metainfo, band_ids=band_ids)
    
    def calibrate(self, metainfo=None, band_ids=None):
        self._calibrate_metainfo(metainfo)
        self._calibrate_bands(band_ids)

    def _calibrate_metainfo(self, metainfo):
        """ set which metadata is passed with the sample"""
        # if not hasattr(self, '_default_meta_info'):
        #     self._default_meta_info = metainfo

        # metainfo = metainfo or self._default_meta_info

        metainfo_bands = {}
        for new_k, old_k in metainfo.items():
            data = [OmegaConf.select(band, old_k) for band in self.ds_config.bands]
            if not isinstance(data[0], str):
                data = [d or torch.nan for d in data]
                data = torch.tensor(data, dtype=torch.float32)
            
            metainfo_bands[new_k] = data
        self.metainfo_bands = metainfo_bands

    def _calibrate_bands(self, band_ids):
        """ set which bands are passed """

        if band_ids is None:
            self.band_ids = list(range(len(self.ds_config['bands'])))
            self._metainfo_bands_output_order = deepcopy(self.metainfo_bands)
            return

        _metainfo_bands_output_order = {}
        for k, v in self.metainfo_bands.items():
            if isinstance(v, list):
                v = [v[i] for i in band_ids]
            elif isinstance(v, torch.Tensor):
                v = v[band_ids]
            _metainfo_bands_output_order[k] = v
        self._metainfo_bands_output_order = _metainfo_bands_output_order

        # logger feedback
        band_names = [self.ds_config['bands'][i].get('name','_unknown_') for i in band_ids]
        band_names_str = ''.join([f'\n  {i:03d}: {name}' for i, name in enumerate(band_names)])
        n_all_bands = len(self.ds_config['bands'])
        n_subsampled_bands = len(band_ids)
        logger.info(f"Subsampled {n_subsampled_bands}/{n_all_bands} bands with ids {band_ids}. Band names are:{band_names_str}")

        self.band_ids = band_ids


    def __getitem__(self, idx):
        x, label = self._getitem(idx)
        
        if self.band_ids is not None:
            x = x[self.band_ids]

        x = dict(imgs=x, band_ids=torch.tensor(self.band_ids), **self._metainfo_bands_output_order)
        return x, label

    def __len__(self):
        return self._len()

    def _getitem(self, idx):
        raise NotImplementedError()
    
    def _len(self):
        raise NotImplementedError()


def collate_fn(data_list):

    with_label = isinstance(data_list[0], tuple)
    batch = torch.utils.data.default_collate(data_list)
    if with_label:
        batch_label = batch[1]
        batch = batch[0]

    if 'band_ids' in batch:
        batch['band_ids'] = batch['band_ids'][0]
    if 'srf_filename' in batch:
        batch['srf_filename'] = [d[0] for d in batch['srf_filename']]

    if with_label:
        return batch, batch_label
    return batch

def batch_to_device(batch, device, **kwargs):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, **kwargs)
    return batch 

