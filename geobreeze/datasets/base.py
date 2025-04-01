from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
import torch
import torch
import logging
from omegaconf import OmegaConf
from copy import deepcopy

logger = logging.getLogger('eval')


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, band_ids=None, metainfo: dict={'chn_ids': 'bands.gaussian.mu'}):

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
        if not hasattr(self, '_default_meta_info'):
            self._default_meta_info = metainfo

        metainfo = metainfo or self._default_meta_info

        metainfo_bands = {}
        band_keys = {new_k: old_k[6:] for new_k, old_k in metainfo.items() 
                      if old_k.startswith('bands.')}
        for new_k, old_k in band_keys.items():
            metainfo_bands[new_k] = torch.tensor(
                [OmegaConf.select(band, old_k) or torch.nan for band in self.ds_config.bands])
        self.metainfo_bands = metainfo_bands

        sensor_keys = {new_k: old_k for new_k, old_k in metainfo.items() 
                        if not old_k.startswith('bands.')}
        self.metainfo_sensor = {new_k: OmegaConf.select(self.ds_config, old_k) 
                           for new_k, old_k in sensor_keys.items()}

    def _calibrate_bands(self, band_ids, verbose=True):
        """ set which bands are passed """

        if band_ids is None:
            self.band_ids = list(range(len(self.ds_config['bands'])))
            self._metainfo_bands_output_order = deepcopy(self.metainfo_bands)
            return

        self._metainfo_bands_output_order = {
            k: v[band_ids] for k, v in self.metainfo_bands.items()}

        # logger feedback
        band_names = [self.ds_config['bands'][i]['name'] for i in band_ids]
        band_names_str = ''.join([f'\n  {i:03d}: {name}' for i, name in enumerate(band_names)])
        n_all_bands = len(self.ds_config['bands'])
        n_subsampled_bands = len(band_ids)
        logger.info(f"Subsampled {n_subsampled_bands}/{n_all_bands} bands with ids {band_ids}. Band names are:{band_names_str}")

        self.band_ids = band_ids


    def __getitem__(self, idx):
        x, label = self._getitem(idx)
        
        if self.band_ids is not None:
            x = x[self.band_ids]

        x = dict(imgs=x, **self._metainfo_bands_output_order, **self.metainfo_sensor)
        return x, label

    def __len__(self):
        return self._len()

    def _getitem(self, idx):
        raise NotImplementedError()
    
    def _len(self):
        raise NotImplementedError()
