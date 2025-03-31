from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
import torch
import torch
import logging

logger = logging.getLogger('eval')


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, ds_name, band_ids=None):

        self.ds_name = ds_name
        ds_config = load_ds_cfg(ds_name)
        self.ds_config = ds_config

        self.band_ids = band_ids
        self.chn_ids = torch.tensor(extract_wavemus(ds_config, True), dtype=torch.long)
        self.num_classes = ds_config['metainfo']['num_classes']

        if band_ids is not list(range(self.chn_ids.shape[0])):
            logger.info(f"Subsampling bands: {band_ids}")

        # self.bands_all = dict(
        #     chn_ids = chn_ids
        # )
        # self.bands_output = deepcopy(self.bands_all)
        # if self.band_ids is not None:
        #     self.bands_output['chn_ids'] = chn_ids[self.band_ids]

    def __getitem__(self, idx):
        x, label = self._getitem(idx)
        chn_ids = self.chn_ids
        
        # re-index bands
        if self.band_ids is not None:
            x = x[self.band_ids]
            chn_ids = chn_ids[self.band_ids]

        x = dict(imgs=x, chn_ids=chn_ids)
        return x, label

    def __len__(self):
        return self._len()

    def _getitem(self, idx):
        raise NotImplementedError()
    
    def _len(self):
        raise NotImplementedError()
