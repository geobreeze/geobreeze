from .utils.utils import ChannelSampler, ChannelSimulator, extract_wavemus, load_ds_cfg
import torch



class BaseDataset:
    def __init__(self, config):
        self.config = config
        self.img_size = (config.image_resolution, config.image_resolution)
        self.root_dir = config.data_path
        self.ds_name = config.dataset_name

        self.band_ids = config.get("band_ids", None)
        self.keep_sensors = config.get("keep_sensors", None)
        self.target_ds_name = config.get('target_dataset_name', None)
        self.full_spectra = config.get("full_spectra", False) # only for panopticon
        
        self.source_chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.ds_name), True), dtype=torch.long) #load all bands
        if self.target_ds_name is not None:
            self.target_chn_ids = torch.tensor(extract_wavemus(load_ds_cfg(self.target_ds_name), True), dtype=torch.long) #load all bands
        else:
            self.target_chn_ids = None

    def create_dataset(self):
        pass

    def __len__(self):
        pass
    
    