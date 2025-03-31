import torch
from geobreeze.engine.model import EvalModelWrapper


class Panopticon(EvalModelWrapper):

    def _load_encoder(self, blk_indices):
        self.encoder = torch.hub.load(
            'panopticon-FM/panopticon',
            'panopticon_vitb14',)
        # self.encoder = torch.hub.load(
        #     '/home/hk-project-pai00028/tum_mhj8661/code/PanOpticOn',
        #     'panopticon_vitb14',
        #     source='local')
        self.norm = self.encoder.norm
        self.blk_indices = blk_indices

        # wavelengths = self.data_config.wavelengths_mean_nm
        # sigmas = self.data_config.get('wavelengths_sigma_nm', None)

        # if not self.model_config.get('full_spectra', False) and sigmas is not None:
        #     self.register_buffer('chn_ids',
        #         torch.tensor([wl for wl in wavelengths]).unsqueeze(0)) # (1, C)
        # else:
        #     print('Full Spectra Enabled')
        #     self.register_buffer('chn_ids',
        #         torch.tensor(list(zip(wavelengths, sigmas))).unsqueeze(0)) # (1, C, 2)
        #     print(self.chn_ids, self.chn_ids.shape)
    
    def get_blocks(self, x_dict):
        chn_ids = x_dict['chn_ids']
        if chn_ids.dim() == 2:  # (1, C)
            chn_ids = chn_ids.expand(x_dict['imgs'].size(0), -1) # Expand to (B, C)
        else:  # (1, C, 2)
            chn_ids = chn_ids.expand(x_dict['imgs'].size(0), -1, -1)  # Expand to (B, C, 2)
        x_dict['chn_ids'] = chn_ids

        if self.encoder.chunked_blocks:
            x_blocks_list = self.encoder._get_intermediate_layers_chunked(x_dict, self.blk_indices)
        else:
            x_blocks_list = self.encoder._get_intermediate_layers_not_chunked(x_dict, self.blk_indices)

        return x_blocks_list

    def default_blocks_to_featurevec(self, block_list):
        return self.norm(block_list[-1])[:,0]

    def replace_pe(self, num_channels):
        raise RuntimeError('No need :)')