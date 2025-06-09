import torch
from geobreeze.engine.model import EvalModelWrapper


class Panopticon(EvalModelWrapper):

    def _load_encoder(self, blk_indices, torchhub_id):
        self.encoder = torch.hub.load(
            'panopticon-FM/panopticon',
            torchhub_id,)
        # self.encoder = torch.hub.load(
        #     '/home/hk-project-pai00028/tum_mhj8661/code/PanOpticOn',
        #     'panopticon_vitb14',
        #     source='local')
        self.norm = self.encoder.norm
        self.blk_indices = blk_indices
    
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