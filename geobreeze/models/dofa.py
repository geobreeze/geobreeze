import torch
from torch import Tensor
import os
from .DOFA.models_dwv import vit_base_patch16 as vit_base_patch16_cls
from .DOFA.models_dwv import vit_large_patch16 as vit_large_patch16_cls
from torchvision.datasets.utils import download_url

from geobreeze.engine.model import EvalModelWrapper
from einops import rearrange



class Dofa(EvalModelWrapper):
    URL = "https://huggingface.co/earthflow/dofa/resolve/main/{}"

    def _load_encoder(self, blk_indices, size, hf_filename):

        # build encoder
        if size == "base":
            assert self.patch_size == 16
            encoder = vit_base_patch16_cls()
        elif size == "large":
            assert self.patch_size == 16
            encoder = vit_large_patch16_cls()
        else:
            raise ValueError(f"Size {size} not supported for DOFA")

        url = self.URL.format(hf_filename)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        encoder.load_state_dict(state_dict, strict=False)
        
        # assign variables
        self.encoder = encoder
        if self.encoder.global_pool:
            self.norm = self.encoder.fc_norm
        else:
            self.norm = self.encoder.norm
        
        # prepare hooks
        for idx in blk_indices:
            self.encoder.blocks[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))


    def _process_chn_ids(self, chn_ids: Tensor):
        chn_ids[chn_ids > 0] = chn_ids[chn_ids > 0] / 1e3 # dofa uses microns
        chn_ids[chn_ids < 0] = 5.6 # dofa uses 5600 nm for all SAR channels
        return chn_ids

    def _cache_block(self,x):
        self.cache.append(x)

    def get_blocks(self, x_dict):
        self.cache = []
        chn_ids = self._process_chn_ids(x_dict['chn_ids'])
        chn_ids = chn_ids[0] # same chn_ids for all images in batch
        self.encoder(x_dict['imgs'], chn_ids)
        blocks = self.cache
        self.cache = [] 
        return blocks

    # def default_input_to_feature_list(self, x: Tensor) -> list[torch.Tensor]:
    #     block_list = self.get_blocks(x)
    #     patch_size = int(block_list[0].size(1) ** 0.5)
    #     out = [rearrange(f[:, 1:, :], "b (h w) c -> b c h w", h=patch_size, w=patch_size) for f in block_list]
    #     return out

    def default_blocks_to_featurevec(self, block_list):
        x = block_list[-1][:, 1:,:].mean(dim=1)
        x = self.norm(x)
        return x
    
