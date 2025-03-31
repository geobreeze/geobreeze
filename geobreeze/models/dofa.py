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

    def _load_encoder(self, blk_indices):

        # build encoder
        size = self.model_config.size
        if size == "base":
            encoder = vit_base_patch16_cls()
        elif size == "large":
            encoder = vit_large_patch16_cls()
        else:
            raise ValueError(f"Size {size} not supported for DOFA")

        # download weights
        if self.model_config.get("pretrained_path", None):
            path = self.model_config.pretrained_path
            if not os.path.exists(path):
                # download the weights from HF
                download_url(
                    self.URL.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),)

        # load weights
        check_point = torch.load(path, map_location="cpu")
        encoder.load_state_dict(check_point, strict=False)
        
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
            
        # prepare wavelengths
        wavelengths_mean_microns = []
        for wl in self.data_config['wavelengths_mean_nm']:
            if wl > 0:
                wl = wl / 1e3
            else:
                wl = 5.6 # dofa uses 5600 nm for all SAR channels
            wavelengths_mean_microns.append(wl)
        self.wavelengths_mean_microns = wavelengths_mean_microns

    def _cache_block(self,x):
        self.cache.append(x)

    def get_blocks(self, x_dict):
        self.cache = []
        self.encoder(x_dict['imgs'], self.wavelengths_mean_microns)
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
    
