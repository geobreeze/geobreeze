import torch
from geobreeze.engine.model import EvalModelWrapper
from torch import nn
from einops import rearrange
from torch import Tensor


class DinoV2(EvalModelWrapper):

    def __init__(self, 
            dinov2_torchhub_id: str,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.dinov2_torchhub_id = dinov2_torchhub_id

    def _load_encoder(self, blk_indices):
        print("BLK INDICES: ", blk_indices)
        self.encoder = torch.hub.load("facebookresearch/dinov2", self.dinov2_torchhub_id)
        self.norm = self.encoder.norm
        self.blk_indices = blk_indices

    def get_blocks(self, x_dict):    
        x = x_dict['imgs']
        if self.encoder.chunked_blocks:
            x_blocks = self.encoder._get_intermediate_layers_chunked(x, self.blk_indices)
        else:
            x_blocks = self.encoder._get_intermediate_layers_not_chunked(x, self.blk_indices)

        return x_blocks

    def default_blocks_to_featurevec(self, block_list):
        out = self.norm(block_list[-1])[:,0]
        return out

    # def default_input_to_feature_list(self, x: Tensor) -> list[torch.Tensor]:
    #     block_list = self.get_blocks(x)
    #     patch_size = int(block_list[0].size(1) ** 0.5)
    #     out = [rearrange(f[:, 1:, :], "b (h w) c -> b c h w", h=patch_size, w=patch_size) for f in block_list]
    #     return out

    def replace_pe(self, num_channels):

        patch_size = self.patch_size
        new_conv2d = nn.Conv2d(
            num_channels, 
            self.encoder.num_features, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.encoder.patch_embed.proj = new_conv2d
        return new_conv2d