import torch.nn as nn
import torch
import os

from torchvision.datasets.utils import download_url
from geobreeze.engine.model import EvalModelWrapper

from einops import rearrange

class SoftConWrapper(EvalModelWrapper):
    URL = "https://huggingface.co/wangyi111/softcon/resolve/main/{}"

    def _load_encoder(self, blk_indices): 
        model_config = self.model_config

        # load dino model
        encoder = torch.hub.load("facebookresearch/dinov2", model_config.dinov2_torchhub_id)

        # apply softcon changes
        embed_dim = encoder.num_features
        num_patches = 1 + int((224 / 14) **2)
        encoder.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder.patch_embed.proj = torch.nn.Conv2d(
            model_config.num_channels, embed_dim, kernel_size=(14, 14), stride=(14, 14)
        )

        # download weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                download_url(
                    self.URL.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )

        # load weights
        ckpt = torch.load(path, map_location="cpu")
        msg = encoder.load_state_dict(ckpt)
        print(msg)

        # assign variables
        self.encoder = encoder
        self.norm = self.encoder.norm
        self.blk_indices = blk_indices
    
    def get_blocks(self, x):
        if self.encoder.chunked_blocks:
            x_blocks = self.encoder._get_intermediate_layers_chunked(x, self.blk_indices)
        else:
            x_blocks = self.encoder._get_intermediate_layers_not_chunked(x, self.blk_indices)

        return x_blocks

    def default_blocks_to_featurevec(self, block_list):
        return self.norm(block_list[-1])[:,0]

    def default_blocks_to_feature_list(self, block_list) -> list[torch.Tensor]:
        patch_size = int(block_list[0].size(1) ** 0.5)
        out = [rearrange(f[:, 1:, :], "b (h w) c -> b c h w", h=patch_size, w=patch_size) for f in block_list]
        return out

    def replace_pe(self, num_channels):

        patch_size = self.model_config.patch_size
        new_conv2d = nn.Conv2d(
            num_channels, 
            self.encoder.num_features, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.encoder.patch_embed.proj = new_conv2d
        return new_conv2d
        