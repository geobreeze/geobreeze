import torch.nn as nn
import torch
from torch import Tensor
from einops import rearrange
from geofm_src.foundation_models.galileo.util import construct_galileo_input, MaskedOutput
from .galileo.model import Encoder

from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
import os
from torchvision.datasets.utils import download_url
from geofm_src.engine.model import EvalModelWrapper


EXPECTED_CHANNELS = {
    "s2": 10,
    "s1": 2,
}


class GalileoWrapper(EvalModelWrapper):
    def _load_encoder(self, blk_indices):
        URL = "https://hf.co/nasaharvest/galileo/resolve/main/models/base/{}"
        pretrained_path = self.model_config.get("pretrained_path", None)
        print(f"PRETRAINED PATH: {pretrained_path}")

        if pretrained_path and not os.path.exists(pretrained_path):
            # Download weights and config if they don't exist.
            download_url(
                URL.format("encoder.pt"),
                os.path.dirname(pretrained_path),
                filename=os.path.basename(pretrained_path),
            )
            download_url(
                URL.format("config.json"),
                os.path.dirname(pretrained_path),
                filename="config.json",
            )
        path = pretrained_path if pretrained_path else None

        self.encoder = Encoder.load_from_folder(Path(os.path.dirname(path)), device="cpu")
        
        self.norm = self.encoder.norm

        self.galileo_train_config = OmegaConf.load(os.path.join(os.path.dirname(path), "config.json"))

        print(self.galileo_train_config.keys())
        print(self.galileo_train_config)

        self.exit_token_cfg = self.galileo_train_config.training.token_exit_cfg
        self.target_exit_after = self.galileo_train_config.training.target_exit_after
    
        # prepare hooks
        for idx in blk_indices:
            self.encoder.blocks[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))


    def _cache_block(self,x):
        self.cache.append(x)


    def format_input(self, x: Tensor, input_key: str) -> dict[str, Tensor]:
        """Process the input tensor and stack per-sample outputs to form a single MaskedOutput.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W] (for s1, s1-asc, or s2/spot).
            input_key (str): Specifies how to interpret the channels.

        Returns:
            dict[str, Tensor]: A dictionary mapping names as expected by the encoder.
        """
        # Validate expected channel count.
        expected = EXPECTED_CHANNELS.get(input_key)
        if expected is None:
            raise ValueError(f"Unsupported input key: {input_key}")
        assert x.shape[1] == expected, (
            f"Input tensor for {input_key} should have {expected} channels"
        )

        # For keys that require a time dimension unsqueeze and rearrange.
        if input_key in {"s1", "s2"}:
            x = x.unsqueeze(1)  # Now shape: [B, 1, C, H, W]
            x = rearrange(x, "B T C H W -> B H W T C")

        # Process each element in the batch through the Galileo input function, but perhaps also need
        # batched_collate_fn?
        outputs = [construct_galileo_input(**{input_key: x[i]}) for i in range(x.shape[0])]

        # Stack each field of the MaskedOutput over the batch dimension for batched input.
        batched = MaskedOutput(
            space_time_x=torch.stack([o.space_time_x for o in outputs], dim=0),
            space_x=torch.stack([o.space_x for o in outputs], dim=0),
            time_x=torch.stack([o.time_x for o in outputs], dim=0),
            static_x=torch.stack([o.static_x for o in outputs], dim=0),
            space_time_mask=torch.stack([o.space_time_mask for o in outputs], dim=0),
            space_mask=torch.stack([o.space_mask for o in outputs], dim=0),
            time_mask=torch.stack([o.time_mask for o in outputs], dim=0),
            static_mask=torch.stack([o.static_mask for o in outputs], dim=0),
            months=torch.stack([o.months for o in outputs], dim=0),
        )

        return {
            "s_t_x": batched.space_time_x,
            "sp_x": batched.space_x,
            "t_x": batched.time_x,
            "st_x": batched.static_x,
            "s_t_m": batched.space_time_mask,
            "sp_m": batched.space_mask,
            "t_m": batched.time_mask,
            "st_m": batched.static_mask,
            "months": batched.months,
        }

    def get_blocks(self, x):
        self.cache = []
        self.encoder(patch_size=self.model_config.patch_size, **self.format_input(x, self.model_config.input_key))
        blocks = self.cache
        self.cache = [] 
        return blocks

    def get_segm_blks(self, x: Tensor) -> list[torch.Tensor]:
        # we extract blocks from the vit and average over the channel groups
        B, C, H, W = x.shape
        
        blocks = self.get_blocks(x)

        patch_size = self.model_config.patch_size
        h = H // patch_size
        w = W // patch_size

        out = [rearrange(blk, "b (h w t chngrps) d -> b d h w t chngrps", h=h, w=w, t=1) for blk in blocks]
        out = [blk.squeeze(4) for blk in out] # remove time dimension
        out = [blk.mean(dim=4) for blk in out] # average over channel groups
        
        return out

    def default_blocks_to_featurevec(self, block_list):
        x = block_list[-1][:, 1:,:].mean(dim=1)
        x = self.norm(x)
        return x