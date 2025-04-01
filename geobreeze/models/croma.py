from .CROMA.use_croma import PretrainedCROMA
import torch
from torch import Tensor
import os
from torchvision.datasets.utils import download_url
from einops import rearrange

from geobreeze.engine.model import EvalModelWrapper 


class Croma(EvalModelWrapper):
    URL = "https://huggingface.co/antofuller/CROMA/resolve/main/{}"

    def _load_encoder(self, 
            blk_indices,
            modality,
            hf_filename, 
            size
        ):

        self.modality = modality
        ckpt_path = os.path.join(
            os.environ['MODEL_WEIGHTS_DIR'], hf_filename)

        if not os.path.exists(ckpt_path):
            download_url(
                self.URL.format(hf_filename),
                root = os.environ['MODEL_WEIGHTS_DIR'],
                filename = hf_filename,
            )

        encoder = PretrainedCROMA(
            pretrained_path=ckpt_path,
            size=size,
            modality=modality,
            image_resolution=self.image_resolution,)

        if modality == "optical":
            self.norm = encoder.s2_encoder.transformer.norm_out 
            encoder.s2_encoder.transformer.out_indices = blk_indices 
            encoder.s2_GAP_FFN = torch.nn.Identity()

        elif modality == "SAR":
            self.norm = encoder.s1_encoder.transformer.norm_out
            encoder.s1_encoder.transformer.out_indices = blk_indices 
            encoder.s1_GAP_FFN = torch.nn.Identity()

        self.encoder = encoder
    

    def get_blocks(self, x_dict):
        mod = self.modality
        out = self.encoder(**{f"{mod}_images": x_dict['imgs']})
        out_features = out['out_feats']

        # rearrange to requested dim
        out_features = [
            rearrange(out, "b c p q -> b (p q) c")
            for out in out_features
        ]

        return out_features
    
    def default_blocks_to_featurevec(self, block_list):
        # the following is how the tokens are passed into the {mod}_GAP_FFN networks
        return self.norm(block_list[-1]).mean(dim=1)

    def get_segm_blks(self, x_dict: Tensor) -> list[torch.Tensor]:
        """ need to overwrite because croma has no class token. """
        return self.encoder(**{f"{self.modality}_images": x_dict})['out_feats']
    
    def replace_pe(self, num_channels):
        mod = self.modality
        if mod == 'optical':
            enc = self.encoder.s2_encoder
        elif mod == 'SAR':
            enc = self.encoder.s1_encoder

        pixels_per_patch = int(enc.patch_size * enc.patch_size * num_channels)
        linear_input = torch.nn.Linear(pixels_per_patch, enc.dim)
    
        enc.linear_input = linear_input
        return linear_input