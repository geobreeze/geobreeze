from einops import rearrange
import torch
import os
from .SenPaMAE.model import vit_base_patch16
import numpy as np
from geobreeze.engine.model import EvalModelWrapper
import logging
from torch import Tensor
import time

logger = logging.getLogger()



class SenPaMAE(EvalModelWrapper):
    URL = "https://drive.google.com/file/d/1B2g1nm2oxKVgocW22nvEFkFellKZ6ATX"
    # url = 'https://drive.usercontent.google.com/download?id=16IoG47yzdyUnPqUgaV8ofeja5RgQjlAz&export=download&authuser=0&confirm=t&uuid=9e279667-af3a-4f3a-a648-bec3452a1450&at=AIrpjvMEDRsz82ufHQy8sUmSk5j5%3A1739180929862'

    def _load_encoder(self, 
            blk_indices: list[int], 
            pretrained_path: str,
            srf_dir: str,
            create_srf_from_mu_std: bool = False,
        ):
        self.srf_dir = srf_dir
        self.create_srf_from_mu_std = create_srf_from_mu_std

        encoder = vit_base_patch16(
            image_size=self.image_resolution,
            num_channels=-1, # not needed
            emb_dim=self.embed_dim,)

        # download weights
        if not os.path.exists(pretrained_path):
            raise NotImplementedError('Need to manually download weights from above google drive link')

        # load weights
        ckpt = torch.load(pretrained_path, map_location="cpu")
        encoder.load_state_dict(ckpt, strict=False)

        # register fwd hooks
        for idx in blk_indices:
            encoder.transformer[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))

        # set variables
        self.encoder = encoder
        self.norm = self.encoder.layer_norm
        self.srfs_loaded = {}

    def _cache_block(self,x):
        self.cache.append(x)

    def _get_srf_gsds(self, x_dict, device):
        """ srf_filename is list of length of band_ids (i.e. same on batch), 
            gsd is of size (batch_size, C)"""
        srf_filename_list = x_dict['srf_filename']
        band_ids = x_dict['band_ids']
        assert isinstance(srf_filename_list[0], str), \
            f"SRF filename list must be a list of strings. {type(srf_filename_list[0])} != str"
        assert len(srf_filename_list) == len(band_ids), \
            f"SRF filename list and band ids must be the same length. {len(srf_filename_list)} != {len(band_ids)}"

        srfs = []
        for i, f in enumerate(srf_filename_list):
            if not f in self.srfs_loaded:
                srf_path = os.path.join(self.srf_dir, f,)
                if os.path.exists(srf_path):
                    srf = np.load(srf_path).T
                    srf = torch.from_numpy(srf).float()
                    self.srfs_loaded[f] = srf
                else:
                    raise FileNotFoundError(f"SRF not found at {srf_path}")

            srf = self.srfs_loaded[f][band_ids[i]]
            srfs.append(srf)

        srf = torch.stack(srfs, dim=0).to(device)
        gsds = x_dict['gsd']

        return srf, gsds

    def get_blocks(self, x_dict):
        self.cache = []
        imgs: Tensor = x_dict['imgs']

        srf, gsd = self._get_srf_gsds(x_dict, imgs.device)
        srf = srf.expand(imgs.shape[0], -1, -1)
        self.encoder(imgs, gsd=gsd, rf=srf)

        blocks_list = self.cache
        self.cache = []

        return blocks_list

    def get_segm_blks(self, x):
        # get transformer blocks and average over the channel dimension
        B, C, H, W = x.shape
        x = self.get_blocks(x)

        h = H // self.patch_size
        w = W // self.patch_size

        out = [rearrange(blk, 'b (nchn h w) d -> b d h w nchn', h=h, w=w, nchn=C) for blk in x]
        out = [blk.mean(dim=4) for blk in out]

        return out

    def default_blocks_to_featurevec(self, block_list):
        # no official recommendation by the authors, using this for now
        return self.norm(block_list[-1]).mean(dim=1)


def create_srf_from_mu_std(mus: Tensor, sigmas: Tensor):
    assert mus.shape[0] == sigmas.shape[0]
    x_min, x_max = 0, 2301
    x = torch.linspace(x_min, x_max, 2301).unsqueeze(0).to(mus.device)
    vals = torch.exp(-0.5 * ((x - mus.unsqueeze(1)) / sigmas.unsqueeze(1)) ** 2)
    vals = vals / vals.max(dim=1, keepdim=True)[0]
    return vals.transpose(0,1)