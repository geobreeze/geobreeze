import torch.nn as nn
import torch
from torch import Tensor
from einops import repeat, rearrange
from geobreeze.engine.model import EvalModelWrapper
import logging
import sys
import os

logger = logging.getLogger('eval')


class AnySat(EvalModelWrapper):
    def _load_encoder(
            self, 
            blk_indices: list,
            torchhub_id: str,
            input_key: str,
        ):
        self.input_key = input_key

        p = os.path.join(os.environ['TORCH_HOME'],'hub/gastruc_anysat_main/src/',)
        sys.path.insert(0, p)
        logger.info(f'Inserted {p} at pos 0 into sys.path')
        
        # Save the current state of sys.modules to a file
        # modules = sys.modules.keys()
        # formatted_modules = "\n".join(sorted(modules))
        # modules_file = '/home/hk-project-pai00028/tum_mhj8661/code/test.txt'
        # with open(modules_file, 'w') as f:
        #     f.write(formatted_modules)
        # print(f'Wrote {len(modules)} modules to {modules_file}')

        self.encoder = torch.hub.load(
            "gastruc/anysat", 
            torchhub_id, 
            pretrained=True, 
            flash_attn=False,
            force_reload=False,
            verbose=True)

        # Dont think there is a norm layer to be used here
        self.norm = nn.Identity()
        
        # prepare hooks
        for idx in blk_indices:
            self.encoder.model.blocks[idx].register_forward_hook(
                lambda m, i, o: self._cache_block(o))

    def _cache_block(self,x):
        self.cache.append(x)


    def format_input(self, x: Tensor, input_key: str) -> dict[str, Tensor]:
        """Format input tensor to be passed to the model.
        
        According to https://github.com/gastruc/AnySat?tab=readme-ov-file#format-your-data
        Args:
            x (Tensor): input tensor
        Returns:
            dict[str, Tensor]: formatted input tensor
        """
        dates = None
        if not self.do_replace_pe:
            match input_key:
                case "s2":
                    assert x.shape[1] == 10, "Input tensor for s2 should have 10 channels"
                    # unsqueeze time dimension
                    x = x.unsqueeze(1)
                    dates = torch.arange(x.shape[1]).float()

                case "s1":
                    assert x.shape[1] == 3, "Input tensor for s1 should have 3 channels"
                    # unsqueeze time dimension
                    x = x.unsqueeze(1)
                    dates = torch.arange(x.shape[1]).float()
                case "s1-asc":
                    assert x.shape[1] == 2, "Input tensor for s1-asc should have 2 channels"
                    # unsqueeze time dimension
                    x = x.unsqueeze(1)
                    dates = torch.arange(x.shape[1]).float()

                case "l7":
                    assert x.shape[1] == 6, "Input tensor for l7 should have 6 channels"
                    # unsqueeze time dimension
                    x = x.unsqueeze(1)
                    dates = torch.arange(x.shape[1]).float()

                case "spot":
                    assert x.shape[1] == 3, "Input tensor for spot should have 3 channels"
                    dates = None

                case "naip":
                    assert x.shape[1] == 4, "Input tensor for naip should have 4 channels"
                    dates = None

                case 'aerial':
                    assert x.shape[1] == 4, "Input tensor for aerial should have 4 channels"
                    dates = None

        anysat_input = {input_key: x}
        if dates is not None:
            dates = repeat(dates, 't -> b t', b=x.shape[0])
            anysat_input[f'{input_key}_dates'] = dates.to(x.device)

        return anysat_input

    def get_blocks(self, x_dict):
        self.cache = []
        # TODO these arguments will be different for segmentation 
        self.encoder(
            self.format_input(x_dict['imgs'], self.input_key), 
            patch_size=self.patch_size, 
            output='tile')
        blocks = self.cache
        self.cache = [] 
        return blocks

    # def default_input_to_feature_list(self, x: Tensor) -> list[torch.Tensor]:
    #     # leawldm: I don't think we actually need the dense prediction mode if we 
    #     #   extract the blocks with the hooks. Only if we use the output,
    #     #   this will make a diff.
    #     self.cache = []
    #     self.encoder(self.format_input(x, self.input_key), patch_size=self.patch_size, output='dense', output_modality=self.model_config.input_key)
    #     blocks = self.cache
    #     self.cache = []
    #     patch_size = int(blocks[0].size(1) ** 0.5)
    #     out = [rearrange(f[:, 1:, :], "b (h w) c -> b c h w", h=patch_size, w=patch_size) for f in blocks]
    #     return out

    def default_blocks_to_featurevec(self, block_list):
        # anysats default aggregation is just returning the last 
        x = block_list[-1][:, 1:,:].mean(dim=1)
        return x
    
    def replace_pe(self, num_channels):

        projector_str = f'projector_{self.input_key}'
        projector = getattr(self.encoder.model, projector_str)
        #we want to get to encoder.model.projector_naip.patch_embed
        # patch_embed = projector.patch_embed
        assert projector is not None

        # Use the original patch size for the new conv2d
        if self.input_key == "naip":
            patch_size = (8, 8)
        elif self.input_key == "spot":
            patch_size = (10, 10)
        else:
            raise NotImplementedError(f"Patch size not implemented for {self.input_key}: use either naip or spot")


        new_conv2d = nn.Conv2d(
            num_channels, 
            self.embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=False
        )
        logger.info(f"Replacing patch embed with new conv2d: {new_conv2d}")
        projector.patch_embed = new_conv2d
        logger.info(projector)
        # self.encoder.patch_embed.proj = new_conv2d
        return new_conv2d

