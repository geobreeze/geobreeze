import torch.nn as nn
from einops import rearrange 
import logging

logger = logging.getLogger('eval')

""" 
Design decisions:
- In alignment with dinov2, we freeze the EvalModelWrapper.norm. The norm is used
    for different blocks to feature extractions. To account for potentially changing
    inputs, engine=lightning always adds a (trainable) 1d-batchnorm in the linear head
    and engine=accelerated tries heads both with and without such a norm (specifiable 
    in the config).

"""

class EvalModelWrapper(nn.Module):

    def __init__(self, 
            image_resolution: int,
            embed_dim: int,
            patch_size: int,
            blk_indices: list,
            **kwargs
        ):
        super().__init__()
        
        self.image_resolution = image_resolution
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self._load_encoder(blk_indices, **kwargs)
        assert hasattr(self, "encoder"), "Encoder has not been loaded!"
        assert hasattr(self, "norm"), "Normalization function has not been loaded!"
        print(f'Loaded encoder with blocks {blk_indices} blocks and norm {self.norm}')

    def _load_encoder(self, blk_indices):
        """ 
        Loads the encoder and prepares any functionality needed for extracting the 
        blocks index by blk_indices (e.g. register forward hooks) in get_blocks.
        Also needs to save the encoder and the norm function for  
        normalizing features in self.encoder and self.norm respectively.

        Important: Make sure that self.norm is not a submodule of self.encoder!
        Else, the optimizer might optimize that param twice.

        Input:
            blk_indices: list of indices of intermediate features to return
        """
        self.encoder = None
        self.norm = None
        raise NotImplementedError()
    
    def get_blocks(self, x):
        """ 
        Main function to extract features. Extracting blocks allows segmentation
        and multiple different mappings from blocks to a single representation. 

        Input: 
            x: input tensor of size [b,c,h,w]
            indices: list of indices of intermediate features to return

        Output: 
            list of intermediate features indexed by indices, 
            each element is of size [b,p,d] where p is the number of patches
            and d the embedding dimension
        """
        raise NotImplementedError()


    def default_blocks_to_featurevec(self, block_list):
        """
        Takes the output of get_blocks and returns a single feature vector computed 
        from this list. During classification / regression, essentially this happens:

            block_list = wrapper.get_blocks(x)
            feature_vec = wrapper.default_blocks_to_featurevec(block_list)
            logits = linear_classifier_head(feature_vec)
        
        This function should be the default method suggested by the authors of 
        the model to extract features from the blocks (or only the last block).
        The advantage of separating this function from get_blocks is that we 
        can test different aggregation functions during linear probing.

        This function is only needed for classification / regression!

        Input: 
            block_list: output of forward_feats

        Output:
            feature vector of size [b,d]
        """
        raise NotImplementedError()

    def replace_pe(self, num_channels):
        """
        Replaces the positional encoding of the model with a new positional encoding
        that can handle `num_channels` input channels. The new PE should
        be initialized from scratch. This is useful to adjust models to 
        different number of input channels

        Input:
            num_channels: number of input channels

        Returns:
            module or list of parameters to optimize.
        """
        raise NotImplementedError()
    
    def get_segm_blks(self, x):
        """ 
        extracts segmentation blocks as list of tensors where each 
        tensor has shape [b,d,h,w]. This default function works for standard
        ViT with a single cls token.
        """
        block_list = self.get_blocks(x)
        patch_size = int(block_list[0].size(1) ** 0.5) # assume quadratic images
        block_list = [blk[:, 1:, :] for blk in block_list] # drop cls token
        out = [rearrange(blk, "b (h w) d -> b d h w", h=patch_size, w=patch_size) 
               for blk in block_list]
        return out