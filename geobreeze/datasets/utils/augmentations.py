import torch
from torch import Tensor
import kornia.augmentation as K
from copy import deepcopy

class DownUpSample:
    """ up and downsample images of specific ids"""

    # def __init__(self, *tuples):
    #     self.bandids_to_relgsd = {k: v for k, v in tuples}

    def __init__(self, **kwargs):
        self.bandids_to_relgsd = {int(k): float(v) for k, v in kwargs.items()}
        self.interpol_args = dict(mode='bilinear', align_corners=False)


    def __call__(self, x_dict: dict):
        img: Tensor = x_dict['imgs']    
        assert img.shape[-1] == img.shape[-2], "Image must be square"
        input_size = img.shape[-1]

        for abs_band_id, relgsd in self.bandids_to_relgsd.items():

            band_ids = [i for i, id in enumerate(x_dict['band_ids']) if id == abs_band_id]
            
            for band_id in band_ids:
                img[band_id] = torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        img[band_id][None,None,:,:], scale_factor = relgsd, **self.interpol_args
                    ),
                    size=input_size, **self.interpol_args
                )
                if 'gsd' in x_dict:
                    x_dict['gsd'][band_id] = x_dict['gsd'][band_id] / relgsd

        x_dict['imgs'] = img
        return x_dict
    

class KorniaWrapper:
    def __init__(self, __target__, **kwargs):
        self.trf = K.__dict__[__target__](**kwargs)

    def __call__(self, x_dict: dict):
        x_dict['imgs'] = self.trf(x_dict['imgs'])
        return x_dict