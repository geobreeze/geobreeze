import torch
import warnings


# from torch._six import inf
import torch.nn.functional as F

from timm.utils import accuracy as timm_accuracy
from torchmetrics.functional.classification import (
    multilabel_average_precision,
    multilabel_f1_score,
)
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error

from torchmetrics.functional import jaccard_index, accuracy



def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    """This is useful because align_corners=True can cause some artifacts or misalignment, 
   especially if the sizes don’t match in specific ways. This check is not done in F.interpolate.
    """

    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

