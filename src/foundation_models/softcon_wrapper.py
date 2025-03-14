import torch.nn as nn
import torch
import os

# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from .lightning_task import LightningTask
from einops import rearrange
from ..util.misc import resize, seg_metric, cls_metric, reg_metric
from torchvision.datasets.utils import download_url

from .base import LinearHead


class SoftConClassification(LightningTask):
    """SoftCon model for classification."""

    url = "https://huggingface.co/wangyi111/softcon/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.full_finetune = model_config.get("full_finetune", False)

        # load dino model
        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        # add softcon input layer
        self.encoder.patch_embed.proj = torch.nn.Conv2d(
            model_config.num_channels, 384, kernel_size=(14, 14), stride=(14, 14)
        )

        # look for Softcon pretrained weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                download_url(
                    self.url.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )

            ckpt_vit14 = torch.load(path)
            self.encoder.load_state_dict(ckpt_vit14)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        # TODO this embed dim could be pulled from the encoder, to remove the need for the arg
        self.linear_classifier = LinearHead(
            in_features=model_config.embed_dim, num_classes=data_config.num_classes
        )
        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        out = self.encoder.forward_features(samples)
        global_pooled = out["x_norm_patchtokens"].mean(dim=1)
        out_logits = self.linear_classifier(global_pooled)
        return out_logits, global_pooled

    def params_to_optimize(self):
        if self.full_finetune:
            return self.encoder.parameters() + self.linear_classifier.parameters()
        elif self.model_config.get("trainable_params", None):
            trainable_params = self.model_config.trainable_params
            params_to_optimize = []
            for name, param in self.encoder.named_parameters():
                for layer in trainable_params:
                    if layer in name:
                        params_to_optimize.append(param)

            if not params_to_optimize:
                model_params = [name for name, _ in self.encoder.named_parameters()]
                raise ValueError(
                    f"No trainable layers found. Check the layer names in the model. Looking at `self.encoder.named_parameters()`, we have found {model_params}"
                )
            return params_to_optimize + list(self.linear_classifier.parameters())
        else:
            return self.linear_classifier.parameters()

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)

class SoftConRegression(SoftConClassification):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.criterion = nn.MSELoss()


    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        mse, mae = reg_metric(self.data_config, outputs[0], targets)
        self.log(f"{prefix}_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=True)



class SoftConSegmentation(LightningTask):
    """SoftCon Model for Segmentation."""

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        # load dino model
        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        # add softcon input layer
        self.encoder.patch_embed.proj = torch.nn.Conv2d(
            model_config.num_channels, 384, kernel_size=(14, 14), stride=(14, 14)
        )

        # look for Softcon pretrained weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                download_url(
                    self.url.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )

            ckpt_vit14 = torch.load(path)
            self.encoder.load_state_dict(ckpt_vit14)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        self.aux_head = FCNHead(
            in_channels=edim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=data_config.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(
            outputs[1], labels
        )

    def forward(self, samples):
        outputs = self.encoder.get_intermediate_layers(samples, [4, 6, 10, 11])
        feats = [
            rearrange(out, "n (h w) c -> n c h w", h=int(out.size(1) ** 0.5))
            for out in outputs
        ]
        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)


# Model factory for different dinov2 tasks
def SoftConModel(args, model_config, data_config):
    if args.task == "classification":
        return SoftConClassification(args, model_config, data_config)
    elif args.task == "regression":
        return SoftConRegression(args, model_config, data_config)
    elif args.task == "segmentation":
        return SoftConSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
