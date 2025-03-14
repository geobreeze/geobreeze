from .lightning_task import LightningTask
from .GFM import build_swin_cls, build_swin_seg
import torch.nn as nn

# use mmsegmentation for upernet+mae
from mmseg.models.decode_heads import UPerHead, FCNHead

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from ..util.misc import resize, seg_metric, cls_metric, reg_metric

from .base import LinearHead


class GFMClassification(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.full_finetune = model_config.get("full_finetune", False)

        model_config.num_classes = data_config.num_classes
        self.encoder = build_swin_cls(model_config)

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        trunc_normal_(self.encoder.head.weight, std=0.01)
        self.encoder.head = LinearHead(
            in_features=self.encoder.head.in_features,
            num_classes=data_config.num_classes,
        )
        trunc_normal_(self.encoder.head.head[1].weight, std=0.01)
        self.unfreeze(self.encoder.head)

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        out_logits, feats = self.encoder(samples)
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        if self.full_finetune:
            return list(self.encoder.parameters())
        elif self.model_config.get("trainable_params", None):
            # find layer names of trainable layers and return their parameters
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

            return params_to_optimize + list(self.encoder.head.parameters())
        else:
            return list(self.encoder.head.parameters())

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

    
class GFMRegression(GFMClassification):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.criterion = nn.MSELoss()


    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        mse, mae = reg_metric(self.data_config, outputs[0], targets)
        self.log(f"{prefix}_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=True)


class GFMSegmentation(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.encoder = build_swin_seg(model_config)
        # pretrained weights loaded already

        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        edim = model_config.embed_dim
        self.decoder = UPerHead(
            in_channels=[256, 512, 1024, 1024],
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
        feats = self.encoder(samples)
        out = self.decoder(feats)
        out = resize(out, size=samples.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=samples.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def params_to_optimize(self):
        return list(self.decoder.parameters()) + list(self.aux_head.parameters())

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate mIoU and other segmentation-specific metrics
        miou, acc = seg_metric(self.data_config, outputs[0], targets)
        loss = self.loss(outputs, targets)
        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_miou", miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)


# Model factory for different dinov2 tasks
def GFMModel(args, model_config, data_config):
    if args.task == "classification":
        return GFMClassification(args, model_config, data_config)
    elif args.task == "regression":
        return GFMRegression(args, model_config, data_config)
    elif args.task == "segmentation":
        return GFMSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
