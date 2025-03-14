from .lightning_task import LightningTask
from .CROMA.use_croma import PretrainedCROMA
import torch.nn as nn
import torch
import os
from torchvision.datasets.utils import download_url

# use mmsegmentation for upernet+mae
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from ..util.misc import resize, seg_metric, cls_metric, reg_metric

from .base import LinearHead


class CromaClassification(LightningTask):
    url = "https://huggingface.co/antofuller/CROMA/resolve/main/{}"

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.full_finetune = model_config.get("full_finetune", False)

        # look for pretrained weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                # download the weights from HF
                download_url(
                    self.url.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )
        else:
            path = None

        self.encoder = PretrainedCROMA(
            pretrained_path=path,
            size=model_config.size,
            modality=model_config.modality,
            image_resolution=model_config.image_resolution,
        )

        # pretrained weights loaded
        if model_config.freeze_backbone:
            self.freeze(self.encoder)

        # TODO the original croma model has more layers in the head
        # why do we truncate it?
        self.encoder.s2_GAP_FFN[1] = torch.nn.Linear(
            self.encoder.s2_GAP_FFN[1].in_features, data_config.num_classes
        )
        del self.encoder.s2_GAP_FFN[2:]
        self.unfreeze(self.encoder.s2_GAP_FFN[1])

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        all_output = self.encoder(optical_images=samples)
        out_logits = all_output["optical_GAP"]
        feats = all_output["optical_encodings"]
        return (out_logits, feats) if self.model_config.out_features else out_logits

    def params_to_optimize(self):
        if self.full_finetune:
            return self.encoder.parameters()
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
            return params_to_optimize + list(
                self.self.encoder.s2_GAP_FFN[1].parameters()
            )
        else:
            return self.encoder.s2_GAP_FFN[1].parameters()

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


class CromaRegression(CromaClassification):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.criterion = nn.MSELoss()


    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        mse, mae = reg_metric(self.data_config, outputs[0], targets)
        self.log(f"{prefix}_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=True)


class CromaSegmentation(LightningTask):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.encoder = PretrainedCROMA(
            pretrained_path=model_config.pretrained_path,
            size=model_config.size,
            modality=model_config.modality,
            image_resolution=model_config.image_resolution,
        )

        # pretrained weights Loaded
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
        feats = self.encoder(optical_images=samples)["out_feats"]
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
def CromaModel(args, model_config, data_config):
    if args.task == "classification":
        return CromaClassification(args, model_config, data_config)
    elif args.task == "regression":
        return CromaRegression(args, model_config, data_config)
    elif args.task == "segmentation":
        return CromaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
