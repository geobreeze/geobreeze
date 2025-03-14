import torch
import os
import torch.nn as nn
from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead

from .lightning_task import LightningTask
from timm.models.layers import trunc_normal_
from ..util.misc import resize, seg_metric, cls_metric
from torchvision.datasets.utils import download_url
from peft import LoraConfig, get_peft_model
from .SenPaMAE.model import vit_base_patch16
import numpy as np

from .base import LinearHead


class SenPaMAEClassification(LightningTask):
    url = "https://drive.google.com/file/d/16IoG47yzdyUnPqUgaV8ofeja5RgQjlAz"

    # url = 'https://drive.usercontent.google.com/download?id=16IoG47yzdyUnPqUgaV8ofeja5RgQjlAz&export=download&authuser=0&confirm=t&uuid=9e279667-af3a-4f3a-a648-bec3452a1450&at=AIrpjvMEDRsz82ufHQy8sUmSk5j5%3A1739180929862'

    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)
        self.lora = model_config.get("lora", False)
        self.full_finetune = model_config.get("full_finetune", False)
        # can only be one of the two
        assert not (self.lora and self.full_finetune), (
            "Can only use one of LoRA or full finetune bot not both to true"
        )

        # senpamae_cfg = omegaconf.OmegaConf.load(senpamae_cfg_path)
        # init the encoder part of the model
        # self.encoder = hydra.utils.instantiate(args.senpamae_cfg)
        self.encoder = vit_base_patch16(
            image_size=model_config.image_size,
            num_channels=model_config.num_channels,
            emb_dim=model_config.embed_dim,
        )
        print(self.encoder)

        # look for pretrained weights
        if model_config.get("pretrained_path", None):
            path = model_config.pretrained_path
            if not os.path.exists(path):
                download_url(
                    self.url.format(os.path.basename(path)),
                    os.path.dirname(path),
                    filename=os.path.basename(path),
                )
            # Load pretrained weights
            check_point = torch.load(path)
            self.encoder.load_state_dict(check_point, strict=False)

        # LoRA
        if self.lora and model_config.lora:
            self.apply_peft(self.encoder, lora_cfg=model_config.lora)

        # setup linear head
        self.linear_classifier = LinearHead(
            in_features=model_config.embed_dim, num_classes=data_config.num_classes
        )
        trunc_normal_(self.linear_classifier.head[1].weight, std=0.01)

        self.encoder.head = nn.Sequential(
            nn.BatchNorm1d(model_config.embed_dim, affine=False, eps=1e-6),
            self.linear_classifier,
        )

        if model_config.freeze_backbone:
            if self.lora:
                # TODO not implemented yet I think
                self.freeze_non_lora_params(self.encoder)
            else:
                self.freeze(self.encoder)
                self.unfreeze(self.encoder.head)

        self.criterion = (
            nn.MultiLabelSoftMarginLoss()
            if data_config.multilabel
            else nn.CrossEntropyLoss()
        )

        self.model_config = model_config
        self.data_config = data_config

        self.process_srfs()

    def process_srfs(self):
        # SRF loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Go up one level

        srf_path = os.path.join(
            parent_dir,
            "foundation_models/SenPaMAE/responsefunctions",
            self.data_config.senpamae_srf_name,
        )
        if not os.path.exists(srf_path):
            raise ValueError(f"SRF not found at {srf_path}")
        self.srf = np.load(srf_path).T
        subset_channels = self.data_config.senpamae_channels
        self.srf = self.srf[subset_channels, :]
        self.srf = torch.from_numpy(self.srf).float()
        self.srf = self.srf.unsqueeze(0)

        # Convert band_gsds to numpy array and index it
        self.band_gsds = np.array(self.data_config.band_gsds)[
            self.data_config.senpamae_channels
        ]
        self.band_gsds = torch.tensor(self.band_gsds).float().unsqueeze(0)

        print("SRF shape: ", self.srf.shape)
        print("Selected GSDs: ", self.band_gsds, self.band_gsds.shape)

    def freeze_non_lora_params(self, encoder):
        raise NotImplementedError(
            "Not implemented yet: CANNOT freeze non-LoRA parameters"
        )

    def apply_peft(self, encoder, lora_cfg: dict):
        """
        Apply LoRA to the last few layers of the encoder using PEFT.
        """

        print("LORA: Applying PEFT: ", lora_cfg)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_cfg.get("lora_rank", 16),  # Rank of LoRA
            lora_alpha=lora_cfg.get("lora_alpha", 16),  # Scaling factor for LoRA
            target_modules=lora_cfg.get(
                "lora_target_modules", "blocks.*.attn.qkv"
            ),  # ["qkv", "proj"]
            lora_dropout=lora_cfg.get("lora_dropout", 0.0),  # Dropout rate for LoRA
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get(
                "lora_task_type", None
            ),  # Task type (use appropriate type for your model), "SEQ_CLS"
        )

        # Wrap the encoder with PEFT
        self.encoder = get_peft_model(encoder, peft_config)

    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels)

    def forward(self, samples):
        # subset channels
        samples = samples[:, self.data_config.senpamae_channels, :, :]

        device = samples.device
        srf = self.srf.to(device)
        gsd = self.band_gsds.to(device)

        # apply SRF
        feats, _ = self.encoder(samples, gsd=gsd, rf=srf)
        # swap 0 and 1 dims
        feats = feats.permute(1, 0, 2)  # (batch, tokens, features)

        # mean pool over token dimension
        globally_pooled = feats.mean(dim=1)
        out_logits = self.encoder.head(globally_pooled)

        return out_logits, globally_pooled

    def params_to_optimize(self):
        if self.lora:
            # Include LoRA parameters for optimization
            lora_params = [p for n, p in self.encoder.named_parameters() if "lora" in n]
            return list(self.encoder.head.parameters()) + lora_params
        elif self.full_finetune:
            return list(self.encoder.parameters())
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
            return params_to_optimize + self.encoder.head.parameters()
        else:
            return list(self.encoder.head.parameters())

    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        acc1, acc5 = cls_metric(self.data_config, outputs[0], targets)
        self.log(
            f"{prefix}_loss",
            self.loss(outputs, targets),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{prefix}_acc1", acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_acc5", acc5, on_step=True, on_epoch=True, prog_bar=True)


class SenPaMAERegression(SenPaMAEClassification):
    def __init__(self, args, model_config, data_config):
        super().__init__(args, model_config, data_config)

        self.criterion = nn.MSELoss()


    def log_metrics(self, outputs, targets, prefix="train"):
        # Calculate accuracy and other classification-specific metrics
        mse, mae = reg_metric(self.data_config, outputs[0], targets)
        self.log(f"{prefix}_mse", mse, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=True, on_epoch=True, prog_bar=True)



# Model factory for different dinov2 tasks
def SenPaMAEModel(args, model_config, data_config):
    if args.task == "classification":
        return SenPaMAEClassification(args, model_config, data_config)
    elif args.task == "regression":
        return SenPaMAERegression(args, model_config, data_config)
    # elif args.task == "segmentation":
    #     return DofaSegmentation(args, model_config, data_config)
    else:
        raise NotImplementedError("Task not supported")
