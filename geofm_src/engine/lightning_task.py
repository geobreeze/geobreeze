"""Base Class."""

from lightning import LightningModule
import torch
from geofm_src.util.misc import resize, seg_metric
import torch.nn as nn
from .model import EvalModelWrapper
from einops import rearrange 
from .base import LinearHead
from torch import Tensor

from geofm_src.engine.accelerated.utils.metrics import build_metric, build_criterion

try:
    from mmseg.models.necks import Feature2Pyramid
    from mmseg.models.decode_heads import UPerHead, FCNHead
    MMSEGM_AVAIL = True
except:
    print("MMSEG not installed, skipping imports")
    MMSEGM_AVAIL = False


class LightningTask(LightningModule):
    def __init__(self, args, model_config, data_config, encoder):
        super().__init__()
        self.encoder = encoder
        self.model_config = model_config  # model_config
        self.args = args  # args for optimization params
        self.data_config = data_config  # dataset_config
        self.training_mode = model_config.training_mode
        self.replace_pe = model_config.get('replace_pe', False)
        self.save_hyperparameters()

        self.train_metrics = build_metric(
            args.task_kwargs.train, num_classes=data_config.num_classes, key_prefix='train/') 
        self.val_metrics = build_metric(
            args.task_kwargs.val, num_classes=data_config.num_classes, key_prefix='val/') 
        self.test_metrics = build_metric(
            args.task_kwargs.val, num_classes=data_config.num_classes, key_prefix='test/') 

        if self.replace_pe:
            self.new_pe = self.encoder.replace_pe(data_config.num_channels)
            print('Replaced PE!')

    def freeze_and_return_params(self):
        """ freeze & unfreeze weights according to self.training_mode, also
            returns all parameters to optimize """
        raise NotImplementedError('Subclass must implement this method')

    def forward(self, x):
        raise NotImplementedError('Subclass must implement this method')

    def loss(self, outputs, labels):
        raise NotImplementedError('Subclass must implement this method')

    def log_metrics(self, outputs, targets, loss, prefix="train"):
        metrics = self.__getattr__(f"{prefix}_metrics")
        out_dict = metrics(outputs, targets)

        on_step = True if prefix == 'train' else False
        on_epoch = True

        self.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log_dict(out_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=True)


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix="test")

    def _proc_param_obj(self, obj):
        if isinstance(obj, nn.Module): 
            params = obj.parameters()
        elif isinstance(obj, list):
            if len(obj) == 0:
                params = []
            elif len(obj[0]) == 1: # .parameters()
                params = obj
            elif len(obj[0]) == 2: # .named_parameters()
                params = [p for _, p in obj]
            else:
                raise ValueError(f"Invalid list of parameters: {obj}")
        else:
            raise ValueError(f"Invalid object: {obj}")
        return params

    def freeze(self, obj):
        for p in self._proc_param_obj(obj):
            p.requires_grad = False

    def unfreeze(self, obj):
        for p in self._proc_param_obj(obj):
            p.requires_grad = True

    def configure_optimizers(self):
        if self.data_config.task in ["classification", "regression"]:
            optimizer = torch.optim.SGD(
                self.freeze_and_return_params(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(self.freeze_and_return_params(), lr=self.args.lr)

        world_size = self.args.num_gpus if self.args.num_gpus >= 1 else 1
        num_warmup_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.warmup_epochs
            // world_size)
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            * self.args.epochs
            // world_size)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=total_steps,
            anneal_strategy="cos",  # Cosine annealing
            pct_start=float(num_warmup_steps)
            / float(total_steps),  # Percentage of warmup
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _filter_named_params(self, named_params, targets):
        out = []
        for name, param in named_params:
            for t in targets:
                if t in name:
                    out.append((name, param))
        return out


class LightningClsRegTask(LightningTask):

    encoder: EvalModelWrapper

    def __init__(self, args, model_config, data_config, encoder: EvalModelWrapper):
        super().__init__(args, model_config, data_config, encoder)

        self.criterion = build_criterion(args.task_kwargs.criterion)
        # Batchnorm + Linear
        self.linear_classifier = LinearHead(in_features=model_config.embed_dim, num_classes=data_config.num_classes)

    def freeze_and_return_params(self):
        """ freeze / unfreeze weights & return parameters to optimize 
            according to self.training_mode"""
        mode = self.training_mode

        # prepare encoder 
        if mode == 'linear_probe':
            self.freeze(self.encoder)
            self.encoder.eval()

            params_to_optimize = self.linear_classifier.parameters()

        elif mode == 'partial_finetune':
            self.freeze(self.encoder)
            encoder_params_to_unfreeze = self._filter_named_params(
                self.encoder.named_parameters(), self.model_config.params_to_train)
            self.unfreeze(encoder_params_to_unfreeze)

            params_to_optimize = list([p for _, p in encoder_params_to_unfreeze])

        elif mode == 'lora':
            self.freeze(self.encoder)
            lora_params = self._filter_named_params(
                self.encoder.named_parameters(), ['lora'])
            assert len(lora_params) > 0, "Did not find any LoRA parameters in the encoder"
            self.unfreeze(lora_params)

            params_to_optimize = list([p for _, p in lora_params])

        elif mode == 'full_finetune':
            self.unfreeze(self.encoder) 

            params_to_optimize = list(self.encoder.parameters())

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # prepare linear classifier
        self.unfreeze(self.linear_classifier)
        params_to_optimize += list(self.linear_classifier.parameters())

        if self.replace_pe:
            self.unfreeze(self.new_pe)
            params_to_optimize += list(self.new_pe.parameters())

        return params_to_optimize
    
    def _step(self, batch, batch_idx, prefix="train"):
        images, targets = batch
        
        if self.training_mode == 'linear_probe':
            with torch.no_grad():
                x = self.encoder.get_blocks(images)
        else:
            x = self.encoder.get_blocks(images)
        x = self.encoder.default_blocks_to_featurevec(x)
        outputs = self.linear_classifier(x)

        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, loss, prefix)
        return loss

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def _get_encoder_params_without_head(self, verbose=False, with_name=True):
        if self.dot_str_of_linear_classifier is None:
            out = self.encoder.named_parameters()
        else:
            out = []
            for n,p in self.encoder.named_parameters():
                if self.dot_str_of_linear_classifier not in n:
                    out.append((n,p))
                elif verbose:
                    print(f"Skipping {n} from encoder parameters")

        if not with_name:
            out = [p for _, p in out]
        return out


class LightningSegmentationTask(LightningTask):

    encoder: EvalModelWrapper

    def __init__(self, args, model_config, data_config, encoder: EvalModelWrapper):
        super().__init__(args, model_config, data_config, encoder)
        assert MMSEGM_AVAIL, "MMSEG needs to be installed"
        self.embed_dim = model_config.embed_dim
        self.criterion = nn.CrossEntropyLoss()
        self._build_default_segm_modules()

    def _build_default_segm_modules(self):
        edim = self.embed_dim
        data_config = self.data_config

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

    def freeze_and_return_params(self):
        if self.training_mode == 'full_finetune':
            self.unfreeze(self.encoder)
        elif self.training_mode == 'frozen_backbone':
            self.freeze(self.encoder)
        else:
            raise ValueError(f"Invalid mode: {self.training_mode}")

        self.unfreeze(self.neck)
        self.unfreeze(self.decoder)
        self.unfreeze(self.aux_head)

        params_to_optimize = (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters()))

        if self.replace_pe:
            self.unfreeze(self.new_pe)
            params_to_optimize += list(self.new_pe.parameters())

        return params_to_optimize

    def forward(self, images):
        """Forward pass of the model."""
        if self.training_mode == 'frozen_backbone':
            with torch.no_grad():
                feats = self.encoder.get_segm_blks(images)
        else:
            feats = self.encoder.get_segm_blks(images)

        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=images.shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=images.shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def _step(self, batch, batch_idx, prefix="train"):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs[0], targets, loss, prefix)
        return loss
    
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels.long()) + 0.4 * self.criterion(
            outputs[1], labels.long()
        )