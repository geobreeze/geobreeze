"""Base Class."""

from lightning import LightningModule
import torch
from geobreeze.util.misc import resize
import torch.nn as nn
from .model import EvalModelWrapper
from .base import BatchNormLinearHead
import logging

from geobreeze.engine.accelerated.utils.metrics import build_metric
from geobreeze.factory import make_optimizer, make_criterion

try:
    from mmseg.models.necks import Feature2Pyramid
    from mmseg.models.decode_heads import UPerHead, FCNHead
    MMSEGM_AVAIL = True
except Exception as e:
    print(f"Error importing MMSEG modules: {e}")
    print("Could not import MMSEG, skipping imports")
    MMSEGM_AVAIL = False


logger = logging.getLogger('eval')


class LightningTask(LightningModule):
    
    encoder: EvalModelWrapper
    
    def __init__(self, cfg, encoder: EvalModelWrapper, num_classes, num_channels=-1):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.num_classes = num_classes
        self.training_mode = self.cfg.optim.mode
        self.replace_pe = self.cfg.model.get('replace_pe', False)
        self.criterion = make_criterion(self.cfg.data.task.criterion)
        self.save_hyperparameters(ignore=['encoder'])
        
        self.train_metrics = build_metric(
            cfg.data.task.metrics.train, num_classes=self.num_classes, key_prefix='train/') 
        self.val_metrics = build_metric(
            cfg.data.task.metrics.val, num_classes=self.num_classes, key_prefix='val/') 
        self.test_metrics = build_metric(
            cfg.data.task.metrics.val, num_classes=self.num_classes, key_prefix='test/') 

        if self.replace_pe:
            assert num_channels > 0, "num_channels must be specified if replace_pe is True"
            self.new_pe = self.encoder.replace_pe(num_channels)
            logger.info(f'Replaced PE with {num_channels} channels.')

    def freeze_and_return_params(self):
        """ freeze & unfreeze weights according to self.training_mode, also
            returns all parameters to optimize """
        raise NotImplementedError('Subclass must implement this method')

    def forward(self, x):
        raise NotImplementedError('Subclass must implement this method')

    def loss(self, outputs, labels):
        raise NotImplementedError('Subclass must implement this method')

    def log_metrics(self, outputs, targets, loss, prefix="train", apply_argmax=False):
        metrics = self.__getattr__(f"{prefix}_metrics")
        # print(f'Logging metrics: outputs={tuple(outputs.shape)}, targets={tuple(targets.shape)}')

        if apply_argmax:
            metrics(
                torch.argmax(outputs, axis=1),
                targets.long()
            )
        else:  
            metrics(outputs, targets)

        on_step = True if prefix == 'train' else False
        on_epoch = True

        self.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        # self.log_dict(out_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=True)
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch, prog_bar=True)


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
        optimizer = make_optimizer(self.cfg.optim.optim, lr=self.cfg.optim.lr,
                        params=self.freeze_and_return_params())

        world_size = self.cfg.num_gpus if self.cfg.num_gpus >= 1 else 1
        num_warmup_steps = (
            self.cfg.optim.warmup_epochs * self.train_dl_len // world_size)
        total_steps = (
            self.cfg.optim.epochs * self.train_dl_len // world_size)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.cfg.optim.lr,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_classifier = BatchNormLinearHead(
            in_features = self.cfg.model.embed_dim, 
            num_classes = self.num_classes)

    def freeze_and_return_params(self):
        """ freeze / unfreeze weights & return parameters to optimize 
            according to self.training_mode"""
        mode = self.training_mode

        # prepare encoder 
        if mode == 'frozen_backbone':
            self.freeze(self.encoder)
            self.encoder.eval()

        elif mode == 'partial_finetune':
            self.freeze(self.encoder)
            encoder_params_to_unfreeze = self._filter_named_params(
                self.encoder.named_parameters(), self.cfg.model.params_to_train)
            self.unfreeze(encoder_params_to_unfreeze)

            params_to_optimize = [p for _, p in encoder_params_to_unfreeze]

        elif mode == 'finetune':
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
        x_dict, targets = batch
        
        if self.training_mode == 'frozen_backbone':
            with torch.no_grad():
                x = self.encoder.get_blocks(x_dict)
        else:
            x = self.encoder.get_blocks(x_dict)
        x = self.encoder.default_blocks_to_featurevec(x)
        outputs = self.linear_classifier(x)

        loss = self.loss(outputs, targets)
        self.log_metrics(outputs, targets, loss, prefix)
        return loss

    def loss(self, outputs, labels):
        return self.criterion(outputs, labels)



class LightningSegmentationTask(LightningTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert MMSEGM_AVAIL, "MMSEG needs to be installed"
        self.embed_dim = self.cfg.model.embed_dim
        self._build_default_segm_modules()

    def _build_default_segm_modules(self):
        edim = self.embed_dim

        self.neck = Feature2Pyramid(embed_dim=edim, rescales=[4, 2, 1, 0.5])
        self.decoder = UPerHead(
            in_channels=[edim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=self.num_classes,
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
            num_classes=self.num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )

    def freeze_and_return_params(self):
        if self.training_mode == 'finetune':
            self.unfreeze(self.encoder)
            params_to_optimize = list(self.encoder.parameters())
        elif self.training_mode == 'segm_frozen_backbone':
            self.freeze(self.encoder)
            self.encoder.eval()
            params_to_optimize = []
        else:
            raise ValueError(f"Invalid mode: {self.training_mode}")

        self.unfreeze(self.neck)
        self.unfreeze(self.decoder)
        self.unfreeze(self.aux_head)

        params_to_optimize += (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters()))

        if self.replace_pe:
            self.unfreeze(self.new_pe)
            params_to_optimize += list(self.new_pe.parameters())

        return params_to_optimize

    def forward(self, x_dict):
        """Forward pass of the model."""
        if self.training_mode == 'segm_frozen_backbone':
            with torch.no_grad():
                feats = self.encoder.get_segm_blks(x_dict)
        else:
            feats = self.encoder.get_segm_blks(x_dict)

        feats = self.neck(feats)
        out = self.decoder(feats)
        out = resize(out, size=x_dict['imgs'].shape[2:], mode="bilinear", align_corners=False)
        out_a = self.aux_head(feats)
        out_a = resize(
            out_a, size=x_dict['imgs'].shape[2:], mode="bilinear", align_corners=False
        )
        return out, out_a

    def _step(self, batch, batch_idx, prefix="train"):
        images, targets = batch
        outputs = self(images)
        loss = self.loss(outputs, targets)
        self.log_metrics(outputs[0], targets, loss, prefix, apply_argmax=True)
        return loss
    
    def loss(self, outputs, labels):
        return self.criterion(outputs[0], labels.long()) + 0.4 * self.criterion(
            outputs[1], labels.long()
        )