# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# adjusted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py


import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer


from .utils.logger import MetricLogger, update_linear_probe_metrics, BestValCheckpointer
from .utils.utils import evaluate, blocks_to_cls, remove_ddp_wrapper
from .utils.data import make_data_loader, SamplerType
from .utils.metrics import build_metric
from .utils import distributed
from geobreeze.engine.model import EvalModelWrapper
from geobreeze.factory import make_criterion, make_optimizer

import time
import math

import json

from functools import partial

logger = logging.getLogger("eval")


####### setup

class FeatureModel(nn.Module):
    def __init__(self, model: EvalModelWrapper, autocast_dtype=torch.float32):
        super().__init__()
        self.model = model
        self.autocast_dtype = autocast_dtype

        self.norm = self.model.norm
        self.default_blocks_to_featurevec = self.model.default_blocks_to_featurevec

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.cuda()

    def forward(self, x):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(
                    enabled=(self.autocast_dtype is not None), 
                    dtype=self.autocast_dtype):
                return self.model.get_blocks(x)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, 
            sample_output, 
            use_n_blocks, 
            pooling, 
            num_classes = -1, 
            norm: nn.Module = nn.Identity(),
            default_blocks_to_featurevec = None,
            use_additional_1dbatchnorm = False,
        ):
        super().__init__()

        self.blocks_to_cls = partial(
            blocks_to_cls,
            use_n_blocks = use_n_blocks,
            pooling = pooling,
            default_blocks_to_featurevec = default_blocks_to_featurevec,
            norm = norm)

        out_dim = self.blocks_to_cls(sample_output).shape[-1]
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        if use_additional_1dbatchnorm:
            self.linear = nn.Sequential(nn.BatchNorm1d(out_dim), self.linear)

    def forward(self, x_tokens_list):
        output = self.blocks_to_cls(x_tokens_list)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs, **kwargs):
        return {k: v.forward(inputs, **kwargs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier, class_mapping=None):
        super().__init__()
        self.linear_classifier = linear_classifier
        self.register_buffer("class_mapping", None if class_mapping is None else torch.LongTensor(class_mapping))

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds[:, self.class_mapping] if self.class_mapping is not None else preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(
        sample_output, 
        pooling, 
        n_last_blocks_list,
        learning_rates, 
        batch_size,
        num_classes = -1, 
        norm = None,
        default_blocks_to_featurevec = None,    
        use_additional_1dbatchnorm_list = [False, True],
    ):
    
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for pool in pooling:
            for _lr in learning_rates:
                for use_1dbn in use_additional_1dbatchnorm_list:
                    lr = scale_lr(_lr, batch_size)
                    linear_classifier = LinearClassifier(
                        sample_output, 
                        use_n_blocks=n, 
                        pooling=pool, 
                        num_classes=num_classes, 
                        default_blocks_to_featurevec=default_blocks_to_featurevec,
                        norm=norm,
                        use_additional_1dbatchnorm=use_1dbn,)
                    linear_classifier = linear_classifier.cuda()
                    linear_classifiers_dict[
                        f"blocks_{n}_pooling_{pool}_lr_{lr:.5f}_1dbn_{use_1dbn}".replace(".", "_")
                    ] = linear_classifier
                    optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


####### execution

@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metrics,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    class_mapping=None,
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metrics = build_metric(metrics, num_classes=num_classes)
    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics_for_each_classifier = {k: metrics.clone() for k in linear_classifiers.classifiers_dict}

    _, all_metrics_results_dict = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics_for_each_classifier,
        torch.cuda.current_device(),
    )

    # print metrics
    logger.info("")
    for classifier_string, metric in all_metrics_results_dict.items():
        metric = {k: round(v.item()*100, 2) for k, v in metric.items()}
        print_metrics_str = ", ".join([f"{k}: {v}" for k, v in metric.items()])
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {print_metrics_str}")

    # find best classifiers by metric
    results_list = []
    for target_metric in metrics.keys():
        higher_is_better = metrics[target_metric].higher_is_better
        if higher_is_better:
            best_val = 0
            is_better = lambda newval, bestval: newval > bestval
        else:
            best_val = float("inf")
            is_better = lambda newval, bestval: newval < bestval

        best_classifier = ""
        for classifier_string, metric in all_metrics_results_dict.items():
            
            val = metric[target_metric].item()
            if (
                best_classifier_on_val is None and is_better(val, best_val)
            ) or classifier_string == best_classifier_on_val:
                best_val = val
                best_classifier = classifier_string
        results_list.append(dict(
            best_classifier = best_classifier,
            val = best_val,
            metric_str = target_metric,))
        logger.info(f"Best classifier by {target_metric} (higher_is_better={higher_is_better}) with {best_val}: {best_classifier}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for result_dict in results_list:
                result_dict['prefix'] = prefixstring
                f.write(json.dumps(result_dict) + "\n")
                result_dict.pop('prefix')
            f.write("\n")

    return results_list, all_metrics_results_dict


def eval_linear(
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,

    max_iter,
    iter_per_epoch,
    eval_period_iter,
    checkpoint_period,  # In number of iter, creates a new file every period

    metrics,
    training_num_classes,

    val_monitor = None,
    val_monitor_higher_is_better = True,

    resume=True,
    classifier_fpath=None,
    val_class_mapping=None,
    criterion_cfg = {'id': 'CrossEntropyLoss'},
):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    checkpointer.logger = logger
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter, max_to_keep=1)
    best_val_checkpointer = BestValCheckpointer(checkpointer, val_monitor, val_monitor_higher_is_better)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ", output_dir=output_dir, output_file='training_metrics.json')
    header = "Training"
    criterion = make_criterion(criterion_cfg)
    all_metrics_results_dict = None

    for data, labels in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        iteration,
        epoch_len = iter_per_epoch,
    ):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda(non_blocking=True)
        else:
            data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)

        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item() / len(losses.values()))
            metric_logger.update(lr=optimizer.param_groups[0]["lr"]) # only to get idea of lr cycle
            metric_logger.update_linear_probe_losses(losses)

        if distributed.is_main_process():
            torch.cuda.synchronize()
            periodic_checkpointer.step(iteration)
            torch.cuda.synchronize()

        if (eval_period_iter > 0 and (iteration + 1) % int(eval_period_iter) == 0) or iteration == max_iter - 1:
            results_list, all_metrics_results_dict = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metrics=metrics,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,)
            
            metric_logger.update_linear_probe_metrics(all_metrics_results_dict)
            best_val_checkpointer.update(results_list, iteration)
            
            torch.cuda.synchronize()

        iteration = iteration + 1

    best_classifier_str = best_val_checkpointer.load_best() # also loads weights into linear_classifier

    if all_metrics_results_dict is not None: # eval might already be done
        update_linear_probe_metrics(output_dir, all_metrics_results_dict, prefix='val', iteration=iteration)
    
    return best_classifier_str, feature_model, linear_classifiers, iteration



def test_on_datasets(
    feature_model,
    linear_classifiers,
    test_dataset_lists,
    batch_size,
    num_workers,
    test_metrics_list,
    metrics_file_path,
    training_num_classes,
    iteration,
    best_classifier_on_val,
    test_class_mappings=[None],
    batchwise_spectral_subsampling = False,
):
    results_list = []
    all_metrics_out = {}
    for ds_id, (test_dataset, class_mapping, metrics) in \
            enumerate(zip(test_dataset_lists, test_class_mappings, test_metrics_list)):

        test_data_loader = make_data_loader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_type=SamplerType.EPOCH,
            batchwise_spectral_subsampling=batchwise_spectral_subsampling,)

        metrics_result_list, all_metrics_results_dict = evaluate_linear_classifiers(
            feature_model,
            remove_ddp_wrapper(linear_classifiers),
            test_data_loader,
            metrics,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring=f"TEST (ds_id={ds_id})",
            class_mapping=class_mapping,
            best_classifier_on_val=best_classifier_on_val,
        )

        for result_dict in metrics_result_list:
            results_list.append(result_dict)

        all_metrics_out[ds_id] = all_metrics_results_dict

    return results_list, all_metrics_out


def run_eval_linear(
    model,
    output_dir,
    train_dataset,
    val_dataset,
    test_dataset_lists,
    num_classes,
    dl_cfg,
    
    heads_cfg,

    epochs,
    iter_per_epoch=-1,
    eval_period_epoch=-1,
    eval_period_iter=-1,
    save_checkpoint_frequency_epoch=10,

    val_metrics=[{'id': 'MulticlassAccuracy'}],
    test_metrics_list=None,
    criterion_cfg = {'id': 'CrossEntropyLoss'},
    optim_cfg = {'id': 'SGD', 'momentum': 0.9, 'weight_decay': 0},

    val_monitor='_not_specified', # corresponds to str of above val_metric
    val_monitor_higher_is_better=True,

    resume=True,
    classifier_fpath=None,
    val_class_mapping_fpath=None,
    test_class_mapping_fpaths=[None],
    seed = 21,

    batchwise_spectral_subsampling = {'train': False, 'val': False, 'test': False},
):

    if any(batchwise_spectral_subsampling.values()):
        assert distributed.get_global_size() == 1, "Batchwise spectral subsampling only supported on single GPU"

    if test_metrics_list is None:
        test_metrics_list = [val_metrics] * len(test_dataset_lists)
    if test_class_mapping_fpaths == [None]:
        test_class_mapping_fpaths = [None] * len(test_dataset_lists)
    assert len(test_dataset_lists) == len(test_class_mapping_fpaths)

    training_num_classes = num_classes

    # set max_iter & iter_per_epoch
    if iter_per_epoch == -1:
        ds_len = len(train_dataset) 
        total_nsamples = ds_len * epochs
        eff_bsz = dl_cfg['batch_size'] * distributed.get_global_size()
        max_iter = math.ceil(total_nsamples / eff_bsz)
        iter_per_epoch = max_iter / epochs # float!
    else:
        max_iter = math.ceil(epochs * iter_per_epoch)

    # set eval_period_iter
    assert (eval_period_epoch == -1) != (eval_period_iter == -1), \
        "Exactly one of eval_period_epoch and eval_period_iter must be set"
    if eval_period_iter == -1:
        eval_period_iter = max(1.0, eval_period_epoch * iter_per_epoch)
    logger.info(f"max_iter: {max_iter}, iter_per_epoch: {iter_per_epoch}, eval_period_iter: {eval_period_iter}")
    checkpoint_period = math.ceil(save_checkpoint_frequency_epoch * iter_per_epoch)

    # classifiers

    feature_model = FeatureModel(model)
    x = next(iter(train_dataset))[0]
    x['imgs'] = x['imgs'].unsqueeze(0)
    x = {k: v.cuda() for k, v in x.items()}
    sample_output = feature_model(x)

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        heads_cfg.pooling,
        heads_cfg.n_last_blocks_list,
        heads_cfg.learning_rates,
        dl_cfg['batch_size'],
        training_num_classes,
        norm = feature_model.norm,
        default_blocks_to_featurevec = feature_model.default_blocks_to_featurevec,
        use_additional_1dbatchnorm_list = heads_cfg.use_additional_1dbatchnorm_list
    )
    
    optimizer = make_optimizer(optim_cfg, params=optim_param_groups)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    checkpointer.logger = logger
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=dl_cfg['batch_size'],
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg.get('pin_memory', True),
        persistent_workers=dl_cfg.get('persistent_workers', True),
        seed=seed,
        sampler_type=SamplerType.INFINITE,
        sampler_advance = start_iter,
        batchwise_spectral_subsampling = batchwise_spectral_subsampling['train'],
    )
    val_data_loader = make_data_loader(
        val_dataset, 
        batch_size = dl_cfg['batch_size'],
        num_workers = dl_cfg['num_workers'],
        pin_memory = dl_cfg.get('pin_memory', False),
        persistent_workers = dl_cfg.get('persistent_workers', True),
        seed = seed,
        sampler_type=SamplerType.EPOCH,
        batchwise_spectral_subsampling = batchwise_spectral_subsampling['val'],
    )


    if val_class_mapping_fpath is not None:
        logger.info(f"Using class mapping from {val_class_mapping_fpath}")
        val_class_mapping = np.load(val_class_mapping_fpath)
    else:
        val_class_mapping = None

    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    best_classifier_str, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model = feature_model,
        linear_classifiers = linear_classifiers,
        train_data_loader = train_data_loader,
        val_data_loader = val_data_loader,
        metrics_file_path = metrics_file_path,
        optimizer = optimizer,
        scheduler = scheduler,
        output_dir = output_dir,
        max_iter = max_iter,
        eval_period_iter = eval_period_iter,
        iter_per_epoch = iter_per_epoch,
        checkpoint_period = checkpoint_period,
        metrics = val_metrics,
        training_num_classes = training_num_classes,
        resume = resume,
        val_class_mapping = val_class_mapping,
        classifier_fpath = classifier_fpath,
        criterion_cfg = criterion_cfg,
        val_monitor=val_monitor,
        val_monitor_higher_is_better=val_monitor_higher_is_better,
    )
    results_list = []
    if len(test_dataset_lists) > 0:
        results_list, test_all_metrics_out = test_on_datasets(
            feature_model,
            linear_classifiers,
            test_dataset_lists,
            dl_cfg['batch_size'],
            dl_cfg['num_workers'], 
            test_metrics_list,
            metrics_file_path,
            training_num_classes,
            iteration,
            best_classifier_str,
            test_class_mappings=test_class_mappings,
            batchwise_spectral_subsampling = batchwise_spectral_subsampling['test'],
            )
        
        for ds_id, cls_metric_dict in test_all_metrics_out.items():
            update_linear_probe_metrics(output_dir, cls_metric_dict, prefix=f'test_{ds_id}')

    return results_list


