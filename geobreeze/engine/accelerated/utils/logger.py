# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# adjusted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/

from collections import defaultdict, deque
import datetime
import json
import logging
import time

import torch
import numpy as np
from . import distributed
import wandb 
import os
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from fvcore.common.checkpoint import Checkpointer

logger = logging.getLogger("eval")


def setup_logger(name, filename=None, level=logging.DEBUG, to_sysout=False, simple_prefix=False, reset_logger=True, hard_close=False):

    logger = logging.getLogger(name)
    if reset_logger:
        if hard_close:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        else:
            logger.handlers = [] # removeHandler will close handler but we later need it again!
    logger.setLevel(level)
    logger.propagate = False

    if simple_prefix:
        fmt_prefix = "%(asctime)s %(filename)s:%(lineno)s] "
        datefmt = "%H:%M:%S"
    else:
        fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
        datefmt = "%Y%m%d %H:%M:%S"
        
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if filename:
        handler = logging.StreamHandler(open(filename, "a+"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if to_sysout:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def update_linear_probe_metrics(output_dir, cls_metric_dict, prefix='val', iteration=None):
    val_file = os.path.join(output_dir, 'linear_probe_all_metrics.json')
    with open(val_file, 'a+') as f:
        for classifier_sting, metric in cls_metric_dict.items():
            print_dict = dict(
                iteration = iteration,
                classifier = classifier_sting,
                prefix = prefix)
            print_dict.update({k: round(v.item(), 4) for k, v in metric.items()})
            f.write(json.dumps(print_dict) + '\n')

class MetricLogger(object):
    def __init__(self, delimiter="\t", output_dir=None, output_file=None, use_wandb=False):
        self.meters = defaultdict(SmoothedValue)
        self.expert_meters = {}  # Separate dict for expert metrics
        self.delimiter = delimiter
        self.output_file = os.path.join(output_dir, output_file) if output_dir else output_file
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.epoch_len = None
        self.nsamples_per_iter = None
        self.dataset_len = None
        # Track which metrics should be logged as bar charts
        self.expert_metrics = set()

        if use_wandb:
            assert wandb.run is not None, 'wandb.run needs to be initialized before MetricLogger'
            self.run = wandb.run
        self.iter_time = SmoothedValue(fmt="{avg:.6f}")
        self.data_time = SmoothedValue(fmt="{avg:.6f}")
        self.epoch_time = SmoothedValue(fmt="{avg:.6f}")
        self.print_on_epoch_end = False

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_expert_metrics(self, name, values):
        """Update expert-specific metrics for bar chart visualization.
        Only stores the most recent values - no smoothing applied.
        These values will persist until the next update.
        """
        for i, v in enumerate(values):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                v = float(v)
            self.expert_meters[f"{name}/expert_{i}"] = v

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump['epoch'] = iteration // self.epoch_len
        # Regular metrics use median
        dict_to_dump.update({k: float(v.median) for k, v in self.meters.items()})
        # Expert metrics - ensure all values are Python floats
        dict_to_dump.update({k: float(v) for k, v in self.expert_meters.items()})
    
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

        if self.use_wandb:
            # improved ordering for wandb online GUI
            prefix_map = {
                'params': ['lr','wd','mom','teacher_temp', 'repa_alpha'],
                'loss': ['total_loss','dino_local_crops_loss','dino_global_crops_loss','koleo_loss','ibot_loss','aux_loss','pe_distil_loss'],
                'expert_activations': [k for k in self.expert_meters.keys()]
            }
            for pat, v in prefix_map.items():
                to_log = {}
                for k in v:
                    if k in dict_to_dump:
                        to_log[k] = dict_to_dump.pop(k)
                self.log_wandb(iteration, to_log, prefix=pat)
            if len(dict_to_dump) > 0:
                self.log_wandb(iteration, dict_to_dump)

    def log_wandb(self, iteration, metric_dict, prefix=None, log_step=True):
        if not self.use_wandb:
            return
        if prefix:
            metric_dict = {os.path.join(prefix,str(k)): v for k, v in metric_dict.items()}

        # define progress values
        metric_dict['iteration'] = iteration
        metric_dict['epoch'] = iteration // self.epoch_len
        if self.nsamples_per_iter is not None:
            metric_dict['nsamples'] = iteration * self.nsamples_per_iter
            if self.dataset_len is not None:
                metric_dict['epoch_act'] = (iteration * self.nsamples_per_iter) // self.dataset_len

        if log_step:
            self.run.log(metric_dict, step=iteration)
        else:
            self.run.log(metric_dict)
    
    def update_linear_probe_losses(self, loss_dict):
        iteration = self.iteration
        loss_file = os.path.join(self.output_dir, 'linear_probe_all_losses.csv')
        loss_dict = {k[5:]: v for k, v in loss_dict.items()}
        classifiers = sorted(list(loss_dict.keys()))
        if not os.path.exists(loss_file):
            with open(loss_file, 'w') as f:
                f.write(','.join(['iteration'] + classifiers) + '\n')
        with open(loss_file, 'a') as f:
            f.write(','.join([str(iteration)] + [str(round(loss_dict[k].item(),4)) for k in classifiers]) + '\n')

    def update_linear_probe_metrics(self, cls_metric_dict):
        iteration = self.iteration
        update_linear_probe_metrics(self.output_dir, cls_metric_dict, iteration=iteration)

    def log_every(self, 
                  iterable, 
                  print_freq, 
                  header='', 
                  n_iterations=None, 
                  start_iteration=0, 
                  use_self_timemeters=False, 

                  # kwargs for logging progress from different viewpoints
                  nsamples_per_iter=None,
                  dataset_len=None,
                  epoch_len=None):

        start_time = time.time()
        end = time.time()
        epoch_start = time.time()
        iter_time = self.iter_time if use_self_timemeters else SmoothedValue(fmt="{avg:.6f}")
        data_time = self.data_time if use_self_timemeters else SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)
        epoch_len = epoch_len or n_iterations
        self.epoch_len = epoch_len
        self.nsamples_per_iter = nsamples_per_iter
        self.dataset_len = dataset_len
        n_epochs = math.ceil(n_iterations / epoch_len)
        self.print_freq = print_freq

        i = start_iteration
        self.iteration = i
        epoch = int(i // epoch_len)

        iter_space_fmt = ":" + str(len(str(n_iterations))) + "d"
        epoch_space_fmt = ":" + str(len(str(n_epochs))) + "d"

        log_list = [
            header,
            "[iter: {0" + iter_space_fmt + "}/{1}, ",
            "epoch: {2" + epoch_space_fmt + "}/{3}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.iteration = i
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            epoch,
                            n_epochs,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            if i // epoch_len > epoch: # this allows float epoch_len
                epoch_time_current = time.time() - epoch_start
                if self.print_on_epoch_end:
                    logger.info(f"Epoch {epoch}/{n_epochs} done in {epoch_time_current:.2f}s\n")
                epoch_start = time.time()
                epoch += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {} ({:.6f} s / it)\n".format(header, total_time_str, total_time / n_iterations))


class BestValCheckpointer:
    def __init__(self, checkpointer: Checkpointer, metric_str: str, higher_is_better: bool):
        self.checkpointer = checkpointer
        self.metric_str = metric_str
        self.higher_is_better = higher_is_better
        self.best_classifier_str = ''
        self.filename = 'best_model_by_val'

        if self.higher_is_better:
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

        ckpt_path = self._get_ckpt_path()
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self._assert_correct_ckpt(ckpt)
            self.best_metric = ckpt['val']
            iteration = ckpt['iteration']
            logger.info(f'Loaded best value {self.best_metric} by {metric_str} from iteration {iteration}')

    def update(self, results_list, iteration):
        """ built to work with output of linear.py/evaluate_lienar_classifiers """
        res_dict = [r for r in results_list if r['metric_str'] == self.metric_str]
        assert len(res_dict) == 1, f'Found {len(res_dict)} values for {self.metric_str}'
        best_classifier_str = res_dict[0]['best_classifier']
        val = res_dict[0]['val']

        if self.higher_is_better:
            is_better = val > self.best_metric
        else:
            is_better = val < self.best_metric

        if is_better:
            logger.info(f"New best {self.metric_str}: {val:.4f} (old best: {self.best_metric:.4f})")
            self.best_metric = val
            self.checkpointer.save(self.filename, 
                iteration=iteration, 
                best_classifier_str=best_classifier_str, 
                metric_str=self.metric_str, 
                higher_is_better=self.higher_is_better,
                val=val)

    def _get_ckpt_path(self):
        ckpt_path = os.path.join(self.checkpointer.save_dir, f'{self.filename}.pth')
        if not os.path.exists(ckpt_path):
            logger.info(f'Checkpoint {ckpt_path} does not exist')
            return None
        return ckpt_path

    def _assert_correct_ckpt(self, ckpt):
        assert ckpt['higher_is_better'] == self.higher_is_better, 'Loaded model has different higher_is_better setting'
        assert ckpt['metric_str'] == self.metric_str, 'Loaded model has different metric_str'

    def load_best(self):
        """ load best ckpt into the model of self.checkpointer and return the dict """
        ckpt_path = self._get_ckpt_path()
        if ckpt_path is None:
            return None
        ckpt = self.checkpointer.load(ckpt_path) # also loads model weights in self.checkpointer.model
        self._assert_correct_ckpt(ckpt)

        best_classifier_str = ckpt['best_classifier_str']
        metric_str = ckpt['metric_str']
        iteration = ckpt['iteration']
        val = ckpt['val']

        logger.info(f'Loaded best model with {val:.4f} by {metric_str} from iteration {iteration}')
        return best_classifier_str

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def plot_curves(output_dir, suppress_print=False):
    """ for linear_probe only"""
    if not suppress_print:
        print(f'Plotting curves for {output_dir}')

    # extract train values

    train_metrics = []
    train_metrics_path = os.path.join(output_dir, 'training_metrics.json')

    if os.path.exists(train_metrics_path): # extract fom json file
        with open(train_metrics_path, 'r') as f:
            for line in f.readlines():
                train_metrics.append(json.loads(line))

    else: # extract from log
        log_file = os.path.join(output_dir, 'log') 
        loss_pattern = 'loss:.*\('
        iter_pattern = '\[iter:\s*\d*/'
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for l in lines:

                if not 'Training' in l:
                    continue

                match = re.search(iter_pattern, l)
                if match:
                    iteration = int(match.group().split(':')[-1][:-1].strip())

                    match = re.search(loss_pattern, l).group()
                    loss = float(match.split('(')[0].split(':')[-1])
                    
                    train_metrics.append({'iteration': iteration, 'loss': loss})

    # extract validation values

    eval_metrics = []
    test_metrics = []
    eval_metrics_path = os.path.join(output_dir, 'results_eval_linear.json')
    with open(eval_metrics_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == '':
                i += 1
                continue
            if lines[i].startswith('iter:'):
                iteration = int(lines[i].split(':')[-1].strip())
                i += 1

                while i < len(lines) and lines[i].strip() != '':
                    metrics = json.loads(lines[i])
                    metrics['iteration'] = iteration
                    if 'TEST' in metrics.get('prefix', ''):
                        test_metrics.append(metrics)
                    else:
                        eval_metrics.append(metrics)
                    i += 1
                continue
    
    dftrain = pd.DataFrame(train_metrics)
    dfeval = pd.DataFrame(eval_metrics)

    fig, ax = plt.subplots(2, 1, figsize=(7,5), sharex=True)
    task_name = os.path.basename(output_dir)
    ax[0].set_title(task_name)


    # plot train
    ax[0].plot(dftrain['iteration'], dftrain['loss'], label='train loss')
    ax[0].set_ylabel('Train Loss')

    # plot eval
    handles = []
    for metric_str in dfeval['metric_str'].unique():
        dfplot = dfeval[dfeval['metric_str'] == metric_str]
        hdl = ax[1].plot(dfplot['iteration'], dfplot['val'], label=f'{metric_str}')
        handles.append(hdl)

    # plot test
    for metric_dict in test_metrics:
        hdl = ax[1].plot(metric_dict['iteration'], 
                   metric_dict['val'], 
                   label = f'{metric_dict["metric_str"]} {metric_dict.get("prefix","")}',
                   marker = '*' )
        handles.append(hdl)
        
    # ax[1].set_title('Validation Metrics')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Validation Value')
    ax[1].legend()

    plt.savefig(os.path.join(output_dir, 'plots.png'))