import datetime
import os
from pathlib import Path
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
import torch
from lightning.pytorch import seed_everything
from omegaconf import open_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from geobreeze.engine.accelerated.utils.logger import setup_logger, plot_curves

from geobreeze.engine.accelerated.linear import run_eval_linear
from geobreeze.engine.accelerated.knn import eval_knn_with_model
from geobreeze.engine.lightning_task import LightningClsRegTask, LightningSegmentationTask
from geobreeze.engine.model import EvalModelWrapper

import logging
import json
from copy import deepcopy
from factory import make_dataset, make_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

logger = logging.getLogger('eval')


def process_config(cfg):

    # training mode preparation
    training_mode = cfg.optim.mode
    if training_mode == 'knn':
        assert cfg.num_gpus == 1, 'accelerated only supports single gpu for now'
        blks = 'default_cls'

    elif training_mode == 'linear_probe':
        assert cfg.num_gpus == 1, 'accelerated only supports single gpu for now'
        blks = 'linear_probe'

    elif training_mode == 'finetune':

        with open_dict(cfg):
            cfg.optim.lr = cfg.optim.base_lr * cfg.dl.batch_size / 256 * cfg.num_gpus
            # logger.info(f'Scaled learning rate from {cfg.optim.base_lr} to {cfg.optim.lr} (bsz={cfg.dl.batch_size}, num_gpus={cfg.num_gpus})')

        blks = 'segm' if cfg.data.task.id == 'segmentation' else 'default_cls'

    else:
        raise ValueError(f'Unknown training_mode: {training_mode}')
    
    # set blocks
    all_blks = set(cfg.model.blk_indices.keys())
    with open_dict(cfg):
        assert blks in all_blks
        cfg.model.blk_indices = cfg.model.blk_indices[blks]
        all_blks.remove(blks)
        for blk in all_blks:
            cfg.model.pop(blk, None)

    # setup output dir
    experiment_name = os.path.relpath(cfg.output_dir, os.environ['ODIR'])
    if cfg.add_defining_args:
            
        args_defining_run = cfg.optim.args_defining_run
        run_name = "_".join(
            [f"{v}={OmegaConf.select(cfg,k)}" for k, v in args_defining_run.items()])

        cfg.output_dir = os.path.join(
            cfg.output_dir, run_name
        )

    else:
        run_name = (
            f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        )

    with open_dict(cfg):
        cfg.experiment_name = experiment_name
        cfg.run_name = run_name

    # resolve config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.pop('_vars')
    
    return cfg

def setup(cfg):

    cfg = process_config(cfg)

    # check if task already executed
    if os.path.exists(os.path.join(cfg.output_dir, "results.csv")):
        if cfg.overwrite:
            print(f"Overwriting existing output dir: {cfg.output_dir}")
        else:
            print(f"Output dir already exists: {cfg.output_dir}. Skipping job.")
            return cfg, True

    # intialize logger
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    setup_logger('eval', to_sysout=True, filename=os.path.join(cfg.output_dir, 'log.txt'))

    # save config
    logger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    
    seed_everything(cfg.seed)

    return cfg, False


def get_num_classes(datasets):
    ds = datasets['train']
    if isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    return ds.num_classes

def do_knn(cfg, model: EvalModelWrapper, datasets: dict):
    # model.load_encoder(cfg.model.default_cls_blk_indices)

    results_list = eval_knn_with_model(
        model,
        cfg.output_dir,
        datasets['train'],
        datasets['val'],
        nb_knn = cfg.optim.nb_knn,
        normmode_list = cfg.optim.normmode_list,
        temperature_list = cfg.optim.temperature_list,
        autocast_dtype = torch.bfloat16,
        metric_cfg = cfg.data.task.metrics.val,
        dl_cfg = cfg.dl,
        num_classes = get_num_classes(datasets),)

    return pd.DataFrame(results_list)

def do_linear_probe(cfg, model: EvalModelWrapper, datasets: dict):

    experiment_name = cfg.experiment_name
    run_name = cfg.run_name

    # model.load_encoder(model.accel_cls_blk_indices)

    heads_cfg = OmegaConf.create(dict(
        n_last_blocks_list = cfg.optim.n_last_blocks_list,
        pooling = cfg.optim.pooling,
        learning_rates = cfg.optim.lr,
        use_additional_1dbatchnorm_list = cfg.optim.use_additional_1dbatchnorm_list,
    ))

    results_list = run_eval_linear(
        model,
        cfg.output_dir,
        datasets['train'],
        datasets['val'],
        [datasets['test']],
        get_num_classes(datasets),
        cfg.dl,
        heads_cfg,
        cfg.optim.epochs,
        eval_period_epoch = cfg.optim.check_val_every_n_epoch,
        criterion_cfg = cfg.data.task.criterion,
        val_metrics = cfg.data.task.metrics.val,
        optim_cfg=cfg.optim.optim,
        val_monitor = cfg.data.task.metrics.ckpt_monitor,
        val_monitor_higher_is_better = cfg.data.task.metrics.ckpt_monitor_higher_is_better,
        batchwise_spectral_subsampling = cfg.optim.batchwise_spectral_subsampling,
        resume = cfg.resume,
    )

    # process loss file
    loss_file = os.path.join(cfg.output_dir, 'linear_probe_all_losses.csv')
    losses = pd.read_csv(loss_file).reset_index(drop=True)
    classifiers = losses.columns[1:]

    # process metrics file
    metrics_file = os.path.join(cfg.output_dir, 'linear_probe_all_metrics.json')
    metrics_by_cls = {}
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        d = json.loads(l)
        cls = d['classifier']
        if cls not in metrics_by_cls:
            metrics_by_cls[cls] = {}
        if 'test' in d['prefix']:
            key = d['prefix']
        else:
            key = d['iteration']
        metrics_by_cls[cls][key] = {k:v for k,v in d.items() if k not in ['prefix','iteration','classifier']}

    if cfg.logger == 'mlflow':
        logger.info('Logging to mlflow')
        import mlflow
        mlflow.set_tracking_uri(f"file:{os.path.join(os.environ['ODIR'], '_mlruns')}")
        mlflow.set_experiment(os.path.join(experiment_name, run_name))
        
        for cls in classifiers:
            with mlflow.start_run(run_name=cls):
                # example: blocks_4_pooling_default_lr_2_50000
                params = dict(
                    blocks = cls.split('_')[1],
                    pooling = cls.split('_')[3],
                    lr = float('.'.join(cls.split('_')[-4:-2])) ,
                    use_1dbn = cls.split('_')[-1],)
                mlflow.log_params(params)

                for i in range(losses.shape[0]):
                    mlflow.log_metric(f'loss', losses.at[i,cls], step=losses.at[i, 'iteration'])

                if cls not in metrics_by_cls:
                    print(f'Skipping {cls} (probably crashed because of high lr)')
                    continue
                for i, metrics in metrics_by_cls[cls].items():
                    if isinstance(i, int):
                        for name, val in metrics.items():
                            mlflow.log_metric(f'val/{name}', val, step=int(i))
                    else:
                        for name, val in metrics.items():
                            mlflow.log_metric(f'{i}/{name}', val)
            
    else:
        raise NotImplementedError()
    
    plot_curves(cfg.output_dir) # plot average curve into .png file
    return pd.DataFrame(results_list)

def do_finetune(cfg, model: EvalModelWrapper, datasets: dict):

    task = cfg.data.task.id
    experiment_name = cfg.experiment_name
    run_name = cfg.run_name
    num_classes = get_num_classes(datasets)

    if task in ['classification','regression']:
        # model.load_encoder(model.default_cls_blk_indices)
        pl_task = LightningClsRegTask(cfg, model, num_classes)
    elif task == 'segmentation':
        # model.load_encoder(model.segm_blk_indices)
        pl_task = LightningSegmentationTask(cfg, model, num_classes)
    else:
        raise NotImplementedError()

    # Setup logger
    if cfg.logger == "mlflow":
        logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=f"file:{os.path.join(os.environ['ODIR'], '_mlruns')}",)
    else:
        raise NotImplementedError(f'Logger {cfg.logger} not implemented.')


    # Callbacks
    monitor = os.path.join('val',cfg.data.task.metrics.ckpt_monitor)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="best_model-{epoch}",
            monitor=monitor,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.optim.epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.optim.check_val_every_n_epoch,
        devices=cfg.num_gpus,
    )

    # initialize dataloader
    train_dl = torch.utils.data.DataLoader(
        datasets['train'],
        **cfg.dl,
        shuffle=True,
        drop_last=True,
    )
    val_dl = torch.utils.data.DataLoader(
        datasets['val'],
        **cfg.dl,
        shuffle=False,
        drop_last=False,
    )
    test_dl = torch.utils.data.DataLoader(
        datasets['test'],
        **cfg.dl,
        shuffle=False,
        drop_last=False,
    )

    # Train
    ckpt_path = os.path.join(cfg.output_dir, 'checkpoints','last.ckpt')
    trainer.fit(pl_task, train_dl, val_dl, ckpt_path=ckpt_path if cfg.resume else None)

    # Test
    best_checkpoint_path = callbacks[0].best_model_path
    results_per_ds_list = trainer.test(pl_task, test_dl, ckpt_path=best_checkpoint_path)
    results_list = [dict(
        metric_str=k,
        val=v,
        best_classifier=f'lr={cfg.lr}'
    ) for ds_dict in results_per_ds_list for k,v in ds_dict.items()]
    results = pd.DataFrame(results_list)

    return results


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    cfg, skip = setup(cfg)
    if skip:
        return

    datasets = {split: make_dataset(cfg.data[split], seed=cfg.seed) for split in ['train','val','test']}
    model = make_model(cfg.model)

    # execute training with correct engine
    training_mode = cfg.optim.mode
    if training_mode == 'knn':
        logger.info('Running KNN')
        results = do_knn(cfg, model, datasets)

    elif training_mode == 'linear_probe':
        logger.info('Running linear probe')
        results = do_linear_probe(cfg, model, datasets)
    
    elif training_mode == 'finetune':
        logger.info('Running finetune')
        results = do_finetune(cfg, model, datasets)

    else:
        raise ValueError(f'Unknown training_mode: {training_mode}')

    # save results
    results = results[['metric_str','val','best_classifier']]
    results.rename(columns={'metric_str':'metric'}, inplace=True)
    results.reset_index(drop=True, inplace=True)
    logger.info(f'Results: \n\n{results.to_string()}\n')
    results.to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()
