import datetime
import os
from pathlib import Path
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
import torch
from datasets.data_module import BenchmarkDataModule
from lightning.pytorch import seed_everything
from factory import create_model
from omegaconf import open_dict
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from geofm_src.engine.accelerated.utils.logger import setup_logger, plot_curves

from geofm_src.engine.accelerated.linear import run_eval_linear
from geofm_src.engine.accelerated.knn import eval_knn_with_model
from geofm_src.engine.lightning_task import LightningClsRegTask, LightningSegmentationTask
import logging
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    task = cfg.dataset.task
    training_mode = cfg.model.training_mode
    os.environ['CDIR'] = os.path.join(os.environ['REPO_PATH'], 'geofm_src/configs/')
    default_config_dir = os.path.join(os.environ['REPO_PATH'], 'geofm_src/configs/task_defaults/')

    # assign engine
    if training_mode in ['linear_probe','knn']:
        engine = 'accelerated'
    else:
        engine = 'lightning'

    # engine specific input handling
    if engine == 'accelerated':
        if training_mode == 'linear_probe':
            f = 'linear_probe_accel.yaml'
        else:
            f = 'knn_accel.yaml'
        defaults = OmegaConf.load(os.path.join(default_config_dir, f))
        cfg = OmegaConf.merge(defaults, cfg)

        if training_mode == 'linear_probe':
            assert OmegaConf.is_list(cfg.lr), 'lr should be a list for accelerated engine'
            args_defining_run = {
                "batch_size": "bsz",
                "epochs": "e",
                'optim.display_name': 'optim',
            }
        else:
            args_defining_run = {
                'nb_knn': 'k',
                'temperature_list': 'T',
                'normmode_list': 'norm',
            }
        assert cfg.num_gpus == 1, 'accelerated only supports single gpu for now'


    elif engine == 'lightning':
        defaults = OmegaConf.load(os.path.join(default_config_dir, 'lightning.yaml'))
        cfg = OmegaConf.merge(defaults, cfg)

        assert all([k not in cfg for k in ['pooling','n_last_blocks_list']]), 'only for accelerated linear_prob engine'

        args_defining_run = {
            "lr": "lr",
            "batch_size": "bsz",
            "epochs": "e",
        }

        # Scale learning rate
        # assert (cfg.lr == -1) != (cfg.base_lr == -1), "either lr or base_lr should be set"
        assert not 'base_lr' in cfg, 'base_lr is legacy, only provide lr'
        with open_dict(cfg):
            cfg.inputted_lr = cfg.lr
        cfg.lr = cfg.lr * cfg.batch_size / 256 * cfg.num_gpus

    # get metrics
    task_kwargs = OmegaConf.load(os.path.join(default_config_dir, 'metrics_and_criterion.yaml'))
    key = task
    if cfg.dataset.multilabel:
        key = f'multilabel_{key}'
    with open_dict(cfg):
        cfg.task_kwargs = task_kwargs[key]


    # setup output dir
    experiment_name = os.path.relpath(cfg.output_dir, os.environ['ODIR'])
    if cfg.add_defining_args:
            
        run_name = "_".join(
            [f"{v}={OmegaConf.select(cfg,k)}" for k, v in args_defining_run.items()])

        cfg.output_dir = os.path.join(
            cfg.output_dir, run_name
        )

    else:
        run_name = (
            f"{experiment_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # check if task already executed
    if os.path.exists(os.path.join(cfg.output_dir, "results.csv")):
        if cfg.overwrite:
            print(f"Overwriting existing output dir: {cfg.output_dir}")
        else:
            print(f"Output dir already exists: {cfg.output_dir}. Skipping job.")
            return

    seed_everything(cfg.seed)

    # print & save config
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    # create model
    model_wrapper = create_model(cfg.model, cfg.dataset)

    # create datamodule
    cfg.dataset.image_resolution = cfg.model.image_resolution
    data_module = BenchmarkDataModule(
        dataset_config=cfg.dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        seed=cfg.seed,
    )


    # execute training with correct engine
    if engine == 'lightning':

        if task in ['classification','regression']:
            model_wrapper.load_encoder(cfg.model.default_cls_blk_indices)
            pl_task = LightningClsRegTask(cfg, cfg.model, cfg.dataset, model_wrapper)
        elif task == 'segmentation':
            model_wrapper.load_encoder(cfg.model.segm_blk_indices)
            pl_task = LightningSegmentationTask(cfg, cfg.model, cfg.dataset, model_wrapper)
        else:
            raise NotImplementedError()

        # Setup logger
        if cfg.logger == "mlflow":
            logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=f"file:{os.path.join(os.environ['ODIR'], '_mlruns')}",)
        elif cfg.logger == 'wandb':
            raise NotImplementedError()


        # Callbacks
        monitor = os.path.join('val',cfg.task_kwargs.ckpt_monitor)
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

        if cfg.num_gpus == 0: # cpu
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='cpu',
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)

        elif cfg.num_gpus == 1: # single gpu
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='gpu',
                devices=cfg.num_gpus,
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)

        else: # ddp on multiple gpus
            trainer = Trainer(
                logger=logger,
                callbacks=callbacks,
                accelerator='gpu',
                strategy=DDPStrategy(find_unused_parameters=False),
                devices=cfg.num_gpus,
                max_epochs=cfg.epochs,
                num_sanity_val_steps=0,
                **cfg.trainer)


        # Train
        ckpt_path = os.path.join(cfg.output_dir, 'checkpoints','last.ckpt')
        trainer.fit(pl_task, data_module, ckpt_path=ckpt_path if cfg.resume else None)

        if cfg.trainer.get('fast_dev_run', False):
            print('No eval for fastdevrun.')
            return

        # Test
        best_checkpoint_path = callbacks[0].best_model_path
        results_per_ds_list = trainer.test(pl_task, data_module, ckpt_path=best_checkpoint_path)
        results_list = [dict(
            metric_str=k,
            val=v,
            best_classifier=f'lr={cfg.lr}'
        ) for ds_dict in results_per_ds_list for k,v in ds_dict.items()]
        results = pd.DataFrame(results_list)


    elif engine == 'accelerated':
        
        print("CONFIG MODEL")
        print(cfg.model)
        print(cfg.model.keys())

        data_module.setup()
        setup_logger('eval', to_sysout=True, filename=os.path.join(cfg.output_dir, 'log.txt'))
        logger = logging.getLogger("eval")

        dl_cfg = OmegaConf.create(dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ))

        if training_mode == 'linear_probe':
            model_wrapper.load_encoder(cfg.model.accel_cls_blk_indices)

            heads_cfg = OmegaConf.create(dict(
                n_last_blocks_list = cfg.n_last_blocks_list,
                pooling = cfg.pooling,
                learning_rates = cfg.lr,
                use_additional_1dbatchnorm_list = cfg.use_additional_1dbatchnorm_list,
            ))

            results_list = run_eval_linear(
                model_wrapper,
                cfg.output_dir,
                data_module.dataset_train,
                data_module.dataset_val,
                [data_module.dataset_test],
                cfg.dataset.num_classes,
                dl_cfg,
                heads_cfg,
                cfg.epochs,
                eval_period_epoch = cfg.trainer.check_val_every_n_epoch,
                criterion_cfg = cfg.task_kwargs.criterion,
                val_metrics = cfg.task_kwargs.val,
                optim_cfg=cfg.optim,
                val_monitor = cfg.task_kwargs.ckpt_monitor,
                val_monitor_higher_is_better = cfg.task_kwargs.ckpt_monitor_higher_is_better,
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

                        # print('key', cls)
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

        
        elif training_mode == 'knn':
            model_wrapper.load_encoder(cfg.model.default_cls_blk_indices)

            results_list = eval_knn_with_model(
                model_wrapper,
                cfg.output_dir,
                data_module.dataset_train,
                data_module.dataset_test,
                nb_knn = cfg.nb_knn,
                normmode_list = cfg.normmode_list,
                temperature_list = cfg.temperature_list,
                autocast_dtype = torch.bfloat16,
                metric_cfg = cfg.task_kwargs.val,
                dl_cfg = dl_cfg,
                num_classes = cfg.dataset.num_classes,)

        else :
            raise ValueError(f'Unknown training_mode: {training_mode}')
    else:
        raise ValueError(f'Unknown engine: {engine}')

    # save results
    results = pd.DataFrame(results_list)
    results = results[['metric_str','val','best_classifier']]
    results.rename(columns={'metric_str':'metric'}, inplace=True)
    results.reset_index(drop=True, inplace=True)
    print(f'Results: \n\n{results.to_string()}\n')
    results.to_csv(os.path.join(cfg.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    os.environ["MODEL_WEIGHTS_DIR"] = os.getenv("MODEL_WEIGHTS_DIR", "./fm_weights")
    main()
