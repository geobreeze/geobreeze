# Evaluation of Foundation Models for Earth Observation

This repository contains the evaluation code of the Panopticon paper.
The code developed from the `geofm` branch in the [DOFA-pytorch](https://github.com/xiong-zhitong/DOFA-pytorch) repository. The code in `geofm/engine/accelerated` is from [DINOv2](https://github.com/facebookresearch/dinov2) with minor adjustments.

## Setup

Navigate into the root directory of this repository and do
```
conda create -n dofa-pytorch python=3.10 --yes
conda activate dofa-pytorch
pip install -U openmim
pip install torch==2.1.2
mim install mmcv==2.1.0 mmsegmentation==1.2.2
pip install -e .
```

Define the following environment variables in an .env file in the root directory of this repository:
```shell
MODEL_WEIGHTS_DIR=<path/to/your/where/you/want/to/store/weights>
TORCH_HOME=<path/to/your/where/you/want/to/store/torch/hub/weights>
DATASETS_DIR=<path/to/your/where/you/want/to/store/all/other/datasets>
GEO_BENCH_DIR=<path/to/your/where/you/want/to/store/GeoBench>
ODIR=<path/to/your/where/you/want/to/store/logs>
REPO_PATH=<path/to/this/repo>
```

When using any of the FMs, the init method will check whether it can find the pre-trained checkpoint of the respective FM in the above `MODEL_WEIGHTS_DIR` and download it there if not found. If you do not change the env
variable, the default will be `./fm_weights`.

Some models depend on [torch hub](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved), which by default will load models to `~.cache/torch/hub`. If you would like to change the directory if this to
for example have a single place where all weights across the models are stored, you can also change



## Supported Models

- CROMA
- SoftCon
- AnySat
- Galileo
- SenPaMAE
- DOFA
- Panopticon

## Supported Datasets

- GeoBench
- BigEarthNetV2
- Resisc45
- Corine
- SpaceNet1
- FMoW
- HyperView
- DigitalTyphoon
- TropicalCyclone

---


## Running Experiments

To run, e.g., linear probing on a model, execute

```bash
export $(cat .env)

python geofm_src/main.py \
   model=base/panopticon \
   dataset=geobench_eurosat_13b \
   output_dir="${ODIR}/sanity_check/" \
   +model.training_mode=linear_probe \
   ++batch_size=200 \
   num_gpus=1 \
   num_workers=8 \
   epochs=2 \
   warmup_epochs=0 \
   trainer.check_val_every_n_epoch=1 \
   +optim=sgd \
   seed=21 \
```

## Attribution

This codebase originated from a fork network. To give geobreeze a well-defined and fresh start, we opted to not include the full commit history. To give attribution, we highlight important prior milestones in our initial commits:
- [fm-playground](https://github.com/ando-shah/fm-playground/tree/ecfa7b8c04f28f62ec01a4f7fe8ff8be8c5f53a5) as 2nd commit of geobreeze, minor changes to save storage: deleted `DOFA-pytorch/src/foundation_models/modules` and `DOFA-pytorch/src/foundation_models/SatMAE`
- [DOFA-pytorch](https://github.com/xiong-zhitong/DOFA-pytorch/tree/b915a2f6d2983c04fd08a270a09e5032e9eb91a9) as 1st commit of geobreeze, minor changes to save storage: deleted `DOFA-pytorch/src/foundation_models/modules` and `DOFA-pytorch/src/foundation_models/SatMAE`
