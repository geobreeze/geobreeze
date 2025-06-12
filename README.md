# Geobreeze: Simple, Fast, and Flexible Evaluation of Remote Sensing Foundation Models

> **Note:** This is the first version of Geobreeze. We highly welcome your feedback! Please feel free to [open an issue](https://github.com/Panopticon-FM/geobreeze/issues) for any questions, or reach out via email at [leonard.waldmann@tum.de](mailto:leonard.waldmann@tum.de).

Geobreeze enables evaluation of remote sensing foundation models (RSFMs). Our unique value proposition is an abstract model wrapper class designed for ViT-based RSFMs. After implementing this wrapper for an RSFM, evaluation can be performed over multiple datasets with different tasks and evaluation protocols.

Core features:
- Simple: Single compact model wrapper (e.g., 44 lines of code for DINOv2) for all tasks
- Fast: Accelerated linear probing (e.g., 78 configurations in parallel at once with 900 batch size on 40 GB GPU RAM for SoftCon)
- Flexible: 10+ model & 17+ dataset wrappers implemented

Geobreeze was created for the evaluation of [Panopticon](https://github.com/Panopticon-FM/panopticon). It evolved from other repositories, see [Attribution](#attribution).

## Setup

Navigate into the root directory of this repository and do
```
conda create -n geobreeze python=3.10 --yes
conda activate geobreeze
pip install torch==2.1.2
pip install -U openmim
mim install mmcv==2.1.0 mmsegmentation==1.2.2
pip install -r requirements.txt
pip install -e .
```

Define the following environment variables in an .env file in the root directory of this repository:
```shell
MODEL_WEIGHTS_DIR=<path/to/store/weights>
TORCH_HOME=<path/to/store/torch/hub/weights>
GEO_BENCH_DIR=<path/to/store/GeoBench>
DATASETS_DIR=<path/to/store/all/other/datasets>
ODIR=<path/to/store/logs>
REPO_PATH=<path/to/this/repo>
```

When using any of the RSFMs, the init method will check whether it can find the pre-trained checkpoint of the respective RSFM in `MODEL_WEIGHTS_DIR` and download it there if not found. If you do not change the env variable, the default will be `./fm_weights`.

Some models depend on [torch hub](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved), which by default will load models to `~.cache/torch/hub`. If you would like to change the directory, set `TORCH_HOME`.



## Supported Models and Datasets

### Models
- CROMA
- SoftCon
- AnySat
- Galileo
- SenPaMAE
- DOFA
- Panopticon

### Datasets
- GeoBench
- BigEarthNetV2
- Resisc45
- Corine
- SpaceNet1
- FMoW
- HyperView
- DigitalTyphoon
- TropicalCyclone




## Running Experiments

To run, e.g., linear probing on a model, execute

```bash
export $(cat .env)

python geobreeze/main.py \
   +model=base/panopticon \
   +data=m-eurosat \
   +optim=linear_probe \
   +output_dir="$ODIR/sanity_check/" \
   dl.batch_size=100 \
   dl.num_workers=5 \
   num_gpus=1 \
   optim.epochs=10 \
   optim.check_val_every_n_epoch=2 \
   seed=21 
```
In `scripts/`, there are bash files for computing the evaluation of Panopticon. You can call these with
1. `bash /scripts/your_choice.sh` to executed all tasks defined in the script.
2. `bash /scripts/your_choice.sh id` to executed the task with integer id `id`
3. Add your slurm configuration on top of `.sh` files and execute with slurm via array jobs, where each jobs the task with the id `$SLURM_ARRAY_TASK_ID`.
To collect (nested) results of your computations into a .csv, call `geobreeze/collect_results.py /path/to/your/folder`.

## Add Models and Datasets
### Models
1. Create a file in `models/` and implement your class inheriting from `engine/model.py/EvalModelWrapper`.
2. Add an import statement into `models/__init__.py`.
3. Create a config file at `config/model/`.

### Datasets
1. Create a file in `datasets/` and implement your class inferiting from `datasets/base.py/BaseDataset`.
2. Create a metadata file in `datasets/metadata/` containing metainformation on the dataset and individual bands.
3. Create a config file in `config/data/`.

## Attribution

This codebase originated from a fork network. To give geobreeze a well-defined and fresh start, we opted to not include the full commit history. To give attribution, we highlight important prior milestones in our initial commits:
- [fm-playground](https://github.com/ando-shah/fm-playground/tree/ecfa7b8c04f28f62ec01a4f7fe8ff8be8c5f53a5) as 2nd commit of geobreeze, minor changes to save storage: deleted `DOFA-pytorch/src/models/modules` and `DOFA-pytorch/src/models/SatMAE`
- [DOFA-pytorch](https://github.com/xiong-zhitong/DOFA-pytorch/tree/b915a2f6d2983c04fd08a270a09e5032e9eb91a9) as 1st commit of geobreeze, minor changes to save storage: deleted `DOFA-pytorch/src/models/modules` and `DOFA-pytorch/src/models/SatMAE`
