PY_EXECUTABLE=/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze

export $(cat $REPO_PATH/.env)



# $PY_EXECUTABLE "$REPO_PATH/geobreeze/main.py" \
#     +model=base/panopticon \
#     +data=m-eurosat_1-2-4 \
#     +optim=knn \
#     +output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug/eval/3 \
#     dl.batch_size=100 \
#     dl.num_workers=8 \
#     num_gpus=1 \
#     seed=21 \
#     overwrite=true \
#     # +optim.epochs=2 \
#     # +data.train.subset=0.1 \



$PY_EXECUTABLE "$REPO_PATH/geobreeze/main.py" \
    +model=base/panopticon \
    +data=m-eurosat_1-2-4 \
    +optim=knn \
    +output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug/eval/3 \
    dl.batch_size=100 \
    dl.num_workers=8 \
    num_gpus=1 \
    seed=21 \
    overwrite=true \
    # +optim.epochs=2 \
    # +data.train.subset=0.1 \
