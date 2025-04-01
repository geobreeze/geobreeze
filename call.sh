# ---------- HOREKA ------------
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


model=panopticon
dataset=resisc45
size=32


$cmd \
    +model=base/$model \
    +data=$dataset\
    +optim=knn \
    +output_dir=\'$ODIR/gsd_inv/also_train/$dataset/$model/$size/\' \
    dl.batch_size=100 \
    dl.num_workers=8 \
    num_gpus=1 \
    seed=21 \
    +data=augms/gsd_inv \
    data.train.transform.0.size=$size \
    data.val.transform.0.size=$size \
    data.test.transform.0.size=$size \
    $add_kwargs \



# knn
# $PY_EXECUTABLE "$REPO_PATH/geobreeze/main.py" \
#     +model=base/panopticon \
#     +data=m-eurosat_1-2-4 \
#     +optim=knn \
#     +output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug/eval/6 \
#     dl.batch_size=100 \
#     dl.num_workers=8 \
#     num_gpus=1 \
#     seed=21 \
#     overwrite=true \
#     # +optim.epochs=2 \
#     # +data.train.subset=0.1 \



# segmentation
# $PY_EXECUTABLE "$REPO_PATH/geobreeze/main.py" \
#     +model=base/panopticon \
#     +data=m-cashew \
#     +optim=segmentation \
#     +output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug/eval/4 \
#     dl.batch_size=100 \
#     dl.num_workers=8 \
#     num_gpus=1 \
#     seed=21 \
#     overwrite=true \
#     # +optim.epochs=2 \
#     # +data.train.subset=0.1 \


# linear_probe
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



