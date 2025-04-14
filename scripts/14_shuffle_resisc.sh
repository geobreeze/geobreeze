#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=adapt_dinov2
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=02:00:00
#SBATCH --array=0-11


# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------

# m-eurosat

all_tasks=(
    "resisc45 linear_probe dinov2 [0,1,2] 200"
    "resisc45 linear_probe dinov2 [0,2,1] 200"
    "resisc45 linear_probe dinov2 [1,0,2] 200"
    "resisc45 linear_probe dinov2 [1,2,0] 200"
    "resisc45 linear_probe dinov2 [2,0,1] 200"
    "resisc45 linear_probe dinov2 [2,1,0] 200"

    "resisc45 knn dinov2 [0,1,2] 200"
    "resisc45 knn dinov2 [0,2,1] 200"
    "resisc45 knn dinov2 [1,0,2] 200"
    "resisc45 knn dinov2 [1,2,0] 200"
    "resisc45 knn dinov2 [2,0,1] 200"
    "resisc45 knn dinov2 [2,1,0] 200"
)



# process which tasks to execute
if [ $# -eq 0 ]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        task_ids=($SLURM_ARRAY_TASK_ID)
    else
        task_ids=($(seq 0 $((${#all_tasks[@]}-1))))
    fi
else
    task_ids=("$@")
fi



for task_id in "${task_ids[@]}"
do
    task=${all_tasks[$task_id]}
    echo "Running Task: $task"
    set $task
    dataset=$1
    train_mode=$2
    model=$3
    ids=$4
    bsz=$5

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    if [ "$train_mode" == "knn" ]; then
        add_kwargs="$add_kwargs \
            +data.train.transform=\${_vars.augm.cls.val}"
    elif [ "$train_mode" == "linear_probe" ]; then
        add_kwargs="$add_kwargs \
            +data.train.transform=\${_vars.augm.cls.train}"
    else
        echo "Unknown training mode: $train_mode"
        exit 1
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$train_mode \
        +output_dir=\'$ODIR/domain_adapt/shuffle/$dataset/$train_mode/$model/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \
        # overwrite=true \

done