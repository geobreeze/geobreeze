#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=dinvo2_segm
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=2:00:00
#SBATCH --array=0-23


# ---------- HOREKA ------------
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
PYTHON=/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/e3/bin/python
# PYTHON=python
cmd="$PYTHON $REPO_PATH/geobreeze/main.py"
# -----------------------------
echo $(which $PYTHON)

# list all tasks with argument as string separated by spaces:
#  - model
#  - dataset
#  - batch_size
#  - channel ids to pass from dataset, -1 or empty for all
#  - subset in percentage to use, -1 for no subset

base_output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug


tasks=(
    # m-eurosat, 0-7 (50e = 0h15 (lp)) 
    'base/dinov2 m-pv4ger-seg 200'
    'base/dinov2 m-chesapeake 200 [0,1,2]'
    'base/dinov2 m-cashew-plant 200 [3,2,1]'
    'base/dinov2 m-SA-crop-type 200 [3,2,1]'
    'base/dinov2 m-nz-cattle 200'
    'base/dinov2 m-NeonTree 200'

)

lrs="1e-3 1e-4 1e-5 1e-6"
# lrs="1e-4"
mode=segmentation


# build all tasks as the cross product of tasks and lrs
all_tasks=()
for task in "${tasks[@]}"; do
    for lr in $lrs; do
        all_tasks+=("$lr $task")
    done
done



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


# execute tasks
for task_id in "${task_ids[@]}"
do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set $task
    base_lr=$1
    model=$2
    dataset=$3
    batch_size=$4
    ids=$5
    subset=$6

    # potentially subset channels
    add_kwargs=""
    if [ "$ids" == "" ]; then
        print_ids=-1
    elif [ "$ids" != "-1" ]; then
        print_ids=$ids
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # potentially subset samples
    if [ "$subset" == "" ]; then
        subset=-1
    elif [ "$subset" != "-1" ]; then
        add_kwargs="$add_kwargs \
            ++data.train.subset=$subset \
            ++data.val.subset=$subset \
            ++data.test.subset=$subset "
    fi

    # main command
    $cmd \
        +model=$model \
        +data=$dataset\
        +optim=$mode \
        optim.base_lr=$base_lr \
        +output_dir=\'$base_output_dir/$mode/$dataset/$model/$print_ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \


done