#!/bin/bash
# optional: your sbatch options
# ...
# SBATCH array=0-30


# setup
REPO_PATH=/path/to/geobreeze
PYTHON=/path/to/your/python/bin

export $(cat $REPO_PATH/.env)
cmd="$PYTHON $REPO_PATH/geobreeze/main.py"


"""
This file executes experiments for linear probing or kNN classification. 
Below, we only list m-eurosat and resisc45. Other datasets can be added similarly.
Most of the dataset configurations are already created in `config/data`.
"""


# list all tasks with argument as string separated by spaces:
#  - model
#  - dataset
#  - batch_size
#  - channel ids to pass from dataset, -1 or empty for all
#  - subset in percentage to use, -1 for no subset

base_output_dir=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/debug
mode=linear_Probe # or knn


all_tasks=(

    # m-eurosat, 0-7 (50e = 0h15 (lp)) 
    'base/dinov2 m-eurosat 800 [3,2,1]'
    'base/croma_12b m-eurosat 900 [0,1,2,3,4,5,6,7,8,9,11,12]'
    'base/softcon_13b m-eursat 900'
    'base/anysat_s2 m-eurosat 100 [1,2,3,4,5,6,7,8,11,12]'
    'base/galileo_s2 m-eurosat 500 [1,2,3,4,5,6,7,8,11,12]'
    'base/senpamae m-eurosat 500'
    'base/dofa m-eurosat 800'
    'base/panopticon m-eurosat 200'

    # resisc45, 8-12 (50e = 1h30 (lp)) 
    'base/dinov2 resisc45 900'
    'base/anysat_spot resisc45 100' # needs ~3h
    'base/senpamae resisc45 900'
    'base/dofa resisc45 900'
    'base/panopticon resisc45 400'

    # other datasets work analogously
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


# execute tasks
for task_id in "${task_ids[@]}"
do

    task=${all_tasks[$task_id]}
    echo "Training Mode: $mode"
    echo "Running Task: $task"

    set $task
    model=$1
    dataset=$2
    batch_size=$3
    ids=$4
    subset=$5

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "" ] && [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    if [ "$subset" != "" ] && [ "$subset" != "-1" ]; then
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
        +output_dir=\'$base_output_dir/$mode/$dataset/$model/$ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \


done