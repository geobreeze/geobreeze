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
This file executes experiments of subsampling channels. The code below is only
for m-eurosat, other datasets can be executed similarly.
"""



dataset=m-eurosat_resize
dataset_folder_name=m-eurosat
gsd_mode=only_val
full_size=64
tasks=(
    "100 64"
    "50 32"
    "25 16"
    "12.5 8"
)

models=(
    "panopticon -1 200"
    "dofa -1 700"
    "senpamae -1 400"
    "dinov2 [3,2,1] 300"
    "croma_12b [0,1,2,3,4,5,6,7,8,9,11,12] 200"
    "softcon_13b -1 300"
    "anysat_s2 [1,2,3,4,5,6,7,8,11,12] 100"
)





# Generate all tasks as the cross product of models and tasks
all_tasks=()
for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        all_tasks+=("$model $task")
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


# execution
for task_id in "${task_ids[@]}"
do
    task=${all_tasks[$task_id]}
    echo "Running Task: $task with $gsd_mode"
    set $task
    model=$1
    ids=$2
    bsz=$3
    prc=$4
    size=$5

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi
    # nchns=$(echo "$ids" | awk -F',' '{print NF}')

    # set gsdmode
    if [ "$gsd_mode" == "only_val" ]; then
        train_size=$full_size
        val_size=$size
        test_size=$size
    elif [ "$gsd_mode" == "also_train" ]; then
        train_size=$size
        val_size=$size
        test_size=$size
    elif [ "$gsd_mode" == "only_train" ]; then
        train_size=$size
        val_size=$full_size
        test_size=$full_size
    else
        echo "Error: Invalid gsd_mode value. Must be 'only_val' or 'also_train'."
        exit 1
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=linear_probe \
        +output_dir=\'$ODIR/gsd_inv/$gsd_mode/$dataset_folder_name/linear_probe/$model/$prc/\' \
        dl.batch_size=$bsz \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        data.train.transform.0.size=$train_size \
        data.val.transform.0.size=$val_size \
        data.test.transform.0.size=$test_size \
        $add_kwargs \
        # data.train.transform.0.size=$size \
        # overwrite=true \

done