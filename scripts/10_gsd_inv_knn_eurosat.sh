#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=gsd_inv
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=00:40:00
#SBATCH --array=0-15


# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------

# m-eurosat

dataset=m-eurosat_gsdinv
dataset_folder_name=m-eurosat
gsd_mode=only_val
full_size=64
tasks=(
    "100 64"
    "50 32"
    "25 16"
    "12.5 8"
    # "16.6 11"
)


models=(
    # "panopticon -1 200"
    # "dofa -1 700"
    "senpamae -1 400"
    # "dinov2 [3,2,1] 300"
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



for task_id in "${task_ids[@]}"
do
    task=${all_tasks[$task_id]}
    echo "Running Task: $task"
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
    nchns=$(echo "$ids" | awk -F',' '{print NF}')

    # set gsdmode
    if [ "$gsd_mode" == "only_val" ]; then
        train_size=$full_size
    elif [ "$gsd_mode" == "also_train" ]; then
        train_size=$size
    else
        echo "Error: Invalid gsd_mode value. Must be 'only_val' or 'also_train'."
        exit 1
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=knn \
        +output_dir=\'$ODIR/gsd_inv/$gsd_mode/$dataset_folder_name/knn/$model/$prc/\' \
        dl.batch_size=100 \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        data.train.transform.0.size=$train_size \
        data.val.transform.0.size=$size \
        data.test.transform.0.size=$size \
        $add_kwargs \
        # data.train.transform.0.size=$size \
        # overwrite=true \

done