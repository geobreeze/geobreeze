#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=gsd_inv_resisc
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=02:15:00
#SBATCH --array=0-15


# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------

# m-eurosat

dataset=resisc45_resize
dataset_folder_name=resisc45
tasks=(
    "100 224"
    "50 112"
    "25 56"
    "12.5 28"
    # "16.6 11"
)


models=(
    "panopticon -1 200"
    "dofa -1 700"
    "senpamae -1 400"
    "dinov2 -1 300"
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
    # nchns=$(echo "$ids" | awk -F',' '{print NF}')

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=linear_probe \
        +output_dir=\'$ODIR/gsd_inv/also_train/$dataset_folder_name/linear_probe/$model/$prc/\' \
        dl.batch_size=$bsz \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        data.train.transform.0.size=$size \
        data.val.transform.0.size=$size \
        data.test.transform.0.size=$size \
        $add_kwargs \
        # data.train.transform.0.size=$size \
        # overwrite=true \

done