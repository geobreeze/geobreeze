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
#SBATCH --time=00:30:00
#SBATCH --array=0-185


# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------

# m-eurosat




dataset=m-eurosat_resize
models=(
    "panopticon [1] 200"
    "panopticon [8] 200"
    "panopticon [7] 200"
    "panopticon [0] 200"
    "panopticon [12] 200"
    "panopticon [0,5,7] 200"
    "panopticon [9,10,12] 200"
    "panopticon [2,5,7] 200"
    "panopticon [3,4,11] 200"
    "panopticon [2,5,10] 200"
    "panopticon [2,4,10,11,12] 200"
    "panopticon [0,3,4,7,8] 200"
    "panopticon [1,2,4,6,12] 200"
    "panopticon [1,2,4,7,9] 200"
    "panopticon [0,1,10,11,12] 200"
    "panopticon [3,5,6,7,8,10,11] 200"
    "panopticon [0,1,2,3,4,5,7] 200"
    "panopticon [1,2,3,4,7,8,9] 200"
    "panopticon [0,4,6,7,8,11,12] 200"
    "panopticon [2,5,6,7,8,9,12] 200"
    "panopticon [0,1,3,4,5,6,8,9,11] 200"
    "panopticon [0,1,2,4,5,7,8,9,11] 200"
    "panopticon [1,2,4,5,7,9,10,11,12] 200"
    "panopticon [0,2,3,4,5,7,10,11,12] 200"
    "panopticon [1,2,3,4,5,6,8,10,11] 200"
    "panopticon [0,1,4,5,6,7,8,9,10,11,12] 200"
    "panopticon [0,1,3,4,5,6,7,8,9,11,12] 200"
    "panopticon [0,1,2,3,4,5,6,8,9,11,12] 200"
    "panopticon [0,1,2,3,4,5,7,8,9,11,12] 200"
    "panopticon [0,1,2,3,5,6,8,9,10,11,12] 200"
    "panopticon [0,1,2,3,4,5,6,7,8,9,10,11,12] 200"
    "dofa [1] 700"
    "dofa [8] 700"
    "dofa [7] 700"
    "dofa [0] 700"
    "dofa [12] 700"
    "dofa [0,5,7] 700"
    "dofa [9,10,12] 700"
    "dofa [2,5,7] 700"
    "dofa [3,4,11] 700"
    "dofa [2,5,10] 700"
    "dofa [2,4,10,11,12] 700"
    "dofa [0,3,4,7,8] 700"
    "dofa [1,2,4,6,12] 700"
    "dofa [1,2,4,7,9] 700"
    "dofa [0,1,10,11,12] 700"
    "dofa [3,5,6,7,8,10,11] 700"
    "dofa [0,1,2,3,4,5,7] 700"
    "dofa [1,2,3,4,7,8,9] 700"
    "dofa [0,4,6,7,8,11,12] 700"
    "dofa [2,5,6,7,8,9,12] 700"
    "dofa [0,1,3,4,5,6,8,9,11] 700"
    "dofa [0,1,2,4,5,7,8,9,11] 700"
    "dofa [1,2,4,5,7,9,10,11,12] 700"
    "dofa [0,2,3,4,5,7,10,11,12] 700"
    "dofa [1,2,3,4,5,6,8,10,11] 700"
    "dofa [0,1,4,5,6,7,8,9,10,11,12] 700"
    "dofa [0,1,3,4,5,6,7,8,9,11,12] 700"
    "dofa [0,1,2,3,4,5,6,8,9,11,12] 700"
    "dofa [0,1,2,3,4,5,7,8,9,11,12] 700"
    "dofa [0,1,2,3,5,6,8,9,10,11,12] 700"
    "dofa [0,1,2,3,4,5,6,7,8,9,10,11,12] 700"
)

tasks=(
    "50 32"
    "25 16"
    "16.6 11"
    # "12.5 8"
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

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=linear_probe \
        +output_dir=\'$ODIR/gsd_spec_inv/also_train/m-eurosat/$model/linear_probe/$prc/$nchns/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        data.train.transform.0.size=$size \
        data.val.transform.0.size=$size \
        data.test.transform.0.size=$size \
        $add_kwargs \
        # overwrite=true \

done