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
#SBATCH --time=00:10:00
#SBATCH --array=0-25


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------

# m-eurosat

dataset=m-eurosat_gsdinv
models=(

    "panopticon true [0]"
    "panopticon true [1]"
    "panopticon true [2]"
    "panopticon true [3]"
    "panopticon true [4]"
    "panopticon true [5]"
    "panopticon true [6]"
    "panopticon true [7]"
    "panopticon true [8]"
    "panopticon true [9]"
    "panopticon true [10]"
    "panopticon true [11]"
    "panopticon true [12]"

    # "dofa true [0]"
    # "dofa true [1]" 
    # "dofa true [2]"
    # "dofa true [3]"
    # "dofa true [4]"
    # "dofa true [5]"
    # "dofa true [6]"
    # "dofa true [7]"
    # "dofa true [8]"
    # "dofa true [9]"
    # "dofa true [10]"
    # "dofa true [11]"
    # "dofa true [12]"

    # "dinov2 true [3,2,1]"
    # "dofa true [3,2,1]"
    # "panopticon true [3,2,1]"

    # "dofa false -1"
    # "panopticon false -1"
)

tasks=(
    # "50 32"
    # "25 16"
    # "16.6 11"
    "33.3 21"
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
    subset=$2
    ids=$3
    prc=$4
    size=$5

    # potentially subset
    add_kwargs=""
    if [ "$subset" = true ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=knn \
        +output_dir=\'$ODIR/gsd_spec_inv/also_train/m-eurosat/$model/knn/$prc/$ids\' \
        dl.batch_size=100 \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        data.train.transform.0.size=$size \
        data.val.transform.0.size=$size \
        data.test.transform.0.size=$size \
        $add_kwargs \
        # overwrite=true \

done