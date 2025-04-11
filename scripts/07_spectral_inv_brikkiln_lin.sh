#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=sinv_brikiln
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=2:15:00
#SBATCH --array=0-62


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------



dataset=m-brick-kiln
ds_tasks=(
  " 1 [12]"
  " 1 [11]"
  " 1 [2]"
  " 1 [4]"
  " 1 [10]"
  " 3 [1,9,11]"
  " 3 [5,8,9]"
  " 3 [1,8,11]"
  " 3 [1,4,8]"
  " 3 [3,6,12]"
  " 5 [1,2,6,8,9]"
  " 5 [1,4,6,9,12]"
  " 5 [0,1,3,4,10]"
  " 5 [2,3,6,11,12]"
  " 5 [2,3,5,7,9]"
  " 9 [2,3,4,5,6,7,9,10,12]"
  " 9 [1,3,4,5,6,7,8,10,12]"
  " 9 [0,2,4,6,7,8,10,11,12]"
  " 9 [0,1,2,3,5,6,9,10,11]"
  " 9 [0,1,4,5,7,8,9,10,12]"
  "13 [0,1,2,3,4,5,6,7,8,9,10,11,12]"
)

model_tasks=(
    "panopticon 200"
    "dofa 700"
    "senpamae 400"
)

# create all tasks as cross product
all_tasks=()
for model in "${model_tasks[@]}"; do
    for task in "${ds_tasks[@]}"; do
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


# execute tasks
for task_id in "${task_ids[@]}"
do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"
    set -- $task
    model=$1
    bsz=$2
    num_bands=$3
    ids=$4


    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=linear_probe \
        +output_dir=\'$ODIR/spec_inv/$dataset/linear_probe/$model/$num_bands/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        +data.train.band_ids=$ids \
        +data.val.band_ids=$ids \
        +data.test.band_ids=$ids \
        optim.check_val_every_n_epoch=100 \
        # overwrite=true \

done