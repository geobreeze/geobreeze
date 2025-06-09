#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=sinv_senpamae
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=01:00:00
#SBATCH --array=0-15


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------



dataset=m-forestnet
all_tasks=(
  " 1 [3]"
  " 1 [5]"
  " 1 [1]"
  " 1 [0]"
  " 1 [2]"
  " 3 [2,4,5]"
  " 3 [2,3,5]"
  " 3 [2,3,4]"
  " 3 [0,1,2]"
  " 3 [0,1,4]"
  " 5 [1,2,3,4,5]"
  " 5 [0,1,2,4,5]"
  " 5 [0,1,2,3,4]"
  " 5 [0,1,3,4,5]"
  " 5 [0,1,2,3,5]"
  " 6 [0,1,2,3,4,5]"
)

# model=panopticon
# bsz=300

# model=dofa
# bsz=900

model=senpamae
bsz=900



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
    echo "Running Task: $model $bsz $task"
    set -- $task
    num_bands=$1
    ids=$2


    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=linear_probe \
        +output_dir=\'$ODIR/spec_inv/$dataset/linear_probe/$model/$num_bands/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=12 \
        num_gpus=1 \
        seed=21 \
        +data.train.band_ids=$ids \
        +data.val.band_ids=$ids \
        +data.test.band_ids=$ids \
        # overwrite=true \

done