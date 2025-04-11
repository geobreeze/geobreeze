#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=sinv_hyp_dofa
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=00:30:00
#SBATCH --array=3,7,12,13


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------



dataset=hyperview
all_tasks=(
  " 1 [46]"
  " 1 [125]"
  " 1 [14]"
  " 1 [49]"
  " 1 [41]"
  " 3 [66,86,95]"
  " 3 [74,95,131]"
  " 3 [41,119,149]"
  " 3 [43,76,135]"
  " 3 [23,35,69]"
  " 5 [5,24,39,96,131]"
  " 5 [3,8,32,59,124]"
  " 5 [44,50,91,119,133]"
  " 5 [20,48,74,85,132]"
  " 5 [7,29,52,57,60]"
  " 9 [13,46,60,79,80,84,133,136,143]"
  " 9 [52,55,72,86,88,92,117,122,134]"
  " 9 [23,28,30,46,54,60,84,91,121]"
  " 9 [3,35,50,56,94,97,127,135,138]"
  " 9 [20,35,37,51,66,73,82,118,147]"
  "13 [5,15,29,32,33,37,60,70,72,107,110,122,138]"
  "13 [24,28,31,43,49,50,57,58,72,109,112,132,149]"
  "13 [1,9,15,26,53,66,71,75,79,88,116,117,149]"
  "13 [8,15,16,22,59,64,67,77,85,88,109,117,122]"
  "13 [14,25,47,70,90,94,104,106,108,122,125,133,136]"
)

# model=panopticon
# bsz=200

model=dofa
bsz=400

# model=senpamae
# bsz=500



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
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        +data.train.band_ids=$ids \
        +data.val.band_ids=$ids \
        +data.test.band_ids=$ids \
        # overwrite=true \

done