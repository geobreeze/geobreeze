#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=sinv_cor_senpa
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=03:00:00
#SBATCH --array=7-9,16,10,12


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------



dataset=corine
all_tasks=(
  " 1 [179]"
  " 1 [138]"
  " 1 [36]"
  " 1 [76]"
  " 1 [93]"
  " 3 [80,161,189]"
  " 3 [166,190,197]"
  " 3 [3,26,123]"
  " 3 [6,8,77]"
  " 3 [26,28,88]"
  " 5 [41,47,76,177,186]"
  " 5 [73,79,112,145,201]"
  " 5 [6,49,50,65,201]"
  " 5 [11,33,58,62,198]"
  " 5 [2,71,133,146,177]"
  " 9 [50,58,90,106,136,150,151,152,194]"
  " 9 [5,58,63,80,98,142,155,162,170]"
  " 9 [15,38,63,104,132,139,163,173,200]"
  " 9 [6,23,72,100,137,151,157,177,180]"
  " 9 [13,50,86,108,109,143,162,165,174]"
  "13 [0,22,52,72,80,122,126,143,145,155,178,185,192]"
  "13 [2,36,43,64,78,86,92,101,107,127,170,190,196]"
  "13 [1,39,44,64,78,87,106,127,139,144,147,153,179]"
  "13 [18,57,70,71,118,121,155,158,159,181,186,187,193]"
  "13 [1,5,6,9,43,77,130,141,151,161,163,165,177]"
)

# model=panopticon
# bsz=200

# model=dofa
# bsz=600

model=senpamae
bsz=300



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