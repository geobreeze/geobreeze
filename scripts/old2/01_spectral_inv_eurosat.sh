#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=eurosat_knn
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=00:05:00
#SBATCH --array=201-300

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


all_tasks_eurosat=(
  # original bands
  # " 1 [1]"
  # " 1 [8]"
  # " 1 [7]"
  # " 1 [0]"
  # " 1 [12]"
  # " 3 [0,5,7]"
  # " 3 [9,10,12]"
  # " 3 [2,5,7]"
  # " 3 [3,4,11]"
  # " 3 [2,5,10]"
  # " 5 [2,4,10,11,12]"
  # " 5 [0,3,4,7,8]"
  # " 5 [1,2,4,6,12]"
  # " 5 [1,2,4,7,9]"
  # " 5 [0,1,10,11,12]"
  # " 7 [3,5,6,7,8,10,11]"
  # " 7 [0,1,2,3,4,5,7]"
  # " 7 [1,2,3,4,7,8,9]"
  # " 7 [0,4,6,7,8,11,12]"
  # " 7 [2,5,6,7,8,9,12]"
  # " 9 [0,1,3,4,5,6,8,9,11]"
  # " 9 [0,1,2,4,5,7,8,9,11]"
  # " 9 [1,2,4,5,7,9,10,11,12]"
  # " 9 [0,2,3,4,5,7,10,11,12]"
  # " 9 [1,2,3,4,5,6,8,10,11]"
  # "11 [0,1,4,5,6,7,8,9,10,11,12]"
  # "11 [0,1,3,4,5,6,7,8,9,11,12]"
  # "11 [0,1,2,3,4,5,6,8,9,11,12]"
  # "11 [0,1,2,3,4,5,7,8,9,11,12]"
  # "11 [0,1,2,3,5,6,8,9,10,11,12]"
  # "13 [0,1,2,3,4,5,6,7,8,9,10,11,12]"

  # second versoin
  # " 1 [6]"
  # " 1 [1]"
  # " 1 [5]"
  # " 1 [12]"
  # " 1 [2]"
  # " 1 [10]"
  # " 1 [11]"
  # " 1 [7]"
  # " 1 [3]"
  # " 1 [4]"
  # " 1 [0]"
  # " 1 [9]"
  # " 1 [8]"
  # " 2 [8,10]"
  # " 2 [3,4]"
  # " 2 [3,5]"
  # " 2 [1,2]"
  # " 2 [5,10]"
  # " 2 [0,5]"
  # " 2 [3,7]"
  # " 2 [4,12]"
  # " 2 [0,12]"
  # " 2 [7,10]"
  # " 2 [4,7]"
  # " 2 [10,11]"
  # " 2 [9,10]"
  # " 2 [6,12]"
  # " 2 [6,11]"
  # " 2 [2,10]"
  # " 2 [8,11]"
  # " 2 [0,3]"
  # " 2 [2,7]"
  # " 2 [6,8]"
  # " 2 [1,6]"
  # " 2 [8,9]"
  # " 2 [1,12]"
  # " 2 [2,11]"
  # " 2 [2,6]"
  # " 3 [6,11,12]"
  # " 3 [2,8,12]"
  # " 3 [5,6,11]"
  # " 3 [6,10,12]"
  # " 3 [1,2,11]"
  # " 3 [0,6,10]"
  # " 3 [2,7,12]"
  # " 3 [2,6,10]"
  # " 3 [1,4,11]"
  # " 3 [4,5,12]"
  # " 3 [2,3,4]"
  # " 3 [6,10,11]"
  # " 3 [2,10,11]"
  # " 3 [5,7,12]"
  # " 3 [5,7,11]"
  # " 3 [1,3,12]"
  # " 3 [1,8,9]"
  # " 3 [2,9,11]"
  # " 3 [2,4,11]"
  # " 3 [0,2,5]"
  # " 3 [1,8,12]"
  # " 3 [1,3,8]"
  # " 3 [5,11,12]"
  # " 3 [2,3,7]"
  # " 3 [0,5,12]"
  " 3 [0,9,10]"
)

all_tasks_corine=(
  "[0]"
)

model=panopticon
dataset=corine-01
all_tasks=$all_tasks_corine
train_mode=knn

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



# execute
for task_id in "${task_ids[@]}"
do
    task=${all_tasks[$task_id]}
    echo $task
    set -- $task
    ids=$1


    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # adjust train mode args
    if [ "$train_mode" == "knn" ]; then
        add_kwargs="$add_kwargs \
            +data.train.transform=\${_vars.augm.cls.val}"
    elif [ "$train_mode" == "linear_probe" ]; then
        add_kwargs="$add_kwargs \
            +data.train.transform=\${_vars.augm.cls.train}"
    else
        echo "Unknown training mode: $train_mode"
        exit 1
    fi

    nchns=$(echo "$ids" | awk -F',' '{print NF}')

    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$train_mode \
        +output_dir=\'$ODIR/investigate_chn_influence/$dataset/$train_mode/$model/$nchns/$ids\' \
        dl.batch_size=200 \
        dl.num_workers=16 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \
        
        # overwrite=true \

done