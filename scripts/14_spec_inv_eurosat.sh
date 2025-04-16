#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=eurosat
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=02:30:00
#SBATCH --array=0-53

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


all_tasks=(
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
    # "[6]"
    # "[1]"
    # "[5]"
    # "[12]"
    # "[2]"
    # "[10]"
    # "[11]"
    # "[7]"
    # "[3]"
    # "[4]"
    # "[0]"
    # "[9]"
    # "[8]"
    # "[8,10]"
    # "[3,4]"
    # "[3,5]"
    # "[1,2]"
    # "[5,10]"
    # "[0,5]"
    # "[3,7]"
    # "[4,12]"
    # "[0,12]"
    # "[7,10]"
    # "[4,7]"
    # "[10,11]"
    # "[9,10]"
    # "[6,12]"
    # "[6,11]"
    # "[2,10]"
    # "[8,11]"
    # "[0,3]"
    # "[2,7]"
    # "[6,8]"
    # "[1,6]"
    # "[8,9]"
    # "[1,12]"
    # "[2,11]"
    # "[2,6]"
    # "[6,11,12]"
    # "[2,8,12]"
    # "[5,6,11]"
    # "[6,10,12]"
    # "[1,2,11]"
    # "[0,6,10]"
    # "[2,7,12]"
    # "[2,6,10]"
    # "[1,4,11]"
    # "[4,5,12]"
    # "[2,3,4]"
    # "[6,10,11]"
    # "[2,10,11]"
    # "[5,7,12]"
    # "[5,7,11]"
    # "[1,3,12]"
    # "[1,8,9]"
    # "[2,9,11]"
    # "[2,4,11]"
    # "[0,2,5]"
    # "[1,8,12]"
    # "[1,3,8]"
    # "[5,11,12]"
    # "[2,3,7]"
    # "[0,5,12]"
    # "[0,9,10]"


    # new with 3 channels
  "[0,1,5]"
  "[2,5,10]"
  "[2,7,9]"
  "[5,7,11]"
  "[2,8,9]"
  "[0,3,10]"
  "[4,5,9]"
  "[7,9,11]"
  "[3,5,7]"
  "[0,2,8]"
  "[4,5,6]"
  "[0,6,7]"
  "[2,3,8]"
  "[6,9,10]"
  "[5,8,10]"
  "[2,4,9]"
  "[1,3,10]"
  "[1,8,11]"
  "[0,4,11]"
  "[0,7,10]"
  "[0,6,9]"
  "[0,5,8]"
  "[0,5,9]"
  "[4,5,11]"
  "[4,9,10]"
  "[2,6,10]"
  "[1,3,4]"
  "[1,8,9]"
  "[2,4,10]"
  "[1,2,7]"
  "[5,6,10]"
  "[0,1,2]"
  "[1,4,10]"
  "[2,4,7]"
  "[1,9,10]"
  "[3,8,9]"
  "[1,3,7]"
  "[1,6,7]"
  "[2,10,11]"
  "[8,9,11]"
  "[0,7,8]"
  "[5,6,7]"
  "[1,10,11]"
  "[2,9,11]"
  "[5,9,11]"
  "[1,5,7]"
  "[3,7,11]"
  "[3,7,10]"
  "[6,7,11]"
  "[1,6,10]"
  "[2,7,8]"
  "[8,10,11]"
  "[0,4,7]"
  "[3,7,8]"
  "[3,8,11]"
  "[4,6,10]"
  "[3,4,11]"
  "[1,7,10]"
  "[8,9,10]"
  "[0,3,5]"
  "[0,1,3]"
  "[4,6,11]"
  "[0,8,10]"
  "[2,3,6]"
  "[4,6,7]"
  "[1,2,9]"
  "[2,7,11]"
  "[6,7,10]"
  "[2,4,5]"
  "[3,5,10]"
  "[1,7,8]"
  "[9,10,11]"
  "[3,6,9]"
  "[3,4,6]"
  "[2,3,11]"
  "[5,6,11]"
  "[0,9,10]"
  "[0,1,8]"
  "[0,1,11]"
  "[1,5,9]"
  "[4,5,8]"
  "[0,6,11]"
  "[1,4,9]"
  "[2,5,6]"
  "[6,8,9]"
  "[1,4,8]"
  "[2,7,10]"
  "[3,4,10]"
  "[3,4,5]"
  "[1,3,8]"
  "[0,2,10]"
  "[1,4,6]"
  "[1,6,8]"
  "[3,4,9]"
  "[1,2,8]"
  "[1,4,5]"
  "[7,10,11]"
  "[5,8,9]"
  "[0,2,5]"
  "[0,1,7]"

)

model=panopticon
# dataset="m-eurosat_bandgsd_10to60_knn"
dataset="m-eurosat"
train_mode=linear_probe

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
    echo $task $train_mode
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
            data.train.transform=\${_vars.augm.cls.val}"
        echo "KNN mode"
    elif [ "$train_mode" == "linear_probe" ]; then
        add_kwargs="$add_kwargs \
            data.train.transform=\${_vars.augm.cls.train}\
            optim.check_val_every_n_epoch=100 \
            optim.save_checkpoint_frequency_epoch=100 \
            logger=none \
            "
    else
        echo "Unknown training mode: $train_mode"
        exit 1
    fi

    nchns=$(echo "$ids" | awk -F',' '{print NF}')

    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$train_mode \
        +output_dir=\'$ODIR/gsd_spec_inv/also_train/$dataset/$model/$train_mode/100/$nchns/$ids\' \
        dl.batch_size=200 \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \

        # overwrite=true \
        # +output_dir=\'$ODIR/investigate_chn_influence/$dataset/$train_mode/$model/$nchns/$ids\' \

done