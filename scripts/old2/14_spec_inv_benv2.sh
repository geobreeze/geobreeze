#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=benv2
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=00:30:00
#SBATCH --array=0-63

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


all_tasks=(
  "[7]"
  "[5]"
  "[6]"
  "[0]"
  "[11]"
  "[2]"
  "[9]"
  "[3]"
  "[4]"
  "[8]"
  "[10]"
  "[1]"
  "[4,7]"
  "[2,10]"
  "[1,2]"
  "[6,9]"
  "[7,10]"
  "[2,9]"
  "[9,10]"
  "[1,10]"
  "[2,5]"
  "[1,7]"
  "[5,8]"
  "[4,6]"
  "[3,8]"
  "[5,7]"
  "[9,11]"
  "[1,8]"
  "[0,9]"
  "[0,11]"
  "[4,11]"
  "[2,11]"
  "[3,5]"
  "[1,11]"
  "[7,8]"
  "[3,10]"
  "[0,8]"
  "[0,1]"
  "[0,6,9]"
  "[2,4,8]"
  "[0,2,8]"
  "[2,9,10]"
  "[1,9,10]"
  "[1,6,7]"
  "[1,8,11]"
  "[1,7,9]"
  "[6,8,11]"
  "[3,6,7]"
  "[2,5,6]"
  "[1,7,8]"
  "[3,4,6]"
  "[0,1,8]"
  "[1,2,5]"
  "[5,6,10]"
  "[4,5,11]"
  "[3,7,9]"
  "[4,8,10]"
  "[4,5,10]"
  "[1,4,6]"
  "[0,8,11]"
  "[1,3,9]"
  "[3,6,8]"
  "[1,3,6]"
  "[0,4,7]"
)

dataset=benv2-s2-01
train_mode=linear_probe
model=panopticon
bsz=200
epochs=25



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
    echo $task $train_mode $dataset
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
        +output_dir=\'$ODIR/investigate_chn_influence/$dataset/$train_mode/$model/$nchns/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=12 \
        num_gpus=1 \
        seed=21 \
        optim.epochs=$epochs \
        $add_kwargs \
        
        # overwrite=true \

done