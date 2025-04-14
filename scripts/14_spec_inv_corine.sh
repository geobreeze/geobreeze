#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=corine_1band
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=02:30:00
#SBATCH --array=0-26

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


all_tasks=(
  "[0]"
  "[7]"
  "[15]"
  "[23]"
  "[30]"
  "[38]"
  "[46]"
  "[54]"
  "[61]"
  "[69]"
  "[77]"
  "[84]"
  "[92]"
  "[100]"
  "[108]"
  "[115]"
  "[123]"
  "[131]"
  "[139]"
  "[146]"
  "[154]"
  "[162]"
  "[169]"
  "[177]"
  "[185]"
  "[193]"
  "[200]"
)

dataset=corine
train_mode=linear_probe
model=panopticon
bsz=200
epochs=50

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