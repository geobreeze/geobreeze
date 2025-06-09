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
#SBATCH --array=0-53

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------


task_ids=(
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
gsd_mode=also_train
dataset=corine-10
full_size=128
task_gsds=(
    # "100 128"
    "50 64"
    "25 32"
    # "12.5 16"
)


train_mode=linear_probe
model=panopticon
bsz=200
epochs=50


# merge tasks
all_tasks=()
for ids in "${task_ids[@]}"
do
    for gsds in "${task_gsds[@]}"
    do
        all_tasks+=("$ids $gsds")
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


# execute
for task_id in "${task_ids[@]}"
do
    task=${all_tasks[$task_id]}
    echo $task $train_mode
    set -- $task
    ids=$1
    prc=$2
    size=$3

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
            data.train.transform=\${_vars.augm.cls.val_resize} \
            data.val.transform=\${_vars.augm.cls.val_resize} \
            data.test.transform=\${_vars.augm.cls.val_resize} \
            "
    elif [ "$train_mode" == "linear_probe" ]; then
        add_kwargs="$add_kwargs \
            data.train.transform=\${_vars.augm.cls.train_resize} \
            data.val.transform=\${_vars.augm.cls.val_resize} \
            data.test.transform=\${_vars.augm.cls.val_resize} \
            optim.check_val_every_n_epoch=100 \
            optim.save_checkpoint_frequency_epoch=100 \
            logger=none \
            "
    else
        echo "Unknown training mode: $train_mode"
        exit 1
    fi

    # set gsdmode
    if [ "$gsd_mode" == "only_val" ]; then
        train_size=$full_size
        val_size=$size
        test_size=$size
    elif [ "$gsd_mode" == "also_train" ]; then
        train_size=$size
        val_size=$size
        test_size=$size
    elif [ "$gsd_mode" == "only_train" ]; then
        train_size=$size
        val_size=$full_size
        test_size=$full_size
    else
        echo "Error: Invalid gsd_mode value. Must be 'only_val' or 'also_train'."
        exit 1
    fi

    nchns=$(echo "$ids" | awk -F',' '{print NF}')

    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$train_mode \
        +output_dir=\'$ODIR/gsd_spec_inv/$gsd_mode/$dataset/$model/$train_mode/$prc/$nchns/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=12 \
        num_gpus=1 \
        seed=21 \
        optim.epochs=$epochs \
        $add_kwargs \
        _vars.augm.cls.train_resize.0.size=$train_size \
        _vars.augm.cls.val_resize.0.size=$val_size \
        # data.train.transform.0.size=$train_size \
        # data.val.transform.0.size=$val_size \
        # data.test.transform.0.size=$test_size \
        
        # overwrite=true \

done