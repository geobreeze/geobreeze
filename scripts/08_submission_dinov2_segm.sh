#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=dinov2
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=02:00:00
#SBATCH --array=0-1

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python $REPO_PATH/geobreeze/main.py"
OLD_ODIR=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/fmplayground
# -----------------------------


all_tasks=(
    "spacenet1 dinov2 [4,2,1] 600 spacenet1 -1"
)

optim=segmentation
lrs="1e-1 1e-2 1e-3 1e-4 1e-5 1e-6"

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

    set $task
    dataset=$1
    model=$2
    ids=$3
    batch_size=$4
    ds_str_output_dir=$5
    train_subset=$6
    val_subset=$6

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    for lr in $lrs
    do
        echo "Running with lr: $lr"
        # main command
        $cmd \
            +model=base/$model \
            +data=$dataset\
            +optim=$optim \
            +output_dir=\'$ODIR/submission/$ds_str_output_dir/base/$model/\' \
            dl.batch_size=$batch_size \
            dl.num_workers=8 \
            num_gpus=1 \
            seed=21 \
            ++data.train.subset=$train_subset \
            ++data.val.subset=$val_subset \
            optim.base_lr=$lr \
            $add_kwargs \
            # optim.epochs=1 \
            # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/base/$model/\' \
            # overwrite=true \
    done
done