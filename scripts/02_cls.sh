#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=s1
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=00:30:00
#SBATCH --array=0

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
OLD_ODIR=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/fmplayground
# -----------------------------


all_tasks=(

    # "m-so2sat-s1 softcon_2b -1 900"
    # "m-so2sat-s1 croma_s1 -1 900"
    # "m-so2sat-s1 panopticon -1 400"
    # "m-so2sat-s1 dofa -1 500"
    # "m-so2sat-s1 dinov2 [0,4,4] 900"

    # "eurosat-sar softcon_2b -1 900"
    # "eurosat-sar croma_s1 -1 900"
    # "eurosat-sar panopticon -1 400"
    # "eurosat-sar dofa -1 500"
    "eurosat-sar dinov2 [0,1,1] 900"

    # "m-eurosat senpamae -1 500"
)

mode=linear_probe


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

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$mode \
        +output_dir=\'$OLD_ODIR/debug/$dataset/base/$model/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \
        # optim.epochs=1 \
        # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/base/$model/\' \
        # overwrite=true \

done