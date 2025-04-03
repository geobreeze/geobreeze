#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=xsensor
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=01:00:00
#SBATCH --array=14,17

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
OLD_ODIR=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/fmplayground
# -----------------------------


all_tasks=(

    "x_hypv_SD_MD panopticon -1 300"
    "x_hypv_SD_MD dofa -1 500"
    "x_hypv_SD_MD_rgb dinov2 -1 500"

    "x_hypv_MD_SD panopticon -1 200"
    "x_hypv_MD_SD dofa -1 500"
    "x_hypv_MD_SD_rgb dinov2 -1 500"

    "x_so2sat_s1_s2 panopticon -1 200"
    "x_so2sat_s1_s2 dofa -1 500"
    "x_so2sat_s1_s2_rgb dinov2 -1 500"

    "x_so2sat_s2_s1 panopticon -1 200"
    "x_so2sat_s2_s1 dofa -1 500"
    "x_so2sat_s2_s1_rgb dinov2 -1 500"

    "x_eurosat_s1_s2 panopticon -1 200"
    "x_eurosat_s1_s2 dofa -1 500"
    "x_eurosat_s1_s2_rgb dinov2 -1 500"

    "x_eurosat_s2_s1 panopticon -1 200"
    "x_eurosat_s2_s1 dofa -1 500"
    "x_eurosat_s2_s1_rgb dinov2 -1 500"
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
    # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/base/$model/\' \
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$mode \
        +output_dir=\'$ODIR/xsensor/$dataset/base/$model/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=16 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \
        # optim.epochs=100 \
        # ++data.train.subset=100 \
        # ++data.train.subset=10 \
        # ++data.train.subset=10 \
        # overwrite=true \

done