#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=cls_corine_dinov2
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=03:00:00
#SBATCH --array=0-1

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
    # "eurosat-sar dinov2 [0,1,1] 900"


    # "corine-sd dofa -1 300 -1 corine-sd"
    # "corine-sd dofa -1 300 0.1 corine-sd-0.1"
    # "corine-md dofa -1 300 -1 corine-md"
    # "corine-md dofa -1 300 0.1 corine-md-0.1"

    # "m-brick-kiln anysat_s2 [1,2,3,4,5,6,7,8,11,12] 100 -1 m-brick-kiln"
    # "m-brick-kiln anysat_s2 [0,1,2,3,4,5,6,7,8,9] 100 -1 m-brick-kiln"

    # "corine-sd senpamae -1 300 -1 corine-sd"
    # "corine-sd senpamae -1 300 0.1 corine-sd-0.1"
    # "corine-md senpamae -1 300 -1 corine-md"
    # "corine-md senpamae -1 300 0.1 corine-md-0.1"

    # "benv2-s1-10 dinov2 [0,1,1] 400 -1 benv2-s1-10"
    # "benv2-s1-10 dinov2 [0,1,0] 400 -1 benv2-s1-10"
    # "benv2-s1-10 dinov2 [1,0,0] 400 -1 benv2-s1-10"
    # "benv2-s1-10 dinov2 [1,0,1] 400 -1 benv2-s1-10"

    "corine dinov2 [47,28,14] 400 -1 corine-MD"
    "corine dinov2 [47,30,15] 400 -1 corine-SD"
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
    subset=$5
    ds_name_output_dir=$6

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    if [ "$subset" != "-1" ]; then
        add_kwargs="$add_kwargs \
            ++data.train.subset=$subset \
            ++data.val.subset=$subset \
            ++data.test.subset=$subset "
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=$mode \
        +output_dir=\'$ODIR/doublecheck/$ds_name_output_dir/base/$model/$ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        optim.check_val_every_n_epoch=100 \
        $add_kwargs \
        # optim.epochs=1 \
        # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/base/$model/\' \
        # overwrite=true \

done