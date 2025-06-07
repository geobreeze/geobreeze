#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=senpamae_dt_tc
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=38        # default: 38
#SBATCH --time=3:00:00
#SBATCH --array=0-1

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
OLD_ODIR=/hkfs/work/workspace/scratch/tum_mhj8661-panopticon/dino_logs/fmplayground
# -----------------------------


# list all tasks with argument as string separated by spaces:
#  - model
#  - dataset
#  - batch_size
#  - channel ids to pass from dataset, -1 for all

all_tasks=(

    # 'base/dinov2 spacenet1_3b 800'

    # m-eurosat, 0-7 (50e = 0h15 (lp)) 
    'base/dinov2 eurosat 800 -1'
    'base/croma_s2 geobench_eurosat_12b 900 -1'
    'base/softcon_13b geobench_eurosat_13b 900'
    'base/anysat_s2 geobench_eurosat_10b 100 [1,2,3,4,5,6,7,8,11,12]'
    'base/galileo_s2 geobench_eurosat_10b 500'
    'base/senpamae geobench_eurosat_13b 500'
    'base/dofa geobench_eurosat_13b 800'
    'base/panopticon geobench_eurosat_13b 200'

    # resisc45, 8-12 (50e = 1h30 (lp)) 
    'base/dinov2 resisc45 900'
    'base/anysat_spot resisc45 100' # beeds ~3h
    'base/senpamae resisc45 900'
    'base/dofa resisc45 900'
    'base/panopticon resisc45 400'

    # benv2-s1, 13-18
    'base/croma_s1 benv2_s1 900'
    'base/softcon_2b benv2_s1 900'
    'base/anysat_s1-asc benv2_s1 100' # needs ~12h 
    'base/galileo_s1 benv2_s1 100' 
    'base/dofa benv2_s1 500'
    'base/panopticon benv2_s1 400'


    # benv2-s2, 19-26 (50e = ~4h)
    'base/dinov2 benv2_rgb 900'
    'base/croma_s2 benv2_s2_12b 900'
    'base/softcon_13b benv2_s2_13b 800'
    'base/softcon_13b benv2_s2_13b_scnorm 800'
    'base/anysat_s2 benv2_s2_10b 100' 
    'base/galileo_s2 benv2_s2_10b 200'
    'base/senpamae benv2_s2_12b 300' 
    'base/dofa benv2_s2_12b 500'
    'base/panopticon benv2_s2_12b 200'

    # benv2-s1, softcon norm: (4-9)
    'base/softcon_2b benv2_s1_scnorm 900'
    'base/softcon_13b benv2_s2_13b_scnorm 800'
    'base/panopticon benv2_s1_scnorm 400'
    'base/panopticon benv2_s2_12b_scnorm 200'
    'base/croma_s1 benv2_s1_scnorm 900'
    'base/croma_s2 benv2_s2_12b_scnorm 900'
    'base/anysat_s1-asc benv2_s1_scnorm 100' # needs ~12h
    'base/galileo_s1 benv2_s1_scnorm 100' 
    'base/dofa benv2_s1_scnorm 500'

    # benv2-s2, 19-26 (50e = ~4h) (10-17)
    'base/dinov2 benv2_rgb_scnorm 900'
    'base/anysat_s2 benv2_s2_10b_scnorm 100' 
    'base/galileo_s2 benv2_s2_10b_scnorm 200'
    'base/senpamae benv2_s2_12b_scnorm 500' 
    'base/dofa benv2_s2_12b_scnorm 500'

    # forestnet, 27-31 (1h) 
    'base/dinov2 geobench_forestnet_rgb 900'
    'base/anysat_naip geobench_forestnet_4b 100'
    'base/senpamae geobench_forestnet_6b 600'
    'base/dofa geobench_forestnet_6b 900'
    'base/panopticon geobench_forestnet_6b 200'

    # fmow-wv, 32-34 (50e = ~4h) 
    'base/senpamae fmow_8b 600'
    'base/dofa fmow_8b 900'
    'base/panopticon fmow_8b 200'
    'base/dinov2 fmow_8b_rgb 900'

    # fmow-rgb, 35-39 (50e = ~4h) 
    'base/dinov2 fmow_rgb 900'
    'base/anysat_spot fmow_rgb 100'
    'base/senpamae fmow_rgb 900'
    'base/dofa fmow_rgb 900'
    'base/panopticon fmow_rgb 400'

    ######################################
    #### other geobench cls tasks
    ######################################

    # so2sat-s2 (40-47)
    'base/dinov2 geobench_so2sat_rgb 800'
    'base/croma_s2 geobench_so2sat_12b 900'
    'base/softcon_13b geobench_so2sat_13b 900'
    'base/anysat_s2 geobench_so2sat_10b 100'
    'base/galileo_s2 geobench_so2sat_10b 500'
    'base/senpamae geobench_so2sat_10b 500'
    'base/dofa geobench_so2sat_10b 800'
    'base/panopticon geobench_so2sat_10b 200'

    # brik kiln (48-55)
    'base/dinov2 geobench_brick_kiln_rgb 800'
    'base/croma_s2 geobench_brick_kiln_12b 900'
    'base/softcon_13b geobench_brick_kiln_13b 900'
    'base/anysat_s2 geobench_brick_kiln_10b 100'
    'base/galileo_s2 geobench_brick_kiln_10b 500'
    'base/senpamae geobench_brick_kiln_13b 500'
    'base/dofa geobench_brick_kiln_13b 800'
    'base/panopticon_v2 geobench_brick_kiln_13b 200'

    # ben-opt (56-63) (skipped because of benv2)
    # 'base/dinov2 geobench_ben_rgb 800'
    # 'base/croma_s2 geobench_ben_12b 900'
    # 'base/softcon_13b geobench_ben_13b 900'
    # 'base/anysat_s2 geobench_ben_10b 100'
    # 'base/galileo_s2 geobench_ben_10b 500'
    # 'base/senpamae geobench_ben_13b 500'
    # 'base/dofa geobench_ben_13b 800'
    # 'base/panopticon geobench_ben_13b 200'

    # geobench_pv4ger_cls (64-68)
    'base/dinov2 geobench_pv4ger_cls 900'
    'base/anysat_spot geobench_pv4ger_cls 100'
    'base/senpamae geobench_pv4ger_cls 900'
    'base/dofa geobench_pv4ger_cls 900'
    'base/panopticon geobench_pv4ger_cls 400'
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
    model=$1
    dataset=$2
    ids=$3
    batch_size=$4
    subset=$5

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
        +model=$model \
        +data=$dataset\
        +optim=$mode \
        +output_dir=\'$ODIR/$mode/$dataset/$model/$ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        optim.check_val_every_n_epoch=100 \
        $add_kwargs \
        # optim.epochs=1 \
        # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/$model/\' \
        # overwrite=true \

done