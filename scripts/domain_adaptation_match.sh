#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out
#!/bin/bash
# optional: your sbatch options
# ...
# SBATCH array=0-30


# setup
REPO_PATH=/path/to/geobreeze
PYTHON=/path/to/your/python/bin

export $(cat $REPO_PATH/.env)
cmd="$PYTHON $REPO_PATH/geobreeze/main.py"


"""
This file executes experiments for domain adaptation with matching channels.
We match the channels to the closest channel in central wavelength. 
"""


all_tasks=(
    "fmow-10 anysat_s2 [1,2,4,5,5,6,6,6,7,7] 100 -1 fmow-10"
    "hyperview anysat_s2 [9,32,64,77,77,77,126,126,126,126] 100 -1 hyperview-sd"
    "hyperview anysat_s2 [8,29,64,67,89,89,124,127,149,149] 100 -1 hyperview-md"
    "corine anysat_s2 [15,30,47,53,53,53,74,74,74,74] 100 -1 corine-sd"
    "corine anysat_s2 [14,28,47,49,59,59,73,74,135,160] 100 -1 corine-md"

    "fmow-10 croma_12b [0,1,2,4,5,5,6,6,6,7,7,7] 200 -1 fmow-10"
    "hyperview croma_12b [0,9,32,64,77,77,77,126,126,126,126,126] 200 -1 hyperview-sd"
    "hyperview croma_12b [0,8,29,64,67,89,89,124,127,127,149,149] 200 -1 hyperview-md"
    "corine croma_12b [5,15,30,47,53,53,53,74,74,74,74,74] 200 -1 corine-sd"
    "corine croma_12b [4,14,28,47,49,59,59,73,74,74,135,160] 200 -1 corine-md"

    "fmow-10 softcon_13b [0,1,2,4,5,5,6,6,6,7,7,7,7] 100 -1 fmow-10"
    "hyperview softcon_13b [0,9,32,64,77,77,77,126,126,126,126,126,126] 300 -1 hyperview-sd"
    "hyperview softcon_13b [0,8,29,64,67,89,89,124,127,127,149,149,149] 300 -1 hyperview-md"
    "corine softcon_13b [5,15,30,47,53,53,53,74,74,74,74,74,74] 300 -1 corine-sd"
    "corine softcon_13b [4,14,28,47,49,59,59,73,74,74,122,135,160] 300 -1 corine-md"

    "digital_typhoon-10 anysat_s2 [0,0,0,0,0,0,0,0,0,0] 100 -1 digital_typhoon-10"
    "digital_typhoon-10 croma_12b [0,0,0,0,0,0,0,0,0,0,0,0] 200 -1 digital_typhoon-10"
    "digital_typhoon-10 softcon_13b [0,0,0,0,0,0,0,0,0,0,0,0,0] 300 -1 digital_typhoon-10"

    "tropical_cyclone-10 anysat_s2 [0,0,0,0,0,0,0,0,0,0] 100 -1 tropical_cyclone-10"
    "tropical_cyclone-10 croma_12b [0,0,0,0,0,0,0,0,0,0,0,0] 200 -1 tropical_cyclone-10"
    "tropical_cyclone-10 softcon_13b [0,0,0,0,0,0,0,0,0,0,0,0,0] 300 -1 tropical_cyclone-10"

    "digital_typhoon-10 dinov2 [0,0,0] 300 -1 digital_typhoon-10"
    "tropical_cyclone-10 dinov2 [0,0,0] 300 -1 tropical_cyclone-10"
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
        +output_dir=\'$ODIR/domain_adapt/match/$ds_name_output_dir/$mode/$model/$ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        optim.epochs=50 \
        optim.check_val_every_n_epoch=100 \
        $add_kwargs \
        # optim.epochs=1 \
        # +output_dir=\'$OLD_ODIR/t1_v3/$dataset/base/$model/\' \
        # overwrite=true \

done