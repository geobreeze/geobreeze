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
base_output_dir=/path/to/your/output/dir

export $(cat $REPO_PATH/.env)
cmd="$PYTHON $REPO_PATH/geobreeze/main.py"



# This file executes experiments for segmentation with frozen backbone. 
# Below, we only list geobench datasets with dinov2. Other datasets and models can 
# be added similarly. All other model configurations are already created in `config/models`.




tasks=(
    'base/dinov2 m-pv4ger-seg 200'
    'base/dinov2 m-chesapeake 200 [0,1,2]'
    'base/dinov2 m-cashew-plant 200 [3,2,1]'
    'base/dinov2 m-SA-crop-type 200 [3,2,1]'
    'base/dinov2 m-nz-cattle 200'
    'base/dinov2 m-NeonTree 200'

    'base/panopticon m-pv4ger-seg 200'
    'base/panopticon m-chesapeake 200'
    'base/panopticon m-cashew-plant 200'
    'base/panopticon m-SA-crop-type 200'
    'base/panopticon m-nz-cattle 200'
    'base/panopticon m-NeonTree 200'

    'base/senpamae m-pv4ger-seg 200'
    'base/senpamae m-chesapeake 200'
    'base/senpamae m-cashew-plant 200'
    'base/senpamae m-SA-crop-type 200'
    'base/senpamae m-nz-cattle 200'
    'base/senpamae m-NeonTree 200'

    'base/dofa m-pv4ger-seg 200'
    'base/dofa m-chesapeake 200'
    'base/dofa m-cashew-plant 200'
    'base/dofa m-SA-crop-type 200'
    'base/dofa m-nz-cattle 200'
    'base/dofa m-NeonTree 200'

    # 'base/softcon_13b m-pv4ger-seg 200'
    # 'base/softcon_13b m-chesapeake 200'
    'base/softcon_13b m-cashew-plant 200'
    'base/softcon_13b m-SA-crop-type 200'
    # 'base/softcon_13b m-nz-cattle 200'
    # 'base/softcon_13b m-NeonTree 200'

     # 'base/croma_12b m-pv4ger-seg 200'
    # 'base/croma_12b m-chesapeake 200'
    'base/croma_12b m-cashew-plant 200 [0,1,2,3,4,5,6,7,8,9,11,12]'
    'base/croma_12b m-SA-crop-type 200 [0,1,2,3,4,5,6,7,8,9,11,12]'
    # 'base/croma_12b m-nz-cattle 200'
    # 'base/croma_12b m-NeonTree 200'

    'base/anysat_spot m-pv4ger-seg 100'
    'base/anysat_naip m-chesapeake 100'
    'base/anysat_s2 m-cashew-plant 100'
    'base/anysat_s2 m-SA-crop-type 100'
    'base/anysat_spot m-nz-cattle 100'
    'base/anysat_spot m-NeonTree 100'

        # 'base/croma_12b m-pv4ger-seg 200'
    # 'base/croma_12b m-chesapeake 200'
    'base/galileo_s2 m-cashew-plant 100 [1,2,3,4,5,6,7,8,11,12]'
    'base/galileo_s2 m-SA-crop-type 100 [1,2,3,4,5,6,7,8,11,12]'
    # 'base/croma_12b m-nz-cattle 200'
    # 'base/croma_12b m-NeonTree 200'
)

lrs="1e-3 1e-4 1e-5 1e-6"
mode=segmentation


# build all tasks as the cross product of tasks and lrs
all_tasks=()
for task in "${tasks[@]}"; do
    for lr in $lrs; do
        all_tasks+=("$lr $task")
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


# execute tasks
for task_id in "${task_ids[@]}"
do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set $task
    base_lr=$1
    model=$2
    dataset=$3
    batch_size=$4
    ids=$5
    subset=$6

    add_kwargs=""

    # uncomment if fastdevrun
    # add_kwargs+="optim.epochs=1 dl.batch_size=20 overwrite=True"

    # potentially subset channels
    if [ "$ids" == "" ]; then
        print_ids=-1
    elif [ "$ids" != "-1" ]; then
        print_ids=$ids
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # potentially subset samples
    if [ "$subset" == "" ]; then
        subset=-1
    elif [ "$subset" != "-1" ]; then
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
        optim.base_lr=$base_lr \
        +output_dir=\'$base_output_dir/$mode/$dataset/$model/$print_ids/\' \
        dl.batch_size=$batch_size \
        dl.num_workers=10 \
        num_gpus=1 \
        seed=21 \
        $add_kwargs \


done