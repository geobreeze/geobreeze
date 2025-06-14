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
This file executes experiments for domain adaptation with re-training a conv2d as
patch embedding with the correct and fixed number of channels.
"""



tasks=(
    "corine-sd frozen_backbone croma_12b -1 200"
    "corine-sd frozen_backbone softcon_13b -1 200"
    "corine-sd frozen_backbone anysat_naip -1 100"

    "corine-md frozen_backbone croma_12b -1 200"
    "corine-md frozen_backbone softcon_13b -1 200"
    "corine-md frozen_backbone anysat_naip -1 100"

    "hyperview-sd frozen_backbone croma_12b -1 200"
    "hyperview-sd frozen_backbone softcon_13b -1 200"
    "hyperview-sd frozen_backbone anysat_naip -1 100"

    "hyperview-md frozen_backbone croma_12b -1 200"
    "hyperview-md frozen_backbone softcon_13b -1 200"
    "hyperview-md frozen_backbone anysat_naip -1 100"

    "digital_typhoon-10 frozen_backbone croma_12b -1 200"
    "digital_typhoon-10 frozen_backbone softcon_13b -1 200"
    "digital_typhoon-10 frozen_backbone anysat_naip -1 100"

    "tropical_cyclone-10 frozen_backbone croma_12b -1 200"
    "tropical_cyclone-10 frozen_backbone softcon_13b -1 200"
    "tropical_cyclone-10 frozen_backbone anysat_naip -1 100"

    # 72-
    "corine-sd frozen_backbone dinov2 -1 200"
    "corine-md frozen_backbone dinov2 -1 200"
    "hyperview-sd frozen_backbone dinov2 -1 200"
    "hyperview-md frozen_backbone dinov2 -1 200"
    "digital_typhoon-10 frozen_backbone dinov2 -1 200"
    "tropical_cyclone-10 frozen_backbone dinov2 -1 200"

    # 96-
    "fmow-10 frozen_backbone croma_12b -1 200"
    "fmow-10 frozen_backbone softcon_13b -1 200"
    "fmow-10 frozen_backbone anysat_naip -1 100"
    "fmow-10 frozen_backbone dinov2 -1 200"

)

lrs="1e-2 1e-3 5e-4 1e-4"  # original
# lrs="1e-1 5e-1 5e-2 5e-3 1e-5 1e-6"



# generate all tasks
all_tasks=()
for t in "${tasks[@]}"; do
    for lr in $lrs; do
        all_tasks+=("$t $lr")
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
    echo "Running Task: $task"
    set $task
    dataset=$1
    optim=$2
    model=$3
    ids=$4
    bsz=$5
    lr=$6

    # potentially subset
    add_kwargs=""
    if [ "$ids" != "-1" ]; then
        add_kwargs="$add_kwargs \
            +data.train.band_ids=$ids \
            +data.val.band_ids=$ids \
            +data.test.band_ids=$ids "
    fi

    # if [ "$optim" == "knn" ]; then
    #     add_kwargs="$add_kwargs \
    #         +data.train.transform=\${_vars.augm.cls.val}"
    # elif [ "$optim" == "linear_probe" ]; then
    #     add_kwargs="$add_kwargs \
    #         +data.train.transform=\${_vars.augm.cls.train}"
    if [ "$optim" == "frozen_backbone" ]; then
        add_kwargs="$add_kwargs \
            data.train.transform=\${_vars.augm.cls.train}"
    else
        echo "Unknown training mode: $optim"
        exit 1
    fi

    # main command
    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=frozen_backbone_adamw \
        +output_dir=\'$ODIR/domain_adapt/pe_adamw/$dataset/$optim/$model/$ids\' \
        dl.batch_size=$bsz \
        dl.num_workers=8 \
        optim.base_lr=$lr \
        num_gpus=1 \
        seed=21 \
        +model.replace_pe=True \
        optim.check_val_every_n_epoch=100 \
        logger=none \
        $add_kwargs \
        # overwrite=true \

done