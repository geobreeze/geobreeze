#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=eval
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
# -----------------------------

# m-eurosat

# dataset=m-eurosat_gsdinv
# models=(
#     "croma_12b true [0,1,2,3,4,5,6,7,8,9,11,12]"
#     "dinov2 true [3,2,1]"
#     "softcon_13b false -1"
#     "panopticon false -1"
#     "dofa false -1"
# )

# tasks=(
#     "100 64"
#     "50 32"
#     "25 16"
#     "12.5 8"
# )



# resisc45
dataset=resisc45_gsdinv
models=(
    "dinov2 false "
    "panopticon false -1"
    "dofa false -1"
)

tasks=(
    "100 224"
    # "50 112"
    # "25 56"
    # "12.5 28"
)


for model in "${models[@]}"
do
    set $model
    model=$1
    subset=$2
    ids=$3

    for task in "${tasks[@]}"
    do
        echo $task
        set -- $task
        prc=$1
        size=$2

        # potentially subset
        add_kwargs=""
        if [ "$subset" = true ]; then
            add_kwargs="$add_kwargs \
                +data.train.band_ids=$ids \
                +data.val.band_ids=$ids \
                +data.test.band_ids=$ids "
        fi

        # main command
        $cmd \
            +model=base/$model \
            +data=$dataset\
            +optim=knn \
            +output_dir=\'$ODIR/gsd_inv/only_val/$dataset/$model/$prc/\' \
            dl.batch_size=100 \
            dl.num_workers=8 \
            num_gpus=1 \
            seed=21 \
            data.train.transform.0.size=224 \
            data.val.transform.0.size=$size \
            data.test.transform.0.size=$size \
            $add_kwargs \
            # overwrite=true \

    done
done