#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=sinv
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=01:00:00


# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True"

# ---------- HOREKA ------------
# eval_cmd='srun -K1 --export=ALL /home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval2/bin/python /home/hk-project-pai00028/tum_mhj8661/code/geobreeze/geobreeze/main.py'
REPO_PATH=/home/hk-project-pai00028/tum_mhj8661/code/geobreeze
export $(cat $REPO_PATH/.env)
cmd="/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python $REPO_PATH/geobreeze/main.py"
# -----------------------------



tasks=(
  " 1 [3]"
  " 1 [5]"
  " 1 [1]"
  " 1 [0]"
  " 1 [2]"
  " 3 [2,4,5]"
  " 3 [2,3,5]"
  " 3 [2,3,4]"
  " 3 [0,1,2]"
  " 3 [0,1,4]"
  " 5 [1,2,3,4,5]"
  " 5 [0,1,2,4,5]"
  " 5 [0,1,2,3,4]"
  " 5 [0,1,3,4,5]"
  " 5 [0,1,2,3,5]"
  " 6 [0,1,2,3,4,5]"
)

models=(
  # "panopticon"
  # "dofa"
  "senpamae"
  # 'dinov2'
)
dataset=m-forestnet




for model in "${models[@]}"
do
  echo model
  for task in "${tasks[@]}"
  do
    echo $task
    set -- $task
    num_bands=$1
    ids=$2


    $cmd \
        +model=base/$model \
        +data=$dataset\
        +optim=knn \
        +output_dir=\'$ODIR/spec_inv/$dataset/$model/$num_bands/$ids\' \
        dl.batch_size=100 \
        dl.num_workers=8 \
        num_gpus=1 \
        seed=21 \
        data.train.transform=\${_vars.augm.cls.val} \
        +data.train.band_ids=$ids \
        +data.val.band_ids=$ids \
        +data.test.band_ids=$ids \
        # overwrite=true \

  done
done