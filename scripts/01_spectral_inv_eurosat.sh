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
  " 1 [1]"
  " 1 [8]"
  " 1 [7]"
  " 1 [0]"
  " 1 [12]"
  " 3 [0,5,7]"
  " 3 [9,10,12]"
  " 3 [2,5,7]"
  " 3 [3,4,11]"
  " 3 [2,5,10]"
  " 5 [2,4,10,11,12]"
  " 5 [0,3,4,7,8]"
  " 5 [1,2,4,6,12]"
  " 5 [1,2,4,7,9]"
  " 5 [0,1,10,11,12]"
  " 7 [3,5,6,7,8,10,11]"
  " 7 [0,1,2,3,4,5,7]"
  " 7 [1,2,3,4,7,8,9]"
  " 7 [0,4,6,7,8,11,12]"
  " 7 [2,5,6,7,8,9,12]"
  " 9 [0,1,3,4,5,6,8,9,11]"
  " 9 [0,1,2,4,5,7,8,9,11]"
  " 9 [1,2,4,5,7,9,10,11,12]"
  " 9 [0,2,3,4,5,7,10,11,12]"
  " 9 [1,2,3,4,5,6,8,10,11]"
  "11 [0,1,4,5,6,7,8,9,10,11,12]"
  "11 [0,1,3,4,5,6,7,8,9,11,12]"
  "11 [0,1,2,3,4,5,6,8,9,11,12]"
  "11 [0,1,2,3,4,5,7,8,9,11,12]"
  "11 [0,1,2,3,5,6,8,9,10,11,12]"
  "13 [0,1,2,3,4,5,6,7,8,9,10,11,12]"
)



model=senpamae
dataset=m-eurosat

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