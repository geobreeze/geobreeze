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



tasks=(
  " 3 [7,11,12]"
  " 3 [3,4,10]"
  " 3 [0,7,12]"
  " 3 [5,6,7]"
  " 3 [10,11,12]"
  " 3 [0,9,12]"
  " 3 [4,6,10]"
  " 3 [7,9,12]"
  " 3 [0,1,10]"
  " 3 [8,10,12]"
  " 3 [5,8,11]"
  " 3 [0,9,10]"
  " 3 [2,9,10]"
  " 3 [7,9,10]"
  " 3 [0,4,11]"
  " 3 [1,6,9]"
  " 3 [4,11,12]"
  " 3 [3,7,11]"
  " 3 [1,8,9]"
  " 3 [0,10,12]"
)



model=dinov2
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
        +output_dir=\'$ODIR/random_chns/$dataset/$model/$num_bands/$ids\' \
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