export $(cat .env) 

fastdevrun=no
exp_base_name=t4

all_tasks=(

    # m-pv4seg (rgb) 0-4
    'base/dinov2 frozen_backbone geobench_pv4ger_seg 100'
    'base/anysat_spot frozen_backbone geobench_pv4ger_seg 100'
    'base/senpamae frozen_backbone geobench_pv4ger_seg 100' 
    'base/dofa frozen_backbone geobench_pv4ger_seg 100'
    'base/panopticon_v4 frozen_backbone geobench_pv4ger_seg 100'

    # m-cashew (s2) 5-12 (<2h)
    'base/dinov2 frozen_backbone geobench_cashew_rgb 100'
    'base/croma_s2 frozen_backbone geobench_cashew_12b 100'
    'base/softcon_13b frozen_backbone geobench_cashew_13b 100'
    'base/anysat_s2 frozen_backbone geobench_cashew_10b 100' 
    'base/galileo_s2 frozen_backbone geobench_cashew_10b 100' 
    'base/senpamae frozen_backbone geobench_cashew_12b 100' 
    'base/dofa frozen_backbone geobench_cashew_12b 100' 
    'base/panopticon_v4 frozen_backbone geobench_cashew_12b 100'

    # chesapeak (rgb / naip) 13-17
    'base/dinov2 frozen_backbone geobench_chesapeake_rgb 100'
    'base/anysat_naip frozen_backbone geobench_chesapeake_4b 100'
    'base/senpamae frozen_backbone geobench_chesapeake_rgb 100'
    'base/dofa frozen_backbone geobench_chesapeake_4b 100'
    'base/panopticon_v4 frozen_backbone geobench_chesapeake_4b 100'

    # m-nzcattle (rgb) 18-22
    'base/dinov2 frozen_backbone geobench_nzcattle 100'
    'base/anysat_spot frozen_backbone geobench_nzcattle 100'
    'base/senpamae frozen_backbone geobench_nzcattle 100'
    'base/dofa frozen_backbone geobench_nzcattle 100'
    'base/panopticon_v4 frozen_backbone geobench_nzcattle 100'

    # neontree (only rgb) 23-27
    'base/dinov2 frozen_backbone geobench_neontree 100'
    'base/anysat_spot frozen_backbone geobench_neontree 100'
    'base/senpamae frozen_backbone geobench_neontree 100'
    'base/dofa frozen_backbone geobench_neontree 100'
    'base/panopticon_v4 frozen_backbone geobench_neontree 100'

    # sa crop (s2) 28-35
    'base/croma_s2 frozen_backbone geobench_sacrop_12b 100'
    'base/softcon_13b frozen_backbone geobench_sacrop_13b 100'
    'base/dinov2 frozen_backbone geobench_sacrop_rgb 100'
    'base/anysat_s2 frozen_backbone geobench_sacrop_10b 100'
    'base/galileo_s2 frozen_backbone geobench_sacrop_10b 100'
    'base/senpamae frozen_backbone geobench_sacrop_12b 100'
    'base/dofa frozen_backbone geobench_sacrop_12b 100'
    'base/panopticon_v4 frozen_backbone geobench_sacrop_12b 100'
)



########## defaults 

epochs=50
num_workers=10
check_val_every_n_epoch=1
val_subset=-1
lrs='0.1 0.01 0.001 0.0001 0.00001 0.000001'
warmup_epochs=0

########## get tasks

if [ $# -eq 0 ]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        task_ids=($SLURM_ARRAY_TASK_ID)
    else
        task_ids=($(seq 0 $((${#all_tasks[@]}-1))))
    fi
else
    task_ids=("$@")
fi


########## execution

for task_id in "${task_ids[@]}"; do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    batch_size=$4

    cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$exp_base_name/$ds/$model/$training_mode/ \
        +model.training_mode=$training_mode \
        ++batch_size=$batch_size \
        num_workers=$num_workers \
        num_gpus=1 \
        seed=21 \
        "

    if [ $fastdevrun == 'no' ]; then
        cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch dataset.subset.val=$val_subset"

    elif [ $fastdevrun == 'fast' ]; then
        echo "fastdevrun 'fast'!"
        cmd="$cmd epochs=1 batch_size=32 trainer.check_val_every_n_epoch=1"
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=32"
        lrs="0.1"

    elif [ $fastdevrun == 'bsz' ]; then
        echo "fastdevrun 'bsz'!"
        cmd="$cmd epochs=1 batch_size=$batch_size trainer.check_val_every_n_epoch=1"
        s=$((batch_size * 1))
        echo $s
        cmd="$cmd dataset.subset.train=$s dataset.subset.val=$s dataset.subset.test=$s"
        lrs="0.1"

    else
        echo "fastdevrun not recognized"
        exit 1
    fi


    for lr in $lrs; do
        echo "subtask with lr=$lr"
        lr_cmd="$cmd \
            +lr=$lr \
            warmup_epochs=$warmup_epochs \
            "
        echo $lr_cmd
        $lr_cmd
    done

            

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done