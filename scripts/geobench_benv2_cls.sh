export $(cat .env) 

fastdevrun=no
exp_base_name=t1_v3

all_tasks=(

    'base/dinov2 linear_probe spacenet1_3b 800'

    # m-eurosat, 0-7 (50e = 0h15 (lp)) 
    'base/dinov2 linear_probe geobench_eurosat_rgb 800'
    'base/croma_s2 linear_probe geobench_eurosat_12b 900'
    'base/softcon_13b linear_probe geobench_eurosat_13b 900'
    'base/anysat_s2 linear_probe geobench_eurosat_10b 100'
    'base/galileo_s2 linear_probe geobench_eurosat_10b 500'
    'base/senpamae linear_probe geobench_eurosat_13b 500'
    'base/dofa linear_probe geobench_eurosat_13b 800'
    'base/panopticon linear_probe geobench_eurosat_13b 200'

    # resisc45, 8-12 (50e = 1h30 (lp)) 
    'base/dinov2 linear_probe resisc45 900'
    'base/anysat_spot linear_probe resisc45 100' # beeds ~3h
    'base/senpamae linear_probe resisc45 900'
    'base/dofa linear_probe resisc45 900'
    'base/panopticon linear_probe resisc45 400'

    # benv2-s1, 13-18
    'base/croma_s1 linear_probe benv2_s1 900'
    'base/softcon_2b linear_probe benv2_s1 900'
    'base/anysat_s1-asc linear_probe benv2_s1 100' # needs ~12h 
    'base/galileo_s1 linear_probe benv2_s1 100' 
    'base/dofa linear_probe benv2_s1 500'
    'base/panopticon linear_probe benv2_s1 400'


    # benv2-s2, 19-26 (50e = ~4h)
    'base/dinov2 linear_probe benv2_rgb 900'
    'base/croma_s2 linear_probe benv2_s2_12b 900'
    'base/softcon_13b linear_probe benv2_s2_13b 800'
    'base/softcon_13b linear_probe benv2_s2_13b_scnorm 800'
    'base/anysat_s2 linear_probe benv2_s2_10b 100' 
    'base/galileo_s2 linear_probe benv2_s2_10b 200'
    'base/senpamae linear_probe benv2_s2_12b 300' 
    'base/dofa linear_probe benv2_s2_12b 500'
    'base/panopticon linear_probe benv2_s2_12b 200'

    # benv2-s1, softcon norm: (4-9)
    'base/softcon_2b linear_probe benv2_s1_scnorm 900'
    'base/softcon_13b linear_probe benv2_s2_13b_scnorm 800'
    'base/panopticon linear_probe benv2_s1_scnorm 400'
    'base/panopticon linear_probe benv2_s2_12b_scnorm 200'
    'base/croma_s1 linear_probe benv2_s1_scnorm 900'
    'base/croma_s2 linear_probe benv2_s2_12b_scnorm 900'
    'base/anysat_s1-asc linear_probe benv2_s1_scnorm 100' # needs ~12h
    'base/galileo_s1 linear_probe benv2_s1_scnorm 100' 
    'base/dofa linear_probe benv2_s1_scnorm 500'

    # benv2-s2, 19-26 (50e = ~4h) (10-17)
    'base/dinov2 linear_probe benv2_rgb_scnorm 900'
    'base/anysat_s2 linear_probe benv2_s2_10b_scnorm 100' 
    'base/galileo_s2 linear_probe benv2_s2_10b_scnorm 200'
    'base/senpamae linear_probe benv2_s2_12b_scnorm 500' 
    'base/dofa linear_probe benv2_s2_12b_scnorm 500'

    # forestnet, 27-31 (1h) 
    'base/dinov2 linear_probe geobench_forestnet_rgb 900'
    'base/anysat_naip linear_probe geobench_forestnet_4b 100'
    'base/senpamae linear_probe geobench_forestnet_6b 600'
    'base/dofa linear_probe geobench_forestnet_6b 900'
    'base/panopticon linear_probe geobench_forestnet_6b 200'

    # fmow-wv, 32-34 (50e = ~4h) 
    'base/senpamae linear_probe fmow_8b 600'
    'base/dofa linear_probe fmow_8b 900'
    'base/panopticon linear_probe fmow_8b 200'
    'base/dinov2 linear_probe fmow_8b_rgb 900'

    # fmow-rgb, 35-39 (50e = ~4h) 
    'base/dinov2 linear_probe fmow_rgb 900'
    'base/anysat_spot linear_probe fmow_rgb 100'
    'base/senpamae linear_probe fmow_rgb 900'
    'base/dofa linear_probe fmow_rgb 900'
    'base/panopticon linear_probe fmow_rgb 400'

    ######################################
    #### other geobench cls tasks
    ######################################

    # so2sat-s2 (40-47)
    'base/dinov2 linear_probe geobench_so2sat_rgb 800'
    'base/croma_s2 linear_probe geobench_so2sat_12b 900'
    'base/softcon_13b linear_probe geobench_so2sat_13b 900'
    'base/anysat_s2 linear_probe geobench_so2sat_10b 100'
    'base/galileo_s2 linear_probe geobench_so2sat_10b 500'
    'base/senpamae linear_probe geobench_so2sat_10b 500'
    'base/dofa linear_probe geobench_so2sat_10b 800'
    'base/panopticon linear_probe geobench_so2sat_10b 200'

    # brik kiln (48-55)
    'base/dinov2 linear_probe geobench_brick_kiln_rgb 800'
    'base/croma_s2 linear_probe geobench_brick_kiln_12b 900'
    'base/softcon_13b linear_probe geobench_brick_kiln_13b 900'
    'base/anysat_s2 linear_probe geobench_brick_kiln_10b 100'
    'base/galileo_s2 linear_probe geobench_brick_kiln_10b 500'
    'base/senpamae linear_probe geobench_brick_kiln_13b 500'
    'base/dofa linear_probe geobench_brick_kiln_13b 800'
    'base/panopticon_v2 linear_probe geobench_brick_kiln_13b 200'

    # ben-opt (56-63) (skipped because of benv2)
    # 'base/dinov2 linear_probe geobench_ben_rgb 800'
    # 'base/croma_s2 linear_probe geobench_ben_12b 900'
    # 'base/softcon_13b linear_probe geobench_ben_13b 900'
    # 'base/anysat_s2 linear_probe geobench_ben_10b 100'
    # 'base/galileo_s2 linear_probe geobench_ben_10b 500'
    # 'base/senpamae linear_probe geobench_ben_13b 500'
    # 'base/dofa linear_probe geobench_ben_13b 800'
    # 'base/panopticon linear_probe geobench_ben_13b 200'

    # geobench_pv4ger_cls (64-68)
    'base/dinov2 linear_probe geobench_pv4ger_cls 900'
    'base/anysat_spot linear_probe geobench_pv4ger_cls 100'
    'base/senpamae linear_probe geobench_pv4ger_cls 900'
    'base/dofa linear_probe geobench_pv4ger_cls 900'
    'base/panopticon linear_probe geobench_pv4ger_cls 400'
)


########## linear probe defaults

optim=sgd

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=8
check_val_every_n_epoch=10
val_subset=0.2

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
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
        lrs_partial_ft="0.1"

    elif [ $fastdevrun == 'bsz' ]; then
        echo "fastdevrun 'bsz'!"
        cmd="$cmd epochs=1 batch_size=$batch_size trainer.check_val_every_n_epoch=1"
        s=$((batch_size * 1))
        echo $s
        cmd="$cmd dataset.subset.train=$s dataset.subset.val=$s dataset.subset.test=$s"
        lrs_partial_ft="0.1"

    else
        echo "fastdevrun not recognized"
        exit 1
    fi


    if [ $training_mode == 'linear_probe' ]; then
        
        cmd="$cmd \
            +optim=\${_optims.$optim} \
            "
        echo $cmd
        $cmd
            
    elif [ $training_mode == 'partial_finetune' ]; then

        for lr in $lrs_partial_ft; do
            echo "partial finetune with lr=$lr"
            lr_cmd="$cmd \
                +lr=$lr \
                +model.params_to_train=[] \
                warmup_epochs=$warmup_epochs \
                "
            echo $lr_cmd
            $lr_cmd
        done
    fi
            

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done