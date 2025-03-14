
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

all_tasks=(
    ######## classification
    'base/croma_s2 linear_probe geobench_eurosat_12b'
    # 'base/dofa linear_probe geobench_eurosat'
    'base/dinov2 linear_probe geobench_eurosat_rgb'    
    'base/softcon_B13 linear_probe geobench_eurosat'
    'base/panopticon linear_probe geobench_eurosat'
    # 'base/senpamae linear_probe geobench_eurosat'

    ######## 
    # 'base/croma_s2 full_finetune geobench_eurosat_12b'

    ######## not checked
    # 'base/croma_s1 linear_probe geobench_eurosat_2b'
    # 'base/anysat linear_probe geobench_eurosat_rgb'
)

suffix='debug2'

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3

    $cmd \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$model/$training_mode/$ds/$suffix \
        +model.training_mode=$training_mode \
        \
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=600 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \
        +trainer.fast_dev_run=False \
        \
        +model.params_to_train=[]
        # dataset.subset.train=0.4 # just as example how to use subset 
        
done