#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export $(cat /home/ando/fm-playground/.env)
export MODEL_SIZE=base
echo "Output Directory": $ODIR
echo "Model Size": $MODEL_SIZE

model=senpamae_cls
dataset=geobench_eurosat
batch_size=512
lr=0.002
epochs=30
warmup_epochs=0
task=classification
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python /home/ando/fm-playground/src/main.py \
output_dir=${ODIR}/exps/${model}_${dataset} \
model=${model} \
dataset=${dataset} \
lr=${lr} \
task=${task} \
num_gpus=${num_gpus} \
num_workers=8 \
epochs=${epochs} \
warmup_epochs=${warmup_epochs} \
seed=13 \
batch_size=512 \
