#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
STEPS=5000
READER=2stage_fewshot/bert150_opt-13b
# CHECKPOINT=/share/project/guozhicheng/FiD/checkpoint/${READER}/SST-2_lr8e-05_bsz8_steps5000_k16/checkpoint/best_dev
# CHECKPOINT=/share/project/guozhicheng/FiD/checkpoint/bm25/SST-2_lr8e-05_bsz8_steps3000_k16/checkpoint/best_dev
CHECKPOINT=/share/project/guozhicheng/FiD/checkpoint/2stage_fewshot/bert150_opt-13b/SST-2_lr8e-05_bsz8_steps5000_k16/checkpoint/best_dev

LR=8e-5
BSZ=8
K=16
TASKS=SST-2
python  evaluate_reader.py \
    --tasks ${TASKS} \
    --name ${READER}/${TASKS} \
    --train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
    --eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
    --model_size base \
    --model_checkpoint $CHECKPOINT \
    --lr ${LR} \
    --per_gpu_batch_size ${BSZ} \
    --total_step ${STEPS} \
    --n_context ${K} \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 128 \
    --warmup_step 0