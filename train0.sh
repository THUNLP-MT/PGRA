#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
STEPS=5000
READER=bert150
# READER=bert
# READER=fever
# READER=get_data_bm25_150
K=48

LR=8e-5
BSZ=4
TASKS=SST-2
python  train_reader.py \
        --tasks ${TASKS} \
        --name ${READER}/${TASKS} \
        --train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
        --eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
        --test_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
        --model_size base \
        --lr ${LR} \
        --per_gpu_batch_size ${BSZ} \
        --total_step ${STEPS} \
        --n_context ${K} \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 128 \
        --warmup_step 0


# TASKS=cr
# LR=1e-5
# BSZ=8
# python  train_reader.py \
# 		--tasks ${TASKS} \
# 		--name ${READER}/${TASKS} \
# 		--train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
# 		--eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
# 		--model_size base \
# 		--lr ${LR} \
# 		--per_gpu_batch_size ${BSZ} \
# 		--total_step ${STEPS} \
# 		--n_context ${K} \
# 		--optim adamw \
# 		--scheduler linear \
# 		--weight_decay 0.01 \
# 		--text_maxlength 128 \
# 		--warmup_step 0



# LR=8e-5
# BSZ=4
# TASKS=mpqa
# python  train_reader.py \
#         --tasks ${TASKS} \
#         --name ${READER}/${TASKS} \
#         --train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
#         --eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
#         --model_size base \
#         --lr ${LR} \
#         --per_gpu_batch_size ${BSZ} \
#         --total_step ${STEPS} \
#         --n_context ${K} \
#         --optim adamw \
#         --scheduler linear \
#         --weight_decay 0.01 \
#         --text_maxlength 128 \
#         --warmup_step 0


# LR=8e-5
# BSZ=8
# # K=8
# TASKS=trec
# python  train_reader.py \
#         --tasks ${TASKS} \
#         --name ${READER}/${TASKS} \
#         --train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
#         --eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
#         --model_size base \
#         --lr ${LR} \
#         --per_gpu_batch_size ${BSZ} \
#         --total_step ${STEPS} \
#         --n_context ${K} \
#         --optim adamw \
#         --scheduler linear \
#         --weight_decay 0.01 \
#         --text_maxlength 128 \
#         --warmup_step 0

# LR=8e-5
# BSZ=8
# K=16
# TASKS=trec
# python  train_reader.py \
#         --tasks ${TASKS} \
#         --name ${READER}/${TASKS} \
#         --train_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.train.json \
#         --eval_data /share/project/guozhicheng/FiD/get_data_${READER}/${TASKS}.dev.json \
#         --model_size base \
#         --lr ${LR} \
#         --per_gpu_batch_size ${BSZ} \
#         --total_step ${STEPS} \
#         --n_context ${K} \
#         --optim adamw \
#         --scheduler linear \
#         --weight_decay 0.01 \
#         --text_maxlength 128 \
#         --warmup_step 0

