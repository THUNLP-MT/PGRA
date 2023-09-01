# PGRA
Prompt-Guided Retrieval For Non-Knowledge-Intensive Tasks

This is the code for [Prompt-Guided Retrieval For Non-Knowledge-Intensive Tasks](https://arxiv.org/abs/2305.17653)

## Data
In this work, we use the Wiki1m dataset in SimCSE as the retrieval dataset. You can download this dataset from the [downloading scripts](https://github.com/princeton-nlp/SimCSE/blob/main/data/download_wiki.sh) provided by them.

As for the training and testing data, we follow the setup of [LM-BFF](https://github.com/princeton-nlp/SimCSE/blob/main/data/download_wiki.sh). Please follow the data preparation process provided by them. After processing the data, you will get a folder named k-shot with training and testing files of all tasks inside. 

## First Stage Retrieval
We provide first-stage retrieval with BM25, BERT and SimCSE. The codes for each retriever are in `get_data_bm25`, `get_data_bert`, `get_data_simcse` respectively. 

Firstly, embeddings of sentences in Wiki1m need to be generated if you use BERT or SimCSE. For example, if you want to use the BERT model as the retriever. You can run the following code:
```
WIKI1M_PATH=
OUTPUT_FILE=
MODEL_NAME_OR_PATH=bert-base-uncased

python get_data_bert/sbert.py \
    --wiki1m_path $WIKI1M_PATH \
    --output_sents_file $OUTPUT_FILE \
    --sbert $MODEL_NAME_OR_PATH
```
An embedding file of Wiki1m with BERT will be save to `OUTPUT_FILE`.

Then, this embedding file can be used to perform retrieval. To do this, you need to firstly fill in the train and test files of tasks in the `single_sentence.py` file in each folder. Then, take BERT retriever as an example, you can run the following command:
```
task=
python get_data_bert/single_sentence.py ${task}
```
This will generate the top 150 retrieved evidence for each sentence in the task and save this into `${task}.train.json` or `${task}.test.json`.

We add examples of final CoLA examples in the `get_data_bert` folder.


## Second-stage retrieval
In this stage, pretrained language models (in our case OPT models) are used to further rerank the retrieval process. For efficiency purposes, we split this process into two stages. 

On the first stage, the OPT models are used to compute embeddings of retrieved evidence for a given task. Due to the fact that many retrieved evidence will appear more than once in a task, this step can save a lot of time. To run this, run the following command:
```
export CUDA_VISIBLE_DEVICES=
task=
SPLIT=
MODEL_NAME_OR_PATH=

python second_stage.py \
        --task $task \
        --split $SPLIT \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --src_tag bert \
        --save_tag bert \
        --d -1
```
This script will process `get_data_bert/${task}.${SPLIT}.json` file and output `get_data_2stage_fewshot/bert_${MODEL_NAME}/${TASK}-${SPLIT}_ctxs_embeddings.pt`.

After this, this embedding can be used to generate the rerank the output. To do this, run
```
python write_from_embeddings.py \
    --task $task \
    --split $split \
    --model opt-13b \
    --src_tag bert \
    --embedding_tag bert \
    --save_tag bert
    --d -1
```

At this time, the model is the stem of `MODEL_NAME_OR_PATH` at the last step. This time, files generated before will be used and final reranked evidence will be saved at `get_data_2stage_fewshot/bert_${MODEL_NAME}/${TASK}.${SPLIT}.json`. We also provided such example files in the `get_data_2stage_fewshot/bert_opt-13b` folder. 


## Train the reader
We copy the [FID](https://github.com/facebookresearch/FiD) implementation to train the reader. Before doing this, you may need to split the train file into train and dev splits.

To run the trainer, you can run the following command:
```
STEPS=5000
READER=2stage_fewshot/bert_opt-13b
export TRANSFORMERS_VERBOSITY=error

LR=1e-4
BSZ=8
K=16
TASKS=cola
SEED=
T5_path=

python  train_reader.py \
    --tasks ${TASKS} \
    --name ${READER}/${TASKS} \
    --train_data get_data_${READER}/${TASKS}.train.json \
    --eval_data get_data_${READER}/${TASKS}.dev.json \
    --model_name_or_path $T5_path \
    --model_size base \
    --seed $seed \
    --lr ${LR} \
    --per_gpu_batch_size ${BSZ} \
    --total_step ${STEPS} \
    --n_context ${K} \
    --optim adamw \
    --scheduler linear \
    --weight_decay 0.01 \
    --text_maxlength 128 \
    --warmup_step 0
```